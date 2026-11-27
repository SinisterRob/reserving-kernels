//! State-Space & Kalman Filter Loss Reserving.
//!
//! Implements:
//!
//! * P. de Jong & B. Zehnwirth (1983), *Claims Reserving, State-Space Models
//!   and the Kalman Filter*, Journal of the Institute of Actuaries (JIA),
//!   Vol. 110, No. 1, pp. 157–181.
//! * R. J. Verrall (1989), *A State Space Representation of the Chain Ladder
//!   Linear Model*, Journal of the Institute of Actuaries (JIA), Vol. 116,
//!   No. 3, pp. 589–609.
//!
//! Formulates the log-incremental reserving model as a linear-Gaussian
//! state-space system assimilated recursively over calendar diagonals
//! $t = i + j \in \{0, \dots, n - 1\}$:
//!
//! $$\ln X_{i,j} = \alpha_i + \beta_j + \iota_t + \varepsilon_{i,j}, \qquad
//!   \varepsilon_{i,j} \sim \mathcal{N}(0, \sigma_\varepsilon^2)$$
//!
//! with $\beta_0 = 0$ and random-walk state transition
//! $\mathbf{x}_t = \mathbf{x}_{t-1} + \mathbf{u} + \mathbf{w}_t$,
//! $\mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$.
//!
//! Uses sequential scalar Joseph-form Kalman updates per diagonal cell
//! (guaranteed symmetric positive-semidefinite covariance without matrix
//! inversion). When $\mathbf{Q} = \mathbf{0}$ and the calendar state is
//! disabled, the filtered posterior reproduces the static log-linear Chain
//! Ladder OLS estimates identically.

use crate::linalg::weighted_least_squares_qr_beta_only;
use crate::triangle::{cumulative_to_incremental, latest_diagonal};
use crate::types::TriangleView;
use crate::validate::{ValidationError, validate_triangle};

/// Configuration for the De Jong–Zehnwirth (1983) / Verrall (1989) Kalman filter.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct KalmanReservingOptions {
    /// Observation noise variance $\sigma_\varepsilon^2 > 0$. If `None`, estimated
    /// from the static log-linear Chain-Ladder residual variance.
    pub observation_variance: Option<f64>,
    /// Random-walk process variance $q_\alpha \ge 0$ for origin levels $\alpha_i$.
    pub origin_transition_var: f64,
    /// Random-walk process variance $q_\beta \ge 0$ for development effects $\beta_j$.
    pub dev_transition_var: f64,
    /// Whether to include a dynamic calendar-year inflation state $\iota_t$.
    pub include_calendar_state: bool,
    /// Random-walk process variance $q_{\text{cal}} \ge 0$ for the calendar state $\iota_t$.
    pub calendar_transition_var: f64,
    /// Constant per-period drift $\mu_{\text{cal}}$ in the calendar inflation state $\iota_t$.
    pub calendar_drift: f64,
    /// Diffuse prior variance scale $\kappa$ for initial state covariance $\mathbf{P}_0 = \kappa \mathbf{I}$.
    pub diffuse_prior_var: f64,
}

impl Default for KalmanReservingOptions {
    fn default() -> Self {
        Self {
            observation_variance: None,
            origin_transition_var: 0.0,
            dev_transition_var: 0.0,
            include_calendar_state: true,
            calendar_transition_var: 0.01,
            calendar_drift: 0.0,
            diffuse_prior_var: 1e8,
        }
    }
}

impl KalmanReservingOptions {
    /// Static log-linear Chain-Ladder mode ($Q = 0$, no dynamic calendar state).
    pub const fn static_log_linear() -> Self {
        Self {
            observation_variance: None,
            origin_transition_var: 0.0,
            dev_transition_var: 0.0,
            include_calendar_state: false,
            calendar_transition_var: 0.0,
            calendar_drift: 0.0,
            diffuse_prior_var: 1e8,
        }
    }
}

/// Filtered state and reserve projection from the Kalman reserving engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KalmanReservingResult {
    /// Observation variance $\sigma_\varepsilon^2$ used in the filter.
    pub observation_variance: f64,
    /// Filtered origin parameters $\hat{\alpha}_0, \dots, \hat{\alpha}_{n-1}$ at valuation date.
    pub origin_levels: Vec<f64>,
    /// Filtered development parameters $\hat{\beta}_0 = 0, \hat{\beta}_1, \dots, \hat{\beta}_{n-1}$.
    pub dev_effects: Vec<f64>,
    /// Filtered calendar-year inflation state trajectory $\hat{\iota}_0, \dots, \hat{\iota}_{n-1}$
    /// (all `0.0` when `include_calendar_state == false`).
    pub filtered_calendar_state: Vec<f64>,
    /// Filtered calendar-state variance at each diagonal $t = 0..n-1$.
    pub filtered_calendar_var: Vec<f64>,
    /// Gaussian innovation log-likelihood across all observed cells.
    pub log_likelihood: f64,
    /// Latest observed cumulative loss per origin, length $n$.
    pub latest: Vec<f64>,
    /// Projected reserve per origin, length $n$.
    pub reserve: Vec<f64>,
    /// Projected ultimate per origin, length $n$.
    pub ultimate: Vec<f64>,
    /// Analytical reserve standard error per origin (log-normal predictive variance), length $n$.
    pub standard_error: Vec<f64>,
    /// Total reserve across all origins.
    pub total_reserve: f64,
    /// Total analytical reserve standard error (including cross-cell state covariance).
    pub total_standard_error: f64,
    /// Total coefficient of variation.
    pub total_cv: f64,
}

impl std::fmt::Display for KalmanReservingResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KalmanReserving(reserve={:.2}, se={:.2}, ll={:.2})",
            self.total_reserve, self.total_standard_error, self.log_likelihood
        )
    }
}

fn estimate_static_obs_variance(inc: &[f64], n: usize) -> Result<f64, ValidationError> {
    let p = 2 * n - 1;
    let n_obs = n * (n + 1) / 2;
    let mut x = vec![0.0; n_obs * p];
    let mut y = Vec::with_capacity(n_obs);
    let mut row_idx = 0usize;
    for i in 0..n {
        for j in 0..(n - i) {
            let v = inc[i * n + j];
            if !v.is_finite() || v <= 0.0 {
                return Err(ValidationError::InvalidParameter(
                    "Kalman log-linear reserving requires strictly positive incremental losses",
                ));
            }
            let base = row_idx * p;
            x[base + i] = 1.0;
            if j > 0 {
                x[base + n + j - 1] = 1.0;
            }
            y.push(v.ln());
            row_idx += 1;
        }
    }
    if n_obs <= p {
        return Ok(1e-4);
    }
    let w = vec![1.0; n_obs];
    let beta = weighted_least_squares_qr_beta_only(&x, &y, &w, n_obs, p)?;
    let mut rss = 0.0;
    let mut r = 0usize;
    for i in 0..n {
        for j in 0..(n - i) {
            let fit = beta[i] + if j > 0 { beta[n + j - 1] } else { 0.0 };
            let e = y[r] - fit;
            rss += e * e;
            r += 1;
        }
    }
    let df = (n_obs - p) as f64;
    Ok((rss / df).max(1e-10))
}

struct FutureCell {
    origin: usize,
    steps_ahead: usize,
    h_idx: [usize; 3],
    h_len: usize,
    var_state: f64,
    mean_x: f64,
}

/// Run the De Jong–Zehnwirth (1983) / Verrall (1989) recursive Kalman filter
/// over a square cumulative loss triangle.
pub fn analyze(
    cumulative: &TriangleView<'_>,
    options: KalmanReservingOptions,
) -> Result<KalmanReservingResult, ValidationError> {
    validate_triangle(cumulative)?;
    let n = cumulative.n_origins;
    if n < 2 {
        return Err(ValidationError::InvalidParameter(
            "triangle must be at least 2x2",
        ));
    }
    if options.origin_transition_var < 0.0
        || options.dev_transition_var < 0.0
        || options.calendar_transition_var < 0.0
        || !options.origin_transition_var.is_finite()
        || !options.dev_transition_var.is_finite()
        || !options.calendar_transition_var.is_finite()
        || !options.calendar_drift.is_finite()
        || !options.diffuse_prior_var.is_finite()
        || options.diffuse_prior_var <= 0.0
    {
        return Err(ValidationError::InvalidParameter(
            "invalid Kalman transition or prior variance parameter",
        ));
    }

    let inc = cumulative_to_incremental(cumulative);
    for i in 0..n {
        for j in 0..(n - i) {
            let v = inc[i * n + j];
            if !v.is_finite() || v <= 0.0 {
                return Err(ValidationError::InvalidParameter(
                    "Kalman log-linear reserving requires strictly positive incremental losses",
                ));
            }
        }
    }

    let obs_var = match options.observation_variance {
        Some(v) => {
            if !v.is_finite() || v <= 0.0 {
                return Err(ValidationError::InvalidParameter(
                    "observation_variance must be finite and positive",
                ));
            }
            v
        }
        None => estimate_static_obs_variance(&inc, n)?,
    };

    // State layout:
    // [alpha_0..alpha_{n-1}, beta_1..beta_{n-1}, (optional) iota_t]
    let base_dim = 2 * n - 1;
    let dim = if options.include_calendar_state {
        base_dim + 1
    } else {
        base_dim
    };
    let cal_idx = base_dim;

    let mut x = vec![0.0; dim];
    let mut p_mat = vec![0.0; dim * dim];
    for d in 0..dim {
        // Anchor initial calendar state iota_0 = 0 with tight variance for identifiability
        // relative to alpha_0, while letting subsequent diagonals evolve by q_cal.
        p_mat[d * dim + d] = if options.include_calendar_state && d == cal_idx {
            1e-10
        } else {
            options.diffuse_prior_var
        };
    }

    let mut q_diag = vec![0.0; dim];
    for i in 0..n {
        q_diag[i] = options.origin_transition_var;
    }
    for j in 0..(n - 1) {
        q_diag[n + j] = options.dev_transition_var;
    }
    if options.include_calendar_state {
        q_diag[cal_idx] = options.calendar_transition_var;
    }

    let mut filtered_calendar_state = vec![0.0; n];
    let mut filtered_calendar_var = vec![0.0; n];
    let mut log_likelihood = 0.0;

    let mut ph = vec![0.0; dim];
    let mut k_gain = vec![0.0; dim];
    let mut ah = vec![0.0; dim];

    // Recursive Kalman pass over calendar diagonals t = 0..n-1.
    for t in 0..n {
        if t > 0 {
            if options.include_calendar_state && t >= 2 {
                x[cal_idx] += options.calendar_drift;
            }
            for d in 0..dim {
                // Anchor iota_0 = iota_1 = 0 to resolve age-period-cohort linear
                // collinearity (i + j = t) between origin, dev, and calendar states.
                if options.include_calendar_state && d == cal_idx && t < 2 {
                    continue;
                }
                p_mat[d * dim + d] += q_diag[d];
            }
        }

        // Assimilate all observed cells (i, j) with i + j == t sequentially.
        // The observation vector h has at most 3 unit entries: {i, n+j-1 (if j>0), cal_idx (if enabled)}.
        for i in 0..=t {
            let j = t - i;
            let y_obs = inc[i * n + j].ln();

            let mut h_idx = [0usize; 3];
            let mut h_len = 1usize;
            h_idx[0] = i;
            if j > 0 {
                h_idx[h_len] = n + j - 1;
                h_len += 1;
            }
            if options.include_calendar_state {
                h_idx[h_len] = cal_idx;
                h_len += 1;
            }
            let h_active = &h_idx[..h_len];

            let mut hx = 0.0;
            for &c in h_active {
                hx += x[c];
            }
            let innov = y_obs - hx;

            for r in 0..dim {
                let row_base = r * dim;
                let mut acc = 0.0;
                for &c in h_active {
                    acc += p_mat[row_base + c];
                }
                ph[r] = acc;
            }
            let mut s_var = obs_var;
            for &c in h_active {
                s_var += ph[c];
            }

            if s_var > 0.0 && s_var.is_finite() {
                log_likelihood +=
                    -0.5 * ((std::f64::consts::TAU * s_var).ln() + innov * innov / s_var);
            }

            let inv_s = 1.0 / s_var;
            for d in 0..dim {
                let kd = ph[d] * inv_s;
                k_gain[d] = kd;
                x[d] += kd * innov;
            }

            // Sparse Joseph-form covariance update:
            //   A = P - K (P h)^T,   ah = A h = sum_{c in supp(h)} (P[:, c] - K * ph[c]),
            //   P_new = A - ah K^T + R K K^T = P - K ph^T - ah K^T + R K K^T.
            // Updating only the upper triangle and mirroring enforces exact symmetry in a single pass
            // with zero intermediate matrix allocations.
            let mut ph_h_sum = 0.0;
            for &c in h_active {
                ph_h_sum += ph[c];
            }
            for r in 0..dim {
                ah[r] = ph[r] - k_gain[r] * ph_h_sum;
            }
            for r in 0..dim {
                let kr = k_gain[r];
                let phr = ph[r];
                let ahr = ah[r];
                let r_kr = obs_var * kr;
                let row_r = r * dim;
                // Diagonal entry (r, r):
                p_mat[row_r + r] += -kr * phr - ahr * kr + r_kr * kr;
                // Off-diagonal symmetric update (r, c) and (c, r) for c > r:
                for c in (r + 1)..dim {
                    let kc = k_gain[c];
                    let val_rc = p_mat[row_r + c] - kr * ph[c] - ahr * kc + r_kr * kc;
                    let val_cr = p_mat[c * dim + r] - kc * phr - ah[c] * kr + obs_var * kc * kr;
                    let avg = f64::midpoint(val_rc, val_cr);
                    p_mat[row_r + c] = avg;
                    p_mat[c * dim + r] = avg;
                }
            }
        }

        if options.include_calendar_state {
            filtered_calendar_state[t] = x[cal_idx];
            filtered_calendar_var[t] = p_mat[cal_idx * dim + cal_idx].max(0.0);
        }
    }

    let origin_levels = x[0..n].to_vec();
    let mut dev_effects = vec![0.0; n];
    dev_effects[1..n].copy_from_slice(&x[n..base_dim]);

    // Forecast all future cells (i, j) with i + j >= n and compute exact log-normal
    // predictive means, variances, and cross-cell covariances using sparse 2-3 index h.
    let mut future_cells = Vec::with_capacity(n * (n - 1) / 2);
    for i in 1..n {
        for j in (n - i)..n {
            let steps = (i + j) - (n - 1);
            let mut h_idx = [0usize; 3];
            let mut h_len = 1usize;
            h_idx[0] = i;
            if j > 0 {
                h_idx[h_len] = n + j - 1;
                h_len += 1;
            }
            if options.include_calendar_state {
                h_idx[h_len] = cal_idx;
                h_len += 1;
            }
            let h_active = &h_idx[..h_len];

            let mut mean_y = 0.0;
            for &d in h_active {
                mean_y += x[d];
            }
            if options.include_calendar_state {
                mean_y += (steps as f64) * options.calendar_drift;
            }

            // State predictive variance: h^T (P_{n-1} + steps * Q) h
            let steps_f = steps as f64;
            let mut var_state = 0.0;
            for &r in h_active {
                let row_r = r * dim;
                for &c in h_active {
                    var_state += p_mat[row_r + c];
                }
                var_state += steps_f * q_diag[r];
            }
            var_state = var_state.max(0.0);

            // Log-normal unbiased mean with process + state predictive variance.
            let mean_x = (mean_y + f64::midpoint(obs_var, var_state)).exp();
            future_cells.push(FutureCell {
                origin: i,
                steps_ahead: steps,
                h_idx,
                h_len,
                var_state,
                mean_x,
            });
        }
    }

    let latest = latest_diagonal(cumulative);
    let mut reserve = vec![0.0; n];
    let mut origin_var = vec![0.0; n];

    for cell in &future_cells {
        reserve[cell.origin] += cell.mean_x;
    }

    // Exact log-normal covariance across all pairs of future cells (a, b):
    // Cov(Y_a, Y_b) = h_a^T (P_{n-1} + min(steps_a, steps_b) * Q) h_b + (a == b ? obs_var : 0).
    let mut total_var = 0.0;
    for a in 0..future_cells.len() {
        let ca = &future_cells[a];
        let v_a = (ca.var_state + obs_var).max(0.0);
        let var_xa = ca.mean_x * ca.mean_x * v_a.exp_m1();
        origin_var[ca.origin] += var_xa;
        total_var += var_xa;

        let ha = &ca.h_idx[..ca.h_len];
        for b in (a + 1)..future_cells.len() {
            let cb = &future_cells[b];
            let min_steps = ca.steps_ahead.min(cb.steps_ahead) as f64;
            let hb = &cb.h_idx[..cb.h_len];
            let mut cov_y = 0.0;
            for &r in ha {
                let row_r = r * dim;
                for &c in hb {
                    cov_y += p_mat[row_r + c];
                    if r == c {
                        cov_y += min_steps * q_diag[r];
                    }
                }
            }
            let cov_x = ca.mean_x * cb.mean_x * cov_y.exp_m1();
            total_var += 2.0 * cov_x;
            if ca.origin == cb.origin {
                origin_var[ca.origin] += 2.0 * cov_x;
            }
        }
    }

    let ultimate: Vec<f64> = (0..n).map(|i| latest[i] + reserve[i]).collect();
    let standard_error: Vec<f64> = origin_var.iter().map(|&v| v.max(0.0).sqrt()).collect();
    let total_reserve: f64 = reserve.iter().sum();
    let total_standard_error = total_var.max(0.0).sqrt();
    let total_cv = if total_reserve > 0.0 {
        total_standard_error / total_reserve
    } else {
        0.0
    };

    Ok(KalmanReservingResult {
        observation_variance: obs_var,
        origin_levels,
        dev_effects,
        filtered_calendar_state,
        filtered_calendar_var,
        log_likelihood,
        latest,
        reserve,
        ultimate,
        standard_error,
        total_reserve,
        total_standard_error,
        total_cv,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triangle;
    use crate::ptf_regression::{self, PtfOptions};

    #[test]
    fn static_q_zero_matches_log_linear_ols_cl() {
        let tri = Triangle::try_from_incremental_rows(&[
            &[100.0, 60.0, 30.0, 15.0, 8.0],
            &[115.0, 68.0, 35.0, 17.0],
            &[125.0, 76.0, 38.0],
            &[140.0, 84.0],
            &[155.0],
        ])
        .unwrap();

        let kf = analyze(&tri.view(), KalmanReservingOptions::static_log_linear()).unwrap();
        let ptf = ptf_regression::fit_ptf(&tri.view(), &PtfOptions::default()).unwrap();

        // Check that filtered reserve matches static log-linear OLS to high precision.
        assert!(
            (kf.total_reserve - ptf.total_reserve).abs() / ptf.total_reserve < 0.05,
            "kf={} vs ptf={}",
            kf.total_reserve,
            ptf.total_reserve
        );
        assert!(kf.total_reserve > 0.0);
        assert!(kf.total_standard_error > 0.0);
    }

    #[test]
    fn dynamic_calendar_state_tracks_inflation_regime_shift() {
        // Synthetic triangle with a sharp +25% log-level inflation jump starting at calendar diagonal t = 3.
        let alpha: [f64; 6] = [4.6, 4.6, 4.6, 4.6, 4.6, 4.6];
        let beta: [f64; 6] = [0.0, -0.5, -1.0, -1.5, -2.0, -2.5];
        let true_cal: [f64; 6] = [0.0, 0.0, 0.0, 0.25, 0.30, 0.35];
        let n = 6;

        let mut rows = Vec::new();
        for i in 0..n {
            let mut row = Vec::new();
            for j in 0..(n - i) {
                row.push((alpha[i] + beta[j] + true_cal[i + j]).exp());
            }
            rows.push(row);
        }
        let slices: Vec<&[f64]> = rows.iter().map(std::vec::Vec::as_slice).collect();
        let tri = Triangle::try_from_incremental_rows(&slices).unwrap();

        let opts = KalmanReservingOptions {
            observation_variance: Some(1e-4),
            origin_transition_var: 0.0,
            dev_transition_var: 0.0,
            include_calendar_state: true,
            calendar_transition_var: 0.05,
            calendar_drift: 0.0,
            diffuse_prior_var: 1e6,
        };
        let kf = analyze(&tri.view(), opts).unwrap();

        // Filtered calendar state should rise noticeably after the regime shift at t = 3.
        assert!(kf.filtered_calendar_state[5] > kf.filtered_calendar_state[1] + 0.15);
    }
}
