//! Murphy (1994) generalized weighted-regression loss development factors.
//!
//! Implements:
//!
//! * D. M. Murphy (1994), *Unbiased Loss Development Factors*, Proceedings of
//!   the Casualty Actuarial Society (PCAS), Vol. LXXXI, pp. 154–222.
//!
//! For each development link $j \to j+1$, Murphy models:
//!
//! $$C_{i,j+1} = \alpha_j + \beta_j C_{i,j} + \varepsilon_{i,j}, \qquad
//!   \mathrm{Var}(\varepsilon_{i,j} \mid C_{i,j}) = \sigma_j^2 C_{i,j}^\delta$$
//!
//! with variance power $\delta \ge 0$ (canonical values $\delta \in \{0, 1, 2\}$)
//! and optional intercept $\alpha_j$:
//!
//! * $\alpha_j = 0, \delta = 1$: Volume-weighted Chain Ladder — **exact
//!   equivalence to [`crate::stochastic::mack`]** (LDFs, $\hat{\sigma}_j^2$, per-origin
//!   MSE, and total-reserve MSE).
//! * $\alpha_j = 0, \delta = 2$: Simple average of link ratios
//!   $\hat{\beta}_j = \frac{1}{n_j} \sum_i \frac{C_{i,j+1}}{C_{i,j}}$.
//! * $\alpha_j = 0, \delta = 0$: Regression-through-the-origin OLS LDF
//!   $\hat{\beta}_j = \frac{\sum_i C_{i,j} C_{i,j+1}}{\sum_i C_{i,j}^2}$.
//! * $\alpha_j \ne 0, \delta = 0$: Unweighted linear regression — **exact
//!   equivalence to [`crate::deterministic::brosius`]** point estimates, plus analytical MSE.

use crate::error::ReservingError;
use crate::preprocessing::triangle::latest_diagonal;
use crate::types::TriangleView;
use crate::validate::{
    MonotonicityPolicy, validate_monotone_cumulative, validate_tail, validate_triangle,
};

/// Options controlling the Murphy (1994) weighted-regression model.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MurphyOptions {
    /// Variance power $\delta$ in $\mathrm{Var}(\varepsilon_{i,j} \mid C_{i,j}) = \sigma_j^2 C_{i,j}^\delta$.
    /// Canonical values: `0.0` (homoscedastic OLS), `1.0` (Mack / volume-weighted CL),
    /// `2.0` (simple-average LDF).
    pub variance_power: f64,
    /// Whether to estimate a free intercept $\alpha_j$ (`true`) or constrain
    /// $\alpha_j = 0$ (regression through the origin, `false`).
    pub include_intercept: bool,
    /// Multiplicative tail factor $\ge 1.0$.
    pub tail: f64,
    /// Cumulative monotonicity policy.
    pub policy: MonotonicityPolicy,
}

impl Default for MurphyOptions {
    fn default() -> Self {
        Self::mack_equivalent()
    }
}

impl MurphyOptions {
    /// Volume-weighted Chain Ladder ($\delta = 1$, no intercept) — matches Mack (1993).
    pub const fn mack_equivalent() -> Self {
        Self {
            variance_power: 1.0,
            include_intercept: false,
            tail: 1.0,
            policy: MonotonicityPolicy::Strict,
        }
    }

    /// Simple-average link ratio model ($\delta = 2$, no intercept).
    pub const fn simple_average() -> Self {
        Self {
            variance_power: 2.0,
            include_intercept: false,
            tail: 1.0,
            policy: MonotonicityPolicy::Strict,
        }
    }

    /// Regression-through-origin OLS model ($\delta = 0$, no intercept).
    pub const fn ols_through_origin() -> Self {
        Self {
            variance_power: 0.0,
            include_intercept: false,
            tail: 1.0,
            policy: MonotonicityPolicy::Strict,
        }
    }

    /// Brosius (1993) OLS with intercept ($\delta = 0$, with intercept).
    pub const fn brosius_equivalent() -> Self {
        Self {
            variance_power: 0.0,
            include_intercept: true,
            tail: 1.0,
            policy: MonotonicityPolicy::Strict,
        }
    }
}

/// Fitted weighted-regression diagnostics for a single development link.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MurphyLinkFit {
    /// Development link start index $j$.
    pub dev: usize,
    /// Number of valid observed pairs.
    pub n_obs: usize,
    /// Fitted intercept $\hat{\alpha}_j$ (`0.0` when `include_intercept == false`).
    pub intercept: f64,
    /// Fitted slope $\hat{\beta}_j$.
    pub slope: f64,
    /// Estimated residual variance scale $\hat{\sigma}_j^2$.
    pub sigma_sq: f64,
    /// Parameter variance $\mathrm{Var}(\hat{\alpha}_j)$.
    pub var_intercept: f64,
    /// Parameter variance $\mathrm{Var}(\hat{\beta}_j)$.
    pub var_slope: f64,
    /// Parameter covariance $\mathrm{Cov}(\hat{\alpha}_j, \hat{\beta}_j)$.
    pub cov_intercept_slope: f64,
}

/// Full Murphy (1994) weighted-regression reserving result with analytical MSE.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MurphyResult {
    /// Options used to fit the model.
    pub options: MurphyOptions,
    /// Per-link regression parameters and parameter covariances, length $n - 1$.
    pub links: Vec<MurphyLinkFit>,
    /// Latest observed cumulative loss per origin, length $n$.
    pub latest: Vec<f64>,
    /// Projected ultimate per origin, length $n$.
    pub ultimate: Vec<f64>,
    /// Projected reserve per origin, length $n$.
    pub reserve: Vec<f64>,
    /// Analytical process variance per origin, length $n$.
    pub process_variance: Vec<f64>,
    /// Analytical parameter estimation variance per origin, length $n$.
    pub parameter_variance: Vec<f64>,
    /// Analytical total MSE per origin (`process_variance + parameter_variance`), length $n$.
    pub mse: Vec<f64>,
    /// Analytical standard error per origin, length $n$.
    pub standard_error: Vec<f64>,
    /// Total reserve across all origins.
    pub total_reserve: f64,
    /// Total analytical MSE across all origins (including cross-origin parameter covariance).
    pub total_mse: f64,
    /// Total analytical standard error.
    pub total_standard_error: f64,
    /// Total coefficient of variation.
    pub total_cv: f64,
}

impl std::fmt::Display for MurphyResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Murphy(delta={}, intercept={}, reserve={:.2}, se={:.2})",
            self.options.variance_power,
            self.options.include_intercept,
            self.total_reserve,
            self.total_standard_error
        )
    }
}

struct RawLink {
    n_obs: usize,
    alpha: f64,
    beta: f64,
    sigma_sq: Option<f64>,
    // Unscaled (X^T W X)^{-1} entries: [[u_aa, u_ab], [u_ab, u_bb]]
    u_aa: f64,
    u_ab: f64,
    u_bb: f64,
}

/// Fit Murphy (1994) weighted-regression LDFs and compute analytical reserve MSE.
pub fn analyze(
    t: &TriangleView<'_>,
    options: MurphyOptions,
) -> Result<MurphyResult, ReservingError> {
    validate_triangle(t)?;
    validate_tail(options.tail)?;
    if !options.variance_power.is_finite() || options.variance_power < 0.0 {
        return Err(ReservingError::InvalidParameter(
            "variance_power must be finite and non-negative",
        ));
    }
    if matches!(options.policy, MonotonicityPolicy::Strict) {
        validate_monotone_cumulative(t)?;
    }

    let n = t.n_origins;
    let n_links = n.saturating_sub(1);
    let delta = options.variance_power;

    // Step 1: Fit per-link weighted OLS parameters and raw sigma^2.
    let mut raw_links = Vec::with_capacity(n_links);
    for j in 0..n_links {
        let nk = n - j - 1;
        let mut xs = Vec::with_capacity(nk);
        let mut ys = Vec::with_capacity(nk);
        let mut ws = Vec::with_capacity(nk);

        for i in 0..nk {
            let x = t.get(i, j);
            let y = t.get(i, j + 1);
            if x > 0.0 {
                xs.push(x);
                ys.push(y);
                ws.push(if delta == 0.0 { 1.0 } else { x.powf(-delta) });
            } else if x == 0.0 && y == 0.0 {
                // Zero-to-zero pairs contribute no ratio information.
            } else if delta == 0.0 {
                xs.push(x);
                ys.push(y);
                ws.push(1.0);
            } else {
                return Err(ReservingError::DivideByZero(
                    "zero prior cumulative with positive variance_power",
                ));
            }
        }

        let m = xs.len();
        if m == 0 {
            return Err(ReservingError::DivideByZero(
                "no valid observations for development link",
            ));
        }

        if !options.include_intercept || m == 1 {
            // Regression through origin: beta = sum(w x y) / sum(w x^2).
            let mut num = 0.0;
            let mut den = 0.0;
            for k in 0..m {
                num += ws[k] * xs[k] * ys[k];
                den += ws[k] * xs[k] * xs[k];
            }
            if den <= 0.0 {
                return Err(ReservingError::DivideByZero("weighted sum of squares"));
            }
            let beta = num / den;
            let sigma_sq = if m >= 2 {
                let mut rss = 0.0;
                for k in 0..m {
                    let r = ys[k] - beta * xs[k];
                    rss += ws[k] * r * r;
                }
                Some(rss / (m as f64 - 1.0))
            } else {
                None
            };
            raw_links.push(RawLink {
                n_obs: m,
                alpha: 0.0,
                beta,
                sigma_sq,
                u_aa: 0.0,
                u_ab: 0.0,
                u_bb: 1.0 / den,
            });
        } else {
            // Weighted OLS with intercept using centered sums for numerical stability.
            let sum_w: f64 = ws.iter().sum();
            let mean_x = (0..m).map(|k| ws[k] * xs[k]).sum::<f64>() / sum_w;
            let mean_y = (0..m).map(|k| ws[k] * ys[k]).sum::<f64>() / sum_w;
            let mut sxx = 0.0;
            let mut sxy = 0.0;
            for k in 0..m {
                let dx = xs[k] - mean_x;
                let dy = ys[k] - mean_y;
                sxx += ws[k] * dx * dx;
                sxy += ws[k] * dx * dy;
            }
            let var_tol = 1e-12 * sum_w * mean_x * mean_x;
            let (alpha, beta, u_aa, u_ab, u_bb) = if sxx > var_tol {
                let b = sxy / sxx;
                let a = mean_y - b * mean_x;
                (
                    a,
                    b,
                    1.0 / sum_w + (mean_x * mean_x) / sxx,
                    -mean_x / sxx,
                    1.0 / sxx,
                )
            } else {
                // Constant regressor fallback: pure intercept model.
                (mean_y, 0.0, 1.0 / sum_w, 0.0, 0.0)
            };
            let sigma_sq = if m >= 3 {
                let mut rss = 0.0;
                for k in 0..m {
                    let r = ys[k] - (alpha + beta * xs[k]);
                    rss += ws[k] * r * r;
                }
                Some(rss / (m as f64 - 2.0))
            } else {
                None
            };
            raw_links.push(RawLink {
                n_obs: m,
                alpha,
                beta,
                sigma_sq,
                u_aa,
                u_ab,
                u_bb,
            });
        }
    }

    // Extrapolate missing tail sigma^2 via Mack's min-rule.
    let raw_sigmas: Vec<Option<f64>> = raw_links.iter().map(|r| r.sigma_sq).collect();
    let sigma_sq_extrap = crate::deterministic::ldf::extrapolate_mack_min_rule(&raw_sigmas);

    let mut links = Vec::with_capacity(n_links);
    for j in 0..n_links {
        let s2 = sigma_sq_extrap[j];
        let r = &raw_links[j];
        links.push(MurphyLinkFit {
            dev: j,
            n_obs: r.n_obs,
            intercept: r.alpha,
            slope: r.beta,
            sigma_sq: s2,
            var_intercept: s2 * r.u_aa,
            var_slope: s2 * r.u_bb,
            cov_intercept_slope: s2 * r.u_ab,
        });
    }

    // Step 2: Project full cumulative grid and recursive process/parameter variances.
    let latest = latest_diagonal(t);
    let mut proj = vec![0.0; n * n];
    let mut ultimate = vec![0.0; n];
    let mut reserve = vec![0.0; n];
    let mut process_variance = vec![0.0; n];
    let mut parameter_variance = vec![0.0; n];
    let mut mse = vec![0.0; n];
    let mut standard_error = vec![0.0; n];

    let tail_sq = options.tail * options.tail;

    for i in 0..n {
        let last_obs = n - 1 - i;
        for k in 0..=last_obs {
            proj[i * n + k] = t.get(i, k);
        }
        let mut c = latest[i];
        let mut pv = 0.0;
        let mut ev = 0.0;
        for k in last_obs..n_links {
            let lk = &links[k];
            let b2 = lk.slope * lk.slope;
            let c_pow = if delta == 0.0 {
                1.0
            } else if c > 0.0 {
                c.powf(delta)
            } else {
                0.0
            };
            pv = b2 * pv + lk.sigma_sq * c_pow;
            let step_param_var =
                lk.var_intercept + 2.0 * c * lk.cov_intercept_slope + c * c * lk.var_slope;
            ev = b2 * ev + step_param_var;
            c = lk.intercept + lk.slope * c;
            proj[i * n + k + 1] = c;
        }
        let ult = c * options.tail;
        let pv_ult = pv * tail_sq;
        let ev_ult = ev * tail_sq;
        let m_ult = (pv_ult + ev_ult).max(0.0);

        ultimate[i] = ult;
        reserve[i] = ult - latest[i];
        process_variance[i] = pv_ult;
        parameter_variance[i] = ev_ult;
        mse[i] = m_ult;
        standard_error[i] = m_ult.sqrt();
    }

    // Step 3: Cross-origin parameter covariance (recursive propagation over shared links).
    let mut total_mse: f64 = mse.iter().sum();
    for i in 1..n {
        let start_k = n - 1 - i;
        for j in (i + 1)..n {
            let mut cov_ij = 0.0;
            for k in start_k..n_links {
                let lk = &links[k];
                let ci = proj[i * n + k];
                let cj = proj[j * n + k];
                let step_cov =
                    lk.var_intercept + (ci + cj) * lk.cov_intercept_slope + ci * cj * lk.var_slope;
                cov_ij = lk.slope * lk.slope * cov_ij + step_cov;
            }
            total_mse += 2.0 * cov_ij * tail_sq;
        }
    }
    total_mse = total_mse.max(0.0);
    let total_standard_error = total_mse.sqrt();
    let total_reserve: f64 = reserve.iter().sum();
    if !total_mse.is_finite()
        || !total_reserve.is_finite()
        || ultimate.iter().any(|v| !v.is_finite())
        || reserve.iter().any(|v| !v.is_finite())
    {
        return Err(ReservingError::NotFinite);
    }
    let total_cv = if total_reserve > 0.0 {
        total_standard_error / total_reserve
    } else {
        0.0
    };

    Ok(MurphyResult {
        options,
        links,
        latest,
        ultimate,
        reserve,
        process_variance,
        parameter_variance,
        mse,
        standard_error,
        total_reserve,
        total_mse,
        total_standard_error,
        total_cv,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triangle;
    use crate::deterministic::brosius;
    use crate::stochastic::mack;

    #[test]
    fn exact_equivalence_to_mack_at_delta_1_no_intercept() {
        let tri = Triangle::try_from_upper_rows(&[
            &[100.0, 165.0, 200.0, 212.0, 218.0],
            &[120.0, 190.0, 238.0, 250.0],
            &[135.0, 222.0, 268.0],
            &[150.0, 240.0],
            &[160.0],
        ])
        .unwrap();

        let m = mack::analyze(&tri.view(), 1.0).unwrap();
        let mu = analyze(&tri.view(), MurphyOptions::mack_equivalent()).unwrap();

        for j in 0..4 {
            assert!((mu.links[j].slope - m.ldf[j]).abs() < 1e-12);
            assert!((mu.links[j].sigma_sq - m.sigma_sq[j]).abs() < 1e-10);
        }
        for i in 0..5 {
            assert!((mu.reserve[i] - m.reserve[i]).abs() < 1e-10);
            assert!((mu.standard_error[i] - m.standard_error[i]).abs() < 1e-10);
        }
        assert!((mu.total_reserve - m.total_reserve).abs() < 1e-10);
        assert!((mu.total_standard_error - m.total_standard_error).abs() < 1e-10);
    }

    #[test]
    fn exact_equivalence_to_brosius_at_delta_0_with_intercept() {
        let tri = Triangle::try_from_upper_rows(&[
            &[100.0, 165.0, 200.0, 212.0, 218.0],
            &[120.0, 190.0, 238.0, 250.0],
            &[135.0, 222.0, 268.0],
            &[150.0, 240.0],
            &[160.0],
        ])
        .unwrap();

        let br = brosius::project(&tri.view(), 1.0).unwrap();
        let mu = analyze(&tri.view(), MurphyOptions::brosius_equivalent()).unwrap();

        for j in 0..4 {
            assert!((mu.links[j].intercept - br.intercept[j]).abs() < 1e-10);
            assert!((mu.links[j].slope - br.slope[j]).abs() < 1e-10);
        }
        for i in 0..5 {
            assert!((mu.ultimate[i] - br.ultimate[i]).abs() < 1e-9);
            assert!((mu.reserve[i] - br.reserve[i]).abs() < 1e-9);
        }
        assert!(mu.total_standard_error > 0.0);
    }

    #[test]
    fn delta_2_recovers_simple_average_ldf() {
        let tri =
            Triangle::try_from_upper_rows(&[&[100.0, 150.0, 180.0], &[200.0, 340.0], &[150.0]])
                .unwrap();

        let mu = analyze(&tri.view(), MurphyOptions::simple_average()).unwrap();
        let expected_f0 = f64::midpoint(150.0 / 100.0, 340.0 / 200.0);
        assert!((mu.links[0].slope - expected_f0).abs() < 1e-12);
    }
}
