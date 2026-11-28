//! Merz & Wüthrich (2010) Paid-Incurred Chain (PIC) Bayesian log-normal joint
//! paid/incurred reserving model.
//!
//! Implements:
//!
//! * M. Merz & M. V. Wüthrich, *"Paid-Incurred Chain Claims Reserving Method"*,
//!   *Insurance: Mathematics and Economics*, Vol. 46, No. 3 (2010), pp. 568–579.
//!
//! ## Model Structure
//!
//! Given aligned `n × n` strictly positive cumulative paid (`P_{i,j}`) and
//! incurred (`I_{i,j}`) triangles (`J = n - 1`), define log-development
//! increments:
//!
//! * Incurred: `zeta_{i,0} = ln I_{i,0}`, `zeta_{i,j} = ln(I_{i,j} / I_{i,j-1})`
//!   for `j = 1..J`, with `zeta_{i,j} | Theta ~ N(Phi_j, sigma_j^2)`.
//! * Paid: `xi_{i,j} = ln(P_{i,j} / P_{i,j-1})` for `j = 1..J`, with
//!   `xi_{i,j} | Theta ~ N(Psi_j, tau_j^2)` (plus an optional closing bridge
//!   `xi_{i,close} = ln(I_{i,J} / P_{i,J}) ~ N(Psi_close, tau_close^2)` when
//!   ultimate paid has not yet converged to incurred at age `J`).
//!
//! Coupling paid and incurred at ultimate (`P_{i, ult} = I_{i, J}`) implies
//! that the observed log incurred-to-paid ratio along the latest diagonal
//! `Z_i = ln(I_{i, d_i} / P_{i, d_i})` satisfies:
//!
//! `Z_i = sum_{k=d_i+1}^J (xi_{i,k} - zeta_{i,k}) + xi_{i,close}`
//!
//! yielding an exact conjugate multivariate Gaussian posterior
//! `Theta | D_I ~ N(Theta_post, V_Theta)` and a closed-form log-normal
//! predictive distribution for `(P_{1,ult}, ..., P_{J,ult}) | D_I`:
//!
//! * Conditional precision weight `w_i = v_{P,i} / (v_{I,i} + v_{P,i})` blends
//!   the incurred and paid chains with strictly lower process variance
//!   `s_i^2 = (v_{I,i} * v_{P,i}) / (v_{I,i} + v_{P,i})`.
//! * Posterior mean: `E[P_{i,ult} | D_I] = exp(m_i + 0.5 * t_i^2)`.
//! * Exact analytical MSEP (per-origin and aggregate with cross-origin
//!   log-normal covariance `E[P_i] E[P_k] (exp(t_{ik}) - 1)`).

use crate::linalg::{invert_spd, quad_form};
use crate::types::TriangleView;
use crate::validate::{ValidationError, validate_positive_finite_slice, validate_triangle};

/// Optional informative diagonal Gaussian prior on `Theta = (Phi, Psi)`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PicPrior {
    /// Prior mean for incurred log-link parameters `Phi_0..Phi_{n-1}` (length `n`).
    pub phi_mean: Vec<f64>,
    /// Prior variances for `Phi_0..Phi_{n-1}` (length `n`, strictly positive).
    pub phi_var: Vec<f64>,
    /// Prior mean for paid log-link parameters `Psi_1..Psi_{n-1}` (length `n - 1`).
    pub psi_mean: Vec<f64>,
    /// Prior variances for `Psi_1..Psi_{n-1}` (length `n - 1`, strictly positive).
    pub psi_var: Vec<f64>,
}

/// Configuration options for [`analyze`].
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct PicOptions {
    /// Optional explicit incurred log-increment variances `sigma_0^2..sigma_{n-1}^2`
    /// (length `n`). Estimated empirically from the incurred triangle when `None`.
    pub incurred_variances: Option<Vec<f64>>,
    /// Optional explicit paid log-increment variances `tau_1^2..tau_{n-1}^2`
    /// (length `n - 1`). Estimated empirically from the paid triangle when `None`.
    pub paid_variances: Option<Vec<f64>>,
    /// Optional informative Gaussian prior on `(Phi, Psi)`. When `None`, uses
    /// Merz & Wüthrich (2010) non-informative (diffuse) conjugate posterior.
    pub prior: Option<PicPrior>,
    /// Floor on estimated log-increment variances to avoid degenerate zero
    /// variance on noiseless columns (default `1e-10`).
    pub variance_floor: Option<f64>,
}

impl PicOptions {
    /// Supply explicit incurred log-increment variances `sigma_0^2..sigma_{n-1}^2`.
    pub fn with_incurred_variances(mut self, variances: impl Into<Vec<f64>>) -> Self {
        self.incurred_variances = Some(variances.into());
        self
    }

    /// Supply explicit paid log-increment variances `tau_1^2..tau_{n-1}^2`.
    pub fn with_paid_variances(mut self, variances: impl Into<Vec<f64>>) -> Self {
        self.paid_variances = Some(variances.into());
        self
    }

    /// Supply an informative Gaussian conjugate prior on `(Phi, Psi)`.
    pub fn with_prior(mut self, prior: PicPrior) -> Self {
        self.prior = Some(prior);
        self
    }

    /// Set the floor on estimated log-increment variances.
    pub const fn with_variance_floor(mut self, floor: f64) -> Self {
        self.variance_floor = Some(floor);
        self
    }
}

/// Complete output of the Merz & Wüthrich (2010) Paid-Incurred Chain model.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PaidIncurredChainResult {
    /// Posterior means for incurred log-increments `Phi_0..Phi_{n-1}` (length `n`).
    pub posterior_phi: Vec<f64>,
    /// Posterior means for paid log-increments `Psi_1..Psi_{n-1}` (length `n - 1`).
    pub posterior_psi: Vec<f64>,
    /// Posterior closing log-ratio `ln(I_{0,J} / P_{0,J})` (`0.0` when closed at age `J`).
    pub posterior_closing_psi: f64,
    /// Incurred log-increment variances `sigma_0^2..sigma_{n-1}^2` (length `n`).
    pub incurred_variances: Vec<f64>,
    /// Paid log-increment variances `tau_1^2..tau_{n-1}^2` (length `n - 1`).
    pub paid_variances: Vec<f64>,
    /// Closing paid-to-incurred bridge variance `tau_close^2`.
    pub closing_variance: f64,
    /// Per-origin precision weight `w_i in [0, 1]` placed on the incurred chain
    /// (`1 - w_i` on the paid chain).
    pub incurred_weights: Vec<f64>,
    /// Posterior predictive log-ultimate means `m_i = E[ln P_{i,ult} | D_I]`.
    pub predictive_log_mean: Vec<f64>,
    /// Posterior predictive log-ultimate variances `t_i^2 = Var(ln P_{i,ult} | D_I)`.
    pub predictive_log_variance: Vec<f64>,
    /// Exact Bayesian posterior expected ultimates `E[P_{i,ult} | D_I]` (length `n`).
    pub ultimates: Vec<f64>,
    /// Paid-Incurred Chain reserves `E[P_{i,ult} | D_I] - P_{i, d_i}` (length `n`).
    pub reserves: Vec<f64>,
    /// Analytical MSEP per origin `Var(P_{i,ult} | D_I)` (length `n`).
    pub msep: Vec<f64>,
    /// Analytical prediction standard error `sqrt(msep[i])` per origin (length `n`).
    pub standard_error: Vec<f64>,
    /// Coefficient of variation `standard_error[i] / reserves[i]` per origin.
    pub cv: Vec<f64>,
    /// Total Paid-Incurred Chain ultimate across origins.
    pub total_ultimate: f64,
    /// Total Paid-Incurred Chain reserve across origins.
    pub total_reserve: f64,
    /// Analytical aggregate MSEP including cross-origin log-normal covariances.
    pub total_msep: f64,
    /// Analytical aggregate prediction standard error `sqrt(total_msep)`.
    pub total_standard_error: f64,
    /// Aggregate coefficient of variation (`total_standard_error / total_reserve`).
    pub total_cv: f64,
}

/// Fit the Merz & Wüthrich (2010) Paid-Incurred Chain model to aligned square
/// cumulative `paid` and `incurred` triangles.
pub fn analyze(
    paid: &TriangleView<'_>,
    incurred: &TriangleView<'_>,
    options: &PicOptions,
) -> Result<PaidIncurredChainResult, ValidationError> {
    validate_triangle(paid)?;
    validate_triangle(incurred)?;

    let n = paid.n_origins;
    if n < 3 || incurred.n_origins != n {
        return Err(ValidationError::ShapeMismatch);
    }

    // Verify strict positivity of all observed cells (required by log-normal model).
    for i in 0..n {
        for j in 0..(n - i) {
            if paid.get(i, j) <= 0.0 || incurred.get(i, j) <= 0.0 {
                return Err(ValidationError::InvalidParameter(
                    "Paid-Incurred Chain requires strictly positive observed paid and incurred cells",
                ));
            }
        }
    }

    let j_max = n - 1;
    let var_floor = match options.variance_floor {
        Some(v) => {
            if !v.is_finite() || v <= 0.0 {
                return Err(ValidationError::InvalidParameter(
                    "variance_floor must be finite and strictly positive",
                ));
            }
            v.max(1e-14)
        }
        None => 1e-10,
    };

    // Compute observed incurred log-increments zeta_{i, j} and paid log-increments xi_{i, j}.
    let mut zeta = vec![f64::NAN; n * n];
    let mut xi = vec![f64::NAN; n * n];
    for i in 0..n {
        let d_i = j_max - i;
        zeta[i * n] = incurred.get(i, 0).ln();
        for j in 1..=d_i {
            zeta[i * n + j] = (incurred.get(i, j) / incurred.get(i, j - 1)).ln();
            xi[i * n + j] = (paid.get(i, j) / paid.get(i, j - 1)).ln();
        }
    }

    // Estimate or validate incurred variances sigma_0^2..sigma_J^2 (length n)
    // and paid variances tau_1^2..tau_J^2 (length n - 1).
    let sigma_sq = match &options.incurred_variances {
        Some(v) => {
            if v.len() != n {
                return Err(ValidationError::ShapeMismatch);
            }
            validate_positive_finite_slice(v, "incurred_variances")?;
            v.clone()
        }
        None => estimate_column_variances(&zeta, n, 0, var_floor),
    };
    let tau_sq = match &options.paid_variances {
        Some(v) => {
            if v.len() != n - 1 {
                return Err(ValidationError::ShapeMismatch);
            }
            validate_positive_finite_slice(v, "paid_variances")?;
            v.clone()
        }
        None => estimate_column_variances(&xi, n, 1, var_floor),
    };

    // Determine whether origin 0 still has an open case reserve at age J (I_{0,J} != P_{0,J}).
    let closing_obs = (incurred.get(0, j_max) / paid.get(0, j_max)).ln();
    let has_closing_bridge = closing_obs.abs() > 1e-12;
    let tau_close_sq = if has_closing_bridge {
        extrapolate_last_variance(&tau_sq, var_floor)
    } else {
        0.0
    };

    // Parameter vector Theta layout:
    //   0..n               -> Phi_0 .. Phi_J          (length n)
    //   n..(2*n - 1)       -> Psi_1 .. Psi_J          (length n - 1)
    //   2*n - 1 (optional) -> Psi_close               (if has_closing_bridge)
    let p_dim = 2 * n - 1 + usize::from(has_closing_bridge);
    let psi_offset = n; // Psi_j (j >= 1) lives at index psi_offset + (j - 1)
    let close_idx = 2 * n - 1;

    // Assemble precision matrix Q = Sigma_0^{-1} + X^T D^{-1} X and info vector h = Sigma_0^{-1} mu_0 + X^T D^{-1} y.
    let mut prec = vec![0.0; p_dim * p_dim];
    let mut rhs = vec![0.0; p_dim];

    if let Some(prior) = &options.prior {
        if prior.phi_mean.len() != n
            || prior.phi_var.len() != n
            || prior.psi_mean.len() != n - 1
            || prior.psi_var.len() != n - 1
        {
            return Err(ValidationError::ShapeMismatch);
        }
        validate_positive_finite_slice(&prior.phi_var, "prior.phi_var")?;
        validate_positive_finite_slice(&prior.psi_var, "prior.psi_var")?;
        for j in 0..n {
            if !prior.phi_mean[j].is_finite() {
                return Err(ValidationError::NotFinite);
            }
            let inv_v = 1.0 / prior.phi_var[j];
            prec[j * p_dim + j] += inv_v;
            rhs[j] += inv_v * prior.phi_mean[j];
        }
        for k in 0..(n - 1) {
            if !prior.psi_mean[k].is_finite() {
                return Err(ValidationError::NotFinite);
            }
            let inv_v = 1.0 / prior.psi_var[k];
            let idx = psi_offset + k;
            prec[idx * p_dim + idx] += inv_v;
            rhs[idx] += inv_v * prior.psi_mean[k];
        }
    }

    // 1. Direct incurred log-increment observations zeta_{i, j} ~ N(Phi_j, sigma_j^2).
    for j in 0..n {
        let inv_v = 1.0 / sigma_sq[j];
        for i in 0..(n - j) {
            prec[j * p_dim + j] += inv_v;
            rhs[j] += inv_v * zeta[i * n + j];
        }
    }

    // 2. Direct paid log-increment observations xi_{i, j} ~ N(Psi_j, tau_j^2), j = 1..J.
    for j in 1..n {
        let inv_v = 1.0 / tau_sq[j - 1];
        let idx = psi_offset + (j - 1);
        for i in 0..(n - j) {
            prec[idx * p_dim + idx] += inv_v;
            rhs[idx] += inv_v * xi[i * n + j];
        }
    }

    // 3. Bridge observations Z_i = ln(I_{i, d_i} / P_{i, d_i}) for i = 0..J.
    // Z_i = sum_{k=d_i+1}^J (Psi_k - Phi_k) + [Psi_close if open] + eps_i,
    // Var(eps_i) = v_{I, i} + v_{P, i}.
    let mut v_inc = vec![0.0; n];
    let mut v_paid = vec![0.0; n];
    let mut active_entries: Vec<(usize, f64)> = Vec::with_capacity(p_dim);
    for i in 0..n {
        let d_i = j_max - i;
        let vi: f64 = sigma_sq[(d_i + 1)..n].iter().sum();
        let vp: f64 = tau_sq[d_i..(n - 1)].iter().sum::<f64>() + tau_close_sq;
        v_inc[i] = vi;
        v_paid[i] = vp;

        let total_v = vi + vp;
        if total_v <= 0.0 {
            continue;
        }
        let z_obs = (incurred.get(i, d_i) / paid.get(i, d_i)).ln();
        active_entries.clear();
        for k in (d_i + 1)..n {
            active_entries.push((k, -1.0)); // -Phi_k
            active_entries.push((psi_offset + (k - 1), 1.0)); // +Psi_k
        }
        if has_closing_bridge {
            active_entries.push((close_idx, 1.0));
        }
        let inv_v = 1.0 / total_v;
        for &(a, xa) in &active_entries {
            rhs[a] += inv_v * xa * z_obs;
            let row_a = a * p_dim;
            let scale_a = inv_v * xa;
            for &(b, xb) in &active_entries {
                prec[row_a + b] += scale_a * xb;
            }
        }
    }

    // Posterior covariance V_Theta = prec^{-1} and posterior mean Theta_post = V_Theta * rhs.
    let v_theta = invert_spd(&prec, p_dim)?;
    let mut theta_post = vec![0.0; p_dim];
    for a in 0..p_dim {
        let mut acc = 0.0;
        for b in 0..p_dim {
            acc += v_theta[a * p_dim + b] * rhs[b];
        }
        theta_post[a] = acc;
    }

    // Build predictive distribution of ln(P_{i, ult}) | D_I for every origin i = 0..n-1.
    let mut incurred_weights = vec![0.0; n];
    let mut pred_log_mean = vec![0.0; n];
    let mut pred_log_var = vec![0.0; n];
    let mut c_vecs: Vec<Vec<f64>> = vec![vec![0.0; p_dim]; n];

    for i in 0..n {
        let d_i = j_max - i;
        let vi = v_inc[i];
        let vp = v_paid[i];
        if vi == 0.0 {
            // Origin 0: I_{0, J} is already observed at age J -> exact ultimate I_{0, J}.
            incurred_weights[i] = 1.0;
            pred_log_mean[i] = incurred.get(i, d_i).ln();
            pred_log_var[i] = 0.0;
            continue;
        }
        let w_i = vp / (vi + vp);
        let s_i_sq = (vi * vp) / (vi + vp);
        incurred_weights[i] = w_i;

        for k in (d_i + 1)..n {
            c_vecs[i][k] = w_i;
            c_vecs[i][psi_offset + (k - 1)] = 1.0 - w_i;
        }
        if has_closing_bridge {
            c_vecs[i][close_idx] = 1.0 - w_i;
        }

        let mut linear_mean = 0.0;
        for a in 0..p_dim {
            linear_mean += c_vecs[i][a] * theta_post[a];
        }
        pred_log_mean[i] =
            w_i * incurred.get(i, d_i).ln() + (1.0 - w_i) * paid.get(i, d_i).ln() + linear_mean;
        let param_var = quad_form(&c_vecs[i], &v_theta, p_dim).max(0.0);
        pred_log_var[i] = s_i_sq + param_var;
    }

    let mut ultimates = vec![0.0; n];
    let mut reserves = vec![0.0; n];
    let mut msep = vec![0.0; n];
    let mut standard_error = vec![0.0; n];
    let mut cv = vec![0.0; n];

    for i in 0..n {
        let d_i = j_max - i;
        let ult = (pred_log_mean[i] + 0.5 * pred_log_var[i]).exp();
        let res = ult - paid.get(i, d_i);
        let var_i = if pred_log_var[i] > 0.0 {
            ult * ult * pred_log_var[i].exp_m1()
        } else {
            0.0
        };
        if !ult.is_finite() || !var_i.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        ultimates[i] = ult;
        reserves[i] = res;
        msep[i] = var_i.max(0.0);
        standard_error[i] = var_i.max(0.0).sqrt();
        cv[i] = if res > 0.0 {
            standard_error[i] / res
        } else {
            0.0
        };
    }

    // Aggregate MSEP with exact log-normal cross-origin covariances.
    // Precompute V_Theta * c_k for each origin so pairwise covariances run in O(n^2 * p_dim).
    let mut vc_vecs: Vec<Vec<f64>> = Vec::with_capacity(n);
    for k in 0..n {
        if pred_log_var[k] == 0.0 {
            vc_vecs.push(Vec::new());
            continue;
        }
        let mut vc = vec![0.0; p_dim];
        for a in 0..p_dim {
            let mut row_acc = 0.0;
            for b in 0..p_dim {
                row_acc += v_theta[a * p_dim + b] * c_vecs[k][b];
            }
            vc[a] = row_acc;
        }
        vc_vecs.push(vc);
    }

    let mut total_msep: f64 = msep.iter().sum();
    for i in 0..n {
        if pred_log_var[i] == 0.0 {
            continue;
        }
        for k in (i + 1)..n {
            if pred_log_var[k] == 0.0 {
                continue;
            }
            let mut cov_log = 0.0;
            let vck = &vc_vecs[k];
            for a in 0..p_dim {
                cov_log += c_vecs[i][a] * vck[a];
            }
            total_msep += 2.0 * ultimates[i] * ultimates[k] * cov_log.exp_m1();
        }
    }

    let total_ultimate: f64 = ultimates.iter().sum();
    let total_reserve: f64 = reserves.iter().sum();
    if !total_ultimate.is_finite() || !total_reserve.is_finite() || !total_msep.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    let total_standard_error = total_msep.max(0.0).sqrt();
    let total_cv = if total_reserve > 0.0 {
        total_standard_error / total_reserve
    } else {
        0.0
    };

    Ok(PaidIncurredChainResult {
        posterior_phi: theta_post[0..n].to_vec(),
        posterior_psi: theta_post[psi_offset..(psi_offset + n - 1)].to_vec(),
        posterior_closing_psi: if has_closing_bridge {
            theta_post[close_idx]
        } else {
            0.0
        },
        incurred_variances: sigma_sq,
        paid_variances: tau_sq,
        closing_variance: tau_close_sq,
        incurred_weights,
        predictive_log_mean: pred_log_mean,
        predictive_log_variance: pred_log_var,
        ultimates,
        reserves,
        msep,
        standard_error,
        cv,
        total_ultimate,
        total_reserve,
        total_msep: total_msep.max(0.0),
        total_standard_error,
        total_cv,
    })
}

fn estimate_column_variances(mat: &[f64], n: usize, start_col: usize, floor: f64) -> Vec<f64> {
    let count = n - start_col;
    let mut vars = vec![f64::NAN; count];
    for (idx, j) in (start_col..n).enumerate() {
        let m_j = n - j;
        if m_j >= 2 {
            let mut sum = 0.0;
            for i in 0..m_j {
                sum += mat[i * n + j];
            }
            let mean = sum / m_j as f64;
            let mut ss = 0.0;
            for i in 0..m_j {
                let d = mat[i * n + j] - mean;
                ss += d * d;
            }
            vars[idx] = (ss / (m_j as f64 - 1.0)).max(floor);
        }
    }
    // Extrapolate the final single-observation column via Mack's min-rule.
    if let Some(last) = vars.last_mut()
        && !last.is_finite()
    {
        let known = &vars[..count - 1];
        let ext = extrapolate_last_variance(known, floor);
        *vars.last_mut().expect("non-empty") = ext;
    }
    vars
}

fn extrapolate_last_variance(known: &[f64], floor: f64) -> f64 {
    let m = known.len();
    if m >= 2 {
        let a = known[m - 1];
        let b = known[m - 2];
        if a > 0.0 && b > 0.0 {
            return (a * a / b).min(a).min(b).max(floor);
        }
    }
    if m >= 1 {
        return known[m - 1].max(floor);
    }
    floor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Triangle;

    const NAN: f64 = f64::NAN;

    #[test]
    fn pic_recovers_exact_ultimates_on_consistent_noiseless_triangles() {
        // Exact multiplicative paid & incurred chains closing at age 4 (P_{i,4} == I_{i,4}).
        let n = 5usize;
        let ult = [1000.0, 1100.0, 1200.0, 1300.0, 1400.0];
        let paid_pct = [0.25, 0.55, 0.80, 0.94, 1.00];
        let inc_pct = [0.65, 0.85, 0.95, 0.99, 1.00];

        let mut p_data = vec![NAN; n * n];
        let mut i_data = vec![NAN; n * n];
        for i in 0..n {
            for j in 0..(n - i) {
                p_data[i * n + j] = ult[i] * paid_pct[j];
                i_data[i * n + j] = ult[i] * inc_pct[j];
            }
        }
        let paid = Triangle::new(n, n, p_data);
        let inc = Triangle::new(n, n, i_data);

        let res = analyze(&paid.view(), &inc.view(), &PicOptions::default()).expect("pic");
        for i in 0..n {
            assert!(
                (res.ultimates[i] - ult[i]).abs() < 1e-3,
                "origin {i}: got {} expected {}",
                res.ultimates[i],
                ult[i]
            );
        }
        assert!(res.posterior_closing_psi.abs() < 1e-12);
    }

    #[test]
    fn pic_process_variance_strictly_below_individual_chains() {
        // When both chains have positive variance, precision weighting yields
        // incurred_weights in (0, 1) and finite positive MSEP.
        #[rustfmt::skip]
        let paid_data = vec![
            100.0, 210.0, 285.0, 325.0, 340.0,
            115.0, 235.0, 315.0, 358.0,   NAN,
            125.0, 260.0, 348.0,   NAN,   NAN,
            135.0, 272.0,   NAN,   NAN,   NAN,
            150.0,   NAN,   NAN,   NAN,   NAN,
        ];
        #[rustfmt::skip]
        let inc_data = vec![
            220.0, 290.0, 322.0, 338.0, 340.0,
            245.0, 325.0, 355.0, 372.0,   NAN,
            270.0, 350.0, 392.0,   NAN,   NAN,
            290.0, 378.0,   NAN,   NAN,   NAN,
            315.0,   NAN,   NAN,   NAN,   NAN,
        ];
        let paid = Triangle::new(5, 5, paid_data);
        let inc = Triangle::new(5, 5, inc_data);
        let res = analyze(&paid.view(), &inc.view(), &PicOptions::default()).expect("pic");

        for i in 1..5 {
            assert!(res.incurred_weights[i] > 0.0 && res.incurred_weights[i] < 1.0);
            assert!(res.standard_error[i] > 0.0);
        }
        assert!(res.total_standard_error > 0.0);
        assert!(res.total_reserve > 0.0);
    }
}
