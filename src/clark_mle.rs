//! Clark (2003) LDF & Cape Cod parametric growth-curve MLE and stochastic
//! reserving kernel.
//!
//! Implements:
//!
//! * D. R. Clark, *"LDF Curve-Fitting and Stochastic Reserving: A Maximum
//!   Likelihood Approach"*, *CAS Fall Forum* (2003), pp. 41–92.
//!
//! ## Growth Curves
//!
//! Incremental loss emergence across development ages `0 < x_0 < x_1 < ... < x_{n-1}`
//! is modeled via a two-parameter parametric cumulative growth curve `G(x; omega, theta)`
//! (`omega > 0`, `theta > 0`, `G(0) = 0`, `lim_{x -> inf} G(x) = 1`):
//!
//! * **Loglogistic**: `G(x) = x^omega / (x^omega + theta^omega)`.
//! * **Weibull**: `G(x) = 1 - exp(-(x / theta)^omega)`.
//!
//! Incremental emergence fractions are `dG_j = G(x_j) - G(x_{j-1})` (with
//! `G(x_{-1}) = 0`).
//!
//! ## Estimation & Variance Decomposition
//!
//! * **Clark LDF (`fit_clark_ldf`)**: `mu_{i,j} = U_i · dG_j` with `p = n + 2`
//!   parameters `(U_0, ..., U_{n-1}, omega, theta)`. Exact profile MLE
//!   `U_hat_i(omega, theta) = C_{i, d_i} / G(x_{d_i})` reduces optimization to
//!   2D damped Newton–Raphson over `(ln omega, ln theta)`.
//! * **Clark Cape Cod (`fit_clark_cape_cod`)**: `mu_{i,j} = P_i · ELR · dG_j`
//!   with `p = 3` parameters `(ELR, omega, theta)` and profile MLE
//!   `ELR_hat(omega, theta) = sum_i C_{i, d_i} / sum_i (P_i · G(x_{d_i}))`.
//! * **Over-Dispersed Poisson scale**:
//!   `sigma^2 = (1 / (N - p)) * sum_{(i,j)} (c_{i,j} - mu_{i,j})^2 / mu_{i,j}`.
//! * **Cramér–Rao parameter variance**: computed from the inverse Fisher
//!   information matrix `Cov(psi_hat) = sigma^2 * I(psi_hat)^{-1}` via delta
//!   method on `R_i(psi)`, combined with process variance `sigma^2 * R_i`:
//!   `SE_total = sqrt(ProcessVar + ParamVar)`.

use crate::linalg::{invert_spd, quad_form};
use crate::triangle::{cumulative_to_incremental, latest_diagonal};
use crate::types::TriangleView;
use crate::validate::{
    MonotonicityPolicy, ValidationError, validate_monotone_cumulative, validate_premium,
    validate_triangle,
};

/// Parametric loss-emergence growth curve `G(x; omega, theta)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GrowthCurve {
    /// Loglogistic: `G(x) = x^omega / (x^omega + theta^omega)` (heavier tail).
    Loglogistic,
    /// Weibull: `G(x) = 1 - exp(-(x / theta)^omega)` (lighter tail).
    Weibull,
}

impl GrowthCurve {
    /// Evaluate `G(x; omega, theta)` for `x >= 0`, `omega > 0`, `theta > 0`.
    #[inline]
    pub fn eval(self, x: f64, omega: f64, theta: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let ratio = x / theta;
        match self {
            Self::Loglogistic => {
                // Stable computation via log-odds: 1 / (1 + exp(-omega * ln(x / theta))).
                let z = omega * ratio.ln();
                if z >= 0.0 {
                    1.0 / (1.0 + (-z).exp())
                } else {
                    let ez = z.exp();
                    ez / (1.0 + ez)
                }
            }
            Self::Weibull => {
                let v = (omega * ratio.ln()).exp();
                -(-v).exp_m1()
            }
        }
    }

    /// Evaluate `(G(x), dG/domega, dG/dtheta)` analytically.
    #[inline]
    pub fn eval_with_grad(self, x: f64, omega: f64, theta: f64) -> (f64, f64, f64) {
        if x <= 0.0 {
            return (0.0, 0.0, 0.0);
        }
        let ln_r = (x / theta).ln();
        match self {
            Self::Loglogistic => {
                let g = self.eval(x, omega, theta);
                let g1mg = g * (1.0 - g);
                let dg_domega = g1mg * ln_r;
                let dg_dtheta = -g1mg * (omega / theta);
                (g, dg_domega, dg_dtheta)
            }
            Self::Weibull => {
                let exponent = omega * ln_r;
                if exponent > 7.0 {
                    // exp(-exp(7.0)) < 1e-475 underflows to 0 in f64; avoids 0.0 * inf = NaN.
                    return (1.0, 0.0, 0.0);
                }
                let v = exponent.exp();
                let surv = (-v).exp();
                let g = -(-v).exp_m1();
                let sv = surv * v;
                let dg_domega = sv * ln_r;
                let dg_dtheta = -sv * (omega / theta);
                (g, dg_domega, dg_dtheta)
            }
        }
    }
}

/// Configuration options for Clark (2003) MLE reserving.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClarkOptions {
    /// Parametric growth curve family (`Loglogistic` or `Weibull`).
    pub curve: GrowthCurve,
    /// Optional explicit development ages `0 < x_0 < x_1 < ... < x_{n-1}`.
    /// When `None`, defaults to Clark (2003) average-date-of-loss midpoint ages
    /// `[1 - origin_offset, 2 - origin_offset, ..., n - origin_offset]`.
    pub ages: Option<Vec<f64>>,
    /// Average accident date offset within period 1 (default `0.5`, giving
    /// midpoint ages `0.5, 1.5, ..., n - 0.5` years as in Clark 2003). Ignored
    /// when `ages` is explicitly provided.
    pub origin_offset: f64,
    /// Optional finite ultimate age truncation `max_age >= x_{n-1}`. When `None`,
    /// projects to `infinity` (`G(inf) = 1.0`, Clark 2003 default including tail).
    pub max_age: Option<f64>,
    /// Cumulative monotonicity validation policy.
    pub monotonicity_policy: MonotonicityPolicy,
}

impl Default for ClarkOptions {
    fn default() -> Self {
        Self::loglogistic()
    }
}

impl ClarkOptions {
    /// Standard Clark (2003) specification with midpoint ages (`0.5, 1.5, ...`)
    /// and projection to ultimate (`max_age = None`).
    pub fn new(curve: GrowthCurve) -> Self {
        Self {
            curve,
            ages: None,
            origin_offset: 0.5,
            max_age: None,
            monotonicity_policy: MonotonicityPolicy::Strict,
        }
    }

    /// Clark (2003) Loglogistic growth curve with default midpoint ages.
    pub fn loglogistic() -> Self {
        Self::new(GrowthCurve::Loglogistic)
    }

    /// Clark (2003) Weibull growth curve with default midpoint ages.
    pub fn weibull() -> Self {
        Self::new(GrowthCurve::Weibull)
    }

    /// Supply explicit development ages `0 < x_0 < ... < x_{n-1}`.
    pub fn with_ages(mut self, ages: impl Into<Vec<f64>>) -> Self {
        self.ages = Some(ages.into());
        self
    }

    /// Set the average accident-date offset within period 1 (default `0.5`).
    pub const fn with_origin_offset(mut self, origin_offset: f64) -> Self {
        self.origin_offset = origin_offset;
        self
    }

    /// Truncate ultimate projection at a finite maximum development age.
    pub const fn with_max_age(mut self, max_age: f64) -> Self {
        self.max_age = Some(max_age);
        self
    }

    /// Set the triangle monotonicity policy.
    pub const fn with_policy(mut self, policy: MonotonicityPolicy) -> Self {
        self.monotonicity_policy = policy;
        self
    }
}

/// Full output from Clark (2003) LDF or Cape Cod parametric MLE reserving.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ClarkMleResult {
    /// Growth curve family fitted.
    pub curve: GrowthCurve,
    /// Development ages `x_0..x_{n-1}` used in fitting.
    pub ages: Vec<f64>,
    /// Estimated shape parameter `omega`.
    pub omega: f64,
    /// Estimated scale parameter `theta`.
    pub theta: f64,
    /// Estimated Cape Cod ELR (`Some(elr)` for Cape Cod, `None` for LDF).
    pub elr: Option<f64>,
    /// Fitted growth percentages `G(x_j)` at each development age (length `n`).
    pub growth_curve: Vec<f64>,
    /// Implied age-to-ultimate development factors `G_max / G(x_j)` (length `n`).
    pub implied_cdfs: Vec<f64>,
    /// Estimated ODP scale parameter `sigma^2 = sum((c - mu)^2 / mu) / (N - p)`.
    pub sigma_sq: f64,
    /// Number of observed incremental cells `N`.
    pub n_observed: usize,
    /// Number of estimated parameters `p` (`n + 2` for LDF, `3` for Cape Cod).
    pub n_params: usize,
    /// Residual degrees of freedom `N - p`.
    pub degrees_of_freedom: usize,
    /// Maximum ODP quasi-log-likelihood achieved (`sum(c ln mu - mu)`).
    pub log_likelihood: f64,
    /// Maximum absolute gradient component at optimum (convergence diagnostic).
    pub max_grad_norm: f64,
    /// Latest observed cumulative losses by origin.
    pub latest: Vec<f64>,
    /// Estimated ultimate losses by origin (`latest + reserve` for LDF,
    /// `latest + P_i * ELR * (G_max - G(x_{d_i}))` for Cape Cod).
    pub ultimates: Vec<f64>,
    /// Point-estimate future reserves by origin.
    pub reserves: Vec<f64>,
    /// ODP process standard error by origin `sqrt(sigma^2 * R_i)`.
    pub process_se: Vec<f64>,
    /// Cramér–Rao parameter estimation standard error by origin.
    pub parameter_se: Vec<f64>,
    /// Total prediction standard error by origin `sqrt(process_se^2 + parameter_se^2)`.
    pub total_se: Vec<f64>,
    /// Total portfolio reserve across origins.
    pub total_reserve: f64,
    /// Total portfolio process standard error `sqrt(sigma^2 * total_reserve)`.
    pub total_process_se: f64,
    /// Total portfolio parameter estimation standard error.
    pub total_parameter_se: f64,
    /// Total portfolio prediction standard error.
    pub total_se_aggregate: f64,
    /// Total portfolio coefficient of variation (`total_se_aggregate / total_reserve`).
    pub total_cv: f64,
    /// Parameter covariance matrix `sigma^2 * I(psi_hat)^{-1}` (`p × p` row-major).
    /// Parameter ordering is `[U_0..U_{n-1}, omega, theta]` for LDF and
    /// `[ELR, omega, theta]` for Cape Cod.
    pub parameter_covariance: Vec<f64>,
}

/// Fit Clark (2003) parametric LDF model (`mu_{i,j} = U_i * dG_j`) to a
/// cumulative loss triangle.
pub fn fit_clark_ldf(
    triangle: &TriangleView<'_>,
    options: &ClarkOptions,
) -> Result<ClarkMleResult, ValidationError> {
    let prep = prepare_inputs(triangle, options)?;
    let n = prep.n;
    let n_obs = n * (n + 1) / 2;
    let n_params = n + 2;
    if n_obs <= n_params {
        return Err(ValidationError::InvalidParameter(
            "Clark LDF requires at least a 4x4 triangle for positive degrees of freedom",
        ));
    }
    if prep.latest.iter().any(|&v| v <= 0.0) {
        return Err(ValidationError::InvalidParameter(
            "Clark LDF requires strictly positive latest diagonal losses for every origin",
        ));
    }

    let (omega, theta, log_lik, max_grad) = optimize_growth_params(&prep, None, options.curve)?;

    let (g_vals, dg_dw, dg_dt, dg_inc, d_dg_dw, d_dg_dt) =
        evaluate_curve_arrays(&prep.ages, omega, theta, options.curve);
    let (g_max, dg_max_dw, dg_max_dt) = match options.max_age {
        Some(mx) => options.curve.eval_with_grad(mx, omega, theta),
        None => (1.0, 0.0, 0.0),
    };

    let mut u_hat = vec![0.0; n];
    for i in 0..n {
        u_hat[i] = prep.latest[i] / g_vals[n - 1 - i];
    }

    // Compute ODP scale sigma^2 and unscaled Fisher information matrix J (p × p).
    // Ordering: [U_0..U_{n-1}, omega, theta].
    let mut pearson_ss = 0.0;
    let mut info = vec![0.0; n_params * n_params];
    for i in 0..n {
        let ui = u_hat[i];
        for j in 0..(n - i) {
            let obs = prep.inc[i * n + j];
            let dg = dg_inc[j];
            let mu = ui * dg;
            if !(mu > 0.0) || !mu.is_finite() {
                return Err(ValidationError::InvalidParameter(
                    "Clark MLE produced non-positive fitted increment",
                ));
            }
            let diff = obs - mu;
            pearson_ss += (diff * diff) / mu;

            let d_u = dg;
            let d_w = ui * d_dg_dw[j];
            let d_t = ui * d_dg_dt[j];
            let inv_mu = 1.0 / mu;

            let idx_w = n;
            let idx_t = n + 1;
            info[i * n_params + i] += inv_mu * d_u * d_u;
            info[i * n_params + idx_w] += inv_mu * d_u * d_w;
            info[idx_w * n_params + i] += inv_mu * d_u * d_w;
            info[i * n_params + idx_t] += inv_mu * d_u * d_t;
            info[idx_t * n_params + i] += inv_mu * d_u * d_t;
            info[idx_w * n_params + idx_w] += inv_mu * d_w * d_w;
            info[idx_w * n_params + idx_t] += inv_mu * d_w * d_t;
            info[idx_t * n_params + idx_w] += inv_mu * d_w * d_t;
            info[idx_t * n_params + idx_t] += inv_mu * d_t * d_t;
        }
    }

    let dof = n_obs - n_params;
    let sigma_sq = pearson_ss / dof as f64;
    let j_inv = invert_spd(&info, n_params)?;
    let cov: Vec<f64> = j_inv.iter().map(|&v| sigma_sq * v).collect();

    let mut reserves = vec![0.0; n];
    let mut ultimates = vec![0.0; n];
    let mut process_se = vec![0.0; n];
    let mut parameter_se = vec![0.0; n];
    let mut total_se = vec![0.0; n];
    let mut grad_total = vec![0.0; n_params];
    let mut grad_i = vec![0.0; n_params];

    for i in 0..n {
        let d_i = n - 1 - i;
        let rem_g = (g_max - g_vals[d_i]).max(0.0);
        let r_i = u_hat[i] * rem_g;
        reserves[i] = r_i;
        ultimates[i] = prep.latest[i] + r_i;

        grad_i.fill(0.0);
        grad_i[i] = rem_g;
        grad_i[n] = u_hat[i] * (dg_max_dw - dg_dw[d_i]);
        grad_i[n + 1] = u_hat[i] * (dg_max_dt - dg_dt[d_i]);
        for k in 0..n_params {
            grad_total[k] += grad_i[k];
        }

        let proc_var = (sigma_sq * r_i).max(0.0);
        let param_var = quad_form(&grad_i, &cov, n_params).max(0.0);
        process_se[i] = proc_var.sqrt();
        parameter_se[i] = param_var.sqrt();
        total_se[i] = (proc_var + param_var).sqrt();
    }

    let total_reserve: f64 = reserves.iter().sum();
    let total_proc_var = (sigma_sq * total_reserve).max(0.0);
    let total_param_var = quad_form(&grad_total, &cov, n_params).max(0.0);
    let total_se_aggregate = (total_proc_var + total_param_var).sqrt();
    let total_cv = if total_reserve > 0.0 {
        total_se_aggregate / total_reserve
    } else {
        0.0
    };
    let implied_cdfs = g_vals.iter().map(|&g| g_max / g).collect();

    Ok(ClarkMleResult {
        curve: options.curve,
        ages: prep.ages,
        omega,
        theta,
        elr: None,
        growth_curve: g_vals,
        implied_cdfs,
        sigma_sq,
        n_observed: n_obs,
        n_params,
        degrees_of_freedom: dof,
        log_likelihood: log_lik,
        max_grad_norm: max_grad,
        latest: prep.latest,
        ultimates,
        reserves,
        process_se,
        parameter_se,
        total_se,
        total_reserve,
        total_process_se: total_proc_var.sqrt(),
        total_parameter_se: total_param_var.sqrt(),
        total_se_aggregate,
        total_cv,
        parameter_covariance: cov,
    })
}

/// Fit Clark (2003) parametric Cape Cod model (`mu_{i,j} = P_i * ELR * dG_j`)
/// to a cumulative loss triangle and earned-premium vector `premium`.
pub fn fit_clark_cape_cod(
    triangle: &TriangleView<'_>,
    premium: &[f64],
    options: &ClarkOptions,
) -> Result<ClarkMleResult, ValidationError> {
    let prep = prepare_inputs(triangle, options)?;
    let n = prep.n;
    validate_premium(premium, n)?;
    let n_obs = n * (n + 1) / 2;
    let n_params = 3usize;
    if n_obs <= n_params {
        return Err(ValidationError::InvalidParameter(
            "Clark Cape Cod requires at least a 3x3 triangle for positive degrees of freedom",
        ));
    }

    let (omega, theta, log_lik, max_grad) =
        optimize_growth_params(&prep, Some(premium), options.curve)?;

    let (g_vals, dg_dw, dg_dt, dg_inc, d_dg_dw, d_dg_dt) =
        evaluate_curve_arrays(&prep.ages, omega, theta, options.curve);
    let (g_max, dg_max_dw, dg_max_dt) = match options.max_age {
        Some(mx) => options.curve.eval_with_grad(mx, omega, theta),
        None => (1.0, 0.0, 0.0),
    };

    let sum_latest: f64 = prep.latest.iter().sum();
    let sum_used_prem: f64 = (0..n).map(|i| premium[i] * g_vals[n - 1 - i]).sum();
    if !(sum_used_prem > 0.0) {
        return Err(ValidationError::DivideByZero("Clark Cape Cod used premium"));
    }
    let elr = sum_latest / sum_used_prem;

    // 3×3 Fisher information for [ELR, omega, theta].
    let mut pearson_ss = 0.0;
    let mut info = vec![0.0; 9];
    for i in 0..n {
        let p_i = premium[i];
        for j in 0..(n - i) {
            let obs = prep.inc[i * n + j];
            let dg = dg_inc[j];
            let mu = p_i * elr * dg;
            if !(mu > 0.0) || !mu.is_finite() {
                return Err(ValidationError::InvalidParameter(
                    "Clark Cape Cod produced non-positive fitted increment",
                ));
            }
            let diff = obs - mu;
            pearson_ss += (diff * diff) / mu;

            let d = [p_i * dg, p_i * elr * d_dg_dw[j], p_i * elr * d_dg_dt[j]];
            let inv_mu = 1.0 / mu;
            for a in 0..3 {
                for b in 0..3 {
                    info[a * 3 + b] += inv_mu * d[a] * d[b];
                }
            }
        }
    }

    let dof = n_obs - n_params;
    let sigma_sq = pearson_ss / dof as f64;
    let j_inv = invert_spd(&info, 3)?;
    let cov: Vec<f64> = j_inv.iter().map(|&v| sigma_sq * v).collect();

    let mut reserves = vec![0.0; n];
    let mut ultimates = vec![0.0; n];
    let mut process_se = vec![0.0; n];
    let mut parameter_se = vec![0.0; n];
    let mut total_se = vec![0.0; n];
    let mut grad_total = [0.0; 3];

    for i in 0..n {
        let d_i = n - 1 - i;
        let rem_g = (g_max - g_vals[d_i]).max(0.0);
        let r_i = premium[i] * elr * rem_g;
        reserves[i] = r_i;
        ultimates[i] = prep.latest[i] + r_i;

        let grad_i = [
            premium[i] * rem_g,
            premium[i] * elr * (dg_max_dw - dg_dw[d_i]),
            premium[i] * elr * (dg_max_dt - dg_dt[d_i]),
        ];
        for k in 0..3 {
            grad_total[k] += grad_i[k];
        }

        let proc_var = (sigma_sq * r_i).max(0.0);
        let param_var = quad_form(&grad_i, &cov, 3).max(0.0);
        process_se[i] = proc_var.sqrt();
        parameter_se[i] = param_var.sqrt();
        total_se[i] = (proc_var + param_var).sqrt();
    }

    let total_reserve: f64 = reserves.iter().sum();
    let total_proc_var = (sigma_sq * total_reserve).max(0.0);
    let total_param_var = quad_form(&grad_total, &cov, 3).max(0.0);
    let total_se_aggregate = (total_proc_var + total_param_var).sqrt();
    let total_cv = if total_reserve > 0.0 {
        total_se_aggregate / total_reserve
    } else {
        0.0
    };
    let implied_cdfs = g_vals.iter().map(|&g| g_max / g).collect();

    Ok(ClarkMleResult {
        curve: options.curve,
        ages: prep.ages,
        omega,
        theta,
        elr: Some(elr),
        growth_curve: g_vals,
        implied_cdfs,
        sigma_sq,
        n_observed: n_obs,
        n_params,
        degrees_of_freedom: dof,
        log_likelihood: log_lik,
        max_grad_norm: max_grad,
        latest: prep.latest,
        ultimates,
        reserves,
        process_se,
        parameter_se,
        total_se,
        total_reserve,
        total_process_se: total_proc_var.sqrt(),
        total_parameter_se: total_param_var.sqrt(),
        total_se_aggregate,
        total_cv,
        parameter_covariance: cov,
    })
}

struct PreparedClarkInputs {
    n: usize,
    ages: Vec<f64>,
    ln_ages: Vec<f64>,
    inc: Vec<f64>,
    latest: Vec<f64>,
    sum_latest: f64,
    /// Column sums of observed increments: `col_inc_sum[j] = sum_{i=0}^{n-1-j} inc[i, j]`.
    col_inc_sum: Vec<f64>,
}

/// Reusable scratch buffers for zero-allocation Clark MLE profile evaluations.
struct ClarkWorkspace {
    g_vals: Vec<f64>,
    dg_inc: Vec<f64>,
    d_dg_dw: Vec<f64>,
    d_dg_dt: Vec<f64>,
    scale_prefix: Vec<f64>,
}

impl ClarkWorkspace {
    fn new(n: usize) -> Self {
        Self {
            g_vals: vec![0.0; n],
            dg_inc: vec![0.0; n],
            d_dg_dw: vec![0.0; n],
            d_dg_dt: vec![0.0; n],
            scale_prefix: vec![0.0; n + 1],
        }
    }
}

fn prepare_inputs(
    triangle: &TriangleView<'_>,
    options: &ClarkOptions,
) -> Result<PreparedClarkInputs, ValidationError> {
    validate_triangle(triangle)?;
    if options.monotonicity_policy == MonotonicityPolicy::Strict {
        validate_monotone_cumulative(triangle)?;
    }
    let n = triangle.n_origins;
    if n < 3 {
        return Err(ValidationError::InvalidParameter(
            "Clark MLE requires at least a 3x3 triangle",
        ));
    }
    let ages = if let Some(custom) = &options.ages {
        if custom.len() != n {
            return Err(ValidationError::ShapeMismatch);
        }
        for k in 0..n {
            if !custom[k].is_finite() || custom[k] <= 0.0 {
                return Err(ValidationError::InvalidParameter(
                    "Clark development ages must be strictly positive and finite",
                ));
            }
            if k > 0 && !(custom[k] > custom[k - 1]) {
                return Err(ValidationError::InvalidParameter(
                    "Clark development ages must be strictly increasing",
                ));
            }
        }
        custom.clone()
    } else {
        if !options.origin_offset.is_finite() || options.origin_offset >= 1.0 {
            return Err(ValidationError::InvalidParameter(
                "origin_offset must be finite and < 1.0",
            ));
        }
        (0..n)
            .map(|j| (j as f64 + 1.0) - options.origin_offset)
            .collect()
    };
    if let Some(mx) = options.max_age
        && (!mx.is_finite() || mx < *ages.last().expect("n >= 3"))
    {
        return Err(ValidationError::InvalidParameter(
            "max_age must be finite and >= largest observed development age",
        ));
    }

    let inc = cumulative_to_incremental(triangle);
    let latest = latest_diagonal(triangle);
    let ln_ages: Vec<f64> = ages.iter().map(|&a| a.ln()).collect();
    let sum_latest: f64 = latest.iter().sum();
    let mut col_inc_sum = vec![0.0; n];
    for i in 0..n {
        let row = &inc[i * n..i * n + (n - i)];
        for (j, &v) in row.iter().enumerate() {
            col_inc_sum[j] += v;
        }
    }
    Ok(PreparedClarkInputs {
        n,
        ages,
        ln_ages,
        inc,
        latest,
        sum_latest,
        col_inc_sum,
    })
}

/// Evaluate only `G(x_j)` and `dG_j` into `ws` using precomputed `ln_ages` (zero `ln()` calls).
#[inline]
fn eval_curve_values_into(
    ln_ages: &[f64],
    omega: f64,
    v: f64,
    curve: GrowthCurve,
    ws: &mut ClarkWorkspace,
) -> bool {
    let n = ln_ages.len();
    let mut prev_g = 0.0;
    for j in 0..n {
        let z = omega * (ln_ages[j] - v);
        let g = match curve {
            GrowthCurve::Loglogistic => {
                if z >= 0.0 {
                    1.0 / (1.0 + (-z).exp())
                } else {
                    let ez = z.exp();
                    ez / (1.0 + ez)
                }
            }
            GrowthCurve::Weibull => {
                if z > 7.0 {
                    1.0
                } else {
                    -(-(z.exp())).exp_m1()
                }
            }
        };
        let dg = g - prev_g;
        if !g.is_finite() || g <= 1e-15 || !dg.is_finite() || dg <= 1e-15 {
            return false;
        }
        ws.g_vals[j] = g;
        ws.dg_inc[j] = dg;
        prev_g = g;
    }
    true
}

/// Evaluate `G(x_j)`, `dG_j`, `d(dG_j)/domega`, and `d(dG_j)/dtheta` into `ws`
/// using precomputed `ln_ages` (zero `ln()` calls, zero heap allocations).
#[inline]
fn eval_curve_and_grad_into(
    ln_ages: &[f64],
    omega: f64,
    theta: f64,
    v: f64,
    curve: GrowthCurve,
    ws: &mut ClarkWorkspace,
) -> bool {
    let n = ln_ages.len();
    let omega_over_theta = omega / theta;
    let mut prev_g = 0.0;
    let mut prev_dw = 0.0;
    let mut prev_dt = 0.0;
    for j in 0..n {
        let ln_r = ln_ages[j] - v;
        let z = omega * ln_r;
        let (g, dw, dt) = match curve {
            GrowthCurve::Loglogistic => {
                let g = if z >= 0.0 {
                    1.0 / (1.0 + (-z).exp())
                } else {
                    let ez = z.exp();
                    ez / (1.0 + ez)
                };
                let g1mg = g * (1.0 - g);
                (g, g1mg * ln_r, -g1mg * omega_over_theta)
            }
            GrowthCurve::Weibull => {
                if z > 7.0 {
                    (1.0, 0.0, 0.0)
                } else {
                    let ev = z.exp();
                    let surv = (-ev).exp();
                    let g = -(-ev).exp_m1();
                    let sv = surv * ev;
                    (g, sv * ln_r, -sv * omega_over_theta)
                }
            }
        };
        let dg = g - prev_g;
        if !g.is_finite() || g <= 1e-15 || !dg.is_finite() || dg <= 1e-15 {
            return false;
        }
        ws.g_vals[j] = g;
        ws.dg_inc[j] = dg;
        ws.d_dg_dw[j] = dw - prev_dw;
        ws.d_dg_dt[j] = dt - prev_dt;
        prev_g = g;
        prev_dw = dw;
        prev_dt = dt;
    }
    true
}

/// Evaluate O(n) profile ODP log-likelihood only (for grid search and line search).
fn profile_ll_only(
    u: f64,
    v: f64,
    prep: &PreparedClarkInputs,
    premium: Option<&[f64]>,
    curve: GrowthCurve,
    ws: &mut ClarkWorkspace,
) -> Option<f64> {
    if !(-6.0..=4.0).contains(&u) || !(-12.0..=12.0).contains(&v) {
        return None;
    }
    let n = prep.n;
    let omega = u.exp();
    if !eval_curve_values_into(&prep.ln_ages, omega, v, curve, ws) {
        return None;
    }

    // Algebraic O(n) reduction of sum_{i,j} [c_{i,j} ln(U_i dG_j) - U_i dG_j]:
    //   = sum_i L_i ln(U_i) + sum_j S_j ln(dG_j) - sum_i U_i G(x_{n-1-i}).
    let mut ll = 0.0;
    match premium {
        None => {
            for i in 0..n {
                let li = prep.latest[i];
                let ui = li / ws.g_vals[n - 1 - i];
                if !(ui > 0.0) || !ui.is_finite() {
                    return None;
                }
                ll += li * ui.ln() - li;
            }
        }
        Some(prem) => {
            let mut sum_used = 0.0;
            for i in 0..n {
                sum_used += prem[i] * ws.g_vals[n - 1 - i];
            }
            if !(sum_used > 0.0) || !sum_used.is_finite() {
                return None;
            }
            let elr = prep.sum_latest / sum_used;
            if !(elr > 0.0) || !elr.is_finite() {
                return None;
            }
            for i in 0..n {
                let ui = prem[i] * elr;
                if !(ui > 0.0) || !ui.is_finite() {
                    return None;
                }
                ll += prep.latest[i] * ui.ln() - ui * ws.g_vals[n - 1 - i];
            }
        }
    }
    for j in 0..n {
        ll += prep.col_inc_sum[j] * ws.dg_inc[j].ln();
    }
    ll.is_finite().then_some(ll)
}

/// Evaluate O(n) profile ODP log-likelihood and its exact analytic gradient w.r.t.
/// `(u, v) = (ln omega, ln theta)` via the Envelope Theorem (zero allocations).
fn profile_ll_and_grad(
    u: f64,
    v: f64,
    prep: &PreparedClarkInputs,
    premium: Option<&[f64]>,
    curve: GrowthCurve,
    ws: &mut ClarkWorkspace,
) -> Option<(f64, [f64; 2])> {
    if !(-6.0..=4.0).contains(&u) || !(-12.0..=12.0).contains(&v) {
        return None;
    }
    let n = prep.n;
    let omega = u.exp();
    let theta = v.exp();
    if !eval_curve_and_grad_into(&prep.ln_ages, omega, theta, v, curve, ws) {
        return None;
    }

    let mut ll = 0.0;
    ws.scale_prefix[0] = 0.0;
    match premium {
        None => {
            let mut acc_u = 0.0;
            for i in 0..n {
                let li = prep.latest[i];
                let ui = li / ws.g_vals[n - 1 - i];
                if !(ui > 0.0) || !ui.is_finite() {
                    return None;
                }
                acc_u += ui;
                ws.scale_prefix[i + 1] = acc_u;
                ll += li * ui.ln() - li;
            }
        }
        Some(prem) => {
            let mut sum_used = 0.0;
            for i in 0..n {
                sum_used += prem[i] * ws.g_vals[n - 1 - i];
            }
            if !(sum_used > 0.0) || !sum_used.is_finite() {
                return None;
            }
            let elr = prep.sum_latest / sum_used;
            if !(elr > 0.0) || !elr.is_finite() {
                return None;
            }
            let mut acc_u = 0.0;
            for i in 0..n {
                let ui = prem[i] * elr;
                if !(ui > 0.0) || !ui.is_finite() {
                    return None;
                }
                acc_u += ui;
                ws.scale_prefix[i + 1] = acc_u;
                ll += prep.latest[i] * ui.ln() - ui * ws.g_vals[n - 1 - i];
            }
        }
    }

    let mut grad_omega = 0.0;
    let mut grad_theta = 0.0;
    for j in 0..n {
        let dg = ws.dg_inc[j];
        let col_sum = prep.col_inc_sum[j];
        ll += col_sum * dg.ln();
        let weight = col_sum / dg - ws.scale_prefix[n - j];
        grad_omega += weight * ws.d_dg_dw[j];
        grad_theta += weight * ws.d_dg_dt[j];
    }
    if !ll.is_finite() || !grad_omega.is_finite() || !grad_theta.is_finite() {
        return None;
    }
    // Chain rule: d/du = omega * d/domega, d/dv = theta * d/dtheta.
    Some((ll, [grad_omega * omega, grad_theta * theta]))
}

#[allow(clippy::type_complexity)]
fn evaluate_curve_arrays(
    ages: &[f64],
    omega: f64,
    theta: f64,
    curve: GrowthCurve,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = ages.len();
    let mut g_vals = vec![0.0; n];
    let mut dg_dw = vec![0.0; n];
    let mut dg_dt = vec![0.0; n];
    let mut dg_inc = vec![0.0; n];
    let mut d_dg_dw = vec![0.0; n];
    let mut d_dg_dt = vec![0.0; n];

    for j in 0..n {
        let (g, dw, dt) = curve.eval_with_grad(ages[j], omega, theta);
        g_vals[j] = g;
        dg_dw[j] = dw;
        dg_dt[j] = dt;
        if j == 0 {
            dg_inc[j] = g;
            d_dg_dw[j] = dw;
            d_dg_dt[j] = dt;
        } else {
            dg_inc[j] = g - g_vals[j - 1];
            d_dg_dw[j] = dw - dg_dw[j - 1];
            d_dg_dt[j] = dt - dg_dt[j - 1];
        }
    }
    (g_vals, dg_dw, dg_dt, dg_inc, d_dg_dw, d_dg_dt)
}

/// Deterministic multi-start grid search + damped Newton–Raphson maximization
/// of the 2D profile log-likelihood over `(u, v) = (ln omega, ln theta)`.
fn optimize_growth_params(
    prep: &PreparedClarkInputs,
    premium: Option<&[f64]>,
    curve: GrowthCurve,
) -> Result<(f64, f64, f64, f64), ValidationError> {
    let mid_age = prep.ages[prep.n / 2];
    let omega_grid: [f64; 6] = [0.4, 0.8, 1.2, 1.8, 2.6, 3.8];
    let theta_mults: [f64; 6] = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0];
    let mut ws = ClarkWorkspace::new(prep.n);

    let mut best_u = 0.0_f64;
    let mut best_v = mid_age.ln();
    let mut best_ll = f64::NEG_INFINITY;

    for &om in &omega_grid {
        let u = om.ln();
        for &tm in &theta_mults {
            let v = (mid_age * tm).ln();
            if let Some(ll) = profile_ll_only(u, v, prep, premium, curve, &mut ws)
                && ll > best_ll
            {
                best_ll = ll;
                best_u = u;
                best_v = v;
            }
        }
    }

    if !best_ll.is_finite() {
        return Err(ValidationError::InvalidParameter(
            "failed to find valid starting point for Clark MLE",
        ));
    }

    let mut u = best_u;
    let mut v = best_v;
    let mut cur_ll = best_ll;
    let mut cur_grad = [0.0; 2];

    for _iter in 0..80 {
        let Some((ll, grad)) = profile_ll_and_grad(u, v, prep, premium, curve, &mut ws) else {
            break;
        };
        cur_ll = ll;
        cur_grad = grad;

        let grad_norm = grad[0].abs().max(grad[1].abs());
        if grad_norm < 1e-9 {
            break;
        }

        // Numerical 2×2 Hessian of log-likelihood via central differences of exact gradient.
        let h = 1e-5;
        let g_up = profile_ll_and_grad(u + h, v, prep, premium, curve, &mut ws).map(|x| x.1);
        let g_um = profile_ll_and_grad(u - h, v, prep, premium, curve, &mut ws).map(|x| x.1);
        let g_vp = profile_ll_and_grad(u, v + h, prep, premium, curve, &mut ws).map(|x| x.1);
        let g_vm = profile_ll_and_grad(u, v - h, prep, premium, curve, &mut ws).map(|x| x.1);

        let step = if let (Some(gup), Some(gum), Some(gvp), Some(gvm)) = (g_up, g_um, g_vp, g_vm) {
            let h00 = (gup[0] - gum[0]) / (2.0 * h);
            let h01 = f64::midpoint((gvp[0] - gvm[0]) / (2.0 * h), (gup[1] - gum[1]) / (2.0 * h));
            let h11 = (gvp[1] - gvm[1]) / (2.0 * h);
            let mut chosen = None;
            let mut damp = 0.0_f64;
            for _ in 0..12 {
                let m00 = -h00 + damp;
                let m01 = -h01;
                let m11 = -h11 + damp;
                let det = m00 * m11 - m01 * m01;
                if m00 > 0.0 && det > 1e-18 {
                    let d0 = (m11 * grad[0] - m01 * grad[1]) / det;
                    let d1 = (-m01 * grad[0] + m00 * grad[1]) / det;
                    if d0.is_finite() && d1.is_finite() && d0 * grad[0] + d1 * grad[1] > 0.0 {
                        chosen = Some([d0, d1]);
                        break;
                    }
                }
                damp = if damp == 0.0 {
                    (h00.abs() + h11.abs()).max(1.0) * 1e-3
                } else {
                    damp * 10.0
                };
            }
            chosen.unwrap_or_else(|| {
                let norm = grad[0].hypot(grad[1]).max(1.0);
                [0.1 * grad[0] / norm, 0.1 * grad[1] / norm]
            })
        } else {
            let norm = grad[0].hypot(grad[1]).max(1.0);
            [0.05 * grad[0] / norm, 0.05 * grad[1] / norm]
        };

        let max_step = step[0].abs().max(step[1].abs());
        let cap = if max_step > 1.5 { 1.5 / max_step } else { 1.0 };
        let mut alpha = cap;
        let mut accepted = false;
        for _ in 0..30 {
            let nu = u + alpha * step[0];
            let nv = v + alpha * step[1];
            if let Some(nll) = profile_ll_only(nu, nv, prep, premium, curve, &mut ws)
                && nll >= cur_ll - 1e-12
            {
                u = nu;
                v = nv;
                cur_ll = nll;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted || (alpha * max_step) < 1e-13 {
            // Near the optimum, 1 ULP of |ll| (~1e-7 for ~1e9 triangles) can mask
            // tiny improvements; accept a full damped Newton step if it reduces |grad|.
            let nu = u + cap * step[0];
            let nv = v + cap * step[1];
            if let Some((nll, ngrad)) = profile_ll_and_grad(nu, nv, prep, premium, curve, &mut ws) {
                let next_gnorm = ngrad[0].abs().max(ngrad[1].abs());
                if next_gnorm < grad_norm {
                    u = nu;
                    v = nv;
                    cur_ll = nll;
                    continue;
                }
            }
            break;
        }
    }

    if let Some((ll, grad)) = profile_ll_and_grad(u, v, prep, premium, curve, &mut ws) {
        cur_ll = ll;
        cur_grad = grad;
    }

    let omega = u.exp();
    let theta = v.exp();
    let max_grad = cur_grad[0].abs().max(cur_grad[1].abs());
    Ok((omega, theta, cur_ll, max_grad))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Triangle;

    const NAN: f64 = f64::NAN;

    #[test]
    fn clark_ldf_recovers_exact_synthetic_loglogistic_and_weibull_curves() {
        for curve in [GrowthCurve::Loglogistic, GrowthCurve::Weibull] {
            let n = 8usize;
            let ages: Vec<f64> = (0..n).map(|j| j as f64 + 0.5).collect();
            let omega_true = 1.65;
            let theta_true = 2.40;
            let ultimates_true: Vec<f64> = (0..n).map(|i| 1000.0 + 120.0 * i as f64).collect();

            let mut data = vec![NAN; n * n];
            for i in 0..n {
                for j in 0..(n - i) {
                    data[i * n + j] =
                        ultimates_true[i] * curve.eval(ages[j], omega_true, theta_true);
                }
            }
            let tri = Triangle::new(n, n, data);
            let res = fit_clark_ldf(&tri.view(), &ClarkOptions::new(curve)).expect("clark");

            assert!((res.omega - omega_true).abs() < 1e-5, "omega={}", res.omega);
            assert!((res.theta - theta_true).abs() < 1e-5, "theta={}", res.theta);
            assert!(res.sigma_sq < 1e-16, "sigma_sq={}", res.sigma_sq);
            for i in 0..n {
                assert!(
                    (res.ultimates[i] - ultimates_true[i]).abs() < 1e-4,
                    "origin {i}: {} vs {}",
                    res.ultimates[i],
                    ultimates_true[i]
                );
            }
        }
    }

    #[test]
    fn clark_cape_cod_recovers_exact_elr_and_curve() {
        let n = 7usize;
        let ages: Vec<f64> = (0..n).map(|j| j as f64 + 0.5).collect();
        let omega_true = 1.4;
        let theta_true = 2.1;
        let elr_true = 0.68;
        let premium: Vec<f64> = (0..n).map(|i| 5000.0 + 300.0 * i as f64).collect();

        let mut data = vec![NAN; n * n];
        for i in 0..n {
            for j in 0..(n - i) {
                data[i * n + j] = premium[i]
                    * elr_true
                    * GrowthCurve::Weibull.eval(ages[j], omega_true, theta_true);
            }
        }
        let tri = Triangle::new(n, n, data);
        let res = fit_clark_cape_cod(
            &tri.view(),
            &premium,
            &ClarkOptions::new(GrowthCurve::Weibull),
        )
        .expect("clark cc");

        assert!((res.elr.unwrap() - elr_true).abs() < 1e-6);
        assert!((res.omega - omega_true).abs() < 1e-5);
        assert!((res.theta - theta_true).abs() < 1e-5);
    }
}
