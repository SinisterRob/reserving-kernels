//! Generalized Linear Model (GLM) stochastic loss reserving via Iteratively
//! Reweighted Least Squares (IRLS) — Taylor & McGuire (2016).
//!
//! Implements:
//!
//! * G. Taylor & G. McGuire, *Stochastic Loss Reserving Using Generalized
//!   Linear Models*, CAS Monograph No. 3 (2016), Chapters 3–5.
//!
//! ## Supported GLM Families (Log Link `eta_{i,j} = ln(mu_{i,j})`)
//!
//! * [`GlmFamily::Odp`] — Over-Dispersed Poisson (`V(mu) = mu`, Tweedie `p = 1`).
//! * [`GlmFamily::Gamma`] — Gamma (`V(mu) = mu^2`, Tweedie `p = 2`).
//! * [`GlmFamily::Tweedie`] — General Compound Poisson–Gamma Tweedie
//!   (`V(mu) = mu^p`, `p in [1, 2]`).
//! * [`GlmFamily::LogNormal`] — Gaussian on `ln Y_{i,j}` with bias-corrected
//!   expectations `exp(eta + 0.5 * sigma^2)` and exact log-normal MSEP.
//!
//! ## Supported Design Specifications
//!
//! * [`GlmDesign::CrossClassified`] — `eta_{i,j} = mu + alpha_i + beta_j`
//!   (for ODP, reproduces volume-weighted chain ladder and `fit_odp` identically).
//! * [`GlmDesign::Hoerl`] — Taylor & McGuire Ch. 4 Hoerl development curve
//!   `eta_{i,j} = mu + alpha_i + beta_1 ln(j + 1) + beta_2 (j + 1)`.
//! * [`GlmDesign::HoerlWithCalendarSplines`] — Hoerl development curve augmented
//!   with piecewise-linear calendar-year hinge splines `max(0, (i + j) - knot_m)`
//!   (CAS Monograph No. 3, Ch. 5).
//!
//! ## Solver & Uncertainty
//!
//! Each IRLS step solves the weighted normal system via Householder QR
//! (`W_k = mu_k^{2 - p}`, working response `z_k = eta_k + (y_k - mu_k) / mu_k`)
//! with deviance-monotone step-halving, returning exact Pearson & deviance
//! dispersions, hat-matrix leverages, and analytic process + parameter MSEP.

use crate::linalg::{quad_form, weighted_least_squares_qr};
use crate::triangle::{cumulative_to_incremental, latest_diagonal};
use crate::types::TriangleView;
use crate::validate::{
    MonotonicityPolicy, ValidationError, validate_monotone_cumulative,
    validate_positive_finite_slice, validate_triangle,
};

/// Exponential-dispersion GLM family for [`fit_glm`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum GlmFamily {
    /// Over-Dispersed Poisson (`V(mu) = mu`, Tweedie `p = 1.0`).
    Odp,
    /// Gamma (`V(mu) = mu^2`, Tweedie `p = 2.0`).
    Gamma,
    /// Tweedie power-variance family `V(mu) = mu^p` with `p in [1.0, 2.0]`.
    Tweedie {
        /// Tweedie variance power `p` (`1.0 <= p <= 2.0`).
        power: f64,
    },
    /// Log-normal GLM (`ln Y_{i,j} ~ N(eta_{i,j}, sigma^2)`).
    LogNormal,
}

impl GlmFamily {
    fn tweedie_power(self) -> Option<f64> {
        match self {
            Self::Odp => Some(1.0),
            Self::Gamma => Some(2.0),
            Self::Tweedie { power } => Some(power),
            Self::LogNormal => None,
        }
    }
}

/// Linear predictor design specification for [`fit_glm`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum GlmDesign {
    /// Saturated cross-classified model `eta_{i,j} = mu + alpha_i + beta_j`
    /// (`2n - 1` parameters; ODP recovers chain ladder identically).
    CrossClassified,
    /// Hoerl curve development `eta_{i,j} = mu + alpha_i + b1 * ln(j + 1) + b2 * (j + 1)`
    /// (`n + 2` parameters when `per_origin_intercepts = true`, else `3` parameters).
    Hoerl {
        /// Include separate accident-year factors `alpha_1..alpha_{n-1}`.
        per_origin_intercepts: bool,
    },
    /// Hoerl curve with piecewise-linear calendar-year splines (CAS Monograph 3, Ch. 5).
    ///
    /// * When `per_origin_intercepts = false`: includes a baseline calendar
    ///   inflation slope `c0 * (i + j)` plus hinge terms `max(0, i + j - knot_m)`.
    /// * When `per_origin_intercepts = true`: includes hinge change-points
    ///   `max(0, i + j - knot_m)` (`knot_m >= 1`) so the design remains full-rank.
    HoerlWithCalendarSplines {
        /// Include separate accident-year factors `alpha_1..alpha_{n-1}`.
        per_origin_intercepts: bool,
        /// Strictly increasing calendar-period spline knots in `(0, n - 1)`.
        calendar_knots: Vec<f64>,
    },
}

/// Configuration options for [`fit_glm`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GlmOptions {
    /// Exponential-dispersion family (`Odp`, `Gamma`, `Tweedie`, or `LogNormal`).
    pub family: GlmFamily,
    /// Linear predictor design matrix structure.
    pub design: GlmDesign,
    /// Optional positive exposure vector `E_i` (length `n`) entered as `ln(E_i)` offset.
    pub exposure: Option<Vec<f64>>,
    /// Maximum IRLS iterations (default `100`).
    pub max_iterations: usize,
    /// Relative parameter convergence tolerance (default `1e-10`).
    pub tolerance: f64,
    /// Cumulative monotonicity validation policy.
    pub monotonicity_policy: MonotonicityPolicy,
}

impl Default for GlmOptions {
    fn default() -> Self {
        Self::odp()
    }
}

impl GlmOptions {
    /// Construct default IRLS options for a given family and design.
    pub fn new(family: GlmFamily, design: GlmDesign) -> Self {
        Self {
            family,
            design,
            exposure: None,
            max_iterations: 100,
            tolerance: 1e-10,
            monotonicity_policy: MonotonicityPolicy::Strict,
        }
    }

    /// Over-Dispersed Poisson cross-classified GLM (reproduces chain ladder).
    pub fn odp() -> Self {
        Self::new(GlmFamily::Odp, GlmDesign::CrossClassified)
    }

    /// Gamma cross-classified GLM (`V(mu) = mu^2`).
    pub fn gamma() -> Self {
        Self::new(GlmFamily::Gamma, GlmDesign::CrossClassified)
    }

    /// Tweedie compound-Poisson-Gamma cross-classified GLM (`p in [1, 2]`).
    pub fn tweedie(power: f64) -> Self {
        Self::new(GlmFamily::Tweedie { power }, GlmDesign::CrossClassified)
    }

    /// Log-normal cross-classified GLM (closed-form OLS on `ln(Y)`).
    pub fn log_normal() -> Self {
        Self::new(GlmFamily::LogNormal, GlmDesign::CrossClassified)
    }

    /// Set the linear predictor design (`CrossClassified`, `Hoerl`, or `HoerlWithCalendarSplines`).
    pub fn with_design(mut self, design: GlmDesign) -> Self {
        self.design = design;
        self
    }

    /// Supply an exposure vector `E_i` entered as a `ln(E_i)` offset.
    pub fn with_exposure(mut self, exposure: impl Into<Vec<f64>>) -> Self {
        self.exposure = Some(exposure.into());
        self
    }

    /// Set the maximum number of IRLS iterations.
    pub const fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the relative convergence tolerance.
    pub const fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the triangle monotonicity policy.
    pub const fn with_policy(mut self, policy: MonotonicityPolicy) -> Self {
        self.monotonicity_policy = policy;
        self
    }
}

/// Complete output from [`fit_glm`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GlmReservingResult {
    /// GLM family fitted.
    pub family: GlmFamily,
    /// Names of fitted regression parameters (length `q`).
    pub param_names: Vec<String>,
    /// Fitted parameter vector `beta_hat` (length `q`).
    pub coefficients: Vec<f64>,
    /// Parameter standard errors `sqrt(diag(Cov(beta_hat)))` (length `q`).
    pub coefficient_se: Vec<f64>,
    /// Full `q × q` parameter covariance matrix `phi * (X^T W X)^{-1}` (row-major).
    pub coefficient_covariance: Vec<f64>,
    /// Number of IRLS iterations performed (`1` for closed-form `LogNormal`).
    pub iterations: usize,
    /// `true` if IRLS converged within `tolerance`.
    pub converged: bool,
    /// Number of observed cells entering the fit (`N`).
    pub n_observed: usize,
    /// Number of parameters (`q`).
    pub n_params: usize,
    /// Residual degrees of freedom (`N - q`).
    pub degrees_of_freedom: usize,
    /// Total unscaled unit deviance `sum_k d(y_k, mu_k)`.
    pub deviance: f64,
    /// Pearson chi-square statistic `sum_k (y_k - mu_k)^2 / V(mu_k)`.
    pub pearson_chi_sq: f64,
    /// Pearson dispersion estimate `phi_P = pearson_chi_sq / (N - q)`
    /// (equals `sigma^2` for `LogNormal`).
    pub pearson_dispersion: f64,
    /// Deviance dispersion estimate `phi_D = deviance / (N - q)`.
    pub deviance_dispersion: f64,
    /// Diagonal hat-matrix leverages `h_k` for each observed cell in row-major
    /// `(origin, dev)` order (`sum(h_k) == q`).
    pub leverages: Vec<f64>,
    /// Full `n × n` fitted/projected incremental expectations (row-major).
    pub fitted_incremental: Vec<f64>,
    /// Point-estimate reserves by origin (length `n`).
    pub reserves: Vec<f64>,
    /// Point-estimate ultimates by origin (`latest + reserves`, length `n`).
    pub ultimates: Vec<f64>,
    /// Analytic process standard error by origin.
    pub process_se: Vec<f64>,
    /// Analytic parameter estimation standard error by origin.
    pub parameter_se: Vec<f64>,
    /// Analytic total prediction error by origin `sqrt(process_se^2 + parameter_se^2)`.
    pub prediction_error: Vec<f64>,
    /// Total reserve across origins.
    pub total_reserve: f64,
    /// Analytic aggregate process standard error.
    pub total_process_se: f64,
    /// Analytic aggregate parameter estimation standard error.
    pub total_parameter_se: f64,
    /// Analytic aggregate prediction error.
    pub total_prediction_error: f64,
    /// Aggregate coefficient of variation (`total_prediction_error / total_reserve`).
    pub total_cv: f64,
}

/// Fit a GLM reserving model via IRLS to a cumulative loss triangle.
pub fn fit_glm(
    triangle: &TriangleView<'_>,
    options: &GlmOptions,
) -> Result<GlmReservingResult, ValidationError> {
    validate_triangle(triangle)?;
    if options.monotonicity_policy == MonotonicityPolicy::Strict {
        validate_monotone_cumulative(triangle)?;
    }
    let n = triangle.n_origins;
    if n < 3 {
        return Err(ValidationError::InvalidParameter(
            "GLM reserving requires at least a 3x3 triangle",
        ));
    }
    if let GlmFamily::Tweedie { power } = options.family {
        if !power.is_finite() || !(1.0..=2.0).contains(&power) {
            return Err(ValidationError::InvalidParameter(
                "Tweedie variance power must lie in [1.0, 2.0]",
            ));
        }
    }
    if let Some(exp) = &options.exposure {
        if exp.len() != n {
            return Err(ValidationError::ShapeMismatch);
        }
        validate_positive_finite_slice(exp, "exposure")?;
    }
    if options.max_iterations == 0 || !options.tolerance.is_finite() || options.tolerance <= 0.0 {
        return Err(ValidationError::InvalidParameter(
            "invalid IRLS convergence settings",
        ));
    }

    let inc = cumulative_to_incremental(triangle);
    let latest = latest_diagonal(triangle);
    let schema = GlmDesignSchema::build(n, &options.design)?;
    let q = schema.param_names.len();

    // Build observed cell table.
    let mut obs_cells: Vec<(usize, usize)> = Vec::new();
    let mut x_obs: Vec<f64> = Vec::new();
    let mut y_obs: Vec<f64> = Vec::new();
    let mut offsets_obs: Vec<f64> = Vec::new();
    let mut row_buf = vec![0.0; q];

    for i in 0..n {
        let log_exp = options.exposure.as_ref().map_or(0.0, |e| e[i].ln());
        for j in 0..(n - i) {
            let y = inc[i * n + j];
            if !y.is_finite() {
                return Err(ValidationError::NotFinite);
            }
            match options.family {
                GlmFamily::Gamma | GlmFamily::LogNormal => {
                    if y <= 0.0 {
                        return Err(ValidationError::InvalidParameter(
                            "Gamma and LogNormal GLMs require strictly positive observed increments",
                        ));
                    }
                }
                GlmFamily::Tweedie { power } if power > 1.0 => {
                    if y < 0.0 {
                        return Err(ValidationError::Negative);
                    }
                }
                _ => {
                    if y < 0.0 && options.monotonicity_policy == MonotonicityPolicy::Strict {
                        return Err(ValidationError::Negative);
                    }
                }
            }
            schema.fill_row(i, j, &mut row_buf);
            x_obs.extend_from_slice(&row_buf);
            y_obs.push(y);
            offsets_obs.push(log_exp);
            obs_cells.push((i, j));
        }
    }

    let n_obs = obs_cells.len();
    if n_obs <= q {
        return Err(ValidationError::InvalidParameter(
            "GLM model has non-positive residual degrees of freedom",
        ));
    }
    let dof = n_obs - q;

    if options.family == GlmFamily::LogNormal {
        return fit_log_normal_glm(
            n,
            &schema,
            &x_obs,
            &y_obs,
            &offsets_obs,
            &latest,
            options.exposure.as_deref(),
            n_obs,
            q,
            dof,
        );
    }

    let power = options.family.tweedie_power().expect("Tweedie family");

    // Initialize eta from log(max(y, positive_floor)) - offset via OLS.
    let positive_mean = y_obs.iter().copied().filter(|&v| v > 0.0).sum::<f64>()
        / (y_obs.iter().filter(|&&v| v > 0.0).count().max(1) as f64);
    let y_floor = (positive_mean * 1e-4).max(1e-8);
    let init_z: Vec<f64> = y_obs
        .iter()
        .zip(offsets_obs.iter())
        .map(|(&y, &off)| y.max(y_floor).ln() - off)
        .collect();
    let ones = vec![1.0; n_obs];
    let mut beta = crate::linalg::weighted_least_squares_qr_beta_only(&x_obs, &init_z, &ones, n_obs, q)?;

    let mut eta = eval_linear_predictor(&x_obs, &offsets_obs, &beta, n_obs, q);
    let mut mu: Vec<f64> = eta.iter().map(|&e| e.clamp(-50.0, 50.0).exp()).collect();
    let mut cur_obj = total_tweedie_objective(&y_obs, &mu, power);

    let mut converged = false;
    let mut iterations = 0usize;

    let mut weights = vec![0.0; n_obs];
    let mut working_z = vec![0.0; n_obs];

    for iter in 1..=options.max_iterations {
        iterations = iter;
        for k in 0..n_obs {
            let mk = mu[k];
            weights[k] = if (power - 1.0).abs() < 1e-12 {
                mk
            } else if (power - 2.0).abs() < 1e-12 {
                1.0
            } else {
                mk.powf(2.0 - power)
            };
            // Working response minus offset so QR solves X * beta_new = working_z.
            working_z[k] = (eta[k] - offsets_obs[k]) + (y_obs[k] - mk) / mk;
        }

        let beta_target =
            crate::linalg::weighted_least_squares_qr_beta_only(&x_obs, &working_z, &weights, n_obs, q)?;
        let delta: Vec<f64> = beta_target
            .iter()
            .zip(beta.iter())
            .map(|(&bt, &b)| bt - b)
            .collect();

        // Step-halving line search guaranteeing non-increasing quasi-deviance.
        let mut step_len = 1.0_f64;
        let mut next_beta = beta_target;
        let mut next_eta = eval_linear_predictor(&x_obs, &offsets_obs, &next_beta, n_obs, q);
        let mut next_mu: Vec<f64> = next_eta
            .iter()
            .map(|&e| e.clamp(-50.0, 50.0).exp())
            .collect();
        let mut next_obj = total_tweedie_objective(&y_obs, &next_mu, power);

        let mut halvings = 0usize;
        while (!next_obj.is_finite() || next_obj > cur_obj + 1e-9 * (1.0 + cur_obj.abs()))
            && halvings < 25
        {
            step_len *= 0.5;
            halvings += 1;
            for j in 0..q {
                next_beta[j] = beta[j] + step_len * delta[j];
            }
            next_eta = eval_linear_predictor(&x_obs, &offsets_obs, &next_beta, n_obs, q);
            next_mu = next_eta
                .iter()
                .map(|&e| e.clamp(-50.0, 50.0).exp())
                .collect();
            next_obj = total_tweedie_objective(&y_obs, &next_mu, power);
        }

        let max_rel_change = next_beta
            .iter()
            .zip(beta.iter())
            .map(|(&bn, &bo)| (bn - bo).abs() / (1.0 + bo.abs()))
            .fold(0.0_f64, f64::max);
        let rel_obj_change = (cur_obj - next_obj).abs() / (1.0 + cur_obj.abs());

        beta = next_beta;
        eta = next_eta;
        mu = next_mu;
        cur_obj = next_obj;

        if max_rel_change < options.tolerance && rel_obj_change < options.tolerance {
            converged = true;
            break;
        }
    }

    let cur_dev = total_tweedie_deviance(&y_obs, &mu, power);

    // Compute exact final (X^T W X)^{-1} and hat leverages once at converged mu.
    for k in 0..n_obs {
        let mk = mu[k];
        weights[k] = if (power - 1.0).abs() < 1e-12 {
            mk
        } else if (power - 2.0).abs() < 1e-12 {
            1.0
        } else {
            mk.powf(2.0 - power)
        };
        working_z[k] = eta[k] - offsets_obs[k];
    }
    let final_sol = weighted_least_squares_qr(&x_obs, &working_z, &weights, n_obs, q)?;
    let final_xtwx_inv = final_sol.xtwx_inv;
    let final_leverages = final_sol.leverages;

    // Pearson chi-square & dispersion estimates.
    let mut pearson_chi_sq = 0.0;
    for k in 0..n_obs {
        let diff = y_obs[k] - mu[k];
        let v_mu = mu[k].powf(power);
        pearson_chi_sq += (diff * diff) / v_mu;
    }
    let pearson_dispersion = (pearson_chi_sq / dof as f64).max(0.0);
    let deviance_dispersion = (cur_dev / dof as f64).max(0.0);

    let cov: Vec<f64> = final_xtwx_inv
        .iter()
        .map(|&v| pearson_dispersion * v)
        .collect();
    let coefficient_se: Vec<f64> = (0..q).map(|k| cov[k * q + k].max(0.0).sqrt()).collect();

    // Project full n × n incremental grid and analytic MSEP.
    let mut fitted_incremental = vec![0.0; n * n];
    let mut reserves = vec![0.0; n];
    let mut process_se = vec![0.0; n];
    let mut parameter_se = vec![0.0; n];
    let mut prediction_error = vec![0.0; n];
    let mut grad_total = vec![0.0; q];
    let mut total_proc_var = 0.0;

    let mut x_cell = vec![0.0; q];
    for i in 0..n {
        let log_exp = options.exposure.as_ref().map_or(0.0, |e| e[i].ln());
        let mut grad_origin = vec![0.0; q];
        let mut proc_var_i = 0.0;
        let mut r_i = 0.0;

        for j in 0..n {
            schema.fill_row(i, j, &mut x_cell);
            let mut e_ij = log_exp;
            for col in 0..q {
                e_ij += x_cell[col] * beta[col];
            }
            let mu_ij = e_ij.clamp(-50.0, 50.0).exp();
            fitted_incremental[i * n + j] = mu_ij;

            if i + j >= n {
                r_i += mu_ij;
                proc_var_i += pearson_dispersion * mu_ij.powf(power);
                for col in 0..q {
                    let g = mu_ij * x_cell[col];
                    grad_origin[col] += g;
                    grad_total[col] += g;
                }
            }
        }
        reserves[i] = r_i;
        total_proc_var += proc_var_i;
        let param_var_i = quad_form(&grad_origin, &cov, q).max(0.0);
        process_se[i] = proc_var_i.max(0.0).sqrt();
        parameter_se[i] = param_var_i.sqrt();
        prediction_error[i] = (proc_var_i.max(0.0) + param_var_i).sqrt();
    }

    let ultimates: Vec<f64> = latest
        .iter()
        .zip(reserves.iter())
        .map(|(&l, &r)| l + r)
        .collect();
    let total_reserve: f64 = reserves.iter().sum();
    let total_param_var = quad_form(&grad_total, &cov, q).max(0.0);
    let total_proc_var = total_proc_var.max(0.0);
    let total_prediction_error = (total_proc_var + total_param_var).sqrt();
    let total_cv = if total_reserve > 0.0 {
        total_prediction_error / total_reserve
    } else {
        0.0
    };

    Ok(GlmReservingResult {
        family: options.family,
        param_names: schema.param_names,
        coefficients: beta,
        coefficient_se,
        coefficient_covariance: cov,
        iterations,
        converged,
        n_observed: n_obs,
        n_params: q,
        degrees_of_freedom: dof,
        deviance: cur_dev,
        pearson_chi_sq,
        pearson_dispersion,
        deviance_dispersion,
        leverages: final_leverages,
        fitted_incremental,
        reserves,
        ultimates,
        process_se,
        parameter_se,
        prediction_error,
        total_reserve,
        total_process_se: total_proc_var.sqrt(),
        total_parameter_se: total_param_var.sqrt(),
        total_prediction_error,
        total_cv,
    })
}

#[allow(clippy::too_many_arguments)]
fn fit_log_normal_glm(
    n: usize,
    schema: &GlmDesignSchema,
    x_obs: &[f64],
    y_obs: &[f64],
    offsets_obs: &[f64],
    latest: &[f64],
    exposure: Option<&[f64]>,
    n_obs: usize,
    q: usize,
    dof: usize,
) -> Result<GlmReservingResult, ValidationError> {
    let z_obs: Vec<f64> = y_obs
        .iter()
        .zip(offsets_obs.iter())
        .map(|(&y, &off)| y.ln() - off)
        .collect();
    let ones = vec![1.0; n_obs];
    let sol = weighted_least_squares_qr(x_obs, &z_obs, &ones, n_obs, q)?;

    let mut rss = 0.0;
    for k in 0..n_obs {
        let mut eta = 0.0;
        for col in 0..q {
            eta += x_obs[k * q + col] * sol.beta[col];
        }
        let d = z_obs[k] - eta;
        rss += d * d;
    }
    let sigma_sq = (rss / dof as f64).max(0.0);
    let cov: Vec<f64> = sol.xtwx_inv.iter().map(|&v| sigma_sq * v).collect();
    let coefficient_se: Vec<f64> = (0..q).map(|k| cov[k * q + k].max(0.0).sqrt()).collect();

    let mut fitted_incremental = vec![0.0; n * n];
    let mut future_cells: Vec<(usize, f64, Vec<f64>)> = Vec::new();
    let mut reserves = vec![0.0; n];

    for i in 0..n {
        let log_exp = exposure.map_or(0.0, |e| e[i].ln());
        for j in 0..n {
            let mut x_cell = vec![0.0; q];
            schema.fill_row(i, j, &mut x_cell);
            let mut eta = log_exp;
            for col in 0..q {
                eta += x_cell[col] * sol.beta[col];
            }
            let mu = (eta + 0.5 * sigma_sq).exp();
            if !mu.is_finite() {
                return Err(ValidationError::NotFinite);
            }
            fitted_incremental[i * n + j] = mu;
            if i + j >= n {
                reserves[i] += mu;
                future_cells.push((i, mu, x_cell));
            }
        }
    }

    // Exact log-normal process + parameter variance decomposition.
    // Precompute Cov * x_b for every future cell so pairwise covariances run in O(N_fut^2 * q).
    let mut proc_var_origin = vec![0.0; n];
    let mut param_var_origin = vec![0.0; n];
    let mut total_proc_var = 0.0;
    let mut total_param_var = 0.0;
    let n_fut = future_cells.len();

    let mut cov_x: Vec<Vec<f64>> = Vec::with_capacity(n_fut);
    for (_, _, xb) in &future_cells {
        let mut v = vec![0.0; q];
        for r in 0..q {
            let mut acc = 0.0;
            for c in 0..q {
                acc += cov[r * q + c] * xb[c];
            }
            v[r] = acc;
        }
        cov_x.push(v);
    }

    for a in 0..n_fut {
        let (ia, mu_a, ref xa) = future_cells[a];
        let mut v_est_a = 0.0;
        let cva = &cov_x[a];
        for r in 0..q {
            v_est_a += xa[r] * cva[r];
        }
        let v_est_a = v_est_a.max(0.0);
        let pv = mu_a * mu_a * sigma_sq.exp_m1();
        let tot_a = mu_a * mu_a * (sigma_sq + v_est_a).exp_m1();
        let pmv = (tot_a - pv).max(0.0);
        proc_var_origin[ia] += pv;
        param_var_origin[ia] += pmv;
        total_proc_var += pv;
        total_param_var += pmv;

        for b in (a + 1)..n_fut {
            let (ib, mu_b, _) = future_cells[b];
            let mut cov_eta = 0.0;
            let cvb = &cov_x[b];
            for r in 0..q {
                cov_eta += xa[r] * cvb[r];
            }
            let cov_ab = mu_a * mu_b * cov_eta.exp_m1();
            total_param_var += 2.0 * cov_ab;
            if ia == ib {
                param_var_origin[ia] += 2.0 * cov_ab;
            }
        }
    }

    let process_se: Vec<f64> = proc_var_origin.iter().map(|v| v.max(0.0).sqrt()).collect();
    let parameter_se: Vec<f64> = param_var_origin.iter().map(|v| v.max(0.0).sqrt()).collect();
    let prediction_error: Vec<f64> = proc_var_origin
        .iter()
        .zip(param_var_origin.iter())
        .map(|(&p1, &p2)| (p1 + p2).max(0.0).sqrt())
        .collect();
    let ultimates: Vec<f64> = latest
        .iter()
        .zip(reserves.iter())
        .map(|(&l, &r)| l + r)
        .collect();
    let total_reserve: f64 = reserves.iter().sum();
    let total_prediction_error = (total_proc_var + total_param_var).max(0.0).sqrt();
    let total_cv = if total_reserve > 0.0 {
        total_prediction_error / total_reserve
    } else {
        0.0
    };

    Ok(GlmReservingResult {
        family: GlmFamily::LogNormal,
        param_names: schema.param_names.clone(),
        coefficients: sol.beta,
        coefficient_se,
        coefficient_covariance: cov,
        iterations: 1,
        converged: true,
        n_observed: n_obs,
        n_params: q,
        degrees_of_freedom: dof,
        deviance: rss,
        pearson_chi_sq: rss,
        pearson_dispersion: sigma_sq,
        deviance_dispersion: sigma_sq,
        leverages: sol.leverages,
        fitted_incremental,
        reserves,
        ultimates,
        process_se,
        parameter_se,
        prediction_error,
        total_reserve,
        total_process_se: total_proc_var.max(0.0).sqrt(),
        total_parameter_se: total_param_var.max(0.0).sqrt(),
        total_prediction_error,
        total_cv,
    })
}

fn eval_linear_predictor(
    x: &[f64],
    offsets: &[f64],
    beta: &[f64],
    n_obs: usize,
    q: usize,
) -> Vec<f64> {
    let mut eta = vec![0.0; n_obs];
    for k in 0..n_obs {
        let mut s = offsets[k];
        for j in 0..q {
            s += x[k * q + j] * beta[j];
        }
        eta[k] = s;
    }
    eta
}

/// Exact unit deviance for Tweedie family `p in [1, 2]`, written with `exp_m1`
/// to avoid catastrophic cancellation when `p` is close to `1` or `2`.
fn tweedie_unit_deviance(y: f64, mu: f64, p: f64) -> f64 {
    if (p - 1.0).abs() < 1e-12 {
        // Poisson / ODP deviance: 2 * (y ln(y / mu) - (y - mu)).
        let term = if y > 0.0 { y * (y / mu).ln() } else { 0.0 };
        2.0 * (term - (y - mu))
    } else if (p - 2.0).abs() < 1e-12 {
        // Gamma deviance: 2 * (-ln(y / mu) + (y - mu) / mu).
        2.0 * (-(y / mu).ln() + (y - mu) / mu)
    } else {
        let one_mp = 1.0 - p;
        let two_mp = 2.0 - p;
        if y > 0.0 {
            let ln_ratio = (y / mu).ln();
            // Rewrite y^{2-p}/((1-p)(2-p)) - y*mu^{1-p}/(1-p) + mu^{2-p}/(2-p)
            // = y * mu^{1-p} * (exp((1-p)*ln(y/mu)) - 1)/(1-p)
            //   - mu^{2-p} * (exp((2-p)*ln(y/mu)) - 1)/(2-p)
            // Note: both (exp_m1(a*x)/a) terms remain smooth and accurate as a -> 0.
            let t1 = y * mu.powf(one_mp) * (one_mp * ln_ratio).exp_m1() / one_mp;
            let t2 = mu.powf(two_mp) * (two_mp * ln_ratio).exp_m1() / two_mp;
            2.0 * (t1 - t2)
        } else {
            2.0 * mu.powf(two_mp) / two_mp
        }
    }
}

/// Line-search quasi-deviance objective matching the IRLS score equation even
/// when `y < 0` under ODP (`p == 1`) with `AllowNegativeIncrements`.
fn tweedie_line_search_objective(y: f64, mu: f64, p: f64) -> f64 {
    if (p - 1.0).abs() < 1e-12 && y < 0.0 {
        // Negative twice ODP quasi-log-likelihood: 2 * (mu - y * ln(mu)).
        // Its derivative w.r.t. mu is 2 * (1 - y / mu), matching the IRLS score.
        2.0 * (mu - y * mu.ln())
    } else {
        tweedie_unit_deviance(y, mu, p)
    }
}

fn total_tweedie_objective(y: &[f64], mu: &[f64], p: f64) -> f64 {
    y.iter()
        .zip(mu.iter())
        .map(|(&yk, &mk)| tweedie_line_search_objective(yk, mk, p))
        .sum()
}

fn total_tweedie_deviance(y: &[f64], mu: &[f64], p: f64) -> f64 {
    y.iter()
        .zip(mu.iter())
        .map(|(&yk, &mk)| tweedie_unit_deviance(yk, mk, p))
        .sum()
}

struct GlmDesignSchema {
    n: usize,
    design: GlmDesign,
    param_names: Vec<String>,
}

impl GlmDesignSchema {
    fn build(n: usize, design: &GlmDesign) -> Result<Self, ValidationError> {
        let mut names = vec!["intercept".to_string()];
        match design {
            GlmDesign::CrossClassified => {
                for i in 1..n {
                    names.push(format!("origin_{i}"));
                }
                for j in 1..n {
                    names.push(format!("dev_{j}"));
                }
            }
            GlmDesign::Hoerl {
                per_origin_intercepts,
            } => {
                if *per_origin_intercepts {
                    for i in 1..n {
                        names.push(format!("origin_{i}"));
                    }
                }
                names.push("hoerl_ln_j".to_string());
                names.push("hoerl_j".to_string());
            }
            GlmDesign::HoerlWithCalendarSplines {
                per_origin_intercepts,
                calendar_knots,
            } => {
                let mut prev = 0.0_f64;
                for &k in calendar_knots {
                    if !k.is_finite() || k <= 0.0 || k >= (n - 1) as f64 {
                        return Err(ValidationError::InvalidParameter(
                            "calendar spline knots must lie strictly inside (0, n - 1)",
                        ));
                    }
                    if k <= prev {
                        return Err(ValidationError::InvalidParameter(
                            "calendar spline knots must be strictly increasing",
                        ));
                    }
                    prev = k;
                }
                if *per_origin_intercepts {
                    for i in 1..n {
                        names.push(format!("origin_{i}"));
                    }
                }
                names.push("hoerl_ln_j".to_string());
                names.push("hoerl_j".to_string());
                if !*per_origin_intercepts {
                    names.push("cal_slope_0".to_string());
                }
                for &k in calendar_knots {
                    names.push(format!("cal_hinge_{k}"));
                }
            }
        }
        Ok(Self {
            n,
            design: design.clone(),
            param_names: names,
        })
    }

    fn fill_row(&self, i: usize, j: usize, out: &mut [f64]) {
        out.fill(0.0);
        out[0] = 1.0;
        let d = (j + 1) as f64; // 1-based development period for Hoerl curve
        let t = (i + j) as f64;

        match &self.design {
            GlmDesign::CrossClassified => {
                if i > 0 {
                    out[i] = 1.0;
                }
                if j > 0 {
                    out[self.n + (j - 1)] = 1.0;
                }
            }
            GlmDesign::Hoerl {
                per_origin_intercepts,
            } => {
                let mut col = 1;
                if *per_origin_intercepts {
                    if i > 0 {
                        out[i] = 1.0;
                    }
                    col = self.n;
                }
                out[col] = d.ln();
                out[col + 1] = d;
            }
            GlmDesign::HoerlWithCalendarSplines {
                per_origin_intercepts,
                calendar_knots,
            } => {
                let mut col = 1;
                if *per_origin_intercepts {
                    if i > 0 {
                        out[i] = 1.0;
                    }
                    col = self.n;
                }
                out[col] = d.ln();
                out[col + 1] = d;
                col += 2;
                if !*per_origin_intercepts {
                    out[col] = t;
                    col += 1;
                }
                for &knot in calendar_knots {
                    out[col] = (t - knot).max(0.0);
                    col += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bootstrap_odp::fit_odp;
    use crate::types::Triangle;

    const NAN: f64 = f64::NAN;

    fn sample_tri() -> Triangle {
        #[rustfmt::skip]
        let data = vec![
            100.0, 220.0, 310.0, 360.0, 385.0,
            120.0, 255.0, 365.0, 420.0,   NAN,
            135.0, 290.0, 410.0,   NAN,   NAN,
            150.0, 318.0,   NAN,   NAN,   NAN,
            175.0,   NAN,   NAN,   NAN,   NAN,
        ];
        Triangle::new(5, 5, data)
    }

    #[test]
    fn cross_classified_odp_irls_matches_exact_fit_odp_to_machine_precision() {
        let tri = sample_tri();
        let odp_exact = fit_odp(&tri.view(), 1.0, MonotonicityPolicy::Strict).unwrap();
        let glm = fit_glm(
            &tri.view(),
            &GlmOptions::new(GlmFamily::Odp, GlmDesign::CrossClassified),
        )
        .unwrap();

        assert!(glm.converged);
        assert!((glm.total_reserve - odp_exact.total_reserve).abs() < 1e-8);
        assert!((glm.pearson_dispersion - odp_exact.dispersion).abs() < 1e-8);
        assert!(
            (glm.total_prediction_error - odp_exact.analytic_total_prediction_error).abs() < 1e-6
        );
        let h_sum: f64 = glm.leverages.iter().sum();
        assert!((h_sum - glm.n_params as f64).abs() < 1e-10);
    }

    #[test]
    fn hoerl_gamma_and_tweedie_recover_exact_synthetic_hoerl_triangle() {
        let n = 7usize;
        let b0 = 6.0;
        let b1 = 1.25;
        let b2 = -0.65;
        let mut data = vec![NAN; n * n];
        let mut true_reserve = 0.0;
        for i in 0..n {
            let alpha_i = 0.08 * i as f64;
            let mut cum = 0.0;
            for j in 0..n {
                let d = (j + 1) as f64;
                let inc = (b0 + alpha_i + b1 * d.ln() + b2 * d).exp();
                if i + j < n {
                    cum += inc;
                    data[i * n + j] = cum;
                } else {
                    true_reserve += inc;
                }
            }
        }
        let tri = Triangle::new(n, n, data);

        for family in [
            GlmFamily::Odp,
            GlmFamily::Tweedie { power: 1.5 },
            GlmFamily::Gamma,
            GlmFamily::LogNormal,
        ] {
            let res = fit_glm(
                &tri.view(),
                &GlmOptions::new(
                    family,
                    GlmDesign::Hoerl {
                        per_origin_intercepts: true,
                    },
                ),
            )
            .expect("hoerl fit");
            assert!(res.converged);
            assert!(
                (res.total_reserve - true_reserve).abs() / true_reserve < 1e-8,
                "family={family:?}: got {} expected {}",
                res.total_reserve,
                true_reserve
            );
            assert!(
                res.deviance < 1e-12,
                "family={family:?}: dev={}",
                res.deviance
            );
        }
    }
}
