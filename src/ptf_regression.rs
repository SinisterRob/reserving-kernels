//! Barnett & Zehnwirth (2000) Probabilistic Trend Family (PTF) log-linear
//! regression reserving kernel.
//!
//! Implements:
//!
//! * G. Barnett & B. Zehnwirth, *"Best Estimates for Reserves"*, *PCAS*,
//!   Vol. LXXXVII (2000), pp. 245–321.
//!
//! ## Model Specification
//!
//! For strictly positive incremental claims `Y_{i,j}` (origin `i = 0..n-1`,
//! development `j = 0..n-1`, calendar period `t = i + j`), optionally divided
//! by exposure `E_i`:
//!
//! `ln(Y_{i,j} / E_i) = alpha_i + sum_{k=1}^j gamma_k + sum_{t=1}^{i+j} iota_t + eps_{i,j}`
//!
//! with `eps_{i,j} ~ N(0, sigma^2 / w_{i,j})`.
//!
//! Because `t = i + j`, saturating all three directions simultaneously creates
//! an age–period–cohort collinearity. PTF parameterizes each direction via
//! parsimonious piecewise-linear trends with explicit knots (solved via
//! Householder QR weighted least squares):
//!
//! * **Accident direction (`AccidentSpec`)**: constant intercept `alpha`,
//!   piecewise-linear accident trend with knots, or per-origin levels
//!   `alpha_0..alpha_{n-1}`.
//! * **Development direction (`DevelopmentSpec`)**: piecewise-linear
//!   development slopes `gamma` with knots, or full per-lag increments
//!   `gamma_1..gamma_{n-1}`.
//! * **Calendar direction (`CalendarSpec`)**: zero calendar trend, constant
//!   calendar inflation `iota * (i + j)`, piecewise-linear inflation regimes
//!   with knots `kappa_m` (`sum_m delta_m * max(0, i + j - kappa_m)`), or
//!   inflation change-points only (identifiable even alongside saturated
//!   per-origin `alpha_i` and per-lag `gamma_k`).
//!
//! ## Forecasting & Analytical MSEP
//!
//! Future incremental cells `(i, j)` (`i + j >= n`) are projected on the
//! log scale as `eta_hat_{i,j} = x_{i,j}^T beta_hat + ln(E_i)` with parameter
//! covariance `Cov(eta_hat_a, eta_hat_b) = sigma^2 * x_a^T (X^T W X)^{-1} x_b`,
//! log-normal bias correction, and exact analytical log-normal reserve MSEP
//! including full cross-cell parameter covariance.

use crate::linalg::{quad_form, weighted_least_squares_qr};
use crate::triangle::cumulative_to_incremental;
use crate::types::TriangleView;
use crate::validate::{ValidationError, validate_positive_finite_slice, validate_triangle};

/// Accident-year direction specification for PTF regression.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AccidentSpec {
    /// Single constant level `alpha` shared across origins (allows full
    /// identification of a baseline calendar inflation rate `iota`).
    Constant,
    /// Piecewise-linear accident trend with slope change knots (e.g. `&[]` for
    /// linear accident trend, `&[4]` for slope change at origin 4).
    PiecewiseLinear {
        /// Interior knot origin indices `k` (`1 <= k < n - 1`).
        knots: Vec<usize>,
    },
    /// Separate level `alpha_i` for every origin `i = 0..n-1`.
    PerOrigin,
}

/// Development-period direction specification for PTF regression.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DevelopmentSpec {
    /// Separate development step `gamma_k` for every lag `k = 1..n-1`.
    PerPeriod,
    /// Piecewise-linear development trend with slope change knots at lags `k`.
    PiecewiseLinear {
        /// Interior knot development lag indices `k` (`1 <= k < n - 1`).
        knots: Vec<usize>,
    },
}

/// Calendar-year inflation direction specification for PTF regression.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CalendarSpec {
    /// No calendar-year inflation term (`iota_t = 0`).
    None,
    /// Constant calendar inflation rate `iota` across all calendar periods `t = i + j`.
    ConstantRate,
    /// Piecewise-linear calendar inflation with baseline rate `iota_0` plus
    /// rate shifts `delta_m` after calendar knot periods `knots`
    /// (`iota_0 * t + sum_m delta_m * max(0, t - knot_m)`).
    PiecewiseLinear {
        /// Calendar-period knots `t_m` (`1 <= t_m < n - 1`) where inflation rate shifts.
        knots: Vec<usize>,
    },
    /// Calendar inflation regime shifts only (`sum_m delta_m * max(0, t - knot_m)`),
    /// omitting the baseline linear slope so it remains uniquely identifiable
    /// even when both `AccidentSpec::PerOrigin` and `DevelopmentSpec::PerPeriod`
    /// are active.
    ChangePointsOnly {
        /// Calendar-period knots `t_m` (`1 <= t_m < n - 1`).
        knots: Vec<usize>,
    },
}

/// Log-normal bias-correction convention when transforming log-scale
/// predictions `eta_hat` back to dollar incremental expectations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum LogNormalMeanMode {
    /// Plug-in log-normal mean: `exp(eta_hat + 0.5 * sigma^2)`.
    #[default]
    ProcessVariance,
    /// Full predictive log-normal mean: `exp(eta_hat + 0.5 * (sigma^2 + v_est))`.
    FullPredictive,
}

/// Configuration for [`fit_ptf`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PtfOptions {
    /// Accident-year trend parameterization.
    pub accident: AccidentSpec,
    /// Development-period trend parameterization.
    pub development: DevelopmentSpec,
    /// Calendar-year inflation trend parameterization.
    pub calendar: CalendarSpec,
    /// Optional positive exposure vector `E_i` (length `n`) used as log offset.
    pub exposure: Option<Vec<f64>>,
    /// Optional per-development-period observation weights `w_j > 0` (length `n`)
    /// for heteroscedastic weighted OLS (Barnett & Zehnwirth 2000, §5).
    pub dev_weights: Option<Vec<f64>>,
    /// Future calendar inflation assumption beyond the last observed diagonal
    /// (`t >= n`): when `Some(r)`, overrides future calendar slope beyond `t = n - 1`
    /// with constant rate `r`; when `None`, continues the final estimated
    /// calendar-year inflation regime.
    pub future_calendar_rate: Option<f64>,
    /// Log-normal mean bias-correction convention.
    pub mean_mode: LogNormalMeanMode,
}

impl Default for PtfOptions {
    fn default() -> Self {
        Self {
            accident: AccidentSpec::Constant,
            development: DevelopmentSpec::PerPeriod,
            calendar: CalendarSpec::ConstantRate,
            exposure: None,
            dev_weights: None,
            future_calendar_rate: None,
            mean_mode: LogNormalMeanMode::ProcessVariance,
        }
    }
}

impl PtfOptions {
    /// Set the accident-year trend specification.
    pub fn with_accident(mut self, accident: AccidentSpec) -> Self {
        self.accident = accident;
        self
    }

    /// Set the development-period trend specification.
    pub fn with_development(mut self, development: DevelopmentSpec) -> Self {
        self.development = development;
        self
    }

    /// Set the calendar-year inflation specification.
    pub fn with_calendar(mut self, calendar: CalendarSpec) -> Self {
        self.calendar = calendar;
        self
    }

    /// Supply a positive exposure vector `E_i` used as a `ln(E_i)` offset.
    pub fn with_exposure(mut self, exposure: impl Into<Vec<f64>>) -> Self {
        self.exposure = Some(exposure.into());
        self
    }

    /// Supply per-development-period observation weights `w_j > 0`.
    pub fn with_dev_weights(mut self, dev_weights: impl Into<Vec<f64>>) -> Self {
        self.dev_weights = Some(dev_weights.into());
        self
    }

    /// Override the future calendar inflation rate beyond the last observed diagonal.
    pub const fn with_future_calendar_rate(mut self, rate: f64) -> Self {
        self.future_calendar_rate = Some(rate);
        self
    }

    /// Set the log-normal mean bias-correction convention.
    pub const fn with_mean_mode(mut self, mean_mode: LogNormalMeanMode) -> Self {
        self.mean_mode = mean_mode;
        self
    }
}

/// Output from Barnett & Zehnwirth (2000) PTF log-linear regression reserving.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PtfResult {
    /// Human-readable names of the fitted regression parameters (length `p`).
    pub param_names: Vec<String>,
    /// Estimated regression parameter vector `beta_hat` (length `p`).
    pub coefficients: Vec<f64>,
    /// Parameter standard errors `sqrt(diag(Cov(beta_hat)))` (length `p`).
    pub coefficient_se: Vec<f64>,
    /// Full `p × p` parameter covariance matrix `sigma^2 * (X^T W X)^{-1}` (row-major).
    pub coefficient_covariance: Vec<f64>,
    /// Implied incremental calendar-year inflation rates `iota_t` for `t = 1..n-1`
    /// (on the log scale; annual percentage rate is `exp(iota_t) - 1`).
    pub calendar_inflation_rates: Vec<f64>,
    /// Residual variance `sigma^2 = weighted_rss / (N - p)`.
    pub sigma_sq: f64,
    /// Residual standard error `sigma = sqrt(sigma_sq)`.
    pub sigma: f64,
    /// Number of observed positive incremental cells `N`.
    pub n_observed: usize,
    /// Number of fitted parameters `p`.
    pub n_params: usize,
    /// Residual degrees of freedom `N - p`.
    pub degrees_of_freedom: usize,
    /// Weighted coefficient of determination `R^2`.
    pub r_squared: f64,
    /// Akaike Information Criterion (`N * ln(weighted_rss / N) + 2 * p`, Gaussian log-scale).
    pub aic: f64,
    /// Bayesian Information Criterion (`N * ln(weighted_rss / N) + p * ln(N)`).
    pub bic: f64,
    /// Full `n × n` fitted/projected incremental expectations (row-major).
    pub fitted_incremental: Vec<f64>,
    /// Projected reserves by origin (length `n`).
    pub reserves: Vec<f64>,
    /// Projected ultimates by origin (`latest_cumulative + reserves[i]`, length `n`).
    pub ultimates: Vec<f64>,
    /// Analytical prediction standard error by origin (length `n`).
    pub standard_error: Vec<f64>,
    /// Total projected reserve across origins.
    pub total_reserve: f64,
    /// Analytical aggregate prediction MSEP (including cross-cell parameter covariance).
    pub total_msep: f64,
    /// Analytical aggregate prediction standard error `sqrt(total_msep)`.
    pub total_standard_error: f64,
    /// Aggregate coefficient of variation (`total_standard_error / total_reserve`).
    pub total_cv: f64,
}

/// Fit the Barnett & Zehnwirth (2000) Probabilistic Trend Family (PTF) model
/// to a cumulative loss triangle.
pub fn fit_ptf(
    triangle: &TriangleView<'_>,
    options: &PtfOptions,
) -> Result<PtfResult, ValidationError> {
    validate_triangle(triangle)?;
    let n = triangle.n_origins;
    if n < 3 {
        return Err(ValidationError::InvalidParameter(
            "PTF regression requires at least a 3x3 triangle",
        ));
    }

    if let Some(exp) = &options.exposure {
        if exp.len() != n {
            return Err(ValidationError::ShapeMismatch);
        }
        validate_positive_finite_slice(exp, "exposure")?;
    }
    if let Some(dw) = &options.dev_weights {
        if dw.len() != n {
            return Err(ValidationError::ShapeMismatch);
        }
        validate_positive_finite_slice(dw, "dev_weights")?;
    }
    if let Some(fcr) = options.future_calendar_rate {
        if !fcr.is_finite() {
            return Err(ValidationError::NotFinite);
        }
    }

    let inc = cumulative_to_incremental(triangle);
    let schema = PtfDesignSchema::build(
        n,
        &options.accident,
        &options.development,
        &options.calendar,
    )?;
    let p = schema.param_names.len();

    // Assemble observed rows for all strictly positive observed incremental cells.
    let mut obs_cells: Vec<(usize, usize)> = Vec::new();
    let mut x_flat: Vec<f64> = Vec::new();
    let mut y_vec: Vec<f64> = Vec::new();
    let mut w_vec: Vec<f64> = Vec::new();
    let mut row_buf = vec![0.0; p];

    for i in 0..n {
        let log_exp = options.exposure.as_ref().map_or(0.0, |e| e[i].ln());
        for j in 0..(n - i) {
            let val = inc[i * n + j];
            if !(val > 0.0) || !val.is_finite() {
                return Err(ValidationError::InvalidParameter(
                    "PTF log-linear regression requires strictly positive observed incremental losses",
                ));
            }
            schema.fill_row(i, j, None, &mut row_buf);
            x_flat.extend_from_slice(&row_buf);
            y_vec.push(val.ln() - log_exp);
            w_vec.push(options.dev_weights.as_ref().map_or(1.0, |w| w[j]));
            obs_cells.push((i, j));
        }
    }

    let n_obs = obs_cells.len();
    if n_obs <= p {
        return Err(ValidationError::InvalidParameter(
            "PTF model has non-positive residual degrees of freedom",
        ));
    }
    let dof = n_obs - p;

    let sol = weighted_least_squares_qr(&x_flat, &y_vec, &w_vec, n_obs, p)?;

    // Weighted residual sum of squares and R^2.
    let w_sum: f64 = w_vec.iter().sum();
    let y_bar = y_vec
        .iter()
        .zip(w_vec.iter())
        .map(|(&y, &w)| w * y)
        .sum::<f64>()
        / w_sum;

    let mut rss = 0.0;
    let mut tss = 0.0;
    for (row_idx, &y_val) in y_vec.iter().enumerate() {
        let mut eta = 0.0;
        for col in 0..p {
            eta += x_flat[row_idx * p + col] * sol.beta[col];
        }
        let resid = y_val - eta;
        rss += w_vec[row_idx] * resid * resid;
        let dy = y_val - y_bar;
        tss += w_vec[row_idx] * dy * dy;
    }

    let sigma_sq = (rss / dof as f64).max(0.0);
    let sigma = sigma_sq.sqrt();
    let r_squared = if tss > 0.0 {
        (1.0 - rss / tss).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let rss_floor = (rss / n_obs as f64).max(1e-30);
    let sum_ln_w: f64 = w_vec.iter().map(|w| w.ln()).sum();
    let aic = n_obs as f64 * rss_floor.ln() - sum_ln_w + 2.0 * p as f64;
    let bic = n_obs as f64 * rss_floor.ln() - sum_ln_w + (p as f64) * (n_obs as f64).ln();

    let cov: Vec<f64> = sol.xtwx_inv.iter().map(|&v| sigma_sq * v).collect();
    let coefficient_se: Vec<f64> = (0..p).map(|k| cov[k * p + k].max(0.0).sqrt()).collect();

    // Extract implied incremental calendar-year inflation rates iota_t for t = 1..n-1.
    let calendar_inflation_rates = schema.calendar_rates(&sol.beta, n);

    // Compute fitted incremental values across full n × n grid and forecast future cells.
    let mut fitted_incremental = vec![0.0; n * n];
    let mut future_rows: Vec<(usize, usize, f64, f64, f64, Vec<f64>)> = Vec::new();
    let mut latest_cum = vec![0.0; n];

    for i in 0..n {
        let log_exp = options.exposure.as_ref().map_or(0.0, |e| e[i].ln());
        latest_cum[i] = triangle.get(i, n - 1 - i);
        for j in 0..n {
            let w_j = options.dev_weights.as_ref().map_or(1.0, |w| w[j]);
            let mut x_cell = vec![0.0; p];
            let extra_cal = schema.fill_row(i, j, options.future_calendar_rate, &mut x_cell);
            let mut eta = log_exp + extra_cal;
            for col in 0..p {
                eta += x_cell[col] * sol.beta[col];
            }
            let v_proc = sigma_sq / w_j;
            let v_est = quad_form(&x_cell, &cov, p).max(0.0);
            let mean_var = match options.mean_mode {
                LogNormalMeanMode::ProcessVariance => v_proc,
                LogNormalMeanMode::FullPredictive => v_proc + v_est,
            };
            let mu = (eta + 0.5 * mean_var).exp();
            if !mu.is_finite() {
                return Err(ValidationError::NotFinite);
            }
            fitted_incremental[i * n + j] = mu;

            if i + j >= n {
                future_rows.push((i, j, mu, v_proc, v_est, x_cell));
            }
        }
    }

    // Per-origin and aggregate reserves + exact analytical log-normal MSEP.
    let mut reserves = vec![0.0; n];
    let mut origin_msep = vec![0.0; n];
    for &(i, _j, mu, _vp, _ve, _) in &future_rows {
        reserves[i] += mu;
    }

    // Precompute Cov * x_b for every future cell so pairwise covariances run in O(N_fut^2 * p).
    let n_fut = future_rows.len();
    let mut cov_x: Vec<Vec<f64>> = Vec::with_capacity(n_fut);
    for (_, _, _, _, _, xb) in &future_rows {
        let mut v = vec![0.0; p];
        for r in 0..p {
            let mut acc = 0.0;
            for c in 0..p {
                acc += cov[r * p + c] * xb[c];
            }
            v[r] = acc;
        }
        cov_x.push(v);
    }

    let mut total_msep = 0.0;
    for a in 0..n_fut {
        let (ia, _ja, mu_a, vp_a, v_est_a, ref xa) = future_rows[a];
        let var_a = mu_a * mu_a * (vp_a + v_est_a).exp_m1();
        if !var_a.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        origin_msep[ia] += var_a;
        total_msep += var_a;

        for b in (a + 1)..n_fut {
            let (ib, _jb, mu_b, _vp_b, _ve_b, _) = future_rows[b];
            let mut cov_eta = 0.0;
            let cvb = &cov_x[b];
            for r in 0..p {
                cov_eta += xa[r] * cvb[r];
            }
            let cov_y = mu_a * mu_b * cov_eta.exp_m1();
            if !cov_y.is_finite() {
                return Err(ValidationError::NotFinite);
            }
            total_msep += 2.0 * cov_y;
            if ia == ib {
                origin_msep[ia] += 2.0 * cov_y;
            }
        }
    }

    if !total_msep.is_finite() {
        return Err(ValidationError::NotFinite);
    }

    let ultimates: Vec<f64> = latest_cum
        .iter()
        .zip(reserves.iter())
        .map(|(&l, &r)| l + r)
        .collect();
    let standard_error: Vec<f64> = origin_msep.iter().map(|v| v.max(0.0).sqrt()).collect();
    let total_reserve: f64 = reserves.iter().sum();
    let total_msep = total_msep.max(0.0);
    let total_standard_error = total_msep.sqrt();
    let total_cv = if total_reserve > 0.0 {
        total_standard_error / total_reserve
    } else {
        0.0
    };

    Ok(PtfResult {
        param_names: schema.param_names,
        coefficients: sol.beta,
        coefficient_se,
        coefficient_covariance: cov,
        calendar_inflation_rates,
        sigma_sq,
        sigma,
        n_observed: n_obs,
        n_params: p,
        degrees_of_freedom: dof,
        r_squared,
        aic,
        bic,
        fitted_incremental,
        reserves,
        ultimates,
        standard_error,
        total_reserve,
        total_msep,
        total_standard_error,
        total_cv,
    })
}

struct PtfDesignSchema {
    n: usize,
    accident: AccidentSpec,
    development: DevelopmentSpec,
    calendar: CalendarSpec,
    param_names: Vec<String>,
    dev_offset: usize,
    cal_offset: usize,
}

impl PtfDesignSchema {
    fn build(
        n: usize,
        accident: &AccidentSpec,
        development: &DevelopmentSpec,
        calendar: &CalendarSpec,
    ) -> Result<Self, ValidationError> {
        let mut names = vec!["intercept".to_string()];

        match accident {
            AccidentSpec::Constant => {}
            AccidentSpec::PiecewiseLinear { knots } => {
                validate_knots(knots, n - 1, "accident knots")?;
                names.push("alpha_slope_0".to_string());
                for &k in knots {
                    names.push(format!("alpha_knot_{k}"));
                }
            }
            AccidentSpec::PerOrigin => {
                for i in 1..n {
                    names.push(format!("alpha_{i}"));
                }
            }
        }

        let dev_offset = names.len();
        match development {
            DevelopmentSpec::PerPeriod => {
                for j in 1..n {
                    names.push(format!("gamma_{j}"));
                }
            }
            DevelopmentSpec::PiecewiseLinear { knots } => {
                validate_knots(knots, n - 1, "development knots")?;
                names.push("gamma_slope_0".to_string());
                for &k in knots {
                    names.push(format!("gamma_knot_{k}"));
                }
            }
        }

        let cal_offset = names.len();
        match calendar {
            CalendarSpec::None => {}
            CalendarSpec::ConstantRate => {
                names.push("iota_0".to_string());
            }
            CalendarSpec::PiecewiseLinear { knots } => {
                validate_knots(knots, n - 1, "calendar knots")?;
                names.push("iota_0".to_string());
                for &k in knots {
                    names.push(format!("iota_knot_{k}"));
                }
            }
            CalendarSpec::ChangePointsOnly { knots } => {
                validate_knots(knots, n - 1, "calendar change-point knots")?;
                for &k in knots {
                    names.push(format!("iota_knot_{k}"));
                }
            }
        }

        Ok(Self {
            n,
            accident: accident.clone(),
            development: development.clone(),
            calendar: calendar.clone(),
            param_names: names,
            dev_offset,
            cal_offset,
        })
    }

    /// Populate design row `out` for cell `(i, j)`. Returns any deterministic
    /// future calendar offset (when `future_calendar_rate` overrides `t > n - 1`).
    fn fill_row(
        &self,
        i: usize,
        j: usize,
        future_calendar_rate: Option<f64>,
        out: &mut [f64],
    ) -> f64 {
        out.fill(0.0);
        out[0] = 1.0;

        // Accident direction.
        match &self.accident {
            AccidentSpec::Constant => {}
            AccidentSpec::PiecewiseLinear { knots } => {
                out[1] = i as f64;
                for (m, &k) in knots.iter().enumerate() {
                    if i > k {
                        out[2 + m] = (i - k) as f64;
                    }
                }
            }
            AccidentSpec::PerOrigin => {
                if i > 0 {
                    out[i] = 1.0;
                }
            }
        }

        // Development direction: sum_{k=1}^j gamma_k or piecewise linear in j.
        match &self.development {
            DevelopmentSpec::PerPeriod => {
                // Parameter gamma_k represents incremental change from k-1 to k,
                // so cumulative sum_{k=1}^j gamma_k places 1.0 in columns 1..=j.
                for k in 1..=j {
                    out[self.dev_offset + (k - 1)] = 1.0;
                }
            }
            DevelopmentSpec::PiecewiseLinear { knots } => {
                out[self.dev_offset] = j as f64;
                for (m, &k) in knots.iter().enumerate() {
                    if j > k {
                        out[self.dev_offset + 1 + m] = (j - k) as f64;
                    }
                }
            }
        }

        // Calendar direction: t = i + j.
        let t = i + j;
        let last_obs_t = self.n - 1;
        let (t_model, extra_offset) = match future_calendar_rate {
            Some(rate) if t > last_obs_t => (last_obs_t, rate * (t - last_obs_t) as f64),
            _ => (t, 0.0),
        };

        match &self.calendar {
            CalendarSpec::None => {}
            CalendarSpec::ConstantRate => {
                out[self.cal_offset] = t_model as f64;
            }
            CalendarSpec::PiecewiseLinear { knots } => {
                out[self.cal_offset] = t_model as f64;
                for (m, &k) in knots.iter().enumerate() {
                    if t_model > k {
                        out[self.cal_offset + 1 + m] = (t_model - k) as f64;
                    }
                }
            }
            CalendarSpec::ChangePointsOnly { knots } => {
                for (m, &k) in knots.iter().enumerate() {
                    if t_model > k {
                        out[self.cal_offset + m] = (t_model - k) as f64;
                    }
                }
            }
        }
        extra_offset
    }

    fn calendar_rates(&self, beta: &[f64], n: usize) -> Vec<f64> {
        let mut rates = vec![0.0; n.saturating_sub(1)];
        for t in 1..n {
            // Incremental calendar trend from t-1 to t.
            let rate = match &self.calendar {
                CalendarSpec::None => 0.0,
                CalendarSpec::ConstantRate => beta[self.cal_offset],
                CalendarSpec::PiecewiseLinear { knots } => {
                    let mut r = beta[self.cal_offset];
                    for (m, &k) in knots.iter().enumerate() {
                        if t > k {
                            r += beta[self.cal_offset + 1 + m];
                        }
                    }
                    r
                }
                CalendarSpec::ChangePointsOnly { knots } => {
                    let mut r = 0.0;
                    for (m, &k) in knots.iter().enumerate() {
                        if t > k {
                            r += beta[self.cal_offset + m];
                        }
                    }
                    r
                }
            };
            rates[t - 1] = rate;
        }
        rates
    }
}

fn validate_knots(
    knots: &[usize],
    max_exclusive: usize,
    _ctx: &str,
) -> Result<(), ValidationError> {
    let mut prev = 0usize;
    for &k in knots {
        if k == 0 || k >= max_exclusive {
            return Err(ValidationError::InvalidParameter(
                "PTF knot index must lie strictly inside (0, n - 1)",
            ));
        }
        if k <= prev {
            return Err(ValidationError::InvalidParameter(
                "PTF knots must be strictly increasing",
            ));
        }
        prev = k;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Triangle;

    const NAN: f64 = f64::NAN;

    #[test]
    fn ptf_recovers_exact_inflation_shock_parameters_and_future_reserves() {
        // Synthetic 8x8 incremental triangle with known development curve and a
        // sharp calendar-year inflation shock at calendar year t = 4:
        //   iota_t = 0.04 for t <= 4, jumping to 0.14 (+0.10 shock) for t > 4.
        let n = 8usize;
        let alpha0 = 10.0_f64;
        let dev_cum = [0.0, 0.35, 0.15, -0.25, -0.70, -1.20, -1.75, -2.30];
        let iota_pre = 0.04_f64;
        let iota_shift = 0.10_f64;
        let knot = 4usize;

        let mut cum_data = vec![NAN; n * n];
        let mut true_reserve = 0.0;
        for i in 0..n {
            let mut running = 0.0;
            for j in 0..n {
                let t = i + j;
                let cal = iota_pre * t as f64 + iota_shift * t.saturating_sub(knot) as f64;
                let inc = (alpha0 + dev_cum[j] + cal).exp();
                if i + j < n {
                    running += inc;
                    cum_data[i * n + j] = running;
                } else {
                    true_reserve += inc;
                }
            }
        }
        let tri = Triangle::new(n, n, cum_data);
        let opts = PtfOptions {
            accident: AccidentSpec::Constant,
            development: DevelopmentSpec::PerPeriod,
            calendar: CalendarSpec::PiecewiseLinear { knots: vec![knot] },
            ..PtfOptions::default()
        };
        let res = fit_ptf(&tri.view(), &opts).expect("ptf");

        assert!(res.sigma < 1e-10, "sigma={}", res.sigma);
        assert!((res.calendar_inflation_rates[0] - iota_pre).abs() < 1e-10);
        assert!((res.calendar_inflation_rates[knot - 1] - iota_pre).abs() < 1e-10);
        assert!((res.calendar_inflation_rates[knot] - (iota_pre + iota_shift)).abs() < 1e-10);
        assert!(
            (res.total_reserve - true_reserve).abs() / true_reserve < 1e-9,
            "got {} expected {}",
            res.total_reserve,
            true_reserve
        );
    }
}
