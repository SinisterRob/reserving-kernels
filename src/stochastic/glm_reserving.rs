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
//! * For [`GlmDesign::CrossClassified`] log-link models, every IRLS weight
//!   matrix `W_{i,j} = mu_{i,j}^{2 - p} = a_i b_j` is strictly rank-1 on the
//!   upper triangle, so the Schur complement of `X^T W X` in cumulative
//!   development coordinates is symmetric positive-definite tridiagonal.
//!   - **ODP (`p = 1`):** exact closed-form volume-weighted chain-ladder MLE
//!     and tridiagonal Green's-function inverse covariance, hat leverages, and
//!     analytic MSEP in `O(n^2)` time and `O(n^2)` space.
//!   - **Gamma / Tweedie (`p in (1, 2]`):** exact `O(K n^2)` rank-1 tridiagonal
//!     IRLS (with single-factorization constant-weight reuse for Gamma) and
//!     `O(n^2)` Green's-function covariance, leverages, and MSEP.
//!   - **Log-normal:** exact `O(n^2)` unit-weight tridiagonal OLS, covariance,
//!     and leverages with `O(1)`-per-pair covariance lookup for exact MSEP.
//! * For parametric [`GlmDesign::Hoerl`] / [`GlmDesign::HoerlWithCalendarSplines`]
//!   designs (and degenerate cross-classified fallbacks), each IRLS step solves
//!   the weighted normal system via column-major Householder QR with
//!   deviance-monotone step-halving (reusing stored Householder reflectors across
//!   iterations for constant-weight Gamma).

use crate::linalg::{
    QrWorkspace, qr_cov_and_leverages_from_factored, qr_factorize_unit_cm_store_reflectors,
    qr_solve_stored_unit_into, quad_form, weighted_least_squares_qr_cm_beta_into,
    weighted_least_squares_qr_cm_cov_and_leverages,
};
use crate::types::TriangleView;
use crate::validate::{MonotonicityPolicy, ValidationError, validate_positive_finite_slice};

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
#[non_exhaustive]
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
    /// Number of IRLS iterations performed (`1` for closed-form `LogNormal` and exact `Odp`).
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

/// Stack-backed `f64` scratch buffer avoiding heap allocations for small/medium
/// triangles (`len <= CAP`) while transparently falling back to `Vec<f64>` for
/// larger sizes. Uses zero `unsafe` code.
enum SmallBuf<const CAP: usize> {
    Stack { len: usize, data: [f64; CAP] },
    Heap(Vec<f64>),
}

impl<const CAP: usize> SmallBuf<CAP> {
    #[inline]
    fn zeroed(len: usize) -> Self {
        if len <= CAP {
            Self::Stack {
                len,
                data: [0.0; CAP],
            }
        } else {
            Self::Heap(vec![0.0; len])
        }
    }

    #[inline]
    fn filled(len: usize, value: f64) -> Self {
        if len <= CAP {
            Self::Stack {
                len,
                data: [value; CAP],
            }
        } else {
            Self::Heap(vec![value; len])
        }
    }

    #[inline]
    fn as_slice(&self) -> &[f64] {
        match self {
            Self::Stack { len, data } => &data[..*len],
            Self::Heap(v) => v.as_slice(),
        }
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [f64] {
        match self {
            Self::Stack { len, data } => &mut data[..*len],
            Self::Heap(v) => v.as_mut_slice(),
        }
    }
}

/// Fit a GLM reserving model to a cumulative loss triangle.
pub fn fit_glm(
    triangle: &TriangleView<'_>,
    options: &GlmOptions,
) -> Result<GlmReservingResult, ValidationError> {
    let strict = options.monotonicity_policy == MonotonicityPolicy::Strict;
    let is_cross = matches!(options.design, GlmDesign::CrossClassified);
    let is_odp_family = match options.family {
        GlmFamily::Odp => true,
        GlmFamily::Tweedie { power } => (power - 1.0).abs() < 1e-12,
        _ => false,
    };

    if triangle.data.len() != triangle.n_origins * triangle.n_devs {
        return Err(ValidationError::ShapeMismatch);
    }
    if triangle.n_origins != triangle.n_devs {
        return Err(ValidationError::NotSquare);
    }
    let n = triangle.n_origins;
    let mut latest_buf = SmallBuf::<16>::zeroed(n);
    let mut col_sum_buf = SmallBuf::<16>::zeroed(if is_cross && is_odp_family { n } else { 0 });

    // Single-pass triangle validation + latest diagonal + column cumulative sums (for ODP fast path).
    validate_triangle_single_pass_into(
        triangle,
        strict,
        latest_buf.as_mut_slice(),
        if is_cross && is_odp_family {
            Some(col_sum_buf.as_mut_slice())
        } else {
            None
        },
    )?;
    if n < 3 {
        return Err(ValidationError::InvalidParameter(
            "GLM reserving requires at least a 3x3 triangle",
        ));
    }
    if let GlmFamily::Tweedie { power } = options.family
        && (!power.is_finite() || !(1.0..=2.0).contains(&power))
    {
        return Err(ValidationError::InvalidParameter(
            "Tweedie variance power must lie in [1.0, 2.0]",
        ));
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

    if is_cross {
        if is_odp_family {
            if let Some(res) = try_fit_cross_classified_odp(
                triangle,
                options,
                latest_buf.as_slice(),
                col_sum_buf.as_slice(),
            ) {
                return Ok(res);
            }
        } else if options.family == GlmFamily::LogNormal {
            if let Some(res) =
                try_fit_cross_classified_log_normal(triangle, options, latest_buf.as_slice())?
            {
                return Ok(res);
            }
        } else if let Some(res) =
            try_fit_cross_classified_tweedie(triangle, options, latest_buf.as_slice())?
        {
            return Ok(res);
        }
    }

    fit_glm_general_qr(triangle, options, latest_buf.as_slice())
}

/// Single-pass triangle validation preserving exact error precedence
/// (`ShapeMismatch` → `NotSquare` → `NotFinite`/`Negative` → `NonMonotoneCumulative`),
/// writing `latest` and optional `col_sum` in-place (zero return-value copies).
fn validate_triangle_single_pass_into(
    t: &TriangleView<'_>,
    strict: bool,
    latest_out: &mut [f64],
    mut col_sum_out: Option<&mut [f64]>,
) -> Result<(), ValidationError> {
    if t.data.len() != t.n_origins * t.n_devs {
        return Err(ValidationError::ShapeMismatch);
    }
    if t.n_origins != t.n_devs {
        return Err(ValidationError::NotSquare);
    }
    let n = t.n_origins;
    let mut first_non_monotone: Option<ValidationError> = None;

    for i in 0..n {
        let row_len = n - i;
        let row = &t.data[i * n..i * n + row_len];
        let mut prev = 0.0_f64;
        if let Some(cs) = col_sum_out.as_deref_mut() {
            for j in 0..row_len {
                let v = row[j];
                if !v.is_finite() {
                    return Err(ValidationError::NotFinite);
                }
                if v < 0.0 {
                    return Err(ValidationError::Negative);
                }
                if strict && j > 0 && v < prev && first_non_monotone.is_none() {
                    first_non_monotone = Some(ValidationError::NonMonotoneCumulative {
                        origin: i,
                        dev: j,
                        previous: prev,
                        value: v,
                    });
                }
                cs[j] += v;
                prev = v;
            }
        } else {
            for j in 0..row_len {
                let v = row[j];
                if !v.is_finite() {
                    return Err(ValidationError::NotFinite);
                }
                if v < 0.0 {
                    return Err(ValidationError::Negative);
                }
                if strict && j > 0 && v < prev && first_non_monotone.is_none() {
                    first_non_monotone = Some(ValidationError::NonMonotoneCumulative {
                        origin: i,
                        dev: j,
                        previous: prev,
                        value: v,
                    });
                }
                prev = v;
            }
        }
        latest_out[i] = prev;
    }

    if let Some(err) = first_non_monotone {
        return Err(err);
    }
    Ok(())
}

#[inline]
fn validate_observed_increment(
    y: f64,
    family: GlmFamily,
    policy: MonotonicityPolicy,
) -> Result<(), ValidationError> {
    if !y.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    match family {
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
            if y < 0.0 && policy == MonotonicityPolicy::Strict {
                return Err(ValidationError::Negative);
            }
        }
    }
    Ok(())
}

/// Fast allocation-exact formatter for `"origin_{idx}"` / `"dev_{idx}"`,
/// avoiding `core::fmt` dispatch and `String` growth reallocations.
#[inline]
fn indexed_name(prefix: &str, idx: usize) -> String {
    if idx < 10 {
        let mut s = String::with_capacity(prefix.len() + 1);
        s.push_str(prefix);
        s.push((b'0' + idx as u8) as char);
        return s;
    }
    let mut buf = [0u8; 20];
    let mut v = idx;
    let mut pos = buf.len();
    loop {
        pos -= 1;
        buf[pos] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    let digits = &buf[pos..];
    let mut s = String::with_capacity(prefix.len() + digits.len());
    s.push_str(prefix);
    for &b in digits {
        s.push(b as char);
    }
    s
}

fn cross_classified_param_names(n: usize) -> Vec<String> {
    let mut names = Vec::with_capacity(2 * n - 1);
    names.push(String::from("intercept"));
    for i in 1..n {
        names.push(indexed_name("origin_", i));
    }
    for j in 1..n {
        names.push(indexed_name("dev_", j));
    }
    names
}

// ===========================================================================
// Universal O(n) Tridiagonal Rank-1 Schur-Complement Factor for Cross-Classified
// GLMs (exact for any positive rank-1 weight matrix W_{i,j} = a_i * b_j).
// ===========================================================================

/// Compact single-slab `L D L^T` factorization of the `(n - 1) × (n - 1)`
/// tridiagonal Schur complement `K` of `X^T W X` for any strictly positive
/// rank-1 weight matrix `W_{i,j} = a_i * b_j` on an `n × n` upper triangle.
///
/// Stack-allocated for `n <= 16` (`7 * 16 - 4 = 108 <= 112` f64s), with automatic
/// heap fallback for larger triangles.
struct TridiagonalRank1Factor {
    n: usize,
    slab: SmallBuf<112>,
}

impl TridiagonalRank1Factor {
    fn new(n: usize) -> Self {
        debug_assert!(n >= 3);
        Self {
            n,
            slab: SmallBuf::<112>::zeroed(7 * n - 4),
        }
    }

    #[inline]
    fn offsets(n: usize) -> (usize, usize, usize, usize, usize, usize) {
        let m = n - 1;
        let off_inv_b = n;
        let off_inv_beta = 2 * n;
        let off_dt = 3 * n;
        let off_inv_dt = 3 * n + m;
        let off_gamma = 3 * n + 2 * m;
        let off_work = 3 * n + 3 * m - 1;
        (
            off_inv_b,
            off_inv_beta,
            off_dt,
            off_inv_dt,
            off_gamma,
            off_work,
        )
    }

    /// Factor `K = L D L^T` in `O(n)` time and zero allocations. Returns `false`
    /// if any weight is non-positive/non-finite or any pivot is ill-conditioned.
    fn refactor(&mut self, a: &[f64], b: &[f64]) -> bool {
        let n = self.n;
        let m = n - 1;
        let (off_inv_b, off_inv_beta, off_dt, off_inv_dt, off_gamma, off_work) = Self::offsets(n);
        let slab = self.slab.as_mut_slice();

        // Cumulative sums of b and reciprocals.
        let mut b_acc = 0.0;
        for j in 0..n {
            let bj = b[j];
            if !(bj > 0.0) || !bj.is_finite() {
                return false;
            }
            b_acc += bj;
            if !(b_acc > 0.0) || !b_acc.is_finite() {
                return false;
            }
            slab[j] = b_acc;
            slab[off_inv_b + j] = 1.0 / b_acc;
            slab[off_inv_beta + j] = 1.0 / bj;
        }

        // Prefix/reverse sums A[k] = sum_{i=0}^{m-k} a_i stored in `work[0..n]`.
        let mut a_acc = 0.0;
        for k in (0..n).rev() {
            let ai = a[m - k];
            if !(ai > 0.0) || !ai.is_finite() {
                return false;
            }
            a_acc += ai;
            slab[off_work + k] = a_acc;
        }

        // Fused tridiagonal assembly and L D L^T sweep.
        let mut prev_off = 0.0;
        let mut prev_gamma = 0.0;
        for k in 1..n {
            let idx = k - 1;
            let off = if k < m {
                slab[off_work + k + 1] * slab[off_inv_beta + k + 1]
            } else {
                0.0
            };
            let d_idx =
                slab[off_work + k] * slab[off_inv_beta + k] + off - a[m - k] * slab[off_inv_b + k];
            let dt = if idx == 0 {
                d_idx
            } else {
                d_idx - prev_off * prev_gamma
            };
            if !dt.is_finite() || dt <= 1e-14 * d_idx.abs().max(1e-30) {
                return false;
            }
            let inv_dt = 1.0 / dt;
            slab[off_dt + idx] = dt;
            slab[off_inv_dt + idx] = inv_dt;
            if k < m {
                let g = off * inv_dt;
                if !g.is_finite() {
                    return false;
                }
                slab[off_gamma + idx] = g;
                prev_off = off;
                prev_gamma = g;
            }
        }
        true
    }

    /// Solve `(X_{uv}^T W X_{uv}) (u, v) = (ru, rv)` in `O(n)` time and zero allocations.
    fn solve_uv(
        &mut self,
        a: &[f64],
        ru: &[f64],
        rv: &[f64],
        u_out: &mut [f64],
        v_out: &mut [f64],
    ) {
        let n = self.n;
        let m = n - 1;
        let (off_inv_b, off_inv_beta, _, off_inv_dt, off_gamma, off_work) = Self::offsets(n);
        let slab = self.slab.as_mut_slice();

        // Forward solve L y = r_tilde into `work[1..n]`.
        let mut y_prev = 0.0;
        for idx in 0..m {
            let k = idx + 1;
            let rt = if k < m {
                rv[k] * slab[off_inv_beta + k]
                    - rv[k + 1] * slab[off_inv_beta + k + 1]
                    - ru[m - k] * slab[off_inv_b + k]
            } else {
                rv[m] * slab[off_inv_beta + m] - ru[0] * slab[off_inv_b + m]
            };
            let yk = if idx == 0 {
                rt
            } else {
                rt + slab[off_gamma + idx - 1] * y_prev
            };
            slab[off_work + k] = yk;
            y_prev = yk;
        }

        // Backward solve L^T T = D^{-1} y in-place in `work[0..n]` (`T[0] = 0`).
        slab[off_work] = 0.0;
        let mut t_next = slab[off_work + m] * slab[off_inv_dt + m - 1];
        slab[off_work + m] = t_next;
        for idx in (0..(m - 1)).rev() {
            let tk =
                slab[off_work + idx + 1] * slab[off_inv_dt + idx] + slab[off_gamma + idx] * t_next;
            slab[off_work + idx + 1] = tk;
            t_next = tk;
        }

        // Recover v and u.
        v_out[0] = 0.0;
        for j in 1..n {
            v_out[j] = (slab[off_work + j] - slab[off_work + j - 1]) * slab[off_inv_beta + j];
        }
        for i in 0..n {
            let ki = m - i;
            u_out[i] = (ru[i] / a[i] - slab[off_work + ki]) * slab[off_inv_b + ki];
        }
    }

    /// Fill the symmetric `n × n` zero-bordered Green's function matrix `G = K^{-1}`
    /// (`g_mat[a * n + b] = G_{a,b}` for `a, b in 1..n`, and `0.0` when `a == 0 || b == 0`)
    /// in-place in `O(n^2)` time (`m(m-1)/2` multiplications, zero return-value copy).
    fn compute_g_mat_into(&self, g_mat: &mut [f64]) {
        let n = self.n;
        let m = n - 1;
        let (_, _, _, off_inv_dt, off_gamma, _) = Self::offsets(n);
        let slab = self.slab.as_slice();

        g_mat[m * n + m] = slab[off_inv_dt + m - 1];
        for idx in (0..(m - 1)).rev() {
            let b = idx + 1;
            let g = slab[off_gamma + idx];
            g_mat[b * n + b] = slab[off_inv_dt + idx] + g * g * g_mat[(b + 1) * n + (b + 1)];
        }

        for b in 2..n {
            let mut val = g_mat[b * n + b];
            for a in (1..b).rev() {
                val *= slab[off_gamma + a - 1];
                g_mat[a * n + b] = val;
                g_mat[b * n + a] = val;
            }
        }
    }

    /// Compute all `N = n(n+1)/2` exact hat-matrix leverages in `O(n^2)` time.
    fn compute_leverages(&mut self, a: &[f64], b: &[f64], g_mat: &[f64]) -> Vec<f64> {
        let n = self.n;
        let m = n - 1;
        let (off_inv_b, off_inv_beta, _, _, _, off_work) = Self::offsets(n);
        let slab = self.slab.as_mut_slice();

        // Precompute w_j^T G w_j into `work[1..n]`.
        for j in 1..n {
            let ib = slab[off_inv_beta + j];
            slab[off_work + j] = (g_mat[j * n + j] - 2.0 * g_mat[j * n + (j - 1)]
                + g_mat[(j - 1) * n + (j - 1)])
                * ib
                * ib;
        }

        let n_obs = n * (n + 1) / 2;
        let mut leverages = Vec::with_capacity(n_obs);
        for origin in 0..n {
            let ki = m - origin;
            let inv_b_ki = slab[off_inv_b + ki];
            let base_phi = inv_b_ki / a[origin];
            let g_row = &g_mat[ki * n..ki * n + n];
            let z_g_z = g_row[ki] * inv_b_ki * inv_b_ki;
            let ai = a[origin];

            leverages.push((ai * b[0] * (base_phi + z_g_z)).clamp(0.0, 1.0));
            for dev in 1..=ki {
                let w_g_z = -(g_row[dev] - g_row[dev - 1]) * inv_b_ki * slab[off_inv_beta + dev];
                let y_g_y = slab[off_work + dev] + 2.0 * w_g_z + z_g_z;
                leverages.push((ai * b[dev] * (base_phi + y_g_y)).clamp(0.0, 1.0));
            }
        }
        leverages
    }

    /// Compute the full `(2n - 1) × (2n - 1)` symmetric covariance matrix
    /// `scale * (X^T W X)^{-1}` in corner coding `(intercept, origin_1.., dev_1..)`
    /// in exact `O(n^2)` time.
    fn compute_cov_scaled(&mut self, inv_a: &[f64], g_mat: &[f64], scale: f64) -> Vec<f64> {
        let n = self.n;
        let m = n - 1;
        let q = 2 * n - 1;
        let (off_inv_b, off_inv_beta, _, _, _, off_work) = Self::offsets(n);
        let slab = self.slab.as_mut_slice();

        let inv_bm = slab[off_inv_b + m];
        let g_mm = g_mat[m * n + m];
        let c_u00 = inv_bm * inv_a[0] + g_mm * inv_bm * inv_bm;

        // Store `c_u0[i] = Cov_unscaled(u_i, u_0)` in `work[0..n]`.
        slab[off_work] = c_u00;
        for i in 1..n {
            let ki = m - i;
            slab[off_work + i] = g_mat[ki * n + m] * slab[off_inv_b + ki] * inv_bm;
        }

        let mut cov = vec![0.0; q * q];
        cov[0] = scale * c_u00;

        // Intercept / origin block.
        for i in 1..n {
            let c_ui0 = slab[off_work + i];
            let v0i = scale * (c_ui0 - c_u00);
            cov[i * q] = v0i;
            cov[i] = v0i;

            let ki = m - i;
            let inv_b_ki = slab[off_inv_b + ki];
            let diag_i = inv_b_ki * inv_a[i];
            for r in i..n {
                let kr = m - r;
                let diag = if i == r { diag_i } else { 0.0 };
                let c_uir = diag + g_mat[ki * n + kr] * inv_b_ki * slab[off_inv_b + kr];
                let val = scale * (c_uir - c_ui0 - slab[off_work + r] + c_u00);
                cov[i * q + r] = val;
                cov[r * q + i] = val;
            }
        }

        // Cross block between (intercept, origin_1..) and dev_1..
        for j in 1..n {
            let col = n + j - 1;
            let inv_bj = slab[off_inv_beta + j];
            let c_u0vj = -(g_mat[m * n + j] - g_mat[m * n + (j - 1)]) * inv_bm * inv_bj;
            let val0 = scale * c_u0vj;
            cov[col] = val0;
            cov[col * q] = val0;

            for i in 1..n {
                let ki = m - i;
                let c_uivj =
                    -(g_mat[ki * n + j] - g_mat[ki * n + (j - 1)]) * slab[off_inv_b + ki] * inv_bj;
                let val = scale * (c_uivj - c_u0vj);
                cov[i * q + col] = val;
                cov[col * q + i] = val;
            }
        }

        // Development block dev_1.. × dev_1..
        for a_idx in 1..n {
            let row = n + a_idx - 1;
            let inv_ba = slab[off_inv_beta + a_idx];
            for b_idx in a_idx..n {
                let col = n + b_idx - 1;
                let c_vab = (g_mat[a_idx * n + b_idx]
                    - g_mat[(a_idx - 1) * n + b_idx]
                    - g_mat[a_idx * n + (b_idx - 1)]
                    + g_mat[(a_idx - 1) * n + (b_idx - 1)])
                    * inv_ba
                    * slab[off_inv_beta + b_idx];
                let val = scale * c_vab;
                cov[row * q + col] = val;
                cov[col * q + row] = val;
            }
        }

        cov
    }

    /// Exact unscaled parameter estimation quadratic form `g_i^T (X^T W X)^{-1} g_i`
    /// for origin `i` under arbitrary rank-1 future expectations `mu_{i,j} = au_i * bv_j`
    /// in `O(n)` time and zero allocations.
    fn origin_param_quad_general(
        &self,
        inv_a: &[f64],
        origin: usize,
        r_i: f64,
        au_i: f64,
        bv: &[f64],
    ) -> f64 {
        if origin == 0 {
            return 0.0;
        }
        let n = self.n;
        let m = n - 1;
        let ki = m - origin;
        let (off_inv_b, off_inv_beta, _, off_inv_dt, off_gamma, _) = Self::offsets(n);
        let slab = self.slab.as_slice();

        let mut quad = (r_i * r_i) * slab[off_inv_b + ki] * inv_a[origin];
        let mut z_prev = 0.0;
        let start_idx = ki.saturating_sub(1);
        for idx in start_idx..m {
            let k = idx + 1;
            let gv_k = if k > ki {
                au_i * bv[k] * slab[off_inv_beta + k]
            } else {
                0.0
            };
            let gv_next = if k < m && k + 1 > ki {
                au_i * bv[k + 1] * slab[off_inv_beta + k + 1]
            } else {
                0.0
            };
            let gu_term = if m - k == origin {
                r_i * slab[off_inv_b + k]
            } else {
                0.0
            };
            let qt = gv_k - gv_next - gu_term;
            let z = if idx == start_idx {
                qt
            } else {
                qt + slab[off_gamma + idx - 1] * z_prev
            };
            quad += (z * z) * slab[off_inv_dt + idx];
            z_prev = z;
        }
        quad
    }

    /// Exact unscaled aggregate parameter estimation quadratic form
    /// `g_total^T (X^T W X)^{-1} g_total` in `O(n)` time and zero allocations.
    fn total_param_quad_general(
        &self,
        inv_a: &[f64],
        reserves: &[f64],
        au: &[f64],
        bv: &[f64],
    ) -> f64 {
        let n = self.n;
        let m = n - 1;
        let (off_inv_b, off_inv_beta, _, off_inv_dt, off_gamma, _) = Self::offsets(n);
        let slab = self.slab.as_slice();

        let mut quad = 0.0;
        for i in 1..n {
            let ki = m - i;
            quad += (reserves[i] * reserves[i]) * slab[off_inv_b + ki] * inv_a[i];
        }

        // `s_au_k = sum_{i = n - k}^{n - 1} au[i]` increases by `au[n - k]` each step.
        let mut s_au_k = 0.0;
        let mut z_prev = 0.0;
        for idx in 0..m {
            let k = idx + 1;
            s_au_k += au[n - k];
            let gv_k = s_au_k * bv[k] * slab[off_inv_beta + k];
            let gv_next = if k < m {
                let s_au_next = s_au_k + au[n - k - 1];
                s_au_next * bv[k + 1] * slab[off_inv_beta + k + 1]
            } else {
                0.0
            };
            let gu_term = reserves[m - k] * slab[off_inv_b + k];
            let qt = gv_k - gv_next - gu_term;
            let z = if idx == 0 {
                qt
            } else {
                qt + slab[off_gamma + idx - 1] * z_prev
            };
            quad += (z * z) * slab[off_inv_dt + idx];
            z_prev = z;
        }
        quad
    }

    #[inline]
    fn b_cum(&self, k: usize) -> f64 {
        self.slab.as_slice()[k]
    }

    #[inline]
    fn inv_b(&self, k: usize) -> f64 {
        self.slab.as_slice()[self.n + k]
    }

    #[inline]
    fn inv_beta(&self, j: usize) -> f64 {
        self.slab.as_slice()[2 * self.n + j]
    }

    #[inline]
    fn inv_d_tilde(&self, idx: usize) -> f64 {
        self.slab.as_slice()[3 * self.n + (self.n - 1) + idx]
    }

    #[inline]
    fn gamma(&self, idx: usize) -> f64 {
        self.slab.as_slice()[3 * self.n + 2 * (self.n - 1) + idx]
    }
}

// ===========================================================================
// Fast Path A: Exact O(n^2) Time & Space Cross-Classified ODP GLM Solver
// ===========================================================================

fn try_fit_cross_classified_odp(
    triangle: &TriangleView<'_>,
    options: &GlmOptions,
    latest: &[f64],
    col_sum: &[f64],
) -> Option<GlmReservingResult> {
    // Clamp guard thresholds matching `eta.clamp(-50.0, 50.0)` with margin:
    // exp(-49.0) and exp(49.0).
    const MIN_UNCLAMPED_MU: f64 = 5.242_885_663_363_464e-22;
    const MAX_UNCLAMPED_MU: f64 = 1.907_346_572_495_099_8e21;

    let n = triangle.n_origins;
    let m = n - 1;
    let q = 2 * n - 1;
    let n_obs = n * (n + 1) / 2;
    let dof = n_obs - q;

    // Validate latest > 0 and compute volume-weighted LDFs in O(n) from `col_sum` & `latest`.
    let mut cdf_buf = SmallBuf::<16>::zeroed(n);
    let cdf = cdf_buf.as_mut_slice();
    cdf[m] = 1.0;
    for k in (0..m).rev() {
        let den = col_sum[k] - latest[m - k];
        let num = col_sum[k + 1];
        if !(den > 0.0) || !den.is_finite() || !(num > 0.0) || !num.is_finite() {
            return None;
        }
        let ldf = num / den;
        if !(ldf > 1.0) || !ldf.is_finite() {
            return None;
        }
        cdf[k] = cdf[k + 1] * ldf;
        if !cdf[k].is_finite() || !(cdf[k] > 0.0) {
            return None;
        }
    }

    let mut pi_buf = SmallBuf::<16>::zeroed(n);
    let pi = pi_buf.as_mut_slice();
    let mut prev_b = 0.0;
    let mut min_pi = f64::INFINITY;
    let mut max_pi = 0.0_f64;
    for j in 0..n {
        let b_j = 1.0 / cdf[j];
        let pj = b_j - prev_b;
        if !(pj > 0.0) || !pj.is_finite() {
            return None;
        }
        pi[j] = pj;
        min_pi = min_pi.min(pj);
        max_pi = max_pi.max(pj);
        prev_b = b_j;
    }

    let mut u_buf = SmallBuf::<16>::zeroed(n);
    let mut inv_u_buf = SmallBuf::<16>::zeroed(n);
    let u_ult = u_buf.as_mut_slice();
    let inv_u = inv_u_buf.as_mut_slice();
    let mut min_u = f64::INFINITY;
    let mut max_u = 0.0_f64;
    for i in 0..n {
        let lat = latest[i];
        if !(lat > 0.0) || !lat.is_finite() {
            return None;
        }
        let ui = lat * cdf[m - i];
        if !(ui > 0.0) || !ui.is_finite() {
            return None;
        }
        u_ult[i] = ui;
        inv_u[i] = 1.0 / ui;
        min_u = min_u.min(ui);
        max_u = max_u.max(ui);
    }

    // Ensure no cell approaches the IRLS linear-predictor clamp `[-50, 50]` (no extra ln).
    let min_mu = min_u * min_pi;
    let max_mu = max_u * max_pi;
    if !(min_mu > MIN_UNCLAMPED_MU) || !(max_mu < MAX_UNCLAMPED_MU) {
        return None;
    }

    // Factor exact tridiagonal Schur complement in O(n) (zero heap alloc for n <= 16).
    let mut factor = TridiagonalRank1Factor::new(n);
    if !factor.refactor(u_ult, pi) {
        return None;
    }

    // Exact corner-coded coefficients beta_hat in O(n) (invariant means under exposure offset).
    let mut coefficients = Vec::with_capacity(q);
    let ln_u0 = u_ult[0].ln() - options.exposure.as_ref().map_or(0.0, |e| e[0].ln());
    let ln_pi0 = pi[0].ln();
    coefficients.push(ln_u0 + ln_pi0);
    for i in 1..n {
        let log_exp_i = options.exposure.as_ref().map_or(0.0, |e| e[i].ln());
        coefficients.push((u_ult[i].ln() - log_exp_i) - ln_u0);
    }
    for j in 1..n {
        coefficients.push(pi[j].ln() - ln_pi0);
    }

    // Compute exact symmetric Green's function G = K^{-1} in-place (stack-backed for n <= 12).
    let mut g_buf = SmallBuf::<144>::zeroed(n * n);
    factor.compute_g_mat_into(g_buf.as_mut_slice());
    let g_mat = g_buf.as_slice();

    // Single fused O(n^2) pass with zero inner-loop divisions:
    // fitted_incremental, leverages, deviance, and Pearson chi-square.
    let mut fitted_incremental = vec![0.0; n * n];
    let mut leverages = Vec::with_capacity(n_obs);
    let mut w_g_w_buf = SmallBuf::<16>::zeroed(n);
    let w_g_w = w_g_w_buf.as_mut_slice();
    for j in 1..n {
        let inv_pj = factor.inv_beta(j);
        w_g_w[j] = (g_mat[j * n + j] - 2.0 * g_mat[j * n + (j - 1)] + g_mat[(j - 1) * n + (j - 1)])
            * inv_pj
            * inv_pj;
    }

    let mut half_deviance = 0.0;
    let mut pearson_chi_sq = 0.0;
    for i in 0..n {
        let ki = m - i;
        let ui = u_ult[i];
        let inv_ui = inv_u[i];
        let row_off = i * n;
        let cum_row = &triangle.data[row_off..=(row_off + ki)];
        let fit_row = &mut fitted_incremental[row_off..row_off + n];
        let inv_b_ki = factor.inv_b(ki);
        let base_phi = inv_b_ki * inv_ui;
        let g_row = &g_mat[ki * n..ki * n + n];
        let z_g_z = g_row[ki] * inv_b_ki * inv_b_ki;

        for j in 0..=ki {
            let y = if j == 0 {
                cum_row[0]
            } else {
                cum_row[j] - cum_row[j - 1]
            };

            let mu = ui * pi[j];
            let inv_pj = factor.inv_beta(j);
            let inv_mu = inv_ui * inv_pj;
            fit_row[j] = mu;

            let diff = y - mu;
            pearson_chi_sq += (diff * diff) * inv_mu;
            let term = if y > 0.0 { y * (y * inv_mu).ln() } else { 0.0 };
            half_deviance += term - diff;

            let y_g_y = if j == 0 {
                z_g_z
            } else {
                let w_g_z = -(g_row[j] - g_row[j - 1]) * inv_b_ki * inv_pj;
                w_g_w[j] + 2.0 * w_g_z + z_g_z
            };
            leverages.push((mu * (base_phi + y_g_y)).clamp(0.0, 1.0));
        }
        for j in (ki + 1)..n {
            fit_row[j] = ui * pi[j];
        }
    }
    let deviance = 2.0 * half_deviance;

    let pearson_dispersion = (pearson_chi_sq / dof as f64).max(0.0);
    let deviance_dispersion = (deviance / dof as f64).max(0.0);

    // Exact full q × q scaled covariance matrix in O(n^2).
    let cov = factor.compute_cov_scaled(inv_u, g_mat, pearson_dispersion);
    let mut coefficient_se = Vec::with_capacity(q);
    for k in 0..q {
        coefficient_se.push(cov[k * q + k].max(0.0).sqrt());
    }

    // Closed-form O(n) origin reserves, ultimates, and analytic ODP MSEP.
    let mut reserves = Vec::with_capacity(n);
    let mut ultimates = Vec::with_capacity(n);
    let mut process_se = Vec::with_capacity(n);
    let mut parameter_se = Vec::with_capacity(n);
    let mut prediction_error = Vec::with_capacity(n);

    let g_mm = g_mat[m * n + m];
    let mut total_reserve = 0.0;
    let mut r_diag_sum = 0.0;
    let mut u_sum = 0.0;
    for i in 0..n {
        let ki = m - i;
        let ui = u_ult[i];
        let inv_ui = inv_u[i];
        u_sum += ui;
        let r_i = if i == 0 {
            0.0
        } else {
            ui * (1.0 - factor.b_cum(ki))
        };
        reserves.push(r_i);
        ultimates.push(latest[i] + r_i);
        total_reserve += r_i;

        let param_quad = if i == 0 {
            0.0
        } else if i < m {
            let inv_b = factor.inv_b(ki);
            let diag_term = (r_i * r_i) * inv_b * inv_ui;
            r_diag_sum += diag_term;
            diag_term
                + ui * ui
                    * (g_mat[ki * n + ki] * inv_b * inv_b - 2.0 * g_mat[ki * n + m] * inv_b + g_mm)
        } else {
            let diag_term = (r_i * r_i) * factor.inv_b(0) * inv_ui;
            r_diag_sum += diag_term;
            diag_term + ui * ui * g_mm
        };
        let proc_var = (pearson_dispersion * r_i).max(0.0);
        let param_var = (pearson_dispersion * param_quad).max(0.0);
        process_se.push(proc_var.sqrt());
        parameter_se.push(param_var.sqrt());
        prediction_error.push((proc_var + param_var).sqrt());
    }

    // Portfolio total parameter variance via single O(n) forward sweep.
    let mut total_quad = r_diag_sum;
    let mut z_prev = 0.0;
    for idx in 0..m {
        let k = idx + 1;
        let y_k = if k < m {
            -u_ult[m - k] * factor.inv_b(k)
        } else {
            u_sum - u_ult[0]
        };
        let z_k = if idx == 0 {
            y_k
        } else {
            y_k + factor.gamma(idx - 1) * z_prev
        };
        total_quad += (z_k * z_k) * factor.inv_d_tilde(idx);
        z_prev = z_k;
    }

    let total_proc_var = (pearson_dispersion * total_reserve).max(0.0);
    let total_param_var = (pearson_dispersion * total_quad).max(0.0);
    let total_prediction_error = (total_proc_var + total_param_var).sqrt();
    let total_cv = if total_reserve > 0.0 {
        total_prediction_error / total_reserve
    } else {
        0.0
    };

    Some(GlmReservingResult {
        family: options.family,
        param_names: cross_classified_param_names(n),
        coefficients,
        coefficient_se,
        coefficient_covariance: cov,
        iterations: 1,
        converged: true,
        n_observed: n_obs,
        n_params: q,
        degrees_of_freedom: dof,
        deviance,
        pearson_chi_sq,
        pearson_dispersion,
        deviance_dispersion,
        leverages,
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

// ===========================================================================
// Fast Path B: Exact O(K n^2) Tridiagonal Rank-1 IRLS for Cross-Classified
// Gamma & Tweedie (p in (1, 2])
// ===========================================================================

fn try_fit_cross_classified_tweedie(
    triangle: &TriangleView<'_>,
    options: &GlmOptions,
    latest: &[f64],
) -> Result<Option<GlmReservingResult>, ValidationError> {
    let n = triangle.n_origins;
    let q = 2 * n - 1;
    let n_obs = n * (n + 1) / 2;
    let dof = n_obs - q;
    let power = options.family.tweedie_power().expect("Tweedie family");
    let is_gamma = (power - 2.0).abs() < 1e-12;
    let one_mp = 1.0 - power;
    let two_mp = 2.0 - power;

    // Extract packed upper-triangle increments + precompute ln(y) once (stack-backed for n <= 15).
    let mut y_buf = SmallBuf::<128>::zeroed(n_obs);
    let mut ln_y_buf = SmallBuf::<128>::zeroed(n_obs);
    let y_obs = y_buf.as_mut_slice();
    let ln_y = ln_y_buf.as_mut_slice();
    let mut pos_sum = 0.0;
    let mut pos_cnt = 0usize;
    let mut obs_idx = 0usize;
    for i in 0..n {
        let row_len = n - i;
        let cum_row = &triangle.data[i * n..i * n + row_len];
        for j in 0..row_len {
            let y = if j == 0 {
                cum_row[0]
            } else {
                cum_row[j] - cum_row[j - 1]
            };
            validate_observed_increment(y, options.family, options.monotonicity_policy)?;
            y_obs[obs_idx] = y;
            if y > 0.0 {
                pos_sum += y;
                pos_cnt += 1;
                ln_y[obs_idx] = y.ln();
            }
            obs_idx += 1;
        }
    }

    let positive_mean = pos_sum / (pos_cnt.max(1) as f64);
    let y_floor = (positive_mean * 1e-4).max(1e-8);
    let ln_y_floor = y_floor.ln();

    let mut log_exp_buf = SmallBuf::<16>::zeroed(n);
    let log_exp = log_exp_buf.as_mut_slice();
    if let Some(exp) = &options.exposure {
        for i in 0..n {
            log_exp[i] = exp[i].ln();
        }
    }

    // Exact O(n^2) OLS initialization using unit-weight tridiagonal factor.
    let ones_buf = SmallBuf::<16>::filled(n, 1.0);
    let ones = ones_buf.as_slice();
    let mut unit_factor = TridiagonalRank1Factor::new(n);
    if !unit_factor.refactor(ones, ones) {
        return Ok(None);
    }

    let mut ru_store_a = SmallBuf::<16>::zeroed(n);
    let mut rv_store_a = SmallBuf::<16>::zeroed(n);
    let mut ru_store_b = SmallBuf::<16>::zeroed(n);
    let mut rv_store_b = SmallBuf::<16>::zeroed(n);
    let (mut ru, mut next_ru) = (ru_store_a.as_mut_slice(), ru_store_b.as_mut_slice());
    let (mut rv, mut next_rv) = (rv_store_a.as_mut_slice(), rv_store_b.as_mut_slice());

    let mut idx = 0usize;
    for i in 0..n {
        let off_i = log_exp[i];
        let mut rui = 0.0;
        for j in 0..(n - i) {
            let ly = if y_obs[idx] > y_floor {
                ln_y[idx]
            } else {
                ln_y_floor
            };
            let z = ly - off_i;
            rui += z;
            rv[j] += z;
            idx += 1;
        }
        ru[i] = rui;
    }

    let mut u_store_a = SmallBuf::<16>::zeroed(n);
    let mut v_store_a = SmallBuf::<16>::zeroed(n);
    let mut u_store_b = SmallBuf::<16>::zeroed(n);
    let mut v_store_b = SmallBuf::<16>::zeroed(n);
    let (mut u, mut next_u) = (u_store_a.as_mut_slice(), u_store_b.as_mut_slice());
    let (mut v, mut next_v) = (v_store_a.as_mut_slice(), v_store_b.as_mut_slice());
    unit_factor.solve_uv(ones, ru, rv, u, v);

    let mut inv_bv_buf = SmallBuf::<16>::zeroed(n);
    let mut b1_buf = SmallBuf::<16>::zeroed(n);
    let mut b2_buf = SmallBuf::<16>::zeroed(n);

    // Evaluate rank-1 factors, objective, and (for Gamma) fused Newton score `(ru, rv)`
    // in a single division-free pass.
    let eval_step = |u_vec: &[f64],
                     v_vec: &[f64],
                     au_out: &mut [f64],
                     bv_out: &mut [f64],
                     ru_out: &mut [f64],
                     rv_out: &mut [f64],
                     inv_bv: &mut [f64],
                     b1_arr: &mut [f64],
                     b2_arr: &mut [f64]|
     -> Option<f64> {
        let mut u_min = f64::INFINITY;
        let mut u_max = f64::NEG_INFINITY;
        let mut v_min = f64::INFINITY;
        let mut v_max = f64::NEG_INFINITY;
        for i in 0..n {
            let ui = u_vec[i] + log_exp[i];
            if !ui.is_finite() {
                return None;
            }
            u_min = u_min.min(ui);
            u_max = u_max.max(ui);
            au_out[i] = ui.exp();
        }
        for j in 0..n {
            let vj = v_vec[j];
            if !vj.is_finite() {
                return None;
            }
            v_min = v_min.min(vj);
            v_max = v_max.max(vj);
            let bvj = vj.exp();
            bv_out[j] = bvj;
            if is_gamma {
                inv_bv[j] = 1.0 / bvj;
            } else {
                b1_arr[j] = (one_mp * vj).exp();
                b2_arr[j] = (two_mp * vj).exp();
            }
        }
        if u_min + v_min <= -49.0 || u_max + v_max >= 49.0 {
            return None;
        }

        let mut half_obj = 0.0;
        let mut k = 0usize;
        if is_gamma {
            rv_out.fill(0.0);
            for i in 0..n {
                let ui_off = u_vec[i] + log_exp[i];
                let inv_aui = 1.0 / au_out[i];
                let mut rui = 0.0;
                for j in 0..(n - i) {
                    let eta = ui_off + v_vec[j];
                    let s = y_obs[k] * inv_aui * inv_bv[j] - 1.0;
                    half_obj += (eta - ln_y[k]) + s;
                    rui += s;
                    rv_out[j] += s;
                    k += 1;
                }
                ru_out[i] = rui;
            }
        } else {
            let inv_one_mp = 1.0 / one_mp;
            let inv_two_mp = 1.0 / two_mp;
            for i in 0..n {
                let ui_off = u_vec[i] + log_exp[i];
                let a1 = (one_mp * ui_off).exp();
                let a2 = (two_mp * ui_off).exp();
                for j in 0..(n - i) {
                    let eta = ui_off + v_vec[j];
                    let y = y_obs[k];
                    let b2 = b2_arr[j];
                    if y > 0.0 {
                        let ln_ratio = ln_y[k] - eta;
                        let t1 = y * (a1 * b1_arr[j]) * (one_mp * ln_ratio).exp_m1() * inv_one_mp;
                        let t2 = (a2 * b2) * (two_mp * ln_ratio).exp_m1() * inv_two_mp;
                        half_obj += t1 - t2;
                    } else {
                        half_obj += (a2 * b2) * inv_two_mp;
                    }
                    k += 1;
                }
            }
        }
        Some(2.0 * half_obj)
    };

    let mut au_store_a = SmallBuf::<16>::zeroed(n);
    let mut bv_store_a = SmallBuf::<16>::zeroed(n);
    let mut au_store_b = SmallBuf::<16>::zeroed(n);
    let mut bv_store_b = SmallBuf::<16>::zeroed(n);
    let (mut au, mut next_au) = (au_store_a.as_mut_slice(), au_store_b.as_mut_slice());
    let (mut bv, mut next_bv) = (bv_store_a.as_mut_slice(), bv_store_b.as_mut_slice());

    let Some(mut cur_obj) = eval_step(
        u,
        v,
        au,
        bv,
        ru,
        rv,
        inv_bv_buf.as_mut_slice(),
        b1_buf.as_mut_slice(),
        b2_buf.as_mut_slice(),
    ) else {
        return Ok(None);
    };

    let mut du_buf = SmallBuf::<16>::zeroed(n);
    let mut dv_buf = SmallBuf::<16>::zeroed(n);
    let du = du_buf.as_mut_slice();
    let dv = dv_buf.as_mut_slice();

    // Extra rank-1 weight buffers for general Tweedie (1 < p < 2).
    let mut a_w_buf = SmallBuf::<16>::zeroed(if is_gamma { 0 } else { n });
    let mut b_w_buf = SmallBuf::<16>::zeroed(if is_gamma { 0 } else { n });
    let mut a_score_buf = SmallBuf::<16>::zeroed(if is_gamma { 0 } else { n });
    let mut b_score_buf = SmallBuf::<16>::zeroed(if is_gamma { 0 } else { n });
    let mut tw_factor = if is_gamma {
        None
    } else {
        Some(TridiagonalRank1Factor::new(n))
    };

    let mut converged = false;
    let mut iterations = 0usize;

    for iter in 1..=options.max_iterations {
        iterations = iter;

        // Solve Newton step `(du, dv)` in O(n) (for Gamma, `ru, rv` were already fused above!).
        if is_gamma {
            unit_factor.solve_uv(ones, ru, rv, du, dv);
        } else {
            let a_w = a_w_buf.as_mut_slice();
            let b_w = b_w_buf.as_mut_slice();
            let a_sc = a_score_buf.as_mut_slice();
            let b_sc = b_score_buf.as_mut_slice();
            for i in 0..n {
                let ui_off = u[i] + log_exp[i];
                a_w[i] = (two_mp * ui_off).exp();
                a_sc[i] = (one_mp * ui_off).exp();
            }
            for j in 0..n {
                b_w[j] = (two_mp * v[j]).exp();
                b_sc[j] = (one_mp * v[j]).exp();
            }
            let fac = tw_factor.as_mut().expect("tweedie factor");
            if !fac.refactor(a_w, b_w) {
                return Ok(None);
            }
            rv.fill(0.0);
            let mut k = 0usize;
            for i in 0..n {
                let aui = au[i];
                let asi = a_sc[i];
                let mut rui = 0.0;
                for j in 0..(n - i) {
                    let s = (asi * b_sc[j]) * (y_obs[k] - aui * bv[j]);
                    rui += s;
                    rv[j] += s;
                    k += 1;
                }
                ru[i] = rui;
            }
            fac.solve_uv(a_w, ru, rv, du, dv);
        }

        // Step-halving line search guaranteeing non-increasing quasi-deviance.
        let mut step_len = 1.0_f64;
        for i in 0..n {
            next_u[i] = u[i] + du[i];
            next_v[i] = v[i] + dv[i];
        }
        let Some(mut next_obj) = eval_step(
            next_u,
            next_v,
            next_au,
            next_bv,
            next_ru,
            next_rv,
            inv_bv_buf.as_mut_slice(),
            b1_buf.as_mut_slice(),
            b2_buf.as_mut_slice(),
        ) else {
            return Ok(None);
        };

        let mut halvings = 0usize;
        while (!next_obj.is_finite() || next_obj > cur_obj + 1e-9 * (1.0 + cur_obj.abs()))
            && halvings < 25
        {
            step_len *= 0.5;
            halvings += 1;
            for i in 0..n {
                next_u[i] = u[i] + step_len * du[i];
                next_v[i] = v[i] + step_len * dv[i];
            }
            next_obj = match eval_step(
                next_u,
                next_v,
                next_au,
                next_bv,
                next_ru,
                next_rv,
                inv_bv_buf.as_mut_slice(),
                b1_buf.as_mut_slice(),
                b2_buf.as_mut_slice(),
            ) {
                Some(v_obj) => v_obj,
                None => return Ok(None),
            };
        }

        // Relative change measured in corner-coded `beta = (u_0, u_i - u_0, v_j)`.
        let mut max_rel_change = (next_u[0] - u[0]).abs() / (1.0 + u[0].abs());
        for i in 1..n {
            let bo = u[i] - u[0];
            let bn = next_u[i] - next_u[0];
            max_rel_change = max_rel_change.max((bn - bo).abs() / (1.0 + bo.abs()));
        }
        for j in 1..n {
            max_rel_change = max_rel_change.max((next_v[j] - v[j]).abs() / (1.0 + v[j].abs()));
        }
        let rel_obj_change = (cur_obj - next_obj).abs() / (1.0 + cur_obj.abs());

        // Zero-copy 16-byte slice-pointer swaps (no stack buffer memcpy).
        std::mem::swap(&mut u, &mut next_u);
        std::mem::swap(&mut v, &mut next_v);
        std::mem::swap(&mut au, &mut next_au);
        std::mem::swap(&mut bv, &mut next_bv);
        if is_gamma {
            std::mem::swap(&mut ru, &mut next_ru);
            std::mem::swap(&mut rv, &mut next_rv);
        }
        cur_obj = next_obj;

        if max_rel_change < options.tolerance && rel_obj_change < options.tolerance {
            converged = true;
            break;
        }
    }

    // Final exact tridiagonal factor at converged mu.
    let mut inv_a_buf = SmallBuf::<16>::filled(n, 1.0);
    let (final_factor, a_fin, b_fin, inv_a_fin): (
        &mut TridiagonalRank1Factor,
        &[f64],
        &[f64],
        &[f64],
    ) = if is_gamma {
        (&mut unit_factor, ones, ones, ones)
    } else {
        let a_w = a_w_buf.as_mut_slice();
        let b_w = b_w_buf.as_mut_slice();
        let inv_a = inv_a_buf.as_mut_slice();
        for i in 0..n {
            let awi = (two_mp * (u[i] + log_exp[i])).exp();
            a_w[i] = awi;
            inv_a[i] = 1.0 / awi;
        }
        for j in 0..n {
            b_w[j] = (two_mp * v[j]).exp();
        }
        let fac = tw_factor.as_mut().expect("tweedie factor");
        if !fac.refactor(a_w, b_w) {
            return Ok(None);
        }
        (fac, a_w, b_w, inv_a)
    };

    let mut g_buf = SmallBuf::<144>::zeroed(n * n);
    final_factor.compute_g_mat_into(g_buf.as_mut_slice());
    let g_mat = g_buf.as_slice();
    let leverages = final_factor.compute_leverages(a_fin, b_fin, g_mat);

    // Pearson chi-square & full n × n fitted incremental grid.
    let mut fitted_incremental = vec![0.0; n * n];
    let mut pearson_chi_sq = 0.0;
    let mut k = 0usize;
    if is_gamma {
        let inv_bv = inv_bv_buf.as_mut_slice();
        for j in 0..n {
            inv_bv[j] = 1.0 / bv[j];
        }
        for i in 0..n {
            let aui = au[i];
            let inv_aui = 1.0 / aui;
            let fit_row = &mut fitted_incremental[i * n..i * n + n];
            for j in 0..(n - i) {
                let mu = aui * bv[j];
                fit_row[j] = mu;
                let r = y_obs[k] * inv_aui * inv_bv[j] - 1.0;
                pearson_chi_sq += r * r;
                k += 1;
            }
            for j in (n - i)..n {
                fit_row[j] = aui * bv[j];
            }
        }
    } else {
        for i in 0..n {
            let aui = au[i];
            let fit_row = &mut fitted_incremental[i * n..i * n + n];
            for j in 0..(n - i) {
                let mu = aui * bv[j];
                fit_row[j] = mu;
                let diff = y_obs[k] - mu;
                pearson_chi_sq += (diff * diff) / mu.powf(power);
                k += 1;
            }
            for j in (n - i)..n {
                fit_row[j] = aui * bv[j];
            }
        }
    }

    let pearson_dispersion = (pearson_chi_sq / dof as f64).max(0.0);
    let deviance_dispersion = (cur_obj / dof as f64).max(0.0);

    let cov = final_factor.compute_cov_scaled(inv_a_fin, g_mat, pearson_dispersion);
    let mut coefficient_se = Vec::with_capacity(q);
    for col in 0..q {
        coefficient_se.push(cov[col * q + col].max(0.0).sqrt());
    }

    let mut coefficients = Vec::with_capacity(q);
    coefficients.push(u[0]);
    for i in 1..n {
        coefficients.push(u[i] - u[0]);
    }
    for j in 1..n {
        coefficients.push(v[j]);
    }

    // Reserves & O(n^2) analytic MSEP using exact O(n) tridiagonal forward sweeps.
    let mut reserves = vec![0.0; n];
    let mut ultimates = vec![0.0; n];
    let mut process_se = vec![0.0; n];
    let mut parameter_se = vec![0.0; n];
    let mut prediction_error = vec![0.0; n];
    let mut total_proc_var = 0.0;
    let mut total_reserve = 0.0;

    for i in 0..n {
        let aui = au[i];
        let mut r_i = 0.0;
        let mut proc_var_i = 0.0;
        for j in (n - i)..n {
            let mu_ij = aui * bv[j];
            r_i += mu_ij;
            let v_mu = if is_gamma {
                mu_ij * mu_ij
            } else {
                mu_ij.powf(power)
            };
            proc_var_i += pearson_dispersion * v_mu;
        }
        reserves[i] = r_i;
        ultimates[i] = latest[i] + r_i;
        total_reserve += r_i;
        total_proc_var += proc_var_i;

        let param_quad = final_factor.origin_param_quad_general(inv_a_fin, i, r_i, aui, bv);
        let param_var_i = (pearson_dispersion * param_quad).max(0.0);
        let pv = proc_var_i.max(0.0);
        process_se[i] = pv.sqrt();
        parameter_se[i] = param_var_i.sqrt();
        prediction_error[i] = (pv + param_var_i).sqrt();
    }

    let total_param_quad = final_factor.total_param_quad_general(inv_a_fin, &reserves, au, bv);
    let total_param_var = (pearson_dispersion * total_param_quad).max(0.0);
    let total_proc_var = total_proc_var.max(0.0);
    let total_prediction_error = (total_proc_var + total_param_var).sqrt();
    let total_cv = if total_reserve > 0.0 {
        total_prediction_error / total_reserve
    } else {
        0.0
    };

    Ok(Some(GlmReservingResult {
        family: options.family,
        param_names: cross_classified_param_names(n),
        coefficients,
        coefficient_se,
        coefficient_covariance: cov,
        iterations,
        converged,
        n_observed: n_obs,
        n_params: q,
        degrees_of_freedom: dof,
        deviance: cur_obj,
        pearson_chi_sq,
        pearson_dispersion,
        deviance_dispersion,
        leverages,
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
    }))
}

// ===========================================================================
// Fast Path C: Cross-Classified Log-Normal GLM (O(n^2) OLS/Cov/Leverages)
// ===========================================================================

fn try_fit_cross_classified_log_normal(
    triangle: &TriangleView<'_>,
    options: &GlmOptions,
    latest: &[f64],
) -> Result<Option<GlmReservingResult>, ValidationError> {
    let n = triangle.n_origins;
    let m = n - 1;
    let q = 2 * n - 1;
    let n_obs = n * (n + 1) / 2;
    let dof = n_obs - q;

    let mut log_exp_buf = SmallBuf::<32>::zeroed(n);
    let log_exp = log_exp_buf.as_mut_slice();
    if let Some(exp) = &options.exposure {
        for i in 0..n {
            log_exp[i] = exp[i].ln();
        }
    }

    let mut z_buf = SmallBuf::<256>::zeroed(n_obs);
    let z_obs = z_buf.as_mut_slice();
    let mut ru_buf = SmallBuf::<32>::zeroed(n);
    let mut rv_buf = SmallBuf::<32>::zeroed(n);
    let ru = ru_buf.as_mut_slice();
    let rv = rv_buf.as_mut_slice();
    let mut obs_idx = 0usize;
    for i in 0..n {
        let row_len = n - i;
        let cum_row = &triangle.data[i * n..i * n + row_len];
        let off_i = log_exp[i];
        let mut rui = 0.0;
        for j in 0..row_len {
            let y = if j == 0 {
                cum_row[0]
            } else {
                cum_row[j] - cum_row[j - 1]
            };
            validate_observed_increment(y, options.family, options.monotonicity_policy)?;
            let z = y.ln() - off_i;
            z_obs[obs_idx] = z;
            obs_idx += 1;
            rui += z;
            rv[j] += z;
        }
        ru[i] = rui;
    }

    let ones_buf = SmallBuf::<32>::filled(n, 1.0);
    let ones = ones_buf.as_slice();
    let mut factor = TridiagonalRank1Factor::new(n);
    if !factor.refactor(ones, ones) {
        return Ok(None);
    }

    let mut u_buf = SmallBuf::<32>::zeroed(n);
    let mut v_buf = SmallBuf::<32>::zeroed(n);
    factor.solve_uv(ones, ru, rv, u_buf.as_mut_slice(), v_buf.as_mut_slice());
    let u = u_buf.as_slice();
    let v = v_buf.as_slice();

    let mut rss = 0.0;
    let mut k = 0usize;
    for i in 0..n {
        let ui = u[i];
        for j in 0..(n - i) {
            let d = z_obs[k] - (ui + v[j]);
            rss += d * d;
            k += 1;
        }
    }
    let sigma_sq = (rss / dof as f64).max(0.0);

    let mut g_buf = SmallBuf::<144>::zeroed(n * n);
    factor.compute_g_mat_into(g_buf.as_mut_slice());
    let g_mat = g_buf.as_slice();
    let leverages = factor.compute_leverages(ones, ones, g_mat);
    let cov = factor.compute_cov_scaled(ones, g_mat, sigma_sq);

    let mut coefficient_se = Vec::with_capacity(q);
    for col in 0..q {
        coefficient_se.push(cov[col * q + col].max(0.0).sqrt());
    }
    let mut coefficients = Vec::with_capacity(q);
    coefficients.push(u[0]);
    for i in 1..n {
        coefficients.push(u[i] - u[0]);
    }
    for j in 1..n {
        coefficients.push(v[j]);
    }

    let mut fitted_incremental = vec![0.0; n * n];
    let mut reserves = vec![0.0; n];
    let n_fut = n * (n - 1) / 2;
    let mut fut_i = Vec::with_capacity(n_fut);
    let mut fut_j = Vec::with_capacity(n_fut);
    let mut fut_mu = Vec::with_capacity(n_fut);

    let half_sig = 0.5 * sigma_sq;
    for i in 0..n {
        let ui_off = u[i] + log_exp[i] + half_sig;
        for j in 0..n {
            let mu = (ui_off + v[j]).exp();
            if !mu.is_finite() {
                return Err(ValidationError::NotFinite);
            }
            fitted_incremental[i * n + j] = mu;
            if i + j >= n {
                reserves[i] += mu;
                fut_i.push(i);
                fut_j.push(j);
                fut_mu.push(mu);
            }
        }
    }

    // Precompute small (n × n) (u, v) covariance blocks scaled by sigma_sq so
    // every future-pair covariance Cov(eta_{i,j}, eta_{r,s}) is O(1) (4 adds).
    let mut cov_u_buf = SmallBuf::<256>::zeroed(n * n);
    let mut cov_uv_buf = SmallBuf::<256>::zeroed(n * n);
    let mut cov_v_buf = SmallBuf::<256>::zeroed(n * n);
    let cov_u = cov_u_buf.as_mut_slice();
    let cov_uv = cov_uv_buf.as_mut_slice();
    let cov_v = cov_v_buf.as_mut_slice();
    for i in 0..n {
        let ki = m - i;
        let inv_bi = factor.inv_b(ki);
        for r in i..n {
            let kr = m - r;
            let diag = if i == r { inv_bi } else { 0.0 };
            let val = sigma_sq * (diag + g_mat[ki * n + kr] * inv_bi * factor.inv_b(kr));
            cov_u[i * n + r] = val;
            cov_u[r * n + i] = val;
        }
        for j in 1..n {
            cov_uv[i * n + j] = -sigma_sq * (g_mat[ki * n + j] - g_mat[ki * n + (j - 1)]) * inv_bi;
        }
    }
    for a in 1..n {
        for b in a..n {
            let val = sigma_sq
                * (g_mat[a * n + b] - g_mat[(a - 1) * n + b] - g_mat[a * n + (b - 1)]
                    + g_mat[(a - 1) * n + (b - 1)]);
            cov_v[a * n + b] = val;
            cov_v[b * n + a] = val;
        }
    }

    let mut proc_var_origin = SmallBuf::<32>::zeroed(n);
    let mut param_var_origin = SmallBuf::<32>::zeroed(n);
    let pvo = proc_var_origin.as_mut_slice();
    let pmvo = param_var_origin.as_mut_slice();
    let mut total_proc_var = 0.0;
    let mut total_param_var = 0.0;
    let exp_m1_sig = sigma_sq.exp_m1();

    for a in 0..n_fut {
        let ia = fut_i[a];
        let ja = fut_j[a];
        let mu_a = fut_mu[a];
        let v_est_a =
            (cov_u[ia * n + ia] + 2.0 * cov_uv[ia * n + ja] + cov_v[ja * n + ja]).max(0.0);
        let pv = mu_a * mu_a * exp_m1_sig;
        let tot_a = mu_a * mu_a * (sigma_sq + v_est_a).exp_m1();
        let pmv = (tot_a - pv).max(0.0);
        pvo[ia] += pv;
        pmvo[ia] += pmv;
        total_proc_var += pv;
        total_param_var += pmv;

        let cu_row = &cov_u[ia * n..ia * n + n];
        let cuv_row = &cov_uv[ia * n..ia * n + n];
        let cv_row = &cov_v[ja * n..ja * n + n];
        let two_mu_a = 2.0 * mu_a;

        for b in (a + 1)..n_fut {
            let ib = fut_i[b];
            let jb = fut_j[b];
            let cov_eta = cu_row[ib] + cuv_row[jb] + cov_uv[ib * n + ja] + cv_row[jb];
            let two_cov_ab = two_mu_a * fut_mu[b] * cov_eta.exp_m1();
            total_param_var += two_cov_ab;
            if ia == ib {
                pmvo[ia] += two_cov_ab;
            }
        }
    }

    let process_se: Vec<f64> = pvo.iter().map(|v| v.max(0.0).sqrt()).collect();
    let parameter_se: Vec<f64> = pmvo.iter().map(|v| v.max(0.0).sqrt()).collect();
    let prediction_error: Vec<f64> = pvo
        .iter()
        .zip(pmvo.iter())
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

    Ok(Some(GlmReservingResult {
        family: GlmFamily::LogNormal,
        param_names: cross_classified_param_names(n),
        coefficients,
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
        leverages,
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
    }))
}

// ===========================================================================
// General Householder QR IRLS Solver (Hoerl, Calendar Splines, & Fallbacks)
// ===========================================================================

fn fit_glm_general_qr(
    triangle: &TriangleView<'_>,
    options: &GlmOptions,
    latest: &[f64],
) -> Result<GlmReservingResult, ValidationError> {
    let n = triangle.n_origins;
    let schema = GlmDesignSchema::build(n, &options.design)?;
    let q = schema.param_names.len();
    let n_obs = n * (n + 1) / 2;
    if n_obs <= q {
        return Err(ValidationError::InvalidParameter(
            "GLM model has non-positive residual degrees of freedom",
        ));
    }
    let dof = n_obs - q;

    // Build exact-capacity row-major `x_obs`, column-major `x_cm`, `y_obs`, and `offsets_obs`
    // directly from `triangle.data` in one pass (no `cumulative_to_incremental` or `obs_cells`).
    let mut x_obs = vec![0.0; n_obs * q];
    let mut x_cm = vec![0.0; n_obs * q];
    let mut y_obs = Vec::with_capacity(n_obs);
    let mut offsets_obs = Vec::with_capacity(n_obs);
    let mut has_negative_y = false;

    let mut obs_idx = 0usize;
    for i in 0..n {
        let log_exp = options.exposure.as_ref().map_or(0.0, |e| e[i].ln());
        let row_len = n - i;
        let cum_row = &triangle.data[i * n..i * n + row_len];
        for j in 0..row_len {
            let y = if j == 0 {
                cum_row[0]
            } else {
                cum_row[j] - cum_row[j - 1]
            };
            validate_observed_increment(y, options.family, options.monotonicity_policy)?;
            if y < 0.0 {
                has_negative_y = true;
            }
            let row_slice = &mut x_obs[obs_idx * q..(obs_idx + 1) * q];
            schema.fill_row(i, j, row_slice);
            for col in 0..q {
                x_cm[col * n_obs + obs_idx] = row_slice[col];
            }
            y_obs.push(y);
            offsets_obs.push(log_exp);
            obs_idx += 1;
        }
    }

    if options.family == GlmFamily::LogNormal {
        return fit_log_normal_glm(
            n,
            &schema,
            &x_obs,
            &x_cm,
            &y_obs,
            &offsets_obs,
            latest,
            options.exposure.as_deref(),
            n_obs,
            q,
            dof,
        );
    }

    let power = options.family.tweedie_power().expect("Tweedie family");
    let is_odp = (power - 1.0).abs() < 1e-12;
    let is_gamma = (power - 2.0).abs() < 1e-12;

    // Single-pass positive mean + precomputed ln(y) for ln-free IRLS objective checks.
    let mut pos_sum = 0.0;
    let mut pos_cnt = 0usize;
    let mut ln_y = Vec::with_capacity(n_obs);
    for &y in &y_obs {
        if y > 0.0 {
            pos_sum += y;
            pos_cnt += 1;
            ln_y.push(y.ln());
        } else {
            ln_y.push(0.0);
        }
    }
    let positive_mean = pos_sum / (pos_cnt.max(1) as f64);
    let y_floor = (positive_mean * 1e-4).max(1e-8);
    let ln_y_floor = y_floor.ln();

    let mut working_z: Vec<f64> = Vec::with_capacity(n_obs);
    for k in 0..n_obs {
        let ly = if y_obs[k] > y_floor {
            ln_y[k]
        } else {
            ln_y_floor
        };
        working_z.push(ly - offsets_obs[k]);
    }

    let mut qr_ws = QrWorkspace::with_capacity(n_obs, q);
    let mut beta = vec![0.0; q];

    // For Gamma (W = I), factor X ONCE and reuse Householder reflectors across OLS init,
    // all IRLS iterations, and final covariance/leverages.
    let mut weights = if is_gamma {
        Vec::new()
    } else {
        vec![1.0; n_obs]
    };

    if is_gamma {
        qr_factorize_unit_cm_store_reflectors(&x_cm, n_obs, q, &mut qr_ws)?;
        qr_solve_stored_unit_into(&working_z, n_obs, q, &mut qr_ws, &mut beta)?;
    } else {
        weighted_least_squares_qr_cm_beta_into(
            &x_cm, &working_z, &weights, n_obs, q, &mut qr_ws, &mut beta,
        )?;
    }

    let mut eta = vec![0.0; n_obs];
    let mut mu = vec![0.0; n_obs];
    eval_linear_predictor_and_mu(&x_obs, &offsets_obs, &beta, n_obs, q, &mut eta, &mut mu);
    let mut cur_obj =
        total_tweedie_objective_fast(&y_obs, &ln_y, &eta, &mu, power, is_odp, is_gamma);

    let mut converged = false;
    let mut iterations = 0usize;

    let mut beta_target = vec![0.0; q];
    let mut delta = vec![0.0; q];
    let mut next_beta = vec![0.0; q];
    let mut next_eta = vec![0.0; n_obs];
    let mut next_mu = vec![0.0; n_obs];

    for iter in 1..=options.max_iterations {
        iterations = iter;
        if is_gamma {
            for k in 0..n_obs {
                let mk = mu[k];
                working_z[k] = (eta[k] - offsets_obs[k]) + (y_obs[k] - mk) / mk;
            }
            qr_solve_stored_unit_into(&working_z, n_obs, q, &mut qr_ws, &mut beta_target)?;
        } else {
            for k in 0..n_obs {
                let mk = mu[k];
                weights[k] = if is_odp { mk } else { mk.powf(2.0 - power) };
                working_z[k] = (eta[k] - offsets_obs[k]) + (y_obs[k] - mk) / mk;
            }
            weighted_least_squares_qr_cm_beta_into(
                &x_cm,
                &working_z,
                &weights,
                n_obs,
                q,
                &mut qr_ws,
                &mut beta_target,
            )?;
        }

        for j in 0..q {
            delta[j] = beta_target[j] - beta[j];
            next_beta[j] = beta_target[j];
        }

        // Step-halving line search guaranteeing non-increasing quasi-deviance.
        let mut step_len = 1.0_f64;
        eval_linear_predictor_and_mu(
            &x_obs,
            &offsets_obs,
            &next_beta,
            n_obs,
            q,
            &mut next_eta,
            &mut next_mu,
        );
        let mut next_obj = total_tweedie_objective_fast(
            &y_obs, &ln_y, &next_eta, &next_mu, power, is_odp, is_gamma,
        );

        let mut halvings = 0usize;
        while (!next_obj.is_finite() || next_obj > cur_obj + 1e-9 * (1.0 + cur_obj.abs()))
            && halvings < 25
        {
            step_len *= 0.5;
            halvings += 1;
            for j in 0..q {
                next_beta[j] = beta[j] + step_len * delta[j];
            }
            eval_linear_predictor_and_mu(
                &x_obs,
                &offsets_obs,
                &next_beta,
                n_obs,
                q,
                &mut next_eta,
                &mut next_mu,
            );
            next_obj = total_tweedie_objective_fast(
                &y_obs, &ln_y, &next_eta, &next_mu, power, is_odp, is_gamma,
            );
        }

        let max_rel_change = next_beta
            .iter()
            .zip(beta.iter())
            .map(|(&bn, &bo)| (bn - bo).abs() / (1.0 + bo.abs()))
            .fold(0.0_f64, f64::max);
        let rel_obj_change = (cur_obj - next_obj).abs() / (1.0 + cur_obj.abs());

        std::mem::swap(&mut beta, &mut next_beta);
        std::mem::swap(&mut eta, &mut next_eta);
        std::mem::swap(&mut mu, &mut next_mu);
        cur_obj = next_obj;

        if max_rel_change < options.tolerance && rel_obj_change < options.tolerance {
            converged = true;
            break;
        }
    }

    let cur_dev = if is_odp && has_negative_y {
        total_tweedie_deviance(&y_obs, &mu, power)
    } else {
        cur_obj
    };

    // Exact final (X^T W X)^{-1} and hat leverages once at converged mu (without discarding beta).
    let (mut cov, final_leverages) = if is_gamma {
        qr_cov_and_leverages_from_factored(&x_obs, n_obs, q, &mut qr_ws)
    } else {
        for k in 0..n_obs {
            let mk = mu[k];
            weights[k] = if is_odp { mk } else { mk.powf(2.0 - power) };
        }
        weighted_least_squares_qr_cm_cov_and_leverages(
            &x_obs, &x_cm, &weights, n_obs, q, &mut qr_ws,
        )?
    };

    // Pearson chi-square & dispersion estimates.
    let mut pearson_chi_sq = 0.0;
    for k in 0..n_obs {
        let diff = y_obs[k] - mu[k];
        let v_mu = if is_odp {
            mu[k]
        } else if is_gamma {
            mu[k] * mu[k]
        } else {
            mu[k].powf(power)
        };
        pearson_chi_sq += (diff * diff) / v_mu;
    }
    let pearson_dispersion = (pearson_chi_sq / dof as f64).max(0.0);
    let deviance_dispersion = (cur_dev / dof as f64).max(0.0);

    // Scale (X^T W X)^{-1} in place to form Cov(beta_hat).
    for v in &mut cov {
        *v *= pearson_dispersion;
    }
    let coefficient_se: Vec<f64> = (0..q).map(|k| cov[k * q + k].max(0.0).sqrt()).collect();

    // Project full n × n incremental grid (reusing already-computed observed `mu`) & analytic MSEP.
    let mut fitted_incremental = vec![0.0; n * n];
    let mut reserves = vec![0.0; n];
    let mut process_se = vec![0.0; n];
    let mut parameter_se = vec![0.0; n];
    let mut prediction_error = vec![0.0; n];
    let mut grad_total = vec![0.0; q];
    let mut grad_origin = vec![0.0; q];
    let mut total_proc_var = 0.0;

    let mut x_cell = vec![0.0; q];
    let mut obs_cursor = 0usize;
    for i in 0..n {
        let log_exp = options.exposure.as_ref().map_or(0.0, |e| e[i].ln());
        grad_origin.fill(0.0);
        let mut proc_var_i = 0.0;
        let mut r_i = 0.0;

        let obs_len = n - i;
        fitted_incremental[i * n..i * n + obs_len]
            .copy_from_slice(&mu[obs_cursor..obs_cursor + obs_len]);
        obs_cursor += obs_len;

        for j in obs_len..n {
            schema.fill_row(i, j, &mut x_cell);
            let mut e_ij = log_exp;
            for (&xc, &bc) in x_cell.iter().zip(beta.iter()) {
                e_ij += xc * bc;
            }
            let mu_ij = e_ij.clamp(-50.0, 50.0).exp();
            fitted_incremental[i * n + j] = mu_ij;

            r_i += mu_ij;
            let v_mu = if is_odp {
                mu_ij
            } else if is_gamma {
                mu_ij * mu_ij
            } else {
                mu_ij.powf(power)
            };
            proc_var_i += pearson_dispersion * v_mu;
            for col in 0..q {
                let g = mu_ij * x_cell[col];
                grad_origin[col] += g;
                grad_total[col] += g;
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

/// Candidate result from [`select_tweedie_power`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TweediePowerCandidate {
    /// Evaluated Tweedie variance power $p \in [1, 2]$.
    pub power: f64,
    /// Extended quasi-likelihood (EQL) negative log-likelihood score (lower is better):
    /// $\frac{1}{2}\big(n_{\text{obs}} \ln(2\pi \hat{\phi}) + p \sum \ln y_k + D(y, \hat{\mu})/\hat{\phi}\big)$.
    pub eql_score: f64,
    /// Fitted Pearson dispersion $\hat{\phi}$.
    pub pearson_dispersion: f64,
    /// Fitted total reserve at this power.
    pub total_reserve: f64,
    /// Fitted total prediction error at this power.
    pub total_prediction_error: f64,
}

/// Output of profile extended quasi-likelihood Tweedie power selection
/// (mirroring `mages/ChainLadder`'s `tweedieReserve` power estimation).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TweediePowerSelection {
    /// Best Tweedie power $p^*$ minimizing the extended quasi-likelihood score.
    pub best_power: f64,
    /// Full GLM fit at `best_power`.
    pub best_fit: GlmReservingResult,
    /// All successfully evaluated candidates across `p_grid`.
    pub candidates: Vec<TweediePowerCandidate>,
}

/// Select the Tweedie variance power $p \in [1, 2]$ by minimizing the Nelder &
/// Pregibon (1987) extended quasi-likelihood profile criterion over `p_grid`.
#[allow(clippy::needless_pass_by_value)]
pub fn select_tweedie_power(
    triangle: &TriangleView<'_>,
    design: GlmDesign,
    p_grid: &[f64],
) -> Result<TweediePowerSelection, ValidationError> {
    if p_grid.is_empty() {
        return Err(ValidationError::InvalidParameter(
            "p_grid must contain at least one power in [1, 2]",
        ));
    }
    let n = triangle.n_origins.min(triangle.n_devs);
    let mut sum_log_y = 0.0;
    let mut n_pos = 0usize;
    for i in 0..n {
        let row_len = n - i;
        let row = &triangle.data[i * triangle.n_devs..i * triangle.n_devs + row_len];
        for j in 0..row_len {
            let y = if j == 0 { row[0] } else { row[j] - row[j - 1] };
            if y.is_finite() && y > 0.0 {
                sum_log_y += y.ln();
                n_pos += 1;
            }
        }
    }
    if n_pos == 0 {
        return Err(ValidationError::ShapeMismatch);
    }

    let mut candidates = Vec::with_capacity(p_grid.len());
    let mut best_score = f64::INFINITY;
    let mut best_power = f64::NAN;
    let mut best_fit: Option<GlmReservingResult> = None;

    for &p in p_grid {
        if !p.is_finite() || !(1.0..=2.0).contains(&p) {
            return Err(ValidationError::InvalidParameter(
                "Tweedie power must lie in [1.0, 2.0]",
            ));
        }
        let opts = GlmOptions::tweedie(p).with_design(design.clone());
        if let Ok(fit) = fit_glm(triangle, &opts) {
            let phi = fit.deviance_dispersion.max(1e-12);
            let n_obs = fit.n_observed as f64;
            let eql_score = 0.5
                * (n_obs * (2.0 * std::f64::consts::PI * phi).ln()
                    + p * sum_log_y
                    + fit.deviance / phi);
            candidates.push(TweediePowerCandidate {
                power: p,
                eql_score,
                pearson_dispersion: fit.pearson_dispersion,
                total_reserve: fit.total_reserve,
                total_prediction_error: fit.total_prediction_error,
            });
            if eql_score < best_score {
                best_score = eql_score;
                best_power = p;
                best_fit = Some(fit);
            }
        }
    }

    let best_fit = best_fit.ok_or(ValidationError::NotFinite)?;
    Ok(TweediePowerSelection {
        best_power,
        best_fit,
        candidates,
    })
}

#[allow(clippy::too_many_arguments)]
fn fit_log_normal_glm(
    n: usize,
    schema: &GlmDesignSchema,
    x_obs: &[f64],
    x_cm: &[f64],
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
    let mut qr_ws = QrWorkspace::with_capacity(n_obs, q);
    let mut beta = vec![0.0; q];
    qr_factorize_unit_cm_store_reflectors(x_cm, n_obs, q, &mut qr_ws)?;
    qr_solve_stored_unit_into(&z_obs, n_obs, q, &mut qr_ws, &mut beta)?;
    let (mut cov, leverages) = qr_cov_and_leverages_from_factored(x_obs, n_obs, q, &mut qr_ws);

    let mut rss = 0.0;
    for k in 0..n_obs {
        let mut eta = 0.0;
        let x_row = &x_obs[k * q..(k + 1) * q];
        for (&xj, &bj) in x_row.iter().zip(beta.iter()) {
            eta += xj * bj;
        }
        let d = z_obs[k] - eta;
        rss += d * d;
    }
    let sigma_sq = (rss / dof as f64).max(0.0);
    for v in &mut cov {
        *v *= sigma_sq;
    }
    let coefficient_se: Vec<f64> = (0..q).map(|k| cov[k * q + k].max(0.0).sqrt()).collect();

    let mut fitted_incremental = vec![0.0; n * n];
    let n_fut = n * (n - 1) / 2;
    let mut fut_i = Vec::with_capacity(n_fut);
    let mut fut_mu = Vec::with_capacity(n_fut);
    let mut fut_x = vec![0.0; n_fut * q];
    let mut reserves = vec![0.0; n];
    let mut x_cell = vec![0.0; q];
    let mut fut_idx = 0usize;

    for i in 0..n {
        let log_exp = exposure.map_or(0.0, |e| e[i].ln());
        for j in 0..n {
            schema.fill_row(i, j, &mut x_cell);
            let mut eta = log_exp;
            for col in 0..q {
                eta += x_cell[col] * beta[col];
            }
            let mu = (eta + 0.5 * sigma_sq).exp();
            if !mu.is_finite() {
                return Err(ValidationError::NotFinite);
            }
            fitted_incremental[i * n + j] = mu;
            if i + j >= n {
                reserves[i] += mu;
                fut_i.push(i);
                fut_mu.push(mu);
                fut_x[fut_idx * q..(fut_idx + 1) * q].copy_from_slice(&x_cell);
                fut_idx += 1;
            }
        }
    }

    // Exact log-normal process + parameter variance decomposition using flat `cov_x`.
    let mut proc_var_origin = vec![0.0; n];
    let mut param_var_origin = vec![0.0; n];
    let mut total_proc_var = 0.0;
    let mut total_param_var = 0.0;

    let mut cov_x = vec![0.0; n_fut * q];
    for a in 0..n_fut {
        let xa = &fut_x[a * q..(a + 1) * q];
        let dst = &mut cov_x[a * q..(a + 1) * q];
        for r in 0..q {
            let cov_row = &cov[r * q..(r + 1) * q];
            let mut acc = 0.0;
            for (&c_rc, &x_c) in cov_row.iter().zip(xa.iter()) {
                acc += c_rc * x_c;
            }
            dst[r] = acc;
        }
    }

    let exp_m1_sig = sigma_sq.exp_m1();
    for a in 0..n_fut {
        let ia = fut_i[a];
        let mu_a = fut_mu[a];
        let xa = &fut_x[a * q..(a + 1) * q];
        let cva = &cov_x[a * q..(a + 1) * q];
        let mut v_est_a = 0.0;
        for (&xr, &cr) in xa.iter().zip(cva.iter()) {
            v_est_a += xr * cr;
        }
        let v_est_a = v_est_a.max(0.0);
        let pv = mu_a * mu_a * exp_m1_sig;
        let tot_a = mu_a * mu_a * (sigma_sq + v_est_a).exp_m1();
        let pmv = (tot_a - pv).max(0.0);
        proc_var_origin[ia] += pv;
        param_var_origin[ia] += pmv;
        total_proc_var += pv;
        total_param_var += pmv;

        let two_mu_a = 2.0 * mu_a;
        for b in (a + 1)..n_fut {
            let ib = fut_i[b];
            let cvb = &cov_x[b * q..(b + 1) * q];
            let mut cov_eta = 0.0;
            for (&xr, &cr) in xa.iter().zip(cvb.iter()) {
                cov_eta += xr * cr;
            }
            let two_cov_ab = two_mu_a * fut_mu[b] * cov_eta.exp_m1();
            total_param_var += two_cov_ab;
            if ia == ib {
                param_var_origin[ia] += two_cov_ab;
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
        coefficients: beta,
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
        leverages,
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

#[inline]
fn eval_linear_predictor_and_mu(
    x: &[f64],
    offsets: &[f64],
    beta: &[f64],
    n_obs: usize,
    q: usize,
    eta_out: &mut [f64],
    mu_out: &mut [f64],
) {
    for k in 0..n_obs {
        let mut s = offsets[k];
        let x_row = &x[k * q..(k + 1) * q];
        for (&xj, &bj) in x_row.iter().zip(beta.iter()) {
            s += xj * bj;
        }
        eta_out[k] = s;
        mu_out[k] = s.clamp(-50.0, 50.0).exp();
    }
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
            let t1 = y * mu.powf(one_mp) * (one_mp * ln_ratio).exp_m1() / one_mp;
            let t2 = mu.powf(two_mp) * (two_mp * ln_ratio).exp_m1() / two_mp;
            2.0 * (t1 - t2)
        } else {
            2.0 * mu.powf(two_mp) / two_mp
        }
    }
}

/// Fast `ln`-free line-search objective using precomputed `ln_y` and `eta`
/// (since `mu = eta.clamp(-50, 50).exp()`, `ln(mu) == eta.clamp(-50, 50)`).
fn total_tweedie_objective_fast(
    y: &[f64],
    ln_y: &[f64],
    eta: &[f64],
    mu: &[f64],
    p: f64,
    is_odp: bool,
    is_gamma: bool,
) -> f64 {
    let mut sum = 0.0;
    if is_odp {
        for k in 0..y.len() {
            let yk = y[k];
            let mk = mu[k];
            let ec = eta[k].clamp(-50.0, 50.0);
            if yk > 0.0 {
                sum += 2.0 * (yk * (ln_y[k] - ec) - (yk - mk));
            } else if yk < 0.0 {
                sum += 2.0 * (mk - yk * ec);
            } else {
                sum += 2.0 * mk;
            }
        }
    } else if is_gamma {
        for k in 0..y.len() {
            let yk = y[k];
            let mk = mu[k];
            let ec = eta[k].clamp(-50.0, 50.0);
            sum += 2.0 * ((ec - ln_y[k]) + (yk - mk) / mk);
        }
    } else {
        let one_mp = 1.0 - p;
        let two_mp = 2.0 - p;
        let inv_one_mp = 1.0 / one_mp;
        let inv_two_mp = 1.0 / two_mp;
        for k in 0..y.len() {
            let yk = y[k];
            let ec = eta[k].clamp(-50.0, 50.0);
            let mu_2mp = (two_mp * ec).exp();
            if yk > 0.0 {
                let ln_ratio = ln_y[k] - ec;
                let mu_1mp = (one_mp * ec).exp();
                let t1 = yk * mu_1mp * (one_mp * ln_ratio).exp_m1() * inv_one_mp;
                let t2 = mu_2mp * (two_mp * ln_ratio).exp_m1() * inv_two_mp;
                sum += 2.0 * (t1 - t2);
            } else {
                sum += 2.0 * mu_2mp * inv_two_mp;
            }
        }
    }
    sum
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
        let cap = match design {
            GlmDesign::CrossClassified => 2 * n - 1,
            GlmDesign::Hoerl {
                per_origin_intercepts,
            } => {
                if *per_origin_intercepts {
                    n + 2
                } else {
                    3
                }
            }
            GlmDesign::HoerlWithCalendarSplines {
                per_origin_intercepts,
                calendar_knots,
            } => {
                if *per_origin_intercepts {
                    n + 2 + calendar_knots.len()
                } else {
                    4 + calendar_knots.len()
                }
            }
        };
        let mut names = Vec::with_capacity(cap);
        names.push(String::from("intercept"));

        match design {
            GlmDesign::CrossClassified => {
                for i in 1..n {
                    names.push(indexed_name("origin_", i));
                }
                for j in 1..n {
                    names.push(indexed_name("dev_", j));
                }
            }
            GlmDesign::Hoerl {
                per_origin_intercepts,
            } => {
                if *per_origin_intercepts {
                    for i in 1..n {
                        names.push(indexed_name("origin_", i));
                    }
                }
                names.push(String::from("hoerl_ln_j"));
                names.push(String::from("hoerl_j"));
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
                        names.push(indexed_name("origin_", i));
                    }
                }
                names.push(String::from("hoerl_ln_j"));
                names.push(String::from("hoerl_j"));
                if !*per_origin_intercepts {
                    names.push(String::from("cal_slope_0"));
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
                let d = (j + 1) as f64;
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
                let d = (j + 1) as f64;
                let t = (i + j) as f64;
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
    use crate::datasets::{genins, raa};
    use crate::stochastic::bootstrap_odp::fit_odp;
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

    fn assert_close(a: f64, b: f64, rel_tol: f64, label: &str) {
        let scale = 1.0_f64.max(a.abs()).max(b.abs());
        assert!(
            (a - b).abs() <= rel_tol * scale,
            "{label}: fast={a} ref={b} diff={} rel={}",
            (a - b).abs(),
            (a - b).abs() / scale
        );
    }

    fn assert_glm_result_parity(fast: &GlmReservingResult, qr: &GlmReservingResult, tol: f64) {
        assert_eq!(fast.family, qr.family);
        assert_eq!(fast.param_names, qr.param_names);
        assert_eq!(fast.converged, qr.converged);
        assert_eq!(fast.n_observed, qr.n_observed);
        assert_eq!(fast.n_params, qr.n_params);
        assert_eq!(fast.degrees_of_freedom, qr.degrees_of_freedom);
        assert_close(fast.deviance, qr.deviance, tol, "deviance");
        assert_close(
            fast.pearson_chi_sq,
            qr.pearson_chi_sq,
            tol,
            "pearson_chi_sq",
        );
        assert_close(
            fast.pearson_dispersion,
            qr.pearson_dispersion,
            tol,
            "pearson_dispersion",
        );
        assert_close(
            fast.deviance_dispersion,
            qr.deviance_dispersion,
            tol,
            "deviance_dispersion",
        );
        assert_close(fast.total_reserve, qr.total_reserve, tol, "total_reserve");
        assert_close(
            fast.total_process_se,
            qr.total_process_se,
            tol,
            "total_process_se",
        );
        assert_close(
            fast.total_parameter_se,
            qr.total_parameter_se,
            tol,
            "total_parameter_se",
        );
        assert_close(
            fast.total_prediction_error,
            qr.total_prediction_error,
            tol,
            "total_prediction_error",
        );
        assert_close(fast.total_cv, qr.total_cv, tol, "total_cv");

        for (k, (&a, &b)) in fast
            .coefficients
            .iter()
            .zip(qr.coefficients.iter())
            .enumerate()
        {
            assert_close(a, b, tol, &format!("coefficients[{k}]"));
        }
        for (k, (&a, &b)) in fast
            .coefficient_se
            .iter()
            .zip(qr.coefficient_se.iter())
            .enumerate()
        {
            assert_close(a, b, tol, &format!("coefficient_se[{k}]"));
        }
        let cov_scale = qr
            .coefficient_covariance
            .iter()
            .fold(1.0_f64, |acc, &v| acc.max(v.abs()));
        for (k, (&a, &b)) in fast
            .coefficient_covariance
            .iter()
            .zip(qr.coefficient_covariance.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() <= tol * cov_scale,
                "cov[{k}]: fast={a} ref={b} rel={}",
                (a - b).abs() / cov_scale
            );
        }
        for (k, (&a, &b)) in fast.leverages.iter().zip(qr.leverages.iter()).enumerate() {
            assert_close(a, b, tol, &format!("leverages[{k}]"));
        }
        for (k, (&a, &b)) in fast
            .fitted_incremental
            .iter()
            .zip(qr.fitted_incremental.iter())
            .enumerate()
        {
            assert_close(a, b, tol, &format!("fitted_incremental[{k}]"));
        }
        for i in 0..fast.reserves.len() {
            assert_close(
                fast.reserves[i],
                qr.reserves[i],
                tol,
                &format!("reserves[{i}]"),
            );
            assert_close(
                fast.ultimates[i],
                qr.ultimates[i],
                tol,
                &format!("ultimates[{i}]"),
            );
            assert_close(
                fast.process_se[i],
                qr.process_se[i],
                tol,
                &format!("process_se[{i}]"),
            );
            assert_close(
                fast.parameter_se[i],
                qr.parameter_se[i],
                tol,
                &format!("parameter_se[{i}]"),
            );
            assert_close(
                fast.prediction_error[i],
                qr.prediction_error[i],
                tol,
                &format!("prediction_error[{i}]"),
            );
        }
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

    fn test_latest(view: &TriangleView<'_>, strict: bool) -> Vec<f64> {
        let mut latest = vec![0.0; view.n_origins];
        validate_triangle_single_pass_into(view, strict, &mut latest, None).unwrap();
        latest
    }

    #[test]
    fn cross_classified_fast_paths_match_dense_qr_reference_all_fields() {
        let g = genins();
        let view = g.view();
        let latest = test_latest(&view, true);
        let exp: Vec<f64> = (0..10).map(|i| 1_000.0 + 175.0 * f64::from(i)).collect();

        let cases = vec![
            GlmOptions::odp(),
            GlmOptions::tweedie(1.0),
            GlmOptions::odp().with_exposure(exp.clone()),
            GlmOptions::gamma(),
            GlmOptions::gamma().with_exposure(exp.clone()),
            GlmOptions::tweedie(1.5),
            GlmOptions::tweedie(1.5).with_exposure(exp.clone()),
            GlmOptions::log_normal(),
            GlmOptions::log_normal().with_exposure(exp),
        ];

        for opts in cases {
            let fast = fit_glm(&view, &opts).expect("fast fit");
            let qr = fit_glm_general_qr(&view, &opts, &latest).expect("qr ref fit");
            assert_glm_result_parity(&fast, &qr, 1e-8);
        }
    }

    #[test]
    fn cross_classified_odp_matches_qr_on_raa_with_negative_increments() {
        let tri = raa();
        let view = tri.view();
        let opts = GlmOptions::odp().with_policy(MonotonicityPolicy::AllowNegativeIncrements);
        let latest = test_latest(&view, false);
        let fast = fit_glm(&view, &opts).expect("fast RAA");
        let qr = fit_glm_general_qr(&view, &opts, &latest).expect("qr RAA");
        assert_glm_result_parity(&fast, &qr, 1e-8);
    }

    #[test]
    fn cross_classified_odp_falls_back_cleanly_to_qr_on_degenerate_or_clamped_inputs() {
        // 1. Constant cumulative last column -> LDF[3] == 1.0 -> zero column sum -> fast path
        //    returns None and QR fallback reports exact same collinearity error.
        #[rustfmt::skip]
        let flat_data = vec![
            100.0, 220.0, 310.0, 360.0, 360.0,
            120.0, 255.0, 365.0, 420.0,   NAN,
            135.0, 290.0, 410.0,   NAN,   NAN,
            150.0, 318.0,   NAN,   NAN,   NAN,
            175.0,   NAN,   NAN,   NAN,   NAN,
        ];
        let flat_tri = Triangle::new(5, 5, flat_data);
        let flat_latest = test_latest(&flat_tri.view(), true);
        let fast_err = fit_glm(&flat_tri.view(), &GlmOptions::odp()).unwrap_err();
        let qr_err =
            fit_glm_general_qr(&flat_tri.view(), &GlmOptions::odp(), &flat_latest).unwrap_err();
        assert_eq!(fast_err, qr_err);

        // 2. Tiny triangle near the `eta = -50` clamp triggers the fast-path clamp guard
        //    and falls back to QR IRLS cleanly.
        let tiny_data: Vec<f64> = sample_tri()
            .data()
            .iter()
            .map(|&v| if v.is_finite() { v * 1e-23 } else { v })
            .collect();
        let tiny_tri = Triangle::new(5, 5, tiny_data);
        let tiny_latest = test_latest(&tiny_tri.view(), true);
        let fast_tiny = fit_glm(&tiny_tri.view(), &GlmOptions::odp()).expect("tiny fallback");
        let qr_tiny = fit_glm_general_qr(&tiny_tri.view(), &GlmOptions::odp(), &tiny_latest)
            .expect("tiny qr");
        assert_glm_result_parity(&fast_tiny, &qr_tiny, 1e-12);
    }

    #[test]
    fn validation_error_precedence_preserved_in_single_pass() {
        // Non-monotone at (0, 1) AND NaN at (1, 1) -> must report NotFinite first.
        #[rustfmt::skip]
        let data = vec![
            200.0, 150.0, 300.0,
            100.0,   NAN,   NAN,
            120.0,   NAN,   NAN,
        ];
        let tri = Triangle::new(3, 3, data);
        let err = fit_glm(&tri.view(), &GlmOptions::odp()).unwrap_err();
        assert_eq!(err, ValidationError::NotFinite);
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

    #[test]
    fn larger_triangle_cross_classified_parity_and_scaling() {
        let n = 20usize;
        let mut data = vec![NAN; n * n];
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..(n - i) {
                let inc = 500.0
                    * (1.0 + 0.03 * i as f64)
                    * (-(j as f64) * 0.25).exp()
                    * (1.0 + 0.02 * (((i * 7 + j * 11) % 5) as f64 - 2.0));
                acc += inc;
                data[i * n + j] = acc;
            }
        }
        let tri = Triangle::new(n, n, data);
        let view = tri.view();
        let latest = test_latest(&view, true);

        for opts in [
            GlmOptions::odp(),
            GlmOptions::gamma(),
            GlmOptions::tweedie(1.5),
            GlmOptions::log_normal(),
        ] {
            let fast = fit_glm(&view, &opts).expect("20x20 fast");
            let qr = fit_glm_general_qr(&view, &opts, &latest).expect("20x20 qr");
            assert_glm_result_parity(&fast, &qr, 1e-7);
        }

        // Also exercise SmallBuf heap-fallback path (n = 35 > 32) against exact fit_odp.
        let n35 = 35usize;
        let mut data35 = vec![NAN; n35 * n35];
        for i in 0..n35 {
            let mut acc = 0.0;
            for j in 0..(n35 - i) {
                let inc = 800.0
                    * (1.0 + 0.02 * i as f64)
                    * (-(j as f64) * 0.15).exp()
                    * (1.0 + 0.015 * (((i * 5 + j * 13) % 7) as f64 - 3.0));
                acc += inc;
                data35[i * n35 + j] = acc;
            }
        }
        let tri35 = Triangle::new(n35, n35, data35);
        let odp35 = fit_odp(&tri35.view(), 1.0, MonotonicityPolicy::Strict).expect("35x35 odp");
        let glm35 = fit_glm(&tri35.view(), &GlmOptions::odp()).expect("35x35 glm");
        assert_close(
            glm35.total_reserve,
            odp35.total_reserve,
            1e-10,
            "35x35 reserve",
        );
        assert_close(
            glm35.pearson_dispersion,
            odp35.dispersion,
            1e-10,
            "35x35 dispersion",
        );
        assert_close(
            glm35.total_prediction_error,
            odp35.analytic_total_prediction_error,
            1e-9,
            "35x35 pred_err",
        );
        let h_sum35: f64 = glm35.leverages.iter().sum();
        assert_close(h_sum35, (2 * n35 - 1) as f64, 1e-10, "35x35 lev sum");
    }
}
