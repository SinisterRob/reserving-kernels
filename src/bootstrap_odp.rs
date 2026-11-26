//! Over-Dispersed Poisson (ODP) GLM and residual bootstrap reserving kernel.
//!
//! Implements the classical ODP reserving and bootstrap framework from:
//!
//! * England & Verrall (1999), *Analytic and Bootstrap Estimates of Prediction
//!   Errors in Claims Reserving*, Insurance: Mathematics and Economics,
//!   Vol. 25, No. 3, pp. 281–293.
//! * England & Verrall (2002), *Stochastic Claims Reserving in General
//!   Insurance*, British Actuarial Journal, Vol. 8, No. 3, pp. 443–518.
//! * Shapland (2016), *Using the ODP Bootstrap Model: A Practitioner's Guide*,
//!   CAS Monograph No. 4.
//!
//! ## Model Specification
//!
//! For incremental payments `q_{i,j}` with `E[q_{i,j}] = μ_{i,j}` and
//! `Var(q_{i,j}) = φ · μ_{i,j}`, the log-linear ODP GLM
//! `log(μ_{i,j}) = c + α_i + β_j` produces fitted expectations identical to
//! volume-weighted chain ladder via backwards recursion from the observed
//! latest diagonal.
//!
//! Two mutually exclusive residual adjustments are supported:
//!
//! * [`ResidualAdjustment::DegreesOfFreedom`] — England & Verrall (1999, 2002)
//!   global degrees-of-freedom scaling `r_adj = sqrt(N / (N - p)) * r`.
//! * [`ResidualAdjustment::HatMatrixLeverage`] — Shapland (2016, CAS Monograph
//!   4, §3.4) leverage scaling `r_H = r / sqrt(1 - h_{ii})` computed from the
//!   exact GLM hat matrix `H = W^{1/2} X (X^T W X)^{-1} X^T W^{1/2}`, excluding
//!   structurally saturated corner cells (`1 - h_{ii} ≈ 0`).
//!
//! Process variance in future cells is sampled once using either a moment-
//! matched Gamma distribution (`shape = μ* / φ`, `scale = φ`) or a scaled
//! Over-Dispersed Poisson (`φ · Poisson(μ* / φ)`), driven by a deterministic
//! SplitMix64-seeded Xoshiro256** PRNG.

use std::f64::consts::TAU;

use crate::ldf::{DevelopmentFactorEngine, volume_weighted_ldfs};
use crate::reserve_distribution::{
    ReserveDistributionSummary, summarize as summarize_distribution,
};
use crate::triangle::{cumulative_to_incremental, latest_diagonal};
use crate::types::TriangleView;
use crate::validate::{
    MonotonicityPolicy, ValidationError, validate_monotone_cumulative, validate_tail,
    validate_triangle,
};

// ===========================================================================
// Public Configuration & Result Types
// ===========================================================================

/// Compact bootstrap configuration for the legacy [`bootstrap`] entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BootstrapConfig {
    /// Number of bootstrap iterations.
    pub iterations: usize,
    /// Deterministic seed for reproducible resampling.
    pub seed: u64,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            iterations: 1000,
            seed: 0xC0DE5EED,
        }
    }
}

/// Residual degrees-of-freedom / leverage adjustment convention.
///
/// The two conventions are mutually exclusive (Shapland 2016, §3.4): do not
/// apply both simultaneously.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum ResidualAdjustment {
    /// England & Verrall (1999, 2002): multiply Pearson residuals by
    /// `sqrt(N / (N - p))` over all `N` active observed cells.
    #[default]
    DegreesOfFreedom,
    /// Shapland (2016, CAS Monograph 4, §3.4): divide non-saturated Pearson
    /// residuals by `sqrt(1 - h_{ii})` and exclude structurally saturated
    /// corner cells (`1 - h_{ii} <= 1e-10`) from the resampling pool.
    HatMatrixLeverage,
}

/// Future incremental process-variance distribution for ODP bootstrap draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum ProcessDistribution {
    /// Moment-matched Gamma: `Gamma(shape = μ* / φ, scale = φ)` with mean `μ*`
    /// and variance `φ · μ*` (England & Verrall 2002; Shapland 2016 §3.5).
    #[default]
    Gamma,
    /// Scaled Poisson: `φ · Poisson(μ* / φ)` with mean `μ*` and variance `φ · μ*`.
    OverDispersedPoisson,
    /// Omit process variance (parameter estimation uncertainty only).
    None,
}

/// Full configuration for [`bootstrap_odp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OdpBootstrapOptions {
    /// Number of bootstrap iterations.
    pub iterations: usize,
    /// Deterministic PRNG seed.
    pub seed: u64,
    /// Residual scaling convention (degrees of freedom vs. hat-matrix leverage).
    pub residual_adjustment: ResidualAdjustment,
    /// Process-variance distribution for future cells.
    pub process_distribution: ProcessDistribution,
    /// Whether to center the eligible adjusted residual pool to zero mean
    /// before resampling (Shapland 2016, §3.4.3).
    pub center_residuals: bool,
    /// Monotonicity policy for triangle validation.
    pub monotonicity_policy: MonotonicityPolicy,
}

impl Default for OdpBootstrapOptions {
    fn default() -> Self {
        Self::england_verrall(1000, 42069)
    }
}

impl OdpBootstrapOptions {
    /// Classic England & Verrall (1999, 2002) specification: global
    /// `sqrt(N / (N - p))` residual scaling, Gamma process variance, uncentered.
    pub const fn england_verrall(iterations: usize, seed: u64) -> Self {
        Self {
            iterations,
            seed,
            residual_adjustment: ResidualAdjustment::DegreesOfFreedom,
            process_distribution: ProcessDistribution::Gamma,
            center_residuals: false,
            monotonicity_policy: MonotonicityPolicy::Strict,
        }
    }

    /// Shapland (2016, CAS Monograph 4) practitioner specification: exact
    /// hat-matrix leverage `1 / sqrt(1 - h_{ii})`, centered eligible pool,
    /// Gamma process variance.
    pub const fn shapland(iterations: usize, seed: u64) -> Self {
        Self {
            iterations,
            seed,
            residual_adjustment: ResidualAdjustment::HatMatrixLeverage,
            process_distribution: ProcessDistribution::Gamma,
            center_residuals: true,
            monotonicity_policy: MonotonicityPolicy::Strict,
        }
    }

    /// Set the number of bootstrap iterations.
    pub const fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set the deterministic PRNG seed.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the residual adjustment convention (`DegreesOfFreedom` vs. `HatMatrixLeverage`).
    pub const fn with_residual_adjustment(mut self, adjustment: ResidualAdjustment) -> Self {
        self.residual_adjustment = adjustment;
        self
    }

    /// Set the future process-variance distribution (`Gamma`, `OverDispersedPoisson`, or `None`).
    pub const fn with_process_distribution(mut self, dist: ProcessDistribution) -> Self {
        self.process_distribution = dist;
        self
    }

    /// Enable or disable centering the eligible adjusted residual pool before resampling.
    pub const fn with_centered_residuals(mut self, center: bool) -> Self {
        self.center_residuals = center;
        self
    }

    /// Set the triangle monotonicity policy.
    pub const fn with_policy(mut self, policy: MonotonicityPolicy) -> Self {
        self.monotonicity_policy = policy;
        self
    }
}

/// Bootstrap reserve summary shared with the generic reserve-distribution layer.
pub type BootstrapSummary = ReserveDistributionSummary;

/// Compact bootstrap output returned by [`bootstrap`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BootstrapResult {
    /// Original chain-ladder / ODP total reserve.
    pub base_total_reserve: f64,
    /// Simulated reserve totals by iteration.
    pub simulated_totals: Vec<f64>,
    /// Summary percentiles and moments.
    pub summary: BootstrapSummary,
}

/// Cell-level residual and hat-matrix leverage record from [`fit_odp`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OdpCellResidual {
    /// Origin row index.
    pub origin: usize,
    /// Development column index.
    pub dev: usize,
    /// Observed incremental payment `q_{i,j}`.
    pub observed_incremental: f64,
    /// ODP GLM / chain-ladder fitted incremental expectation `μ_{i,j}`.
    pub fitted_incremental: f64,
    /// Unscaled Pearson residual `(q_{i,j} - μ_{i,j}) / sqrt(μ_{i,j})`.
    pub raw_pearson_residual: f64,
    /// Diagonal entry `h_{ii}` of the GLM hat matrix `H`.
    pub leverage: f64,
    /// `true` when `1 - h_{ii} <= 1e-10` (structurally saturated corner cell).
    pub saturated: bool,
}

/// Immutable ODP GLM fit, dispersion, leverages, and analytic prediction errors.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OdpFit {
    /// Number of origin rows.
    pub n_origins: usize,
    /// Number of development columns.
    pub n_devs: usize,
    /// Volume-weighted chain-ladder LDFs (length `n - 1`).
    pub ldfs: Vec<f64>,
    /// Multiplicative tail factor (`>= 1.0`).
    pub tail: f64,
    /// Full `n × n` fitted cumulative grid (row-major; observed region by
    /// backward recursion, future region by chain-ladder projection).
    pub fitted_cumulative: Vec<f64>,
    /// Full `n × n` fitted incremental grid (row-major).
    pub fitted_incremental: Vec<f64>,
    /// Observed `n × n` incremental grid (`NaN` in unobserved lower-right).
    pub observed_incremental: Vec<f64>,
    /// Residual and leverage diagnostics for every active observed cell.
    pub residuals: Vec<OdpCellResidual>,
    /// Count of active observed cells entering the ODP fit (`N`).
    pub n_observed: usize,
    /// Effective parameter rank of the ODP design matrix (`p = 2n - 1` for full triangles).
    pub rank: usize,
    /// Residual degrees of freedom (`N - p`).
    pub degrees_of_freedom: usize,
    /// Pearson dispersion parameter `φ = sum(r^2) / (N - p)`.
    pub dispersion: f64,
    /// Point-estimate reserves by origin (including tail when `tail > 1`).
    pub origin_reserves: Vec<f64>,
    /// Aggregate point-estimate reserve.
    pub total_reserve: f64,
    /// Analytic ODP process standard error by origin (pre-tail).
    pub analytic_origin_process_se: Vec<f64>,
    /// Analytic ODP parameter estimation standard error by origin (pre-tail).
    pub analytic_origin_param_se: Vec<f64>,
    /// Analytic ODP total prediction error by origin (England & Verrall 1999, Table 2).
    pub analytic_origin_prediction_error: Vec<f64>,
    /// Analytic ODP aggregate process standard error `sqrt(φ · R)`.
    pub analytic_total_process_se: f64,
    /// Analytic ODP aggregate parameter estimation standard error.
    pub analytic_total_param_se: f64,
    /// Analytic ODP aggregate prediction error (England & Verrall 1999, 2002).
    pub analytic_total_prediction_error: f64,
}

/// Full ODP bootstrap output with fit diagnostics, per-origin distributions,
/// and explicit failure accounting.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OdpBootstrapResult {
    /// Underlying ODP GLM fit and analytic prediction errors.
    pub fit: OdpFit,
    /// Options used to generate this simulation.
    pub options: OdpBootstrapOptions,
    /// Base chain-ladder / ODP total reserve.
    pub base_total_reserve: f64,
    /// Simulated aggregate reserves for successful iterations.
    pub simulated_totals: Vec<f64>,
    /// Simulated reserves per origin: `simulated_by_origin[origin][iter]`.
    pub simulated_by_origin: Vec<Vec<f64>>,
    /// Distribution summary for each origin year.
    pub origin_summaries: Vec<ReserveDistributionSummary>,
    /// Distribution summary for the aggregate portfolio reserve.
    pub summary: ReserveDistributionSummary,
    /// Number of bootstrap draws rejected due to degenerate pseudo-triangles
    /// (non-positive column sums or non-finite refits). Never silently clipped.
    pub failed_iterations: usize,
    /// Arithmetic mean of the adjusted residual pool prior to optional centering.
    pub residual_pool_mean_before_centering: f64,
}

// ===========================================================================
// Deterministic PRNG & Exact Samplers (SplitMix64 + Xoshiro256**)
// ===========================================================================

/// Deterministic SplitMix64-seeded Xoshiro256** PRNG with exact Normal, Gamma,
/// and Poisson samplers. Shared with [`crate::dependence`].
#[derive(Debug, Clone)]
pub(crate) struct DeterministicRng {
    s: [u64; 4],
    cached_normal: Option<f64>,
}

impl DeterministicRng {
    pub(crate) fn from_seed(seed: u64) -> Self {
        let mut sm = seed;
        let mut next_sm = || {
            sm = sm.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        };
        let mut s = [next_sm(), next_sm(), next_sm(), next_sm()];
        if s == [0; 4] {
            s[0] = 1;
        }
        Self {
            s,
            cached_normal: None,
        }
    }

    #[inline]
    pub(crate) fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform float strictly in `(0, 1)` (53-bit mantissa precision).
    #[inline]
    pub(crate) fn next_open01(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Unbiased uniform index in `0..upper`.
    #[inline]
    pub(crate) fn next_index(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        let bound = upper as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let r = self.next_u64();
            if r >= threshold {
                return (r % bound) as usize;
            }
        }
    }

    /// Standard normal `N(0, 1)` via paired Box–Muller transform.
    pub(crate) fn next_standard_normal(&mut self) -> f64 {
        if let Some(z) = self.cached_normal.take() {
            return z;
        }
        let u1 = self.next_open01();
        let u2 = self.next_open01();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = TAU * u2;
        self.cached_normal = Some(r * theta.sin());
        r * theta.cos()
    }

    /// Exact Gamma(shape, scale) via Marsaglia & Tsang (2000).
    pub(crate) fn sample_gamma(&mut self, shape: f64, scale: f64) -> f64 {
        if !(shape > 0.0) || !(scale > 0.0) || !shape.is_finite() || !scale.is_finite() {
            return 0.0;
        }
        if shape < 1.0 {
            let g = self.sample_gamma_ge1(shape + 1.0);
            let u = self.next_open01();
            return g * u.powf(1.0 / shape) * scale;
        }
        self.sample_gamma_ge1(shape) * scale
    }

    fn sample_gamma_ge1(&mut self, shape: f64) -> f64 {
        let d = shape - (1.0 / 3.0);
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let z = self.next_standard_normal();
            let v_root = 1.0 + c * z;
            if v_root <= 0.0 {
                continue;
            }
            let v = v_root * v_root * v_root;
            let u = self.next_open01();
            let z2 = z * z;
            if u < 1.0 - 0.0331 * (z2 * z2) {
                return d * v;
            }
            if u.ln() < 0.5 * z2 + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    /// Exact Poisson(lambda) via Knuth (lambda < 30) and Hörmann PTRD (lambda >= 30).
    pub(crate) fn sample_poisson(&mut self, lambda: f64) -> f64 {
        if !(lambda > 0.0) || !lambda.is_finite() {
            return 0.0;
        }
        if lambda < 30.0 {
            let limit = (-lambda).exp();
            let mut k = 0u64;
            let mut p = 1.0;
            loop {
                k += 1;
                p *= self.next_open01();
                if p <= limit {
                    return (k - 1) as f64;
                }
            }
        }
        // Hörmann (1993), Insurance: Mathematics and Economics 12(1), pp. 39–45.
        let s = lambda.sqrt();
        let b = 0.931 + 2.53 * s;
        let a = -0.059 + 0.02483 * b;
        let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
        let v_r = 0.9277 - 3.6224 / (b - 2.0);
        let ln_lambda = lambda.ln();
        loop {
            let u = self.next_open01() - 0.5;
            let v = self.next_open01();
            let us = 0.5 - u.abs();
            let k = ((2.0 * a / us + b) * u + lambda + 0.43).floor();
            if us >= 0.07 && v <= v_r {
                return k;
            }
            if k < 0.0 || (us < 0.013 && v > us) {
                continue;
            }
            let lhs = (v * inv_alpha / (a / (us * us) + b)).ln();
            let rhs = -lambda + k * ln_lambda - ln_gamma(k + 1.0);
            if lhs <= rhs {
                return k;
            }
        }
    }
}

/// Lanczos approximation to `ln Γ(z)` for `z > 0` (accuracy ~1e-14).
pub(crate) fn ln_gamma(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = z - 1.0;
    let mut sum = COEFFS[0];
    for (idx, &coeff) in COEFFS.iter().enumerate().skip(1) {
        sum += coeff / (x + idx as f64);
    }
    let t = x + 7.5;
    0.5 * TAU.ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

// Re-use the crate's scale-invariant Cholesky and triangular solvers.
pub(crate) use crate::linalg::{cholesky_lower, solve_lower};

// ===========================================================================
// ODP GLM Fit (Backward Chain-Ladder Recursion + Exact Hat Matrix)
// ===========================================================================

/// Fit the Over-Dispersed Poisson GLM (`log μ_{i,j} = c + α_i + β_j`) to a
/// cumulative loss triangle.
///
/// Produces fitted expectations, unscaled Pearson residuals, exact GLM hat-
/// matrix diagonal leverages `h_{ii}`, Pearson dispersion `φ`, and England &
/// Verrall (1999, 2002) analytic prediction errors.
pub fn fit_odp(
    triangle: &TriangleView<'_>,
    tail: f64,
    policy: MonotonicityPolicy,
) -> Result<OdpFit, ValidationError> {
    validate_triangle(triangle)?;
    if policy == MonotonicityPolicy::Strict {
        validate_monotone_cumulative(triangle)?;
    }
    validate_tail(tail)?;

    let n = triangle.n_origins;
    if n < 3 {
        return Err(ValidationError::InvalidParameter(
            "ODP model requires at least a 3x3 triangle for positive degrees of freedom",
        ));
    }

    let factors = DevelopmentFactorEngine::volume_weighted_with_policy(triangle, tail, policy)?;
    if factors.ldfs.iter().any(|&f| !f.is_finite() || f < 1.0) {
        return Err(ValidationError::InvalidParameter(
            "ODP GLM requires non-negative fitted expectations (all volume-weighted LDFs >= 1.0)",
        ));
    }
    let observed_incremental = cumulative_to_incremental(triangle);
    let latest = latest_diagonal(triangle);

    // Exact ODP GLM MLE via backward + forward chain-ladder recursion.
    let mut fitted_cumulative = vec![0.0; n * n];
    for origin in 0..n {
        let d = n - 1 - origin;
        fitted_cumulative[origin * n + d] = latest[origin];
        for dev in (0..d).rev() {
            fitted_cumulative[origin * n + dev] =
                fitted_cumulative[origin * n + dev + 1] / factors.ldfs[dev];
        }
        for dev in (d + 1)..n {
            fitted_cumulative[origin * n + dev] =
                fitted_cumulative[origin * n + dev - 1] * factors.ldfs[dev - 1];
        }
    }

    let mut fitted_incremental = vec![0.0; n * n];
    for origin in 0..n {
        fitted_incremental[origin * n] = fitted_cumulative[origin * n];
        for dev in 1..n {
            fitted_incremental[origin * n + dev] =
                fitted_cumulative[origin * n + dev] - fitted_cumulative[origin * n + dev - 1];
        }
    }

    // Identify active origin rows and development columns with strictly positive fitted mass.
    let mut active_origins = vec![false; n];
    let mut active_devs = vec![false; n];
    let mut active_cells: Vec<(usize, usize)> = Vec::new();
    for origin in 0..n {
        for dev in 0..(n - origin) {
            let mu = fitted_incremental[origin * n + dev];
            if mu.is_finite() && mu > 0.0 {
                active_origins[origin] = true;
                active_devs[dev] = true;
                active_cells.push((origin, dev));
            }
        }
    }

    // Build compact parameter index map: col 0 = intercept, then active origins
    // (skipping first active origin as baseline), then active devs (skipping first).
    let mut origin_col = vec![None; n];
    let mut dev_col = vec![None; n];
    let mut param_count = 1usize;
    let mut first_origin_skipped = false;
    for origin in 0..n {
        if active_origins[origin] {
            if !first_origin_skipped {
                first_origin_skipped = true;
            } else {
                origin_col[origin] = Some(param_count);
                param_count += 1;
            }
        }
    }
    let mut first_dev_skipped = false;
    for dev in 0..n {
        if active_devs[dev] {
            if !first_dev_skipped {
                first_dev_skipped = true;
            } else {
                dev_col[dev] = Some(param_count);
                param_count += 1;
            }
        }
    }

    let n_observed = active_cells.len();
    if n_observed <= param_count {
        return Err(ValidationError::InvalidParameter(
            "ODP model has non-positive residual degrees of freedom",
        ));
    }
    let degrees_of_freedom = n_observed - param_count;

    // Assemble Fisher information matrix M = X^T W X where W = diag(mu_{i,j}).
    let mut info = vec![0.0; param_count * param_count];
    for &(origin, dev) in &active_cells {
        let w = fitted_incremental[origin * n + dev];
        let mut cols = [0usize; 3];
        let mut ncols = 1usize;
        if let Some(c) = origin_col[origin] {
            cols[ncols] = c;
            ncols += 1;
        }
        if let Some(c) = dev_col[dev] {
            cols[ncols] = c;
            ncols += 1;
        }
        for a in 0..ncols {
            for b in 0..ncols {
                info[cols[a] * param_count + cols[b]] += w;
            }
        }
    }

    let chol = cholesky_lower(&info, param_count)?;

    // Compute exact hat-matrix diagonal leverages h_{ii} = w_k * ||L^{-1} x_k||^2
    // and unscaled Pearson residuals r_{i,j} = (q_{i,j} - mu_{i,j}) / sqrt(mu_{i,j}).
    let mut residuals = Vec::with_capacity(n_observed);
    let mut pearson_sq_sum = 0.0;
    let mut x_vec = vec![0.0; param_count];
    for &(origin, dev) in &active_cells {
        let obs = observed_incremental[origin * n + dev];
        let mu = fitted_incremental[origin * n + dev];
        let raw_residual = (obs - mu) / mu.sqrt();
        pearson_sq_sum += raw_residual * raw_residual;

        x_vec.fill(0.0);
        x_vec[0] = 1.0;
        if let Some(c) = origin_col[origin] {
            x_vec[c] = 1.0;
        }
        if let Some(c) = dev_col[dev] {
            x_vec[c] = 1.0;
        }
        let y = solve_lower(&chol, param_count, &x_vec);
        let leverage = (mu * y.iter().map(|v| v * v).sum::<f64>()).clamp(0.0, 1.0);
        let saturated = (1.0 - leverage) <= 1e-10;

        residuals.push(OdpCellResidual {
            origin,
            dev,
            observed_incremental: obs,
            fitted_incremental: mu,
            raw_pearson_residual: raw_residual,
            leverage,
            saturated,
        });
    }

    let dispersion = pearson_sq_sum / degrees_of_freedom as f64;

    // Point reserves and England & Verrall (1999, 2002) analytic ODP prediction errors.
    let mut origin_reserves = vec![0.0; n];
    let mut analytic_origin_process_se = vec![0.0; n];
    let mut analytic_origin_param_se = vec![0.0; n];
    let mut analytic_origin_prediction_error = vec![0.0; n];
    let mut g_total = vec![0.0; param_count];
    let mut pre_tail_total_reserve = 0.0;

    for origin in 0..n {
        let mut g_origin = vec![0.0; param_count];
        let mut r_pre_tail = 0.0;
        for dev in (n - origin)..n {
            let mu = fitted_incremental[origin * n + dev];
            r_pre_tail += mu;
            g_origin[0] += mu;
            g_total[0] += mu;
            if let Some(c) = origin_col[origin] {
                g_origin[c] += mu;
                g_total[c] += mu;
            }
            if let Some(c) = dev_col[dev] {
                g_origin[c] += mu;
                g_total[c] += mu;
            }
        }
        pre_tail_total_reserve += r_pre_tail;
        let tail_increment = fitted_cumulative[origin * n + (n - 1)] * (tail - 1.0);
        origin_reserves[origin] = r_pre_tail + tail_increment;

        let y_origin = solve_lower(&chol, param_count, &g_origin);
        let param_var = (dispersion * y_origin.iter().map(|v| v * v).sum::<f64>()).max(0.0);
        let proc_var = (dispersion * r_pre_tail).max(0.0);
        analytic_origin_process_se[origin] = proc_var.sqrt();
        analytic_origin_param_se[origin] = param_var.sqrt();
        analytic_origin_prediction_error[origin] = (proc_var + param_var).sqrt();
    }

    let total_reserve: f64 = origin_reserves.iter().sum();
    let y_total = solve_lower(&chol, param_count, &g_total);
    let total_param_var = (dispersion * y_total.iter().map(|v| v * v).sum::<f64>()).max(0.0);
    let total_proc_var = (dispersion * pre_tail_total_reserve).max(0.0);

    Ok(OdpFit {
        n_origins: n,
        n_devs: n,
        ldfs: factors.ldfs,
        tail,
        fitted_cumulative,
        fitted_incremental,
        observed_incremental,
        residuals,
        n_observed,
        rank: param_count,
        degrees_of_freedom,
        dispersion,
        origin_reserves,
        total_reserve,
        analytic_origin_process_se,
        analytic_origin_param_se,
        analytic_origin_prediction_error,
        analytic_total_process_se: total_proc_var.sqrt(),
        analytic_total_param_se: total_param_var.sqrt(),
        analytic_total_prediction_error: (total_proc_var + total_param_var).sqrt(),
    })
}

/// Build the adjusted residual pool for resampling along with its pre-centering mean.
pub(crate) fn build_adjusted_residual_pool(
    fit: &OdpFit,
    adjustment: ResidualAdjustment,
    center: bool,
) -> Result<(Vec<f64>, f64), ValidationError> {
    let mut pool = Vec::with_capacity(fit.residuals.len());
    match adjustment {
        ResidualAdjustment::DegreesOfFreedom => {
            let scale = (fit.n_observed as f64 / fit.degrees_of_freedom as f64).sqrt();
            for cell in &fit.residuals {
                pool.push(cell.raw_pearson_residual * scale);
            }
        }
        ResidualAdjustment::HatMatrixLeverage => {
            for cell in &fit.residuals {
                if !cell.saturated {
                    pool.push(cell.raw_pearson_residual / (1.0 - cell.leverage).sqrt());
                }
            }
        }
    }
    if pool.is_empty() {
        return Err(ValidationError::InvalidParameter(
            "no eligible residuals available for ODP bootstrap resampling",
        ));
    }
    let pool_mean = pool.iter().sum::<f64>() / pool.len() as f64;
    if center {
        for r in &mut pool {
            *r -= pool_mean;
        }
    }
    Ok((pool, pool_mean))
}

/// Reusable scratch buffers and precomputed `sqrt(mu)` for zero-allocation ODP
/// bootstrap iterations.
pub(crate) struct OdpBootstrapScratch {
    sqrt_mu: Vec<f64>,
    pseudo_cum: Vec<f64>,
    pseudo_ldfs: Vec<f64>,
    origin_reserves: Vec<f64>,
}

impl OdpBootstrapScratch {
    pub(crate) fn for_fit(fit: &OdpFit) -> Self {
        let n = fit.n_origins;
        let sqrt_mu: Vec<f64> = fit
            .fitted_incremental
            .iter()
            .map(|&mu| if mu > 0.0 { mu.sqrt() } else { 0.0 })
            .collect();
        Self {
            sqrt_mu,
            pseudo_cum: vec![f64::NAN; n * n],
            pseudo_ldfs: vec![0.0; n.saturating_sub(1)],
            origin_reserves: vec![0.0; n],
        }
    }
}

/// Execute one ODP bootstrap iteration into reusable `scratch` buffers.
///
/// Returns `Some(&scratch.origin_reserves)` on valid refit or `None` if the
/// pseudo-triangle has a degenerate non-positive column sum.
pub(crate) fn run_odp_iteration_with_scratch<'a>(
    fit: &OdpFit,
    sampled_residuals: &[f64],
    process_distribution: ProcessDistribution,
    rng: &mut DeterministicRng,
    scratch: &'a mut OdpBootstrapScratch,
) -> Option<&'a [f64]> {
    let n = fit.n_origins;
    scratch.pseudo_cum.fill(f64::NAN);
    let mut idx = 0usize;

    for origin in 0..n {
        let mut running = 0.0;
        let row_off = origin * n;
        for dev in 0..(n - origin) {
            let mu = fit.fitted_incremental[row_off + dev];
            let inc = if mu > 0.0 {
                let r = sampled_residuals[idx];
                idx += 1;
                mu + r * scratch.sqrt_mu[row_off + dev]
            } else {
                0.0
            };
            running += inc;
            scratch.pseudo_cum[row_off + dev] = running;
        }
    }

    // Refit volume-weighted LDFs on the pseudo-cumulative triangle.
    volume_weighted_ldfs(&scratch.pseudo_cum, n, n, &mut scratch.pseudo_ldfs);
    if scratch
        .pseudo_ldfs
        .iter()
        .any(|f| !f.is_finite() || *f <= 0.0)
    {
        return None;
    }

    // Square the pseudo-triangle forward and add process variance once per future cell.
    let phi = fit.dispersion;
    for origin in 0..n {
        let latest_dev = n - 1 - origin;
        let mut prev_cum = scratch.pseudo_cum[origin * n + latest_dev];
        if !prev_cum.is_finite() {
            return None;
        }
        let mut reserve_i = 0.0;
        for dev in (latest_dev + 1)..n {
            let next_cum = prev_cum * scratch.pseudo_ldfs[dev - 1];
            let future_mu = next_cum - prev_cum;
            reserve_i += sample_process_increment(future_mu, phi, process_distribution, rng);
            prev_cum = next_cum;
        }
        if fit.tail > 1.0 {
            let tail_mu = prev_cum * (fit.tail - 1.0);
            reserve_i += sample_process_increment(tail_mu, phi, process_distribution, rng);
        }
        if !reserve_i.is_finite() {
            return None;
        }
        scratch.origin_reserves[origin] = reserve_i;
    }

    Some(&scratch.origin_reserves)
}

/// Execute one ODP bootstrap iteration given a pre-sampled array of adjusted
/// residuals (one per observed cell in `(origin, dev)` row-major order).
///
/// Returns `Some(origin_reserves)` on valid refit or `None` if the pseudo-
/// triangle has a degenerate non-positive column sum.
pub(crate) fn run_odp_iteration_from_residuals(
    fit: &OdpFit,
    sampled_residuals: &[f64],
    process_distribution: ProcessDistribution,
    rng: &mut DeterministicRng,
) -> Option<Vec<f64>> {
    let mut scratch = OdpBootstrapScratch::for_fit(fit);
    run_odp_iteration_with_scratch(fit, sampled_residuals, process_distribution, rng, &mut scratch)
        .map(|slice| slice.to_vec())
}

#[inline]
fn sample_process_increment(
    future_mu: f64,
    phi: f64,
    dist: ProcessDistribution,
    rng: &mut DeterministicRng,
) -> f64 {
    if !(phi > 0.0) || future_mu == 0.0 || !future_mu.is_finite() {
        return future_mu;
    }
    // Per Shapland (2016) §3.5.3, preserve process variance on negative pseudo-
    // incremental expectations by reflecting the draw: -Sample(|mu|, phi).
    let sign = if future_mu < 0.0 { -1.0 } else { 1.0 };
    let mag = future_mu.abs();
    let draw = match dist {
        ProcessDistribution::Gamma => rng.sample_gamma(mag / phi, phi),
        ProcessDistribution::OverDispersedPoisson => phi * rng.sample_poisson(mag / phi),
        ProcessDistribution::None => mag,
    };
    sign * draw
}

/// Run the full England & Verrall (1999, 2002) / Shapland (2016) ODP bootstrap.
pub fn bootstrap_odp(
    triangle: &TriangleView<'_>,
    tail: f64,
    options: OdpBootstrapOptions,
) -> Result<OdpBootstrapResult, ValidationError> {
    if options.iterations == 0 {
        return Err(ValidationError::ShapeMismatch);
    }

    let fit = fit_odp(triangle, tail, options.monotonicity_policy)?;
    let (pool, pool_mean) =
        build_adjusted_residual_pool(&fit, options.residual_adjustment, options.center_residuals)?;

    let mut rng = DeterministicRng::from_seed(options.seed);
    let n = fit.n_origins;
    let mut simulated_totals = Vec::with_capacity(options.iterations);
    let mut simulated_by_origin = vec![Vec::with_capacity(options.iterations); n];
    let mut sampled_residuals = vec![0.0; fit.n_observed];
    let mut scratch = OdpBootstrapScratch::for_fit(&fit);
    let mut failed_iterations = 0usize;

    for _ in 0..options.iterations {
        for slot in &mut sampled_residuals {
            *slot = pool[rng.next_index(pool.len())];
        }
        if let Some(origin_reserves) = run_odp_iteration_with_scratch(
            &fit,
            &sampled_residuals,
            options.process_distribution,
            &mut rng,
            &mut scratch,
        ) {
            let total: f64 = origin_reserves.iter().sum();
            if total.is_finite() {
                simulated_totals.push(total);
                for (origin, &r) in origin_reserves.iter().enumerate() {
                    simulated_by_origin[origin].push(r);
                }
            } else {
                failed_iterations += 1;
            }
        } else {
            failed_iterations += 1;
        }
    }

    if simulated_totals.is_empty() {
        return Err(ValidationError::NotFinite);
    }

    let summary = summarize_distribution(&simulated_totals)?;
    let mut origin_summaries = Vec::with_capacity(n);
    for series in &simulated_by_origin {
        origin_summaries.push(summarize_distribution(series)?);
    }

    Ok(OdpBootstrapResult {
        base_total_reserve: fit.total_reserve,
        fit,
        options,
        simulated_totals,
        simulated_by_origin,
        origin_summaries,
        summary,
        failed_iterations,
        residual_pool_mean_before_centering: pool_mean,
    })
}

/// Run a deterministic ODP bootstrap using the classic England & Verrall
/// (1999, 2002) specification (`DegreesOfFreedom` scaling + Gamma process variance).
pub fn bootstrap(
    triangle: &TriangleView<'_>,
    tail: f64,
    config: BootstrapConfig,
) -> Result<BootstrapResult, ValidationError> {
    let full = bootstrap_odp(
        triangle,
        tail,
        OdpBootstrapOptions::england_verrall(config.iterations, config.seed),
    )?;
    Ok(BootstrapResult {
        base_total_reserve: full.base_total_reserve,
        simulated_totals: full.simulated_totals,
        summary: full.summary,
    })
}

/// Reserve-only convenience wrapper around [`bootstrap`].
pub fn bootstrap_summary(
    triangle: &TriangleView<'_>,
    tail: f64,
    config: BootstrapConfig,
) -> Result<BootstrapSummary, ValidationError> {
    Ok(bootstrap(triangle, tail, config)?.summary)
}

/// Output of the 1-year Solvency II Claims Development Result (CDR)
/// re-reserving ODP bootstrap (England & Verrall / Merz & Wüthrich;
/// mirrors `mages/ChainLadder`'s `CDR(BootChainLadder)`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OdpCdrBootstrapResult {
    /// Initial deterministic reserve $R^{(0)}$ summed across origins.
    pub base_total_reserve: f64,
    /// Initial deterministic reserve $R_i^{(0)}$ per origin.
    pub base_origin_reserves: Vec<f64>,
    /// Simulated total 1-year CDR samples: $R^{(0)} - (Y_{\text{next}} + R^{(1)})$.
    /// Negative values represent adverse 1-year development.
    pub simulated_total_cdr: Vec<f64>,
    /// Simulated 1-year CDR samples per origin (`n_origins` vectors).
    pub simulated_cdr_by_origin: Vec<Vec<f64>>,
    /// Empirical summary statistics of total 1-year CDR.
    pub summary: BootstrapSummary,
    /// Empirical summary statistics of 1-year CDR per origin.
    pub origin_summaries: Vec<BootstrapSummary>,
    /// Count of skipped iterations due to degenerate pseudo-triangles.
    pub failed_iterations: usize,
}

/// Run a 1-year Solvency II Claims Development Result (CDR) re-reserving
/// bootstrap on top of the ODP bootstrap model.
///
/// For each simulation $b$:
/// 1. Resample scaled Pearson residuals and refit pseudo-LDFs $\hat{f}_k^{(b)}$.
/// 2. Simulate the next calendar-year incremental payments $Y_{i, n-i}^{(b)}$
///    (with process variance from `options.process_distribution`) for each open
///    origin $i \ge 1$.
/// 3. Append the simulated diagonal $C_{i, n-i}^{(b)} = C_{i, n-1-i} + Y_{i, n-i}^{(b)}$
///    to the observed triangle, re-estimate volume-weighted chain-ladder LDFs
///    on the extended 1-year-later triangle, and compute the updated reserve $R_i^{(1,b)}$.
/// 4. Record $\text{CDR}_i^{(b)}(1) = R_i^{(0)} - (Y_{i, n-i}^{(b)} + R_i^{(1,b)})$.
pub fn bootstrap_odp_cdr(
    triangle: &TriangleView<'_>,
    tail: f64,
    options: OdpBootstrapOptions,
) -> Result<OdpCdrBootstrapResult, ValidationError> {
    if options.iterations == 0 {
        return Err(ValidationError::ShapeMismatch);
    }

    let fit = fit_odp(triangle, tail, options.monotonicity_policy)?;
    let (pool, _) =
        build_adjusted_residual_pool(&fit, options.residual_adjustment, options.center_residuals)?;

    let n = fit.n_origins;
    let m = n.saturating_sub(1);
    let mut rng = DeterministicRng::from_seed(options.seed);

    let mut simulated_total_cdr = Vec::with_capacity(options.iterations);
    let mut simulated_cdr_by_origin = vec![Vec::with_capacity(options.iterations); n];
    let mut sampled_residuals = vec![0.0; fit.n_observed];
    let mut scratch = OdpBootstrapScratch::for_fit(&fit);
    let mut ext_cum = vec![f64::NAN; n * n];
    let mut ext_ldfs = vec![1.0; m];
    let mut next_pay = vec![0.0; n];
    let mut failed_iterations = 0usize;

    for _ in 0..options.iterations {
        for slot in &mut sampled_residuals {
            *slot = pool[rng.next_index(pool.len())];
        }

        // 1. Build pseudo-triangle & refit pseudo-LDFs into scratch (matching
        //    run_odp_iteration_with_scratch so zero-fitted cells don't poison rows).
        scratch.pseudo_cum.fill(f64::NAN);
        let mut r_idx = 0usize;
        for origin in 0..n {
            let mut running = 0.0;
            let row_off = origin * n;
            for dev in 0..(n - origin) {
                let mu = fit.fitted_incremental[row_off + dev];
                let pseudo_inc = if mu > 0.0 {
                    let r = sampled_residuals[r_idx];
                    r_idx += 1;
                    mu + r * scratch.sqrt_mu[row_off + dev]
                } else {
                    0.0
                };
                running += pseudo_inc;
                scratch.pseudo_cum[row_off + dev] = running;
            }
        }
        volume_weighted_ldfs(&scratch.pseudo_cum, n, n, &mut scratch.pseudo_ldfs);
        if scratch
            .pseudo_ldfs
            .iter()
            .any(|f| !f.is_finite() || *f <= 0.0)
        {
            failed_iterations += 1;
            continue;
        }

        // 2. Copy observed triangle & simulate next calendar diagonal (i + j == n).
        ext_cum.copy_from_slice(triangle.data);
        next_pay.fill(0.0);
        for i in 1..n {
            let j_next = n - i;
            let c_prev = triangle.get(i, j_next - 1);
            let expected_inc = c_prev * (scratch.pseudo_ldfs[j_next - 1] - 1.0);
            let sim_inc = sample_process_increment(
                expected_inc,
                fit.dispersion,
                options.process_distribution,
                &mut rng,
            );
            next_pay[i] = sim_inc;
            ext_cum[i * n + j_next] = c_prev + sim_inc;
        }

        // 3. Re-estimate LDFs on the 1-year-extended triangle (observed + simulated diagonal).
        let mut valid_ext = true;
        for k in 0..m {
            let nk_ext = (n - k).min(n);
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..nk_ext {
                let c0 = ext_cum[i * n + k];
                let c1 = ext_cum[i * n + (k + 1)];
                if c0.is_finite() && c1.is_finite() {
                    den += c0;
                    num += c1;
                }
            }
            if !den.is_finite() || den <= 0.0 || !num.is_finite() || num <= 0.0 {
                valid_ext = false;
                break;
            }
            ext_ldfs[k] = num / den;
        }
        if !valid_ext {
            failed_iterations += 1;
            continue;
        }

        // 4. Compute updated reserve R^{(1)} and 1-year CDR = R^{(0)} - (Y_next + R^{(1)}).
        let mut total_cdr = 0.0;
        let mut iter_cdrs = [0.0_f64; 0];
        let _ = &mut iter_cdrs;
        let mut ok = true;
        for i in 0..n {
            let r1 = if i == 0 {
                // Origin 0 is already at age n - 1; only tail remains.
                let c_last = triangle.get(0, n - 1);
                c_last * (tail - 1.0)
            } else {
                let j_next = n - i;
                let mut proj = ext_cum[i * n + j_next];
                for &f in &ext_ldfs[j_next..m] {
                    proj *= f;
                }
                proj * tail - ext_cum[i * n + j_next]
            };
            let cdr_i = fit.origin_reserves[i] - (next_pay[i] + r1);
            if !cdr_i.is_finite() {
                ok = false;
                break;
            }
            scratch.origin_reserves[i] = cdr_i;
            total_cdr += cdr_i;
        }
        if !ok || !total_cdr.is_finite() {
            failed_iterations += 1;
            continue;
        }
        simulated_total_cdr.push(total_cdr);
        for i in 0..n {
            simulated_cdr_by_origin[i].push(scratch.origin_reserves[i]);
        }
    }

    if simulated_total_cdr.is_empty() {
        return Err(ValidationError::NotFinite);
    }

    let summary = summarize_distribution(&simulated_total_cdr)?;
    let mut origin_summaries = Vec::with_capacity(n);
    for series in &simulated_cdr_by_origin {
        origin_summaries.push(summarize_distribution(series)?);
    }

    Ok(OdpCdrBootstrapResult {
        base_total_reserve: fit.total_reserve,
        base_origin_reserves: fit.origin_reserves,
        simulated_total_cdr,
        simulated_cdr_by_origin,
        summary,
        origin_summaries,
        failed_iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain_ladder::ChainLadderEngine;
    use crate::types::Triangle;

    fn triangle_4x4() -> Triangle {
        #[rustfmt::skip]
        let data = vec![
            1000.0, 1500.0, 1700.0, 1800.0,
            1200.0, 1800.0, 2000.0, f64::NAN,
            1500.0, 2100.0, f64::NAN, f64::NAN,
            1800.0, f64::NAN, f64::NAN, f64::NAN,
        ];
        Triangle::new(4, 4, data)
    }

    #[test]
    fn bootstrap_is_deterministic_for_same_seed() {
        let triangle = triangle_4x4();
        let config = BootstrapConfig {
            iterations: 50,
            seed: 42,
        };

        let first = bootstrap(&triangle.view(), 1.0, config).expect("bootstrap");
        let second = bootstrap(&triangle.view(), 1.0, config).expect("bootstrap");

        assert_eq!(first.simulated_totals, second.simulated_totals);
        assert_eq!(first.summary, second.summary);
    }

    #[test]
    fn bootstrap_returns_ordered_quantiles() {
        let triangle = triangle_4x4();
        let summary = bootstrap_summary(
            &triangle.view(),
            1.0,
            BootstrapConfig {
                iterations: 100,
                seed: 7,
            },
        )
        .expect("summary");

        assert!(summary.mean >= 0.0);
        assert!(summary.std_dev >= 0.0);
        assert!(summary.p50 <= summary.p75);
        assert!(summary.p75 <= summary.p90);
        assert!(summary.p90 <= summary.p95);
        assert!(summary.p95 <= summary.p99);
    }

    #[test]
    fn bootstrap_rejects_zero_iterations() {
        let triangle = triangle_4x4();
        assert_eq!(
            bootstrap(
                &triangle.view(),
                1.0,
                BootstrapConfig {
                    iterations: 0,
                    seed: 1
                }
            ),
            Err(ValidationError::ShapeMismatch)
        );
    }

    #[test]
    fn bootstrap_base_reserve_matches_chain_ladder() {
        let triangle = triangle_4x4();
        let result = bootstrap(
            &triangle.view(),
            1.0,
            BootstrapConfig {
                iterations: 10,
                seed: 9,
            },
        )
        .expect("bootstrap");

        let factors =
            DevelopmentFactorEngine::volume_weighted(&triangle.view(), 1.0).expect("factors");
        let projection =
            ChainLadderEngine::project(&triangle.view(), &factors).expect("projection");
        assert!((result.base_total_reserve - projection.total_reserve).abs() < 1e-9);
    }

    #[test]
    fn odp_fit_leverages_sum_to_rank_and_saturate_corners() {
        let triangle = triangle_4x4();
        let fit = fit_odp(&triangle.view(), 1.0, MonotonicityPolicy::Strict).expect("fit");

        assert_eq!(fit.n_observed, 10);
        assert_eq!(fit.rank, 7);
        assert_eq!(fit.degrees_of_freedom, 3);
        let h_sum: f64 = fit.residuals.iter().map(|r| r.leverage).sum();
        assert!((h_sum - 7.0).abs() < 1e-10, "sum(h_ii)={h_sum}");
        let saturated_corners: Vec<(usize, usize)> = fit
            .residuals
            .iter()
            .filter(|r| r.saturated)
            .map(|r| (r.origin, r.dev))
            .collect();
        assert_eq!(saturated_corners, vec![(0, 3), (3, 0)]);
    }

    #[test]
    fn gamma_and_poisson_samplers_match_theoretical_moments() {
        let mut rng = DeterministicRng::from_seed(42069);
        let n = 40_000usize;

        for &(shape, scale) in &[(0.4, 2.5), (3.5, 10.0), (25.0, 50.0)] {
            let samples: Vec<f64> = (0..n).map(|_| rng.sample_gamma(shape, scale)).collect();
            let mean = samples.iter().sum::<f64>() / n as f64;
            let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            let expected_mean = shape * scale;
            let expected_var = shape * scale * scale;
            assert!(
                (mean - expected_mean).abs() / expected_mean < 0.03,
                "gamma({shape},{scale}) mean={mean} expected={expected_mean}"
            );
            assert!(
                (var - expected_var).abs() / expected_var < 0.08,
                "gamma({shape},{scale}) var={var} expected={expected_var}"
            );
        }

        for &lambda in &[0.7, 12.0, 45.0, 300.0] {
            let samples: Vec<f64> = (0..n).map(|_| rng.sample_poisson(lambda)).collect();
            let mean = samples.iter().sum::<f64>() / n as f64;
            let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            assert!(
                (mean - lambda).abs() / lambda < 0.03,
                "poisson({lambda}) mean={mean}"
            );
            assert!(
                (var - lambda).abs() / lambda < 0.06,
                "poisson({lambda}) var={var}"
            );
        }
    }
}
