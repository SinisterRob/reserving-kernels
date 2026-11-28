//! Tail-dependent copulas for multi-line reserve capital aggregation.
//!
//! Complements Iman–Conover (which has zero extreme tail dependence) with:
//!
//! * **Multivariate Student-$t$ copula (`t_{ν, Σ}`)** with exact Kendall `τ`
//!   calibration (`ρ = sin(πτ/2)`, valid for every elliptical copula),
//!   explicit Spearman rejection (the Gaussian map is inexact for `t_ν`),
//!   opt-in Higham (2002) nearest-correlation repair,
//!   Lentz continued-fraction regularized incomplete beta `I_x(a, b)`, and
//!   exact analytical tail dependence `λ_U = λ_L`.
//! * **$d$-Dimensional Clayton and Gumbel Archimedean copulas** via exact
//!   Marshall–Olkin frailty sampling (log-space Gamma frailty for Clayton; log-domain
//!   Chambers–Mallows–Stuck positive `α`-stable draws for Gumbel).
//! * **Hierarchical / Nested Archimedean Copulas (HAC)** for multi-line
//!   aggregation enforcing McNeil's (2008) sufficient nesting condition
//!   `θ_parent <= θ_child`.

use std::f64::consts::PI;

use crate::capital::correlation::{
    CorrelationRepair, CorrelationType, PreparedCorrelation, prepare_latent_correlation,
    rank_match_reorder, validate_and_convert_correlation_matrix,
};
use crate::stochastic::bootstrap_odp::{DeterministicRng, ln_gamma};
use crate::validate::ValidationError;

pub use crate::capital::correlation::repair_correlation_matrix;

// ===========================================================================
// Special Functions: Regularized Incomplete Beta & Student-t CDF
// ===========================================================================

/// Evaluate the regularized incomplete beta function `I_x(a, b)` for
/// `x ∈ [0, 1]`, `a > 0`, `b > 0` via Lentz's modified continued fraction.
pub(crate) fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_bt = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (-x).ln_1p();
    let bt = ln_bt.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 400;
    const EPS: f64 = 3.0e-14;
    const FPMIN: f64 = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITER {
        let mf = m as f64;
        let m2 = 2.0 * mf;
        // Even step.
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step.
        let aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Student-$t$ cumulative distribution function `F_{t_ν}(x)` for `ν > 0`.
pub fn student_t_cdf(x: f64, nu: f64) -> f64 {
    if !x.is_finite() || !nu.is_finite() || nu <= 0.0 {
        return f64::NAN;
    }
    if nu > 1.0e8 {
        return crate::excess::severity::normal_cdf_exact(x);
    }
    let t_arg = nu / (nu + x * x);
    let ib = regularized_incomplete_beta(t_arg, 0.5 * nu, 0.5);
    if x >= 0.0 { 1.0 - 0.5 * ib } else { 0.5 * ib }
}

/// Exact analytical symmetric upper/lower tail-dependence coefficient
/// `λ_U = λ_L = 2 * F_{t_{ν+1}}(-sqrt((ν + 1)(1 - ρ) / (1 + ρ)))`
/// for a bivariate Student-$t$ copula with `ν > 0` and `ρ ∈ (-1, 1)`.
pub fn student_t_tail_dependence(nu: f64, rho: f64) -> Result<f64, ValidationError> {
    if !nu.is_finite() || !rho.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if nu <= 0.0 {
        return Err(ValidationError::NonPositive("degrees of freedom nu"));
    }
    if !(-1.0..=1.0).contains(&rho) {
        return Err(ValidationError::InvalidParameter(
            "correlation rho must lie in [-1, 1]",
        ));
    }
    if rho >= 1.0 {
        return Ok(1.0);
    }
    if rho <= -1.0 {
        return Ok(0.0);
    }
    let arg = -(((nu + 1.0) * (1.0 - rho)) / (1.0 + rho)).sqrt();
    Ok(2.0 * student_t_cdf(arg, nu + 1.0))
}

// ===========================================================================
// Log-Space Gamma Sampling (frailty / chi-square mixing variables)
// ===========================================================================

/// Draw `ln G` for `G ~ Gamma(shape, 1)` entirely in log space.
///
/// * `shape >= 1`: Marsaglia & Tsang (2000) squeeze/acceptance, returning
///   `ln(d) + 3 ln(v)`.
/// * `shape < 1`: boosting identity `G = G_{shape+1} · U^{1/shape}` evaluated
///   as `ln G_{shape+1} + ln(U) / shape`, which stays finite for arbitrarily
///   small `shape` (e.g. Clayton frailty `shape = 1/θ` with `θ >> 200`), where
///   the linear-domain product underflows to zero.
pub(crate) fn sample_ln_gamma(rng: &mut DeterministicRng, shape: f64) -> f64 {
    if shape < 1.0 {
        let ln_g = sample_ln_gamma_ge1(rng, shape + 1.0);
        let ln_u = rng.next_open01().ln();
        return ln_g + ln_u / shape;
    }
    sample_ln_gamma_ge1(rng, shape)
}

fn sample_ln_gamma_ge1(rng: &mut DeterministicRng, shape: f64) -> f64 {
    let d = shape - (1.0 / 3.0);
    let c = (1.0 / 3.0) / d.sqrt();
    loop {
        let z = rng.next_standard_normal();
        let v_root = 1.0 + c * z;
        if v_root <= 0.0 {
            continue;
        }
        let ln_v = 3.0 * v_root.ln();
        let v = v_root * v_root * v_root;
        let u = rng.next_open01();
        let z2 = z * z;
        if u < 1.0 - 0.0331 * z2 * z2 || u.ln() < 0.5 * z2 + d * (1.0 - v + ln_v) {
            return d.ln() + ln_v;
        }
    }
}

/// Numerically stable softplus `ln(1 + e^x)`.
#[inline]
fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// Clamp a copula draw into the open unit interval without collapsing tails:
/// `[f64::MIN_POSITIVE, 1 - 2^-53]`.
#[inline]
fn clamp_open_unit(u: f64) -> f64 {
    u.clamp(f64::MIN_POSITIVE, 1.0 - 0.5 * f64::EPSILON)
}

// ===========================================================================
// Copula Sample Output & Empirical Marginal Application
// ===========================================================================

/// Uniform `[0, 1]^d` draws produced by a copula sampler.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CopulaSampleResult {
    /// Dimension `d` (number of variables/lines).
    pub dim: usize,
    /// Number of simulated scenarios `N`.
    pub n_samples: usize,
    /// Column-major uniform draws: `columns[var][scenario] ∈ (0, 1)`.
    pub columns: Vec<Vec<f64>>,
    /// Effective correlation matrix (populated for Student-$t$ copulas; empty
    /// for Archimedean / HAC copulas).
    pub effective_correlation: Vec<f64>,
    /// Frobenius distance `‖Σ_repaired − Σ_target‖_F` when the latent target
    /// was repaired under [`CorrelationRepair::Higham`]; `None` otherwise.
    #[serde(default)]
    pub repair_frobenius_distance: Option<f64>,
}

/// Apply copula uniform draws to empirical marginal scenario vectors by rank
/// matching, preserving each marginal multiset exactly.
pub fn apply_copula_reordering(
    marginals: &[&[f64]],
    copula_uniforms: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, ValidationError> {
    let d = marginals.len();
    if d == 0 || copula_uniforms.len() != d {
        return Err(ValidationError::ShapeMismatch);
    }
    let n = marginals[0].len();
    if n == 0 {
        return Err(ValidationError::ShapeMismatch);
    }
    for col in marginals {
        if col.len() != n {
            return Err(ValidationError::ShapeMismatch);
        }
        if col.iter().any(|v| !v.is_finite()) {
            return Err(ValidationError::NotFinite);
        }
    }
    for u in copula_uniforms {
        if u.len() != n {
            return Err(ValidationError::ShapeMismatch);
        }
        if u.iter().any(|v| !v.is_finite()) {
            return Err(ValidationError::NotFinite);
        }
    }

    // Values-only output: within-value tie order is irrelevant, so no RNG is
    // used (see `rank_match_reorder` tie policy).
    let out = marginals
        .iter()
        .zip(copula_uniforms.iter())
        .map(|(vals, u)| rank_match_reorder(vals, u, None).1)
        .collect();
    Ok(out)
}

// ===========================================================================
// 3.1 Multivariate Student-t Copula
// ===========================================================================

/// Sample `n_samples` vectors from a `dim`-dimensional Student-$t$ copula
/// `t_{ν, Σ}`, rejecting non-positive-definite latent targets.
///
/// Equivalent to [`student_t_copula_sample_with_repair`] with
/// [`CorrelationRepair::Reject`].
pub fn student_t_copula_sample(
    target_corr: &[f64],
    dim: usize,
    nu: f64,
    corr_type: CorrelationType,
    n_samples: usize,
    seed: u64,
) -> Result<CopulaSampleResult, ValidationError> {
    student_t_copula_sample_with_repair(
        target_corr,
        dim,
        nu,
        corr_type,
        n_samples,
        seed,
        CorrelationRepair::Reject,
    )
}

/// Sample `n_samples` vectors from a `dim`-dimensional Student-$t$ copula
/// `t_{ν, Σ}` with an explicit non-positive-definite [`CorrelationRepair`] policy.
///
/// * [`CorrelationType::LatentGaussian`]: `target_corr` is `Σ` directly.
/// * [`CorrelationType::KendallTarget`]: `Σ_ij = sin(π τ_ij / 2)`, exact for
///   every elliptical copula including `t_ν` (Lindskog, McNeil & Schmock 2003).
/// * [`CorrelationType::SpearmanTarget`] is **rejected**: the Gaussian map
///   `2 sin(π ρ_S / 6)` is not the Student-$t$ relation (it depends on `ν`;
///   e.g. at `ν = 2` a `ρ_S = 0.5` target achieves only `≈ 0.47`). Supply
///   Kendall's `τ` instead.
///
/// The chi-square mixing variable is drawn in log space, so tiny `ν` cannot
/// underflow the radial scale.
pub fn student_t_copula_sample_with_repair(
    target_corr: &[f64],
    dim: usize,
    nu: f64,
    corr_type: CorrelationType,
    n_samples: usize,
    seed: u64,
    repair: CorrelationRepair,
) -> Result<CopulaSampleResult, ValidationError> {
    if dim < 2 || target_corr.len() != dim * dim {
        return Err(ValidationError::ShapeMismatch);
    }
    if n_samples == 0 {
        return Err(ValidationError::InvalidIterations);
    }
    if !nu.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if nu <= 0.0 {
        return Err(ValidationError::NonPositive("degrees of freedom nu"));
    }
    if corr_type == CorrelationType::SpearmanTarget {
        return Err(ValidationError::InvalidParameter(
            "Spearman targets are not supported for the Student-t copula (the Gaussian map 2 sin(pi rho_S / 6) is inexact for t_nu); use CorrelationType::KendallTarget, which is exact for elliptical copulas",
        ));
    }

    let latent = validate_and_convert_correlation_matrix(target_corr, dim, corr_type)?;
    let (sigma, chol, repair_frobenius_distance): PreparedCorrelation =
        prepare_latent_correlation(latent, dim, repair)?;

    let mut rng = DeterministicRng::from_seed(seed);
    let mut columns: Vec<Vec<f64>> = (0..dim).map(|_| Vec::with_capacity(n_samples)).collect();
    let mut z = vec![0.0; dim];
    let ln_nu = nu.ln();

    for _ in 0..n_samples {
        for zi in &mut z {
            *zi = rng.next_standard_normal();
        }
        // W ~ Chi^2(nu) = 2 * Gamma(nu / 2, 1), drawn as ln W.
        let ln_w = std::f64::consts::LN_2 + sample_ln_gamma(&mut rng, 0.5 * nu);
        let scale = (0.5 * (ln_nu - ln_w)).exp();

        for i in 0..dim {
            let mut lx = 0.0;
            let row = &chol[i * dim..i * dim + (i + 1)];
            for (&lik, &zk) in row.iter().zip(z[..=i].iter()) {
                lx += lik * zk;
            }
            let x_i = scale * lx;
            columns[i].push(student_t_cdf(x_i, nu));
        }
    }

    Ok(CopulaSampleResult {
        dim,
        n_samples,
        columns,
        effective_correlation: sigma,
        repair_frobenius_distance,
    })
}

// ===========================================================================
// 3.2 d-Dimensional Clayton & Gumbel Archimedean Copulas (Marshall–Olkin)
// ===========================================================================

/// Convert Kendall's `τ ∈ (0, 1)` to Clayton parameter `θ = 2τ / (1 - τ)`.
pub fn clayton_theta_from_tau(tau: f64) -> Result<f64, ValidationError> {
    if !tau.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if !(0.0..1.0).contains(&tau) {
        return Err(ValidationError::InvalidParameter(
            "Clayton Kendall tau must lie in [0, 1)",
        ));
    }
    Ok(2.0 * tau / (1.0 - tau))
}

/// Analytical lower tail-dependence coefficient `λ_L = 2^{-1/θ}` for Clayton (`θ > 0`).
pub fn clayton_lower_tail_dependence(theta: f64) -> Result<f64, ValidationError> {
    if !theta.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if theta < 0.0 {
        return Err(ValidationError::Negative);
    }
    if theta < 1e-12 {
        return Ok(0.0);
    }
    Ok(2.0_f64.powf(-1.0 / theta))
}

/// Convert Kendall's `τ ∈ [0, 1)` to Gumbel parameter `θ = 1 / (1 - τ)`.
pub fn gumbel_theta_from_tau(tau: f64) -> Result<f64, ValidationError> {
    if !tau.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if !(0.0..1.0).contains(&tau) {
        return Err(ValidationError::InvalidParameter(
            "Gumbel Kendall tau must lie in [0, 1)",
        ));
    }
    Ok(1.0 / (1.0 - tau))
}

/// Analytical upper tail-dependence coefficient `λ_U = 2 - 2^{1/θ}` for Gumbel (`θ >= 1`).
pub fn gumbel_upper_tail_dependence(theta: f64) -> Result<f64, ValidationError> {
    if !theta.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if theta < 1.0 {
        return Err(ValidationError::InvalidParameter(
            "Gumbel parameter theta must be >= 1.0",
        ));
    }
    Ok(2.0 - 2.0_f64.powf(1.0 / theta))
}

/// Draw `ln V` for a positive `α`-stable random variable (`α ∈ (0, 1]`) via
/// log-domain Chambers–Mallows–Stuck (1976) with underflow/overflow guards.
#[inline]
fn sample_log_positive_stable(alpha: f64, rng: &mut DeterministicRng) -> f64 {
    if alpha >= 1.0 - 1e-12 {
        return 0.0; // Degenerate point mass at V = 1 (ln V = 0).
    }
    let phi = (rng.next_open01() * PI).clamp(1e-12, PI - 1e-12);
    let w = (-rng.next_open01().ln()).max(1e-300);
    let one_minus_a = 1.0 - alpha;
    let term1 = (alpha * phi).sin().max(1e-300).ln();
    let term2 = phi.sin().max(1e-300).ln() / alpha;
    let term3 = (one_minus_a / alpha) * ((one_minus_a * phi).sin().max(1e-300).ln() - w.ln());
    term1 - term2 + term3
}

/// Sample `n_samples` vectors from a `dim`-dimensional Clayton copula (`θ >= 0`)
/// via exact Marshall–Olkin Gamma frailty sampling.
pub fn clayton_copula_sample(
    dim: usize,
    theta: f64,
    n_samples: usize,
    seed: u64,
) -> Result<CopulaSampleResult, ValidationError> {
    if dim < 2 {
        return Err(ValidationError::InvalidParameter(
            "Clayton copula requires dim >= 2",
        ));
    }
    if n_samples == 0 {
        return Err(ValidationError::InvalidIterations);
    }
    if !theta.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if theta < 0.0 {
        return Err(ValidationError::Negative);
    }

    let mut rng = DeterministicRng::from_seed(seed);
    let mut columns: Vec<Vec<f64>> = (0..dim).map(|_| Vec::with_capacity(n_samples)).collect();

    if theta < 1e-9 {
        for _ in 0..n_samples {
            for col in &mut columns {
                col.push(rng.next_open01());
            }
        }
        return Ok(CopulaSampleResult {
            dim,
            n_samples,
            columns,
            effective_correlation: Vec::new(),
            repair_frobenius_distance: None,
        });
    }

    // Marshall–Olkin: V ~ Gamma(1/θ, 1), E_j ~ Exp(1), U_j = ψ(E_j / V) with
    // ψ(t) = (1 + t)^{-1/θ}. Everything is evaluated in log space:
    // ln(1 + E/V) = softplus(ln E − ln V), so huge θ (V ≈ e^{-θ}) never
    // underflows.
    let inv_theta = 1.0 / theta;
    for _ in 0..n_samples {
        let ln_v = sample_ln_gamma(&mut rng, inv_theta);
        for col in &mut columns {
            let ln_e = (-rng.next_open01().ln()).ln();
            let u = (-inv_theta * softplus(ln_e - ln_v)).exp();
            col.push(clamp_open_unit(u));
        }
    }

    Ok(CopulaSampleResult {
        dim,
        n_samples,
        columns,
        effective_correlation: Vec::new(),
        repair_frobenius_distance: None,
    })
}

/// Sample `n_samples` vectors from a `dim`-dimensional Gumbel copula (`θ >= 1`)
/// via log-domain Chambers–Mallows–Stuck Marshall–Olkin sampling.
pub fn gumbel_copula_sample(
    dim: usize,
    theta: f64,
    n_samples: usize,
    seed: u64,
) -> Result<CopulaSampleResult, ValidationError> {
    if dim < 2 {
        return Err(ValidationError::InvalidParameter(
            "Gumbel copula requires dim >= 2",
        ));
    }
    if n_samples == 0 {
        return Err(ValidationError::InvalidIterations);
    }
    if !theta.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if theta < 1.0 {
        return Err(ValidationError::InvalidParameter(
            "Gumbel parameter theta must be >= 1.0",
        ));
    }

    let mut rng = DeterministicRng::from_seed(seed);
    let mut columns: Vec<Vec<f64>> = (0..dim).map(|_| Vec::with_capacity(n_samples)).collect();

    if theta < 1.0 + 1e-9 {
        for _ in 0..n_samples {
            for col in &mut columns {
                col.push(rng.next_open01());
            }
        }
        return Ok(CopulaSampleResult {
            dim,
            n_samples,
            columns,
            effective_correlation: Vec::new(),
            repair_frobenius_distance: None,
        });
    }

    let alpha = 1.0 / theta;
    for _ in 0..n_samples {
        let ln_v = sample_log_positive_stable(alpha, &mut rng);
        for col in &mut columns {
            let e = (-rng.next_open01().ln()).max(1e-300);
            let inner = (alpha * (e.ln() - ln_v)).clamp(-700.0, 700.0);
            let u = (-inner.exp()).exp();
            col.push(clamp_open_unit(u));
        }
    }

    Ok(CopulaSampleResult {
        dim,
        n_samples,
        columns,
        effective_correlation: Vec::new(),
        repair_frobenius_distance: None,
    })
}

// ===========================================================================
// 3.3 Hierarchical / Nested Archimedean Copula (Gumbel HAC)
// ===========================================================================

/// A cluster (child branch) in a two-level Gumbel Hierarchical Archimedean
/// Copula (HAC).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GumbelHacCluster {
    /// Child Gumbel parameter `θ_c >= θ_root`.
    pub theta: f64,
    /// Number of leaf variables/lines belonging to this cluster (`>= 1`).
    pub size: usize,
}

/// Two-level Gumbel Hierarchical Archimedean Copula (McNeil 2008) enforcing
/// the sufficient nesting condition `1.0 <= θ_root <= θ_c` for all clusters.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GumbelHac {
    root_theta: f64,
    clusters: Vec<GumbelHacCluster>,
    total_dim: usize,
}

impl GumbelHac {
    /// Construct and validate a Gumbel HAC tree, enforcing `1 <= θ_root <= θ_c`.
    pub fn try_new(
        root_theta: f64,
        clusters: Vec<GumbelHacCluster>,
    ) -> Result<Self, ValidationError> {
        if !root_theta.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if root_theta < 1.0 {
            return Err(ValidationError::InvalidParameter(
                "Gumbel HAC root_theta must be >= 1.0",
            ));
        }
        if clusters.is_empty() {
            return Err(ValidationError::InvalidParameter(
                "Gumbel HAC requires at least one cluster",
            ));
        }
        let mut total_dim = 0usize;
        for c in &clusters {
            if !c.theta.is_finite() {
                return Err(ValidationError::NotFinite);
            }
            if c.size == 0 {
                return Err(ValidationError::InvalidParameter(
                    "Gumbel HAC cluster size must be >= 1",
                ));
            }
            if c.theta < root_theta {
                return Err(ValidationError::InvalidParameter(
                    "nesting condition violated: child theta must be >= parent root_theta",
                ));
            }
            total_dim += c.size;
        }
        if total_dim < 2 {
            return Err(ValidationError::InvalidParameter(
                "Gumbel HAC requires at least 2 total variables",
            ));
        }
        Ok(Self {
            root_theta,
            clusters,
            total_dim,
        })
    }

    /// Root Gumbel parameter `θ_0`.
    pub const fn root_theta(&self) -> f64 {
        self.root_theta
    }

    /// Total number of leaf variables across all clusters.
    pub const fn total_dim(&self) -> usize {
        self.total_dim
    }

    /// Cluster definitions.
    pub fn clusters(&self) -> &[GumbelHacCluster] {
        &self.clusters
    }

    /// Draw `n_samples` vectors from the Gumbel HAC in log-domain.
    pub fn sample(
        &self,
        n_samples: usize,
        seed: u64,
    ) -> Result<CopulaSampleResult, ValidationError> {
        if n_samples == 0 {
            return Err(ValidationError::InvalidIterations);
        }
        let mut rng = DeterministicRng::from_seed(seed);
        let mut columns: Vec<Vec<f64>> = (0..self.total_dim)
            .map(|_| Vec::with_capacity(n_samples))
            .collect();

        let alpha_0 = 1.0 / self.root_theta;
        for _ in 0..n_samples {
            let ln_v0 = sample_log_positive_stable(alpha_0, &mut rng);
            let mut col_idx = 0usize;
            for c in &self.clusters {
                let alpha_c = 1.0 / c.theta;
                let alpha_0c = (self.root_theta / c.theta).clamp(1e-12, 1.0);
                let ln_s = sample_log_positive_stable(alpha_0c, &mut rng);
                let ln_v0c = ln_s + ln_v0 / alpha_0c;
                for _ in 0..c.size {
                    let e = (-rng.next_open01().ln()).max(1e-300);
                    let inner = (alpha_c * (e.ln() - ln_v0c)).clamp(-700.0, 700.0);
                    let u = (-inner.exp()).exp();
                    columns[col_idx].push(clamp_open_unit(u));
                    col_idx += 1;
                }
            }
        }

        Ok(CopulaSampleResult {
            dim: self.total_dim,
            n_samples,
            columns,
            effective_correlation: Vec::new(),
            repair_frobenius_distance: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ks_uniform_statistic(vals: &[f64]) -> f64 {
        let mut sorted = vals.to_vec();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len() as f64;
        let mut d_max = 0.0_f64;
        for (i, &u) in sorted.iter().enumerate() {
            let f_hi = (i + 1) as f64 / n;
            let f_lo = i as f64 / n;
            d_max = d_max.max((f_hi - u).abs()).max((u - f_lo).abs());
        }
        d_max
    }

    fn empirical_upper_tail_dep(u1: &[f64], u2: &[f64], q: f64) -> f64 {
        let mut denom = 0usize;
        let mut num = 0usize;
        for (&a, &b) in u1.iter().zip(u2.iter()) {
            if a > q {
                denom += 1;
                if b > q {
                    num += 1;
                }
            }
        }
        num as f64 / denom.max(1) as f64
    }

    #[test]
    fn analytical_tail_dependence_identities_match_specification() {
        // Gumbel theta = 2.0 => lambda_U = 2 - sqrt(2) ~= 0.5857864376.
        let lam_g = gumbel_upper_tail_dependence(2.0).expect("gumbel");
        assert!((lam_g - (2.0 - 2.0_f64.sqrt())).abs() < 1e-12);

        // Student-t (nu = 4, rho = 0.5) => 2 * F_{t_5}(-sqrt(5/3)) = 0.2531699951...
        let lam_t = student_t_tail_dependence(4.0, 0.5).expect("student-t");
        assert!((lam_t - 0.253_169_995_1).abs() < 1e-8, "lam_t={lam_t}");

        // Clayton theta = 2.0 => lambda_L = 2^{-1/2} ~= 0.70710678.
        let lam_c = clayton_lower_tail_dependence(2.0).expect("clayton");
        assert!((lam_c - 2.0_f64.powf(-0.5)).abs() < 1e-12);
    }

    #[test]
    fn student_t_and_gumbel_marginals_and_tail_dependence_hold() {
        let n = 50_000usize;
        let ks_crit = 1.63 / (n as f64).sqrt(); // 1% critical value

        // Gumbel (theta = 2.0)
        let g = gumbel_copula_sample(2, 2.0, n, 42069).expect("gumbel sample");
        for col in &g.columns {
            assert!(ks_uniform_statistic(col) < ks_crit);
        }
        let emp_g = empirical_upper_tail_dep(&g.columns[0], &g.columns[1], 0.99);
        let ana_g = gumbel_upper_tail_dependence(2.0).expect("ana");
        assert!(
            (emp_g - ana_g).abs() < 0.06,
            "gumbel empirical={emp_g}, analytical={ana_g}"
        );

        // Student-t (nu = 4, rho = 0.5)
        let corr = [1.0, 0.5, 0.5, 1.0];
        let t = student_t_copula_sample(&corr, 2, 4.0, CorrelationType::LatentGaussian, n, 42069)
            .expect("student-t sample");
        for col in &t.columns {
            assert!(ks_uniform_statistic(col) < ks_crit);
        }
        let emp_t = empirical_upper_tail_dep(&t.columns[0], &t.columns[1], 0.99);
        let ana_t = student_t_tail_dependence(4.0, 0.5).expect("ana");
        assert!(
            (emp_t - ana_t).abs() < 0.06,
            "student-t empirical={emp_t}, analytical={ana_t}"
        );
    }

    #[test]
    fn student_t_rejects_non_pd_by_default_and_reports_repair_distance() {
        #[rustfmt::skip]
        let bad = [
            1.0,  0.9,  0.9,
            0.9,  1.0, -0.9,
            0.9, -0.9,  1.0,
        ];
        assert!(
            student_t_copula_sample(&bad, 3, 4.0, CorrelationType::LatentGaussian, 10, 42069)
                .is_err()
        );
        let res = student_t_copula_sample_with_repair(
            &bad,
            3,
            4.0,
            CorrelationType::LatentGaussian,
            10,
            42069,
            CorrelationRepair::Higham(crate::capital::correlation::HighamOptions::default()),
        )
        .expect("repaired");
        let dist = res.repair_frobenius_distance.expect("distance");
        assert!(dist > 0.1 && dist.is_finite());
    }

    #[test]
    fn non_psd_correlation_repair_produces_valid_spd_correlation() {
        #[rustfmt::skip]
        let bad = [
            1.0,  0.9,  0.9,
            0.9,  1.0, -0.9,
            0.9, -0.9,  1.0,
        ];
        assert!(crate::linalg::cholesky_lower(&bad, 3).is_err());
        let fixed = repair_correlation_matrix(&bad, 3).expect("repair");
        assert!(crate::linalg::cholesky_lower(&fixed, 3).is_ok());
        for i in 0..3 {
            assert!((fixed[i * 3 + i] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn gumbel_hac_enforces_nesting_condition_and_preserves_cluster_hierarchy() {
        // Violates child >= parent.
        let err = GumbelHac::try_new(
            2.5,
            vec![
                GumbelHacCluster {
                    theta: 2.0,
                    size: 2,
                },
                GumbelHacCluster {
                    theta: 3.0,
                    size: 2,
                },
            ],
        );
        assert!(err.is_err());

        // Valid hierarchy: within-cluster theta=3.0 > across-cluster theta=1.5.
        let hac = GumbelHac::try_new(
            1.5,
            vec![
                GumbelHacCluster {
                    theta: 3.0,
                    size: 2,
                },
                GumbelHacCluster {
                    theta: 2.5,
                    size: 2,
                },
            ],
        )
        .expect("valid hac");
        let draws = hac.sample(15_000, 42069).expect("hac sample");
        let within_c1 =
            crate::capital::correlation::spearman_correlation(&draws.columns[0], &draws.columns[1]);
        let across =
            crate::capital::correlation::spearman_correlation(&draws.columns[0], &draws.columns[2]);
        assert!(
            within_c1 > across + 0.25,
            "within={within_c1}, across={across}"
        );
    }
}
