//! Severity kernels: empirical claim-count ratios and parametric severity
//! distributions for Limited Expected Values (LEV), excess ratios, and
//! layer loss development.
//!
//! Parametric closed-form formulas follow:
//!
//! * Klugman, Panjer & Willmot, *Loss Models: From Data to Decisions* (5th ed.),
//!   Appendix A.
//! * Miccolis (1977), *On the Theory of Increased Limits and Excess of Loss
//!   Pricing*, PCAS Vol. LXIV, pp. 27–59.
//! * Lee (1988), *The Mathematics of Excess of Loss Coverages and Retrospective
//!   Rating — A Graphical Approach*, PCAS Vol. LXXV, pp. 49–77.
//! * Keatinge (1999), *Modeling Losses with the Mixed Exponential Distribution*,
//!   PCAS Vol. LXXXVI, pp. 654–698.

use std::f64::consts::TAU;

use crate::error::ReservingError;
use crate::special::normal_cdf;
use crate::validate::validate_nonnegative_finite_vector;

fn divide_vectors(numerator: &[f64], denominator: &[f64]) -> Result<Vec<f64>, ReservingError> {
    let n = numerator.len();
    validate_nonnegative_finite_vector(numerator, n)?;
    validate_nonnegative_finite_vector(denominator, denominator.len())?;
    if denominator.len() != n {
        return Err(ReservingError::ShapeMismatch);
    }

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if denominator[i] <= 0.0 {
            return Err(ReservingError::DivideByZero("count"));
        }
        out.push(numerator[i] / denominator[i]);
    }
    Ok(out)
}

/// Paid severity by origin, `paid / counts`.
pub fn paid_severity(paid: &[f64], counts: &[f64]) -> Result<Vec<f64>, ReservingError> {
    divide_vectors(paid, counts)
}

/// Incurred severity by origin, `incurred / counts`.
pub fn incurred_severity(incurred: &[f64], counts: &[f64]) -> Result<Vec<f64>, ReservingError> {
    divide_vectors(incurred, counts)
}

/// Ultimate severity by origin, `ultimate_loss / ultimate_count`.
pub fn ultimate_severity(
    ultimate_loss: &[f64],
    ultimate_count: &[f64],
) -> Result<Vec<f64>, ReservingError> {
    divide_vectors(ultimate_loss, ultimate_count)
}

/// Open-claim severity by origin, `case_reserve / open_claims`.
pub fn open_claim_severity(
    case_reserve: &[f64],
    open_claims: &[f64],
) -> Result<Vec<f64>, ReservingError> {
    divide_vectors(case_reserve, open_claims)
}

/// Closed-claim severity by origin, `paid / closed_claims`.
pub fn closed_claim_severity(
    paid: &[f64],
    closed_claims: &[f64],
) -> Result<Vec<f64>, ReservingError> {
    divide_vectors(paid, closed_claims)
}

/// Frequency-severity ultimate by origin, `ultimate_count * severity`.
pub fn frequency_severity_ultimate(
    ultimate_count: &[f64],
    severity: &[f64],
) -> Result<Vec<f64>, ReservingError> {
    let n = ultimate_count.len();
    validate_nonnegative_finite_vector(ultimate_count, n)?;
    validate_nonnegative_finite_vector(severity, severity.len())?;
    if severity.len() != n {
        return Err(ReservingError::ShapeMismatch);
    }
    Ok(ultimate_count
        .iter()
        .zip(severity.iter())
        .map(|(&count, &sev)| count * sev)
        .collect())
}

// ===========================================================================
// Parametric Severity Distributions, Limited Expected Values & Excess Ratios
// ===========================================================================

// ===========================================================================
// Direct Survival-Integral Closed Forms (cancellation-free layers)
// ===========================================================================

/// `|ξ|` below which the GPD is treated as exponential with mean `β`.
const GPD_EXPONENTIAL_XI: f64 = 1e-10;

/// Relative layer width `l / a` below which lognormal layers use Gauss–Legendre.
const LOGNORMAL_NARROW_LAYER_RATIO: f64 = 1e-2;

/// 8-point Gauss–Legendre nodes (positive half) and weights on `[-1, 1]`.
const GL8_NODES: [f64; 4] = [
    0.183_434_642_495_649_8,
    0.525_532_409_916_329,
    0.796_666_477_413_626_7,
    0.960_289_856_497_536_3,
];
const GL8_WEIGHTS: [f64; 4] = [
    0.362_683_783_378_362,
    0.313_706_645_877_887_3,
    0.222_381_034_453_374_5,
    0.101_228_536_290_376_3,
];

/// Exponential layer `∫_a^{a+l} e^{-x/θ} dx = θ e^{-a/θ} (1 − e^{-l/θ})`.
#[inline]
fn exponential_layer(theta: f64, a: f64, l: f64) -> f64 {
    theta * (-a / theta).exp() * -((-l / theta).exp_m1())
}

/// Lomax survival `S(a) = (1 + a/θ)^{-α}`.
#[inline]
fn lomax_survival(alpha: f64, theta: f64, a: f64) -> f64 {
    (-alpha * (a / theta).ln_1p()).exp()
}

/// Lomax layer `(a + θ) S(a) (1 − r^{1-α}) / (α − 1)` with `r = 1 + l / (a + θ)`.
fn lomax_layer(alpha: f64, theta: f64, a: f64, l: f64) -> f64 {
    let base = a + theta;
    let s_a = lomax_survival(alpha, theta, a);
    let ln_r = (l / base).ln_1p();
    if (alpha - 1.0).abs() <= 1e-10 {
        base * s_a * ln_r
    } else {
        let am1 = alpha - 1.0;
        base * s_a * -((-am1 * ln_r).exp_m1()) / am1
    }
}

/// Single-parameter Pareto layer above the threshold (`a >= θ`):
/// `a S(a) (1 − (1 + l/a)^{1-α}) / (α − 1)`.
fn spp_tail_layer(alpha: f64, theta: f64, a: f64, l: f64) -> f64 {
    let s_a = (theta / a).powf(alpha);
    let ln_r = (l / a).ln_1p();
    if (alpha - 1.0).abs() <= 1e-10 {
        a * s_a * ln_r
    } else {
        let am1 = alpha - 1.0;
        a * s_a * -((-am1 * ln_r).exp_m1()) / am1
    }
}

/// GPD survival `S(a) = (1 + ξ a / β)^{-1/ξ}` (`ξ > 0`).
#[inline]
fn gpd_survival(xi: f64, beta: f64, a: f64) -> f64 {
    (-((xi * a / beta).ln_1p()) / xi).exp()
}

/// GPD layer `(β + ξ a) S(a) (1 − r^{-(1-ξ)/ξ}) / (1 − ξ)` with `r = 1 + ξ l / (β + ξ a)`.
fn gpd_layer(xi: f64, beta: f64, a: f64, l: f64) -> f64 {
    if xi.abs() <= GPD_EXPONENTIAL_XI {
        return exponential_layer(beta, a, l);
    }
    let base = beta + xi * a;
    let s_a = gpd_survival(xi, beta, a);
    let ln_r = (xi * l / base).ln_1p();
    if (xi - 1.0).abs() <= 1e-10 {
        base * s_a * ln_r / xi
    } else {
        let omx = 1.0 - xi;
        base * s_a * -((-(omx / xi) * ln_r).exp_m1()) / omx
    }
}

/// `exp(log_mean) * Q(w)` with `Q(w) = 1 − Φ(w)`, evaluated in log space so
/// neither factor overflows or underflows on its own.
fn lognormal_scaled_tail(log_mean: f64, w: f64) -> f64 {
    let q = normal_cdf(-w);
    let ln_q = if q > 1e-300 {
        q.ln()
    } else if w > 0.0 {
        // Mills-ratio asymptotic Q(w) ≈ φ(w)/w · (1 − 1/w² + 3/w⁴) for w >= ~37.
        let inv_w2 = 1.0 / (w * w);
        -0.5 * w * w - (w * TAU.sqrt()).ln() + (1.0 - inv_w2 * (1.0 - 3.0 * inv_w2)).ln()
    } else {
        return 0.0;
    };
    (log_mean + ln_q).exp()
}

/// Lognormal limited expected value `E[X ∧ L]` (single shared implementation,
/// also used by [`crate::excess::excess_layers::LognormalLayerModel`]).
pub(crate) fn lognormal_lev(mu: f64, sigma: f64, limit: f64) -> f64 {
    if limit.is_nan() {
        return f64::NAN;
    }
    if limit <= 0.0 {
        return 0.0;
    }
    let log_mean = mu + 0.5 * sigma * sigma;
    if !limit.is_finite() {
        return log_mean.exp();
    }
    let z = (limit.ln() - mu) / sigma;
    let term1 = if log_mean <= 700.0 {
        let cdf_shifted = normal_cdf(z - sigma);
        if cdf_shifted == 0.0 {
            0.0
        } else {
            log_mean.exp() * cdf_shifted
        }
    } else {
        // Asymptotic Mills-ratio expansion: exp(mu + sigma^2/2) * Phi(z - sigma)
        // cancels exp(sigma^2/2) analytically against exp(-(z - sigma)^2 / 2).
        let t = sigma - z;
        if t > 0.0 {
            let inv_t2 = 1.0 / (t * t);
            let series = 1.0 - inv_t2 * (1.0 - 3.0 * inv_t2);
            (limit * (-0.5 * z * z).exp() / (TAU.sqrt() * t) * series).max(0.0)
        } else {
            limit
        }
    };
    (term1 + limit * normal_cdf(-z)).min(limit)
}

/// Lognormal expected excess `E[(X − c)_+] = e^{μ+σ²/2} Q(z_c − σ) − c Q(z_c)`.
pub(crate) fn lognormal_excess(mu: f64, sigma: f64, c: f64) -> f64 {
    let log_mean = mu + 0.5 * sigma * sigma;
    if c <= 0.0 {
        return log_mean.exp();
    }
    if !c.is_finite() {
        return 0.0;
    }
    let z = (c.ln() - mu) / sigma;
    (lognormal_scaled_tail(log_mean, z - sigma) - c * normal_cdf(-z)).max(0.0)
}

/// Lognormal layer `∫_a^{a+l} S(x) dx` without `LEV` differencing.
pub(crate) fn lognormal_layer(mu: f64, sigma: f64, a: f64, l: f64) -> f64 {
    if !(l > 0.0) {
        return 0.0;
    }
    if a > 0.0 && l <= LOGNORMAL_NARROW_LAYER_RATIO * a {
        // Narrow layer: S is smooth on [a, a + l]; 8-point Gauss–Legendre is
        // accurate to ~machine precision and avoids differencing two excesses.
        let half = 0.5 * l;
        let mid = a + half;
        let surv = |x: f64| normal_cdf(-((x.ln() - mu) / sigma));
        let mut acc = 0.0;
        for (&node, &w) in GL8_NODES.iter().zip(GL8_WEIGHTS.iter()) {
            acc += w * (surv(mid - half * node) + surv(mid + half * node));
        }
        return (half * acc).clamp(0.0, l);
    }
    // Below the median LEV values are far from E[X], so differencing LEVs is
    // benign; above the median difference the (small) complementary excesses.
    let s_a = normal_cdf(-((a.ln() - mu) / sigma));
    if a <= 0.0 || s_a >= 0.5 {
        (lognormal_lev(mu, sigma, a + l) - lognormal_lev(mu, sigma, a)).clamp(0.0, l)
    } else {
        (lognormal_excess(mu, sigma, a) - lognormal_excess(mu, sigma, a + l)).clamp(0.0, l)
    }
}

/// Parametric P&C claim severity distributions with exact closed-form Limited
/// Expected Values `E[X ∧ L]`, survival functions, layer losses, and excess ratios.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SeverityDistribution {
    /// Lognormal `ln(X) ~ N(mu, sigma^2)` (`sigma > 0`).
    Lognormal {
        /// Log-scale location parameter `μ`.
        mu: f64,
        /// Log-scale standard deviation `σ > 0`.
        sigma: f64,
    },
    /// Two-parameter Pareto (Type II / Lomax): `S(x) = (θ / (x + θ))^α`
    /// for `x >= 0` (`α > 0`, `θ > 0`).
    Pareto {
        /// Tail shape parameter `α > 0`.
        alpha: f64,
        /// Scale parameter `θ > 0`.
        theta: f64,
    },
    /// Single-parameter Pareto: `S(x) = (θ / x)^α` for `x >= θ` (`α > 0`, `θ > 0`).
    SingleParameterPareto {
        /// Tail shape parameter `α > 0`.
        alpha: f64,
        /// Threshold scale parameter `θ > 0`.
        theta: f64,
    },
    /// Exponential with mean `θ > 0`: `S(x) = exp(-x / θ)`.
    Exponential {
        /// Mean severity `θ > 0`.
        mean: f64,
    },
    /// Mixed Exponential (Keatinge 1999; ISO ILF standard):
    /// `S(x) = Σ_k w_k · exp(-x / θ_k)` with `w_k > 0`, `Σ w_k = 1`, `θ_k > 0`.
    MixedExponential {
        /// Positive mixture weights summing to `1.0`.
        weights: Vec<f64>,
        /// Positive component exponential means `θ_k`.
        means: Vec<f64>,
    },
    /// Generalized Pareto Distribution (GPD, Peaks-Over-Threshold):
    /// `S(x) = (1 + ξ · x / β)^{-1/ξ}` (`β > 0`, `ξ >= 0` heavy/exponential tail).
    GeneralizedPareto {
        /// Shape parameter `ξ` (`ξ = 0` is exponential; `ξ > 0` is heavy-tailed).
        xi: f64,
        /// Scale parameter `β > 0`.
        beta: f64,
    },
}

impl SeverityDistribution {
    /// Construct a Lognormal severity `ln(X) ~ N(mu, sigma^2)`.
    pub const fn lognormal(mu: f64, sigma: f64) -> Self {
        Self::Lognormal { mu, sigma }
    }

    /// Construct a two-parameter Pareto Type II (Lomax) severity.
    pub const fn pareto(alpha: f64, theta: f64) -> Self {
        Self::Pareto { alpha, theta }
    }

    /// Construct a single-parameter Pareto severity (`x >= theta`).
    pub const fn single_parameter_pareto(alpha: f64, theta: f64) -> Self {
        Self::SingleParameterPareto { alpha, theta }
    }

    /// Construct an Exponential severity with given `mean`.
    pub const fn exponential(mean: f64) -> Self {
        Self::Exponential { mean }
    }

    /// Construct a Mixed Exponential severity from weights and component means.
    pub fn mixed_exponential(weights: impl Into<Vec<f64>>, means: impl Into<Vec<f64>>) -> Self {
        Self::MixedExponential {
            weights: weights.into(),
            means: means.into(),
        }
    }

    /// Construct a Generalized Pareto (GPD) severity with shape `xi` and scale `beta`.
    pub const fn gpd(xi: f64, beta: f64) -> Self {
        Self::GeneralizedPareto { xi, beta }
    }

    /// Discretize onto `0, step, ..., max_index * step` via the Method of Rounding.
    pub fn discretize_rounding(
        &self,
        step: f64,
        max_index: usize,
        capture_upper_tail: bool,
    ) -> Result<Vec<f64>, ReservingError> {
        crate::capital::aggregate_loss::discretize_rounding(
            self,
            step,
            max_index,
            capture_upper_tail,
        )
    }

    /// Discretize onto `0, step, ..., max_index * step` via mean-preserving LEV local moment matching.
    pub fn discretize_mean_preserving(
        &self,
        step: f64,
        max_index: usize,
    ) -> Result<Vec<f64>, ReservingError> {
        crate::capital::aggregate_loss::discretize_mean_preserving(self, step, max_index)
    }

    /// Construct a Lognormal distribution from ground-up mean and coefficient
    /// of variation `cv = sd / mean > 0`.
    pub fn lognormal_from_mean_cv(mean: f64, cv: f64) -> Result<Self, ReservingError> {
        if !mean.is_finite() || !cv.is_finite() {
            return Err(ReservingError::NotFinite);
        }
        if mean <= 0.0 || cv <= 0.0 {
            return Err(ReservingError::InvalidParameter(
                "lognormal mean and CV must be strictly positive",
            ));
        }
        let sigma = (cv * cv).ln_1p().sqrt();
        let mu = mean.ln() - 0.5 * sigma * sigma;
        Ok(Self::Lognormal { mu, sigma })
    }

    /// Validate distribution parameters.
    pub fn validate(&self) -> Result<(), ReservingError> {
        match self {
            Self::Lognormal { mu, sigma } => {
                if !mu.is_finite() || !sigma.is_finite() {
                    return Err(ReservingError::NotFinite);
                }
                if *sigma <= 0.0 {
                    return Err(ReservingError::InvalidParameter(
                        "lognormal sigma must be > 0",
                    ));
                }
            }
            Self::Pareto { alpha, theta } | Self::SingleParameterPareto { alpha, theta } => {
                if !alpha.is_finite() || !theta.is_finite() {
                    return Err(ReservingError::NotFinite);
                }
                if *alpha <= 0.0 || *theta <= 0.0 {
                    return Err(ReservingError::InvalidParameter(
                        "Pareto alpha and theta must be > 0",
                    ));
                }
            }
            Self::Exponential { mean } => {
                if !mean.is_finite() {
                    return Err(ReservingError::NotFinite);
                }
                if *mean <= 0.0 {
                    return Err(ReservingError::InvalidParameter(
                        "exponential mean must be > 0",
                    ));
                }
            }
            Self::MixedExponential { weights, means } => {
                if weights.is_empty() || weights.len() != means.len() {
                    return Err(ReservingError::ShapeMismatch);
                }
                let mut sum_w = 0.0;
                for (&w, &m) in weights.iter().zip(means.iter()) {
                    if !w.is_finite() || !m.is_finite() {
                        return Err(ReservingError::NotFinite);
                    }
                    if w <= 0.0 || m <= 0.0 {
                        return Err(ReservingError::InvalidParameter(
                            "mixed exponential weights and means must be > 0",
                        ));
                    }
                    sum_w += w;
                }
                if (sum_w - 1.0).abs() > 1e-6 {
                    return Err(ReservingError::InvalidParameter(
                        "mixed exponential weights must sum to 1.0",
                    ));
                }
            }
            Self::GeneralizedPareto { xi, beta } => {
                if !xi.is_finite() || !beta.is_finite() {
                    return Err(ReservingError::NotFinite);
                }
                if *beta <= 0.0 {
                    return Err(ReservingError::InvalidParameter(
                        "GPD scale beta must be > 0",
                    ));
                }
                if *xi < 0.0 {
                    return Err(ReservingError::InvalidParameter(
                        "GPD shape xi must be >= 0 for unbounded positive support",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Survival probability `S(x) = P(X > x)`.
    pub fn survival(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        match self {
            Self::Lognormal { mu, sigma } => {
                let z = (x.ln() - mu) / sigma;
                normal_cdf(-z)
            }
            Self::Pareto { alpha, theta } => (theta / (x + theta)).powf(*alpha),
            Self::SingleParameterPareto { alpha, theta } => {
                if x <= *theta {
                    1.0
                } else {
                    (theta / x).powf(*alpha)
                }
            }
            Self::Exponential { mean } => (-x / mean).exp(),
            Self::MixedExponential { weights, means } => weights
                .iter()
                .zip(means.iter())
                .map(|(&w, &m)| w * (-x / m).exp())
                .sum(),
            Self::GeneralizedPareto { xi, beta } => {
                if xi.abs() <= 1e-12 {
                    (-x / beta).exp()
                } else {
                    (-((xi * x / beta).ln_1p()) / xi).exp()
                }
            }
        }
    }

    /// Cumulative distribution function `F(x) = P(X <= x)`.
    pub fn cdf(&self, x: f64) -> f64 {
        1.0 - self.survival(x)
    }

    /// Probability density function `f(x)`.
    pub fn pdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x <= 0.0 || !x.is_finite() {
            return 0.0;
        }
        match self {
            Self::Lognormal { mu, sigma } => {
                let ln_x = x.ln();
                let z = (ln_x - mu) / sigma;
                (-0.5 * z * z - ln_x - (sigma * TAU.sqrt()).ln()).exp()
            }
            Self::Pareto { alpha, theta } => {
                // Avoid intermediate theta^alpha overflow on realistic dollar scales.
                (alpha / (x + theta)) * (theta / (x + theta)).powf(*alpha)
            }
            Self::SingleParameterPareto { alpha, theta } => {
                if x < *theta {
                    0.0
                } else {
                    (alpha / x) * (theta / x).powf(*alpha)
                }
            }
            Self::Exponential { mean } => (-x / mean).exp() / mean,
            Self::MixedExponential { weights, means } => weights
                .iter()
                .zip(means.iter())
                .map(|(&w, &m)| w * (-x / m).exp() / m)
                .sum(),
            Self::GeneralizedPareto { xi, beta } => {
                if xi.abs() <= 1e-12 {
                    (-x / beta).exp() / beta
                } else {
                    (-(1.0 / xi + 1.0) * (xi * x / beta).ln_1p()).exp() / beta
                }
            }
        }
    }

    /// Ground-up unlimited mean `E[X]`, or `None` when the first moment diverges
    /// (Pareto `alpha <= 1` or GPD `xi >= 1`).
    pub fn unbounded_mean(&self) -> Option<f64> {
        let mean = match self {
            Self::Lognormal { mu, sigma } => Some((mu + 0.5 * sigma * sigma).exp()),
            Self::Pareto { alpha, theta } => {
                if *alpha > 1.0 {
                    Some(theta / (alpha - 1.0))
                } else {
                    None
                }
            }
            Self::SingleParameterPareto { alpha, theta } => {
                if *alpha > 1.0 {
                    Some(alpha * theta / (alpha - 1.0))
                } else {
                    None
                }
            }
            Self::Exponential { mean } => Some(*mean),
            Self::MixedExponential { weights, means } => {
                Some(weights.iter().zip(means.iter()).map(|(&w, &m)| w * m).sum())
            }
            Self::GeneralizedPareto { xi, beta } => {
                if *xi < 1.0 {
                    Some(beta / (1.0 - xi))
                } else {
                    None
                }
            }
        };
        mean.filter(|m| m.is_finite())
    }

    /// Exact closed-form Limited Expected Value `E[X ∧ limit]`
    /// (Klugman, Panjer & Willmot, Appendix A; Lee 1988).
    pub fn limited_expected_value(&self, limit: f64) -> Result<f64, ReservingError> {
        self.validate()?;
        if !limit.is_finite() {
            return Err(ReservingError::NotFinite);
        }
        if limit < 0.0 {
            return Err(ReservingError::Negative);
        }
        if limit == 0.0 {
            return Ok(0.0);
        }

        let lev = match self {
            Self::Lognormal { mu, sigma } => lognormal_lev(*mu, *sigma, limit),
            Self::Pareto { alpha, theta } => {
                let ln_ratio = (limit / theta).ln_1p();
                if (alpha - 1.0).abs() <= 1e-10 {
                    theta * ln_ratio
                } else {
                    let am1 = alpha - 1.0;
                    theta * -((-am1 * ln_ratio).exp_m1()) / am1
                }
            }
            Self::SingleParameterPareto { alpha, theta } => {
                if limit <= *theta {
                    limit
                } else {
                    let ln_ratio = (limit / theta).ln();
                    if (alpha - 1.0).abs() <= 1e-10 {
                        theta * (1.0 + ln_ratio)
                    } else {
                        let am1 = alpha - 1.0;
                        theta + theta * -((-am1 * ln_ratio).exp_m1()) / am1
                    }
                }
            }
            Self::Exponential { mean } => mean * -((-limit / mean).exp_m1()),
            Self::MixedExponential { weights, means } => weights
                .iter()
                .zip(means.iter())
                .map(|(&w, &m)| w * m * -((-limit / m).exp_m1()))
                .sum(),
            Self::GeneralizedPareto { xi, beta } => {
                if xi.abs() <= 1e-10 {
                    beta * -((-limit / beta).exp_m1())
                } else {
                    let ln_u = (xi * limit / beta).ln_1p();
                    if (xi - 1.0).abs() <= 1e-10 {
                        beta * ln_u
                    } else {
                        let omx = 1.0 - xi;
                        beta * -(((-omx / xi) * ln_u).exp_m1()) / omx
                    }
                }
            }
        };
        Ok(lev)
    }

    /// Expected loss in layer `[attachment, attachment + limit]`:
    /// `E[min(max(X - attachment, 0), limit)] = ∫_a^{a+l} S(x) dx`.
    ///
    /// Evaluated with **direct closed forms** of the survival integral (never
    /// as the difference `LEV(a + l) − LEV(a)` of two nearly equal numbers), so
    /// high layers retain full relative precision:
    ///
    /// * Exponential / Mixed Exponential: `θ e^{-a/θ} (1 − e^{-l/θ})` via `exp_m1`.
    /// * Pareto (Lomax): `(a + θ) S(a) (1 − r^{1-α}) / (α − 1)`, `r = 1 + l / (a + θ)`,
    ///   via `ln_1p` / `exp_m1` (`α = 1`: `(a + θ) S(a) ln r`).
    /// * Single-parameter Pareto: the same form above `θ` plus the flat
    ///   `S = 1` segment below `θ`.
    /// * GPD: `(β + ξ a) S(a) (1 − r^{-(1-ξ)/ξ}) / (1 − ξ)`, `r = 1 + ξ l / (β + ξ a)`.
    /// * Lognormal: complementary-tail form
    ///   `E[(X − a)_+] − E[(X − a − l)_+]` with `E[(X − c)_+] = e^{μ+σ²/2} Q(z_c − σ) − c Q(z_c)`
    ///   evaluated in log space (no `e^{μ+σ²/2}` overflow); narrow layers
    ///   (`l <= 0.01 a`) use 8-point Gauss–Legendre on `S`.
    pub fn layer_expected_loss(&self, attachment: f64, limit: f64) -> Result<f64, ReservingError> {
        self.validate()?;
        if !attachment.is_finite() || !limit.is_finite() {
            return Err(ReservingError::NotFinite);
        }
        if attachment < 0.0 || limit < 0.0 {
            return Err(ReservingError::Negative);
        }
        if limit == 0.0 {
            return Ok(0.0);
        }
        let a = attachment;
        let l = limit;
        let layer = match self {
            Self::Lognormal { mu, sigma } => lognormal_layer(*mu, *sigma, a, l),
            Self::Pareto { alpha, theta } => lomax_layer(*alpha, *theta, a, l),
            Self::SingleParameterPareto { alpha, theta } => {
                let top = a + l;
                if top <= *theta {
                    l
                } else if a >= *theta {
                    spp_tail_layer(*alpha, *theta, a, l)
                } else {
                    (theta - a) + spp_tail_layer(*alpha, *theta, *theta, top - theta)
                }
            }
            Self::Exponential { mean } => exponential_layer(*mean, a, l),
            Self::MixedExponential { weights, means } => weights
                .iter()
                .zip(means.iter())
                .map(|(&w, &m)| w * exponential_layer(m, a, l))
                .sum(),
            Self::GeneralizedPareto { xi, beta } => gpd_layer(*xi, *beta, a, l),
        };
        if !layer.is_finite() {
            return Err(ReservingError::NotFinite);
        }
        Ok(layer.clamp(0.0, l))
    }

    /// Expected loss in the typed [`crate::excess::excess_layers::LayerSpec`] layer.
    pub fn layer_expected_loss_spec(
        &self,
        layer: crate::excess::excess_layers::LayerSpec,
    ) -> Result<f64, ReservingError> {
        self.layer_expected_loss(layer.attachment, layer.limit)
    }

    /// Expected excess loss `E[(X - retention)_+] = ∫_retention^∞ S(x) dx`,
    /// computed directly from the survival-integral closed form (no
    /// `E[X] − E[X ∧ retention]` cancellation).
    ///
    /// Returns [`ReservingError::InvalidParameter`] when the unlimited mean
    /// diverges (Pareto `α <= 1`, GPD `ξ >= 1`).
    pub fn excess_expected_value(&self, retention: f64) -> Result<f64, ReservingError> {
        self.validate()?;
        if !retention.is_finite() {
            return Err(ReservingError::NotFinite);
        }
        if retention < 0.0 {
            return Err(ReservingError::Negative);
        }
        let mean = self
            .unbounded_mean()
            .ok_or(ReservingError::InvalidParameter(
                "excess expected value requires a finite unbounded mean",
            ))?;
        let a = retention;
        let excess = match self {
            Self::Lognormal { mu, sigma } => lognormal_excess(*mu, *sigma, a),
            Self::Pareto { alpha, theta } => {
                (a + theta) * lomax_survival(*alpha, *theta, a) / (alpha - 1.0)
            }
            Self::SingleParameterPareto { alpha, theta } => {
                if a >= *theta {
                    a * (theta / a).powf(*alpha) / (alpha - 1.0)
                } else {
                    (theta - a) + theta / (alpha - 1.0)
                }
            }
            Self::Exponential { mean } => mean * (-a / mean).exp(),
            Self::MixedExponential { weights, means } => weights
                .iter()
                .zip(means.iter())
                .map(|(&w, &m)| w * m * (-a / m).exp())
                .sum(),
            Self::GeneralizedPareto { xi, beta } => {
                if xi.abs() <= GPD_EXPONENTIAL_XI {
                    beta * (-a / beta).exp()
                } else {
                    (beta + xi * a) * gpd_survival(*xi, *beta, a) / (1.0 - xi)
                }
            }
        };
        if !excess.is_finite() {
            return Err(ReservingError::NotFinite);
        }
        Ok(excess.clamp(0.0, mean))
    }

    /// Excess ratio `R(retention) = E[(X - retention)_+] / E[X]`, computed
    /// from [`Self::excess_expected_value`] (full relative precision for high
    /// retentions).
    pub fn excess_ratio(&self, retention: f64) -> Result<f64, ReservingError> {
        self.validate()?;
        let mean = self
            .unbounded_mean()
            .ok_or(ReservingError::InvalidParameter(
                "excess ratio requires a finite unbounded mean",
            ))?;
        let excess = self.excess_expected_value(retention)?;
        Ok((excess / mean).clamp(0.0, 1.0))
    }

    /// Layer loss share `(R(a) − R(a + l)) = E[layer] / E[X]`, computed from the
    /// direct layer closed form (no difference of excess ratios).
    pub fn layer_ratio(&self, attachment: f64, limit: f64) -> Result<f64, ReservingError> {
        self.validate()?;
        let mean = self
            .unbounded_mean()
            .ok_or(ReservingError::InvalidParameter(
                "layer ratio requires a finite unbounded mean",
            ))?;
        let layer = self.layer_expected_loss(attachment, limit)?;
        Ok((layer / mean).clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_paid_and_ultimate_severity() {
        let paid = paid_severity(&[100.0, 120.0], &[10.0, 12.0]).expect("paid severity");
        let ultimate =
            ultimate_severity(&[150.0, 180.0], &[10.0, 12.0]).expect("ultimate severity");

        assert_eq!(paid, vec![10.0, 10.0]);
        assert_eq!(ultimate, vec![15.0, 15.0]);
    }

    #[test]
    fn computes_frequency_severity_ultimate() {
        let loss =
            frequency_severity_ultimate(&[10.0, 12.0], &[15.0, 20.0]).expect("frequency severity");
        assert_eq!(loss, vec![150.0, 240.0]);
    }

    #[test]
    fn parametric_lev_matches_numerical_survival_integral() {
        let dists = [
            SeverityDistribution::lognormal_from_mean_cv(100_000.0, 1.5).unwrap(),
            SeverityDistribution::Pareto {
                alpha: 2.5,
                theta: 150_000.0,
            },
            SeverityDistribution::Pareto {
                alpha: 1.0,
                theta: 100_000.0,
            },
            SeverityDistribution::SingleParameterPareto {
                alpha: 2.0,
                theta: 50_000.0,
            },
            SeverityDistribution::Exponential { mean: 100_000.0 },
            SeverityDistribution::MixedExponential {
                weights: vec![0.7, 0.3],
                means: vec![40_000.0, 240_000.0],
            },
            SeverityDistribution::GeneralizedPareto {
                xi: 0.35,
                beta: 80_000.0,
            },
        ];

        // Lee (1988) identity: E[X ∧ L] = ∫_0^L S(x) dx.
        let limit = 250_000.0;
        let steps = 5_000usize;
        let dx = limit / steps as f64;
        for dist in &dists {
            let exact = dist.limited_expected_value(limit).expect("lev");
            let mut trap = 0.0;
            for i in 0..steps {
                let x_mid = (i as f64 + 0.5) * dx;
                trap += dist.survival(x_mid) * dx;
            }
            assert!(
                (exact - trap).abs() / exact < 5e-4,
                "dist={dist:?} exact={exact} numerical={trap}"
            );
        }
    }
}
