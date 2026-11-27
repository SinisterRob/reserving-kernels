//! Severity kernels: empirical claim-count ratios and parametric severity
//! distributions for Limited Expected Values (LEV), excess ratios, and
//! Increased Limit Factors (ILFs).
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

use crate::validate::{ValidationError, validate_nonnegative_finite_vector};

fn divide_vectors(numerator: &[f64], denominator: &[f64]) -> Result<Vec<f64>, ValidationError> {
    let n = numerator.len();
    validate_nonnegative_finite_vector(numerator, n)?;
    validate_nonnegative_finite_vector(denominator, denominator.len())?;
    if denominator.len() != n {
        return Err(ValidationError::ShapeMismatch);
    }

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if denominator[i] <= 0.0 {
            return Err(ValidationError::DivideByZero("count"));
        }
        out.push(numerator[i] / denominator[i]);
    }
    Ok(out)
}

/// Paid severity by origin, `paid / counts`.
pub fn paid_severity(paid: &[f64], counts: &[f64]) -> Result<Vec<f64>, ValidationError> {
    divide_vectors(paid, counts)
}

/// Incurred severity by origin, `incurred / counts`.
pub fn incurred_severity(incurred: &[f64], counts: &[f64]) -> Result<Vec<f64>, ValidationError> {
    divide_vectors(incurred, counts)
}

/// Ultimate severity by origin, `ultimate_loss / ultimate_count`.
pub fn ultimate_severity(
    ultimate_loss: &[f64],
    ultimate_count: &[f64],
) -> Result<Vec<f64>, ValidationError> {
    divide_vectors(ultimate_loss, ultimate_count)
}

/// Open-claim severity by origin, `case_reserve / open_claims`.
pub fn open_claim_severity(
    case_reserve: &[f64],
    open_claims: &[f64],
) -> Result<Vec<f64>, ValidationError> {
    divide_vectors(case_reserve, open_claims)
}

/// Closed-claim severity by origin, `paid / closed_claims`.
pub fn closed_claim_severity(
    paid: &[f64],
    closed_claims: &[f64],
) -> Result<Vec<f64>, ValidationError> {
    divide_vectors(paid, closed_claims)
}

/// Frequency-severity ultimate by origin, `ultimate_count * severity`.
pub fn frequency_severity_ultimate(
    ultimate_count: &[f64],
    severity: &[f64],
) -> Result<Vec<f64>, ValidationError> {
    let n = ultimate_count.len();
    validate_nonnegative_finite_vector(ultimate_count, n)?;
    validate_nonnegative_finite_vector(severity, severity.len())?;
    if severity.len() != n {
        return Err(ValidationError::ShapeMismatch);
    }
    Ok(ultimate_count
        .iter()
        .zip(severity.iter())
        .map(|(&count, &sev)| count * sev)
        .collect())
}

// ===========================================================================
// Parametric Severity Distributions, Limited Expected Values & ILFs
// ===========================================================================

/// High-accuracy standard normal CDF `Φ(z)` via Cody (1969) rational `erfc`
/// approximation (max absolute error `< 1e-15`).
pub(crate) fn normal_cdf_exact(z: f64) -> f64 {
    0.5 * erfc_approx(-z / std::f64::consts::SQRT_2)
}

/// Complementary error function `erfc(x)` accurate to ~1e-15 (Cody 1969).
fn erfc_approx(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return 2.0 - erfc_approx(-x);
    }
    if x < 0.46875 {
        const A: [f64; 5] = [
            3.161_123_743_870_565_6e0,
            1.138_641_541_510_501_6e2,
            3.774_852_376_853_02e2,
            3.209_377_589_138_469_5e3,
            1.857_777_061_846_031_5e-1,
        ];
        const B: [f64; 4] = [
            2.360_129_095_234_412_1e1,
            2.440_246_379_344_441_7e2,
            1.282_616_526_077_372_3e3,
            2.844_236_833_439_171e3,
        ];
        let y = x * x;
        let num = (((A[4] * y + A[0]) * y + A[1]) * y + A[2]) * y + A[3];
        let den = (((y + B[0]) * y + B[1]) * y + B[2]) * y + B[3];
        1.0 - x * num / den
    } else if x <= 4.0 {
        const C: [f64; 9] = [
            5.641_884_969_886_701e-1,
            8.883_149_794_388_375,
            6.611_919_063_714_163e1,
            2.986_351_381_974_001e2,
            8.819_522_212_417_69e2,
            1.712_047_612_634_070_6e3,
            2.051_078_377_826_071_5e3,
            1.230_339_354_797_997_2e3,
            2.153_115_354_744_038_5e-8,
        ];
        const D: [f64; 8] = [
            1.574_492_611_070_983_5e1,
            1.176_939_508_913_125e2,
            5.371_811_018_620_099e2,
            1.621_389_574_566_690_2e3,
            3.290_799_235_733_459_6e3,
            4.362_619_090_143_247e3,
            3.439_367_674_143_721_6e3,
            1.230_339_354_803_749_4e3,
        ];
        let num = (((((((C[8] * x + C[0]) * x + C[1]) * x + C[2]) * x + C[3]) * x + C[4]) * x
            + C[5])
            * x
            + C[6])
            * x
            + C[7];
        let den = (((((((x + D[0]) * x + D[1]) * x + D[2]) * x + D[3]) * x + D[4]) * x + D[5]) * x
            + D[6])
            * x
            + D[7];
        (-x * x).exp() * num / den
    } else if x < 26.5 {
        const P: [f64; 6] = [
            3.053_266_349_612_323_4e-1,
            3.603_448_999_498_044_4e-1,
            1.257_817_261_112_292_5e-1,
            1.608_378_514_874_228e-2,
            6.587_491_615_298_378e-4,
            1.631_538_713_730_209_8e-2,
        ];
        const Q: [f64; 5] = [
            2.568_520_192_289_822,
            1.872_952_849_923_460_5e0,
            5.279_051_029_514_285e-1,
            6.051_834_131_244_132e-2,
            2.335_204_976_268_691_9e-3,
        ];
        let y = 1.0 / (x * x);
        let num = ((((P[5] * y + P[0]) * y + P[1]) * y + P[2]) * y + P[3]) * y + P[4];
        let den = ((((y + Q[0]) * y + Q[1]) * y + Q[2]) * y + Q[3]) * y + Q[4];
        let inv_sqrt_pi = std::f64::consts::FRAC_2_SQRT_PI * 0.5;
        (-x * x).exp() / x * (inv_sqrt_pi - y * num / den)
    } else {
        0.0
    }
}

/// Parametric P&C claim severity distributions with exact closed-form Limited
/// Expected Values `E[X ∧ L]`, survival functions, excess ratios, and ILFs.
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
    ) -> Result<Vec<f64>, ValidationError> {
        crate::aggregate_loss::discretize_rounding(self, step, max_index, capture_upper_tail)
    }

    /// Discretize onto `0, step, ..., max_index * step` via mean-preserving LEV local moment matching.
    pub fn discretize_mean_preserving(
        &self,
        step: f64,
        max_index: usize,
    ) -> Result<Vec<f64>, ValidationError> {
        crate::aggregate_loss::discretize_mean_preserving(self, step, max_index)
    }

    /// Construct a Lognormal distribution from ground-up mean and coefficient
    /// of variation `cv = sd / mean > 0`.
    pub fn lognormal_from_mean_cv(mean: f64, cv: f64) -> Result<Self, ValidationError> {
        if !mean.is_finite() || !cv.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if mean <= 0.0 || cv <= 0.0 {
            return Err(ValidationError::InvalidParameter(
                "lognormal mean and CV must be strictly positive",
            ));
        }
        let sigma = (1.0 + cv * cv).ln().sqrt();
        let mu = mean.ln() - 0.5 * sigma * sigma;
        Ok(Self::Lognormal { mu, sigma })
    }

    /// Validate distribution parameters.
    pub fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::Lognormal { mu, sigma } => {
                if !mu.is_finite() || !sigma.is_finite() {
                    return Err(ValidationError::NotFinite);
                }
                if *sigma <= 0.0 {
                    return Err(ValidationError::InvalidParameter(
                        "lognormal sigma must be > 0",
                    ));
                }
            }
            Self::Pareto { alpha, theta } | Self::SingleParameterPareto { alpha, theta } => {
                if !alpha.is_finite() || !theta.is_finite() {
                    return Err(ValidationError::NotFinite);
                }
                if *alpha <= 0.0 || *theta <= 0.0 {
                    return Err(ValidationError::InvalidParameter(
                        "Pareto alpha and theta must be > 0",
                    ));
                }
            }
            Self::Exponential { mean } => {
                if !mean.is_finite() {
                    return Err(ValidationError::NotFinite);
                }
                if *mean <= 0.0 {
                    return Err(ValidationError::InvalidParameter(
                        "exponential mean must be > 0",
                    ));
                }
            }
            Self::MixedExponential { weights, means } => {
                if weights.is_empty() || weights.len() != means.len() {
                    return Err(ValidationError::ShapeMismatch);
                }
                let mut sum_w = 0.0;
                for (&w, &m) in weights.iter().zip(means.iter()) {
                    if !w.is_finite() || !m.is_finite() {
                        return Err(ValidationError::NotFinite);
                    }
                    if w <= 0.0 || m <= 0.0 {
                        return Err(ValidationError::InvalidParameter(
                            "mixed exponential weights and means must be > 0",
                        ));
                    }
                    sum_w += w;
                }
                if (sum_w - 1.0).abs() > 1e-6 {
                    return Err(ValidationError::InvalidParameter(
                        "mixed exponential weights must sum to 1.0",
                    ));
                }
            }
            Self::GeneralizedPareto { xi, beta } => {
                if !xi.is_finite() || !beta.is_finite() {
                    return Err(ValidationError::NotFinite);
                }
                if *beta <= 0.0 {
                    return Err(ValidationError::InvalidParameter(
                        "GPD scale beta must be > 0",
                    ));
                }
                if *xi < 0.0 {
                    return Err(ValidationError::InvalidParameter(
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
                normal_cdf_exact(-z)
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
        if x <= 0.0 {
            return 0.0;
        }
        match self {
            Self::Lognormal { mu, sigma } => {
                let z = (x.ln() - mu) / sigma;
                (-0.5 * z * z).exp() / (x * sigma * TAU.sqrt())
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
        match self {
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
        }
    }

    /// Exact closed-form Limited Expected Value `E[X ∧ limit]`
    /// (Klugman, Panjer & Willmot, Appendix A; Lee 1988).
    pub fn limited_expected_value(&self, limit: f64) -> Result<f64, ValidationError> {
        self.validate()?;
        if !limit.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if limit < 0.0 {
            return Err(ValidationError::Negative);
        }
        if limit == 0.0 {
            return Ok(0.0);
        }

        let lev = match self {
            Self::Lognormal { mu, sigma } => {
                let z = (limit.ln() - mu) / sigma;
                let cdf_shifted = normal_cdf_exact(z - sigma);
                let term1 = if cdf_shifted == 0.0 {
                    0.0
                } else {
                    (mu + 0.5 * sigma * sigma).exp() * cdf_shifted
                };
                term1 + limit * normal_cdf_exact(-z)
            }
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
    /// `E[min(max(X - attachment, 0), limit)] = E[X ∧ (a + w)] - E[X ∧ a]`.
    pub fn layer_expected_loss(&self, attachment: f64, limit: f64) -> Result<f64, ValidationError> {
        self.validate()?;
        if !attachment.is_finite() || !limit.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if attachment < 0.0 || limit < 0.0 {
            return Err(ValidationError::Negative);
        }
        if limit == 0.0 {
            return Ok(0.0);
        }
        let upper = self.limited_expected_value(attachment + limit)?;
        let lower = self.limited_expected_value(attachment)?;
        Ok((upper - lower).max(0.0))
    }

    /// Excess ratio `R(retention) = E[(X - retention)_+] / E[X] = 1 - E[X ∧ retention] / E[X]`.
    pub fn excess_ratio(&self, retention: f64) -> Result<f64, ValidationError> {
        self.validate()?;
        let mean = self
            .unbounded_mean()
            .ok_or(ValidationError::InvalidParameter(
                "excess ratio requires a finite unbounded mean",
            ))?;
        let lev = self.limited_expected_value(retention)?;
        Ok(((mean - lev) / mean).clamp(0.0, 1.0))
    }

    /// Pure-loss Increased Limit Factor `ILF(L; B) = E[X ∧ L] / E[X ∧ B]`
    /// (Miccolis 1977; Lee 1988).
    pub fn ilf(&self, basic_limit: f64, increased_limit: f64) -> Result<f64, ValidationError> {
        if basic_limit <= 0.0 || increased_limit < 0.0 {
            return Err(ValidationError::InvalidParameter(
                "basic_limit must be > 0 and increased_limit >= 0",
            ));
        }
        let lev_b = self.limited_expected_value(basic_limit)?;
        if !(lev_b > 0.0) {
            return Err(ValidationError::DivideByZero("basic limit LEV"));
        }
        let lev_l = self.limited_expected_value(increased_limit)?;
        Ok(lev_l / lev_b)
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
            assert!(dist.ilf(100_000.0, 250_000.0).unwrap() >= 1.0);
        }
    }
}
