//! Benktander–Hovinen iterated Bornhuetter–Ferguson credibility reserving method.
//!
//! Implements:
//!
//! * Benktander (1976), *An Approach to Credibility in Calculating IBNR for
//!   Casualty Excess Reinsurance*, Actuarial Review / ASTIN.
//! * Hovinen (1981), *Additive and Continuous IBNR*, ASTIN Colloquium.
//! * Mack (2000), *Credible Claims Reserve: The Benktander Method*, ASTIN
//!   Bulletin, Vol. 30, No. 2, pp. 333–347.
//! * Friedland (2010), *Estimating Unpaid Claims Using Basic Techniques*, CAS,
//!   Chapter 10.
//!
//! ## Closed-Form Recursion
//!
//! Starting from an a priori ultimate `U_i^{(0)} = E_i` and maturity percentage
//! `p_i = 1 / CDF_{d_i}` (unreported fraction `q_i = 1 - p_i`), the `m`-step
//! Benktander ultimate is:
//!
//! ```text
//! U_i^{(m)} = C_{i, d_i} + q_i · U_i^{(m - 1)}
//!           = (1 - q_i^m) · U_i^{CL} + q_i^m · E_i
//! ```
//!
//! Special cases:
//! * `m = 0`: pure a priori Expected Claims (`Z_i = 0`).
//! * `m = 1`: Bornhuetter–Ferguson (`Z_i = p_i`).
//! * `m = 2`: classical Benktander (`Z_i = 1 - q_i^2 = p_i (2 - p_i)`).
//! * `m → ∞`: classical Chain Ladder (`Z_i → 1`).

use crate::apriori::{AprioriInput, build_apriori};
use crate::chain_ladder::latest_observed_by_origin;
use crate::ldf::DevelopmentFactors;
use crate::types::TriangleView;
use crate::validate::{
    MonotonicityPolicy, ValidationError, validate_development_factor_slices,
    validate_latest_and_dev, validate_monotone_cumulative, validate_nonnegative_finite_slice,
    validate_nonnegative_finite_vector, validate_triangle,
};

/// Benktander–Hovinen projection result.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BenktanderProjection {
    /// Number of iterations `m` applied (`m = 1` is BF, `m = 2` is Benktander).
    pub iterations: usize,
    /// Latest observed cumulative value by origin.
    pub latest: Vec<f64>,
    /// Latest observed development index by origin.
    pub latest_dev: Vec<usize>,
    /// Maturity percentage `p_i = 1 / cdf[latest_dev[i]]` by origin.
    pub maturity: Vec<f64>,
    /// Implicit credibility weight `Z_i = 1 - (1 - p_i)^m` placed on Chain Ladder.
    pub credibility_weights: Vec<f64>,
    /// A priori expected ultimate loss by origin.
    pub expected_ultimate: Vec<f64>,
    /// Benktander reserve by origin (`ultimates[i] - latest[i]`).
    pub reserves: Vec<f64>,
    /// Benktander ultimate by origin.
    pub ultimates: Vec<f64>,
    /// Total latest observed value.
    pub total_latest: f64,
    /// Total a priori expected ultimate.
    pub total_expected_ultimate: f64,
    /// Total Benktander reserve.
    pub total_reserve: f64,
    /// Total Benktander ultimate.
    pub total_ultimate: f64,
}

/// Mack (2000, Theorem 1) MSE-optimal credibility weight `Z_i^* = p_i / (p_i + t_i)`
/// for combining Chain Ladder and a priori ultimates, where `p_i` is maturity
/// (`1 / CDF`) and `t_i = Var(process) / Var(prior)` is the variance ratio.
///
/// Notice `t_i = 0` yields Chain Ladder (`Z = 1`), `t_i = 1 - p_i` yields BF
/// (`Z = p_i`), and `t_i = (1 - p_i)^2 / (2 - p_i)` yields Benktander.
#[inline]
pub fn mack_optimal_credibility(
    maturity: f64,
    variance_ratio_t: f64,
) -> Result<f64, ValidationError> {
    if !maturity.is_finite() || !variance_ratio_t.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if !(0.0..=1.0).contains(&maturity) {
        return Err(ValidationError::InvalidMaturity {
            index: 0,
            value: maturity,
        });
    }
    if variance_ratio_t < 0.0 {
        return Err(ValidationError::Negative);
    }
    let denom = maturity + variance_ratio_t;
    if denom <= 0.0 {
        Ok(0.0)
    } else {
        Ok(maturity / denom)
    }
}

/// Checked Benktander–Hovinen reserving service.
pub struct BenktanderEngine;

impl BenktanderEngine {
    /// Project Benktander ultimates from explicit expected ultimates with
    /// `iterations` steps (`iterations = 2` is standard Benktander).
    pub fn project_from_expected_ultimate(
        cumulative: &TriangleView<'_>,
        factors: &DevelopmentFactors,
        expected_ultimate: &[f64],
        iterations: usize,
    ) -> Result<BenktanderProjection, ValidationError> {
        Self::project_with_policy(
            cumulative,
            factors,
            expected_ultimate,
            iterations,
            MonotonicityPolicy::Strict,
        )
    }

    /// Project Benktander ultimates under an explicit [`MonotonicityPolicy`].
    pub fn project_with_policy(
        cumulative: &TriangleView<'_>,
        factors: &DevelopmentFactors,
        expected_ultimate: &[f64],
        iterations: usize,
        policy: MonotonicityPolicy,
    ) -> Result<BenktanderProjection, ValidationError> {
        validate_triangle(cumulative)?;
        if policy == MonotonicityPolicy::Strict {
            validate_monotone_cumulative(cumulative)?;
        }
        validate_development_factor_slices(
            &factors.ldfs,
            &factors.cdfs,
            factors.tail,
            cumulative.n_devs,
            "Benktander factor",
        )?;
        validate_nonnegative_finite_vector(expected_ultimate, cumulative.n_origins)?;

        let n = cumulative.n_origins;
        let mut latest = vec![0.0; n];
        let mut latest_dev = vec![0usize; n];
        latest_observed_by_origin(
            cumulative.data,
            cumulative.n_origins,
            cumulative.n_devs,
            &mut latest,
            &mut latest_dev,
        );
        validate_latest_and_dev(&latest, &latest_dev, &factors.cdfs)?;

        let mut maturity = vec![0.0; n];
        let mut credibility_weights = vec![0.0; n];
        let mut ultimates = vec![0.0; n];
        let mut reserves = vec![0.0; n];

        for i in 0..n {
            let cdf = factors.cdfs[latest_dev[i]];
            let p = 1.0 / cdf;
            let valid_p = match policy {
                MonotonicityPolicy::Strict => p.is_finite() && (0.0..=1.0).contains(&p),
                MonotonicityPolicy::AllowNegativeIncrements => {
                    p.is_finite() && (0.0..2.0).contains(&p)
                }
            };
            if !valid_p {
                return Err(ValidationError::InvalidMaturity { index: i, value: p });
            }
            let q = 1.0 - p;
            let q_pow = q.powi(iterations.min(i32::MAX as usize) as i32);
            let z = 1.0 - q_pow;
            let cl_ult = latest[i] * cdf;
            let ult = z * cl_ult + q_pow * expected_ultimate[i];
            let res = ult - latest[i];

            maturity[i] = p;
            credibility_weights[i] = z;
            ultimates[i] = ult;
            reserves[i] = res;
        }

        validate_nonnegative_finite_slice(&ultimates)?;
        let total_latest: f64 = latest.iter().sum();
        let total_expected_ultimate: f64 = expected_ultimate.iter().sum();
        let total_reserve: f64 = reserves.iter().sum();
        let total_ultimate: f64 = ultimates.iter().sum();

        if !total_latest.is_finite()
            || !total_expected_ultimate.is_finite()
            || !total_reserve.is_finite()
            || !total_ultimate.is_finite()
        {
            return Err(ValidationError::NotFinite);
        }

        Ok(BenktanderProjection {
            iterations,
            latest,
            latest_dev,
            maturity,
            credibility_weights,
            expected_ultimate: expected_ultimate.to_vec(),
            reserves,
            ultimates,
            total_latest,
            total_expected_ultimate,
            total_reserve,
            total_ultimate,
        })
    }

    /// Project Benktander ultimates using a shared [`AprioriInput`] builder.
    pub fn project_from_apriori(
        cumulative: &TriangleView<'_>,
        factors: &DevelopmentFactors,
        premium_or_exposure: &[f64],
        apriori: AprioriInput<'_>,
        iterations: usize,
    ) -> Result<BenktanderProjection, ValidationError> {
        let apriori_vec = build_apriori(premium_or_exposure, apriori)?;
        Self::project_from_expected_ultimate(
            cumulative,
            factors,
            &apriori_vec.expected_ultimate,
            iterations,
        )
    }

    /// Project Benktander ultimates from earned premium and a scalar ELR.
    pub fn project_from_scalar_elr(
        cumulative: &TriangleView<'_>,
        factors: &DevelopmentFactors,
        earned_premium: &[f64],
        expected_loss_ratio: f64,
        iterations: usize,
    ) -> Result<BenktanderProjection, ValidationError> {
        Self::project_from_apriori(
            cumulative,
            factors,
            earned_premium,
            AprioriInput::ScalarElr(expected_loss_ratio),
            iterations,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf::BornhuetterFergusonEngine;
    use crate::chain_ladder::ChainLadderEngine;
    use crate::ldf::DevelopmentFactorEngine;
    use crate::types::Triangle;

    fn sample_triangle() -> Triangle {
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
    fn benktander_interpolates_between_bf_and_chain_ladder() {
        let tri = sample_triangle();
        let factors = DevelopmentFactorEngine::volume_weighted(&tri.view(), 1.0).expect("factors");
        let expected = [2000.0, 2200.0, 2600.0, 3200.0];

        let bf = BornhuetterFergusonEngine::project_from_expected_ultimate(
            &tri.view(),
            &factors,
            &expected,
        )
        .expect("bf");
        let cl = ChainLadderEngine::project(&tri.view(), &factors).expect("cl");
        let bk1 =
            BenktanderEngine::project_from_expected_ultimate(&tri.view(), &factors, &expected, 1)
                .expect("bk1");
        let bk2 =
            BenktanderEngine::project_from_expected_ultimate(&tri.view(), &factors, &expected, 2)
                .expect("bk2");
        let bk_inf =
            BenktanderEngine::project_from_expected_ultimate(&tri.view(), &factors, &expected, 200)
                .expect("bk200");

        for i in 0..4 {
            assert!((bk1.ultimates[i] - bf.ultimates[i]).abs() < 1e-10);
            assert!((bk_inf.ultimates[i] - cl.ultimates[i]).abs() < 1e-9);
            // m = 2 recursion check: U^{(2)} = latest + (1 - p) * U^{BF}.
            let manual = bk2.latest[i] + (1.0 - bk2.maturity[i]) * bf.ultimates[i];
            assert!((bk2.ultimates[i] - manual).abs() < 1e-10);
        }
    }

    #[test]
    fn mack_optimal_credibility_matches_special_cases() {
        let p = 0.4;
        assert!((mack_optimal_credibility(p, 0.0).unwrap() - 1.0).abs() < 1e-12);
        assert!((mack_optimal_credibility(p, 1.0 - p).unwrap() - p).abs() < 1e-12);
        let t_bk = (1.0 - p).powi(2) / (2.0 - p);
        assert!((mack_optimal_credibility(p, t_bk).unwrap() - p * (2.0 - p)).abs() < 1e-12);
    }
}
