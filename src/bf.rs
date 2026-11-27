//! Bornhuetter–Ferguson reserving method.
//!
//! Given selected LDFs and an *a priori* expected ultimate loss per origin
//! (usually `premium × ELR` from pricing), BF produces:
//!
//! `U_i^{BF} = C_{i, n−1−i} + (1 − r_{n−1−i}) × E_i`,
//!
//! where `r_j = 1 / cdf_j` is the proportion of ultimate losses expected to
//! be reported by development age `j`, and `E_i` is the expected ultimate
//! for origin `i`. The reserve is `(1 − r_{n−1−i}) × E_i`.
//!
//! BF is a credibility-weighted blend between chain-ladder (full weight on
//! actual development) and a priori expectation (full weight on the budget).
//! It is the standard antidote to chain-ladder instability at the youngest
//! origin years.
//!
//! Three layers, all in this file:
//!
//! * **L0 columnar kernels** — flat slices and caller-provided output
//!   buffers ([`expected_ultimate_from_elr`], [`bf_reserves`],
//!   [`bf_ultimates`]).
//! * **L1 view convenience** — [`project_unchecked`] takes a [`TriangleView`],
//!   raw `ldfs` slice, `tail`, and `expected_ultimate`, returning an owned
//!   [`BornhuetterFergusonProjection`] without validation.
//! * **L2 service / engine** — [`BornhuetterFergusonEngine`] performs the
//!   same projection from a validated [`DevelopmentFactors`].

use crate::apriori::{AprioriInput, build_apriori};
use crate::chain_ladder::latest_observed_by_origin;
use crate::ldf::{DevelopmentFactors, cdf_from_ldf, pct_reported};
use crate::triangle::latest_diagonal;
use crate::types::TriangleView;
use crate::validate::{
    MonotonicityPolicy, ValidationError, validate_development_factor_slices,
    validate_latest_and_dev, validate_monotone_cumulative, validate_nonnegative_finite_slice,
    validate_nonnegative_finite_vector, validate_triangle,
};

// ===========================================================================
// Types
// ===========================================================================

/// Bornhuetter-Ferguson projection result.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct BornhuetterFergusonProjection {
    /// Latest observed cumulative value by origin.
    pub latest: Vec<f64>,
    /// Latest observed development index by origin.
    pub latest_dev: Vec<usize>,
    /// Maturity percentage at latest observed development by origin.
    pub maturity: Vec<f64>,
    /// Expected ultimate loss by origin.
    pub expected_ultimate: Vec<f64>,
    /// BF reserve by origin.
    pub reserves: Vec<f64>,
    /// BF ultimate by origin.
    pub ultimates: Vec<f64>,
    /// Total latest observed value.
    pub total_latest: f64,
    /// Total expected ultimate loss.
    pub total_expected_ultimate: f64,
    /// Total BF reserve.
    pub total_reserve: f64,
    /// Total BF ultimate.
    pub total_ultimate: f64,
}

// ===========================================================================
// L0 — columnar kernels
// ===========================================================================

/// Build expected ultimate losses from earned premium and expected loss ratio.
pub fn expected_ultimate_from_elr(
    earned_premium: &[f64],
    expected_loss_ratio: &[f64],
    out: &mut [f64],
) {
    debug_assert_eq!(earned_premium.len(), expected_loss_ratio.len());
    debug_assert_eq!(earned_premium.len(), out.len());

    for i in 0..earned_premium.len() {
        out[i] = earned_premium[i] * expected_loss_ratio[i];
    }
}

/// BF reserves: `expected_ultimate[i] * (1 - maturity[i])`.
pub fn bf_reserves(expected_ultimate: &[f64], maturity: &[f64], out: &mut [f64]) {
    debug_assert_eq!(expected_ultimate.len(), maturity.len());
    debug_assert_eq!(expected_ultimate.len(), out.len());

    for i in 0..expected_ultimate.len() {
        out[i] = expected_ultimate[i] * (1.0 - maturity[i]);
    }
}

/// BF ultimates: `latest[i] + reserves[i]`.
pub fn bf_ultimates(latest: &[f64], reserves: &[f64], out: &mut [f64]) {
    debug_assert_eq!(latest.len(), reserves.len());
    debug_assert_eq!(latest.len(), out.len());

    for i in 0..latest.len() {
        out[i] = latest[i] + reserves[i];
    }
}

// ===========================================================================
// L1 — unchecked view-based projection
// ===========================================================================

/// Build the expected-ultimate vector from `premium × elr`.
pub fn expected_from_premium(premium: &[f64], elr: f64) -> Vec<f64> {
    premium.iter().map(|&p| p * elr).collect()
}

/// Project ultimates using Bornhuetter–Ferguson with an explicit per-origin
/// expected-ultimate vector (e.g. `premium × ELR`).
///
/// Performs no validation. Prefer [`BornhuetterFergusonEngine::project_from_expected_ultimate`]
/// at external ingress boundaries.
pub fn project_unchecked(
    t: &TriangleView<'_>,
    ldfs: &[f64],
    tail: f64,
    expected_ultimate: &[f64],
) -> BornhuetterFergusonProjection {
    let n = t.n_origins.min(t.n_devs);
    debug_assert_eq!(expected_ultimate.len(), n);
    let cdfs = cdf_from_ldf(ldfs, tail);
    let pct = pct_reported(&cdfs);
    let latest = latest_diagonal(t);

    let mut latest_dev = vec![0_usize; n];
    let mut maturity = vec![0.0; n];
    let mut ultimates = vec![0.0; n];
    let mut reserve_values = vec![0.0; n];
    for i in 0..n {
        let age = n - 1 - i;
        latest_dev[i] = age;
        maturity[i] = pct[age];
        let unreported = 1.0 - pct[age];
        let r = unreported * expected_ultimate[i];
        reserve_values[i] = r;
        ultimates[i] = latest[i] + r;
    }

    let total_latest: f64 = latest.iter().sum();
    let total_expected_ultimate: f64 = expected_ultimate.iter().sum();
    let total_reserve: f64 = reserve_values.iter().sum();
    let total_ultimate: f64 = ultimates.iter().sum();

    BornhuetterFergusonProjection {
        latest,
        latest_dev,
        maturity,
        expected_ultimate: expected_ultimate.to_vec(),
        reserves: reserve_values,
        ultimates,
        total_latest,
        total_expected_ultimate,
        total_reserve,
        total_ultimate,
    }
}

// ===========================================================================
// L2 — validated engine
// ===========================================================================

/// Section 4 Bornhuetter-Ferguson service.
///
/// # Example
/// ```rust
/// use reserving_kernels::prelude::*;
///
/// let tri = Triangle::try_from_upper_rows(&[
///     &[100.0, 160.0, 195.0],
///     &[120.0, 190.0],
///     &[135.0],
/// ])?;
/// let factors = DevelopmentFactorEngine::volume_weighted(&tri.view(), 1.0)?;
/// let bf = BornhuetterFergusonEngine::project_from_expected_ultimate(
///     &tri.view(),
///     &factors,
///     &[200.0, 235.0, 260.0],
/// )?;
/// assert!(bf.total_reserve > 0.0);
/// # Ok::<(), ValidationError>(())
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct BornhuetterFergusonEngine;

impl BornhuetterFergusonEngine {
    /// Project BF ultimates and reserves from a shared a priori input builder.
    pub fn project_from_apriori(
        cumulative: &TriangleView<'_>,
        factors: &DevelopmentFactors,
        premium_or_exposure: &[f64],
        apriori: AprioriInput<'_>,
    ) -> Result<BornhuetterFergusonProjection, ValidationError> {
        let apriori_vector = build_apriori(premium_or_exposure, apriori)?;
        Self::project_from_expected_ultimate(cumulative, factors, &apriori_vector.expected_ultimate)
    }

    /// Project BF ultimates and reserves from earned premium and a scalar ELR.
    pub fn project_from_scalar_elr(
        cumulative: &TriangleView<'_>,
        factors: &DevelopmentFactors,
        earned_premium: &[f64],
        expected_loss_ratio: f64,
    ) -> Result<BornhuetterFergusonProjection, ValidationError> {
        validate_nonnegative_finite_vector(earned_premium, cumulative.n_origins)?;
        if !expected_loss_ratio.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if expected_loss_ratio < 0.0 {
            return Err(ValidationError::Negative);
        }

        let expected_ultimate = expected_from_premium(earned_premium, expected_loss_ratio);
        validate_nonnegative_finite_slice(&expected_ultimate)?;

        Self::project_from_expected_ultimate(cumulative, factors, &expected_ultimate)
    }

    /// Project BF ultimates and reserves from explicit expected ultimate losses.
    pub fn project_from_expected_ultimate(
        cumulative: &TriangleView<'_>,
        factors: &DevelopmentFactors,
        expected_ultimate: &[f64],
    ) -> Result<BornhuetterFergusonProjection, ValidationError> {
        Self::project_from_expected_ultimate_with_policy(
            cumulative,
            factors,
            expected_ultimate,
            MonotonicityPolicy::Strict,
        )
    }

    /// Project BF ultimates and reserves from explicit expected ultimate losses
    /// under an explicit [`MonotonicityPolicy`].
    pub fn project_from_expected_ultimate_with_policy(
        cumulative: &TriangleView<'_>,
        factors: &DevelopmentFactors,
        expected_ultimate: &[f64],
        policy: MonotonicityPolicy,
    ) -> Result<BornhuetterFergusonProjection, ValidationError> {
        validate_bf_inputs_with_policy(cumulative, factors, policy)?;
        validate_nonnegative_finite_vector(expected_ultimate, cumulative.n_origins)?;

        let latest_maturity = latest_and_maturity(cumulative, factors)?;
        validate_maturity_with_policy(&latest_maturity.maturity, policy)?;

        let mut reserve_values = vec![0.0; cumulative.n_origins];
        bf_reserves(
            expected_ultimate,
            &latest_maturity.maturity,
            &mut reserve_values,
        );
        if policy == MonotonicityPolicy::Strict {
            validate_nonnegative_finite_slice(&reserve_values)?;
        } else {
            crate::validate::validate_finite_slice(&reserve_values)?;
        }

        let mut ultimates = vec![0.0; cumulative.n_origins];
        bf_ultimates(&latest_maturity.latest, &reserve_values, &mut ultimates);
        if policy == MonotonicityPolicy::Strict {
            validate_nonnegative_finite_slice(&ultimates)?;
        } else {
            crate::validate::validate_finite_slice(&ultimates)?;
        }

        let total_latest: f64 = latest_maturity.latest.iter().sum();
        let total_expected_ultimate: f64 = expected_ultimate.iter().sum();
        let total_reserve: f64 = reserve_values.iter().sum();
        let total_ultimate: f64 = ultimates.iter().sum();

        if !total_latest.is_finite()
            || !total_expected_ultimate.is_finite()
            || !total_reserve.is_finite()
            || !total_ultimate.is_finite()
        {
            return Err(ValidationError::NotFinite);
        }

        Ok(BornhuetterFergusonProjection {
            latest: latest_maturity.latest,
            latest_dev: latest_maturity.latest_dev,
            maturity: latest_maturity.maturity,
            expected_ultimate: expected_ultimate.to_vec(),
            reserves: reserve_values,
            ultimates,
            total_latest,
            total_expected_ultimate,
            total_reserve,
            total_ultimate,
        })
    }

    /// Project BF ultimates and reserves from earned premium and ELR vectors.
    pub fn project_from_elr(
        cumulative: &TriangleView<'_>,
        factors: &DevelopmentFactors,
        earned_premium: &[f64],
        expected_loss_ratio: &[f64],
    ) -> Result<BornhuetterFergusonProjection, ValidationError> {
        validate_nonnegative_finite_vector(earned_premium, cumulative.n_origins)?;
        validate_nonnegative_finite_vector(expected_loss_ratio, cumulative.n_origins)?;

        let mut expected_ultimate = vec![0.0; cumulative.n_origins];
        expected_ultimate_from_elr(earned_premium, expected_loss_ratio, &mut expected_ultimate);
        validate_nonnegative_finite_slice(&expected_ultimate)?;

        Self::project_from_expected_ultimate(cumulative, factors, &expected_ultimate)
    }
}

struct LatestMaturity {
    latest: Vec<f64>,
    latest_dev: Vec<usize>,
    maturity: Vec<f64>,
}

fn latest_and_maturity(
    cumulative: &TriangleView<'_>,
    factors: &DevelopmentFactors,
) -> Result<LatestMaturity, ValidationError> {
    let n = cumulative.n_origins;
    let mut latest = vec![0.0; n];
    let mut latest_dev = vec![0; n];
    latest_observed_by_origin(
        cumulative.data,
        cumulative.n_origins,
        cumulative.n_devs,
        &mut latest,
        &mut latest_dev,
    );
    validate_latest_and_dev(&latest, &latest_dev, &factors.cdfs)?;

    let mut maturity = vec![0.0; n];
    for i in 0..n {
        let cdf = factors.cdfs[latest_dev[i]];
        maturity[i] = if cdf > 0.0 { 1.0 / cdf } else { f64::NAN };
    }

    Ok(LatestMaturity {
        latest,
        latest_dev,
        maturity,
    })
}

fn validate_bf_inputs_with_policy(
    cumulative: &TriangleView<'_>,
    factors: &DevelopmentFactors,
    policy: MonotonicityPolicy,
) -> Result<(), ValidationError> {
    validate_triangle(cumulative)?;
    if policy == MonotonicityPolicy::Strict {
        validate_monotone_cumulative(cumulative)?;
    }
    validate_development_factor_slices(
        &factors.ldfs,
        &factors.cdfs,
        factors.tail,
        cumulative.n_devs,
        "BF factor",
    )
}

fn validate_maturity_with_policy(
    maturity: &[f64],
    policy: MonotonicityPolicy,
) -> Result<(), ValidationError> {
    for (index, &value) in maturity.iter().enumerate() {
        if !value.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        let valid = match policy {
            MonotonicityPolicy::Strict => (0.0..=1.0).contains(&value),
            MonotonicityPolicy::AllowNegativeIncrements => value >= 0.0,
        };
        if !valid {
            return Err(ValidationError::InvalidMaturity { index, value });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ldf::DevelopmentFactors;
    use crate::types::Triangle;

    fn tiny_triangle() -> Triangle {
        #[rustfmt::skip]
        let data = vec![
            100.0, 150.0, 180.0,
            120.0, 180.0, f64::NAN,
            140.0, f64::NAN, f64::NAN,
        ];
        Triangle::new(3, 3, data)
    }

    fn factors() -> DevelopmentFactors {
        DevelopmentFactors {
            ldfs: vec![1.5, 1.2],
            cdfs: vec![1.8, 1.2, 1.0],
            tail: 1.0,
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * (1.0 + b.abs())
    }

    #[test]
    fn expected_ultimate_from_elr_kernel_multiplies_vectors() {
        let premium = [100.0, 200.0, 300.0];
        let elr = [0.6, 0.7, 0.8];
        let mut out = [0.0; 3];

        expected_ultimate_from_elr(&premium, &elr, &mut out);

        assert_eq!(out, [60.0, 140.0, 240.0]);
    }

    #[test]
    fn bf_reserves_kernel_uses_unreported_percentage() {
        let expected = [200.0, 240.0, 300.0];
        let maturity = [1.0, 1.0 / 1.2, 1.0 / 1.8];
        let mut out = [0.0; 3];

        bf_reserves(&expected, &maturity, &mut out);

        assert!(close(out[0], 0.0));
        assert!(close(out[1], 40.0));
        assert!(close(out[2], 133.33333333333334));
    }

    #[test]
    fn bf_ultimates_kernel_adds_latest_and_reserve() {
        let latest = [180.0, 180.0, 140.0];
        let reserves = [0.0, 40.0, 133.33333333333334];
        let mut out = [0.0; 3];

        bf_ultimates(&latest, &reserves, &mut out);

        assert!(close(out[0], 180.0));
        assert!(close(out[1], 220.0));
        assert!(close(out[2], 273.33333333333337));
    }

    #[test]
    fn project_unchecked_matches_engine_on_valid_input() {
        let tri = tiny_triangle();
        let expected = [200.0, 240.0, 300.0];

        let unchecked = project_unchecked(&tri.view(), &factors().ldfs, factors().tail, &expected);
        let engine = BornhuetterFergusonEngine::project_from_expected_ultimate(
            &tri.view(),
            &factors(),
            &expected,
        )
        .expect("engine");

        assert_eq!(unchecked, engine);
    }

    #[test]
    fn bf_projection_known_example() {
        let tri = tiny_triangle();
        let expected = [200.0, 240.0, 300.0];

        let projected = BornhuetterFergusonEngine::project_from_expected_ultimate(
            &tri.view(),
            &factors(),
            &expected,
        )
        .expect("projection");

        assert_eq!(projected.latest, vec![180.0, 180.0, 140.0]);
        assert_eq!(projected.latest_dev, vec![2, 1, 0]);
        assert!(close(projected.maturity[0], 1.0));
        assert!(close(projected.maturity[1], 1.0 / 1.2));
        assert!(close(projected.maturity[2], 1.0 / 1.8));
        assert!(close(projected.reserves[0], 0.0));
        assert!(close(projected.reserves[1], 40.0));
        assert!(close(projected.reserves[2], 133.33333333333334));
        assert!(close(projected.ultimates[0], 180.0));
        assert!(close(projected.ultimates[1], 220.0));
        assert!(close(projected.ultimates[2], 273.33333333333337));
        assert!(close(projected.total_latest, 500.0));
        assert!(close(projected.total_expected_ultimate, 740.0));
        assert!(close(projected.total_reserve, 173.33333333333334));
        assert!(close(projected.total_ultimate, 673.3333333333334));
    }

    #[test]
    fn bf_projection_from_elr_matches_expected_ultimate() {
        let tri = tiny_triangle();
        let premium = [400.0, 300.0, 600.0];
        let elr = [0.5, 0.8, 0.5];

        let projected =
            BornhuetterFergusonEngine::project_from_elr(&tri.view(), &factors(), &premium, &elr)
                .expect("projection");

        assert_eq!(projected.expected_ultimate, vec![200.0, 240.0, 300.0]);
        assert!(close(projected.total_ultimate, 673.3333333333334));
    }

    #[test]
    fn bf_projection_from_scalar_elr_matches_vector_elr() {
        let tri = tiny_triangle();
        let premium = [400.0, 300.0, 600.0];

        let scalar = BornhuetterFergusonEngine::project_from_scalar_elr(
            &tri.view(),
            &factors(),
            &premium,
            0.5,
        )
        .expect("scalar projection");

        let vector = BornhuetterFergusonEngine::project_from_elr(
            &tri.view(),
            &factors(),
            &premium,
            &[0.5, 0.5, 0.5],
        )
        .expect("vector projection");

        assert_eq!(scalar, vector);
    }

    #[test]
    fn bf_projection_from_apriori_matches_scalar_elr() {
        let tri = tiny_triangle();
        let premium = [400.0, 300.0, 600.0];

        let from_apriori = BornhuetterFergusonEngine::project_from_apriori(
            &tri.view(),
            &factors(),
            &premium,
            AprioriInput::ScalarElr(0.5),
        )
        .expect("apriori projection");

        let scalar = BornhuetterFergusonEngine::project_from_scalar_elr(
            &tri.view(),
            &factors(),
            &premium,
            0.5,
        )
        .expect("scalar projection");

        assert_eq!(from_apriori, scalar);
    }

    #[test]
    fn bf_rejects_expected_ultimate_length_mismatch() {
        let tri = tiny_triangle();

        assert_eq!(
            BornhuetterFergusonEngine::project_from_expected_ultimate(
                &tri.view(),
                &factors(),
                &[200.0, 240.0],
            ),
            Err(ValidationError::ShapeMismatch)
        );
    }

    #[test]
    fn bf_rejects_negative_premium() {
        let tri = tiny_triangle();

        assert_eq!(
            BornhuetterFergusonEngine::project_from_elr(
                &tri.view(),
                &factors(),
                &[400.0, -300.0, 600.0],
                &[0.5, 0.8, 0.5],
            ),
            Err(ValidationError::Negative)
        );
    }

    #[test]
    fn bf_rejects_negative_elr() {
        let tri = tiny_triangle();

        assert_eq!(
            BornhuetterFergusonEngine::project_from_elr(
                &tri.view(),
                &factors(),
                &[400.0, 300.0, 600.0],
                &[0.5, -0.8, 0.5],
            ),
            Err(ValidationError::Negative)
        );
    }

    #[test]
    fn bf_rejects_non_finite_maturity_from_cdf() {
        let tri = tiny_triangle();
        let mut bad_factors = factors();
        bad_factors.cdfs[1] = f64::NAN;

        assert_eq!(
            BornhuetterFergusonEngine::project_from_expected_ultimate(
                &tri.view(),
                &bad_factors,
                &[200.0, 240.0, 300.0],
            ),
            Err(ValidationError::NotFinite)
        );
    }

    #[test]
    fn bf_rejects_non_cumulative_triangle() {
        #[rustfmt::skip]
        let tri = Triangle::new(3, 3, vec![
            100.0, 90.0, 180.0,
            120.0, 180.0, f64::NAN,
            140.0, f64::NAN, f64::NAN,
        ]);

        assert_eq!(
            BornhuetterFergusonEngine::project_from_expected_ultimate(
                &tri.view(),
                &factors(),
                &[200.0, 240.0, 300.0],
            ),
            Err(ValidationError::NonMonotoneCumulative {
                origin: 0,
                dev: 1,
                previous: 100.0,
                value: 90.0,
            })
        );
    }
}
