//! Cape Cod / Stanard–Bühlmann method.
//!
//! Cape Cod is BF where the *a priori* expected loss ratio is derived from
//! the triangle itself, using only on-level (used) premium:
//!
//! `ELR = Σ_i C_{i, n−1−i} / Σ_i (P_i × r_{n−1−i})`,
//!
//! where `r_j = 1 / cdf_j` is the percent-reported at development age `j`.
//! The used premium `P_i × r_{n−1−i}` is the fraction of each origin's
//! premium that "matches" losses actually observed. Once `ELR` is in hand,
//! the method projects exactly like BF.

use crate::deterministic::bf::{
    BornhuetterFergusonEngine, BornhuetterFergusonProjection,
    project_unchecked as bf_project_unchecked,
};
use crate::deterministic::chain_ladder::latest_observed_by_origin;
use crate::deterministic::ldf::{DevelopmentFactors, cdf_from_ldf, pct_reported};
use crate::error::ReservingError;
use crate::preprocessing::triangle::latest_diagonal;
use crate::types::TriangleView;
use crate::validate::{
    MonotonicityPolicy, validate_development_factor_slices_with_policy, validate_latest_and_dev,
    validate_monotone_cumulative, validate_premium, validate_triangle,
};

/// Cape Cod ELR (scalar). Returns `NaN` when the used-premium total is zero
/// or when `premium.len()` / `ldf.len()` do not match the triangle (never panics).
pub fn elr(t: &TriangleView<'_>, ldf: &[f64], tail: f64, premium: &[f64]) -> f64 {
    let n = t.n_origins.min(t.n_devs);
    if premium.len() != n || ldf.len() + 1 != n {
        return f64::NAN;
    }
    let cdf = cdf_from_ldf(ldf, tail);
    let pct = pct_reported(&cdf);
    let latest = latest_diagonal(t);
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        let age = n - 1 - i;
        num += latest[i];
        den += premium[i] * pct[age];
    }
    if den > 0.0 { num / den } else { f64::NAN }
}

/// Cape Cod projection: compute the data-implied ELR, then run BF with
/// `expected_i = premium_i × ELR`. Performs no validation. Prefer
/// [`CapeCodEngine::project`] at external ingress boundaries.
pub fn project_unchecked(
    t: &TriangleView<'_>,
    ldf: &[f64],
    tail: f64,
    premium: &[f64],
) -> (f64, BornhuetterFergusonProjection) {
    let r = elr(t, ldf, tail, premium);
    let expected: Vec<f64> = premium.iter().map(|&p| p * r).collect();
    (r, bf_project_unchecked(t, ldf, tail, &expected))
}

/// Checked Cape Cod projection result.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct CapeCodProjection {
    /// Data-implied expected loss ratio.
    pub elr: f64,
    /// Latest observed cumulative value by origin.
    pub latest: Vec<f64>,
    /// Latest observed development index by origin.
    pub latest_dev: Vec<usize>,
    /// Reported percentage at each origin's latest observed development.
    pub reported_percentage: Vec<f64>,
    /// Premium used in the Cape Cod denominator by origin.
    pub used_premium: Vec<f64>,
    /// Expected ultimate by origin: `premium[i] * elr`.
    pub expected_ultimate: Vec<f64>,
    /// Total latest observed loss.
    pub total_latest: f64,
    /// Total used premium.
    pub total_used_premium: f64,
    /// BF projection run with the Cape Cod expected ultimates.
    pub bf_projection: BornhuetterFergusonProjection,
}

/// Checked Cape Cod service.
#[derive(Debug, Clone, Copy, Default)]
pub struct CapeCodEngine;

impl CapeCodEngine {
    /// Derive the Cape Cod ELR and project BF ultimates/reserves.
    pub fn project(
        cumulative: &TriangleView<'_>,
        factors: &DevelopmentFactors,
        premium: &[f64],
    ) -> Result<CapeCodProjection, ReservingError> {
        Self::project_with_policy(cumulative, factors, premium, MonotonicityPolicy::Strict)
    }

    /// Derive the Cape Cod ELR and project BF ultimates/reserves under an
    /// explicit [`MonotonicityPolicy`].
    pub fn project_with_policy(
        cumulative: &TriangleView<'_>,
        factors: &DevelopmentFactors,
        premium: &[f64],
        policy: MonotonicityPolicy,
    ) -> Result<CapeCodProjection, ReservingError> {
        validate_cape_cod_inputs(cumulative, factors, premium, policy)?;

        let n = cumulative.n_origins;
        let mut latest = vec![0.0; n];
        let mut latest_dev = vec![0; n];
        latest_observed_by_origin(
            cumulative.data,
            cumulative.n_origins,
            cumulative.n_devs,
            &mut latest,
            &mut latest_dev,
        )?;
        validate_latest_and_dev(&latest, &latest_dev, &factors.cdfs)?;

        let mut reported_percentage = vec![0.0; n];
        let mut used_premium = vec![0.0; n];
        for i in 0..n {
            let cdf = factors.cdfs[latest_dev[i]];
            reported_percentage[i] = if cdf > 0.0 { 1.0 / cdf } else { f64::NAN };
            if !reported_percentage[i].is_finite() {
                return Err(ReservingError::NotFinite);
            }
            used_premium[i] = premium[i] * reported_percentage[i];
        }

        let total_latest: f64 = latest.iter().sum();
        let total_used_premium: f64 = used_premium.iter().sum();
        if !total_latest.is_finite() || !total_used_premium.is_finite() {
            return Err(ReservingError::NotFinite);
        }
        if total_used_premium <= 0.0 {
            return Err(ReservingError::DivideByZero("Cape Cod used premium"));
        }

        let elr = total_latest / total_used_premium;
        if !elr.is_finite() || elr < 0.0 {
            return Err(ReservingError::NotFinite);
        }

        let expected_ultimate: Vec<f64> = premium.iter().map(|&p| p * elr).collect();
        let bf_projection = BornhuetterFergusonEngine::project_from_expected_ultimate_with_policy(
            cumulative,
            factors,
            &expected_ultimate,
            policy,
        )?;

        Ok(CapeCodProjection {
            elr,
            latest,
            latest_dev,
            reported_percentage,
            used_premium,
            expected_ultimate,
            total_latest,
            total_used_premium,
            bf_projection,
        })
    }
}

fn validate_cape_cod_inputs(
    cumulative: &TriangleView<'_>,
    factors: &DevelopmentFactors,
    premium: &[f64],
    policy: MonotonicityPolicy,
) -> Result<(), ReservingError> {
    validate_triangle(cumulative)?;
    if policy == MonotonicityPolicy::Strict {
        validate_monotone_cumulative(cumulative)?;
    }
    validate_premium(premium, cumulative.n_origins)?;
    validate_development_factor_slices_with_policy(
        &factors.ldfs,
        &factors.cdfs,
        factors.tail,
        cumulative.n_devs,
        "Cape Cod factor",
        policy,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deterministic::ldf::DevelopmentFactorEngine;
    use crate::types::Triangle;

    #[test]
    fn cape_cod_elr_and_used_premium_conservation() {
        let tri = Triangle::try_from_upper_rows(&[
            &[100.0, 160.0, 195.0, 205.0],
            &[120.0, 190.0, 232.0],
            &[135.0, 218.0],
            &[150.0],
        ])
        .expect("triangle");
        let prem = [300.0, 340.0, 380.0, 420.0];
        let factors = DevelopmentFactorEngine::volume_weighted(&tri.view(), 1.0).expect("factors");
        let proj = CapeCodEngine::project(&tri.view(), &factors, &prem).expect("cape cod");

        assert!((proj.elr - proj.total_latest / proj.total_used_premium).abs() < 1e-12);
        assert!(proj.bf_projection.total_reserve > 0.0);
        for (i, (&p, &eu)) in prem.iter().zip(proj.expected_ultimate.iter()).enumerate() {
            assert!((eu - p * proj.elr).abs() < 1e-10, "origin {i}");
        }
    }

    #[test]
    fn cape_cod_rejects_non_positive_premium() {
        let tri =
            Triangle::try_from_upper_rows(&[&[100.0, 160.0, 195.0], &[120.0, 190.0], &[135.0]])
                .expect("triangle");
        let factors = DevelopmentFactorEngine::volume_weighted(&tri.view(), 1.0).expect("factors");
        assert_eq!(
            CapeCodEngine::project(&tri.view(), &factors, &[300.0, 0.0, 380.0]),
            Err(ReservingError::NonPositive("premium"))
        );
    }
}
