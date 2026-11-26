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

use crate::bf::{
    BornhuetterFergusonEngine, BornhuetterFergusonProjection,
    project_unchecked as bf_project_unchecked,
};
use crate::chain_ladder::latest_observed_by_origin;
use crate::ldf::{DevelopmentFactors, cdf_from_ldf, pct_reported};
use crate::triangle::latest_diagonal;
use crate::types::TriangleView;
use crate::validate::{
    ValidationError, validate_development_factor_slices, validate_latest_and_dev,
    validate_monotone_cumulative, validate_premium, validate_triangle,
};

/// Cape Cod ELR (scalar). Returns `NaN` when the used-premium total is zero.
pub fn elr(t: &TriangleView<'_>, ldf: &[f64], tail: f64, premium: &[f64]) -> f64 {
    let n = t.n_origins.min(t.n_devs);
    debug_assert_eq!(premium.len(), n);
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
pub struct CapeCodEngine;

impl CapeCodEngine {
    /// Derive the Cape Cod ELR and project BF ultimates/reserves.
    pub fn project(
        cumulative: &TriangleView<'_>,
        factors: &DevelopmentFactors,
        premium: &[f64],
    ) -> Result<CapeCodProjection, ValidationError> {
        validate_cape_cod_inputs(cumulative, factors, premium)?;

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

        let mut reported_percentage = vec![0.0; n];
        let mut used_premium = vec![0.0; n];
        for i in 0..n {
            let cdf = factors.cdfs[latest_dev[i]];
            reported_percentage[i] = if cdf > 0.0 { 1.0 / cdf } else { f64::NAN };
            if !reported_percentage[i].is_finite() {
                return Err(ValidationError::NotFinite);
            }
            used_premium[i] = premium[i] * reported_percentage[i];
        }

        let total_latest: f64 = latest.iter().sum();
        let total_used_premium: f64 = used_premium.iter().sum();
        if !total_latest.is_finite() || !total_used_premium.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if total_used_premium <= 0.0 {
            return Err(ValidationError::DivideByZero("Cape Cod used premium"));
        }

        let elr = total_latest / total_used_premium;
        if !elr.is_finite() || elr < 0.0 {
            return Err(ValidationError::NotFinite);
        }

        let expected_ultimate: Vec<f64> = premium.iter().map(|&p| p * elr).collect();
        let bf_projection = BornhuetterFergusonEngine::project_from_expected_ultimate(
            cumulative,
            factors,
            &expected_ultimate,
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
) -> Result<(), ValidationError> {
    validate_triangle(cumulative)?;
    validate_monotone_cumulative(cumulative)?;
    validate_premium(premium, cumulative.n_origins)?;
    validate_development_factor_slices(
        &factors.ldfs,
        &factors.cdfs,
        factors.tail,
        cumulative.n_devs,
        "Cape Cod factor",
    )
}
