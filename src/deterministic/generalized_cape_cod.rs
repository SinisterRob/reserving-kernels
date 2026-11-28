//! Generalized (Decay-Weighted) Cape Cod reserving method — Gluck (1997).
//!
//! Implements:
//!
//! * S. M. Gluck, *"Balancing Development and Trend in Loss Reserve Analysis"*,
//!   *PCAS*, Vol. LXXXIV (1997), pp. 482–532.
//! * J. Friedland, *Estimating Unpaid Claims Using Basic Techniques*, CAS
//!   (2010), Chapter 10 (Generalized Cape Cod).
//!
//! ## Formulation
//!
//! Standard Cape Cod weights every origin year equally to estimate a single
//! constant expected loss ratio across the triangle. Gluck (1997) introduces
//! geometric decay weights `w_{i,k} = delta^{|i - k|}` (`delta in (0, 1]`)
//! centered on each target origin year `i`:
//!
//! `ELR_i = [ sum_k w_{i,k} · C_{k, d_k} · tau_k ] / [ tau_i · sum_k w_{i,k} · P_k · p_{d_k} ]`
//!
//! where `p_{d_k} = 1 / CDF_{d_k}` is the percent reported at origin `k`'s
//! latest observed age `d_k`, `P_k` is on-level earned premium / exposure, and
//! `tau_k` is an optional loss-trend factor trending origin `k` to a common
//! cost level (`tau_k = 1` when omitted).
//!
//! ## Limiting Identities
//!
//! * **`delta = 1.0`** (with unit trend): reduces identically to standard
//!   Stanard–Bühlmann Cape Cod ([`crate::deterministic::cape_cod`]).
//! * **`delta -> 0+`**: puts full weight on origin `i` alone, so
//!   `ELR_i -> C_{i, d_i} / (P_i · p_{d_i}) = U_i^{CL} / P_i` and the resulting
//!   BF projection reduces identically to Chain Ladder ([`crate::deterministic::chain_ladder`]).

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
    MonotonicityPolicy, validate_development_factor_slices, validate_latest_and_dev,
    validate_monotone_cumulative, validate_positive_finite_slice, validate_premium,
    validate_triangle,
};

/// Result of a Generalized (Decay-Weighted) Cape Cod projection.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GeneralizedCapeCodProjection {
    /// Decay factor `delta` in `(0, 1]`.
    pub decay: f64,
    /// Origin-specific expected loss ratios `ELR_i` (length `n`).
    pub elr_by_origin: Vec<f64>,
    /// Row-major `n × n` normalized weight matrix (`row i` sums to `1.0`),
    /// showing the relative contribution of origin `k`'s used premium to `ELR_i`.
    pub normalized_weights: Vec<f64>,
    /// Latest observed cumulative loss by origin.
    pub latest: Vec<f64>,
    /// Latest observed development index by origin.
    pub latest_dev: Vec<usize>,
    /// Percent reported `p_{d_i} = 1 / CDF_{d_i}` by origin.
    pub reported_percentage: Vec<f64>,
    /// Used-up premium `P_i * p_{d_i}` by origin.
    pub used_premium: Vec<f64>,
    /// Expected ultimate `P_i * ELR_i` by origin.
    pub expected_ultimate: Vec<f64>,
    /// Underlying BF projection driven by `expected_ultimate`.
    pub bf_projection: BornhuetterFergusonProjection,
}

/// Compute origin-specific Gluck (1997) decay-weighted expected loss ratios
/// without input validation.
pub fn elr_by_origin_unchecked(
    t: &TriangleView<'_>,
    ldf: &[f64],
    tail: f64,
    premium: &[f64],
    decay: f64,
) -> Vec<f64> {
    let n = t.n_origins.min(t.n_devs);
    debug_assert_eq!(premium.len(), n);
    let cdf = cdf_from_ldf(ldf, tail);
    let pct = pct_reported(&cdf);
    let latest = latest_diagonal(t);

    let mut used = vec![0.0; n];
    for k in 0..n {
        used[k] = premium[k] * pct[n - 1 - k];
    }

    let mut elrs = vec![0.0; n];
    for i in 0..n {
        let mut num = 0.0;
        let mut den = 0.0;
        for k in 0..n {
            let w = decay.powi(i.abs_diff(k) as i32);
            num += w * latest[k];
            den += w * used[k];
        }
        elrs[i] = if den > 0.0 { num / den } else { f64::NAN };
    }
    elrs
}

/// Unchecked Generalized Cape Cod projection returning `(elr_by_origin, bf_projection)`.
pub fn project_unchecked(
    t: &TriangleView<'_>,
    ldf: &[f64],
    tail: f64,
    premium: &[f64],
    decay: f64,
) -> (Vec<f64>, BornhuetterFergusonProjection) {
    let elrs = elr_by_origin_unchecked(t, ldf, tail, premium, decay);
    let expected: Vec<f64> = premium
        .iter()
        .zip(elrs.iter())
        .map(|(&p, &r)| p * r)
        .collect();
    (elrs, bf_project_unchecked(t, ldf, tail, &expected))
}

/// Checked Generalized Cape Cod service (Gluck 1997).
#[derive(Debug, Clone, Copy, Default)]
pub struct GeneralizedCapeCodEngine;

impl GeneralizedCapeCodEngine {
    /// Run decay-weighted Cape Cod with strict cumulative monotonicity and no
    /// explicit loss-trend factors (`tau_k = 1.0`).
    pub fn project(
        cumulative: &TriangleView<'_>,
        factors: &DevelopmentFactors,
        premium: &[f64],
        decay: f64,
    ) -> Result<GeneralizedCapeCodProjection, ReservingError> {
        Self::project_with_options(
            cumulative,
            factors,
            premium,
            decay,
            None,
            MonotonicityPolicy::Strict,
        )
    }

    /// Run decay-weighted Cape Cod with optional loss-trend factors `tau_k`
    /// (trending origin `k` losses to a common base level) and explicit
    /// [`MonotonicityPolicy`].
    pub fn project_with_options(
        cumulative: &TriangleView<'_>,
        factors: &DevelopmentFactors,
        premium: &[f64],
        decay: f64,
        loss_trend_to_base: Option<&[f64]>,
        policy: MonotonicityPolicy,
    ) -> Result<GeneralizedCapeCodProjection, ReservingError> {
        validate_triangle(cumulative)?;
        if policy == MonotonicityPolicy::Strict {
            validate_monotone_cumulative(cumulative)?;
        }
        let n = cumulative.n_origins;
        validate_premium(premium, n)?;
        validate_development_factor_slices(
            &factors.ldfs,
            &factors.cdfs,
            factors.tail,
            cumulative.n_devs,
            "Generalized Cape Cod factor",
        )?;
        if !decay.is_finite() {
            return Err(ReservingError::NotFinite);
        }
        if !(decay > 0.0 && decay <= 1.0) {
            return Err(ReservingError::InvalidParameter(
                "Generalized Cape Cod decay factor must lie in (0, 1]",
            ));
        }
        if let Some(trend) = loss_trend_to_base {
            if trend.len() != n {
                return Err(ReservingError::ShapeMismatch);
            }
            validate_positive_finite_slice(trend, "loss_trend_to_base")?;
        }

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
            reported_percentage[i] = 1.0 / cdf;
            used_premium[i] = premium[i] * reported_percentage[i];
        }

        let mut decay_powers = vec![1.0; n];
        for d in 1..n {
            decay_powers[d] = decay_powers[d - 1] * decay;
        }

        let mut elr_by_origin = vec![0.0; n];
        let mut normalized_weights = vec![0.0; n * n];
        let mut expected_ultimate = vec![0.0; n];

        for i in 0..n {
            let mut num = 0.0;
            let mut den = 0.0;
            for k in 0..n {
                let w = decay_powers[i.abs_diff(k)];
                let tau_k = loss_trend_to_base.map_or(1.0, |t| t[k]);
                let eff_w = w * used_premium[k];
                normalized_weights[i * n + k] = eff_w;
                num += w * latest[k] * tau_k;
                den += eff_w;
            }
            if !den.is_finite() || den <= 0.0 {
                return Err(ReservingError::DivideByZero(
                    "Generalized Cape Cod used premium",
                ));
            }
            for k in 0..n {
                normalized_weights[i * n + k] /= den;
            }
            let tau_i = loss_trend_to_base.map_or(1.0, |t| t[i]);
            let elr_i = (num / den) / tau_i;
            if !elr_i.is_finite() || elr_i < 0.0 {
                return Err(ReservingError::NotFinite);
            }
            elr_by_origin[i] = elr_i;
            expected_ultimate[i] = premium[i] * elr_i;
        }

        let bf_projection = BornhuetterFergusonEngine::project_from_expected_ultimate_with_policy(
            cumulative,
            factors,
            &expected_ultimate,
            policy,
        )?;

        Ok(GeneralizedCapeCodProjection {
            decay,
            elr_by_origin,
            normalized_weights,
            latest,
            latest_dev,
            reported_percentage,
            used_premium,
            expected_ultimate,
            bf_projection,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deterministic::cape_cod::CapeCodEngine;
    use crate::deterministic::chain_ladder::ChainLadderEngine;
    use crate::deterministic::ldf::DevelopmentFactorEngine;
    use crate::types::Triangle;

    const NAN: f64 = f64::NAN;

    fn sample_triangle() -> (Triangle, Vec<f64>) {
        #[rustfmt::skip]
        let data = vec![
            100.0, 160.0, 200.0, 210.0,
            120.0, 190.0, 245.0,   NAN,
            150.0, 245.0,   NAN,   NAN,
            200.0,   NAN,   NAN,   NAN,
        ];
        (Triangle::new(4, 4, data), vec![300.0, 320.0, 350.0, 400.0])
    }

    #[test]
    fn decay_one_matches_standard_cape_cod_exactly() {
        let (tri, prem) = sample_triangle();
        let factors = DevelopmentFactorEngine::volume_weighted(&tri.view(), 1.0).unwrap();
        let cc = CapeCodEngine::project(&tri.view(), &factors, &prem).unwrap();
        let gcc = GeneralizedCapeCodEngine::project(&tri.view(), &factors, &prem, 1.0).unwrap();

        for i in 0..4 {
            assert!((gcc.elr_by_origin[i] - cc.elr).abs() < 1e-12);
            assert!((gcc.bf_projection.reserves[i] - cc.bf_projection.reserves[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn decay_near_zero_converges_to_chain_ladder() {
        let (tri, prem) = sample_triangle();
        let factors = DevelopmentFactorEngine::volume_weighted(&tri.view(), 1.0).unwrap();
        let cl = ChainLadderEngine::project(&tri.view(), &factors).unwrap();
        let gcc = GeneralizedCapeCodEngine::project(&tri.view(), &factors, &prem, 1e-12).unwrap();

        for i in 0..4 {
            assert!(
                (gcc.bf_projection.reserves[i] - cl.reserves[i]).abs() < 1e-6,
                "origin {i}: gcc={} cl={}",
                gcc.bf_projection.reserves[i],
                cl.reserves[i]
            );
        }
    }

    #[test]
    fn rejects_invalid_decay() {
        let (tri, prem) = sample_triangle();
        let factors = DevelopmentFactorEngine::volume_weighted(&tri.view(), 1.0).unwrap();
        assert!(GeneralizedCapeCodEngine::project(&tri.view(), &factors, &prem, 0.0).is_err());
        assert!(GeneralizedCapeCodEngine::project(&tri.view(), &factors, &prem, 1.1).is_err());
    }
}
