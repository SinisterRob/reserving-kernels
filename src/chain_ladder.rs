//! Chain-ladder projection.
//!
//! Section 3 consumes a cumulative triangle, latest observed development
//! positions, and CDFs to produce ultimates and reserves.
//!
//! Three layers, all in this file:
//!
//! * **L0 columnar kernels** — flat slices and caller-provided output buffers
//!   (e.g. [`project_ultimates`], [`reserves`], [`latest_observed_by_origin`]).
//! * **L1 view convenience** — [`project_unchecked`] takes a [`TriangleView`],
//!   raw `ldfs` slice, and `tail`, and returns an owned
//!   [`ChainLadderProjection`] without validation. Used by the WASM hot path
//!   and the v0.1 unchecked surface.
//! * **L2 service / engine** — [`ChainLadderEngine`] performs the same
//!   projection from a validated [`DevelopmentFactors`] and returns
//!   `Result<ChainLadderProjection, ValidationError>`.

use crate::ldf::{DevelopmentFactors, cdf_from_ldf};
use crate::types::TriangleView;
use crate::validate::{
    MonotonicityPolicy, ValidationError, validate_development_factor_slices, validate_finite_slice,
    validate_latest_and_dev, validate_monotone_cumulative, validate_triangle,
};

// ===========================================================================
// Types
// ===========================================================================

/// Chain-ladder projection result.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ChainLadderProjection {
    /// Latest observed cumulative value by origin.
    pub latest: Vec<f64>,
    /// Latest observed development index by origin.
    pub latest_dev: Vec<usize>,
    /// Projected ultimate by origin.
    pub ultimates: Vec<f64>,
    /// Reserve by origin: ultimate minus latest.
    pub reserves: Vec<f64>,
    /// Total latest observed value.
    pub total_latest: f64,
    /// Total projected ultimate.
    pub total_ultimate: f64,
    /// Total reserve.
    pub total_reserve: f64,
}

// ===========================================================================
// L0 — columnar kernels
// ===========================================================================

/// Project latest observed values to ultimate using CDFs indexed by latest dev.
///
/// # Safety Contract
///
/// This is a raw L0 kernel. Callers must ensure:
///
/// * `latest.len() == latest_dev.len() == out.len()`
/// * every `latest_dev[i] < cdfs.len()`
/// * `latest` and `cdfs` contain finite values
///
/// Use [`ChainLadderEngine::project`] for checked projection.
pub fn project_ultimates(latest: &[f64], latest_dev: &[usize], cdfs: &[f64], out: &mut [f64]) {
    debug_assert_eq!(latest.len(), latest_dev.len());
    debug_assert_eq!(latest.len(), out.len());

    for i in 0..latest.len() {
        out[i] = latest[i] * cdfs[latest_dev[i]];
    }
}

/// Reserve by origin: ultimate minus latest.
///
/// # Safety Contract
///
/// This is a raw L0 kernel. Callers must ensure
/// `latest.len() == ultimates.len() == out.len()` and finite inputs. Use
/// [`ChainLadderEngine::project`] for checked projection.
pub fn reserves(latest: &[f64], ultimates: &[f64], out: &mut [f64]) {
    debug_assert_eq!(latest.len(), ultimates.len());
    debug_assert_eq!(latest.len(), out.len());

    for i in 0..latest.len() {
        out[i] = ultimates[i] - latest[i];
    }
}

/// Extract latest observed values and development indices under right-triangle geometry.
///
/// # Safety Contract
///
/// This is a raw L0 kernel. Callers must ensure `cum.len() == n_origin * n_dev`,
/// `latest.len() == n_origin`, and `latest_dev.len() == n_origin`.
pub fn latest_observed_by_origin(
    cum: &[f64],
    n_origin: usize,
    n_dev: usize,
    latest: &mut [f64],
    latest_dev: &mut [usize],
) {
    debug_assert_eq!(cum.len(), n_origin * n_dev);
    debug_assert_eq!(latest.len(), n_origin);
    debug_assert_eq!(latest_dev.len(), n_origin);

    for i in 0..n_origin {
        if i >= n_dev {
            latest[i] = f64::NAN;
            latest_dev[i] = 0;
            continue;
        }
        let dev = n_dev - 1 - i;
        latest_dev[i] = dev;
        latest[i] = cum[i * n_dev + dev];
    }
}

/// Extract latest observed values and development indices from observed row widths.
///
/// # Safety Contract
///
/// This is a raw L0 kernel. Callers must ensure `cum.len() == n_origin * n_dev`,
/// `observed_counts.len() == n_origin`, output lengths match `n_origin`, and
/// each non-zero observed count is at most `n_dev`.
pub fn latest_observed_by_origin_observed_counts(
    cum: &[f64],
    n_origin: usize,
    n_dev: usize,
    observed_counts: &[usize],
    latest: &mut [f64],
    latest_dev: &mut [usize],
) {
    debug_assert_eq!(cum.len(), n_origin * n_dev);
    debug_assert_eq!(observed_counts.len(), n_origin);
    debug_assert_eq!(latest.len(), n_origin);
    debug_assert_eq!(latest_dev.len(), n_origin);

    for i in 0..n_origin {
        if observed_counts[i] == 0 {
            latest[i] = f64::NAN;
            latest_dev[i] = 0;
            continue;
        }
        let dev = observed_counts[i] - 1;
        debug_assert!(dev < n_dev);
        latest_dev[i] = dev;
        latest[i] = cum[i * n_dev + dev];
    }
}

// ===========================================================================
// L1 — unchecked view-based projection
// ===========================================================================

/// Project ultimates using the classical chain-ladder identity
/// `U_i = C_{i, n−1−i} × cdf_{n−1−i}` without validation.
///
/// `ldfs` must have length `n − 1` (age-to-age factors). `tail` is the factor
/// from development age `n − 1` to ultimate; pass `1.0` if fully developed.
///
/// Prefer [`ChainLadderEngine::project`] at external ingress boundaries.
pub fn project_unchecked(t: &TriangleView<'_>, ldfs: &[f64], tail: f64) -> ChainLadderProjection {
    let n = t.n_origins.min(t.n_devs);
    let cdfs = cdf_from_ldf(ldfs, tail);

    let mut latest = vec![0.0; n];
    let mut latest_dev = vec![0; n];
    for i in 0..n {
        let d = n - 1 - i;
        latest[i] = t.get(i, d);
        latest_dev[i] = d;
    }

    let mut ultimates = vec![0.0; n];
    project_ultimates(&latest, &latest_dev, &cdfs, &mut ultimates);

    let mut reserve_values = vec![0.0; n];
    reserves(&latest, &ultimates, &mut reserve_values);

    let total_latest: f64 = latest.iter().sum();
    let total_ultimate: f64 = ultimates.iter().sum();
    let total_reserve: f64 = reserve_values.iter().sum();

    ChainLadderProjection {
        latest,
        latest_dev,
        ultimates,
        reserves: reserve_values,
        total_latest,
        total_ultimate,
        total_reserve,
    }
}

// ===========================================================================
// L2 — validated engine
// ===========================================================================

/// Section 3 chain-ladder projection service.
pub struct ChainLadderEngine;

impl ChainLadderEngine {
    /// Project ultimates and reserves from a cumulative triangle and CDFs.
    pub fn project(
        cumulative: &TriangleView<'_>,
        factors: &DevelopmentFactors,
    ) -> Result<ChainLadderProjection, ValidationError> {
        Self::project_with_policy(cumulative, factors, MonotonicityPolicy::Strict)
    }

    /// Project ultimates and reserves under an explicit [`MonotonicityPolicy`].
    pub fn project_with_policy(
        cumulative: &TriangleView<'_>,
        factors: &DevelopmentFactors,
        policy: MonotonicityPolicy,
    ) -> Result<ChainLadderProjection, ValidationError> {
        validate_projection_inputs_with_policy(cumulative, factors, policy)?;

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

        let mut ultimates = vec![0.0; n];
        project_ultimates(&latest, &latest_dev, &factors.cdfs, &mut ultimates);
        validate_finite_slice(&ultimates)?;

        let mut reserve_values = vec![0.0; n];
        reserves(&latest, &ultimates, &mut reserve_values);
        validate_finite_slice(&reserve_values)?;

        let total_latest: f64 = latest.iter().sum();
        let total_ultimate: f64 = ultimates.iter().sum();
        let total_reserve: f64 = reserve_values.iter().sum();

        if !total_latest.is_finite() || !total_ultimate.is_finite() || !total_reserve.is_finite() {
            return Err(ValidationError::NotFinite);
        }

        Ok(ChainLadderProjection {
            latest,
            latest_dev,
            ultimates,
            reserves: reserve_values,
            total_latest,
            total_ultimate,
            total_reserve,
        })
    }
}

fn validate_projection_inputs_with_policy(
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
        "projection factor",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ldf::{DevelopmentFactorEngine, DevelopmentFactors};
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

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * (1.0 + b.abs())
    }

    #[test]
    fn project_ultimates_kernel_projects_by_latest_dev() {
        let latest = [180.0, 180.0, 140.0];
        let latest_dev = [2, 1, 0];
        let cdfs = [1.8, 1.2, 1.0];
        let mut out = [0.0; 3];

        project_ultimates(&latest, &latest_dev, &cdfs, &mut out);

        assert_eq!(out, [180.0, 216.0, 252.0]);
    }

    #[test]
    fn reserves_kernel_subtracts_latest() {
        let latest = [180.0, 180.0, 140.0];
        let ultimates = [180.0, 216.0, 252.0];
        let mut out = [0.0; 3];

        reserves(&latest, &ultimates, &mut out);

        assert_eq!(out, [0.0, 36.0, 112.0]);
    }

    #[test]
    fn latest_observed_counts_handles_rectangular_schedule() {
        #[rustfmt::skip]
        let cum = [
            100.0, 150.0, 180.0, 198.0,
            120.0, 180.0, f64::NAN, f64::NAN,
            140.0, f64::NAN, f64::NAN, f64::NAN,
        ];
        let observed_counts = [4, 2, 1];
        let mut latest = [0.0; 3];
        let mut latest_dev = [0; 3];

        latest_observed_by_origin_observed_counts(
            &cum,
            3,
            4,
            &observed_counts,
            &mut latest,
            &mut latest_dev,
        );

        assert_eq!(latest, [198.0, 180.0, 140.0]);
        assert_eq!(latest_dev, [3, 1, 0]);
    }

    #[test]
    fn project_unchecked_matches_engine_on_valid_input() {
        let tri = tiny_triangle();
        let factors = DevelopmentFactorEngine::volume_weighted(&tri.view(), 1.0).expect("factors");

        let unchecked = project_unchecked(&tri.view(), &factors.ldfs, factors.tail);
        let engine = ChainLadderEngine::project(&tri.view(), &factors).expect("engine");

        assert_eq!(unchecked, engine);
    }

    #[test]
    fn chain_ladder_projection_known_triangle() {
        let tri = tiny_triangle();
        let factors = DevelopmentFactorEngine::volume_weighted(&tri.view(), 1.0).expect("factors");

        let projected = ChainLadderEngine::project(&tri.view(), &factors).expect("projection");

        assert_eq!(projected.latest, vec![180.0, 180.0, 140.0]);
        assert_eq!(projected.latest_dev, vec![2, 1, 0]);
        assert!(close(projected.ultimates[0], 180.0));
        assert!(close(projected.ultimates[1], 216.0));
        assert!(close(projected.ultimates[2], 252.0));
        assert!(close(projected.reserves[0], 0.0));
        assert!(close(projected.reserves[1], 36.0));
        assert!(close(projected.reserves[2], 112.0));
        assert!(close(projected.total_latest, 500.0));
        assert!(close(projected.total_ultimate, 648.0));
        assert!(close(projected.total_reserve, 148.0));
    }

    #[test]
    fn project_rejects_cdf_length_mismatch() {
        let tri = tiny_triangle();
        let factors = DevelopmentFactors {
            ldfs: vec![1.5],
            cdfs: vec![1.8, 1.2],
            tail: 1.0,
        };

        assert_eq!(
            ChainLadderEngine::project(&tri.view(), &factors),
            Err(ValidationError::ShapeMismatch)
        );
    }

    #[test]
    fn project_rejects_non_cumulative_triangle() {
        #[rustfmt::skip]
        let tri = Triangle::new(3, 3, vec![
            100.0, 90.0, 180.0,
            120.0, 180.0, f64::NAN,
            140.0, f64::NAN, f64::NAN,
        ]);
        let factors = DevelopmentFactors {
            ldfs: vec![1.5, 1.2],
            cdfs: vec![1.8, 1.2, 1.0],
            tail: 1.0,
        };

        assert_eq!(
            ChainLadderEngine::project(&tri.view(), &factors),
            Err(ValidationError::NonMonotoneCumulative {
                origin: 0,
                dev: 1,
                previous: 100.0,
                value: 90.0,
            })
        );
    }

    #[test]
    fn project_rejects_non_finite_cdf() {
        let tri = tiny_triangle();
        let factors = DevelopmentFactors {
            ldfs: vec![1.5, f64::NAN],
            cdfs: vec![1.8, f64::NAN, 1.0],
            tail: 1.0,
        };

        assert_eq!(
            ChainLadderEngine::project(&tri.view(), &factors),
            Err(ValidationError::NotFinite)
        );
    }

    #[test]
    fn project_rejects_non_finite_latest() {
        #[rustfmt::skip]
        let tri = Triangle::new(3, 3, vec![
            100.0, 150.0, f64::NAN,
            120.0, 180.0, f64::NAN,
            140.0, f64::NAN, f64::NAN,
        ]);
        let factors = DevelopmentFactors {
            ldfs: vec![1.5, 1.2],
            cdfs: vec![1.8, 1.2, 1.0],
            tail: 1.0,
        };

        assert_eq!(
            ChainLadderEngine::project(&tri.view(), &factors),
            Err(ValidationError::NotFinite)
        );
    }
}
