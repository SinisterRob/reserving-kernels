//! Paid-versus-incurred reconciliation diagnostics.

use crate::chain_ladder::{ChainLadderEngine, ChainLadderProjection};
use crate::ldf::{DevelopmentFactorEngine, DevelopmentFactors};
use crate::types::{Triangle, TriangleView};
use crate::validate::{ValidationError, validate_triangle};

/// Paid-versus-incurred reserve reconciliation by origin.
#[derive(Debug, Clone, PartialEq)]
pub struct PaidIncurredReconciliation {
    /// Latest paid cumulative values by origin.
    pub paid_latest: Vec<f64>,
    /// Latest incurred cumulative values by origin.
    pub incurred_latest: Vec<f64>,
    /// Latest case reserve by origin, `incurred_latest - paid_latest`.
    pub case_reserve_latest: Vec<f64>,
    /// Latest paid-to-incurred ratio by origin.
    pub paid_to_incurred_ratio: Vec<f64>,
    /// Paid chain-ladder ultimates by origin.
    pub paid_ldf_ultimate: Vec<f64>,
    /// Incurred chain-ladder ultimates by origin.
    pub incurred_ldf_ultimate: Vec<f64>,
    /// Paid minus incurred ultimate by origin.
    pub ultimate_gap: Vec<f64>,
    /// Current case-adequacy signal by origin, equal to `ultimate_gap`.
    pub case_adequacy_signal: Vec<f64>,
}

fn validate_compatible_triangles(
    paid: &TriangleView<'_>,
    incurred: &TriangleView<'_>,
) -> Result<(), ValidationError> {
    validate_triangle(paid)?;
    validate_triangle(incurred)?;
    if paid.n_origins != incurred.n_origins || paid.n_devs != incurred.n_devs {
        return Err(ValidationError::ShapeMismatch);
    }
    Ok(())
}

/// Case reserve triangle `incurred - paid`, preserving `NaN` where either cell is missing.
pub fn case_triangle(
    paid: &TriangleView<'_>,
    incurred: &TriangleView<'_>,
) -> Result<Triangle, ValidationError> {
    validate_compatible_triangles(paid, incurred)?;

    let mut data = vec![f64::NAN; paid.n_origins * paid.n_devs];
    for origin in 0..paid.n_origins {
        for dev in 0..paid.n_devs {
            let paid_value = paid.get(origin, dev);
            let incurred_value = incurred.get(origin, dev);
            if paid_value.is_finite() && incurred_value.is_finite() {
                data[origin * paid.n_devs + dev] = incurred_value - paid_value;
            }
        }
    }
    Ok(Triangle::new(paid.n_origins, paid.n_devs, data))
}

/// Paid-to-incurred ratio triangle `paid / incurred`, preserving `NaN` where undefined.
pub fn paid_to_incurred_ratio_triangle(
    paid: &TriangleView<'_>,
    incurred: &TriangleView<'_>,
) -> Result<Triangle, ValidationError> {
    validate_compatible_triangles(paid, incurred)?;

    let mut data = vec![f64::NAN; paid.n_origins * paid.n_devs];
    for origin in 0..paid.n_origins {
        for dev in 0..paid.n_devs {
            let paid_value = paid.get(origin, dev);
            let incurred_value = incurred.get(origin, dev);
            if paid_value.is_finite() && incurred_value.is_finite() && incurred_value != 0.0 {
                data[origin * paid.n_devs + dev] = paid_value / incurred_value;
            }
        }
    }
    Ok(Triangle::new(paid.n_origins, paid.n_devs, data))
}

fn build_reconciliation(
    paid_projection: ChainLadderProjection,
    incurred_projection: ChainLadderProjection,
) -> PaidIncurredReconciliation {
    let n = paid_projection.latest.len();
    let mut case_reserve_latest = Vec::with_capacity(n);
    let mut paid_to_incurred_ratio = Vec::with_capacity(n);
    let mut ultimate_gap = Vec::with_capacity(n);
    let mut case_adequacy_signal = Vec::with_capacity(n);

    for origin in 0..n {
        let paid_latest = paid_projection.latest[origin];
        let incurred_latest = incurred_projection.latest[origin];
        let case_latest = incurred_latest - paid_latest;
        let ratio = if incurred_latest == 0.0 {
            f64::NAN
        } else {
            paid_latest / incurred_latest
        };
        let gap = paid_projection.ultimates[origin] - incurred_projection.ultimates[origin];

        case_reserve_latest.push(case_latest);
        paid_to_incurred_ratio.push(ratio);
        ultimate_gap.push(gap);
        case_adequacy_signal.push(gap);
    }

    PaidIncurredReconciliation {
        paid_latest: paid_projection.latest,
        incurred_latest: incurred_projection.latest,
        case_reserve_latest,
        paid_to_incurred_ratio,
        paid_ldf_ultimate: paid_projection.ultimates,
        incurred_ldf_ultimate: incurred_projection.ultimates,
        ultimate_gap,
        case_adequacy_signal,
    }
}

/// Reconcile paid and incurred triangles using caller-supplied development factors.
pub fn reconcile_with_factors(
    paid: &TriangleView<'_>,
    incurred: &TriangleView<'_>,
    paid_factors: &DevelopmentFactors,
    incurred_factors: &DevelopmentFactors,
) -> Result<PaidIncurredReconciliation, ValidationError> {
    validate_compatible_triangles(paid, incurred)?;

    let paid_projection = ChainLadderEngine::project(paid, paid_factors)?;
    let incurred_projection = ChainLadderEngine::project(incurred, incurred_factors)?;
    Ok(build_reconciliation(paid_projection, incurred_projection))
}

/// Reconcile paid and incurred triangles using volume-weighted factors for each basis.
pub fn reconcile_volume_weighted(
    paid: &TriangleView<'_>,
    incurred: &TriangleView<'_>,
    tail: f64,
) -> Result<PaidIncurredReconciliation, ValidationError> {
    validate_compatible_triangles(paid, incurred)?;

    let paid_factors = DevelopmentFactorEngine::volume_weighted(paid, tail)?;
    let incurred_factors = DevelopmentFactorEngine::volume_weighted(incurred, tail)?;
    reconcile_with_factors(paid, incurred, &paid_factors, &incurred_factors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paid_triangle() -> Triangle {
        #[rustfmt::skip]
        let data = vec![
            1000.0, 1500.0, 1700.0, 1800.0,
            1200.0, 1800.0, 2000.0, f64::NAN,
            1500.0, 2100.0, f64::NAN, f64::NAN,
            1800.0, f64::NAN, f64::NAN, f64::NAN,
        ];
        Triangle::new(4, 4, data)
    }

    fn incurred_triangle() -> Triangle {
        #[rustfmt::skip]
        let data = vec![
            1200.0, 1650.0, 1785.0, 1840.0,
            1400.0, 1950.0, 2120.0, f64::NAN,
            1700.0, 2280.0, f64::NAN, f64::NAN,
            2000.0, f64::NAN, f64::NAN, f64::NAN,
        ];
        Triangle::new(4, 4, data)
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * (1.0 + b.abs())
    }

    #[test]
    fn case_and_ratio_triangles_are_derived_cellwise() {
        let paid = paid_triangle();
        let incurred = incurred_triangle();

        let case = case_triangle(&paid.view(), &incurred.view()).expect("case triangle");
        let ratio = paid_to_incurred_ratio_triangle(&paid.view(), &incurred.view())
            .expect("ratio triangle");

        assert!(close(case.data()[0], 200.0));
        assert!(close(case.data()[1], 150.0));
        assert!(close(ratio.data()[0], 1000.0 / 1200.0));
        assert!(ratio.data()[7].is_nan());
    }

    #[test]
    fn reconcile_volume_weighted_returns_origin_level_signals() {
        let paid = paid_triangle();
        let incurred = incurred_triangle();

        let reconciliation =
            reconcile_volume_weighted(&paid.view(), &incurred.view(), 1.0).expect("reconciliation");

        assert_eq!(reconciliation.paid_latest.len(), 4);
        assert_eq!(reconciliation.incurred_latest.len(), 4);
        assert!(close(
            reconciliation.case_reserve_latest[0],
            reconciliation.incurred_latest[0] - reconciliation.paid_latest[0]
        ));
        assert!(
            reconciliation
                .paid_to_incurred_ratio
                .iter()
                .all(|value| value.is_finite())
        );
        assert_eq!(
            reconciliation.ultimate_gap,
            reconciliation.case_adequacy_signal
        );
    }

    #[test]
    fn reconcile_rejects_shape_mismatch() {
        let paid = paid_triangle();
        let incurred = Triangle::new(
            3,
            3,
            vec![1.0, 2.0, 3.0, 2.0, 3.0, f64::NAN, 3.0, f64::NAN, f64::NAN],
        );

        assert_eq!(
            reconcile_volume_weighted(&paid.view(), &incurred.view(), 1.0),
            Err(ValidationError::ShapeMismatch)
        );
    }
}
