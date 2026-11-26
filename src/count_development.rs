//! Claim-count development kernels.

use crate::chain_ladder::{ChainLadderEngine, ChainLadderProjection};
use crate::ldf::{DevelopmentFactorEngine, DevelopmentFactors};
use crate::types::{Triangle, TriangleView};
use crate::validate::{ValidationError, validate_triangle};

/// Open-count triangle `reported - closed`, preserving `NaN` where either cell is missing
/// or outside the scheduled observed region.
pub fn open_count_triangle(
    reported: &TriangleView<'_>,
    closed: &TriangleView<'_>,
) -> Result<Triangle, ValidationError> {
    validate_triangle(reported)?;
    validate_triangle(closed)?;
    if reported.n_origins != closed.n_origins || reported.n_devs != closed.n_devs {
        return Err(ValidationError::ShapeMismatch);
    }
    let mut data = vec![f64::NAN; reported.n_origins * reported.n_devs];
    for origin in 0..reported.n_origins {
        for dev in 0..reported.n_devs {
            if !reported.is_scheduled_observed(origin, dev) {
                continue;
            }
            let reported_value = reported.get(origin, dev);
            let closed_value = closed.get(origin, dev);
            if reported_value.is_finite() && closed_value.is_finite() {
                data[origin * reported.n_devs + dev] = (reported_value - closed_value).max(0.0);
            }
        }
    }
    Ok(Triangle::new(reported.n_origins, reported.n_devs, data))
}

/// Development factors for a reported-count triangle.
pub fn reported_count_ldf(
    reported: &TriangleView<'_>,
    tail: f64,
) -> Result<DevelopmentFactors, ValidationError> {
    DevelopmentFactorEngine::volume_weighted(reported, tail)
}

/// Development factors for a closed-count triangle.
pub fn closed_count_ldf(
    closed: &TriangleView<'_>,
    tail: f64,
) -> Result<DevelopmentFactors, ValidationError> {
    DevelopmentFactorEngine::volume_weighted(closed, tail)
}

/// Ultimate claim counts by origin from a cumulative count triangle.
pub fn ultimate_claim_count(
    counts: &TriangleView<'_>,
    factors: &DevelopmentFactors,
) -> Result<Vec<f64>, ValidationError> {
    Ok(ChainLadderEngine::project(counts, factors)?.ultimates)
}

/// IBNR claim counts by origin from a cumulative reported count triangle.
pub fn ibnr_claim_count(
    reported: &TriangleView<'_>,
    factors: &DevelopmentFactors,
) -> Result<Vec<f64>, ValidationError> {
    Ok(ChainLadderEngine::project(reported, factors)?.reserves)
}

/// Convenience wrapper for count chain-ladder projection with volume-weighted factors.
pub fn project_volume_weighted(
    counts: &TriangleView<'_>,
    tail: f64,
) -> Result<ChainLadderProjection, ValidationError> {
    let factors = DevelopmentFactorEngine::volume_weighted(counts, tail)?;
    ChainLadderEngine::project(counts, &factors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reported_triangle() -> Triangle {
        #[rustfmt::skip]
        let data = vec![
            100.0, 140.0, 150.0, 155.0,
            110.0, 145.0, 153.0, f64::NAN,
            120.0, 150.0, f64::NAN, f64::NAN,
            130.0, f64::NAN, f64::NAN, f64::NAN,
        ];
        Triangle::new(4, 4, data)
    }

    fn closed_triangle() -> Triangle {
        #[rustfmt::skip]
        let data = vec![
            80.0, 120.0, 140.0, 150.0,
            90.0, 125.0, 142.0, f64::NAN,
            100.0, 130.0, f64::NAN, f64::NAN,
            110.0, f64::NAN, f64::NAN, f64::NAN,
        ];
        Triangle::new(4, 4, data)
    }

    #[test]
    fn builds_open_count_triangle() {
        let open = open_count_triangle(&reported_triangle().view(), &closed_triangle().view())
            .expect("open");
        assert_eq!(open.data[0], 20.0);
        assert_eq!(open.data[1], 20.0);
    }

    #[test]
    fn count_projection_returns_ultimate_and_ibnr() {
        let reported = reported_triangle();
        let factors = reported_count_ldf(&reported.view(), 1.0).expect("factors");
        let ultimate = ultimate_claim_count(&reported.view(), &factors).expect("ultimate");
        let ibnr = ibnr_claim_count(&reported.view(), &factors).expect("ibnr");

        assert_eq!(ultimate.len(), 4);
        assert_eq!(ibnr.len(), 4);
        assert!(ultimate[3] >= reported.data[12]);
        assert!(ibnr[0] >= 0.0);
    }
}
