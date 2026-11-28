//! Link-ratio diagnostics and fragility measures.
//!
//! All factor computations delegate to the shared averaging kernels used by
//! [`crate::deterministic::ldf`], so `volume_weighted` here is bit-identical to
//! [`crate::deterministic::ldf::DevelopmentFactorEngine::volume_weighted`]
//! (rows with `C_{i,j} = 0` contribute their `C_{i,j+1}` mass, exactly like
//! the engine).

use crate::deterministic::averaging::{self, LinkPair};
use crate::deterministic::ldf::LdfSelectionMethod;
use crate::error::ReservingError;
use crate::types::TriangleView;

/// Diagnostics for one development-age link ratio selection.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct LinkRatioDiagnostics {
    /// Development link start age `j` for `j -> j + 1`.
    pub dev_age: usize,
    /// Factor selected by the requested method (or the caller-selected factor).
    pub selected_factor: f64,
    /// Simple average of individual observed ratios.
    pub simple_average: f64,
    /// Volume-weighted link ratio (same definition as the LDF engine).
    pub volume_weighted: f64,
    /// Median observed ratio.
    pub median: f64,
    /// Minimum observed ratio.
    pub min: f64,
    /// Maximum observed ratio.
    pub max: f64,
    /// Sample standard deviation of observed ratios.
    pub std_dev: f64,
    /// Coefficient of variation of observed ratios.
    pub coeff_var: f64,
    /// Number of defined individual ratios (`C_{i,j} > 0`).
    pub n_obs: usize,
    /// Number of observed origin rows feeding the link (including rows with
    /// `C_{i,j} = 0`, which still enter the volume-weighted factor).
    pub n_pairs: usize,
    /// Origin with the largest non-zero leave-one-out delta versus the
    /// selected factor (`None` when no removal moves the factor).
    pub high_leverage_origin: Option<usize>,
    /// Leave-one-out selected factors, one per observed origin row in
    /// ascending origin order (length `n_pairs`).
    pub leave_one_out_factors: Vec<f64>,
}

fn method_selector(
    method: LdfSelectionMethod,
) -> Result<impl Fn(&[LinkPair]) -> f64, ReservingError> {
    let selector = move |pairs: &[LinkPair]| match method {
        LdfSelectionMethod::VolumeWeighted => averaging::volume_weighted(pairs),
        LdfSelectionMethod::SimpleAverage => averaging::simple_average(pairs),
        LdfSelectionMethod::Median => averaging::median(pairs),
        LdfSelectionMethod::ExHiLo => averaging::ex_hi_lo(pairs),
        LdfSelectionMethod::Geometric => averaging::geometric(pairs),
        LdfSelectionMethod::LastNVolumeWeighted { last_n } => {
            averaging::volume_weighted(averaging::recent(pairs, last_n))
        }
        LdfSelectionMethod::LastNSimpleAverage { last_n } => {
            averaging::simple_average(averaging::recent(pairs, last_n))
        }
        LdfSelectionMethod::Selected => f64::NAN,
    };
    if method == LdfSelectionMethod::Selected {
        return Err(ReservingError::InvalidParameter(
            "LdfSelectionMethod::Selected has no computable factor; use diagnose_link_selected / diagnose_all_selected",
        ));
    }
    Ok(selector)
}

fn diagnose_with(
    t: &TriangleView<'_>,
    dev_age: usize,
    select: impl Fn(&[LinkPair]) -> f64,
) -> LinkRatioDiagnostics {
    let pairs = averaging::link_pairs(t, dev_age);
    let ratios = averaging::ratios(&pairs);
    let simple_average = averaging::simple_average(&pairs);
    let volume_weighted = averaging::volume_weighted(&pairs);
    let median = averaging::median(&pairs);
    let selected_factor = select(&pairs);
    let min = ratios
        .iter()
        .copied()
        .min_by(f64::total_cmp)
        .unwrap_or(f64::NAN);
    let max = ratios
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .unwrap_or(f64::NAN);
    let std_dev = averaging::ratio_sample_std_dev(&pairs, simple_average);
    let coeff_var = if simple_average.abs() > 0.0 {
        std_dev / simple_average.abs()
    } else {
        f64::NAN
    };

    let mut high_leverage_origin = None;
    let mut max_delta = 0.0;
    let mut leave_one_out_factors = Vec::with_capacity(pairs.len());
    let mut scratch = Vec::with_capacity(pairs.len());
    for (idx, removed) in pairs.iter().enumerate() {
        scratch.clear();
        scratch.extend(
            pairs
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != idx)
                .map(|(_, p)| *p),
        );
        let factor = select(&scratch);
        let delta = (selected_factor - factor).abs();
        if delta.is_finite() && delta > max_delta {
            max_delta = delta;
            high_leverage_origin = Some(removed.origin);
        }
        leave_one_out_factors.push(factor);
    }

    LinkRatioDiagnostics {
        dev_age,
        selected_factor,
        simple_average,
        volume_weighted,
        median,
        min,
        max,
        std_dev,
        coeff_var,
        n_obs: ratios.len(),
        n_pairs: pairs.len(),
        high_leverage_origin,
        leave_one_out_factors,
    }
}

/// Diagnose one development-age link for a computed selection method.
///
/// Returns `InvalidParameter` for [`LdfSelectionMethod::Selected`] (it carries
/// no factor values); use [`diagnose_link_selected`] with the explicit factor.
pub fn diagnose_link(
    t: &TriangleView<'_>,
    dev_age: usize,
    method: LdfSelectionMethod,
) -> Result<LinkRatioDiagnostics, ReservingError> {
    let select = method_selector(method)?;
    Ok(diagnose_with(t, dev_age, select))
}

/// Diagnose one development-age link against an explicit caller-selected factor.
///
/// Leave-one-out factors equal `selected_factor` (a judgmental selection does
/// not move when one origin is removed), so `high_leverage_origin` is `None`.
pub fn diagnose_link_selected(
    t: &TriangleView<'_>,
    dev_age: usize,
    selected_factor: f64,
) -> LinkRatioDiagnostics {
    diagnose_with(t, dev_age, |_| selected_factor)
}

/// Diagnose all development-age links for a computed selection method.
///
/// Returns `InvalidParameter` for [`LdfSelectionMethod::Selected`]; use
/// [`diagnose_all_selected`].
pub fn diagnose_all(
    t: &TriangleView<'_>,
    method: LdfSelectionMethod,
) -> Result<Vec<LinkRatioDiagnostics>, ReservingError> {
    let select = method_selector(method)?;
    let n = t.n_origins.min(t.n_devs);
    if n < 2 {
        return Ok(Vec::new());
    }
    Ok((0..n - 1)
        .map(|dev_age| diagnose_with(t, dev_age, &select))
        .collect())
}

/// Diagnose all development-age links against explicit caller-selected
/// factors (`selected.len()` must equal `min(n_origins, n_devs) - 1`).
pub fn diagnose_all_selected(
    t: &TriangleView<'_>,
    selected: &[f64],
) -> Result<Vec<LinkRatioDiagnostics>, ReservingError> {
    let n = t.n_origins.min(t.n_devs);
    if selected.len() != n.saturating_sub(1) {
        return Err(ReservingError::ShapeMismatch);
    }
    Ok(selected
        .iter()
        .enumerate()
        .map(|(dev_age, &factor)| diagnose_link_selected(t, dev_age, factor))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deterministic::ldf::DevelopmentFactorEngine;
    use crate::types::Triangle;

    fn triangle_4x4() -> Triangle {
        #[rustfmt::skip]
        let data = vec![
            1000.0, 1500.0, 1700.0, 1800.0,
            1200.0, 1800.0, 2000.0, f64::NAN,
            1500.0, 2100.0, f64::NAN, f64::NAN,
            1800.0, f64::NAN, f64::NAN, f64::NAN,
        ];
        Triangle::new(4, 4, data)
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * (1.0 + b.abs())
    }

    fn vw(t: &TriangleView<'_>, dev: usize) -> LinkRatioDiagnostics {
        diagnose_link(t, dev, LdfSelectionMethod::VolumeWeighted).expect("diagnose")
    }

    #[test]
    fn diagnose_link_reports_core_statistics() {
        let tri = triangle_4x4();
        let diagnostics = vw(&tri.view(), 0);

        assert_eq!(diagnostics.dev_age, 0);
        assert_eq!(diagnostics.n_obs, 3);
        assert_eq!(diagnostics.n_pairs, 3);
        assert!(close(diagnostics.simple_average, (1.5 + 1.5 + 1.4) / 3.0));
        assert!(close(diagnostics.volume_weighted, 5400.0 / 3700.0));
        assert!(close(diagnostics.median, 1.5));
        assert!(close(diagnostics.min, 1.4));
        assert!(close(diagnostics.max, 1.5));
        assert_eq!(diagnostics.leave_one_out_factors.len(), 3);
    }

    #[test]
    fn leave_one_out_highlights_high_leverage_origin() {
        let tri = triangle_4x4();
        let diagnostics = vw(&tri.view(), 0);

        assert_eq!(diagnostics.high_leverage_origin, Some(2));
        assert!(
            diagnostics
                .leave_one_out_factors
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn method_specific_selected_factor_changes_with_selector() {
        let tri = triangle_4x4();
        let simple =
            diagnose_link(&tri.view(), 0, LdfSelectionMethod::SimpleAverage).expect("simple");
        let volume = vw(&tri.view(), 0);

        assert!(simple.selected_factor.is_finite());
        assert!(volume.selected_factor.is_finite());
        assert_ne!(simple.selected_factor, volume.selected_factor);
    }

    #[test]
    fn zero_denominator_row_matches_engine_volume_weighted() {
        // Origin 2 has C_{2,0} = 0 -> 50: it has no individual ratio but its
        // numerator mass enters the classical volume-weighted factor.
        #[rustfmt::skip]
        let tri = Triangle::new(4, 4, vec![
            100.0, 150.0, 170.0, 180.0,
            120.0, 180.0, 200.0, f64::NAN,
              0.0,  50.0, f64::NAN, f64::NAN,
            140.0, f64::NAN, f64::NAN, f64::NAN,
        ]);
        let engine = DevelopmentFactorEngine::volume_weighted(&tri.view(), 1.0).expect("engine");
        let diag = diagnose_all(&tri.view(), LdfSelectionMethod::VolumeWeighted).expect("diag");
        for (d, &f) in diag.iter().zip(&engine.ldfs) {
            assert_eq!(d.volume_weighted.to_bits(), f.to_bits());
            assert_eq!(d.selected_factor.to_bits(), f.to_bits());
        }
        // (150 + 180 + 50) / (100 + 120 + 0) = 380 / 220, not 330 / 220.
        assert!(close(diag[0].volume_weighted, 380.0 / 220.0));
        assert_eq!(diag[0].n_obs, 2);
        assert_eq!(diag[0].n_pairs, 3);
        assert_eq!(diag[0].leave_one_out_factors.len(), 3);
        assert!(close(diag[0].leave_one_out_factors[2], 330.0 / 220.0));
    }

    #[test]
    fn selected_method_requires_explicit_factors() {
        let tri = triangle_4x4();
        assert!(matches!(
            diagnose_link(&tri.view(), 0, LdfSelectionMethod::Selected),
            Err(ReservingError::InvalidParameter(_))
        ));
        assert!(matches!(
            diagnose_all(&tri.view(), LdfSelectionMethod::Selected),
            Err(ReservingError::InvalidParameter(_))
        ));
        let manual = diagnose_all_selected(&tri.view(), &[1.45, 1.12, 1.05]).expect("manual");
        assert_eq!(manual.len(), 3);
        assert!(close(manual[0].selected_factor, 1.45));
        assert!(
            manual[0]
                .leave_one_out_factors
                .iter()
                .all(|&f| close(f, 1.45))
        );
        assert_eq!(manual[0].high_leverage_origin, None);
        assert_eq!(
            diagnose_all_selected(&tri.view(), &[1.45]),
            Err(ReservingError::ShapeMismatch)
        );
    }
}
