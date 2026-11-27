//! Link-ratio diagnostics and fragility measures.

use crate::ldf::LdfSelectionMethod;
use crate::triangle::age_to_age;
use crate::types::TriangleView;

#[derive(Debug, Clone, Copy, PartialEq)]
struct LinkRatioObservation {
    origin: usize,
    ratio: f64,
    current: f64,
    next: f64,
}

/// Diagnostics for one development-age link ratio selection.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkRatioDiagnostics {
    /// Development link start age `j` for `j -> j + 1`.
    pub dev_age: usize,
    /// Factor selected by the requested method.
    pub selected_factor: f64,
    /// Simple average of individual observed ratios.
    pub simple_average: f64,
    /// Volume-weighted link ratio.
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
    /// Number of observed ratios.
    pub n_obs: usize,
    /// Origin with the largest leave-one-out delta versus the selected factor.
    pub high_leverage_origin: Option<usize>,
    /// Leave-one-out selected factors in ascending observed-origin order.
    pub leave_one_out_factors: Vec<f64>,
}

fn observations_for_dev(t: &TriangleView<'_>, dev: usize) -> Vec<LinkRatioObservation> {
    let n = t.n_origins.min(t.n_devs);
    if dev + 1 >= n {
        return Vec::new();
    }

    let mut out = Vec::new();
    for origin in 0..n - dev - 1 {
        let current = t.get(origin, dev);
        let next = t.get(origin, dev + 1);
        let ratio = age_to_age(t, origin, dev);
        if current.is_finite() && next.is_finite() && ratio.is_finite() {
            out.push(LinkRatioObservation {
                origin,
                ratio,
                current,
                next,
            });
        }
    }
    out
}

fn mean_ratio(observations: &[LinkRatioObservation]) -> f64 {
    if observations.is_empty() {
        return f64::NAN;
    }
    observations.iter().map(|obs| obs.ratio).sum::<f64>() / observations.len() as f64
}

fn volume_weighted_ratio(observations: &[LinkRatioObservation]) -> f64 {
    let denominator: f64 = observations.iter().map(|obs| obs.current).sum();
    if denominator <= 0.0 {
        return f64::NAN;
    }
    let numerator: f64 = observations.iter().map(|obs| obs.next).sum();
    numerator / denominator
}

fn median_ratio(observations: &[LinkRatioObservation]) -> f64 {
    if observations.is_empty() {
        return f64::NAN;
    }
    let mut ratios: Vec<f64> = observations.iter().map(|obs| obs.ratio).collect();
    ratios.sort_by(f64::total_cmp);
    let n = ratios.len();
    if n % 2 == 1 {
        ratios[n / 2]
    } else {
        f64::midpoint(ratios[n / 2 - 1], ratios[n / 2])
    }
}

fn ex_hi_lo_ratio(observations: &[LinkRatioObservation]) -> f64 {
    if observations.len() < 3 {
        return mean_ratio(observations);
    }
    let mut ratios: Vec<f64> = observations.iter().map(|obs| obs.ratio).collect();
    ratios.sort_by(f64::total_cmp);
    let trimmed = &ratios[1..ratios.len() - 1];
    trimmed.iter().sum::<f64>() / trimmed.len() as f64
}

fn sample_std_dev(observations: &[LinkRatioObservation], mean: f64) -> f64 {
    if observations.len() <= 1 || !mean.is_finite() {
        return 0.0;
    }
    let variance = observations
        .iter()
        .map(|obs| (obs.ratio - mean).powi(2))
        .sum::<f64>()
        / (observations.len() as f64 - 1.0);
    variance.sqrt()
}

fn recent_window(observations: &[LinkRatioObservation], last_n: usize) -> &[LinkRatioObservation] {
    if observations.is_empty() {
        return observations;
    }
    let take = if last_n == 0 {
        observations.len()
    } else {
        observations.len().min(last_n)
    };
    &observations[observations.len() - take..]
}

fn select_factor(method: LdfSelectionMethod, observations: &[LinkRatioObservation]) -> f64 {
    match method {
        // `Selected` currently carries no per-link selected values, so the
        // closest auditable baseline is the objective volume-weighted factor.
        LdfSelectionMethod::VolumeWeighted | LdfSelectionMethod::Selected => {
            volume_weighted_ratio(observations)
        }
        LdfSelectionMethod::SimpleAverage => mean_ratio(observations),
        LdfSelectionMethod::Median => median_ratio(observations),
        LdfSelectionMethod::ExHiLo => ex_hi_lo_ratio(observations),
        LdfSelectionMethod::LastNVolumeWeighted { last_n } => {
            volume_weighted_ratio(recent_window(observations, last_n))
        }
        LdfSelectionMethod::LastNSimpleAverage { last_n } => {
            mean_ratio(recent_window(observations, last_n))
        }
    }
}

/// Diagnose one development-age link for a requested selection method.
pub fn diagnose_link(
    t: &TriangleView<'_>,
    dev_age: usize,
    method: LdfSelectionMethod,
) -> LinkRatioDiagnostics {
    let observations = observations_for_dev(t, dev_age);
    let simple_average = mean_ratio(&observations);
    let volume_weighted = volume_weighted_ratio(&observations);
    let median = median_ratio(&observations);
    let selected_factor = select_factor(method, &observations);
    let min = observations
        .iter()
        .map(|obs| obs.ratio)
        .min_by(f64::total_cmp)
        .unwrap_or(f64::NAN);
    let max = observations
        .iter()
        .map(|obs| obs.ratio)
        .max_by(f64::total_cmp)
        .unwrap_or(f64::NAN);
    let std_dev = sample_std_dev(&observations, simple_average);
    let coeff_var = if simple_average.abs() > 0.0 {
        std_dev / simple_average.abs()
    } else {
        f64::NAN
    };

    let mut high_leverage_origin = None;
    let mut max_delta = f64::NEG_INFINITY;
    let mut leave_one_out_factors = Vec::with_capacity(observations.len());
    for idx in 0..observations.len() {
        let mut leave_one_out = observations.clone();
        let removed = leave_one_out.remove(idx);
        let factor = select_factor(method, &leave_one_out);
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
        n_obs: observations.len(),
        high_leverage_origin,
        leave_one_out_factors,
    }
}

/// Diagnose all development-age links for a requested selection method.
pub fn diagnose_all(t: &TriangleView<'_>, method: LdfSelectionMethod) -> Vec<LinkRatioDiagnostics> {
    let n = t.n_origins.min(t.n_devs);
    if n < 2 {
        return Vec::new();
    }
    (0..n - 1)
        .map(|dev_age| diagnose_link(t, dev_age, method))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn diagnose_link_reports_core_statistics() {
        let tri = triangle_4x4();
        let diagnostics = diagnose_link(&tri.view(), 0, LdfSelectionMethod::VolumeWeighted);

        assert_eq!(diagnostics.dev_age, 0);
        assert_eq!(diagnostics.n_obs, 3);
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
        let diagnostics = diagnose_link(&tri.view(), 0, LdfSelectionMethod::VolumeWeighted);

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
        let simple = diagnose_link(&tri.view(), 0, LdfSelectionMethod::SimpleAverage);
        let volume = diagnose_link(&tri.view(), 0, LdfSelectionMethod::VolumeWeighted);

        assert!(simple.selected_factor.is_finite());
        assert!(volume.selected_factor.is_finite());
        assert_ne!(simple.selected_factor, volume.selected_factor);
    }
}
