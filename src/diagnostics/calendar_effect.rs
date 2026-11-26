//! Calendar-diagonal residual diagnostics.

use crate::diagnostics::actual_expected::{residual_scale_by_dev, standardized_residuals};
use crate::types::TriangleView;

/// Calendar-year / diagonal distortion diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct CalendarEffectDiagnostics {
    /// Mean standardized residual score by calendar diagonal index `origin + dev`.
    pub calendar_scores: Vec<f64>,
    /// Calendar diagonal with the largest absolute score.
    pub strongest_calendar: Option<usize>,
    /// Root-mean-square diagonal bias magnitude.
    pub diagonal_bias_score: f64,
    /// Heuristic concentration score in `(0, 1]`; lower means more diagonal distortion.
    pub p_value_like_score: f64,
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Diagnose calendar / diagonal effects from standardized residuals.
pub fn diagnose(t: &TriangleView<'_>, ldfs: &[f64]) -> CalendarEffectDiagnostics {
    if t.n_origins == 0 || t.n_devs < 2 {
        return CalendarEffectDiagnostics {
            calendar_scores: Vec::new(),
            strongest_calendar: None,
            diagonal_bias_score: 0.0,
            p_value_like_score: 1.0,
        };
    }
    let sigma = residual_scale_by_dev(t, ldfs);
    let standardized = standardized_residuals(t, ldfs, &sigma);
    let max_calendar = t.n_origins + t.n_devs - 2;
    let mut buckets = vec![Vec::new(); max_calendar + 1];

    for origin in 0..t.n_origins {
        for dev in 1..t.n_devs {
            if !t.is_scheduled_observed(origin, dev) {
                continue;
            }
            let value = standardized[origin * t.n_devs + dev];
            if value.is_finite() {
                buckets[origin + dev].push(value);
            }
        }
    }

    let calendar_scores: Vec<f64> = buckets.iter().map(|bucket| mean(bucket)).collect();
    let strongest_calendar = calendar_scores
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
        .map(|(index, _)| index);
    let finite_scores: Vec<f64> = calendar_scores
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let diagonal_bias_score = if finite_scores.is_empty() {
        0.0
    } else {
        (finite_scores.iter().map(|value| value.powi(2)).sum::<f64>() / finite_scores.len() as f64)
            .sqrt()
    };
    let p_value_like_score = 1.0 / (1.0 + diagonal_bias_score.max(0.0));

    CalendarEffectDiagnostics {
        calendar_scores,
        strongest_calendar,
        diagonal_bias_score,
        p_value_like_score,
    }
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

    #[test]
    fn diagnoses_calendar_bias_surface() {
        let diagnostics = diagnose(&triangle_4x4().view(), &[1.5, 1.2, 1.1]);
        assert_eq!(diagnostics.calendar_scores.len(), 7);
        assert!(diagnostics.diagonal_bias_score >= 0.0);
        assert!(diagnostics.p_value_like_score > 0.0);
        assert!(diagnostics.strongest_calendar.is_some());
    }
}
