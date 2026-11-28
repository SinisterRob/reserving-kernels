//! Calendar-diagonal residual diagnostics.
//!
//! Standardized chain-ladder residuals `r_{i,j}` are bucketed by calendar
//! diagonal `d = i + j`. Under the no-calendar-effect null the residuals are
//! approximately independent `N(0, 1)`, so each diagonal mean `m_d` over
//! `n_d` cells satisfies `√n_d · m_d ≈ N(0, 1)` and
//!
//! ```text
//! Q = Σ_d n_d · m_d²  ≈  χ²_K,   K = number of diagonals with residuals
//! ```
//!
//! The reported `p_value = P(χ²_K > Q)` is an approximate (asymptotic) test:
//! residuals share plug-in `σ_j` estimates and are weakly dependent, so treat
//! it as a screening statistic. For the exact non-parametric alternative see
//! [`crate::diagnostics::mack_tests::calendar_year_effect_test`].

use crate::diagnostics::actual_expected::{residual_scale_by_dev, standardized_residuals};
use crate::special::chi_square_survival;
use crate::types::TriangleView;

/// Calendar-year / diagonal distortion diagnostics.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CalendarEffectDiagnostics {
    /// Mean standardized residual score by calendar diagonal index `origin + dev`.
    pub calendar_scores: Vec<f64>,
    /// Calendar diagonal with the largest absolute score.
    pub strongest_calendar: Option<usize>,
    /// Root-mean-square diagonal bias magnitude.
    pub diagonal_bias_score: f64,
    /// Chi-square statistic `Q = Σ_d n_d · m_d²` over diagonals with residuals.
    pub chi_square_statistic: f64,
    /// Degrees of freedom `K` (diagonals contributing to `Q`).
    pub degrees_of_freedom: usize,
    /// Approximate p-value `P(χ²_K > Q)`; small values indicate calendar-year
    /// effects. `NaN` when no standardized residual is available.
    pub p_value: f64,
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
            chi_square_statistic: f64::NAN,
            degrees_of_freedom: 0,
            p_value: f64::NAN,
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
    let mut chi_square_statistic = 0.0;
    let mut degrees_of_freedom = 0_usize;
    for bucket in buckets.iter().filter(|bucket| !bucket.is_empty()) {
        let m = mean(bucket);
        chi_square_statistic += bucket.len() as f64 * m * m;
        degrees_of_freedom += 1;
    }
    let (chi_square_statistic, p_value) = if degrees_of_freedom == 0 {
        (f64::NAN, f64::NAN)
    } else {
        (
            chi_square_statistic,
            chi_square_survival(chi_square_statistic, degrees_of_freedom),
        )
    };

    CalendarEffectDiagnostics {
        calendar_scores,
        strongest_calendar,
        diagonal_bias_score,
        chi_square_statistic,
        degrees_of_freedom,
        p_value,
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
        assert!((0.0..=1.0).contains(&diagnostics.p_value));
        assert!(diagnostics.degrees_of_freedom > 0);
        assert!(diagnostics.strongest_calendar.is_some());
    }

    #[test]
    fn p_value_is_chi_square_tail_of_diagonal_means() {
        let tri = triangle_4x4();
        let d = diagnose(&tri.view(), &[1.5, 1.2, 1.1]);
        let sigma = residual_scale_by_dev(&tri.view(), &[1.5, 1.2, 1.1]);
        let z = standardized_residuals(&tri.view(), &[1.5, 1.2, 1.1], &sigma);
        let mut q = 0.0;
        let mut k = 0;
        for cal in 0_usize..7 {
            let cells: Vec<f64> = (0..4)
                .filter_map(|i| {
                    let j = cal.checked_sub(i)?;
                    ((1..4).contains(&j) && i + j < 4).then(|| z[i * 4 + j])
                })
                .filter(|v| v.is_finite())
                .collect();
            if !cells.is_empty() {
                let m = cells.iter().sum::<f64>() / cells.len() as f64;
                q += cells.len() as f64 * m * m;
                k += 1;
            }
        }
        assert_eq!(d.degrees_of_freedom, k);
        assert!((d.chi_square_statistic - q).abs() < 1e-12);
        assert!((d.p_value - chi_square_survival(q, k)).abs() < 1e-15);
    }

    fn shocked_triangle(shocked: bool) -> Triangle {
        // 8x8 triangle with base development plus small deterministic jitter;
        // optionally a +25% shock on the last two calendar diagonals and a
        // −10% shock on early diagonals.
        let n = 8usize;
        let base = [2.0, 1.5, 1.25, 1.15, 1.10, 1.05, 1.02];
        let mut data = vec![f64::NAN; n * n];
        for i in 0..n {
            let mut c = 1000.0 + 37.0 * i as f64;
            data[i * n] = c;
            for j in 1..(n - i) {
                let cal = i + j;
                let jitter = 1.0 + 0.01 * ((i * 7 + j * 3) % 5) as f64;
                let shock = match (shocked, cal) {
                    (true, c) if c >= n - 2 => 1.25,
                    (true, c) if c <= 3 => 0.90,
                    _ => 1.0,
                };
                c *= base[j - 1] * shock * jitter;
                data[i * n + j] = c;
            }
        }
        Triangle::new(n, n, data)
    }

    #[test]
    fn calendar_shock_lowers_p_value_below_five_percent() {
        let p_value = |tri: &Triangle| {
            let ldfs = crate::deterministic::ldf::volume_weighted(&tri.view());
            diagnose(&tri.view(), &ldfs).p_value
        };
        let shocked = p_value(&shocked_triangle(true));
        let clean = p_value(&shocked_triangle(false));
        assert!(shocked < 0.05, "shocked p={shocked}");
        assert!(clean > shocked, "clean p={clean} shocked p={shocked}");
    }
}
