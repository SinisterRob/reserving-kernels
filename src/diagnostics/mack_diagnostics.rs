//! Mack variance decomposition diagnostics.

use crate::preprocessing::triangle::latest_diagonal;
use crate::stochastic::mack::{self, Mack};
use crate::types::TriangleView;
use crate::validate::ValidationError;

/// Per-origin Mack reserve variance decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct MackOriginVariance {
    /// Origin row index.
    pub origin: usize,
    /// Mack process variance contribution.
    pub process_variance: f64,
    /// Mack parameter variance contribution.
    pub parameter_variance: f64,
    /// Total origin variance.
    pub total_variance: f64,
    /// Standard error, `sqrt(total_variance)`.
    pub standard_error: f64,
    /// Coefficient of variation against reserve.
    pub coefficient_of_variation: f64,
}

/// Full Mack diagnostic bundle.
#[derive(Debug, Clone, PartialEq)]
pub struct MackDiagnostics {
    /// Mack sigma-squared by development age.
    pub sigma_by_age: Vec<f64>,
    /// Per-origin reserve variance decomposition.
    pub origin_variance: Vec<MackOriginVariance>,
    /// Sum of per-origin process variance.
    pub total_process_variance: f64,
    /// Sum of per-origin parameter variance.
    pub total_parameter_variance: f64,
    /// Total aggregate variance from Mack, including covariance terms.
    pub total_variance: f64,
    /// Aggregate standard error.
    pub total_standard_error: f64,
    /// Aggregate coefficient of variation.
    pub total_coefficient_of_variation: f64,
}

fn project_cumulative(latest: &[f64], ldf: &[f64]) -> Vec<f64> {
    let n = latest.len();
    let mut projected = vec![0.0; n * n];
    for origin in 0..n {
        let start = n - 1 - origin;
        projected[origin * n + start] = latest[origin];
        for dev in (start + 1)..n {
            projected[origin * n + dev] = projected[origin * n + dev - 1] * ldf[dev - 1];
        }
    }
    projected
}

fn column_sums(t: &TriangleView<'_>) -> Vec<f64> {
    let n = t.n_origins.min(t.n_devs);
    let mut sums = vec![0.0; n.saturating_sub(1)];
    for dev in 0..n.saturating_sub(1) {
        let observed = n - dev - 1;
        let mut acc = 0.0;
        for origin in 0..observed {
            acc += t.get(origin, dev);
        }
        sums[dev] = acc;
    }
    sums
}

fn decompose_origin_variance(t: &TriangleView<'_>, mack: &Mack) -> Vec<MackOriginVariance> {
    let n = t.n_origins.min(t.n_devs);
    let latest = latest_diagonal(t);
    let projected = project_cumulative(&latest, &mack.ldf);
    let col_sums = column_sums(t);

    let mut rows = Vec::with_capacity(n);
    for origin in 0..n {
        let start = n - 1 - origin;
        let ultimate = mack.ultimate[origin];
        let reserve = mack.reserve[origin];
        let mut process_term = 0.0;
        let mut parameter_term = 0.0;

        for dev in start..n.saturating_sub(1) {
            let fk = mack.ldf[dev];
            let sigma_sq = mack.sigma_sq[dev];
            let c_hat = projected[origin * n + dev];
            let sum = col_sums[dev];
            if fk > 0.0 && c_hat > 0.0 && sum > 0.0 {
                let scale = ultimate * ultimate * sigma_sq / (fk * fk);
                process_term += scale * (1.0 / c_hat);
                parameter_term += scale * (1.0 / sum);
            }
        }

        let total_variance = process_term + parameter_term;
        let standard_error = total_variance.max(0.0).sqrt();
        let coefficient_of_variation = if reserve > 0.0 {
            standard_error / reserve
        } else {
            0.0
        };

        rows.push(MackOriginVariance {
            origin,
            process_variance: process_term,
            parameter_variance: parameter_term,
            total_variance,
            standard_error,
            coefficient_of_variation,
        });
    }
    rows
}

/// Mack sigma-squared by development age.
pub fn mack_sigma_by_age(t: &TriangleView<'_>, tail: f64) -> Result<Vec<f64>, ValidationError> {
    Ok(mack::analyze(t, tail)?.sigma_sq)
}

/// Per-origin Mack process variance.
pub fn mack_process_variance(t: &TriangleView<'_>, tail: f64) -> Result<Vec<f64>, ValidationError> {
    let diagnostics = diagnose(t, tail)?;
    Ok(diagnostics
        .origin_variance
        .into_iter()
        .map(|row| row.process_variance)
        .collect())
}

/// Per-origin Mack parameter variance.
pub fn mack_parameter_variance(
    t: &TriangleView<'_>,
    tail: f64,
) -> Result<Vec<f64>, ValidationError> {
    let diagnostics = diagnose(t, tail)?;
    Ok(diagnostics
        .origin_variance
        .into_iter()
        .map(|row| row.parameter_variance)
        .collect())
}

/// Per-origin Mack total variance.
pub fn mack_total_variance(t: &TriangleView<'_>, tail: f64) -> Result<Vec<f64>, ValidationError> {
    let diagnostics = diagnose(t, tail)?;
    Ok(diagnostics
        .origin_variance
        .into_iter()
        .map(|row| row.total_variance)
        .collect())
}

/// Per-origin Mack reserve coefficient of variation.
pub fn mack_reserve_cv(t: &TriangleView<'_>, tail: f64) -> Result<Vec<f64>, ValidationError> {
    Ok(mack::analyze(t, tail)?.cv)
}

/// Full Mack variance decomposition diagnostics.
pub fn diagnose(t: &TriangleView<'_>, tail: f64) -> Result<MackDiagnostics, ValidationError> {
    let result = mack::analyze(t, tail)?;
    let origin_variance = decompose_origin_variance(t, &result);
    let total_process_variance = origin_variance.iter().map(|row| row.process_variance).sum();
    let total_parameter_variance = origin_variance
        .iter()
        .map(|row| row.parameter_variance)
        .sum();

    Ok(MackDiagnostics {
        sigma_by_age: result.sigma_sq.clone(),
        origin_variance,
        total_process_variance,
        total_parameter_variance,
        total_variance: result.total_mse,
        total_standard_error: result.total_standard_error,
        total_coefficient_of_variation: result.total_cv,
    })
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
        (a - b).abs() <= 1e-9 * (1.0 + b.abs())
    }

    #[test]
    fn diagnose_splits_origin_variance() {
        let triangle = triangle_4x4();
        let diagnostics = diagnose(&triangle.view(), 1.0).expect("mack diagnostics");

        assert_eq!(diagnostics.sigma_by_age.len(), 3);
        assert_eq!(diagnostics.origin_variance.len(), 4);
        assert!(diagnostics.total_variance >= diagnostics.total_process_variance);
        assert!(diagnostics.total_variance >= diagnostics.total_parameter_variance);

        for row in &diagnostics.origin_variance {
            assert!(close(
                row.total_variance,
                row.process_variance + row.parameter_variance
            ));
            assert!(close(
                row.standard_error,
                row.total_variance.max(0.0).sqrt()
            ));
        }
    }

    #[test]
    fn helper_vectors_match_full_diagnostics() {
        let triangle = triangle_4x4();
        let diagnostics = diagnose(&triangle.view(), 1.0).expect("mack diagnostics");

        assert_eq!(
            mack_sigma_by_age(&triangle.view(), 1.0).expect("sigma"),
            diagnostics.sigma_by_age
        );
        assert_eq!(
            mack_process_variance(&triangle.view(), 1.0).expect("process"),
            diagnostics
                .origin_variance
                .iter()
                .map(|row| row.process_variance)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            mack_parameter_variance(&triangle.view(), 1.0).expect("parameter"),
            diagnostics
                .origin_variance
                .iter()
                .map(|row| row.parameter_variance)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            mack_total_variance(&triangle.view(), 1.0).expect("total variance"),
            diagnostics
                .origin_variance
                .iter()
                .map(|row| row.total_variance)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn reserve_cv_matches_mack_engine() {
        let triangle = triangle_4x4();
        let reserve_cv = mack_reserve_cv(&triangle.view(), 1.0).expect("reserve cv");
        let diagnostics = diagnose(&triangle.view(), 1.0).expect("mack diagnostics");
        let diagnostic_cv = diagnostics
            .origin_variance
            .iter()
            .map(|row| row.coefficient_of_variation)
            .collect::<Vec<_>>();

        assert_eq!(reserve_cv.len(), diagnostic_cv.len());
        for (left, right) in reserve_cv.iter().zip(diagnostic_cv.iter()) {
            assert!(close(*left, *right));
        }
    }
}
