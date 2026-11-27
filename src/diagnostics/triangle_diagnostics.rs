//! Triangle sanity-check kernels.
//!
//! This module stays below method selection and reserve indication logic. It
//! answers basic structural questions about a row-major triangle before any
//! reserving method consumes it.

use crate::triangle;
use crate::types::TriangleView;

/// Flag emitted for a cell-level diagnostic condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellFlag {
    /// Origin row index.
    pub origin: usize,
    /// Development column index.
    pub dev: usize,
}

/// Flag emitted for a row-level diagnostic condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowFlag {
    /// Origin row index.
    pub origin: usize,
}

/// Summary diagnostics for one triangle.
#[derive(Debug, Clone, PartialEq)]
pub struct TriangleDiagnostics {
    /// Number of origin rows.
    pub n_origin: usize,
    /// Number of development columns.
    pub n_dev: usize,
    /// Count of finite observed cells inside the scheduled triangle.
    pub observed_count: usize,
    /// Count of missing cells (`NaN`) inside the scheduled triangle.
    pub missing_count: usize,
    /// Count of observed incremental cells below zero after cumulative differencing.
    pub negative_increment_count: usize,
    /// Count of origin rows with at least one cumulative decrease.
    pub non_monotone_row_count: usize,
    /// Count of origin rows whose latest observed value is exactly zero.
    pub zero_latest_count: usize,
    /// Count of origin rows with zero or one observed cell.
    pub sparse_origin_count: usize,
    /// Latest observed cumulative value by origin under scheduled triangle geometry.
    pub latest_diagonal: Vec<f64>,
}

/// Build a boolean mask for scheduled observed cells.
pub fn observed_mask(t: &TriangleView<'_>) -> Vec<bool> {
    let mut out = vec![false; t.n_origins * t.n_devs];
    for i in 0..t.n_origins {
        for j in 0..t.n_devs {
            out[i * t.n_devs + j] = t.is_scheduled_observed(i, j);
        }
    }
    out
}

/// Latest finite observed cumulative value by origin.
///
/// Returns `NaN` for rows with no finite scheduled observed cells.
pub fn latest_diagonal(t: &TriangleView<'_>) -> Vec<f64> {
    let ages = development_age_by_origin(t);
    ages.iter()
        .enumerate()
        .map(|(origin, &dev)| {
            if dev < t.n_devs {
                t.get(origin, dev)
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// Convert cumulative to incremental using scheduled triangle geometry.
pub fn incremental_from_cumulative(t: &TriangleView<'_>) -> Vec<f64> {
    triangle::cumulative_to_incremental(t)
}

/// Convert incremental to cumulative using scheduled triangle geometry.
pub fn cumulative_from_incremental(t: &TriangleView<'_>) -> Vec<f64> {
    triangle::incremental_to_cumulative(t)
}

/// Flag cells whose incremental emergence is negative.
pub fn negative_increment_flags(t: &TriangleView<'_>) -> Vec<CellFlag> {
    let mut flags = Vec::new();
    for i in 0..t.n_origins {
        for j in 0..t.n_devs {
            if !t.is_scheduled_observed(i, j) {
                continue;
            }
            let cur = t.get(i, j);
            if !cur.is_finite() {
                continue;
            }
            let inc = if j == 0 {
                cur
            } else {
                let prev = t.get(i, j - 1);
                if !prev.is_finite() {
                    continue;
                }
                cur - prev
            };
            if inc < 0.0 {
                flags.push(CellFlag { origin: i, dev: j });
            }
        }
    }
    flags
}

/// Flag origin rows containing at least one cumulative decrease.
pub fn monotonicity_flags(t: &TriangleView<'_>) -> Vec<RowFlag> {
    let mut flags = Vec::new();
    for origin in 0..t.n_origins {
        let mut previous: Option<f64> = None;
        let mut row_flagged = false;
        for dev in 0..t.n_devs {
            if !t.is_scheduled_observed(origin, dev) {
                continue;
            }
            let value = t.get(origin, dev);
            if !value.is_finite() {
                continue;
            }
            if let Some(prev) = previous
                && value < prev
            {
                row_flagged = true;
                break;
            }
            previous = Some(value);
        }
        if row_flagged {
            flags.push(RowFlag { origin });
        }
    }
    flags
}

/// Latest finite observed development age by origin.
///
/// For rows with no finite scheduled observed cells, returns `n_dev`.
pub fn development_age_by_origin(t: &TriangleView<'_>) -> Vec<usize> {
    let mut out = vec![t.n_devs; t.n_origins];
    for origin in 0..t.n_origins {
        let mut latest = None;
        for dev in 0..t.n_devs {
            if t.is_scheduled_observed(origin, dev) && t.get(origin, dev).is_finite() {
                latest = Some(dev);
            }
        }
        if let Some(dev) = latest {
            out[origin] = dev;
        }
    }
    out
}

/// Summarize the structural state of a triangle.
pub fn analyze(t: &TriangleView<'_>) -> TriangleDiagnostics {
    let mask = observed_mask(t);
    let latest = latest_diagonal(t);
    let negative_increment = negative_increment_flags(t);
    let monotonicity = monotonicity_flags(t);

    let mut observed_count = 0;
    let mut missing_count = 0;
    let mut sparse_origin_count = 0;

    for origin in 0..t.n_origins {
        let mut row_observed = 0;
        for dev in 0..t.n_devs {
            if !mask[origin * t.n_devs + dev] {
                continue;
            }
            let value = t.get(origin, dev);
            if value.is_finite() {
                observed_count += 1;
                row_observed += 1;
            } else {
                missing_count += 1;
            }
        }
        if row_observed <= 1 {
            sparse_origin_count += 1;
        }
    }

    let zero_latest_count = latest
        .iter()
        .filter(|value| value.is_finite() && **value == 0.0)
        .count();

    TriangleDiagnostics {
        n_origin: t.n_origins,
        n_dev: t.n_devs,
        observed_count,
        missing_count,
        negative_increment_count: negative_increment.len(),
        non_monotone_row_count: monotonicity.len(),
        zero_latest_count,
        sparse_origin_count,
        latest_diagonal: latest,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Triangle;

    fn triangle_4x4() -> Triangle {
        #[rustfmt::skip]
        let data = vec![
            100.0, 150.0, 170.0, 175.0,
            110.0, 165.0, 190.0, f64::NAN,
            120.0, 180.0, f64::NAN, f64::NAN,
            130.0, f64::NAN, f64::NAN, f64::NAN,
        ];
        Triangle::new(4, 4, data)
    }

    #[test]
    fn latest_and_development_age_follow_triangle_geometry() {
        let tri = triangle_4x4();

        assert_eq!(development_age_by_origin(&tri.view()), vec![3, 2, 1, 0]);
        assert_eq!(
            latest_diagonal(&tri.view()),
            vec![175.0, 190.0, 180.0, 130.0]
        );
    }

    #[test]
    fn observed_mask_marks_only_scheduled_cells() {
        let tri = triangle_4x4();
        let mask = observed_mask(&tri.view());

        assert_eq!(mask.len(), 16);
        assert!(mask[0]);
        assert!(mask[6]);
        assert!(!mask[7]);
        assert!(!mask[15]);
    }

    #[test]
    fn negative_increment_and_monotonicity_flags_detect_problems() {
        #[rustfmt::skip]
        let tri = Triangle::new(3, 3, vec![
            100.0, 90.0, 120.0,
            80.0, 100.0, f64::NAN,
            70.0, f64::NAN, f64::NAN,
        ]);

        assert_eq!(
            negative_increment_flags(&tri.view()),
            vec![CellFlag { origin: 0, dev: 1 }]
        );
        assert_eq!(monotonicity_flags(&tri.view()), vec![RowFlag { origin: 0 }]);
    }

    #[test]
    fn analyze_counts_missing_sparse_and_zero_latest_rows() {
        #[rustfmt::skip]
        let tri = Triangle::new(4, 4, vec![
            100.0, 150.0, 170.0, 0.0,
            110.0, 165.0, f64::NAN, f64::NAN,
            0.0, f64::NAN, f64::NAN, f64::NAN,
            130.0, f64::NAN, f64::NAN, f64::NAN,
        ]);

        let diagnostics = analyze(&tri.view());

        assert_eq!(diagnostics.observed_count, 8);
        assert_eq!(diagnostics.missing_count, 2);
        assert_eq!(diagnostics.zero_latest_count, 2);
        assert_eq!(diagnostics.sparse_origin_count, 2);
        assert_eq!(diagnostics.latest_diagonal, vec![0.0, 165.0, 0.0, 130.0]);
    }
}
