//! Expected-cell and residual diagnostics for cumulative chain-ladder style fits.

use crate::deterministic::ldf::mack_sigma_sq_raw_by_link;
use crate::types::TriangleView;

/// Cell-level actual-versus-expected diagnostic row.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct CellResidual {
    /// Origin row index.
    pub origin: usize,
    /// Development column index of the actual cell being diagnosed.
    pub dev: usize,
    /// Actual cumulative value at `(origin, dev)`.
    pub actual: f64,
    /// Expected cumulative value implied by the prior development column and selected LDF.
    pub expected: f64,
    /// Raw residual `actual - expected`.
    pub residual: f64,
    /// Actual-to-expected ratio `actual / expected`.
    pub actual_to_expected: f64,
    /// Standardized residual when a scale vector is available.
    pub standardized_residual: f64,
}

fn scheduled_pair_observed(t: &TriangleView<'_>, origin: usize, dev: usize) -> bool {
    dev > 0
        && origin < t.n_origins
        && dev < t.n_devs
        && t.is_scheduled_observed(origin, dev)
        && t.is_scheduled_observed(origin, dev - 1)
        && t.get(origin, dev).is_finite()
        && t.get(origin, dev - 1).is_finite()
}

/// Expected cumulative next cell `C[i, j] * ldf[j]`, stored at target cell `(i, j + 1)`.
pub fn expected_next_cumulative(t: &TriangleView<'_>, ldfs: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; t.n_origins * t.n_devs];
    for origin in 0..t.n_origins {
        for dev in 1..t.n_devs {
            let link = dev - 1;
            if link >= ldfs.len() || !scheduled_pair_observed(t, origin, dev) {
                continue;
            }
            out[origin * t.n_devs + dev] = t.get(origin, dev - 1) * ldfs[link];
        }
    }
    out
}

/// Raw residuals `actual - expected`, stored at target cell `(i, j + 1)`.
pub fn raw_residuals(t: &TriangleView<'_>, ldfs: &[f64]) -> Vec<f64> {
    let expected = expected_next_cumulative(t, ldfs);
    let mut out = vec![f64::NAN; t.n_origins * t.n_devs];
    for origin in 0..t.n_origins {
        for dev in 1..t.n_devs {
            let expected_value = expected[origin * t.n_devs + dev];
            if expected_value.is_finite() {
                out[origin * t.n_devs + dev] = t.get(origin, dev) - expected_value;
            }
        }
    }
    out
}

/// Actual-to-expected ratios `actual / expected`, stored at target cell `(i, j + 1)`.
pub fn actual_to_expected_ratios(t: &TriangleView<'_>, ldfs: &[f64]) -> Vec<f64> {
    let expected = expected_next_cumulative(t, ldfs);
    let mut out = vec![f64::NAN; t.n_origins * t.n_devs];
    for origin in 0..t.n_origins {
        for dev in 1..t.n_devs {
            let expected_value = expected[origin * t.n_devs + dev];
            if expected_value.is_finite() && expected_value != 0.0 {
                out[origin * t.n_devs + dev] = t.get(origin, dev) / expected_value;
            }
        }
    }
    out
}

/// Empirical residual scale by development link (Mack 1993 `σ_j`).
///
/// Returns one `sigma_j` per link `j -> j + 1`:
///
/// ```text
/// σ_j² = 1 / (n_j − 1) · Σ_i e_{i,j}² / C_{i,j},   e_{i,j} = C_{i,j+1} − C_{i,j} f_j
/// ```
///
/// over the `n_j` observed pairs with `C_{i,j} > 0` — the unbiased Mack
/// estimator shared with Munich Chain Ladder. Links with `n_j < 2` (the last
/// link of a square triangle) are `NaN`; no min-rule extrapolation is applied
/// so standardized residuals are only reported where `σ_j` is estimable.
pub fn residual_scale_by_dev(t: &TriangleView<'_>, ldfs: &[f64]) -> Vec<f64> {
    mack_sigma_sq_raw_by_link(t, ldfs)
        .into_iter()
        .map(|s| {
            if s.is_finite() {
                s.max(0.0).sqrt()
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// Standardized residuals `e / (sigma_j * sqrt(current))`, stored at target cell `(i, j + 1)`.
pub fn standardized_residuals(
    t: &TriangleView<'_>,
    ldfs: &[f64],
    sigma_by_dev: &[f64],
) -> Vec<f64> {
    let residuals = raw_residuals(t, ldfs);
    let mut out = vec![f64::NAN; t.n_origins * t.n_devs];
    for origin in 0..t.n_origins {
        for dev in 1..t.n_devs {
            let link = dev - 1;
            if link >= sigma_by_dev.len() || !scheduled_pair_observed(t, origin, dev) {
                continue;
            }
            let sigma = sigma_by_dev[link];
            let current = t.get(origin, dev - 1);
            let residual = residuals[origin * t.n_devs + dev];
            if sigma.is_finite() && sigma > 0.0 && current > 0.0 && residual.is_finite() {
                out[origin * t.n_devs + dev] = residual / (sigma * current.sqrt());
            }
        }
    }
    out
}

/// Dense residual heatmap rows for all observable target cells (single-pass,
/// zero intermediate matrices).
pub fn cell_residual_heatmap(
    t: &TriangleView<'_>,
    ldfs: &[f64],
    sigma_by_dev: Option<&[f64]>,
) -> Vec<CellResidual> {
    let mut out = Vec::new();
    for origin in 0..t.n_origins {
        for dev in 1..t.n_devs {
            if !scheduled_pair_observed(t, origin, dev) {
                continue;
            }
            let prev = t.get(origin, dev - 1);
            let actual = t.get(origin, dev);
            let ldf = ldfs.get(dev - 1).copied().unwrap_or(f64::NAN);
            let expected = if prev.is_finite() && ldf.is_finite() {
                prev * ldf
            } else {
                f64::NAN
            };
            let residual = if actual.is_finite() && expected.is_finite() {
                actual - expected
            } else {
                f64::NAN
            };
            let actual_to_expected = if actual.is_finite() && expected.is_finite() && expected > 0.0
            {
                actual / expected
            } else {
                f64::NAN
            };
            let standardized_residual = if let Some(sigmas) = sigma_by_dev {
                let sigma = sigmas.get(dev - 1).copied().unwrap_or(f64::NAN);
                if sigma.is_finite() && sigma > 0.0 && prev > 0.0 && residual.is_finite() {
                    residual / (sigma * prev.sqrt())
                } else {
                    f64::NAN
                }
            } else {
                f64::NAN
            };
            out.push(CellResidual {
                origin,
                dev,
                actual,
                expected,
                residual,
                actual_to_expected,
                standardized_residual,
            });
        }
    }
    out
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
    fn expected_and_residuals_follow_selected_ldfs() {
        let tri = triangle_4x4();
        let ldfs = [1.5, 1.2, 1.1];

        let expected = expected_next_cumulative(&tri.view(), &ldfs);
        let residuals = raw_residuals(&tri.view(), &ldfs);

        assert!(close(expected[1], 1500.0));
        assert!(close(expected[2], 1800.0));
        assert!(close(residuals[2], -100.0));
        assert!(close(residuals[6], -160.0));
    }

    #[test]
    fn actual_to_expected_and_standardized_residuals_are_computed_by_target_cell() {
        let tri = triangle_4x4();
        let ldfs = [1.5, 1.2, 1.1];
        let sigma = residual_scale_by_dev(&tri.view(), &ldfs);
        let ratios = actual_to_expected_ratios(&tri.view(), &ldfs);
        let standardized = standardized_residuals(&tri.view(), &ldfs, &sigma);

        assert!(close(ratios[1], 1.0));
        assert!(close(ratios[2], 1700.0 / 1800.0));
        assert!(standardized[2].is_finite());
        assert!(standardized[6].is_finite());
    }

    #[test]
    fn heatmap_rows_capture_cell_level_diagnostics() {
        let tri = triangle_4x4();
        let ldfs = [1.5, 1.2, 1.1];
        let sigma = residual_scale_by_dev(&tri.view(), &ldfs);
        let rows = cell_residual_heatmap(&tri.view(), &ldfs, Some(&sigma));

        assert_eq!(rows.len(), 6);
        assert_eq!(rows[0].origin, 0);
        assert_eq!(rows[0].dev, 1);
        assert!(rows.iter().all(|row| row.expected.is_finite()));
    }

    #[test]
    fn residual_scale_uses_n_minus_one_and_skips_single_pair_links() {
        let tri = triangle_4x4();
        let ldfs = [1.5, 1.2, 1.1];
        let sigma = residual_scale_by_dev(&tri.view(), &ldfs);
        // Link 0: ratios 1.5, 1.5, 1.4 around f = 1.5 → Σ C (F − f)² = 1500·0.01 = 15;
        // σ² = 15 / (3 − 1) = 7.5.
        assert!(close(sigma[0], 7.5_f64.sqrt()));
        // Link 1: 1500·(1700/1500 − 1.2)² + 1800·(2000/1800 − 1.2)², divided by (2 − 1).
        let s1 = 1500.0 * (1700.0_f64 / 1500.0 - 1.2).powi(2)
            + 1800.0 * (2000.0_f64 / 1800.0 - 1.2).powi(2);
        assert!(close(sigma[1], s1.sqrt()));
        // Link 2 has a single pair: σ undefined.
        assert!(sigma[2].is_nan());
        let standardized = standardized_residuals(&tri.view(), &ldfs, &sigma);
        assert!(standardized[3].is_nan());
    }
}
