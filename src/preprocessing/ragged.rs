//! Factor selection for rectangular cumulative triangles.
//!
//! Inputs are row-major losses in consistent units; counts identify each row
//! observed prefix at the caller's valuation date. No calendar assumptions are
//! introduced here. Unselectable links produce NaN; callers validate results.
//!
//! All selectors delegate to the crate's shared averaging kernels (the same
//! definitions used by [`crate::deterministic::ldf`]); they never panic. On a
//! shape mismatch (`cum.len() != n_o * n_d`, `counts.len() != n_o`, or
//! `out.len() != n_d - 1`) every element of `out` is set to `NaN`.

use crate::deterministic::averaging::{self, LinkPair};

fn fill_links(
    cum: &[f64],
    n_o: usize,
    n_d: usize,
    counts: &[usize],
    out: &mut [f64],
    average: impl Fn(&[LinkPair]) -> f64,
) {
    if cum.len() != n_o * n_d || counts.len() != n_o || out.len() != n_d.saturating_sub(1) {
        out.fill(f64::NAN);
        return;
    }
    for (j, slot) in out.iter_mut().enumerate() {
        *slot = average(&averaging::link_pairs_observed_counts(
            cum, n_o, n_d, counts, j,
        ));
    }
}

/// Counts-aware simple average. `last_n=Some(k)` restricts to the `k` most
/// recent eligible origins per link (origins are ordered with oldest first);
/// the window is applied before undefined ratios are skipped. `Some(0)` and
/// `None` both use every eligible origin.
pub fn simple_average_observed_counts(
    cum: &[f64],
    n_o: usize,
    n_d: usize,
    counts: &[usize],
    last_n: Option<usize>,
    out: &mut [f64],
) {
    let window = last_n.unwrap_or(0);
    fill_links(cum, n_o, n_d, counts, out, |pairs| {
        averaging::simple_average(averaging::recent(pairs, window))
    });
}

/// Counts-aware ex-hi/lo (drop largest and smallest ratio per link;
/// fall back to simple average when fewer than 3 pairs).
pub fn ex_hi_lo_observed_counts(
    cum: &[f64],
    n_o: usize,
    n_d: usize,
    counts: &[usize],
    out: &mut [f64],
) {
    fill_links(cum, n_o, n_d, counts, out, averaging::ex_hi_lo);
}

/// Counts-aware volume-weighted, restricted to the `last_n` most recent eligible
/// origins (largest index = most recent in the right-triangle convention).
/// `last_n == 0` uses every eligible origin.
pub fn volume_weighted_last_n_observed_counts(
    cum: &[f64],
    n_o: usize,
    n_d: usize,
    counts: &[usize],
    last_n: usize,
    out: &mut [f64],
) {
    fill_links(cum, n_o, n_d, counts, out, |pairs| {
        averaging::volume_weighted(averaging::recent(pairs, last_n))
    });
}

/// Counts-aware median of individual ratios per link.
pub fn median_observed_counts(
    cum: &[f64],
    n_o: usize,
    n_d: usize,
    counts: &[usize],
    out: &mut [f64],
) {
    fill_links(cum, n_o, n_d, counts, out, averaging::median);
}

/// Counts-aware geometric average of strictly positive individual ratios.
pub fn geometric_average_observed_counts(
    cum: &[f64],
    n_o: usize,
    n_d: usize,
    counts: &[usize],
    out: &mut [f64],
) {
    fill_links(cum, n_o, n_d, counts, out, averaging::geometric);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observed_count_selectors_match_manual_ratios() {
        #[rustfmt::skip]
        let cum = [
            100.0, 150.0, 180.0,
            100.0, 200.0, 240.0,
            100.0, 160.0, f64::NAN,
            100.0, f64::NAN, f64::NAN,
        ];
        let counts = [3, 3, 2, 1];
        let mut out = [0.0; 2];

        simple_average_observed_counts(&cum, 4, 3, &counts, None, &mut out);
        assert!((out[0] - (1.5 + 2.0 + 1.6) / 3.0).abs() < 1e-12);
        assert!((out[1] - 1.2).abs() < 1e-12);

        simple_average_observed_counts(&cum, 4, 3, &counts, Some(2), &mut out);
        assert!((out[0] - f64::midpoint(2.0, 1.6)).abs() < 1e-12);

        ex_hi_lo_observed_counts(&cum, 4, 3, &counts, &mut out);
        assert!((out[0] - 1.6).abs() < 1e-12);

        volume_weighted_last_n_observed_counts(&cum, 4, 3, &counts, 2, &mut out);
        assert!((out[0] - (200.0 + 160.0) / 200.0).abs() < 1e-12);

        median_observed_counts(&cum, 4, 3, &counts, &mut out);
        assert!((out[0] - 1.6).abs() < 1e-12);

        geometric_average_observed_counts(&cum, 4, 3, &counts, &mut out);
        assert!((out[0] - (1.5_f64 * 2.0 * 1.6).cbrt()).abs() < 1e-12);
    }

    #[test]
    fn shape_mismatch_fills_nan_instead_of_panicking() {
        let mut out = [0.0; 2];
        simple_average_observed_counts(&[1.0; 5], 2, 3, &[3, 2], None, &mut out);
        assert!(out.iter().all(|v| v.is_nan()));
        let mut out = [0.0; 1];
        ex_hi_lo_observed_counts(&[1.0; 6], 2, 3, &[3, 2], &mut out);
        assert!(out[0].is_nan());
    }

    #[test]
    fn square_counts_match_square_selectors() {
        use crate::deterministic::ldf;
        use crate::types::TriangleView;
        #[rustfmt::skip]
        let cum = [
            100.0, 150.0, 180.0, 190.0,
            110.0, 170.0, 200.0, f64::NAN,
            120.0, 175.0, f64::NAN, f64::NAN,
            130.0, f64::NAN, f64::NAN, f64::NAN,
        ];
        let counts = [4, 3, 2, 1];
        let view = TriangleView {
            n_origins: 4,
            n_devs: 4,
            data: &cum,
        };
        let mut out = [0.0; 3];
        ex_hi_lo_observed_counts(&cum, 4, 4, &counts, &mut out);
        assert_eq!(out.to_vec(), ldf::ex_hi_lo(&view));
        simple_average_observed_counts(&cum, 4, 4, &counts, Some(2), &mut out);
        assert_eq!(out.to_vec(), ldf::simple_average_last_n(&view, 2));
        volume_weighted_last_n_observed_counts(&cum, 4, 4, &counts, 2, &mut out);
        assert_eq!(out.to_vec(), ldf::volume_weighted_last_n(&view, 2));
        median_observed_counts(&cum, 4, 4, &counts, &mut out);
        assert_eq!(out.to_vec(), ldf::median(&view));
    }
}
