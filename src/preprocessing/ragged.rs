//! Factor selection for rectangular cumulative triangles.
//!
//! Inputs are row-major losses in consistent units; counts identify each row
//! observed prefix at the caller's valuation date. No calendar assumptions are
//! introduced here. Unselectable links produce NaN; callers validate results.

/// Counts-aware simple average. `last_n=Some(k)` restricts to the `k` most
/// recent eligible origins per link (origins are ordered with oldest first).
pub fn simple_average_observed_counts(
    cum: &[f64],
    n_o: usize,
    n_d: usize,
    counts: &[usize],
    last_n: Option<usize>,
    out: &mut [f64],
) {
    debug_assert_eq!(cum.len(), n_o * n_d);
    debug_assert_eq!(counts.len(), n_o);
    let n_links = n_d.saturating_sub(1);
    debug_assert_eq!(out.len(), n_links);

    for j in 0..n_links {
        let limit = last_n.unwrap_or(usize::MAX);
        let mut seen_eligible = 0_usize;
        let mut sum = 0.0_f64;
        let mut count = 0_usize;
        for i in (0..n_o).rev() {
            if counts[i] <= j + 1 {
                continue;
            }
            if seen_eligible >= limit {
                break;
            }
            seen_eligible += 1;
            let x = cum[i * n_d + j];
            let y = cum[i * n_d + j + 1];
            if x > 0.0 && x.is_finite() && y.is_finite() {
                sum += y / x;
                count += 1;
            }
        }
        out[j] = if count > 0 {
            sum / count as f64
        } else {
            f64::NAN
        };
    }
}

/// Counts-aware ex-hi/lo (drop largest and smallest ratio per link;
/// fall back to simple average when fewer than 3 pairs).
///
/// Uses a two-pass index-tracking scan (zero heap allocations and free of
/// catastrophic cancellation on extreme outlier ratios).
pub fn ex_hi_lo_observed_counts(
    cum: &[f64],
    n_o: usize,
    n_d: usize,
    counts: &[usize],
    out: &mut [f64],
) {
    debug_assert_eq!(cum.len(), n_o * n_d);
    debug_assert_eq!(counts.len(), n_o);
    let n_links = n_d.saturating_sub(1);
    debug_assert_eq!(out.len(), n_links);

    for j in 0..n_links {
        let mut valid_count = 0_usize;
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        let mut min_idx = 0_usize;
        let mut max_idx = 0_usize;

        for i in 0..n_o {
            if counts[i] <= j + 1 {
                continue;
            }
            let x = cum[i * n_d + j];
            let y = cum[i * n_d + j + 1];
            if x > 0.0 && x.is_finite() && y.is_finite() {
                let r = y / x;
                if valid_count == 0 || r < min_val {
                    min_val = r;
                    min_idx = i;
                }
                if valid_count == 0 || r > max_val {
                    max_val = r;
                    max_idx = i;
                }
                valid_count += 1;
            }
        }

        if valid_count == 0 {
            out[j] = f64::NAN;
            continue;
        }

        // If all valid ratios are equal, min_idx == max_idx; pick any distinct
        // second index when trimming >= 3 elements.
        let trim = valid_count >= 3;
        if trim && min_idx == max_idx {
            for i in 0..n_o {
                if i == min_idx || counts[i] <= j + 1 {
                    continue;
                }
                let x = cum[i * n_d + j];
                let y = cum[i * n_d + j + 1];
                if x > 0.0 && x.is_finite() && y.is_finite() {
                    max_idx = i;
                    break;
                }
            }
        }

        let mut sum = 0.0_f64;
        let mut kept = 0_usize;
        for i in 0..n_o {
            if counts[i] <= j + 1 {
                continue;
            }
            if trim && (i == min_idx || i == max_idx) {
                continue;
            }
            let x = cum[i * n_d + j];
            let y = cum[i * n_d + j + 1];
            if x > 0.0 && x.is_finite() && y.is_finite() {
                sum += y / x;
                kept += 1;
            }
        }
        out[j] = if kept > 0 {
            sum / kept as f64
        } else {
            f64::NAN
        };
    }
}

/// Counts-aware volume-weighted, restricted to the `last_n` most recent eligible
/// origins (largest index = most recent in the right-triangle convention).
pub fn volume_weighted_last_n_observed_counts(
    cum: &[f64],
    n_o: usize,
    n_d: usize,
    counts: &[usize],
    last_n: usize,
    out: &mut [f64],
) {
    debug_assert_eq!(cum.len(), n_o * n_d);
    debug_assert_eq!(counts.len(), n_o);
    let n_links = n_d.saturating_sub(1);
    debug_assert_eq!(out.len(), n_links);

    for j in 0..n_links {
        let mut seen = 0_usize;
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for i in (0..n_o).rev() {
            if counts[i] <= j + 1 {
                continue;
            }
            if seen >= last_n {
                break;
            }
            seen += 1;
            num += cum[i * n_d + j + 1];
            den += cum[i * n_d + j];
        }
        out[j] = if den > 0.0 { num / den } else { f64::NAN };
    }
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
    }
}
