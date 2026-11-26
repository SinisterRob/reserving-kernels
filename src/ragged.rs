//! Factor selection for rectangular cumulative triangles.
//!
//! Inputs are row-major losses in consistent units; counts identify each row
//! observed prefix at the caller's valuation date. No calendar assumptions are
//! introduced here. Unselectable links produce NaN; callers validate results.

/// Returns the indices of origins that have **both** `dev j` and `dev j+1`
/// observed under arbitrary per-row widths.
fn eligible_origins(counts: &[usize], j: usize) -> Vec<usize> {
    counts
        .iter()
        .enumerate()
        .filter_map(|(i, &c)| if c > j + 1 { Some(i) } else { None })
        .collect()
}

/// Counts-aware simple average. `last_n=Some(k)` restricts to the `k` most
/// recent origins per link (origins are ordered with oldest first).
pub fn simple_average_observed_counts(
    cum: &[f64],
    n_o: usize,
    n_d: usize,
    counts: &[usize],
    last_n: Option<usize>,
    out: &mut [f64],
) {
    debug_assert_eq!(cum.len(), n_o * n_d);
    for j in 0..n_d - 1 {
        let mut origins = eligible_origins(counts, j);
        if let Some(k) = last_n {
            let drop = origins.len().saturating_sub(k);
            origins.drain(..drop);
        }
        let mut sum = 0.0_f64;
        let mut count = 0_usize;
        for &i in &origins {
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
pub fn ex_hi_lo_observed_counts(
    cum: &[f64],
    n_o: usize,
    n_d: usize,
    counts: &[usize],
    out: &mut [f64],
) {
    debug_assert_eq!(cum.len(), n_o * n_d);
    for j in 0..n_d - 1 {
        let origins = eligible_origins(counts, j);
        let mut ratios: Vec<f64> = Vec::with_capacity(origins.len());
        for &i in &origins {
            let x = cum[i * n_d + j];
            let y = cum[i * n_d + j + 1];
            if x > 0.0 && x.is_finite() && y.is_finite() {
                ratios.push(y / x);
            }
        }
        if ratios.is_empty() {
            out[j] = f64::NAN;
            continue;
        }
        if ratios.len() < 3 {
            out[j] = ratios.iter().sum::<f64>() / ratios.len() as f64;
            continue;
        }
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let trimmed = &ratios[1..ratios.len() - 1];
        out[j] = trimmed.iter().sum::<f64>() / trimmed.len() as f64;
    }
}

/// Counts-aware volume-weighted, restricted to the `k` most recent origins
/// (largest index = most recent in the right-triangle convention).
pub fn volume_weighted_last_n_observed_counts(
    cum: &[f64],
    n_o: usize,
    n_d: usize,
    counts: &[usize],
    last_n: usize,
    out: &mut [f64],
) {
    debug_assert_eq!(cum.len(), n_o * n_d);
    for j in 0..n_d - 1 {
        let mut origins = eligible_origins(counts, j);
        let drop = origins.len().saturating_sub(last_n);
        origins.drain(..drop);
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for &i in &origins {
            num += cum[i * n_d + j + 1];
            den += cum[i * n_d + j];
        }
        out[j] = if den > 0.0 { num / den } else { f64::NAN };
    }
}
