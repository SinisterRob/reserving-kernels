//! Shared link-ratio averaging kernels.
//!
//! Every age-to-age selection method in the crate (square [`crate::deterministic::ldf`]
//! selectors, ragged/observed-count selectors in
//! [`crate::preprocessing::ragged`], and link-ratio diagnostics) reduces to an
//! average over the `(C_{i,j}, C_{i,j+1})` pairs of one development link. This
//! module is the single definition of those averages so all callers agree
//! bit-for-bit.
//!
//! Conventions:
//!
//! * A [`LinkPair`] is one *scheduled-observed* origin row for the link (both
//!   cells inside the observed geometry). Values are passed through unfiltered:
//!   volume-weighted sums propagate `NaN`, while ratio-based methods skip
//!   pairs whose individual ratio is undefined (`C_{i,j} <= 0` or non-finite
//!   `C_{i,j+1}`), exactly like [`crate::preprocessing::triangle::age_to_age`].
//! * Pairs are ordered oldest origin first; "last `n`" windows take the tail.
//! * Every average returns `NaN` when the link has no usable data, so checked
//!   engines surface `NotFinite` / `DivideByZero` instead of a silent `1.0`.

use crate::types::TriangleView;

/// One scheduled-observed origin row for development link `j -> j + 1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LinkPair {
    /// Origin row index.
    pub(crate) origin: usize,
    /// Cumulative value at development `j` (ratio denominator).
    pub(crate) current: f64,
    /// Cumulative value at development `j + 1` (ratio numerator).
    pub(crate) next: f64,
}

impl LinkPair {
    /// Individual link ratio `next / current`, or `None` when undefined.
    #[inline]
    pub(crate) fn ratio(self) -> Option<f64> {
        if self.current > 0.0 && self.next.is_finite() {
            Some(self.next / self.current)
        } else {
            None
        }
    }
}

/// Scheduled-observed pairs for link `dev -> dev + 1` of a right triangle
/// (`i + dev + 1 < min(n_origins, n_devs)`). Out-of-range `dev` yields no pairs.
pub(crate) fn link_pairs(t: &TriangleView<'_>, dev: usize) -> Vec<LinkPair> {
    let n = t.n_origins.min(t.n_devs);
    if dev + 1 >= n || t.data.len() < t.n_origins * t.n_devs {
        return Vec::new();
    }
    (0..n - dev - 1)
        .map(|origin| LinkPair {
            origin,
            current: t.data[origin * t.n_devs + dev],
            next: t.data[origin * t.n_devs + dev + 1],
        })
        .collect()
}

/// Scheduled-observed pairs for link `dev -> dev + 1` keyed by per-origin
/// observed counts (`counts[i] > dev + 1`). Shape mismatches yield no pairs.
pub(crate) fn link_pairs_observed_counts(
    cum: &[f64],
    n_origin: usize,
    n_dev: usize,
    counts: &[usize],
    dev: usize,
) -> Vec<LinkPair> {
    if cum.len() != n_origin * n_dev || counts.len() != n_origin || dev + 1 >= n_dev {
        return Vec::new();
    }
    (0..n_origin)
        .filter(|&origin| counts[origin] > dev + 1)
        .map(|origin| LinkPair {
            origin,
            current: cum[origin * n_dev + dev],
            next: cum[origin * n_dev + dev + 1],
        })
        .collect()
}

/// Most recent `last_n` pairs (`last_n == 0` keeps all pairs).
#[inline]
pub(crate) fn recent(pairs: &[LinkPair], last_n: usize) -> &[LinkPair] {
    if last_n == 0 || last_n >= pairs.len() {
        pairs
    } else {
        &pairs[pairs.len() - last_n..]
    }
}

/// Defined individual ratios in pair order.
pub(crate) fn ratios(pairs: &[LinkPair]) -> Vec<f64> {
    pairs.iter().filter_map(|pair| pair.ratio()).collect()
}

/// Volume-weighted factor `Σ next / Σ current` over every pair.
///
/// Rows with `current == 0` contribute their `next` mass (the classical
/// chain-ladder definition). Returns `NaN` when `Σ current <= 0`.
pub(crate) fn volume_weighted(pairs: &[LinkPair]) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for pair in pairs {
        denominator += pair.current;
        numerator += pair.next;
    }
    if denominator > 0.0 {
        numerator / denominator
    } else {
        f64::NAN
    }
}

/// Simple (straight) average of defined individual ratios; `NaN` if none.
pub(crate) fn simple_average(pairs: &[LinkPair]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0_u32;
    for ratio in pairs.iter().filter_map(|pair| pair.ratio()) {
        sum += ratio;
        count += 1;
    }
    if count > 0 {
        sum / f64::from(count)
    } else {
        f64::NAN
    }
}

/// Median of defined individual ratios; `NaN` if none.
pub(crate) fn median(pairs: &[LinkPair]) -> f64 {
    let mut values = ratios(pairs);
    median_in_place(&mut values)
}

/// Median of a scratch slice (sorted in place); `NaN` if empty.
pub(crate) fn median_in_place(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let k = values.len();
    if k == 0 {
        f64::NAN
    } else if k % 2 == 1 {
        values[k / 2]
    } else {
        f64::midpoint(values[k / 2 - 1], values[k / 2])
    }
}

/// Mean of defined individual ratios after dropping the first-occurring
/// minimum and maximum (falls back to the simple average below 3 ratios).
/// `NaN` if no ratio is defined.
pub(crate) fn ex_hi_lo(pairs: &[LinkPair]) -> f64 {
    let values = ratios(pairs);
    let k = values.len();
    if k == 0 {
        return f64::NAN;
    }
    if k < 3 {
        return values.iter().sum::<f64>() / k as f64;
    }
    let mut min_idx = 0_usize;
    let mut max_idx = 0_usize;
    for (idx, &v) in values.iter().enumerate().skip(1) {
        if v < values[min_idx] {
            min_idx = idx;
        }
        if v > values[max_idx] {
            max_idx = idx;
        }
    }
    if min_idx == max_idx {
        // All ratios equal: drop any two distinct entries.
        max_idx = 1;
    }
    let mut sum = 0.0;
    for (idx, &v) in values.iter().enumerate() {
        if idx != min_idx && idx != max_idx {
            sum += v;
        }
    }
    sum / ((k - 2) as f64)
}

/// Geometric mean of individual ratios over pairs with strictly positive,
/// finite `current` and `next`; `NaN` if no such pair exists.
pub(crate) fn geometric(pairs: &[LinkPair]) -> f64 {
    let mut log_sum = 0.0;
    let mut count = 0_usize;
    for pair in pairs {
        let (a, b) = (pair.current, pair.next);
        if a.is_finite() && b.is_finite() && a > 0.0 && b > 0.0 {
            log_sum += (b / a).ln();
            count += 1;
        }
    }
    if count > 0 {
        (log_sum / count as f64).exp()
    } else {
        f64::NAN
    }
}

/// Sample (`n − 1`) standard deviation of the defined individual ratios
/// around `mean`; `0.0` with fewer than two ratios.
pub(crate) fn ratio_sample_std_dev(pairs: &[LinkPair], mean: f64) -> f64 {
    let values = ratios(pairs);
    if values.len() <= 1 || !mean.is_finite() {
        return 0.0;
    }
    let variance =
        values.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (values.len() as f64 - 1.0);
    variance.sqrt()
}

/// Mack (1993) link variance estimate
/// `σ²_j = 1/(n_j − 1) · Σ_i C_{i,j} (C_{i,j+1}/C_{i,j} − f_j)²`
/// over pairs with a defined ratio (`n_j` counts those pairs).
///
/// Returns `NaN` when fewer than two pairs contribute (the estimator is
/// undefined; callers extrapolate via the Mack min-rule or skip the link).
pub(crate) fn mack_sigma_sq(pairs: &[LinkPair], factor: f64) -> f64 {
    if !factor.is_finite() {
        return f64::NAN;
    }
    let mut acc = 0.0;
    let mut count = 0_usize;
    for pair in pairs {
        if let Some(ratio) = pair.ratio() {
            acc += pair.current * (ratio - factor).powi(2);
            count += 1;
        }
    }
    if count < 2 {
        f64::NAN
    } else {
        acc / (count as f64 - 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(origin: usize, current: f64, next: f64) -> LinkPair {
        LinkPair {
            origin,
            current,
            next,
        }
    }

    #[test]
    fn averages_match_hand_values() {
        let pairs = [
            pair(0, 100.0, 150.0),
            pair(1, 200.0, 240.0),
            pair(2, 100.0, 160.0),
            pair(3, 0.0, 50.0),
        ];
        // VW includes the zero-denominator row's numerator mass.
        assert!((volume_weighted(&pairs) - 600.0 / 400.0).abs() < 1e-15);
        assert!((simple_average(&pairs) - (1.5 + 1.2 + 1.6) / 3.0).abs() < 1e-15);
        assert!((median(&pairs) - 1.5).abs() < 1e-15);
        assert!((ex_hi_lo(&pairs) - 1.5).abs() < 1e-15);
        let geo = (1.5_f64 * 1.2 * 1.6).cbrt();
        assert!((geometric(&pairs) - geo).abs() < 1e-14);
        assert_eq!(recent(&pairs, 2), &pairs[2..]);
        assert_eq!(recent(&pairs, 0), &pairs[..]);
        assert_eq!(recent(&pairs, 9), &pairs[..]);
    }

    #[test]
    fn no_data_returns_nan() {
        let none: [LinkPair; 0] = [];
        let zero = [pair(0, 0.0, 0.0)];
        for p in [&none[..], &zero[..]] {
            assert!(volume_weighted(p).is_nan());
            assert!(simple_average(p).is_nan());
            assert!(median(p).is_nan());
            assert!(ex_hi_lo(p).is_nan());
            assert!(geometric(p).is_nan());
            assert!(mack_sigma_sq(p, 1.0).is_nan());
        }
    }

    #[test]
    fn mack_sigma_uses_n_minus_one() {
        let pairs = [pair(0, 100.0, 150.0), pair(1, 100.0, 130.0)];
        let f = volume_weighted(&pairs);
        // f = 1.4; Σ C (F - f)² = 100·0.01 + 100·0.01 = 2; σ² = 2 / (2 − 1).
        assert!((mack_sigma_sq(&pairs, f) - 2.0).abs() < 1e-12);
        assert!(mack_sigma_sq(&pairs[..1], f).is_nan());
    }

    #[test]
    fn ex_hi_lo_all_equal_drops_two() {
        let pairs = [pair(0, 1.0, 2.0), pair(1, 1.0, 2.0), pair(2, 1.0, 2.0)];
        assert!((ex_hi_lo(&pairs) - 2.0).abs() < 1e-15);
    }
}
