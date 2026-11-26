//! Triangle helpers: cumulative ↔ incremental conversion, latest-diagonal
//! extraction, individual age-to-age ratios.
//!
//! All functions treat `NaN` as the unobserved sentinel and assume the input
//! has been validated (see [`crate::validate`]).

use crate::types::TriangleView;

/// Extract the latest-observed cumulative value for each origin.
///
/// For a square triangle of size `n`, origin `i`'s latest observed
/// development age is `n − 1 − i` (anti-diagonal). Returns a `Vec<f64>` of
/// length `n`.
pub fn latest_diagonal(t: &TriangleView<'_>) -> Vec<f64> {
    let n = t.n_origins.min(t.n_devs);
    let mut out = vec![0.0; n];
    for i in 0..n {
        out[i] = t.get(i, n - 1 - i);
    }
    out
}

/// Convert cumulative to incremental in-place semantics: returns a new
/// triangle-shaped `Vec<f64>` whose observed cells are period deltas.
/// Cell `(i, 0)` is copied as-is; cell `(i, j)` with `j > 0` is
/// `C_{i,j} − C_{i,j−1}`. Unobserved cells become `NaN`.
pub fn cumulative_to_incremental(t: &TriangleView<'_>) -> Vec<f64> {
    let n = t.n_origins.min(t.n_devs);
    let mut out = vec![f64::NAN; n * n];
    for i in 0..n {
        for j in 0..n {
            if i + j > n - 1 {
                continue;
            }
            let v = if j == 0 {
                t.get(i, 0)
            } else {
                t.get(i, j) - t.get(i, j - 1)
            };
            out[i * n + j] = v;
        }
    }
    out
}

/// Convert incremental to cumulative. Mirror of [`cumulative_to_incremental`].
pub fn incremental_to_cumulative(t: &TriangleView<'_>) -> Vec<f64> {
    let n = t.n_origins.min(t.n_devs);
    let mut out = vec![f64::NAN; n * n];
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..n {
            if i + j > n - 1 {
                continue;
            }
            acc += t.get(i, j);
            out[i * n + j] = acc;
        }
    }
    out
}

/// Individual age-to-age ratio `C_{i, j+1} / C_{i, j}`. Returns `NaN` where
/// the ratio is undefined (pair unobserved or denominator ≤ 0).
#[inline]
pub fn age_to_age(t: &TriangleView<'_>, i: usize, j: usize) -> f64 {
    let n = t.n_origins.min(t.n_devs);
    if n < 2 || i >= n || j >= n || i.saturating_add(j).saturating_add(1) >= n {
        return f64::NAN;
    }
    let a = t.get(i, j);
    let b = t.get(i, j + 1);
    if !(a > 0.0) || !b.is_finite() {
        return f64::NAN;
    }
    b / a
}

/// Number of origins with both `(i, j)` and `(i, j+1)` observed (i.e. the
/// count that feeds a development-`j → j+1` LDF).
#[inline]
pub fn link_count(n: usize, j: usize) -> usize {
    // Need i + j + 1 ≤ n - 1, so i ≤ n - j - 2, giving n - j - 1 observations
    // for j = 0..n-2.
    if j + 2 > n { 0 } else { n - j - 1 }
}
