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
        let row_len = n - i;
        let in_row = &t.data[i * t.n_devs..i * t.n_devs + row_len];
        let out_row = &mut out[i * n..i * n + row_len];
        out_row[0] = in_row[0];
        for j in 1..row_len {
            out_row[j] = in_row[j] - in_row[j - 1];
        }
    }
    out
}

/// Convert incremental to cumulative. Mirror of [`cumulative_to_incremental`].
pub fn incremental_to_cumulative(t: &TriangleView<'_>) -> Vec<f64> {
    let n = t.n_origins.min(t.n_devs);
    let mut out = vec![f64::NAN; n * n];
    for i in 0..n {
        let row_len = n - i;
        let in_row = &t.data[i * t.n_devs..i * t.n_devs + row_len];
        let out_row = &mut out[i * n..i * n + row_len];
        let mut acc = 0.0;
        for j in 0..row_len {
            acc += in_row[j];
            out_row[j] = acc;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Triangle;

    #[test]
    fn latest_diagonal_conversions_and_age_to_age_work() {
        let tri =
            Triangle::try_from_upper_rows(&[&[100.0, 150.0, 180.0], &[120.0, 180.0], &[140.0]])
                .expect("triangle");
        let v = tri.view();
        assert_eq!(latest_diagonal(&v), vec![180.0, 180.0, 140.0]);
        assert!((age_to_age(&v, 0, 0) - 1.5).abs() < 1e-12);
        assert!((age_to_age(&v, 0, 1) - 1.2).abs() < 1e-12);
        assert!(age_to_age(&v, 2, 0).is_nan());
        assert_eq!(link_count(3, 0), 2);
        assert_eq!(link_count(3, 1), 1);
        assert_eq!(link_count(3, 2), 0);

        let inc = cumulative_to_incremental(&v);
        let inc_view = TriangleView {
            n_origins: 3,
            n_devs: 3,
            data: &inc,
        };
        let roundtrip = incremental_to_cumulative(&inc_view);
        assert_eq!(
            v,
            TriangleView {
                n_origins: 3,
                n_devs: 3,
                data: &roundtrip,
            }
        );
    }
}
