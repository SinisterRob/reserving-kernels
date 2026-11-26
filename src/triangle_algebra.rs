//! Allocation-free numeric kernels for triangle algebra.
//!
//! These functions intentionally operate on flat slices and caller-provided
//! output buffers. Public actuarial types can validate and label data before
//! calling into this layer.

/// Returns whether a cell is observed under standard right-triangle geometry.
#[inline]
pub const fn is_observed_right_triangle(
    origin_idx: usize,
    dev_idx: usize,
    _n_origin: usize,
    n_dev: usize,
) -> bool {
    origin_idx + dev_idx < n_dev
}

/// Convert one cumulative row to incremental values.
#[inline]
pub fn cumulative_to_incremental_row(cum: &[f64], out: &mut [f64]) {
    debug_assert_eq!(cum.len(), out.len());
    if cum.is_empty() {
        return;
    }

    out[0] = cum[0];
    for j in 1..cum.len() {
        out[j] = cum[j] - cum[j - 1];
    }
}

/// Convert one incremental row to cumulative values.
#[inline]
pub fn incremental_to_cumulative_row(incr: &[f64], out: &mut [f64]) {
    debug_assert_eq!(incr.len(), out.len());

    let mut acc = 0.0;
    for (j, &value) in incr.iter().enumerate() {
        acc += value;
        out[j] = acc;
    }
}

/// Convert a flat row-major cumulative triangle to incremental values.
#[inline]
pub fn cumulative_to_incremental_flat(cum: &[f64], n_origin: usize, n_dev: usize, out: &mut [f64]) {
    debug_assert_eq!(cum.len(), n_origin * n_dev);
    debug_assert_eq!(out.len(), cum.len());

    for i in 0..n_origin {
        let start = i * n_dev;
        let end = start + n_dev;
        cumulative_to_incremental_row(&cum[start..end], &mut out[start..end]);
    }
}

/// Convert a flat row-major incremental triangle to cumulative values.
#[inline]
pub fn incremental_to_cumulative_flat(
    incr: &[f64],
    n_origin: usize,
    n_dev: usize,
    out: &mut [f64],
) {
    debug_assert_eq!(incr.len(), n_origin * n_dev);
    debug_assert_eq!(out.len(), incr.len());

    for i in 0..n_origin {
        let start = i * n_dev;
        let end = start + n_dev;
        incremental_to_cumulative_row(&incr[start..end], &mut out[start..end]);
    }
}

/// Extract the latest observed value and development index for each origin.
///
/// This assumes standard right-triangle geometry.
#[inline]
pub fn latest_observed_by_origin(
    values: &[f64],
    n_origin: usize,
    n_dev: usize,
    latest_value: &mut [f64],
    latest_dev: &mut [usize],
) {
    debug_assert_eq!(values.len(), n_origin * n_dev);
    debug_assert_eq!(latest_value.len(), n_origin);
    debug_assert_eq!(latest_dev.len(), n_origin);

    if n_dev == 0 {
        return;
    }

    for i in 0..n_origin {
        if i >= n_dev {
            latest_dev[i] = 0;
            latest_value[i] = 0.0;
            continue;
        }
        let last_j = n_dev - 1 - i;
        latest_dev[i] = last_j;
        latest_value[i] = values[i * n_dev + last_j];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_cumulative_to_incremental() {
        let cum = [100.0, 150.0, 170.0, 175.0];
        let mut out = [0.0; 4];

        cumulative_to_incremental_row(&cum, &mut out);

        assert_eq!(out, [100.0, 50.0, 20.0, 5.0]);
    }

    #[test]
    fn row_incremental_to_cumulative() {
        let incr = [100.0, 50.0, 20.0, 5.0];
        let mut out = [0.0; 4];

        incremental_to_cumulative_row(&incr, &mut out);

        assert_eq!(out, [100.0, 150.0, 170.0, 175.0]);
    }

    #[test]
    fn flat_round_trip() {
        let cum = [100.0, 150.0, 170.0, 120.0, 180.0, 0.0];
        let mut incr = [0.0; 6];
        let mut round_trip = [0.0; 6];

        cumulative_to_incremental_flat(&cum, 2, 3, &mut incr);
        incremental_to_cumulative_flat(&incr, 2, 3, &mut round_trip);

        assert_eq!(round_trip, cum);
    }

    #[test]
    fn right_triangle_observed_geometry() {
        assert!(is_observed_right_triangle(0, 4, 5, 5));
        assert!(is_observed_right_triangle(4, 0, 5, 5));
        assert!(!is_observed_right_triangle(1, 4, 5, 5));
        assert!(!is_observed_right_triangle(4, 1, 5, 5));
    }

    #[test]
    fn latest_observed_right_triangle() {
        let values = [
            100.0, 150.0, 170.0, 175.0, 110.0, 165.0, 190.0, 0.0, 120.0, 180.0, 0.0, 0.0, 130.0,
            0.0, 0.0, 0.0,
        ];
        let mut latest_value = [0.0; 4];
        let mut latest_dev = [0; 4];

        latest_observed_by_origin(&values, 4, 4, &mut latest_value, &mut latest_dev);

        assert_eq!(latest_value, [175.0, 190.0, 180.0, 130.0]);
        assert_eq!(latest_dev, [3, 2, 1, 0]);
    }
}
