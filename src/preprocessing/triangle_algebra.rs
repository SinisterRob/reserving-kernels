//! Allocation-free numeric kernels for triangle algebra.
//!
//! These functions intentionally operate on flat slices and caller-provided
//! output buffers. Public actuarial types can validate and label data before
//! calling into this layer. Shape mismatches return
//! [`ReservingError::ShapeMismatch`] instead of panicking.

use crate::error::ReservingError;

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
///
/// Returns `ShapeMismatch` when `cum.len() != out.len()`.
#[inline]
pub fn cumulative_to_incremental_row(cum: &[f64], out: &mut [f64]) -> Result<(), ReservingError> {
    if cum.len() != out.len() {
        return Err(ReservingError::ShapeMismatch);
    }
    let mut prev = 0.0;
    for (slot, &value) in out.iter_mut().zip(cum) {
        *slot = value - prev;
        prev = value;
    }
    Ok(())
}

/// Convert one incremental row to cumulative values.
///
/// Returns `ShapeMismatch` when `incr.len() != out.len()`.
#[inline]
pub fn incremental_to_cumulative_row(incr: &[f64], out: &mut [f64]) -> Result<(), ReservingError> {
    if incr.len() != out.len() {
        return Err(ReservingError::ShapeMismatch);
    }
    let mut acc = 0.0;
    for (slot, &value) in out.iter_mut().zip(incr) {
        acc += value;
        *slot = acc;
    }
    Ok(())
}

/// Apply a row kernel to every `n_dev`-wide row of a flat row-major triangle.
fn map_flat_rows(
    values: &[f64],
    n_origin: usize,
    n_dev: usize,
    out: &mut [f64],
    row_kernel: fn(&[f64], &mut [f64]) -> Result<(), ReservingError>,
) -> Result<(), ReservingError> {
    if n_origin.checked_mul(n_dev) != Some(values.len()) || out.len() != values.len() {
        return Err(ReservingError::ShapeMismatch);
    }
    if n_dev == 0 {
        return Ok(());
    }
    for (row, out_row) in values.chunks_exact(n_dev).zip(out.chunks_exact_mut(n_dev)) {
        row_kernel(row, out_row)?;
    }
    Ok(())
}

/// Convert a flat row-major cumulative triangle to incremental values.
///
/// Returns `ShapeMismatch` unless `cum.len() == out.len() == n_origin * n_dev`.
#[inline]
pub fn cumulative_to_incremental_flat(
    cum: &[f64],
    n_origin: usize,
    n_dev: usize,
    out: &mut [f64],
) -> Result<(), ReservingError> {
    map_flat_rows(cum, n_origin, n_dev, out, cumulative_to_incremental_row)
}

/// Convert a flat row-major incremental triangle to cumulative values.
///
/// Returns `ShapeMismatch` unless `incr.len() == out.len() == n_origin * n_dev`.
#[inline]
pub fn incremental_to_cumulative_flat(
    incr: &[f64],
    n_origin: usize,
    n_dev: usize,
    out: &mut [f64],
) -> Result<(), ReservingError> {
    map_flat_rows(incr, n_origin, n_dev, out, incremental_to_cumulative_row)
}

/// Extract the latest observed value and development index for each origin
/// under standard right-triangle geometry (`latest_dev[i] = n_dev - 1 - i`).
///
/// Origins with no observed cell (`i >= n_dev`) receive `(NaN, 0)`. This is
/// the single crate implementation (re-exported as
/// [`crate::deterministic::chain_ladder::latest_observed_by_origin`]).
///
/// Returns `ShapeMismatch` unless `values.len() == n_origin * n_dev` and both
/// outputs have length `n_origin`.
#[inline]
pub fn latest_observed_by_origin(
    values: &[f64],
    n_origin: usize,
    n_dev: usize,
    latest_value: &mut [f64],
    latest_dev: &mut [usize],
) -> Result<(), ReservingError> {
    if n_origin.checked_mul(n_dev) != Some(values.len())
        || latest_value.len() != n_origin
        || latest_dev.len() != n_origin
    {
        return Err(ReservingError::ShapeMismatch);
    }

    for i in 0..n_origin {
        if i >= n_dev {
            latest_dev[i] = 0;
            latest_value[i] = f64::NAN;
            continue;
        }
        let last_j = n_dev - 1 - i;
        latest_dev[i] = last_j;
        latest_value[i] = values[i * n_dev + last_j];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_cumulative_to_incremental() {
        let cum = [100.0, 150.0, 170.0, 175.0];
        let mut out = [0.0; 4];

        cumulative_to_incremental_row(&cum, &mut out).expect("row");

        assert_eq!(out, [100.0, 50.0, 20.0, 5.0]);
    }

    #[test]
    fn row_incremental_to_cumulative() {
        let incr = [100.0, 50.0, 20.0, 5.0];
        let mut out = [0.0; 4];

        incremental_to_cumulative_row(&incr, &mut out).expect("row");

        assert_eq!(out, [100.0, 150.0, 170.0, 175.0]);
    }

    #[test]
    fn flat_round_trip() {
        let cum = [100.0, 150.0, 170.0, 120.0, 180.0, 0.0];
        let mut incr = [0.0; 6];
        let mut round_trip = [0.0; 6];

        cumulative_to_incremental_flat(&cum, 2, 3, &mut incr).expect("flat");
        incremental_to_cumulative_flat(&incr, 2, 3, &mut round_trip).expect("flat");

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

        latest_observed_by_origin(&values, 4, 4, &mut latest_value, &mut latest_dev)
            .expect("latest");

        assert_eq!(latest_value, [175.0, 190.0, 180.0, 130.0]);
        assert_eq!(latest_dev, [3, 2, 1, 0]);
    }

    #[test]
    fn shape_mismatches_return_errors_instead_of_panicking() {
        let mut out = [0.0; 3];
        assert_eq!(
            cumulative_to_incremental_row(&[1.0, 2.0], &mut out),
            Err(ReservingError::ShapeMismatch)
        );
        assert_eq!(
            incremental_to_cumulative_row(&[1.0, 2.0], &mut out),
            Err(ReservingError::ShapeMismatch)
        );
        let mut flat = [0.0; 5];
        assert_eq!(
            cumulative_to_incremental_flat(&[0.0; 6], 2, 3, &mut flat),
            Err(ReservingError::ShapeMismatch)
        );
        assert_eq!(
            incremental_to_cumulative_flat(&[0.0; 6], 3, 3, &mut [0.0; 6]),
            Err(ReservingError::ShapeMismatch)
        );
        let mut latest = [0.0; 2];
        let mut dev = [0_usize; 3];
        assert_eq!(
            latest_observed_by_origin(&[0.0; 9], 3, 3, &mut latest, &mut dev),
            Err(ReservingError::ShapeMismatch)
        );
    }
}
