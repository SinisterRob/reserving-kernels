//! Prevalidated columnar triangle inputs.
//!
//! A loss-development triangle is an `n_origins × n_devs` grid of cumulative
//! amounts (paid or incurred). Cell `(i, j)` is observed iff `i + j ≤ n - 1`
//! for a square triangle. Unobserved cells are represented as `NaN` so the
//! compute kernels stay branch-light — null-handling lives at the ingress
//! boundary (see [`crate::validate`]).

use std::fmt;

use crate::validate::{ValidationError, validate_triangle};

/// Deserialize a `Vec<f64>` where JSON `null` values map back to `f64::NAN`,
/// enabling lossless `serde_json` round-trips for triangles containing `NaN`
/// sentinels in unobserved cells.
fn deserialize_vec_f64_null_as_nan<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let opts: Vec<Option<f64>> = Vec::deserialize(deserializer)?;
    Ok(opts.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
}

#[derive(serde::Deserialize)]
struct RawTriangle {
    n_origins: usize,
    n_devs: usize,
    #[serde(deserialize_with = "deserialize_vec_f64_null_as_nan")]
    data: Vec<f64>,
}

impl TryFrom<RawTriangle> for Triangle {
    type Error = ValidationError;

    fn try_from(raw: RawTriangle) -> Result<Self, Self::Error> {
        Self::try_new(raw.n_origins, raw.n_devs, raw.data)
    }
}

/// An `n_origins × n_devs` cumulative loss triangle in row-major layout.
///
/// `data[i * n_devs + j]` is the cumulative loss for origin `i` at
/// development age `j`. Unobserved cells are `NaN`. The kernels assume the
/// triangle is *square* (`n_origins == n_devs`) and that cell `(i, j)` is
/// observed for `i + j ≤ n - 1`.
///
/// Equality (`PartialEq`) treats two `NaN` cells at the same position as equal
/// so `tri == tri.clone()` holds for triangles with unobserved cells.
/// Deserialization maps JSON `null` to `NaN` and validates via [`Self::try_new`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "RawTriangle")]
pub struct Triangle {
    n_origins: usize,
    n_devs: usize,
    data: Vec<f64>,
}

impl PartialEq for Triangle {
    fn eq(&self, other: &Self) -> bool {
        self.view() == other.view()
    }
}

impl Triangle {
    /// Construct a triangle from a row-major buffer without validation.
    ///
    /// Crate-internal only: callers must guarantee
    /// `data.len() == n_origins * n_devs` (accessors such as
    /// [`Self::latest_diagonal`] index the buffer directly and would panic in
    /// release builds otherwise). Every in-crate caller derives `data` from an
    /// existing validated triangle of the same shape or from a fixed-size
    /// literal. External users must go through [`Self::try_new`] or the
    /// `try_from_*` constructors.
    #[inline]
    #[must_use]
    pub(crate) fn new(n_origins: usize, n_devs: usize, data: Vec<f64>) -> Self {
        debug_assert_eq!(data.len(), n_origins * n_devs);
        Self {
            n_origins,
            n_devs,
            data,
        }
    }

    /// Construct and validate a square cumulative triangle.
    ///
    /// This enforces the same shape, finiteness, and non-negative observed-cell
    /// contract used by the checked reserving engines. Row monotonicity is
    /// still method-specific and is validated by those engines.
    pub fn try_new(
        n_origins: usize,
        n_devs: usize,
        data: Vec<f64>,
    ) -> Result<Self, ValidationError> {
        // Checked product first: with untrusted (e.g. deserialized) dimensions
        // `n_origins * n_devs` can overflow and wrap to `data.len()`, which
        // would otherwise let a `2^32 × 2^32` header with empty data through.
        if n_origins.checked_mul(n_devs) != Some(data.len()) {
            return Err(ValidationError::ShapeMismatch);
        }
        let triangle = Self {
            n_origins,
            n_devs,
            data,
        };
        validate_triangle(&triangle.view())?;
        Ok(triangle)
    }

    /// Construct and validate a square cumulative triangle directly from
    /// ragged upper-triangle rows (no manual `f64::NAN` padding required).
    ///
    /// Row `i` must have length `n - i` where `n = rows.len() > 0`.
    /// For owned `Vec<Vec<f64>>` inputs, see [`Self::try_from_rows`].
    ///
    /// # Example
    /// ```rust
    /// use reserving_kernels::Triangle;
    ///
    /// let tri = Triangle::try_from_upper_rows(&[
    ///     &[100.0, 160.0, 195.0],
    ///     &[120.0, 190.0],
    ///     &[135.0],
    /// ])?;
    /// assert_eq!(tri.n(), 3);
    /// assert_eq!(tri.latest_diagonal(), vec![195.0, 190.0, 135.0]);
    /// # Ok::<(), reserving_kernels::validate::ValidationError>(())
    /// ```
    pub fn try_from_upper_rows(rows: &[&[f64]]) -> Result<Self, ValidationError> {
        Self::try_from_rows(rows)
    }

    /// Generic variant of [`Self::try_from_upper_rows`] accepting any slice of
    /// row buffers (`&[Vec<f64>]`, `&[&[f64]]`, etc.).
    pub fn try_from_rows<R: AsRef<[f64]>>(rows: &[R]) -> Result<Self, ValidationError> {
        let n = rows.len();
        if n == 0 {
            return Err(ValidationError::ShapeMismatch);
        }
        let mut data = vec![f64::NAN; n * n];
        for (i, row) in rows.iter().enumerate() {
            let row = row.as_ref();
            if row.len() != n - i {
                return Err(ValidationError::ShapeMismatch);
            }
            data[i * n..i * n + row.len()].copy_from_slice(row);
        }
        Self::try_new(n, n, data)
    }

    /// Construct and validate a square cumulative triangle by cumulating
    /// incremental upper-triangle rows along each origin period.
    ///
    /// Row `i` must have length `n - i` where `n = rows.len() > 0`.
    pub fn try_from_incremental_rows(rows: &[&[f64]]) -> Result<Self, ValidationError> {
        let n = rows.len();
        if n == 0 {
            return Err(ValidationError::ShapeMismatch);
        }
        let mut data = vec![f64::NAN; n * n];
        for (i, row) in rows.iter().enumerate() {
            if row.len() != n - i {
                return Err(ValidationError::ShapeMismatch);
            }
            let mut acc = 0.0;
            for (j, &inc) in row.iter().enumerate() {
                if !inc.is_finite() {
                    return Err(ValidationError::NotFinite);
                }
                acc += inc;
                data[i * n + j] = acc;
            }
        }
        Self::try_new(n, n, data)
    }

    /// Cumulative value at origin `i`, development `j`. `NaN` if unobserved.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n_devs + j]
    }

    /// Observed cumulative slice for origin `i` (length `n - i`, excluding trailing `NaN`s).
    #[inline]
    pub fn origin_row(&self, i: usize) -> &[f64] {
        let n = self.n();
        let len = n.saturating_sub(i);
        let start = i * self.n_devs;
        &self.data[start..start + len]
    }

    /// Number of origin (accident / policy) periods.
    #[inline]
    pub fn n_origins(&self) -> usize {
        self.n_origins
    }

    /// Number of development periods.
    #[inline]
    pub fn n_devs(&self) -> usize {
        self.n_devs
    }

    /// Borrow the underlying row-major slice of length `n_origins * n_devs`.
    #[inline]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Square triangle dimension `min(n_origins, n_devs)`.
    #[inline]
    pub fn n(&self) -> usize {
        self.n_origins.min(self.n_devs)
    }

    /// `true` iff the triangle has zero origins or zero development periods.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_origins == 0 || self.n_devs == 0
    }

    /// Borrow as a [`TriangleView`].
    #[inline]
    #[must_use]
    pub fn view(&self) -> TriangleView<'_> {
        TriangleView {
            n_origins: self.n_origins,
            n_devs: self.n_devs,
            data: &self.data,
        }
    }
}

impl fmt::Display for Triangle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.view().fmt(f)
    }
}

/// A non-owning view over a triangle's row-major buffer.
#[derive(Debug, Clone, Copy)]
pub struct TriangleView<'a> {
    /// Number of origin periods.
    pub n_origins: usize,
    /// Number of development periods.
    pub n_devs: usize,
    /// Row-major buffer.
    pub data: &'a [f64],
}

impl PartialEq for TriangleView<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.n_origins == other.n_origins
            && self.n_devs == other.n_devs
            && self.data.len() == other.data.len()
            && self
                .data
                .iter()
                .zip(other.data.iter())
                .all(|(&a, &b)| (a.is_nan() && b.is_nan()) || a == b)
    }
}

impl fmt::Display for TriangleView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.n_origins {
            if i > 0 {
                writeln!(f)?;
            }
            for j in 0..self.n_devs {
                if j > 0 {
                    write!(f, "\t")?;
                }
                let v = self.data[i * self.n_devs + j];
                if v.is_finite() {
                    write!(f, "{v:.2}")?;
                } else {
                    write!(f, "-")?;
                }
            }
        }
        Ok(())
    }
}

impl TriangleView<'_> {
    /// Cumulative value at origin `i`, development `j`. `NaN` if unobserved.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n_devs + j]
    }

    /// `true` iff cell `(i, j)` is scheduled to be observed under the
    /// upper-triangular square-triangle convention (`i + j < n`).
    #[inline]
    pub fn is_scheduled_observed(&self, i: usize, j: usize) -> bool {
        let n = self.n_origins.min(self.n_devs);
        i < n && j < n && i + j < n
    }

    /// `true` iff cell `(i, j)` is scheduled observed and holds a finite value.
    #[inline]
    pub fn is_valid_observed(&self, i: usize, j: usize) -> bool {
        self.is_scheduled_observed(i, j) && self.get(i, j).is_finite()
    }

    /// `true` iff cell `(i, j)` is scheduled observed and holds a finite value.
    ///
    /// This preserves the original hot-path predicate. Use
    /// [`Self::is_scheduled_observed`] when validation needs to distinguish
    /// triangle geometry from numeric validity.
    #[inline]
    pub fn is_observed(&self, i: usize, j: usize) -> bool {
        self.is_valid_observed(i, j)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::ValidationError;

    #[test]
    fn try_new_rejects_shape_mismatch() {
        assert_eq!(
            Triangle::try_new(2, 2, vec![1.0, 2.0, 3.0]),
            Err(ValidationError::ShapeMismatch)
        );
    }

    #[test]
    fn try_new_accepts_valid_square_triangle() {
        let triangle = Triangle::try_new(2, 2, vec![1.0, 2.0, 3.0, f64::NAN]).expect("triangle");

        assert_eq!(triangle.n_origins, 2);
        assert_eq!(triangle.n_devs, 2);
        assert_eq!(triangle, triangle.clone());
        assert_eq!(triangle.view(), triangle.view());
    }

    #[test]
    fn triangle_serde_json_roundtrips_nan_and_validates() {
        let triangle =
            Triangle::try_from_upper_rows(&[&[100.0, 160.0, 195.0], &[120.0, 190.0], &[135.0]])
                .expect("triangle");
        let json = serde_json::to_string(&triangle).expect("serialize");
        let decoded: Triangle = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(triangle, decoded);

        let malformed = r#"{"n_origins":3,"n_devs":3,"data":[1.0]}"#;
        assert!(serde_json::from_str::<Triangle>(malformed).is_err());
    }
}
