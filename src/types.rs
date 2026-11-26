//! Prevalidated columnar triangle inputs.
//!
//! A loss-development triangle is an `n_origins × n_devs` grid of cumulative
//! amounts (paid or incurred). Cell `(i, j)` is observed iff `i + j ≤ n - 1`
//! for a square triangle. Unobserved cells are represented as `NaN` so the
//! compute kernels stay branch-light — null-handling lives at the ingress
//! boundary (see [`crate::validate`]).

use crate::validate::{ValidationError, validate_triangle};

/// An `n_origins × n_devs` cumulative loss triangle in row-major layout.
///
/// `data[i * n_devs + j]` is the cumulative loss for origin `i` at
/// development age `j`. Unobserved cells are `NaN`. The kernels assume the
/// triangle is *square* (`n_origins == n_devs`) and that cell `(i, j)` is
/// observed for `i + j ≤ n - 1`.
#[derive(Debug, Clone, PartialEq)]
pub struct Triangle {
    /// Number of origin (accident / policy) periods.
    pub n_origins: usize,
    /// Number of development periods.
    pub n_devs: usize,
    /// Row-major buffer of length `n_origins * n_devs`.
    pub data: Vec<f64>,
}

impl Triangle {
    /// Construct a triangle from a row-major slice without validation.
    ///
    /// Callers must ensure `data.len() == n_origins * n_devs`. Use
    /// [`Self::try_new`] at external ingress boundaries when full validation
    /// (square, finite, non-negative observed cells) is required.
    #[inline]
    pub fn new(n_origins: usize, n_devs: usize, data: Vec<f64>) -> Self {
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
        let triangle = Self {
            n_origins,
            n_devs,
            data,
        };
        validate_triangle(&triangle.view())?;
        Ok(triangle)
    }

    /// Borrow as a [`TriangleView`].
    #[inline]
    pub fn view(&self) -> TriangleView<'_> {
        TriangleView {
            n_origins: self.n_origins,
            n_devs: self.n_devs,
            data: &self.data,
        }
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

impl<'a> TriangleView<'a> {
    /// Cumulative value at origin `i`, development `j`. `NaN` if unobserved.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n_devs + j]
    }

    /// `true` iff cell `(i, j)` is scheduled to be observed under the
    /// upper-triangular square-triangle convention (`i + j <= n - 1`).
    #[inline]
    pub fn is_scheduled_observed(&self, i: usize, j: usize) -> bool {
        let n = self.n_origins.min(self.n_devs);
        i + j <= n.saturating_sub(1)
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
    }
}
