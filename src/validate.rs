//! Input validation — kept deliberately off the compute hot path.
//!
//! Call these at the ingress boundary (CSV/DB/API → `Triangle`). Kernels in
//! the rest of the crate assume validated inputs and skip defensive checks.

use crate::error::ReservingError;
use crate::types::TriangleView;

/// Policy controlling whether cumulative rows must be monotonically non-decreasing.
///
/// Strict monotonicity (`Strict`) is appropriate for pure gross paid triangles
/// without salvage/subrogation recoveries. Incurred triangles and reinsurance
/// triangles (including Mack's 1993 RAA benchmark) legitimately exhibit downward
/// case-reserve revisions and require [`MonotonicityPolicy::AllowNegativeIncrements`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum MonotonicityPolicy {
    /// Reject any observed cumulative cell strictly lower than its predecessor.
    #[default]
    Strict,
    /// Permit downward cumulative movements (case-reserve releases, recoveries)
    /// while still enforcing finite non-negative cumulative balances.
    AllowNegativeIncrements,
}

/// Validate a triangle view: square shape, finite non-negative observed
/// cells. Unobserved cells (`i + j > n − 1`) are ignored.
pub fn validate_triangle(t: &TriangleView<'_>) -> Result<(), ReservingError> {
    if t.n_origins.checked_mul(t.n_devs) != Some(t.data.len()) {
        return Err(ReservingError::ShapeMismatch);
    }
    if t.n_origins != t.n_devs {
        return Err(ReservingError::NotSquare);
    }
    let n = t.n_origins;
    for i in 0..n {
        let row = &t.data[i * n..i * n + (n - i)];
        for &v in row {
            if !v.is_finite() {
                // NaN is our "unobserved" sentinel — within the observed
                // upper triangle, NaN or infinity is an error.
                return Err(ReservingError::NotFinite);
            }
            if v < 0.0 {
                return Err(ReservingError::Negative);
            }
        }
    }
    Ok(())
}

/// Validate a premium vector: length matches origins, finite positive.
pub fn validate_premium(premium: &[f64], n_origins: usize) -> Result<(), ReservingError> {
    if premium.len() != n_origins {
        return Err(ReservingError::ShapeMismatch);
    }
    for &p in premium {
        if !p.is_finite() {
            return Err(ReservingError::NotFinite);
        }
        if p <= 0.0 {
            return Err(ReservingError::NonPositive("premium"));
        }
    }
    Ok(())
}

/// Validate that observed cumulative rows are non-decreasing.
pub(crate) fn validate_monotone_cumulative(t: &TriangleView<'_>) -> Result<(), ReservingError> {
    let n = t.n_origins.min(t.n_devs);
    for i in 0..n {
        let row_len = n - i;
        let row = &t.data[i * t.n_devs..i * t.n_devs + row_len];
        let mut prev: Option<f64> = None;
        for (j, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                continue;
            }
            if let Some(prev_value) = prev
                && value < prev_value
            {
                return Err(ReservingError::NonMonotoneCumulative {
                    origin: i,
                    dev: j,
                    previous: prev_value,
                    value,
                });
            }
            prev = Some(value);
        }
    }
    Ok(())
}

/// Validate a finite slice.
pub(crate) fn validate_finite_slice(values: &[f64]) -> Result<(), ReservingError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(ReservingError::NotFinite);
    }
    Ok(())
}

/// Validate a tail factor under the crate's cumulative-loss production policy
/// (strict: `tail >= 1.0`).
pub(crate) fn validate_tail(tail: f64) -> Result<(), ReservingError> {
    validate_tail_with_policy(tail, MonotonicityPolicy::Strict)
}

/// Validate a tail factor under an explicit [`MonotonicityPolicy`].
///
/// * [`MonotonicityPolicy::Strict`] — `tail` must be finite and `>= 1.0`.
/// * [`MonotonicityPolicy::AllowNegativeIncrements`] — `tail` must be finite
///   and strictly positive; tails below `1.0` (expected net downward incurred
///   development, salvage/subrogation) are permitted.
pub(crate) fn validate_tail_with_policy(
    tail: f64,
    policy: MonotonicityPolicy,
) -> Result<(), ReservingError> {
    if !tail.is_finite() {
        return Err(ReservingError::NotFinite);
    }
    let valid = match policy {
        MonotonicityPolicy::Strict => tail >= 1.0,
        MonotonicityPolicy::AllowNegativeIncrements => tail > 0.0,
    };
    if !valid {
        return Err(ReservingError::InvalidTail { tail });
    }
    Ok(())
}

/// Validate a finite, non-negative slice.
pub(crate) fn validate_nonnegative_finite_slice(values: &[f64]) -> Result<(), ReservingError> {
    for &value in values {
        if !value.is_finite() {
            return Err(ReservingError::NotFinite);
        }
        if value < 0.0 {
            return Err(ReservingError::Negative);
        }
    }
    Ok(())
}

/// Validate a finite, non-negative vector length.
pub(crate) fn validate_nonnegative_finite_vector(
    values: &[f64],
    expected_len: usize,
) -> Result<(), ReservingError> {
    if values.len() != expected_len {
        return Err(ReservingError::ShapeMismatch);
    }
    validate_nonnegative_finite_slice(values)
}

/// Validate a finite, strictly positive slice.
pub(crate) fn validate_positive_finite_slice(
    values: &[f64],
    zero_context: &'static str,
) -> Result<(), ReservingError> {
    for &value in values {
        if !value.is_finite() {
            return Err(ReservingError::NotFinite);
        }
        if value <= 0.0 {
            return Err(ReservingError::DivideByZero(zero_context));
        }
    }
    Ok(())
}

/// Validate LDF/CDF vector shape and positivity with a policy-aware tail check.
pub(crate) fn validate_development_factor_slices_with_policy(
    ldfs: &[f64],
    cdfs: &[f64],
    tail: f64,
    n_devs: usize,
    zero_context: &'static str,
    policy: MonotonicityPolicy,
) -> Result<(), ReservingError> {
    if cdfs.len() != n_devs {
        return Err(ReservingError::ShapeMismatch);
    }
    if ldfs.len() + 1 != cdfs.len() {
        return Err(ReservingError::ShapeMismatch);
    }
    validate_tail_with_policy(tail, policy)?;
    validate_positive_finite_slice(cdfs, zero_context)?;
    validate_positive_finite_slice(ldfs, zero_context)?;
    Ok(())
}

/// Validate latest values and latest development indices against CDF length.
pub(crate) fn validate_latest_and_dev(
    latest: &[f64],
    latest_dev: &[usize],
    cdfs: &[f64],
) -> Result<(), ReservingError> {
    if latest.len() != latest_dev.len() {
        return Err(ReservingError::ShapeMismatch);
    }
    for i in 0..latest.len() {
        if latest_dev[i] >= cdfs.len() {
            return Err(ReservingError::ShapeMismatch);
        }
        if !latest[i].is_finite() {
            return Err(ReservingError::NotFinite);
        }
    }
    Ok(())
}

/// Validate rectangular/ragged cumulative triangle inputs keyed by per-origin
/// observed counts under strict monotonicity (and strict `tail >= 1.0`).
pub(crate) fn validate_observed_count_inputs(
    t: &TriangleView<'_>,
    observed_counts: &[usize],
    tail: f64,
) -> Result<(), ReservingError> {
    validate_observed_count_inputs_with_policy(t, observed_counts, tail, MonotonicityPolicy::Strict)
}

/// Validate rectangular/ragged cumulative triangle inputs keyed by per-origin
/// observed counts under an explicit [`MonotonicityPolicy`].
///
/// Always enforces: flat shape, `observed_counts.len() == n_origins`,
/// `n_devs >= 2`, every count in `1..=n_devs`, and finite non-negative
/// observed cells. [`MonotonicityPolicy::Strict`] additionally rejects
/// decreasing cumulative rows and tails `< 1.0`;
/// [`MonotonicityPolicy::AllowNegativeIncrements`] permits both (tail must
/// still be finite and `> 0`).
pub(crate) fn validate_observed_count_inputs_with_policy(
    t: &TriangleView<'_>,
    observed_counts: &[usize],
    tail: f64,
    policy: MonotonicityPolicy,
) -> Result<(), ReservingError> {
    if t.n_origins.checked_mul(t.n_devs) != Some(t.data.len()) {
        return Err(ReservingError::ShapeMismatch);
    }
    if observed_counts.len() != t.n_origins || t.n_devs < 2 {
        return Err(ReservingError::ShapeMismatch);
    }
    validate_tail_with_policy(tail, policy)?;
    let strict = policy == MonotonicityPolicy::Strict;
    for (origin, &count) in observed_counts.iter().enumerate() {
        if count == 0 || count > t.n_devs {
            return Err(ReservingError::ShapeMismatch);
        }
        let mut previous = None;
        for dev in 0..count {
            let value = t.get(origin, dev);
            if !value.is_finite() {
                return Err(ReservingError::NotFinite);
            }
            if value < 0.0 {
                return Err(ReservingError::Negative);
            }
            if strict
                && let Some(previous_value) = previous
                && value < previous_value
            {
                return Err(ReservingError::NonMonotoneCumulative {
                    origin,
                    dev,
                    previous: previous_value,
                    value,
                });
            }
            previous = Some(value);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_triangle_checks_shape_square_finite_and_sign() {
        let bad_len = TriangleView {
            n_origins: 2,
            n_devs: 2,
            data: &[1.0, 2.0, 3.0],
        };
        assert_eq!(
            validate_triangle(&bad_len),
            Err(ReservingError::ShapeMismatch)
        );

        let rect = TriangleView {
            n_origins: 2,
            n_devs: 3,
            data: &[1.0, 2.0, 3.0, 4.0, 5.0, f64::NAN],
        };
        assert_eq!(validate_triangle(&rect), Err(ReservingError::NotSquare));

        let nan_obs = TriangleView {
            n_origins: 2,
            n_devs: 2,
            data: &[1.0, f64::NAN, 3.0, f64::NAN],
        };
        assert_eq!(validate_triangle(&nan_obs), Err(ReservingError::NotFinite));

        let neg = TriangleView {
            n_origins: 2,
            n_devs: 2,
            data: &[1.0, -2.0, 3.0, f64::NAN],
        };
        assert_eq!(validate_triangle(&neg), Err(ReservingError::Negative));
    }

    #[test]
    fn validate_premium_checks_length_finiteness_and_positivity() {
        assert_eq!(
            validate_premium(&[100.0], 2),
            Err(ReservingError::ShapeMismatch)
        );
        assert_eq!(
            validate_premium(&[100.0, f64::NAN], 2),
            Err(ReservingError::NotFinite)
        );
        assert_eq!(
            validate_premium(&[100.0, 0.0], 2),
            Err(ReservingError::NonPositive("premium"))
        );
        assert_eq!(validate_premium(&[100.0, 200.0], 2), Ok(()));
    }

    #[test]
    fn tail_policy_allows_positive_sub_unity_tail_only_when_permitted() {
        assert_eq!(
            validate_tail_with_policy(0.97, MonotonicityPolicy::Strict),
            Err(ReservingError::InvalidTail { tail: 0.97 })
        );
        assert_eq!(
            validate_tail_with_policy(0.97, MonotonicityPolicy::AllowNegativeIncrements),
            Ok(())
        );
        for bad in [0.0, -0.5] {
            assert_eq!(
                validate_tail_with_policy(bad, MonotonicityPolicy::AllowNegativeIncrements),
                Err(ReservingError::InvalidTail { tail: bad })
            );
        }
        assert_eq!(
            validate_tail_with_policy(f64::INFINITY, MonotonicityPolicy::AllowNegativeIncrements),
            Err(ReservingError::NotFinite)
        );
    }

    #[test]
    fn observed_count_validation_respects_policy() {
        #[rustfmt::skip]
        let data = [
            100.0, 90.0, 95.0,
            120.0, 130.0, f64::NAN,
            140.0, f64::NAN, f64::NAN,
        ];
        let t = TriangleView {
            n_origins: 3,
            n_devs: 3,
            data: &data,
        };
        let counts = [3, 2, 1];
        assert!(matches!(
            validate_observed_count_inputs(&t, &counts, 1.0),
            Err(ReservingError::NonMonotoneCumulative {
                origin: 0,
                dev: 1,
                ..
            })
        ));
        assert_eq!(
            validate_observed_count_inputs_with_policy(
                &t,
                &counts,
                0.98,
                MonotonicityPolicy::AllowNegativeIncrements
            ),
            Ok(())
        );
        assert_eq!(
            validate_observed_count_inputs_with_policy(
                &t,
                &[3, 2, 0],
                1.0,
                MonotonicityPolicy::AllowNegativeIncrements
            ),
            Err(ReservingError::ShapeMismatch)
        );
    }
}
