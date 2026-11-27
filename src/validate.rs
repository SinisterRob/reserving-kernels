//! Input validation — kept deliberately off the compute hot path.
//!
//! Call these at the ingress boundary (CSV/DB/API → `Triangle`). Kernels in
//! the rest of the crate assume validated inputs and skip defensive checks.

use std::fmt;

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

/// Validation failure modes for triangle inputs.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
#[non_exhaustive]
pub enum ValidationError {
    /// `data.len() != n_origins * n_devs`.
    ShapeMismatch,
    /// Triangle must be square (`n_origins == n_devs`) for the reserving
    /// kernels as implemented here.
    NotSquare,
    /// An observed cell held a non-finite value other than `NaN`.
    NotFinite,
    /// An observed cell held a negative cumulative loss.
    Negative,
    /// A required denominator was zero (column sum or observed-cell value).
    DivideByZero(&'static str),
    /// An observed cumulative row decreased.
    NonMonotoneCumulative {
        /// Origin row index.
        origin: usize,
        /// Development index holding the lower value.
        dev: usize,
        /// Previous cumulative value.
        previous: f64,
        /// Current cumulative value.
        value: f64,
    },
    /// Tail factor is below the supported cumulative-development minimum.
    InvalidTail {
        /// Supplied tail factor.
        tail: f64,
    },
    /// Maturity percentage is outside `[0, 1]`.
    InvalidMaturity {
        /// Maturity vector index.
        index: usize,
        /// Supplied maturity value.
        value: f64,
    },
    /// Model or matrix specification is invalid or numerically singular.
    InvalidParameter(&'static str),
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch => write!(f, "input dimensions do not match expected shape"),
            Self::NotSquare => write!(f, "triangle must be square (n_origins == n_devs)"),
            Self::NotFinite => write!(f, "input contains NaN or infinite values"),
            Self::Negative => write!(
                f,
                "input contains negative values where non-negative required"
            ),
            Self::DivideByZero(ctx) => write!(f, "division by zero in {ctx}"),
            Self::NonMonotoneCumulative {
                origin,
                dev,
                previous,
                value,
            } => write!(
                f,
                "non-monotone cumulative triangle at origin {origin}, dev {dev}: {value} < {previous}"
            ),
            Self::InvalidTail { tail } => write!(f, "invalid tail factor {tail} (must be >= 1.0)"),
            Self::InvalidMaturity { index, value } => {
                write!(f, "invalid maturity/probability {value} at index {index}")
            }
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
        }
    }
}

impl std::error::Error for ValidationError {}

/// Validate a triangle view: square shape, finite non-negative observed
/// cells. Unobserved cells (`i + j > n − 1`) are ignored.
pub fn validate_triangle(t: &TriangleView<'_>) -> Result<(), ValidationError> {
    if t.data.len() != t.n_origins * t.n_devs {
        return Err(ValidationError::ShapeMismatch);
    }
    if t.n_origins != t.n_devs {
        return Err(ValidationError::NotSquare);
    }
    let n = t.n_origins;
    for i in 0..n {
        let row = &t.data[i * n..i * n + (n - i)];
        for &v in row {
            if !v.is_finite() {
                // NaN is our "unobserved" sentinel — within the observed
                // upper triangle, NaN or infinity is an error.
                return Err(ValidationError::NotFinite);
            }
            if v < 0.0 {
                return Err(ValidationError::Negative);
            }
        }
    }
    Ok(())
}

/// Validate a premium vector: length matches origins, finite positive.
pub fn validate_premium(premium: &[f64], n_origins: usize) -> Result<(), ValidationError> {
    if premium.len() != n_origins {
        return Err(ValidationError::ShapeMismatch);
    }
    for &p in premium {
        if !p.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if p <= 0.0 {
            return Err(ValidationError::DivideByZero("premium"));
        }
    }
    Ok(())
}

/// Validate that observed cumulative rows are non-decreasing.
pub(crate) fn validate_monotone_cumulative(t: &TriangleView<'_>) -> Result<(), ValidationError> {
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
                return Err(ValidationError::NonMonotoneCumulative {
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
pub(crate) fn validate_finite_slice(values: &[f64]) -> Result<(), ValidationError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(ValidationError::NotFinite);
    }
    Ok(())
}

/// Validate a tail factor under the crate's cumulative-loss production policy.
pub(crate) fn validate_tail(tail: f64) -> Result<(), ValidationError> {
    if !tail.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if tail < 1.0 {
        return Err(ValidationError::InvalidTail { tail });
    }
    Ok(())
}

/// Validate a finite, non-negative slice.
pub(crate) fn validate_nonnegative_finite_slice(values: &[f64]) -> Result<(), ValidationError> {
    for &value in values {
        if !value.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if value < 0.0 {
            return Err(ValidationError::Negative);
        }
    }
    Ok(())
}

/// Validate a finite, non-negative vector length.
pub(crate) fn validate_nonnegative_finite_vector(
    values: &[f64],
    expected_len: usize,
) -> Result<(), ValidationError> {
    if values.len() != expected_len {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_nonnegative_finite_slice(values)
}

/// Validate a finite, strictly positive slice.
pub(crate) fn validate_positive_finite_slice(
    values: &[f64],
    zero_context: &'static str,
) -> Result<(), ValidationError> {
    for &value in values {
        if !value.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if value <= 0.0 {
            return Err(ValidationError::DivideByZero(zero_context));
        }
    }
    Ok(())
}

/// Validate LDF/CDF vector shape and positivity.
pub(crate) fn validate_development_factor_slices(
    ldfs: &[f64],
    cdfs: &[f64],
    tail: f64,
    n_devs: usize,
    zero_context: &'static str,
) -> Result<(), ValidationError> {
    if cdfs.len() != n_devs {
        return Err(ValidationError::ShapeMismatch);
    }
    if ldfs.len() + 1 != cdfs.len() {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_tail(tail)?;
    validate_positive_finite_slice(cdfs, zero_context)?;
    validate_positive_finite_slice(ldfs, zero_context)?;
    Ok(())
}

/// Validate latest values and latest development indices against CDF length.
pub(crate) fn validate_latest_and_dev(
    latest: &[f64],
    latest_dev: &[usize],
    cdfs: &[f64],
) -> Result<(), ValidationError> {
    if latest.len() != latest_dev.len() {
        return Err(ValidationError::ShapeMismatch);
    }
    for i in 0..latest.len() {
        if latest_dev[i] >= cdfs.len() {
            return Err(ValidationError::ShapeMismatch);
        }
        if !latest[i].is_finite() {
            return Err(ValidationError::NotFinite);
        }
    }
    Ok(())
}

/// Validate rectangular/ragged cumulative triangle inputs keyed by per-origin observed counts.
pub(crate) fn validate_observed_count_inputs(
    t: &TriangleView<'_>,
    observed_counts: &[usize],
    tail: f64,
) -> Result<(), ValidationError> {
    if t.data.len() != t.n_origins * t.n_devs {
        return Err(ValidationError::ShapeMismatch);
    }
    if observed_counts.len() != t.n_origins || t.n_devs < 2 {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_tail(tail)?;
    for (origin, &count) in observed_counts.iter().enumerate() {
        if count == 0 || count > t.n_devs {
            return Err(ValidationError::ShapeMismatch);
        }
        let mut previous = None;
        for dev in 0..count {
            let value = t.get(origin, dev);
            if !value.is_finite() {
                return Err(ValidationError::NotFinite);
            }
            if value < 0.0 {
                return Err(ValidationError::Negative);
            }
            if let Some(previous_value) = previous
                && value < previous_value
            {
                return Err(ValidationError::NonMonotoneCumulative {
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
