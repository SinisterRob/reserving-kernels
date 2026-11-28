//! Error and result types for `reserving-kernels`.

use std::fmt;

/// Convenience result alias using [`ReservingError`].
pub type Result<T> = core::result::Result<T, ReservingError>;

/// Error modes for triangle validation and reserving kernels.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
#[non_exhaustive]
pub enum ReservingError {
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
    /// Probability or confidence level is outside `[0, 1]`.
    InvalidProbability {
        /// Supplied probability value.
        value: f64,
    },
    /// Simulation or bootstrap iteration count is zero.
    InvalidIterations,
    /// A required strictly positive input was zero or negative.
    NonPositive(&'static str),
    /// Model or matrix specification is invalid or numerically singular.
    InvalidParameter(&'static str),
}

impl fmt::Display for ReservingError {
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
            Self::InvalidProbability { value } => {
                write!(f, "invalid probability {value} (must be in [0, 1])")
            }
            Self::InvalidIterations => write!(f, "iteration count must be at least 1"),
            Self::NonPositive(ctx) => write!(f, "{ctx} must be strictly positive"),
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
        }
    }
}

impl std::error::Error for ReservingError {}
