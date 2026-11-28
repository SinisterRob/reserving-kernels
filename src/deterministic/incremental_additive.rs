//! Incremental Additive reserving kernel (Schmidt 2006 / Friedland).
//!
//! Projects unobserved future incremental losses additively from exposure-weighted
//! incremental loss ratios:
//!
//! 1. Detrend observed incremental losses $Y_{i,j}$ to the latest origin period
//!    using an annual severity/loss trend rate $\tau$:
//!    $$\tilde{Y}_{i,j} = Y_{i,j} (1 + \tau)^{(n-1) - i}.$$
//! 2. Compute the selected incremental-to-exposure factor $\hat{\zeta}_j$ at
//!    each development age $j$:
//!    $$\hat{\zeta}_j = \frac{\sum_{i \in \mathcal{O}_j} \tilde{Y}_{i,j}}{\sum_{i \in \mathcal{O}_j} E_i}$$
//!    (or unweighted simple average of $\tilde{Y}_{i,j} / E_i$).
//! 3. Project unobserved cells $(i, j)$ by re-trending back to origin $i$:
//!    $$\hat{Y}_{i,j} = E_i \, \hat{\zeta}_j \, (1 + \tau)^{i - (n-1)}.$$

use serde::{Deserialize, Serialize};

use crate::{
    ReservingError,
    preprocessing::triangle::{cumulative_to_incremental, latest_diagonal},
    types::TriangleView,
    validate::{MonotonicityPolicy, validate_monotone_cumulative, validate_triangle},
};

/// Averaging rule for incremental-to-exposure ratios.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IncrementalRatioAverage {
    /// Exposure-weighted average $\sum \tilde{Y}_{i,j} / \sum E_i$.
    ExposureWeighted,
    /// Unweighted arithmetic mean of $\tilde{Y}_{i,j} / E_i$.
    Simple,
}

/// Configuration for [`project_incremental_additive`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IncrementalAdditiveOptions {
    /// Per-origin-period trend rate $\tau$ (e.g. `0.03` for 3% annual trend;
    /// `0.0` disables trending).
    pub trend: f64,
    /// Averaging rule for age-specific incremental-to-exposure factors.
    pub average: IncrementalRatioAverage,
    /// Optional trailing window length (`Some(k)` uses at most the latest `k`
    /// observed origins per development age).
    pub last_n: Option<usize>,
    /// Monotonicity policy for validating the input cumulative triangle.
    pub policy: MonotonicityPolicy,
}

impl Default for IncrementalAdditiveOptions {
    fn default() -> Self {
        Self {
            trend: 0.0,
            average: IncrementalRatioAverage::ExposureWeighted,
            last_n: None,
            policy: MonotonicityPolicy::AllowNegativeIncrements,
        }
    }
}

/// Output of the Incremental Additive reserving projection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IncrementalAdditiveProjection {
    /// Selected incremental-to-exposure factors $\hat{\zeta}_j$ (at latest origin trend level).
    pub zeta: Vec<f64>,
    /// Full completed incremental triangle (row-major `n_origins × n_devs`).
    pub completed_incremental: Vec<f64>,
    /// Full completed cumulative triangle (row-major `n_origins × n_devs`).
    pub completed_cumulative: Vec<f64>,
    /// Projected ultimate loss per origin.
    pub ultimate: Vec<f64>,
    /// Projected unpaid reserve (`ultimate - latest`) per origin.
    pub reserve: Vec<f64>,
    /// Sum of `reserve` across all origins.
    pub total_reserve: f64,
}

/// Project ultimates and reserves using the Incremental Additive method from a
/// cumulative loss triangle and per-origin `exposures`.
pub fn project_incremental_additive(
    cumulative: &TriangleView<'_>,
    exposures: &[f64],
    options: &IncrementalAdditiveOptions,
) -> Result<IncrementalAdditiveProjection, ReservingError> {
    validate_triangle(cumulative)?;
    if options.policy == MonotonicityPolicy::Strict {
        validate_monotone_cumulative(cumulative)?;
    }
    let n_origins = cumulative.n_origins;
    let n_devs = cumulative.n_devs;
    if exposures.len() != n_origins {
        return Err(ReservingError::ShapeMismatch);
    }
    if !options.trend.is_finite() || options.trend <= -1.0 {
        return Err(ReservingError::InvalidParameter(
            "trend must be finite and greater than -1.0",
        ));
    }
    for &e in exposures {
        if !e.is_finite() || e <= 0.0 {
            return Err(ReservingError::InvalidParameter(
                "exposures must be finite and strictly positive",
            ));
        }
    }

    let inc = cumulative_to_incremental(cumulative);
    let factor = 1.0 + options.trend;
    let mut zeta = vec![0.0; n_devs];

    for j in 0..n_devs {
        let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(n_origins);
        for i in 0..n_origins {
            let y = inc[i * n_devs + j];
            if y.is_finite() {
                let detrend = factor.powi((n_origins - 1 - i) as i32);
                pairs.push((y * detrend, exposures[i]));
            }
        }
        if pairs.is_empty() {
            return Err(ReservingError::InvalidParameter(
                "incremental additive factor unestimable: no observed incremental loss at a \
                 development age",
            ));
        }
        let slice = match options.last_n {
            Some(k) if k > 0 && k < pairs.len() => &pairs[pairs.len() - k..],
            _ => &pairs[..],
        };
        let z = match options.average {
            IncrementalRatioAverage::ExposureWeighted => {
                let num: f64 = slice.iter().map(|(y, _)| *y).sum();
                let den: f64 = slice.iter().map(|(_, e)| *e).sum();
                if !(den > 0.0) {
                    return Err(ReservingError::DivideByZero(
                        "incremental additive exposure sum",
                    ));
                }
                num / den
            }
            IncrementalRatioAverage::Simple => {
                let sum: f64 = slice.iter().map(|(y, e)| *y / *e).sum();
                sum / slice.len() as f64
            }
        };
        if !z.is_finite() {
            return Err(ReservingError::NotFinite);
        }
        zeta[j] = z;
    }

    let mut completed_incremental = inc;
    for i in 0..n_origins {
        let retrend = factor.powi(i as i32 - (n_origins - 1) as i32);
        for j in (n_origins - i)..n_devs {
            let idx = i * n_devs + j;
            completed_incremental[idx] = exposures[i] * zeta[j] * retrend;
        }
    }

    let mut completed_cumulative = vec![0.0; n_origins * n_devs];
    let mut ultimate = Vec::with_capacity(n_origins);
    let latest = latest_diagonal(cumulative);
    let mut reserve = Vec::with_capacity(n_origins);

    for i in 0..n_origins {
        let mut acc = 0.0;
        for j in 0..n_devs {
            acc += completed_incremental[i * n_devs + j];
            completed_cumulative[i * n_devs + j] = acc;
        }
        ultimate.push(acc);
        // Latest diagonal cells are observed and were validated finite above.
        reserve.push(acc - latest[i]);
    }

    let total_reserve: f64 = reserve.iter().sum();
    if !total_reserve.is_finite()
        || ultimate.iter().any(|v| !v.is_finite())
        || reserve.iter().any(|v| !v.is_finite())
    {
        return Err(ReservingError::NotFinite);
    }

    Ok(IncrementalAdditiveProjection {
        zeta,
        completed_incremental,
        completed_cumulative,
        ultimate,
        reserve,
        total_reserve,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Triangle;

    #[test]
    fn exact_recovery_with_trend() {
        let n = 5;
        let exposures = vec![1000.0, 1100.0, 1200.0, 1300.0, 1400.0];
        let base_zeta = [0.30, 0.20, 0.10, 0.05, 0.02];
        let trend = 0.05_f64;
        let mut data = vec![f64::NAN; n * n];
        let mut expected_res = 0.0;

        for i in 0..n {
            let tr = (1.0 + trend).powi(i as i32 - (n - 1) as i32);
            let mut cum = 0.0;
            for j in 0..n {
                let y = exposures[i] * base_zeta[j] * tr;
                if i + j < n {
                    cum += y;
                    data[i * n + j] = cum;
                } else {
                    expected_res += y;
                }
            }
        }

        let tri = Triangle::new(n, n, data);
        let opts = IncrementalAdditiveOptions {
            trend,
            ..Default::default()
        };
        let res = project_incremental_additive(&tri.view(), &exposures, &opts).expect("inc add");
        for (a, b) in res.zeta.iter().zip(base_zeta.iter()) {
            assert!((a - b).abs() < 1e-12, "zeta mismatch: {a} vs {b}");
        }
        assert!(
            (res.total_reserve - expected_res).abs() < 1e-8,
            "reserve={} vs expected={}",
            res.total_reserve,
            expected_res
        );
    }
}
