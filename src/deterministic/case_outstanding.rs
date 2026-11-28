//! Case Outstanding (Paid-to-Case & Case-to-Case) reserving kernel.
//!
//! Implements the Friedland (2010, Ch. 11) / CAS Case Outstanding development
//! technique (corresponding to `chainladder-python`'s `CaseOutstanding`
//! estimator):
//!
//! 1. Given aligned cumulative paid ($P_{i,j}$) and case-outstanding
//!    ($O_{i,j} = I_{i,j} - P_{i,j}$) triangles, compute age-to-age ratios:
//!    - **Paid-to-prior-case ratio** for incremental paid $p_{i,j} = P_{i,j} - P_{i,j-1}$:
//!      $$r^{\text{paid}}_j = \frac{\sum_i p_{i,j}}{\sum_i O_{i,j-1}}$$
//!    - **Case-to-prior-case ratio**:
//!      $$r^{\text{case}}_j = \frac{\sum_i O_{i,j}}{\sum_i O_{i,j-1}}$$
//! 2. Project future incremental paid and future case reserves cell-by-cell
//!    across the lower-right triangle:
//!    $$\hat{p}_{i,j} = \hat{O}_{i,j-1} \cdot r^{\text{paid}}_j, \qquad
//!    \hat{O}_{i,j} = \hat{O}_{i,j-1} \cdot r^{\text{case}}_j.$$
//! 3. Accumulate projected incremental paid (plus any remaining terminal case
//!    reserve if `include_terminal_case` is set) to obtain paid ultimates and
//!    total unpaid reserves.

use serde::{Deserialize, Serialize};

use crate::{
    preprocessing::triangle::{cumulative_to_incremental, latest_diagonal},
    types::TriangleView,
    validate::{
        MonotonicityPolicy, ValidationError, validate_monotone_cumulative, validate_triangle,
    },
};

/// Ratio averaging rule for [`CaseOutstandingOptions`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CaseRatioAverage {
    /// Volume-weighted ratio $\sum_i y_{i,j} / \sum_i O_{i,j-1}$.
    VolumeWeighted,
    /// Unweighted arithmetic mean of cell ratios $y_{i,j} / O_{i,j-1}$.
    Simple,
}

/// Configuration for [`project_case_outstanding`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaseOutstandingOptions {
    /// Averaging method for paid-to-case and case-to-case ratios.
    pub average: CaseRatioAverage,
    /// Optional trailing window length (`Some(k)` uses at most the latest `k`
    /// observed origins at each development age; `None` uses all observed).
    pub last_n: Option<usize>,
    /// Whether to add any residual projected case reserve at the final
    /// development age to the ultimate paid projection (default: `true`,
    /// assuming remaining terminal case reserves eventually pay out 1:1).
    pub include_terminal_case: bool,
    /// Monotonicity policy for validating the cumulative paid triangle.
    pub paid_policy: MonotonicityPolicy,
}

impl Default for CaseOutstandingOptions {
    fn default() -> Self {
        Self {
            average: CaseRatioAverage::VolumeWeighted,
            last_n: None,
            include_terminal_case: true,
            paid_policy: MonotonicityPolicy::AllowNegativeIncrements,
        }
    }
}

/// Output of the Case Outstanding reserving projection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseOutstandingProjection {
    /// Selected paid-to-prior-case ratios for development transitions `j = 1..n_devs`.
    pub paid_to_case_ratios: Vec<f64>,
    /// Selected case-to-prior-case ratios for development transitions `j = 1..n_devs`.
    pub case_to_case_ratios: Vec<f64>,
    /// Full completed cumulative paid triangle (row-major `n_origins × n_devs`).
    pub completed_paid: Vec<f64>,
    /// Full completed case-outstanding triangle (row-major `n_origins × n_devs`).
    pub completed_case: Vec<f64>,
    /// Projected ultimate loss per origin.
    pub ultimate: Vec<f64>,
    /// Projected total unpaid reserve (`ultimate - latest_paid`) per origin.
    pub reserve: Vec<f64>,
    /// Projected IBNR (`reserve - latest_case`) per origin.
    pub ibnr: Vec<f64>,
    /// Sum of `reserve` across all origins.
    pub total_reserve: f64,
    /// Sum of `ibnr` across all origins.
    pub total_ibnr: f64,
}

fn select_ratio(
    numerators: &[f64],
    denominators: &[f64],
    average: CaseRatioAverage,
    last_n: Option<usize>,
) -> f64 {
    let pairs: Vec<(f64, f64)> = numerators
        .iter()
        .zip(denominators.iter())
        .filter(|&(num, den)| num.is_finite() && den.is_finite() && *den > 0.0)
        .map(|(&num, &den)| (num, den))
        .collect();
    if pairs.is_empty() {
        return 0.0;
    }
    let slice = match last_n {
        Some(k) if k > 0 && k < pairs.len() => &pairs[pairs.len() - k..],
        _ => &pairs[..],
    };
    match average {
        CaseRatioAverage::VolumeWeighted => {
            let num_sum: f64 = slice.iter().map(|(n, _)| *n).sum();
            let den_sum: f64 = slice.iter().map(|(_, d)| *d).sum();
            if den_sum > 0.0 {
                num_sum / den_sum
            } else {
                0.0
            }
        }
        CaseRatioAverage::Simple => {
            let ratio_sum: f64 = slice.iter().map(|(n, d)| *n / *d).sum();
            ratio_sum / slice.len() as f64
        }
    }
}

/// Project ultimates and reserves using the Case Outstanding method from
/// aligned cumulative `paid` and `case_outstanding` triangles.
pub fn project_case_outstanding(
    paid: &TriangleView<'_>,
    case_outstanding: &TriangleView<'_>,
    options: &CaseOutstandingOptions,
) -> Result<CaseOutstandingProjection, ValidationError> {
    validate_triangle(paid)?;
    if options.paid_policy == MonotonicityPolicy::Strict {
        validate_monotone_cumulative(paid)?;
    }
    if paid.n_origins != case_outstanding.n_origins
        || paid.n_devs != case_outstanding.n_devs
        || case_outstanding.data.len() != case_outstanding.n_origins * case_outstanding.n_devs
    {
        return Err(ValidationError::ShapeMismatch);
    }
    let n_origins = paid.n_origins;
    let n_devs = paid.n_devs;
    if n_origins < 2 || n_devs < 2 {
        return Err(ValidationError::ShapeMismatch);
    }
    for i in 0..n_origins {
        for j in 0..(n_origins - i) {
            if !case_outstanding.get(i, j).is_finite() {
                return Err(ValidationError::NotFinite);
            }
        }
    }

    let inc_paid = cumulative_to_incremental(paid);
    let mut paid_to_case_ratios = Vec::with_capacity(n_devs - 1);
    let mut case_to_case_ratios = Vec::with_capacity(n_devs - 1);

    for j in 1..n_devs {
        let mut p_nums = Vec::with_capacity(n_origins);
        let mut c_nums = Vec::with_capacity(n_origins);
        let mut dens = Vec::with_capacity(n_origins);
        for i in 0..(n_origins - j) {
            let prev_case = case_outstanding.get(i, j - 1);
            let cur_inc_paid = inc_paid[i * n_devs + j];
            let cur_case = case_outstanding.get(i, j);
            if prev_case.is_finite() && cur_inc_paid.is_finite() && cur_case.is_finite() {
                p_nums.push(cur_inc_paid);
                c_nums.push(cur_case);
                dens.push(prev_case);
            }
        }
        paid_to_case_ratios.push(select_ratio(
            &p_nums,
            &dens,
            options.average,
            options.last_n,
        ));
        case_to_case_ratios.push(select_ratio(
            &c_nums,
            &dens,
            options.average,
            options.last_n,
        ));
    }

    let mut completed_paid = paid.data.to_vec();
    let mut completed_case = case_outstanding.data.to_vec();

    for i in 0..n_origins {
        for j in (n_origins - i)..n_devs {
            let idx = i * n_devs + j;
            let prev_paid = completed_paid[i * n_devs + (j - 1)];
            let prev_case = completed_case[i * n_devs + (j - 1)];
            let r_p = paid_to_case_ratios[j - 1];
            let r_c = case_to_case_ratios[j - 1];
            completed_paid[idx] = prev_paid + prev_case * r_p;
            completed_case[idx] = prev_case * r_c;
        }
    }

    let latest_paid = latest_diagonal(paid);
    let latest_case = latest_diagonal(case_outstanding);
    let mut ultimate = Vec::with_capacity(n_origins);
    let mut reserve = Vec::with_capacity(n_origins);
    let mut ibnr = Vec::with_capacity(n_origins);

    for i in 0..n_origins {
        let end_paid = completed_paid[i * n_devs + (n_devs - 1)];
        let end_case = completed_case[i * n_devs + (n_devs - 1)];
        let ult = if options.include_terminal_case {
            end_paid + end_case
        } else {
            end_paid
        };
        let lp = if latest_paid[i].is_finite() {
            latest_paid[i]
        } else {
            0.0
        };
        let lc = if latest_case[i].is_finite() {
            latest_case[i]
        } else {
            0.0
        };
        let res = ult - lp;
        ultimate.push(ult);
        reserve.push(res);
        ibnr.push(res - lc);
    }

    let total_reserve = reserve.iter().sum();
    let total_ibnr = ibnr.iter().sum();

    Ok(CaseOutstandingProjection {
        paid_to_case_ratios,
        case_to_case_ratios,
        completed_paid,
        completed_case,
        ultimate,
        reserve,
        ibnr,
        total_reserve,
        total_ibnr,
    })
}

/// Convenience wrapper deriving `case_outstanding = incurred - paid` from
/// aligned cumulative `paid` and `incurred` triangles.
pub fn project_from_paid_incurred(
    paid: &TriangleView<'_>,
    incurred: &TriangleView<'_>,
    options: &CaseOutstandingOptions,
) -> Result<CaseOutstandingProjection, ValidationError> {
    validate_triangle(paid)?;
    validate_triangle(incurred)?;
    if paid.n_origins != incurred.n_origins || paid.n_devs != incurred.n_devs {
        return Err(ValidationError::ShapeMismatch);
    }
    let case_data: Vec<f64> = incurred
        .data
        .iter()
        .zip(paid.data.iter())
        .map(|(&inc, &p)| {
            if inc.is_finite() && p.is_finite() {
                inc - p
            } else {
                f64::NAN
            }
        })
        .collect();
    let case_view = TriangleView {
        n_origins: paid.n_origins,
        n_devs: paid.n_devs,
        data: &case_data,
    };
    project_case_outstanding(paid, &case_view, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Triangle;

    #[test]
    fn exact_multiplicative_paid_and_case_recovery() {
        // Construct exact paid & case triangles with constant r_p = 0.5, r_c = 0.4.
        let n = 4;
        let mut paid = vec![f64::NAN; n * n];
        let mut case = vec![f64::NAN; n * n];
        for i in 0..n {
            let mut p = 100.0 * (1.0 + 0.1 * i as f64);
            let mut c = 200.0 * (1.0 + 0.1 * i as f64);
            for j in 0..(n - i) {
                if j > 0 {
                    p += 0.5 * c;
                    c *= 0.4;
                }
                paid[i * n + j] = p;
                case[i * n + j] = c;
            }
        }
        let pt = Triangle::new(n, n, paid);
        let ct = Triangle::new(n, n, case);
        let res =
            project_case_outstanding(&pt.view(), &ct.view(), &CaseOutstandingOptions::default())
                .expect("case outstanding");
        for &rp in &res.paid_to_case_ratios {
            assert!((rp - 0.5).abs() < 1e-12, "rp={rp}");
        }
        for &rc in &res.case_to_case_ratios {
            assert!((rc - 0.4).abs() < 1e-12, "rc={rc}");
        }
        assert!(res.total_reserve > 0.0);
    }
}
