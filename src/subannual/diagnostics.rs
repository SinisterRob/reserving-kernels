//! Sub-annual interpolation diagnostics and consistency checks.
//!
//! * [`equilibrium_ibnr_check`] — Robbin (2018): tests an interpolated
//!   sub-annual development pattern against a steady-state exposure stream to
//!   verify monotonic IBNR runoff with no artificial sawtooth oscillations.
//! * [`subannual_product_parity`] — Internal / Identity: verifies that the
//!   product of interpolated sub-annual LDFs across each 12-month window
//!   matches the source annual LDF within floating-point tolerance.

use crate::validate::{ValidationError, validate_positive_finite_slice};

/// Output of [`equilibrium_ibnr_check`] (Robbin 2018).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EquilibriumIbnrCheckResult {
    /// Sub-period emergence rates `(P(t_k) - P(t_{k-1})) / (t_k - t_{k-1})` (length `m - 1`).
    pub emergence_rates: Vec<f64>,
    /// Number of upward jumps (`rate[k] > rate[k - 1] + tolerance`) indicating
    /// sawtooth oscillation in expected IBNR runoff.
    pub oscillation_count: usize,
    /// Largest upward jump `max_k (rate[k] - rate[k - 1])_+` (0.0 if perfectly monotonic).
    pub max_upward_jump: f64,
    /// Minimum emergence rate observed (negative values indicate non-monotone CDFs).
    pub min_emergence_rate: f64,
    /// Total steady-state IBNR across the observed sub-annual maturity grid
    /// (per unit ultimate loss per sub-period).
    pub steady_state_ibnr: f64,
    /// Steady-state IBNR by intra-year calendar phase (`0..sub_periods_per_year`).
    pub phase_ibnr: Vec<f64>,
    /// Relative sawtooth amplitude `(max(phase_ibnr) - min(phase_ibnr)) / mean(phase_ibnr)`.
    pub relative_phase_oscillation: f64,
    /// `true` iff `oscillation_count == 0` (monotonically non-increasing emergence rates).
    pub is_monotonic_runoff: bool,
    /// `true` iff `is_monotonic_runoff && min_emergence_rate >= -tolerance`.
    pub passed: bool,
}

/// Test an interpolated sub-annual development pattern against a steady-state
/// exposure stream to verify monotonic IBNR runoff without artificial sawtooth
/// oscillations (Robbin 2018).
///
/// # Parameters
/// * `subannual_ages_months` — strictly increasing sub-annual ages in months (`len >= 2`).
/// * `subannual_cdfs` — interpolated CDFs at each sub-annual age (`len == subannual_ages_months.len()`).
/// * `sub_periods_per_year` — number of sub-periods per year (e.g., `2`, `4`, or `12`).
/// * `tolerance` — numerical tolerance (`>= 0.0`) for monotonicity checks (e.g. `1e-9`).
pub fn equilibrium_ibnr_check(
    subannual_ages_months: &[f64],
    subannual_cdfs: &[f64],
    sub_periods_per_year: usize,
    tolerance: f64,
) -> Result<EquilibriumIbnrCheckResult, ValidationError> {
    if subannual_ages_months.len() < 2
        || subannual_ages_months.len() != subannual_cdfs.len()
        || sub_periods_per_year == 0
    {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_positive_finite_slice(subannual_ages_months, "subannual_ages_months")?;
    validate_positive_finite_slice(subannual_cdfs, "subannual_cdfs")?;
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(ValidationError::InvalidParameter(
            "tolerance must be finite and non-negative",
        ));
    }
    for k in 1..subannual_ages_months.len() {
        if subannual_ages_months[k] <= subannual_ages_months[k - 1] {
            return Err(ValidationError::InvalidParameter(
                "subannual_ages_months must be strictly increasing",
            ));
        }
    }

    let pct_reported: Vec<f64> = subannual_cdfs.iter().map(|&c| 1.0 / c).collect();
    let mut emergence_rates = Vec::with_capacity(pct_reported.len() - 1);
    let mut min_emergence_rate = f64::INFINITY;
    for k in 1..pct_reported.len() {
        let dt = subannual_ages_months[k] - subannual_ages_months[k - 1];
        let rate = (pct_reported[k] - pct_reported[k - 1]) / dt;
        if rate < min_emergence_rate {
            min_emergence_rate = rate;
        }
        emergence_rates.push(rate);
    }

    let mut oscillation_count = 0usize;
    let mut max_upward_jump = 0.0_f64;
    for k in 1..emergence_rates.len() {
        let jump = emergence_rates[k] - emergence_rates[k - 1];
        if jump > max_upward_jump {
            max_upward_jump = jump;
        }
        if jump > tolerance {
            oscillation_count += 1;
        }
    }

    // Steady-state IBNR under unit ultimate exposure per sub-period.
    let mut steady_state_ibnr = 0.0;
    let mut phase_ibnr = vec![0.0; sub_periods_per_year];
    for (k, &p) in pct_reported.iter().enumerate() {
        let rem = (1.0 - p).max(0.0);
        steady_state_ibnr += rem;
        phase_ibnr[k % sub_periods_per_year] += rem;
    }

    let phase_mean = phase_ibnr.iter().sum::<f64>() / sub_periods_per_year as f64;
    let phase_min = phase_ibnr.iter().copied().fold(f64::INFINITY, f64::min);
    let phase_max = phase_ibnr.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let relative_phase_oscillation = if phase_mean > 0.0 {
        (phase_max - phase_min) / phase_mean
    } else {
        0.0
    };

    let is_monotonic_runoff = oscillation_count == 0;
    let passed = is_monotonic_runoff && min_emergence_rate >= -tolerance;

    Ok(EquilibriumIbnrCheckResult {
        emergence_rates,
        oscillation_count,
        max_upward_jump,
        min_emergence_rate,
        steady_state_ibnr,
        phase_ibnr,
        relative_phase_oscillation,
        is_monotonic_runoff,
        passed,
    })
}

/// Product-parity diagnostic for a single annual window `[start_age_months, end_age_months]`.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AnnualWindowParity {
    /// Window index (`0..annual_ldfs.len()`).
    pub window_index: usize,
    /// Window start age in months.
    pub start_age_months: f64,
    /// Window end age in months.
    pub end_age_months: f64,
    /// Source annual LDF across this window.
    pub annual_ldf: f64,
    /// Product of interpolated sub-annual LDFs spanning this window.
    pub subannual_product: f64,
    /// Number of sub-annual LDF steps multiplied in this window.
    pub n_sub_steps: usize,
    /// Absolute difference `|subannual_product - annual_ldf|`.
    pub abs_error: f64,
    /// Relative difference `abs_error / |annual_ldf|`.
    pub rel_error: f64,
    /// `true` iff `abs_error <= tolerance`.
    pub within_tolerance: bool,
}

/// Output of [`subannual_product_parity`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SubannualProductParityResult {
    /// Per-window parity diagnostics.
    pub windows: Vec<AnnualWindowParity>,
    /// Maximum absolute error across all windows.
    pub max_abs_error: f64,
    /// Maximum relative error across all windows.
    pub max_rel_error: f64,
    /// `true` iff every annual window is within `tolerance`.
    pub passed: bool,
}

/// Verify that the product of interpolated sub-annual LDFs across each annual
/// window `[annual_ages_months[k], annual_ages_months[k + 1]]` matches the
/// source `annual_ldfs[k]` within `tolerance`.
pub fn subannual_product_parity(
    annual_ages_months: &[f64],
    annual_ldfs: &[f64],
    subannual_ages_months: &[f64],
    subannual_ldfs: &[f64],
    tolerance: f64,
) -> Result<SubannualProductParityResult, ValidationError> {
    if annual_ages_months.len() < 2
        || annual_ldfs.len() + 1 != annual_ages_months.len()
        || subannual_ages_months.len() < 2
        || subannual_ldfs.len() + 1 != subannual_ages_months.len()
    {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_positive_finite_slice(annual_ages_months, "annual_ages_months")?;
    validate_positive_finite_slice(annual_ldfs, "annual_ldfs")?;
    validate_positive_finite_slice(subannual_ages_months, "subannual_ages_months")?;
    validate_positive_finite_slice(subannual_ldfs, "subannual_ldfs")?;
    if annual_ages_months.windows(2).any(|w| w[1] <= w[0]) {
        return Err(ValidationError::InvalidParameter(
            "annual_ages_months must be strictly increasing",
        ));
    }
    if subannual_ages_months.windows(2).any(|w| w[1] <= w[0]) {
        return Err(ValidationError::InvalidParameter(
            "subannual_ages_months must be strictly increasing",
        ));
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(ValidationError::InvalidParameter(
            "tolerance must be finite and non-negative",
        ));
    }

    let age_tol = 1e-6;
    let mut windows = Vec::with_capacity(annual_ldfs.len());
    let mut max_abs_error = 0.0_f64;
    let mut max_rel_error = 0.0_f64;
    let mut all_passed = true;

    for (w_idx, &ann_ldf) in annual_ldfs.iter().enumerate() {
        let t_start = annual_ages_months[w_idx];
        let t_end = annual_ages_months[w_idx + 1];
        let idx_start = subannual_ages_months
            .iter()
            .position(|&a| (a - t_start).abs() <= age_tol)
            .ok_or(ValidationError::InvalidParameter(
                "subannual_ages_months must include every annual knot age",
            ))?;
        let idx_end = subannual_ages_months
            .iter()
            .position(|&a| (a - t_end).abs() <= age_tol)
            .ok_or(ValidationError::InvalidParameter(
                "subannual_ages_months must include every annual knot age",
            ))?;
        if idx_end <= idx_start {
            return Err(ValidationError::InvalidParameter(
                "annual window endpoints must be strictly increasing in subannual_ages_months",
            ));
        }

        let mut prod = 1.0;
        for j in idx_start..idx_end {
            prod *= subannual_ldfs[j];
        }
        let abs_error = (prod - ann_ldf).abs();
        let rel_error = abs_error / ann_ldf.abs().max(1e-15);
        let within_tolerance = abs_error <= tolerance;
        if abs_error > max_abs_error {
            max_abs_error = abs_error;
        }
        if rel_error > max_rel_error {
            max_rel_error = rel_error;
        }
        if !within_tolerance {
            all_passed = false;
        }

        windows.push(AnnualWindowParity {
            window_index: w_idx,
            start_age_months: t_start,
            end_age_months: t_end,
            annual_ldf: ann_ldf,
            subannual_product: prod,
            n_sub_steps: idx_end - idx_start,
            abs_error,
            rel_error,
            within_tolerance,
        });
    }

    Ok(SubannualProductParityResult {
        windows,
        max_abs_error,
        max_rel_error,
        passed: all_passed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subannual::interp::{
        SubannualInterpMethod, interp_exponential, interp_swimon, interpolate_pattern,
    };

    #[test]
    fn swimon_passes_equilibrium_ibnr_check_while_exponential_oscillates() {
        let annual_ages = [12.0, 24.0, 36.0, 48.0, 60.0];
        let annual_cdfs = [2.50, 1.45, 1.18, 1.06, 1.00];
        let monthly_ages: Vec<f64> = (12..=60).map(f64::from).collect();

        let sw = interp_swimon(&annual_ages, &annual_cdfs, &monthly_ages).unwrap();
        let sw_check = equilibrium_ibnr_check(&sw.ages_months, &sw.cdfs, 12, 1e-9).unwrap();
        assert!(
            sw_check.passed,
            "SWIMON should pass equilibrium check: {sw_check:?}"
        );
        assert_eq!(sw_check.oscillation_count, 0);

        let ex = interp_exponential(&annual_ages, &annual_cdfs, &monthly_ages).unwrap();
        let ex_check = equilibrium_ibnr_check(&ex.ages_months, &ex.cdfs, 12, 1e-9).unwrap();
        assert!(
            !ex_check.passed && ex_check.oscillation_count > 0,
            "Exponential CDF should exhibit intra-year sawtooth oscillations"
        );
    }

    #[test]
    fn all_interpolators_satisfy_subannual_product_parity() {
        let annual_ages = [12.0, 24.0, 36.0, 48.0, 60.0];
        let annual_cdfs = [2.50, 1.45, 1.18, 1.06, 1.00];
        let annual_ldfs: Vec<f64> = annual_cdfs.windows(2).map(|w| w[0] / w[1]).collect();
        let quarterly_ages: Vec<f64> = (4..=20).map(|q| f64::from(q * 3)).collect();

        for method in [
            SubannualInterpMethod::LinearCdf,
            SubannualInterpMethod::LinearPctReported,
            SubannualInterpMethod::Exponential,
            SubannualInterpMethod::InversePower,
            SubannualInterpMethod::Swimon,
            SubannualInterpMethod::SplicedWeibull,
            SubannualInterpMethod::Ivf12,
        ] {
            let pat = interpolate_pattern(method, &annual_ages, &annual_cdfs, &quarterly_ages)
                .expect("interp");
            let parity = subannual_product_parity(
                &annual_ages,
                &annual_ldfs,
                &pat.ages_months,
                &pat.ldfs,
                1e-9,
            )
            .expect("parity");
            assert!(
                parity.passed,
                "{method:?} failed product parity: max_abs_error={}",
                parity.max_abs_error
            );
        }
    }
}
