//! Standalone frequency–severity reserving kernels.
//!
//! Implements two classic P&C frequency–severity methods designed to handle
//! settlement-speed shifts and claim-severity inflation:
//!
//! * **Friedland Method 2 / Adler–Kline (1988)** — projects future closed claim
//!   counts from selected cumulative disposal rates (`C_{i,j} / N_hat_i`) and
//!   multiplies by trend-adjusted incremental paid-per-closed-claim severities.
//! * **Wright's Operational Time Reserving Model (1990)** — replaces calendar
//!   development age with operational time `τ_{i,j} = C_{i,j} / N_hat_i` and
//!   fits a claim-count-weighted Gamma log-link GLM for incremental severity as
//!   a Hoerl curve in midpoint operational time plus calendar-year inflation.

use crate::linalg::{cholesky_lower, solve_general, solve_lower_into};
use crate::types::TriangleView;
use crate::validate::{ValidationError, validate_triangle};

// ===========================================================================
// 2.1 Friedland Method 2 / Adler–Kline (1988)
// ===========================================================================

/// Convention for projecting future incremental closed claim counts from
/// selected cumulative disposal rates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum CountProjectionMode {
    /// Rescale remaining future disposal increments proportionally so that
    /// `Σ_{k >= j} c_hat_{i,k} = N_hat_i - C_{i, j-1}` holds exactly for every
    /// origin `i` (guarantees exact claim-count conservation).
    #[default]
    ExactConservation,
    /// Anchor the first future cell to actual closed counts
    /// `N_hat_i * (d_hat_j - d_{i, j-1})` and use `N_hat_i * (d_hat_k - d_hat_{k-1})`
    /// for subsequent lags.
    AnchoredToActual,
}

/// Configuration for [`adler_kline_reserve`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AdlerKlineOptions {
    /// Annual accident-year severity trend rate `g > -1.0`.
    pub severity_trend: f64,
    /// Optional trailing window length for averaging observed disposal rates
    /// per lag (`None` = all observed origins at that lag).
    pub disposal_averaging_window: Option<usize>,
    /// Optional trailing window length for averaging trend-adjusted severities
    /// per lag (`None` = all observed origins with `c_{i,j} > 0`).
    pub severity_averaging_window: Option<usize>,
    /// Count-projection convention for future lags.
    pub count_projection_mode: CountProjectionMode,
    /// Optional trend-adjusted tail severity (at the latest accident-year level)
    /// applied to remaining counts when `d_hat_J < 1.0`. Defaults to the last
    /// lag's selected trended severity.
    pub tail_severity: Option<f64>,
    /// If `true` (default), clamp selected trended severities to `>= 0.0`.
    pub clamp_nonnegative_severity: bool,
}

impl Default for AdlerKlineOptions {
    fn default() -> Self {
        Self {
            severity_trend: 0.0,
            disposal_averaging_window: None,
            severity_averaging_window: None,
            count_projection_mode: CountProjectionMode::ExactConservation,
            tail_severity: None,
            clamp_nonnegative_severity: true,
        }
    }
}

/// Output of [`adler_kline_reserve`] (Friedland Method 2 / Adler–Kline 1988).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AdlerKlineResult {
    /// Observed cumulative disposal rates `d_{i,j} = C_{i,j} / N_hat_i` (`n × n`, `NaN` in future).
    pub observed_disposal_rates: Vec<f64>,
    /// Selected monotone cumulative disposal rates `d_hat_j` (length `n`, in `[0, 1]`).
    pub selected_disposal_rates: Vec<f64>,
    /// Selected trend-adjusted incremental severities `s_tilde_hat_j` at the
    /// latest accident-year level (length `n`).
    pub selected_trended_severities: Vec<f64>,
    /// Trend-adjusted tail severity used when `d_hat_J < 1.0`.
    pub tail_severity: f64,
    /// Full `n × n` incremental closed count grid (observed upper + projected lower).
    pub projected_closed_counts: Vec<f64>,
    /// Projected tail closed counts per origin (`N_hat_i * (1 - d_hat_J)` or rescaled).
    pub tail_closed_counts: Vec<f64>,
    /// Indicated reserve per origin year.
    pub reserves: Vec<f64>,
    /// Total indicated reserve across all origins.
    pub total_reserve: f64,
}

fn validate_square_incremental_paid(t: &TriangleView<'_>) -> Result<(), ValidationError> {
    if t.data.len() != t.n_origins * t.n_devs {
        return Err(ValidationError::ShapeMismatch);
    }
    if t.n_origins != t.n_devs {
        return Err(ValidationError::NotSquare);
    }
    let n = t.n_origins;
    for i in 0..n {
        for j in 0..(n - i) {
            if !t.get(i, j).is_finite() {
                return Err(ValidationError::NotFinite);
            }
        }
    }
    Ok(())
}

/// Fill lags without any observed closings: carry the prior lag's selected
/// severity forward; leading empty lags take the first observed lag's value.
/// Errors when no lag has any observed closing.
fn fill_missing_lag_severities(lags: &[Option<f64>]) -> Result<Vec<f64>, ValidationError> {
    let Some(first) = lags.iter().find_map(|v| *v) else {
        return Err(ValidationError::InvalidParameter(
            "Adler-Kline severity unestimable: no lag has a positive incremental closed count",
        ));
    };
    let mut prev = first;
    Ok(lags
        .iter()
        .map(|v| {
            if let Some(s) = *v {
                prev = s;
            }
            prev
        })
        .collect())
}

/// Friedland Method 2 / Adler–Kline (1988) frequency–severity reserving kernel.
///
/// Lags with no positive incremental closed count in any observed origin
/// carry the previous lag's selected severity forward (leading such lags
/// take the first observed lag's severity); if no lag has any closing the
/// function errors. Observed disposal rates are clamped to `[0, 1]`
/// (closed counts above the selected ultimate count are treated as fully
/// closed).
///
/// * `cumulative_closed_counts` — square `n × n` cumulative closed claim count triangle.
/// * `incremental_paid_loss` — square `n × n` incremental paid loss triangle
///   (negative incrementals are permitted for salvage/subrogation).
/// * `ultimate_counts` — pre-computed ultimate claim counts `N_hat_i > 0` (length `n`).
pub fn adler_kline_reserve(
    cumulative_closed_counts: &TriangleView<'_>,
    incremental_paid_loss: &TriangleView<'_>,
    ultimate_counts: &[f64],
    options: AdlerKlineOptions,
) -> Result<AdlerKlineResult, ValidationError> {
    validate_triangle(cumulative_closed_counts)?;
    validate_square_incremental_paid(incremental_paid_loss)?;
    let n = cumulative_closed_counts.n_origins;
    if incremental_paid_loss.n_origins != n || ultimate_counts.len() != n {
        return Err(ValidationError::ShapeMismatch);
    }
    if n < 2 {
        return Err(ValidationError::InvalidParameter(
            "Adler-Kline requires at least a 2x2 triangle",
        ));
    }
    if !options.severity_trend.is_finite() || options.severity_trend <= -1.0 {
        return Err(ValidationError::InvalidParameter(
            "severity_trend must be finite and > -1.0",
        ));
    }
    if let Some(ts) = options.tail_severity
        && !ts.is_finite()
    {
        return Err(ValidationError::NotFinite);
    }
    for &u in ultimate_counts {
        if !u.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if u <= 0.0 {
            return Err(ValidationError::NonPositive("ultimate_counts"));
        }
    }

    // 1. Observed disposal rates d_{i,j} = C_{i,j} / N_hat_i.
    let mut observed_disposal_rates = vec![f64::NAN; n * n];
    for i in 0..n {
        let u = ultimate_counts[i];
        for j in 0..(n - i) {
            observed_disposal_rates[i * n + j] =
                (cumulative_closed_counts.get(i, j) / u).clamp(0.0, 1.0);
        }
    }

    // Select monotone cumulative disposal rates d_hat_j.
    let mut selected_disposal_rates = vec![0.0; n];
    let mut prev_d = 0.0;
    for j in 0..n {
        let count_rows = n - j;
        let start = match options.disposal_averaging_window {
            Some(w) if w > 0 && w < count_rows => count_rows - w,
            _ => 0,
        };
        let mut sum = 0.0;
        let mut k = 0usize;
        for i in start..count_rows {
            sum += observed_disposal_rates[i * n + j];
            k += 1;
        }
        let avg = (sum / k as f64).clamp(prev_d, 1.0);
        selected_disposal_rates[j] = avg;
        prev_d = avg;
    }

    // 2. Trend-adjusted incremental severities s_tilde_{i,j} at latest AY (i = n - 1).
    let growth = 1.0 + options.severity_trend;
    let mut lag_severities: Vec<Option<f64>> = vec![None; n];
    let mut projected_closed_counts = vec![0.0; n * n];

    for j in 0..n {
        let count_rows = n - j;
        let mut valid_sevs: Vec<f64> = Vec::with_capacity(count_rows);
        for i in 0..count_rows {
            let c_cum = cumulative_closed_counts.get(i, j);
            let c_prev = if j == 0 {
                0.0
            } else {
                cumulative_closed_counts.get(i, j - 1)
            };
            let inc_c = c_cum - c_prev;
            projected_closed_counts[i * n + j] = inc_c;
            if inc_c > 0.0 {
                let s_ij = incremental_paid_loss.get(i, j) / inc_c;
                let trend_exp = (n - 1 - i) as i32;
                valid_sevs.push(s_ij * growth.powi(trend_exp));
            }
        }
        if !valid_sevs.is_empty() {
            let start = match options.severity_averaging_window {
                Some(w) if w > 0 && w < valid_sevs.len() => valid_sevs.len() - w,
                _ => 0,
            };
            let slice = &valid_sevs[start..];
            let sel = slice.iter().sum::<f64>() / slice.len() as f64;
            if !sel.is_finite() {
                return Err(ValidationError::NotFinite);
            }
            lag_severities[j] = Some(if options.clamp_nonnegative_severity {
                sel.max(0.0)
            } else {
                sel
            });
        }
    }
    let selected_trended_severities = fill_missing_lag_severities(&lag_severities)?;

    let tail_severity = options
        .tail_severity
        .unwrap_or(selected_trended_severities[n - 1]);
    let tail_severity = if options.clamp_nonnegative_severity {
        tail_severity.max(0.0)
    } else {
        tail_severity
    };

    // 3. Project future incremental closed counts and reserves.
    let mut tail_closed_counts = vec![0.0; n];
    let mut reserves = vec![0.0; n];
    let mut total_reserve = 0.0;

    for i in 0..n {
        let last_obs_j = n - 1 - i;
        let detrend = growth.powi(-((n - 1 - i) as i32));
        let u = ultimate_counts[i];
        let closed_to_date = cumulative_closed_counts.get(i, last_obs_j);
        let remaining_counts = (u - closed_to_date).max(0.0);

        if last_obs_j + 1 < n {
            match options.count_projection_mode {
                CountProjectionMode::ExactConservation => {
                    let d_last_sel = selected_disposal_rates[last_obs_j];
                    let unscaled_rem = 1.0 - d_last_sel;
                    if unscaled_rem > 1e-15 {
                        let scale = remaining_counts / (u * unscaled_rem);
                        for j in (last_obs_j + 1)..n {
                            let dd = selected_disposal_rates[j] - selected_disposal_rates[j - 1];
                            let c_hat = u * dd * scale;
                            projected_closed_counts[i * n + j] = c_hat;
                            reserves[i] += c_hat * selected_trended_severities[j] * detrend;
                        }
                        let tail_c = u * (1.0 - selected_disposal_rates[n - 1]) * scale;
                        tail_closed_counts[i] = tail_c;
                        reserves[i] += tail_c * tail_severity * detrend;
                    } else if remaining_counts > 0.0 {
                        // Selected disposal already reached 1.0 at `last_obs_j`, yet open claims
                        // remain for this slower-closing origin: route them to the tail so exact
                        // count conservation holds.
                        tail_closed_counts[i] = remaining_counts;
                        reserves[i] += remaining_counts * tail_severity * detrend;
                    }
                }
                CountProjectionMode::AnchoredToActual => {
                    let d_actual_last = observed_disposal_rates[i * n + last_obs_j];
                    for j in (last_obs_j + 1)..n {
                        let dd = if j == last_obs_j + 1 {
                            (selected_disposal_rates[j] - d_actual_last).max(0.0)
                        } else {
                            selected_disposal_rates[j] - selected_disposal_rates[j - 1]
                        };
                        let c_hat = u * dd;
                        projected_closed_counts[i * n + j] = c_hat;
                        reserves[i] += c_hat * selected_trended_severities[j] * detrend;
                    }
                    let tail_c = u * (1.0 - selected_disposal_rates[n - 1]);
                    tail_closed_counts[i] = tail_c;
                    reserves[i] += tail_c * tail_severity * detrend;
                }
            }
        } else {
            // Most mature origin (i = 0): only tail remains if d_hat_J < 1.0.
            let tail_c = match options.count_projection_mode {
                CountProjectionMode::ExactConservation => {
                    if selected_disposal_rates[n - 1] < 1.0 - 1e-12 {
                        remaining_counts
                    } else {
                        0.0
                    }
                }
                CountProjectionMode::AnchoredToActual => u * (1.0 - selected_disposal_rates[n - 1]),
            };
            tail_closed_counts[i] = tail_c;
            reserves[i] = tail_c * tail_severity * detrend;
        }
        total_reserve += reserves[i];
    }

    Ok(AdlerKlineResult {
        observed_disposal_rates,
        selected_disposal_rates,
        selected_trended_severities,
        tail_severity,
        projected_closed_counts,
        tail_closed_counts,
        reserves,
        total_reserve,
    })
}

// ===========================================================================
// 2.2 Wright's Operational Time Reserving Model (1990)
// ===========================================================================

/// Configuration for [`wright_operational_time_reserve`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WrightOperationalTimeOptions {
    /// Include the calendar-year inflation term `δ * (i + j)` in the Gamma GLM.
    pub include_calendar_inflation: bool,
    /// Optional trailing window length for selecting target disposal rates.
    pub disposal_averaging_window: Option<usize>,
    /// Maximum IRLS iterations.
    pub max_iterations: usize,
    /// `L_inf` parameter convergence tolerance.
    pub tolerance: f64,
}

impl Default for WrightOperationalTimeOptions {
    fn default() -> Self {
        Self {
            include_calendar_inflation: true,
            disposal_averaging_window: None,
            max_iterations: 200,
            tolerance: 1e-10,
        }
    }
}

/// Output of [`wright_operational_time_reserve`] (Wright 1990).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WrightOperationalTimeResult {
    /// Fitted intercept `β_0`.
    pub beta_0: f64,
    /// Fitted linear operational-time slope `β_1`.
    pub beta_1: f64,
    /// Fitted log operational-time slope `β_2`.
    pub beta_2: f64,
    /// Fitted calendar-year inflation parameter `δ` (`0.0` when disabled).
    pub calendar_inflation: f64,
    /// Selected monotone target cumulative disposal rates `d_hat_j` (length `n`).
    pub selected_disposal_rates: Vec<f64>,
    /// Indicated reserve per origin year.
    pub reserves: Vec<f64>,
    /// Total indicated reserve across all origins.
    pub total_reserve: f64,
    /// Number of IRLS iterations executed.
    pub iterations: usize,
    /// Whether IRLS met the convergence tolerance.
    pub converged: bool,
}

fn solve_symmetric_system(a: &[f64], b: &[f64], dim: usize) -> Result<Vec<f64>, ValidationError> {
    if let Ok(l) = cholesky_lower(a, dim) {
        let mut y = vec![0.0; dim];
        solve_lower_into(&l, dim, b, &mut y);
        let mut x = vec![0.0; dim];
        for i in (0..dim).rev() {
            let mut sum = y[i];
            for j in (i + 1)..dim {
                sum -= l[j * dim + i] * x[j];
            }
            x[i] = sum / l[i * dim + i];
        }
        Ok(x)
    } else {
        solve_general(a, dim, b)
    }
}

/// Wright's (1990) operational-time Gamma GLM reserving kernel.
///
/// Models incremental paid severity `s_{i,j} = p_{i,j} / c_{i,j}` as a
/// count-weighted Gamma log-link GLM in midpoint operational time
/// `τ_bar_{i,j} = (τ_{i,j-1} + τ_{i,j}) / 2`:
/// `ln E[s_{i,j}] = β_0 + β_1 τ_bar_{i,j} + β_2 ln(τ_bar_{i,j}) + δ (i + j)`.
pub fn wright_operational_time_reserve(
    cumulative_closed_counts: &TriangleView<'_>,
    incremental_paid_loss: &TriangleView<'_>,
    ultimate_counts: &[f64],
    options: WrightOperationalTimeOptions,
) -> Result<WrightOperationalTimeResult, ValidationError> {
    validate_triangle(cumulative_closed_counts)?;
    validate_square_incremental_paid(incremental_paid_loss)?;
    let n = cumulative_closed_counts.n_origins;
    if incremental_paid_loss.n_origins != n || ultimate_counts.len() != n {
        return Err(ValidationError::ShapeMismatch);
    }
    if n < 3 {
        return Err(ValidationError::InvalidParameter(
            "Wright operational time model requires at least a 3x3 triangle",
        ));
    }
    if options.max_iterations == 0 {
        return Err(ValidationError::InvalidIterations);
    }
    if !options.tolerance.is_finite() || options.tolerance <= 0.0 {
        return Err(ValidationError::NonPositive("tolerance"));
    }
    for &u in ultimate_counts {
        if !u.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if u <= 0.0 {
            return Err(ValidationError::NonPositive("ultimate_counts"));
        }
    }

    let p = if options.include_calendar_inflation {
        4usize
    } else {
        3usize
    };

    // Build operational times and GLM training rows (cells with c_{i,j} > 0 and s_{i,j} > 0).
    let mut tau = vec![f64::NAN; n * n];
    let mut x_rows: Vec<[f64; 4]> = Vec::new();
    let mut s_vals: Vec<f64> = Vec::new();
    let mut w_vals: Vec<f64> = Vec::new();

    for i in 0..n {
        let u = ultimate_counts[i];
        let mut prev_tau = 0.0;
        let mut prev_c = 0.0;
        for j in 0..(n - i) {
            let c_cum = cumulative_closed_counts.get(i, j);
            let t_ij = (c_cum / u).clamp(0.0, 1.0);
            tau[i * n + j] = t_ij;
            let c_inc = c_cum - prev_c;
            let p_inc = incremental_paid_loss.get(i, j);
            let tau_mid = f64::midpoint(prev_tau, t_ij).clamp(1e-9, 1.0 - 1e-9);
            if c_inc > 0.0 && p_inc > 0.0 {
                let s_ij = p_inc / c_inc;
                x_rows.push([1.0, tau_mid, tau_mid.ln(), (i + j) as f64]);
                s_vals.push(s_ij);
                w_vals.push(c_inc);
            }
            prev_tau = t_ij;
            prev_c = c_cum;
        }
    }

    if x_rows.len() < p {
        return Err(ValidationError::InvalidParameter(
            "insufficient positive incremental severity observations for Wright GLM",
        ));
    }

    // Precompute constant X^T W X (for Gamma log-link GLM, working weights are constant W = c_{i,j}).
    let mut xtwx = vec![0.0; p * p];
    let mut rhs0 = vec![0.0; p];
    for ((x, &s), &w) in x_rows.iter().zip(s_vals.iter()).zip(w_vals.iter()) {
        let log_s = s.max(1e-12).ln();
        for a in 0..p {
            rhs0[a] += w * x[a] * log_s;
            for b in 0..p {
                xtwx[a * p + b] += w * x[a] * x[b];
            }
        }
    }

    // Factor constant X^T W X once before IRLS iterations.
    let chol = crate::linalg::cholesky_lower(&xtwx, p).ok();
    let solve_with_xtwx = |rhs_vec: &[f64]| -> Result<Vec<f64>, ValidationError> {
        if let Some(ref l) = chol {
            let mut y = vec![0.0; p];
            for i in 0..p {
                let mut sum = rhs_vec[i];
                for j in 0..i {
                    sum -= l[i * p + j] * y[j];
                }
                y[i] = sum / l[i * p + i];
            }
            let mut x = vec![0.0; p];
            for i in (0..p).rev() {
                let mut sum = y[i];
                for j in (i + 1)..p {
                    sum -= l[j * p + i] * x[j];
                }
                x[i] = sum / l[i * p + i];
            }
            Ok(x)
        } else {
            solve_symmetric_system(&xtwx, rhs_vec, p)
        }
    };

    let mut theta = solve_with_xtwx(&rhs0)?;
    let mut converged = false;
    let mut iterations = 0usize;
    let mut rhs = vec![0.0; p];

    for iter in 0..options.max_iterations {
        iterations = iter + 1;
        rhs.fill(0.0);
        for ((x, &s), &w) in x_rows.iter().zip(s_vals.iter()).zip(w_vals.iter()) {
            let mut eta = 0.0;
            for a in 0..p {
                eta += x[a] * theta[a];
            }
            let eta_clamped = eta.clamp(-50.0, 50.0);
            let mu = eta_clamped.exp();
            let z = eta_clamped + (s - mu) / mu;
            for a in 0..p {
                rhs[a] += w * x[a] * z;
            }
        }
        let next_theta = solve_with_xtwx(&rhs)?;
        let max_diff = theta
            .iter()
            .zip(next_theta.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        theta = next_theta;
        if !max_diff.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if max_diff < options.tolerance {
            converged = true;
            break;
        }
    }

    let beta_0 = theta[0];
    let beta_1 = theta[1];
    let beta_2 = theta[2];
    let calendar_inflation = if options.include_calendar_inflation {
        theta[3]
    } else {
        0.0
    };

    // Select target monotone disposal rates for projecting future operational time.
    let mut selected_disposal_rates = vec![0.0; n];
    let mut prev_d = 0.0;
    for j in 0..n {
        let count_rows = n - j;
        let start = match options.disposal_averaging_window {
            Some(w) if w > 0 && w < count_rows => count_rows - w,
            _ => 0,
        };
        let mut sum = 0.0;
        let mut k = 0usize;
        for i in start..count_rows {
            sum += tau[i * n + j];
            k += 1;
        }
        let avg = (sum / k as f64).clamp(prev_d, 1.0);
        selected_disposal_rates[j] = avg;
        prev_d = avg;
    }

    // Project future reserves along operational time with exact count conservation.
    let mut reserves = vec![0.0; n];
    let mut total_reserve = 0.0;
    for i in 0..n {
        let last_obs_j = n - 1 - i;
        let u = ultimate_counts[i];
        let mut prev_tau = tau[i * n + last_obs_j];
        let remaining_frac = (1.0 - prev_tau).max(0.0);
        let d_last_sel = selected_disposal_rates[last_obs_j];
        let unscaled_rem = 1.0 - d_last_sel;
        let scale = if unscaled_rem > 1e-15 {
            remaining_frac / unscaled_rem
        } else {
            0.0
        };

        for j in (last_obs_j + 1)..n {
            let d_inc = (selected_disposal_rates[j] - selected_disposal_rates[j - 1]) * scale;
            let next_tau = (prev_tau + d_inc).clamp(prev_tau, 1.0);
            let c_hat = u * (next_tau - prev_tau);
            if c_hat > 0.0 {
                let tau_mid = f64::midpoint(prev_tau, next_tau).clamp(1e-9, 1.0 - 1e-9);
                let eta = beta_0
                    + beta_1 * tau_mid
                    + beta_2 * tau_mid.ln()
                    + calendar_inflation * (i + j) as f64;
                reserves[i] += c_hat * eta.clamp(-50.0, 50.0).exp();
            }
            prev_tau = next_tau;
        }
        // Tail segment if prev_tau < 1.0.
        if prev_tau < 1.0 - 1e-12 {
            let c_tail = u * (1.0 - prev_tau);
            let tau_mid = f64::midpoint(prev_tau, 1.0).clamp(1e-9, 1.0 - 1e-9);
            let eta = beta_0
                + beta_1 * tau_mid
                + beta_2 * tau_mid.ln()
                + calendar_inflation * (i + n) as f64;
            reserves[i] += c_tail * eta.clamp(-50.0, 50.0).exp();
        }
        total_reserve += reserves[i];
    }

    Ok(WrightOperationalTimeResult {
        beta_0,
        beta_1,
        beta_2,
        calendar_inflation,
        selected_disposal_rates,
        reserves,
        total_reserve,
        iterations,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Triangle;

    #[test]
    fn golden_adler_kline_friedland_method_2() {
        const NAN: f64 = f64::NAN;
        let closed = Triangle::new(
            3,
            3,
            vec![50.0, 80.0, 100.0, 55.0, 88.0, NAN, 60.0, NAN, NAN],
        );
        let inc_paid = Triangle::new(
            3,
            3,
            vec![
                50_000.0, 60_000.0, 50_000.0, 57_750.0, 69_300.0, NAN, 66_150.0, NAN, NAN,
            ],
        );
        let ults = [100.0, 110.0, 120.0];
        let opts = AdlerKlineOptions {
            severity_trend: 0.05,
            ..AdlerKlineOptions::default()
        };

        let res = adler_kline_reserve(&closed.view(), &inc_paid.view(), &ults, opts)
            .expect("Adler-Kline");

        assert!((res.selected_disposal_rates[0] - 0.50).abs() < 1e-12);
        assert!((res.selected_disposal_rates[1] - 0.80).abs() < 1e-12);
        assert!((res.selected_disposal_rates[2] - 1.00).abs() < 1e-12);

        assert!((res.selected_trended_severities[0] - 1102.50).abs() < 1e-9);
        assert!((res.selected_trended_severities[1] - 2205.00).abs() < 1e-9);
        assert!((res.selected_trended_severities[2] - 2756.25).abs() < 1e-9);

        assert!(res.reserves[0].abs() < 1e-9);
        assert!((res.reserves[1] - 57_750.0).abs() < 1e-6);
        assert!((res.reserves[2] - 145_530.0).abs() < 1e-6);
        assert!((res.total_reserve - 203_280.0).abs() < 1e-6);
    }

    #[test]
    fn wright_operational_time_recovers_exact_synthetic_hoerl_severity() {
        // Exact synthetic triangle generated from Wright's Gamma log-link model
        // with varying settlement speeds across origins.
        let n = 5usize;
        let b0 = 6.5;
        let b1 = 0.8;
        let b2 = 0.3;
        let delta = 0.04;
        let ults = [200.0, 220.0, 240.0, 260.0, 280.0];
        let base_disp = [0.25, 0.55, 0.78, 0.92, 1.0];

        let mut closed_data = vec![f64::NAN; n * n];
        let mut inc_paid_data = vec![f64::NAN; n * n];

        for i in 0..n {
            // Slight settlement speedup in newer accident years.
            let speed = 1.0 + 0.03 * i as f64;
            let mut prev_t = 0.0;
            for j in 0..(n - i) {
                let t = (base_disp[j] * speed).min(1.0);
                let t = t.max(prev_t + 1e-4).min(1.0);
                closed_data[i * n + j] = ults[i] * t;
                let c_inc = ults[i] * (t - prev_t);
                let tau_mid = f64::midpoint(prev_t, t);
                let sev = (b0 + b1 * tau_mid + b2 * tau_mid.ln() + delta * (i + j) as f64).exp();
                inc_paid_data[i * n + j] = c_inc * sev;
                prev_t = t;
            }
        }
        let closed = Triangle::new(n, n, closed_data);
        let inc_paid = Triangle::new(n, n, inc_paid_data);

        let res = wright_operational_time_reserve(
            &closed.view(),
            &inc_paid.view(),
            &ults,
            WrightOperationalTimeOptions::default(),
        )
        .expect("Wright operational time");

        assert!(res.converged);
        assert!((res.beta_0 - b0).abs() < 1e-6, "b0={}", res.beta_0);
        assert!((res.beta_1 - b1).abs() < 1e-6, "b1={}", res.beta_1);
        assert!((res.beta_2 - b2).abs() < 1e-6, "b2={}", res.beta_2);
        assert!(
            (res.calendar_inflation - delta).abs() < 1e-6,
            "delta={}",
            res.calendar_inflation
        );
        assert!(res.total_reserve > 0.0);
    }

    #[test]
    fn fill_missing_lag_severities_carries_forward_and_backward() {
        let filled =
            fill_missing_lag_severities(&[None, Some(2.0), None, Some(5.0), None]).expect("filled");
        assert_eq!(filled, vec![2.0, 2.0, 2.0, 5.0, 5.0]);
        assert!(fill_missing_lag_severities(&[None, None]).is_err());
    }
}
