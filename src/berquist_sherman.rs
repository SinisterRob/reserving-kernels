//! Berquist & Sherman (1977) operational loss-reserve adjustments.
//!
//! Implements the two classical operational diagnostic and restatement
//! adjustments from:
//!
//! * J. R. Berquist & R. E. Sherman, *"Loss Reserve Adequacy Testing: A
//!   Comprehensive, Systematic Approach"*, *PCAS*, Vol. LXIV (1977),
//!   pp. 123–184.
//! * J. Friedland, *Estimating Unpaid Claims Using Basic Techniques*, CAS
//!   (2010), Chapter 13.
//!
//! ## 1. Case-Reserve Adequacy Adjustment ([`adjust_case_reserves`])
//!
//! When case-reserve adequacy shifts over calendar time (e.g., systematic case
//! strengthening), historical reported triangles mix incompatible case-reserve
//! levels. Berquist & Sherman anchor each development column `j` at its latest
//! observed diagonal average case reserve `S_{i_max, j}` (where
//! `i_max = n - 1 - j`) and restate earlier origin rows `i <= i_max` along an
//! exponential severity trend `r`:
//!
//! `S_{i, j} = S_{i_max, j} · exp(-r · (i_max - i))`
//!
//! (equivalently `(1 + g)^{-(i_max - i)}` with `r = ln(1 + g)`), reconstruct
//! adjusted case reserves `Case*_{i,j} = S_{i,j} · Open_{i,j}`, add observed
//! cumulative paid losses to form `Reported*_{i,j} = Paid_{i,j} + Case*_{i,j}`,
//! and project ultimates via chain ladder. The latest diagonal is preserved
//! identically.
//!
//! ## 2. Paid Claim Disposal-Rate Adjustment ([`adjust_paid_disposal_rate`])
//!
//! When claim settlement velocity changes, paid development factors are
//! distorted. Berquist & Sherman compute historical disposal rates
//! `d_{i,j} = Closed_{i,j} / N_i` (relative to selected ultimate claim counts
//! `N_i`), select target disposal rates `d*_j` (by default from the latest
//! diagonal `d_{n - 1 - j, j}`), form adjusted closed counts
//! `Closed*_{i,j} = N_i · d*_j`, and interpolate adjusted cumulative paid
//! losses `Paid*_{i,j}` along each origin's `(Closed, Paid)` curve using
//! piecewise-linear or Berquist–Sherman exponential interpolation before
//! applying chain-ladder development.

use crate::chain_ladder::ChainLadderEngine;
use crate::ldf::DevelopmentFactorEngine;
use crate::types::{Triangle, TriangleView};
use crate::validate::{
    MonotonicityPolicy, ValidationError, validate_positive_finite_slice, validate_tail,
    validate_triangle,
};

// ===========================================================================
// Case-Reserve Adequacy Adjustment Types
// ===========================================================================

/// Severity trend specification for [`adjust_case_reserves`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CaseReserveTrend {
    /// Continuous exponential trend rate `r` in `S_{i,j} = S_{i_max,j} * exp(-r * (i_max - i))`.
    ExponentialRate(f64),
    /// Discrete annual compound trend `g > -1` in `S_{i,j} = S_{i_max,j} * (1 + g)^{-(i_max - i)}`
    /// (converted internally to `r = ln(1 + g)`).
    AnnualCompoundRate(f64),
    /// Fit the continuous exponential rate `r` via pooled log-linear OLS across
    /// all observed columns with strictly positive average case reserves
    /// (`ln(A_{i,j}) = alpha_j + r * i`).
    FitFromAverageCase,
}

/// Output of the Berquist–Sherman case-reserve adequacy adjustment.
#[derive(Debug, Clone, PartialEq)]
pub struct CaseReserveAdjustmentResult {
    /// Effective continuous exponential severity trend `r` used in `exp(-r * (i_max - i))`.
    pub exponential_trend_rate: f64,
    /// Equivalent annual compound rate `exp(r) - 1`.
    pub annual_compound_rate: f64,
    /// Raw observed case outstanding triangle (`Reported - Paid`).
    pub raw_case_reserves: Triangle,
    /// Raw observed average case outstanding (`Case / Open`, `0` where `Open == 0`).
    pub raw_average_case: Triangle,
    /// Restated average case outstanding (`S_{i_max,j} * exp(-r * (i_max - i))`).
    pub adjusted_average_case: Triangle,
    /// Restated case outstanding triangle (`adjusted_average_case * Open`).
    pub adjusted_case_reserves: Triangle,
    /// Restated cumulative reported loss triangle (`Paid + adjusted_case_reserves`).
    pub adjusted_reported: Triangle,
    /// Volume-weighted age-to-age LDFs computed on `adjusted_reported`.
    pub ldfs: Vec<f64>,
    /// Cumulative development factors (including tail).
    pub cdfs: Vec<f64>,
    /// Adjusted chain-ladder ultimates by origin.
    pub ultimates: Vec<f64>,
    /// Adjusted chain-ladder reserves by origin (`ultimates[i] - latest_reported[i]`).
    pub reserves: Vec<f64>,
    /// Total adjusted ultimate loss.
    pub total_ultimate: f64,
    /// Total adjusted reserve.
    pub total_reserve: f64,
}

/// Perform the Berquist & Sherman (1977) case-reserve adequacy adjustment on
/// aligned cumulative `reported`, cumulative `paid`, and `open_counts`
/// triangles.
pub fn adjust_case_reserves(
    reported: &TriangleView<'_>,
    paid: &TriangleView<'_>,
    open_counts: &TriangleView<'_>,
    trend: CaseReserveTrend,
    tail: f64,
    policy: MonotonicityPolicy,
) -> Result<CaseReserveAdjustmentResult, ValidationError> {
    validate_triangle(reported)?;
    validate_triangle(paid)?;
    validate_triangle(open_counts)?;
    validate_tail(tail)?;

    let n = reported.n_origins;
    if n < 2 || paid.n_origins != n || open_counts.n_origins != n {
        return Err(ValidationError::ShapeMismatch);
    }

    let mut raw_case = vec![f64::NAN; n * n];
    let mut raw_avg = vec![f64::NAN; n * n];
    for i in 0..n {
        for j in 0..(n - i) {
            let idx = i * n + j;
            let case_val = reported.data[idx] - paid.data[idx];
            if case_val < -1e-9 {
                return Err(ValidationError::InvalidParameter(
                    "reported loss must be >= paid loss for case-reserve adjustment",
                ));
            }
            let case_clamped = case_val.max(0.0);
            raw_case[idx] = case_clamped;
            let open = open_counts.data[idx];
            raw_avg[idx] = if open > 0.0 { case_clamped / open } else { 0.0 };
        }
    }

    let rate = match trend {
        CaseReserveTrend::ExponentialRate(r) => {
            if !r.is_finite() {
                return Err(ValidationError::NotFinite);
            }
            r
        }
        CaseReserveTrend::AnnualCompoundRate(g) => {
            if !g.is_finite() {
                return Err(ValidationError::NotFinite);
            }
            if g <= -1.0 {
                return Err(ValidationError::InvalidParameter(
                    "annual compound trend must be > -1.0",
                ));
            }
            (1.0 + g).ln()
        }
        CaseReserveTrend::FitFromAverageCase => fit_pooled_log_linear_trend(&raw_avg, n)?,
    };

    let mut adj_avg = vec![f64::NAN; n * n];
    let mut adj_case = vec![f64::NAN; n * n];
    let mut adj_reported = vec![f64::NAN; n * n];

    for j in 0..n {
        let i_max = n - 1 - j;
        let anchor_avg = raw_avg[i_max * n + j];
        for i in 0..=i_max {
            let idx = i * n + j;
            let lag = (i_max - i) as f64;
            let s_ij = if i == i_max {
                anchor_avg
            } else {
                anchor_avg * (-rate * lag).exp()
            };
            let c_ij = if i == i_max {
                raw_case[idx]
            } else {
                s_ij * open_counts.data[idx]
            };
            adj_avg[idx] = s_ij;
            adj_case[idx] = c_ij;
            adj_reported[idx] = paid.data[idx] + c_ij;
        }
    }

    let adjusted_reported = Triangle::new(n, n, adj_reported);
    let factors = DevelopmentFactorEngine::volume_weighted_with_policy(
        &adjusted_reported.view(),
        tail,
        policy,
    )?;
    let cl = ChainLadderEngine::project_with_policy(&adjusted_reported.view(), &factors, policy)?;

    Ok(CaseReserveAdjustmentResult {
        exponential_trend_rate: rate,
        annual_compound_rate: rate.exp() - 1.0,
        raw_case_reserves: Triangle::new(n, n, raw_case),
        raw_average_case: Triangle::new(n, n, raw_avg),
        adjusted_average_case: Triangle::new(n, n, adj_avg),
        adjusted_case_reserves: Triangle::new(n, n, adj_case),
        adjusted_reported,
        ldfs: factors.ldfs,
        cdfs: factors.cdfs,
        ultimates: cl.ultimates,
        reserves: cl.reserves,
        total_ultimate: cl.total_ultimate,
        total_reserve: cl.total_reserve,
    })
}

/// Fit pooled within-column log-linear trend `r` from `ln(A_{i,j}) = alpha_j + r * i`.
fn fit_pooled_log_linear_trend(raw_avg: &[f64], n: usize) -> Result<f64, ValidationError> {
    let mut num = 0.0;
    let mut den = 0.0;
    for j in 0..n {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for i in 0..(n - j) {
            let v = raw_avg[i * n + j];
            if v.is_finite() && v > 0.0 {
                xs.push(i as f64);
                ys.push(v.ln());
            }
        }
        if xs.len() < 2 {
            continue;
        }
        let m = xs.len() as f64;
        let x_bar = xs.iter().sum::<f64>() / m;
        let y_bar = ys.iter().sum::<f64>() / m;
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            let dx = x - x_bar;
            num += dx * (y - y_bar);
            den += dx * dx;
        }
    }
    if den <= 0.0 {
        return Err(ValidationError::InvalidParameter(
            "insufficient positive average case reserves to estimate severity trend",
        ));
    }
    Ok(num / den)
}

// ===========================================================================
// Paid Claim Disposal-Rate Adjustment Types
// ===========================================================================

/// Interpolation rule mapping adjusted closed claim counts to adjusted
/// cumulative paid losses in [`adjust_paid_disposal_rate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DisposalInterpolation {
    /// Piecewise-linear interpolation along each origin's observed
    /// `(Closed_{i,k}, Paid_{i,k})` curve (anchored at `(0, 0)` below age 0).
    #[default]
    PiecewiseLinear,
    /// Berquist & Sherman (1977) piecewise-exponential interpolation
    /// `Paid = a * exp(b * Closed)` on positive bracketing segments (falling
    /// back to linear when anchored at `(0, 0)` or non-positive endpoints).
    Exponential,
}

/// Output of the Berquist–Sherman paid disposal-rate adjustment.
#[derive(Debug, Clone, PartialEq)]
pub struct PaidDisposalAdjustmentResult {
    /// Observed closed-claim disposal rate triangle (`Closed_{i,j} / N_i`).
    pub observed_disposal_rates: Triangle,
    /// Selected target disposal rate per development age `j` (length `n`).
    pub target_disposal_rates: Vec<f64>,
    /// Adjusted cumulative closed claim counts (`N_i * target_disposal_rates[j]`).
    pub adjusted_closed_counts: Triangle,
    /// Adjusted cumulative paid loss triangle (latest diagonal preserved).
    pub adjusted_paid: Triangle,
    /// Volume-weighted age-to-age LDFs computed on `adjusted_paid`.
    pub ldfs: Vec<f64>,
    /// Cumulative development factors (including tail).
    pub cdfs: Vec<f64>,
    /// Adjusted chain-ladder paid ultimates by origin.
    pub ultimates: Vec<f64>,
    /// Adjusted chain-ladder paid reserves by origin.
    pub reserves: Vec<f64>,
    /// Total adjusted paid ultimate.
    pub total_ultimate: f64,
    /// Total adjusted paid reserve.
    pub total_reserve: f64,
}

/// Perform the Berquist & Sherman (1977) paid claim disposal-rate adjustment.
///
/// * `paid` — cumulative paid loss triangle (`n × n`).
/// * `closed_counts` — cumulative closed claim count triangle (`n × n`).
/// * `ultimate_counts` — selected ultimate claim counts `N_i > 0` per origin (length `n`).
/// * `target_disposal_rates` — optional override of target disposal rates per
///   development age (length `n`). When `None`, the latest diagonal disposal
///   rates `Closed_{n - 1 - j, j} / N_{n - 1 - j}` are selected.
/// * `interpolation` — piecewise-linear or exponential paid-vs-closed curve interpolation.
/// * `tail` — multiplicative tail factor (`>= 1.0`).
pub fn adjust_paid_disposal_rate(
    paid: &TriangleView<'_>,
    closed_counts: &TriangleView<'_>,
    ultimate_counts: &[f64],
    target_disposal_rates: Option<&[f64]>,
    interpolation: DisposalInterpolation,
    tail: f64,
) -> Result<PaidDisposalAdjustmentResult, ValidationError> {
    validate_triangle(paid)?;
    validate_triangle(closed_counts)?;
    validate_tail(tail)?;

    let n = paid.n_origins;
    if n < 2 || closed_counts.n_origins != n || ultimate_counts.len() != n {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_positive_finite_slice(ultimate_counts, "ultimate_counts")?;

    let mut obs_rates = vec![f64::NAN; n * n];
    for i in 0..n {
        let ult_n = ultimate_counts[i];
        let mut prev_c = -1.0_f64;
        for j in 0..(n - i) {
            let c = closed_counts.get(i, j);
            if c < prev_c - 1e-12 {
                return Err(ValidationError::InvalidParameter(
                    "closed_counts must be non-decreasing across development periods",
                ));
            }
            prev_c = c;
            obs_rates[i * n + j] = c / ult_n;
        }
    }

    let targets: Vec<f64> = match target_disposal_rates {
        Some(custom) => {
            if custom.len() != n {
                return Err(ValidationError::ShapeMismatch);
            }
            for (idx, &d) in custom.iter().enumerate() {
                if !d.is_finite() {
                    return Err(ValidationError::NotFinite);
                }
                if d < 0.0 || d > 1.0 + 1e-9 {
                    return Err(ValidationError::InvalidMaturity {
                        index: idx,
                        value: d,
                    });
                }
            }
            custom.to_vec()
        }
        None => (0..n).map(|j| obs_rates[(n - 1 - j) * n + j]).collect(),
    };

    let mut adj_closed = vec![f64::NAN; n * n];
    let mut adj_paid = vec![f64::NAN; n * n];

    for i in 0..n {
        let d_max = n - 1 - i;
        let ult_n = ultimate_counts[i];
        // Collect observed curve points (closed, paid) for origin i.
        let mut xs = Vec::with_capacity(d_max + 1);
        let mut ys = Vec::with_capacity(d_max + 1);
        for k in 0..=d_max {
            xs.push(closed_counts.get(i, k));
            ys.push(paid.get(i, k));
        }

        for j in 0..=d_max {
            let idx = i * n + j;
            let c_target = ult_n * targets[j];
            adj_closed[idx] = c_target;

            if j == d_max && target_disposal_rates.is_none() {
                // Latest diagonal already operates at the current disposal rate.
                adj_paid[idx] = paid.get(i, j);
            } else {
                adj_paid[idx] = interpolate_paid_from_closed(&xs, &ys, c_target, interpolation);
            }
        }
    }

    let adjusted_paid = Triangle::new(n, n, adj_paid);
    let policy = MonotonicityPolicy::AllowNegativeIncrements;
    let factors =
        DevelopmentFactorEngine::volume_weighted_with_policy(&adjusted_paid.view(), tail, policy)?;
    let cl = ChainLadderEngine::project_with_policy(&adjusted_paid.view(), &factors, policy)?;

    Ok(PaidDisposalAdjustmentResult {
        observed_disposal_rates: Triangle::new(n, n, obs_rates),
        target_disposal_rates: targets,
        adjusted_closed_counts: Triangle::new(n, n, adj_closed),
        adjusted_paid,
        ldfs: factors.ldfs,
        cdfs: factors.cdfs,
        ultimates: cl.ultimates,
        reserves: cl.reserves,
        total_ultimate: cl.total_ultimate,
        total_reserve: cl.total_reserve,
    })
}

fn interpolate_paid_from_closed(
    xs: &[f64],
    ys: &[f64],
    x_target: f64,
    mode: DisposalInterpolation,
) -> f64 {
    let m = xs.len();
    debug_assert!(m >= 1 && xs.len() == ys.len());

    if x_target <= 0.0 {
        return 0.0;
    }
    if x_target <= xs[0] {
        if xs[0] <= 0.0 {
            return ys[0];
        }
        return (x_target / xs[0]) * ys[0];
    }

    // Search for bracketing segment [k, k+1] with xs[k] <= x_target <= xs[k+1].
    for k in 0..(m - 1) {
        if x_target <= xs[k + 1] {
            return eval_segment(xs[k], ys[k], xs[k + 1], ys[k + 1], x_target, mode);
        }
    }

    // Extrapolate beyond the last observed closed count for this origin.
    if m == 1 {
        return if xs[0] > 0.0 {
            (x_target / xs[0]) * ys[0]
        } else {
            ys[0]
        };
    }
    // Find last segment with strictly positive closed-count width.
    for k in (0..(m - 1)).rev() {
        if (xs[k + 1] - xs[k]) > 1e-12 {
            return eval_segment(xs[k], ys[k], xs[k + 1], ys[k + 1], x_target, mode);
        }
    }
    ys[m - 1]
}

fn eval_segment(x0: f64, y0: f64, x1: f64, y1: f64, x: f64, mode: DisposalInterpolation) -> f64 {
    let dx = x1 - x0;
    if dx.abs() <= 1e-12 {
        return y1;
    }
    match mode {
        DisposalInterpolation::PiecewiseLinear => {
            let t = (x - x0) / dx;
            (y0 + t * (y1 - y0)).max(0.0)
        }
        DisposalInterpolation::Exponential => {
            if y0 > 0.0 && y1 > 0.0 {
                let b = (y1 / y0).ln() / dx;
                y0 * (b * (x - x0)).exp()
            } else {
                let t = (x - x0) / dx;
                (y0 + t * (y1 - y0)).max(0.0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAN: f64 = f64::NAN;

    #[test]
    fn case_reserve_adjustment_removes_case_strengthening_bias() {
        // Construct a 5x5 triangle where true severity grows at 5% continuous rate,
        // paid & open counts are stationary, and a 25% case-strengthening jump occurs
        // on the final diagonal (calendar year 4).
        let n = 5usize;
        let r_true = 0.05_f64;
        let base_paid = [100.0, 220.0, 320.0, 380.0, 410.0];
        let base_open = [20.0, 12.0, 6.0, 2.0, 0.0];
        let base_avg_case = [10.0, 11.0, 12.0, 13.0, 0.0];

        let mut paid = vec![NAN; n * n];
        let mut reported = vec![NAN; n * n];
        let mut open = vec![NAN; n * n];
        let mut true_ult = vec![0.0; n];

        for i in 0..n {
            let inflation = (r_true * i as f64).exp();
            true_ult[i] = base_paid[n - 1] * inflation;
            for j in 0..(n - i) {
                let idx = i * n + j;
                let p = base_paid[j] * inflation;
                let o = base_open[j];
                // Case strengthening +25% on latest calendar diagonal (i + j == n - 1).
                let adequacy = if i + j == n - 1 { 1.25 } else { 1.00 };
                let c = o * base_avg_case[j] * inflation * adequacy;
                paid[idx] = p;
                open[idx] = o;
                reported[idx] = p + c;
            }
        }

        let paid_tri = Triangle::new(n, n, paid);
        let rep_tri = Triangle::new(n, n, reported);
        let open_tri = Triangle::new(n, n, open);

        let naive_f = DevelopmentFactorEngine::volume_weighted(&rep_tri.view(), 1.0).unwrap();
        let naive_cl = ChainLadderEngine::project(&rep_tri.view(), &naive_f).unwrap();

        let adj = adjust_case_reserves(
            &rep_tri.view(),
            &paid_tri.view(),
            &open_tri.view(),
            CaseReserveTrend::ExponentialRate(r_true),
            1.0,
            MonotonicityPolicy::AllowNegativeIncrements,
        )
        .expect("case adj");

        // Latest diagonal must be preserved identically.
        for i in 0..n {
            let d = n - 1 - i;
            assert!(
                (adj.adjusted_reported.view().get(i, d) - rep_tri.view().get(i, d)).abs() < 1e-10
            );
        }

        // True total ultimate; case-strengthening inflates naive CL, whereas BS corrects it.
        let true_total: f64 = true_ult.iter().sum();
        let naive_err = (naive_cl.total_ultimate - true_total).abs();
        let adj_err = (adj.total_ultimate - true_total).abs();
        assert!(
            adj_err < naive_err * 0.25,
            "adj_err={adj_err} naive_err={naive_err}"
        );
    }

    #[test]
    fn paid_disposal_rate_adjustment_corrects_settlement_speedup() {
        // True claim payment curve P(c) = 10 * c + 0.02 * c^2 for closed count c in [0, 100].
        // Historical years closed at [20, 50, 75, 90, 100], while latest calendar diagonal
        // sped up by +10% disposal rate.
        let n = 5usize;
        let base_d = [0.20, 0.50, 0.75, 0.90, 1.00];
        let fast_d = [0.28, 0.60, 0.82, 0.94, 1.00];
        let ult_counts = [100.0, 110.0, 120.0, 130.0, 140.0];

        let mut paid = vec![NAN; n * n];
        let mut closed = vec![NAN; n * n];
        let mut true_ult = vec![0.0; n];

        for i in 0..n {
            let n_i = ult_counts[i];
            true_ult[i] = 10.0 * n_i + 0.02 * n_i * n_i;
            for j in 0..(n - i) {
                let d = if i + j == n - 1 { fast_d[j] } else { base_d[j] };
                let c = n_i * d;
                closed[i * n + j] = c;
                paid[i * n + j] = 10.0 * c + 0.02 * c * c;
            }
        }

        let paid_tri = Triangle::new(n, n, paid);
        let closed_tri = Triangle::new(n, n, closed);

        let naive_f = DevelopmentFactorEngine::volume_weighted(&paid_tri.view(), 1.0).unwrap();
        let naive_cl = ChainLadderEngine::project(&paid_tri.view(), &naive_f).unwrap();

        let adj = adjust_paid_disposal_rate(
            &paid_tri.view(),
            &closed_tri.view(),
            &ult_counts,
            None,
            DisposalInterpolation::PiecewiseLinear,
            1.0,
        )
        .expect("paid adj");

        let true_total: f64 = true_ult.iter().sum();
        let naive_err = (naive_cl.total_ultimate - true_total).abs();
        let adj_err = (adj.total_ultimate - true_total).abs();
        assert!(
            adj_err < naive_err * 0.20,
            "adj_err={adj_err} naive_err={naive_err} adj_tot={} true_tot={true_total}",
            adj.total_ultimate
        );
    }
}
