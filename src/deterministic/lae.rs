//! Loss Adjustment Expense (ULAE & ALAE) reserving kernels.
//!
//! Implements classical and modern P&C LAE reserving methods:
//!
//! * **Classical Paid-to-Paid ULAE** (50/50 split and generalized opening weight
//!   `w_open ∈ [0, 1]`).
//! * **Kittel's Refinement (1981)** — effort-weighted incurred/paid denominator
//!   to eliminate growth and inflation distortion.
//! * **Mango–Allen (1995)** — expected-payment / expected-incurred smoothing
//!   to remove calendar-year large-loss settlement noise.
//! * **Wendy Johnson (1989)** — claim-count-based ULAE projection with cost
//!   inflation.
//! * **Paid-to-Paid ALAE Ratio Development Triangle** — ratio-weighted and
//!   loss-weighted age-to-age development of `A_{i,j} / L_{i,j}`.
//! * **Clark (2003) / Johnson Ratio-to-Loss ALAE ODP-MLE** — Over-Dispersed
//!   Poisson profile MLE linking incremental paid ALAE to ultimate losses via
//!   Loglogistic or Weibull emergence curves.

use crate::stochastic::clark_mle::GrowthCurve;
use crate::types::TriangleView;
use crate::validate::{
    ValidationError, validate_finite_slice, validate_nonnegative_finite_slice, validate_tail,
    validate_triangle,
};

// ===========================================================================
// Shared ULAE Options & Result Types
// ===========================================================================

/// Selection rule for aggregating historical calendar-year ULAE ratios.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum UlaeRatioSelection {
    /// Volume-weighted average `Σ U_t / Σ D_t` over the selected calendar years.
    #[default]
    VolumeWeighted,
    /// Arithmetic mean of annual ratios `r_t = U_t / D_t` over selected years.
    SimpleAverage,
}

/// Configuration for paid-to-paid, Kittel, and Mango–Allen ULAE estimators.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct UlaeOptions {
    /// Fraction of ULAE effort incurred at claim opening/reporting (`[0, 1]`).
    /// Classical 50/50 default is `0.5`.
    pub opening_weight: f64,
    /// Rule used to average historical calendar-year ratios.
    pub selection: UlaeRatioSelection,
    /// If `false` (default), clamp the effective reserve base
    /// `(1 - w_open) * Case + IBNR` at `0.0` when negative IBNR exceeds
    /// weighted case reserves.
    pub allow_negative_reserve: bool,
}

impl Default for UlaeOptions {
    fn default() -> Self {
        Self {
            opening_weight: 0.5,
            selection: UlaeRatioSelection::VolumeWeighted,
            allow_negative_reserve: false,
        }
    }
}

impl UlaeOptions {
    /// Set the opening/reporting effort weight `w_open ∈ [0, 1]`.
    pub const fn with_opening_weight(mut self, opening_weight: f64) -> Self {
        self.opening_weight = opening_weight;
        self
    }

    /// Set the ratio selection rule (`VolumeWeighted` vs `SimpleAverage`).
    pub const fn with_selection(mut self, selection: UlaeRatioSelection) -> Self {
        self.selection = selection;
        self
    }

    /// Allow or clamp negative effective ULAE reserve bases.
    pub const fn with_allow_negative_reserve(mut self, allow: bool) -> Self {
        self.allow_negative_reserve = allow;
        self
    }
}

/// Output of Classical Paid-to-Paid, Kittel, or Mango–Allen ULAE estimation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct UlaeRatioResult {
    /// Annual ULAE-to-denominator ratios `r_t = U_t / D_t` (length `T`).
    pub annual_ratios: Vec<f64>,
    /// Annual denominators `D_t` used in the ratio calculation (length `T`).
    pub denominators: Vec<f64>,
    /// Selected overall ULAE ratio `r_hat`.
    pub selected_ratio: f64,
    /// Effective reserve base `(1 - w_open) * Case + IBNR` (clamped to `>= 0`
    /// unless `allow_negative_reserve` is set).
    pub reserve_base: f64,
    /// Indicated ULAE reserve `r_hat * reserve_base`.
    pub reserve: f64,
}

/// Input claim counts for one calendar year in Wendy Johnson's (1989) ULAE method.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClaimCountYear {
    /// Newly reported claims during the calendar year.
    pub newly_reported: f64,
    /// Open claims (year-end or average open count).
    pub open: f64,
    /// Closed claims during the calendar year.
    pub closed: f64,
}

/// Effort weights `(w_new, w_open, w_close)` for Wendy Johnson's ULAE method.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct JohnsonUlaeWeights {
    /// Weight on newly reported claims.
    pub newly_reported: f64,
    /// Weight on open claims.
    pub open: f64,
    /// Weight on closed claims.
    pub closed: f64,
}

impl Default for JohnsonUlaeWeights {
    fn default() -> Self {
        Self {
            newly_reported: 1.0,
            open: 1.0,
            closed: 0.0,
        }
    }
}

/// Output of Wendy Johnson's (1989) claim-count-based ULAE method.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct JohnsonUlaeResult {
    /// Historical weighted claim counts `N*_t`.
    pub historical_weighted_counts: Vec<f64>,
    /// Historical unit costs `c_t = U_t / N*_t`.
    pub historical_unit_costs: Vec<f64>,
    /// Trend-adjusted valuation-year unit cost `c_T`.
    pub base_unit_cost: f64,
    /// Projected future weighted claim counts `N*_{T+k}` for `k = 1..=K`.
    pub projected_weighted_counts: Vec<f64>,
    /// Projected inflated unit costs `c_T * (1 + g)^k` for `k = 1..=K`.
    pub projected_unit_costs: Vec<f64>,
    /// Projected annual ULAE cash flows for `k = 1..=K`.
    pub projected_ulae_payments: Vec<f64>,
    /// Total indicated ULAE reserve `Σ_{k=1}^K projected_ulae_payments[k-1]`.
    pub reserve: f64,
}

// ===========================================================================
// ALAE Types
// ===========================================================================

/// Weighting convention for age-to-age factors on an ALAE/Loss ratio triangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum AlaeRatioWeighting {
    /// Sum of ratios: `Σ_i Q_{i,j+1} / Σ_i Q_{i,j}`.
    #[default]
    RatioSum,
    /// Loss-weighted ratio of aggregate ratios:
    /// `(Σ_i A_{i,j+1} / Σ_i L_{i,j+1}) / (Σ_i A_{i,j} / Σ_i L_{i,j})`.
    LossWeighted,
}

/// Output of paid-to-paid ALAE ratio triangle development.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AlaeRatioTriangleResult {
    /// Square `n × n` observed ratio grid `Q_{i,j} = A_{i,j} / L_{i,j}` (`NaN` where unobserved or `L_{i,j} <= 0`).
    pub ratio_grid: Vec<f64>,
    /// Age-to-age ratio development factors `f^(Q)_j` (length `n - 1`).
    pub ldfs: Vec<f64>,
    /// Cumulative ratio development factors `F^(Q)_j` including tail (length `n`).
    pub cdfs: Vec<f64>,
    /// Tail factor applied to the ratio triangle.
    pub tail: f64,
    /// Projected ultimate ALAE-to-loss ratios `Q_hat_{i,∞}` per origin.
    pub ultimate_ratios: Vec<f64>,
    /// Projected ultimate ALAE `A_hat_{i,∞} = Q_hat_{i,∞} * L_hat_i` per origin.
    pub ultimate_alae: Vec<f64>,
    /// Indicated ALAE reserve `A_hat_{i,∞} - A_{i, n - 1 - i}` per origin.
    pub reserves: Vec<f64>,
    /// Total indicated ALAE reserve across all origins.
    pub total_reserve: f64,
}

/// Configuration for Clark (2003) / Johnson ratio-to-loss ALAE ODP-MLE.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClarkAlaeOptions {
    /// Parametric emergence curve (`LogLogistic` or `Weibull`).
    pub curve: GrowthCurve,
    /// Initial `omega` parameter (`> 0`).
    pub initial_omega: f64,
    /// Initial `theta` parameter (`> 0`).
    pub initial_theta: f64,
    /// Maximum Nelder–Mead iterations.
    pub max_iterations: usize,
    /// Parameter convergence tolerance.
    pub tolerance: f64,
}

impl Default for ClarkAlaeOptions {
    fn default() -> Self {
        Self {
            curve: GrowthCurve::Loglogistic,
            initial_omega: 1.5,
            initial_theta: 2.0,
            max_iterations: 500,
            tolerance: 1e-10,
        }
    }
}

/// Output of Clark (2003) / Johnson ratio-to-loss ALAE ODP-MLE.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClarkAlaeResult {
    /// Emergence curve family fitted.
    pub curve: GrowthCurve,
    /// Fitted shape parameter `ω > 0`.
    pub omega: f64,
    /// Fitted scale parameter `θ > 0`.
    pub theta: f64,
    /// Profile MLE expected ALAE-to-loss ratio `ELR_A`.
    pub elr_alae: f64,
    /// Concentrated ODP log-likelihood at the optimum (up to additive constant).
    pub log_likelihood: f64,
    /// Fitted cumulative emergence `G(t_j; ω, θ)` at each development age `j = 1..=n`.
    pub emergence_curve: Vec<f64>,
    /// Indicated ALAE reserve per origin `L_hat_i * ELR_A * (1 - G(t_{n-i}))`.
    pub reserves: Vec<f64>,
    /// Projected ultimate ALAE per origin (observed paid ALAE + indicated reserve).
    pub ultimate_alae: Vec<f64>,
    /// Total indicated ALAE reserve across all origins.
    pub total_reserve: f64,
    /// Whether the optimizer met the convergence tolerance.
    pub converged: bool,
}

// ===========================================================================
// Internal Helpers
// ===========================================================================

fn validate_ulae_common(
    paid_ulae: &[f64],
    denominators: &[f64],
    case_reserves: f64,
    ibnr: f64,
    options: UlaeOptions,
) -> Result<(), ValidationError> {
    if paid_ulae.is_empty() || paid_ulae.len() != denominators.len() {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_nonnegative_finite_slice(paid_ulae)?;
    validate_finite_slice(denominators)?;
    if !case_reserves.is_finite() || !ibnr.is_finite() || !options.opening_weight.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if case_reserves < 0.0 {
        return Err(ValidationError::Negative);
    }
    if !(0.0..=1.0).contains(&options.opening_weight) {
        return Err(ValidationError::InvalidProbability {
            value: options.opening_weight,
        });
    }
    Ok(())
}

fn compute_ulae_from_denominators(
    paid_ulae: &[f64],
    denominators: &[f64],
    case_reserves: f64,
    ibnr: f64,
    options: UlaeOptions,
) -> Result<UlaeRatioResult, ValidationError> {
    validate_ulae_common(paid_ulae, denominators, case_reserves, ibnr, options)?;

    let mut annual_ratios = Vec::with_capacity(paid_ulae.len());
    let mut sum_u = 0.0;
    let mut sum_d = 0.0;
    for (&u, &d) in paid_ulae.iter().zip(denominators.iter()) {
        if d <= 0.0 {
            return Err(ValidationError::NonPositive("ULAE denominator"));
        }
        annual_ratios.push(u / d);
        sum_u += u;
        sum_d += d;
    }
    if sum_d <= 0.0 {
        return Err(ValidationError::NonPositive("ULAE denominator sum"));
    }

    let selected_ratio = match options.selection {
        UlaeRatioSelection::VolumeWeighted => sum_u / sum_d,
        UlaeRatioSelection::SimpleAverage => {
            annual_ratios.iter().sum::<f64>() / annual_ratios.len() as f64
        }
    };

    let raw_base = (1.0 - options.opening_weight) * case_reserves + ibnr;
    let reserve_base = if options.allow_negative_reserve {
        raw_base
    } else {
        raw_base.max(0.0)
    };
    let reserve = selected_ratio * reserve_base;

    Ok(UlaeRatioResult {
        annual_ratios,
        denominators: denominators.to_vec(),
        selected_ratio,
        reserve_base,
        reserve,
    })
}

// ===========================================================================
// 1.1 Classical Paid-to-Paid ULAE
// ===========================================================================

/// Classical Paid-to-Paid ULAE estimator (50/50 default or generalized
/// `opening_weight ∈ [0, 1]`).
pub fn ulae_paid_to_paid(
    paid_ulae: &[f64],
    paid_loss: &[f64],
    case_reserves: f64,
    ibnr: f64,
    options: UlaeOptions,
) -> Result<UlaeRatioResult, ValidationError> {
    compute_ulae_from_denominators(paid_ulae, paid_loss, case_reserves, ibnr, options)
}

// ===========================================================================
// 1.2 Kittel's Refinement (1981)
// ===========================================================================

/// Kittel's (1981) ULAE refinement using effort-weighted incurred and paid
/// calendar-year losses `D_t = w_open * I_t + (1 - w_open) * P_t`.
pub fn ulae_kittel(
    paid_ulae: &[f64],
    paid_loss: &[f64],
    incurred_loss: &[f64],
    case_reserves: f64,
    ibnr: f64,
    options: UlaeOptions,
) -> Result<UlaeRatioResult, ValidationError> {
    if paid_loss.len() != incurred_loss.len() || paid_loss.len() != paid_ulae.len() {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_finite_slice(paid_loss)?;
    validate_finite_slice(incurred_loss)?;
    if !options.opening_weight.is_finite() || !(0.0..=1.0).contains(&options.opening_weight) {
        return Err(ValidationError::InvalidProbability {
            value: options.opening_weight,
        });
    }
    let w = options.opening_weight;
    let denominators: Vec<f64> = incurred_loss
        .iter()
        .zip(paid_loss.iter())
        .map(|(&inc, &paid)| w * inc + (1.0 - w) * paid)
        .collect();
    compute_ulae_from_denominators(paid_ulae, &denominators, case_reserves, ibnr, options)
}

// ===========================================================================
// 1.3 Mango–Allen (1995) Smoothing / Expected-Payment-Weighted ULAE
// ===========================================================================

/// Compute expected calendar-year loss emergence `Σ_i L_hat_i * Δp_{t - i + 1}`
/// for calendar diagonals `t = 0..n` from accident-year ultimates `ultimates`
/// and incremental emergence percentages `incremental_pattern` (both length `n`).
pub fn expected_calendar_emergence(
    ultimates: &[f64],
    incremental_pattern: &[f64],
) -> Result<Vec<f64>, ValidationError> {
    let n = ultimates.len();
    if n == 0 || incremental_pattern.len() != n {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_finite_slice(ultimates)?;
    validate_finite_slice(incremental_pattern)?;

    let mut cal = vec![0.0; n];
    for t in 0..n {
        let mut sum = 0.0;
        for i in 0..=t {
            let lag = t - i;
            sum += ultimates[i] * incremental_pattern[lag];
        }
        cal[t] = sum;
    }
    Ok(cal)
}

/// Mango–Allen (1995) ULAE estimator using expected calendar-year paid and
/// incurred losses `D_t = w_open * I_hat_t + (1 - w_open) * P_hat_t`.
pub fn ulae_mango_allen(
    paid_ulae: &[f64],
    expected_paid_loss: &[f64],
    expected_incurred_loss: &[f64],
    case_reserves: f64,
    ibnr: f64,
    options: UlaeOptions,
) -> Result<UlaeRatioResult, ValidationError> {
    ulae_kittel(
        paid_ulae,
        expected_paid_loss,
        expected_incurred_loss,
        case_reserves,
        ibnr,
        options,
    )
}

// ===========================================================================
// 1.4 Wendy Johnson (1989) Claim-Count-Based ULAE
// ===========================================================================

/// Wendy Johnson (1989) claim-count-based ULAE estimator with annual ULAE
/// cost inflation `inflation_rate > -1.0`.
pub fn ulae_johnson_count_based(
    historical_ulae: &[f64],
    historical_counts: &[ClaimCountYear],
    projected_counts: &[ClaimCountYear],
    weights: JohnsonUlaeWeights,
    inflation_rate: f64,
) -> Result<JohnsonUlaeResult, ValidationError> {
    let t_len = historical_ulae.len();
    if t_len == 0 || historical_counts.len() != t_len || projected_counts.is_empty() {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_nonnegative_finite_slice(historical_ulae)?;
    if !weights.newly_reported.is_finite()
        || !weights.open.is_finite()
        || !weights.closed.is_finite()
        || !inflation_rate.is_finite()
    {
        return Err(ValidationError::NotFinite);
    }
    if weights.newly_reported < 0.0 || weights.open < 0.0 || weights.closed < 0.0 {
        return Err(ValidationError::Negative);
    }
    if inflation_rate <= -1.0 {
        return Err(ValidationError::InvalidParameter(
            "ULAE inflation rate must be strictly greater than -1.0",
        ));
    }

    let weighted_count = |c: ClaimCountYear| -> Result<f64, ValidationError> {
        if !c.newly_reported.is_finite() || !c.open.is_finite() || !c.closed.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if c.newly_reported < 0.0 || c.open < 0.0 || c.closed < 0.0 {
            return Err(ValidationError::Negative);
        }
        Ok(weights.newly_reported * c.newly_reported
            + weights.open * c.open
            + weights.closed * c.closed)
    };

    let mut historical_weighted_counts = Vec::with_capacity(t_len);
    let mut historical_unit_costs = Vec::with_capacity(t_len);
    let mut trended_sum = 0.0;
    let growth = 1.0 + inflation_rate;

    for (idx, (&u, &cnt)) in historical_ulae
        .iter()
        .zip(historical_counts.iter())
        .enumerate()
    {
        let n_star = weighted_count(cnt)?;
        if n_star <= 0.0 {
            return Err(ValidationError::NonPositive(
                "historical weighted claim count",
            ));
        }
        let unit_cost = u / n_star;
        historical_weighted_counts.push(n_star);
        historical_unit_costs.push(unit_cost);
        let years_to_val = (t_len - 1 - idx) as i32;
        trended_sum += unit_cost * growth.powi(years_to_val);
    }

    let base_unit_cost = trended_sum / t_len as f64;

    let mut projected_weighted_counts = Vec::with_capacity(projected_counts.len());
    let mut projected_unit_costs = Vec::with_capacity(projected_counts.len());
    let mut projected_ulae_payments = Vec::with_capacity(projected_counts.len());
    let mut reserve = 0.0;

    for (k_idx, &cnt) in projected_counts.iter().enumerate() {
        let n_star = weighted_count(cnt)?;
        let unit_cost = base_unit_cost * growth.powi((k_idx + 1) as i32);
        let payment = n_star * unit_cost;
        projected_weighted_counts.push(n_star);
        projected_unit_costs.push(unit_cost);
        projected_ulae_payments.push(payment);
        reserve += payment;
    }

    Ok(JohnsonUlaeResult {
        historical_weighted_counts,
        historical_unit_costs,
        base_unit_cost,
        projected_weighted_counts,
        projected_unit_costs,
        projected_ulae_payments,
        reserve,
    })
}

// ===========================================================================
// 1.5 ALAE Reserving: Paid-to-Paid Ratio Triangle
// ===========================================================================

/// Paid-to-paid ALAE ratio triangle development.
///
/// Forms `Q_{i,j} = A_{i,j} / L_{i,j}` over observed cells with `L_{i,j} > 0`,
/// develops the ratio triangle to ultimate with `tail >= 1.0`, and multiplies
/// ultimate ratios by `ultimate_loss`.
pub fn alae_ratio_triangle(
    paid_loss: &TriangleView<'_>,
    paid_alae: &TriangleView<'_>,
    ultimate_loss: &[f64],
    tail: f64,
    weighting: AlaeRatioWeighting,
) -> Result<AlaeRatioTriangleResult, ValidationError> {
    validate_triangle(paid_loss)?;
    validate_triangle(paid_alae)?;
    validate_tail(tail)?;
    let n = paid_loss.n_origins;
    if paid_alae.n_origins != n || paid_alae.n_devs != n || ultimate_loss.len() != n {
        return Err(ValidationError::ShapeMismatch);
    }
    if n < 2 {
        return Err(ValidationError::InvalidParameter(
            "ALAE ratio triangle requires at least a 2x2 triangle",
        ));
    }
    validate_nonnegative_finite_slice(ultimate_loss)?;

    let mut ratio_grid = vec![f64::NAN; n * n];
    for i in 0..n {
        for j in 0..(n - i) {
            let l = paid_loss.get(i, j);
            let a = paid_alae.get(i, j);
            if l > 0.0 {
                ratio_grid[i * n + j] = a / l;
            }
        }
    }

    let mut ldfs = Vec::with_capacity(n - 1);
    for j in 0..(n - 1) {
        let rows = n - 1 - j;
        let ldf = match weighting {
            AlaeRatioWeighting::RatioSum => {
                let mut num = 0.0;
                let mut den = 0.0;
                for i in 0..rows {
                    let q0 = ratio_grid[i * n + j];
                    let q1 = ratio_grid[i * n + (j + 1)];
                    if q0.is_finite() && q1.is_finite() {
                        num += q1;
                        den += q0;
                    }
                }
                if den <= 0.0 {
                    return Err(ValidationError::DivideByZero("ALAE ratio column sum"));
                }
                num / den
            }
            AlaeRatioWeighting::LossWeighted => {
                let mut a0 = 0.0;
                let mut l0 = 0.0;
                let mut a1 = 0.0;
                let mut l1 = 0.0;
                for i in 0..rows {
                    let lv0 = paid_loss.get(i, j);
                    let lv1 = paid_loss.get(i, j + 1);
                    if lv0 > 0.0 && lv1 > 0.0 {
                        a0 += paid_alae.get(i, j);
                        l0 += lv0;
                        a1 += paid_alae.get(i, j + 1);
                        l1 += lv1;
                    }
                }
                if l0 <= 0.0 || l1 <= 0.0 || a0 <= 0.0 {
                    return Err(ValidationError::DivideByZero("loss-weighted ALAE ratio"));
                }
                (a1 / l1) / (a0 / l0)
            }
        };
        ldfs.push(ldf);
    }

    let mut cdfs = vec![tail; n];
    for j in (0..(n - 1)).rev() {
        cdfs[j] = ldfs[j] * cdfs[j + 1];
    }

    let mut ultimate_ratios = Vec::with_capacity(n);
    let mut ultimate_alae = Vec::with_capacity(n);
    let mut reserves = Vec::with_capacity(n);
    let mut total_reserve = 0.0;

    for i in 0..n {
        let last_j = n - 1 - i;
        let q_last = ratio_grid[i * n + last_j];
        if !q_last.is_finite() {
            return Err(ValidationError::DivideByZero(
                "latest diagonal paid loss in ALAE ratio triangle",
            ));
        }
        let q_ult = q_last * cdfs[last_j];
        let a_ult = q_ult * ultimate_loss[i];
        let a_paid = paid_alae.get(i, last_j);
        let res = a_ult - a_paid;
        ultimate_ratios.push(q_ult);
        ultimate_alae.push(a_ult);
        reserves.push(res);
        total_reserve += res;
    }

    Ok(AlaeRatioTriangleResult {
        ratio_grid,
        ldfs,
        cdfs,
        tail,
        ultimate_ratios,
        ultimate_alae,
        reserves,
        total_reserve,
    })
}

// ===========================================================================
// 1.5b Clark (2003) / Johnson Ratio-to-Loss ALAE ODP-MLE
// ===========================================================================

#[inline]
fn eval_emergence(curve: GrowthCurve, t: f64, omega: f64, theta: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    match curve {
        GrowthCurve::Loglogistic => {
            let ratio = (t / theta).powf(omega);
            if ratio.is_infinite() {
                1.0
            } else {
                ratio / (1.0 + ratio)
            }
        }
        GrowthCurve::Weibull => -((-(t / theta).powf(omega)).exp_m1()),
    }
}

/// Clark (2003) / Johnson ratio-to-loss ALAE ODP-MLE model.
///
/// Models incremental paid ALAE `a_{i,j}` with
/// `E[a_{i,j}] = L_hat_i * ELR_A * (G(t_j; ω, θ) - G(t_{j-1}; ω, θ))`
/// using closed-form profile MLE for `ELR_A` and deterministic Nelder–Mead
/// optimization over `(ln ω, ln θ)`.
pub fn clark_alae_mle(
    cumulative_paid_alae: &TriangleView<'_>,
    ultimate_loss: &[f64],
    options: ClarkAlaeOptions,
) -> Result<ClarkAlaeResult, ValidationError> {
    validate_triangle(cumulative_paid_alae)?;
    let n = cumulative_paid_alae.n_origins;
    if ultimate_loss.len() != n {
        return Err(ValidationError::ShapeMismatch);
    }
    if n < 2 {
        return Err(ValidationError::InvalidParameter(
            "Clark ALAE MLE requires at least a 2x2 triangle",
        ));
    }
    if !options.initial_omega.is_finite()
        || options.initial_omega <= 0.0
        || !options.initial_theta.is_finite()
        || options.initial_theta <= 0.0
        || !options.tolerance.is_finite()
        || options.tolerance <= 0.0
    {
        return Err(ValidationError::InvalidParameter(
            "Clark ALAE initial parameters and tolerance must be finite and positive",
        ));
    }
    if options.max_iterations == 0 {
        return Err(ValidationError::InvalidIterations);
    }
    for &u in ultimate_loss {
        if !u.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if u <= 0.0 {
            return Err(ValidationError::NonPositive("ultimate_loss"));
        }
    }

    // Extract observed incremental ALAE cells (i, j, a_{i,j}).
    let mut obs: Vec<(usize, usize, f64)> = Vec::with_capacity(n * (n + 1) / 2);
    let mut sum_a = 0.0;
    for i in 0..n {
        let mut prev = 0.0;
        for j in 0..(n - i) {
            let cum = cumulative_paid_alae.get(i, j);
            let inc = cum - prev;
            prev = cum;
            obs.push((i, j, inc));
            sum_a += inc;
        }
    }
    if sum_a <= 0.0 {
        return Err(ValidationError::NonPositive("total observed paid ALAE"));
    }

    let mut g_vals = vec![0.0; n + 1];
    let mut neg_profile_ll = |log_omega: f64, log_theta: f64| -> (f64, f64) {
        if log_omega < -8.0 || log_omega > 6.0 || log_theta < -8.0 || log_theta > 8.0 {
            return (f64::INFINITY, f64::NAN);
        }
        let omega = log_omega.exp();
        let theta = log_theta.exp();
        g_vals[0] = 0.0;
        for j in 1..=n {
            g_vals[j] = eval_emergence(options.curve, j as f64, omega, theta);
        }
        let mut denom = 0.0;
        for &(i, j, _) in &obs {
            let dg = g_vals[j + 1] - g_vals[j];
            if !dg.is_finite() || dg <= 0.0 {
                return (f64::INFINITY, f64::NAN);
            }
            denom += ultimate_loss[i] * dg;
        }
        if !denom.is_finite() || denom <= 0.0 {
            return (f64::INFINITY, f64::NAN);
        }
        let elr = sum_a / denom;
        let mut ll = 0.0;
        for &(i, j, a_ij) in &obs {
            let dg = g_vals[j + 1] - g_vals[j];
            let mu = ultimate_loss[i] * elr * dg;
            if !mu.is_finite() || mu <= 0.0 {
                return (f64::INFINITY, f64::NAN);
            }
            ll += a_ij * mu.ln() - mu;
        }
        if !ll.is_finite() {
            return (f64::INFINITY, f64::NAN);
        }
        (-ll, elr)
    };

    // Deterministic 2D Nelder–Mead on (ln omega, ln theta).
    let x0 = [options.initial_omega.ln(), options.initial_theta.ln()];
    let mut simplex = [
        (x0, neg_profile_ll(x0[0], x0[1]).0),
        ([x0[0] + 0.35, x0[1]], neg_profile_ll(x0[0] + 0.35, x0[1]).0),
        ([x0[0], x0[1] + 0.35], neg_profile_ll(x0[0], x0[1] + 0.35).0),
    ];

    let mut converged = false;
    for _ in 0..options.max_iterations {
        simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
        let spread_f = (simplex[2].1 - simplex[0].1).abs();
        let spread_x = (simplex[2].0[0] - simplex[0].0[0])
            .abs()
            .max((simplex[2].0[1] - simplex[0].0[1]).abs())
            .max((simplex[1].0[0] - simplex[0].0[0]).abs())
            .max((simplex[1].0[1] - simplex[0].0[1]).abs());
        if spread_x < options.tolerance && spread_f < options.tolerance {
            converged = true;
            break;
        }

        let centroid = [
            f64::midpoint(simplex[0].0[0], simplex[1].0[0]),
            f64::midpoint(simplex[0].0[1], simplex[1].0[1]),
        ];
        let xr = [
            centroid[0] + (centroid[0] - simplex[2].0[0]),
            centroid[1] + (centroid[1] - simplex[2].0[1]),
        ];
        let fr = neg_profile_ll(xr[0], xr[1]).0;

        if fr < simplex[0].1 {
            let xe = [
                centroid[0] + 2.0 * (xr[0] - centroid[0]),
                centroid[1] + 2.0 * (xr[1] - centroid[1]),
            ];
            let fe = neg_profile_ll(xe[0], xe[1]).0;
            simplex[2] = if fe < fr { (xe, fe) } else { (xr, fr) };
        } else if fr < simplex[1].1 {
            simplex[2] = (xr, fr);
        } else {
            let xc = if fr < simplex[2].1 {
                [
                    centroid[0] + 0.5 * (xr[0] - centroid[0]),
                    centroid[1] + 0.5 * (xr[1] - centroid[1]),
                ]
            } else {
                [
                    centroid[0] + 0.5 * (simplex[2].0[0] - centroid[0]),
                    centroid[1] + 0.5 * (simplex[2].0[1] - centroid[1]),
                ]
            };
            let fc = neg_profile_ll(xc[0], xc[1]).0;
            if fc < simplex[2].1.min(fr) {
                simplex[2] = (xc, fc);
            } else {
                let best = simplex[0].0;
                for idx in 1..3 {
                    let pt = [
                        f64::midpoint(best[0], simplex[idx].0[0]),
                        f64::midpoint(best[1], simplex[idx].0[1]),
                    ];
                    simplex[idx] = (pt, neg_profile_ll(pt[0], pt[1]).0);
                }
            }
        }
    }

    simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
    let best_x = simplex[0].0;
    let (neg_ll, elr_alae) = neg_profile_ll(best_x[0], best_x[1]);
    if !neg_ll.is_finite() || !elr_alae.is_finite() {
        return Err(ValidationError::NotFinite);
    }

    let omega = best_x[0].exp();
    let theta = best_x[1].exp();
    let emergence_curve: Vec<f64> = (1..=n)
        .map(|j| eval_emergence(options.curve, j as f64, omega, theta))
        .collect();

    let mut reserves = Vec::with_capacity(n);
    let mut ultimate_alae = Vec::with_capacity(n);
    let mut total_reserve = 0.0;
    for i in 0..n {
        let last_j = n - 1 - i;
        let paid_to_date = cumulative_paid_alae.get(i, last_j);
        let g_last = emergence_curve[last_j];
        let res = ultimate_loss[i] * elr_alae * (1.0 - g_last);
        reserves.push(res);
        ultimate_alae.push(paid_to_date + res);
        total_reserve += res;
    }

    Ok(ClarkAlaeResult {
        curve: options.curve,
        omega,
        theta,
        elr_alae,
        log_likelihood: -neg_ll,
        emergence_curve,
        reserves,
        ultimate_alae,
        total_reserve,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Triangle;

    #[test]
    fn golden_ulae_classical_kittel_and_johnson() {
        // 1. Classical 50/50 (Kittel 1981 comparison)
        let classical =
            ulae_paid_to_paid(&[110.0], &[1000.0], 300.0, 500.0, UlaeOptions::default())
                .expect("classical ULAE");
        assert!((classical.selected_ratio - 0.11).abs() < 1e-12);
        assert!((classical.reserve - 71.5).abs() < 1e-12);

        // 2. Kittel (1981) growth-adjusted
        let kittel = ulae_kittel(
            &[110.0],
            &[1000.0],
            &[1200.0],
            300.0,
            500.0,
            UlaeOptions::default(),
        )
        .expect("Kittel ULAE");
        assert!((kittel.selected_ratio - 0.10).abs() < 1e-12);
        assert!((kittel.reserve - 65.0).abs() < 1e-12);

        // 3. Wendy Johnson (1989) count-based 2-yr runoff
        let johnson = ulae_johnson_count_based(
            &[200.0],
            &[ClaimCountYear {
                newly_reported: 1000.0,
                open: 0.0,
                closed: 0.0,
            }],
            &[
                ClaimCountYear {
                    newly_reported: 600.0,
                    open: 0.0,
                    closed: 0.0,
                },
                ClaimCountYear {
                    newly_reported: 200.0,
                    open: 0.0,
                    closed: 0.0,
                },
            ],
            JohnsonUlaeWeights {
                newly_reported: 1.0,
                open: 0.0,
                closed: 0.0,
            },
            0.05,
        )
        .expect("Johnson ULAE");
        assert!((johnson.base_unit_cost - 0.20).abs() < 1e-12);
        assert!((johnson.projected_unit_costs[0] - 0.21).abs() < 1e-12);
        assert!((johnson.projected_unit_costs[1] - 0.2205).abs() < 1e-12);
        assert!((johnson.reserve - 170.1).abs() < 1e-12);
    }

    #[test]
    fn mango_allen_expected_emergence_smooths_denominators() {
        let ults = [1000.0, 1100.0, 1200.0];
        let pay_pat = [0.5, 0.3, 0.2];
        let inc_pat = [0.8, 0.15, 0.05];
        let exp_paid = expected_calendar_emergence(&ults, &pay_pat).expect("exp_paid");
        let exp_inc = expected_calendar_emergence(&ults, &inc_pat).expect("exp_inc");
        // t=0: 1000*0.5=500; t=1: 1000*0.3 + 1100*0.5=850; t=2: 1000*0.2 + 1100*0.3 + 1200*0.5=1130.
        assert!((exp_paid[0] - 500.0).abs() < 1e-12);
        assert!((exp_paid[1] - 850.0).abs() < 1e-12);
        assert!((exp_paid[2] - 1130.0).abs() < 1e-12);

        let ma = ulae_mango_allen(
            &[65.0, 90.0, 115.0],
            &exp_paid,
            &exp_inc,
            300.0,
            500.0,
            UlaeOptions::default(),
        )
        .expect("Mango-Allen");
        assert!(ma.reserve > 0.0 && ma.selected_ratio.is_finite());
    }

    #[test]
    fn alae_ratio_triangle_and_clark_alae_recover_exact_synthetic_alae() {
        // Exact synthetic triangle where ALAE follows LogLogistic(omega=1.8, theta=2.2), ELR_A=0.12.
        let ults = [10000.0, 11000.0, 12000.0, 13000.0, 14000.0];
        let omega_true = 1.8_f64;
        let theta_true = 2.2_f64;
        let elr_true = 0.12_f64;
        let n = ults.len();

        let mut alae_data = vec![f64::NAN; n * n];
        let mut loss_data = vec![f64::NAN; n * n];
        for i in 0..n {
            for j in 0..(n - i) {
                let t = (j + 1) as f64;
                let g_alae = eval_emergence(GrowthCurve::Loglogistic, t, omega_true, theta_true);
                let g_loss = eval_emergence(GrowthCurve::Loglogistic, t, 1.4, 1.5);
                alae_data[i * n + j] = ults[i] * elr_true * g_alae;
                loss_data[i * n + j] = ults[i] * g_loss;
            }
        }
        let alae_tri = Triangle::new(n, n, alae_data);
        let loss_tri = Triangle::new(n, n, loss_data);

        let clark = clark_alae_mle(&alae_tri.view(), &ults, ClarkAlaeOptions::default())
            .expect("Clark ALAE MLE");
        assert!(clark.converged);
        assert!(
            (clark.omega - omega_true).abs() < 1e-3,
            "omega={}",
            clark.omega
        );
        assert!(
            (clark.theta - theta_true).abs() < 1e-3,
            "theta={}",
            clark.theta
        );
        assert!(
            (clark.elr_alae - elr_true).abs() < 1e-4,
            "elr={}",
            clark.elr_alae
        );

        let ratio = alae_ratio_triangle(
            &loss_tri.view(),
            &alae_tri.view(),
            &ults,
            1.0,
            AlaeRatioWeighting::LossWeighted,
        )
        .expect("ALAE ratio triangle");
        assert!(ratio.total_reserve > 0.0);
        assert_eq!(ratio.reserves.len(), n);
    }
}
