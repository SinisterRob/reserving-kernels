//! Sub-annual LDF and CDF interpolation kernels.
//!
//! Converts annual (or arbitrary strictly increasing) maturity anchor nodes
//! `(anchor_ages, anchor_cdfs)` into smooth sub-annual CDFs, percent reported,
//! and implied sub-period LDFs:
//!
//! * [`interp_linear_cdf`] — direct piecewise-linear CDF interpolation.
//! * [`interp_linear_pct_reported`] — piecewise-linear percent-reported
//!   (`1 / CDF`) interpolation, avoiding convexity bias.
//! * [`interp_exponential`] — constant-rate geometric interpolation (Bloom 2015),
//!   preserving exact multiplicative LDF parity.
//! * [`interp_inverse_power`] — Sherman (1984) inverse power curve
//!   `f(t) - 1 = a · t^{-b}` integrated over sub-annual intervals.
//! * [`interp_swimon`] — Robbin (2018) smooth, monotonic, concavity-preserving
//!   interpolation designed to eliminate sawtooth oscillations in expected IBNR runoff.
//! * [`interp_spliced_weibull`] — Robbin (2018) piecewise Weibull curve spliced
//!   across annual knots.
//! * [`interp_ivf12`] — Robbin (2018) Interpolated Volume Factor (12-month basis)
//!   method preserving exact 12-month factor parity with smooth emergence.

use crate::validate::{ValidationError, validate_positive_finite_slice};

/// Sub-annual CDF/LDF interpolation method selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SubannualInterpMethod {
    /// Piecewise-linear interpolation of CDFs.
    LinearCdf,
    /// Piecewise-linear interpolation of percent reported (`1 / CDF`).
    LinearPctReported,
    /// Constant-rate geometric interpolation (`CDF(t) = CDF_0^(1-w) * CDF_1^w`).
    Exponential,
    /// Sherman (1984) inverse power curve `f(t) - 1 = a * t^{-b}`.
    InversePower,
    /// Robbin (2018) smooth monotonic interpolation (eliminates IBNR sawtooth).
    Swimon,
    /// Robbin (2018) piecewise Weibull spliced across annual knots.
    SplicedWeibull,
    /// Robbin (2018) Interpolated Volume Factor (12-month basis).
    Ivf12,
}

/// Interpolated sub-annual development pattern evaluated at `target_ages`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InterpolatedPattern {
    /// Interpolation method used.
    pub method: SubannualInterpMethod,
    /// Evaluation ages in months (same order as caller's `target_ages`).
    pub ages_months: Vec<f64>,
    /// Interpolated cumulative development factors `CDF(t)`.
    pub cdfs: Vec<f64>,
    /// Implied percent reported `1 / CDF(t)`.
    pub pct_reported: Vec<f64>,
    /// Implied step-to-step LDFs `CDF(t_k) / CDF(t_{k+1})` (length `target_ages.len() - 1`).
    pub ldfs: Vec<f64>,
}

/// Result of Sherman (1984) inverse-power sub-annual interpolation, including
/// the fitted curve parameters `f(t) - 1 = a · t^{-b}`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InversePowerInterpResult {
    /// Evaluated sub-annual pattern (`cdfs`, `pct_reported`, `ldfs`).
    pub pattern: InterpolatedPattern,
    /// Scale parameter `a > 0` in `f(t) - 1 = a · t^{-b}`.
    pub a: f64,
    /// Decay exponent `b > 0` in `f(t) - 1 = a · t^{-b}`.
    pub b: f64,
    /// Coefficient of determination `R²` of the log-linear fit `ln(f_k - 1) = ln(a) - b ln(t_k)`.
    pub r_squared: f64,
}

/// Dispatch to any of the 7 sub-annual interpolation kernels via [`SubannualInterpMethod`].
pub fn interpolate_pattern(
    method: SubannualInterpMethod,
    anchor_ages: &[f64],
    anchor_cdfs: &[f64],
    target_ages: &[f64],
) -> Result<InterpolatedPattern, ValidationError> {
    match method {
        SubannualInterpMethod::LinearCdf => {
            interp_linear_cdf(anchor_ages, anchor_cdfs, target_ages)
        }
        SubannualInterpMethod::LinearPctReported => {
            interp_linear_pct_reported(anchor_ages, anchor_cdfs, target_ages)
        }
        SubannualInterpMethod::Exponential => {
            interp_exponential(anchor_ages, anchor_cdfs, target_ages)
        }
        SubannualInterpMethod::InversePower => {
            Ok(interp_inverse_power(anchor_ages, anchor_cdfs, target_ages)?.pattern)
        }
        SubannualInterpMethod::Swimon => interp_swimon(anchor_ages, anchor_cdfs, target_ages),
        SubannualInterpMethod::SplicedWeibull => {
            interp_spliced_weibull(anchor_ages, anchor_cdfs, target_ages)
        }
        SubannualInterpMethod::Ivf12 => interp_ivf12(anchor_ages, anchor_cdfs, target_ages),
    }
}

fn validate_anchors_and_targets(
    anchor_ages: &[f64],
    anchor_cdfs: &[f64],
    target_ages: &[f64],
) -> Result<(), ValidationError> {
    if anchor_ages.len() < 2 || anchor_ages.len() != anchor_cdfs.len() || target_ages.is_empty() {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_positive_finite_slice(anchor_ages, "anchor_ages")?;
    validate_positive_finite_slice(anchor_cdfs, "anchor_cdfs")?;
    validate_positive_finite_slice(target_ages, "target_ages")?;
    for k in 1..anchor_ages.len() {
        if anchor_ages[k] <= anchor_ages[k - 1] {
            return Err(ValidationError::InvalidParameter(
                "anchor_ages must be strictly increasing",
            ));
        }
    }
    Ok(())
}

fn build_pattern(
    method: SubannualInterpMethod,
    target_ages: &[f64],
    cdfs: Vec<f64>,
) -> InterpolatedPattern {
    let pct_reported: Vec<f64> = cdfs.iter().map(|&c| 1.0 / c).collect();
    let mut ldfs = Vec::with_capacity(cdfs.len().saturating_sub(1));
    for k in 0..cdfs.len().saturating_sub(1) {
        ldfs.push(cdfs[k] / cdfs[k + 1]);
    }
    InterpolatedPattern {
        method,
        ages_months: target_ages.to_vec(),
        cdfs,
        pct_reported,
        ldfs,
    }
}

/// Find bracket index `k` such that `ages[k] <= t <= ages[k + 1]`.
/// Assumes `ages[0] <= t <= ages[m - 1]`.
#[inline]
fn find_bracket(ages: &[f64], t: f64) -> usize {
    let m = ages.len();
    let pos = ages.partition_point(|&a| a < t);
    if pos == 0 {
        0
    } else if pos >= m {
        m - 2
    } else if (ages[pos] - t).abs() <= 1e-12 * t.max(1.0) && pos + 1 < m {
        pos
    } else {
        pos - 1
    }
}

// ===========================================================================
// 1. Linear CDF
// ===========================================================================

/// Linearly interpolate CDFs between adjacent anchor maturities:
/// `CDF(t) = (1 - w) * CDF(t_k) + w * CDF(t_{k+1})`.
///
/// Requires `target_ages` within `[anchor_ages[0], anchor_ages[last]]`.
pub fn interp_linear_cdf(
    anchor_ages: &[f64],
    anchor_cdfs: &[f64],
    target_ages: &[f64],
) -> Result<InterpolatedPattern, ValidationError> {
    validate_anchors_and_targets(anchor_ages, anchor_cdfs, target_ages)?;
    let t_min = anchor_ages[0];
    let t_max = *anchor_ages.last().expect("len >= 2");
    let tol = 1e-9 * t_max;

    let mut cdfs = Vec::with_capacity(target_ages.len());
    for &t in target_ages {
        if t < t_min - tol || t > t_max + tol {
            return Err(ValidationError::InvalidParameter(
                "target age outside [anchor_ages[0], anchor_ages[last]] for linear CDF interpolation",
            ));
        }
        let tc = t.clamp(t_min, t_max);
        let k = find_bracket(anchor_ages, tc);
        let w = (tc - anchor_ages[k]) / (anchor_ages[k + 1] - anchor_ages[k]);
        let cdf = (1.0 - w) * anchor_cdfs[k] + w * anchor_cdfs[k + 1];
        cdfs.push(cdf);
    }
    Ok(build_pattern(
        SubannualInterpMethod::LinearCdf,
        target_ages,
        cdfs,
    ))
}

// ===========================================================================
// 2. Linear Percent Reported (1 / CDF)
// ===========================================================================

/// Linearly interpolate percent reported `p(t) = 1 / CDF(t)` between adjacent
/// anchor nodes, then invert back to `CDF(t) = 1 / p(t)`.
///
/// Avoids the convexity bias of direct linear CDF interpolation. For ages in
/// `(0, anchor_ages[0])`, interpolates linearly from the origin `(0, p(0) = 0)`.
pub fn interp_linear_pct_reported(
    anchor_ages: &[f64],
    anchor_cdfs: &[f64],
    target_ages: &[f64],
) -> Result<InterpolatedPattern, ValidationError> {
    validate_anchors_and_targets(anchor_ages, anchor_cdfs, target_ages)?;
    let t_max = *anchor_ages.last().expect("len >= 2");
    let tol = 1e-9 * t_max;

    let mut cdfs = Vec::with_capacity(target_ages.len());
    for &t in target_ages {
        if t > t_max + tol {
            return Err(ValidationError::InvalidParameter(
                "target age exceeds maximum anchor age",
            ));
        }
        let tc = t.min(t_max);
        let p = if tc < anchor_ages[0] {
            (tc / anchor_ages[0]) * (1.0 / anchor_cdfs[0])
        } else {
            let k = find_bracket(anchor_ages, tc);
            let w = (tc - anchor_ages[k]) / (anchor_ages[k + 1] - anchor_ages[k]);
            let p0 = 1.0 / anchor_cdfs[k];
            let p1 = 1.0 / anchor_cdfs[k + 1];
            (1.0 - w) * p0 + w * p1
        };
        if !(p > 0.0) || !p.is_finite() {
            return Err(ValidationError::DivideByZero("percent reported"));
        }
        cdfs.push(1.0 / p);
    }
    Ok(build_pattern(
        SubannualInterpMethod::LinearPctReported,
        target_ages,
        cdfs,
    ))
}

// ===========================================================================
// 3. Exponential / Geometric CDF (Bloom 2015)
// ===========================================================================

/// Constant-rate geometric CDF interpolation (Bloom 2015):
/// `CDF(t) = CDF(t_k)^(1 - w) * CDF(t_{k+1})^w`.
///
/// Ensures that sub-period LDFs within each interval are constant and multiply
/// cleanly back to the exact anchor-to-anchor LDF.
pub fn interp_exponential(
    anchor_ages: &[f64],
    anchor_cdfs: &[f64],
    target_ages: &[f64],
) -> Result<InterpolatedPattern, ValidationError> {
    validate_anchors_and_targets(anchor_ages, anchor_cdfs, target_ages)?;
    let t_min = anchor_ages[0];
    let t_max = *anchor_ages.last().expect("len >= 2");
    let tol = 1e-9 * t_max;

    let mut cdfs = Vec::with_capacity(target_ages.len());
    for &t in target_ages {
        if t < t_min - tol || t > t_max + tol {
            return Err(ValidationError::InvalidParameter(
                "target age outside [anchor_ages[0], anchor_ages[last]] for exponential interpolation",
            ));
        }
        let tc = t.clamp(t_min, t_max);
        let k = find_bracket(anchor_ages, tc);
        let w = (tc - anchor_ages[k]) / (anchor_ages[k + 1] - anchor_ages[k]);
        let ln_cdf = (1.0 - w) * anchor_cdfs[k].ln() + w * anchor_cdfs[k + 1].ln();
        cdfs.push(ln_cdf.exp());
    }
    Ok(build_pattern(
        SubannualInterpMethod::Exponential,
        target_ages,
        cdfs,
    ))
}

// ===========================================================================
// 4. Sherman (1984) Inverse Power Curve
// ===========================================================================

/// 8-point Gauss–Legendre quadrature nodes and weights on `[-1, 1]`.
const GL8_NODES: [f64; 8] = [
    -0.960_289_856_497_536_3,
    -0.796_666_477_413_626_7,
    -0.525_532_409_916_329,
    -0.183_434_642_495_649_8,
    0.183_434_642_495_649_8,
    0.525_532_409_916_329,
    0.796_666_477_413_626_7,
    0.960_289_856_497_536_3,
];
const GL8_WEIGHTS: [f64; 8] = [
    0.101_228_536_290_376_26,
    0.222_381_034_453_374_47,
    0.313_706_645_877_887_3,
    0.362_683_783_378_362,
    0.362_683_783_378_362,
    0.313_706_645_877_887_3,
    0.222_381_034_453_374_47,
    0.101_228_536_290_376_26,
];

/// Integrate `ln(1 + a * u^{-b})` over `[lo, hi]` (`0 < lo <= hi`) via 8-point Gauss–Legendre.
fn integrate_sherman_log_rate(lo: f64, hi: f64, a: f64, b: f64) -> f64 {
    if hi <= lo {
        return 0.0;
    }
    let mid = 0.5 * (lo + hi);
    let half = 0.5 * (hi - lo);
    let mut acc = 0.0;
    for (&node, &weight) in GL8_NODES.iter().zip(GL8_WEIGHTS.iter()) {
        let u = mid + half * node;
        let val = (a * u.powf(-b)).ln_1p();
        acc += weight * val;
    }
    half * acc
}

/// Fit Sherman (1984) inverse power curve `f(t) - 1 = a · t^{-b}` across annual
/// nodes and evaluate smooth sub-annual LDFs and CDFs at arbitrary ages `t > 0`.
///
/// The continuous log-development rate `λ(u) ∝ ln(1 + a · u^{-b})` is integrated
/// over each anchor interval `[t_k, t_{k+1}]` and normalized to preserve exact
/// anchor CDFs at every knot (ensuring exact sub-annual product parity) while
/// front-loading emergence realistically within each period. For target ages
/// outside `[t_0, t_{m-1}]`, the fitted curve extrapolates smoothly via the
/// unnormalized integral scaled by the adjacent anchor step width.
pub fn interp_inverse_power(
    anchor_ages: &[f64],
    anchor_cdfs: &[f64],
    target_ages: &[f64],
) -> Result<InversePowerInterpResult, ValidationError> {
    validate_anchors_and_targets(anchor_ages, anchor_cdfs, target_ages)?;

    // Collect (ln t_k, ln(f_k - 1)) for annual links where f_k = CDF_k / CDF_{k+1} > 1.
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for k in 0..(anchor_ages.len() - 1) {
        let f_k = anchor_cdfs[k] / anchor_cdfs[k + 1];
        if f_k > 1.0 + 1e-12 {
            xs.push(anchor_ages[k].ln());
            ys.push((f_k - 1.0).ln());
        }
    }
    if xs.len() < 2 {
        return Err(ValidationError::InvalidParameter(
            "interp_inverse_power requires at least two anchor links with LDF > 1.0",
        ));
    }

    let n = xs.len() as f64;
    let x_bar = xs.iter().sum::<f64>() / n;
    let y_bar = ys.iter().sum::<f64>() / n;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut syy = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let dx = x - x_bar;
        let dy = y - y_bar;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    if sxx <= 0.0 {
        return Err(ValidationError::InvalidParameter(
            "degenerate anchor ages in inverse power fit",
        ));
    }
    let slope = sxy / sxx; // = -b
    let intercept = y_bar - slope * x_bar; // = ln(a)
    let b = -slope;
    let a = intercept.exp();
    if !(a > 0.0) || !a.is_finite() || !(b > 0.0) || !b.is_finite() {
        return Err(ValidationError::InvalidParameter(
            "inverse power fit requires decaying development factors (b > 0)",
        ));
    }
    let r_squared = if syy > 0.0 {
        ((sxy * sxy) / (sxx * syy)).clamp(0.0, 1.0)
    } else {
        1.0
    };

    let t_min = anchor_ages[0];
    let t_max = *anchor_ages.last().expect("len >= 2");
    let first_step = anchor_ages[1] - anchor_ages[0];
    let last_step = anchor_ages[anchor_ages.len() - 1] - anchor_ages[anchor_ages.len() - 2];

    let mut cdfs = Vec::with_capacity(target_ages.len());
    for &t in target_ages {
        let cdf = if t < t_min {
            let integral = integrate_sherman_log_rate(t, t_min, a, b) / first_step;
            anchor_cdfs[0] * integral.exp()
        } else if t > t_max {
            let integral = integrate_sherman_log_rate(t_max, t, a, b) / last_step;
            (*anchor_cdfs.last().expect("non-empty") * (-integral).exp()).max(1.0)
        } else {
            let k = find_bracket(anchor_ages, t);
            let t0 = anchor_ages[k];
            let t1 = anchor_ages[k + 1];
            let full = integrate_sherman_log_rate(t0, t1, a, b);
            let w = if full > 0.0 {
                (integrate_sherman_log_rate(t0, t, a, b) / full).clamp(0.0, 1.0)
            } else {
                (t - t0) / (t1 - t0)
            };
            ((1.0 - w) * anchor_cdfs[k].ln() + w * anchor_cdfs[k + 1].ln()).exp()
        };
        cdfs.push(cdf);
    }

    Ok(InversePowerInterpResult {
        pattern: build_pattern(SubannualInterpMethod::InversePower, target_ages, cdfs),
        a,
        b,
        r_squared,
    })
}

// ===========================================================================
// 5. SWIMON — Smooth Monotonic Interpolation (Robbin 2018)
// ===========================================================================

/// Evaluate cubic Hermite spline on `[x0, x1]` with values `(y0, y1)` and
/// endpoint derivatives `(m0, m1)`.
#[inline]
fn eval_cubic_hermite(x0: f64, x1: f64, y0: f64, y1: f64, m0: f64, m1: f64, x: f64) -> f64 {
    let h = x1 - x0;
    let t = ((x - x0) / h).clamp(0.0, 1.0);
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
}

/// Robbin (2018) SWIMON — Smooth, monotonic sub-annual interpolation designed
/// to eliminate artificial sawtooth oscillations in expected IBNR runoff.
///
/// Interpolates percent reported `P(t) = 1 / CDF(t)` anchored at `(0, 0)` and
/// every annual knot `(anchor_ages[k], 1 / anchor_cdfs[k])` using a `C¹`
/// monotonic, concavity-preserving cubic Hermite spline:
///
/// * Exact preservation of all annual anchor CDFs.
/// * Continuous first derivative `P'(t) >= 0` across every annual boundary (no
///   kinks or sawtooth jumps in incremental emergence).
/// * When the annual emergence increments are non-increasing (concave `P`),
///   enforces `m_k >= s_k >= m_{k+1}` and the cubic concavity condition
///   `m_k + 2 m_{k+1} <= 3 s_k <= 2 m_k + m_{k+1}` on every interval so that
///   `P''(t) <= 0` everywhere and sub-annual IBNR runoff is strictly monotonic!
pub fn interp_swimon(
    anchor_ages: &[f64],
    anchor_cdfs: &[f64],
    target_ages: &[f64],
) -> Result<InterpolatedPattern, ValidationError> {
    validate_anchors_and_targets(anchor_ages, anchor_cdfs, target_ages)?;
    let t_max = *anchor_ages.last().expect("len >= 2");
    let tol = 1e-9 * t_max;

    // Prepend origin knot (0.0, 0.0) so P(t) is smoothly defined on (0, t_max].
    let n_knots = anchor_ages.len() + 1;
    let mut xs = Vec::with_capacity(n_knots);
    let mut ys = Vec::with_capacity(n_knots);
    xs.push(0.0);
    ys.push(0.0);
    for (&a, &c) in anchor_ages.iter().zip(anchor_cdfs.iter()) {
        xs.push(a);
        ys.push(1.0 / c);
    }

    // Secant slopes s[k] = (y[k+1] - y[k]) / (x[k+1] - x[k]) for k = 0..n_knots-1.
    let n_seg = n_knots - 1;
    let mut h = Vec::with_capacity(n_seg);
    let mut s = Vec::with_capacity(n_seg);
    for k in 0..n_seg {
        let hk = xs[k + 1] - xs[k];
        h.push(hk);
        s.push((ys[k + 1] - ys[k]) / hk);
    }

    // Step 1: Fritsch–Butland weighted harmonic-mean interior tangents for C¹ monotonicity.
    let mut m = vec![0.0; n_knots];
    for k in 1..n_seg {
        if s[k - 1] * s[k] <= 0.0 {
            m[k] = 0.0;
        } else {
            let w1 = 2.0 * h[k] + h[k - 1];
            let w2 = h[k] + 2.0 * h[k - 1];
            m[k] = (w1 + w2) / (w1 / s[k - 1] + w2 / s[k]);
        }
    }

    // Endpoint tangents via non-centered 3-point formula, clamped for monotonicity/concavity.
    m[0] = ((2.0 * h[0] + h[1]) * s[0] - h[0] * s[1]) / (h[0] + h[1]);
    if m[0] * s[0] <= 0.0 {
        m[0] = 0.0;
    } else if s[0] > 0.0 && m[0] > 2.0 * s[0] {
        m[0] = 2.0 * s[0];
    }

    let last = n_seg - 1;
    m[n_knots - 1] = ((2.0 * h[last] + h[last - 1]) * s[last] - h[last] * s[last - 1])
        / (h[last] + h[last - 1]);
    if m[n_knots - 1] * s[last] <= 0.0 {
        m[n_knots - 1] = 0.0;
    } else if s[last] > 0.0 && m[n_knots - 1] > 2.0 * s[last] {
        m[n_knots - 1] = 2.0 * s[last];
    }

    // Step 2: Check if secants are non-increasing (concave emergence) and enforce
    // exact cubic concavity P''(t) <= 0 on every segment:
    //   P''(x_k) <= 0  <=>  2 m_k + m_{k+1} >= 3 s_k
    //   P''(x_{k+1}) <= 0  <=>  m_k + 2 m_{k+1} <= 3 s_k
    // Combined with m_k >= s_k >= m_{k+1} >= 0, this guarantees P'(t) is
    // monotonically non-increasing across the entire domain (zero sawtooth!).
    let is_concave_data = s.windows(2).all(|w| w[0] >= w[1] - 1e-12 && w[1] >= -1e-12);
    if is_concave_data {
        // Clamp interior tangents into [s[k], s[k-1]] first.
        for k in 1..n_seg {
            m[k] = m[k].clamp(s[k].max(0.0), s[k - 1].max(0.0));
        }
        m[0] = m[0].clamp(s[0].max(0.0), (2.0 * s[0]).max(0.0));
        m[n_knots - 1] = m[n_knots - 1].clamp(0.0, s[last].max(0.0));

        // Iterative projection onto the linear concavity inequalities (converges in a few sweeps).
        for _ in 0..8 {
            for k in 0..n_seg {
                let sk = s[k].max(0.0);
                // Enforce m[k] + 2 m[k+1] <= 3 s_k (reduce m[k+1] if needed, or raise m[k]).
                if m[k] + 2.0 * m[k + 1] > 3.0 * sk {
                    let target_m_next = 0.5 * (3.0 * sk - m[k]);
                    let lo_next = if k + 1 < n_seg { s[k + 1].max(0.0) } else { 0.0 };
                    m[k + 1] = target_m_next.clamp(lo_next, sk);
                }
                // Enforce 2 m[k] + m[k+1] >= 3 s_k (raise m[k] within [s_k, s_{k-1}]).
                if 2.0 * m[k] + m[k + 1] < 3.0 * sk {
                    let target_m_cur = 0.5 * (3.0 * sk - m[k + 1]);
                    let hi_cur = if k > 0 { s[k - 1].max(0.0) } else { 2.0 * sk };
                    m[k] = target_m_cur.clamp(sk, hi_cur);
                }
            }
        }
    } else {
        // General Fritsch–Carlson monotonicity constraint (alpha^2 + beta^2 <= 9).
        for k in 0..n_seg {
            if s[k].abs() < 1e-15 {
                m[k] = 0.0;
                m[k + 1] = 0.0;
            } else {
                let alpha = m[k] / s[k];
                let beta = m[k + 1] / s[k];
                let r2 = alpha * alpha + beta * beta;
                if r2 > 9.0 {
                    let tau = 3.0 / r2.sqrt();
                    m[k] = tau * alpha * s[k];
                    m[k + 1] = tau * beta * s[k];
                }
            }
        }
    }

    let mut cdfs = Vec::with_capacity(target_ages.len());
    for &t in target_ages {
        if t > t_max + tol {
            return Err(ValidationError::InvalidParameter(
                "target age exceeds maximum anchor age for SWIMON interpolation",
            ));
        }
        let tc = t.min(t_max);
        let k = find_bracket(&xs, tc);
        let p = eval_cubic_hermite(xs[k], xs[k + 1], ys[k], ys[k + 1], m[k], m[k + 1], tc);
        if !(p > 0.0) || !p.is_finite() {
            return Err(ValidationError::DivideByZero("SWIMON percent reported"));
        }
        cdfs.push(1.0 / p);
    }

    Ok(build_pattern(
        SubannualInterpMethod::Swimon,
        target_ages,
        cdfs,
    ))
}

// ===========================================================================
// 6. Spliced Weibull (Robbin 2018)
// ===========================================================================

/// Robbin (2018) Spliced Weibull sub-annual interpolation.
///
/// Splices a two-parameter Weibull emergence curve `P(t) = 1 - exp(-a_k · t^{b_k})`
/// across each consecutive pair of annual knots `(t_k, P_k)` and `(t_{k+1}, P_{k+1})`,
/// preserving exact annual anchor points while producing smooth sub-annual emergence.
///
/// For `t in (0, anchor_ages[0])`, evaluates the first segment's Weibull curve
/// (which naturally satisfies `P(0) = 0`). If a knot has `P = 1.0` (i.e. `CDF = 1.0`,
/// fully developed), falls back cleanly to linear percent-reported on that final segment.
pub fn interp_spliced_weibull(
    anchor_ages: &[f64],
    anchor_cdfs: &[f64],
    target_ages: &[f64],
) -> Result<InterpolatedPattern, ValidationError> {
    validate_anchors_and_targets(anchor_ages, anchor_cdfs, target_ages)?;
    let t_max = *anchor_ages.last().expect("len >= 2");
    let tol = 1e-9 * t_max;

    let pcts: Vec<f64> = anchor_cdfs.iter().map(|&c| 1.0 / c).collect();
    let n_seg = anchor_ages.len() - 1;

    // Fit (ln_a, b) for each interior segment where 0 < p0 < p1 < 1.
    let mut seg_params: Vec<Option<(f64, f64)>> = Vec::with_capacity(n_seg);
    for k in 0..n_seg {
        let p0 = pcts[k];
        let p1 = pcts[k + 1];
        if p0 > 0.0 && p0 < 1.0 - 1e-12 && p1 > p0 && p1 < 1.0 - 1e-12 {
            let y0 = (-(1.0 - p0).ln()).ln();
            let y1 = (-(1.0 - p1).ln()).ln();
            let x0 = anchor_ages[k].ln();
            let x1 = anchor_ages[k + 1].ln();
            let b = (y1 - y0) / (x1 - x0);
            let ln_a = y0 - b * x0;
            seg_params.push(Some((ln_a, b)));
        } else {
            seg_params.push(None);
        }
    }

    let mut cdfs = Vec::with_capacity(target_ages.len());
    for &t in target_ages {
        if t > t_max + tol {
            return Err(ValidationError::InvalidParameter(
                "target age exceeds maximum anchor age for spliced Weibull interpolation",
            ));
        }
        let tc = t.min(t_max);
        let p = if tc < anchor_ages[0] {
            if let Some((ln_a, b)) = seg_params[0] {
                let v = (ln_a + b * tc.ln()).exp();
                -(-v).exp_m1()
            } else {
                (tc / anchor_ages[0]) * pcts[0]
            }
        } else {
            let k = find_bracket(anchor_ages, tc);
            if let Some((ln_a, b)) = seg_params[k] {
                let v = (ln_a + b * tc.ln()).exp();
                -(-v).exp_m1()
            } else {
                // Fully mature boundary (P = 1.0) or flat segment: linear fallback in P-space.
                let w = (tc - anchor_ages[k]) / (anchor_ages[k + 1] - anchor_ages[k]);
                (1.0 - w) * pcts[k] + w * pcts[k + 1]
            }
        };
        if !(p > 0.0) || !p.is_finite() {
            return Err(ValidationError::DivideByZero("Weibull percent reported"));
        }
        cdfs.push(1.0 / p);
    }

    Ok(build_pattern(
        SubannualInterpMethod::SplicedWeibull,
        target_ages,
        cdfs,
    ))
}

// ===========================================================================
// 7. IVF12 — Interpolated Volume Factor, 12-Month Basis (Robbin 2018)
// ===========================================================================

/// Robbin (2018) Interpolated Volume Factor (12-month basis) method.
///
/// Computes the 12-month development factor `F_12(t_k) = CDF(t_k) / CDF(t_k + 12)`
/// at each annual knot, smoothly interpolates `F_12(t)` geometrically across
/// intermediate sub-annual ages, and recursively constructs
/// `CDF(t) = F_12(t) * CDF(t + 12)` backward from the final 12-month window.
///
/// Preserves every annual anchor CDF identically and guarantees exact 12-month
/// multiplicative product parity across every annual window.
///
/// Requires equally spaced annual anchor ages (12-month spacing, e.g. `12, 24, 36, ...`).
pub fn interp_ivf12(
    anchor_ages: &[f64],
    anchor_cdfs: &[f64],
    target_ages: &[f64],
) -> Result<InterpolatedPattern, ValidationError> {
    validate_anchors_and_targets(anchor_ages, anchor_cdfs, target_ages)?;

    let m = anchor_ages.len();
    let step = anchor_ages[1] - anchor_ages[0];
    for k in 1..m {
        if ((anchor_ages[k] - anchor_ages[k - 1]) - step).abs() > 1e-6 * step {
            return Err(ValidationError::InvalidParameter(
                "interp_ivf12 requires equally spaced anchor ages (e.g. 12, 24, 36, ...)",
            ));
        }
    }

    let t_min = anchor_ages[0];
    let t_max = anchor_ages[m - 1];
    let tol = 1e-9 * t_max;

    // 12-month volume factors at each knot: F(t_k) = CDF(t_k) / CDF(t_{k+1}),
    // with F(t_{m-1}) = 1.0 (beyond the final knot, incremental annual development is 1.0).
    let mut f12 = Vec::with_capacity(m);
    for k in 0..(m - 1) {
        f12.push(anchor_cdfs[k] / anchor_cdfs[k + 1]);
    }
    f12.push(1.0);

    // Evaluate F_12(t) via log-linear interpolation between adjacent knots.
    let eval_f12 = |t: f64| -> f64 {
        if t >= t_max {
            return 1.0;
        }
        let tc = t.max(t_min);
        let k = find_bracket(anchor_ages, tc);
        let w = (tc - anchor_ages[k]) / (anchor_ages[k + 1] - anchor_ages[k]);
        ((1.0 - w) * f12[k].ln() + w * f12[k + 1].ln()).exp()
    };

    // Evaluate CDF(t) by stepping forward in `step` (12m) increments until landing
    // inside the final window `[t_{m-2}, t_{m-1}]`, where CDF is anchored cleanly.
    let eval_cdf = |t: f64| -> f64 {
        let mut prod = 1.0;
        let mut cur = t;
        let last_window_start = anchor_ages[m - 2];
        while cur < last_window_start - tol {
            prod *= eval_f12(cur);
            cur += step;
        }
        // Now cur is in [t_{m-2}, t_{m-1}]: geometric anchor inside the final window.
        let w = ((cur - last_window_start) / step).clamp(0.0, 1.0);
        let tail_cdf = ((1.0 - w) * anchor_cdfs[m - 2].ln() + w * anchor_cdfs[m - 1].ln()).exp();
        prod * tail_cdf
    };

    let mut cdfs = Vec::with_capacity(target_ages.len());
    for &t in target_ages {
        if t < t_min - tol || t > t_max + tol {
            return Err(ValidationError::InvalidParameter(
                "target age outside [anchor_ages[0], anchor_ages[last]] for IVF12 interpolation",
            ));
        }
        cdfs.push(eval_cdf(t.clamp(t_min, t_max)));
    }

    Ok(build_pattern(
        SubannualInterpMethod::Ivf12,
        target_ages,
        cdfs,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_annual() -> ([f64; 5], [f64; 5]) {
        // Annual ages 12..60 months, smooth decelerating pattern.
        let ages = [12.0, 24.0, 36.0, 48.0, 60.0];
        let cdfs = [2.50, 1.45, 1.18, 1.06, 1.00];
        (ages, cdfs)
    }

    #[test]
    fn all_interpolators_preserve_exact_annual_knots() {
        let (ages, cdfs) = sample_annual();
        let targets = [12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0, 54.0, 60.0];
        let methods = [
            SubannualInterpMethod::LinearCdf,
            SubannualInterpMethod::LinearPctReported,
            SubannualInterpMethod::Exponential,
            SubannualInterpMethod::InversePower,
            SubannualInterpMethod::Swimon,
            SubannualInterpMethod::SplicedWeibull,
            SubannualInterpMethod::Ivf12,
        ];
        for method in methods {
            let pat = interpolate_pattern(method, &ages, &cdfs, &targets).expect("interp");
            for (k, &expected) in cdfs.iter().enumerate() {
                let got = pat.cdfs[k * 2];
                assert!(
                    (got - expected).abs() < 1e-10,
                    "{method:?} knot {k}: got {got}, expected {expected}"
                );
            }
            // Monotonic non-increasing CDFs across all targets.
            for w in pat.cdfs.windows(2) {
                assert!(
                    w[0] >= w[1] - 1e-10,
                    "{method:?} non-monotone: {} < {}",
                    w[0],
                    w[1]
                );
            }
        }
    }

    #[test]
    fn linear_cdf_exceeds_linear_pct_reported_between_knots() {
        // Jensen's inequality: 1/p is strictly convex, so linear CDF >= linear % reported.
        let (ages, cdfs) = sample_annual();
        let targets = [18.0, 30.0, 42.0, 54.0];
        let lin_cdf = interp_linear_cdf(&ages, &cdfs, &targets).unwrap();
        let lin_pct = interp_linear_pct_reported(&ages, &cdfs, &targets).unwrap();
        for (a, b) in lin_cdf.cdfs.iter().zip(lin_pct.cdfs.iter()) {
            assert!(a > b, "expected linear CDF ({a}) > linear pct reported ({b})");
        }
    }

    #[test]
    fn swimon_produces_non_increasing_monthly_emergence() {
        let (ages, cdfs) = sample_annual();
        let targets: Vec<f64> = (1..=60).map(|m| m as f64).collect();
        let pat = interp_swimon(&ages, &cdfs, &targets).unwrap();
        let mut prev_inc = pat.pct_reported[0];
        for k in 1..pat.pct_reported.len() {
            let inc = pat.pct_reported[k] - pat.pct_reported[k - 1];
            assert!(
                inc <= prev_inc + 1e-9,
                "SWIMON sawtooth at month {}: inc={inc} > prev={prev_inc}",
                k + 1
            );
            prev_inc = inc;
        }
    }
}
