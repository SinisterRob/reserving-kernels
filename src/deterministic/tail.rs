//! Parametric and heuristic tail-factor fitting kernels.
//!
//! Implements tail-factor extrapolation curves and tail-variance estimators
//! from primary actuarial literature:
//!
//! * **Bondy (1966)** & **Generalized Bondy** — CAS Tail Factor Working Party
//!   (2013), *The Estimation of Loss Development Tail Factors: A Summary
//!   Report*, CAS E-Forum, Fall 2013, §4.2.
//! * **Exponential Decay** — log-linear regression `ln(f_t - 1) = a + b · t`
//!   (CAS TFWP 2013, §4.3).
//! * **Inverse Power Curve** — Sherman (1984), *Extrapolating, Smoothing, and
//!   Interpolating Development Factors*, PCAS Vol. LXXI, pp. 122–155:
//!   `f_t = 1 + a · t^{-b}`.
//! * **Weibull Curve** — CAS TFWP (2013):
//!   `ln(ln(f_t / (f_t - 1))) = a + b · ln(t)`.
//! * **Mack (1999) Tail Variance** — Mack (1999), *The Standard Error of Chain
//!   Ladder Reserve Estimates: Recursive Calculation and Inclusion of a Tail
//!   Factor*, ASTIN Bulletin, Vol. 29, No. 2, pp. 361–366.

use crate::error::ReservingError;
use crate::stochastic::mack::MackTailSpec;
use crate::validate::validate_positive_finite_slice;

/// Tail curve family fitted to empirical development factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TailCurveKind {
    /// Classic Bondy method (`b = 0.5`, tail equals final observed LDF).
    Bondy,
    /// Generalized Bondy geometric decay in log-LDF space (`ln f_k = c · b^k`).
    GeneralizedBondy,
    /// Exponential decay `f_t = 1 + exp(a + b · t)` with `b < 0`.
    Exponential,
    /// Sherman (1984) inverse power curve `f_t = 1 + exp(a) · t^b` with `b < 0`.
    InversePower,
    /// Weibull curve `f_t = 1 / (1 - exp(-exp(a) · t^b))`.
    Weibull,
}

/// Result of fitting and extrapolating a parametric tail curve.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct TailFit {
    /// Curve family used.
    pub curve: TailCurveKind,
    /// Compounded multiplicative tail factor (`>= 1.0`).
    pub tail_factor: f64,
    /// Curve intercept parameter (`a`, or `ln(f_start)` for Bondy).
    pub intercept: f64,
    /// Curve slope/decay parameter (`b`).
    pub slope: f64,
    /// Coefficient of determination `R²` on the transformed regression scale.
    pub r_squared: f64,
    /// Number of observed LDF points included in the fit.
    pub n_points_used: usize,
    /// Extrapolated age-to-age factors beyond the triangle edge.
    pub extrapolated_ldfs: Vec<f64>,
}

/// Classic Bondy (1966) tail factor: repeats the final observed development
/// factor (`b = 0.5`, so `prod_{m>=1} f_last^{0.5^m} = f_last`).
pub fn bondy(ldfs: &[f64]) -> Result<TailFit, ReservingError> {
    if ldfs.is_empty() {
        return Err(ReservingError::ShapeMismatch);
    }
    validate_positive_finite_slice(ldfs, "LDF")?;
    let last = ldfs[ldfs.len() - 1];
    let tail_factor = last.max(1.0);
    // Geometric Bondy decay: f_{last+m} = f_last^{0.5^m}, whose infinite product equals f_last.
    let mut extrapolated_ldfs = Vec::with_capacity(5);
    let mut weight = 0.5;
    for _ in 0..5 {
        extrapolated_ldfs.push(tail_factor.powf(weight));
        weight *= 0.5;
    }
    Ok(TailFit {
        curve: TailCurveKind::Bondy,
        tail_factor,
        intercept: tail_factor.ln(),
        slope: 0.5,
        r_squared: 1.0,
        n_points_used: 1,
        extrapolated_ldfs,
    })
}

/// Generalized Bondy tail factor (CAS Tail Factor Working Party 2013, §4.2).
///
/// Minimizes `Σ_k (ln(f_{start+k}) - c · b^k)²` over `b ∈ (0, 1)` and `c > 0`
/// via exact profile least squares (golden-section search over `b`), yielding
/// the closed-form infinite tail product `exp(c · b^m / (1 - b))`.
pub fn generalized_bondy(
    ldfs: &[f64],
    fit_start: usize,
    extrap_periods: usize,
) -> Result<TailFit, ReservingError> {
    if ldfs.is_empty() || fit_start >= ldfs.len() {
        return Err(ReservingError::ShapeMismatch);
    }
    validate_positive_finite_slice(ldfs, "LDF")?;

    let slice = &ldfs[fit_start..];
    if slice.len() == 1 {
        return bondy(slice);
    }
    let ys: Vec<f64> = slice.iter().map(|&f| f.ln()).collect();
    if ys.iter().all(|&y| y <= 0.0) {
        return Err(ReservingError::InvalidParameter(
            "generalized Bondy requires at least one LDF > 1.0",
        ));
    }

    // For fixed b, optimal c(b) = sum(y_k b^k) / sum(b^{2k}) minimizes SSE(b).
    let eval_sse = |b: f64| -> (f64, f64) {
        let mut num = 0.0;
        let mut den = 0.0;
        let mut bk = 1.0;
        for &y in &ys {
            num += y * bk;
            den += bk * bk;
            bk *= b;
        }
        let c = if den > 0.0 { (num / den).max(0.0) } else { 0.0 };
        let mut sse = 0.0;
        let mut bk = 1.0;
        for &y in &ys {
            let err = y - c * bk;
            sse += err * err;
            bk *= b;
        }
        (sse, c)
    };

    // Coarse grid scan to bracket the global minimum, then golden-section refine.
    let grid_n = 200usize;
    let mut best_idx = 1usize;
    let mut best_sse = f64::INFINITY;
    for i in 1..grid_n {
        let b = i as f64 / grid_n as f64;
        let (sse, _) = eval_sse(b);
        if sse < best_sse {
            best_sse = sse;
            best_idx = i;
        }
    }
    let mut lo = ((best_idx - 1) as f64 / grid_n as f64).max(1e-6);
    let mut hi = ((best_idx + 1) as f64 / grid_n as f64).min(1.0 - 1e-6);
    let phi = 0.5 * (5.0_f64.sqrt() - 1.0);
    let mut x1 = hi - phi * (hi - lo);
    let mut x2 = lo + phi * (hi - lo);
    let mut f1 = eval_sse(x1).0;
    let mut f2 = eval_sse(x2).0;
    for _ in 0..80 {
        if f1 < f2 {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - phi * (hi - lo);
            f1 = eval_sse(x1).0;
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + phi * (hi - lo);
            f2 = eval_sse(x2).0;
        }
    }
    let b = f64::midpoint(lo, hi);
    if b >= 1.0 - 1e-5 {
        return Err(ReservingError::InvalidParameter(
            "generalized Bondy requires decaying LDFs (b < 1.0)",
        ));
    }
    let (sse, c) = eval_sse(b);

    let y_mean = ys.iter().sum::<f64>() / ys.len() as f64;
    let sst: f64 = ys.iter().map(|&y| (y - y_mean).powi(2)).sum();
    let r_squared = if sst > 0.0 {
        (1.0 - sse / sst).clamp(0.0, 1.0)
    } else {
        1.0
    };

    let m = ys.len();
    let mut b_pow = b.powi(m.min(i32::MAX as usize) as i32);
    let tail_log = c * b_pow / (1.0 - b);
    let tail_factor = tail_log.exp().max(1.0);
    if !tail_factor.is_finite() {
        return Err(ReservingError::NotFinite);
    }
    let mut extrapolated_ldfs = Vec::with_capacity(extrap_periods);
    for _ in 0..extrap_periods {
        let log_f = c * b_pow;
        let f_step = log_f.exp();
        if !f_step.is_finite() {
            return Err(ReservingError::NotFinite);
        }
        extrapolated_ldfs.push(f_step);
        b_pow *= b;
    }

    Ok(TailFit {
        curve: TailCurveKind::GeneralizedBondy,
        tail_factor,
        intercept: c,
        slope: b,
        r_squared,
        n_points_used: m,
        extrapolated_ldfs,
    })
}

/// Ordinary least squares on `(xs, ys)` returning `(intercept, slope, r_squared)`.
fn ols_fit(xs: &[f64], ys: &[f64]) -> Result<(f64, f64, f64), ReservingError> {
    let n = xs.len();
    if n < 2 || n != ys.len() {
        return Err(ReservingError::InvalidParameter(
            "tail curve regression requires at least two valid LDF points > 1.0",
        ));
    }
    let nf = n as f64;
    let mx = xs.iter().sum::<f64>() / nf;
    let my = ys.iter().sum::<f64>() / nf;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut syy = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let dx = x - mx;
        let dy = y - my;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    if !(sxx > 0.0) {
        return Err(ReservingError::DivideByZero("tail regression variance"));
    }
    let slope = sxy / sxx;
    let intercept = my - slope * mx;
    let sse: f64 = xs
        .iter()
        .zip(ys.iter())
        .map(|(&x, &y)| (y - (intercept + slope * x)).powi(2))
        .sum();
    let r_squared = if syy > 0.0 {
        (1.0 - sse / syy).clamp(0.0, 1.0)
    } else {
        1.0
    };
    Ok((intercept, slope, r_squared))
}

/// Log-linear tail-curve family: OLS of `y(f_t) = a + b · x(t)` over
/// 1-indexed links with `f_t > 1.00001`, then `f_t = link(a + b · x(t))`
/// beyond the triangle edge.
struct LogLinearCurve {
    kind: TailCurveKind,
    /// Regressor transform of the 1-indexed development link `t`.
    x: fn(f64) -> f64,
    /// Response transform of an observed LDF `f_t > 1`.
    y: fn(f64) -> f64,
    /// Inverse link from the linear predictor to an LDF (`None` if undefined).
    ldf_from_eta: fn(f64) -> Option<f64>,
    /// Required sign of the fitted slope for a convergent (decaying) tail.
    slope_ok: fn(f64) -> bool,
    slope_error: &'static str,
}

fn log_excess(f: f64) -> f64 {
    (f - 1.0).ln()
}

fn fit_log_linear_curve(
    ldfs: &[f64],
    fit_start: usize,
    extrap_periods: usize,
    curve: &LogLinearCurve,
) -> Result<TailFit, ReservingError> {
    if ldfs.is_empty() || fit_start >= ldfs.len() {
        return Err(ReservingError::ShapeMismatch);
    }
    validate_positive_finite_slice(ldfs, "LDF")?;

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (idx, &f) in ldfs.iter().enumerate().skip(fit_start) {
        if f > 1.000_01 {
            xs.push((curve.x)((idx + 1) as f64));
            ys.push((curve.y)(f));
        }
    }

    let (intercept, slope, r_squared) = ols_fit(&xs, &ys)?;
    if !(curve.slope_ok)(slope) {
        return Err(ReservingError::InvalidParameter(curve.slope_error));
    }

    let m = ldfs.len();
    let mut extrapolated_ldfs = Vec::with_capacity(extrap_periods);
    let mut tail_factor = 1.0;
    for step in 1..=extrap_periods {
        let t = (m + step) as f64;
        // Evaluate in log-space to avoid intermediate exp(intercept) overflow/underflow.
        let eta = intercept + slope * (curve.x)(t);
        let f_t = (curve.ldf_from_eta)(eta).ok_or(ReservingError::NotFinite)?;
        if !f_t.is_finite() {
            return Err(ReservingError::NotFinite);
        }
        extrapolated_ldfs.push(f_t);
        tail_factor *= f_t;
    }
    if !tail_factor.is_finite() {
        return Err(ReservingError::NotFinite);
    }

    Ok(TailFit {
        curve: curve.kind,
        tail_factor,
        intercept,
        slope,
        r_squared,
        n_points_used: xs.len(),
        extrapolated_ldfs,
    })
}

/// Fit an exponential decay tail curve: `ln(f_t - 1) = a + b · t` for 1-indexed
/// development links `t = 1..=ldfs.len()` (CAS TFWP 2013, §4.3).
///
/// Only links at indices `>= fit_start` with `f_t > 1.00001` enter the
/// log-linear regression. Extrapolates `extrap_periods` links beyond the
/// triangle edge and compounds them into `tail_factor`.
pub fn fit_exponential_decay(
    ldfs: &[f64],
    fit_start: usize,
    extrap_periods: usize,
) -> Result<TailFit, ReservingError> {
    fit_log_linear_curve(
        ldfs,
        fit_start,
        extrap_periods,
        &LogLinearCurve {
            kind: TailCurveKind::Exponential,
            x: |t| t,
            y: log_excess,
            ldf_from_eta: |eta| Some(1.0 + eta.exp()),
            slope_ok: |b| b < 0.0,
            slope_error: "exponential tail curve requires negative decay slope",
        },
    )
}

/// Fit a Sherman (1984) inverse power tail curve:
/// `f_t = 1 + a · t^b`, estimated via OLS of `ln(f_t - 1) = ln(a) + b · ln(t)`
/// for 1-indexed development links `t = 1..=ldfs.len()`.
pub fn fit_inverse_power(
    ldfs: &[f64],
    fit_start: usize,
    extrap_periods: usize,
) -> Result<TailFit, ReservingError> {
    fit_log_linear_curve(
        ldfs,
        fit_start,
        extrap_periods,
        &LogLinearCurve {
            kind: TailCurveKind::InversePower,
            x: f64::ln,
            y: log_excess,
            ldf_from_eta: |eta| Some(1.0 + eta.exp()),
            slope_ok: |b| b < 0.0,
            slope_error: "inverse power tail curve requires negative decay slope",
        },
    )
}

/// Fit a Weibull tail curve (CAS TFWP 2013):
/// `ln(ln(f_t / (f_t - 1))) = a + b · ln(t)`, yielding
/// `f_t = 1 / (1 - exp(-exp(a) · t^b))`.
pub fn fit_weibull(
    ldfs: &[f64],
    fit_start: usize,
    extrap_periods: usize,
) -> Result<TailFit, ReservingError> {
    fit_log_linear_curve(
        ldfs,
        fit_start,
        extrap_periods,
        &LogLinearCurve {
            kind: TailCurveKind::Weibull,
            x: f64::ln,
            y: |f| (1.0 / (f - 1.0)).ln_1p().ln(),
            ldf_from_eta: |eta| {
                let denom = -(-eta.exp()).exp_m1();
                (denom > 0.0 && denom.is_finite()).then_some(1.0 / denom)
            },
            slope_ok: |b| b > 0.0,
            slope_error: "Weibull tail curve requires positive slope",
        },
    )
}

/// Estimate Mack (1999) tail variance parameters (`tail_sigma_sq` and `tail_se`)
/// by log-linear exponential regression of observed Mack `sigma_sq` and
/// development-factor standard errors against development age (Mack 1999, §3).
///
/// When `tail_factor <= 1.0`, returns zero tail variance.
pub fn estimate_mack_tail_spec(
    sigma_sq: &[f64],
    ldf_se: &[f64],
    tail_factor: f64,
) -> Result<MackTailSpec, ReservingError> {
    if !tail_factor.is_finite() {
        return Err(ReservingError::NotFinite);
    }
    if tail_factor < 1.0 {
        return Err(ReservingError::InvalidTail { tail: tail_factor });
    }
    if sigma_sq.len() != ldf_se.len() || sigma_sq.is_empty() {
        return Err(ReservingError::ShapeMismatch);
    }
    if (tail_factor - 1.0).abs() <= 1e-12 {
        return Ok(MackTailSpec::fixed(tail_factor));
    }

    let mut xs_sig = Vec::new();
    let mut ys_sig = Vec::new();
    for (idx, &s2) in sigma_sq.iter().enumerate() {
        if s2.is_finite() && s2 > 0.0 {
            xs_sig.push((idx + 1) as f64);
            ys_sig.push(s2.ln());
        }
    }
    let next_t = (sigma_sq.len() + 1) as f64;
    let last_pos_sig = sigma_sq
        .iter()
        .rev()
        .copied()
        .find(|&v| v.is_finite() && v > 0.0)
        .unwrap_or(0.0);
    let tail_sigma_sq = if xs_sig.len() >= 2 {
        let (a, b, _) = ols_fit(&xs_sig, &ys_sig)?;
        let extrapolated = (a + b * next_t).exp();
        if b < 0.0 && extrapolated.is_finite() {
            extrapolated.min(last_pos_sig)
        } else {
            last_pos_sig
        }
    } else {
        last_pos_sig
    };

    let mut xs_se = Vec::new();
    let mut ys_se = Vec::new();
    for (idx, &se) in ldf_se.iter().enumerate() {
        if se.is_finite() && se > 0.0 {
            xs_se.push((idx + 1) as f64);
            ys_se.push(se.ln());
        }
    }
    let last_pos_se = ldf_se
        .iter()
        .rev()
        .copied()
        .find(|&v| v.is_finite() && v > 0.0)
        .unwrap_or(0.0);
    let tail_se = if xs_se.len() >= 2 {
        let (a, b, _) = ols_fit(&xs_se, &ys_se)?;
        let extrapolated = (a + b * next_t).exp();
        if b < 0.0 && extrapolated.is_finite() {
            extrapolated.min(last_pos_se)
        } else {
            last_pos_se
        }
    } else {
        last_pos_se
    };

    Ok(MackTailSpec {
        tail_factor,
        tail_sigma_sq,
        tail_se,
    })
}

/// Fit a Clark (2003) growth curve (`Loglogistic` or `Weibull`) to `triangle`
/// and return the implied tail factor and its parameter standard error as a
/// [`MackTailSpec`] (mirroring `chainladder-python`'s `TailClark`).
pub fn clark_tail(
    triangle: &crate::types::TriangleView<'_>,
    curve: crate::stochastic::clark_mle::GrowthCurve,
) -> Result<MackTailSpec, ReservingError> {
    let opts = crate::stochastic::clark_mle::ClarkOptions::new(curve);
    let fit = crate::stochastic::clark_mle::fit_clark_ldf(triangle, &opts)?;
    let n_devs = triangle.n_devs;
    let g_end = fit.growth_curve[n_devs - 1];
    if !g_end.is_finite() || g_end <= 0.0 || g_end > 1.0 {
        return Err(ReservingError::NotFinite);
    }
    let tail_factor = (1.0 / g_end).max(1.0);
    // Approximate tail factor SE from Clark's origin-0 (oldest maturity) parameter SE:
    // R_0 = C_{0, n-1} * (tail - 1) => se(tail) ≈ parameter_se[0] / C_{0, n-1}.
    let c0_last = triangle.get(0, n_devs - 1);
    let tail_se = if c0_last.is_finite() && c0_last > 0.0 {
        let v = fit.parameter_se[0] / c0_last;
        if !v.is_finite() {
            return Err(ReservingError::NotFinite);
        }
        v.max(0.0)
    } else {
        0.0
    };
    let tail_sigma_sq = if c0_last.is_finite() && c0_last > 0.0 {
        let proc_var = fit.process_se[0] * fit.process_se[0];
        let v = proc_var / c0_last;
        if !v.is_finite() {
            return Err(ReservingError::NotFinite);
        }
        v.max(0.0)
    } else {
        0.0
    };
    Ok(MackTailSpec {
        tail_factor,
        tail_sigma_sq,
        tail_se,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TAIL_SAMPLE_PAID_LDFS: [f64; 9] = [
        2.026_309_092_4,
        1.559_086_954_1,
        1.320_122_537_1,
        1.184_491_104_5,
        1.107_263_962_6,
        1.074_001_009_1,
        1.046_207_298_4,
        1.032_158_277_7,
        1.023_924_521_2,
    ];

    const RAA_LDFS: [f64; 9] = [
        2.999_359, 1.623_523, 1.270_888, 1.171_675, 1.113_385, 1.041_935, 1.033_264, 1.016_936,
        1.009_217,
    ];

    #[test]
    fn bondy_repeats_last_ldf() {
        let fit = bondy(&RAA_LDFS).expect("bondy");
        assert!((fit.tail_factor - 1.009_217).abs() < 1e-9);
    }

    #[test]
    fn generalized_bondy_matches_raa_exponent() {
        let fit = generalized_bondy(&RAA_LDFS, 0, 5).expect("gen bondy");
        assert!((fit.slope - 0.4845).abs() < 0.002, "b={}", fit.slope);
        assert!(
            (fit.tail_factor - 1.0031).abs() < 0.0005,
            "tail={}",
            fit.tail_factor
        );
    }

    #[test]
    fn exponential_and_inverse_power_match_tail_sample() {
        let exp_fit = fit_exponential_decay(&TAIL_SAMPLE_PAID_LDFS, 0, 100).expect("exp");
        let inv_fit = fit_inverse_power(&TAIL_SAMPLE_PAID_LDFS, 0, 100).expect("inv");
        let weibull_fit = fit_weibull(&TAIL_SAMPLE_PAID_LDFS, 0, 100).expect("weibull");

        // Reference values from CAS TFWP / chainladder TailCurve on tail_sample paid.
        assert!(
            (exp_fit.tail_factor - 1.0324).abs() < 0.001,
            "exp tail={}",
            exp_fit.tail_factor
        );
        assert!(
            (inv_fit.tail_factor - 1.3615).abs() < 0.005,
            "inv tail={}",
            inv_fit.tail_factor
        );
        assert!(weibull_fit.tail_factor > 1.0 && weibull_fit.tail_factor < 1.2);
        assert!(exp_fit.r_squared > 0.98);
    }

    #[test]
    fn estimate_mack_tail_spec_returns_positive_variance_for_tail_gt_one() {
        let sigma_sq = [1000.0, 500.0, 250.0, 125.0, 62.5];
        let ldf_se = [0.20, 0.12, 0.07, 0.04, 0.025];
        let spec = estimate_mack_tail_spec(&sigma_sq, &ldf_se, 1.05).expect("spec");
        assert!(spec.tail_sigma_sq > 0.0 && spec.tail_sigma_sq <= 62.5);
        assert!(spec.tail_se > 0.0 && spec.tail_se <= 0.025);

        let fixed = estimate_mack_tail_spec(&sigma_sq, &ldf_se, 1.0).expect("fixed");
        assert_eq!(fixed, MackTailSpec::fixed(1.0));
    }
}
