//! Clark (2003) Loglogistic / Weibull ODP MLE generalized to per-cell exact ages
//! and stub exposure weights (`clark_mle_exact_age`).
//!
//! Unlike standard square-triangle Clark MLE (where every origin shares the same
//! column development ages), [`clark_mle_exact_age`] accepts an arbitrary
//! rectangular/asymmetric triangle with per-cell exact average claim ages
//! `x_{i,j} > 0` (e.g., from [`crate::subannual::grid::AsymmetricAgeGrid`] and
//! [`crate::subannual::grid::stub_average_age`]) and optional per-origin
//! stub-adjusted exposure weights `P_i`.
//!
//! This handles asymmetric grids and interim valuations directly from the exact
//! emergence curve `ΔG_{i,j}(ω, θ) = G(x_{i,j}) - G(x_{i,j-1})` without needing
//! LDF interpolation.

use crate::linalg::{invert_spd, quad_form};
use crate::stochastic::clark_mle::GrowthCurve;
use crate::validate::{MonotonicityPolicy, ValidationError, validate_positive_finite_slice};

/// Configuration options for [`clark_mle_exact_age`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClarkExactAgeOptions {
    /// Parametric growth curve family (`Loglogistic` or `Weibull`).
    pub curve: GrowthCurve,
    /// Optional finite ultimate age truncation `max_age`. When `None`, projects
    /// to infinity (`G(inf) = 1.0`, Clark 2003 default including tail).
    pub max_age: Option<f64>,
    /// Cumulative monotonicity validation policy.
    pub monotonicity_policy: MonotonicityPolicy,
}

impl Default for ClarkExactAgeOptions {
    fn default() -> Self {
        Self {
            curve: GrowthCurve::Loglogistic,
            max_age: None,
            monotonicity_policy: MonotonicityPolicy::Strict,
        }
    }
}

/// Full output from [`clark_mle_exact_age`] (both LDF and Cape Cod variants).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClarkExactAgeResult {
    /// Growth curve family fitted.
    pub curve: GrowthCurve,
    /// Estimated shape parameter `omega`.
    pub omega: f64,
    /// Estimated scale parameter `theta` (in the same units as `cell_ages`).
    pub theta: f64,
    /// Estimated Cape Cod ELR (`Some(elr)` when `exposure_weights` provided, `None` for LDF).
    pub elr: Option<f64>,
    /// Estimated ODP scale parameter `sigma^2 = sum((c - mu)^2 / mu) / (N - p)`.
    pub sigma_sq: f64,
    /// Number of observed incremental cells `N`.
    pub n_observed: usize,
    /// Number of estimated parameters `p` (`n_origins + 2` for LDF, `3` for Cape Cod).
    pub n_params: usize,
    /// Residual degrees of freedom `N - p`.
    pub degrees_of_freedom: usize,
    /// Maximum ODP quasi-log-likelihood achieved (`sum(c ln mu - mu)`).
    pub log_likelihood: f64,
    /// Maximum absolute gradient component in `(ln omega, ln theta)` at optimum.
    pub max_grad_norm: f64,
    /// Latest observed cumulative losses by origin.
    pub latest: Vec<f64>,
    /// Exact average claim age of the latest observed cell for each origin.
    pub latest_ages: Vec<f64>,
    /// Fitted growth percentage `G(latest_ages[i])` for each origin.
    pub latest_growth_pct: Vec<f64>,
    /// Implied age-to-ultimate CDF `G_max / G(latest_ages[i])` for each origin.
    pub implied_cdfs: Vec<f64>,
    /// Estimated ultimate losses by origin.
    pub ultimates: Vec<f64>,
    /// Point-estimate future reserves by origin.
    pub reserves: Vec<f64>,
    /// ODP process standard error by origin `sqrt(sigma^2 * R_i)`.
    pub process_se: Vec<f64>,
    /// Cramér–Rao parameter estimation standard error by origin.
    pub parameter_se: Vec<f64>,
    /// Total prediction standard error by origin `sqrt(process_se^2 + parameter_se^2)`.
    pub total_se: Vec<f64>,
    /// Total portfolio reserve across origins.
    pub total_reserve: f64,
    /// Total portfolio process standard error `sqrt(sigma^2 * total_reserve)`.
    pub total_process_se: f64,
    /// Total portfolio parameter estimation standard error.
    pub total_parameter_se: f64,
    /// Total portfolio prediction standard error.
    pub total_se_aggregate: f64,
    /// Total portfolio coefficient of variation (`total_se_aggregate / total_reserve`).
    pub total_cv: f64,
    /// Parameter covariance matrix `sigma^2 * I(psi_hat)^{-1}` (`p × p` row-major).
    pub parameter_covariance: Vec<f64>,
}

struct ExactAgeData {
    n_origins: usize,
    n_devs: usize,
    observed_counts: Vec<usize>,
    inc: Vec<f64>,
    ages: Vec<f64>,
    latest: Vec<f64>,
    latest_ages: Vec<f64>,
    n_obs: usize,
    age_scale: f64,
}

fn prepare_exact_age_data(
    cum: &[f64],
    cell_ages: &[f64],
    n_origins: usize,
    n_devs: usize,
    observed_counts: &[usize],
    options: &ClarkExactAgeOptions,
) -> Result<ExactAgeData, ValidationError> {
    if n_origins == 0 || n_devs == 0 {
        return Err(ValidationError::ShapeMismatch);
    }
    if cum.len() != n_origins * n_devs
        || cell_ages.len() != n_origins * n_devs
        || observed_counts.len() != n_origins
    {
        return Err(ValidationError::ShapeMismatch);
    }

    let mut inc = vec![f64::NAN; n_origins * n_devs];
    let mut ages = vec![f64::NAN; n_origins * n_devs];
    let mut latest = Vec::with_capacity(n_origins);
    let mut latest_ages = Vec::with_capacity(n_origins);
    let mut n_obs = 0usize;
    let mut age_sum = 0.0;
    let mut max_obs_age = 0.0_f64;

    for i in 0..n_origins {
        let cnt = observed_counts[i];
        if cnt == 0 || cnt > n_devs {
            return Err(ValidationError::ShapeMismatch);
        }
        let mut prev_c = 0.0;
        let mut prev_age = 0.0;
        for j in 0..cnt {
            let idx = i * n_devs + j;
            let c = cum[idx];
            let x = cell_ages[idx];
            if !c.is_finite() || !x.is_finite() {
                return Err(ValidationError::NotFinite);
            }
            if c < 0.0 {
                return Err(ValidationError::Negative);
            }
            if x <= prev_age {
                return Err(ValidationError::InvalidParameter(
                    "cell_ages must be strictly positive and strictly increasing within each origin row",
                ));
            }
            if j > 0 && options.monotonicity_policy == MonotonicityPolicy::Strict && c < prev_c {
                return Err(ValidationError::NonMonotoneCumulative {
                    origin: i,
                    dev: j,
                    previous: prev_c,
                    value: c,
                });
            }
            inc[idx] = if j == 0 { c } else { c - prev_c };
            ages[idx] = x;
            age_sum += x;
            max_obs_age = max_obs_age.max(x);
            n_obs += 1;
            prev_c = c;
            prev_age = x;
        }
        latest.push(prev_c);
        latest_ages.push(prev_age);
    }

    if let Some(mx) = options.max_age
        && (!mx.is_finite() || mx < max_obs_age - 1e-9)
    {
        return Err(ValidationError::InvalidParameter(
            "max_age must be finite and >= maximum observed cell age",
        ));
    }

    let age_scale = (age_sum / n_obs as f64).max(1e-6);

    Ok(ExactAgeData {
        n_origins,
        n_devs,
        observed_counts: observed_counts.to_vec(),
        inc,
        ages,
        latest,
        latest_ages,
        n_obs,
        age_scale,
    })
}

/// Evaluate profile ODP quasi-log-likelihood at `(ln_omega, ln_theta)`.
fn profile_log_lik(
    data: &ExactAgeData,
    exposure_weights: Option<&[f64]>,
    curve: GrowthCurve,
    u: f64,
    v: f64,
) -> f64 {
    let omega = u.exp();
    let theta = v.exp();
    if !omega.is_finite() || !theta.is_finite() || omega <= 1e-4 || omega > 100.0 {
        return f64::NEG_INFINITY;
    }

    // Compute profile scale factor(s): either per-origin U_i (LDF) or global ELR (Cape Cod).
    let elr = if let Some(weights) = exposure_weights {
        let mut sum_lat = 0.0;
        let mut sum_wg = 0.0;
        for i in 0..data.n_origins {
            let g_last = curve.eval(data.latest_ages[i], omega, theta);
            if !(g_last > 1e-15) || !g_last.is_finite() {
                return f64::NEG_INFINITY;
            }
            sum_lat += data.latest[i];
            sum_wg += weights[i] * g_last;
        }
        if !(sum_wg > 0.0) || !(sum_lat > 0.0) {
            return f64::NEG_INFINITY;
        }
        Some(sum_lat / sum_wg)
    } else {
        None
    };

    let mut ll = 0.0;
    for i in 0..data.n_origins {
        let scale_i = if let (Some(w), Some(e)) = (exposure_weights, elr) {
            w[i] * e
        } else {
            let g_last = curve.eval(data.latest_ages[i], omega, theta);
            if !(g_last > 1e-15) || !g_last.is_finite() {
                return f64::NEG_INFINITY;
            }
            data.latest[i] / g_last
        };
        let mut prev_g = 0.0;
        for j in 0..data.observed_counts[i] {
            let idx = i * data.n_devs + j;
            let g = curve.eval(data.ages[idx], omega, theta);
            let dg = g - prev_g;
            prev_g = g;
            let mu = scale_i * dg;
            if !(mu > 1e-300) || !mu.is_finite() {
                return f64::NEG_INFINITY;
            }
            let c = data.inc[idx];
            ll += c * mu.ln() - mu;
        }
    }
    ll
}

/// Evaluate profile ODP quasi-log-likelihood and its exact analytic gradient
/// `(ll, d ll / d u, d ll / d v)` at `(u, v) = (ln_omega, ln_theta)` via the
/// envelope theorem (`partial ll / partial scale_i == 0` at the profile optimum).
fn profile_log_lik_and_grad(
    data: &ExactAgeData,
    exposure_weights: Option<&[f64]>,
    curve: GrowthCurve,
    u: f64,
    v: f64,
) -> (f64, f64, f64) {
    let omega = u.exp();
    let theta = v.exp();
    if !omega.is_finite() || !theta.is_finite() || omega <= 1e-4 || omega > 100.0 {
        return (f64::NEG_INFINITY, f64::NAN, f64::NAN);
    }

    let elr = if let Some(weights) = exposure_weights {
        let mut sum_lat = 0.0;
        let mut sum_wg = 0.0;
        for i in 0..data.n_origins {
            let g_last = curve.eval(data.latest_ages[i], omega, theta);
            if !(g_last > 1e-15) || !g_last.is_finite() {
                return (f64::NEG_INFINITY, f64::NAN, f64::NAN);
            }
            sum_lat += data.latest[i];
            sum_wg += weights[i] * g_last;
        }
        if !(sum_wg > 0.0) || !(sum_lat > 0.0) {
            return (f64::NEG_INFINITY, f64::NAN, f64::NAN);
        }
        Some(sum_lat / sum_wg)
    } else {
        None
    };

    let mut ll = 0.0;
    let mut gu = 0.0;
    let mut gv = 0.0;
    for i in 0..data.n_origins {
        let scale_i = if let (Some(w), Some(e)) = (exposure_weights, elr) {
            w[i] * e
        } else {
            let g_last = curve.eval(data.latest_ages[i], omega, theta);
            if !(g_last > 1e-15) || !g_last.is_finite() {
                return (f64::NEG_INFINITY, f64::NAN, f64::NAN);
            }
            data.latest[i] / g_last
        };
        let mut prev_g = 0.0;
        let mut prev_dg_dw = 0.0;
        let mut prev_dg_dt = 0.0;
        for j in 0..data.observed_counts[i] {
            let idx = i * data.n_devs + j;
            let (g, dg_dw, dg_dt) = curve.eval_with_grad(data.ages[idx], omega, theta);
            let dg = g - prev_g;
            let d_dg_du = (dg_dw - prev_dg_dw) * omega;
            let d_dg_dv = (dg_dt - prev_dg_dt) * theta;
            prev_g = g;
            prev_dg_dw = dg_dw;
            prev_dg_dt = dg_dt;
            let mu = scale_i * dg;
            if !(mu > 1e-300) || !mu.is_finite() {
                return (f64::NEG_INFINITY, f64::NAN, f64::NAN);
            }
            let c = data.inc[idx];
            ll += c * mu.ln() - mu;
            let coeff = c / dg - scale_i;
            gu += coeff * d_dg_du;
            gv += coeff * d_dg_dv;
        }
    }
    (ll, gu, gv)
}

/// Maximize profile ODP quasi-log-likelihood over `(ln_omega, ln_theta)` using
/// multi-start grid search followed by damped Newton–Raphson with exact analytic
/// profile gradients (envelope theorem) and gradient finite-difference Hessian.
fn optimize_exact_age_params(
    data: &ExactAgeData,
    exposure_weights: Option<&[f64]>,
    curve: GrowthCurve,
) -> Result<(f64, f64, f64, f64), ValidationError> {
    let omega_grid = [0.4_f64, 0.8, 1.3, 2.0, 3.2];
    let theta_mults = [0.25_f64, 0.5, 1.0, 2.0, 4.0];

    let mut best_u = 0.0_f64;
    let mut best_v = data.age_scale.ln();
    let mut best_ll = f64::NEG_INFINITY;

    for &om in &omega_grid {
        let u = om.ln();
        for &tm in &theta_mults {
            let v = (data.age_scale * tm).ln();
            let ll = profile_log_lik(data, exposure_weights, curve, u, v);
            if ll > best_ll {
                best_ll = ll;
                best_u = u;
                best_v = v;
            }
        }
    }

    if !best_ll.is_finite() {
        return Err(ValidationError::InvalidParameter(
            "Clark exact-age MLE failed to find a valid starting point",
        ));
    }

    let mut u = best_u;
    let mut v = best_v;
    let mut ll = best_ll;
    let mut max_grad = f64::INFINITY;
    let h_hess = 1e-5;

    for _iter in 0..80 {
        let (ll_cur, gu, gv) = profile_log_lik_and_grad(data, exposure_weights, curve, u, v);
        if !ll_cur.is_finite() || !gu.is_finite() || !gv.is_finite() {
            break;
        }
        ll = ll_cur;
        max_grad = gu.abs().max(gv.abs());
        if max_grad < 1e-9 {
            break;
        }

        // First-order central FD of the exact analytic gradient (4 evals, O(eps^(1/2)) accuracy).
        let (_, gu_pu, gv_pu) =
            profile_log_lik_and_grad(data, exposure_weights, curve, u + h_hess, v);
        let (_, gu_mu, gv_mu) =
            profile_log_lik_and_grad(data, exposure_weights, curve, u - h_hess, v);
        let (_, gu_pv, gv_pv) =
            profile_log_lik_and_grad(data, exposure_weights, curve, u, v + h_hess);
        let (_, gu_mv, gv_mv) =
            profile_log_lik_and_grad(data, exposure_weights, curve, u, v - h_hess);

        let (huu, hvv, huv) = if gu_pu.is_finite()
            && gu_mu.is_finite()
            && gv_pu.is_finite()
            && gv_mu.is_finite()
            && gu_pv.is_finite()
            && gu_mv.is_finite()
            && gv_pv.is_finite()
            && gv_mv.is_finite()
        {
            (
                (gu_pu - gu_mu) / (2.0 * h_hess),
                (gv_pv - gv_mv) / (2.0 * h_hess),
                f64::midpoint(gv_pu - gv_mu, gu_pv - gu_mv) / (2.0 * h_hess),
            )
        } else {
            (-1.0, -1.0, 0.0)
        };

        // We maximize ll, so negative Hessian N = -H should be SPD.
        let mut n00 = -huu;
        let mut n11 = -hvv;
        let n01 = -huv;
        let det = n00 * n11 - n01 * n01;

        let (du, dv) = if n00 > 1e-12 && n11 > 1e-12 && det > 1e-12 * n00 * n11 {
            ((n11 * gu - n01 * gv) / det, (-n01 * gu + n00 * gv) / det)
        } else {
            // Levenberg–Marquardt damping toward steepest ascent.
            let damp = n00.abs().max(n11.abs()).max(1.0);
            n00 += damp;
            n11 += damp;
            let d2 = n00 * n11 - n01 * n01;
            if d2 > 1e-15 {
                ((n11 * gu - n01 * gv) / d2, (-n01 * gu + n00 * gv) / d2)
            } else {
                let norm = gu.hypot(gv).max(1.0);
                (0.1 * gu / norm, 0.1 * gv / norm)
            }
        };

        // Clamp step norm and backtrack until log-likelihood increases.
        let step_norm = du.hypot(dv);
        let cap = if step_norm > 1.0 {
            1.0 / step_norm
        } else {
            1.0
        };
        let mut alpha = cap;
        let mut improved = false;
        for _ in 0..30 {
            let nu = u + alpha * du;
            let nv = v + alpha * dv;
            let nll = profile_log_lik(data, exposure_weights, curve, nu, nv);
            if nll > ll {
                u = nu;
                v = nv;
                ll = nll;
                improved = true;
                break;
            }
            alpha *= 0.5;
        }
        if !improved {
            break;
        }
    }

    Ok((u.exp(), v.exp(), ll, max_grad))
}

/// Fit Clark (2003) Loglogistic or Weibull ODP MLE on a rectangular/asymmetric
/// triangle with per-cell exact average claim ages and optional stub exposure
/// weights.
///
/// * When `exposure_weights` is `None`, fits the **Clark LDF** model
///   `mu_{i,j} = U_i · [G(x_{i,j}) - G(x_{i,j-1})]` (`p = n_origins + 2`).
/// * When `exposure_weights` is `Some(P)`, fits the **Clark Cape Cod** model
///   `mu_{i,j} = P_i · ELR · [G(x_{i,j}) - G(x_{i,j-1})]` (`p = 3`), where `P_i`
///   incorporates any pro-rated stub exposure scaling.
pub fn clark_mle_exact_age(
    cum: &[f64],
    cell_ages: &[f64],
    n_origins: usize,
    n_devs: usize,
    observed_counts: &[usize],
    exposure_weights: Option<&[f64]>,
    options: &ClarkExactAgeOptions,
) -> Result<ClarkExactAgeResult, ValidationError> {
    let data = prepare_exact_age_data(cum, cell_ages, n_origins, n_devs, observed_counts, options)?;
    if let Some(weights) = exposure_weights {
        if weights.len() != n_origins {
            return Err(ValidationError::ShapeMismatch);
        }
        validate_positive_finite_slice(weights, "exposure_weights")?;
    } else if data.latest.iter().any(|&v| v <= 0.0) {
        return Err(ValidationError::InvalidParameter(
            "Clark exact-age LDF requires strictly positive latest losses for every origin",
        ));
    }

    let n_params = if exposure_weights.is_some() {
        3
    } else {
        n_origins + 2
    };
    if data.n_obs <= n_params {
        return Err(ValidationError::InvalidParameter(
            "Clark exact-age MLE requires more observed cells than parameters for positive degrees of freedom",
        ));
    }

    let (omega, theta, log_lik, max_grad) =
        optimize_exact_age_params(&data, exposure_weights, options.curve)?;

    let (g_max, dg_max_dw, dg_max_dt) = match options.max_age {
        Some(mx) => options.curve.eval_with_grad(mx, omega, theta),
        None => (1.0, 0.0, 0.0),
    };

    // Precompute latest growth values and derivatives per origin.
    let mut latest_g = Vec::with_capacity(n_origins);
    let mut latest_dg_dw = Vec::with_capacity(n_origins);
    let mut latest_dg_dt = Vec::with_capacity(n_origins);
    for i in 0..n_origins {
        let (g, dw, dt) = options
            .curve
            .eval_with_grad(data.latest_ages[i], omega, theta);
        latest_g.push(g);
        latest_dg_dw.push(dw);
        latest_dg_dt.push(dt);
    }

    let (elr_opt, u_hat): (Option<f64>, Vec<f64>) = if let Some(weights) = exposure_weights {
        let sum_lat: f64 = data.latest.iter().sum();
        let sum_wg: f64 = (0..n_origins).map(|i| weights[i] * latest_g[i]).sum();
        let elr = sum_lat / sum_wg;
        let u = weights.iter().map(|&w| w * elr).collect();
        (Some(elr), u)
    } else {
        let u = (0..n_origins)
            .map(|i| data.latest[i] / latest_g[i])
            .collect();
        (None, u)
    };

    // Assemble ODP Pearson chi-square and unscaled Fisher information matrix J (p × p).
    let mut pearson_ss = 0.0;
    let mut info = vec![0.0; n_params * n_params];

    for i in 0..n_origins {
        let mut prev_g = 0.0;
        let mut prev_dw = 0.0;
        let mut prev_dt = 0.0;
        for j in 0..data.observed_counts[i] {
            let idx = i * n_devs + j;
            let (g, dw, dt) = options.curve.eval_with_grad(data.ages[idx], omega, theta);
            let dg = g - prev_g;
            let d_dg_dw = dw - prev_dw;
            let d_dg_dt = dt - prev_dt;
            prev_g = g;
            prev_dw = dw;
            prev_dt = dt;

            let ui = u_hat[i];
            let mu = ui * dg;
            if !(mu > 0.0) || !mu.is_finite() {
                return Err(ValidationError::InvalidParameter(
                    "Clark exact-age MLE produced non-positive fitted increment",
                ));
            }
            let diff = data.inc[idx] - mu;
            pearson_ss += (diff * diff) / mu;
            let inv_mu = 1.0 / mu;

            if let Some(weights) = exposure_weights {
                // Parameters: [ELR, omega, theta]
                let d_elr = weights[i] * dg;
                let d_w = ui * d_dg_dw;
                let d_t = ui * d_dg_dt;
                let gvec = [d_elr, d_w, d_t];
                for a in 0..3 {
                    for b in 0..3 {
                        info[a * 3 + b] += inv_mu * gvec[a] * gvec[b];
                    }
                }
            } else {
                // Parameters: [U_0..U_{n-1}, omega, theta]
                let d_u = dg;
                let d_w = ui * d_dg_dw;
                let d_t = ui * d_dg_dt;
                let idx_w = n_origins;
                let idx_t = n_origins + 1;
                info[i * n_params + i] += inv_mu * d_u * d_u;
                info[i * n_params + idx_w] += inv_mu * d_u * d_w;
                info[idx_w * n_params + i] += inv_mu * d_u * d_w;
                info[i * n_params + idx_t] += inv_mu * d_u * d_t;
                info[idx_t * n_params + i] += inv_mu * d_u * d_t;
                info[idx_w * n_params + idx_w] += inv_mu * d_w * d_w;
                info[idx_w * n_params + idx_t] += inv_mu * d_w * d_t;
                info[idx_t * n_params + idx_w] += inv_mu * d_w * d_t;
                info[idx_t * n_params + idx_t] += inv_mu * d_t * d_t;
            }
        }
    }

    let dof = data.n_obs - n_params;
    let sigma_sq = pearson_ss / dof as f64;
    let j_inv = invert_spd(&info, n_params)?;
    let cov: Vec<f64> = j_inv.iter().map(|&v| sigma_sq * v).collect();

    let mut reserves = vec![0.0; n_origins];
    let mut ultimates = vec![0.0; n_origins];
    let mut process_se = vec![0.0; n_origins];
    let mut parameter_se = vec![0.0; n_origins];
    let mut total_se = vec![0.0; n_origins];
    let mut grad_total = vec![0.0; n_params];
    let mut grad_i = vec![0.0; n_params];

    for i in 0..n_origins {
        let rem_g = (g_max - latest_g[i]).max(0.0);
        let r_i = u_hat[i] * rem_g;
        reserves[i] = r_i;
        ultimates[i] = data.latest[i] + r_i;

        grad_i.fill(0.0);
        if let Some(weights) = exposure_weights {
            grad_i[0] = weights[i] * rem_g;
            grad_i[1] = u_hat[i] * (dg_max_dw - latest_dg_dw[i]);
            grad_i[2] = u_hat[i] * (dg_max_dt - latest_dg_dt[i]);
        } else {
            grad_i[i] = rem_g;
            grad_i[n_origins] = u_hat[i] * (dg_max_dw - latest_dg_dw[i]);
            grad_i[n_origins + 1] = u_hat[i] * (dg_max_dt - latest_dg_dt[i]);
        }
        for k in 0..n_params {
            grad_total[k] += grad_i[k];
        }

        let proc_var = (sigma_sq * r_i).max(0.0);
        let param_var = quad_form(&grad_i, &cov, n_params).max(0.0);
        process_se[i] = proc_var.sqrt();
        parameter_se[i] = param_var.sqrt();
        total_se[i] = (proc_var + param_var).sqrt();
    }

    let total_reserve: f64 = reserves.iter().sum();
    let total_proc_var = (sigma_sq * total_reserve).max(0.0);
    let total_param_var = quad_form(&grad_total, &cov, n_params).max(0.0);
    let total_se_aggregate = (total_proc_var + total_param_var).sqrt();
    let total_cv = if total_reserve > 0.0 {
        total_se_aggregate / total_reserve
    } else {
        0.0
    };
    let implied_cdfs = latest_g.iter().map(|&g| g_max / g).collect();

    Ok(ClarkExactAgeResult {
        curve: options.curve,
        omega,
        theta,
        elr: elr_opt,
        sigma_sq,
        n_observed: data.n_obs,
        n_params,
        degrees_of_freedom: dof,
        log_likelihood: log_lik,
        max_grad_norm: max_grad,
        latest: data.latest,
        latest_ages: data.latest_ages,
        latest_growth_pct: latest_g,
        implied_cdfs,
        ultimates,
        reserves,
        process_se,
        parameter_se,
        total_se,
        total_reserve,
        total_process_se: total_proc_var.sqrt(),
        total_parameter_se: total_param_var.sqrt(),
        total_se_aggregate,
        total_cv,
        parameter_covariance: cov,
    })
}

/// Convenience wrapper for Clark exact-age LDF model (`exposure_weights = None`).
pub fn fit_clark_ldf_exact_age(
    cum: &[f64],
    cell_ages: &[f64],
    n_origins: usize,
    n_devs: usize,
    observed_counts: &[usize],
    options: &ClarkExactAgeOptions,
) -> Result<ClarkExactAgeResult, ValidationError> {
    clark_mle_exact_age(
        cum,
        cell_ages,
        n_origins,
        n_devs,
        observed_counts,
        None,
        options,
    )
}

/// Convenience wrapper for Clark exact-age Cape Cod model with stub/earned exposure weights.
pub fn fit_clark_cape_cod_exact_age(
    cum: &[f64],
    cell_ages: &[f64],
    n_origins: usize,
    n_devs: usize,
    observed_counts: &[usize],
    exposure_weights: &[f64],
    options: &ClarkExactAgeOptions,
) -> Result<ClarkExactAgeResult, ValidationError> {
    clark_mle_exact_age(
        cum,
        cell_ages,
        n_origins,
        n_devs,
        observed_counts,
        Some(exposure_weights),
        options,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::genins;
    use crate::stochastic::clark_mle::{ClarkOptions, fit_clark_cape_cod, fit_clark_ldf};

    #[test]
    fn exact_age_reduces_identically_to_square_clark_mle_on_genins() {
        let tri = genins();
        let n = tri.n();
        let mut cell_ages = vec![f64::NAN; n * n];
        let mut counts = Vec::with_capacity(n);
        for i in 0..n {
            counts.push(n - i);
            for j in 0..(n - i) {
                cell_ages[i * n + j] = (j as f64) + 0.5;
            }
        }

        let sq_ldf = fit_clark_ldf(&tri.view(), &ClarkOptions::loglogistic()).unwrap();
        let ex_ldf = fit_clark_ldf_exact_age(
            tri.data(),
            &cell_ages,
            n,
            n,
            &counts,
            &ClarkExactAgeOptions::default(),
        )
        .unwrap();

        assert!((sq_ldf.omega - ex_ldf.omega).abs() < 1e-4);
        assert!((sq_ldf.theta - ex_ldf.theta).abs() < 1e-4);
        assert!((sq_ldf.total_reserve - ex_ldf.total_reserve).abs() / sq_ldf.total_reserve < 1e-5);
        assert!(
            (sq_ldf.total_se_aggregate - ex_ldf.total_se_aggregate).abs()
                / sq_ldf.total_se_aggregate
                < 1e-4
        );

        let prem = vec![10_000_000.0; n];
        let sq_cc = fit_clark_cape_cod(&tri.view(), &prem, &ClarkOptions::weibull()).unwrap();
        let ex_cc = fit_clark_cape_cod_exact_age(
            tri.data(),
            &cell_ages,
            n,
            n,
            &counts,
            &prem,
            &ClarkExactAgeOptions {
                curve: GrowthCurve::Weibull,
                ..Default::default()
            },
        )
        .unwrap();
        assert!((sq_cc.omega - ex_cc.omega).abs() < 1e-4);
        assert!((sq_cc.total_reserve - ex_cc.total_reserve).abs() / sq_cc.total_reserve < 1e-5);
    }

    #[test]
    fn envelope_theorem_analytic_profile_gradient_matches_central_fd() {
        let tri = genins();
        let n = tri.n_origins();
        let mut cell_ages = vec![f64::NAN; n * n];
        let mut counts = Vec::with_capacity(n);
        for i in 0..n {
            counts.push(n - i);
            for j in 0..(n - i) {
                cell_ages[i * n + j] = (j as f64) + 0.5;
            }
        }
        let data = prepare_exact_age_data(
            tri.data(),
            &cell_ages,
            n,
            n,
            &counts,
            &ClarkExactAgeOptions::default(),
        )
        .unwrap();
        let prem = vec![10_000_000.0; n];
        let h = 1e-6;
        for curve in [GrowthCurve::Loglogistic, GrowthCurve::Weibull] {
            for weights in [None, Some(prem.as_slice())] {
                let u = 1.4_f64.ln();
                let v = 3.2_f64.ln();
                let (ll, gu, gv) = profile_log_lik_and_grad(&data, weights, curve, u, v);
                assert!(ll.is_finite());
                let fd_u = (profile_log_lik(&data, weights, curve, u + h, v)
                    - profile_log_lik(&data, weights, curve, u - h, v))
                    / (2.0 * h);
                let fd_v = (profile_log_lik(&data, weights, curve, u, v + h)
                    - profile_log_lik(&data, weights, curve, u, v - h))
                    / (2.0 * h);
                assert!(
                    (gu - fd_u).abs() < 1e-5 * (1.0 + fd_u.abs()),
                    "gu={gu} fd_u={fd_u}"
                );
                assert!(
                    (gv - fd_v).abs() < 1e-5 * (1.0 + fd_v.abs()),
                    "gv={gv} fd_v={fd_v}"
                );
            }
        }
    }
}
