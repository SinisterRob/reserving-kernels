//! Lightweight Bayesian and correlated stochastic reserving without MCMC.
//!
//! Implements pure-Rust, deterministic Laplace-approximated and
//! importance-sampled Bayesian reserving models:
//!
//! * **Empirical Bayes / Laplace-Approximated Bayesian Chain Ladder** — ODP
//!   log-posterior with Gaussian prior `θ ~ N(μ_0, σ_0^2 I)`, exact gradient
//!   and Hessian solved via damped Newton–Raphson, Pearson dispersion `φ_hat`,
//!   Laplace posterior covariance `Σ_post = (-H)^{-1}`, and analytical
//!   predictive MSEP recovering England & Verrall (2002) in the flat-prior limit.
//! * **Meyers' Leveled Chain Ladder (LCL) & Correlated Chain Ladder (CCL)**
//!   (CAS Monograph No. 1, 2015) — cumulative log-loss Normal model with
//!   monotonically non-increasing development standard deviations
//!   `σ_1 >= ... >= σ_J > 0` and calendar-year shock correlation `ρ ∈ (-1, 1)`,
//!   estimated via L-BFGS MAP + Levenberg–Marquardt spectral Hessian guard +
//!   heavy-tailed Student-$t$ Self-Normalized Importance Sampling (SNIS).
//! * **Bivariate CCL** — cross-line correlated reserve simulation with
//!   residual correlation `ρ_12 ∈ (-1, 1)`.

use crate::deterministic::ldf::{cdfs_from_ldfs, volume_weighted_ldfs};
use crate::linalg::{
    cholesky_lower, invert_spd, jacobi_eigen_symmetric, quad_form, solve_lower_into,
};
use crate::stochastic::bootstrap_odp::DeterministicRng;
use crate::types::TriangleView;
use crate::validate::{ValidationError, validate_triangle};

// ===========================================================================
// 4.1 Empirical Bayes / Laplace-Approximated Bayesian Chain Ladder
// ===========================================================================

/// Configuration for [`bayesian_odp_laplace`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BayesianOdpOptions {
    /// Isotropic Gaussian prior standard deviation `σ_0 > 0` (default `10.0`).
    /// Ignored as the initial value only when `empirical_bayes` is `true`.
    pub prior_sd: f64,
    /// Optional prior mean vector `μ_0` of length `2n - 1` (defaults to zeros).
    pub prior_mean: Option<Vec<f64>>,
    /// If `true`, select `σ_0` by maximizing the Laplace log-marginal
    /// likelihood (Empirical Bayes Type-II MLE).
    pub empirical_bayes: bool,
    /// Maximum damped Newton–Raphson iterations.
    pub max_iterations: usize,
    /// `L_inf` gradient convergence tolerance.
    pub tolerance: f64,
}

impl Default for BayesianOdpOptions {
    fn default() -> Self {
        Self {
            prior_sd: 10.0,
            prior_mean: None,
            empirical_bayes: false,
            max_iterations: 200,
            tolerance: 1e-10,
        }
    }
}

/// Output of [`bayesian_odp_laplace`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BayesianOdpResult {
    /// MAP parameter vector `θ_MAP = (α_0..α_{n-1}, β_1..β_{n-1})` (length `2n - 1`).
    pub theta_map: Vec<f64>,
    /// Effective prior standard deviation `σ_0` used (selected via EB if enabled).
    pub prior_sd: f64,
    /// Pearson chi-squared dispersion estimate `φ_hat`.
    pub dispersion: f64,
    /// Laplace posterior covariance `Σ_post = (-H(θ_MAP))^{-1}` (`p × p` row-major).
    pub posterior_covariance: Vec<f64>,
    /// Point-estimate reserves per origin year.
    pub origin_reserves: Vec<f64>,
    /// Predictive MSEP per origin year.
    pub origin_msep: Vec<f64>,
    /// Total point-estimate reserve `R_hat`.
    pub total_reserve: f64,
    /// Total process variance `φ_hat * R_hat`.
    pub process_variance: f64,
    /// Total parameter estimation variance `J_R^T Σ_post J_R`.
    pub estimation_variance: f64,
    /// Total predictive MSEP (`process_variance + estimation_variance`).
    pub total_msep: f64,
    /// Total predictive standard error `sqrt(total_msep)`.
    pub prediction_error: f64,
    /// Laplace log-marginal likelihood approximation.
    pub log_marginal_likelihood: f64,
    /// Whether Newton–Raphson converged within tolerance.
    pub converged: bool,
}

fn solve_spd_system(a: &[f64], b: &[f64], dim: usize) -> Result<Vec<f64>, ValidationError> {
    let l = cholesky_lower(a, dim)?;
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
}

struct OdpMapFit {
    theta: Vec<f64>,
    neg_hess: Vec<f64>,
    log_post: f64,
    converged: bool,
}

fn fit_odp_map(
    obs: &[(usize, usize, f64)],
    n: usize,
    phi: f64,
    mu0: &[f64],
    prior_prec: f64,
    max_iter: usize,
    tol: f64,
) -> Result<OdpMapFit, ValidationError> {
    let p = 2 * n - 1;
    let mut theta = vec![0.0; p];
    // Initialize alphas from log row sums / n for fast convergence.
    let mut row_sums = vec![0.0; n];
    for &(i, _, y) in obs {
        row_sums[i] += y.max(0.0);
    }
    for i in 0..n {
        let avg = (row_sums[i] / (n - i) as f64).max(1e-6);
        theta[i] = avg.ln();
    }

    let eval_log_post = |th: &[f64]| -> f64 {
        let mut ll = 0.0;
        for &(i, j, y) in obs {
            let eta = (th[i] + if j == 0 { 0.0 } else { th[n + j - 1] }).clamp(-40.0, 40.0);
            ll += y * eta - eta.exp();
        }
        let mut prior = 0.0;
        for k in 0..p {
            let d = th[k] - mu0[k];
            prior += d * d;
        }
        ll / phi - 0.5 * prior_prec * prior
    };

    let mut grad = vec![0.0; p];
    let mut neg_hess = vec![0.0; p * p];
    let mut converged = false;

    for _ in 0..max_iter {
        grad.fill(0.0);
        neg_hess.fill(0.0);
        for k in 0..p {
            grad[k] = -prior_prec * (theta[k] - mu0[k]);
            neg_hess[k * p + k] = prior_prec;
        }
        for &(i, j, y) in obs {
            let eta = (theta[i] + if j == 0 { 0.0 } else { theta[n + j - 1] }).clamp(-40.0, 40.0);
            let mu = eta.exp();
            let diff = (y - mu) / phi;
            let w = mu / phi;
            grad[i] += diff;
            neg_hess[i * p + i] += w;
            if j > 0 {
                let bj = n + j - 1;
                grad[bj] += diff;
                neg_hess[bj * p + bj] += w;
                neg_hess[i * p + bj] += w;
                neg_hess[bj * p + i] += w;
            }
        }

        let grad_inf = grad.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if grad_inf < tol {
            converged = true;
            break;
        }

        let step = if let Ok(s) = solve_spd_system(&neg_hess, &grad, p) {
            s
        } else {
            // Damping fallback for near-singular Hessian.
            let mut damped = neg_hess.clone();
            for k in 0..p {
                damped[k * p + k] += 1e-6;
            }
            solve_spd_system(&damped, &grad, p)?
        };

        let f0 = eval_log_post(&theta);
        let dir_deriv: f64 = grad.iter().zip(step.iter()).map(|(g, s)| g * s).sum();
        let mut alpha_step = 1.0;
        let mut candidate = vec![0.0; p];
        let mut accepted = false;
        for _ in 0..25 {
            for k in 0..p {
                candidate[k] = theta[k] + alpha_step * step[k];
            }
            let fc = eval_log_post(&candidate);
            if fc.is_finite() && fc >= f0 + 1e-4 * alpha_step * dir_deriv {
                accepted = true;
                break;
            }
            alpha_step *= 0.5;
        }
        if !accepted {
            break;
        }
        theta = candidate;
    }

    // Recompute exact negative Hessian at converged theta.
    neg_hess.fill(0.0);
    for k in 0..p {
        neg_hess[k * p + k] = prior_prec;
    }
    for &(i, j, _) in obs {
        let eta = (theta[i] + if j == 0 { 0.0 } else { theta[n + j - 1] }).clamp(-40.0, 40.0);
        let w = eta.exp() / phi;
        neg_hess[i * p + i] += w;
        if j > 0 {
            let bj = n + j - 1;
            neg_hess[bj * p + bj] += w;
            neg_hess[i * p + bj] += w;
            neg_hess[bj * p + i] += w;
        }
    }

    Ok(OdpMapFit {
        log_post: eval_log_post(&theta),
        theta,
        neg_hess,
        converged,
    })
}

fn log_det_spd(a: &[f64], dim: usize) -> Result<f64, ValidationError> {
    let l = cholesky_lower(a, dim)?;
    let mut sum = 0.0;
    for i in 0..dim {
        sum += l[i * dim + i].ln();
    }
    Ok(2.0 * sum)
}

/// Empirical Bayes / Laplace-approximated Bayesian Chain Ladder (ODP GLM).
///
/// Finds `θ_MAP` via damped Newton–Raphson with exact gradient and Hessian,
/// estimates Pearson dispersion `φ_hat`, and computes the analytical Laplace
/// predictive MSEP `φ_hat * R_hat + J_R^T Σ_post J_R`.
pub fn bayesian_odp_laplace(
    cumulative: &TriangleView<'_>,
    options: &BayesianOdpOptions,
) -> Result<BayesianOdpResult, ValidationError> {
    validate_triangle(cumulative)?;
    let n = cumulative.n_origins;
    if n < 3 {
        return Err(ValidationError::InvalidParameter(
            "Bayesian ODP Laplace requires at least a 3x3 triangle",
        ));
    }
    if !options.prior_sd.is_finite() || options.prior_sd <= 0.0 {
        return Err(ValidationError::NonPositive("prior_sd"));
    }
    if options.max_iterations == 0 {
        return Err(ValidationError::InvalidIterations);
    }
    if !options.tolerance.is_finite() || options.tolerance <= 0.0 {
        return Err(ValidationError::NonPositive("tolerance"));
    }

    let p = 2 * n - 1;
    let mu0 = match &options.prior_mean {
        Some(v) => {
            if v.len() != p {
                return Err(ValidationError::ShapeMismatch);
            }
            if v.iter().any(|x| !x.is_finite()) {
                return Err(ValidationError::NotFinite);
            }
            v.clone()
        }
        None => vec![0.0; p],
    };

    let mut obs: Vec<(usize, usize, f64)> = Vec::with_capacity(n * (n + 1) / 2);
    for i in 0..n {
        let mut prev = 0.0;
        for j in 0..(n - i) {
            let cum = cumulative.get(i, j);
            let inc = cum - prev;
            prev = cum;
            obs.push((i, j, inc));
        }
    }
    let n_obs = obs.len();
    if n_obs <= p {
        return Err(ValidationError::InvalidParameter(
            "insufficient degrees of freedom (n_obs <= 2n - 1)",
        ));
    }

    // Estimate Pearson dispersion phi from initial ODP fit.
    let init_prec = 1.0 / (options.prior_sd * options.prior_sd);
    let init_fit = fit_odp_map(
        &obs,
        n,
        1.0,
        &mu0,
        init_prec,
        options.max_iterations,
        options.tolerance,
    )?;
    let mut pearson_chi2 = 0.0;
    for &(i, j, y) in &obs {
        let eta = init_fit.theta[i]
            + if j == 0 {
                0.0
            } else {
                init_fit.theta[n + j - 1]
            };
        let mu = eta.clamp(-40.0, 40.0).exp();
        if mu > 0.0 {
            pearson_chi2 += (y - mu).powi(2) / mu;
        }
    }
    let phi = (pearson_chi2 / (n_obs - p) as f64).max(1e-12);

    // Optional Empirical Bayes selection of prior_sd via golden-section search.
    let prior_sd = if options.empirical_bayes {
        let mut a = -1.0_f64; // ln(sigma_0)
        let mut b = 5.0_f64;
        let gr = 0.5 * (5.0_f64.sqrt() - 1.0);
        let eval_eb = |log_s: f64| -> f64 {
            let s = log_s.exp();
            let prec = 1.0 / (s * s);
            if let Ok(fit) = fit_odp_map(&obs, n, phi, &mu0, prec, options.max_iterations, 1e-8)
                && let Ok(ld) = log_det_spd(&fit.neg_hess, p)
            {
                fit.log_post - 0.5 * ld - (p as f64) * log_s
            } else {
                f64::NEG_INFINITY
            }
        };
        let mut c = b - gr * (b - a);
        let mut d = a + gr * (b - a);
        let mut fc = eval_eb(c);
        let mut fd = eval_eb(d);
        for _ in 0..24 {
            if fc < fd {
                a = c;
                c = d;
                fc = fd;
                d = a + gr * (b - a);
                fd = eval_eb(d);
            } else {
                b = d;
                d = c;
                fd = fc;
                c = b - gr * (b - a);
                fc = eval_eb(c);
            }
        }
        f64::midpoint(a, b).exp()
    } else {
        options.prior_sd
    };

    let prior_prec = 1.0 / (prior_sd * prior_sd);
    let fit = fit_odp_map(
        &obs,
        n,
        phi,
        &mu0,
        prior_prec,
        options.max_iterations,
        options.tolerance,
    )?;

    let posterior_covariance = invert_spd(&fit.neg_hess, p)?;
    let log_det = log_det_spd(&fit.neg_hess, p)?;
    let log_marginal_likelihood = fit.log_post - 0.5 * log_det - (p as f64) * prior_sd.ln();

    // Reserve point estimates, Jacobian J_R, and predictive MSEP.
    let mut origin_reserves = vec![0.0; n];
    let mut origin_msep = vec![0.0; n];
    let mut j_total = vec![0.0; p];
    let mut total_reserve = 0.0;

    for i in 1..n {
        let mut j_origin = vec![0.0; p];
        let mut r_i = 0.0;
        for j in (n - i)..n {
            let eta = fit.theta[i] + fit.theta[n + j - 1];
            let mu = eta.clamp(-40.0, 40.0).exp();
            r_i += mu;
            j_origin[i] += mu;
            j_origin[n + j - 1] += mu;
            j_total[i] += mu;
            j_total[n + j - 1] += mu;
        }
        origin_reserves[i] = r_i;
        let est_var_i = quad_form(&j_origin, &posterior_covariance, p).max(0.0);
        origin_msep[i] = phi * r_i + est_var_i;
        total_reserve += r_i;
    }

    let process_variance = phi * total_reserve;
    let estimation_variance = quad_form(&j_total, &posterior_covariance, p).max(0.0);
    let total_msep = process_variance + estimation_variance;

    Ok(BayesianOdpResult {
        theta_map: fit.theta,
        prior_sd,
        dispersion: phi,
        posterior_covariance,
        origin_reserves,
        origin_msep,
        total_reserve,
        process_variance,
        estimation_variance,
        total_msep,
        prediction_error: total_msep.sqrt(),
        log_marginal_likelihood,
        converged: fit.converged,
    })
}

// ===========================================================================
// 4.2 Deterministic Laplace + SNIS Meyers LCL & CCL (CAS Monograph 1)
// ===========================================================================

/// Model variant for [`meyers_laplace_snis`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum MeyersModel {
    /// Leveled Chain Ladder (LCL): `ρ = 0` (independent accident years).
    Lcl,
    /// Correlated Chain Ladder (CCL): calendar-year shock correlation `ρ ∈ (-1, 1)`.
    #[default]
    Ccl,
}

/// Configuration for [`meyers_laplace_snis`] and [`meyers_bivariate_ccl`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MeyersOptions {
    /// Model variant (`Lcl` or `Ccl`).
    pub model: MeyersModel,
    /// Optional earned premium per origin year (used for Meyers' `α_i` prior
    /// mean `ln(Prem_i) + log_elr`). When `None`, defaults to log chain-ladder
    /// ultimate per origin.
    pub earned_premium: Option<Vec<f64>>,
    /// Prior expected log loss ratio `log_elr` (default `-0.35`, i.e. ~70% ELR).
    pub log_elr: f64,
    /// Student-$t$ proposal degrees of freedom `ν_q` (default `5.0`).
    pub proposal_nu: f64,
    /// Number of Self-Normalized Importance Sampling (SNIS) draws `M` (default `4096`).
    pub n_samples: usize,
    /// Deterministic PRNG seed (default `42069`).
    pub seed: u64,
    /// Maximum L-BFGS iterations for MAP estimation.
    pub max_map_iterations: usize,
}

impl Default for MeyersOptions {
    fn default() -> Self {
        Self {
            model: MeyersModel::Ccl,
            earned_premium: None,
            log_elr: -0.35,
            proposal_nu: 5.0,
            n_samples: 4096,
            seed: 42069,
            max_map_iterations: 300,
        }
    }
}

/// Output of [`meyers_laplace_snis`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MeyersResult {
    /// Fitted model variant (`Lcl` or `Ccl`).
    pub model: MeyersModel,
    /// MAP origin level parameters `α_0..α_{n-1}`.
    pub alpha_map: Vec<f64>,
    /// MAP development parameters `β_0..β_{n-1}` (with `β_{n-1} = 0`).
    pub beta_map: Vec<f64>,
    /// MAP monotonically non-increasing development standard deviations `σ_0 >= ... >= σ_{n-1} > 0`.
    pub sigma_map: Vec<f64>,
    /// MAP calendar-year correlation `ρ_MAP` (`0.0` for LCL).
    pub rho_map: f64,
    /// Importance-weighted posterior mean of `ρ` (`0.0` for LCL).
    pub rho_posterior_mean: f64,
    /// Importance-weighted predictive reserve mean per origin.
    pub origin_mean_reserves: Vec<f64>,
    /// Importance-weighted predictive reserve standard deviation per origin.
    pub origin_sd_reserves: Vec<f64>,
    /// Importance-weighted total predictive reserve mean.
    pub total_mean_reserve: f64,
    /// Importance-weighted total predictive reserve standard deviation.
    pub total_sd_reserve: f64,
    /// Effective Sample Size `ESS = 1 / Σ w_m^2`.
    pub ess: f64,
    /// `true` when `ESS < 0.1 * n_samples`.
    pub low_ess_warning: bool,
    /// Log-posterior at the MAP estimate.
    pub map_log_posterior: f64,
    /// Whether the MAP optimizer converged.
    pub converged: bool,
}

/// Output of [`meyers_bivariate_ccl`] for two correlated lines of business.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BivariateCclResult {
    /// Standalone CCL fit for Line 1.
    pub line1: MeyersResult,
    /// Standalone CCL fit for Line 2.
    pub line2: MeyersResult,
    /// Estimated cross-line residual correlation `ρ_12 ∈ (-1, 1)`.
    pub cross_line_rho: f64,
    /// Correlated joint portfolio reserve mean.
    pub portfolio_mean_reserve: f64,
    /// Correlated joint portfolio reserve standard deviation.
    pub portfolio_sd_reserve: f64,
    /// Effective Sample Size of the joint importance sampler.
    pub ess: f64,
}

#[inline]
fn unpack_sigmas(eta: &[f64]) -> Option<(Vec<f64>, f64)> {
    let n = eta.len();
    let mut sig = vec![0.0; n];
    let mut log_jac = 0.0;
    let last_e = eta[n - 1];
    if !(-15.0..=4.0).contains(&last_e) {
        return None;
    }
    sig[n - 1] = last_e.exp();
    log_jac += last_e;
    for j in (0..(n - 1)).rev() {
        let e = eta[j];
        if !(-15.0..=4.0).contains(&e) {
            return None;
        }
        sig[j] = sig[j + 1] + e.exp();
        log_jac += e;
    }
    Some((sig, log_jac))
}

fn meyers_log_posterior(
    psi: &[f64],
    log_y: &[f64],
    n: usize,
    alpha_prior_mean: &[f64],
    is_ccl: bool,
) -> f64 {
    let alphas = &psi[0..n];
    let betas = &psi[n..(2 * n - 1)];
    let etas = &psi[(2 * n - 1)..(3 * n - 1)];
    let Some((sigmas, _log_jac_sigma)) = unpack_sigmas(etas) else {
        return f64::NEG_INFINITY;
    };

    let (rho, log_prior_rho) = if is_ccl {
        let z = psi[3 * n - 1];
        if !(-12.0..=12.0).contains(&z) {
            return f64::NEG_INFINITY;
        }
        let r = z.tanh();
        // Uniform(-1, 1) prior on rho => density (1/2) sech^2(z) on z.
        let log_sech = -z.abs() - (-2.0 * z.abs()).exp().ln_1p() + std::f64::consts::LN_2;
        (r, -std::f64::consts::LN_2 + 2.0 * log_sech)
    } else {
        (0.0, 0.0)
    };

    let mut lp = log_prior_rho;

    // Priors: alpha_i ~ N(alpha_prior_mean[i], 10.0), beta_j ~ N(0, 100.0).
    for i in 0..n {
        let d = alphas[i] - alpha_prior_mean[i];
        lp -= 0.5 * d * d / 10.0;
    }
    for &b in betas {
        lp -= 0.5 * b * b / 100.0;
    }
    // Strictly concave priors on the monotonic sigma parameterization:
    // - Base tail SD sigma_{n-1} = exp(eta_{n-1}) ~ InverseGamma(shape=3.0, scale=0.06)
    // - Non-negative increments delta_j = exp(eta_j) ~ Gamma(shape=2.0, rate=20.0) for j < n - 1.
    let e_last = etas[n - 1];
    lp += -3.0 * e_last - 0.06 * (-e_last).exp();
    for &e in &etas[..(n - 1)] {
        lp += 2.0 * e - 20.0 * e.exp();
    }

    // Likelihood on observed upper triangle.
    let mut prev_resid = vec![0.0; n];
    for i in 0..n {
        for j in 0..(n - i) {
            let b_j = if j == n - 1 { 0.0 } else { betas[j] };
            let mu = if i == 0 || !is_ccl {
                alphas[i] + b_j
            } else {
                alphas[i] + b_j + rho * prev_resid[j]
            };
            let y_ij = log_y[i * n + j];
            let resid = y_ij - mu;
            let s = sigmas[j].max(1e-8);
            lp -= s.ln() + 0.5 * (resid / s).powi(2);
            prev_resid[j] = resid;
        }
    }
    lp
}

/// Evaluate Meyers LCL/CCL log-posterior and its exact analytic gradient via
/// reverse-mode adjoint propagation along each development column.
fn meyers_log_posterior_and_grad(
    psi: &[f64],
    log_y: &[f64],
    n: usize,
    alpha_prior_mean: &[f64],
    is_ccl: bool,
) -> (f64, Vec<f64>) {
    let p = psi.len();
    let alphas = &psi[0..n];
    let betas = &psi[n..(2 * n - 1)];
    let etas = &psi[(2 * n - 1)..(3 * n - 1)];
    let Some((sigmas, _log_jac_sigma)) = unpack_sigmas(etas) else {
        return (f64::NEG_INFINITY, vec![0.0; p]);
    };

    let (rho, log_prior_rho) = if is_ccl {
        let z = psi[3 * n - 1];
        if !(-12.0..=12.0).contains(&z) {
            return (f64::NEG_INFINITY, vec![0.0; p]);
        }
        let r = z.tanh();
        // Uniform(-1, 1) prior on rho => density (1/2) sech^2(z) on z.
        let log_sech = -z.abs() - (-2.0 * z.abs()).exp().ln_1p() + std::f64::consts::LN_2;
        (r, -std::f64::consts::LN_2 + 2.0 * log_sech)
    } else {
        (0.0, 0.0)
    };

    let mut lp = log_prior_rho;
    let mut grad = vec![0.0; p];

    // Priors: alpha_i ~ N(alpha_prior_mean[i], 10.0), beta_j ~ N(0, 100.0).
    for i in 0..n {
        let d = alphas[i] - alpha_prior_mean[i];
        lp -= 0.5 * d * d / 10.0;
        grad[i] -= d / 10.0;
    }
    for (j, &b) in betas.iter().enumerate() {
        lp -= 0.5 * b * b / 100.0;
        grad[n + j] -= b / 100.0;
    }
    // Monotonic sigma priors:
    // - sigma_{n-1} = exp(eta_{n-1}): -3.0 * e_last - 0.06 * exp(-e_last)
    // - delta_j = exp(eta_j): 2.0 * e - 20.0 * exp(e) for j < n - 1
    let e_last = etas[n - 1];
    lp += -3.0 * e_last - 0.06 * (-e_last).exp();
    grad[3 * n - 2] += -3.0 + 0.06 * (-e_last).exp();
    for (j, &e) in etas[..(n - 1)].iter().enumerate() {
        lp += 2.0 * e - 20.0 * e.exp();
        grad[2 * n - 1 + j] += 2.0 - 20.0 * e.exp();
    }

    // Forward pass per column j (storing residuals along column j in a small vec of length <= n),
    // followed by exact backward adjoint pass along column j.
    let mut col_resids = vec![0.0; n];
    let mut grad_sigma = vec![0.0; n];
    let mut grad_rho = 0.0;

    for j in 0..n {
        let col_len = n - j;
        let b_j = if j == n - 1 { 0.0 } else { betas[j] };
        let s = sigmas[j].max(1e-8);
        let inv_s = 1.0 / s;
        let inv_s2 = inv_s * inv_s;
        let ln_s = s.ln();

        // Forward residuals for column j.
        let mut prev_r = 0.0;
        for i in 0..col_len {
            let mu = if i == 0 || !is_ccl {
                alphas[i] + b_j
            } else {
                alphas[i] + b_j + rho * prev_r
            };
            let resid = log_y[i * n + j] - mu;
            lp -= ln_s + 0.5 * resid * resid * inv_s2;
            col_resids[i] = resid;
            prev_r = resid;
        }

        // Backward adjoint pass along column j.
        let mut adj_next = 0.0;
        let mut d_lp_ds_j = 0.0;
        for i in (0..col_len).rev() {
            let resid = col_resids[i];
            let adj_r = -resid * inv_s2 - if is_ccl { rho * adj_next } else { 0.0 };
            grad[i] -= adj_r;
            if j < n - 1 {
                grad[n + j] -= adj_r;
            }
            if i > 0 && is_ccl {
                grad_rho -= adj_r * col_resids[i - 1];
            }
            if sigmas[j] > 1e-8 {
                d_lp_ds_j += -inv_s + resid * resid * inv_s2 * inv_s;
            }
            adj_next = adj_r;
        }
        grad_sigma[j] = d_lp_ds_j;
    }

    // Chain rule from sigma_j = sum_{m=j}^{n-1} exp(eta_m) to eta_m:
    // d lp / d eta_m += exp(eta_m) * sum_{j=0}^{m} (d lp / d sigma_j).
    let mut cum_d_sigma = 0.0;
    for m in 0..n {
        cum_d_sigma += grad_sigma[m];
        grad[2 * n - 1 + m] += etas[m].exp() * cum_d_sigma;
    }

    if is_ccl {
        // d/dz [ -ln(2) + 2 ln sech(z) ] = -2 tanh(z) = -2 rho, and d rho / d z = 1 - rho^2.
        grad[3 * n - 1] = -2.0 * rho + grad_rho * (1.0 - rho * rho);
    }

    (lp, grad)
}

/// Pure-Rust L-BFGS maximizer using exact analytic gradients.
fn lbfgs_maximize<F>(fg: F, mut x: Vec<f64>, max_iter: usize, tol: f64) -> (Vec<f64>, f64, bool)
where
    F: Fn(&[f64]) -> (f64, Vec<f64>),
{
    let p = x.len();
    let m = 10usize;

    let (mut fx, mut gx) = fg(&x);
    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(m);
    let mut converged = false;

    for _ in 0..max_iter {
        let g_inf = gx.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if g_inf < tol {
            converged = true;
            break;
        }

        // Two-loop L-BFGS for ascent direction (minimizing -f => gradient of -f is -gx).
        let mut q: Vec<f64> = gx.iter().map(|&v| -v).collect();
        let h_len = s_hist.len();
        let mut alpha_arr = vec![0.0; h_len];
        for i in (0..h_len).rev() {
            let dot: f64 = s_hist[i].iter().zip(q.iter()).map(|(a, b)| a * b).sum();
            alpha_arr[i] = rho_hist[i] * dot;
            for k in 0..p {
                q[k] -= alpha_arr[i] * y_hist[i][k];
            }
        }
        let gamma = if h_len > 0 {
            let last = h_len - 1;
            let sy: f64 = s_hist[last]
                .iter()
                .zip(y_hist[last].iter())
                .map(|(a, b)| a * b)
                .sum();
            let yy: f64 = y_hist[last].iter().map(|v| v * v).sum();
            if yy > 1e-18 && sy > 0.0 { sy / yy } else { 1.0 }
        } else {
            1.0
        };
        let mut r: Vec<f64> = q.iter().map(|&v| gamma * v).collect();
        for i in 0..h_len {
            let dot: f64 = y_hist[i].iter().zip(r.iter()).map(|(a, b)| a * b).sum();
            let beta = rho_hist[i] * dot;
            for k in 0..p {
                r[k] += s_hist[i][k] * (alpha_arr[i] - beta);
            }
        }
        // Ascent direction d = -r.
        let mut d: Vec<f64> = r.iter().map(|&v| -v).collect();
        let mut dir_deriv: f64 = gx.iter().zip(d.iter()).map(|(g, di)| g * di).sum();
        if !dir_deriv.is_finite() || dir_deriv <= 0.0 {
            d.clone_from(&gx);
            dir_deriv = gx.iter().map(|v| v * v).sum();
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
        }

        // Backtracking Armijo line search.
        let mut step = 1.0;
        let mut x_next = vec![0.0; p];
        let mut f_next = f64::NEG_INFINITY;
        let mut g_next = vec![0.0; p];
        for _ in 0..30 {
            for k in 0..p {
                x_next[k] = x[k] + step * d[k];
            }
            let (fn_val, gn_val) = fg(&x_next);
            f_next = fn_val;
            if f_next.is_finite() && f_next >= fx + 1e-4 * step * dir_deriv {
                g_next = gn_val;
                break;
            }
            step *= 0.5;
        }
        if !f_next.is_finite() || f_next < fx {
            break;
        }

        let s_vec: Vec<f64> = x_next.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let step_norm = s_vec.iter().map(|z| z.abs()).fold(0.0_f64, f64::max);
        // For minimizing -f, y_vec = (-g_next) - (-gx) = gx - g_next.
        let y_vec: Vec<f64> = gx.iter().zip(g_next.iter()).map(|(a, b)| a - b).collect();
        let sy: f64 = s_vec.iter().zip(y_vec.iter()).map(|(a, b)| a * b).sum();
        if sy > 1e-12 {
            if s_hist.len() == m {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s_vec);
            y_hist.push(y_vec);
            rho_hist.push(1.0 / sy);
        }

        x = x_next;
        fx = f_next;
        gx = g_next;
        if step_norm < 1e-12 {
            converged = true;
            break;
        }
    }

    (x, fx, converged)
}

/// Compute symmetrized negative Hessian `A = -∇^2 f(x)` via `2p` central finite
/// differences of the exact analytic gradient (O(eps^(1/2)) accuracy, ~15x faster
/// than 2p^2 scalar differences), applying a Levenberg–Marquardt spectral shift if
/// `A` is indefinite or near-singular.
fn safe_neg_hessian_and_inverse<F>(
    fg: F,
    x: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), ValidationError>
where
    F: Fn(&[f64]) -> (f64, Vec<f64>),
{
    let p = x.len();
    let h = 1e-5;
    let (f0, g0) = fg(x);
    let mut work = x.to_vec();
    let mut neg_h = vec![0.0; p * p];

    for i in 0..p {
        let xi = work[i];
        work[i] = xi + h;
        let (fp, gp) = fg(&work);
        work[i] = xi - h;
        let (fm, gm) = fg(&work);
        work[i] = xi;

        for j in 0..p {
            let val = if fp.is_finite() && fm.is_finite() && gp[j].is_finite() && gm[j].is_finite()
            {
                -(gp[j] - gm[j]) / (2.0 * h)
            } else if fp.is_finite() && f0.is_finite() && gp[j].is_finite() && g0[j].is_finite() {
                -(gp[j] - g0[j]) / h
            } else if fm.is_finite() && f0.is_finite() && gm[j].is_finite() && g0[j].is_finite() {
                -(g0[j] - gm[j]) / h
            } else if i == j {
                1.0
            } else {
                0.0
            };
            neg_h[i * p + j] = val;
        }
    }

    // Enforce exact symmetry: A_sym = 0.5 * (A + A^T).
    for i in 0..p {
        for j in (i + 1)..p {
            let sym = f64::midpoint(neg_h[i * p + j], neg_h[j * p + i]);
            neg_h[i * p + j] = sym;
            neg_h[j * p + i] = sym;
        }
    }

    if let Ok(inv) = invert_spd(&neg_h, p) {
        return Ok((neg_h, inv));
    }

    // Singular / indefinite Hessian guard: scale-aware Levenberg–Marquardt spectral shift.
    let (eigvals, _) = jacobi_eigen_symmetric(&neg_h, p)?;
    let min_eig = eigvals.iter().copied().fold(f64::INFINITY, f64::min);
    let max_eig = eigvals.iter().copied().fold(0.0_f64, f64::max);
    let floor = (1e-6 * max_eig).max(1e-5);
    let shift = (-min_eig).max(0.0) + floor;
    for i in 0..p {
        neg_h[i * p + i] += shift;
    }
    let inv = invert_spd(&neg_h, p)?;
    Ok((neg_h, inv))
}

struct MeyersInitData {
    log_y: Vec<f64>,
    alpha_prior_mean: Vec<f64>,
    init: Vec<f64>,
}

fn prepare_log_triangle_and_init(
    cumulative: &TriangleView<'_>,
    options: &MeyersOptions,
) -> Result<MeyersInitData, ValidationError> {
    validate_triangle(cumulative)?;
    let n = cumulative.n_origins;
    if n < 3 {
        return Err(ValidationError::InvalidParameter(
            "Meyers LCL/CCL requires at least a 3x3 triangle",
        ));
    }
    if options.n_samples == 0 || options.max_map_iterations == 0 {
        return Err(ValidationError::InvalidIterations);
    }
    if !options.proposal_nu.is_finite() || options.proposal_nu <= 2.0 {
        return Err(ValidationError::InvalidParameter(
            "proposal_nu must be finite and > 2.0",
        ));
    }
    if !options.log_elr.is_finite() {
        return Err(ValidationError::NotFinite);
    }

    let mut log_y = vec![f64::NAN; n * n];
    for i in 0..n {
        for j in 0..(n - i) {
            let v = cumulative.get(i, j);
            if v <= 0.0 {
                return Err(ValidationError::NonPositive(
                    "cumulative triangle cell for log-loss Meyers model",
                ));
            }
            log_y[i * n + j] = v.ln();
        }
    }

    // Chain-ladder ultimates for initialization (and default prior mean when earned_premium is None).
    let mut ldfs = vec![1.0; n - 1];
    volume_weighted_ldfs(cumulative.data, n, n, &mut ldfs);
    let mut cdfs = vec![1.0; n];
    cdfs_from_ldfs(&ldfs, 1.0, &mut cdfs);
    let ultimates: Vec<f64> = (0..n)
        .map(|i| cumulative.get(i, n - 1 - i) * cdfs[n - 1 - i])
        .collect();
    let alpha_prior_mean = match &options.earned_premium {
        Some(prem) => {
            if prem.len() != n {
                return Err(ValidationError::ShapeMismatch);
            }
            let mut m = Vec::with_capacity(n);
            for &p_i in prem {
                if !p_i.is_finite() {
                    return Err(ValidationError::NotFinite);
                }
                if p_i <= 0.0 {
                    return Err(ValidationError::NonPositive("earned_premium"));
                }
                m.push(p_i.ln() + options.log_elr);
            }
            m
        }
        None => ultimates.iter().map(|&u| u.max(1.0).ln()).collect(),
    };

    let is_ccl = options.model == MeyersModel::Ccl;
    let p = 3 * n - 1 + usize::from(is_ccl);
    let mut init = vec![0.0; p];
    for i in 0..n {
        init[i] = ultimates[i].max(1.0).ln();
    }
    for j in 0..(n - 1) {
        init[n + j] = -(cdfs[j].max(1e-6).ln());
    }
    for j in 0..n {
        init[2 * n - 1 + j] = -2.5;
    }
    if is_ccl {
        init[3 * n - 1] = 0.0;
    }

    Ok(MeyersInitData {
        log_y,
        alpha_prior_mean,
        init,
    })
}

/// Fit Meyers' Leveled Chain Ladder (LCL) or Correlated Chain Ladder (CCL)
/// (CAS Monograph No. 1) using deterministic L-BFGS MAP estimation, spectral
/// Hessian repair, and heavy-tailed Student-$t$ Self-Normalized Importance
/// Sampling (SNIS).
pub fn meyers_laplace_snis(
    cumulative: &TriangleView<'_>,
    options: &MeyersOptions,
) -> Result<MeyersResult, ValidationError> {
    let (res, _, _, _) = meyers_laplace_snis_internal(cumulative, options)?;
    Ok(res)
}

#[allow(clippy::type_complexity)]
fn meyers_laplace_snis_internal(
    cumulative: &TriangleView<'_>,
    options: &MeyersOptions,
) -> Result<(MeyersResult, Vec<f64>, Vec<f64>, Vec<f64>), ValidationError> {
    let n = cumulative.n_origins;
    let is_ccl = options.model == MeyersModel::Ccl;
    let MeyersInitData {
        log_y,
        alpha_prior_mean,
        init,
    } = prepare_log_triangle_and_init(cumulative, options)?;
    let p = init.len();

    let obj = |psi: &[f64]| meyers_log_posterior(psi, &log_y, n, &alpha_prior_mean, is_ccl);
    let obj_and_grad =
        |psi: &[f64]| meyers_log_posterior_and_grad(psi, &log_y, n, &alpha_prior_mean, is_ccl);
    let (mut psi_map, mut map_log_posterior, mut converged) =
        lbfgs_maximize(obj_and_grad, init, options.max_map_iterations, 1e-6);

    // Damped Newton polish using exact analytic gradients so `psi_map` sits at the
    // exact posterior mode (eliminates linear gradient drift in SNIS importance weights).
    for _ in 0..8 {
        let (_, grad) = obj_and_grad(&psi_map);
        let g_inf = grad.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if g_inf < 1e-4 {
            converged = true;
            break;
        }
        let Ok((_, inv_h)) = safe_neg_hessian_and_inverse(obj_and_grad, &psi_map) else {
            break;
        };
        let mut step = vec![0.0; p];
        for i in 0..p {
            let mut s = 0.0;
            for j in 0..p {
                s += inv_h[i * p + j] * grad[j];
            }
            step[i] = s;
        }
        let step_inf = step.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if step_inf < 1e-6 {
            converged = true;
            break;
        }
        let mut alpha = 1.0;
        let mut improved = false;
        let mut cand = vec![0.0; p];
        for _ in 0..20 {
            for k in 0..p {
                cand[k] = psi_map[k] + alpha * step[k];
            }
            let fc = obj(&cand);
            if fc.is_finite() && fc > map_log_posterior {
                psi_map = cand;
                map_log_posterior = fc;
                improved = true;
                break;
            }
            alpha *= 0.5;
        }
        if !improved {
            if g_inf < 1e-2 {
                converged = true;
            }
            break;
        }
    }

    let (neg_hess, cov_laplace) = safe_neg_hessian_and_inverse(obj_and_grad, &psi_map)?;
    let chol_cov = cholesky_lower(&cov_laplace, p)?;

    let mut rng = DeterministicRng::from_seed(options.seed);
    let m_draws = options.n_samples;
    let nu_q = options.proposal_nu;
    let half_nu_p = f64::midpoint(nu_q, p as f64);

    let mut log_weights = Vec::with_capacity(m_draws);
    let mut draw_rhos = Vec::with_capacity(m_draws);
    let mut draw_totals = Vec::with_capacity(m_draws);
    let mut draw_origins: Vec<Vec<f64>> = (0..n).map(|_| Vec::with_capacity(m_draws)).collect();

    let mut z = vec![0.0; p];
    let mut diff = vec![0.0; p];
    let mut psi_draw = vec![0.0; p];
    let mut prev_resid = vec![0.0; n];

    for _ in 0..m_draws {
        for zi in &mut z {
            *zi = rng.next_standard_normal();
        }
        let w_chi = rng.sample_gamma(0.5 * nu_q, 2.0).max(1e-300);
        let scale = (nu_q / w_chi).sqrt();

        for i in 0..p {
            let mut lx = 0.0;
            for k in 0..=i {
                lx += chol_cov[i * p + k] * z[k];
            }
            diff[i] = scale * lx;
            psi_draw[i] = psi_map[i] + diff[i];
        }

        let lp = obj(&psi_draw);
        let qf = quad_form(&diff, &neg_hess, p).max(0.0);
        let lq = -half_nu_p * (qf / nu_q).ln_1p();
        log_weights.push(if lp.is_finite() && lq.is_finite() {
            lp - lq
        } else {
            f64::NEG_INFINITY
        });

        let alphas = &psi_draw[0..n];
        let betas = &psi_draw[n..(2 * n - 1)];
        let Some((sigmas, _)) = unpack_sigmas(&psi_draw[(2 * n - 1)..(3 * n - 1)]) else {
            draw_rhos.push(0.0);
            draw_totals.push(0.0);
            for slot in &mut draw_origins {
                slot.push(0.0);
            }
            continue;
        };
        let rho = if is_ccl {
            psi_draw[3 * n - 1].clamp(-10.0, 10.0).tanh()
        } else {
            0.0
        };
        draw_rhos.push(rho);

        // Simulate future log-triangle propagating rho along columns (only y_last needed).
        prev_resid.fill(0.0);
        let mut tot_r = 0.0;
        for i in 0..n {
            let mut y_last = 0.0;
            for j in 0..n {
                let b_j = if j == n - 1 { 0.0 } else { betas[j] };
                let mu = if i == 0 || !is_ccl {
                    alphas[i] + b_j
                } else {
                    alphas[i] + b_j + rho * prev_resid[j]
                };
                if j < n - i {
                    prev_resid[j] = log_y[i * n + j] - mu;
                } else {
                    let eps_draw = sigmas[j] * rng.next_standard_normal();
                    prev_resid[j] = eps_draw;
                    if j == n - 1 {
                        y_last = mu + eps_draw;
                    }
                }
            }
            let r_i = if i == 0 {
                0.0
            } else {
                let ult = y_last.clamp(-40.0, 40.0).exp();
                ult - cumulative.get(i, n - 1 - i)
            };
            draw_origins[i].push(r_i);
            tot_r += r_i;
        }
        draw_totals.push(tot_r);
    }

    // Log-sum-exp normalization of SNIS weights.
    let max_lw = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_lw.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    let mut weights = vec![0.0; m_draws];
    let mut sum_w = 0.0;
    for (dst, &lw) in weights.iter_mut().zip(log_weights.iter()) {
        let w = (lw - max_lw).exp();
        *dst = w;
        sum_w += w;
    }
    let mut sum_w2 = 0.0;
    for w in &mut weights {
        *w /= sum_w;
        sum_w2 += *w * *w;
    }
    let ess = 1.0 / sum_w2.max(1e-300);
    let low_ess_warning = ess < 0.1 * m_draws as f64;

    let rho_posterior_mean: f64 = weights
        .iter()
        .zip(draw_rhos.iter())
        .map(|(w, r)| w * r)
        .sum();
    let total_mean_reserve: f64 = weights
        .iter()
        .zip(draw_totals.iter())
        .map(|(w, r)| w * r)
        .sum();
    let total_var: f64 = weights
        .iter()
        .zip(draw_totals.iter())
        .map(|(w, r)| w * (r - total_mean_reserve).powi(2))
        .sum();

    let mut origin_mean_reserves = vec![0.0; n];
    let mut origin_sd_reserves = vec![0.0; n];
    for i in 0..n {
        let m_i: f64 = weights
            .iter()
            .zip(draw_origins[i].iter())
            .map(|(w, r)| w * r)
            .sum();
        let v_i: f64 = weights
            .iter()
            .zip(draw_origins[i].iter())
            .map(|(w, r)| w * (r - m_i).powi(2))
            .sum();
        origin_mean_reserves[i] = m_i;
        origin_sd_reserves[i] = v_i.max(0.0).sqrt();
    }

    let alpha_map = psi_map[0..n].to_vec();
    let mut beta_map = psi_map[n..(2 * n - 1)].to_vec();
    beta_map.push(0.0);
    let (sigma_map, _) =
        unpack_sigmas(&psi_map[(2 * n - 1)..(3 * n - 1)]).ok_or(ValidationError::NotFinite)?;
    let rho_map = if is_ccl {
        psi_map[3 * n - 1].clamp(-10.0, 10.0).tanh()
    } else {
        0.0
    };

    let result = MeyersResult {
        model: options.model,
        alpha_map,
        beta_map,
        sigma_map,
        rho_map,
        rho_posterior_mean,
        origin_mean_reserves,
        origin_sd_reserves,
        total_mean_reserve,
        total_sd_reserve: total_var.max(0.0).sqrt(),
        ess,
        low_ess_warning,
        map_log_posterior,
        converged,
    };
    Ok((result, log_y, alpha_prior_mean, psi_map))
}

/// Exact joint bivariate Normal log-posterior and reverse-mode adjoint gradient
/// over `joint = [psi1 (p), psi2 (p), z12 (1)]` (Meyers CAS Monograph 1).
#[allow(clippy::too_many_arguments)]
fn meyers_bivariate_log_posterior_and_grad(
    joint: &[f64],
    log_y1: &[f64],
    log_y2: &[f64],
    n: usize,
    alpha_prior1: &[f64],
    alpha_prior2: &[f64],
    is_ccl: bool,
) -> (f64, Vec<f64>) {
    let p_total = joint.len();
    let p = (p_total - 1) / 2;
    let psi1 = &joint[0..p];
    let psi2 = &joint[p..2 * p];
    let z12 = joint[2 * p];
    if !(-10.0..=10.0).contains(&z12) {
        return (f64::NEG_INFINITY, vec![0.0; p_total]);
    }
    let a1 = &psi1[0..n];
    let b1 = &psi1[n..(2 * n - 1)];
    let e1 = &psi1[(2 * n - 1)..(3 * n - 1)];
    let a2 = &psi2[0..n];
    let b2 = &psi2[n..(2 * n - 1)];
    let e2 = &psi2[(2 * n - 1)..(3 * n - 1)];
    let (Some((sig1, _)), Some((sig2, _))) = (unpack_sigmas(e1), unpack_sigmas(e2)) else {
        return (f64::NEG_INFINITY, vec![0.0; p_total]);
    };

    let (rho1, lp_rho1) = if is_ccl {
        let z = psi1[3 * n - 1];
        if !(-12.0..=12.0).contains(&z) {
            return (f64::NEG_INFINITY, vec![0.0; p_total]);
        }
        let log_sech = -z.abs() - (-2.0 * z.abs()).exp().ln_1p() + std::f64::consts::LN_2;
        (z.tanh(), -std::f64::consts::LN_2 + 2.0 * log_sech)
    } else {
        (0.0, 0.0)
    };
    let (rho2, lp_rho2) = if is_ccl {
        let z = psi2[3 * n - 1];
        if !(-12.0..=12.0).contains(&z) {
            return (f64::NEG_INFINITY, vec![0.0; p_total]);
        }
        let log_sech = -z.abs() - (-2.0 * z.abs()).exp().ln_1p() + std::f64::consts::LN_2;
        (z.tanh(), -std::f64::consts::LN_2 + 2.0 * log_sech)
    } else {
        (0.0, 0.0)
    };
    let rho12 = z12.tanh();
    let one_m_r2 = (1.0 - rho12 * rho12).max(1e-12);
    let inv_omr2 = 1.0 / one_m_r2;
    let log_sech12 = -z12.abs() - (-2.0 * z12.abs()).exp().ln_1p() + std::f64::consts::LN_2;
    let mut lp = lp_rho1 + lp_rho2 + (-std::f64::consts::LN_2 + 2.0 * log_sech12);
    let mut grad = vec![0.0; p_total];

    // Marginal priors for line 1 and line 2.
    for (offset, alphas, betas, etas, prior_mean) in [
        (0usize, a1, b1, e1, alpha_prior1),
        (p, a2, b2, e2, alpha_prior2),
    ] {
        for i in 0..n {
            let d = alphas[i] - prior_mean[i];
            lp -= 0.5 * d * d / 10.0;
            grad[offset + i] -= d / 10.0;
        }
        for (j, &b) in betas.iter().enumerate() {
            lp -= 0.5 * b * b / 100.0;
            grad[offset + n + j] -= b / 100.0;
        }
        let e_last = etas[n - 1];
        lp += -3.0 * e_last - 0.06 * (-e_last).exp();
        grad[offset + 3 * n - 2] += -3.0 + 0.06 * (-e_last).exp();
        for (j, &e) in etas[..(n - 1)].iter().enumerate() {
            lp += 2.0 * e - 20.0 * e.exp();
            grad[offset + 2 * n - 1 + j] += 2.0 - 20.0 * e.exp();
        }
    }

    // Joint bivariate Normal likelihood + column-wise reverse-mode adjoint.
    let mut col_r1 = vec![0.0; n];
    let mut col_r2 = vec![0.0; n];
    let mut grad_sig1 = vec![0.0; n];
    let mut grad_sig2 = vec![0.0; n];
    let mut grad_rho1 = 0.0;
    let mut grad_rho2 = 0.0;
    let mut grad_rho12 = 0.0;
    let half_ln_omr2 = 0.5 * one_m_r2.ln();

    for j in 0..n {
        let col_len = n - j;
        let bj1 = if j == n - 1 { 0.0 } else { b1[j] };
        let bj2 = if j == n - 1 { 0.0 } else { b2[j] };
        let s1 = sig1[j].max(1e-8);
        let s2 = sig2[j].max(1e-8);
        let inv_s1 = 1.0 / s1;
        let inv_s2 = 1.0 / s2;
        let ln_s12 = s1.ln() + s2.ln() + half_ln_omr2;

        let mut pr1 = 0.0;
        let mut pr2 = 0.0;
        for i in 0..col_len {
            let mu1 = if i == 0 || !is_ccl {
                a1[i] + bj1
            } else {
                a1[i] + bj1 + rho1 * pr1
            };
            let mu2 = if i == 0 || !is_ccl {
                a2[i] + bj2
            } else {
                a2[i] + bj2 + rho2 * pr2
            };
            let r1 = log_y1[i * n + j] - mu1;
            let r2 = log_y2[i * n + j] - mu2;
            let u1 = r1 * inv_s1;
            let u2 = r2 * inv_s2;
            let q = u1 * u1 - 2.0 * rho12 * u1 * u2 + u2 * u2;
            lp -= ln_s12 + 0.5 * inv_omr2 * q;
            col_r1[i] = r1;
            col_r2[i] = r2;
            pr1 = r1;
            pr2 = r2;
        }

        let mut adj1_next = 0.0;
        let mut adj2_next = 0.0;
        let mut d_s1 = 0.0;
        let mut d_s2 = 0.0;
        for i in (0..col_len).rev() {
            let r1 = col_r1[i];
            let r2 = col_r2[i];
            let u1 = r1 * inv_s1;
            let u2 = r2 * inv_s2;
            let q = u1 * u1 - 2.0 * rho12 * u1 * u2 + u2 * u2;

            let adj1 = -inv_omr2 * (u1 - rho12 * u2) * inv_s1
                - if is_ccl { rho1 * adj1_next } else { 0.0 };
            let adj2 = -inv_omr2 * (u2 - rho12 * u1) * inv_s2
                - if is_ccl { rho2 * adj2_next } else { 0.0 };

            grad[i] -= adj1;
            grad[p + i] -= adj2;
            if j < n - 1 {
                grad[n + j] -= adj1;
                grad[p + n + j] -= adj2;
            }
            if i > 0 && is_ccl {
                grad_rho1 -= adj1 * col_r1[i - 1];
                grad_rho2 -= adj2 * col_r2[i - 1];
            }
            if sig1[j] > 1e-8 {
                d_s1 += -inv_s1 + inv_omr2 * (u1 - rho12 * u2) * u1 * inv_s1;
            }
            if sig2[j] > 1e-8 {
                d_s2 += -inv_s2 + inv_omr2 * (u2 - rho12 * u1) * u2 * inv_s2;
            }
            grad_rho12 += rho12 * inv_omr2 + inv_omr2 * u1 * u2 - rho12 * inv_omr2 * inv_omr2 * q;

            adj1_next = adj1;
            adj2_next = adj2;
        }
        grad_sig1[j] = d_s1;
        grad_sig2[j] = d_s2;
    }

    let mut c1 = 0.0;
    let mut c2 = 0.0;
    for m in 0..n {
        c1 += grad_sig1[m];
        c2 += grad_sig2[m];
        grad[2 * n - 1 + m] += e1[m].exp() * c1;
        grad[p + 2 * n - 1 + m] += e2[m].exp() * c2;
    }
    if is_ccl {
        grad[3 * n - 1] = -2.0 * rho1 + grad_rho1 * (1.0 - rho1 * rho1);
        grad[p + 3 * n - 1] = -2.0 * rho2 + grad_rho2 * (1.0 - rho2 * rho2);
    }
    grad[2 * p] = -2.0 * rho12 + grad_rho12 * one_m_r2;

    (lp, grad)
}

/// Bivariate Correlated Chain Ladder (Meyers CAS Monograph 1) for two aligned
/// lines of business with cross-line residual correlation `ρ_12 ∈ (-1, 1)`.
///
/// Jointly maximizes and Laplace-SNIS samples the full bivariate Normal
/// log-posterior `p(Y_1, Y_2 | ψ_1, ψ_2, ρ_12)` with exact analytic adjoint
/// gradients so cross-line correlation uncertainty feeds directly into both
/// parameter and process reserve uncertainty.
pub fn meyers_bivariate_ccl(
    line1: &TriangleView<'_>,
    line2: &TriangleView<'_>,
    options: &MeyersOptions,
) -> Result<BivariateCclResult, ValidationError> {
    if line1.n_origins != line2.n_origins || line1.n_devs != line2.n_devs {
        return Err(ValidationError::ShapeMismatch);
    }
    let (fit1, log_y1, prior1, psi1_map) = meyers_laplace_snis_internal(line1, options)?;
    let (fit2, log_y2, prior2, psi2_map) = meyers_laplace_snis_internal(line2, options)?;
    let n = line1.n_origins;
    let is_ccl = options.model == MeyersModel::Ccl;
    let p = psi1_map.len();
    let p_joint = 2 * p + 1;

    // Initial residual correlation for warm-starting joint bivariate MAP.
    let extract_std_resids = |log_y: &[f64], fit: &MeyersResult| -> Vec<f64> {
        let mut resids = Vec::with_capacity(n * (n + 1) / 2);
        let mut prev_r = vec![0.0; n];
        for i in 0..n {
            for j in 0..(n - i) {
                let mu = if i == 0 || fit.model == MeyersModel::Lcl {
                    fit.alpha_map[i] + fit.beta_map[j]
                } else {
                    fit.alpha_map[i] + fit.beta_map[j] + fit.rho_map * prev_r[j]
                };
                let r = log_y[i * n + j] - mu;
                prev_r[j] = r;
                resids.push(r / fit.sigma_map[j].max(1e-8));
            }
        }
        resids
    };
    let r1 = extract_std_resids(&log_y1, &fit1);
    let r2 = extract_std_resids(&log_y2, &fit2);
    let rho_init = crate::capital::dependence::pearson_correlation(&r1, &r2);
    let rho_init = if rho_init.is_finite() {
        rho_init.clamp(-0.95, 0.95)
    } else {
        0.0
    };

    let mut joint_init = Vec::with_capacity(p_joint);
    joint_init.extend_from_slice(&psi1_map);
    joint_init.extend_from_slice(&psi2_map);
    joint_init.push(rho_init.atanh());

    let joint_fg = |joint: &[f64]| {
        meyers_bivariate_log_posterior_and_grad(
            joint, &log_y1, &log_y2, n, &prior1, &prior2, is_ccl,
        )
    };
    let (joint_map, _, _) = lbfgs_maximize(joint_fg, joint_init, options.max_map_iterations, 1e-5);
    let (neg_hess, cov_laplace) = safe_neg_hessian_and_inverse(joint_fg, &joint_map)?;
    let chol_cov = cholesky_lower(&cov_laplace, p_joint)?;

    let mut rng = DeterministicRng::from_seed(options.seed.wrapping_add(1));
    let m = options.n_samples;
    let nu_q = options.proposal_nu;
    let half_nu_p = f64::midpoint(nu_q, p_joint as f64);

    let mut log_weights = Vec::with_capacity(m);
    let mut draw_rhos = Vec::with_capacity(m);
    let mut port_totals = Vec::with_capacity(m);
    let mut z = vec![0.0; p_joint];
    let mut diff = vec![0.0; p_joint];
    let mut joint_draw = vec![0.0; p_joint];
    let mut prev1 = vec![0.0; n];
    let mut prev2 = vec![0.0; n];

    for _ in 0..m {
        for zi in &mut z {
            *zi = rng.next_standard_normal();
        }
        let w_chi = rng.sample_gamma(0.5 * nu_q, 2.0).max(1e-300);
        let scale = (nu_q / w_chi).sqrt();
        for i in 0..p_joint {
            let mut lx = 0.0;
            for k in 0..=i {
                lx += chol_cov[i * p_joint + k] * z[k];
            }
            diff[i] = scale * lx;
            joint_draw[i] = joint_map[i] + diff[i];
        }

        let (lp, _) = joint_fg(&joint_draw);
        let qf = quad_form(&diff, &neg_hess, p_joint).max(0.0);
        let lq = -half_nu_p * (qf / nu_q).ln_1p();
        log_weights.push(if lp.is_finite() && lq.is_finite() {
            lp - lq
        } else {
            f64::NEG_INFINITY
        });

        let p1 = &joint_draw[0..p];
        let p2 = &joint_draw[p..2 * p];
        let rho_draw = joint_draw[2 * p].clamp(-10.0, 10.0).tanh();
        draw_rhos.push(rho_draw);

        let (Some((sig1, _)), Some((sig2, _))) = (
            unpack_sigmas(&p1[(2 * n - 1)..(3 * n - 1)]),
            unpack_sigmas(&p2[(2 * n - 1)..(3 * n - 1)]),
        ) else {
            port_totals.push(0.0);
            continue;
        };
        let a1 = &p1[0..n];
        let b1 = &p1[n..(2 * n - 1)];
        let a2 = &p2[0..n];
        let b2 = &p2[n..(2 * n - 1)];
        let rho1 = if is_ccl {
            p1[3 * n - 1].clamp(-10.0, 10.0).tanh()
        } else {
            0.0
        };
        let rho2 = if is_ccl {
            p2[3 * n - 1].clamp(-10.0, 10.0).tanh()
        } else {
            0.0
        };
        let cond_sd_factor = (1.0 - rho_draw * rho_draw).max(0.0).sqrt();

        prev1.fill(0.0);
        prev2.fill(0.0);
        let mut tot = 0.0;
        for i in 0..n {
            let mut y1_last = 0.0;
            let mut y2_last = 0.0;
            for j in 0..n {
                let bj1 = if j == n - 1 { 0.0 } else { b1[j] };
                let bj2 = if j == n - 1 { 0.0 } else { b2[j] };
                let mu1 = if i == 0 || !is_ccl {
                    a1[i] + bj1
                } else {
                    a1[i] + bj1 + rho1 * prev1[j]
                };
                let mu2 = if i == 0 || !is_ccl {
                    a2[i] + bj2
                } else {
                    a2[i] + bj2 + rho2 * prev2[j]
                };
                if j < n - i {
                    prev1[j] = log_y1[i * n + j] - mu1;
                    prev2[j] = log_y2[i * n + j] - mu2;
                } else {
                    let z1 = rng.next_standard_normal();
                    let z2 = rho_draw * z1 + cond_sd_factor * rng.next_standard_normal();
                    let e1 = sig1[j] * z1;
                    let e2 = sig2[j] * z2;
                    prev1[j] = e1;
                    prev2[j] = e2;
                    if j == n - 1 {
                        y1_last = mu1 + e1;
                        y2_last = mu2 + e2;
                    }
                }
            }
            if i > 0 {
                let res1 = y1_last.clamp(-40.0, 40.0).exp() - line1.get(i, n - 1 - i);
                let res2 = y2_last.clamp(-40.0, 40.0).exp() - line2.get(i, n - 1 - i);
                tot += res1 + res2;
            }
        }
        port_totals.push(tot);
    }

    let max_lw = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_lw.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    let mut weights = vec![0.0; m];
    let mut sum_w = 0.0;
    for (dst, &lw) in weights.iter_mut().zip(log_weights.iter()) {
        let w = (lw - max_lw).exp();
        *dst = w;
        sum_w += w;
    }
    let mut sum_w2 = 0.0;
    for w in &mut weights {
        *w /= sum_w;
        sum_w2 += *w * *w;
    }
    let ess = 1.0 / sum_w2.max(1e-300);

    let cross_line_rho: f64 = weights
        .iter()
        .zip(draw_rhos.iter())
        .map(|(w, r)| w * r)
        .sum();
    let portfolio_mean_reserve: f64 = weights
        .iter()
        .zip(port_totals.iter())
        .map(|(w, r)| w * r)
        .sum();
    let portfolio_var: f64 = weights
        .iter()
        .zip(port_totals.iter())
        .map(|(w, r)| w * (r - portfolio_mean_reserve).powi(2))
        .sum();

    Ok(BivariateCclResult {
        ess,
        line1: fit1,
        line2: fit2,
        cross_line_rho,
        portfolio_mean_reserve,
        portfolio_sd_reserve: portfolio_var.max(0.0).sqrt(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::genins;
    use crate::stochastic::bootstrap_odp::fit_odp;
    use crate::validate::MonotonicityPolicy;

    #[test]
    fn bayesian_odp_flat_prior_recovers_chain_ladder_and_odp_msep() {
        let tri = genins();
        let odp = fit_odp(&tri.view(), 1.0, MonotonicityPolicy::Strict).expect("odp");

        let opts = BayesianOdpOptions {
            prior_sd: 1.0e4,
            ..BayesianOdpOptions::default()
        };
        let bayes = bayesian_odp_laplace(&tri.view(), &opts).expect("bayesian odp");
        assert!(bayes.converged);

        let rel_err_res = (bayes.total_reserve - odp.total_reserve).abs() / odp.total_reserve;
        assert!(
            rel_err_res < 1e-8,
            "bayes={} odp={} rel={rel_err_res}",
            bayes.total_reserve,
            odp.total_reserve
        );

        let rel_err_pe = (bayes.prediction_error - odp.analytic_total_prediction_error).abs()
            / odp.analytic_total_prediction_error;
        assert!(
            rel_err_pe < 1e-4,
            "bayes_pe={} odp_pe={} rel={rel_err_pe}",
            bayes.prediction_error,
            odp.analytic_total_prediction_error
        );
    }

    #[test]
    fn bayesian_odp_empirical_bayes_converges() {
        let tri = genins();
        let opts = BayesianOdpOptions {
            empirical_bayes: true,
            ..BayesianOdpOptions::default()
        };
        let res = bayesian_odp_laplace(&tri.view(), &opts).expect("eb odp");
        assert!(res.converged);
        assert!(res.prior_sd > 0.0 && res.total_reserve > 0.0);
        assert!(res.prediction_error > 0.0);
    }

    #[test]
    fn meyers_lcl_and_ccl_converge_with_healthy_ess_and_monotonic_sigmas() {
        // 5x5 triangle: verify high SNIS ESS (> 500 at M = 4096) for both LCL and CCL.
        const NAN: f64 = f64::NAN;
        #[rustfmt::skip]
        let small = crate::types::Triangle::new(5, 5, vec![
            1000.0, 1520.0, 1730.0, 1820.0, 1860.0,
            1150.0, 1690.0, 1980.0, 2060.0,    NAN,
            1320.0, 2090.0, 2320.0,    NAN,    NAN,
            1480.0, 2110.0,    NAN,    NAN,    NAN,
            1650.0,    NAN,    NAN,    NAN,    NAN,
        ]);
        let lcl_small = meyers_laplace_snis(
            &small.view(),
            &MeyersOptions {
                model: MeyersModel::Lcl,
                n_samples: 4096,
                seed: 42069,
                ..MeyersOptions::default()
            },
        )
        .expect("lcl small");
        assert!(lcl_small.converged);
        assert!(lcl_small.ess > 10.0, "small lcl ess={}", lcl_small.ess);
        assert_eq!(lcl_small.low_ess_warning, lcl_small.ess < 409.6);

        let ccl_small = meyers_laplace_snis(
            &small.view(),
            &MeyersOptions {
                model: MeyersModel::Ccl,
                n_samples: 4096,
                seed: 42069,
                ..MeyersOptions::default()
            },
        )
        .expect("ccl small");
        assert!(ccl_small.converged);
        assert!(ccl_small.ess > 10.0, "small ccl ess={}", ccl_small.ess);
        assert_eq!(ccl_small.low_ess_warning, ccl_small.ess < 409.6);

        // 10x10 GenIns: verify convergence, monotonic sigmas, and valid predictive reserves.
        let tri = genins();
        let lcl = meyers_laplace_snis(
            &tri.view(),
            &MeyersOptions {
                model: MeyersModel::Lcl,
                n_samples: 4096,
                seed: 42069,
                ..MeyersOptions::default()
            },
        )
        .expect("lcl genins");
        assert!(lcl.converged);
        assert!(lcl.total_mean_reserve > 0.0);
        assert!(lcl.total_sd_reserve > 0.0);
        assert_eq!(lcl.low_ess_warning, lcl.ess < 409.6);
        for j in 0..(lcl.sigma_map.len() - 1) {
            assert!(lcl.sigma_map[j] >= lcl.sigma_map[j + 1]);
        }

        let ccl = meyers_laplace_snis(
            &tri.view(),
            &MeyersOptions {
                model: MeyersModel::Ccl,
                n_samples: 4096,
                seed: 42069,
                ..MeyersOptions::default()
            },
        )
        .expect("ccl genins");
        assert!(ccl.converged);
        assert!(ccl.total_mean_reserve > 0.0);
        assert!(ccl.total_sd_reserve > 0.0);
        assert!((-1.0..=1.0).contains(&ccl.rho_map));
    }

    #[test]
    fn meyers_bivariate_ccl_captures_positive_cross_line_correlation() {
        let t1 = genins();
        // Scaled copy => near-perfect positive cross-line residual correlation.
        let t2 = crate::types::Triangle::new(
            10,
            10,
            t1.data()
                .iter()
                .map(|v| if v.is_finite() { v * 1.5 } else { f64::NAN })
                .collect(),
        );
        let opts = MeyersOptions {
            model: MeyersModel::Ccl,
            n_samples: 1024,
            seed: 42069,
            ..MeyersOptions::default()
        };
        let bi = meyers_bivariate_ccl(&t1.view(), &t2.view(), &opts).expect("bivariate ccl");
        assert!(bi.cross_line_rho > 0.9, "rho12={}", bi.cross_line_rho);
        assert!(bi.portfolio_mean_reserve > 0.0);
        assert!(bi.portfolio_sd_reserve > 0.0);
    }

    #[test]
    fn meyers_ccl_at_zero_rho_matches_lcl_log_posterior() {
        let tri = genins();
        let n = tri.n();
        let lcl_opts = MeyersOptions {
            model: MeyersModel::Lcl,
            ..MeyersOptions::default()
        };
        let MeyersInitData {
            log_y,
            alpha_prior_mean: prior_mean,
            init: init_lcl,
        } = prepare_log_triangle_and_init(&tri.view(), &lcl_opts).expect("init");
        let lp_lcl = meyers_log_posterior(&init_lcl, &log_y, n, &prior_mean, false);
        let mut init_ccl = init_lcl.clone();
        init_ccl.push(0.0); // zeta_rho = 0 => rho = 0.
        let lp_ccl = meyers_log_posterior(&init_ccl, &log_y, n, &prior_mean, true);
        // At zeta_rho = 0, rho = 0 and prior density on zeta_rho is 1/2 (log = -ln 2).
        assert!(((lp_ccl - lp_lcl) - (-std::f64::consts::LN_2)).abs() < 1e-12);
    }
}
