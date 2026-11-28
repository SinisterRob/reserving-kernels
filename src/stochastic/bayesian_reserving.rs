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
//!   estimated via L-BFGS MAP + Newton polish + Levenberg–Marquardt spectral
//!   Hessian guard, then Rao-Blackwellised adaptive Self-Normalized Importance
//!   Sampling: `(α, β)` are integrated exactly (Gaussian conditional on
//!   `σ, ρ, logelr`), and only the low-dimensional non-linear block is
//!   importance-sampled with a tempered population-Monte-Carlo Student-$t$
//!   proposal plus a defensive Laplace component. Priors default to Meyers
//!   (2015) Monograph 1 ([`MeyersPriors`]); a minimum-ESS gate
//!   ([`MeyersOptions::min_ess_fraction`]) turns weight collapse into an error.
//! * **Bivariate CCL** — cross-line correlated reserve simulation with
//!   residual correlation `ρ_12 ∈ (-1, 1)`.

use crate::deterministic::ldf::{cdfs_from_ldfs, volume_weighted_ldfs};
use crate::error::ReservingError;
use crate::linalg::{
    cholesky_lower, invert_spd, jacobi_eigen_symmetric, quad_form, solve_lower_into,
};
use crate::stochastic::bootstrap_odp::DeterministicRng;
use crate::types::TriangleView;
use crate::validate::validate_triangle;

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

fn solve_spd_system(a: &[f64], b: &[f64], dim: usize) -> Result<Vec<f64>, ReservingError> {
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
) -> Result<OdpMapFit, ReservingError> {
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

fn log_det_spd(a: &[f64], dim: usize) -> Result<f64, ReservingError> {
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
) -> Result<BayesianOdpResult, ReservingError> {
    validate_triangle(cumulative)?;
    let n = cumulative.n_origins;
    if n < 3 {
        return Err(ReservingError::InvalidParameter(
            "Bayesian ODP Laplace requires at least a 3x3 triangle",
        ));
    }
    if !options.prior_sd.is_finite() || options.prior_sd <= 0.0 {
        return Err(ReservingError::NonPositive("prior_sd"));
    }
    if options.max_iterations == 0 {
        return Err(ReservingError::InvalidIterations);
    }
    if !options.tolerance.is_finite() || options.tolerance <= 0.0 {
        return Err(ReservingError::NonPositive("tolerance"));
    }

    let p = 2 * n - 1;
    let mu0 = match &options.prior_mean {
        Some(v) => {
            if v.len() != p {
                return Err(ReservingError::ShapeMismatch);
            }
            if v.iter().any(|x| !x.is_finite()) {
                return Err(ReservingError::NotFinite);
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
        return Err(ReservingError::InvalidParameter(
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
// 4.2 Deterministic Laplace + adaptive SNIS Meyers LCL & CCL (CAS Monograph 1)
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

/// Prior family for Meyers LCL / CCL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum MeyersPriorFamily {
    /// Meyers (2015), CAS Monograph No. 1, §§ 4–5 priors:
    ///
    /// * `logelr ~ U(log_elr_lower, log_elr_upper)` (default `U(-1, 0.5)`),
    /// * `α_i ~ N(ln Prem_i + logelr, alpha_sd)` (default `alpha_sd = √10`),
    /// * `β_j ~ U(-beta_bound, beta_bound)` for `j < n − 1`, `β_{n−1} = 0`
    ///   (default `beta_bound = 5`),
    /// * `a_i ~ U(0, sigma_sq_increment_upper)` with `σ_j² = Σ_{i ≥ j} a_i`
    ///   (default upper bound `1`),
    /// * `ρ ~ U(-1, 1)` (CCL only).
    ///
    /// Uniform priors have non-smooth support boundaries, so every bounded
    /// parameter is sampled and optimised on an unconstrained logit scale
    /// (`x = lo + (hi − lo)·logistic(u)`) with the exact log-Jacobian
    /// `ln logistic(u) + ln(1 − logistic(u))` added to the log-posterior. The
    /// Laplace target on the transformed scale is therefore smooth and the
    /// importance weights remain exact for the Monograph posterior.
    #[default]
    Monograph2015,
    /// Smooth Gaussian / Gamma preset (the crate's pre-0.5 default):
    /// `α_i ~ N(ln Prem_i + log_elr, alpha_sd)` with fixed `log_elr`,
    /// `β_j ~ N(0, smooth_beta_sd)`, tail SD `σ_{n−1} ~ InvGamma(shape, scale)`
    /// and SD increments `σ_j − σ_{j+1} ~ Gamma(shape, rate)`.
    SmoothGaussian,
}

/// Configurable priors for [`meyers_laplace_snis`] and [`meyers_bivariate_ccl`].
///
/// When [`MeyersOptions::earned_premium`] is `None`, the log chain-ladder
/// ultimate is used in place of `ln Prem_i`. This double-uses the data
/// (the prior is centred on a statistic of the same triangle), which
/// understates parameter uncertainty slightly; supply premium where available.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MeyersPriors {
    /// Prior family (default [`MeyersPriorFamily::Monograph2015`]).
    pub family: MeyersPriorFamily,
    /// Prior standard deviation of `α_i` (default `√10`, both families).
    pub alpha_sd: f64,
    /// Lower bound of the uniform `logelr` prior (Monograph; default `-1.0`).
    pub log_elr_lower: f64,
    /// Upper bound of the uniform `logelr` prior (Monograph; default `0.5`).
    pub log_elr_upper: f64,
    /// Half-width of the uniform `β_j` prior (Monograph; default `5.0`).
    pub beta_bound: f64,
    /// Upper bound of the uniform `a_i` variance-increment prior (Monograph; default `1.0`).
    pub sigma_sq_increment_upper: f64,
    /// Prior SD of `β_j` (`SmoothGaussian`; default `10.0`).
    pub smooth_beta_sd: f64,
    /// Inverse-gamma shape on the tail SD `σ_{n−1}` (`SmoothGaussian`; default `3.0`).
    pub smooth_sigma_tail_shape: f64,
    /// Inverse-gamma scale on the tail SD `σ_{n−1}` (`SmoothGaussian`; default `0.06`).
    pub smooth_sigma_tail_scale: f64,
    /// Gamma shape on SD increments (`SmoothGaussian`; default `2.0`).
    pub smooth_sigma_increment_shape: f64,
    /// Gamma rate on SD increments (`SmoothGaussian`; default `20.0`).
    pub smooth_sigma_increment_rate: f64,
}

impl Default for MeyersPriors {
    fn default() -> Self {
        Self::monograph_2015()
    }
}

impl MeyersPriors {
    /// Meyers (2015) CAS Monograph No. 1 priors (see [`MeyersPriorFamily::Monograph2015`]).
    pub fn monograph_2015() -> Self {
        Self {
            family: MeyersPriorFamily::Monograph2015,
            alpha_sd: 10.0_f64.sqrt(),
            log_elr_lower: -1.0,
            log_elr_upper: 0.5,
            beta_bound: 5.0,
            sigma_sq_increment_upper: 1.0,
            smooth_beta_sd: 10.0,
            smooth_sigma_tail_shape: 3.0,
            smooth_sigma_tail_scale: 0.06,
            smooth_sigma_increment_shape: 2.0,
            smooth_sigma_increment_rate: 20.0,
        }
    }

    /// Smooth Gaussian / Gamma preset (see [`MeyersPriorFamily::SmoothGaussian`]).
    pub fn smooth_gaussian() -> Self {
        Self {
            family: MeyersPriorFamily::SmoothGaussian,
            ..Self::monograph_2015()
        }
    }

    fn validate(&self) -> Result<(), ReservingError> {
        let vals = [
            self.alpha_sd,
            self.log_elr_lower,
            self.log_elr_upper,
            self.beta_bound,
            self.sigma_sq_increment_upper,
            self.smooth_beta_sd,
            self.smooth_sigma_tail_shape,
            self.smooth_sigma_tail_scale,
            self.smooth_sigma_increment_shape,
            self.smooth_sigma_increment_rate,
        ];
        if vals.iter().any(|v| !v.is_finite()) {
            return Err(ReservingError::NotFinite);
        }
        if self.alpha_sd <= 0.0
            || self.beta_bound <= 0.0
            || self.sigma_sq_increment_upper <= 0.0
            || self.smooth_beta_sd <= 0.0
            || self.smooth_sigma_tail_shape <= 0.0
            || self.smooth_sigma_tail_scale <= 0.0
            || self.smooth_sigma_increment_shape <= 0.0
            || self.smooth_sigma_increment_rate <= 0.0
        {
            return Err(ReservingError::NonPositive("Meyers prior scale"));
        }
        if self.log_elr_lower >= self.log_elr_upper {
            return Err(ReservingError::InvalidParameter(
                "Meyers log_elr_lower must be < log_elr_upper",
            ));
        }
        Ok(())
    }
}

/// Configuration for [`meyers_laplace_snis`] and [`meyers_bivariate_ccl`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MeyersOptions {
    /// Model variant (`Lcl` or `Ccl`).
    pub model: MeyersModel,
    /// Optional earned premium per origin year (used for Meyers' `α_i` prior
    /// mean `ln(Prem_i) + logelr`). When `None`, the log chain-ladder ultimate
    /// replaces `ln(Prem_i)` (see [`MeyersPriors`] on the data double-use).
    pub earned_premium: Option<Vec<f64>>,
    /// Fixed prior log loss ratio for [`MeyersPriorFamily::SmoothGaussian`]
    /// (default `-0.35`). Ignored by the Monograph family, where `logelr` is a
    /// free parameter with a uniform prior. Without premium the smooth family
    /// centres `α_i` on the log chain-ladder ultimate and ignores this value.
    pub log_elr: f64,
    /// Prior specification (default: Meyers 2015 Monograph priors).
    pub priors: MeyersPriors,
    /// Student-$t$ proposal degrees of freedom `ν_q` (default `5.0`, must be `> 2`).
    pub proposal_nu: f64,
    /// Multiplicative inflation of the proposal scale matrix (default `1.2`).
    pub proposal_inflation: f64,
    /// Number of adaptive importance-sampling rounds that re-fit the proposal
    /// mean and covariance from tempered weighted draws (default `4`; `0`
    /// keeps the plain Laplace-$t$ proposal). Draws from every round are
    /// pooled, so `(adaptation_rounds + 1) · n_samples` draws are used.
    pub adaptation_rounds: usize,
    /// Tempered ESS fraction targeted when re-fitting the proposal (default `0.5`).
    pub tempering_ess_fraction: f64,
    /// Weight `δ` of the defensive Laplace-$t$ component in the final
    /// mixture proposal `(1 − δ) q_adapt + δ q_laplace` (Hesterberg 1995; default `0.1`).
    pub defensive_weight: f64,
    /// Number of Self-Normalized Importance Sampling (SNIS) draws `M` per round (default `4096`).
    pub n_samples: usize,
    /// Minimum acceptable `ESS / M` (default `0.025`, i.e. ESS ≥ ~102 at `M = 4096`).
    /// Below it the fit returns [`ReservingError::InvalidParameter`] unless
    /// [`Self::fallback_to_laplace`] is set.
    pub min_ess_fraction: f64,
    /// When `true` and the final ESS is below the minimum, fall back to
    /// unweighted draws from the Gaussian Laplace approximation and flag the
    /// result with `laplace_fallback = true` and `converged = false`.
    pub fallback_to_laplace: bool,
    /// Deterministic PRNG seed (default `42069`).
    pub seed: u64,
    /// Maximum L-BFGS iterations for MAP estimation.
    pub max_map_iterations: usize,
    /// Maximum damped-Newton polish iterations after L-BFGS (default `40`).
    pub max_newton_iterations: usize,
}

impl Default for MeyersOptions {
    fn default() -> Self {
        Self {
            model: MeyersModel::Ccl,
            earned_premium: None,
            log_elr: -0.35,
            priors: MeyersPriors::default(),
            proposal_nu: 5.0,
            proposal_inflation: 1.2,
            adaptation_rounds: 4,
            tempering_ess_fraction: 0.5,
            defensive_weight: 0.1,
            n_samples: 4096,
            min_ess_fraction: 0.025,
            fallback_to_laplace: false,
            seed: 42069,
            max_map_iterations: 300,
            max_newton_iterations: 40,
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
    /// MAP log expected loss ratio (`logelr`; the fixed `log_elr` option for the smooth family).
    pub log_elr_map: f64,
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
    /// Effective Sample Size `ESS = 1 / Σ w_m^2` of the final SNIS round.
    pub ess: f64,
    /// Minimum ESS required by [`MeyersOptions::min_ess_fraction`].
    pub min_ess: f64,
    /// `true` when `ESS < 0.1 * n_samples`.
    pub low_ess_warning: bool,
    /// `true` when ESS fell below `min_ess` and the unweighted Laplace
    /// fallback was used (only with [`MeyersOptions::fallback_to_laplace`]).
    pub laplace_fallback: bool,
    /// Log-posterior at the MAP estimate.
    pub map_log_posterior: f64,
    /// Whether the MAP optimizer converged (`false` whenever `laplace_fallback`).
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
    /// `true` when the joint fit used the unweighted Laplace fallback.
    pub laplace_fallback: bool,
    /// Whether the joint MAP (L-BFGS + Newton polish) converged.
    pub converged: bool,
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

/// Unconstrained logit coordinates are confined to `|u| <= LOGIT_COORD_BOUND`
/// so `logistic(u)` never saturates to `0` / `1` in `f64`.
const LOGIT_COORD_BOUND: f64 = 30.0;

/// Numerically stable logistic function.
#[inline]
fn logistic(u: f64) -> f64 {
    if u >= 0.0 {
        1.0 / (1.0 + (-u).exp())
    } else {
        let e = u.exp();
        e / (1.0 + e)
    }
}

/// `ln σ(u) + ln(1 − σ(u))`, the log-Jacobian of the logit transform up to `ln(hi − lo)`.
#[inline]
fn logit_log_jacobian(u: f64) -> f64 {
    -u.abs() - 2.0 * (-u.abs()).exp().ln_1p()
}

/// `ln[(1/2) sech²(z)]` — density of `z = atanh(ρ)` under `ρ ~ U(-1, 1)`.
#[inline]
fn log_uniform_rho_density(z: f64) -> f64 {
    let log_sech = -z.abs() - (-2.0 * z.abs()).exp().ln_1p() + std::f64::consts::LN_2;
    -std::f64::consts::LN_2 + 2.0 * log_sech
}

/// Natural-scale Meyers parameters decoded from an unconstrained vector `ψ`.
struct MeyersParams {
    alphas: Vec<f64>,
    /// Length `n`, with `betas[n − 1] = 0`.
    betas: Vec<f64>,
    sigmas: Vec<f64>,
    rho: f64,
    log_elr: f64,
    /// Log prior density (including transform Jacobians) on the `ψ` scale.
    log_prior: f64,
}

/// Natural-scale non-linear block (σ, ρ, logelr) decoded from `θ = ψ[2n−1..]`.
struct NlParams {
    sigmas: Vec<f64>,
    rho: f64,
    log_elr: f64,
    /// Log prior density of `θ` (σ / ρ / logelr priors and Jacobians).
    log_prior: f64,
}

/// Gradients of the log-likelihood with respect to the natural parameters.
struct MeyersLikGrad {
    alpha: Vec<f64>,
    /// Length `n − 1`.
    beta: Vec<f64>,
    sigma: Vec<f64>,
    rho: f64,
}

impl MeyersLikGrad {
    fn zeros(n: usize) -> Self {
        Self {
            alpha: vec![0.0; n],
            beta: vec![0.0; n - 1],
            sigma: vec![0.0; n],
            rho: 0.0,
        }
    }
}

/// Model specification: layout, priors and `α` prior centres.
///
/// Unconstrained layout `ψ = [α (n) | β-coords (n−1) | σ-coords (n) | z_ρ (CCL) | logelr-coord (Monograph)]`.
#[derive(Clone)]
struct MeyersSpec {
    n: usize,
    is_ccl: bool,
    priors: MeyersPriors,
    /// `ln Prem_i` (Monograph; `logelr` added at evaluation) or
    /// `ln Prem_i + log_elr` (smooth family).
    alpha_center: Vec<f64>,
}

impl MeyersSpec {
    fn is_monograph(&self) -> bool {
        self.priors.family == MeyersPriorFamily::Monograph2015
    }

    fn n_params(&self) -> usize {
        3 * self.n - 1 + usize::from(self.is_ccl) + usize::from(self.is_monograph())
    }

    fn log_elr_index(&self) -> usize {
        3 * self.n - 1 + usize::from(self.is_ccl)
    }

    /// Decode the non-linear block `θ = ψ[2n−1..]` (σ coordinates, `z_ρ`,
    /// logelr coordinate) with its log prior density and Jacobians.
    fn unpack_nonlinear(&self, theta: &[f64]) -> Option<NlParams> {
        let n = self.n;
        let pr = &self.priors;
        let mut log_prior = 0.0;
        let (sigmas, log_elr) = if self.is_monograph() {
            let mut sigma_sq = vec![0.0; n];
            let mut acc = 0.0;
            for i in (0..n).rev() {
                let u = theta[i];
                if u.abs() > LOGIT_COORD_BOUND {
                    return None;
                }
                acc += pr.sigma_sq_increment_upper * logistic(u);
                sigma_sq[i] = acc;
                log_prior += logit_log_jacobian(u);
            }
            let t = theta[n + usize::from(self.is_ccl)];
            if t.abs() > LOGIT_COORD_BOUND {
                return None;
            }
            let log_elr = pr.log_elr_lower + (pr.log_elr_upper - pr.log_elr_lower) * logistic(t);
            log_prior += logit_log_jacobian(t);
            (sigma_sq.iter().map(|v| v.sqrt()).collect(), log_elr)
        } else {
            let etas = &theta[0..n];
            let (sig, _) = unpack_sigmas(etas)?;
            let e_last = etas[n - 1];
            log_prior +=
                -pr.smooth_sigma_tail_shape * e_last - pr.smooth_sigma_tail_scale * (-e_last).exp();
            for &e in &etas[..(n - 1)] {
                log_prior +=
                    pr.smooth_sigma_increment_shape * e - pr.smooth_sigma_increment_rate * e.exp();
            }
            (sig, 0.0)
        };
        let rho = if self.is_ccl {
            let z = theta[n];
            if !(-12.0..=12.0).contains(&z) {
                return None;
            }
            log_prior += log_uniform_rho_density(z);
            z.tanh()
        } else {
            0.0
        };
        Some(NlParams {
            sigmas,
            rho,
            log_elr,
            log_prior,
        })
    }

    /// Assemble natural parameters from a linear-block draw `γ` and decoded θ.
    /// Returns `None` when a Monograph `β_j` falls outside `[−B, B]`.
    fn params_from_linear(&self, gamma: &[f64], nl: NlParams) -> Option<MeyersParams> {
        let n = self.n;
        let mut betas = vec![0.0; n];
        betas[..(n - 1)].copy_from_slice(&gamma[n..(2 * n - 1)]);
        if self.is_monograph() && betas.iter().any(|b| b.abs() > self.priors.beta_bound) {
            return None;
        }
        Some(MeyersParams {
            alphas: gamma[0..n].to_vec(),
            betas,
            sigmas: nl.sigmas,
            rho: nl.rho,
            log_elr: nl.log_elr,
            log_prior: 0.0,
        })
    }

    fn unpack(&self, psi: &[f64]) -> Option<MeyersParams> {
        let n = self.n;
        let pr = &self.priors;
        let nl = self.unpack_nonlinear(&psi[(2 * n - 1)..])?;
        let alphas = psi[0..n].to_vec();
        let mut betas = vec![0.0; n];
        let mut log_prior = nl.log_prior;
        if self.is_monograph() {
            let width_beta = 2.0 * pr.beta_bound;
            for j in 0..(n - 1) {
                let v = psi[n + j];
                if v.abs() > LOGIT_COORD_BOUND {
                    return None;
                }
                betas[j] = -pr.beta_bound + width_beta * logistic(v);
                log_prior += logit_log_jacobian(v);
            }
        } else {
            let inv_var_b = 1.0 / (pr.smooth_beta_sd * pr.smooth_beta_sd);
            for j in 0..(n - 1) {
                let b = psi[n + j];
                betas[j] = b;
                log_prior -= 0.5 * b * b * inv_var_b;
            }
        }
        let inv_var_a = 1.0 / (pr.alpha_sd * pr.alpha_sd);
        for i in 0..n {
            let d = alphas[i] - self.alpha_center[i] - nl.log_elr;
            log_prior -= 0.5 * d * d * inv_var_a;
        }
        Some(MeyersParams {
            alphas,
            betas,
            sigmas: nl.sigmas,
            rho: nl.rho,
            log_elr: nl.log_elr,
            log_prior,
        })
    }

    /// Chain-rule the likelihood gradient through the transforms and add the
    /// log-prior gradient, writing `∂ log p / ∂ψ` into `grad`.
    fn backprop(&self, psi: &[f64], params: &MeyersParams, lik: &MeyersLikGrad, grad: &mut [f64]) {
        let n = self.n;
        let pr = &self.priors;
        let inv_var_a = 1.0 / (pr.alpha_sd * pr.alpha_sd);
        let mut d_log_elr = 0.0;
        for i in 0..n {
            let d = params.alphas[i] - self.alpha_center[i] - params.log_elr;
            grad[i] = lik.alpha[i] - d * inv_var_a;
            d_log_elr += d * inv_var_a;
        }
        if self.is_monograph() {
            let width_beta = 2.0 * pr.beta_bound;
            for j in 0..(n - 1) {
                let s = logistic(psi[n + j]);
                grad[n + j] = lik.beta[j] * width_beta * s * (1.0 - s) + (1.0 - 2.0 * s);
            }
            // σ_j = sqrt(Σ_{i>=j} a_i)  =>  ∂/∂a_i = Σ_{j<=i} g_σj / (2 σ_j).
            let mut cum = 0.0;
            for i in 0..n {
                let sj = params.sigmas[i];
                if sj > 0.0 {
                    cum += lik.sigma[i] / (2.0 * sj);
                }
                let s = logistic(psi[2 * n - 1 + i]);
                grad[2 * n - 1 + i] =
                    cum * pr.sigma_sq_increment_upper * s * (1.0 - s) + (1.0 - 2.0 * s);
            }
            let idx = self.log_elr_index();
            let s = logistic(psi[idx]);
            grad[idx] =
                d_log_elr * (pr.log_elr_upper - pr.log_elr_lower) * s * (1.0 - s) + (1.0 - 2.0 * s);
        } else {
            let inv_var_b = 1.0 / (pr.smooth_beta_sd * pr.smooth_beta_sd);
            for j in 0..(n - 1) {
                grad[n + j] = lik.beta[j] - psi[n + j] * inv_var_b;
            }
            let etas = &psi[(2 * n - 1)..(3 * n - 1)];
            let mut cum = 0.0;
            for m in 0..n {
                cum += lik.sigma[m];
                let e = etas[m];
                let prior_d = if m == n - 1 {
                    -pr.smooth_sigma_tail_shape + pr.smooth_sigma_tail_scale * (-e).exp()
                } else {
                    pr.smooth_sigma_increment_shape - pr.smooth_sigma_increment_rate * e.exp()
                };
                grad[2 * n - 1 + m] = e.exp() * cum + prior_d;
            }
        }
        if self.is_ccl {
            let rho = params.rho;
            // d/dz [ -ln 2 + 2 ln sech z ] = -2 tanh z, and dρ/dz = 1 - ρ².
            grad[3 * n - 1] = -2.0 * rho + lik.rho * (1.0 - rho * rho);
        }
    }
}

/// Univariate LCL / CCL Normal log-likelihood on the observed upper log-triangle
/// with reverse-mode adjoint gradients along each development column.
fn meyers_log_likelihood(
    params: &MeyersParams,
    log_y: &[f64],
    n: usize,
    is_ccl: bool,
) -> (f64, MeyersLikGrad) {
    let mut lp = 0.0;
    let mut g = MeyersLikGrad::zeros(n);
    let rho = params.rho;
    let mut col_resids = vec![0.0; n];
    for j in 0..n {
        let col_len = n - j;
        let b_j = params.betas[j];
        let s = params.sigmas[j].max(1e-8);
        let inv_s = 1.0 / s;
        let inv_s2 = inv_s * inv_s;
        let ln_s = s.ln();

        let mut prev_r = 0.0;
        for i in 0..col_len {
            let mu = if i == 0 || !is_ccl {
                params.alphas[i] + b_j
            } else {
                params.alphas[i] + b_j + rho * prev_r
            };
            let resid = log_y[i * n + j] - mu;
            lp -= ln_s + 0.5 * resid * resid * inv_s2;
            col_resids[i] = resid;
            prev_r = resid;
        }

        let mut adj_next = 0.0;
        let mut d_lp_ds_j = 0.0;
        for i in (0..col_len).rev() {
            let resid = col_resids[i];
            let adj_r = -resid * inv_s2 - if is_ccl { rho * adj_next } else { 0.0 };
            g.alpha[i] -= adj_r;
            if j < n - 1 {
                g.beta[j] -= adj_r;
            }
            if i > 0 && is_ccl {
                g.rho -= adj_r * col_resids[i - 1];
            }
            if params.sigmas[j] > 1e-8 {
                d_lp_ds_j += -inv_s + resid * resid * inv_s2 * inv_s;
            }
            adj_next = adj_r;
        }
        g.sigma[j] = d_lp_ds_j;
    }
    (lp, g)
}

/// Meyers LCL/CCL log-posterior on the unconstrained scale `ψ`.
#[cfg(test)]
fn meyers_log_posterior(psi: &[f64], log_y: &[f64], spec: &MeyersSpec) -> f64 {
    meyers_log_posterior_and_grad(psi, log_y, spec).0
}

/// Meyers LCL/CCL log-posterior and its exact analytic gradient.
fn meyers_log_posterior_and_grad(psi: &[f64], log_y: &[f64], spec: &MeyersSpec) -> (f64, Vec<f64>) {
    let p = psi.len();
    let Some(params) = spec.unpack(psi) else {
        return (f64::NEG_INFINITY, vec![0.0; p]);
    };
    let (ll, lik) = meyers_log_likelihood(&params, log_y, spec.n, spec.is_ccl);
    let mut grad = vec![0.0; p];
    spec.backprop(psi, &params, &lik, &mut grad);
    (params.log_prior + ll, grad)
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
fn safe_neg_hessian_and_inverse<F>(fg: F, x: &[f64]) -> Result<(Vec<f64>, Vec<f64>), ReservingError>
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
    spec: MeyersSpec,
    init: Vec<f64>,
}

#[inline]
fn logit(x: f64) -> f64 {
    (x / (1.0 - x)).ln()
}

fn validate_meyers_options(options: &MeyersOptions) -> Result<(), ReservingError> {
    if options.n_samples == 0 || options.max_map_iterations == 0 {
        return Err(ReservingError::InvalidIterations);
    }
    if !options.proposal_nu.is_finite() || options.proposal_nu <= 2.0 {
        return Err(ReservingError::InvalidParameter(
            "proposal_nu must be finite and > 2.0",
        ));
    }
    if !options.log_elr.is_finite()
        || !options.proposal_inflation.is_finite()
        || !options.tempering_ess_fraction.is_finite()
        || !options.defensive_weight.is_finite()
        || !options.min_ess_fraction.is_finite()
    {
        return Err(ReservingError::NotFinite);
    }
    if options.proposal_inflation <= 0.0 {
        return Err(ReservingError::NonPositive("proposal_inflation"));
    }
    for value in [
        options.tempering_ess_fraction,
        options.defensive_weight,
        options.min_ess_fraction,
    ] {
        if !(0.0..=1.0).contains(&value) {
            return Err(ReservingError::InvalidProbability { value });
        }
    }
    if options.tempering_ess_fraction <= 0.0 {
        return Err(ReservingError::NonPositive("tempering_ess_fraction"));
    }
    options.priors.validate()
}

fn prepare_log_triangle_and_init(
    cumulative: &TriangleView<'_>,
    options: &MeyersOptions,
) -> Result<MeyersInitData, ReservingError> {
    validate_triangle(cumulative)?;
    let n = cumulative.n_origins;
    if n < 3 {
        return Err(ReservingError::InvalidParameter(
            "Meyers LCL/CCL requires at least a 3x3 triangle",
        ));
    }
    validate_meyers_options(options)?;

    let mut log_y = vec![f64::NAN; n * n];
    for i in 0..n {
        for j in 0..(n - i) {
            let v = cumulative.get(i, j);
            if v <= 0.0 {
                return Err(ReservingError::NonPositive(
                    "cumulative triangle cell for log-loss Meyers model",
                ));
            }
            log_y[i * n + j] = v.ln();
        }
    }

    // Chain-ladder ultimates for initialization (and premium proxy when earned_premium is None).
    let mut ldfs = vec![1.0; n - 1];
    volume_weighted_ldfs(cumulative.data, n, n, &mut ldfs);
    let mut cdfs = vec![1.0; n];
    cdfs_from_ldfs(&ldfs, 1.0, &mut cdfs);
    let ultimates: Vec<f64> = (0..n)
        .map(|i| cumulative.get(i, n - 1 - i) * cdfs[n - 1 - i])
        .collect();
    let priors = options.priors;
    let monograph = priors.family == MeyersPriorFamily::Monograph2015;
    let alpha_center = match &options.earned_premium {
        Some(prem) => {
            if prem.len() != n {
                return Err(ReservingError::ShapeMismatch);
            }
            let mut m = Vec::with_capacity(n);
            for &p_i in prem {
                if !p_i.is_finite() {
                    return Err(ReservingError::NotFinite);
                }
                if p_i <= 0.0 {
                    return Err(ReservingError::NonPositive("earned_premium"));
                }
                m.push(if monograph {
                    p_i.ln()
                } else {
                    p_i.ln() + options.log_elr
                });
            }
            m
        }
        None => ultimates.iter().map(|&u| u.max(1.0).ln()).collect(),
    };

    let is_ccl = options.model == MeyersModel::Ccl;
    let spec = MeyersSpec {
        n,
        is_ccl,
        priors,
        alpha_center,
    };
    let p = spec.n_params();
    let mut init = vec![0.0; p];
    for i in 0..n {
        init[i] = ultimates[i].max(1.0).ln();
    }
    if monograph {
        let bb = priors.beta_bound;
        for j in 0..(n - 1) {
            let beta = (-(cdfs[j].max(1e-6).ln())).clamp(-0.98 * bb, 0.98 * bb);
            init[n + j] = logit((beta + bb) / (2.0 * bb));
        }
        // Small equal variance increments a_i (σ_0² ≈ 0.01 n).
        let a0 = (0.01 / priors.sigma_sq_increment_upper).clamp(1e-6, 0.5);
        for j in 0..n {
            init[2 * n - 1 + j] = logit(a0);
        }
        let mean_gap = (0..n).map(|i| init[i] - spec.alpha_center[i]).sum::<f64>() / n as f64;
        let lo = priors.log_elr_lower;
        let hi = priors.log_elr_upper;
        let frac = ((mean_gap - lo) / (hi - lo)).clamp(0.02, 0.98);
        init[spec.log_elr_index()] = logit(frac);
    } else {
        for j in 0..(n - 1) {
            init[n + j] = -(cdfs[j].max(1e-6).ln());
        }
        for j in 0..n {
            init[2 * n - 1 + j] = -2.5;
        }
    }
    if is_ccl {
        init[3 * n - 1] = 0.0;
    }

    Ok(MeyersInitData { log_y, spec, init })
}

/// Damped Newton polish using exact analytic gradients and a finite-difference
/// Hessian so the MAP sits at the exact posterior mode.
fn newton_polish<F>(
    fg: F,
    mut psi: Vec<f64>,
    mut lp: f64,
    mut converged: bool,
    max_iter: usize,
) -> (Vec<f64>, f64, bool)
where
    F: Fn(&[f64]) -> (f64, Vec<f64>),
{
    let p = psi.len();
    for _ in 0..max_iter {
        let (_, grad) = fg(&psi);
        let g_inf = grad.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if g_inf < 1e-4 {
            converged = true;
            break;
        }
        let Ok((_, inv_h)) = safe_neg_hessian_and_inverse(&fg, &psi) else {
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
                cand[k] = psi[k] + alpha * step[k];
            }
            let fc = fg(&cand).0;
            if fc.is_finite() && fc > lp {
                psi.clone_from(&cand);
                lp = fc;
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
    (psi, lp, converged)
}

/// Multivariate Student-$t$ proposal component `t_ν(μ, Σ)`.
#[derive(Clone)]
struct TProposal {
    mean: Vec<f64>,
    chol: Vec<f64>,
    half_log_det: f64,
}

impl TProposal {
    fn new(mean: Vec<f64>, cov: &[f64], p: usize) -> Result<Self, ReservingError> {
        let chol = cholesky_lower(cov, p)?;
        let half_log_det = (0..p).map(|i| chol[i * p + i].ln()).sum();
        Ok(Self {
            mean,
            chol,
            half_log_det,
        })
    }

    /// Log density up to the additive constant shared by all components with the same `ν`, `p`.
    fn log_density(&self, x: &[f64], nu: f64, diff: &mut [f64], y: &mut [f64]) -> f64 {
        let p = x.len();
        for k in 0..p {
            diff[k] = x[k] - self.mean[k];
        }
        solve_lower_into(&self.chol, p, diff, y);
        let q: f64 = y.iter().map(|v| v * v).sum();
        -self.half_log_det - f64::midpoint(nu, p as f64) * (q / nu).ln_1p()
    }

    fn sample_into(&self, rng: &mut DeterministicRng, nu: f64, z: &mut [f64], out: &mut [f64]) {
        let p = out.len();
        for zi in z.iter_mut() {
            *zi = rng.next_standard_normal();
        }
        let w_chi = rng.sample_gamma(0.5 * nu, 2.0).max(1e-300);
        let scale = (nu / w_chi).sqrt();
        for i in 0..p {
            let mut lx = 0.0;
            for k in 0..=i {
                lx += self.chol[i * p + k] * z[k];
            }
            out[i] = self.mean[i] + scale * lx;
        }
    }
}

/// Self-normalized weights and ESS from log-weights (`-inf` allowed).
fn normalize_log_weights(log_w: &[f64], temperature: f64) -> Option<(Vec<f64>, f64)> {
    let max_lw = log_w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max_lw.is_finite() {
        return None;
    }
    let mut w: Vec<f64> = log_w
        .iter()
        .map(|&lw| (temperature * (lw - max_lw)).exp())
        .collect();
    let sum: f64 = w.iter().sum();
    let mut sum2 = 0.0;
    for wi in &mut w {
        *wi /= sum;
        sum2 += *wi * *wi;
    }
    Some((w, 1.0 / sum2.max(1e-300)))
}

/// Final importance sample: `m × p` draws (row-major) with normalized weights.
struct ImportanceSample {
    draws: Vec<f64>,
    weights: Vec<f64>,
    ess: f64,
    laplace_fallback: bool,
}

/// Laplace approximation of a low-dimensional target from its value alone:
/// L-BFGS on central-difference gradients, then a finite-difference Hessian.
/// Returns `None` if the refined mode is not better than `start` or the
/// curvature cannot be inverted.
fn marginal_laplace<F>(f: &F, start: &[f64], max_iter: usize) -> Option<(Vec<f64>, Vec<f64>)>
where
    F: Fn(&[f64]) -> f64,
{
    let f0 = f(start);
    if !f0.is_finite() {
        return None;
    }
    let fg = |x: &[f64]| -> (f64, Vec<f64>) {
        let fx = f(x);
        let mut g = vec![0.0; x.len()];
        let mut w = x.to_vec();
        for k in 0..x.len() {
            let h = 1e-5 * (1.0 + x[k].abs());
            w[k] = x[k] + h;
            let fp = f(&w);
            w[k] = x[k] - h;
            let fm = f(&w);
            w[k] = x[k];
            g[k] = if fp.is_finite() && fm.is_finite() {
                (fp - fm) / (2.0 * h)
            } else {
                0.0
            };
        }
        (fx, g)
    };
    let (x, fx, _) = lbfgs_maximize(fg, start.to_vec(), max_iter, 1e-5);
    if !(fx >= f0) {
        return None;
    }
    let (_, cov) = safe_neg_hessian_and_inverse(fg, &x).ok()?;
    Some((x, cov))
}

/// Deterministic adaptive importance sampling (population Monte Carlo,
/// Cappé et al. 2004) with tempered moment matching and a defensive
/// Laplace-$t$ mixture component (Hesterberg 1995).
///
/// Round 0 uses `t_ν(ψ_MAP, c² Σ_Laplace)`. Each adaptation round draws `M`
/// samples from the current mixture, tempers the log-weights by the largest
/// `τ ∈ (0, 1]` keeping `ESS(w^τ) >= tempering_ess_fraction · M`, and
/// re-fits the adaptive component's mean and scale matrix
/// `(ν − 2)/ν · c² · Cov_w`. The final `M` draws from
/// `(1 − δ) t(μ_R, Σ_R) + δ t(ψ_MAP, c² Σ_Laplace)` carry exact mixture weights.
fn adaptive_importance_sample<F>(
    log_target: F,
    mode: &[f64],
    cov_laplace: &[f64],
    options: &MeyersOptions,
    rng: &mut DeterministicRng,
) -> Result<ImportanceSample, ReservingError>
where
    F: Fn(&[f64]) -> f64,
{
    let p = mode.len();
    let m = options.n_samples;
    let nu = options.proposal_nu;
    let infl2 = options.proposal_inflation * options.proposal_inflation;
    // Re-centre on the mode of the (marginal) target itself with its own
    // Laplace curvature; keep the supplied block if that fails.
    let (mode, cov_laplace) = marginal_laplace(&log_target, mode, options.max_map_iterations)
        .unwrap_or_else(|| (mode.to_vec(), cov_laplace.to_vec()));
    let (mode, cov_laplace) = (mode.as_slice(), cov_laplace.as_slice());
    let base_cov: Vec<f64> = cov_laplace.iter().map(|v| v * infl2).collect();
    let base = TProposal::new(mode.to_vec(), &base_cov, p)?;
    let mut adapt = base.clone();
    let delta = options.defensive_weight;
    let ln_delta = delta.ln();
    let ln_one_m_delta = (1.0 - delta).ln();

    let mut z = vec![0.0; p];
    let mut diff = vec![0.0; p];
    let mut y = vec![0.0; p];
    let mut draws = vec![0.0; m * p];
    let mut log_w = vec![0.0; m];

    let mut run_round =
        |adapt: &TProposal, rng: &mut DeterministicRng, draws: &mut [f64], log_w: &mut [f64]| {
            for (r, lw) in log_w.iter_mut().enumerate() {
                let x = &mut draws[r * p..(r + 1) * p];
                let from_base = delta > 0.0 && rng.next_open01() < delta;
                if from_base {
                    base.sample_into(rng, nu, &mut z, x);
                } else {
                    adapt.sample_into(rng, nu, &mut z, x);
                }
                let lp = log_target(x);
                let lq_a = adapt.log_density(x, nu, &mut diff, &mut y);
                let lq = if delta > 0.0 {
                    let lq_b = base.log_density(x, nu, &mut diff, &mut y);
                    let a = ln_one_m_delta + lq_a;
                    let b = ln_delta + lq_b;
                    let hi = a.max(b);
                    hi + ((a - hi).exp() + (b - hi).exp()).ln()
                } else {
                    lq_a
                };
                *lw = if lp.is_finite() && lq.is_finite() {
                    lp - lq
                } else {
                    f64::NEG_INFINITY
                };
            }
        };

    let mut pooled_draws: Vec<f64> = Vec::with_capacity((options.adaptation_rounds + 1) * m * p);
    let mut pooled_log_w: Vec<f64> = Vec::with_capacity((options.adaptation_rounds + 1) * m);
    for _ in 0..options.adaptation_rounds {
        run_round(&adapt, rng, &mut draws, &mut log_w);
        pooled_draws.extend_from_slice(&draws);
        pooled_log_w.extend_from_slice(&log_w);
        let target_ess = options.tempering_ess_fraction * m as f64;
        // Largest temperature τ with ESS(w^τ) >= target (ESS is non-increasing in τ).
        let tau = match normalize_log_weights(&log_w, 1.0) {
            None => break,
            Some((_, ess1)) if ess1 >= target_ess => 1.0,
            Some(_) => {
                let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
                for _ in 0..40 {
                    let mid = f64::midpoint(lo, hi);
                    match normalize_log_weights(&log_w, mid) {
                        Some((_, e)) if e >= target_ess => lo = mid,
                        _ => hi = mid,
                    }
                }
                lo
            }
        };
        if tau <= 0.0 {
            break;
        }
        let Some((w, _)) = normalize_log_weights(&log_w, tau) else {
            break;
        };
        let mut mean = vec![0.0; p];
        for (r, &wr) in w.iter().enumerate() {
            for k in 0..p {
                mean[k] += wr * draws[r * p + k];
            }
        }
        let mut cov = vec![0.0; p * p];
        for (r, &wr) in w.iter().enumerate() {
            if wr == 0.0 {
                continue;
            }
            let x = &draws[r * p..(r + 1) * p];
            for a in 0..p {
                let da = x[a] - mean[a];
                for b in 0..=a {
                    cov[a * p + b] += wr * da * (x[b] - mean[b]);
                }
            }
        }
        let scale = (nu - 2.0) / nu * infl2;
        for a in 0..p {
            for b in 0..=a {
                let v = cov[a * p + b] * scale;
                cov[a * p + b] = v;
                cov[b * p + a] = v;
            }
        }
        if let Ok(next) = TProposal::new(mean, &cov, p) {
            adapt = next;
        } else {
            break;
        }
    }

    run_round(&adapt, rng, &mut draws, &mut log_w);
    pooled_draws.extend_from_slice(&draws);
    pooled_log_w.extend_from_slice(&log_w);
    // Population Monte Carlo pooling (Cappé et al. 2004): each round is a
    // valid self-normalised IS estimator w.r.t. its own mixture proposal; the
    // rounds are combined with weights proportional to their ESS so poorly
    // adapted early rounds cannot dominate.
    let n_rounds = pooled_log_w.len() / m;
    let mut weights = Vec::with_capacity(pooled_log_w.len());
    let mut round_ess = Vec::with_capacity(n_rounds);
    let mut round_w = Vec::with_capacity(n_rounds);
    for r in 0..n_rounds {
        let (w_r, e_r) = normalize_log_weights(&pooled_log_w[r * m..(r + 1) * m], 1.0)
            .unwrap_or_else(|| (vec![0.0; m], 0.0));
        round_ess.push(e_r);
        round_w.push(w_r);
    }
    let ess_sum: f64 = round_ess.iter().sum();
    if !(ess_sum > 0.0) {
        return Err(ReservingError::NotFinite);
    }
    for (w_r, e_r) in round_w.iter().zip(&round_ess) {
        weights.extend(w_r.iter().map(|w| w * e_r / ess_sum));
    }
    let ess = 1.0 / weights.iter().map(|w| w * w).sum::<f64>().max(1e-300);
    let draws = pooled_draws;
    let min_ess = options.min_ess_fraction * m as f64;
    if ess >= min_ess {
        return Ok(ImportanceSample {
            draws,
            weights,
            ess,
            laplace_fallback: false,
        });
    }
    if !options.fallback_to_laplace {
        return Err(ReservingError::InvalidParameter(
            "Meyers SNIS effective sample size fell below min_ess_fraction * n_samples; increase \
             n_samples / adaptation_rounds or set fallback_to_laplace",
        ));
    }
    // Unweighted Gaussian Laplace draws N(ψ_MAP, Σ_Laplace).
    let mut draws = vec![0.0; m * p];
    let laplace_chol = cholesky_lower(cov_laplace, p)?;
    for r in 0..m {
        for zi in &mut z {
            *zi = rng.next_standard_normal();
        }
        for i in 0..p {
            let mut lx = 0.0;
            for k in 0..=i {
                lx += laplace_chol[i * p + k] * z[k];
            }
            draws[r * p + i] = mode[i] + lx;
        }
    }
    Ok(ImportanceSample {
        draws,
        weights: vec![1.0 / m as f64; m],
        ess: m as f64,
        laplace_fallback: true,
    })
}

/// Extract the `idx × idx` sub-matrix of a row-major `dim × dim` matrix.
fn sub_matrix(a: &[f64], dim: usize, idx: &[usize]) -> Vec<f64> {
    let k = idx.len();
    let mut out = vec![0.0; k * k];
    for (r, &i) in idx.iter().enumerate() {
        for (c, &j) in idx.iter().enumerate() {
            out[r * k + c] = a[i * dim + j];
        }
    }
    out
}

/// One line's inputs to [`conditional_linear_block`].
struct LineBlockInput<'a> {
    spec: &'a MeyersSpec,
    log_y: &'a [f64],
    nl: &'a NlParams,
}

/// Exact Gaussian conditional `γ | θ, Y` of the linear block
/// `γ = [α_0..α_{n−1}, β_0..β_{n−2}]` (stacked per line) with the
/// log marginal `ln ∫ p(Y, γ, θ) dγ` up to a θ-independent constant.
struct LinearBlock {
    mean: Vec<f64>,
    /// Lower Cholesky factor of the conditional precision `A = L Lᵀ`.
    chol: Vec<f64>,
    log_marginal: f64,
}

impl LinearBlock {
    /// Draw `γ = mean + L^{-T} z`, `z ~ N(0, I)`.
    fn sample_into(&self, rng: &mut DeterministicRng, out: &mut [f64]) {
        let dim = self.mean.len();
        let z: Vec<f64> = (0..dim).map(|_| rng.next_standard_normal()).collect();
        for i in (0..dim).rev() {
            let mut sum = z[i];
            for j in (i + 1)..dim {
                sum -= self.chol[j * dim + i] * out[j];
            }
            out[i] = sum / self.chol[i * dim + i];
        }
        for (o, m) in out.iter_mut().zip(&self.mean) {
            *o += m;
        }
    }
}

/// Build the conditional Gaussian of the linear block given the non-linear
/// parameters of one or two lines (cross-line correlation `rho12` for two).
///
/// Conditional on `(σ, ρ)`, every CCL residual
/// `r_{i,j} = y_{i,j} − α_i − β_j − ρ r_{i−1,j}` is affine in `γ`, so the
/// log-likelihood is an exact quadratic form in `γ`. Combined with the Gaussian
/// `α` (and smooth-family `β`) priors this yields `γ | θ ~ N(A⁻¹ b, A⁻¹)` and
/// `ln p(Y, θ) = c + ½ bᵀA⁻¹b − ½ ln|A| + Σ cell log-normalisers + ln p(θ)`.
/// Monograph `β ~ U(−B, B)` contributes a constant density; draws outside the
/// box are given zero importance weight by the caller, which keeps the
/// importance sampler exact.
fn conditional_linear_block(lines: &[LineBlockInput<'_>], rho12: f64) -> Option<LinearBlock> {
    let n = lines[0].spec.n;
    let k = 2 * n - 1;
    let n_lines = lines.len();
    let dim = k * n_lines;
    let omr2 = 1.0 - rho12 * rho12;
    if n_lines == 2 && omr2 <= 1e-12 {
        return None;
    }
    let mut a = vec![0.0; dim * dim];
    let mut b = vec![0.0; dim];
    let mut c = 0.0;
    let mut log_norm = 0.0;
    let mut d_cur = [0.0; 2];
    let mut m_cur = vec![0.0; n_lines * k];
    let mut d_prev = vec![0.0; n_lines * n];
    let mut m_prev = vec![0.0; n_lines * n * k];
    let mut prec = [[0.0; 2]; 2];

    for i in 0..n {
        for j in 0..(n - i) {
            for (l, line) in lines.iter().enumerate() {
                let rho = if line.spec.is_ccl { line.nl.rho } else { 0.0 };
                let slot = (l * n + j) * k;
                let mrow = &mut m_cur[l * k..(l + 1) * k];
                if i > 0 && rho != 0.0 {
                    d_cur[l] = line.log_y[i * n + j] - rho * d_prev[l * n + j];
                    for (dst, &pv) in mrow.iter_mut().zip(&m_prev[slot..slot + k]) {
                        *dst = -rho * pv;
                    }
                } else {
                    d_cur[l] = line.log_y[i * n + j];
                    mrow.fill(0.0);
                }
                mrow[i] += 1.0;
                if j < n - 1 {
                    mrow[n + j] += 1.0;
                }
                d_prev[l * n + j] = d_cur[l];
                m_prev[slot..slot + k].copy_from_slice(mrow);
            }
            if n_lines == 1 {
                let s = lines[0].nl.sigmas[j].max(1e-8);
                prec[0][0] = 1.0 / (s * s);
                log_norm -= s.ln();
            } else {
                let s1 = lines[0].nl.sigmas[j].max(1e-8);
                let s2 = lines[1].nl.sigmas[j].max(1e-8);
                prec[0][0] = 1.0 / (omr2 * s1 * s1);
                prec[1][1] = 1.0 / (omr2 * s2 * s2);
                prec[0][1] = -rho12 / (omr2 * s1 * s2);
                prec[1][0] = prec[0][1];
                log_norm -= s1.ln() + s2.ln() + 0.5 * omr2.ln();
            }
            for la in 0..n_lines {
                for lb in 0..n_lines {
                    let pab = prec[la][lb];
                    c -= 0.5 * pab * d_cur[la] * d_cur[lb];
                    for u in 0..k {
                        let mu_a = m_cur[la * k + u];
                        if mu_a == 0.0 {
                            continue;
                        }
                        let w = pab * mu_a;
                        b[la * k + u] += w * d_cur[lb];
                        let row = (la * k + u) * dim + lb * k;
                        for v in 0..k {
                            a[row + v] += w * m_cur[lb * k + v];
                        }
                    }
                }
            }
        }
    }

    let mut log_prior = 0.0;
    for (l, line) in lines.iter().enumerate() {
        let pr = &line.spec.priors;
        log_prior += line.nl.log_prior;
        let inv_var_a = 1.0 / (pr.alpha_sd * pr.alpha_sd);
        for i in 0..n {
            let idx = l * k + i;
            let mu = line.spec.alpha_center[i] + line.nl.log_elr;
            a[idx * dim + idx] += inv_var_a;
            b[idx] += mu * inv_var_a;
            c -= 0.5 * mu * mu * inv_var_a;
        }
        if !line.spec.is_monograph() {
            let inv_var_b = 1.0 / (pr.smooth_beta_sd * pr.smooth_beta_sd);
            for j in 0..(n - 1) {
                let idx = l * k + n + j;
                a[idx * dim + idx] += inv_var_b;
            }
        }
    }

    let chol = cholesky_lower(&a, dim).ok()?;
    let mut y = vec![0.0; dim];
    solve_lower_into(&chol, dim, &b, &mut y);
    let half_log_det: f64 = (0..dim).map(|i| chol[i * dim + i].ln()).sum();
    let quad: f64 = y.iter().map(|v| v * v).sum();
    let mut mean = vec![0.0; dim];
    for i in (0..dim).rev() {
        let mut sum = y[i];
        for j in (i + 1)..dim {
            sum -= chol[j * dim + i] * mean[j];
        }
        mean[i] = sum / chol[i * dim + i];
    }
    // `c + ½ bᵀA⁻¹b` cancels catastrophically (both terms ~ Σ y²/σ²), so the
    // minimised quadratic is re-evaluated from residuals at the conditional mean.
    let _ = (c, quad);
    let mut q_min = 0.0;
    let mut r_prev = vec![0.0; n_lines * n];
    let mut r_cur = [0.0; 2];
    for i in 0..n {
        for j in 0..(n - i) {
            for (l, line) in lines.iter().enumerate() {
                let rho = if line.spec.is_ccl { line.nl.rho } else { 0.0 };
                let beta = if j < n - 1 { mean[l * k + n + j] } else { 0.0 };
                let carry = if i > 0 { rho * r_prev[l * n + j] } else { 0.0 };
                r_cur[l] = line.log_y[i * n + j] - mean[l * k + i] - beta - carry;
                r_prev[l * n + j] = r_cur[l];
            }
            if n_lines == 1 {
                let s = lines[0].nl.sigmas[j].max(1e-8);
                q_min += (r_cur[0] / s).powi(2);
            } else {
                let u1 = r_cur[0] / lines[0].nl.sigmas[j].max(1e-8);
                let u2 = r_cur[1] / lines[1].nl.sigmas[j].max(1e-8);
                q_min += (u1 * u1 - 2.0 * rho12 * u1 * u2 + u2 * u2) / omr2;
            }
        }
    }
    for (l, line) in lines.iter().enumerate() {
        let pr = &line.spec.priors;
        for i in 0..n {
            let d = mean[l * k + i] - line.spec.alpha_center[i] - line.nl.log_elr;
            q_min += d * d / (pr.alpha_sd * pr.alpha_sd);
        }
        if !line.spec.is_monograph() {
            for j in 0..(n - 1) {
                let bj = mean[l * k + n + j];
                q_min += bj * bj / (pr.smooth_beta_sd * pr.smooth_beta_sd);
            }
        }
    }
    let log_marginal = -0.5 * q_min - half_log_det + log_norm + log_prior;
    if !log_marginal.is_finite() {
        return None;
    }
    Some(LinearBlock {
        mean,
        chol,
        log_marginal,
    })
}

/// Weighted mean and standard deviation.
fn weighted_mean_sd(weights: &[f64], values: &[f64]) -> (f64, f64) {
    let mean: f64 = weights.iter().zip(values).map(|(w, v)| w * v).sum();
    let var: f64 = weights
        .iter()
        .zip(values)
        .map(|(w, v)| w * (v - mean).powi(2))
        .sum();
    (mean, var.max(0.0).sqrt())
}

/// Fit Meyers' Leveled Chain Ladder (LCL) or Correlated Chain Ladder (CCL)
/// (CAS Monograph No. 1) using deterministic L-BFGS MAP estimation with a
/// Newton polish, spectral Hessian repair, and adaptive Student-$t$
/// Self-Normalized Importance Sampling (SNIS) with a defensive Laplace
/// mixture component.
///
/// # Errors
///
/// Returns [`ReservingError::InvalidParameter`] when the final ESS is below
/// `min_ess_fraction · n_samples` and `fallback_to_laplace` is `false`.
pub fn meyers_laplace_snis(
    cumulative: &TriangleView<'_>,
    options: &MeyersOptions,
) -> Result<MeyersResult, ReservingError> {
    let (res, _, _, _) = meyers_laplace_snis_internal(cumulative, options)?;
    Ok(res)
}

#[allow(clippy::type_complexity)]
fn meyers_laplace_snis_internal(
    cumulative: &TriangleView<'_>,
    options: &MeyersOptions,
) -> Result<(MeyersResult, Vec<f64>, MeyersSpec, Vec<f64>), ReservingError> {
    let n = cumulative.n_origins;
    let MeyersInitData { log_y, spec, init } = prepare_log_triangle_and_init(cumulative, options)?;
    let is_ccl = spec.is_ccl;
    let p = init.len();

    let obj_and_grad = |psi: &[f64]| meyers_log_posterior_and_grad(psi, &log_y, &spec);
    let (psi_lbfgs, lp_lbfgs, conv_lbfgs) =
        lbfgs_maximize(obj_and_grad, init, options.max_map_iterations, 1e-6);
    let (psi_map, map_log_posterior, converged) = newton_polish(
        obj_and_grad,
        psi_lbfgs,
        lp_lbfgs,
        conv_lbfgs,
        options.max_newton_iterations,
    );

    let (_, cov_laplace) = safe_neg_hessian_and_inverse(obj_and_grad, &psi_map)?;
    // Rao-Blackwellised SNIS: (α, β) are Gaussian conditional on the
    // non-linear block θ = (σ, ρ, logelr), so only θ is importance-sampled.
    let lin = 2 * n - 1;
    let nl_idx: Vec<usize> = (lin..p).collect();
    let theta_map = psi_map[lin..].to_vec();
    let cov_nl = sub_matrix(&cov_laplace, p, &nl_idx);
    let block_for = |theta: &[f64]| -> Option<(NlParams, LinearBlock)> {
        let nl = spec.unpack_nonlinear(theta)?;
        let blk = conditional_linear_block(
            &[LineBlockInput {
                spec: &spec,
                log_y: &log_y,
                nl: &nl,
            }],
            0.0,
        )?;
        Some((nl, blk))
    };
    let mut rng = DeterministicRng::from_seed(options.seed);
    let sample = adaptive_importance_sample(
        |theta: &[f64]| block_for(theta).map_or(f64::NEG_INFINITY, |(_, b)| b.log_marginal),
        &theta_map,
        &cov_nl,
        options,
        &mut rng,
    )?;
    let p_nl = nl_idx.len();
    let m_draws = sample.weights.len();

    let mut draw_rhos = Vec::with_capacity(m_draws);
    let mut draw_totals = Vec::with_capacity(m_draws);
    let mut draw_origins: Vec<Vec<f64>> = (0..n).map(|_| Vec::with_capacity(m_draws)).collect();
    let mut prev_resid = vec![0.0; n];
    let mut weights = sample.weights;
    let mut gamma = vec![0.0; lin];

    for r in 0..m_draws {
        let theta = &sample.draws[r * p_nl..(r + 1) * p_nl];
        let params = block_for(theta).and_then(|(nl, blk)| {
            blk.sample_into(&mut rng, &mut gamma);
            spec.params_from_linear(&gamma, nl)
        });
        let Some(params) = params else {
            // Outside the support (or β outside the Monograph box): zero weight.
            weights[r] = 0.0;
            draw_rhos.push(0.0);
            draw_totals.push(0.0);
            for slot in &mut draw_origins {
                slot.push(0.0);
            }
            continue;
        };
        let rho = params.rho;
        draw_rhos.push(rho);

        // Simulate future log-triangle propagating rho along columns (only y_last needed).
        prev_resid.fill(0.0);
        let mut tot_r = 0.0;
        for i in 0..n {
            let mut y_last = 0.0;
            for j in 0..n {
                let b_j = params.betas[j];
                let mu = if i == 0 || !is_ccl {
                    params.alphas[i] + b_j
                } else {
                    params.alphas[i] + b_j + rho * prev_resid[j]
                };
                if j < n - i {
                    prev_resid[j] = log_y[i * n + j] - mu;
                } else {
                    let eps_draw = params.sigmas[j] * rng.next_standard_normal();
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
    let w_sum: f64 = weights.iter().sum();
    if !(w_sum > 0.0) {
        return Err(ReservingError::NotFinite);
    }
    for w in &mut weights {
        *w /= w_sum;
    }

    let (rho_posterior_mean, _) = weighted_mean_sd(&weights, &draw_rhos);
    let (total_mean_reserve, total_sd_reserve) = weighted_mean_sd(&weights, &draw_totals);
    let mut origin_mean_reserves = vec![0.0; n];
    let mut origin_sd_reserves = vec![0.0; n];
    for i in 0..n {
        let (m_i, s_i) = weighted_mean_sd(&weights, &draw_origins[i]);
        origin_mean_reserves[i] = m_i;
        origin_sd_reserves[i] = s_i;
    }

    let map = spec.unpack(&psi_map).ok_or(ReservingError::NotFinite)?;
    let result = MeyersResult {
        model: options.model,
        alpha_map: map.alphas,
        beta_map: map.betas,
        sigma_map: map.sigmas,
        rho_map: map.rho,
        log_elr_map: if spec.is_monograph() {
            map.log_elr
        } else {
            options.log_elr
        },
        rho_posterior_mean,
        origin_mean_reserves,
        origin_sd_reserves,
        total_mean_reserve,
        total_sd_reserve,
        ess: sample.ess,
        min_ess: options.min_ess_fraction * options.n_samples as f64,
        low_ess_warning: sample.ess < 0.1 * options.n_samples as f64,
        laplace_fallback: sample.laplace_fallback,
        map_log_posterior,
        converged: converged && !sample.laplace_fallback,
    };
    Ok((result, log_y, spec, psi_map))
}

/// Bivariate Normal LCL/CCL log-likelihood for two aligned lines with
/// cross-line residual correlation `ρ_12`, returning per-line natural-scale
/// gradients and `∂/∂ρ_12`.
fn meyers_bivariate_log_likelihood(
    u1: &MeyersParams,
    u2: &MeyersParams,
    rho12: f64,
    log_y1: &[f64],
    log_y2: &[f64],
    n: usize,
    is_ccl: bool,
) -> (f64, MeyersLikGrad, MeyersLikGrad, f64) {
    let mut g1 = MeyersLikGrad::zeros(n);
    let mut g2 = MeyersLikGrad::zeros(n);
    let mut grad_rho12 = 0.0;
    let one_m_r2 = (1.0 - rho12 * rho12).max(1e-12);
    let inv_omr2 = 1.0 / one_m_r2;
    let half_ln_omr2 = 0.5 * one_m_r2.ln();
    let (rho1, rho2) = (u1.rho, u2.rho);
    let mut lp = 0.0;
    let mut col_r1 = vec![0.0; n];
    let mut col_r2 = vec![0.0; n];

    for j in 0..n {
        let col_len = n - j;
        let bj1 = u1.betas[j];
        let bj2 = u2.betas[j];
        let s1 = u1.sigmas[j].max(1e-8);
        let s2 = u2.sigmas[j].max(1e-8);
        let inv_s1 = 1.0 / s1;
        let inv_s2 = 1.0 / s2;
        let ln_s12 = s1.ln() + s2.ln() + half_ln_omr2;

        let mut pr1 = 0.0;
        let mut pr2 = 0.0;
        for i in 0..col_len {
            let mu1 = if i == 0 || !is_ccl {
                u1.alphas[i] + bj1
            } else {
                u1.alphas[i] + bj1 + rho1 * pr1
            };
            let mu2 = if i == 0 || !is_ccl {
                u2.alphas[i] + bj2
            } else {
                u2.alphas[i] + bj2 + rho2 * pr2
            };
            let r1 = log_y1[i * n + j] - mu1;
            let r2 = log_y2[i * n + j] - mu2;
            let a = r1 * inv_s1;
            let b = r2 * inv_s2;
            let q = a * a - 2.0 * rho12 * a * b + b * b;
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
            let a = col_r1[i] * inv_s1;
            let b = col_r2[i] * inv_s2;
            let q = a * a - 2.0 * rho12 * a * b + b * b;
            let adj1 =
                -inv_omr2 * (a - rho12 * b) * inv_s1 - if is_ccl { rho1 * adj1_next } else { 0.0 };
            let adj2 =
                -inv_omr2 * (b - rho12 * a) * inv_s2 - if is_ccl { rho2 * adj2_next } else { 0.0 };
            g1.alpha[i] -= adj1;
            g2.alpha[i] -= adj2;
            if j < n - 1 {
                g1.beta[j] -= adj1;
                g2.beta[j] -= adj2;
            }
            if i > 0 && is_ccl {
                g1.rho -= adj1 * col_r1[i - 1];
                g2.rho -= adj2 * col_r2[i - 1];
            }
            if u1.sigmas[j] > 1e-8 {
                d_s1 += -inv_s1 + inv_omr2 * (a - rho12 * b) * a * inv_s1;
            }
            if u2.sigmas[j] > 1e-8 {
                d_s2 += -inv_s2 + inv_omr2 * (b - rho12 * a) * b * inv_s2;
            }
            grad_rho12 += rho12 * inv_omr2 + inv_omr2 * a * b - rho12 * inv_omr2 * inv_omr2 * q;
            adj1_next = adj1;
            adj2_next = adj2;
        }
        g1.sigma[j] = d_s1;
        g2.sigma[j] = d_s2;
    }
    (lp, g1, g2, grad_rho12)
}

/// Exact joint bivariate Normal log-posterior and reverse-mode adjoint gradient
/// over `joint = [ψ_1 (p), ψ_2 (p), z_12 (1)]` (Meyers CAS Monograph 1).
fn meyers_bivariate_log_posterior_and_grad(
    joint: &[f64],
    log_y1: &[f64],
    log_y2: &[f64],
    spec1: &MeyersSpec,
    spec2: &MeyersSpec,
) -> (f64, Vec<f64>) {
    let p_total = joint.len();
    let p = (p_total - 1) / 2;
    let psi1 = &joint[0..p];
    let psi2 = &joint[p..2 * p];
    let z12 = joint[2 * p];
    if !(-10.0..=10.0).contains(&z12) {
        return (f64::NEG_INFINITY, vec![0.0; p_total]);
    }
    let (Some(u1), Some(u2)) = (spec1.unpack(psi1), spec2.unpack(psi2)) else {
        return (f64::NEG_INFINITY, vec![0.0; p_total]);
    };
    let rho12 = z12.tanh();
    let (ll, g1, g2, g_rho12) =
        meyers_bivariate_log_likelihood(&u1, &u2, rho12, log_y1, log_y2, spec1.n, spec1.is_ccl);
    let mut grad = vec![0.0; p_total];
    spec1.backprop(psi1, &u1, &g1, &mut grad[0..p]);
    spec2.backprop(psi2, &u2, &g2, &mut grad[p..2 * p]);
    grad[2 * p] = -2.0 * rho12 + g_rho12 * (1.0 - rho12 * rho12);
    (
        u1.log_prior + u2.log_prior + log_uniform_rho_density(z12) + ll,
        grad,
    )
}

/// Bivariate Correlated Chain Ladder (Meyers CAS Monograph 1) for two aligned
/// lines of business with cross-line residual correlation `ρ_12 ∈ (-1, 1)`.
///
/// Jointly maximizes (L-BFGS + Newton polish) and adaptively importance-samples
/// the full bivariate Normal log-posterior `p(Y_1, Y_2 | ψ_1, ψ_2, ρ_12)` with
/// exact analytic adjoint gradients so cross-line correlation uncertainty feeds
/// directly into both parameter and process reserve uncertainty.
///
/// # Errors
///
/// As [`meyers_laplace_snis`], including the `min_ess_fraction` gate applied
/// to each standalone fit and to the joint sampler.
pub fn meyers_bivariate_ccl(
    line1: &TriangleView<'_>,
    line2: &TriangleView<'_>,
    options: &MeyersOptions,
) -> Result<BivariateCclResult, ReservingError> {
    if line1.n_origins != line2.n_origins || line1.n_devs != line2.n_devs {
        return Err(ReservingError::ShapeMismatch);
    }
    let (fit1, log_y1, spec1, psi1_map) = meyers_laplace_snis_internal(line1, options)?;
    let (fit2, log_y2, spec2, psi2_map) = meyers_laplace_snis_internal(line2, options)?;
    let n = line1.n_origins;
    let is_ccl = spec1.is_ccl;
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
        meyers_bivariate_log_posterior_and_grad(joint, &log_y1, &log_y2, &spec1, &spec2)
    };
    let (lbfgs_map, lbfgs_lp, lbfgs_conv) =
        lbfgs_maximize(joint_fg, joint_init, options.max_map_iterations, 1e-5);
    let (joint_map, _, converged) = newton_polish(
        joint_fg,
        lbfgs_map,
        lbfgs_lp,
        lbfgs_conv,
        options.max_newton_iterations,
    );
    let (_, cov_laplace) = safe_neg_hessian_and_inverse(joint_fg, &joint_map)?;

    // Rao-Blackwellised joint SNIS over θ = (θ_1, θ_2, z_12); (α, β) of both
    // lines are jointly Gaussian conditional on θ.
    let lin = 2 * n - 1;
    let p_nl_line = p - lin;
    let nl_idx: Vec<usize> = (lin..p)
        .chain((p + lin)..(2 * p))
        .chain(std::iter::once(2 * p))
        .collect();
    let theta_map: Vec<f64> = nl_idx.iter().map(|&k| joint_map[k]).collect();
    let cov_nl = sub_matrix(&cov_laplace, p_joint, &nl_idx);
    let joint_block = |theta: &[f64]| -> Option<(NlParams, NlParams, f64, LinearBlock)> {
        let z12 = theta[2 * p_nl_line];
        if !(-10.0..=10.0).contains(&z12) {
            return None;
        }
        let nl1 = spec1.unpack_nonlinear(&theta[0..p_nl_line])?;
        let nl2 = spec2.unpack_nonlinear(&theta[p_nl_line..2 * p_nl_line])?;
        let rho12 = z12.tanh();
        let mut blk = conditional_linear_block(
            &[
                LineBlockInput {
                    spec: &spec1,
                    log_y: &log_y1,
                    nl: &nl1,
                },
                LineBlockInput {
                    spec: &spec2,
                    log_y: &log_y2,
                    nl: &nl2,
                },
            ],
            rho12,
        )?;
        blk.log_marginal += log_uniform_rho_density(z12);
        Some((nl1, nl2, rho12, blk))
    };
    let mut rng = DeterministicRng::from_seed(options.seed.wrapping_add(1));
    let sample = adaptive_importance_sample(
        |theta: &[f64]| joint_block(theta).map_or(f64::NEG_INFINITY, |(_, _, _, b)| b.log_marginal),
        &theta_map,
        &cov_nl,
        options,
        &mut rng,
    )?;
    let p_nl = nl_idx.len();
    let m = sample.weights.len();
    let mut weights = sample.weights;
    let mut draw_rhos = Vec::with_capacity(m);
    let mut port_totals = Vec::with_capacity(m);
    let mut prev1 = vec![0.0; n];
    let mut prev2 = vec![0.0; n];
    let mut gamma = vec![0.0; 2 * lin];

    for r in 0..m {
        let theta = &sample.draws[r * p_nl..(r + 1) * p_nl];
        let drawn = joint_block(theta).and_then(|(nl1, nl2, rho12, blk)| {
            blk.sample_into(&mut rng, &mut gamma);
            let u1 = spec1.params_from_linear(&gamma[0..lin], nl1)?;
            let u2 = spec2.params_from_linear(&gamma[lin..], nl2)?;
            Some((u1, u2, rho12))
        });
        let Some((u1, u2, rho_draw)) = drawn else {
            weights[r] = 0.0;
            draw_rhos.push(0.0);
            port_totals.push(0.0);
            continue;
        };
        draw_rhos.push(rho_draw);
        let cond_sd_factor = (1.0 - rho_draw * rho_draw).max(0.0).sqrt();

        prev1.fill(0.0);
        prev2.fill(0.0);
        let mut tot = 0.0;
        for i in 0..n {
            let mut y1_last = 0.0;
            let mut y2_last = 0.0;
            for j in 0..n {
                let mu1 = if i == 0 || !is_ccl {
                    u1.alphas[i] + u1.betas[j]
                } else {
                    u1.alphas[i] + u1.betas[j] + u1.rho * prev1[j]
                };
                let mu2 = if i == 0 || !is_ccl {
                    u2.alphas[i] + u2.betas[j]
                } else {
                    u2.alphas[i] + u2.betas[j] + u2.rho * prev2[j]
                };
                if j < n - i {
                    prev1[j] = log_y1[i * n + j] - mu1;
                    prev2[j] = log_y2[i * n + j] - mu2;
                } else {
                    let z1 = rng.next_standard_normal();
                    let z2 = rho_draw * z1 + cond_sd_factor * rng.next_standard_normal();
                    let e1 = u1.sigmas[j] * z1;
                    let e2 = u2.sigmas[j] * z2;
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
    let w_sum: f64 = weights.iter().sum();
    if !(w_sum > 0.0) {
        return Err(ReservingError::NotFinite);
    }
    for w in &mut weights {
        *w /= w_sum;
    }

    let (cross_line_rho, _) = weighted_mean_sd(&weights, &draw_rhos);
    let (portfolio_mean_reserve, portfolio_sd_reserve) = weighted_mean_sd(&weights, &port_totals);

    Ok(BivariateCclResult {
        ess: sample.ess,
        laplace_fallback: sample.laplace_fallback,
        converged: converged && !sample.laplace_fallback,
        line1: fit1,
        line2: fit2,
        cross_line_rho,
        portfolio_mean_reserve,
        portfolio_sd_reserve,
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

    fn raa() -> crate::types::Triangle {
        const NAN: f64 = f64::NAN;
        #[rustfmt::skip]
        let data = vec![
            5012.0,  8269.0, 10907.0, 11805.0, 13539.0, 16181.0, 18009.0, 18608.0, 18662.0, 18834.0,
             106.0,  4285.0,  5396.0, 10666.0, 13782.0, 15599.0, 15496.0, 16169.0, 16704.0,     NAN,
            3410.0,  8992.0, 13873.0, 16141.0, 18735.0, 22214.0, 22863.0, 23466.0,     NAN,     NAN,
            5655.0, 11555.0, 15766.0, 21266.0, 23425.0, 26083.0, 27067.0,     NAN,     NAN,     NAN,
            1092.0,  9565.0, 15836.0, 22169.0, 25955.0, 26180.0,     NAN,     NAN,     NAN,     NAN,
            1513.0,  6445.0, 11702.0, 12935.0, 15852.0,     NAN,     NAN,     NAN,     NAN,     NAN,
             557.0,  4020.0, 10946.0, 12314.0,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,
            1351.0,  6947.0, 13112.0,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,
            3133.0,  5395.0,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,
            2063.0,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,
        ];
        crate::types::Triangle::new(10, 10, data)
    }

    #[test]
    fn meyers_lcl_and_ccl_converge_with_healthy_ess_and_monotonic_sigmas() {
        // 10x10 GenIns and RAA, both models and both prior families: the
        // adaptive SNIS must clear min_ess (fit would otherwise return Err).
        for (name, tri) in [("genins", genins()), ("raa", raa())] {
            for priors in [
                MeyersPriors::monograph_2015(),
                MeyersPriors::smooth_gaussian(),
            ] {
                for model in [MeyersModel::Lcl, MeyersModel::Ccl] {
                    let opts = MeyersOptions {
                        model,
                        priors,
                        ..MeyersOptions::default()
                    };
                    let fit = meyers_laplace_snis(&tri.view(), &opts)
                        .unwrap_or_else(|e| panic!("{name} {model:?} {:?}: {e}", priors.family));
                    eprintln!(
                        "{name} {model:?} {:?}: ess={:.1} mean={:.0} sd={:.0} rho={:.3}",
                        priors.family,
                        fit.ess,
                        fit.total_mean_reserve,
                        fit.total_sd_reserve,
                        fit.rho_map
                    );
                    assert!(fit.converged, "{name} {model:?}");
                    assert!(!fit.laplace_fallback);
                    assert!(fit.ess >= fit.min_ess, "{name} {model:?} ess={}", fit.ess);
                    assert!(fit.ess >= 200.0, "{name} {model:?} ess={}", fit.ess);
                    assert!(fit.total_mean_reserve > 0.0 && fit.total_sd_reserve > 0.0);
                    assert!((-1.0..=1.0).contains(&fit.rho_map));
                    for j in 0..(fit.sigma_map.len() - 1) {
                        assert!(fit.sigma_map[j] >= fit.sigma_map[j + 1]);
                    }
                }
            }
        }
    }

    #[test]
    fn meyers_min_ess_gate_errors_or_falls_back_to_laplace() {
        let tri = genins();
        let strict = MeyersOptions {
            n_samples: 64,
            adaptation_rounds: 0,
            min_ess_fraction: 1.0,
            ..MeyersOptions::default()
        };
        assert!(matches!(
            meyers_laplace_snis(&tri.view(), &strict),
            Err(ReservingError::InvalidParameter(_))
        ));
        let fallback = MeyersOptions {
            fallback_to_laplace: true,
            ..strict
        };
        let fit = meyers_laplace_snis(&tri.view(), &fallback).expect("fallback");
        assert!(fit.laplace_fallback);
        assert!(!fit.converged);
        assert!(fit.total_mean_reserve > 0.0);
    }

    #[test]
    fn meyers_bivariate_ccl_captures_positive_cross_line_correlation() {
        let t1 = genins();
        // Scaled copy with small deterministic jitter => strong (but not
        // degenerate) positive cross-line residual correlation. An exact scaled
        // copy makes the ρ_12 posterior improper (mass piles up at ρ_12 → 1).
        let n = 10;
        let mut data2 = vec![f64::NAN; n * n];
        for i in 0..n {
            for j in 0..(n - i) {
                let jitter = 1.0 + 0.02 * (((i * 7 + j * 13) % 5) as f64 - 2.0);
                data2[i * n + j] = t1.get(i, j) * 1.5 * jitter;
            }
        }
        let t2 = crate::types::Triangle::new(n, n, data2);
        let opts = MeyersOptions {
            model: MeyersModel::Ccl,
            n_samples: 4096,
            seed: 42069,
            ..MeyersOptions::default()
        };
        let bi = meyers_bivariate_ccl(&t1.view(), &t2.view(), &opts).expect("bivariate ccl");
        assert!(bi.cross_line_rho > 0.5, "rho12={}", bi.cross_line_rho);
        assert!(bi.ess >= opts.min_ess_fraction * opts.n_samples as f64);
        assert!(bi.converged);
        assert!(bi.portfolio_mean_reserve > 0.0);
        assert!(bi.portfolio_sd_reserve > 0.0);
    }

    #[test]
    fn meyers_ccl_at_zero_rho_matches_lcl_log_posterior() {
        let tri = genins();
        for priors in [
            MeyersPriors::monograph_2015(),
            MeyersPriors::smooth_gaussian(),
        ] {
            let lcl_opts = MeyersOptions {
                model: MeyersModel::Lcl,
                priors,
                ..MeyersOptions::default()
            };
            let ccl_opts = MeyersOptions {
                model: MeyersModel::Ccl,
                ..lcl_opts.clone()
            };
            let lcl = prepare_log_triangle_and_init(&tri.view(), &lcl_opts).expect("init");
            let ccl = prepare_log_triangle_and_init(&tri.view(), &ccl_opts).expect("init");
            let lp_lcl = meyers_log_posterior(&lcl.init, &lcl.log_y, &lcl.spec);
            // CCL init equals LCL init with z_rho = 0 inserted before the logelr coordinate.
            let lp_ccl = meyers_log_posterior(&ccl.init, &ccl.log_y, &ccl.spec);
            // At z_rho = 0, rho = 0 and prior density on z_rho is 1/2 (log = -ln 2).
            assert!(((lp_ccl - lp_lcl) - (-std::f64::consts::LN_2)).abs() < 1e-12);
        }
    }

    #[test]
    fn meyers_univariate_and_bivariate_analytic_gradients_match_central_fd() {
        let tri = genins();
        for priors in [
            MeyersPriors::monograph_2015(),
            MeyersPriors::smooth_gaussian(),
        ] {
            let opts = MeyersOptions {
                model: MeyersModel::Ccl,
                priors,
                ..MeyersOptions::default()
            };
            let MeyersInitData {
                log_y,
                spec,
                mut init,
            } = prepare_log_triangle_and_init(&tri.view(), &opts).expect("init");
            // Perturb z_rho away from 0 so the AR(1) adjoint chain is fully exercised.
            let n = spec.n;
            init[3 * n - 1] = 0.35;

            let (lp0, grad_a) = meyers_log_posterior_and_grad(&init, &log_y, &spec);
            let h = 1e-6;
            let mut work = init.clone();
            for k in 0..init.len() {
                let xk = work[k];
                work[k] = xk + h;
                let fp = meyers_log_posterior(&work, &log_y, &spec);
                work[k] = xk - h;
                let fm = meyers_log_posterior(&work, &log_y, &spec);
                work[k] = xk;
                let fd = (fp - fm) / (2.0 * h);
                let scale = 1.0 + fd.abs();
                assert!(
                    (grad_a[k] - fd).abs() < 1e-5 * scale,
                    "{:?} univariate grad[{k}] analytic={} fd={} (lp0={lp0})",
                    priors.family,
                    grad_a[k],
                    fd
                );
            }

            // Bivariate joint log-posterior gradient check with nonzero rho12.
            let log_y2: Vec<f64> = log_y
                .iter()
                .enumerate()
                .map(|(idx, &v)| {
                    if v.is_finite() {
                        v + 0.25 + 0.03 * ((idx % 7) as f64 - 3.0)
                    } else {
                        f64::NAN
                    }
                })
                .collect();
            let mut joint = Vec::with_capacity(2 * init.len() + 1);
            joint.extend_from_slice(&init);
            joint.extend_from_slice(&init);
            joint.push(0.45);

            let (_, g_joint) =
                meyers_bivariate_log_posterior_and_grad(&joint, &log_y, &log_y2, &spec, &spec);
            let mut wj = joint.clone();
            for k in 0..joint.len() {
                let xk = wj[k];
                wj[k] = xk + h;
                let (fp, _) =
                    meyers_bivariate_log_posterior_and_grad(&wj, &log_y, &log_y2, &spec, &spec);
                wj[k] = xk - h;
                let (fm, _) =
                    meyers_bivariate_log_posterior_and_grad(&wj, &log_y, &log_y2, &spec, &spec);
                wj[k] = xk;
                let fd = (fp - fm) / (2.0 * h);
                let scale = 1.0 + fd.abs();
                assert!(
                    (g_joint[k] - fd).abs() < 1e-4 * scale,
                    "{:?} bivariate grad[{k}] analytic={} fd={}",
                    priors.family,
                    g_joint[k],
                    fd
                );
            }
        }
    }
}
