//! Antonio & Plat (2014) micro-level individual-claim stochastic reserving.
//!
//! Implements:
//!
//! * K. Antonio & R. Plat (2014), *Micro-Level Stochastic Loss Reserving for
//!   General Insurance*, Scandinavian Actuarial Journal, Vol. 2014, No. 7,
//!   pp. 649–669.
//!
//! Models individual claim life-cycles directly rather than aggregating into
//! run-off triangles:
//!
//! 1. **Occurrence process**: piecewise-constant Non-Homogeneous Poisson
//!    Process (NHPP) with intensity $\lambda_k$ over accident period $k$,
//!    estimated with exact right-truncation correction
//!    $\hat{\lambda}_k = N_k^{\text{obs}} / \int_{t_k}^{t_{k+1}} F_R(\tau - t)\,dt$.
//! 2. **Reporting delay**: exponential ($F_R(r) = 1 - e^{-\rho r}$) delay from
//!    occurrence to reporting, estimated via right-truncated maximum likelihood.
//! 3. **Multi-state development & settlement hazard**: open claims evolve in
//!    continuous time under competing Poisson hazards:
//!    * Interim payment hazard $h_{\text{pay}}$ with lognormal severity $(\mu_{\text{pay}}, \sigma_{\text{pay}})$,
//!    * Settlement (closure) hazard $h_{\text{settle}}$ with probability $p_{\text{close}}$
//!      of a final settlement payment with lognormal severity $(\mu_{\text{close}}, \sigma_{\text{close}})$.
//! 4. **RBNS + IBNR Monte Carlo simulation**: deterministic simulation of
//!    open RBNS claims to settlement and thinned-NHPP IBNR claims from
//!    occurrence through reporting to settlement.

use crate::bootstrap_odp::Rng;
use crate::reserve_distribution::empirical_quantile;
use crate::validate::ValidationError;

/// A single payment transaction on an individual claim.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClaimPayment {
    /// Calendar time of the payment ($t \ge \text{report\_time}$).
    pub time: f64,
    /// Payment amount ($> 0$).
    pub amount: f64,
}

/// Individual claim record observed as of the valuation date.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClaimRecord {
    /// Unique claim identifier.
    pub claim_id: u64,
    /// Calendar time of loss occurrence ($0 \le t_{\text{occ}} \le \tau$).
    pub occurrence_time: f64,
    /// Calendar time of claim reporting ($t_{\text{rep}} \ge t_{\text{occ}}$).
    pub report_time: f64,
    /// Settlement/closure calendar time if settled on or before $\tau$; `None` if open (RBNS).
    pub settlement_time: Option<f64>,
    /// Historical payments made on or before $\tau$.
    pub payments: Vec<ClaimPayment>,
}

/// Parametric Antonio & Plat (2014) micro-level reserving model.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MicroReservingModel {
    /// Piecewise-constant NHPP occurrence intensities $\lambda_k$ per unit period, length `n_periods`.
    pub occurrence_intensities: Vec<f64>,
    /// Exponential reporting-delay rate $\rho > 0$ (mean reporting delay $= 1 / \rho$).
    pub reporting_delay_rate: f64,
    /// Interim payment hazard rate $h_{\text{pay}} \ge 0$ per unit time while open.
    pub interim_payment_hazard: f64,
    /// Settlement (closure) hazard rate $h_{\text{settle}} > 0$ per unit time while open.
    pub settlement_hazard: f64,
    /// Lognormal $\mu$ parameter for interim payments.
    pub interim_severity_mu: f64,
    /// Lognormal $\sigma \ge 0$ parameter for interim payments.
    pub interim_severity_sigma: f64,
    /// Probability $p_{\text{close}} \in [0, 1]$ of a final payment upon settlement.
    pub settlement_payment_prob: f64,
    /// Lognormal $\mu$ parameter for settlement payments.
    pub settlement_severity_mu: f64,
    /// Lognormal $\sigma \ge 0$ parameter for settlement payments.
    pub settlement_severity_sigma: f64,
}

impl MicroReservingModel {
    /// Validate all model parameters.
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.occurrence_intensities.is_empty() {
            return Err(ValidationError::InvalidParameter(
                "occurrence_intensities must not be empty",
            ));
        }
        for &lam in &self.occurrence_intensities {
            if !lam.is_finite() || lam < 0.0 {
                return Err(ValidationError::InvalidParameter(
                    "occurrence intensities must be finite and non-negative",
                ));
            }
        }
        if !self.reporting_delay_rate.is_finite() || self.reporting_delay_rate <= 0.0 {
            return Err(ValidationError::InvalidParameter(
                "reporting_delay_rate must be finite and positive",
            ));
        }
        if !self.interim_payment_hazard.is_finite() || self.interim_payment_hazard < 0.0 {
            return Err(ValidationError::InvalidParameter(
                "interim_payment_hazard must be finite and non-negative",
            ));
        }
        if !self.settlement_hazard.is_finite() || self.settlement_hazard <= 0.0 {
            return Err(ValidationError::InvalidParameter(
                "settlement_hazard must be finite and positive",
            ));
        }
        if !self.interim_severity_mu.is_finite()
            || !self.interim_severity_sigma.is_finite()
            || self.interim_severity_sigma < 0.0
            || !self.settlement_severity_mu.is_finite()
            || !self.settlement_severity_sigma.is_finite()
            || self.settlement_severity_sigma < 0.0
            || !self.settlement_payment_prob.is_finite()
            || self.settlement_payment_prob < 0.0
            || self.settlement_payment_prob > 1.0
        {
            return Err(ValidationError::InvalidParameter(
                "invalid severity or settlement payment probability parameter",
            ));
        }
        Ok(())
    }

    /// Exact theoretical expected total cost per claim from reporting to settlement.
    pub fn expected_claim_severity(&self) -> f64 {
        let mean_open_duration = 1.0 / self.settlement_hazard;
        let expected_interim_count = self.interim_payment_hazard * mean_open_duration;
        let mean_interim =
            (self.interim_severity_mu + 0.5 * self.interim_severity_sigma.powi(2)).exp();
        let mean_close =
            (self.settlement_severity_mu + 0.5 * self.settlement_severity_sigma.powi(2)).exp();
        expected_interim_count * mean_interim + self.settlement_payment_prob * mean_close
    }

    /// Exact theoretical expected unreported IBNR claim count over `[0, tau]`
    /// assuming unit-length periods `0..n_periods` and `tau = n_periods as f64`.
    pub fn expected_ibnr_count(&self, valuation_time: f64) -> f64 {
        let rho = self.reporting_delay_rate;
        let mut total = 0.0;
        for (k, &lam) in self.occurrence_intensities.iter().enumerate() {
            let a = k as f64;
            let b = ((k + 1) as f64).min(valuation_time);
            if b <= a {
                continue;
            }
            // Integral_{a}^{b} exp(-rho * (tau - t)) dt = (exp(-rho*(tau-b)) - exp(-rho*(tau-a))) / rho
            let integral =
                ((-rho * (valuation_time - b)).exp() - (-rho * (valuation_time - a)).exp()) / rho;
            total += lam * integral;
        }
        total
    }
}

/// Summary statistics of a micro-level stochastic reserve simulation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MicroReservingResult {
    /// Number of Monte Carlo scenarios simulated.
    pub n_simulations: usize,
    /// Number of open RBNS claims at valuation date.
    pub n_open_claims: usize,
    /// Mean simulated IBNR claim count.
    pub mean_ibnr_count: f64,
    /// Mean RBNS reserve.
    pub mean_rbns: f64,
    /// Mean IBNR reserve.
    pub mean_ibnr: f64,
    /// Mean total reserve (`mean_rbns + mean_ibnr`).
    pub mean_total: f64,
    /// Standard deviation of RBNS reserve.
    pub sd_rbns: f64,
    /// Standard deviation of IBNR reserve.
    pub sd_ibnr: f64,
    /// Standard deviation of total reserve.
    pub sd_total: f64,
    /// 50th, 75th, 90th, 95th, and 99th percentiles of total reserve.
    pub total_quantiles: [f64; 5],
    /// Simulated total reserve draws (length `n_simulations`).
    pub total_draws: Vec<f64>,
}

impl std::fmt::Display for MicroReservingResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MicroReserving(rbns={:.2}, ibnr={:.2}, total={:.2}, sd={:.2})",
            self.mean_rbns, self.mean_ibnr, self.mean_total, self.sd_total
        )
    }
}

/// Fit an Antonio & Plat (2014) [`MicroReservingModel`] from an observed
/// individual-claim event log up to `valuation_time` across `n_periods` unit
/// accident periods `[0, 1), ..., [n_periods - 1, n_periods)`.
pub fn fit_from_claims(
    claims: &[ClaimRecord],
    valuation_time: f64,
    n_periods: usize,
) -> Result<MicroReservingModel, ValidationError> {
    if claims.is_empty() || n_periods == 0 || !valuation_time.is_finite() || valuation_time <= 0.0 {
        return Err(ValidationError::InvalidParameter(
            "non-empty claims, positive valuation_time, and n_periods >= 1 required",
        ));
    }

    // 1. Right-truncated MLE for exponential reporting-delay rate rho.
    // Log-likelihood: sum_i [ ln(rho) - rho * r_i - ln(1 - exp(-rho * T_i)) ], T_i = tau - t_occ_i.
    let mut delays = Vec::new();
    let mut truncs = Vec::new();
    for c in claims {
        if c.report_time <= valuation_time && c.report_time >= c.occurrence_time {
            let t_cap = valuation_time - c.occurrence_time;
            if t_cap > 1e-9 {
                delays.push(c.report_time - c.occurrence_time);
                truncs.push(t_cap);
            }
        }
    }
    if delays.is_empty() {
        return Err(ValidationError::InvalidParameter(
            "no valid reported claims before valuation_time",
        ));
    }
    let mean_delay = delays.iter().sum::<f64>() / delays.len() as f64;
    let mut rho = if mean_delay > 1e-9 {
        1.0 / mean_delay
    } else {
        10.0
    };
    // Damped Newton-Raphson for right-truncated exponential MLE.
    for _ in 0..40 {
        let mut grad = 0.0;
        let mut hess = 0.0;
        for (&r, &cap) in delays.iter().zip(truncs.iter()) {
            let x = (rho * cap).min(500.0);
            let ex = (-x).exp();
            let one_minus_ex = (1.0 - ex).max(1e-15);
            grad += 1.0 / rho - r - (cap * ex) / one_minus_ex;
            hess += -1.0 / (rho * rho) + (cap * cap * ex) / (one_minus_ex * one_minus_ex);
        }
        if hess.abs() < 1e-15 {
            break;
        }
        let step = (grad / hess).clamp(-0.5 * rho, 0.5 * rho);
        rho = (rho - step).max(1e-6);
        if step.abs() < 1e-10 * rho {
            break;
        }
    }

    // 2. Truncation-corrected NHPP occurrence intensities per period k.
    let mut period_counts = vec![0.0; n_periods];
    for c in claims {
        if c.report_time <= valuation_time && c.occurrence_time >= 0.0 {
            let k = (c.occurrence_time.floor() as usize).min(n_periods - 1);
            period_counts[k] += 1.0;
        }
    }
    let mut occurrence_intensities = vec![0.0; n_periods];
    for k in 0..n_periods {
        let a = k as f64;
        let b = ((k + 1) as f64).min(valuation_time);
        if b <= a {
            continue;
        }
        // Observed fraction integral: int_a^b (1 - exp(-rho * (tau - t))) dt
        let unreported_int =
            ((-rho * (valuation_time - b)).exp() - (-rho * (valuation_time - a)).exp()) / rho;
        let obs_exposure = ((b - a) - unreported_int).max(1e-9);
        occurrence_intensities[k] = period_counts[k] / obs_exposure;
    }

    // 3. Multi-state settlement and interim payment hazards & lognormal severities.
    let mut total_exposure_open = 0.0;
    let mut n_settled = 0usize;
    let mut n_settled_with_pay = 0usize;
    let mut interim_logs = Vec::new();
    let mut close_logs = Vec::new();

    for c in claims {
        if c.report_time > valuation_time {
            continue;
        }
        let end_time = match c.settlement_time {
            Some(ts) if ts <= valuation_time => {
                n_settled += 1;
                ts
            }
            _ => valuation_time,
        };
        total_exposure_open += (end_time - c.report_time).max(0.0);

        let settled_at = c.settlement_time.filter(|&ts| ts <= valuation_time);
        let mut had_close_pay = false;
        for p in &c.payments {
            if p.time > valuation_time || p.amount <= 0.0 {
                continue;
            }
            if let Some(ts) = settled_at {
                if (p.time - ts).abs() <= 1e-6 {
                    close_logs.push(p.amount.ln());
                    had_close_pay = true;
                    continue;
                }
            }
            interim_logs.push(p.amount.ln());
        }
        if had_close_pay {
            n_settled_with_pay += 1;
        }
    }

    if total_exposure_open <= 0.0 || n_settled == 0 {
        return Err(ValidationError::InvalidParameter(
            "insufficient settled claim exposure to estimate settlement hazard",
        ));
    }
    let settlement_hazard = n_settled as f64 / total_exposure_open;
    let interim_payment_hazard = interim_logs.len() as f64 / total_exposure_open;
    let settlement_payment_prob = n_settled_with_pay as f64 / n_settled as f64;

    let (interim_severity_mu, interim_severity_sigma) = fit_lognormal(&interim_logs);
    let (settlement_severity_mu, settlement_severity_sigma) = fit_lognormal(&close_logs);

    let model = MicroReservingModel {
        occurrence_intensities,
        reporting_delay_rate: rho,
        interim_payment_hazard,
        settlement_hazard,
        interim_severity_mu,
        interim_severity_sigma,
        settlement_payment_prob,
        settlement_severity_mu,
        settlement_severity_sigma,
    };
    model.validate()?;
    Ok(model)
}

fn fit_lognormal(log_amounts: &[f64]) -> (f64, f64) {
    if log_amounts.is_empty() {
        return (0.0, 0.0);
    }
    let m = log_amounts.len() as f64;
    let mu = log_amounts.iter().sum::<f64>() / m;
    if log_amounts.len() < 2 {
        return (mu, 0.0);
    }
    let var = log_amounts.iter().map(|&v| (v - mu).powi(2)).sum::<f64>() / (m - 1.0);
    (mu, var.max(0.0).sqrt())
}

fn simulate_single_open_claim(model: &MicroReservingModel, rng: &mut Rng) -> f64 {
    let total_rate = model.interim_payment_hazard + model.settlement_hazard;
    let p_interim = model.interim_payment_hazard / total_rate;
    let mut paid = 0.0;
    // Competing exponential hazards: geometrically distributed number of interim
    // payments before settlement (memoryless property).
    loop {
        if rng.next_open01() < p_interim {
            let z = rng.next_standard_normal();
            paid += (model.interim_severity_mu + model.interim_severity_sigma * z).exp();
        } else {
            if model.settlement_payment_prob > 0.0
                && rng.next_open01() < model.settlement_payment_prob
            {
                let z = rng.next_standard_normal();
                paid += (model.settlement_severity_mu + model.settlement_severity_sigma * z).exp();
            }
            break;
        }
    }
    paid
}

/// Simulate RBNS and IBNR reserve distributions from a fitted [`MicroReservingModel`]
/// and the observed claim log as of `valuation_time`.
pub fn simulate_reserves(
    claims: &[ClaimRecord],
    model: &MicroReservingModel,
    valuation_time: f64,
    n_simulations: usize,
    seed: u64,
) -> Result<MicroReservingResult, ValidationError> {
    model.validate()?;
    if n_simulations == 0 || !valuation_time.is_finite() || valuation_time <= 0.0 {
        return Err(ValidationError::InvalidParameter(
            "n_simulations must be >= 1 and valuation_time > 0",
        ));
    }

    let n_open_claims = claims
        .iter()
        .filter(|c| {
            c.report_time <= valuation_time
                && match c.settlement_time {
                    Some(ts) => ts > valuation_time,
                    None => true,
                }
        })
        .count();

    let expected_ibnr = model.expected_ibnr_count(valuation_time);
    let mut rng = Rng::from_seed(seed);

    let mut rbns_draws = Vec::with_capacity(n_simulations);
    let mut ibnr_draws = Vec::with_capacity(n_simulations);
    let mut total_draws = Vec::with_capacity(n_simulations);
    let mut ibnr_count_sum = 0.0;

    for _ in 0..n_simulations {
        let mut rbns = 0.0;
        for _ in 0..n_open_claims {
            rbns += simulate_single_open_claim(model, &mut rng);
        }

        let n_ibnr = rng.sample_poisson(expected_ibnr).round() as usize;
        ibnr_count_sum += n_ibnr as f64;
        let mut ibnr = 0.0;
        for _ in 0..n_ibnr {
            ibnr += simulate_single_open_claim(model, &mut rng);
        }

        rbns_draws.push(rbns);
        ibnr_draws.push(ibnr);
        total_draws.push(rbns + ibnr);
    }

    let (mean_rbns, sd_rbns) = mean_sd(&rbns_draws);
    let (mean_ibnr, sd_ibnr) = mean_sd(&ibnr_draws);
    let (mean_total, sd_total) = mean_sd(&total_draws);

    let total_quantiles = [
        empirical_quantile(&total_draws, 0.50),
        empirical_quantile(&total_draws, 0.75),
        empirical_quantile(&total_draws, 0.90),
        empirical_quantile(&total_draws, 0.95),
        empirical_quantile(&total_draws, 0.99),
    ];

    Ok(MicroReservingResult {
        n_simulations,
        n_open_claims,
        mean_ibnr_count: ibnr_count_sum / n_simulations as f64,
        mean_rbns,
        mean_ibnr,
        mean_total,
        sd_rbns,
        sd_ibnr,
        sd_total,
        total_quantiles,
        total_draws,
    })
}

/// Generate a synthetic individual-claim event log from a known [`MicroReservingModel`]
/// up to `valuation_time` (for validation, backtesting, and benchmark tests).
///
/// Returns `(observed_claims, true_unpaid_rbns, true_unpaid_ibnr)`.
pub fn simulate_synthetic_portfolio(
    model: &MicroReservingModel,
    valuation_time: f64,
    seed: u64,
) -> Result<(Vec<ClaimRecord>, f64, f64), ValidationError> {
    model.validate()?;
    let mut rng = Rng::from_seed(seed);
    let mut observed = Vec::new();
    let mut true_rbns = 0.0;
    let mut true_ibnr = 0.0;
    let mut next_id = 1u64;

    let total_rate = model.interim_payment_hazard + model.settlement_hazard;
    let p_interim = model.interim_payment_hazard / total_rate;

    for (k, &lam) in model.occurrence_intensities.iter().enumerate() {
        let a = k as f64;
        let b = ((k + 1) as f64).min(valuation_time);
        if b <= a {
            continue;
        }
        let n_occ = rng.sample_poisson(lam * (b - a)).round() as usize;
        for _ in 0..n_occ {
            let t_occ = a + (b - a) * rng.next_open01();
            let rep_delay = -rng.next_open01().ln() / model.reporting_delay_rate;
            let t_rep = t_occ + rep_delay;

            let mut cur_t = t_rep;
            let mut obs_payments = Vec::new();
            let mut future_paid = 0.0;
            let t_settle;

            loop {
                let dt = -rng.next_open01().ln() / total_rate;
                cur_t += dt;
                if rng.next_open01() < p_interim {
                    let amt = (model.interim_severity_mu
                        + model.interim_severity_sigma * rng.next_standard_normal())
                    .exp();
                    if cur_t <= valuation_time {
                        obs_payments.push(ClaimPayment {
                            time: cur_t,
                            amount: amt,
                        });
                    } else {
                        future_paid += amt;
                    }
                } else {
                    if model.settlement_payment_prob > 0.0
                        && rng.next_open01() < model.settlement_payment_prob
                    {
                        let amt = (model.settlement_severity_mu
                            + model.settlement_severity_sigma * rng.next_standard_normal())
                        .exp();
                        if cur_t <= valuation_time {
                            obs_payments.push(ClaimPayment {
                                time: cur_t,
                                amount: amt,
                            });
                        } else {
                            future_paid += amt;
                        }
                    }
                    t_settle = cur_t;
                    break;
                }
            }

            if t_rep <= valuation_time {
                let settlement_time = if t_settle <= valuation_time {
                    Some(t_settle)
                } else {
                    true_rbns += future_paid;
                    None
                };
                observed.push(ClaimRecord {
                    claim_id: next_id,
                    occurrence_time: t_occ,
                    report_time: t_rep,
                    settlement_time,
                    payments: obs_payments,
                });
                next_id += 1;
            } else {
                true_ibnr += future_paid;
            }
        }
    }

    Ok((observed, true_rbns, true_ibnr))
}

fn mean_sd(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let m = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / m;
    if xs.len() < 2 {
        return (mean, 0.0);
    }
    let var = xs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (m - 1.0);
    (mean, var.max(0.0).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_event_log_parameter_recovery_and_reserve_unbiasedness() {
        let true_model = MicroReservingModel {
            occurrence_intensities: vec![250.0, 250.0, 250.0, 250.0, 250.0],
            reporting_delay_rate: 2.0,
            interim_payment_hazard: 0.8,
            settlement_hazard: 1.2,
            interim_severity_mu: 7.0,
            interim_severity_sigma: 0.4,
            settlement_payment_prob: 0.9,
            settlement_severity_mu: 8.0,
            settlement_severity_sigma: 0.5,
        };
        let tau = 5.0;
        let (claims, true_rbns, true_ibnr) =
            simulate_synthetic_portfolio(&true_model, tau, 12345).unwrap();

        let fitted = fit_from_claims(&claims, tau, 5).unwrap();
        assert!((fitted.reporting_delay_rate - true_model.reporting_delay_rate).abs() < 0.35);
        assert!((fitted.settlement_hazard - true_model.settlement_hazard).abs() < 0.20);
        assert!((fitted.interim_payment_hazard - true_model.interim_payment_hazard).abs() < 0.20);

        let sim = simulate_reserves(&claims, &fitted, tau, 2_000, 99).unwrap();
        let true_total = true_rbns + true_ibnr;
        assert!(
            (sim.mean_total - true_total).abs() / true_total < 0.15,
            "sim.mean_total={} vs true_total={}",
            sim.mean_total,
            true_total
        );
        // Quantiles must be monotonically non-decreasing.
        for w in sim.total_quantiles.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }
}
