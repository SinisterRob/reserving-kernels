//! Hertig (1985) exact closed-form log-normal link-ratio reserving model.
//!
//! Implements:
//!
//! * J. Hertig (1985), *A Statistical Approach to IBNR-Reserves in Marine
//!   Reinsurance*, ASTIN Bulletin, Vol. 15, No. 2, pp. 171–183.
//!
//! Models individual log link ratios as independent Gaussian variables:
//!
//! $$\ln\left(\frac{C_{i,k+1}}{C_{i,k}}\right) \sim \mathcal{N}(\mu_k, \sigma_k^2)$$
//!
//! Because the future cumulative development factor for origin $i$ is a product
//! of independent log-normal link ratios, $\ln(U_i / C_{i, n-1-i})$ is **exact
//! Gaussian** with predictive mean $M_i = \sum_{k=n-1-i}^{n-2} \hat{\mu}_k$ and
//! predictive variance:
//!
//! $$V_i = \sum_{k=n-1-i}^{n-2} \hat{\sigma}_k^2 \left(1 + \frac{\mathbb{I}_{\text{param}}}{n_k}\right)$$
//!
//! yielding **exact closed-form** mean, variance, skewness, cross-origin
//! covariance, and skewed quantiles without Monte Carlo simulation.

use crate::dependence::inv_normal_cdf;
use crate::triangle::latest_diagonal;
use crate::types::TriangleView;
use crate::validate::{
    MonotonicityPolicy, ValidationError, validate_monotone_cumulative, validate_triangle,
};

/// Options for Hertig (1985) log-normal link-ratio reserving.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HertigOptions {
    /// Whether to include parameter estimation variance $\hat{\sigma}_k^2 / n_k$
    /// in predictive variance and cross-origin covariance (default `true`).
    pub include_parameter_error: bool,
    /// Monotonicity policy (`AllowNegativeIncrements` permits link ratios `< 1.0`,
    /// provided cumulative cells remain strictly positive).
    pub policy: MonotonicityPolicy,
}

impl Default for HertigOptions {
    fn default() -> Self {
        Self {
            include_parameter_error: true,
            policy: MonotonicityPolicy::AllowNegativeIncrements,
        }
    }
}

/// Full Hertig (1985) closed-form log-normal reserving result.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HertigResult {
    /// Estimated log-link means $\hat{\mu}_k$ per link $k = 0..n-2$, length $n - 1$.
    pub mu: Vec<f64>,
    /// Estimated log-link variances $\hat{\sigma}_k^2$ per link $k = 0..n-2$, length $n - 1$.
    pub sigma_sq: Vec<f64>,
    /// Number of observed link ratios $n_k$ per link, length $n - 1$.
    pub n_obs: Vec<usize>,
    /// Latest observed cumulative loss per origin, length $n$.
    pub latest: Vec<f64>,
    /// Predictive log-development mean $M_i$ per origin, length $n$.
    pub log_mean: Vec<f64>,
    /// Predictive log-development variance $V_i$ per origin, length $n$.
    pub log_variance: Vec<f64>,
    /// Exact analytical expected ultimate $E\[U_i\]$ per origin, length $n$.
    pub ultimate: Vec<f64>,
    /// Exact analytical expected reserve $E\[R_i\] = E\[U_i\] - C_{i, n-1-i}$ per origin, length $n$.
    pub reserve: Vec<f64>,
    /// Exact analytical reserve variance $\mathrm{Var}(R_i)$ per origin, length $n$.
    pub variance: Vec<f64>,
    /// Exact analytical reserve standard error per origin, length $n$.
    pub standard_error: Vec<f64>,
    /// Exact analytical reserve skewness per origin, length $n$.
    pub skewness: Vec<f64>,
    /// Total expected reserve across all origins.
    pub total_reserve: f64,
    /// Exact total reserve variance (including cross-origin parameter covariance).
    pub total_variance: f64,
    /// Exact total reserve standard error.
    pub total_standard_error: f64,
    /// Total coefficient of variation.
    pub total_cv: f64,
}

impl HertigResult {
    /// Exact closed-form reserve quantile at probability `p` in `(0, 1)` for origin `origin`.
    pub fn origin_reserve_quantile(&self, origin: usize, p: f64) -> Result<f64, ValidationError> {
        if origin >= self.latest.len() {
            return Err(ValidationError::ShapeMismatch);
        }
        if !p.is_finite() || p <= 0.0 || p >= 1.0 {
            return Err(ValidationError::InvalidParameter(
                "quantile probability must be strictly in (0, 1)",
            ));
        }
        let z = inv_normal_cdf(p);
        let ult_q = self.latest[origin]
            * (self.log_mean[origin] + z * self.log_variance[origin].sqrt()).exp();
        Ok(ult_q - self.latest[origin])
    }

    /// Fenton–Wilkinson moment-matched log-normal approximation to the aggregate
    /// total reserve quantile at probability `p` in `(0, 1)`.
    pub fn aggregate_reserve_quantile(&self, p: f64) -> Result<f64, ValidationError> {
        if !p.is_finite() || p <= 0.0 || p >= 1.0 {
            return Err(ValidationError::InvalidParameter(
                "quantile probability must be strictly in (0, 1)",
            ));
        }
        let mean_u: f64 = self.ultimate.iter().sum();
        let sum_lat: f64 = self.latest.iter().sum();
        if mean_u <= 0.0 {
            return Ok(0.0);
        }
        let cv2 = self.total_variance / (mean_u * mean_u);
        let sigma2 = (1.0 + cv2).ln();
        let mu = mean_u.ln() - 0.5 * sigma2;
        let z = inv_normal_cdf(p);
        Ok((mu + z * sigma2.sqrt()).exp() - sum_lat)
    }
}

impl std::fmt::Display for HertigResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Hertig(reserve={:.2}, se={:.2}, cv={:.2}%)",
            self.total_reserve,
            self.total_standard_error,
            self.total_cv * 100.0
        )
    }
}

/// Fit Hertig (1985) closed-form log-normal link-ratio model to a square cumulative triangle.
pub fn analyze(
    t: &TriangleView<'_>,
    options: HertigOptions,
) -> Result<HertigResult, ValidationError> {
    validate_triangle(t)?;
    if matches!(options.policy, MonotonicityPolicy::Strict) {
        validate_monotone_cumulative(t)?;
    }

    let n = t.n_origins;
    if n < 2 {
        return Err(ValidationError::InvalidParameter(
            "triangle must be at least 2x2",
        ));
    }
    let n_links = n - 1;

    let mut mu = vec![0.0; n_links];
    let mut raw_sigma_sq: Vec<Option<f64>> = vec![None; n_links];
    let mut n_obs = vec![0usize; n_links];

    for k in 0..n_links {
        let nk = n - k - 1;
        let mut logs = Vec::with_capacity(nk);
        for i in 0..nk {
            let c0 = t.get(i, k);
            let c1 = t.get(i, k + 1);
            if c0 <= 0.0 || c1 <= 0.0 {
                return Err(ValidationError::InvalidParameter(
                    "Hertig log-normal model requires strictly positive cumulative cells",
                ));
            }
            logs.push((c1 / c0).ln());
        }
        let m = logs.len();
        n_obs[k] = m;
        let mean = logs.iter().sum::<f64>() / m as f64;
        mu[k] = mean;
        if m >= 2 {
            let ss: f64 = logs.iter().map(|&v| (v - mean).powi(2)).sum();
            raw_sigma_sq[k] = Some(ss / (m as f64 - 1.0));
        }
    }

    // Extrapolate last-link variance via Mack's min-rule.
    let mut sigma_sq = vec![0.0; n_links];
    for k in 0..n_links {
        if let Some(v) = raw_sigma_sq[k] {
            sigma_sq[k] = v.max(0.0);
        } else if k >= 2 {
            let a = sigma_sq[k - 1];
            let b = sigma_sq[k - 2];
            sigma_sq[k] = if a > 0.0 && b > 0.0 {
                (a * a / b).min(a).min(b).max(0.0)
            } else {
                0.0
            };
        } else if k == 1 {
            sigma_sq[k] = sigma_sq[0].max(0.0);
        } else {
            sigma_sq[k] = 0.0;
        }
    }

    let latest = latest_diagonal(t);
    let mut log_mean = vec![0.0; n];
    let mut log_variance = vec![0.0; n];
    let mut ultimate = vec![0.0; n];
    let mut reserve = vec![0.0; n];
    let mut variance = vec![0.0; n];
    let mut standard_error = vec![0.0; n];
    let mut skewness = vec![0.0; n];

    let param_mult = if options.include_parameter_error {
        1.0
    } else {
        0.0
    };

    for i in 0..n {
        let start_k = n - 1 - i;
        let mut m_i = 0.0;
        let mut v_i = 0.0;
        for k in start_k..n_links {
            m_i += mu[k];
            v_i += sigma_sq[k] * (1.0 + param_mult / n_obs[k] as f64);
        }
        log_mean[i] = m_i;
        log_variance[i] = v_i;
        let u_i = latest[i] * (m_i + 0.5 * v_i).exp();
        let var_i = u_i * u_i * v_i.exp_m1();
        let skew_i = if v_i > 0.0 {
            (v_i.exp() + 2.0) * v_i.exp_m1().sqrt()
        } else {
            0.0
        };
        ultimate[i] = u_i;
        reserve[i] = u_i - latest[i];
        variance[i] = var_i.max(0.0);
        standard_error[i] = variance[i].sqrt();
        skewness[i] = skew_i;
    }

    // Exact aggregate variance including bivariate log-normal cross-origin covariance.
    let mut total_variance: f64 = variance.iter().sum();
    if options.include_parameter_error {
        for i in 1..n {
            let start_k = n - 1 - i;
            let mut shared_cov = 0.0;
            for k in start_k..n_links {
                shared_cov += sigma_sq[k] / n_obs[k] as f64;
            }
            for j in (i + 1)..n {
                total_variance += 2.0 * ultimate[i] * ultimate[j] * shared_cov.exp_m1();
            }
        }
    }
    total_variance = total_variance.max(0.0);
    let total_standard_error = total_variance.sqrt();
    let total_reserve: f64 = reserve.iter().sum();
    let total_cv = if total_reserve.abs() > 0.0 {
        total_standard_error / total_reserve.abs()
    } else {
        0.0
    };

    Ok(HertigResult {
        mu,
        sigma_sq,
        n_obs,
        latest,
        log_mean,
        log_variance,
        ultimate,
        reserve,
        variance,
        standard_error,
        skewness,
        total_reserve,
        total_variance,
        total_standard_error,
        total_cv,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triangle;
    use crate::bootstrap_odp::DeterministicRng;

    #[test]
    fn analytical_moments_and_quantiles_match_monte_carlo_lognormal_links() {
        let tri = Triangle::try_from_upper_rows(&[
            &[100.0, 165.0, 200.0, 212.0, 218.0],
            &[120.0, 190.0, 238.0, 250.0],
            &[135.0, 222.0, 268.0],
            &[150.0, 240.0],
            &[160.0],
        ])
        .unwrap();

        // Process-only mode for direct MC verification of exact moments & quantiles.
        let opts = HertigOptions {
            include_parameter_error: false,
            policy: MonotonicityPolicy::Strict,
        };
        let h = analyze(&tri.view(), opts).unwrap();

        let mut rng = DeterministicRng::from_seed(42);
        let n_sim = 40_000usize;
        let mut samples = Vec::with_capacity(n_sim);
        let origin = 4; // youngest origin develops through all 4 links
        for _ in 0..n_sim {
            let mut log_cdf = 0.0;
            for k in 0..4 {
                log_cdf += h.mu[k] + h.sigma_sq[k].sqrt() * rng.next_standard_normal();
            }
            samples.push(h.latest[origin] * (log_cdf.exp() - 1.0));
        }

        let mc_mean: f64 = samples.iter().sum::<f64>() / n_sim as f64;
        let mc_var: f64 =
            samples.iter().map(|&x| (x - mc_mean).powi(2)).sum::<f64>() / (n_sim as f64 - 1.0);
        samples.sort_by(|a, b| a.total_cmp(b));
        let mc_q90 = samples[(0.90 * n_sim as f64) as usize];

        assert!((mc_mean - h.reserve[origin]).abs() / h.reserve[origin] < 0.02);
        assert!((mc_var.sqrt() - h.standard_error[origin]).abs() / h.standard_error[origin] < 0.03);
        let exact_q90 = h.origin_reserve_quantile(origin, 0.90).unwrap();
        assert!((mc_q90 - exact_q90).abs() / exact_q90 < 0.02);
        assert!(h.skewness[origin] > 0.0);
    }
}
