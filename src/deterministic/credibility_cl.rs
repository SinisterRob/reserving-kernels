//! Bühlmann–Straub credibility for the Chain-Ladder reserving method.
//!
//! Implements:
//!
//! * A. Gisler & M. V. Wüthrich (2008), *Credibility for the Chain Ladder
//!   Reserving Method*, ASTIN Bulletin, Vol. 38, No. 2, pp. 565–600.
//!
//! Shrinks empirical volume-weighted development factors $\hat{f}_k^{\text{CL}}$
//! toward an external or collective prior $f_k^{(0)}$ using Bühlmann–Straub
//! credibility weights:
//!
//! $$\alpha_k = \frac{S_k}{S_k + \kappa_k}, \qquad \kappa_k = \frac{\sigma_k^2}{\tau_k^2}, \qquad
//!   \hat{f}_k^{\text{cred}} = \alpha_k \hat{f}_k^{\text{CL}} + (1 - \alpha_k) f_k^{(0)}$$
//!
//! where $S_k = \sum_{i=0}^{n-k-2} C_{i,k}$ is the column volume at link $k$,
//! $\sigma_k^2$ is the within-risk Mack variance parameter, and $\tau_k^2$ is
//! the between-risk structural variance of the link factor across the
//! collective.
//!
//! Exact analytical conditional MSEP (Gisler & Wüthrich 2008, §5):
//!
//! $$\mathrm{MSEP}(\hat{U}_i^{\text{cred}}) = (\hat{U}_i^{\text{cred}})^2
//!   \sum_{k=n-1-i}^{n-2} \frac{1}{(\hat{f}_k^{\text{cred}})^2}
//!   \left(\frac{\sigma_k^2}{\hat{C}_{i,k}^{\text{cred}}} + \alpha_k \frac{\sigma_k^2}{S_k}\right)$$
//!
//! Limits:
//! * $\tau_k^2 \to \infty$ ($\alpha_k \to 1$): reduces identically to Mack (1993).
//! * $\tau_k^2 \to 0$ ($\alpha_k \to 0$): reduces to the known external prior
//!   $f_k^{(0)}$ with zero parameter estimation error (pure process variance).

use crate::preprocessing::triangle::latest_diagonal;
use crate::stochastic::mack;
use crate::types::TriangleView;
use crate::validate::{
    MonotonicityPolicy, ValidationError, validate_monotone_cumulative, validate_tail,
    validate_triangle,
};

/// Prior specification for Gisler & Wüthrich (2008) credibility chain ladder.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CredibilityClPrior {
    /// External / collective prior LDFs $f_k^{(0)}$, length $n - 1$.
    pub prior_ldf: Vec<f64>,
    /// Structural between-risk variances $\tau_k^2 \ge 0$, length $n - 1$.
    /// Use `0.0` for full trust in the prior ($\alpha_k = 0$) and `f64::INFINITY`
    /// for pure empirical Mack CL ($\alpha_k = 1$).
    pub tau_sq: Vec<f64>,
    /// Optional override for within-risk variance parameters $\sigma_k^2$, length $n - 1$.
    /// If `None`, estimated from the target triangle via Mack (1993).
    pub sigma_sq_override: Option<Vec<f64>>,
    /// Multiplicative tail factor $\ge 1.0$.
    pub tail: f64,
    /// Monotonicity policy.
    pub policy: MonotonicityPolicy,
}

/// Collective Bühlmann–Straub empirical Bayes parameters estimated across peer triangles.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CollectiveCredibilityParams {
    /// Collective prior LDFs $f_k^{(0)}$, length $n - 1$.
    pub prior_ldf: Vec<f64>,
    /// Collective within-risk variance $\hat{\sigma}_k^2$, length $n - 1$.
    pub sigma_sq: Vec<f64>,
    /// Collective between-risk structural variance $\hat{\tau}_k^2$, length $n - 1$.
    pub tau_sq: Vec<f64>,
}

/// Full Gisler & Wüthrich (2008) credibility chain-ladder result.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CredibilityClResult {
    /// Empirical volume-weighted LDFs $\hat{f}_k^{\text{CL}}$, length $n - 1$.
    pub empirical_ldf: Vec<f64>,
    /// External / collective prior LDFs $f_k^{(0)}$, length $n - 1$.
    pub prior_ldf: Vec<f64>,
    /// Bühlmann–Straub credibility weights $\alpha_k \in [0, 1]$, length $n - 1$.
    pub credibility_weight: Vec<f64>,
    /// Credibility-shrunk LDFs $\hat{f}_k^{\text{cred}}$, length $n - 1$.
    pub credibility_ldf: Vec<f64>,
    /// Within-risk variance parameters $\sigma_k^2$, length $n - 1$.
    pub sigma_sq: Vec<f64>,
    /// Between-risk variance parameters $\tau_k^2$, length $n - 1$.
    pub tau_sq: Vec<f64>,
    /// Projected ultimates per origin, length $n$.
    pub ultimate: Vec<f64>,
    /// Projected reserves per origin, length $n$.
    pub reserve: Vec<f64>,
    /// Analytical process variance per origin, length $n$.
    pub process_variance: Vec<f64>,
    /// Analytical credibility parameter estimation variance per origin, length $n$.
    pub parameter_variance: Vec<f64>,
    /// Standard error per origin, length $n$.
    pub standard_error: Vec<f64>,
    /// Total reserve across origins.
    pub total_reserve: f64,
    /// Total analytical MSEP across origins (including cross-origin covariance).
    pub total_mse: f64,
    /// Total analytical standard error.
    pub total_standard_error: f64,
    /// Total coefficient of variation.
    pub total_cv: f64,
}

impl std::fmt::Display for CredibilityClResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CredibilityCL(reserve={:.2}, se={:.2}, cv={:.2}%)",
            self.total_reserve,
            self.total_standard_error,
            self.total_cv * 100.0
        )
    }
}

/// Compute Gisler & Wüthrich (2008) credibility chain-ladder reserves and
/// exact analytical MSEP.
pub fn analyze(
    t: &TriangleView<'_>,
    prior: &CredibilityClPrior,
) -> Result<CredibilityClResult, ValidationError> {
    validate_triangle(t)?;
    validate_tail(prior.tail)?;
    if matches!(prior.policy, MonotonicityPolicy::Strict) {
        validate_monotone_cumulative(t)?;
    }

    let n = t.n_origins;
    let n_links = n.saturating_sub(1);
    if prior.prior_ldf.len() != n_links || prior.tau_sq.len() != n_links {
        return Err(ValidationError::ShapeMismatch);
    }
    for &f0 in &prior.prior_ldf {
        if !f0.is_finite() || f0 <= 0.0 {
            return Err(ValidationError::InvalidParameter(
                "prior_ldf entries must be finite and positive",
            ));
        }
    }
    for &t2 in &prior.tau_sq {
        if t2.is_nan() || t2 < 0.0 {
            return Err(ValidationError::InvalidParameter(
                "tau_sq entries must be non-negative (or +inf)",
            ));
        }
    }

    let base_mack = mack::analyze_with_policy(t, prior.tail, prior.policy)?;
    let sigma_sq = if let Some(ref ov) = prior.sigma_sq_override {
        if ov.len() != n_links {
            return Err(ValidationError::ShapeMismatch);
        }
        for &s2 in ov {
            if !s2.is_finite() || s2 < 0.0 {
                return Err(ValidationError::InvalidParameter(
                    "sigma_sq_override entries must be finite and non-negative",
                ));
            }
        }
        ov.clone()
    } else {
        base_mack.sigma_sq.clone()
    };

    let mut col_sums = vec![0.0; n_links];
    for k in 0..n_links {
        let nk = n - k - 1;
        let mut s = 0.0;
        for i in 0..nk {
            s += t.get(i, k);
        }
        col_sums[k] = s;
    }

    let mut credibility_weight = Vec::with_capacity(n_links);
    let mut credibility_ldf = Vec::with_capacity(n_links);
    let mut est_var_ldf = Vec::with_capacity(n_links);

    for k in 0..n_links {
        let sk = col_sums[k];
        let s2 = sigma_sq[k];
        let t2 = prior.tau_sq[k];
        let f_emp = base_mack.ldf[k];
        let f0 = prior.prior_ldf[k];

        let (alpha, var_f) = if t2 == 0.0 {
            (0.0, 0.0)
        } else if t2.is_infinite() {
            let vf = if sk > 0.0 { s2 / sk } else { 0.0 };
            (1.0, vf)
        } else if s2 == 0.0 {
            (1.0, 0.0)
        } else if sk > 0.0 {
            let kappa = s2 / t2;
            let a = sk / (sk + kappa);
            (a, a * s2 / sk)
        } else {
            (0.0, t2)
        };

        credibility_weight.push(alpha);
        credibility_ldf.push(alpha * f_emp + (1.0 - alpha) * f0);
        est_var_ldf.push(var_f);
    }

    // Project cumulatives using credibility LDFs.
    let latest = latest_diagonal(t);
    let mut proj = vec![0.0; n * n];
    let mut ultimate = vec![0.0; n];
    let mut reserve = vec![0.0; n];

    for i in 0..n {
        let last_obs = n - 1 - i;
        for k in 0..=last_obs {
            proj[i * n + k] = t.get(i, k);
        }
        let mut c = latest[i];
        for k in last_obs..n_links {
            c *= credibility_ldf[k];
            proj[i * n + k + 1] = c;
        }
        ultimate[i] = c * prior.tail;
        reserve[i] = ultimate[i] - latest[i];
    }

    // Analytical MSEP per origin and aggregate (Gisler & Wüthrich 2008 §5).
    let mut process_variance = vec![0.0; n];
    let mut parameter_variance = vec![0.0; n];
    let mut standard_error = vec![0.0; n];

    for i in 1..n {
        let start_k = n - 1 - i;
        let mut proc_sum = 0.0;
        let mut param_sum = 0.0;
        for k in start_k..n_links {
            let fk = credibility_ldf[k];
            let cik = proj[i * n + k];
            if fk > 0.0 {
                let inv_f2 = 1.0 / (fk * fk);
                if cik > 0.0 {
                    proc_sum += inv_f2 * (sigma_sq[k] / cik);
                }
                param_sum += inv_f2 * est_var_ldf[k];
            }
        }
        let u2 = ultimate[i] * ultimate[i];
        process_variance[i] = u2 * proc_sum;
        parameter_variance[i] = u2 * param_sum;
        standard_error[i] = (process_variance[i] + parameter_variance[i])
            .max(0.0)
            .sqrt();
    }

    let mut total_mse: f64 = (0..n)
        .map(|i| process_variance[i] + parameter_variance[i])
        .sum();
    for i in 1..n {
        let start_k = n - 1 - i;
        let mut shared = 0.0;
        for k in start_k..n_links {
            let fk = credibility_ldf[k];
            if fk > 0.0 {
                shared += est_var_ldf[k] / (fk * fk);
            }
        }
        for j in (i + 1)..n {
            total_mse += 2.0 * ultimate[i] * ultimate[j] * shared;
        }
    }
    total_mse = total_mse.max(0.0);
    let total_standard_error = total_mse.sqrt();
    let total_reserve: f64 = reserve.iter().sum();
    let total_cv = if total_reserve == 0.0 {
        0.0
    } else {
        total_standard_error / total_reserve
    };

    Ok(CredibilityClResult {
        empirical_ldf: base_mack.ldf,
        prior_ldf: prior.prior_ldf.clone(),
        credibility_weight,
        credibility_ldf,
        sigma_sq,
        tau_sq: prior.tau_sq.clone(),
        ultimate,
        reserve,
        process_variance,
        parameter_variance,
        standard_error,
        total_reserve,
        total_mse,
        total_standard_error,
        total_cv,
    })
}

/// Estimate collective Bühlmann–Straub empirical Bayes parameters
/// $(f_k^{(0)}, \hat{\sigma}_k^2, \hat{\tau}_k^2)$ across $M \ge 2$ aligned
/// peer triangles (Gisler & Wüthrich 2008, §3.2).
pub fn estimate_collective_parameters(
    triangles: &[TriangleView<'_>],
    policy: MonotonicityPolicy,
) -> Result<CollectiveCredibilityParams, ValidationError> {
    if triangles.len() < 2 {
        return Err(ValidationError::InvalidParameter(
            "at least 2 triangles are required to estimate collective credibility parameters",
        ));
    }
    let n = triangles[0].n_origins;
    for t in triangles {
        validate_triangle(t)?;
        if t.n_origins != n || t.n_devs != n {
            return Err(ValidationError::ShapeMismatch);
        }
        if matches!(policy, MonotonicityPolicy::Strict) {
            validate_monotone_cumulative(t)?;
        }
    }

    let m = triangles.len();
    let n_links = n.saturating_sub(1);
    let mut macks = Vec::with_capacity(m);
    for t in triangles {
        macks.push(mack::analyze_with_policy(t, 1.0, policy)?);
    }

    let mut prior_ldf = vec![0.0; n_links];
    let mut sigma_sq = vec![0.0; n_links];
    let mut tau_sq = vec![0.0; n_links];

    for k in 0..n_links {
        let nk = n - k - 1;
        let mut vols = vec![0.0; m];
        for r in 0..m {
            let mut s = 0.0;
            for i in 0..nk {
                s += triangles[r].get(i, k);
            }
            vols[r] = s;
        }
        let total_vol: f64 = vols.iter().sum();
        if total_vol <= 0.0 {
            return Err(ValidationError::DivideByZero("collective volume"));
        }

        // Volume-weighted collective LDF f_k^{(0)}.
        let f0: f64 = (0..m).map(|r| vols[r] * macks[r].ldf[k]).sum::<f64>() / total_vol;
        prior_ldf[k] = f0;

        // Average within-risk variance \hat{\sigma}_k^2.
        let s2: f64 = macks.iter().map(|mk| mk.sigma_sq[k]).sum::<f64>() / m as f64;
        sigma_sq[k] = s2;

        // Bühlmann–Straub between-risk structural variance estimator \hat{\tau}_k^2.
        // Sum v_r * (1 - v_r / total_vol) to avoid catastrophic cancellation when one peer dominates.
        let c_star: f64 = vols.iter().map(|&v| v * (1.0 - v / total_vol)).sum();
        let weighted_ss: f64 = (0..m)
            .map(|r| vols[r] * (macks[r].ldf[k] - f0).powi(2))
            .sum();
        let t2 = if c_star > 0.0 {
            ((weighted_ss - (m as f64 - 1.0) * s2) / c_star).max(0.0)
        } else {
            0.0
        };
        tau_sq[k] = t2;
    }

    Ok(CollectiveCredibilityParams {
        prior_ldf,
        sigma_sq,
        tau_sq,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triangle;

    #[test]
    fn limits_recover_mack_and_pure_prior() {
        let tri = Triangle::try_from_upper_rows(&[
            &[100.0, 165.0, 200.0, 212.0, 218.0],
            &[120.0, 190.0, 238.0, 250.0],
            &[135.0, 222.0, 268.0],
            &[150.0, 240.0],
            &[160.0],
        ])
        .unwrap();

        let base = mack::analyze(&tri.view(), 1.0).unwrap();

        // Limit 1: tau_sq = +inf => alpha = 1 => exact Mack (1993).
        let prior_inf = CredibilityClPrior {
            prior_ldf: vec![1.50, 1.20, 1.05, 1.02],
            tau_sq: vec![f64::INFINITY; 4],
            sigma_sq_override: None,
            tail: 1.0,
            policy: MonotonicityPolicy::Strict,
        };
        let res_inf = analyze(&tri.view(), &prior_inf).unwrap();
        assert!((res_inf.total_reserve - base.total_reserve).abs() < 1e-10);
        assert!((res_inf.total_standard_error - base.total_standard_error).abs() < 1e-10);

        // Limit 2: tau_sq = 0 => alpha = 0 => pure prior LDF, zero parameter variance.
        let prior_zero = CredibilityClPrior {
            prior_ldf: vec![1.50, 1.20, 1.05, 1.02],
            tau_sq: vec![0.0; 4],
            sigma_sq_override: None,
            tail: 1.0,
            policy: MonotonicityPolicy::Strict,
        };
        let res_zero = analyze(&tri.view(), &prior_zero).unwrap();
        for k in 0..4 {
            assert_eq!(res_zero.credibility_weight[k], 0.0);
            assert!((res_zero.credibility_ldf[k] - prior_zero.prior_ldf[k]).abs() < 1e-12);
        }
        for i in 0..5 {
            assert_eq!(res_zero.parameter_variance[i], 0.0);
        }

        // Intermediate tau_sq shrinks LDFs strictly between prior and empirical,
        // with lower total MSEP than Mack when prior is aligned.
        let prior_mid = CredibilityClPrior {
            prior_ldf: base.ldf.clone(),
            tau_sq: vec![0.01; 4],
            sigma_sq_override: None,
            tail: 1.0,
            policy: MonotonicityPolicy::Strict,
        };
        let res_mid = analyze(&tri.view(), &prior_mid).unwrap();
        assert!(res_mid.total_standard_error < base.total_standard_error);
    }

    #[test]
    fn collective_empirical_bayes_estimates_valid_parameters() {
        let t1 = Triangle::try_from_upper_rows(&[
            &[100.0, 160.0, 195.0, 205.0],
            &[110.0, 178.0, 216.0],
            &[120.0, 192.0],
            &[130.0],
        ])
        .unwrap();
        let t2 = Triangle::try_from_upper_rows(&[
            &[150.0, 255.0, 315.0, 335.0],
            &[160.0, 270.0, 334.0],
            &[170.0, 289.0],
            &[180.0],
        ])
        .unwrap();

        let params =
            estimate_collective_parameters(&[t1.view(), t2.view()], MonotonicityPolicy::Strict)
                .unwrap();
        assert_eq!(params.prior_ldf.len(), 3);
        assert!(params.tau_sq[0] > 0.0);
    }
}
