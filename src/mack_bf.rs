//! Mack (2008) distribution-free analytical MSEP for Bornhuetter–Ferguson.
//!
//! Implements:
//!
//! * T. Mack, *"The Prediction Error of Bornhuetter/Ferguson"*, *ASTIN
//!   Bulletin*, Vol. 38, No. 1 (2008), pp. 87–109.
//!
//! ## Model Assumptions (BF1)–(BF3)
//!
//! For incremental losses `S_{i,k}` (`i = 0..n-1`, `k = 0..n-1`):
//!
//! * **(BF1)** All increments `S_{i,k}` are mutually independent.
//! * **(BF2)** `E[S_{i,k}] = nu_i · y_k` where `y_k = z_k - z_{k-1}` is the
//!   incremental payout pattern (`z_k` cumulative payout proportion,
//!   `z_{n-1} = 1`).
//! * **(BF3)** `Var(S_{i,k}) = nu_i · s_k^2` where `s_k^2 >= 0` is the
//!   development-period process variance parameter.
//!
//! ## Analytical MSEP Decomposition
//!
//! Given *a priori* ultimates `E_i` (`nu_hat_i`) with standard errors
//! `s(E_i)`, optional inter-origin prior correlation `rho`, estimated
//! cumulative payout proportions `z_k` with standard errors `s(z_k)`, and
//! process variance parameters `s_k^2`:
//!
//! For each origin `i` (latest observed age `d_i = n - 1 - i`, reserve
//! `R_hat_i = E_i · (1 - z_{d_i})`):
//!
//! * **Process variance**: `Var(R_i) = E_i · sum_{k = d_i + 1}^{n - 1} s_k^2`.
//! * **Estimation variance** (exact Goodman variance of independent product):
//!   `Var(R_hat_i) = E_i^2 · s(z_{d_i})^2 + (1 - z_{d_i})^2 · s(E_i)^2 + s(E_i)^2 · s(z_{d_i})^2`.
//! * **Total MSEP**: `msep(R_hat_i) = Var(R_i) + Var(R_hat_i)`.
//!
//! Across origins `0 <= i < j < n`, shared payout-pattern covariance
//! `Cov(1 - z_{d_i}, 1 - z_{d_j}) = s(z_{d_i})^2` and optional prior covariance
//! `Cov(E_i, E_j) = rho_{ij} · s(E_i) · s(E_j)` enter the aggregate MSEP:
//!
//! `Cov(R_hat_i, R_hat_j) = (E_i E_j + Cov(E_i, E_j)) · Cov(1 - z_{d_i}, 1 - z_{d_j})
//!                          + (1 - z_{d_i})(1 - z_{d_j}) · Cov(E_i, E_j)`.

use crate::ldf::{DevelopmentFactorEngine, pct_reported};
use crate::triangle::{cumulative_to_incremental, latest_diagonal};
use crate::types::TriangleView;
use crate::validate::{
    MonotonicityPolicy, ValidationError, validate_monotone_cumulative,
    validate_nonnegative_finite_vector, validate_positive_finite_slice, validate_triangle,
};

/// How the cumulative payout pattern `z_k` is estimated from the triangle in
/// [`analyze_from_triangle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BfPayoutEstimator {
    /// Mack (2008) §2 direct incremental BF payout shares:
    /// `y_k = sum_{i=0}^{n-1-k} S_{i,k} / sum_{i=0}^{n-1-k} E_i`,
    /// `z_k = 1 - sum_{j=k+1}^{n-1} y_j` (normalized if needed so `z_{n-1} = 1`).
    #[default]
    DirectIncremental,
    /// Chain-ladder implied payout pattern `z_k = 1 / CDF_k` from volume-weighted LDFs.
    ChainLadderImplied,
}

/// Configuration options for [`analyze_from_triangle`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MackBfOptions {
    /// Payout pattern estimation method.
    pub payout_estimator: BfPayoutEstimator,
    /// Uniform pairwise correlation `rho in [0, 1]` between *a priori* ultimate
    /// estimators `E_i` and `E_j` (`0.0` = independent priors; `1.0` = fully
    /// comonotonic pricing-cycle uncertainty, as compared in Mack 2008 Table 3).
    pub prior_correlation: f64,
    /// Cumulative monotonicity validation policy.
    pub monotonicity_policy: MonotonicityPolicy,
}

impl Default for MackBfOptions {
    fn default() -> Self {
        Self {
            payout_estimator: BfPayoutEstimator::DirectIncremental,
            prior_correlation: 0.0,
            monotonicity_policy: MonotonicityPolicy::Strict,
        }
    }
}

/// Complete Mack (2008) analytical BF MSEP decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct MackBfResult {
    /// Latest observed cumulative losses by origin (length `n`).
    pub latest: Vec<f64>,
    /// *A priori* ultimate estimates `E_i` (length `n`).
    pub prior_ultimate: Vec<f64>,
    /// *A priori* ultimate standard errors `s(E_i)` (length `n`).
    pub prior_se: Vec<f64>,
    /// Cumulative payout pattern `z_k` for development ages `k = 0..n-1` (length `n`).
    pub payout_z: Vec<f64>,
    /// Payout pattern standard errors `s(z_k)` for `k = 0..n-1` (length `n`).
    pub payout_se: Vec<f64>,
    /// Development-period process variance parameters `s_k^2` (length `n`).
    pub process_sigma_sq: Vec<f64>,
    /// BF reserves `R_hat_i = E_i * (1 - z_{n - 1 - i})` by origin (length `n`).
    pub reserves: Vec<f64>,
    /// BF ultimates `latest[i] + reserves[i]` by origin (length `n`).
    pub ultimates: Vec<f64>,
    /// Process variance `Var(R_i)` by origin.
    pub process_variance: Vec<f64>,
    /// Prior-uncertainty component `(1 - z_{d_i})^2 * s(E_i)^2` by origin.
    pub prior_variance: Vec<f64>,
    /// Payout-pattern uncertainty component `E_i^2 * s(z_{d_i})^2` by origin.
    pub pattern_variance: Vec<f64>,
    /// Exact product interaction term `s(E_i)^2 * s(z_{d_i})^2` by origin.
    pub interaction_variance: Vec<f64>,
    /// Total estimation variance `Var(R_hat_i)` by origin.
    pub estimation_variance: Vec<f64>,
    /// Total MSEP `Var(R_i) + Var(R_hat_i)` by origin.
    pub msep: Vec<f64>,
    /// Prediction standard error `sqrt(msep[i])` by origin.
    pub standard_error: Vec<f64>,
    /// Coefficient of variation `standard_error[i] / reserves[i]` by origin.
    pub cv: Vec<f64>,
    /// Total BF reserve across origins.
    pub total_reserve: f64,
    /// Aggregate process variance `sum_i Var(R_i)`.
    pub total_process_variance: f64,
    /// Aggregate estimation variance (including cross-origin covariances).
    pub total_estimation_variance: f64,
    /// Aggregate cross-origin covariance contribution `2 * sum_{i<j} Cov(R_hat_i, R_hat_j)`.
    pub total_covariance: f64,
    /// Aggregate MSEP (`total_process_variance + total_estimation_variance`).
    pub total_msep: f64,
    /// Aggregate prediction standard error `sqrt(total_msep)`.
    pub total_standard_error: f64,
    /// Aggregate coefficient of variation (`total_standard_error / total_reserve`).
    pub total_cv: f64,
}

/// Run Mack (2008) analytical BF MSEP directly from explicit model components:
/// `latest`, `prior_ultimate` (`E_i`), `prior_se` (`s(E_i)`), `payout_z` (`z_k`),
/// `payout_se` (`s(z_k)`), `process_sigma_sq` (`s_k^2`), and pairwise prior
/// correlation `prior_correlation in [0, 1]`.
///
/// This entry point matches Mack (2008) Tables 1–3 directly when the actuary
/// supplies judgmental or externally calibrated `s(E_i)`, `s(z_k)`, and `s_k^2`.
pub fn analyze_from_components(
    latest: &[f64],
    prior_ultimate: &[f64],
    prior_se: &[f64],
    payout_z: &[f64],
    payout_se: &[f64],
    process_sigma_sq: &[f64],
    prior_correlation: f64,
) -> Result<MackBfResult, ValidationError> {
    let n = latest.len();
    if n < 2
        || prior_ultimate.len() != n
        || prior_se.len() != n
        || payout_z.len() != n
        || payout_se.len() != n
        || process_sigma_sq.len() != n
    {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_nonnegative_finite_vector(latest, n)?;
    validate_positive_finite_slice(prior_ultimate, "prior_ultimate")?;
    validate_nonnegative_finite_vector(prior_se, n)?;
    validate_nonnegative_finite_vector(payout_se, n)?;
    validate_nonnegative_finite_vector(process_sigma_sq, n)?;
    for (idx, &z) in payout_z.iter().enumerate() {
        if !z.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if !(0.0..=1.0).contains(&z) {
            return Err(ValidationError::InvalidMaturity {
                index: idx,
                value: z,
            });
        }
    }
    if !prior_correlation.is_finite() || !(0.0..=1.0).contains(&prior_correlation) {
        return Err(ValidationError::InvalidParameter(
            "prior_correlation must lie in [0, 1]",
        ));
    }

    let mut reserves = vec![0.0; n];
    let mut ultimates = vec![0.0; n];
    let mut process_variance = vec![0.0; n];
    let mut prior_variance = vec![0.0; n];
    let mut pattern_variance = vec![0.0; n];
    let mut interaction_variance = vec![0.0; n];
    let mut estimation_variance = vec![0.0; n];
    let mut msep = vec![0.0; n];
    let mut standard_error = vec![0.0; n];
    let mut cv = vec![0.0; n];

    for i in 0..n {
        let d_i = n - 1 - i;
        let q_i = 1.0 - payout_z[d_i];
        let e_i = prior_ultimate[i];
        let se_e = prior_se[i];
        let se_z = payout_se[d_i];

        reserves[i] = e_i * q_i;
        ultimates[i] = latest[i] + reserves[i];

        let fut_s_sq: f64 = process_sigma_sq[(d_i + 1)..n].iter().sum();
        let proc_v = (e_i * fut_s_sq).max(0.0);
        let prior_v = (q_i * se_e).powi(2);
        let pat_v = (e_i * se_z).powi(2);
        let inter_v = (se_e * se_z).powi(2);
        let est_v = prior_v + pat_v + inter_v;
        let tot_v = proc_v + est_v;

        process_variance[i] = proc_v;
        prior_variance[i] = prior_v;
        pattern_variance[i] = pat_v;
        interaction_variance[i] = inter_v;
        estimation_variance[i] = est_v;
        msep[i] = tot_v;
        standard_error[i] = tot_v.sqrt();
        cv[i] = if reserves[i] > 0.0 {
            standard_error[i] / reserves[i]
        } else {
            0.0
        };
    }

    let total_reserve: f64 = reserves.iter().sum();
    let total_process_variance: f64 = process_variance.iter().sum();

    // Cross-origin covariances for 0 <= i < j < n:
    // origin i is older (d_i > d_j), so shared future pattern columns are k = d_i+1..n-1,
    // whose cumulative variance is s(z_{d_i})^2.
    let mut total_covariance = 0.0;
    for i in 0..n {
        let d_i = n - 1 - i;
        let q_i = 1.0 - payout_z[d_i];
        let e_i = prior_ultimate[i];
        let cov_z_ij = payout_se[d_i] * payout_se[d_i];

        for j in (i + 1)..n {
            let d_j = n - 1 - j;
            let q_j = 1.0 - payout_z[d_j];
            let e_j = prior_ultimate[j];
            let cov_e_ij = prior_correlation * prior_se[i] * prior_se[j];

            let cov_r_ij = (e_i * e_j + cov_e_ij) * cov_z_ij + q_i * q_j * cov_e_ij;
            total_covariance += 2.0 * cov_r_ij;
        }
    }

    let sum_est_var: f64 = estimation_variance.iter().sum();
    let total_estimation_variance = (sum_est_var + total_covariance).max(0.0);
    let total_msep = total_process_variance + total_estimation_variance;
    let total_standard_error = total_msep.sqrt();
    let total_cv = if total_reserve > 0.0 {
        total_standard_error / total_reserve
    } else {
        0.0
    };

    Ok(MackBfResult {
        latest: latest.to_vec(),
        prior_ultimate: prior_ultimate.to_vec(),
        prior_se: prior_se.to_vec(),
        payout_z: payout_z.to_vec(),
        payout_se: payout_se.to_vec(),
        process_sigma_sq: process_sigma_sq.to_vec(),
        reserves,
        ultimates,
        process_variance,
        prior_variance,
        pattern_variance,
        interaction_variance,
        estimation_variance,
        msep,
        standard_error,
        cv,
        total_reserve,
        total_process_variance,
        total_estimation_variance,
        total_covariance,
        total_msep,
        total_standard_error,
        total_cv,
    })
}

/// Estimate Mack (2008) parameters (`z_k`, `s(z_k)`, `s_k^2`) directly from a
/// cumulative loss triangle and compute the analytical BF MSEP decomposition.
pub fn analyze_from_triangle(
    triangle: &TriangleView<'_>,
    prior_ultimate: &[f64],
    prior_se: &[f64],
    options: MackBfOptions,
) -> Result<MackBfResult, ValidationError> {
    validate_triangle(triangle)?;
    if options.monotonicity_policy == MonotonicityPolicy::Strict {
        validate_monotone_cumulative(triangle)?;
    }
    let n = triangle.n_origins;
    if n < 3 {
        return Err(ValidationError::InvalidParameter(
            "Mack BF triangle estimation requires at least a 3x3 triangle",
        ));
    }
    if prior_ultimate.len() != n || prior_se.len() != n {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_positive_finite_slice(prior_ultimate, "prior_ultimate")?;
    validate_nonnegative_finite_vector(prior_se, n)?;

    let inc = cumulative_to_incremental(triangle);
    let latest = latest_diagonal(triangle);

    // Column-wise prior volume V_k = sum_{i=0}^{n-1-k} E_i and incremental shares y_k.
    let mut vol_k = vec![0.0; n];
    let mut y_raw = vec![0.0; n];
    for k in 0..n {
        let m_k = n - k;
        let mut s_sum = 0.0;
        let mut e_sum = 0.0;
        for i in 0..m_k {
            s_sum += inc[i * n + k];
            e_sum += prior_ultimate[i];
        }
        vol_k[k] = e_sum;
        y_raw[k] = s_sum / e_sum;
    }

    // Build cumulative payout pattern z_k (z_{n-1} = 1.0).
    let payout_z: Vec<f64> = match options.payout_estimator {
        BfPayoutEstimator::DirectIncremental => {
            let sum_y: f64 = y_raw.iter().sum();
            let scale = if sum_y > 0.0 { 1.0 / sum_y } else { 1.0 };
            let mut z = vec![0.0; n];
            let mut cum = 0.0;
            for k in 0..n {
                cum += (y_raw[k] * scale).max(0.0);
                z[k] = cum.clamp(0.0, 1.0);
            }
            z[n - 1] = 1.0;
            z
        }
        BfPayoutEstimator::ChainLadderImplied => {
            let factors = DevelopmentFactorEngine::volume_weighted_with_policy(
                triangle,
                1.0,
                options.monotonicity_policy,
            )?;
            let mut z = pct_reported(&factors.cdfs);
            for v in &mut z {
                *v = v.clamp(0.0, 1.0);
            }
            z[n - 1] = 1.0;
            z
        }
    };

    // Incremental shares implied by payout_z for variance estimation.
    let mut y_k = vec![0.0; n];
    y_k[0] = payout_z[0];
    for k in 1..n {
        y_k[k] = payout_z[k] - payout_z[k - 1];
    }

    // Mack (2008) §3 process variance parameters s_k^2:
    // s_k^2 = 1 / (m_k - 1) * sum_{i=0}^{m_k - 1} E_i * (S_{i,k} / E_i - y_k)^2.
    let mut s_sq = vec![0.0; n];
    for k in 0..(n - 1) {
        let m_k = n - k;
        let mut acc = 0.0;
        for i in 0..m_k {
            let diff = inc[i * n + k] / prior_ultimate[i] - y_k[k];
            acc += prior_ultimate[i] * diff * diff;
        }
        s_sq[k] = acc / (m_k as f64 - 1.0);
    }
    // Extrapolate final column k = n - 1 (single observation) via Mack's min-rule.
    s_sq[n - 1] = if n >= 3 {
        let a = s_sq[n - 2];
        let b = s_sq[n - 3];
        if a > 0.0 && b > 0.0 {
            (a * a / b).min(a).min(b).max(0.0)
        } else {
            0.0
        }
    } else {
        s_sq[0].max(0.0)
    };

    // Payout-pattern standard errors s(z_k) = sqrt(sum_{j=k+1}^{n-1} s_j^2 / V_j).
    let mut payout_se = vec![0.0; n];
    for k in 0..(n - 1) {
        let mut var_z = 0.0;
        for j in (k + 1)..n {
            if vol_k[j] > 0.0 {
                var_z += s_sq[j] / vol_k[j];
            }
        }
        payout_se[k] = var_z.max(0.0).sqrt();
    }
    payout_se[n - 1] = 0.0;

    analyze_from_components(
        &latest,
        prior_ultimate,
        prior_se,
        &payout_z,
        &payout_se,
        &s_sq,
        options.prior_correlation,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mack;
    use crate::types::Triangle;

    const NAN: f64 = f64::NAN;

    #[test]
    fn analytical_component_identities_hold_exactly() {
        let latest = [1000.0, 850.0, 600.0, 300.0];
        let prior = [1000.0, 1020.0, 1050.0, 1100.0];
        let prior_se = [50.0, 60.0, 70.0, 80.0];
        let z = [0.30, 0.60, 0.85, 1.00];
        let z_se = [0.04, 0.03, 0.015, 0.0];
        let s_sq = [40.0, 25.0, 12.0, 4.0];

        let indep = analyze_from_components(&latest, &prior, &prior_se, &z, &z_se, &s_sq, 0.0)
            .expect("indep");
        let corr = analyze_from_components(&latest, &prior, &prior_se, &z, &z_se, &s_sq, 1.0)
            .expect("corr");

        // Origin 0 is fully developed (z_3 = 1, s(z_3) = 0) -> zero reserve & zero SE.
        assert_eq!(indep.reserves[0], 0.0);
        assert_eq!(indep.standard_error[0], 0.0);

        // Per-origin variance decomposition identity.
        for i in 0..4 {
            let sum_parts = indep.process_variance[i]
                + indep.prior_variance[i]
                + indep.pattern_variance[i]
                + indep.interaction_variance[i];
            assert!((indep.msep[i] - sum_parts).abs() < 1e-10);
        }

        // Positively correlated priors strictly increase aggregate SE while
        // leaving single-origin SEs unchanged.
        for i in 0..4 {
            assert!((indep.standard_error[i] - corr.standard_error[i]).abs() < 1e-12);
        }
        assert!(corr.total_standard_error > indep.total_standard_error);
        assert!(indep.total_covariance > 0.0);
    }

    #[test]
    fn bf_msep_dominates_chain_ladder_on_noisy_youngest_origin() {
        // Construct a noisy 5x5 triangle where origin 4 has an unusually volatile first cell.
        #[rustfmt::skip]
        let data = vec![
            100.0, 220.0, 310.0, 360.0, 380.0,
            150.0, 250.0, 340.0, 390.0,   NAN,
             70.0, 210.0, 300.0,   NAN,   NAN,
            130.0, 260.0,   NAN,   NAN,   NAN,
            180.0,   NAN,   NAN,   NAN,   NAN,
        ];
        let tri = Triangle::new(5, 5, data);
        let prior = [380.0, 395.0, 390.0, 405.0, 410.0];
        let prior_se = [20.0; 5]; // 5% a priori CoV

        let m_cl = mack::analyze(&tri.view(), 1.0).unwrap();
        let m_bf = analyze_from_triangle(&tri.view(), &prior, &prior_se, MackBfOptions::default())
            .unwrap();

        assert!(
            m_bf.standard_error[4] < m_cl.standard_error[4],
            "BF SE={} should be < CL SE={}",
            m_bf.standard_error[4],
            m_cl.standard_error[4]
        );
    }
}
