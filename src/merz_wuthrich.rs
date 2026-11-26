//! Merz & Wüthrich (2008) 1-year Claims Development Result (CDR) analytical
//! MSEP for Solvency II / 1-year capital horizons.
//!
//! Implements:
//!
//! * M. Merz & M. V. Wüthrich, *"Modelling the Claims Development Result for
//!   Solvency Purposes"*, *CAS E-Forum*, Fall 2008, pp. 542–568.
//!
//! ## Formulation
//!
//! Under Mack's (1993) distribution-free chain-ladder assumptions for an
//! `n × n` cumulative triangle (`I = J = n - 1`), the 1-year Claims
//! Development Result for origin `i` (`1 <= i <= I`, latest observed age
//! `d_i = I - i`) is the update in ultimate claims over the next accounting
//! year `I -> I + 1`:
//!
//! `CDR_{i, I + 1} = C_hat_{i, J}^{(I)} - C_hat_{i, J}^{(I + 1)}`.
//!
//! Let `S_j^{[I]} = sum_{m=0}^{I-j-1} C_{m, j}`,
//! `S_j^{[I+1]} = S_j^{[I]} + C_{I-j, j}`, and credibility weight
//! `alpha_j = C_{I-j, j} / S_j^{[I+1]}` for link `j = 0..J-1`.
//!
//! ### Single-Origin 1-Year MSEP (Merz & Wüthrich 2008, Theorem 1)
//!
//! `msep(CDR_{i, I+1} | D_I) = (C_hat_{i, J}^{(I)})^2 * [`
//! `  (sigma_hat_{d_i}^2 / f_hat_{d_i}^2) * (1 / C_{i, d_i} + 1 / S_{d_i}^{[I]})`
//! `  + sum_{j = d_i + 1}^{J - 1} (sigma_hat_j^2 / f_hat_j^2) * (alpha_j / S_j^{[I]}) ]`
//!
//! ### Aggregate 1-Year MSEP with Cross-Origin Covariance (Theorem 2)
//!
//! `msep(sum_i CDR_{i, I+1} | D_I) = sum_i msep(CDR_{i, I+1})`
//! `  + 2 * sum_{1 <= i < k <= I} C_hat_{i, J}^{(I)} * C_hat_{k, J}^{(I)} * [`
//! `      sigma_hat_{d_i}^2 / (f_hat_{d_i}^2 * S_{d_i}^{[I]})`
//! `      + sum_{j = d_i + 1}^{J - 1} (sigma_hat_j^2 / f_hat_j^2) * (alpha_j / S_j^{[I]}) ]`
//!
//! ### Multi-Year Run-Off Telescoping Identity
//!
//! [`runoff_decomposition`] computes the expected calendar-year CDR MSEP for
//! every future accounting year `t = 1, ..., n - 1` along the projected square.
//! Because `1 / S_j^{[I+t-1]} - 1 / S_j^{[I+t]}` telescopes across `t`, the sum
//! of calendar-year CDR MSEPs equals Mack's (1993) ultimate MSEP exactly:
//! `sum_{t=1}^{n-1} msep(CDR_t) == msep_Mack(Ultimate)`.

use crate::mack;
use crate::triangle::latest_diagonal;
use crate::types::TriangleView;
use crate::validate::{MonotonicityPolicy, ValidationError};

/// Result of the Merz & Wüthrich (2008) 1-year CDR MSEP calculation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MerzWuthrichResult {
    /// Volume-weighted chain-ladder LDFs (length `n - 1`).
    pub ldf: Vec<f64>,
    /// Mack (1993) link variance parameters `sigma_j^2` (length `n - 1`).
    pub sigma_sq: Vec<f64>,
    /// One-year LDF update weights `alpha_j = C_{I-j, j} / S_j^{[I+1]}` (length `n - 1`).
    pub alpha: Vec<f64>,
    /// Chain-ladder ultimates by origin (length `n`).
    pub ultimate: Vec<f64>,
    /// Chain-ladder reserves by origin (length `n`).
    pub reserve: Vec<f64>,
    /// 1-year CDR MSEP by origin (length `n`; origin `0` is `0.0`).
    pub cdr_msep: Vec<f64>,
    /// 1-year CDR standard error `sqrt(cdr_msep[i])` by origin.
    pub cdr_standard_error: Vec<f64>,
    /// Mack (1993) ultimate reserve standard error by origin (for emergence comparison).
    pub ultimate_standard_error: Vec<f64>,
    /// Emergence ratio `cdr_standard_error[i] / ultimate_standard_error[i]` by origin.
    pub emergence_ratio: Vec<f64>,
    /// Total chain-ladder reserve.
    pub total_reserve: f64,
    /// Sum of single-origin 1-year CDR MSEPs (excluding cross-origin covariance).
    pub sum_origin_cdr_msep: f64,
    /// Cross-origin covariance contribution to aggregate 1-year CDR MSEP.
    pub total_covariance: f64,
    /// Aggregate 1-year CDR MSEP.
    pub total_cdr_msep: f64,
    /// Aggregate 1-year CDR standard error `sqrt(total_cdr_msep)`.
    pub total_cdr_standard_error: f64,
    /// Aggregate Mack (1993) ultimate standard error.
    pub total_ultimate_standard_error: f64,
    /// Aggregate 1-year emergence factor `total_cdr_se / total_ultimate_se` (`<= 1.0`).
    pub total_emergence_ratio: f64,
    /// Aggregate 1-year coefficient of variation `total_cdr_se / total_reserve`.
    pub total_cdr_cv: f64,
}

impl std::fmt::Display for MerzWuthrichResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MerzWuthrich1Yr(reserve={:.2}, cdr_se={:.2}, cdr_cv={:.2}%, emergence={:.1}%)",
            self.total_reserve,
            self.total_cdr_standard_error,
            self.total_cdr_cv * 100.0,
            self.total_emergence_ratio * 100.0
        )
    }
}

/// Multi-year run-off decomposition of Mack's ultimate MSEP into calendar-year
/// CDR MSEPs `t = 1, ..., n - 1`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RunoffCdrDecomposition {
    /// Calendar step `t` (`1..=n-1`) aggregate CDR MSEP (length `n - 1`).
    pub step_total_msep: Vec<f64>,
    /// Calendar step `t` aggregate CDR standard error `sqrt(step_total_msep[t-1])`.
    pub step_total_se: Vec<f64>,
    /// Per-origin CDR MSEP by calendar step: `step_origin_msep[t - 1][origin]`.
    pub step_origin_msep: Vec<Vec<f64>>,
    /// Sum of `step_total_msep` across all future calendar years `t = 1..=n-1`.
    pub cumulative_cdr_msep: f64,
    /// Mack (1993) total ultimate MSEP (matches `cumulative_cdr_msep` algebraically).
    pub mack_ultimate_msep: f64,
}

/// Compute Merz & Wüthrich (2008) 1-year CDR MSEP with strict cumulative monotonicity.
pub fn analyze(t: &TriangleView<'_>) -> Result<MerzWuthrichResult, ValidationError> {
    analyze_with_policy(t, MonotonicityPolicy::Strict)
}

/// Compute Merz & Wüthrich (2008) 1-year CDR MSEP under an explicit [`MonotonicityPolicy`].
pub fn analyze_with_policy(
    t: &TriangleView<'_>,
    policy: MonotonicityPolicy,
) -> Result<MerzWuthrichResult, ValidationError> {
    let m = mack::analyze_with_policy(t, 1.0, policy)?;
    let n = t.n_origins;
    if n < 2 {
        return Err(ValidationError::InvalidParameter(
            "Merz & Wuthrich CDR requires at least a 2x2 triangle",
        ));
    }

    let last_idx = n - 1;
    let ldf = &m.ldf;
    let sigma_sq = &m.sigma_sq;
    let latest = latest_diagonal(t);

    // Precompute S_j^{[I]} = sum_{r=0}^{I-j-1} C_{r,j} and alpha_j = C_{I-j, j} / (S_j^{[I]} + C_{I-j, j}).
    let mut s_col = vec![0.0; n - 1];
    let mut alpha = vec![0.0; n - 1];
    let mut param_update_term = vec![0.0; n - 1]; // sigma_j^2 / f_j^2 * (alpha_j / S_j^{[I]})
    let mut link_param_base = vec![0.0; n - 1]; // sigma_j^2 / (f_j^2 * S_j^{[I]})

    for j in 0..(n - 1) {
        let nk = last_idx - j; // rows 0..nk are in S_j^{[I]}
        let mut s_i = 0.0;
        for r in 0..nk {
            s_i += t.get(r, j);
        }
        let diag_c = t.get(nk, j); // C_{I - j, j}, sits on latest diagonal
        let s_ip1 = s_i + diag_c;
        s_col[j] = s_i;
        alpha[j] = if s_ip1 > 0.0 { diag_c / s_ip1 } else { 0.0 };

        let fk = ldf[j];
        let sk2 = sigma_sq[j];
        if fk > 0.0 && s_i > 0.0 {
            let base = sk2 / (fk * fk * s_i);
            link_param_base[j] = base;
            param_update_term[j] = base * alpha[j];
        }
    }

    // Suffix sums of param_update_term for contiguous zero-alloc inner indexing.
    // suffix_update[j] = sum_{l = j}^{n - 2} param_update_term[l].
    let mut suffix_update = vec![0.0; n];
    for j in (0..(n - 1)).rev() {
        suffix_update[j] = suffix_update[j + 1] + param_update_term[j];
    }

    let mut cdr_msep = vec![0.0; n];
    for i in 1..n {
        let d_i = last_idx - i;
        let fk = ldf[d_i];
        let sk2 = sigma_sq[d_i];
        let c_diag = latest[i];
        let s_di = s_col[d_i];

        let mut bracket = suffix_update[d_i + 1];
        if fk > 0.0 && c_diag > 0.0 && s_di > 0.0 {
            bracket += (sk2 / (fk * fk)) * (1.0 / c_diag + 1.0 / s_di);
        }
        let u_i = m.ultimate[i];
        cdr_msep[i] = (u_i * u_i * bracket).max(0.0);
    }

    let cdr_standard_error: Vec<f64> = cdr_msep.iter().map(|v| v.sqrt()).collect();
    let emergence_ratio: Vec<f64> = cdr_standard_error
        .iter()
        .zip(m.standard_error.iter())
        .map(|(&s1, &su)| if su > 0.0 { s1 / su } else { 0.0 })
        .collect();

    let sum_origin_cdr_msep: f64 = cdr_msep.iter().sum();

    // Cross-origin covariance (Theorem 2):
    // For 1 <= i < k <= I, d_i = I - i > d_k = I - k.
    let mut total_covariance = 0.0;
    for i in 1..n {
        let d_i = last_idx - i;
        let cov_bracket = link_param_base[d_i] + suffix_update[d_i + 1];
        let u_i = m.ultimate[i];
        for k in (i + 1)..n {
            total_covariance += 2.0 * u_i * m.ultimate[k] * cov_bracket;
        }
    }

    let total_cdr_msep = (sum_origin_cdr_msep + total_covariance).max(0.0);
    let total_cdr_standard_error = total_cdr_msep.sqrt();
    let total_emergence_ratio = if m.total_standard_error > 0.0 {
        total_cdr_standard_error / m.total_standard_error
    } else {
        0.0
    };
    let total_cdr_cv = if m.total_reserve > 0.0 {
        total_cdr_standard_error / m.total_reserve
    } else {
        0.0
    };

    Ok(MerzWuthrichResult {
        ldf: m.ldf,
        sigma_sq: m.sigma_sq,
        alpha,
        ultimate: m.ultimate,
        reserve: m.reserve,
        cdr_msep,
        cdr_standard_error,
        ultimate_standard_error: m.standard_error,
        emergence_ratio,
        total_reserve: m.total_reserve,
        sum_origin_cdr_msep,
        total_covariance,
        total_cdr_msep,
        total_cdr_standard_error,
        total_ultimate_standard_error: m.total_standard_error,
        total_emergence_ratio,
        total_cdr_cv,
    })
}

/// Decompose Mack's (1993) ultimate MSEP across all future calendar years
/// `t = 1, ..., n - 1` using Merz & Wüthrich's telescoping run-off recursion.
///
/// The step `t = 1` entry equals [`analyze_with_policy`], and the sum across
/// `t = 1..=n-1` reproduces Mack's ultimate MSEP (`mack_ultimate_msep`)
/// identically (both per-origin and aggregate).
pub fn runoff_decomposition(
    t: &TriangleView<'_>,
    policy: MonotonicityPolicy,
) -> Result<RunoffCdrDecomposition, ValidationError> {
    let m = mack::analyze_with_policy(t, 1.0, policy)?;
    let n = t.n_origins;
    if n < 2 {
        return Err(ValidationError::InvalidParameter(
            "run-off decomposition requires at least a 2x2 triangle",
        ));
    }

    let last_idx = n - 1;
    let ldf = &m.ldf;
    let sigma_sq = &m.sigma_sq;

    // Build full projected square C_hat_{i, j} (observed upper-left + CL projected lower-right).
    let mut c_hat = vec![0.0; n * n];
    for i in 0..n {
        let d_i = last_idx - i;
        for j in 0..=d_i {
            c_hat[i * n + j] = t.get(i, j);
        }
        for j in (d_i + 1)..n {
            c_hat[i * n + j] = c_hat[i * n + j - 1] * ldf[j - 1];
        }
    }

    // Track evolving column sums S_j^{[I + t - 1]} across calendar steps t = 1..=n-1.
    let mut s_prev = vec![0.0; n - 1];
    for j in 0..(n - 1) {
        let mut s = 0.0;
        for r in 0..(last_idx - j) {
            s += c_hat[r * n + j];
        }
        s_prev[j] = s;
    }

    let mut step_total_msep = Vec::with_capacity(n - 1);
    let mut step_total_se = Vec::with_capacity(n - 1);
    let mut step_origin_msep = Vec::with_capacity(n - 1);

    for step in 1..n {
        // Calendar diagonal index moving from (last_idx + step - 1) -> (last_idx + step).
        // For each link j with last_idx + step - 1 - j < n (i.e. j >= step - 1),
        // the newly observed origin crossing link j -> j + 1 is r_new = last_idx + step - 1 - j.
        let mut s_next = s_prev.clone();
        let mut update_delta = vec![0.0; n - 1]; // (sigma_j^2 / f_j^2) * (1 / S_prev - 1 / S_next)
        for j in (step - 1)..(n - 1) {
            let r_new = last_idx + step - 1 - j;
            if r_new < n {
                s_next[j] = s_prev[j] + c_hat[r_new * n + j];
                let fk = ldf[j];
                let sk2 = sigma_sq[j];
                if fk > 0.0 && s_prev[j] > 0.0 && s_next[j] > 0.0 {
                    update_delta[j] = (sk2 / (fk * fk)) * (1.0 / s_prev[j] - 1.0 / s_next[j]);
                }
            }
        }

        let mut origin_msep_t = vec![0.0; n];
        for i in step..n {
            // Origin i has current known age d_cur = last_idx + step - 1 - i.
            let d_cur = last_idx + step - 1 - i;
            let fk = ldf[d_cur];
            let sk2 = sigma_sq[d_cur];
            let c_cur = c_hat[i * n + d_cur];
            let s_cur = s_prev[d_cur];
            let mut bracket = 0.0;
            if fk > 0.0 && c_cur > 0.0 && s_cur > 0.0 {
                bracket += (sk2 / (fk * fk)) * (1.0 / c_cur + 1.0 / s_cur);
            }
            for j in (d_cur + 1)..(n - 1) {
                bracket += update_delta[j];
            }
            let u_i = m.ultimate[i];
            origin_msep_t[i] = (u_i * u_i * bracket).max(0.0);
        }

        let mut total_msep_t: f64 = origin_msep_t.iter().sum();
        for i in step..n {
            let d_cur = last_idx + step - 1 - i;
            let fk = ldf[d_cur];
            let sk2 = sigma_sq[d_cur];
            let s_cur = s_prev[d_cur];
            let mut cov_bracket = if fk > 0.0 && s_cur > 0.0 {
                sk2 / (fk * fk * s_cur)
            } else {
                0.0
            };
            for j in (d_cur + 1)..(n - 1) {
                cov_bracket += update_delta[j];
            }
            let u_i = m.ultimate[i];
            for k in (i + 1)..n {
                total_msep_t += 2.0 * u_i * m.ultimate[k] * cov_bracket;
            }
        }

        step_total_msep.push(total_msep_t.max(0.0));
        step_total_se.push(total_msep_t.max(0.0).sqrt());
        step_origin_msep.push(origin_msep_t);
        s_prev = s_next;
    }

    let cumulative_cdr_msep: f64 = step_total_msep.iter().sum();

    Ok(RunoffCdrDecomposition {
        step_total_msep,
        step_total_se,
        step_origin_msep,
        cumulative_cdr_msep,
        mack_ultimate_msep: m.total_mse,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Triangle;

    const NAN: f64 = f64::NAN;

    fn mini_triangle() -> Triangle {
        #[rustfmt::skip]
        let data = vec![
            100.0, 160.0, 200.0, 215.0, 220.0,
            115.0, 178.0, 225.0, 240.0,   NAN,
            130.0, 212.0, 260.0,   NAN,   NAN,
            145.0, 228.0,   NAN,   NAN,   NAN,
            165.0,   NAN,   NAN,   NAN,   NAN,
        ];
        Triangle::new(5, 5, data)
    }

    #[test]
    fn runoff_sum_equals_mack_ultimate_msep_to_machine_precision() {
        let tri = mini_triangle();
        let mw1 = analyze(&tri.view()).expect("mw");
        let decomp = runoff_decomposition(&tri.view(), MonotonicityPolicy::Strict).expect("decomp");

        // Step 1 of run-off matches standalone 1-year Merz-Wuthrich identically.
        assert!((decomp.step_total_msep[0] - mw1.total_cdr_msep).abs() < 1e-10);
        for i in 0..5 {
            assert!((decomp.step_origin_msep[0][i] - mw1.cdr_msep[i]).abs() < 1e-10);
        }

        // Exact telescoping identity: sum_t msep(CDR_t) == msep_Mack(Ultimate).
        assert!(
            (decomp.cumulative_cdr_msep - decomp.mack_ultimate_msep).abs()
                / decomp.mack_ultimate_msep
                < 1e-12
        );
        // 1-year emergence factor lies strictly in (0, 1).
        assert!(mw1.total_emergence_ratio > 0.0 && mw1.total_emergence_ratio < 1.0);
    }
}
