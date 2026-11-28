//! Multivariate Chain-Ladder analytical MSEP for correlated run-off triangles.
//!
//! Implements:
//!
//! * Braun (2004), *The Prediction Error of the Chain Ladder Method Applied to
//!   Correlated Run-Off Triangles*, ASTIN Bulletin, Vol. 34, No. 2, pp. 399–423.
//! * Merz & Wüthrich (2008), *Prediction Error of the Multivariate Chain Ladder
//!   Reserving Method*, North American Actuarial Journal, Vol. 12, No. 2,
//!   pp. 175–197.
//!
//! This module is the exact distribution-free analytical counterpart to
//! [`crate::capital::dependence::synchronized_odp_bootstrap`]. Given $N$ aligned square
//! cumulative triangles $C^{(1)}, \dots, C^{(N)}$ of dimension $n \times n$:
//!
//! * Each triangle $r$ has standard Mack (1993) LDFs $\hat{f}_k^{(r)}$,
//!   variances $\hat{\sigma}_k^{(r)2}$, ultimates $\hat{U}_i^{(r)}$, and
//!   reserves $\hat{R}_i^{(r)}$.
//! * Cross-triangle link covariance at development link $k$ ($n_k = n - k - 1$):
//!
//!   $$\hat{\sigma}_k^{(r,s)} = \frac{1}{w_k^{(r,s)}}
//!   \sum_{i=0}^{n_k-1} \sqrt{C_{i,k}^{(r)} C_{i,k}^{(s)}}
//!   \left(\frac{C_{i,k+1}^{(r)}}{C_{i,k}^{(r)}} - \hat{f}_k^{(r)}\right)
//!   \left(\frac{C_{i,k+1}^{(s)}}{C_{i,k}^{(s)}} - \hat{f}_k^{(s)}\right)$$
//!
//!   with unbiased denominator (Braun 2004, Eq. 3.4):
//!
//!   $$w_k^{(r,s)} = n_k - 2 + \frac{\left(\sum_{i=0}^{n_k-1} \sqrt{C_{i,k}^{(r)} C_{i,k}^{(s)}}\right)^2}{S_k^{(r)} S_k^{(s)}}$$
//!
//!   which reduces identically to $n_k - 1$ when $r = s$.
//! * Cross-triangle total-reserve covariance between lines $r$ and $s$ sums
//!   same-origin process + parameter covariance and cross-origin parameter
//!   covariance across shared links $k \ge n - 1 - i$.
//! * When $N = 1$ or all cross-covariances are zero, the aggregate MSEP
//!   reduces identically to [`crate::stochastic::mack`].

use crate::stochastic::mack::{self, Mack};
use crate::types::TriangleView;
use crate::validate::{
    MonotonicityPolicy, ValidationError, validate_monotone_cumulative, validate_tail,
    validate_triangle,
};

/// Result of Braun (2004) / Merz & Wüthrich (2008) multivariate chain-ladder MSEP.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MultivariateMackResult {
    /// Number of correlated triangles $N$.
    pub n_triangles: usize,
    /// Square triangle dimension $n$.
    pub n_origins: usize,
    /// Individual Mack (1993) results per triangle, length $N$.
    pub marginal_mack: Vec<Mack>,
    /// Per-link cross-covariance matrices $\hat{\sigma}_k^{(r,s)}$ (length $n - 1$,
    /// each an $N \times N$ row-major matrix).
    pub link_covariances: Vec<Vec<f64>>,
    /// Per-link cross-correlation matrices $\hat{\rho}_k^{(r,s)}$ (length $n - 1$,
    /// each an $N \times N$ row-major matrix).
    pub link_correlations: Vec<Vec<f64>>,
    /// $N \times N$ row-major covariance matrix of total reserves across triangles.
    /// Diagonal entry `(r, r)` equals `marginal_mack[r].total_mse`.
    pub reserve_covariance_matrix: Vec<f64>,
    /// $N \times N$ row-major implied correlation matrix of total reserves across triangles.
    pub reserve_correlation_matrix: Vec<f64>,
    /// Aggregate reserve per origin summed across all triangles, length $n$.
    pub origin_total_reserve: Vec<f64>,
    /// Aggregate MSEP per origin across all triangles (including cross-line covariances), length $n$.
    pub origin_total_mse: Vec<f64>,
    /// Aggregate standard error per origin across all triangles, length $n$.
    pub origin_total_se: Vec<f64>,
    /// Portfolio total reserve summed across all triangles.
    pub total_reserve: f64,
    /// Portfolio aggregate MSEP including all cross-triangle covariances.
    pub total_mse: f64,
    /// Portfolio aggregate standard error `sqrt(total_mse)`.
    pub total_standard_error: f64,
    /// Portfolio aggregate coefficient of variation `total_standard_error / total_reserve`.
    pub total_cv: f64,
    /// Sum of standalone triangle standard errors (comonotonic upper bound).
    pub undiversified_se: f64,
    /// Diversification benefit: `undiversified_se - total_standard_error`.
    pub diversification_benefit: f64,
}

impl std::fmt::Display for MultivariateMackResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MultivariateMack(n_lines={}, reserve={:.2}, se={:.2}, cv={:.2}%, div={:.2})",
            self.n_triangles,
            self.total_reserve,
            self.total_standard_error,
            self.total_cv * 100.0,
            self.diversification_benefit
        )
    }
}

/// Compute Braun (2004) / Merz & Wüthrich (2008) multivariate chain-ladder MSEP
/// across $N$ aligned square triangles under strict cumulative monotonicity.
pub fn analyze(
    triangles: &[TriangleView<'_>],
    tail: f64,
) -> Result<MultivariateMackResult, ValidationError> {
    analyze_with_policy(triangles, tail, MonotonicityPolicy::Strict)
}

/// Compute Braun (2004) / Merz & Wüthrich (2008) multivariate chain-ladder MSEP
/// with an explicit [`MonotonicityPolicy`].
pub fn analyze_with_policy(
    triangles: &[TriangleView<'_>],
    tail: f64,
    policy: MonotonicityPolicy,
) -> Result<MultivariateMackResult, ValidationError> {
    if triangles.is_empty() {
        return Err(ValidationError::InvalidParameter(
            "at least one triangle is required",
        ));
    }
    validate_tail(tail)?;

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

    let n_lines = triangles.len();
    let mut marginal_mack = Vec::with_capacity(n_lines);
    for t in triangles {
        marginal_mack.push(mack::analyze_with_policy(t, tail, policy)?);
    }

    let n_links = n.saturating_sub(1);

    // Precompute column sums S_k^{(r)} and projected full cumulative grids \hat{C}_{i,k}^{(r)}.
    let mut col_sums = vec![vec![0.0; n_links]; n_lines];
    let mut projected = vec![vec![0.0; n * n]; n_lines];
    for r in 0..n_lines {
        let t = &triangles[r];
        let f = &marginal_mack[r].ldf;
        for k in 0..n_links {
            let nk = n - k - 1;
            let mut s = 0.0;
            for i in 0..nk {
                s += t.get(i, k);
            }
            col_sums[r][k] = s;
        }
        for i in 0..n {
            let last_obs = n - 1 - i;
            for k in 0..=last_obs {
                projected[r][i * n + k] = t.get(i, k);
            }
            let mut c = t.get(i, last_obs);
            for k in (last_obs + 1)..n {
                c *= f[k - 1];
                projected[r][i * n + k] = c;
            }
        }
    }

    // Cross-triangle covariance \hat{\sigma}_k^{(r,s)} and correlation \hat{\rho}_k^{(r,s)}.
    let mut link_covariances = vec![vec![0.0; n_lines * n_lines]; n_links];
    let mut link_correlations = vec![vec![0.0; n_lines * n_lines]; n_links];

    // Precompute sum_m sqrt(C_{m,k}^{(r)} C_{m,k}^{(s)}) for each link k and pair (r, s).
    let mut sum_sqrt_prod = vec![vec![0.0; n_lines * n_lines]; n_links];

    for k in 0..n_links {
        let nk = n - k - 1;
        for r in 0..n_lines {
            link_covariances[k][r * n_lines + r] = marginal_mack[r].sigma_sq[k];
            link_correlations[k][r * n_lines + r] = 1.0;
            sum_sqrt_prod[k][r * n_lines + r] = col_sums[r][k];
        }
        for r in 0..n_lines {
            for s in (r + 1)..n_lines {
                let mut ssp = 0.0;
                for i in 0..nk {
                    let cr = triangles[r].get(i, k);
                    let cs = triangles[s].get(i, k);
                    if cr > 0.0 && cs > 0.0 {
                        ssp += (cr * cs).sqrt();
                    }
                }
                sum_sqrt_prod[k][r * n_lines + s] = ssp;
                sum_sqrt_prod[k][s * n_lines + r] = ssp;

                let sig_r = marginal_mack[r].sigma_sq[k];
                let sig_s = marginal_mack[s].sigma_sq[k];
                let max_abs_cov = if sig_r > 0.0 && sig_s > 0.0 {
                    (sig_r * sig_s).sqrt()
                } else {
                    0.0
                };

                let cov_rs = if nk >= 2 {
                    let mut acc = 0.0;
                    let mut valid = 0usize;
                    let mut ssp_valid = 0.0;
                    let mut sr_valid = 0.0;
                    let mut ss_valid = 0.0;
                    for i in 0..nk {
                        let cr = triangles[r].get(i, k);
                        let cs = triangles[s].get(i, k);
                        if cr > 0.0 && cs > 0.0 {
                            let dr = triangles[r].get(i, k + 1) / cr - marginal_mack[r].ldf[k];
                            let ds = triangles[s].get(i, k + 1) / cs - marginal_mack[s].ldf[k];
                            let g = (cr * cs).sqrt();
                            acc += g * dr * ds;
                            ssp_valid += g;
                            sr_valid += cr;
                            ss_valid += cs;
                            valid += 1;
                        }
                    }
                    if valid >= 2 && sr_valid > 0.0 && ss_valid > 0.0 {
                        let denom =
                            (valid as f64 - 2.0) + (ssp_valid * ssp_valid) / (sr_valid * ss_valid);
                        if denom > 0.0 {
                            (acc / denom).clamp(-max_abs_cov, max_abs_cov)
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    }
                } else if k >= 2 {
                    // Extrapolate last-link covariance via Braun (2004) Mack-style min-rule
                    // preserving sign and Cauchy-Schwarz bound.
                    let prev1 = link_covariances[k - 1][r * n_lines + s];
                    let prev2 = link_covariances[k - 2][r * n_lines + s];
                    let a = prev1.abs();
                    let b = prev2.abs();
                    let mag = if a > 0.0 && b > 0.0 {
                        (a * a / b).min(a).min(b)
                    } else {
                        0.0
                    };
                    (prev1.signum() * mag).clamp(-max_abs_cov, max_abs_cov)
                } else if k == 1 {
                    link_covariances[0][r * n_lines + s].clamp(-max_abs_cov, max_abs_cov)
                } else {
                    0.0
                };

                link_covariances[k][r * n_lines + s] = cov_rs;
                link_covariances[k][s * n_lines + r] = cov_rs;
                let rho = if max_abs_cov > 0.0 {
                    (cov_rs / max_abs_cov).clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                link_correlations[k][r * n_lines + s] = rho;
                link_correlations[k][s * n_lines + r] = rho;
            }
        }
    }

    // Reserve covariance matrix across triangles and origin-level aggregate MSE.
    let mut reserve_cov = vec![0.0; n_lines * n_lines];
    let mut origin_total_reserve = vec![0.0; n];
    let mut origin_total_mse = vec![0.0; n];

    for r in 0..n_lines {
        reserve_cov[r * n_lines + r] = marginal_mack[r].total_mse;
        for i in 0..n {
            origin_total_reserve[i] += marginal_mack[r].reserve[i];
            origin_total_mse[i] += marginal_mack[r].standard_error[i].powi(2);
        }
    }

    let mut param_terms = vec![0.0; n_links];
    let mut link_scale = vec![0.0; n_links];
    for r in 0..n_lines {
        let ur = &marginal_mack[r].ultimate;
        let fr_vec = &marginal_mack[r].ldf;
        for s in (r + 1)..n_lines {
            let us = &marginal_mack[s].ultimate;
            let fs_vec = &marginal_mack[s].ldf;

            // Precompute per-link cross-covariance scales once for pair (r, s).
            for k in 0..n_links {
                let fr = fr_vec[k];
                let fs = fs_vec[k];
                let sr = col_sums[r][k];
                let ss = col_sums[s][k];
                let sig_rs = link_covariances[k][r * n_lines + s];
                if fr > 0.0 && fs > 0.0 && sr > 0.0 && ss > 0.0 {
                    let scale = sig_rs / (fr * fs);
                    link_scale[k] = scale;
                    param_terms[k] = scale * (sum_sqrt_prod[k][r * n_lines + s] / (sr * ss));
                } else {
                    link_scale[k] = 0.0;
                    param_terms[k] = 0.0;
                }
            }

            let mut cov_total_rs = 0.0;
            // Same-origin cross-triangle covariance (process + parameter), i = 1..n-1.
            // Suffix sum of param_terms[start_k..n_links] is maintained incrementally as i grows
            // (since start_k = n - 1 - i decreases by 1 each step).
            let mut shared_param = 0.0;
            let mut suf_ur = 0.0;
            let mut suf_us = 0.0;
            for i in (1..n).rev() {
                suf_ur += ur[i];
                suf_us += us[i];
            }
            for i in 1..n {
                let start_k = n - 1 - i;
                shared_param += param_terms[start_k];

                let mut proc_sum = 0.0;
                let row_base = i * n;
                for k in start_k..n_links {
                    let scale = link_scale[k];
                    if scale != 0.0 {
                        let cr = projected[r][row_base + k];
                        let cs = projected[s][row_base + k];
                        if cr > 0.0 && cs > 0.0 {
                            proc_sum += scale / (cr * cs).sqrt();
                        }
                    }
                }
                let cov_ii = ur[i] * us[i] * (proc_sum + shared_param);
                cov_total_rs += cov_ii;
                origin_total_mse[i] += 2.0 * cov_ii;

                // Cross-origin cross-triangle parameter covariance for j > i in O(1):
                // sum_{j = i+1}^{n-1} (ur[i] * us[j] + ur[j] * us[i]) * shared_param
                suf_ur -= ur[i];
                suf_us -= us[i];
                cov_total_rs += (ur[i] * suf_us + suf_ur * us[i]) * shared_param;
            }

            reserve_cov[r * n_lines + s] = cov_total_rs;
            reserve_cov[s * n_lines + r] = cov_total_rs;
        }
    }

    let mut reserve_corr = vec![0.0; n_lines * n_lines];
    for r in 0..n_lines {
        reserve_corr[r * n_lines + r] = 1.0;
        let se_r = marginal_mack[r].total_standard_error;
        for s in (r + 1)..n_lines {
            let se_s = marginal_mack[s].total_standard_error;
            let rho = if se_r > 0.0 && se_s > 0.0 {
                (reserve_cov[r * n_lines + s] / (se_r * se_s)).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            reserve_corr[r * n_lines + s] = rho;
            reserve_corr[s * n_lines + r] = rho;
        }
    }

    let mut total_mse = 0.0;
    for &c in &reserve_cov {
        total_mse += c;
    }
    total_mse = total_mse.max(0.0);
    let total_standard_error = total_mse.sqrt();
    let total_reserve: f64 = marginal_mack.iter().map(|m| m.total_reserve).sum();
    let total_cv = if total_reserve > 0.0 {
        total_standard_error / total_reserve
    } else {
        0.0
    };
    let undiversified_se: f64 = marginal_mack.iter().map(|m| m.total_standard_error).sum();
    let diversification_benefit = undiversified_se - total_standard_error;

    for v in &mut origin_total_mse {
        *v = v.max(0.0);
    }
    let origin_total_se = origin_total_mse.iter().map(|&v| v.sqrt()).collect();

    Ok(MultivariateMackResult {
        n_triangles: n_lines,
        n_origins: n,
        marginal_mack,
        link_covariances,
        link_correlations,
        reserve_covariance_matrix: reserve_cov,
        reserve_correlation_matrix: reserve_corr,
        origin_total_reserve,
        origin_total_mse,
        origin_total_se,
        total_reserve,
        total_mse,
        total_standard_error,
        total_cv,
        undiversified_se,
        diversification_benefit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triangle;

    #[test]
    fn single_triangle_reduces_identically_to_mack() {
        let tri = Triangle::try_from_upper_rows(&[
            &[100.0, 160.0, 195.0, 210.0, 215.0],
            &[120.0, 195.0, 240.0, 255.0],
            &[135.0, 215.0, 260.0],
            &[150.0, 245.0],
            &[165.0],
        ])
        .unwrap();

        let solo = mack::analyze(&tri.view(), 1.0).unwrap();
        let mv = analyze(&[tri.view()], 1.0).unwrap();

        assert!((mv.total_reserve - solo.total_reserve).abs() < 1e-10);
        assert!((mv.total_mse - solo.total_mse).abs() < 1e-8);
        assert!((mv.total_standard_error - solo.total_standard_error).abs() < 1e-10);
        assert!(mv.diversification_benefit.abs() < 1e-10);
    }

    #[test]
    fn identical_scaled_triangles_have_unit_correlation_and_zero_diversification() {
        let t1 = Triangle::try_from_upper_rows(&[
            &[100.0, 165.0, 200.0, 212.0, 218.0],
            &[120.0, 190.0, 238.0, 250.0],
            &[135.0, 222.0, 268.0],
            &[150.0, 240.0],
            &[160.0],
        ])
        .unwrap();
        let data2: Vec<f64> = t1.data().iter().map(|&v| v * 2.5).collect();
        let t2 = Triangle::try_new(5, 5, data2).unwrap();

        let mv = analyze(&[t1.view(), t2.view()], 1.0).unwrap();
        assert!((mv.reserve_correlation_matrix[1] - 1.0).abs() < 1e-10);
        assert!(mv.diversification_benefit.abs() < 1e-6);
        assert!((mv.total_standard_error - mv.undiversified_se).abs() < 1e-6);
    }

    #[test]
    fn negatively_correlated_triangles_yield_positive_diversification() {
        let t1 = Triangle::try_from_upper_rows(&[
            &[100.0, 170.0, 210.0, 220.0, 225.0],
            &[100.0, 150.0, 195.0, 205.0],
            &[100.0, 175.0, 215.0],
            &[100.0, 145.0],
            &[100.0],
        ])
        .unwrap();
        let t2 = Triangle::try_from_upper_rows(&[
            &[100.0, 150.0, 195.0, 205.0, 210.0],
            &[100.0, 170.0, 210.0, 220.0],
            &[100.0, 145.0, 190.0],
            &[100.0, 175.0],
            &[100.0],
        ])
        .unwrap();

        let mv = analyze(&[t1.view(), t2.view()], 1.0).unwrap();
        assert!(mv.reserve_correlation_matrix[1] < 0.0);
        assert!(mv.diversification_benefit > 0.0);
        assert!(mv.total_standard_error < mv.undiversified_se);
    }
}
