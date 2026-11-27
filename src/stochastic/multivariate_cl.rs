//! Multivariate Chain Ladder (Zhang 2010 / Pröhl & Schmidt 2005).
//!
//! Models $K$ correlated cumulative loss triangles simultaneously:
//! $$Y_{i,k+1} = B_k Y_{i,k} + \varepsilon_{i,k+1}, \qquad
//!   \text{Cov}(\varepsilon_{i,k+1} \mid Y_{i,k}) = D(Y_{i,k})^{1/2} \, \Sigma_k \, D(Y_{i,k})^{1/2},$$
//! where $Y_{i,k} \in \mathbb{R}^K$ is the vector of cumulative losses across
//! the $K$ lines of business at origin $i$ and development age $k$.
//!
//! Two structural variants are supported (mirroring `mages/ChainLadder`'s
//! `MultiChainLadder`):
//! - [`MultivariateModel::DiagonalMcl`]: diagonal development matrices
//!   $B_k = \text{diag}(f_{k,1}, \dots, f_{k,K})$ with full cross-line
//!   residual covariance $\Sigma_k$ (Pröhl & Schmidt 2005; Braun 2004).
//!   Matches univariate Mack reserves line-by-line while capturing exact
//!   cross-line process and parameter covariance for portfolio MSEP.
//! - [`MultivariateModel::GeneralGmcl`]: full $K \times K$ development matrices
//!   $B_k$ estimated via weighted multivariate regression, allowing cross-line
//!   development spillovers (Zhang 2010).

use serde::{Deserialize, Serialize};

use crate::{
    linalg::solve_general,
    preprocessing::triangle::latest_diagonal,
    types::TriangleView,
    validate::{
        MonotonicityPolicy, ValidationError, validate_monotone_cumulative, validate_triangle,
    },
};

/// Structural specification for [`analyze_multivariate_cl`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MultivariateModel {
    /// Diagonal development matrices with full $K \times K$ residual covariance
    /// (Braun 2004 / Pröhl & Schmidt 2005 / `MultiChainLadder(..., model="MCL")`).
    DiagonalMcl,
    /// Full $K \times K$ development matrices with cross-line regression
    /// (Zhang 2010 / `MultiChainLadder(..., model="GMCL")`).
    GeneralGmcl,
}

/// Options for [`analyze_multivariate_cl`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultivariateClOptions {
    /// Diagonal MCL vs. General GMCL.
    pub model: MultivariateModel,
    /// Monotonicity policy applied to each input triangle.
    pub policy: MonotonicityPolicy,
}

impl Default for MultivariateClOptions {
    fn default() -> Self {
        Self {
            model: MultivariateModel::DiagonalMcl,
            policy: MonotonicityPolicy::AllowNegativeIncrements,
        }
    }
}

/// Result of a $K$-line Multivariate Chain Ladder analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultivariateClResult {
    /// Number of lines of business $K$.
    pub n_lines: usize,
    /// Number of origin / development periods $n$.
    pub n_origins: usize,
    /// Estimated $K \times K$ development matrices $B_k$ (each row-major `K × K`) for `k = 0..n-1`.
    pub development_matrices: Vec<Vec<f64>>,
    /// Estimated $K \times K$ residual covariance matrices $\Sigma_k$ (row-major `K × K`) for `k = 0..n-1`.
    pub sigma_matrices: Vec<Vec<f64>>,
    /// Projected ultimates per line (`n_lines` vectors of length `n_origins`).
    pub ultimates: Vec<Vec<f64>>,
    /// Projected reserves per line (`n_lines` vectors of length `n_origins`).
    pub reserves: Vec<Vec<f64>>,
    /// Total reserve per line.
    pub total_reserves_by_line: Vec<f64>,
    /// Portfolio total reserve summed across all $K$ lines.
    pub portfolio_total_reserve: f64,
    /// Full $K \times K$ total-reserve MSEP covariance matrix (row-major `K × K`).
    /// Diagonal entry `(a, a)` is the MSEP of line `a`; off-diagonal `(a, b)` is
    /// the cross-line reserve covariance.
    pub total_msep_matrix: Vec<f64>,
    /// Portfolio total reserve standard error $\sqrt{\mathbf{1}^\top \text{MSEP} \, \mathbf{1}}$.
    pub portfolio_standard_error: f64,
    /// Per-line reserve standard errors $\sqrt{\text{MSEP}_{a,a}}$.
    pub line_standard_errors: Vec<f64>,
}

/// Analyze $K$ aligned square $n \times n$ cumulative triangles jointly under
/// the Multivariate Chain Ladder framework.
pub fn analyze_multivariate_cl(
    triangles: &[TriangleView<'_>],
    options: &MultivariateClOptions,
) -> Result<MultivariateClResult, ValidationError> {
    let k_lines = triangles.len();
    if k_lines == 0 {
        return Err(ValidationError::ShapeMismatch);
    }
    let n = triangles[0].n_origins;
    if n < 3 || triangles[0].n_devs != n {
        return Err(ValidationError::ShapeMismatch);
    }
    for tri in triangles {
        validate_triangle(tri)?;
        if options.policy == MonotonicityPolicy::Strict {
            validate_monotone_cumulative(tri)?;
        }
        if tri.n_origins != n || tri.n_devs != n {
            return Err(ValidationError::ShapeMismatch);
        }
    }

    let n_steps = n - 1;
    let mut b_mats: Vec<Vec<f64>> = Vec::with_capacity(n_steps);
    let mut sigma_mats: Vec<Vec<f64>> = Vec::with_capacity(n_steps);
    let mut s_col: Vec<Vec<f64>> = vec![vec![0.0; k_lines]; n_steps];

    let mut resid = vec![0.0; k_lines];
    // 1. Estimate development matrices B_k and residual covariance Sigma_k.
    for k in 0..n_steps {
        let nk = n - 1 - k;
        for a in 0..k_lines {
            let mut sum = 0.0;
            for i in 0..nk {
                sum += triangles[a].get(i, k);
            }
            s_col[k][a] = sum;
        }

        let mut bk = vec![0.0; k_lines * k_lines];
        match options.model {
            MultivariateModel::DiagonalMcl => {
                fill_diagonal_mcl(triangles, k, nk, &mut bk);
            }
            MultivariateModel::GeneralGmcl => {
                // If nk <= k_lines, fall back to diagonal MCL for numerical stability.
                if nk <= k_lines {
                    fill_diagonal_mcl(triangles, k, nk, &mut bk);
                } else {
                    // Weighted multivariate regression for each target line a:
                    // minimize sum_i w_i * (Y_{i,k+1,a} - sum_b B_{a,b} Y_{i,k,b})^2
                    // with w_i = 1 / Y_{i,k,a} (matching Mack's alpha = 1 weighting).
                    let mut xtx = vec![0.0; k_lines * k_lines];
                    let mut xty = vec![0.0; k_lines];
                    for a in 0..k_lines {
                        xtx.fill(0.0);
                        xty.fill(0.0);
                        for i in 0..nk {
                            let y_a = triangles[a].get(i, k);
                            let w = if y_a > 0.0 { 1.0 / y_a } else { 0.0 };
                            let target = triangles[a].get(i, k + 1);
                            for b1 in 0..k_lines {
                                let x1 = triangles[b1].get(i, k);
                                xty[b1] += w * x1 * target;
                                for b2 in 0..k_lines {
                                    let x2 = triangles[b2].get(i, k);
                                    xtx[b1 * k_lines + b2] += w * x1 * x2;
                                }
                            }
                        }
                        match solve_general(&xtx, k_lines, &xty) {
                            Ok(beta) => {
                                bk[a * k_lines..(a + 1) * k_lines].copy_from_slice(&beta);
                            }
                            Err(_) => {
                                // Ridge/diagonal fallback on collinear lines.
                                bk[a * k_lines + a] = diagonal_ldf(&triangles[a], k, nk);
                            }
                        }
                    }
                }
            }
        }

        // Estimate K × K residual covariance Sigma_k:
        // e_{i,a} = (Y_{i,k+1,a} - (B_k Y_{i,k})_a) / sqrt(Y_{i,k,a})
        let mut sig = vec![0.0; k_lines * k_lines];
        let dof = match options.model {
            MultivariateModel::DiagonalMcl => nk.saturating_sub(1),
            MultivariateModel::GeneralGmcl => {
                if nk > k_lines {
                    nk - k_lines
                } else {
                    nk.saturating_sub(1)
                }
            }
        };
        if dof >= 1 {
            for i in 0..nk {
                for a in 0..k_lines {
                    let mut pred = 0.0;
                    for b in 0..k_lines {
                        pred += bk[a * k_lines + b] * triangles[b].get(i, k);
                    }
                    let y_a = triangles[a].get(i, k);
                    resid[a] = if y_a > 0.0 {
                        (triangles[a].get(i, k + 1) - pred) / y_a.sqrt()
                    } else {
                        0.0
                    };
                }
                for a in 0..k_lines {
                    let ra = resid[a];
                    for b in 0..k_lines {
                        sig[a * k_lines + b] += ra * resid[b];
                    }
                }
            }
            let inv_dof = 1.0 / dof as f64;
            for v in &mut sig {
                *v *= inv_dof;
            }
        }
        b_mats.push(bk);
        sigma_mats.push(sig);
    }

    // Mack (1993) extrapolation for final development step k = n - 2 (where nk = 1, dof = 0).
    if n_steps >= 3 {
        let last = n_steps - 1;
        let prev1 = n_steps - 2;
        let prev2 = n_steps - 3;
        for a in 0..k_lines {
            let s1 = sigma_mats[prev1][a * k_lines + a];
            let s2 = sigma_mats[prev2][a * k_lines + a];
            let s_last = if s2 > 0.0 {
                (s1 * s1 / s2).min(s1.min(s2))
            } else {
                0.0
            };
            sigma_mats[last][a * k_lines + a] = s_last.max(0.0);
        }
        // Extrapolate off-diagonal covariances preserving correlation from prev1.
        for a in 0..k_lines {
            for b in (a + 1)..k_lines {
                let va1 = sigma_mats[prev1][a * k_lines + a];
                let vb1 = sigma_mats[prev1][b * k_lines + b];
                let rho = if va1 > 0.0 && vb1 > 0.0 {
                    (sigma_mats[prev1][a * k_lines + b] / (va1 * vb1).sqrt()).clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                let va = sigma_mats[last][a * k_lines + a];
                let vb = sigma_mats[last][b * k_lines + b];
                let cov = rho * (va * vb).sqrt();
                sigma_mats[last][a * k_lines + b] = cov;
                sigma_mats[last][b * k_lines + a] = cov;
            }
        }
    }

    // 2. Project cumulative triangles jointly into a flat n × n × K buffer.
    let latest_by_line: Vec<Vec<f64>> = triangles.iter().map(latest_diagonal).collect();
    let mut proj = vec![0.0; n * n * k_lines];
    let proj_idx = |i: usize, k: usize, a: usize| (i * n + k) * k_lines + a;
    for i in 0..n {
        let start = n - 1 - i;
        for a in 0..k_lines {
            proj[proj_idx(i, start, a)] = latest_by_line[a][i];
        }
        for k in start..n_steps {
            let src_base = (i * n + k) * k_lines;
            let dst_base = (i * n + k + 1) * k_lines;
            let bk = &b_mats[k];
            for a in 0..k_lines {
                let mut acc = 0.0;
                for b in 0..k_lines {
                    acc += bk[a * k_lines + b] * proj[src_base + b];
                }
                proj[dst_base + a] = acc;
            }
        }
    }

    let mut ultimates = vec![vec![0.0; n]; k_lines];
    let mut reserves = vec![vec![0.0; n]; k_lines];
    for a in 0..k_lines {
        for i in 0..n {
            let ult = proj[proj_idx(i, n - 1, a)];
            ultimates[a][i] = ult;
            reserves[a][i] = ult - latest_by_line[a][i];
        }
    }
    let total_reserves_by_line: Vec<f64> = reserves.iter().map(|r| r.iter().sum()).collect();
    let portfolio_total_reserve: f64 = total_reserves_by_line.iter().sum();

    // 3. Portfolio K × K MSEP covariance matrix via Braun (2004) / Pröhl–Schmidt (2005)
    //    recursive propagation of process and parameter covariance matrices.
    let kk = k_lines * k_lines;
    let mut total_msep = vec![0.0; kk];

    // Precompute parameter-estimation covariance scale per step k:
    // Exact Braun (2004) Eq. (3.5) for cross-line parameter covariance:
    //   Cov(f_{k,a}, f_{k,b}) = (sum_{i=0}^{nk-1} sqrt(C_{i,k,a} C_{i,k,b})) / (S_{k,a} S_{k,b}) * Sigma_{k,ab}.
    let mut param_scale = vec![0.0; n_steps * kk];
    for k in 0..n_steps {
        let nk = n - 1 - k;
        let ps_base = k * kk;
        for a in 0..k_lines {
            let sa = s_col[k][a];
            if sa <= 0.0 {
                continue;
            }
            for b in a..k_lines {
                let sb = s_col[k][b];
                if sb > 0.0 {
                    let mut cross_sqrt = 0.0;
                    for i in 0..nk {
                        let ca = triangles[a].get(i, k).max(0.0);
                        let cb = triangles[b].get(i, k).max(0.0);
                        cross_sqrt += (ca * cb).sqrt();
                    }
                    let val = (cross_sqrt / (sa * sb)) * sigma_mats[k][a * k_lines + b];
                    param_scale[ps_base + a * k_lines + b] = val;
                    param_scale[ps_base + b * k_lines + a] = val;
                }
            }
        }
    }

    // Precompute cumulative transition products P_{k -> n-1} = B_{n-2} ... B_k (K × K).
    let mut trans_to_ult = vec![vec![0.0; kk]; n];
    for a in 0..k_lines {
        trans_to_ult[n - 1][a * k_lines + a] = 1.0;
    }
    for k in (0..n_steps).rev() {
        trans_to_ult[k] = mat_mul(&trans_to_ult[k + 1], &b_mats[k], k_lines);
    }

    // Hoisted scratch buffers for zero-allocation MSEP accumulation.
    let mut step_mat = vec![0.0; kk];
    let mut pm_scratch = vec![0.0; kk];
    let mut sqrt_c = vec![0.0; k_lines];
    let mut z_k = vec![0.0; k_lines];

    // Per-origin process variance:
    // sum_{k=start}^{n-2} P_{k+1->ult} D(C_{i,k})^{1/2} Sigma_k D(C_{i,k})^{1/2} P_{k+1->ult}^T
    for i in 1..n {
        let start = n - 1 - i;
        for k in start..n_steps {
            let p_next = &trans_to_ult[k + 1];
            let base = (i * n + k) * k_lines;
            for a in 0..k_lines {
                sqrt_c[a] = proj[base + a].max(0.0).sqrt();
            }
            let sig_k = &sigma_mats[k];
            for a in 0..k_lines {
                let sa = sqrt_c[a];
                for b in 0..k_lines {
                    step_mat[a * k_lines + b] = sa * sqrt_c[b] * sig_k[a * k_lines + b];
                }
            }
            add_mat_mul_ab_at(p_next, &step_mat, k_lines, &mut pm_scratch, &mut total_msep);
        }
    }

    // Parameter covariance summed across all origins (including cross-origin pairs i, j):
    //   P_{k+1->ult} [ (Z_k Z_k^T) ∘ Cov(f_k) ] P_{k+1->ult}^T.
    for k in 0..n_steps {
        let first_origin = n - 1 - k;
        z_k.fill(0.0);
        for i in first_origin..n {
            let base = (i * n + k) * k_lines;
            for a in 0..k_lines {
                z_k[a] += proj[base + a];
            }
        }
        let ps_base = k * kk;
        for a in 0..k_lines {
            let za = z_k[a];
            for b in 0..k_lines {
                step_mat[a * k_lines + b] = za * z_k[b] * param_scale[ps_base + a * k_lines + b];
            }
        }
        add_mat_mul_ab_at(
            &trans_to_ult[k + 1],
            &step_mat,
            k_lines,
            &mut pm_scratch,
            &mut total_msep,
        );
    }

    // Symmetrize numerically.
    for a in 0..k_lines {
        for b in (a + 1)..k_lines {
            let avg = f64::midpoint(total_msep[a * k_lines + b], total_msep[b * k_lines + a]);
            total_msep[a * k_lines + b] = avg;
            total_msep[b * k_lines + a] = avg;
        }
    }

    let portfolio_var: f64 = total_msep.iter().sum();
    let portfolio_standard_error = portfolio_var.max(0.0).sqrt();
    let line_standard_errors: Vec<f64> = (0..k_lines)
        .map(|a| total_msep[a * k_lines + a].max(0.0).sqrt())
        .collect();

    Ok(MultivariateClResult {
        n_lines: k_lines,
        n_origins: n,
        development_matrices: b_mats,
        sigma_matrices: sigma_mats,
        ultimates,
        reserves,
        total_reserves_by_line,
        portfolio_total_reserve,
        total_msep_matrix: total_msep,
        portfolio_standard_error,
        line_standard_errors,
    })
}

fn diagonal_ldf(tri: &TriangleView<'_>, k: usize, nk: usize) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..nk {
        num += tri.get(i, k + 1);
        den += tri.get(i, k);
    }
    if den > 0.0 { num / den } else { 1.0 }
}

fn fill_diagonal_mcl(triangles: &[TriangleView<'_>], k: usize, nk: usize, bk: &mut [f64]) {
    let k_lines = triangles.len();
    for (a, tri) in triangles.iter().enumerate() {
        bk[a * k_lines + a] = diagonal_ldf(tri, k, nk);
    }
}

fn mat_mul(a: &[f64], b: &[f64], k: usize) -> Vec<f64> {
    let mut out = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut acc = 0.0;
            for m in 0..k {
                acc += a[i * k + m] * b[m * k + j];
            }
            out[i * k + j] = acc;
        }
    }
    out
}

fn add_mat_mul_ab_at(p: &[f64], m: &[f64], k: usize, pm: &mut [f64], out: &mut [f64]) {
    for i in 0..k {
        for j in 0..k {
            let mut acc = 0.0;
            for r in 0..k {
                acc += p[i * k + r] * m[r * k + j];
            }
            pm[i * k + j] = acc;
        }
    }
    for i in 0..k {
        for j in 0..k {
            let mut acc = 0.0;
            for r in 0..k {
                acc += pm[i * k + r] * p[j * k + r];
            }
            out[i * k + j] += acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{stochastic::mack, types::Triangle};

    #[test]
    fn diagonal_mcl_matches_univariate_mack_exactly() {
        #[rustfmt::skip]
        let genins = Triangle::new(10, 10, vec![
            357848.0, 1124788.0, 1735330.0, 2218270.0, 2745596.0, 3319994.0, 3466336.0, 3606286.0, 3833515.0, 3901463.0,
            352118.0, 1236139.0, 2170033.0, 3353322.0, 3799067.0, 4120063.0, 4647867.0, 4914039.0, 5339085.0, f64::NAN,
            290507.0, 1292306.0, 2218525.0, 3235179.0, 3985995.0, 4132918.0, 4628910.0, 4909315.0, f64::NAN, f64::NAN,
            310608.0, 1418858.0, 2195047.0, 3757447.0, 4029929.0, 4381982.0, 4588268.0, f64::NAN, f64::NAN, f64::NAN,
            443160.0, 1136350.0, 2128333.0, 2897821.0, 3402672.0, 3873311.0, f64::NAN, f64::NAN, f64::NAN, f64::NAN,
            396132.0, 1333217.0, 2180715.0, 2985752.0, 3691712.0, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN,
            440832.0, 1288463.0, 2419861.0, 3483130.0, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN,
            359480.0, 1421128.0, 2864498.0, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN,
            376686.0, 1363294.0, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN,
            344014.0, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN,
        ]);
        let m_uni = mack::analyze(&genins.view(), 1.0).expect("mack");
        let m_multi = analyze_multivariate_cl(&[genins.view()], &MultivariateClOptions::default())
            .expect("mcl");
        assert!((m_multi.portfolio_total_reserve - m_uni.total_reserve).abs() < 1e-6);
        assert!(
            (m_multi.portfolio_standard_error - m_uni.total_standard_error).abs() < 1e-6,
            "multi SE={} vs uni SE={}",
            m_multi.portfolio_standard_error,
            m_uni.total_standard_error
        );
    }
}
