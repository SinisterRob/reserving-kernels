//! Multi-line portfolio dependence: Iman–Conover rank reordering and
//! synchronized multi-line ODP bootstrapping.
//!
//! Implements two distinct dependence integration points from primary
//! actuarial literature:
//!
//! * **Iman–Conover (1982) / Mildenhall (2006)** — *Correlation and Aggregate
//!   Loss Distributions With an Emphasis on the Iman–Conover Method*, CAS
//!   Winter 2006 Forum, pp. 43–82. Post-simulation batch permutation that
//!   induces target rank correlation across marginal scenario vectors while
//!   preserving each empirical marginal distribution exactly. Returns whole-
//!   scenario index permutations so multi-origin and multi-line scenario
//!   records move coherently.
//! * **Synchronized ODP Bootstrap** — Kirschner, Kerley & Isaacs (2002),
//!   *Two Approaches to Calculating Correlated Reserve Indications Across
//!   Multiple Lines of Business*, CAS Forum, pp. 211–246; Shapland (2016),
//!   CAS Monograph No. 4, §9. Samples a single shared triangle coordinate map
//!   per iteration across aligned lines of business so empirical calendar/
//!   residual cross-line dependence is preserved during parameter resampling.

use std::f64::consts::PI;

use crate::bootstrap_odp::{
    DeterministicRng, OdpBootstrapOptions, OdpBootstrapScratch, OdpFit, ResidualAdjustment,
    build_adjusted_residual_pool, cholesky_lower, fit_odp, run_odp_iteration_with_scratch,
};
use crate::reserve_distribution::{ReserveDistributionSummary, summarize_with_scratch};
use crate::types::TriangleView;
use crate::validate::ValidationError;

/// Semantic interpretation of the supplied target correlation matrix for
/// [`iman_conover`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum CorrelationType {
    /// Target matrix is directly the Pearson correlation of latent Gaussian
    /// van der Waerden reference scores (Mildenhall 2006, §3).
    #[default]
    LatentGaussian,
    /// Target matrix specifies Spearman rank correlations `ρ_S`, converted via
    /// the continuous Gaussian copula relation `ρ = 2 · sin(π · ρ_S / 6)`.
    SpearmanTarget,
    /// Target matrix specifies Kendall's tau `τ`, converted via the continuous
    /// Gaussian copula relation `ρ = sin(π · τ / 2)`.
    KendallTarget,
}

/// Output of [`iman_conover`] batch rank reordering.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ImanConoverResult {
    /// Number of scenarios `N`.
    pub n_scenarios: usize,
    /// Number of correlated variables/lines `K`.
    pub n_variables: usize,
    /// Whole-scenario permutations: `permutations[col][row]` is the original
    /// scenario index placed into output position `row` for variable `col`.
    pub permutations: Vec<Vec<usize>>,
    /// Reordered marginal columns preserving exact empirical multisets.
    pub reordered_columns: Vec<Vec<f64>>,
    /// Row-major `K × K` effective latent Gaussian correlation matrix `S`.
    pub effective_latent_correlation: Vec<f64>,
    /// Row-major `K × K` achieved empirical Spearman rank correlation matrix.
    pub achieved_spearman: Vec<f64>,
    /// Row-major `K × K` achieved empirical monetary Pearson correlation matrix.
    pub achieved_pearson: Vec<f64>,
}

/// Output of [`synchronized_odp_bootstrap`] across `K` aligned lines of business.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SynchronizedBootstrapResult {
    /// Underlying ODP fits per line.
    pub line_fits: Vec<OdpFit>,
    /// Simulated total reserves per line: `line_totals[line][iter]`.
    pub line_totals: Vec<Vec<f64>>,
    /// Coherent portfolio aggregate reserves: `aggregate_totals[iter] = Σ_k line_totals[k][iter]`.
    pub aggregate_totals: Vec<f64>,
    /// Standalone reserve distribution summary for each line.
    pub line_summaries: Vec<ReserveDistributionSummary>,
    /// Correlated portfolio reserve distribution summary.
    pub aggregate_summary: ReserveDistributionSummary,
    /// Sum of standalone line standard deviations minus correlated portfolio
    /// standard deviation (`>= 0` under diversification).
    pub std_dev_diversification_benefit: f64,
    /// Row-major `K × K` empirical Spearman rank correlation of simulated line reserves.
    pub achieved_spearman: Vec<f64>,
    /// Row-major `K × K` empirical Pearson correlation of simulated line reserves.
    pub achieved_pearson: Vec<f64>,
    /// Count of rejected joint iterations (any line failing refit rejects the joint draw).
    pub failed_iterations: usize,
}

// ===========================================================================
// Normal Quantile (Acklam 2003, max relative error < 1.15e-9)
// ===========================================================================

/// Inverse standard normal CDF `Φ^{-1}(p)` for `p ∈ (0, 1)`.
pub(crate) fn inv_normal_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

// ===========================================================================
// Correlation Matrix Helpers (Pearson & Tie-Adjusted Spearman)
// ===========================================================================

/// Compute tie-adjusted average ranks (1-based midranks) for a slice.
pub(crate) fn average_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| values[a].total_cmp(&values[b]));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && values[order[j]] == values[order[i]] {
            j += 1;
        }
        // 1-based midrank for indices i..j is (i + 1 + j) / 2.
        let avg_rank = 0.5 * ((i + 1) as f64 + j as f64);
        for &idx in &order[i..j] {
            ranks[idx] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Sample Pearson correlation between two equal-length slices.
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 || n != y.len() {
        return f64::NAN;
    }
    let nf = n as f64;
    let mx = x.iter().sum::<f64>() / nf;
    let my = y.iter().sum::<f64>() / nf;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for (&a, &b) in x.iter().zip(y.iter()) {
        let dx = a - mx;
        let dy = b - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let denom = (sxx * syy).sqrt();
    if denom > 0.0 {
        (sxy / denom).clamp(-1.0, 1.0)
    } else {
        f64::NAN
    }
}

/// Tie-adjusted Spearman rank correlation between two equal-length slices.
pub fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() < 2 || x.len() != y.len() {
        return f64::NAN;
    }
    if x.iter().any(|v| v.is_nan()) || y.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let rx = average_ranks(x);
    let ry = average_ranks(y);
    pearson_correlation(&rx, &ry)
}

fn correlation_matrix(columns: &[Vec<f64>], spearman: bool) -> Vec<f64> {
    let k = columns.len();
    let mut out = vec![0.0; k * k];
    let ranked: Vec<Vec<f64>> = if spearman {
        columns.iter().map(|c| average_ranks(c)).collect()
    } else {
        Vec::new()
    };
    let slice_at = |idx: usize| -> &[f64] {
        if spearman {
            &ranked[idx]
        } else {
            &columns[idx]
        }
    };

    // Precompute column means and inverse norms once so each pair is a single dot product.
    let mut means = vec![0.0; k];
    let mut inv_norms = vec![0.0; k];
    for i in 0..k {
        let col = slice_at(i);
        let nf = col.len() as f64;
        let m = col.iter().sum::<f64>() / nf;
        means[i] = m;
        let ss: f64 = col.iter().map(|&v| (v - m) * (v - m)).sum();
        inv_norms[i] = if ss > 0.0 { 1.0 / ss.sqrt() } else { f64::NAN };
    }

    for i in 0..k {
        out[i * k + i] = 1.0;
        let ci = slice_at(i);
        let mi = means[i];
        let ni = inv_norms[i];
        for j in (i + 1)..k {
            let cj = slice_at(j);
            let mj = means[j];
            let nj = inv_norms[j];
            let r = if ni.is_finite() && nj.is_finite() {
                let mut sxy = 0.0;
                for (&a, &b) in ci.iter().zip(cj.iter()) {
                    sxy += (a - mi) * (b - mj);
                }
                (sxy * ni * nj).clamp(-1.0, 1.0)
            } else {
                f64::NAN
            };
            out[i * k + j] = r;
            out[j * k + i] = r;
        }
    }
    out
}

/// Solve `U * x = b` where `U` is upper-triangular (represented as `L^T` from
/// lower-triangular `l` of size `dim × dim`).
fn solve_upper_from_lower_transpose(l: &[f64], dim: usize, b: &[f64]) -> Vec<f64> {
    let mut x = vec![0.0; dim];
    for i in (0..dim).rev() {
        let mut sum = b[i];
        for j in (i + 1)..dim {
            // U[i, j] = L[j, i].
            sum -= l[j * dim + i] * x[j];
        }
        x[i] = sum / l[i * dim + i];
    }
    x
}

fn validate_and_convert_correlation_matrix(
    target_corr: &[f64],
    k: usize,
    corr_type: CorrelationType,
) -> Result<Vec<f64>, ValidationError> {
    if target_corr.len() != k * k {
        return Err(ValidationError::ShapeMismatch);
    }
    let mut s = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            let v = target_corr[i * k + j];
            if !v.is_finite() {
                return Err(ValidationError::NotFinite);
            }
            if i == j {
                if (v - 1.0).abs() > 1e-10 {
                    return Err(ValidationError::InvalidParameter(
                        "correlation matrix diagonal entries must equal 1.0",
                    ));
                }
                s[i * k + j] = 1.0;
            } else {
                let sym = target_corr[j * k + i];
                if (v - sym).abs() > 1e-10 {
                    return Err(ValidationError::InvalidParameter(
                        "correlation matrix must be symmetric",
                    ));
                }
                if !(-1.0..=1.0).contains(&v) {
                    return Err(ValidationError::InvalidParameter(
                        "correlation matrix entries must lie in [-1, 1]",
                    ));
                }
                s[i * k + j] = match corr_type {
                    CorrelationType::LatentGaussian => v,
                    CorrelationType::SpearmanTarget => 2.0 * (PI * v / 6.0).sin(),
                    CorrelationType::KendallTarget => (PI * v / 2.0).sin(),
                };
            }
        }
    }
    Ok(s)
}

// ===========================================================================
// Iman–Conover (1982) / Mildenhall (2006) Batch Rank Permutation
// ===========================================================================

/// Apply Iman–Conover (1982) batch rank reordering following Mildenhall (2006,
/// CAS Winter Forum, pp. 43–50).
///
/// * Preserves each column's empirical marginal multiset **exactly**.
/// * Induces the positive-definite target correlation matrix `target_corr`
///   (row-major `K × K`) on van der Waerden normal reference scores without
///   explicitly inverting triangular Cholesky factors (`F · A = C`).
/// * Returns whole-scenario [`ImanConoverResult::permutations`] so callers can
///   move full multi-origin / multi-layer scenario structures coherently.
pub fn iman_conover(
    columns: &[&[f64]],
    target_corr: &[f64],
    corr_type: CorrelationType,
    seed: u64,
) -> Result<ImanConoverResult, ValidationError> {
    let k = columns.len();
    if k < 2 {
        return Err(ValidationError::InvalidParameter(
            "Iman-Conover requires at least 2 variables",
        ));
    }
    let n = columns[0].len();
    if n <= k {
        return Err(ValidationError::InvalidParameter(
            "Iman-Conover requires more scenarios than variables (N > K)",
        ));
    }
    for col in columns {
        if col.len() != n {
            return Err(ValidationError::ShapeMismatch);
        }
        if col.iter().any(|v| !v.is_finite()) {
            return Err(ValidationError::NotFinite);
        }
    }

    let s = validate_and_convert_correlation_matrix(target_corr, k, corr_type)?;
    // S = L_S * L_S^T => upper-triangular C = L_S^T satisfies S = C^T * C.
    let l_s = cholesky_lower(&s, k)?;

    // Build centered, unit-variance van der Waerden normal scores.
    let mut base_scores: Vec<f64> = (0..n)
        .map(|r| inv_normal_cdf((r as f64 + 1.0) / (n as f64 + 1.0)))
        .collect();
    let mean_z = base_scores.iter().sum::<f64>() / n as f64;
    let var_z = base_scores
        .iter()
        .map(|z| (z - mean_z).powi(2))
        .sum::<f64>()
        / (n as f64 - 1.0);
    let sd_z = var_z.sqrt();
    for z in &mut base_scores {
        *z = (*z - mean_z) / sd_z;
    }

    // Construct shuffled N × K score matrix M with positive-definite empirical correlation E.
    let mut rng = DeterministicRng::from_seed(seed);
    let mut m_cols: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut l_e_opt = None;
    for _attempt in 0..16 {
        m_cols.clear();
        for _ in 0..k {
            let mut col = base_scores.clone();
            // Fisher–Yates shuffle.
            for i in (1..n).rev() {
                let j = rng.next_index(i + 1);
                col.swap(i, j);
            }
            m_cols.push(col);
        }
        let e = correlation_matrix(&m_cols, false);
        if let Ok(l_e) = cholesky_lower(&e, k) {
            l_e_opt = Some(l_e);
            break;
        }
    }
    let l_e = l_e_opt.ok_or(ValidationError::InvalidParameter(
        "failed to generate full-rank normal score matrix E",
    ))?;

    // Solve F * A = C column-by-column where F = L_E^T and C = L_S^T.
    // Store A column-major as a_cols[j][i] = A[i, j].
    let mut a_cols: Vec<Vec<f64>> = Vec::with_capacity(k);
    for col_j in 0..k {
        let mut c_col = vec![0.0; k];
        for row_i in 0..=col_j {
            // C[row_i, col_j] = L_S[col_j, row_i].
            c_col[row_i] = l_s[col_j * k + row_i];
        }
        a_cols.push(solve_upper_from_lower_transpose(&l_e, k, &c_col));
    }

    // Form T = M * A and reorder each input column to match the rank order of T[:, col].
    let mut permutations = Vec::with_capacity(k);
    let mut reordered_columns = Vec::with_capacity(k);
    let mut t_col = vec![0.0; n];
    let mut t_order: Vec<usize> = (0..n).collect();
    let mut val_order: Vec<usize> = (0..n).collect();

    for col_j in 0..k {
        let a_col = &a_cols[col_j];
        t_col.fill(0.0);
        for p in 0..k {
            let ap = a_col[p];
            if ap != 0.0 {
                for (dst, &mp) in t_col.iter_mut().zip(m_cols[p].iter()) {
                    *dst += mp * ap;
                }
            }
        }

        for (idx, slot) in t_order.iter_mut().enumerate() {
            *slot = idx;
        }
        t_order.sort_unstable_by(|&a, &b| t_col[a].total_cmp(&t_col[b]));

        let col_vals = columns[col_j];
        for (idx, slot) in val_order.iter_mut().enumerate() {
            *slot = idx;
        }
        val_order.sort_unstable_by(|&a, &b| col_vals[a].total_cmp(&col_vals[b]));
        // Randomly break ties within equal-value runs to avoid inheriting original row order.
        let mut tie_start = 0usize;
        while tie_start < n {
            let v0 = col_vals[val_order[tie_start]];
            let mut tie_end = tie_start + 1;
            while tie_end < n && col_vals[val_order[tie_end]] == v0 {
                tie_end += 1;
            }
            if tie_end - tie_start > 1 {
                for m in (tie_start + 1..tie_end).rev() {
                    let swap_idx = tie_start + rng.next_index(m - tie_start + 1);
                    val_order.swap(m, swap_idx);
                }
            }
            tie_start = tie_end;
        }

        let mut perm = vec![0usize; n];
        let mut reordered = vec![0.0; n];
        for rank_idx in 0..n {
            let dst_row = t_order[rank_idx];
            let src_row = val_order[rank_idx];
            perm[dst_row] = src_row;
            reordered[dst_row] = col_vals[src_row];
        }
        permutations.push(perm);
        reordered_columns.push(reordered);
    }

    let achieved_spearman = correlation_matrix(&reordered_columns, true);
    let achieved_pearson = correlation_matrix(&reordered_columns, false);

    Ok(ImanConoverResult {
        n_scenarios: n,
        n_variables: k,
        permutations,
        reordered_columns,
        effective_latent_correlation: s,
        achieved_spearman,
        achieved_pearson,
    })
}

// ===========================================================================
// Synchronized Multi-Line ODP Bootstrap (Kirschner, Kerley & Isaacs 2002)
// ===========================================================================

/// Run a synchronized multi-line ODP bootstrap across `K >= 2` aligned loss
/// triangles (Kirschner, Kerley & Isaacs 2002; Shapland 2016, §9).
///
/// On each bootstrap draw, one shared triangle coordinate resampling map is
/// drawn and applied identically to every line, pulling each line's own
/// adjusted Pearson residual from the same `(origin, dev)` source location.
/// Requires identical square dimensions and aligned eligible residual pools
/// across all lines; mismatched geometry or missing pairs fail explicitly.
pub fn synchronized_odp_bootstrap(
    triangles: &[TriangleView<'_>],
    tails: &[f64],
    options: OdpBootstrapOptions,
) -> Result<SynchronizedBootstrapResult, ValidationError> {
    let k = triangles.len();
    if k < 2 {
        return Err(ValidationError::InvalidParameter(
            "synchronized ODP bootstrap requires at least 2 triangles",
        ));
    }
    if tails.len() != k || options.iterations == 0 {
        return Err(ValidationError::ShapeMismatch);
    }

    let n0 = triangles[0].n_origins;
    for t in triangles {
        if t.n_origins != n0 || t.n_devs != n0 {
            return Err(ValidationError::ShapeMismatch);
        }
    }

    let mut line_fits = Vec::with_capacity(k);
    let mut pools = Vec::with_capacity(k);
    let mut coord_signature: Option<Vec<(usize, usize)>> = None;

    for (t, &tail) in triangles.iter().zip(tails.iter()) {
        let fit = fit_odp(t, tail, options.monotonicity_policy)?;
        let coords: Vec<(usize, usize)> = match options.residual_adjustment {
            ResidualAdjustment::DegreesOfFreedom => {
                fit.residuals.iter().map(|r| (r.origin, r.dev)).collect()
            }
            ResidualAdjustment::HatMatrixLeverage => fit
                .residuals
                .iter()
                .filter(|r| !r.saturated)
                .map(|r| (r.origin, r.dev))
                .collect(),
        };
        if let Some(ref expected) = coord_signature {
            if &coords != expected {
                return Err(ValidationError::InvalidParameter(
                    "synchronized bootstrap requires identical eligible residual coordinates across lines",
                ));
            }
        } else {
            coord_signature = Some(coords);
        }
        let (pool, _) = build_adjusted_residual_pool(
            &fit,
            options.residual_adjustment,
            options.center_residuals,
        )?;
        line_fits.push(fit);
        pools.push(pool);
    }

    let pool_len = pools[0].len();
    let n_obs = line_fits[0].n_observed;
    let mut rng = DeterministicRng::from_seed(options.seed);

    let mut line_totals: Vec<Vec<f64>> = (0..k)
        .map(|_| Vec::with_capacity(options.iterations))
        .collect();
    let mut aggregate_totals = Vec::with_capacity(options.iterations);
    let mut shared_indices = vec![0usize; n_obs];
    let mut line_residuals = vec![0.0; n_obs];
    let mut scratches: Vec<OdpBootstrapScratch> =
        line_fits.iter().map(OdpBootstrapScratch::for_fit).collect();
    let mut draw_totals = vec![0.0; k];
    let mut failed_iterations = 0usize;

    for _ in 0..options.iterations {
        for slot in &mut shared_indices {
            *slot = rng.next_index(pool_len);
        }
        let mut all_ok = true;
        for line in 0..k {
            for (dst, &pool_idx) in line_residuals.iter_mut().zip(shared_indices.iter()) {
                *dst = pools[line][pool_idx];
            }
            match run_odp_iteration_with_scratch(
                &line_fits[line],
                &line_residuals,
                options.process_distribution,
                &mut rng,
                &mut scratches[line],
            ) {
                Some(origin_res) => {
                    let tot: f64 = origin_res.iter().sum();
                    if tot.is_finite() {
                        draw_totals[line] = tot;
                    } else {
                        all_ok = false;
                        break;
                    }
                }
                None => {
                    all_ok = false;
                    break;
                }
            }
        }
        if all_ok {
            let agg: f64 = draw_totals.iter().sum();
            aggregate_totals.push(agg);
            for (line, &tot) in draw_totals.iter().enumerate() {
                line_totals[line].push(tot);
            }
        } else {
            failed_iterations += 1;
        }
    }

    if aggregate_totals.is_empty() {
        return Err(ValidationError::NotFinite);
    }

    let mut summary_buf = Vec::with_capacity(aggregate_totals.len());
    let mut line_summaries = Vec::with_capacity(k);
    for series in &line_totals {
        line_summaries.push(summarize_with_scratch(series, &mut summary_buf, true)?);
    }
    let aggregate_summary = summarize_with_scratch(&aggregate_totals, &mut summary_buf, true)?;
    let sum_standalone_sd: f64 = line_summaries.iter().map(|s| s.std_dev).sum();
    let std_dev_diversification_benefit = sum_standalone_sd - aggregate_summary.std_dev;
    let achieved_spearman = correlation_matrix(&line_totals, true);
    let achieved_pearson = correlation_matrix(&line_totals, false);

    Ok(SynchronizedBootstrapResult {
        line_fits,
        line_totals,
        aggregate_totals,
        line_summaries,
        aggregate_summary,
        std_dev_diversification_benefit,
        achieved_spearman,
        achieved_pearson,
        failed_iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Triangle;

    #[test]
    fn iman_conover_preserves_marginals_and_induces_target_rank_correlation() {
        let mut rng = DeterministicRng::from_seed(42069);
        let n = 4_000usize;
        // Skewed lognormal marginals (heterogeneous severities).
        let x: Vec<f64> = (0..n)
            .map(|_| (10.0 + 0.8 * rng.next_standard_normal()).exp())
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|_| (11.0 + 1.1 * rng.next_standard_normal()).exp())
            .collect();
        let z: Vec<f64> = (0..n)
            .map(|_| (9.5 + 0.5 * rng.next_standard_normal()).exp())
            .collect();

        #[rustfmt::skip]
        let target = [
            1.0,  0.65, -0.30,
            0.65, 1.0,   0.20,
           -0.30, 0.20,  1.0,
        ];

        let res = iman_conover(
            &[&x, &y, &z],
            &target,
            CorrelationType::SpearmanTarget,
            12345,
        )
        .expect("iman_conover");

        // Exact multiset preservation of every empirical marginal.
        for (orig, reord) in [&x, &y, &z].iter().zip(res.reordered_columns.iter()) {
            let mut a = (*orig).clone();
            let mut b = reord.clone();
            a.sort_by(f64::total_cmp);
            b.sort_by(f64::total_cmp);
            assert_eq!(a, b);
        }

        // Permutation coherence check: reordered[col][r] == orig[col][perm[col][r]].
        for (col, orig) in [&x, &y, &z].iter().enumerate() {
            for r in 0..n {
                assert_eq!(
                    res.reordered_columns[col][r],
                    orig[res.permutations[col][r]]
                );
            }
        }

        // Achieved Spearman rank correlations match target within tight finite-sample tolerance.
        assert!(
            (res.achieved_spearman[1] - 0.65).abs() < 0.025,
            "rho_01={}",
            res.achieved_spearman[1]
        );
        assert!(
            (res.achieved_spearman[2] - (-0.30)).abs() < 0.025,
            "rho_02={}",
            res.achieved_spearman[2]
        );
        assert!(
            (res.achieved_spearman[5] - 0.20).abs() < 0.025,
            "rho_12={}",
            res.achieved_spearman[5]
        );
    }

    #[test]
    fn iman_conover_rejects_non_positive_definite_matrix() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0];
        let c = [3.0, 4.0, 5.0, 6.0, 7.0];
        #[rustfmt::skip]
        let bad = [
            1.0,  0.9,  0.9,
            0.9,  1.0, -0.9,
            0.9, -0.9,  1.0,
        ];
        assert!(matches!(
            iman_conover(&[&a, &b, &c], &bad, CorrelationType::LatentGaussian, 1),
            Err(ValidationError::InvalidParameter(_))
        ));
    }

    #[test]
    fn synchronized_bootstrap_preserves_positive_dependence_on_comonotonic_lines() {
        const NAN: f64 = f64::NAN;
        #[rustfmt::skip]
        let t1 = Triangle::new(5, 5, vec![
            1000.0, 1520.0, 1730.0, 1820.0, 1860.0,
            1150.0, 1690.0, 1980.0, 2060.0,    NAN,
            1320.0, 2090.0, 2320.0,    NAN,    NAN,
            1480.0, 2110.0,    NAN,    NAN,    NAN,
            1650.0,    NAN,    NAN,    NAN,    NAN,
        ]);
        // Identically scaled line -> identical residual signs -> strong positive parameter correlation.
        let t2 = Triangle::new(
            5,
            5,
            t1.data
                .iter()
                .map(|v| if v.is_finite() { v * 2.0 } else { NAN })
                .collect(),
        );
        let opts = OdpBootstrapOptions::shapland(400, 42069);
        let sync = synchronized_odp_bootstrap(&[t1.view(), t2.view()], &[1.0, 1.0], opts)
            .expect("synchronized bootstrap");

        assert!(
            sync.achieved_spearman[1] > 0.30,
            "spearman={}",
            sync.achieved_spearman[1]
        );
        assert!(sync.std_dev_diversification_benefit >= 0.0);
    }
}
