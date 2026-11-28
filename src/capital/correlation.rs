//! Shared correlation utilities for the capital-modelling dependence layer.
//!
//! This module is the single leaf dependency of both
//! [`crate::capital::copulas`] and [`crate::capital::dependence`]
//! (`correlation <- copulas <- dependence`), so neither of those modules
//! imports the other's internals. It provides:
//!
//! * [`CorrelationType`] target semantics and their conversion to a latent
//!   (Gaussian / elliptical) correlation matrix.
//! * Pearson, tie-adjusted Spearman, and correlation-matrix helpers.
//! * A single (crate-internal) rank-matching reorder helper used by both
//!   Iman–Conover and copula reordering, with one tie-handling policy: both
//!   sorts are total and deterministic (ties ordered by original index), and
//!   runs of equal *values* are additionally shuffled with the caller's seeded
//!   RNG only when the caller exposes source indices (Iman–Conover
//!   permutations). Copula reordering returns values only, for which the
//!   within-tie order is provably irrelevant, so it uses no RNG.
//! * Higham (2002) nearest-correlation-matrix repair via alternating
//!   projections with Dykstra's correction ([`nearest_correlation_matrix`]),
//!   and the explicit [`CorrelationRepair`] policy shared by every sampler that
//!   consumes a target correlation matrix.
//!
//! Reference: N. J. Higham, *"Computing the nearest correlation matrix — a
//! problem from finance"*, IMA Journal of Numerical Analysis 22 (2002),
//! pp. 329–343 (Algorithm 3.3).

use std::f64::consts::PI;

use crate::error::ReservingError;
use crate::linalg::{cholesky_lower, jacobi_eigen_symmetric};
use crate::stochastic::bootstrap_odp::DeterministicRng;

/// Semantic interpretation of a supplied target correlation matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum CorrelationType {
    /// Target matrix is directly the Pearson correlation of latent Gaussian
    /// van der Waerden reference scores (Mildenhall 2006, §3) or the latent
    /// elliptical correlation `Σ` of a Student-$t$ copula.
    #[default]
    LatentGaussian,
    /// Target matrix specifies Spearman rank correlations `ρ_S`, converted via
    /// the **Gaussian-copula** relation `ρ = 2 · sin(π · ρ_S / 6)`.
    ///
    /// This map is exact only for the Gaussian copula (Iman–Conover); it is
    /// rejected by the Student-$t$ copula sampler, whose Spearman–`ρ` relation
    /// depends on `ν`.
    SpearmanTarget,
    /// Target matrix specifies Kendall's tau `τ`, converted via
    /// `ρ = sin(π · τ / 2)` — exact for every elliptical copula (Gaussian and
    /// Student-$t$ alike; Lindskog, McNeil & Schmock 2003).
    KendallTarget,
}

// ===========================================================================
// Pearson / Spearman Helpers
// ===========================================================================

/// Compute tie-adjusted average ranks (1-based midranks) for a slice.
pub(crate) fn average_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| values[a].total_cmp(&values[b]).then(a.cmp(&b)));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && values[order[j]] == values[order[i]] {
            j += 1;
        }
        // 1-based midrank for indices i..j is (i + 1 + j) / 2.
        let avg_rank = f64::midpoint((i + 1) as f64, j as f64);
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

/// Row-major `K × K` Pearson (or Spearman when `spearman`) correlation matrix of `columns`.
pub(crate) fn correlation_matrix(columns: &[Vec<f64>], spearman: bool) -> Vec<f64> {
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

/// Validate a row-major `k × k` target matrix (unit diagonal, symmetric,
/// entries in `[-1, 1]`) and convert it to a latent correlation matrix
/// according to `corr_type`.
pub(crate) fn validate_and_convert_correlation_matrix(
    target_corr: &[f64],
    k: usize,
    corr_type: CorrelationType,
) -> Result<Vec<f64>, ReservingError> {
    if target_corr.len() != k * k {
        return Err(ReservingError::ShapeMismatch);
    }
    let mut s = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            let v = target_corr[i * k + j];
            if !v.is_finite() {
                return Err(ReservingError::NotFinite);
            }
            if i == j {
                if (v - 1.0).abs() > 1e-10 {
                    return Err(ReservingError::InvalidParameter(
                        "correlation matrix diagonal entries must equal 1.0",
                    ));
                }
                s[i * k + j] = 1.0;
            } else {
                let sym = target_corr[j * k + i];
                if (v - sym).abs() > 1e-10 {
                    return Err(ReservingError::InvalidParameter(
                        "correlation matrix must be symmetric",
                    ));
                }
                if !(-1.0..=1.0).contains(&v) {
                    return Err(ReservingError::InvalidParameter(
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
// Rank-Matching Reorder (shared by Iman–Conover and copula reordering)
// ===========================================================================

/// Reorder `values` so that their ranks match the ranks of `scores`.
///
/// Returns `(perm, reordered)` with `reordered[row] = values[perm[row]]`, where
/// the `r`-th smallest score position receives the `r`-th smallest value.
///
/// **Tie policy (single policy for the whole crate):**
///
/// * Both sorts are total and deterministic: ties are ordered by
///   `(value.total_cmp, original index)`, so results never depend on the
///   sort algorithm or platform.
/// * When `tie_rng` is `Some`, each run of *equal values* is additionally
///   shuffled with that seeded RNG (Fisher–Yates). This only changes *which
///   source index* fills a tied slot, never the reordered values themselves,
///   so it matters only when callers consume `perm` (Iman–Conover returns
///   whole-scenario permutations; copula reordering returns values only and
///   therefore passes `None`).
pub(crate) fn rank_match_reorder(
    values: &[f64],
    scores: &[f64],
    tie_rng: Option<&mut DeterministicRng>,
) -> (Vec<usize>, Vec<f64>) {
    let n = values.len().min(scores.len());
    let mut score_order: Vec<usize> = (0..n).collect();
    score_order.sort_unstable_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(a.cmp(&b)));
    let mut value_order: Vec<usize> = (0..n).collect();
    value_order.sort_unstable_by(|&a, &b| values[a].total_cmp(&values[b]).then(a.cmp(&b)));

    if let Some(rng) = tie_rng {
        let mut tie_start = 0usize;
        while tie_start < n {
            let v0 = values[value_order[tie_start]];
            let mut tie_end = tie_start + 1;
            while tie_end < n && values[value_order[tie_end]] == v0 {
                tie_end += 1;
            }
            if tie_end - tie_start > 1 {
                for m in (tie_start + 1..tie_end).rev() {
                    let swap_idx = tie_start + rng.next_index(m - tie_start + 1);
                    value_order.swap(m, swap_idx);
                }
            }
            tie_start = tie_end;
        }
    }

    let mut perm = vec![0usize; n];
    let mut reordered = vec![0.0; n];
    for rank in 0..n {
        let dst = score_order[rank];
        let src = value_order[rank];
        perm[dst] = src;
        reordered[dst] = values[src];
    }
    (perm, reordered)
}

// ===========================================================================
// Higham (2002) Nearest Correlation Matrix
// ===========================================================================

/// Tunables for Higham (2002) alternating-projections nearest-correlation repair.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HighamOptions {
    /// Relative Frobenius convergence tolerance on successive iterates and on
    /// the gap between the PSD and unit-diagonal projections (default `1e-10`).
    pub tolerance: f64,
    /// Maximum number of alternating-projection iterations (default `10_000`).
    pub max_iterations: usize,
    /// Eigenvalue floor `δ >= 0` of the PSD projection `{X : λ_min(X) >= δ}`
    /// (default `1e-8`). `δ > 0` keeps the repaired matrix strictly positive
    /// definite so it admits a Cholesky factor; `δ = 0` is Higham's original
    /// positive-semidefinite projection.
    pub min_eigenvalue: f64,
}

impl Default for HighamOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-10,
            max_iterations: 10_000,
            min_eigenvalue: 1e-8,
        }
    }
}

/// Output of [`nearest_correlation_matrix`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NearestCorrelationResult {
    /// Repaired row-major `dim × dim` correlation matrix (unit diagonal, symmetric).
    pub matrix: Vec<f64>,
    /// Frobenius distance `‖repaired − input‖_F` (the repair magnitude).
    pub frobenius_distance: f64,
    /// Number of alternating-projection iterations performed.
    pub iterations: usize,
}

/// Explicit policy for a target correlation matrix that is not positive definite.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum CorrelationRepair {
    /// Reject non-positive-definite targets with
    /// [`ReservingError::InvalidParameter`] (default for every sampler).
    #[default]
    Reject,
    /// Repair via Higham (2002) nearest correlation matrix and report the
    /// Frobenius repair distance in the sampler output.
    Higham(HighamOptions),
}

fn frobenius_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

fn frobenius_norm(a: &[f64]) -> f64 {
    a.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Project symmetric `r` onto `{X : λ_min(X) >= floor}` by eigenvalue clipping.
fn project_psd(r: &[f64], dim: usize, floor: f64) -> Result<Vec<f64>, ReservingError> {
    let (eigvals, q) = jacobi_eigen_symmetric(r, dim)?;
    let clipped: Vec<f64> = eigvals.iter().map(|&v| v.max(floor)).collect();
    let mut x = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in i..dim {
            let mut sum = 0.0;
            for k in 0..dim {
                sum += q[i * dim + k] * clipped[k] * q[j * dim + k];
            }
            x[i * dim + j] = sum;
            x[j * dim + i] = sum;
        }
    }
    Ok(x)
}

/// Compute the nearest correlation matrix (unweighted Frobenius norm) to the
/// symmetric `dim × dim` matrix `corr` via Higham's (2002, Algorithm 3.3)
/// alternating projections with Dykstra's correction:
///
/// `R_k = Y_{k-1} − ΔS_{k-1}`, `X_k = P_S(R_k)`, `ΔS_k = X_k − R_k`,
/// `Y_k = P_U(X_k)`,
///
/// where `P_S` clips eigenvalues at `options.min_eigenvalue` and `P_U` resets
/// the diagonal to one. Fully deterministic (cyclic Jacobi eigensolver, fixed
/// iteration order). Returns [`ReservingError::InvalidParameter`] when the
/// iteration does not reach `options.tolerance` within
/// `options.max_iterations`.
pub fn nearest_correlation_matrix(
    corr: &[f64],
    dim: usize,
    options: &HighamOptions,
) -> Result<NearestCorrelationResult, ReservingError> {
    if dim == 0 || dim.checked_mul(dim) != Some(corr.len()) {
        return Err(ReservingError::ShapeMismatch);
    }
    if corr.iter().any(|v| !v.is_finite()) {
        return Err(ReservingError::NotFinite);
    }
    if !options.tolerance.is_finite()
        || !(options.tolerance > 0.0)
        || !options.min_eigenvalue.is_finite()
        || options.min_eigenvalue < 0.0
        || options.min_eigenvalue >= 1.0
        || options.max_iterations == 0
    {
        return Err(ReservingError::InvalidParameter(
            "Higham options require tolerance > 0, max_iterations >= 1 and min_eigenvalue in [0, 1)",
        ));
    }

    // Symmetrize the input once; the target A of the projection problem.
    let mut a = vec![0.0; dim * dim];
    for i in 0..dim {
        a[i * dim + i] = corr[i * dim + i];
        for j in (i + 1)..dim {
            let s = f64::midpoint(corr[i * dim + j], corr[j * dim + i]);
            a[i * dim + j] = s;
            a[j * dim + i] = s;
        }
    }

    let mut y = a.clone();
    let mut x_prev = a.clone();
    let mut delta_s = vec![0.0; dim * dim];
    let mut r = vec![0.0; dim * dim];
    for iter in 1..=options.max_iterations {
        for ((ri, &yi), &di) in r.iter_mut().zip(y.iter()).zip(delta_s.iter()) {
            *ri = yi - di;
        }
        let x = project_psd(&r, dim, options.min_eigenvalue)?;
        for ((di, &xi), &ri) in delta_s.iter_mut().zip(x.iter()).zip(r.iter()) {
            *di = xi - ri;
        }
        let y_prev = std::mem::replace(&mut y, x.clone());
        for i in 0..dim {
            y[i * dim + i] = 1.0;
        }

        let norm_x = frobenius_norm(&x).max(f64::MIN_POSITIVE);
        let norm_y = frobenius_norm(&y).max(f64::MIN_POSITIVE);
        let rel_x = frobenius_diff(&x, &x_prev) / norm_x;
        let rel_y = frobenius_diff(&y, &y_prev) / norm_y;
        let rel_xy = frobenius_diff(&y, &x) / norm_y;
        x_prev = x;
        if rel_x.max(rel_y).max(rel_xy) <= options.tolerance {
            // Enforce exact symmetry and clamp to [-1, 1].
            for i in 0..dim {
                for j in (i + 1)..dim {
                    let v = f64::midpoint(y[i * dim + j], y[j * dim + i]).clamp(-1.0, 1.0);
                    y[i * dim + j] = v;
                    y[j * dim + i] = v;
                }
            }
            let frobenius_distance = frobenius_diff(&y, &a);
            return Ok(NearestCorrelationResult {
                matrix: y,
                frobenius_distance,
                iterations: iter,
            });
        }
    }
    Err(ReservingError::InvalidParameter(
        "Higham nearest-correlation iteration did not converge within max_iterations",
    ))
}

/// Repair a symmetric `dim × dim` correlation matrix to the nearest valid
/// positive-definite correlation matrix using Higham (2002) alternating
/// projections with default [`HighamOptions`].
///
/// Use [`nearest_correlation_matrix`] to control tolerances and to obtain the
/// Frobenius repair distance.
pub fn repair_correlation_matrix(corr: &[f64], dim: usize) -> Result<Vec<f64>, ReservingError> {
    nearest_correlation_matrix(corr, dim, &HighamOptions::default()).map(|r| r.matrix)
}

/// Latent correlation prepared for sampling: `(sigma, cholesky_lower, repair_distance)`.
pub(crate) type PreparedCorrelation = (Vec<f64>, Vec<f64>, Option<f64>);

/// Cholesky-factor a latent correlation matrix, applying `repair` when it is
/// not positive definite. Returns the (possibly repaired) matrix, its lower
/// Cholesky factor, and the Frobenius repair distance (`None` if unrepaired).
pub(crate) fn prepare_latent_correlation(
    latent: Vec<f64>,
    dim: usize,
    repair: CorrelationRepair,
) -> Result<PreparedCorrelation, ReservingError> {
    match cholesky_lower(&latent, dim) {
        Ok(chol) => Ok((latent, chol, None)),
        Err(err) => match repair {
            CorrelationRepair::Reject => Err(err),
            CorrelationRepair::Higham(opts) => {
                let repaired = nearest_correlation_matrix(&latent, dim, &opts)?;
                let chol = cholesky_lower(&repaired.matrix, dim)?;
                Ok((repaired.matrix, chol, Some(repaired.frobenius_distance)))
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_match_reorder_matches_score_ranks_and_preserves_multiset() {
        let values = [5.0, 1.0, 3.0, 3.0, 9.0];
        let scores = [0.9, 0.1, 0.5, 0.3, 0.7];
        let (perm, reordered) = rank_match_reorder(&values, &scores, None);
        assert_eq!(reordered, vec![9.0, 1.0, 3.0, 3.0, 5.0]);
        for (row, &src) in perm.iter().enumerate() {
            assert_eq!(reordered[row], values[src]);
        }
    }

    #[test]
    fn higham_leaves_valid_correlation_unchanged() {
        let c = [1.0, 0.3, 0.3, 1.0];
        let res = nearest_correlation_matrix(&c, 2, &HighamOptions::default()).expect("higham");
        assert!(res.frobenius_distance < 1e-12);
        assert_eq!(res.iterations, 1);
    }
}
