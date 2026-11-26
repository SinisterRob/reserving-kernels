//! Pure-Rust dense linear algebra helpers shared across MLE, GLM/IRLS, and
//! regression reserving kernels.
//!
//! All routines operate on row-major `&[f64]` / `Vec<f64>` buffers with zero
//! unsafe code and zero external linear-algebra dependencies, keeping the crate
//! WASM-compatible and deterministic across platforms.

use crate::validate::ValidationError;

/// Compute the lower-triangular Cholesky factor `L` of symmetric positive-definite
/// `dim × dim` matrix `a` (`A = L L^T`).
pub(crate) fn cholesky_lower(a: &[f64], dim: usize) -> Result<Vec<f64>, ValidationError> {
    if a.len() != dim * dim || dim == 0 {
        return Err(ValidationError::ShapeMismatch);
    }
    let mut l = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..=i {
            let mut sum = a[i * dim + j];
            for k in 0..j {
                sum -= l[i * dim + k] * l[j * dim + k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 1e-14 {
                    return Err(ValidationError::InvalidParameter(
                        "matrix is not positive definite",
                    ));
                }
                l[i * dim + j] = sum.sqrt();
            } else {
                l[i * dim + j] = sum / l[j * dim + j];
            }
        }
    }
    Ok(l)
}

/// Solve lower-triangular `L y = b` by forward substitution.
pub(crate) fn solve_lower(l: &[f64], dim: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; dim];
    for i in 0..dim {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * dim + k] * y[k];
        }
        y[i] = sum / l[i * dim + i];
    }
    y
}

/// Solve upper-triangular `L^T x = y` by backward substitution given lower-triangular `L`.
pub(crate) fn solve_lower_transpose(l: &[f64], dim: usize, y: &[f64]) -> Vec<f64> {
    let mut x = vec![0.0; dim];
    for i in (0..dim).rev() {
        let mut sum = y[i];
        for k in (i + 1)..dim {
            sum -= l[k * dim + i] * x[k];
        }
        x[i] = sum / l[i * dim + i];
    }
    x
}

/// Invert a symmetric positive-definite `dim × dim` matrix `A` via Cholesky factorization.
pub(crate) fn invert_spd(a: &[f64], dim: usize) -> Result<Vec<f64>, ValidationError> {
    let l = cholesky_lower(a, dim)?;
    let mut inv = vec![0.0; dim * dim];
    let mut rhs = vec![0.0; dim];
    for col in 0..dim {
        rhs.fill(0.0);
        rhs[col] = 1.0;
        let y = solve_lower(&l, dim, &rhs);
        let x = solve_lower_transpose(&l, dim, &y);
        for row in 0..dim {
            inv[row * dim + col] = x[row];
        }
    }
    // Enforce exact symmetry against floating-point rounding drift.
    for i in 0..dim {
        for j in (i + 1)..dim {
            let avg = 0.5 * (inv[i * dim + j] + inv[j * dim + i]);
            inv[i * dim + j] = avg;
            inv[j * dim + i] = avg;
        }
    }
    Ok(inv)
}

/// Solve a general `dim × dim` linear system `A x = b` via Gaussian elimination
/// with partial pivoting.
pub(crate) fn solve_general(a: &[f64], dim: usize, b: &[f64]) -> Result<Vec<f64>, ValidationError> {
    if a.len() != dim * dim || b.len() != dim || dim == 0 {
        return Err(ValidationError::ShapeMismatch);
    }
    let mut mat = a.to_vec();
    let mut rhs = b.to_vec();

    for col in 0..dim {
        let mut pivot_row = col;
        let mut max_abs = mat[col * dim + col].abs();
        for row in (col + 1)..dim {
            let candidate = mat[row * dim + col].abs();
            if candidate > max_abs {
                max_abs = candidate;
                pivot_row = row;
            }
        }
        if !max_abs.is_finite() || max_abs <= 1e-14 {
            return Err(ValidationError::InvalidParameter(
                "linear system is singular or rank-deficient",
            ));
        }
        if pivot_row != col {
            for k in col..dim {
                mat.swap(col * dim + k, pivot_row * dim + k);
            }
            rhs.swap(col, pivot_row);
        }
        let pivot = mat[col * dim + col];
        for row in (col + 1)..dim {
            let factor = mat[row * dim + col] / pivot;
            mat[row * dim + col] = 0.0;
            for k in (col + 1)..dim {
                mat[row * dim + k] -= factor * mat[col * dim + k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    let mut x = vec![0.0; dim];
    for row in (0..dim).rev() {
        let mut sum = rhs[row];
        for k in (row + 1)..dim {
            sum -= mat[row * dim + k] * x[k];
        }
        let diag = mat[row * dim + row];
        if !diag.is_finite() || diag.abs() <= 1e-14 {
            return Err(ValidationError::InvalidParameter(
                "linear system is singular or rank-deficient",
            ));
        }
        x[row] = sum / diag;
    }
    Ok(x)
}

/// Output of a Householder QR weighted least squares solve.
#[derive(Debug, Clone)]
pub(crate) struct WlsQrSolution {
    /// Regression coefficient vector `beta` (length `p`).
    pub beta: Vec<f64>,
    /// Unscaled covariance matrix `(X^T W X)^{-1}` (`p × p` row-major).
    pub xtwx_inv: Vec<f64>,
    /// Hat-matrix diagonal leverages `h_i = w_i x_i^T (X^T W X)^{-1} x_i` (length `n`).
    pub leverages: Vec<f64>,
}

/// Solve weighted least squares `min_beta sum_i w_i (y_i - x_i^T beta)^2` via
/// Householder QR factorization of `W^{1/2} X`, avoiding normal-equation
/// condition-number squaring.
///
/// * `x` — row-major `n × p` design matrix.
/// * `y` — response vector of length `n`.
/// * `weights` — strictly positive weights of length `n`.
pub(crate) fn weighted_least_squares_qr(
    x: &[f64],
    y: &[f64],
    weights: &[f64],
    n: usize,
    p: usize,
) -> Result<WlsQrSolution, ValidationError> {
    if n == 0 || p == 0 || p > n || x.len() != n * p || y.len() != n || weights.len() != n {
        return Err(ValidationError::ShapeMismatch);
    }

    let mut a = vec![0.0; n * p];
    let mut b = vec![0.0; n];
    let mut sqrt_w = vec![0.0; n];
    for i in 0..n {
        let w = weights[i];
        if !w.is_finite() || w < 0.0 || !y[i].is_finite() {
            return Err(ValidationError::InvalidParameter(
                "WLS requires finite non-negative weights and finite responses",
            ));
        }
        let sw = w.sqrt();
        sqrt_w[i] = sw;
        b[i] = sw * y[i];
        for j in 0..p {
            let xij = x[i * p + j];
            if !xij.is_finite() {
                return Err(ValidationError::NotFinite);
            }
            a[i * p + j] = sw * xij;
        }
    }

    // In-place Householder QR: transforms `a` into upper-triangular `R` in its
    // top `p` rows and transforms `b` into `Q^T b`.
    let mut v = vec![0.0; n];
    for k in 0..p {
        let mut norm_sq = 0.0;
        for i in k..n {
            let val = a[i * p + k];
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt();
        if !norm.is_finite() || norm <= 1e-13 {
            return Err(ValidationError::InvalidParameter(
                "design matrix is rank-deficient or collinear",
            ));
        }
        let x0 = a[k * p + k];
        let sign = if x0 >= 0.0 { 1.0 } else { -1.0 };
        let alpha = -sign * norm;

        let mut v_norm_sq = 0.0;
        for i in k..n {
            let vi = if i == k { x0 - alpha } else { a[i * p + k] };
            v[i] = vi;
            v_norm_sq += vi * vi;
        }
        if v_norm_sq <= 1e-30 {
            continue;
        }
        let tau = 2.0 / v_norm_sq;

        // Apply reflector H = I - tau v v^T to remaining columns k..p.
        a[k * p + k] = alpha;
        for i in (k + 1)..n {
            a[i * p + k] = 0.0;
        }
        for j in (k + 1)..p {
            let mut dot = 0.0;
            for i in k..n {
                dot += v[i] * a[i * p + j];
            }
            let scale = tau * dot;
            for i in k..n {
                a[i * p + j] -= scale * v[i];
            }
        }
        // Apply reflector to RHS `b`.
        let mut dot_b = 0.0;
        for i in k..n {
            dot_b += v[i] * b[i];
        }
        let scale_b = tau * dot_b;
        for i in k..n {
            b[i] -= scale_b * v[i];
        }
    }

    // Check R conditioning / rank deficiency.
    let mut max_diag = 0.0_f64;
    let mut min_diag = f64::INFINITY;
    for k in 0..p {
        let d = a[k * p + k].abs();
        if !d.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        max_diag = max_diag.max(d);
        min_diag = min_diag.min(d);
    }
    if min_diag <= 1e-12 * max_diag.max(1.0) {
        return Err(ValidationError::InvalidParameter(
            "design matrix is rank-deficient or collinear",
        ));
    }

    // Back-substitute R * beta = (Q^T b)[0..p].
    let mut beta = vec![0.0; p];
    for i in (0..p).rev() {
        let mut sum = b[i];
        for j in (i + 1)..p {
            sum -= a[i * p + j] * beta[j];
        }
        beta[i] = sum / a[i * p + i];
    }

    // Invert upper-triangular R to obtain R^{-1} and (X^T W X)^{-1} = R^{-1} R^{-T}.
    let mut r_inv = vec![0.0; p * p];
    for col in 0..p {
        for row in (0..=col).rev() {
            let mut sum = if row == col { 1.0 } else { 0.0 };
            for k in (row + 1)..=col {
                sum -= a[row * p + k] * r_inv[k * p + col];
            }
            r_inv[row * p + col] = sum / a[row * p + row];
        }
    }

    let mut xtwx_inv = vec![0.0; p * p];
    for i in 0..p {
        for j in i..p {
            let mut acc = 0.0;
            let k_start = i.max(j);
            for k in k_start..p {
                acc += r_inv[i * p + k] * r_inv[j * p + k];
            }
            xtwx_inv[i * p + j] = acc;
            xtwx_inv[j * p + i] = acc;
        }
    }

    // Leverages h_i = || (W^{1/2} X)_i R^{-1} ||^2.
    let mut leverages = vec![0.0; n];
    for i in 0..n {
        let sw = sqrt_w[i];
        let mut h_i = 0.0;
        for col in 0..p {
            let mut z = 0.0;
            for k in 0..=col {
                z += sw * x[i * p + k] * r_inv[k * p + col];
            }
            h_i += z * z;
        }
        leverages[i] = h_i.clamp(0.0, 1.0);
    }

    Ok(WlsQrSolution {
        beta,
        xtwx_inv,
        leverages,
    })
}

/// Evaluate quadratic form `v^T M v` for symmetric `dim × dim` matrix `M`.
#[inline]
pub(crate) fn quad_form(v: &[f64], m: &[f64], dim: usize) -> f64 {
    let mut total = 0.0;
    for i in 0..dim {
        let vi = v[i];
        if vi == 0.0 {
            continue;
        }
        let mut row_acc = 0.0;
        for j in 0..dim {
            row_acc += m[i * dim + j] * v[j];
        }
        total += vi * row_acc;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qr_wls_recovers_exact_ols_and_leverages() {
        // y = 2 + 3*x
        let x = vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
        let y = vec![2.0, 5.0, 8.0, 11.0];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        let sol = weighted_least_squares_qr(&x, &y, &w, 4, 2).expect("wls");
        assert!((sol.beta[0] - 2.0).abs() < 1e-12);
        assert!((sol.beta[1] - 3.0).abs() < 1e-12);
        let h_sum: f64 = sol.leverages.iter().sum();
        assert!((h_sum - 2.0).abs() < 1e-12);
    }

    #[test]
    fn spd_invert_produces_identity_product() {
        let a = vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 5.0];
        let inv = invert_spd(&a, 3).expect("inv");
        for i in 0..3 {
            for j in 0..3 {
                let mut prod = 0.0;
                for k in 0..3 {
                    prod += a[i * 3 + k] * inv[k * 3 + j];
                }
                let target = if i == j { 1.0 } else { 0.0 };
                assert!((prod - target).abs() < 1e-12);
            }
        }
    }
}
