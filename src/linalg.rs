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
    for i in 0..dim {
        let d = a[i * dim + i];
        if !d.is_finite() || d <= 0.0 {
            return Err(ValidationError::InvalidParameter(
                "matrix is not positive definite",
            ));
        }
    }
    let mut l = vec![0.0; dim * dim];
    for i in 0..dim {
        // Row-relative tolerance is invariant to diagonal unit rescaling D A D
        // (e.g. mixed-unit Fisher information matrices in Clark MLE).
        let row_tol = 1e-14 * a[i * dim + i];
        for j in 0..=i {
            let mut sum = a[i * dim + j];
            for k in 0..j {
                sum -= l[i * dim + k] * l[j * dim + k];
            }
            if i == j {
                if !sum.is_finite() || sum <= row_tol {
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

/// Solve lower-triangular `L y = b` by forward substitution into `y` (zero allocation).
#[allow(dead_code)]
pub(crate) fn solve_lower_into(l: &[f64], dim: usize, b: &[f64], y: &mut [f64]) {
    for i in 0..dim {
        let mut sum = b[i];
        let row = &l[i * dim..i * dim + i];
        for (lij, &yk) in row.iter().zip(y[..i].iter()) {
            sum -= lij * yk;
        }
        y[i] = sum / l[i * dim + i];
    }
}

/// Solve lower-triangular `L y = b` by forward substitution.
#[allow(dead_code)]
pub(crate) fn solve_lower(l: &[f64], dim: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; dim];
    solve_lower_into(l, dim, b, &mut y);
    y
}

/// Solve upper-triangular `L^T x = y` by backward substitution given lower-triangular `L`.
#[allow(dead_code)]
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
    let mut y = vec![0.0; dim];
    let mut x = vec![0.0; dim];
    for col in 0..dim {
        // Forward substitution for L y = e_col; y[0..col] == 0 identically.
        y[..col].fill(0.0);
        y[col] = 1.0 / l[col * dim + col];
        for i in (col + 1)..dim {
            let mut sum = 0.0;
            for k in col..i {
                sum -= l[i * dim + k] * y[k];
            }
            y[i] = sum / l[i * dim + i];
        }
        // Backward substitution for L^T x = y.
        for i in (0..dim).rev() {
            let mut sum = y[i];
            for k in (i + 1)..dim {
                sum -= l[k * dim + i] * x[k];
            }
            x[i] = sum / l[i * dim + i];
        }
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
#[allow(dead_code)]
pub(crate) fn solve_general(a: &[f64], dim: usize, b: &[f64]) -> Result<Vec<f64>, ValidationError> {
    if a.len() != dim * dim || b.len() != dim || dim == 0 {
        return Err(ValidationError::ShapeMismatch);
    }
    let mut max_entry = 0.0_f64;
    for &v in a {
        if !v.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        max_entry = max_entry.max(v.abs());
    }
    if max_entry == 0.0 {
        return Err(ValidationError::InvalidParameter(
            "linear system is singular or rank-deficient",
        ));
    }
    let piv_tol = 1e-14 * max_entry;
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
        if !max_abs.is_finite() || max_abs <= piv_tol {
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
        if !diag.is_finite() || diag.abs() <= piv_tol {
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

/// Reusable workspace for allocation-free column-major Householder QR WLS solves
/// inside iterative loops such as GLM IRLS.
#[derive(Debug, Default)]
pub(crate) struct QrWorkspace {
    /// Column-major `n × p` weighted matrix (`col j` at `j * n .. (j + 1) * n`).
    a_cm: Vec<f64>,
    b: Vec<f64>,
    sqrt_w: Vec<f64>,
    v: Vec<f64>,
}

impl QrWorkspace {
    pub(crate) fn with_capacity(n: usize, p: usize) -> Self {
        Self {
            a_cm: vec![0.0; n * p],
            b: vec![0.0; n],
            sqrt_w: vec![0.0; n],
            v: vec![0.0; n],
        }
    }
}

/// Factorize `W^{1/2} X` via in-place column-major Householder QR into `ws`.
///
/// Storing `A = W^{1/2} X` column-major makes every Householder dot-product and
/// AXPY update a contiguous slice loop that auto-vectorizes cleanly.
fn qr_factorize_weighted_into(
    x: &[f64],
    y: &[f64],
    weights: &[f64],
    n: usize,
    p: usize,
    ws: &mut QrWorkspace,
) -> Result<(), ValidationError> {
    if n == 0 || p == 0 || p > n || x.len() != n * p || y.len() != n || weights.len() != n {
        return Err(ValidationError::ShapeMismatch);
    }

    if ws.a_cm.len() != n * p {
        ws.a_cm.resize(n * p, 0.0);
    }
    if ws.b.len() != n {
        ws.b.resize(n, 0.0);
    }
    if ws.sqrt_w.len() != n {
        ws.sqrt_w.resize(n, 0.0);
    }
    if ws.v.len() != n {
        ws.v.resize(n, 0.0);
    }

    for i in 0..n {
        let w = weights[i];
        let yi = y[i];
        if !w.is_finite() || w < 0.0 || !yi.is_finite() {
            return Err(ValidationError::InvalidParameter(
                "WLS requires finite non-negative weights and finite responses",
            ));
        }
        let sw = w.sqrt();
        ws.sqrt_w[i] = sw;
        ws.b[i] = sw * yi;
        let x_row = &x[i * p..(i + 1) * p];
        for j in 0..p {
            let xij = x_row[j];
            if !xij.is_finite() {
                return Err(ValidationError::NotFinite);
            }
            ws.a_cm[j * n + i] = sw * xij;
        }
    }

    // In-place column-major Householder QR: `ws.a_cm[j * n + i]` holds R[i, j] for i <= j.
    for k in 0..p {
        let len = n - k;
        let col_k_start = k * n + k;
        let col_k = &ws.a_cm[col_k_start..col_k_start + len];

        let mut norm_sq = 0.0;
        for &val in col_k {
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return Err(ValidationError::InvalidParameter(
                "design matrix is rank-deficient or collinear",
            ));
        }
        let x0 = col_k[0];
        let sign = if x0 >= 0.0 { 1.0 } else { -1.0 };
        let alpha = -sign * norm;

        let v_slice = &mut ws.v[..len];
        v_slice[0] = x0 - alpha;
        let mut v_norm_sq = v_slice[0] * v_slice[0];
        for (dst, &src) in v_slice[1..].iter_mut().zip(col_k[1..].iter()) {
            *dst = src;
            v_norm_sq += src * src;
        }
        if v_norm_sq <= 1e-30 * norm_sq {
            continue;
        }
        let tau = 2.0 / v_norm_sq;

        ws.a_cm[col_k_start] = alpha;
        ws.a_cm[col_k_start + 1..col_k_start + len].fill(0.0);

        // Apply reflector H = I - tau v v^T to remaining columns (k + 1)..p contiguously.
        let v_ref = &ws.v[..len];
        for j in (k + 1)..p {
            let col_j_start = j * n + k;
            let col_j = &mut ws.a_cm[col_j_start..col_j_start + len];
            let mut dot = 0.0;
            for (&vi, &aij) in v_ref.iter().zip(col_j.iter()) {
                dot += vi * aij;
            }
            let scale = tau * dot;
            for (aij, &vi) in col_j.iter_mut().zip(v_ref.iter()) {
                *aij -= scale * vi;
            }
        }

        // Apply reflector to RHS `b[k..n]` contiguously.
        let b_slice = &mut ws.b[k..n];
        let mut dot_b = 0.0;
        for (&vi, &bi) in v_ref.iter().zip(b_slice.iter()) {
            dot_b += vi * bi;
        }
        let scale_b = tau * dot_b;
        for (bi, &vi) in b_slice.iter_mut().zip(v_ref.iter()) {
            *bi -= scale_b * vi;
        }
    }

    // Check R conditioning / rank deficiency (scale-invariant).
    let mut max_diag = 0.0_f64;
    let mut min_diag = f64::INFINITY;
    for k in 0..p {
        let d = ws.a_cm[k * n + k].abs();
        if !d.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        max_diag = max_diag.max(d);
        min_diag = min_diag.min(d);
    }
    if max_diag <= 0.0 || min_diag <= 1e-12 * max_diag {
        return Err(ValidationError::InvalidParameter(
            "design matrix is rank-deficient or collinear",
        ));
    }

    Ok(())
}

/// Zero-allocation Householder QR weighted least squares writing `beta` into `beta_out`.
pub(crate) fn weighted_least_squares_qr_beta_into(
    x: &[f64],
    y: &[f64],
    weights: &[f64],
    n: usize,
    p: usize,
    ws: &mut QrWorkspace,
    beta_out: &mut [f64],
) -> Result<(), ValidationError> {
    qr_factorize_weighted_into(x, y, weights, n, p, ws)?;
    for i in (0..p).rev() {
        let mut sum = ws.b[i];
        for j in (i + 1)..p {
            sum -= ws.a_cm[j * n + i] * beta_out[j];
        }
        beta_out[i] = sum / ws.a_cm[i * n + i];
    }
    Ok(())
}

/// Fast Householder QR weighted least squares returning only `beta` (skipping
/// `R^{-1}`, `(X^T W X)^{-1}`, and hat leverages). Ideal for inner IRLS loops.
#[allow(dead_code)]
pub(crate) fn weighted_least_squares_qr_beta_only(
    x: &[f64],
    y: &[f64],
    weights: &[f64],
    n: usize,
    p: usize,
) -> Result<Vec<f64>, ValidationError> {
    let mut ws = QrWorkspace::with_capacity(n, p);
    let mut beta = vec![0.0; p];
    weighted_least_squares_qr_beta_into(x, y, weights, n, p, &mut ws, &mut beta)?;
    Ok(beta)
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
    let mut ws = QrWorkspace::with_capacity(n, p);
    weighted_least_squares_qr_with_workspace(x, y, weights, n, p, &mut ws)
}

/// Solve weighted least squares via Householder QR reusing `ws`.
pub(crate) fn weighted_least_squares_qr_with_workspace(
    x: &[f64],
    y: &[f64],
    weights: &[f64],
    n: usize,
    p: usize,
    ws: &mut QrWorkspace,
) -> Result<WlsQrSolution, ValidationError> {
    qr_factorize_weighted_into(x, y, weights, n, p, ws)?;

    // Back-substitute R * beta = (Q^T b)[0..p].
    let mut beta = vec![0.0; p];
    for i in (0..p).rev() {
        let mut sum = ws.b[i];
        for j in (i + 1)..p {
            sum -= ws.a_cm[j * n + i] * beta[j];
        }
        beta[i] = sum / ws.a_cm[i * n + i];
    }

    // Invert upper-triangular R (stored column-major in ws.a_cm) to obtain R^{-1}
    // (stored row-major in r_inv) and (X^T W X)^{-1} = R^{-1} R^{-T}.
    let mut r_inv = vec![0.0; p * p];
    for col in 0..p {
        for row in (0..=col).rev() {
            let mut sum = if row == col { 1.0 } else { 0.0 };
            for k in (row + 1)..=col {
                sum -= ws.a_cm[k * n + row] * r_inv[k * p + col];
            }
            r_inv[row * p + col] = sum / ws.a_cm[row * n + row];
        }
    }

    let mut xtwx_inv = vec![0.0; p * p];
    for i in 0..p {
        let r_i = &r_inv[i * p..(i + 1) * p];
        for j in i..p {
            let r_j = &r_inv[j * p..(j + 1) * p];
            let k_start = i.max(j);
            let mut acc = 0.0;
            for (&a_ik, &a_jk) in r_i[k_start..].iter().zip(r_j[k_start..].iter()) {
                acc += a_ik * a_jk;
            }
            xtwx_inv[i * p + j] = acc;
            xtwx_inv[j * p + i] = acc;
        }
    }

    // Column-major copy of R^{-1} for contiguous leverage dot products.
    let mut r_inv_cm = vec![0.0; p * p];
    for col in 0..p {
        for row in 0..=col {
            r_inv_cm[col * p + row] = r_inv[row * p + col];
        }
    }

    // Leverages h_i = w_i * || X_i R^{-1} ||^2.
    let mut leverages = vec![0.0; n];
    for i in 0..n {
        let w_i = ws.sqrt_w[i] * ws.sqrt_w[i];
        let x_row = &x[i * p..(i + 1) * p];
        let mut norm_sq = 0.0;
        for col in 0..p {
            let col_slice = &r_inv_cm[col * p..col * p + col + 1];
            let mut z = 0.0;
            for (&xk, &rk) in x_row[..=col].iter().zip(col_slice.iter()) {
                z += xk * rk;
            }
            norm_sq += z * z;
        }
        leverages[i] = (w_i * norm_sq).clamp(0.0, 1.0);
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
