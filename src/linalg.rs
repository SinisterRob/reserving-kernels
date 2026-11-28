//! Pure-Rust dense linear algebra helpers shared across MLE, GLM/IRLS, and
//! regression reserving kernels.
//!
//! All routines operate on row-major `&[f64]` / `Vec<f64>` buffers with zero
//! unsafe code and zero external linear-algebra dependencies, keeping the crate
//! WASM-compatible. Results are deterministic run-to-run on a given platform
//! (fixed operation order, no threading); bit-for-bit reproducibility *across*
//! platforms is not guaranteed because transcendental helpers such as
//! `f64::hypot`, `sqrt`-adjacent libm calls and compiler FMA contraction may
//! differ in the last ulp between targets.

use crate::error::ReservingError;

/// Compute the lower-triangular Cholesky factor `L` of symmetric positive-definite
/// `dim × dim` matrix `a` (`A = L L^T`).
pub(crate) fn cholesky_lower(a: &[f64], dim: usize) -> Result<Vec<f64>, ReservingError> {
    if a.len() != dim * dim || dim == 0 {
        return Err(ReservingError::ShapeMismatch);
    }
    for i in 0..dim {
        let d = a[i * dim + i];
        if !d.is_finite() || d <= 0.0 {
            return Err(ReservingError::InvalidParameter(
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
                    return Err(ReservingError::InvalidParameter(
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

/// Invert a symmetric positive-definite `dim × dim` matrix `A` via Cholesky factorization.
///
/// Computes lower-triangular `L^{-1}` directly (O(dim^3 / 6) flops) and forms
/// `A^{-1} = L^{-T} L^{-1}` with contiguous dot products.
pub(crate) fn invert_spd(a: &[f64], dim: usize) -> Result<Vec<f64>, ReservingError> {
    let l = cholesky_lower(a, dim)?;
    // Store L^{-1} column-major (`l_inv_cm[col * dim + row]` for `row >= col`) so
    // the final A^{-1}_{i, j} = sum_{k >= max(i, j)} L^{-1}_{k, i} L^{-1}_{k, j}
    // is a contiguous slice dot product that auto-vectorizes cleanly.
    let mut l_inv_cm = vec![0.0; dim * dim];
    for col in 0..dim {
        let base = col * dim;
        l_inv_cm[base + col] = 1.0 / l[col * dim + col];
        for row in (col + 1)..dim {
            let l_row = &l[row * dim + col..row * dim + row];
            let inv_col = &l_inv_cm[base + col..base + row];
            let mut sum = 0.0;
            for (&lrk, &yk) in l_row.iter().zip(inv_col.iter()) {
                sum -= lrk * yk;
            }
            l_inv_cm[base + row] = sum / l[row * dim + row];
        }
    }

    let mut inv = vec![0.0; dim * dim];
    for i in 0..dim {
        let col_i = &l_inv_cm[i * dim..(i + 1) * dim];
        for j in i..dim {
            let col_j = &l_inv_cm[j * dim..(j + 1) * dim];
            let mut acc = 0.0;
            for (&a_ki, &a_kj) in col_i[j..].iter().zip(col_j[j..].iter()) {
                acc += a_ki * a_kj;
            }
            inv[i * dim + j] = acc;
            inv[j * dim + i] = acc;
        }
    }
    Ok(inv)
}

/// Solve a general `dim × dim` linear system `A x = b` via Gaussian elimination
/// with partial pivoting.
pub(crate) fn solve_general(a: &[f64], dim: usize, b: &[f64]) -> Result<Vec<f64>, ReservingError> {
    if dim == 0 || dim.checked_mul(dim) != Some(a.len()) || b.len() != dim {
        return Err(ReservingError::ShapeMismatch);
    }
    if b.iter().any(|v| !v.is_finite()) {
        return Err(ReservingError::NotFinite);
    }
    let mut row_scale = vec![0.0_f64; dim];
    for r in 0..dim {
        let mut r_max = 0.0_f64;
        for &v in &a[r * dim..(r + 1) * dim] {
            if !v.is_finite() {
                return Err(ReservingError::NotFinite);
            }
            r_max = r_max.max(v.abs());
        }
        if r_max == 0.0 {
            return Err(ReservingError::InvalidParameter(
                "linear system is singular or rank-deficient",
            ));
        }
        row_scale[r] = r_max;
    }
    let mut mat = a.to_vec();
    let mut rhs = b.to_vec();

    for col in 0..dim {
        let mut pivot_row = col;
        let mut max_scaled = mat[col * dim + col].abs() / row_scale[col];
        for row in (col + 1)..dim {
            let candidate = mat[row * dim + col].abs() / row_scale[row];
            if candidate > max_scaled {
                max_scaled = candidate;
                pivot_row = row;
            }
        }
        if !max_scaled.is_finite() || max_scaled <= 1e-14 {
            return Err(ReservingError::InvalidParameter(
                "linear system is singular or rank-deficient",
            ));
        }
        if pivot_row != col {
            for k in col..dim {
                mat.swap(col * dim + k, pivot_row * dim + k);
            }
            rhs.swap(col, pivot_row);
            row_scale.swap(col, pivot_row);
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
        if !diag.is_finite() || diag.abs() <= 1e-14 * row_scale[row] {
            return Err(ReservingError::InvalidParameter(
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
    #[allow(dead_code)]
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
    /// Stored Householder scaling factors `tau_k` for constant-weight QR reuse.
    tau: Vec<f64>,
    /// Row-major `p × p` copy of upper-triangular `R` for contiguous back-sub / `R^{-1}`.
    r_rm: Vec<f64>,
    /// Column-major `p × p` inverse `R^{-1}`.
    r_inv_cm: Vec<f64>,
}

impl QrWorkspace {
    pub(crate) fn with_capacity(n: usize, p: usize) -> Self {
        Self {
            a_cm: vec![0.0; n * p],
            b: vec![0.0; n],
            sqrt_w: vec![0.0; n],
            v: vec![0.0; n],
            tau: vec![0.0; p],
            r_rm: vec![0.0; p * p],
            r_inv_cm: vec![0.0; p * p],
        }
    }

    #[inline]
    fn ensure_dims(&mut self, n: usize, p: usize) {
        if self.a_cm.len() != n * p {
            self.a_cm.resize(n * p, 0.0);
        }
        if self.b.len() != n {
            self.b.resize(n, 0.0);
        }
        if self.sqrt_w.len() != n {
            self.sqrt_w.resize(n, 0.0);
        }
        if self.v.len() != n {
            self.v.resize(n, 0.0);
        }
        if self.tau.len() != p {
            self.tau.resize(p, 0.0);
        }
    }
}

/// Core in-place column-major Householder QR factorization of `ws.a_cm` (`n × p`),
/// optionally applying reflectors to `ws.b` (`apply_rhs = true`) and optionally
/// storing normalized reflectors in the lower triangle of `ws.a_cm` (`store_reflectors = true`).
fn qr_factorize_a_cm_in_place(
    n: usize,
    p: usize,
    ws: &mut QrWorkspace,
    apply_rhs: bool,
    store_reflectors: bool,
) -> Result<(), ReservingError> {
    for k in 0..p {
        let len = n - k;
        let col_k_start = k * n + k;
        let col_k = &ws.a_cm[col_k_start..col_k_start + len];

        // Scale by column infinity norm before squaring to prevent overflow (> 1e154)
        // or underflow (< 1e-154) on extreme-magnitude design matrices.
        let mut max_abs = 0.0_f64;
        for &val in col_k {
            max_abs = max_abs.max(val.abs());
        }
        if !max_abs.is_finite() || max_abs <= 0.0 {
            return Err(ReservingError::InvalidParameter(
                "design matrix is rank-deficient or collinear",
            ));
        }
        let inv_scale = 1.0 / max_abs;
        let mut scaled_norm_sq = 0.0;
        for &val in col_k {
            let s = val * inv_scale;
            scaled_norm_sq += s * s;
        }
        let norm = max_abs * scaled_norm_sq.sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return Err(ReservingError::InvalidParameter(
                "design matrix is rank-deficient or collinear",
            ));
        }
        let x0 = col_k[0];
        let sign = if x0 >= 0.0 { 1.0 } else { -1.0 };
        let alpha = -sign * norm;

        let v_slice = &mut ws.v[..len];
        let mut sub_norm_sq = 0.0;
        for (dst, &src) in v_slice[1..].iter_mut().zip(col_k[1..].iter()) {
            let s = src * inv_scale;
            *dst = s;
            sub_norm_sq += s * s;
        }
        if sub_norm_sq <= 1e-30 * scaled_norm_sq {
            ws.a_cm[col_k_start] = x0;
            if store_reflectors {
                ws.tau[k] = 0.0;
                ws.a_cm[col_k_start + 1..col_k_start + len].fill(0.0);
            }
            continue;
        }
        v_slice[0] = (x0 - alpha) * inv_scale;
        let v_norm_sq = v_slice[0] * v_slice[0] + sub_norm_sq;
        let tau = 2.0 / v_norm_sq;

        ws.a_cm[col_k_start] = alpha;
        if store_reflectors {
            let inv_v0 = 1.0 / v_slice[0];
            ws.tau[k] = tau * v_slice[0] * v_slice[0];
            for (dst, &src) in ws.a_cm[col_k_start + 1..col_k_start + len]
                .iter_mut()
                .zip(v_slice[1..].iter())
            {
                *dst = src * inv_v0;
            }
        } else {
            ws.a_cm[col_k_start + 1..col_k_start + len].fill(0.0);
        }

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

        if apply_rhs {
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
    }

    // Check R conditioning / rank deficiency (scale-invariant).
    let mut max_diag = 0.0_f64;
    let mut min_diag = f64::INFINITY;
    for k in 0..p {
        let d = ws.a_cm[k * n + k].abs();
        if !d.is_finite() {
            return Err(ReservingError::NotFinite);
        }
        max_diag = max_diag.max(d);
        min_diag = min_diag.min(d);
    }
    if max_diag <= 0.0 || min_diag <= 1e-12 * max_diag {
        return Err(ReservingError::InvalidParameter(
            "design matrix is rank-deficient or collinear",
        ));
    }

    Ok(())
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
) -> Result<(), ReservingError> {
    if n == 0 || p == 0 || p > n || x.len() != n * p || y.len() != n || weights.len() != n {
        return Err(ReservingError::ShapeMismatch);
    }

    ws.ensure_dims(n, p);

    for i in 0..n {
        let w = weights[i];
        let yi = y[i];
        if !w.is_finite() || w < 0.0 || !yi.is_finite() {
            return Err(ReservingError::InvalidParameter(
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
                return Err(ReservingError::NotFinite);
            }
            ws.a_cm[j * n + i] = sw * xij;
        }
    }

    qr_factorize_a_cm_in_place(n, p, ws, true, false)
}

/// Factorize `W^{1/2} X` from a pre-validated column-major design matrix `x_cm`
/// (`col j` at `j * n .. (j + 1) * n`), making the scaling step a contiguous SIMD pass.
fn qr_factorize_weighted_cm_into(
    x_cm: &[f64],
    y: Option<&[f64]>,
    weights: &[f64],
    n: usize,
    p: usize,
    ws: &mut QrWorkspace,
) -> Result<(), ReservingError> {
    if n == 0 || p == 0 || p > n || x_cm.len() != n * p || weights.len() != n {
        return Err(ReservingError::ShapeMismatch);
    }
    if let Some(yv) = y
        && yv.len() != n
    {
        return Err(ReservingError::ShapeMismatch);
    }

    ws.ensure_dims(n, p);

    if let Some(yv) = y {
        for i in 0..n {
            let w = weights[i];
            let yi = yv[i];
            if !w.is_finite() || w < 0.0 || !yi.is_finite() {
                return Err(ReservingError::InvalidParameter(
                    "WLS requires finite non-negative weights and finite responses",
                ));
            }
            let sw = w.sqrt();
            ws.sqrt_w[i] = sw;
            ws.b[i] = sw * yi;
        }
    } else {
        for i in 0..n {
            let w = weights[i];
            if !w.is_finite() || w < 0.0 {
                return Err(ReservingError::InvalidParameter(
                    "WLS requires finite non-negative weights",
                ));
            }
            ws.sqrt_w[i] = w.sqrt();
        }
    }

    let sw = &ws.sqrt_w[..n];
    for j in 0..p {
        let src = &x_cm[j * n..(j + 1) * n];
        let dst = &mut ws.a_cm[j * n..(j + 1) * n];
        for ((d, &s), &w) in dst.iter_mut().zip(src.iter()).zip(sw.iter()) {
            *d = w * s;
        }
    }

    qr_factorize_a_cm_in_place(n, p, ws, y.is_some(), false)
}

/// Factorize unweighted `X` (column-major `x_cm`, pre-validated finite) once and
/// retain its compact Householder reflectors in `ws` for repeated `O(n p)` solves
/// (ideal for constant-weight Gamma IRLS and LogNormal OLS).
pub(crate) fn qr_factorize_unit_cm_store_reflectors(
    x_cm: &[f64],
    n: usize,
    p: usize,
    ws: &mut QrWorkspace,
) -> Result<(), ReservingError> {
    if n == 0 || p == 0 || p > n || x_cm.len() != n * p {
        return Err(ReservingError::ShapeMismatch);
    }
    ws.ensure_dims(n, p);
    ws.sqrt_w.fill(1.0);
    ws.a_cm.copy_from_slice(x_cm);
    qr_factorize_a_cm_in_place(n, p, ws, false, true)
}

/// Solve `min_beta || X beta - y ||_2^2` in `O(n p)` reusing reflectors stored by
/// [`qr_factorize_unit_cm_store_reflectors`].
pub(crate) fn qr_solve_stored_unit_into(
    y: &[f64],
    n: usize,
    p: usize,
    ws: &mut QrWorkspace,
    beta_out: &mut [f64],
) -> Result<(), ReservingError> {
    if y.len() != n || beta_out.len() != p {
        return Err(ReservingError::ShapeMismatch);
    }
    for (dst, &yi) in ws.b[..n].iter_mut().zip(y.iter()) {
        if !yi.is_finite() {
            return Err(ReservingError::InvalidParameter(
                "WLS requires finite responses",
            ));
        }
        *dst = yi;
    }

    for k in 0..p {
        let tau = ws.tau[k];
        if tau == 0.0 {
            continue;
        }
        let tail_len = n - k - 1;
        let col_tail = &ws.a_cm[k * n + k + 1..k * n + k + 1 + tail_len];
        let b_slice = &mut ws.b[k..n];
        let mut dot = b_slice[0];
        for (&vi, &bi) in col_tail.iter().zip(b_slice[1..].iter()) {
            dot += vi * bi;
        }
        let scale = tau * dot;
        b_slice[0] -= scale;
        for (bi, &vi) in b_slice[1..].iter_mut().zip(col_tail.iter()) {
            *bi -= scale * vi;
        }
    }

    for i in (0..p).rev() {
        let mut sum = ws.b[i];
        for j in (i + 1)..p {
            sum -= ws.a_cm[j * n + i] * beta_out[j];
        }
        beta_out[i] = sum / ws.a_cm[i * n + i];
    }
    Ok(())
}

/// Extract `(X^T W X)^{-1} = R^{-1} R^{-T}` and hat-matrix leverages from an
/// already-factored upper-triangular `R` stored in `ws.a_cm` (with `ws.sqrt_w` populated).
pub(crate) fn qr_cov_and_leverages_from_factored(
    x_row_major: &[f64],
    n: usize,
    p: usize,
    ws: &mut QrWorkspace,
) -> (Vec<f64>, Vec<f64>) {
    // Copy upper triangle of R into contiguous row-major `ws.r_rm` so `R^{-1}`
    // column back-substitutions use contiguous inner dot products.
    if ws.r_rm.len() != p * p {
        ws.r_rm.resize(p * p, 0.0);
    }
    if ws.r_inv_cm.len() != p * p {
        ws.r_inv_cm.resize(p * p, 0.0);
    }
    for row in 0..p {
        let r_dst = &mut ws.r_rm[row * p..row * p + p];
        for col in row..p {
            r_dst[col] = ws.a_cm[col * n + row];
        }
    }

    // Invert upper-triangular R directly into column-major `ws.r_inv_cm`.
    ws.r_inv_cm.fill(0.0);
    for col in 0..p {
        let inv_col_base = col * p;
        ws.r_inv_cm[inv_col_base + col] = 1.0 / ws.r_rm[col * p + col];
        for row in (0..col).rev() {
            let r_row = &ws.r_rm[(row * p + row + 1)..=(row * p + col)];
            let inv_col = &ws.r_inv_cm[(inv_col_base + row + 1)..=(inv_col_base + col)];
            let mut sum = 0.0;
            for (&rik, &yk) in r_row.iter().zip(inv_col.iter()) {
                sum -= rik * yk;
            }
            ws.r_inv_cm[inv_col_base + row] = sum / ws.r_rm[row * p + row];
        }
    }

    // Form (X^T W X)^{-1} = R^{-1} R^{-T} = sum_{k=0}^{p-1} c_k c_k^T using
    // contiguous column slices `c_k` of `ws.r_inv_cm` (upper triangle, then mirror).
    let mut xtwx_inv = vec![0.0; p * p];
    for k in 0..p {
        let c_k = &ws.r_inv_cm[(k * p)..=(k * p + k)];
        for i in 0..=k {
            let ci = c_k[i];
            let dst = &mut xtwx_inv[(i * p + i)..=(i * p + k)];
            for (d, &cj) in dst.iter_mut().zip(c_k[i..].iter()) {
                *d += ci * cj;
            }
        }
    }
    for i in 0..p {
        for j in (i + 1)..p {
            xtwx_inv[j * p + i] = xtwx_inv[i * p + j];
        }
    }

    // Leverages h_i = w_i * || X_i R^{-1} ||^2.
    let mut leverages = vec![0.0; n];
    for i in 0..n {
        let w_i = ws.sqrt_w[i] * ws.sqrt_w[i];
        let x_row = &x_row_major[i * p..(i + 1) * p];
        let mut norm_sq = 0.0;
        for col in 0..p {
            let col_slice = &ws.r_inv_cm[(col * p)..=(col * p + col)];
            let mut z = 0.0;
            for (&xk, &rk) in x_row[..=col].iter().zip(col_slice.iter()) {
                z += xk * rk;
            }
            norm_sq += z * z;
        }
        leverages[i] = (w_i * norm_sq).clamp(0.0, 1.0);
    }

    (xtwx_inv, leverages)
}

/// Zero-allocation Householder QR weighted least squares from pre-validated
/// column-major `x_cm`, writing `beta` into `beta_out`.
pub(crate) fn weighted_least_squares_qr_cm_beta_into(
    x_cm: &[f64],
    y: &[f64],
    weights: &[f64],
    n: usize,
    p: usize,
    ws: &mut QrWorkspace,
    beta_out: &mut [f64],
) -> Result<(), ReservingError> {
    if beta_out.len() != p {
        return Err(ReservingError::ShapeMismatch);
    }
    qr_factorize_weighted_cm_into(x_cm, Some(y), weights, n, p, ws)?;
    for i in (0..p).rev() {
        let mut sum = ws.b[i];
        for j in (i + 1)..p {
            sum -= ws.a_cm[j * n + i] * beta_out[j];
        }
        beta_out[i] = sum / ws.a_cm[i * n + i];
    }
    Ok(())
}

/// Factorize `W^{1/2} X` (from column-major `x_cm`) and return `(X^T W X)^{-1}`
/// and hat-matrix leverages without computing an unused `beta`.
pub(crate) fn weighted_least_squares_qr_cm_cov_and_leverages(
    x_row_major: &[f64],
    x_cm: &[f64],
    weights: &[f64],
    n: usize,
    p: usize,
    ws: &mut QrWorkspace,
) -> Result<(Vec<f64>, Vec<f64>), ReservingError> {
    qr_factorize_weighted_cm_into(x_cm, None, weights, n, p, ws)?;
    Ok(qr_cov_and_leverages_from_factored(x_row_major, n, p, ws))
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
) -> Result<(), ReservingError> {
    if beta_out.len() != p {
        return Err(ReservingError::ShapeMismatch);
    }
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
pub(crate) fn weighted_least_squares_qr_beta_only(
    x: &[f64],
    y: &[f64],
    weights: &[f64],
    n: usize,
    p: usize,
) -> Result<Vec<f64>, ReservingError> {
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
) -> Result<WlsQrSolution, ReservingError> {
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
) -> Result<WlsQrSolution, ReservingError> {
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

    let (xtwx_inv, leverages) = qr_cov_and_leverages_from_factored(x, n, p, ws);

    Ok(WlsQrSolution {
        beta,
        xtwx_inv,
        leverages,
    })
}

/// Evaluate quadratic form `v^T M v` for symmetric `dim × dim` matrix `M`.
///
/// Exploits symmetry (`sum_i v_i * (M_{i,i} v_i + 2 * sum_{j > i} M_{i,j} v_j)`)
/// to halve work while keeping the inner slice contiguous and auto-vectorizable.
#[inline]
pub(crate) fn quad_form(v: &[f64], m: &[f64], dim: usize) -> f64 {
    let mut total = 0.0;
    for i in 0..dim {
        let vi = v[i];
        if vi == 0.0 {
            continue;
        }
        let row = &m[i * dim..i * dim + dim];
        let mut upper = 0.0;
        for (&mij, &vj) in row[i + 1..].iter().zip(v[i + 1..].iter()) {
            upper += mij * vj;
        }
        total += vi * (row[i] * vi + 2.0 * upper);
    }
    total
}

/// Compute all eigenvalues and orthonormal eigenvectors of a real symmetric
/// `dim × dim` matrix `A` via cyclic Jacobi rotations (`A = Q diag(lambda) Q^T`).
///
/// Returns `(eigenvalues, q)` where `q` is row-major `dim × dim` with column `k`
/// (`q[row * dim + k]`) holding the unit eigenvector for `eigenvalues[k]`.
/// Returns [`ReservingError::InvalidParameter`] if the off-diagonal mass does
/// not fall below `1e-13 * max|a_ij|` within `64 * dim` sweeps (never returns
/// an unconverged decomposition silently).
pub(crate) fn jacobi_eigen_symmetric(
    a: &[f64],
    dim: usize,
) -> Result<(Vec<f64>, Vec<f64>), ReservingError> {
    if dim == 0 || dim.checked_mul(dim) != Some(a.len()) {
        return Err(ReservingError::ShapeMismatch);
    }
    let mut scale = 0.0_f64;
    for &v in a {
        if !v.is_finite() {
            return Err(ReservingError::NotFinite);
        }
        scale = scale.max(v.abs());
    }
    let mut mat = vec![0.0; dim * dim];
    for i in 0..dim {
        mat[i * dim + i] = a[i * dim + i];
        for j in (i + 1)..dim {
            let sym = f64::midpoint(a[i * dim + j], a[j * dim + i]);
            mat[i * dim + j] = sym;
            mat[j * dim + i] = sym;
        }
    }
    let mut q = vec![0.0; dim * dim];
    for i in 0..dim {
        q[i * dim + i] = 1.0;
    }
    if scale == 0.0 {
        return Ok((vec![0.0; dim], q));
    }

    let off_tol = 1e-13 * scale;
    let pivot_tol = 1e-15 * scale;
    let max_sweeps = 64 * dim.max(1);
    let max_offdiag = |m: &[f64]| {
        let mut max_off = 0.0_f64;
        for p in 0..dim {
            for r in (p + 1)..dim {
                max_off = max_off.max(m[p * dim + r].abs());
            }
        }
        max_off
    };
    let mut converged = false;
    for _sweep in 0..max_sweeps {
        if max_offdiag(&mat) <= off_tol {
            converged = true;
            break;
        }
        for p in 0..dim {
            for r in (p + 1)..dim {
                let apr = mat[p * dim + r];
                if apr.abs() <= pivot_tol {
                    continue;
                }
                let app = mat[p * dim + p];
                let arr = mat[r * dim + r];
                let tau = (arr - app) / (2.0 * apr);
                let t = if tau >= 0.0 {
                    1.0 / (tau + tau.hypot(1.0))
                } else {
                    -1.0 / (-tau + tau.hypot(1.0))
                };
                let c = 1.0 / t.hypot(1.0);
                let s = t * c;

                mat[p * dim + p] = app - t * apr;
                mat[r * dim + r] = arr + t * apr;
                mat[p * dim + r] = 0.0;
                mat[r * dim + p] = 0.0;

                for k in 0..dim {
                    if k != p && k != r {
                        let akp = mat[k * dim + p];
                        let akr = mat[k * dim + r];
                        let new_kp = c * akp - s * akr;
                        let new_kr = s * akp + c * akr;
                        mat[k * dim + p] = new_kp;
                        mat[p * dim + k] = new_kp;
                        mat[k * dim + r] = new_kr;
                        mat[r * dim + k] = new_kr;
                    }
                    let qkp = q[k * dim + p];
                    let qkr = q[k * dim + r];
                    q[k * dim + p] = c * qkp - s * qkr;
                    q[k * dim + r] = s * qkp + c * qkr;
                }
            }
        }
    }

    if !converged && max_offdiag(&mat) > off_tol {
        return Err(ReservingError::InvalidParameter(
            "Jacobi eigen-decomposition did not converge within max_sweeps",
        ));
    }

    let eigenvalues: Vec<f64> = (0..dim).map(|i| mat[i * dim + i]).collect();
    Ok((eigenvalues, q))
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
