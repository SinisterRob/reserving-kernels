//! Shared `O(n)` tridiagonal rank-1 Schur-complement factor for the
//! cross-classified log-link GLM `log μ_{i,j} = u_i + v_j` on an `n × n`
//! upper triangle with rank-1 IRLS weights `W_{i,j} = a_i · b_j`.
//!
//! Used by the ODP fit in [`crate::stochastic::bootstrap_odp`] (weights
//! `μ_{i,j} = α_i β_j`) and by the ODP / Tweedie / log-normal fast paths of
//! [`crate::stochastic::glm_reserving`], so both produce identical Green's
//! functions, hat-matrix leverages and parameter quadratic forms.

/// Stack-backed `f64` scratch buffer avoiding heap allocations for small/medium
/// triangles (`len <= CAP`) while transparently falling back to `Vec<f64>` for
/// larger sizes. Uses zero `unsafe` code.
pub(crate) enum SmallBuf<const CAP: usize> {
    Stack { len: usize, data: [f64; CAP] },
    Heap(Vec<f64>),
}

impl<const CAP: usize> SmallBuf<CAP> {
    #[inline]
    pub(crate) fn zeroed(len: usize) -> Self {
        if len <= CAP {
            Self::Stack {
                len,
                data: [0.0; CAP],
            }
        } else {
            Self::Heap(vec![0.0; len])
        }
    }

    #[inline]
    pub(crate) fn filled(len: usize, value: f64) -> Self {
        if len <= CAP {
            Self::Stack {
                len,
                data: [value; CAP],
            }
        } else {
            Self::Heap(vec![value; len])
        }
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[f64] {
        match self {
            Self::Stack { len, data } => &data[..*len],
            Self::Heap(v) => v.as_slice(),
        }
    }

    #[inline]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [f64] {
        match self {
            Self::Stack { len, data } => &mut data[..*len],
            Self::Heap(v) => v.as_mut_slice(),
        }
    }
}

// ===========================================================================
// Universal O(n) Tridiagonal Rank-1 Schur-Complement Factor for Cross-Classified
// GLMs (exact for any positive rank-1 weight matrix W_{i,j} = a_i * b_j).
// ===========================================================================

/// Compact single-slab `L D L^T` factorization of the `(n - 1) × (n - 1)`
/// tridiagonal Schur complement `K` of `X^T W X` for any strictly positive
/// rank-1 weight matrix `W_{i,j} = a_i * b_j` on an `n × n` upper triangle.
///
/// Stack-allocated for `n <= 16` (`7 * 16 - 4 = 108 <= 112` f64s), with automatic
/// heap fallback for larger triangles.
pub(crate) struct TridiagonalRank1Factor {
    n: usize,
    slab: SmallBuf<112>,
}

impl TridiagonalRank1Factor {
    pub(crate) fn new(n: usize) -> Self {
        debug_assert!(n >= 3);
        Self {
            n,
            slab: SmallBuf::<112>::zeroed(7 * n - 4),
        }
    }

    #[inline]
    fn offsets(n: usize) -> (usize, usize, usize, usize, usize, usize) {
        let m = n - 1;
        let off_inv_b = n;
        let off_inv_beta = 2 * n;
        let off_dt = 3 * n;
        let off_inv_dt = 3 * n + m;
        let off_gamma = 3 * n + 2 * m;
        let off_work = 3 * n + 3 * m - 1;
        (
            off_inv_b,
            off_inv_beta,
            off_dt,
            off_inv_dt,
            off_gamma,
            off_work,
        )
    }

    /// Factor `K = L D L^T` in `O(n)` time and zero allocations. Returns `false`
    /// if any weight is non-positive/non-finite or any pivot is ill-conditioned.
    pub(crate) fn refactor(&mut self, a: &[f64], b: &[f64]) -> bool {
        let n = self.n;
        let m = n - 1;
        let (off_inv_b, off_inv_beta, off_dt, off_inv_dt, off_gamma, off_work) = Self::offsets(n);
        let slab = self.slab.as_mut_slice();

        // Cumulative sums of b and reciprocals.
        let mut b_acc = 0.0;
        for j in 0..n {
            let bj = b[j];
            if !(bj > 0.0) || !bj.is_finite() {
                return false;
            }
            b_acc += bj;
            if !(b_acc > 0.0) || !b_acc.is_finite() {
                return false;
            }
            slab[j] = b_acc;
            slab[off_inv_b + j] = 1.0 / b_acc;
            slab[off_inv_beta + j] = 1.0 / bj;
        }

        // Prefix/reverse sums A[k] = sum_{i=0}^{m-k} a_i stored in `work[0..n]`.
        let mut a_acc = 0.0;
        for k in (0..n).rev() {
            let ai = a[m - k];
            if !(ai > 0.0) || !ai.is_finite() {
                return false;
            }
            a_acc += ai;
            slab[off_work + k] = a_acc;
        }

        // Fused tridiagonal assembly and L D L^T sweep.
        let mut prev_off = 0.0;
        let mut prev_gamma = 0.0;
        for k in 1..n {
            let idx = k - 1;
            let off = if k < m {
                slab[off_work + k + 1] * slab[off_inv_beta + k + 1]
            } else {
                0.0
            };
            let d_idx =
                slab[off_work + k] * slab[off_inv_beta + k] + off - a[m - k] * slab[off_inv_b + k];
            let dt = if idx == 0 {
                d_idx
            } else {
                d_idx - prev_off * prev_gamma
            };
            if !dt.is_finite() || dt <= 1e-14 * d_idx.abs().max(1e-30) {
                return false;
            }
            let inv_dt = 1.0 / dt;
            slab[off_dt + idx] = dt;
            slab[off_inv_dt + idx] = inv_dt;
            if k < m {
                let g = off * inv_dt;
                if !g.is_finite() {
                    return false;
                }
                slab[off_gamma + idx] = g;
                prev_off = off;
                prev_gamma = g;
            }
        }
        true
    }

    /// Solve `(X_{uv}^T W X_{uv}) (u, v) = (ru, rv)` in `O(n)` time and zero allocations.
    pub(crate) fn solve_uv(
        &mut self,
        a: &[f64],
        ru: &[f64],
        rv: &[f64],
        u_out: &mut [f64],
        v_out: &mut [f64],
    ) {
        let n = self.n;
        let m = n - 1;
        let (off_inv_b, off_inv_beta, _, off_inv_dt, off_gamma, off_work) = Self::offsets(n);
        let slab = self.slab.as_mut_slice();

        // Forward solve L y = r_tilde into `work[1..n]`.
        let mut y_prev = 0.0;
        for idx in 0..m {
            let k = idx + 1;
            let rt = if k < m {
                rv[k] * slab[off_inv_beta + k]
                    - rv[k + 1] * slab[off_inv_beta + k + 1]
                    - ru[m - k] * slab[off_inv_b + k]
            } else {
                rv[m] * slab[off_inv_beta + m] - ru[0] * slab[off_inv_b + m]
            };
            let yk = if idx == 0 {
                rt
            } else {
                rt + slab[off_gamma + idx - 1] * y_prev
            };
            slab[off_work + k] = yk;
            y_prev = yk;
        }

        // Backward solve L^T T = D^{-1} y in-place in `work[0..n]` (`T[0] = 0`).
        slab[off_work] = 0.0;
        let mut t_next = slab[off_work + m] * slab[off_inv_dt + m - 1];
        slab[off_work + m] = t_next;
        for idx in (0..(m - 1)).rev() {
            let tk =
                slab[off_work + idx + 1] * slab[off_inv_dt + idx] + slab[off_gamma + idx] * t_next;
            slab[off_work + idx + 1] = tk;
            t_next = tk;
        }

        // Recover v and u.
        v_out[0] = 0.0;
        for j in 1..n {
            v_out[j] = (slab[off_work + j] - slab[off_work + j - 1]) * slab[off_inv_beta + j];
        }
        for i in 0..n {
            let ki = m - i;
            u_out[i] = (ru[i] / a[i] - slab[off_work + ki]) * slab[off_inv_b + ki];
        }
    }

    /// Fill the symmetric `n × n` zero-bordered Green's function matrix `G = K^{-1}`
    /// (`g_mat[a * n + b] = G_{a,b}` for `a, b in 1..n`, and `0.0` when `a == 0 || b == 0`)
    /// in-place in `O(n^2)` time (`m(m-1)/2` multiplications, zero return-value copy).
    pub(crate) fn compute_g_mat_into(&self, g_mat: &mut [f64]) {
        let n = self.n;
        let m = n - 1;
        let (_, _, _, off_inv_dt, off_gamma, _) = Self::offsets(n);
        let slab = self.slab.as_slice();

        g_mat[m * n + m] = slab[off_inv_dt + m - 1];
        for idx in (0..(m - 1)).rev() {
            let b = idx + 1;
            let g = slab[off_gamma + idx];
            g_mat[b * n + b] = slab[off_inv_dt + idx] + g * g * g_mat[(b + 1) * n + (b + 1)];
        }

        for b in 2..n {
            let mut val = g_mat[b * n + b];
            for a in (1..b).rev() {
                val *= slab[off_gamma + a - 1];
                g_mat[a * n + b] = val;
                g_mat[b * n + a] = val;
            }
        }
    }

    /// Compute all `N = n(n+1)/2` exact hat-matrix leverages in `O(n^2)` time.
    pub(crate) fn compute_leverages(&mut self, a: &[f64], b: &[f64], g_mat: &[f64]) -> Vec<f64> {
        let n = self.n;
        let m = n - 1;
        let (off_inv_b, off_inv_beta, _, _, _, off_work) = Self::offsets(n);
        let slab = self.slab.as_mut_slice();

        // Precompute w_j^T G w_j into `work[1..n]`.
        for j in 1..n {
            let ib = slab[off_inv_beta + j];
            slab[off_work + j] = (g_mat[j * n + j] - 2.0 * g_mat[j * n + (j - 1)]
                + g_mat[(j - 1) * n + (j - 1)])
                * ib
                * ib;
        }

        let n_obs = n * (n + 1) / 2;
        let mut leverages = Vec::with_capacity(n_obs);
        for origin in 0..n {
            let ki = m - origin;
            let inv_b_ki = slab[off_inv_b + ki];
            let base_phi = inv_b_ki / a[origin];
            let g_row = &g_mat[ki * n..ki * n + n];
            let z_g_z = g_row[ki] * inv_b_ki * inv_b_ki;
            let ai = a[origin];

            leverages.push((ai * b[0] * (base_phi + z_g_z)).clamp(0.0, 1.0));
            for dev in 1..=ki {
                let w_g_z = -(g_row[dev] - g_row[dev - 1]) * inv_b_ki * slab[off_inv_beta + dev];
                let y_g_y = slab[off_work + dev] + 2.0 * w_g_z + z_g_z;
                leverages.push((ai * b[dev] * (base_phi + y_g_y)).clamp(0.0, 1.0));
            }
        }
        leverages
    }

    /// Compute the full `(2n - 1) × (2n - 1)` symmetric covariance matrix
    /// `scale * (X^T W X)^{-1}` in corner coding `(intercept, origin_1.., dev_1..)`
    /// in exact `O(n^2)` time.
    pub(crate) fn compute_cov_scaled(
        &mut self,
        inv_a: &[f64],
        g_mat: &[f64],
        scale: f64,
    ) -> Vec<f64> {
        let n = self.n;
        let m = n - 1;
        let q = 2 * n - 1;
        let (off_inv_b, off_inv_beta, _, _, _, off_work) = Self::offsets(n);
        let slab = self.slab.as_mut_slice();

        let inv_bm = slab[off_inv_b + m];
        let g_mm = g_mat[m * n + m];
        let c_u00 = inv_bm * inv_a[0] + g_mm * inv_bm * inv_bm;

        // Store `c_u0[i] = Cov_unscaled(u_i, u_0)` in `work[0..n]`.
        slab[off_work] = c_u00;
        for i in 1..n {
            let ki = m - i;
            slab[off_work + i] = g_mat[ki * n + m] * slab[off_inv_b + ki] * inv_bm;
        }

        let mut cov = vec![0.0; q * q];
        cov[0] = scale * c_u00;

        // Intercept / origin block.
        for i in 1..n {
            let c_ui0 = slab[off_work + i];
            let v0i = scale * (c_ui0 - c_u00);
            cov[i * q] = v0i;
            cov[i] = v0i;

            let ki = m - i;
            let inv_b_ki = slab[off_inv_b + ki];
            let diag_i = inv_b_ki * inv_a[i];
            for r in i..n {
                let kr = m - r;
                let diag = if i == r { diag_i } else { 0.0 };
                let c_uir = diag + g_mat[ki * n + kr] * inv_b_ki * slab[off_inv_b + kr];
                let val = scale * (c_uir - c_ui0 - slab[off_work + r] + c_u00);
                cov[i * q + r] = val;
                cov[r * q + i] = val;
            }
        }

        // Cross block between (intercept, origin_1..) and dev_1..
        for j in 1..n {
            let col = n + j - 1;
            let inv_bj = slab[off_inv_beta + j];
            let c_u0vj = -(g_mat[m * n + j] - g_mat[m * n + (j - 1)]) * inv_bm * inv_bj;
            let val0 = scale * c_u0vj;
            cov[col] = val0;
            cov[col * q] = val0;

            for i in 1..n {
                let ki = m - i;
                let c_uivj =
                    -(g_mat[ki * n + j] - g_mat[ki * n + (j - 1)]) * slab[off_inv_b + ki] * inv_bj;
                let val = scale * (c_uivj - c_u0vj);
                cov[i * q + col] = val;
                cov[col * q + i] = val;
            }
        }

        // Development block dev_1.. × dev_1..
        for a_idx in 1..n {
            let row = n + a_idx - 1;
            let inv_ba = slab[off_inv_beta + a_idx];
            for b_idx in a_idx..n {
                let col = n + b_idx - 1;
                let c_vab = (g_mat[a_idx * n + b_idx]
                    - g_mat[(a_idx - 1) * n + b_idx]
                    - g_mat[a_idx * n + (b_idx - 1)]
                    + g_mat[(a_idx - 1) * n + (b_idx - 1)])
                    * inv_ba
                    * slab[off_inv_beta + b_idx];
                let val = scale * c_vab;
                cov[row * q + col] = val;
                cov[col * q + row] = val;
            }
        }

        cov
    }

    /// Exact unscaled parameter estimation quadratic form `g_i^T (X^T W X)^{-1} g_i`
    /// for origin `i` under arbitrary rank-1 future expectations `mu_{i,j} = au_i * bv_j`
    /// in `O(n)` time and zero allocations.
    pub(crate) fn origin_param_quad_general(
        &self,
        inv_a: &[f64],
        origin: usize,
        r_i: f64,
        au_i: f64,
        bv: &[f64],
    ) -> f64 {
        if origin == 0 {
            return 0.0;
        }
        let n = self.n;
        let m = n - 1;
        let ki = m - origin;
        let (off_inv_b, off_inv_beta, _, off_inv_dt, off_gamma, _) = Self::offsets(n);
        let slab = self.slab.as_slice();

        let mut quad = (r_i * r_i) * slab[off_inv_b + ki] * inv_a[origin];
        let mut z_prev = 0.0;
        let start_idx = ki.saturating_sub(1);
        for idx in start_idx..m {
            let k = idx + 1;
            let gv_k = if k > ki {
                au_i * bv[k] * slab[off_inv_beta + k]
            } else {
                0.0
            };
            let gv_next = if k < m && k + 1 > ki {
                au_i * bv[k + 1] * slab[off_inv_beta + k + 1]
            } else {
                0.0
            };
            let gu_term = if m - k == origin {
                r_i * slab[off_inv_b + k]
            } else {
                0.0
            };
            let qt = gv_k - gv_next - gu_term;
            let z = if idx == start_idx {
                qt
            } else {
                qt + slab[off_gamma + idx - 1] * z_prev
            };
            quad += (z * z) * slab[off_inv_dt + idx];
            z_prev = z;
        }
        quad
    }

    /// Exact unscaled aggregate parameter estimation quadratic form
    /// `g_total^T (X^T W X)^{-1} g_total` in `O(n)` time and zero allocations.
    pub(crate) fn total_param_quad_general(
        &self,
        inv_a: &[f64],
        reserves: &[f64],
        au: &[f64],
        bv: &[f64],
    ) -> f64 {
        let n = self.n;
        let m = n - 1;
        let (off_inv_b, off_inv_beta, _, off_inv_dt, off_gamma, _) = Self::offsets(n);
        let slab = self.slab.as_slice();

        let mut quad = 0.0;
        for i in 1..n {
            let ki = m - i;
            quad += (reserves[i] * reserves[i]) * slab[off_inv_b + ki] * inv_a[i];
        }

        // `s_au_k = sum_{i = n - k}^{n - 1} au[i]` increases by `au[n - k]` each step.
        let mut s_au_k = 0.0;
        let mut z_prev = 0.0;
        for idx in 0..m {
            let k = idx + 1;
            s_au_k += au[n - k];
            let gv_k = s_au_k * bv[k] * slab[off_inv_beta + k];
            let gv_next = if k < m {
                let s_au_next = s_au_k + au[n - k - 1];
                s_au_next * bv[k + 1] * slab[off_inv_beta + k + 1]
            } else {
                0.0
            };
            let gu_term = reserves[m - k] * slab[off_inv_b + k];
            let qt = gv_k - gv_next - gu_term;
            let z = if idx == 0 {
                qt
            } else {
                qt + slab[off_gamma + idx - 1] * z_prev
            };
            quad += (z * z) * slab[off_inv_dt + idx];
            z_prev = z;
        }
        quad
    }

    #[inline]
    pub(crate) fn b_cum(&self, k: usize) -> f64 {
        self.slab.as_slice()[k]
    }

    #[inline]
    pub(crate) fn inv_b(&self, k: usize) -> f64 {
        self.slab.as_slice()[self.n + k]
    }

    #[inline]
    pub(crate) fn inv_beta(&self, j: usize) -> f64 {
        self.slab.as_slice()[2 * self.n + j]
    }

    #[inline]
    pub(crate) fn inv_d_tilde(&self, idx: usize) -> f64 {
        self.slab.as_slice()[3 * self.n + (self.n - 1) + idx]
    }

    #[inline]
    pub(crate) fn gamma(&self, idx: usize) -> f64 {
        self.slab.as_slice()[3 * self.n + 2 * (self.n - 1) + idx]
    }
}
