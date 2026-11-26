//! Mack's distribution-free chain-ladder standard error.
//!
//! Implements:
//!
//! * Mack (1993), *Distribution-Free Calculation of the Standard Error of Chain
//!   Ladder Reserve Estimates*, ASTIN Bulletin, Vol. 23, No. 2, pp. 213–225.
//! * Mack (1999), *The Standard Error of Chain Ladder Reserve Estimates:
//!   Recursive Calculation and Inclusion of a Tail Factor*, ASTIN Bulletin,
//!   Vol. 29, No. 2, pp. 361–366.
//!
//! Notation (0-indexed, square triangle of size `n`):
//!
//! * `f_k = Σ_i C_{i, k+1} / Σ_i C_{i, k}` — volume-weighted LDF at link
//!   `k`, summed over `i = 0..n−k−2`.
//! * `σ²_k = 1/(n_k − 1) × Σ_i C_{i, k} × (C_{i, k+1}/C_{i, k} − f_k)²`
//!   with `n_k = n − k − 1` (observed pairs at link `k`).
//! * For the last link (`k = n − 2`, only one pair observed) `σ²` is
//!   extrapolated via Mack's min-rule:
//!   `σ²_{n−2} = min(σ⁴_{n−3}/σ²_{n−4}, σ²_{n−3}, σ²_{n−4})`.
//! * `Ĉ_{i,k}` — projected cumulative for origin `i` at development `k`,
//!   defined observationally for `k ≤ n − 1 − i` and by chain-ladder for
//!   `k > n − 1 − i`.
//! * `S_k = Σ_{i=0}^{n_k−1} C_{i,k}` — column sum entering the LDF.
//!
//! Per-origin MSE (Mack 1993, Theorem 3 / Eq. 7):
//!
//! `MSE(Û_i) = Û_i² × Σ_{k = n−1−i}^{n−2} σ²_k/f²_k × (1/Ĉ_{i,k} + 1/S_k)`.
//!
//! Total-reserve MSE (Mack 1993, Theorem 3 / Eq. 15):
//!
//! `MSE(ΣR) = Σ MSE(R_i) + 2 × Σ_{i<j} Û_i Û_j × Σ_{k=n−1−i}^{n−2}
//!            σ²_k / (f²_k × S_k)`.
//!
//! With a stochastic tail factor `f_tail` carrying process variance
//! `σ²_tail` and parameter standard error `se(f_tail)` (Mack 1999, Eq. 4):
//!
//! `MSE(Û_i) = f_tail² × MSE(Ĉ_{i,n−1}) + Ĉ_{i,n−1} × σ²_tail + Ĉ_{i,n−1}² × se(f_tail)²`,
//!
//! and every pair `0 ≤ i < j < n` gains cross-origin tail covariance
//! `2 × Ĉ_{i,n−1} × Ĉ_{j,n−1} × se(f_tail)²`.

use crate::ldf::volume_weighted;
use crate::ldf::{DevelopmentFactorEngine, volume_weighted_ldfs_observed_counts};
use crate::triangle::latest_diagonal;
use crate::types::TriangleView;
use crate::validate::{
    MonotonicityPolicy, ValidationError, validate_monotone_cumulative,
    validate_observed_count_inputs, validate_tail, validate_triangle,
};

/// Tail-factor mean and variance parameters for Mack (1999) recursive standard errors.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MackTailSpec {
    /// Multiplicative tail factor `f_tail >= 1.0`.
    pub tail_factor: f64,
    /// Tail process variance parameter `sigma_tail^2 >= 0.0` (Mack 1999, Eq. 4).
    pub tail_sigma_sq: f64,
    /// Tail parameter estimation standard error `se(f_tail) >= 0.0` (Mack 1999, Eq. 4).
    pub tail_se: f64,
}

impl MackTailSpec {
    /// Fixed tail factor with zero modeled tail uncertainty.
    #[inline]
    pub const fn fixed(tail_factor: f64) -> Self {
        Self {
            tail_factor,
            tail_sigma_sq: 0.0,
            tail_se: 0.0,
        }
    }
}

/// Full Mack chain-ladder result: point estimates and standard errors.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Mack {
    /// Volume-weighted LDFs, length `n − 1`.
    pub ldf: Vec<f64>,
    /// Mack σ² per development link, length `n − 1`.
    pub sigma_sq: Vec<f64>,
    /// Chain-ladder ultimates per origin, length `n`.
    pub ultimate: Vec<f64>,
    /// Reserves per origin, length `n`.
    pub reserve: Vec<f64>,
    /// Standard error per origin, length `n`.
    pub standard_error: Vec<f64>,
    /// Coefficient of variation per origin (SE / reserve), length `n`.
    pub cv: Vec<f64>,
    /// Total reserve across origins.
    pub total_reserve: f64,
    /// Total MSE (sum of per-origin MSE plus covariance adjustment).
    pub total_mse: f64,
    /// Total standard error: `sqrt(total_mse)`.
    pub total_standard_error: f64,
    /// Total coefficient of variation: `total_se / total_reserve`.
    pub total_cv: f64,
}

impl std::fmt::Display for Mack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Mack(reserve={:.2}, se={:.2}, cv={:.2}%)",
            self.total_reserve,
            self.total_standard_error,
            self.total_cv * 100.0
        )
    }
}

fn sigma_sq_vec(t: &TriangleView<'_>, f: &[f64]) -> Vec<f64> {
    let n = t.n_origins.min(t.n_devs);
    if n < 2 {
        return Vec::new();
    }
    let mut s: Vec<Option<f64>> = vec![None; n - 1];
    // Links k = 0..n-2; observed pairs n_k = n - k - 1.
    for k in 0..n - 1 {
        let nk = n - k - 1;
        let mut acc = 0.0;
        let mut valid = 0usize;
        let mut invalid_zero = false;
        for i in 0..nk {
            let c_k = t.get(i, k);
            let c_kp1 = t.get(i, k + 1);
            if c_k > 0.0 {
                let ratio = c_kp1 / c_k;
                acc += c_k * (ratio - f[k]).powi(2);
                valid += 1;
            } else if c_k == 0.0 && c_kp1 == 0.0 {
                // Zero prior and zero next contributes 0 to variance and does not
                // provide an independent ratio observation.
                continue;
            } else {
                invalid_zero = true;
                break;
            }
        }
        if invalid_zero {
            s[k] = Some(f64::NAN);
        } else if valid >= 2 {
            s[k] = Some(acc / (valid as f64 - 1.0));
        }
        // valid < 2 remains None for Mack's extrapolation below.
    }
    // Extrapolate missing links using Mack's min-rule.
    let mut out = vec![0.0; n - 1];
    for k in 0..n - 1 {
        if let Some(val) = s[k] {
            out[k] = val;
        } else if k >= 2 {
            let a = out[k - 1];
            let b = out[k - 2];
            out[k] = if a.is_finite() && b.is_finite() && a > 0.0 && b > 0.0 {
                (a * a / b).min(a).min(b).max(0.0)
            } else {
                0.0
            };
        } else if k == 1 {
            out[k] = if out[0].is_finite() {
                out[0].max(0.0)
            } else {
                0.0
            };
        } else {
            out[k] = 0.0;
        }
    }
    out
}

fn sigma_sq_vec_observed_counts(
    t: &TriangleView<'_>,
    observed_counts: &[usize],
    f: &[f64],
) -> Vec<f64> {
    let n_dev = t.n_devs;
    if n_dev < 2 {
        return Vec::new();
    }
    let mut s: Vec<Option<f64>> = vec![None; n_dev - 1];
    for k in 0..n_dev - 1 {
        let mut acc = 0.0;
        let mut valid = 0usize;
        let mut invalid_zero = false;
        for (i, &count) in observed_counts.iter().enumerate() {
            if count <= k + 1 {
                continue;
            }
            let c_k = t.get(i, k);
            let c_kp1 = t.get(i, k + 1);
            if c_k > 0.0 {
                let ratio = c_kp1 / c_k;
                acc += c_k * (ratio - f[k]).powi(2);
                valid += 1;
            } else if c_k == 0.0 && c_kp1 == 0.0 {
                continue;
            } else {
                invalid_zero = true;
                break;
            }
        }
        if invalid_zero {
            s[k] = Some(f64::NAN);
        } else if valid >= 2 {
            s[k] = Some(acc / (valid as f64 - 1.0));
        }
    }
    extrapolate_missing_sigma(&s)
}

fn extrapolate_missing_sigma(sigma_sq: &[Option<f64>]) -> Vec<f64> {
    let mut out = vec![0.0; sigma_sq.len()];
    for k in 0..sigma_sq.len() {
        if let Some(val) = sigma_sq[k] {
            out[k] = val;
            continue;
        }
        let mut last_two = [None, None];
        for &value in &out[..k] {
            if value.is_finite() && value >= 0.0 {
                last_two[0] = last_two[1];
                last_two[1] = Some(value);
            }
        }
        out[k] = match last_two {
            [Some(b), Some(a)] => {
                if a > 0.0 && b > 0.0 {
                    (a * a / b).min(a).min(b).max(0.0)
                } else {
                    0.0
                }
            }
            [None, Some(a)] => a,
            _ => 0.0,
        };
    }
    out
}

/// Run the full Mack analysis without validation.
///
/// `tail` is the multiplicative tail factor (pass `1.0` for fully-developed).
/// When a tail > 1 is applied, the reported standard errors capture only the
/// chain-ladder development uncertainty — tail uncertainty is not modelled.
/// Use [`analyze_with_tail_uncertainty`] to include Mack (1999) tail variance.
///
/// Prefer [`analyze`] at external ingress boundaries.
pub fn analyze_unchecked(t: &TriangleView<'_>, tail: f64) -> Mack {
    analyze_with_tail_spec_unchecked(t, MackTailSpec::fixed(tail))
}

/// Run the full Mack (1993 / 1999) analysis with an explicit [`MackTailSpec`]
/// without validation.
pub fn analyze_with_tail_spec_unchecked(t: &TriangleView<'_>, tail_spec: MackTailSpec) -> Mack {
    let n = t.n_origins.min(t.n_devs);
    let ldf = volume_weighted(t);
    let sigma_sq = sigma_sq_vec(t, &ldf);
    let tail = tail_spec.tail_factor;
    let tail_se_sq = tail_spec.tail_se * tail_spec.tail_se;

    // Column sums S_k = Σ_{i=0}^{n_k-1} C_{i,k} for k = 0..n-2.
    let mut s_col = vec![0.0; n.saturating_sub(1)];
    for k in 0..n.saturating_sub(1) {
        let nk = n - k - 1;
        let mut acc = 0.0;
        for i in 0..nk {
            acc += t.get(i, k);
        }
        s_col[k] = acc;
    }

    let latest = latest_diagonal(t);

    // Projected cumulative Ĉ_{i,k} for i in 0..n, k in 0..n.
    let mut proj = vec![0.0; n * n];
    for i in 0..n {
        let start = n - 1 - i;
        proj[i * n + start] = latest[i];
        for k in (start + 1)..n {
            proj[i * n + k] = proj[i * n + k - 1] * ldf[k - 1];
        }
    }

    let mut ultimate = vec![0.0; n];
    let mut reserve = vec![0.0; n];
    for i in 0..n {
        ultimate[i] = proj[i * n + (n - 1)] * tail;
        reserve[i] = ultimate[i] - latest[i];
    }

    // Per-origin MSE up to age n - 1 (Mack 1993 Eq. 7), then Mack (1999) Eq. 4
    // recursive step for the tail factor f_tail with sigma_tail^2 and se(f_tail).
    let mut mse = vec![0.0; n];
    for i in 0..n {
        let c_last = proj[i * n + (n - 1)];
        let mut pre_tail_mse = 0.0;
        if i >= 1 {
            let start = n - 1 - i;
            let mut acc = 0.0;
            for k in start..n - 1 {
                let fk = ldf[k];
                let sk2 = sigma_sq[k];
                let c_hat = proj[i * n + k];
                let s = s_col[k];
                if fk > 0.0 && c_hat > 0.0 && s > 0.0 {
                    acc += (sk2 / (fk * fk)) * (1.0 / c_hat + 1.0 / s);
                }
            }
            pre_tail_mse = c_last * c_last * acc;
        }
        let tail_process = if c_last > 0.0 {
            c_last * tail_spec.tail_sigma_sq
        } else {
            0.0
        };
        let tail_param = c_last * c_last * tail_se_sq;
        mse[i] = tail * tail * pre_tail_mse + tail_process + tail_param;
    }

    let standard_error: Vec<f64> = mse
        .iter()
        .map(|&m| if m.is_finite() { m.max(0.0).sqrt() } else { f64::NAN })
        .collect();
    let cv: Vec<f64> = standard_error
        .iter()
        .zip(reserve.iter())
        .map(|(s, r)| if *r > 0.0 { s / r } else { 0.0 })
        .collect();

    let total_reserve: f64 = reserve.iter().sum();

    // Total MSE (Mack 1993 Theorem 3 / Eq. 15 + Mack 1999 Section 3 tail covariance):
    //   Σ MSE_i + 2 × Σ_{i<j} [ Û_i Û_j Σ_{k=n-1-i}^{n-2} σ²_k/(f²_k × S_k)
    //                         + Ĉ_{i,n-1} Ĉ_{j,n-1} se(f_tail)² ]
    // For i < j, origin i is older and already observed up to age n - 1 - i,
    // so origins i and j share estimated development factors only for k >= n - 1 - i.
    let mut total_mse: f64 = mse.iter().sum();
    for i in 0..n {
        let c_last_i = proj[i * n + (n - 1)];
        let start = n - 1 - i;
        let mut cov_sum = 0.0;
        if i >= 1 {
            for k in start..n - 1 {
                let fk = ldf[k];
                let sk2 = sigma_sq[k];
                let s = s_col[k];
                if fk > 0.0 && s > 0.0 {
                    cov_sum += sk2 / (fk * fk * s);
                }
            }
        }
        for j in (i + 1)..n {
            let c_last_j = proj[j * n + (n - 1)];
            total_mse +=
                2.0 * ultimate[i] * ultimate[j] * cov_sum + 2.0 * c_last_i * c_last_j * tail_se_sq;
        }
    }

    let total_standard_error = if total_mse.is_finite() {
        total_mse.max(0.0).sqrt()
    } else {
        f64::NAN
    };
    let total_cv = if total_reserve > 0.0 {
        total_standard_error / total_reserve
    } else {
        0.0
    };

    Mack {
        ldf,
        sigma_sq,
        ultimate,
        reserve,
        standard_error,
        cv,
        total_reserve,
        total_mse,
        total_standard_error,
        total_cv,
    }
}

/// Run Mack analysis for rectangular/ragged triangles using observed row widths.
///
/// `observed_counts[i]` is the number of observed development periods in
/// origin row `i`. The latest value is taken from `observed_counts[i] - 1`, and
/// each development link uses rows where both adjacent cells are observed.
pub fn analyze_observed_counts(
    t: &TriangleView<'_>,
    observed_counts: &[usize],
    tail: f64,
) -> Result<Mack, ValidationError> {
    validate_observed_count_inputs(t, observed_counts, tail)?;

    let n_origin = t.n_origins;
    let n_dev = t.n_devs;
    let mut ldf = vec![0.0; n_dev.saturating_sub(1)];
    volume_weighted_ldfs_observed_counts(t.data, n_origin, n_dev, observed_counts, &mut ldf);
    if ldf.iter().any(|value| !value.is_finite()) {
        return Err(ValidationError::DivideByZero("development denominator"));
    }

    let sigma_sq = sigma_sq_vec_observed_counts(t, observed_counts, &ldf);
    if sigma_sq.iter().any(|value| !value.is_finite()) {
        return Err(ValidationError::NotFinite);
    }

    let mut s_col = vec![0.0; n_dev.saturating_sub(1)];
    for k in 0..n_dev.saturating_sub(1) {
        let mut acc = 0.0;
        for (i, &count) in observed_counts.iter().enumerate() {
            if count > k + 1 {
                acc += t.get(i, k);
            }
        }
        s_col[k] = acc;
    }

    let mut latest = vec![0.0; n_origin];
    let mut latest_dev = vec![0; n_origin];
    for i in 0..n_origin {
        latest_dev[i] = observed_counts[i] - 1;
        latest[i] = t.get(i, latest_dev[i]);
    }

    let mut proj = vec![0.0; n_origin * n_dev];
    for i in 0..n_origin {
        let start = latest_dev[i];
        proj[i * n_dev + start] = latest[i];
        for k in (start + 1)..n_dev {
            proj[i * n_dev + k] = proj[i * n_dev + k - 1] * ldf[k - 1];
        }
    }

    let mut ultimate = vec![0.0; n_origin];
    let mut reserve = vec![0.0; n_origin];
    for i in 0..n_origin {
        ultimate[i] = proj[i * n_dev + (n_dev - 1)] * tail;
        reserve[i] = ultimate[i] - latest[i];
    }

    let mut mse = vec![0.0; n_origin];
    for i in 0..n_origin {
        let start = latest_dev[i];
        let mut acc = 0.0;
        for k in start..n_dev.saturating_sub(1) {
            let fk = ldf[k];
            let sk2 = sigma_sq[k];
            let c_hat = proj[i * n_dev + k];
            let s = s_col[k];
            if fk > 0.0 && c_hat > 0.0 && s > 0.0 {
                acc += (sk2 / (fk * fk)) * (1.0 / c_hat + 1.0 / s);
            }
        }
        let u = ultimate[i];
        mse[i] = u * u * acc;
    }

    let standard_error: Vec<f64> = mse.iter().map(|value| value.sqrt()).collect();
    let cv: Vec<f64> = standard_error
        .iter()
        .zip(reserve.iter())
        .map(|(standard_error, reserve)| {
            if *reserve > 0.0 {
                standard_error / reserve
            } else {
                0.0
            }
        })
        .collect();
    let total_reserve: f64 = reserve.iter().sum();

    let mut total_mse: f64 = mse.iter().sum();
    for i in 0..n_origin {
        for j in (i + 1)..n_origin {
            let start = latest_dev[i].max(latest_dev[j]);
            let mut cov_sum = 0.0;
            for k in start..n_dev.saturating_sub(1) {
                let fk = ldf[k];
                let sk2 = sigma_sq[k];
                let s = s_col[k];
                if fk > 0.0 && s > 0.0 {
                    cov_sum += sk2 / (fk * fk * s);
                }
            }
            total_mse += 2.0 * ultimate[i] * ultimate[j] * cov_sum;
        }
    }

    let total_standard_error = total_mse.max(0.0).sqrt();
    let total_cv = if total_reserve > 0.0 {
        total_standard_error / total_reserve
    } else {
        0.0
    };

    Ok(Mack {
        ldf,
        sigma_sq,
        ultimate,
        reserve,
        standard_error,
        cv,
        total_reserve,
        total_mse,
        total_standard_error,
        total_cv,
    })
}

/// Run Mack analysis after validating square-triangle and development inputs.
///
/// `tail > 1` scales point estimates, but tail-factor uncertainty is not
/// included in the Mack standard errors. Use [`analyze_with_tail_uncertainty`]
/// to incorporate Mack (1999) tail variance, or [`analyze_with_policy`] when
/// analyzing triangles with negative incremental development (e.g., incurred
/// case-reserve releases or Mack's 1993 RAA dataset).
pub fn analyze(t: &TriangleView<'_>, tail: f64) -> Result<Mack, ValidationError> {
    analyze_with_policy(t, tail, MonotonicityPolicy::Strict)
}

/// Run Mack analysis under an explicit [`MonotonicityPolicy`].
pub fn analyze_with_policy(
    t: &TriangleView<'_>,
    tail: f64,
    policy: MonotonicityPolicy,
) -> Result<Mack, ValidationError> {
    analyze_with_tail_uncertainty_and_policy(t, MackTailSpec::fixed(tail), policy)
}

/// Run Mack (1999) recursive standard-error analysis including tail uncertainty.
pub fn analyze_with_tail_uncertainty(
    t: &TriangleView<'_>,
    tail_spec: MackTailSpec,
) -> Result<Mack, ValidationError> {
    analyze_with_tail_uncertainty_and_policy(t, tail_spec, MonotonicityPolicy::Strict)
}

/// Run Mack (1999) recursive standard-error analysis with explicit tail
/// uncertainty and [`MonotonicityPolicy`].
pub fn analyze_with_tail_uncertainty_and_policy(
    t: &TriangleView<'_>,
    tail_spec: MackTailSpec,
    policy: MonotonicityPolicy,
) -> Result<Mack, ValidationError> {
    validate_triangle(t)?;
    if policy == MonotonicityPolicy::Strict {
        validate_monotone_cumulative(t)?;
    }
    validate_tail_spec(tail_spec)?;
    DevelopmentFactorEngine::volume_weighted_with_policy(t, tail_spec.tail_factor, policy)?;
    let n = t.n_origins;
    for k in 0..n.saturating_sub(1) {
        let nk = n - k - 1;
        for i in 0..nk {
            if t.get(i, k) == 0.0 && t.get(i, k + 1) != 0.0 {
                return Err(ValidationError::DivideByZero("mack link denominator"));
            }
        }
    }
    Ok(analyze_with_tail_spec_unchecked(t, tail_spec))
}

/// Configuration for generalized Mack chain-ladder analysis with custom
/// weighting exponent `alpha` (Murphy 1994; `mages/ChainLadder`) and optional
/// per-cell outlier/exclusion weights.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MackOptions {
    /// Tail factor and optional Mack (1999) tail variance specification.
    pub tail_spec: MackTailSpec,
    /// Variance weighting exponent $\alpha$:
    /// - `1.0` (default): classical volume-weighted chain ladder (Mack 1993)
    /// - `0.0`: OLS regression through the origin
    /// - `2.0`: unweighted simple average of individual link ratios
    pub alpha: f64,
    /// Monotonicity validation policy for the cumulative input triangle.
    pub policy: MonotonicityPolicy,
    /// Optional row-major cell weight mask of length `n_origins * n_devs`
    /// (`0.0` excludes a link from LDF and $\hat{\sigma}_k^2$ estimation;
    /// positive values scale the cell's weight).
    pub weights: Option<Vec<f64>>,
}

impl Default for MackOptions {
    fn default() -> Self {
        Self {
            tail_spec: MackTailSpec::fixed(1.0),
            alpha: 1.0,
            policy: MonotonicityPolicy::Strict,
            weights: None,
        }
    }
}

impl MackOptions {
    /// Set the variance weighting exponent $\alpha$ (`0.0`, `1.0`, or `2.0`).
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the tail specification.
    pub fn with_tail_spec(mut self, tail_spec: MackTailSpec) -> Self {
        self.tail_spec = tail_spec;
        self
    }

    /// Set the monotonicity validation policy.
    pub fn with_policy(mut self, policy: MonotonicityPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Set a row-major cell weight / outlier-exclusion mask.
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self
    }
}

/// Run generalized Mack (1993 / 1999) / Murphy (1994) analysis with explicit
/// `alpha` weighting and optional cell weight/exclusion mask.
pub fn analyze_with_options(
    t: &TriangleView<'_>,
    options: &MackOptions,
) -> Result<Mack, ValidationError> {
    if !options.alpha.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if options.weights.is_none() && (options.alpha - 1.0).abs() <= 1e-12 {
        return analyze_with_tail_uncertainty_and_policy(t, options.tail_spec, options.policy);
    }

    validate_triangle(t)?;
    if options.policy == MonotonicityPolicy::Strict {
        validate_monotone_cumulative(t)?;
    }
    validate_tail_spec(options.tail_spec)?;

    let n = t.n_origins.min(t.n_devs);
    if n < 2 {
        return Err(ValidationError::ShapeMismatch);
    }
    if let Some(ref w) = options.weights {
        if w.len() != t.n_origins * t.n_devs {
            return Err(ValidationError::ShapeMismatch);
        }
        if w.iter().any(|&v| !v.is_finite() || v < 0.0) {
            return Err(ValidationError::Negative);
        }
    }

    let alpha = options.alpha;
    let cell_w = |i: usize, k: usize| -> f64 {
        options
            .weights
            .as_ref()
            .map_or(1.0, |w| w[i * t.n_devs + k])
    };

    let n_steps = n - 1;
    let mut ldf = vec![1.0; n_steps];
    let mut s_col = vec![0.0; n_steps];
    let mut sigma_sq = vec![0.0; n_steps];

    for k in 0..n_steps {
        let nk = n - 1 - k;
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..nk {
            let w = cell_w(i, k);
            if w <= 0.0 {
                continue;
            }
            let c0 = t.get(i, k);
            let c1 = t.get(i, k + 1);
            if !c0.is_finite() || !c1.is_finite() || c0 <= 0.0 {
                continue;
            }
            let c_pow = c0.powf(alpha);
            num += w * c_pow * (c1 / c0);
            den += w * c_pow;
        }
        if den <= 0.0 {
            return Err(ValidationError::DivideByZero("mack weighted denominator"));
        }
        let fk = num / den;
        ldf[k] = fk;
        s_col[k] = den;

        let mut active = 0usize;
        let mut sse = 0.0;
        for i in 0..nk {
            let w = cell_w(i, k);
            if w <= 0.0 {
                continue;
            }
            let c0 = t.get(i, k);
            let c1 = t.get(i, k + 1);
            if !c0.is_finite() || !c1.is_finite() || c0 <= 0.0 {
                continue;
            }
            active += 1;
            let diff = (c1 / c0) - fk;
            sse += w * c0.powf(alpha) * diff * diff;
        }
        if active >= 2 {
            sigma_sq[k] = (sse / (active - 1) as f64).max(0.0);
        }
    }

    // Mack (1993) extrapolation for the last column when fewer than 2 active links exist.
    if n_steps >= 3 && sigma_sq[n_steps - 1] == 0.0 {
        let s1 = sigma_sq[n_steps - 2];
        let s2 = sigma_sq[n_steps - 3];
        if s2 > 0.0 {
            sigma_sq[n_steps - 1] = (s1 * s1 / s2).min(s1.min(s2)).max(0.0);
        }
    }

    let tail = options.tail_spec.tail_factor;
    let tail_se_sq = options.tail_spec.tail_se * options.tail_spec.tail_se;
    let latest = latest_diagonal(t);

    let mut proj = vec![0.0; n * n];
    for i in 0..n {
        let start = n - 1 - i;
        proj[i * n + start] = latest[i];
        for k in (start + 1)..n {
            proj[i * n + k] = proj[i * n + k - 1] * ldf[k - 1];
        }
    }

    let mut ultimate = vec![0.0; n];
    let mut reserve = vec![0.0; n];
    for i in 0..n {
        ultimate[i] = proj[i * n + (n - 1)] * tail;
        reserve[i] = ultimate[i] - latest[i];
    }

    let mut mse = vec![0.0; n];
    for i in 0..n {
        let c_last = proj[i * n + (n - 1)];
        let mut pre_tail_mse = 0.0;
        if i >= 1 {
            let start = n - 1 - i;
            let mut acc = 0.0;
            for k in start..n_steps {
                let fk = ldf[k];
                let sk2 = sigma_sq[k];
                let c_hat = proj[i * n + k];
                let s = s_col[k];
                if fk > 0.0 && c_hat > 0.0 && s > 0.0 {
                    acc += (sk2 / (fk * fk)) * (1.0 / c_hat.powf(alpha) + 1.0 / s);
                }
            }
            pre_tail_mse = c_last * c_last * acc;
        }
        let tail_process = if c_last > 0.0 {
            c_last * options.tail_spec.tail_sigma_sq
        } else {
            0.0
        };
        let tail_param = c_last * c_last * tail_se_sq;
        mse[i] = tail * tail * pre_tail_mse + tail_process + tail_param;
    }

    let standard_error: Vec<f64> = mse
        .iter()
        .map(|&m| if m.is_finite() { m.max(0.0).sqrt() } else { f64::NAN })
        .collect();
    let cv: Vec<f64> = standard_error
        .iter()
        .zip(reserve.iter())
        .map(|(s, r)| if *r > 0.0 { s / r } else { 0.0 })
        .collect();
    let total_reserve: f64 = reserve.iter().sum();

    let mut total_mse: f64 = mse.iter().sum();
    for i in 0..n {
        let c_last_i = proj[i * n + (n - 1)];
        let start = n - 1 - i;
        let mut cov_sum = 0.0;
        if i >= 1 {
            for k in start..n_steps {
                let fk = ldf[k];
                let sk2 = sigma_sq[k];
                let s = s_col[k];
                if fk > 0.0 && s > 0.0 {
                    cov_sum += sk2 / (fk * fk * s);
                }
            }
        }
        for j in (i + 1)..n {
            let c_last_j = proj[j * n + (n - 1)];
            total_mse +=
                2.0 * ultimate[i] * ultimate[j] * cov_sum + 2.0 * c_last_i * c_last_j * tail_se_sq;
        }
    }

    let total_standard_error = if total_mse.is_finite() {
        total_mse.max(0.0).sqrt()
    } else {
        f64::NAN
    };
    let total_cv = if total_reserve > 0.0 {
        total_standard_error / total_reserve
    } else {
        0.0
    };

    Ok(Mack {
        ldf,
        sigma_sq,
        ultimate,
        reserve,
        standard_error,
        cv,
        total_reserve,
        total_mse,
        total_standard_error,
        total_cv,
    })
}

fn validate_tail_spec(spec: MackTailSpec) -> Result<(), ValidationError> {
    validate_tail(spec.tail_factor)?;
    if !spec.tail_sigma_sq.is_finite() || !spec.tail_se.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if spec.tail_sigma_sq < 0.0 || spec.tail_se < 0.0 {
        return Err(ValidationError::Negative);
    }
    Ok(())
}

#[cfg(test)]
mod observed_count_tests {
    use super::*;

    fn rectangular_data() -> Vec<f64> {
        #[rustfmt::skip]
        let data = vec![
            100.0, 150.0, 180.0, 198.0,
            120.0, 180.0, 216.0, f64::NAN,
            140.0, 210.0, f64::NAN, f64::NAN,
        ];
        data
    }

    #[test]
    fn analyze_observed_counts_projects_rectangular_triangle() {
        let data = rectangular_data();
        let view = TriangleView {
            n_origins: 3,
            n_devs: 4,
            data: &data,
        };

        let result = analyze_observed_counts(&view, &[4, 3, 2], 1.0).expect("mack");

        assert_eq!(result.ldf.len(), 3);
        assert_eq!(result.reserve.len(), 3);
        assert_eq!(result.standard_error.len(), 3);
        assert!(result.total_reserve.is_finite());
        assert!(result.total_standard_error.is_finite());
        assert!(result.ultimate[0] >= 198.0);
        assert!(result.ultimate[2] > 210.0);
    }

    #[test]
    fn analyze_observed_counts_rejects_bad_count() {
        let data = rectangular_data();
        let view = TriangleView {
            n_origins: 3,
            n_devs: 4,
            data: &data,
        };

        assert_eq!(
            analyze_observed_counts(&view, &[4, 5, 2], 1.0).map(|_| ()),
            Err(ValidationError::ShapeMismatch)
        );
    }
}
