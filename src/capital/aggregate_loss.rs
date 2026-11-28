//! Compound aggregate loss distributions via Panjer (1981) recursion and
//! Robertson (1992) tilted radix-2 FFT characteristic-function inversion.
//!
//! Implements:
//!
//! * H. H. Panjer, *"Recursive Evaluation of a Family of Compound
//!   Distributions"*, *ASTIN Bulletin*, Vol. 12, No. 1 (1981), pp. 22–26.
//! * J. P. Robertson, *"The Computation of Aggregate Loss Distributions"*,
//!   *PCAS*, Vol. LXXIX (1992), pp. 57–133.
//! * S. A. Klugman, H. H. Panjer & G. E. Willmot, *Loss Models: From Data to
//!   Decisions* (5th ed.), Chapter 9 (Panjer `(a, b, 0)` class, method of
//!   rounding, and mean-preserving LEV discretization).
//!
//! ## Panjer `(a, b, 0)` Recursion
//!
//! For compound sum `S = sum_{i=1}^N X_i` where `N` belongs to the `(a, b, 0)`
//! class (`p_k / p_{k-1} = a + b / k` for `k >= 1`: Poisson, Binomial,
//! Negative Binomial) and discrete severity `X` has PMF `f_0, f_1, ..., f_m`
//! on grid `0, h, 2h, ..., m*h`:
//!
//! `g_0 = P_N(f_0)`,
//! `g_x = (1 / (1 - a * f_0)) * sum_{y=1}^{min(x, m)} (a + b * y / x) * f_y * g_{x - y}`.
//!
//! ## Tilted Radix-2 FFT Inversion
//!
//! For large grids (`N >= 2^12`) where `O(N^2)` Panjer recursion slows down or
//! `g_0 = P_N(f_0)` risks underflow, [`fft_aggregate_loss`] computes
//! `phi_S(t) = P_N(phi_X(t))` in `O(N log N)` using a pure-Rust, zero-unsafe
//! iterative radix-2 Cooley–Tukey FFT with exponential tilting
//! `f_tilde_k = exp(-theta * k) * f_k` to suppress circular wrap-around
//! aliasing by `exp(-theta * N)`.

use std::f64::consts::TAU;

use crate::excess::severity::SeverityDistribution;
use crate::validate::{ValidationError, validate_nonnegative_finite_slice};

/// Claim-count frequency distribution in the Panjer `(a, b, 0)` class.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum FrequencyDistribution {
    /// Poisson(`lambda`): `E[N] = Var(N) = lambda`, `a = 0`, `b = lambda`.
    Poisson {
        /// Poisson mean intensity `lambda > 0`.
        lambda: f64,
    },
    /// Binomial(`m`, `q`): `E[N] = m q`, `Var(N) = m q (1 - q)`,
    /// `a = -q / (1 - q)`, `b = (m + 1) q / (1 - q)`.
    Binomial {
        /// Number of trials `m >= 1`.
        m: u64,
        /// Per-trial claim probability `q in (0, 1)`.
        q: f64,
    },
    /// Negative Binomial(`r`, `beta`) in Klugman–Panjer–Willmot parameterization:
    /// `E[N] = r * beta`, `Var(N) = r * beta * (1 + beta)`,
    /// `a = beta / (1 + beta)`, `b = (r - 1) * beta / (1 + beta)`.
    NegativeBinomial {
        /// Shape / dispersion parameter `r > 0`.
        r: f64,
        /// Scale parameter `beta > 0`.
        beta: f64,
    },
}

impl FrequencyDistribution {
    /// Construct a `Poisson(lambda)` frequency distribution.
    pub const fn poisson(lambda: f64) -> Self {
        Self::Poisson { lambda }
    }

    /// Construct a `Binomial(m, q)` frequency distribution.
    pub const fn binomial(m: u64, q: f64) -> Self {
        Self::Binomial { m, q }
    }

    /// Construct a `NegativeBinomial(r, beta)` frequency distribution.
    pub const fn negative_binomial(r: f64, beta: f64) -> Self {
        Self::NegativeBinomial { r, beta }
    }

    /// Evaluate the compound aggregate distribution via exact Panjer `(a, b, 0)` recursion.
    pub fn compound_panjer(
        self,
        severity_pmf: &[f64],
        step: f64,
        n_steps: usize,
    ) -> Result<AggregateLossDistribution, ValidationError> {
        panjer_recursion(self, severity_pmf, step, n_steps)
    }

    /// Evaluate the compound aggregate distribution via exponentially tilted radix-2 FFT.
    pub fn compound_fft(
        self,
        severity_pmf: &[f64],
        step: f64,
        fft_size: usize,
        tilt: Option<f64>,
    ) -> Result<AggregateLossDistribution, ValidationError> {
        fft_aggregate_loss(self, severity_pmf, step, fft_size, tilt)
    }

    /// Construct a Negative Binomial distribution matching a target `mean > 0`
    /// and overdispersed `variance > mean`.
    pub fn negative_binomial_from_mean_variance(
        mean: f64,
        variance: f64,
    ) -> Result<Self, ValidationError> {
        if !mean.is_finite() || !variance.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if !(mean > 0.0) || !(variance > mean) {
            return Err(ValidationError::InvalidParameter(
                "Negative Binomial requires 0 < mean < variance",
            ));
        }
        let beta = (variance - mean) / mean;
        let r = mean / beta;
        if !beta.is_finite() || !(beta > 0.0) || !r.is_finite() || !(r > 0.0) {
            return Err(ValidationError::NotFinite);
        }
        Ok(Self::NegativeBinomial { r, beta })
    }

    /// Validate distribution parameters.
    pub fn validate(&self) -> Result<(), ValidationError> {
        match *self {
            Self::Poisson { lambda } => {
                if !lambda.is_finite() {
                    return Err(ValidationError::NotFinite);
                }
                if !(lambda > 0.0) {
                    return Err(ValidationError::InvalidParameter(
                        "Poisson lambda must be strictly positive",
                    ));
                }
            }
            Self::Binomial { m, q } => {
                if !q.is_finite() {
                    return Err(ValidationError::NotFinite);
                }
                if m == 0 || !(q > 0.0 && q < 1.0) {
                    return Err(ValidationError::InvalidParameter(
                        "Binomial requires m >= 1 and 0 < q < 1",
                    ));
                }
            }
            Self::NegativeBinomial { r, beta } => {
                if !r.is_finite() || !beta.is_finite() {
                    return Err(ValidationError::NotFinite);
                }
                if !(r > 0.0) || !(beta > 0.0) {
                    return Err(ValidationError::InvalidParameter(
                        "Negative Binomial requires r > 0 and beta > 0",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Panjer `(a, b)` recursion parameters.
    #[inline]
    pub fn panjer_ab(&self) -> (f64, f64) {
        match *self {
            Self::Poisson { lambda } => (0.0, lambda),
            Self::Binomial { m, q } => {
                let ratio = q / (1.0 - q);
                (-ratio, (m as f64 + 1.0) * ratio)
            }
            Self::NegativeBinomial { r, beta } => {
                let p = beta / (1.0 + beta);
                (p, (r - 1.0) * p)
            }
        }
    }

    /// Theoretical frequency mean `E[N]`.
    #[inline]
    pub fn mean(&self) -> f64 {
        match *self {
            Self::Poisson { lambda } => lambda,
            Self::Binomial { m, q } => m as f64 * q,
            Self::NegativeBinomial { r, beta } => r * beta,
        }
    }

    /// Theoretical frequency variance `Var(N)`.
    #[inline]
    pub fn variance(&self) -> f64 {
        match *self {
            Self::Poisson { lambda } => lambda,
            Self::Binomial { m, q } => m as f64 * q * (1.0 - q),
            Self::NegativeBinomial { r, beta } => r * beta * (1.0 + beta),
        }
    }

    /// Real probability generating function `P_N(z)` for real `z in [0, 1]`.
    #[inline]
    pub fn pgf_real(&self, z: f64) -> f64 {
        match *self {
            Self::Poisson { lambda } => (lambda * (z - 1.0)).exp(),
            Self::Binomial { m, q } => (1.0 - q + q * z).powf(m as f64),
            Self::NegativeBinomial { r, beta } => (1.0 - beta * (z - 1.0)).powf(-r),
        }
    }

    /// Complex probability generating function `P_N(re + i * im)`.
    #[inline]
    fn pgf_complex(&self, re: f64, im: f64) -> (f64, f64) {
        match *self {
            Self::Poisson { lambda } => {
                let mag = (lambda * (re - 1.0)).exp();
                let angle = lambda * im;
                (mag * angle.cos(), mag * angle.sin())
            }
            Self::Binomial { m, q } => {
                let wr = 1.0 - q + q * re;
                let wi = q * im;
                let r = wr.hypot(wi);
                let phi = wi.atan2(wr);
                let mag = r.powf(m as f64);
                let ang = (m as f64) * phi;
                (mag * ang.cos(), mag * ang.sin())
            }
            Self::NegativeBinomial { r, beta } => {
                let wr = 1.0 + beta - beta * re;
                let wi = -beta * im;
                let radius = wr.hypot(wi);
                let phi = wi.atan2(wr);
                let mag = radius.powf(-r);
                let ang = -r * phi;
                (mag * ang.cos(), mag * ang.sin())
            }
        }
    }
}

/// Discretized aggregate loss distribution on the uniform lattice `0, step, 2*step, ...`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct AggregateLossDistribution {
    /// Grid step size `h > 0`.
    pub step: f64,
    /// Discrete aggregate PMF `g_k = P(S = k * step)` (length `n_steps`).
    pub pmf: Vec<f64>,
    /// Cumulative distribution function `G_k = P(S <= k * step)` (length `n_steps`).
    pub cdf: Vec<f64>,
    /// Total captured probability mass `sum_k g_k` (`<= 1.0`; `< 1.0` reflects right truncation).
    pub total_mass: f64,
    /// Numerical mean `sum_k (k * step) * g_k`.
    pub mean: f64,
    /// Numerical variance `sum_k (k * step - mean)^2 * g_k`.
    pub variance: f64,
    /// Numerical standard deviation `sqrt(variance)`.
    pub std_dev: f64,
    /// Exact compound theoretical mean `E[N] * E[X]` from the input severity PMF.
    pub exact_theoretical_mean: f64,
    /// Exact compound theoretical variance `E[N] * Var(X) + Var(N) * E[X]^2`.
    pub exact_theoretical_variance: f64,
}

impl AggregateLossDistribution {
    /// Discrete Value at Risk `VaR_alpha(S) = inf { x : F_S(x) >= alpha }`.
    pub fn var(&self, alpha: f64) -> Result<f64, ValidationError> {
        if !alpha.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err(ValidationError::InvalidProbability { value: alpha });
        }
        if alpha < 1.0 && alpha > self.total_mass + 1e-12 {
            return Err(ValidationError::InvalidParameter(
                "truncated aggregate grid has insufficient probability mass above alpha for VaR",
            ));
        }
        for (k, &f) in self.cdf.iter().enumerate() {
            if f >= alpha - 1e-15 {
                return Ok(k as f64 * self.step);
            }
        }
        Ok((self.pmf.len().saturating_sub(1)) as f64 * self.step)
    }

    /// Exact coherent Tail Value at Risk `TVaR_alpha(S)` on the discrete lattice
    /// (`(1 / (1 - alpha)) * int_alpha^1 F^-(u) du`), normalized by `total_mass`.
    pub fn tvar(&self, alpha: f64) -> Result<f64, ValidationError> {
        if !alpha.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err(ValidationError::InvalidProbability { value: alpha });
        }
        if alpha <= 0.0 {
            if !(self.total_mass > 0.0) {
                return Err(ValidationError::InvalidParameter(
                    "aggregate loss distribution has zero captured probability mass",
                ));
            }
            return Ok(self.mean / self.total_mass);
        }
        if alpha >= 1.0 {
            return Ok((self.pmf.len().saturating_sub(1)) as f64 * self.step);
        }
        let mut prev_cdf = 0.0_f64;
        let mut integral = 0.0_f64;
        for (k, &pk) in self.pmf.iter().enumerate() {
            let next_cdf = (prev_cdf + pk).min(1.0);
            let lo = prev_cdf.max(alpha);
            let hi = next_cdf;
            if hi > lo {
                integral += (hi - lo) * (k as f64 * self.step);
            }
            prev_cdf = next_cdf;
        }
        let captured_tail = prev_cdf - alpha;
        if !(captured_tail > 1e-15) {
            return Err(ValidationError::InvalidParameter(
                "truncated aggregate grid has insufficient probability mass above alpha for TVaR",
            ));
        }
        Ok(integral / captured_tail)
    }

    /// Exact net stop-loss premium `E[(S - retention)_+] = sum_k max(0, k*h - retention) * g_k`.
    pub fn stop_loss_premium(&self, retention: f64) -> Result<f64, ValidationError> {
        if !retention.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if retention < 0.0 {
            return Err(ValidationError::Negative);
        }
        let mut premium = 0.0;
        for (k, &pk) in self.pmf.iter().enumerate() {
            let x = k as f64 * self.step;
            if x > retention {
                premium += (x - retention) * pk;
            }
        }
        Ok(premium)
    }
}

/// Evaluate a compound aggregate loss distribution using Panjer's (1981) exact
/// `(a, b, 0)` recursion up to `n_steps` grid points (`0, step, ..., (n_steps - 1) * step`).
///
/// Uses a zero-allocation contiguous inner loop over `g[..n_steps]` suitable
/// for compiler SIMD auto-vectorization.
pub fn panjer_recursion(
    frequency: FrequencyDistribution,
    severity_pmf: &[f64],
    step: f64,
    n_steps: usize,
) -> Result<AggregateLossDistribution, ValidationError> {
    validate_compound_inputs(frequency, severity_pmf, step, n_steps)?;
    let (a, b) = frequency.panjer_ab();
    let f0 = severity_pmf[0];
    let denom = 1.0 - a * f0;
    if !denom.is_finite() || denom <= 1e-15 {
        return Err(ValidationError::InvalidParameter(
            "Panjer recursion denominator 1 - a * f_0 is non-positive",
        ));
    }
    let inv_denom = 1.0 / denom;

    let g0 = frequency.pgf_real(f0);
    if !g0.is_finite() || g0 < f64::MIN_POSITIVE {
        return Err(ValidationError::InvalidParameter(
            "Panjer recursion g_0 underflowed normal f64 range; use fft_aggregate_loss instead",
        ));
    }

    let mut g = vec![0.0_f64; n_steps];
    g[0] = g0;
    let m_max = severity_pmf.len() - 1;
    // Precompute f_y and y * f_y for y = 1..=m_max so the inner loop splits into
    // two pure dot-products without per-element float casts or branchy index math.
    let f_pos = &severity_pmf[1..=m_max];
    let yf_pos: Vec<f64> = f_pos
        .iter()
        .enumerate()
        .map(|(idx, &fy)| (idx + 1) as f64 * fy)
        .collect();
    let is_poisson = a == 0.0;

    for x in 1..n_steps {
        let upper = x.min(m_max);
        let inv_x = 1.0 / x as f64;
        let g_slice = &g[(x - upper)..x];
        let acc = if is_poisson {
            let mut acc_b = 0.0;
            for (&yfy, &g_prev) in yf_pos[..upper].iter().zip(g_slice.iter().rev()) {
                acc_b += yfy * g_prev;
            }
            (b * inv_x) * acc_b
        } else {
            let mut acc_a = 0.0;
            let mut acc_b = 0.0;
            for ((&fy, &yfy), &g_prev) in f_pos[..upper]
                .iter()
                .zip(yf_pos[..upper].iter())
                .zip(g_slice.iter().rev())
            {
                acc_a += fy * g_prev;
                acc_b += yfy * g_prev;
            }
            a * acc_a + (b * inv_x) * acc_b
        };
        let gx = acc * inv_denom;
        if !gx.is_finite() || gx < -1e-6 {
            return Err(ValidationError::InvalidParameter(
                "Panjer recursion became numerically unstable (e.g. high-q Binomial); use fft_aggregate_loss",
            ));
        }
        g[x] = gx.max(0.0);
    }

    Ok(summarize_aggregate(frequency, severity_pmf, step, g))
}

/// Evaluate a compound aggregate loss distribution using Robertson's (1992)
/// radix-2 FFT characteristic-function inversion with exponential tilting.
///
/// * `fft_size` — power of two (`>= severity_pmf.len()` and `>= 2`).
/// * `tilt` — exponential tilting parameter `theta >= 0.0` (`None` uses the
///   automatic `f64`-balanced default `14.0 / fft_size as f64`, suppressing
///   wrap-around aliasing by `exp(-14) ≈ 8.3e-7` while limiting right-edge
///   amplification to ~`1e-10`).
pub fn fft_aggregate_loss(
    frequency: FrequencyDistribution,
    severity_pmf: &[f64],
    step: f64,
    fft_size: usize,
    tilt: Option<f64>,
) -> Result<AggregateLossDistribution, ValidationError> {
    validate_compound_inputs(frequency, severity_pmf, step, fft_size)?;
    if fft_size < 2 || !fft_size.is_power_of_two() {
        return Err(ValidationError::InvalidParameter(
            "fft_size must be a power of two >= 2",
        ));
    }
    if fft_size < severity_pmf.len() {
        return Err(ValidationError::InvalidParameter(
            "fft_size must be >= severity_pmf.len()",
        ));
    }
    let theta = match tilt {
        Some(t) => {
            if !t.is_finite() || t < 0.0 || t * (fft_size - 1) as f64 > 700.0 {
                return Err(ValidationError::InvalidParameter(
                    "exponential tilt must be finite, >= 0, and satisfy tilt * (fft_size - 1) <= 700",
                ));
            }
            t
        }
        None => 14.0 / fft_size as f64,
    };

    let mut re = vec![0.0; fft_size];
    let mut im = vec![0.0; fft_size];
    if theta > 0.0 {
        let step_down = (-theta).exp();
        let mut scale = 1.0;
        for (k, &fk) in severity_pmf.iter().enumerate() {
            if k.trailing_zeros() >= 6 {
                scale = (-theta * k as f64).exp();
            }
            re[k] = fk * scale;
            scale *= step_down;
        }
    } else {
        re[..severity_pmf.len()].copy_from_slice(severity_pmf);
    }

    // Precompute exact N/2 twiddle table once and share across forward & inverse FFTs
    // (cuts sin_cos calls by 4x while preserving exact twiddle precision).
    let (tw_re, tw_im) = build_fft_twiddles(fft_size);

    // Forward radix-2 FFT: severity characteristic function.
    fft_radix2_in_place(&mut re, &mut im, &tw_re, &tw_im, false);

    // Compound PGF evaluation in frequency domain: phi_S(u) = P_N(phi_X(u)).
    for u in 0..fft_size {
        let (gr, gi) = frequency.pgf_complex(re[u], im[u]);
        if !gr.is_finite() || !gi.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        re[u] = gr;
        im[u] = gi;
    }

    // Inverse radix-2 FFT back to loss lattice.
    fft_radix2_in_place(&mut re, &mut im, &tw_re, &tw_im, true);

    // Untilt and suppress sub-noise-floor FFT roundoff dust scaled to the tilted peak
    // so exponential untilting by exp(theta * k) doesn't amplify roundoff without
    // truncating legitimate tail mass on heavily tilted grids.
    let max_re = re
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let noise_floor = if theta > 0.0 {
        4.0 * f64::EPSILON * max_re
    } else {
        0.0
    };
    let mut g = vec![0.0; fft_size];
    if theta > 0.0 {
        let step_up = theta.exp();
        let mut scale = 1.0;
        for k in 0..fft_size {
            if k.trailing_zeros() >= 6 {
                scale = (theta * k as f64).exp();
            }
            let rk = re[k];
            if rk.is_finite() && rk > noise_floor {
                let raw = rk * scale;
                if raw.is_finite() && raw > 0.0 {
                    g[k] = raw;
                }
            }
            scale *= step_up;
        }
    } else {
        for k in 0..fft_size {
            let rk = re[k];
            if rk.is_finite() && rk > 0.0 {
                g[k] = rk;
            }
        }
    }

    Ok(summarize_aggregate(frequency, severity_pmf, step, g))
}

/// Discretize a continuous [`SeverityDistribution`] onto the lattice
/// `0, step, 2*step, ..., max_index * step` using the **Method of Rounding**
/// (Klugman, Panjer & Willmot, §9.6.1):
///
/// `f_0 = F(step / 2)`, `f_k = F((k + 0.5) * step) - F((k - 0.5) * step)`.
///
/// If `capture_upper_tail` is `true`, the remaining survival mass
/// `S((max_index - 0.5) * step)` is placed on `f_{max_index}` so `sum(f) == 1`.
pub fn discretize_rounding(
    dist: &SeverityDistribution,
    step: f64,
    max_index: usize,
    capture_upper_tail: bool,
) -> Result<Vec<f64>, ValidationError> {
    dist.validate()?;
    if !step.is_finite() || step <= 0.0 || max_index == 0 {
        return Err(ValidationError::InvalidParameter(
            "discretization requires step > 0 and max_index >= 1",
        ));
    }
    let mut pmf = vec![0.0; max_index + 1];
    let mut prev_cdf = 0.0_f64;
    for k in 0..=max_index {
        let next_cdf = if k == max_index && capture_upper_tail {
            1.0
        } else {
            let x_hi = (k as f64 + 0.5) * step;
            1.0 - dist.survival(x_hi)
        };
        pmf[k] = (next_cdf - prev_cdf).max(0.0);
        prev_cdf = next_cdf;
    }
    Ok(pmf)
}

/// Discretize a continuous [`SeverityDistribution`] onto the lattice
/// `0, step, 2*step, ..., max_index * step` using **mean-preserving Limited
/// Expected Value (LEV) local moment matching** (Klugman, Panjer & Willmot
/// §9.6.2 / Wang 1998):
///
/// `f_0 = 1 - LEV(step) / step`,
/// `f_k = (2 LEV(k * step) - LEV((k - 1) * step) - LEV((k + 1) * step)) / step`
/// for `1 <= k < max_index`, and remaining mass at `max_index`.
///
/// Preserves `E[X ∧ (max_index * step)]` identically!
pub fn discretize_mean_preserving(
    dist: &SeverityDistribution,
    step: f64,
    max_index: usize,
) -> Result<Vec<f64>, ValidationError> {
    dist.validate()?;
    if !step.is_finite() || step <= 0.0 || max_index < 2 {
        return Err(ValidationError::InvalidParameter(
            "mean-preserving discretization requires step > 0 and max_index >= 2",
        ));
    }
    let mut levs = vec![0.0; max_index + 1];
    for k in 1..=max_index {
        levs[k] = dist.limited_expected_value(k as f64 * step)?;
    }

    let mut pmf = vec![0.0; max_index + 1];
    pmf[0] = (1.0 - levs[1] / step).max(0.0);
    for k in 1..max_index {
        pmf[k] = ((2.0 * levs[k] - levs[k - 1] - levs[k + 1]) / step).max(0.0);
    }
    pmf[max_index] = ((levs[max_index] - levs[max_index - 1]) / step).max(0.0);
    Ok(pmf)
}

fn validate_compound_inputs(
    frequency: FrequencyDistribution,
    severity_pmf: &[f64],
    step: f64,
    n_steps: usize,
) -> Result<(), ValidationError> {
    frequency.validate()?;
    if severity_pmf.is_empty() || n_steps < 2 {
        return Err(ValidationError::ShapeMismatch);
    }
    if !step.is_finite() || step <= 0.0 {
        return Err(ValidationError::InvalidParameter(
            "lattice step size must be finite and strictly positive",
        ));
    }
    validate_nonnegative_finite_slice(severity_pmf)?;
    let sum_f: f64 = severity_pmf.iter().sum();
    if (sum_f - 1.0).abs() > 1e-6 {
        return Err(ValidationError::InvalidParameter(
            "severity PMF must sum to 1.0 within 1e-6",
        ));
    }
    Ok(())
}

fn summarize_aggregate(
    frequency: FrequencyDistribution,
    severity_pmf: &[f64],
    step: f64,
    pmf: Vec<f64>,
) -> AggregateLossDistribution {
    // Exact moments of input discrete severity X (two-pass centered variance avoids cancellation).
    let mut ex = 0.0;
    for (k, &fk) in severity_pmf.iter().enumerate() {
        ex += (k as f64 * step) * fk;
    }
    let mut var_x = 0.0;
    for (k, &fk) in severity_pmf.iter().enumerate() {
        let dx = k as f64 * step - ex;
        var_x += dx * dx * fk;
    }
    let var_x = var_x.max(0.0);
    let en = frequency.mean();
    let var_n = frequency.variance();
    let exact_theoretical_mean = en * ex;
    let exact_theoretical_variance = en * var_x + var_n * ex * ex;

    let mut cdf = Vec::with_capacity(pmf.len());
    let mut running = 0.0;
    let mut mean = 0.0;
    for (k, &gk) in pmf.iter().enumerate() {
        running += gk;
        cdf.push(running);
        mean += (k as f64 * step) * gk;
    }
    let mut variance = 0.0;
    for (k, &gk) in pmf.iter().enumerate() {
        let dx = k as f64 * step - mean;
        variance += dx * dx * gk;
    }
    let variance = variance.max(0.0);

    AggregateLossDistribution {
        step,
        pmf,
        cdf,
        total_mass: running,
        mean,
        variance,
        std_dev: variance.sqrt(),
        exact_theoretical_mean,
        exact_theoretical_variance,
    }
}

/// Build exact forward twiddle factors `exp(-i * 2*pi*k / n)` for `k = 0..n/2`.
fn build_fft_twiddles(n: usize) -> (Vec<f64>, Vec<f64>) {
    let half = n >> 1;
    let mut tw_re = Vec::with_capacity(half);
    let mut tw_im = Vec::with_capacity(half);
    let angle_step = -TAU / n as f64;
    for k in 0..half {
        let (s, c) = (k as f64 * angle_step).sin_cos();
        tw_re.push(c);
        tw_im.push(s);
    }
    (tw_re, tw_im)
}

/// Pure-Rust iterative in-place radix-2 Cooley–Tukey FFT (`len` must be power of 2).
fn fft_radix2_in_place(
    re: &mut [f64],
    im: &mut [f64],
    tw_re: &[f64],
    tw_im: &[f64],
    inverse: bool,
) {
    let n = re.len();
    debug_assert_eq!(n, im.len());
    debug_assert!(n.is_power_of_two());

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // Cooley–Tukey butterfly stages using precomputed exact twiddles (inverse conjugates wi).
    let im_sign = if inverse { -1.0 } else { 1.0 };
    let mut len = 2usize;
    while len <= n {
        let half = len >> 1;
        let stride = n / len;

        for k in 0..half {
            let tw_idx = k * stride;
            let wr = tw_re[tw_idx];
            let wi = im_sign * tw_im[tw_idx];
            let mut start = k;
            while start < n {
                let odd = start + half;
                let tr = wr * re[odd] - wi * im[odd];
                let ti = wr * im[odd] + wi * re[odd];
                let er = re[start];
                let ei = im[start];
                re[start] = er + tr;
                im[start] = ei + ti;
                re[odd] = er - tr;
                im[odd] = ei - ti;
                start += len;
            }
        }
        len <<= 1;
    }

    if inverse {
        let inv_n = 1.0 / n as f64;
        for i in 0..n {
            re[i] *= inv_n;
            im[i] *= inv_n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panjer_and_tilted_fft_agree_to_high_precision_and_match_exact_moments() {
        // Klugman, Panjer & Willmot style discrete severity on {0, 1, 2, 3}.
        let sev = [0.0, 0.5, 0.3, 0.2];
        let step = 100.0;
        let n_grid = 256usize;

        for freq in [
            FrequencyDistribution::Poisson { lambda: 3.5 },
            FrequencyDistribution::NegativeBinomial { r: 2.5, beta: 1.4 },
            FrequencyDistribution::Binomial { m: 10, q: 0.35 },
        ] {
            let panjer = panjer_recursion(freq, &sev, step, n_grid).expect("panjer");
            let fft = fft_aggregate_loss(freq, &sev, step, n_grid, None).expect("fft");

            assert!((panjer.total_mass - 1.0).abs() < 1e-10);
            assert!(
                (panjer.mean - panjer.exact_theoretical_mean).abs() / panjer.exact_theoretical_mean
                    < 1e-10
            );
            assert!(
                (panjer.variance - panjer.exact_theoretical_variance).abs()
                    / panjer.exact_theoretical_variance
                    < 1e-9
            );

            // L_inf agreement between Panjer recursion and tilted FFT.
            let max_diff = panjer
                .pmf
                .iter()
                .zip(fft.pmf.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(max_diff < 1e-10, "freq={freq:?} max_diff={max_diff}");
        }
    }

    #[test]
    fn mean_preserving_discretization_matches_exact_lev() {
        let dist = SeverityDistribution::Lognormal {
            mu: 9.0,
            sigma: 0.8,
        };
        let step = 1_000.0;
        let max_idx = 200usize;
        let pmf = discretize_mean_preserving(&dist, step, max_idx).unwrap();
        let disc_mean: f64 = pmf
            .iter()
            .enumerate()
            .map(|(k, &p)| k as f64 * step * p)
            .sum();
        let exact_lev = dist.limited_expected_value(max_idx as f64 * step).unwrap();
        assert!((disc_mean - exact_lev).abs() / exact_lev < 1e-12);
    }
}
