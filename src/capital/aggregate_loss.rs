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

    /// Evaluate the compound aggregate distribution via exact Panjer `(a, b, 0)`
    /// recursion (with automatic scaling-and-squaring; see [`panjer_recursion_with_options`]).
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

    /// Natural log of the real PGF, `ln P_N(z)` for `z in [0, 1]`, evaluated
    /// without forming `P_N(z)` (no underflow for large `E[N]`).
    #[inline]
    pub fn ln_pgf_real(&self, z: f64) -> f64 {
        match *self {
            Self::Poisson { lambda } => lambda * (z - 1.0),
            Self::Binomial { m, q } => m as f64 * (-q * (1.0 - z)).ln_1p(),
            Self::NegativeBinomial { r, beta } => -r * (beta * (1.0 - z)).ln_1p(),
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

/// Default maximum fraction of the tail probability `1 - alpha` that may be
/// missing (truncated beyond the grid) or misplaced (FFT wrap-around) before
/// [`AggregateLossDistribution::var`] / [`AggregateLossDistribution::tvar`]
/// refuse to answer: `0.01`, i.e. at most 1% of the tail mass.
pub const DEFAULT_MAX_MISSING_TAIL_FRACTION: f64 = 0.01;

/// Absolute probability floor treated as floating-point round-off (`1e-12`):
/// missing / aliased mass below it never triggers an error (needed at `alpha = 1`).
pub const MASS_ROUNDOFF_FLOOR: f64 = 1e-12;

/// Tunables for [`panjer_recursion_with_options`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PanjerOptions {
    /// Maximum total negative probability mass that may be clipped to zero
    /// during the recursion (Binomial `a < 0` cancellation); exceeding it is an
    /// error (default `1e-6`). The clipped amount is always reported in
    /// [`AggregateLossDistribution::clipped_negative_mass`].
    pub max_clipped_negative_mass: f64,
    /// Smallest admissible `ln g_0 = ln P_N(f_0)` for a direct recursion
    /// (default `-300.0`, i.e. `g_0 >= ~5e-131`, well clear of subnormals).
    /// Below it, infinitely divisible frequencies (Poisson, Negative Binomial)
    /// switch to scaling-and-squaring: the recursion runs on `N / 2^k`
    /// (`lambda / 2^k` or `r / 2^k`) and the result is self-convolved `k`
    /// times. Binomial frequencies are not split and error only if `g_0`
    /// actually underflows.
    pub min_ln_g0: f64,
    /// Maximum number of scaling-and-squaring halvings `k` (default `32`).
    pub max_scaling_squaring_steps: u32,
}

impl Default for PanjerOptions {
    fn default() -> Self {
        Self {
            max_clipped_negative_mass: 1e-6,
            min_ln_g0: -300.0,
            max_scaling_squaring_steps: 32,
        }
    }
}

/// Tunables for [`fft_aggregate_loss_with_options`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FftOptions {
    /// Exponential tilt `theta >= 0` (`None` = automatic `14 / fft_size`).
    pub tilt: Option<f64>,
    /// Maximum admissible [`AggregateLossDistribution::aliasing_error_estimate`]
    /// (upper bound on probability mass wrapped around the circular grid);
    /// exceeding it is an error (default `1e-6`).
    pub max_aliasing_error: f64,
}

impl Default for FftOptions {
    fn default() -> Self {
        Self {
            tilt: None,
            max_aliasing_error: 1e-6,
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
    /// Relative mean coverage gap `(exact_theoretical_mean - mean) / exact_theoretical_mean`
    /// (`0` when the exact mean is zero). Positive values reflect truncation
    /// and/or FFT wrap-around.
    #[serde(default)]
    pub relative_mean_error: f64,
    /// Rigorous upper bound on probability mass misplaced by circular FFT
    /// wrap-around (always `0` for Panjer recursion). For a grid of `N` points
    /// it is `min(gap / (N h) - missing, e^{-θN} missing / (1 - e^{-θN}))`,
    /// where `gap = exact mean - mean` and `missing = 1 - total_mass`
    /// (every wrapped unit of mass lowers the mean by at least `N h`; tilting
    /// damps wrapped mass by `e^{-θN}`).
    #[serde(default)]
    pub aliasing_error_estimate: f64,
    /// Total negative probability mass clipped to zero during Panjer recursion
    /// (non-zero only for Binomial frequencies with round-off cancellation).
    #[serde(default)]
    pub clipped_negative_mass: f64,
    /// Number of Panjer scaling-and-squaring halvings applied (`0` = direct recursion).
    #[serde(default)]
    pub scaling_squaring_steps: u32,
}

impl AggregateLossDistribution {
    /// Probability mass not captured on the grid, `max(0, 1 - total_mass)`.
    #[inline]
    #[must_use]
    pub fn missing_mass(&self) -> f64 {
        (1.0 - self.total_mass).max(0.0)
    }

    /// Top of the captured support: `max { k * step : g_k > 0 }`.
    fn top_of_support(&self) -> Result<f64, ValidationError> {
        self.pmf
            .iter()
            .rposition(|&p| p > 0.0)
            .map(|k| k as f64 * self.step)
            .ok_or(ValidationError::InvalidParameter(
                "aggregate loss distribution has zero captured probability mass",
            ))
    }

    /// Reject when missing (truncated) or aliased mass exceeds
    /// `max(max_missing_tail_fraction * (1 - alpha), MASS_ROUNDOFF_FLOOR)`.
    fn check_tail_coverage(
        &self,
        alpha: f64,
        max_missing_tail_fraction: f64,
        check_missing: bool,
    ) -> Result<(), ValidationError> {
        let budget = (max_missing_tail_fraction * (1.0 - alpha)).max(MASS_ROUNDOFF_FLOOR);
        if check_missing && self.missing_mass() > budget {
            return Err(ValidationError::InvalidParameter(
                "truncated aggregate grid: missing probability mass beyond the grid exceeds the admissible fraction of the tail 1 - alpha; extend the grid",
            ));
        }
        if self.aliasing_error_estimate > budget {
            return Err(ValidationError::InvalidParameter(
                "FFT wrap-around aliasing bound exceeds the admissible fraction of the tail 1 - alpha; enlarge fft_size or increase the tilt",
            ));
        }
        Ok(())
    }

    fn validate_alpha_and_fraction(alpha: f64, fraction: f64) -> Result<(), ValidationError> {
        if !alpha.is_finite() || !fraction.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err(ValidationError::InvalidProbability { value: alpha });
        }
        if fraction < 0.0 {
            return Err(ValidationError::Negative);
        }
        Ok(())
    }

    /// Discrete Value at Risk `VaR_alpha(S) = inf { x : F_S(x) >= alpha }`
    /// with the default [`DEFAULT_MAX_MISSING_TAIL_FRACTION`] aliasing budget.
    ///
    /// See [`Self::var_with_tolerance`].
    pub fn var(&self, alpha: f64) -> Result<f64, ValidationError> {
        self.var_with_tolerance(alpha, DEFAULT_MAX_MISSING_TAIL_FRACTION)
    }

    /// Discrete Value at Risk with an explicit tail-coverage tolerance.
    ///
    /// * Errors when `alpha > total_mass` (quantile lies beyond the grid).
    /// * Errors when the FFT aliasing bound exceeds
    ///   `max_missing_tail_fraction * (1 - alpha)`.
    /// * `alpha = 1` returns the top of the captured support
    ///   (`max { k h : g_k > 0 }`), and errors unless the missing mass is
    ///   below [`MASS_ROUNDOFF_FLOOR`].
    pub fn var_with_tolerance(
        &self,
        alpha: f64,
        max_missing_tail_fraction: f64,
    ) -> Result<f64, ValidationError> {
        Self::validate_alpha_and_fraction(alpha, max_missing_tail_fraction)?;
        let top = self.top_of_support()?;
        if alpha >= 1.0 {
            self.check_tail_coverage(1.0, max_missing_tail_fraction, true)?;
            return Ok(top);
        }
        if alpha > self.total_mass + MASS_ROUNDOFF_FLOOR {
            return Err(ValidationError::InvalidParameter(
                "truncated aggregate grid has insufficient probability mass above alpha for VaR",
            ));
        }
        self.check_tail_coverage(alpha, max_missing_tail_fraction, false)?;
        for (k, &f) in self.cdf.iter().enumerate() {
            if f >= alpha - 1e-15 {
                return Ok(k as f64 * self.step);
            }
        }
        Ok(top)
    }

    /// Coherent Tail Value at Risk `TVaR_alpha(S) = (1 / (1 - alpha)) int_alpha^1 F^-(u) du`
    /// with the default [`DEFAULT_MAX_MISSING_TAIL_FRACTION`] (1% of the tail).
    ///
    /// See [`Self::tvar_with_tolerance`].
    pub fn tvar(&self, alpha: f64) -> Result<f64, ValidationError> {
        self.tvar_with_tolerance(alpha, DEFAULT_MAX_MISSING_TAIL_FRACTION)
    }

    /// Coherent Tail Value at Risk with an explicit tail-coverage tolerance.
    ///
    /// The missing (truncated) mass `1 - total_mass` is **not** renormalized
    /// away (which biases TVaR low on truncated grids). Instead:
    ///
    /// * Errors when the missing mass or the FFT aliasing bound exceeds
    ///   `max(max_missing_tail_fraction * (1 - alpha), MASS_ROUNDOFF_FLOOR)`.
    /// * Otherwise the residual missing mass is placed at the top of the
    ///   captured support, making the result a lower bound whose quantile
    ///   function is non-decreasing, so `TVaR` is monotone in `alpha` on
    ///   `[0, 1]` and `TVaR_1 = VaR_1 = max { k h : g_k > 0 }`.
    /// * `alpha = 0` returns the (coverage-checked) mean.
    pub fn tvar_with_tolerance(
        &self,
        alpha: f64,
        max_missing_tail_fraction: f64,
    ) -> Result<f64, ValidationError> {
        Self::validate_alpha_and_fraction(alpha, max_missing_tail_fraction)?;
        let top = self.top_of_support()?;
        if alpha >= 1.0 {
            self.check_tail_coverage(1.0, max_missing_tail_fraction, true)?;
            return Ok(top);
        }
        self.check_tail_coverage(alpha, max_missing_tail_fraction, true)?;
        let mut prev_cdf = 0.0_f64;
        let mut integral = 0.0_f64;
        for (k, &pk) in self.pmf.iter().enumerate() {
            let next_cdf = (prev_cdf + pk).min(1.0);
            let lo = prev_cdf.max(alpha);
            if next_cdf > lo {
                integral += (next_cdf - lo) * (k as f64 * self.step);
            }
            prev_cdf = next_cdf;
        }
        let tail = 1.0 - alpha;
        let captured_tail = (prev_cdf - alpha).max(0.0);
        let residual = (tail - captured_tail).max(0.0);
        integral += residual * top;
        Ok((integral / tail).min(top))
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
/// `(a, b, 0)` recursion up to `n_steps` grid points (`0, step, ..., (n_steps - 1) * step`)
/// with default [`PanjerOptions`].
pub fn panjer_recursion(
    frequency: FrequencyDistribution,
    severity_pmf: &[f64],
    step: f64,
    n_steps: usize,
) -> Result<AggregateLossDistribution, ValidationError> {
    panjer_recursion_with_options(
        frequency,
        severity_pmf,
        step,
        n_steps,
        &PanjerOptions::default(),
    )
}

/// Panjer recursion with explicit [`PanjerOptions`].
///
/// **Scaling and squaring.** When `ln g_0 = ln P_N(f_0) < options.min_ln_g0`
/// (e.g. Poisson `lambda (1 - f_0) >~ 300`, where `g_0` would otherwise drift
/// towards/below `f64::MIN_POSITIVE` near `lambda ~ 708`), infinitely divisible
/// frequencies are split: with `k = ceil(log2(ln g_0 / min_ln_g0))`, the
/// recursion runs for `Poisson(lambda / 2^k)` or `NB(r / 2^k, beta)` and the
/// resulting grid PMF is convolved with itself `k` times, using the identity
/// `S_N = S_{N/2} + S'_{N/2}` for independent halves. Each squaring is an
/// exact truncated direct convolution
/// `h_x = sum_{y=0}^{x} g_y g_{x-y}` for `x < n_steps` (`O(n_steps^2)`;
/// truncation is exact because all terms have non-negative support), so the
/// total cost is `O(n_steps * m + k * n_steps^2)`.
///
/// **Negative-mass clipping.** For Binomial frequencies (`a < 0`) round-off
/// can produce small negative `g_x`; these are clipped to zero, the clipped
/// total is reported in [`AggregateLossDistribution::clipped_negative_mass`],
/// and the call errors when it exceeds `options.max_clipped_negative_mass`.
pub fn panjer_recursion_with_options(
    frequency: FrequencyDistribution,
    severity_pmf: &[f64],
    step: f64,
    n_steps: usize,
    options: &PanjerOptions,
) -> Result<AggregateLossDistribution, ValidationError> {
    validate_compound_inputs(frequency, severity_pmf, step, n_steps)?;
    if !options.max_clipped_negative_mass.is_finite()
        || options.max_clipped_negative_mass < 0.0
        || !options.min_ln_g0.is_finite()
        || !(options.min_ln_g0 < 0.0)
        || options.min_ln_g0 < LN_MIN_POSITIVE
    {
        return Err(ValidationError::InvalidParameter(
            "Panjer options require max_clipped_negative_mass >= 0 and ln(f64::MIN_POSITIVE) <= min_ln_g0 < 0",
        ));
    }
    let f0 = severity_pmf[0];
    let ln_g0 = frequency.ln_pgf_real(f0);
    if ln_g0.is_nan() {
        return Err(ValidationError::NotFinite);
    }

    let mut squarings = 0u32;
    let mut base = frequency;
    if ln_g0 < options.min_ln_g0 {
        match frequency {
            FrequencyDistribution::Binomial { .. } => {
                if ln_g0 < LN_MIN_POSITIVE {
                    return Err(ValidationError::InvalidParameter(
                        "Panjer recursion g_0 underflowed for Binomial frequency (not split by scaling-and-squaring); use fft_aggregate_loss instead",
                    ));
                }
            }
            FrequencyDistribution::Poisson { lambda } => {
                squarings = scaling_steps(ln_g0, options)?;
                base = FrequencyDistribution::Poisson {
                    lambda: lambda / f64::from(squarings).exp2(),
                };
            }
            FrequencyDistribution::NegativeBinomial { r, beta } => {
                squarings = scaling_steps(ln_g0, options)?;
                base = FrequencyDistribution::NegativeBinomial {
                    r: r / f64::from(squarings).exp2(),
                    beta,
                };
            }
        }
    }

    let (mut g, clipped_negative_mass) = panjer_core(base, severity_pmf, n_steps, options)?;
    for _ in 0..squarings {
        g = self_convolve_truncated(&g);
    }

    let mut out = summarize_aggregate(frequency, severity_pmf, step, g);
    out.clipped_negative_mass = clipped_negative_mass;
    out.scaling_squaring_steps = squarings;
    Ok(out)
}

/// `ln(f64::MIN_POSITIVE)`.
const LN_MIN_POSITIVE: f64 = -708.396_418_532_264_1;

/// Smallest `k` with `ln_g0 / 2^k >= options.min_ln_g0`.
fn scaling_steps(ln_g0: f64, options: &PanjerOptions) -> Result<u32, ValidationError> {
    let ratio = ln_g0 / options.min_ln_g0;
    let k = ratio.log2().ceil().max(1.0);
    if !k.is_finite() || k > f64::from(options.max_scaling_squaring_steps) {
        return Err(ValidationError::InvalidParameter(
            "Panjer scaling-and-squaring would exceed max_scaling_squaring_steps",
        ));
    }
    Ok(k as u32)
}

/// Exact truncated self-convolution `h_x = sum_{y <= x} g_y g_{x - y}` for `x < g.len()`.
fn self_convolve_truncated(g: &[f64]) -> Vec<f64> {
    let n = g.len();
    let first = g.iter().position(|&v| v > 0.0).unwrap_or(n);
    let mut out = vec![0.0; n];
    for x in (2 * first)..n {
        let mut acc = 0.0;
        // Symmetric pairs (y, x - y) and (x - y, y) counted twice.
        let half = x / 2;
        for y in first..=half {
            let term = g[y] * g[x - y];
            acc += if 2 * y == x { term } else { 2.0 * term };
        }
        out[x] = acc;
    }
    out
}

/// Direct Panjer `(a, b, 0)` recursion; returns `(g, clipped_negative_mass)`.
fn panjer_core(
    frequency: FrequencyDistribution,
    severity_pmf: &[f64],
    n_steps: usize,
    options: &PanjerOptions,
) -> Result<(Vec<f64>, f64), ValidationError> {
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
    let mut clipped = 0.0_f64;

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
        if !gx.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if gx < 0.0 {
            clipped += -gx;
            if clipped > options.max_clipped_negative_mass {
                return Err(ValidationError::InvalidParameter(
                    "Panjer recursion clipped negative mass exceeds max_clipped_negative_mass (numerically unstable, e.g. high-q Binomial); use fft_aggregate_loss",
                ));
            }
        }
        g[x] = gx.max(0.0);
    }
    Ok((g, clipped))
}

/// Evaluate a compound aggregate loss distribution using Robertson's (1992)
/// radix-2 FFT characteristic-function inversion with exponential tilting.
///
/// * `fft_size` — power of two (`>= severity_pmf.len()` and `>= 2`).
/// * `tilt` — exponential tilting parameter `theta >= 0.0` (`None` uses the
///   automatic `f64`-balanced default `14.0 / fft_size as f64`, suppressing
///   wrap-around aliasing by `exp(-14) ≈ 8.3e-7` while limiting right-edge
///   amplification to ~`1e-10`).
///
/// This legacy entry point never rejects on aliasing; it **flags** it in
/// [`AggregateLossDistribution::aliasing_error_estimate`], and the risk
/// measures [`AggregateLossDistribution::var`] / [`AggregateLossDistribution::tvar`]
/// refuse to answer when that bound exceeds their tail budget. Use
/// [`fft_aggregate_loss_with_options`] to reject aliased grids up front.
pub fn fft_aggregate_loss(
    frequency: FrequencyDistribution,
    severity_pmf: &[f64],
    step: f64,
    fft_size: usize,
    tilt: Option<f64>,
) -> Result<AggregateLossDistribution, ValidationError> {
    fft_aggregate_loss_core(frequency, severity_pmf, step, fft_size, tilt)
}

/// Tilted radix-2 FFT aggregation that errors when the wrap-around aliasing
/// bound [`AggregateLossDistribution::aliasing_error_estimate`] exceeds
/// `options.max_aliasing_error`.
pub fn fft_aggregate_loss_with_options(
    frequency: FrequencyDistribution,
    severity_pmf: &[f64],
    step: f64,
    fft_size: usize,
    options: &FftOptions,
) -> Result<AggregateLossDistribution, ValidationError> {
    if !options.max_aliasing_error.is_finite() || options.max_aliasing_error < 0.0 {
        return Err(ValidationError::InvalidParameter(
            "max_aliasing_error must be finite and >= 0",
        ));
    }
    let out = fft_aggregate_loss_core(frequency, severity_pmf, step, fft_size, options.tilt)?;
    if out.aliasing_error_estimate > options.max_aliasing_error {
        return Err(ValidationError::InvalidParameter(
            "FFT grid is undersized: wrap-around aliasing bound exceeds max_aliasing_error; enlarge fft_size or increase the tilt",
        ));
    }
    Ok(out)
}

fn fft_aggregate_loss_core(
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

    let mut out = summarize_aggregate(frequency, severity_pmf, step, g);
    out.aliasing_error_estimate = fft_aliasing_bound(&out, fft_size, theta);
    Ok(out)
}

/// Upper bound on probability mass wrapped around an `n`-point circular grid
/// with tilt `theta` (see [`AggregateLossDistribution::aliasing_error_estimate`]).
fn fft_aliasing_bound(dist: &AggregateLossDistribution, n: usize, theta: f64) -> f64 {
    let missing = dist.missing_mass();
    let span = n as f64 * dist.step;
    let gap = (dist.exact_theoretical_mean - dist.mean).max(0.0);
    let mean_bound = (gap / span - missing).max(0.0);
    let theta_n = theta * n as f64;
    let tilt_bound = if theta_n > 0.0 {
        missing * (-theta_n).exp() / -((-theta_n).exp_m1())
    } else {
        f64::INFINITY
    };
    mean_bound.min(tilt_bound)
}

/// Discretize a continuous [`SeverityDistribution`] onto the lattice
/// `0, step, 2*step, ..., max_index * step` using the **Method of Rounding**
/// (Klugman, Panjer & Willmot, §9.6.1):
///
/// `f_0 = F(step / 2)`, `f_k = S((k - 0.5) * step) - S((k + 0.5) * step)`.
///
/// Differences are taken on the survival function so tail cells keep full
/// relative precision. If `capture_upper_tail` is `true`, the remaining
/// survival mass `S((max_index - 0.5) * step)` is placed on `f_{max_index}`
/// so `sum(f) == 1`.
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
    let mut prev_surv = 1.0_f64;
    for k in 0..=max_index {
        let next_surv = if k == max_index && capture_upper_tail {
            0.0
        } else {
            dist.survival((k as f64 + 0.5) * step)
        };
        pmf[k] = (prev_surv - next_surv).max(0.0);
        prev_surv = next_surv;
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
/// The second differences are evaluated as differences of direct unit-layer
/// integrals `L_j = int_{j h}^{(j+1) h} S(x) dx`
/// (`f_k = (L_{k-1} - L_k) / h`, `f_max = L_{max-1} / h`), avoiding
/// cancellation between LEVs that are all close to `E[X]` in the tail.
///
/// Preserves `E[X ∧ (max_index * step)] = sum_j L_j` identically.
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
    let mut layers = vec![0.0; max_index];
    for (j, slot) in layers.iter_mut().enumerate() {
        *slot = dist.layer_expected_loss(j as f64 * step, step)?;
    }

    let mut pmf = vec![0.0; max_index + 1];
    pmf[0] = ((step - layers[0]) / step).max(0.0);
    for k in 1..max_index {
        pmf[k] = ((layers[k - 1] - layers[k]) / step).max(0.0);
    }
    pmf[max_index] = (layers[max_index - 1] / step).max(0.0);
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
    let relative_mean_error = if exact_theoretical_mean > 0.0 {
        (exact_theoretical_mean - mean) / exact_theoretical_mean
    } else {
        0.0
    };

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
        relative_mean_error,
        aliasing_error_estimate: 0.0,
        clipped_negative_mass: 0.0,
        scaling_squaring_steps: 0,
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
