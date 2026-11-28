//! One-year re-reserving simulation and lifetime reserve-risk comparison.
//!
//! Implements the one-year re-reserving and lifetime reserve-risk simulation
//! framework of:
//!
//! * P. D. England, R. J. Verrall & M. V. Wüthrich (2019), *"On the Lifetime
//!   and One-Year Views of Reserve Risk, with Application to IFRS 17 and
//!   Solvency II Risk Margins"*, *Insurance: Mathematics and Economics*,
//!   Vol. 85, pp. 74–88, §§3–4.
//!
//! ## Accounting Identity
//!
//! For an observed `n × n` cumulative claims triangle at valuation time `t`,
//! let `R_t` be the opening reserve across the observed origin-period cohort
//! `i = 0, ..., n - 1`. Over the next accounting year `t -> t + 1`:
//!
//! 1. Simulate the next calendar diagonal `C_{i, n - i}^*` (`i = 1, ..., n - 1`),
//!    giving next-year payments `P_{i, t+1} = C_{i, n - i}^* - C_{i, n - 1 - i}`
//!    (with `P_{0, t+1} = 0`).
//! 2. Keep the observed triangle cells fixed, append the simulated diagonal,
//!    and re-estimate the closing reserve `R_{i, t+1}` on the **same
//!    origin-period cohort**.
//! 3. Record one-year adverse development (positive = adverse, i.e., `-CDR`):
//!
//!    `L_{i, t+1} = P_{i, t+1} + R_{i, t+1} - R_{i, t}`,
//!    `L_{t+1}    = P_{t+1}    + R_{t+1}    - R_t`.
//!
//! Note the sign convention: positive `L_{t+1}` represents adverse development
//! (a reserve strengthening / loss to surplus), so right-tail risk measures
//! (`VaR_p(L_{t+1})`, `TVaR_p(L_{t+1})`) directly measure one-year adverse
//! reserve-risk capital.
//!
//! ## Reusable Roll-Forward Architecture
//!
//! [`ChainLadderRollForward`] encapsulates the fixed-history closing-valuation
//! step (`O(n)` per scenario) and is shared across:
//!
//! * [`mack_one_year_runoff`] / [`mack_one_year_runoff_with_options`] — Mack
//!   bootstrap re-reserving (England, Verrall & Wüthrich 2019, §3.2).
//! * [`odp_one_year_runoff`] — Over-Dispersed Poisson (ODP) bootstrap
//!   re-reserving reusing the existing [`crate::stochastic::bootstrap_odp`] engine.
//!
//! The same closing-valuation interface accepts a scenario-level closing tail
//! factor, preparing it for stochastic tail re-estimation (Boumezoued et al.
//! 2011) and multi-line correlated run-off.

use crate::capital::reserve_distribution::{
    ReserveDistributionSummary, summarize_with_scratch, tvar_coherent, var_least,
};
use crate::error::ReservingError;
use crate::preprocessing::triangle::latest_diagonal;
use crate::stochastic::bootstrap_odp::{
    DeterministicRng, FastIndexSampler, OdpBootstrapOptions, OdpBootstrapScratch,
    build_adjusted_residual_pool, fit_odp, refit_pseudo_ldfs_from_pool,
    sample_process_increment_fast,
};
use crate::stochastic::mack;
use crate::types::TriangleView;
use crate::validate::{MonotonicityPolicy, validate_tail};

// ===========================================================================
// Reusable Closing-Valuation Roll-Forward Engine
// ===========================================================================

/// Per-scenario closing-valuation output on the observed origin-period cohort.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RollForwardStep {
    /// Next-year payments `P_{i, t+1}` by origin (length `n_origins`).
    pub origin_payments: Vec<f64>,
    /// Closing reserves `R_{i, t+1}` by origin (length `n_origins`).
    pub origin_closing_reserves: Vec<f64>,
    /// One-year adverse development `L_{i, t+1} = P_{i, t+1} + R_{i, t+1} - R_{i, t}` by origin.
    pub origin_adverse_development: Vec<f64>,
    /// Aggregate next-year payments `P_{t+1}`.
    pub total_payments: f64,
    /// Aggregate closing reserve `R_{t+1}`.
    pub total_closing_reserve: f64,
    /// Aggregate one-year adverse development `L_{t+1} = P_{t+1} + R_{t+1} - R_t`.
    pub total_adverse_development: f64,
}

/// Precomputed fixed-history state for `O(n)` chain-ladder closing valuations.
///
/// The observed cells of the opening triangle remain immutable; each call to
/// [`Self::evaluate`] appends a simulated next calendar diagonal `C_{i, n - i}^*` (`i = 1..n`) to the fixed observed
/// column sums, refits volume-weighted chain-ladder LDFs, and computes
/// `P_{t+1}`, `R_{t+1}`, and `L_{t+1}` on the same origin-period cohort.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainLadderRollForward {
    n_origins: usize,
    opening_tail: f64,
    obs_diag: Vec<f64>,
    obs_den: Vec<f64>,
    obs_num: Vec<f64>,
    opening_origin_reserves: Vec<f64>,
    opening_total_reserve: f64,
}

impl ChainLadderRollForward {
    /// Precompute observed column sums and opening reserves from `triangle`.
    pub fn from_triangle(
        triangle: &TriangleView<'_>,
        opening_tail: f64,
        opening_origin_reserves: Vec<f64>,
    ) -> Result<Self, ReservingError> {
        validate_tail(opening_tail)?;
        let n = triangle.n_origins;
        if n < 2
            || n != triangle.n_devs
            || triangle.data.len() != n * triangle.n_devs
            || opening_origin_reserves.len() != n
        {
            return Err(ReservingError::ShapeMismatch);
        }
        if opening_origin_reserves.iter().any(|v| !v.is_finite()) {
            return Err(ReservingError::NotFinite);
        }
        let m = n - 1;
        let mut obs_den = vec![0.0; m];
        let mut obs_num = vec![0.0; m];
        for i in 0..m {
            let row = &triangle.data[i * n..i * n + (n - i)];
            for k in 0..(m - i) {
                let c0 = row[k];
                let c1 = row[k + 1];
                if c0.is_finite() && c1.is_finite() {
                    obs_den[k] += c0;
                    obs_num[k] += c1;
                }
            }
        }
        let obs_diag = latest_diagonal(triangle);
        let opening_total_reserve = opening_origin_reserves.iter().sum();

        Ok(Self {
            n_origins: n,
            opening_tail,
            obs_diag,
            obs_den,
            obs_num,
            opening_origin_reserves,
            opening_total_reserve,
        })
    }

    /// Number of origin periods in the cohort.
    #[inline]
    pub const fn n_origins(&self) -> usize {
        self.n_origins
    }

    /// Opening tail factor `f_tail >= 1.0`.
    #[inline]
    pub const fn opening_tail(&self) -> f64 {
        self.opening_tail
    }

    /// Observed latest diagonal `C_{i, n - 1 - i}` by origin.
    #[inline]
    pub fn obs_diag(&self) -> &[f64] {
        &self.obs_diag
    }

    /// Opening reserves `R_{i, t}` by origin.
    #[inline]
    pub fn opening_origin_reserves(&self) -> &[f64] {
        &self.opening_origin_reserves
    }

    /// Aggregate opening reserve `R_t`.
    #[inline]
    pub const fn opening_total_reserve(&self) -> f64 {
        self.opening_total_reserve
    }

    /// Evaluate a single simulated next calendar diagonal, returning
    /// [`RollForwardStep`] if all refitted LDFs and reserves are finite and valid.
    pub fn evaluate(
        &self,
        next_cum_diag: &[f64],
        closing_tail: f64,
    ) -> Result<RollForwardStep, ReservingError> {
        validate_tail(closing_tail)?;
        if next_cum_diag.len() != self.n_origins {
            return Err(ReservingError::ShapeMismatch);
        }
        let n = self.n_origins;
        let m = n - 1;
        let mut ext_ldfs = vec![1.0; m];
        let mut ext_cdfs = vec![closing_tail; n];
        let mut origin_payments = vec![0.0; n];
        let mut origin_closing_reserves = vec![0.0; n];
        let mut origin_adverse_development = vec![0.0; n];

        let Some((total_payments, total_closing_reserve, total_adverse_development)) = self
            .evaluate_into(
                next_cum_diag,
                closing_tail,
                &mut ext_ldfs,
                &mut ext_cdfs,
                &mut origin_payments,
                &mut origin_closing_reserves,
                &mut origin_adverse_development,
            )
        else {
            return Err(ReservingError::NotFinite);
        };

        Ok(RollForwardStep {
            origin_payments,
            origin_closing_reserves,
            origin_adverse_development,
            total_payments,
            total_closing_reserve,
            total_adverse_development,
        })
    }

    /// Zero-allocation hot-path closing valuation.
    ///
    /// `next_cum_diag[i]` holds the simulated cumulative value `C_{i, n - i}^*`
    /// for `i = 1..n` (entry `0` is unused because origin `0` is already at
    /// development age `n - 1`). Returns `Some((P_{t+1}, R_{t+1}, L_{t+1}))`
    /// when all closing LDFs and reserves are finite and well-defined.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_into(
        &self,
        next_cum_diag: &[f64],
        closing_tail: f64,
        ext_ldfs: &mut [f64],
        ext_cdfs: &mut [f64],
        origin_payments: &mut [f64],
        origin_closing_reserves: &mut [f64],
        origin_adverse_development: &mut [f64],
    ) -> Option<(f64, f64, f64)> {
        let n = self.n_origins;
        let m = n - 1;

        // 1. Update volume-weighted LDFs by appending the single new pair
        //    (i = n - 1 - k, j = k -> k + 1) to each link k = 0..m.
        for k in 0..m {
            let i = m - k;
            let c_prev = self.obs_diag[i];
            let c_next = next_cum_diag[i];
            let den = self.obs_den[k] + c_prev;
            let num = self.obs_num[k] + c_next;
            if !den.is_finite() || den <= 0.0 || !num.is_finite() || num <= 0.0 {
                return None;
            }
            ext_ldfs[k] = num / den;
        }

        // 2. Compound closing CDFs right-to-left with the scenario closing tail.
        ext_cdfs[m] = closing_tail;
        for k in (0..m).rev() {
            ext_cdfs[k] = ext_ldfs[k] * ext_cdfs[k + 1];
        }

        // 3. Evaluate payments, closing reserves, and adverse development on
        //    the exact same origin-period cohort i = 0..n.
        let r1_origin0 = self.obs_diag[0] * (closing_tail - 1.0);
        let l_origin0 = r1_origin0 - self.opening_origin_reserves[0];
        if !r1_origin0.is_finite() || !l_origin0.is_finite() {
            return None;
        }
        origin_payments[0] = 0.0;
        origin_closing_reserves[0] = r1_origin0;
        origin_adverse_development[0] = l_origin0;

        let mut total_payments = 0.0;
        let mut total_closing_reserve = r1_origin0;

        for i in 1..n {
            let c_prev = self.obs_diag[i];
            let c_next = next_cum_diag[i];
            let j_next = n - i;
            let p_i = c_next - c_prev;
            let r1_i = c_next * (ext_cdfs[j_next] - 1.0);
            let l_i = p_i + r1_i - self.opening_origin_reserves[i];
            if !p_i.is_finite() || !r1_i.is_finite() || !l_i.is_finite() {
                return None;
            }
            origin_payments[i] = p_i;
            origin_closing_reserves[i] = r1_i;
            origin_adverse_development[i] = l_i;
            total_payments += p_i;
            total_closing_reserve += r1_i;
        }

        let total_adverse_development =
            total_payments + total_closing_reserve - self.opening_total_reserve;
        if !total_payments.is_finite()
            || !total_closing_reserve.is_finite()
            || !total_adverse_development.is_finite()
        {
            return None;
        }

        Some((
            total_payments,
            total_closing_reserve,
            total_adverse_development,
        ))
    }
}

// ===========================================================================
// Configuration & Result Types
// ===========================================================================

/// Innovation distribution for the Mack one-year re-reserving simulation
/// (England, Verrall & Wüthrich 2019, §3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum MackInnovation {
    /// Non-parametric Mack bootstrap: resample leverage-adjusted standardized
    /// Mack link-ratio residuals `r_{i,j} / sqrt(1 - h_{i,j})`.
    #[default]
    BootstrapResiduals,
    /// Parametric Mack simulation: draw standard normal `N(0, 1)` innovations
    /// matching Mack's (1993) and Merz & Wüthrich's (2008) exact first two moments.
    Normal,
}

/// Configuration for [`mack_one_year_runoff_with_options`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MackRunoffOptions {
    /// Number of Monte Carlo scenarios.
    pub scenarios: usize,
    /// Deterministic PRNG seed.
    pub seed: u64,
    /// Innovation source for parameter and process uncertainty.
    pub innovation: MackInnovation,
    /// Whether to center the adjusted Mack residual pool to zero mean before
    /// resampling (ignored when `innovation == MackInnovation::Normal`).
    pub center_residuals: bool,
    /// Monotonicity policy for triangle validation.
    pub monotonicity_policy: MonotonicityPolicy,
}

impl Default for MackRunoffOptions {
    fn default() -> Self {
        Self::new(10_000, 42_069)
    }
}

impl MackRunoffOptions {
    /// Create Mack one-year re-reserving options with non-parametric leverage-adjusted
    /// bootstrap residuals, centered residual pool, and Mack's default
    /// [`MonotonicityPolicy::AllowNegativeIncrements`].
    pub const fn new(scenarios: usize, seed: u64) -> Self {
        Self {
            scenarios,
            seed,
            innovation: MackInnovation::BootstrapResiduals,
            center_residuals: true,
            monotonicity_policy: MonotonicityPolicy::AllowNegativeIncrements,
        }
    }

    /// Set the number of Monte Carlo scenarios.
    pub const fn with_scenarios(mut self, scenarios: usize) -> Self {
        self.scenarios = scenarios;
        self
    }

    /// Set the deterministic PRNG seed.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the Mack innovation distribution (`BootstrapResiduals` or `Normal`).
    pub const fn with_innovation(mut self, innovation: MackInnovation) -> Self {
        self.innovation = innovation;
        self
    }

    /// Enable or disable zero-mean centering of the adjusted Mack residual pool.
    pub const fn with_centered_residuals(mut self, center: bool) -> Self {
        self.center_residuals = center;
        self
    }

    /// Set the triangle monotonicity policy.
    pub const fn with_policy(mut self, policy: MonotonicityPolicy) -> Self {
        self.monotonicity_policy = policy;
        self
    }
}

/// Output of a one-year re-reserving and lifetime reserve-risk simulation.
///
/// Every scenario `s` satisfies the exact accounting identity on the observed
/// origin-period cohort `i = 0..n`:
///
/// * Per origin: `origin_adverse_development[i][s] == next_year_origin_payments[i][s] + closing_origin_reserves[i][s] - opening_origin_reserves[i]`
/// * Aggregate:  `total_adverse_development[s]     == next_year_total_payments[s]     + closing_total_reserves[s]     - opening_total_reserve`
///
/// Positive values of `adverse_development` denote adverse development (`-CDR`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct OneYearRunoffResult {
    /// Opening total reserve `R_t` across all origins.
    pub opening_total_reserve: f64,
    /// Opening reserve `R_{i, t}` per origin (length `n_origins`).
    pub opening_origin_reserves: Vec<f64>,
    /// Simulated aggregate next-year payments `P_{t+1}` by scenario.
    pub next_year_total_payments: Vec<f64>,
    /// Simulated next-year payments `P_{i, t+1}` by `[origin][scenario]`.
    pub next_year_origin_payments: Vec<Vec<f64>>,
    /// Simulated aggregate closing reserves `R_{t+1}` by scenario.
    pub closing_total_reserves: Vec<f64>,
    /// Simulated closing reserves `R_{i, t+1}` by `[origin][scenario]`.
    pub closing_origin_reserves: Vec<Vec<f64>>,
    /// Simulated aggregate one-year adverse development `L_{t+1} = P_{t+1} + R_{t+1} - R_t`
    /// by scenario (positive = adverse).
    pub total_adverse_development: Vec<f64>,
    /// Simulated one-year adverse development `L_{i, t+1}` by `[origin][scenario]`.
    pub origin_adverse_development: Vec<Vec<f64>>,
    /// Simulated aggregate lifetime reserves `R_life` on the same scenario paths.
    pub lifetime_total_reserves: Vec<f64>,
    /// Empirical summary of aggregate one-year adverse development `L_{t+1}`.
    pub adverse_development_summary: ReserveDistributionSummary,
    /// Empirical summary of one-year adverse development `L_{i, t+1}` per origin.
    pub origin_adverse_development_summaries: Vec<ReserveDistributionSummary>,
    /// Empirical summary of aggregate lifetime reserves `R_life`.
    pub lifetime_reserve_summary: ReserveDistributionSummary,
    /// Empirical summary of aggregate next-year payments `P_{t+1}`.
    pub payments_summary: ReserveDistributionSummary,
    /// Empirical summary of aggregate closing reserves `R_{t+1}`.
    pub closing_reserve_summary: ReserveDistributionSummary,
    /// Count of skipped scenarios due to degenerate / non-positive pseudo-fits.
    pub failed_iterations: usize,
}

impl OneYearRunoffResult {
    /// Empirical Value at Risk `VaR_p(L_{t+1})` (least quantile, CAS Monograph 15
    /// Algorithm 3.4.2) of aggregate one-year adverse development.
    pub fn var_adverse_development(&self, p: f64) -> Result<f64, ReservingError> {
        var_least(&self.total_adverse_development, p)
    }

    /// Coherent Tail Value at Risk `TVaR_p(L_{t+1})` (exact integral, CAS
    /// Monograph 15 Algorithm 3.4.3) of aggregate one-year adverse development.
    pub fn tvar_adverse_development(&self, p: f64) -> Result<f64, ReservingError> {
        tvar_coherent(&self.total_adverse_development, p)
    }

    /// Simulated one-year emergence ratio: `SD(L_{t+1}) / SD(R_life)`.
    pub fn emergence_ratio(&self) -> f64 {
        let life_sd = self.lifetime_reserve_summary.std_dev;
        if life_sd > 0.0 {
            self.adverse_development_summary.std_dev / life_sd
        } else {
            0.0
        }
    }
}

impl std::fmt::Display for OneYearRunoffResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cv_1yr = if self.opening_total_reserve > 0.0 {
            self.adverse_development_summary.std_dev / self.opening_total_reserve * 100.0
        } else {
            0.0
        };
        let cv_life = if self.opening_total_reserve > 0.0 {
            self.lifetime_reserve_summary.std_dev / self.opening_total_reserve * 100.0
        } else {
            0.0
        };
        write!(
            f,
            "OneYearRunoff(opening_R={:.2}, 1yr_sd={:.2} [{:.2}%], life_sd={:.2} [{:.2}%], emergence={:.1}%)",
            self.opening_total_reserve,
            self.adverse_development_summary.std_dev,
            cv_1yr,
            self.lifetime_reserve_summary.std_dev,
            cv_life,
            self.emergence_ratio() * 100.0,
        )
    }
}

// ===========================================================================
// Mack One-Year Re-Reserving Simulation (England, Verrall & Wüthrich 2019)
// ===========================================================================

/// Build the leverage-adjusted Mack link-ratio residual pool for non-saturated
/// observed links (`j = 0..n - 2`, where `n_j >= 2` and `1 - h_{i,j} > 1e-10`).
fn build_mack_residual_pool(
    triangle: &TriangleView<'_>,
    ldf: &[f64],
    sigma_sq: &[f64],
    obs_den: &[f64],
    center: bool,
) -> Vec<f64> {
    let n = triangle.n_origins;
    let m = n.saturating_sub(1);
    let mut pool = Vec::with_capacity(m * m / 2);

    for k in 0..m {
        let nk = m - k;
        if nk < 2 {
            continue;
        }
        let s_k = obs_den[k];
        let sk2 = sigma_sq[k];
        if !(s_k > 0.0) || !(sk2 > 0.0) || !sk2.is_finite() {
            continue;
        }
        let inv_sigma = 1.0 / sk2.sqrt();
        let fk = ldf[k];
        for i in 0..nk {
            let c0 = triangle.get(i, k);
            let c1 = triangle.get(i, k + 1);
            if !(c0 > 0.0) || !c1.is_finite() {
                continue;
            }
            let leverage = c0 / s_k;
            let one_minus_h = 1.0 - leverage;
            if one_minus_h <= 1e-10 {
                continue;
            }
            let r_raw = c0.sqrt() * (c1 / c0 - fk) * inv_sigma;
            let r_adj = r_raw / one_minus_h.sqrt();
            if r_adj.is_finite() {
                pool.push(r_adj);
            }
        }
    }

    if center && !pool.is_empty() {
        let mean = pool.iter().sum::<f64>() / pool.len() as f64;
        for r in &mut pool {
            *r -= mean;
        }
    }

    pool
}

#[inline]
fn draw_mack_innovation(
    innovation: MackInnovation,
    pool: &[f64],
    sampler: Option<&mut FastIndexSampler>,
    rng: &mut DeterministicRng,
) -> f64 {
    match innovation {
        MackInnovation::Normal => rng.next_standard_normal(),
        MackInnovation::BootstrapResiduals => {
            if pool.is_empty() {
                0.0
            } else if let Some(s) = sampler {
                pool[s.next(rng)]
            } else {
                pool[rng.next_index(pool.len())]
            }
        }
    }
}

/// Run the England, Verrall & Wüthrich (2019, §§3–4) Mack bootstrap one-year
/// re-reserving simulation with default options (tail `1.0`, leverage-adjusted
/// centered Mack bootstrap residuals).
///
/// Emits opening reserve `R_t`, next-year payments `P_{t+1}`, closing reserve
/// `R_{t+1}`, one-year adverse development `L_{t+1} = P_{t+1} + R_{t+1} - R_t`,
/// and lifetime reserves `R_life` on the same observed origin-period cohort.
pub fn mack_one_year_runoff(
    triangle: &TriangleView<'_>,
    scenarios: usize,
    seed: u64,
) -> Result<OneYearRunoffResult, ReservingError> {
    mack_one_year_runoff_with_options(triangle, 1.0, MackRunoffOptions::new(scenarios, seed))
}

/// Run the Mack one-year re-reserving simulation with explicit tail factor and
/// [`MackRunoffOptions`].
pub fn mack_one_year_runoff_with_options(
    triangle: &TriangleView<'_>,
    tail: f64,
    options: MackRunoffOptions,
) -> Result<OneYearRunoffResult, ReservingError> {
    if options.scenarios == 0 {
        return Err(ReservingError::InvalidIterations);
    }
    validate_tail(tail)?;

    let opening_mack = mack::analyze_with_policy(triangle, tail, options.monotonicity_policy)?;
    let n = triangle.n_origins;
    if n < 3 {
        return Err(ReservingError::InvalidParameter(
            "one-year re-reserving requires at least a 3x3 triangle",
        ));
    }
    let m = n - 1;

    let roll_forward =
        ChainLadderRollForward::from_triangle(triangle, tail, opening_mack.reserve.clone())?;

    let pool = match options.innovation {
        MackInnovation::BootstrapResiduals => build_mack_residual_pool(
            triangle,
            &opening_mack.ldf,
            &opening_mack.sigma_sq,
            &roll_forward.obs_den,
            options.center_residuals,
        ),
        MackInnovation::Normal => Vec::new(),
    };
    let mut sampler = if pool.is_empty() {
        None
    } else {
        Some(FastIndexSampler::new(pool.len()))
    };

    // Precompute per-link sqrt(C_{i,k}) weights for fast O(n^2) bootstrap LDF draws
    // and sqrt(sigma_k^2) scales.
    let mut sqrt_c_by_link: Vec<Vec<f64>> = Vec::with_capacity(m);
    for k in 0..m {
        let nk = m - k;
        let mut weights = Vec::with_capacity(nk);
        for i in 0..nk {
            let c0 = triangle.get(i, k);
            weights.push(if c0 > 0.0 && c0.is_finite() {
                c0.sqrt()
            } else {
                0.0
            });
        }
        sqrt_c_by_link.push(weights);
    }
    let sigma: Vec<f64> = opening_mack
        .sigma_sq
        .iter()
        .map(|&s2| if s2 > 0.0 { s2.sqrt() } else { 0.0 })
        .collect();
    let sqrt_obs_diag: Vec<f64> = roll_forward
        .obs_diag
        .iter()
        .map(|&c| if c > 0.0 { c.sqrt() } else { 0.0 })
        .collect();

    let mut rng = DeterministicRng::from_seed(options.seed);
    let mut next_year_total_payments = Vec::with_capacity(options.scenarios);
    let mut closing_total_reserves = Vec::with_capacity(options.scenarios);
    let mut total_adverse_development = Vec::with_capacity(options.scenarios);
    let mut lifetime_total_reserves = Vec::with_capacity(options.scenarios);
    let mut next_year_origin_payments: Vec<Vec<f64>> = (0..n)
        .map(|_| Vec::with_capacity(options.scenarios))
        .collect();
    let mut closing_origin_reserves: Vec<Vec<f64>> = (0..n)
        .map(|_| Vec::with_capacity(options.scenarios))
        .collect();
    let mut origin_adverse_development: Vec<Vec<f64>> = (0..n)
        .map(|_| Vec::with_capacity(options.scenarios))
        .collect();

    let mut pseudo_ldfs = vec![1.0; m];
    let mut next_cum_diag = vec![0.0; n];
    let mut ext_ldfs = vec![1.0; m];
    let mut ext_cdfs = vec![tail; n];
    let mut step_payments = vec![0.0; n];
    let mut step_closing = vec![0.0; n];
    let mut step_adverse = vec![0.0; n];
    let mut failed_iterations = 0usize;

    let r_life_origin0 = roll_forward.obs_diag[0] * (tail - 1.0);

    for _ in 0..options.scenarios {
        // 1. Parameter uncertainty: draw bootstrap pseudo-LDFs f_k^* under Mack's
        //    conditional residual bootstrap (England, Verrall & Wüthrich 2019, §3.2).
        let mut valid_params = true;
        for k in 0..m {
            let s_k = roll_forward.obs_den[k];
            let sk = sigma[k];
            if !(sk > 0.0) || !(s_k > 0.0) {
                pseudo_ldfs[k] = opening_mack.ldf[k];
                continue;
            }
            let mut weighted_z = 0.0;
            for &sqrt_c in &sqrt_c_by_link[k] {
                if sqrt_c > 0.0 {
                    let z =
                        draw_mack_innovation(options.innovation, &pool, sampler.as_mut(), &mut rng);
                    weighted_z += sqrt_c * z;
                }
            }
            let f_star = opening_mack.ldf[k] + (sk / s_k) * weighted_z;
            if !f_star.is_finite() || f_star <= 0.0 {
                valid_params = false;
                break;
            }
            pseudo_ldfs[k] = f_star;
        }
        if !valid_params {
            failed_iterations += 1;
            continue;
        }

        // 2. Process uncertainty on the next calendar diagonal (i + j == n),
        //    conditioning on the actual observed latest diagonal C_{i, n - 1 - i}.
        let mut valid_diag = true;
        for i in 1..n {
            let k = m - i;
            let c_prev = roll_forward.obs_diag[i];
            let eps = if sigma[k] > 0.0 {
                draw_mack_innovation(options.innovation, &pool, sampler.as_mut(), &mut rng)
            } else {
                0.0
            };
            let c_next = c_prev * pseudo_ldfs[k] + sqrt_obs_diag[i] * sigma[k] * eps;
            if !c_next.is_finite() {
                valid_diag = false;
                break;
            }
            next_cum_diag[i] = c_next;
        }
        if !valid_diag {
            failed_iterations += 1;
            continue;
        }

        // 3. Re-reserve on the fixed observed triangle + simulated diagonal.
        let Some((tot_p, tot_r1, tot_l)) = roll_forward.evaluate_into(
            &next_cum_diag,
            tail,
            &mut ext_ldfs,
            &mut ext_cdfs,
            &mut step_payments,
            &mut step_closing,
            &mut step_adverse,
        ) else {
            failed_iterations += 1;
            continue;
        };

        // 4. Continue the same scenario path to ultimate for lifetime reserve risk.
        let mut tot_life_reserve = r_life_origin0;
        let mut valid_life = true;
        for i in 1..n {
            let mut c_cur = next_cum_diag[i];
            for k in (n - i)..m {
                let eps = if sigma[k] > 0.0 {
                    draw_mack_innovation(options.innovation, &pool, sampler.as_mut(), &mut rng)
                } else {
                    0.0
                };
                c_cur = c_cur * pseudo_ldfs[k] + c_cur.max(0.0).sqrt() * sigma[k] * eps;
                if !c_cur.is_finite() {
                    valid_life = false;
                    break;
                }
            }
            if !valid_life {
                break;
            }
            let r_life_i = c_cur * tail - roll_forward.obs_diag[i];
            if !r_life_i.is_finite() {
                valid_life = false;
                break;
            }
            tot_life_reserve += r_life_i;
        }
        if !valid_life || !tot_life_reserve.is_finite() {
            failed_iterations += 1;
            continue;
        }

        next_year_total_payments.push(tot_p);
        closing_total_reserves.push(tot_r1);
        total_adverse_development.push(tot_l);
        lifetime_total_reserves.push(tot_life_reserve);
        for i in 0..n {
            next_year_origin_payments[i].push(step_payments[i]);
            closing_origin_reserves[i].push(step_closing[i]);
            origin_adverse_development[i].push(step_adverse[i]);
        }
    }

    finalize_runoff_result(
        roll_forward,
        next_year_total_payments,
        next_year_origin_payments,
        closing_total_reserves,
        closing_origin_reserves,
        total_adverse_development,
        origin_adverse_development,
        lifetime_total_reserves,
        failed_iterations,
    )
}

// ===========================================================================
// ODP One-Year Re-Reserving Simulation (Reusing ChainLadderRollForward)
// ===========================================================================

/// Run an Over-Dispersed Poisson (ODP) bootstrap one-year re-reserving
/// simulation reusing [`ChainLadderRollForward`] and the existing
/// [`crate::stochastic::bootstrap_odp`] engine.
///
/// Emits opening reserve `R_t`, next-year payments `P_{t+1}`, closing reserve
/// `R_{t+1}`, one-year adverse development `L_{t+1} = P_{t+1} + R_{t+1} - R_t`
/// (positive = adverse, i.e., `-CDR`), and lifetime reserves `R_life` on the
/// same observed origin-period cohort.
pub fn odp_one_year_runoff(
    triangle: &TriangleView<'_>,
    tail: f64,
    options: OdpBootstrapOptions,
) -> Result<OneYearRunoffResult, ReservingError> {
    if options.iterations == 0 {
        return Err(ReservingError::InvalidIterations);
    }
    validate_tail(tail)?;

    let fit = fit_odp(triangle, tail, options.monotonicity_policy)?;
    let (pool, _) =
        build_adjusted_residual_pool(&fit, options.residual_adjustment, options.center_residuals)?;

    let n = fit.n_origins;
    if n < 3 {
        return Err(ReservingError::InvalidParameter(
            "one-year re-reserving requires at least a 3x3 triangle",
        ));
    }
    let m = n - 1;

    let roll_forward =
        ChainLadderRollForward::from_triangle(triangle, tail, fit.origin_reserves.clone())?;

    let mut rng = DeterministicRng::from_seed(options.seed);
    let mut sampler = FastIndexSampler::new(pool.len());
    let mut scratch = OdpBootstrapScratch::for_fit(&fit);
    let inv_phi = scratch.inv_phi;

    let mut next_year_total_payments = Vec::with_capacity(options.iterations);
    let mut closing_total_reserves = Vec::with_capacity(options.iterations);
    let mut total_adverse_development = Vec::with_capacity(options.iterations);
    let mut lifetime_total_reserves = Vec::with_capacity(options.iterations);
    let mut next_year_origin_payments: Vec<Vec<f64>> = (0..n)
        .map(|_| Vec::with_capacity(options.iterations))
        .collect();
    let mut closing_origin_reserves: Vec<Vec<f64>> = (0..n)
        .map(|_| Vec::with_capacity(options.iterations))
        .collect();
    let mut origin_adverse_development: Vec<Vec<f64>> = (0..n)
        .map(|_| Vec::with_capacity(options.iterations))
        .collect();

    let mut pseudo_cdfs = vec![tail; n];
    let mut next_cum_diag = vec![0.0; n];
    let mut ext_ldfs = vec![1.0; m];
    let mut ext_cdfs = vec![tail; n];
    let mut step_payments = vec![0.0; n];
    let mut step_closing = vec![0.0; n];
    let mut step_adverse = vec![0.0; n];
    let mut failed_iterations = 0usize;

    let r_life_origin0_mu = roll_forward.obs_diag[0] * (tail - 1.0);

    for _ in 0..options.iterations {
        // 1. Resample ODP residuals and refit pseudo-LDFs.
        let outcome = refit_pseudo_ldfs_from_pool(&pool, &mut sampler, &mut rng, &mut scratch);
        if !outcome.valid {
            failed_iterations += 1;
            continue;
        }

        // Compound pseudo-CDFs for lifetime remaining-reserve simulation.
        pseudo_cdfs[m] = tail;
        for k in (0..m).rev() {
            pseudo_cdfs[k] = scratch.pseudo_ldfs[k] * pseudo_cdfs[k + 1];
        }

        // 2. Simulate next calendar diagonal incremental payments with ODP process variance.
        let mut valid_diag = true;
        for i in 1..n {
            let k = m - i;
            let c_prev = roll_forward.obs_diag[i];
            let expected_inc = c_prev * (scratch.pseudo_ldfs[k] - 1.0);
            let sim_inc = sample_process_increment_fast(
                expected_inc,
                fit.dispersion,
                inv_phi,
                options.process_distribution,
                &mut rng,
            );
            let c_next = c_prev + sim_inc;
            if !c_next.is_finite() {
                valid_diag = false;
                break;
            }
            next_cum_diag[i] = c_next;
        }
        if !valid_diag {
            failed_iterations += 1;
            continue;
        }

        // 3. Re-reserve on the fixed observed triangle + simulated diagonal.
        let Some((tot_p, tot_r1, tot_l)) = roll_forward.evaluate_into(
            &next_cum_diag,
            tail,
            &mut ext_ldfs,
            &mut ext_cdfs,
            &mut step_payments,
            &mut step_closing,
            &mut step_adverse,
        ) else {
            failed_iterations += 1;
            continue;
        };

        // 4. Simulate remaining future payments beyond the next diagonal on the
        //    same ODP scenario path (exact additive Gamma/ODP family property).
        let mut tot_life_reserve = sample_process_increment_fast(
            r_life_origin0_mu,
            fit.dispersion,
            inv_phi,
            options.process_distribution,
            &mut rng,
        );
        let mut valid_life = tot_life_reserve.is_finite();
        for i in 1..n {
            let k = m - i;
            let c_prev = roll_forward.obs_diag[i];
            let rem_mu = (c_prev * scratch.pseudo_ldfs[k]) * (pseudo_cdfs[k + 1] - 1.0);
            let rem_inc = sample_process_increment_fast(
                rem_mu,
                fit.dispersion,
                inv_phi,
                options.process_distribution,
                &mut rng,
            );
            let r_life_i = step_payments[i] + rem_inc;
            if !r_life_i.is_finite() {
                valid_life = false;
                break;
            }
            tot_life_reserve += r_life_i;
        }
        if !valid_life || !tot_life_reserve.is_finite() {
            failed_iterations += 1;
            continue;
        }

        next_year_total_payments.push(tot_p);
        closing_total_reserves.push(tot_r1);
        total_adverse_development.push(tot_l);
        lifetime_total_reserves.push(tot_life_reserve);
        for i in 0..n {
            next_year_origin_payments[i].push(step_payments[i]);
            closing_origin_reserves[i].push(step_closing[i]);
            origin_adverse_development[i].push(step_adverse[i]);
        }
    }

    finalize_runoff_result(
        roll_forward,
        next_year_total_payments,
        next_year_origin_payments,
        closing_total_reserves,
        closing_origin_reserves,
        total_adverse_development,
        origin_adverse_development,
        lifetime_total_reserves,
        failed_iterations,
    )
}

#[allow(clippy::too_many_arguments)]
fn finalize_runoff_result(
    roll_forward: ChainLadderRollForward,
    next_year_total_payments: Vec<f64>,
    next_year_origin_payments: Vec<Vec<f64>>,
    closing_total_reserves: Vec<f64>,
    closing_origin_reserves: Vec<Vec<f64>>,
    total_adverse_development: Vec<f64>,
    origin_adverse_development: Vec<Vec<f64>>,
    lifetime_total_reserves: Vec<f64>,
    failed_iterations: usize,
) -> Result<OneYearRunoffResult, ReservingError> {
    if total_adverse_development.is_empty() {
        return Err(ReservingError::NotFinite);
    }

    let n = roll_forward.n_origins;
    let mut scratch_buf = Vec::with_capacity(total_adverse_development.len());
    let adverse_development_summary =
        summarize_with_scratch(&total_adverse_development, &mut scratch_buf, true)?;
    let mut origin_adverse_development_summaries = Vec::with_capacity(n);
    for series in &origin_adverse_development {
        origin_adverse_development_summaries.push(summarize_with_scratch(
            series,
            &mut scratch_buf,
            true,
        )?);
    }
    let lifetime_reserve_summary =
        summarize_with_scratch(&lifetime_total_reserves, &mut scratch_buf, true)?;
    let payments_summary =
        summarize_with_scratch(&next_year_total_payments, &mut scratch_buf, true)?;
    let closing_reserve_summary =
        summarize_with_scratch(&closing_total_reserves, &mut scratch_buf, true)?;

    Ok(OneYearRunoffResult {
        opening_total_reserve: roll_forward.opening_total_reserve,
        opening_origin_reserves: roll_forward.opening_origin_reserves,
        next_year_total_payments,
        next_year_origin_payments,
        closing_total_reserves,
        closing_origin_reserves,
        total_adverse_development,
        origin_adverse_development,
        lifetime_total_reserves,
        adverse_development_summary,
        origin_adverse_development_summaries,
        lifetime_reserve_summary,
        payments_summary,
        closing_reserve_summary,
        failed_iterations,
    })
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::genins;
    use crate::stochastic::bootstrap_odp::bootstrap_odp_cdr;
    use crate::stochastic::merz_wuthrich;
    use crate::types::Triangle;

    fn small_integer_triangle() -> Triangle {
        #[rustfmt::skip]
        let data = vec![
            100.0, 150.0, 180.0, 198.0,
            120.0, 192.0, 224.0, f64::NAN,
            110.0, 154.0, f64::NAN, f64::NAN,
            130.0, f64::NAN, f64::NAN, f64::NAN,
        ];
        Triangle::new(4, 4, data)
    }

    fn noiseless_integer_triangle() -> Triangle {
        // Exact LDFs: f_0 = 2.0, f_1 = 1.5, f_2 = 1.1
        #[rustfmt::skip]
        let data = vec![
            100.0, 200.0, 300.0, 330.0,
            150.0, 300.0, 450.0, f64::NAN,
            200.0, 400.0, f64::NAN, f64::NAN,
            250.0, f64::NAN, f64::NAN, f64::NAN,
        ];
        Triangle::new(4, 4, data)
    }

    #[test]
    fn small_integer_fixture_satisfies_exact_accounting_identity_on_every_path() {
        let tri = small_integer_triangle();
        let res = mack_one_year_runoff(&tri.view(), 500, 12_345).expect("mack runoff");
        assert_eq!(res.failed_iterations, 0);
        assert_eq!(res.total_adverse_development.len(), 500);
        assert!(res.adverse_development_summary.std_dev > 0.0);

        let r_t = res.opening_total_reserve;
        for s in 0..res.total_adverse_development.len() {
            let p_tot = res.next_year_total_payments[s];
            let r1_tot = res.closing_total_reserves[s];
            let l_tot = res.total_adverse_development[s];
            let expected_l = p_tot + r1_tot - r_t;
            assert!(
                (l_tot - expected_l).abs() <= 1e-11 * (1.0 + r_t.abs()),
                "scenario {s}: L={l_tot}, P+R1-R0={expected_l}"
            );

            // Origin-level identity and cohort aggregation.
            let mut sum_p = 0.0;
            let mut sum_r1 = 0.0;
            let mut sum_l = 0.0;
            for i in 0..4 {
                let p_i = res.next_year_origin_payments[i][s];
                let r1_i = res.closing_origin_reserves[i][s];
                let l_i = res.origin_adverse_development[i][s];
                let r0_i = res.opening_origin_reserves[i];
                assert!(
                    (l_i - (p_i + r1_i - r0_i)).abs() <= 1e-11 * (1.0 + r0_i.abs()),
                    "origin {i}, scenario {s}: L_i={l_i}, P_i+R1_i-R0_i={}",
                    p_i + r1_i - r0_i
                );
                sum_p += p_i;
                sum_r1 += r1_i;
                sum_l += l_i;
            }
            assert!((sum_p - p_tot).abs() <= 1e-11);
            assert!((sum_r1 - r1_tot).abs() <= 1e-11);
            assert!((sum_l - l_tot).abs() <= 1e-11);
        }

        // Origin 0 is fully developed (tail = 1.0): zero payments, closing reserve, adverse dev.
        assert!(res.next_year_origin_payments[0].iter().all(|&x| x == 0.0));
        assert!(res.closing_origin_reserves[0].iter().all(|&x| x == 0.0));
        assert!(res.origin_adverse_development[0].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn noiseless_triangle_has_zero_adverse_development_and_exact_rollforward() {
        let tri = noiseless_integer_triangle();
        let res = mack_one_year_runoff(&tri.view(), 50, 99).expect("noiseless runoff");
        assert_eq!(res.failed_iterations, 0);
        // Opening reserves:
        // i=0: 0
        // i=1: 450 * (1.1 - 1) = 45
        // i=2: 400 * (1.5 * 1.1 - 1) = 260
        // i=3: 250 * (2.0 * 1.5 * 1.1 - 1) = 575
        // Total R_t = 880
        assert!((res.opening_total_reserve - 880.0).abs() < 1e-10);
        // Next-year payments: 45 + 200 + 250 = 495; closing reserve: 0 + 60 + 325 = 385.
        for s in 0..50 {
            assert!((res.next_year_total_payments[s] - 495.0).abs() < 1e-10);
            assert!((res.closing_total_reserves[s] - 385.0).abs() < 1e-10);
            assert!(res.total_adverse_development[s].abs() < 1e-10);
            assert!((res.lifetime_total_reserves[s] - 880.0).abs() < 1e-10);
        }
        assert!(res.adverse_development_summary.std_dev.abs() < 1e-10);
    }

    #[test]
    fn genins_mack_runoff_matches_england_verrall_wuthrich_2019_table_2() {
        let tri = genins();
        let mw = merz_wuthrich::analyze(&tri.view()).expect("merz wuthrich");
        assert!((mw.total_reserve - 18_680_855.61).abs() < 1.0);
        assert!((mw.total_cdr_standard_error - 1_778_968.0).abs() < 5.0);
        assert!((mw.total_ultimate_standard_error - 2_447_095.0).abs() < 5.0);

        // Parametric Normal Mack simulation: exact moment match within tight MC tolerance.
        let normal_res = mack_one_year_runoff_with_options(
            &tri.view(),
            1.0,
            MackRunoffOptions::new(25_000, 20_190_111).with_innovation(MackInnovation::Normal),
        )
        .expect("normal mack runoff");
        assert_eq!(normal_res.failed_iterations, 0);
        assert!((normal_res.opening_total_reserve - 18_680_855.61).abs() < 1.0);

        let rel_1yr_normal =
            (normal_res.adverse_development_summary.std_dev - 1_778_968.0).abs() / 1_778_968.0;
        let rel_life_normal =
            (normal_res.lifetime_reserve_summary.std_dev - 2_447_095.0).abs() / 2_447_095.0;
        assert!(
            rel_1yr_normal < 0.03,
            "Normal Mack 1yr SD={} vs target 1,778,968 (rel={:.4})",
            normal_res.adverse_development_summary.std_dev,
            rel_1yr_normal
        );
        assert!(
            rel_life_normal < 0.03,
            "Normal Mack life SD={} vs target 2,447,095 (rel={:.4})",
            normal_res.lifetime_reserve_summary.std_dev,
            rel_life_normal
        );
        // Mean adverse development should be near zero (unbiased re-reserving).
        assert!(
            normal_res.adverse_development_summary.mean.abs() < 60_000.0,
            "Normal Mack mean L={}",
            normal_res.adverse_development_summary.mean
        );

        // Non-parametric Mack bootstrap residuals (default).
        let boot_res =
            mack_one_year_runoff(&tri.view(), 25_000, 20_190_111).expect("bootstrap mack runoff");
        assert_eq!(boot_res.failed_iterations, 0);
        let rel_1yr_boot =
            (boot_res.adverse_development_summary.std_dev - 1_778_968.0).abs() / 1_778_968.0;
        let rel_life_boot =
            (boot_res.lifetime_reserve_summary.std_dev - 2_447_095.0).abs() / 2_447_095.0;
        assert!(
            rel_1yr_boot < 0.08,
            "Bootstrap Mack 1yr SD={} vs target 1,778,968 (rel={:.4})",
            boot_res.adverse_development_summary.std_dev,
            rel_1yr_boot
        );
        assert!(
            rel_life_boot < 0.08,
            "Bootstrap Mack life SD={} vs target 2,447,095 (rel={:.4})",
            boot_res.lifetime_reserve_summary.std_dev,
            rel_life_boot
        );

        // VaR and TVaR on one-year adverse development must satisfy TVaR >= VaR.
        let var995 = boot_res.var_adverse_development(0.995).expect("var995");
        let tvar995 = boot_res.tvar_adverse_development(0.995).expect("tvar995");
        assert!(tvar995 >= var995 && var995 > 0.0);
        assert!(boot_res.emergence_ratio() > 0.65 && boot_res.emergence_ratio() < 0.85);
    }

    #[test]
    fn odp_one_year_runoff_reuses_rollforward_and_matches_bootstrap_odp_cdr() {
        let tri = genins();
        let opts = OdpBootstrapOptions::shapland(2_000, 42_069);
        let odp_res = odp_one_year_runoff(&tri.view(), 1.0, opts).expect("odp runoff");
        let cdr_res = bootstrap_odp_cdr(&tri.view(), 1.0, opts).expect("odp cdr");

        assert_eq!(odp_res.failed_iterations, 0);
        assert!((odp_res.opening_total_reserve - cdr_res.base_total_reserve).abs() < 1e-8);

        // Accounting identity holds on every ODP scenario path.
        for s in 0..odp_res.total_adverse_development.len() {
            let lhs = odp_res.total_adverse_development[s];
            let rhs = odp_res.next_year_total_payments[s] + odp_res.closing_total_reserves[s]
                - odp_res.opening_total_reserve;
            assert!((lhs - rhs).abs() <= 1e-8 * (1.0 + odp_res.opening_total_reserve));
        }

        // ODP 1-year SD must agree with bootstrap_odp_cdr SD within MC tolerance
        // (they use the same underlying ODP bootstrap model, differing only in
        // extra RNG draws for the remaining lifetime reserve on each path).
        let rel_sd = (odp_res.adverse_development_summary.std_dev - cdr_res.summary.std_dev).abs()
            / cdr_res.summary.std_dev;
        assert!(
            rel_sd < 0.06,
            "ODP runoff 1yr SD={} vs bootstrap_odp_cdr SD={} (rel={:.4})",
            odp_res.adverse_development_summary.std_dev,
            cdr_res.summary.std_dev,
            rel_sd
        );
        assert!(
            odp_res.adverse_development_summary.std_dev < odp_res.lifetime_reserve_summary.std_dev
        );
    }

    #[test]
    fn deterministic_seed_and_validation_guards() {
        let tri = small_integer_triangle();
        let a = mack_one_year_runoff(&tri.view(), 100, 777).expect("run a");
        let b = mack_one_year_runoff(&tri.view(), 100, 777).expect("run b");
        assert_eq!(a, b);

        assert_eq!(
            mack_one_year_runoff(&tri.view(), 0, 777),
            Err(ReservingError::InvalidIterations)
        );
        assert_eq!(
            mack_one_year_runoff_with_options(&tri.view(), 0.9, MackRunoffOptions::new(10, 1)),
            Err(ReservingError::InvalidTail { tail: 0.9 })
        );
    }
}
