//! Over-Dispersed Poisson (ODP) GLM and residual bootstrap reserving kernel.
//!
//! Implements the classical ODP reserving and bootstrap framework from:
//!
//! * England & Verrall (1999), *Analytic and Bootstrap Estimates of Prediction
//!   Errors in Claims Reserving*, Insurance: Mathematics and Economics,
//!   Vol. 25, No. 3, pp. 281–293.
//! * England & Verrall (2002), *Stochastic Claims Reserving in General
//!   Insurance*, British Actuarial Journal, Vol. 8, No. 3, pp. 443–518.
//! * Shapland (2016), *Using the ODP Bootstrap Model: A Practitioner's Guide*,
//!   CAS Monograph No. 4.
//!
//! ## Model Specification
//!
//! For incremental payments `q_{i,j}` with `E[q_{i,j}] = μ_{i,j}` and
//! `Var(q_{i,j}) = φ · μ_{i,j}`, the log-linear ODP GLM
//! `log(μ_{i,j}) = c + α_i + β_j` produces fitted expectations identical to
//! volume-weighted chain ladder via backwards recursion from the observed
//! latest diagonal.
//!
//! Two mutually exclusive residual adjustments are supported:
//!
//! * [`ResidualAdjustment::DegreesOfFreedom`] — England & Verrall (1999, 2002)
//!   global degrees-of-freedom scaling `r_adj = sqrt(N / (N - p)) * r`.
//! * [`ResidualAdjustment::HatMatrixLeverage`] — Shapland (2016, CAS Monograph
//!   4, §3.4) leverage scaling `r_H = r / sqrt(1 - h_{ii})` computed from the
//!   exact GLM hat matrix `H = W^{1/2} X (X^T W X)^{-1} X^T W^{1/2}`, excluding
//!   structurally saturated corner cells (`1 - h_{ii} ≈ 0`).
//!
//! Process variance in future cells is sampled using either a moment-matched
//! Gamma distribution (`shape = μ* / φ`, `scale = φ`) or a scaled
//! Over-Dispersed Poisson (`φ · Poisson(μ* / φ)`), driven by a deterministic
//! SplitMix64-seeded Xoshiro256** PRNG with a 256-strip Ziggurat normal sampler.
//! Because both Gamma and scaled Poisson are exact additive families under
//! common dispersion `φ`, per-origin future reserves are sampled directly in
//! `O(n)` per iteration while preserving exact distributional equivalence
//! (including Shapland reflection for negative pseudo-increments).

use std::f64::consts::TAU;

use crate::capital::reserve_distribution::{ReserveDistributionSummary, summarize_with_scratch};
use crate::deterministic::ldf::{cdfs_from_ldfs, volume_weighted_ldfs};
use crate::linalg::{invert_spd, solve_lower_into};
use crate::preprocessing::triangle::{cumulative_to_incremental, latest_diagonal};
use crate::types::TriangleView;
use crate::validate::{
    MonotonicityPolicy, ValidationError, validate_monotone_cumulative, validate_tail,
    validate_triangle,
};

// Re-export the crate's Cholesky helper for `crate::capital::dependence`.
pub(crate) use crate::linalg::cholesky_lower;

// ===========================================================================
// Public Configuration & Result Types
// ===========================================================================

/// Compact bootstrap configuration for the legacy [`bootstrap`] entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BootstrapConfig {
    /// Number of bootstrap iterations.
    pub iterations: usize,
    /// Deterministic seed for reproducible resampling.
    pub seed: u64,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            iterations: 1000,
            seed: 0xC0DE5EED,
        }
    }
}

/// Residual degrees-of-freedom / leverage adjustment convention.
///
/// The two conventions are mutually exclusive (Shapland 2016, §3.4): do not
/// apply both simultaneously.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum ResidualAdjustment {
    /// England & Verrall (1999, 2002): multiply Pearson residuals by
    /// `sqrt(N / (N - p))` over all `N` active observed cells.
    #[default]
    DegreesOfFreedom,
    /// Shapland (2016, CAS Monograph 4, §3.4): divide non-saturated Pearson
    /// residuals by `sqrt(1 - h_{ii})` and exclude structurally saturated
    /// corner cells (`1 - h_{ii} <= 1e-10`) from the resampling pool.
    HatMatrixLeverage,
}

/// Future incremental process-variance distribution for ODP bootstrap draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum ProcessDistribution {
    /// Moment-matched Gamma: `Gamma(shape = μ* / φ, scale = φ)` with mean `μ*`
    /// and variance `φ · μ*` (England & Verrall 2002; Shapland 2016 §3.5).
    #[default]
    Gamma,
    /// Scaled Poisson: `φ · Poisson(μ* / φ)` with mean `μ*` and variance `φ · μ*`.
    OverDispersedPoisson,
    /// Omit process variance (parameter estimation uncertainty only).
    None,
}

/// Full configuration for [`bootstrap_odp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OdpBootstrapOptions {
    /// Number of bootstrap iterations.
    pub iterations: usize,
    /// Deterministic PRNG seed.
    pub seed: u64,
    /// Residual scaling convention (degrees of freedom vs. hat-matrix leverage).
    pub residual_adjustment: ResidualAdjustment,
    /// Process-variance distribution for future cells.
    pub process_distribution: ProcessDistribution,
    /// Whether to center the eligible adjusted residual pool to zero mean
    /// before resampling (Shapland 2016, §3.4.3).
    pub center_residuals: bool,
    /// Monotonicity policy for triangle validation.
    pub monotonicity_policy: MonotonicityPolicy,
}

impl Default for OdpBootstrapOptions {
    fn default() -> Self {
        Self::england_verrall(1000, 42069)
    }
}

impl OdpBootstrapOptions {
    /// Classic England & Verrall (1999, 2002) specification: global
    /// `sqrt(N / (N - p))` residual scaling, Gamma process variance, uncentered.
    pub const fn england_verrall(iterations: usize, seed: u64) -> Self {
        Self {
            iterations,
            seed,
            residual_adjustment: ResidualAdjustment::DegreesOfFreedom,
            process_distribution: ProcessDistribution::Gamma,
            center_residuals: false,
            monotonicity_policy: MonotonicityPolicy::Strict,
        }
    }

    /// Shapland (2016, CAS Monograph 4) practitioner specification: exact
    /// hat-matrix leverage `1 / sqrt(1 - h_{ii})`, centered eligible pool,
    /// Gamma process variance.
    pub const fn shapland(iterations: usize, seed: u64) -> Self {
        Self {
            iterations,
            seed,
            residual_adjustment: ResidualAdjustment::HatMatrixLeverage,
            process_distribution: ProcessDistribution::Gamma,
            center_residuals: true,
            monotonicity_policy: MonotonicityPolicy::Strict,
        }
    }

    /// Set the number of bootstrap iterations.
    pub const fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set the deterministic PRNG seed.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the residual adjustment convention (`DegreesOfFreedom` vs. `HatMatrixLeverage`).
    pub const fn with_residual_adjustment(mut self, adjustment: ResidualAdjustment) -> Self {
        self.residual_adjustment = adjustment;
        self
    }

    /// Set the future process-variance distribution (`Gamma`, `OverDispersedPoisson`, or `None`).
    pub const fn with_process_distribution(mut self, dist: ProcessDistribution) -> Self {
        self.process_distribution = dist;
        self
    }

    /// Enable or disable centering the eligible adjusted residual pool before resampling.
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

/// Bootstrap reserve summary shared with the generic reserve-distribution layer.
pub type BootstrapSummary = ReserveDistributionSummary;

/// Compact bootstrap output returned by [`bootstrap`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BootstrapResult {
    /// Original chain-ladder / ODP total reserve.
    pub base_total_reserve: f64,
    /// Simulated reserve totals by iteration.
    pub simulated_totals: Vec<f64>,
    /// Summary percentiles and moments.
    pub summary: BootstrapSummary,
}

/// Cell-level residual and hat-matrix leverage record from [`fit_odp`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OdpCellResidual {
    /// Origin row index.
    pub origin: usize,
    /// Development column index.
    pub dev: usize,
    /// Observed incremental payment `q_{i,j}`.
    pub observed_incremental: f64,
    /// ODP GLM / chain-ladder fitted incremental expectation `μ_{i,j}`.
    pub fitted_incremental: f64,
    /// Unscaled Pearson residual `(q_{i,j} - μ_{i,j}) / sqrt(μ_{i,j})`.
    pub raw_pearson_residual: f64,
    /// Diagonal entry `h_{ii}` of the GLM hat matrix `H`.
    pub leverage: f64,
    /// `true` when `1 - h_{ii} <= 1e-10` (structurally saturated corner cell).
    pub saturated: bool,
}

/// Immutable ODP GLM fit, dispersion, leverages, and analytic prediction errors.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct OdpFit {
    /// Number of origin rows.
    pub n_origins: usize,
    /// Number of development columns.
    pub n_devs: usize,
    /// Volume-weighted chain-ladder LDFs (length `n - 1`).
    pub ldfs: Vec<f64>,
    /// Multiplicative tail factor (`>= 1.0`).
    pub tail: f64,
    /// Full `n × n` fitted cumulative grid (row-major; observed region by
    /// backward recursion, future region by chain-ladder projection).
    pub fitted_cumulative: Vec<f64>,
    /// Full `n × n` fitted incremental grid (row-major).
    pub fitted_incremental: Vec<f64>,
    /// Observed `n × n` incremental grid (`NaN` in unobserved lower-right).
    pub observed_incremental: Vec<f64>,
    /// Residual and leverage diagnostics for every active observed cell.
    pub residuals: Vec<OdpCellResidual>,
    /// Count of active observed cells entering the ODP fit (`N`).
    pub n_observed: usize,
    /// Effective parameter rank of the ODP design matrix (`p = 2n - 1` for full triangles).
    pub rank: usize,
    /// Residual degrees of freedom (`N - p`).
    pub degrees_of_freedom: usize,
    /// Pearson dispersion parameter `φ = sum(r^2) / (N - p)`.
    pub dispersion: f64,
    /// Point-estimate reserves by origin (including tail when `tail > 1`).
    pub origin_reserves: Vec<f64>,
    /// Aggregate point-estimate reserve.
    pub total_reserve: f64,
    /// Analytic ODP process standard error by origin (pre-tail).
    pub analytic_origin_process_se: Vec<f64>,
    /// Analytic ODP parameter estimation standard error by origin (pre-tail).
    pub analytic_origin_param_se: Vec<f64>,
    /// Analytic ODP total prediction error by origin (England & Verrall 1999, Table 2).
    pub analytic_origin_prediction_error: Vec<f64>,
    /// Analytic ODP aggregate process standard error `sqrt(φ · R)`.
    pub analytic_total_process_se: f64,
    /// Analytic ODP aggregate parameter estimation standard error.
    pub analytic_total_param_se: f64,
    /// Analytic ODP aggregate prediction error (England & Verrall 1999, 2002).
    pub analytic_total_prediction_error: f64,
}

/// Full ODP bootstrap output with fit diagnostics, per-origin distributions,
/// and explicit failure accounting.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct OdpBootstrapResult {
    /// Underlying ODP GLM fit and analytic prediction errors.
    pub fit: OdpFit,
    /// Options used to generate this simulation.
    pub options: OdpBootstrapOptions,
    /// Base chain-ladder / ODP total reserve.
    pub base_total_reserve: f64,
    /// Simulated aggregate reserves for successful iterations.
    pub simulated_totals: Vec<f64>,
    /// Simulated reserves per origin: `simulated_by_origin[origin][iter]`.
    pub simulated_by_origin: Vec<Vec<f64>>,
    /// Distribution summary for each origin year.
    pub origin_summaries: Vec<ReserveDistributionSummary>,
    /// Distribution summary for the aggregate portfolio reserve.
    pub summary: ReserveDistributionSummary,
    /// Number of bootstrap draws rejected due to degenerate pseudo-triangles
    /// (non-positive column sums or non-finite refits). Never silently clipped.
    pub failed_iterations: usize,
    /// Arithmetic mean of the adjusted residual pool prior to optional centering.
    pub residual_pool_mean_before_centering: f64,
}

// ===========================================================================
// 256-Strip Ziggurat Tables for Standard Normal N(0, 1)
// (Marsaglia & Tsang 2000; Doornik 2005; exact equal-area strips to ~2e-15)
// ===========================================================================

#[allow(clippy::excessive_precision)]
const ZIG_NORM_R: f64 = 3.65415288536100924e+00;
#[allow(clippy::excessive_precision)]
const ZIG_NORM_INV_R: f64 = 2.73661237329758222e-01;

/// Pre-scaled strip half-widths `x[i] * 2^-51` for exact symmetric 52-bit shifts.
#[allow(clippy::excessive_precision)]
#[rustfmt::skip]
const ZIG_NORM_W: [f64; 256] = [
    1.73672541216026340e-15, 1.62276986753129703e-15, 1.53178727416111455e-15, 1.47448486035975978e-15,
    1.43199898696613284e-15, 1.39794366727752419e-15, 1.36936068351285195e-15, 1.34463009250111733e-15,
    1.32276557701953283e-15, 1.30312066346899871e-15, 1.28524793190961128e-15, 1.26882448142550120e-15,
    1.25360939266025688e-15, 1.23941797891632531e-15, 1.22610544174505652e-15, 1.21355608186668990e-15,
    1.20167593925438206e-15, 1.19038762992828923e-15, 1.17962663529558048e-15, 1.16933857869109619e-15,
    1.15947718914491932e-15, 1.15000275378397944e-15, 1.14088092425827821e-15, 1.13208178401648483e-15,
    1.12357911071108373e-15, 1.11534978658531569e-15, 1.10737332249357560e-15, 1.09963147017856302e-15,
    1.09210790381495648e-15, 1.08478795644034286e-15, 1.07765840026681103e-15, 1.07070726236329957e-15,
    1.06392366906768027e-15, 1.05729771390098922e-15, 1.05082034483551986e-15, 1.04448326760004725e-15,
    1.03827886235154011e-15, 1.03220011154864585e-15, 1.02624053726135694e-15, 1.02039414646831276e-15,
    1.01465538314670881e-15, 1.00901908616374609e-15, 1.00348045214361829e-15, 9.98035002618364886e-16,
    9.92678554880797846e-16, 9.87407196048067304e-16, 9.82217259919054503e-16, 9.77105306270721081e-16,
    9.72068102290162982e-16, 9.67102605882305622e-16, 9.62205950629484033e-16, 9.57375432209745160e-16,
    9.52608496106628067e-16, 9.47902726465174022e-16, 9.43255835967670851e-16, 9.38665656618666373e-16,
    9.34130131342526710e-16, 9.29647306308641870e-16, 9.25215323909569530e-16, 9.20832416326230957e-16,
    9.16496899621913729e-16, 9.12207168313484807e-16, 9.07961690374007196e-16, 9.03759002626012147e-16,
    8.99597706489110330e-16, 8.95476464049507169e-16, 8.91393994422409008e-16, 8.87349070381313749e-16,
    8.83340515230841091e-16, 8.79367199902104669e-16, 8.75428040251717888e-16, 8.71521994547370097e-16,
    8.67648061124558555e-16, 8.63805276200526185e-16, 8.59992711832766756e-16, 8.56209474010623629e-16,
    8.52454700869559916e-16, 8.48727561018615789e-16, 8.45027251972409973e-16, 8.41352998679803121e-16,
    8.37704052142022459e-16, 8.34079688113663175e-16, 8.30479205880537668e-16, 8.26901927108847298e-16,
    8.23347194760605338e-16, 8.19814372070653583e-16, 8.16302841580988043e-16, 8.12812004228450472e-16,
    8.09341278482150286e-16, 8.05890099527265881e-16, 8.02457918492126209e-16, 7.99044201715713301e-16,
    7.95648430052936913e-16, 7.92270098215227480e-16, 7.88908714144175817e-16, 7.85563798416108701e-16,
    7.82234883675641479e-16, 7.78921514096385636e-16, 7.75623244867117738e-16, 7.72339641701830249e-16,
    7.69070280372192306e-16, 7.65814746261049029e-16, 7.62572633935675970e-16, 7.59343546739589345e-16,
    7.56127096401792568e-16, 7.52922902662406231e-16, 7.49730592913698329e-16, 7.46549801855589297e-16,
    7.43380171164765187e-16, 7.40221349176580262e-16, 7.37072990578983395e-16, 7.33934756117740870e-16,
    7.30806312312274682e-16, 7.27687331181469666e-16, 7.24577489978839093e-16, 7.21476470936471065e-16,
    7.18383961017206680e-16, 7.15299651674530694e-16, 7.12223238619678278e-16, 7.09154421595487736e-16,
    7.06092904156548511e-16, 7.03038393455215375e-16, 6.99990600033076791e-16, 6.96949237617483333e-16,
    6.93914022922757200e-16, 6.90884675455717296e-16, 6.87860917325166761e-16, 6.84842473055004125e-16,
    6.81829069400625700e-16, 6.78820435168298322e-16, 6.75816301037190453e-16, 6.72816399383753509e-16,
    6.69820464108156184e-16, 6.66828230462474985e-16, 6.63839434880350960e-16, 6.60853814807826269e-16,
    6.57871108535074527e-16, 6.54891055028741935e-16, 6.51913393764616367e-16, 6.48937864560340148e-16,
    6.45964207407883707e-16, 6.42992162305490100e-16, 6.40021469088802966e-16, 6.37051867260881988e-16,
    6.34083095820806494e-16, 6.31114893090560917e-16, 6.28146996539890023e-16, 6.25179142608800574e-16,
    6.22211066527379383e-16, 6.19242502132585295e-16, 6.16273181681660223e-16, 6.13302835661791692e-16,
    6.10331192595644535e-16, 6.07357978842361158e-16, 6.04382918393613093e-16, 6.01405732664266896e-16,
    5.98426140277203304e-16, 5.95443856841806472e-16, 5.92458594725613986e-16, 5.89470062818587669e-16,
    5.86477966289437117e-16, 5.83482006333389797e-16, 5.80481879910769157e-16, 5.77477279475697074e-16,
    5.74467892694196679e-16, 5.71453402150920989e-16, 5.68433485043682002e-16, 5.65407812864896635e-16,
    5.62376051069004938e-16, 5.59337858724846799e-16, 5.56292888151909609e-16, 5.53240784539278853e-16,
    5.50181185546034217e-16, 5.47113720881736705e-16, 5.44038011865547399e-16, 5.40953670962399925e-16,
    5.37860301294524071e-16, 5.34757496126473645e-16, 5.31644838321662347e-16, 5.28521899768238889e-16,
    5.25388240771946115e-16, 5.22243409413404864e-16, 5.19086940867035028e-16, 5.15918356678574433e-16,
    5.12737163997880354e-16, 5.09542854763389919e-16, 5.06334904834272739e-16, 5.03112773065932658e-16,
    4.99875900324091668e-16, 4.96623708432218620e-16, 4.93355599046542278e-16, 4.90070952452300263e-16,
    4.86769126274221756e-16, 4.83449454093501688e-16, 4.80111243962702438e-16, 4.76753776809086152e-16,
    4.73376304715833357e-16, 4.69978049069420384e-16, 4.66558198560088404e-16, 4.63115907020817356e-16,
    4.59650291088495059e-16, 4.56160427669004794e-16, 4.52645351185714954e-16, 4.49104050588288485e-16,
    4.45535466095792352e-16, 4.41938485644707683e-16, 4.38311941008546798e-16, 4.34654603551288730e-16,
    4.30965179571630542e-16, 4.27242305188998743e-16, 4.23484540715208218e-16, 4.19690364447408414e-16,
    4.15858165808281775e-16, 4.11986237748075556e-16, 4.08072768309605617e-16, 4.04115831241434458e-16,
    4.00113375525468080e-16, 3.96063213662565057e-16, 3.91963008532578115e-16, 3.87810258612503849e-16,
    3.83602281296774122e-16, 3.79336194015496967e-16, 3.75008892787479409e-16, 3.70617027772362203e-16,
    3.66156975296536877e-16, 3.61624805715984870e-16, 3.57016246339536469e-16, 3.52326638460021892e-16,
    3.47550887317299159e-16, 3.42683403531195187e-16, 3.37718034173533949e-16, 3.32647981168418826e-16,
    3.27465704079399184e-16, 3.22162803505500827e-16, 3.16729880185819877e-16, 3.11156363389217596e-16,
    3.05430300071926326e-16, 2.99538093367823101e-16, 2.93464174847290854e-16, 2.87190589041141424e-16,
    2.80696460024850205e-16, 2.73957296851426679e-16, 2.66944074736485216e-16, 2.59621997725283540e-16,
    2.51948798292744913e-16, 2.43872345574290521e-16, 2.35327189140461874e-16, 2.26229403922184264e-16,
    2.16468605768957550e-16, 2.05895006284824550e-16, 1.94297201531359847e-16, 1.81361208101194749e-16,
    1.66587336315867911e-16, 1.49097409624961075e-16, 1.27087048348114564e-16, 9.55866035145697372e-17,
];

/// Inner acceptance thresholds `x[i + 1]` for strip `i` (`x[256] = 0.0`).
#[allow(clippy::excessive_precision)]
#[rustfmt::skip]
const ZIG_NORM_X_INNER: [f64; 256] = [
    3.65415288536100924e+00, 3.44927829856143164e+00, 3.32024473383982599e+00, 3.22457505204780182e+00,
    3.14788928951800129e+00, 3.08352613200214387e+00, 3.02783779176959378e+00, 2.97860327988184359e+00,
    2.93436686720888806e+00, 2.89412105361341299e+00, 2.85713873087322501e+00, 2.82287739682644334e+00,
    2.79092117400192796e+00, 2.76094400527998696e+00, 2.73268535904401189e+00, 2.70593365612306291e+00,
    2.68051464328574562e+00, 2.65628303757674411e+00, 2.63311639363158312e+00, 2.61091051848882438e+00,
    2.58957598670828704e+00, 2.56903545268184441e+00, 2.54922155032478370e+00, 2.53007523215985497e+00,
    2.51154444162669499e+00, 2.49358304127104757e+00, 2.47614993967052355e+00, 2.45920837433470574e+00,
    2.44272531820036498e+00, 2.42667098493714750e+00, 2.41101841390111993e+00, 2.39574311978192789e+00,
    2.38082279517208617e+00, 2.36623705671729168e+00, 2.35196722737914543e+00, 2.33799614879652928e+00,
    2.32430801887113292e+00, 2.31088825060137237e+00, 2.29772334890286434e+00, 2.28480080272449282e+00,
    2.27210899022838264e+00, 2.25963709517378808e+00, 2.24737503294739005e+00, 2.23531338492992182e+00,
    2.22344334009251110e+00, 2.21175664288416174e+00, 2.20024554661127691e+00, 2.18890277162636160e+00,
    2.17772146774029363e+00, 2.16669518035430908e+00, 2.15581781987673793e+00, 2.14508363404788938e+00,
    2.13448718284601746e+00, 2.12402331568952407e+00, 2.11368715068665391e+00, 2.10347405571487789e+00,
    2.09337963113879244e+00, 2.08339969399830505e+00, 2.07353026351874359e+00, 2.06376754781173277e+00,
    2.05410793165065275e+00, 2.04454796521753224e+00, 2.03508435372961971e+00, 2.02571394786385506e+00,
    2.01643373490620492e+00, 2.00724083056052960e+00, 1.99813247135842031e+00, 1.98910600761743894e+00,
    1.98015889690047753e+00, 1.97128869793366013e+00, 1.96249306494436371e+00, 1.95376974238464762e+00,
    1.94511656000867905e+00, 1.93653142827569535e+00, 1.92801233405266648e+00, 1.91955733659318883e+00,
    1.91116456377125399e+00, 1.90283220855042989e+00, 1.89455852567070537e+00, 1.88634182853678345e+00,
    1.87818048629299650e+00, 1.87007292107126744e+00, 1.86201760539967487e+00, 1.85401305976020270e+00,
    1.84605785028518632e+00, 1.83815058658280739e+00, 1.83028991968275778e+00, 1.82247454009388643e+00,
    1.81470317596628328e+00, 1.80697459135082172e+00, 1.79928758454972093e+00, 1.79164098655216342e+00,
    1.78403365954944237e+00, 1.77646449552452368e+00, 1.76893241491126929e+00, 1.76143636531891112e+00,
    1.75397532031767245e+00, 1.74654827828172321e+00, 1.73915426128591255e+00, 1.73179231405296408e+00,
    1.72446150294804568e+00, 1.71716091501782397e+00, 1.70988965707130269e+00, 1.70264685479992406e+00,
    1.69543165193456247e+00, 1.68824320943719641e+00, 1.68108070472517479e+00, 1.67394333092612602e+00,
    1.66683029616166634e+00, 1.65974082285818336e+00, 1.65267414708305682e+00, 1.64562951790478329e+00,
    1.63860619677554875e+00, 1.63160345693487452e+00, 1.62462058283303579e+00, 1.61765686957301646e+00,
    1.61071162236983101e+00, 1.60378415602609548e+00, 1.59687379442278909e+00, 1.58997987002419161e+00,
    1.58310172339603006e+00, 1.57623870273590727e+00, 1.56939016341512461e+00, 1.56255546753104557e+00,
    1.55573398346917724e+00, 1.54892508547417429e+00, 1.54212815322900276e+00, 1.53534257144151498e+00,
    1.52856772943771313e+00, 1.52180302076099894e+00, 1.51504784277671556e+00, 1.50830159628131244e+00,
    1.50156368511546479e+00, 1.49483351578049461e+00, 1.48811049705744858e+00, 1.48139403962818839e+00,
    1.47468355569785659e+00, 1.46797845861808085e+00, 1.46127816251027665e+00, 1.45458208188841143e+00,
    1.44788963128057735e+00, 1.44120022484872523e+00, 1.43451327600589340e+00, 1.42782819703025732e+00,
    1.42114439867531028e+00, 1.41446128977547247e+00, 1.40777827684640022e+00, 1.40109476367925234e+00,
    1.39441015092814236e+00, 1.38772383568997748e+00, 1.38103521107585681e+00, 1.37434366577316758e+00,
    1.36764858359747765e+00, 1.36094934303328441e+00, 1.35424531676263626e+00, 1.34753587118058848e+00,
    1.34082036589640530e+00, 1.33409815321936143e+00, 1.32736857762792715e+00, 1.32063097522105766e+00,
    1.31388467315022184e+00, 1.30712898903073249e+00, 1.30036323033083856e+00, 1.29358669373694912e+00,
    1.28679866449324498e+00, 1.27999841571381934e+00, 1.27318520766535781e+00, 1.26635828701823083e+00,
    1.25951688606371559e+00, 1.25266022189489856e+00, 1.24578749554862855e+00, 1.23889789110568871e+00,
    1.23199057474613749e+00, 1.22506469375653237e+00, 1.21811937548548310e+00, 1.21115372624370066e+00,
    1.20416683014438308e+00, 1.19715774787944307e+00, 1.19012551542669365e+00, 1.18306914268268848e+00,
    1.17598761201545376e+00, 1.16887987673083482e+00, 1.16174485944561323e+00, 1.15458145035992943e+00,
    1.14738850542085080e+00, 1.14016484436815313e+00, 1.13290924865253562e+00, 1.12562045921553522e+00,
    1.11829717411934682e+00, 1.11093804601357782e+00, 1.10354167942464176e+00, 1.09610662785202351e+00,
    1.08863139065398196e+00, 1.08111440970340600e+00, 1.07355406579243851e+00, 1.06594867476212474e+00,
    1.05829648333067716e+00, 1.05059566459093201e+00, 1.04284431314415116e+00, 1.03504043983344318e+00,
    1.02718196603564804e+00, 1.01926671746548658e+00, 1.01129241743999820e+00, 1.00325667954467534e+00,
    9.95156999635093409e-01, 9.86990747099065091e-01, 9.78755155294227297e-01, 9.70447311064227103e-01,
    9.62064143223043233e-01, 9.53602409881088686e-01, 9.45058684468168098e-01, 9.36429340286577871e-01,
    9.27710533402002824e-01, 9.18898183649593303e-01, 9.09987953496721236e-01, 9.00975224461224689e-01,
    8.91855070732944566e-01, 8.82622229585168672e-01, 8.73271068088864011e-01, 8.63795545553312061e-01,
    8.54189171008167158e-01, 8.44444954909157275e-01, 8.34555354086385592e-01, 8.24512208752295694e-01,
    8.14306670135218735e-01, 8.03929116989974824e-01, 7.93369058840627028e-01, 7.82615023307236868e-01,
    7.71654424224571978e-01, 7.60473406430111964e-01, 7.49056662017819397e-01, 7.37387211434299639e-01,
    7.25446140910003923e-01, 7.13212285190980122e-01, 7.00661841106819505e-01, 6.87767892795793090e-01,
    6.74499822837298479e-01, 6.60822574244424588e-01, 6.46695714894998885e-01, 6.32072236386066355e-01,
    6.16896990007756774e-01, 6.01104617755998327e-01, 5.84616766106385355e-01, 5.67338257053825123e-01,
    5.49151702327171809e-01, 5.29909720661565165e-01, 5.09423329602099240e-01, 4.87443966139244012e-01,
    4.63634336790890889e-01, 4.37518402207881185e-01, 4.08389134612001825e-01, 3.75121332878393110e-01,
    3.35737519214440450e-01, 2.86174591792092481e-01, 2.15241895984913806e-01, 0.00000000000000000e+00,
];

/// Interleaved `(W[i], X_INNER[i])` table packed into 16-byte pairs so each
/// Ziggurat normal draw hits a single cache line instead of two separate 2 KB arrays.
const ZIG_NORM_WX: [(f64, f64); 256] = {
    let mut out = [(0.0, 0.0); 256];
    let mut i = 0;
    while i < 256 {
        out[i] = (ZIG_NORM_W[i], ZIG_NORM_X_INNER[i]);
        i += 1;
    }
    out
};

/// Density values `f[i] = exp(-0.5 * x[i]^2)` for wedge checks (`f[256] = 1.0`).
#[allow(clippy::excessive_precision)]
#[rustfmt::skip]
const ZIG_NORM_F: [f64; 257] = [
    1.26028593049859559e-03, 1.26028593049859559e-03, 2.60907274610215883e-03, 4.03797259336302443e-03,
    5.52240329925098895e-03, 7.05087547137321635e-03, 8.61658276939871945e-03, 1.02149714397014556e-02,
    1.18427578579078704e-02, 1.34974506017398587e-02, 1.51770883079353023e-02, 1.68800831525431419e-02,
    1.86051212757246155e-02, 2.03510962300444860e-02, 2.21170627073088294e-02, 2.39022033057958438e-02,
    2.57058040085488548e-02, 2.75272356696030368e-02, 2.93659397581332651e-02, 3.12214171919201963e-02,
    3.30932194585784670e-02, 3.49809414617160280e-02, 3.68842156885672290e-02, 3.88027074045260573e-02,
    4.07361106559408701e-02, 4.26841449164743689e-02, 4.46465522512943802e-02, 4.66230949019302982e-02,
    4.86135532158684519e-02, 5.06177238609476915e-02, 5.26354182767921064e-02, 5.46664613248888376e-02,
    5.67106901062028185e-02, 5.87679529209336748e-02, 6.08381083495397740e-02, 6.29210244377580302e-02,
    6.50165779712427588e-02, 6.71246538277883997e-02, 6.92451443970066721e-02, 7.13779490588902776e-02,
    7.35229737139811712e-02, 7.56801303589269697e-02, 7.78493367020959420e-02, 8.00305158146629447e-02,
    8.22235958132027517e-02, 8.44285095703532634e-02, 8.66451944505578497e-02, 8.88735920682756780e-02,
    9.11136480663735232e-02, 9.33653119126907488e-02, 9.56285367130087077e-02, 9.79032790388621732e-02,
    1.00189498768809698e-01, 1.02487158941934969e-01, 1.04796225622486777e-01, 1.07116667774683511e-01,
    1.09448457146811506e-01, 1.11791568163837868e-01, 1.14145977827838210e-01, 1.16511665625610661e-01,
    1.18888613442909824e-01, 1.21276805484790057e-01, 1.23676228201596392e-01, 1.26086870220185693e-01,
    1.28508722279999349e-01, 1.30941777173644136e-01, 1.33386029691668934e-01, 1.35841476571253533e-01,
    1.38308116448550511e-01, 1.40785949814444478e-01, 1.43274978973513212e-01, 1.45775208005993834e-01,
    1.48286642732574331e-01, 1.50809290681845481e-01, 1.53343161060262634e-01, 1.55888264724479003e-01,
    1.58444614155924091e-01, 1.61012223437510871e-01, 1.63591108232365501e-01, 1.66181285764481851e-01,
    1.68782774801211288e-01, 1.71395595637505727e-01, 1.74019770081838526e-01, 1.76655321443734747e-01,
    1.79302274522847416e-01, 1.81960655599522320e-01, 1.84630492426799020e-01, 1.87311814223800027e-01,
    1.90004651670464736e-01, 1.92709036903588898e-01, 1.95425003514134027e-01, 1.98152586545774861e-01,
    2.00891822494656314e-01, 2.03642749310334603e-01, 2.06405406397880464e-01, 2.09179834621124744e-01,
    2.11966076307029905e-01, 2.14764175251173306e-01, 2.17574176724330853e-01, 2.20396127480151666e-01,
    2.23230075763917152e-01, 2.26076071322379890e-01, 2.28934165414679924e-01, 2.31804410824338281e-01,
    2.34686861872329566e-01, 2.37581574431237619e-01, 2.40488605940500061e-01, 2.43408015422749785e-01,
    2.46339863501263301e-01, 2.49284212418527912e-01, 2.52241126055941567e-01, 2.55210669954661351e-01,
    2.58192911337618569e-01, 2.61187919132720492e-01, 2.64195763997260413e-01, 2.67216518343560749e-01,
    2.70250256365874852e-01, 2.73297054068576517e-01, 2.76356989295667710e-01, 2.79430141761637385e-01,
    2.82516593083707079e-01, 2.85616426815501201e-01, 2.88729728482182368e-01, 2.91856585617094655e-01,
    2.94997087799961311e-01, 2.98151326696684982e-01, 3.01319396100802550e-01, 3.04501391976649438e-01,
    3.07697412504291501e-01, 3.10907558126285954e-01, 3.14131931596336678e-01, 3.17370638029913055e-01,
    3.20623784956904856e-01, 3.23891482376390594e-01, 3.27173842813600846e-01, 3.30470981379162920e-01,
    3.33783015830717733e-01, 3.37110066637005379e-01, 3.40452257044521145e-01, 3.43809713146850049e-01,
    3.47182563956792922e-01, 3.50570941481405329e-01, 3.53974980800076000e-01, 3.57394820145779668e-01,
    3.60830600989647143e-01, 3.64282468129003112e-01, 3.67750569779031644e-01, 3.71235057668238555e-01,
    3.74736087137890195e-01, 3.78253817245618240e-01, 3.81788410873392714e-01, 3.85340034840076340e-01,
    3.88908860018787828e-01, 3.92495061459314676e-01, 3.96098818515831508e-01, 3.99720314980196334e-01,
    4.03359739221113622e-01, 4.07017284329472484e-01, 4.10693148270187325e-01, 4.14387534040890237e-01,
    4.18100649837847282e-01, 4.21832709229495006e-01, 4.25583931338021082e-01, 4.29354541029440595e-01,
    4.33144769112651373e-01, 4.36954852547984607e-01, 4.40785034665803044e-01, 4.44635565395738397e-01,
    4.48506701507202010e-01, 4.52398706861847522e-01, 4.56311852678715379e-01, 4.60246417812841757e-01,
    4.64202689048173189e-01, 4.68180961405692431e-01, 4.72181538467728978e-01, 4.76204732719504698e-01,
    4.80250865909045588e-01, 4.84320269426682048e-01, 4.88413284705456752e-01, 4.92530263643867261e-01,
    4.96671569052488437e-01, 5.00837575126147461e-01, 5.05028667943466902e-01, 5.09245245995746609e-01,
    5.13487720747325516e-01, 5.17756517229754909e-01, 5.22052074672320399e-01, 5.26374847171682925e-01,
    5.30725304403660503e-01, 5.35103932380456060e-01, 5.39511234256950467e-01, 5.43947731190024597e-01,
    5.48413963255264147e-01, 5.52910490425830625e-01, 5.57437893618764280e-01, 5.61996775814522675e-01,
    5.66587763256162669e-01, 5.71211506735251451e-01, 5.75868682972351942e-01, 5.80559996100789122e-01,
    5.85286179263369566e-01, 5.90047996332824121e-01, 5.94846243767985561e-01, 5.99681752619123487e-01,
    6.04555390697466000e-01, 6.09468064925771658e-01, 6.14420723888912002e-01, 6.19414360605832437e-01,
    6.24450015547024617e-01, 6.29528779924834803e-01, 6.34651799287621721e-01, 6.39820277453054698e-01,
    6.45035480820820406e-01, 6.50298743110814814e-01, 6.55611470579695377e-01, 6.60975147776661220e-01,
    6.66391343908748102e-01, 6.71861719897079990e-01, 6.77388036218771306e-01, 6.82972161644992526e-01,
    6.88616083004669477e-01, 6.94321916126114380e-01, 7.00091918136509284e-01, 7.05928501332751868e-01,
    7.11834248878245979e-01, 7.17811932630719518e-01, 7.23864533468627669e-01, 7.29995264561473678e-01,
    7.36207598126860097e-01, 7.42505296340148502e-01, 7.48892447219154267e-01, 7.55373506507093451e-01,
    7.61953346836792611e-01, 7.68637315798483489e-01, 7.75431304981184399e-01, 7.82341832654799618e-01,
    7.89376143566021704e-01, 7.96542330422956080e-01, 8.03849483170961387e-01, 8.11307874312653277e-01,
    8.18929191603699369e-01, 8.26726833946218376e-01, 8.34716292986880326e-01, 8.42915653112200958e-01,
    8.51346258458674621e-01, 8.60033621196328091e-01, 8.69008688036853494e-01, 8.78309655808913736e-01,
    8.87984660755829602e-01, 8.98095921898339533e-01, 9.08726440052126772e-01, 9.19991505039342683e-01,
    9.32060075959225798e-01, 9.45198953442294543e-01, 9.59879091800100892e-01, 9.77101701267664491e-01,
    1.00000000000000000e+00,
];

// ===========================================================================
// Deterministic PRNG & Exact Samplers (SplitMix64 + Xoshiro256** + Ziggurat)
// ===========================================================================

/// Deterministic SplitMix64-seeded Xoshiro256** PRNG with exact Ziggurat
/// Normal, Marsaglia–Tsang Gamma, and Knuth/Hörmann PTRD Poisson samplers.
/// Shared with [`crate::capital::dependence`], [`crate::stochastic::hertig`], [`crate::stochastic::micro_reserving`],
/// and [`crate::subannual::odp_asymmetric`].
#[derive(Debug, Clone)]
pub(crate) struct DeterministicRng {
    s: [u64; 4],
}

impl DeterministicRng {
    pub(crate) fn from_seed(seed: u64) -> Self {
        let mut sm = seed;
        let mut next_sm = || {
            sm = sm.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        };
        let mut s = [next_sm(), next_sm(), next_sm(), next_sm()];
        if s == [0; 4] {
            s[0] = 1;
        }
        Self { s }
    }

    #[inline]
    pub(crate) fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform float strictly in `(0, 1)` (53-bit mantissa precision).
    #[inline]
    pub(crate) fn next_open01(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Unbiased uniform index in `0..upper` via Lemire's nearly-divisionless multiply-high.
    #[inline]
    pub(crate) fn next_index(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        let bound = upper as u64;
        let mut prod = u128::from(self.next_u64()) * u128::from(bound);
        let mut low = prod as u64;
        if low < bound {
            let threshold = bound.wrapping_neg() % bound;
            while low < threshold {
                prod = u128::from(self.next_u64()) * u128::from(bound);
                low = prod as u64;
            }
        }
        (prod >> 64) as usize
    }

    /// Standard normal `N(0, 1)` via 256-strip Ziggurat (98.51% branch-light fast
    /// path requiring 1 `next_u64` and zero transcendental functions).
    #[inline]
    pub(crate) fn next_standard_normal(&mut self) -> f64 {
        loop {
            let bits = self.next_u64();
            let i = (bits & 0xFF) as usize;
            // Signed 52-bit integer in [-2^51, 2^51 - 1] (disjoint from bits 0..8);
            // adding 0.5 is exact in IEEE-754 f64 and symmetric around 0.0.
            let j = ((bits as i64) >> 12) as f64 + 0.5;
            let (w, x_inner) = ZIG_NORM_WX[i];
            let x = j * w;
            if x.abs() < x_inner {
                return x;
            }
            if i == 0 {
                return self.sample_normal_tail(j >= 0.0);
            }
            let u = self.next_open01();
            if ZIG_NORM_F[i] + u * (ZIG_NORM_F[i + 1] - ZIG_NORM_F[i]) < (-0.5 * x * x).exp() {
                return x;
            }
        }
    }

    #[cold]
    fn sample_normal_tail(&mut self, positive: bool) -> f64 {
        // Marsaglia (1964) exact exponential tail for |Z| >= ZIG_NORM_R.
        loop {
            let tx = -self.next_open01().ln() * ZIG_NORM_INV_R;
            let ty = -self.next_open01().ln();
            if ty + ty >= tx * tx {
                let mag = ZIG_NORM_R + tx;
                return if positive { mag } else { -mag };
            }
        }
    }

    /// Exact Gamma(shape, scale) via Marsaglia & Tsang (2000).
    #[inline]
    pub(crate) fn sample_gamma(&mut self, shape: f64, scale: f64) -> f64 {
        if !(shape > 0.0) || !(scale > 0.0) || !shape.is_finite() || !scale.is_finite() {
            return 0.0;
        }
        if shape < 1.0 {
            let g = self.sample_gamma_ge1(shape + 1.0);
            let u = self.next_open01();
            return g * u.powf(1.0 / shape) * scale;
        }
        self.sample_gamma_ge1(shape) * scale
    }

    #[inline]
    fn sample_gamma_ge1(&mut self, shape: f64) -> f64 {
        let d = shape - (1.0 / 3.0);
        let c = (1.0 / 3.0) / d.sqrt();
        loop {
            let z = self.next_standard_normal();
            let v_root = 1.0 + c * z;
            if v_root <= 0.0 {
                continue;
            }
            let v = v_root * v_root * v_root;
            let u = self.next_open01();
            let z2 = z * z;
            if u < 1.0 - 0.0331 * (z2 * z2) {
                return d * v;
            }
            if u.ln() < 0.5 * z2 + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    /// Exact Poisson(lambda) via Knuth (`lambda < 10`) and Hörmann PTRD (`lambda >= 10`).
    pub(crate) fn sample_poisson(&mut self, lambda: f64) -> f64 {
        if !(lambda > 0.0) || !lambda.is_finite() {
            return 0.0;
        }
        if lambda < 10.0 {
            let limit = (-lambda).exp();
            let mut k = 0u64;
            let mut p = 1.0;
            loop {
                k += 1;
                p *= self.next_open01();
                if p <= limit {
                    return (k - 1) as f64;
                }
            }
        }
        if lambda >= 1.0e9 {
            // Avoid catastrophic cancellation in `-lambda + k * ln(lambda) - ln_gamma(k + 1)`
            // for huge rates; Normal(lambda, lambda) is indistinguishable to > 4.5 digits.
            return (lambda + lambda.sqrt() * self.next_standard_normal())
                .max(0.0)
                .round();
        }
        // Hörmann (1993), Insurance: Mathematics and Economics 12(1), pp. 39–45.
        // Exact for all lambda >= 10.
        let s = lambda.sqrt();
        let b = 0.931 + 2.53 * s;
        let a = -0.059 + 0.02483 * b;
        let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
        let v_r = 0.9277 - 3.6224 / (b - 2.0);
        let ln_lambda = lambda.ln();
        loop {
            let u = self.next_open01() - 0.5;
            let v = self.next_open01();
            let us = 0.5 - u.abs();
            let k = ((2.0 * a / us + b) * u + lambda + 0.43).floor();
            if us >= 0.07 && v <= v_r {
                return k;
            }
            if k < 0.0 || (us < 0.013 && v > us) {
                continue;
            }
            let lhs = (v * inv_alpha / (a / (us * us) + b)).ln();
            let rhs = -lambda + k * ln_lambda - ln_gamma(k + 1.0);
            if lhs <= rhs {
                return k;
            }
        }
    }
}

/// 4-wide Structure-of-Arrays (SoA) Xoshiro256++ PRNG for lockstep vectorization
/// across 4 independent Monte Carlo lanes with zero `unsafe` code.
struct DeterministicRng4 {
    s0: [u64; 4],
    s1: [u64; 4],
    s2: [u64; 4],
    s3: [u64; 4],
}

impl DeterministicRng4 {
    fn from_rng(rng: &mut DeterministicRng) -> Self {
        let mut s0 = [0u64; 4];
        let mut s1 = [0u64; 4];
        let mut s2 = [0u64; 4];
        let mut s3 = [0u64; 4];
        for lane in 0..4 {
            s0[lane] = rng.next_u64();
            s1[lane] = rng.next_u64();
            s2[lane] = rng.next_u64();
            s3[lane] = rng.next_u64();
            if (s0[lane] | s1[lane] | s2[lane] | s3[lane]) == 0 {
                s0[lane] = 1;
            }
        }
        Self { s0, s1, s2, s3 }
    }

    #[inline]
    fn next_u64x4(&mut self) -> [u64; 4] {
        let mut out = [0u64; 4];
        for lane in 0..4 {
            let s0 = self.s0[lane];
            let s1 = self.s1[lane];
            out[lane] = s0
                .wrapping_add(self.s3[lane])
                .rotate_left(23)
                .wrapping_add(s0);
            let t = s1 << 17;
            self.s2[lane] ^= s0;
            self.s3[lane] ^= s1;
            self.s1[lane] ^= self.s2[lane];
            self.s0[lane] ^= self.s3[lane];
            self.s2[lane] ^= t;
            self.s3[lane] = self.s3[lane].rotate_left(45);
        }
        out
    }

    #[inline]
    fn next_open01x4(&mut self) -> [f64; 4] {
        const SCALE: f64 = 1.0 / 9_007_199_254_740_992.0;
        let bits = self.next_u64x4();
        [
            ((bits[0] >> 11) as f64 + 0.5) * SCALE,
            ((bits[1] >> 11) as f64 + 0.5) * SCALE,
            ((bits[2] >> 11) as f64 + 0.5) * SCALE,
            ((bits[3] >> 11) as f64 + 0.5) * SCALE,
        ]
    }

    #[cold]
    fn finish_normal_lane(bits: u64, x: f64, fallback: &mut DeterministicRng) -> f64 {
        let i = (bits & 0xFF) as usize;
        if i == 0 {
            let j_nonneg = ((bits as i64) >> 12) >= 0;
            return fallback.sample_normal_tail(j_nonneg);
        }
        let u = fallback.next_open01();
        if ZIG_NORM_F[i] + u * (ZIG_NORM_F[i + 1] - ZIG_NORM_F[i]) < (-0.5 * x * x).exp() {
            return x;
        }
        fallback.next_standard_normal()
    }

    #[inline]
    fn next_standard_normal_x4(&mut self, fallback: &mut DeterministicRng) -> [f64; 4] {
        let bits = self.next_u64x4();
        let i0 = (bits[0] & 0xFF) as usize;
        let i1 = (bits[1] & 0xFF) as usize;
        let i2 = (bits[2] & 0xFF) as usize;
        let i3 = (bits[3] & 0xFF) as usize;
        let (w0, xi0) = ZIG_NORM_WX[i0];
        let (w1, xi1) = ZIG_NORM_WX[i1];
        let (w2, xi2) = ZIG_NORM_WX[i2];
        let (w3, xi3) = ZIG_NORM_WX[i3];
        let x0 = (((bits[0] as i64) >> 12) as f64 + 0.5) * w0;
        let x1 = (((bits[1] as i64) >> 12) as f64 + 0.5) * w1;
        let x2 = (((bits[2] as i64) >> 12) as f64 + 0.5) * w2;
        let x3 = (((bits[3] as i64) >> 12) as f64 + 0.5) * w3;
        let ok0 = x0.abs() < xi0;
        let ok1 = x1.abs() < xi1;
        let ok2 = x2.abs() < xi2;
        let ok3 = x3.abs() < xi3;
        if ok0 & ok1 & ok2 & ok3 {
            return [x0, x1, x2, x3];
        }
        [
            if ok0 {
                x0
            } else {
                Self::finish_normal_lane(bits[0], x0, fallback)
            },
            if ok1 {
                x1
            } else {
                Self::finish_normal_lane(bits[1], x1, fallback)
            },
            if ok2 {
                x2
            } else {
                Self::finish_normal_lane(bits[2], x2, fallback)
            },
            if ok3 {
                x3
            } else {
                Self::finish_normal_lane(bits[3], x3, fallback)
            },
        ]
    }

    #[cold]
    fn finish_gamma_lane(
        shape: f64,
        d: f64,
        vr: f64,
        v: f64,
        u: f64,
        z2: f64,
        fallback: &mut DeterministicRng,
    ) -> f64 {
        if vr > 0.0 && u.ln() < 0.5 * z2 + d * (1.0 - v + v.ln()) {
            return d * v;
        }
        fallback.sample_gamma_ge1(shape)
    }

    /// Vectorized 4-wide Marsaglia–Tsang Gamma(shape >= 1, scale = 1).
    #[inline]
    fn sample_gamma_ge1_x4(
        &mut self,
        shapes: [f64; 4],
        fallback: &mut DeterministicRng,
    ) -> [f64; 4] {
        let d = [
            shapes[0] - (1.0 / 3.0),
            shapes[1] - (1.0 / 3.0),
            shapes[2] - (1.0 / 3.0),
            shapes[3] - (1.0 / 3.0),
        ];
        let c = [
            (1.0 / 3.0) / d[0].sqrt(),
            (1.0 / 3.0) / d[1].sqrt(),
            (1.0 / 3.0) / d[2].sqrt(),
            (1.0 / 3.0) / d[3].sqrt(),
        ];
        let z = self.next_standard_normal_x4(fallback);
        let u = self.next_open01x4();
        let vr = [
            1.0 + c[0] * z[0],
            1.0 + c[1] * z[1],
            1.0 + c[2] * z[2],
            1.0 + c[3] * z[3],
        ];
        let v = [
            vr[0] * vr[0] * vr[0],
            vr[1] * vr[1] * vr[1],
            vr[2] * vr[2] * vr[2],
            vr[3] * vr[3] * vr[3],
        ];
        let z2 = [z[0] * z[0], z[1] * z[1], z[2] * z[2], z[3] * z[3]];
        let ok0 = vr[0] > 0.0 && u[0] < 1.0 - 0.0331 * (z2[0] * z2[0]);
        let ok1 = vr[1] > 0.0 && u[1] < 1.0 - 0.0331 * (z2[1] * z2[1]);
        let ok2 = vr[2] > 0.0 && u[2] < 1.0 - 0.0331 * (z2[2] * z2[2]);
        let ok3 = vr[3] > 0.0 && u[3] < 1.0 - 0.0331 * (z2[3] * z2[3]);
        if ok0 & ok1 & ok2 & ok3 {
            return [d[0] * v[0], d[1] * v[1], d[2] * v[2], d[3] * v[3]];
        }
        [
            if ok0 {
                d[0] * v[0]
            } else {
                Self::finish_gamma_lane(shapes[0], d[0], vr[0], v[0], u[0], z2[0], fallback)
            },
            if ok1 {
                d[1] * v[1]
            } else {
                Self::finish_gamma_lane(shapes[1], d[1], vr[1], v[1], u[1], z2[1], fallback)
            },
            if ok2 {
                d[2] * v[2]
            } else {
                Self::finish_gamma_lane(shapes[2], d[2], vr[2], v[2], u[2], z2[2], fallback)
            },
            if ok3 {
                d[3] * v[3]
            } else {
                Self::finish_gamma_lane(shapes[3], d[3], vr[3], v[3], u[3], z2[3], fallback)
            },
        ]
    }
}

/// Branch-light batched unbiased index sampler with loop-invariant precomputed
/// Lemire rejection threshold.
///
/// For `bound <= 1024` (triangles up to `45 × 45`), `fill_weighted_deltas`
/// extracts 4 independent exact 16-bit Lemire indices per `next_u64()` with
/// zero per-element refill branches and fuses the `pool[idx] * sqrt_mu[k]`
/// product into a single pass. For larger pools (`bound <= u32::MAX`), it
/// extracts 2 independent 32-bit Lemire indices per `next_u64()`.
pub(crate) struct FastIndexSampler {
    bound: usize,
    threshold: u32,
}

impl FastIndexSampler {
    #[inline]
    pub(crate) fn new(bound: usize) -> Self {
        debug_assert!(bound > 0);
        let threshold = if bound <= 1024 {
            65_536u32 % (bound as u32)
        } else if u32::try_from(bound).is_ok() {
            let b = bound as u32;
            b.wrapping_neg() % b
        } else {
            0
        };
        Self { bound, threshold }
    }

    #[cold]
    fn sample_16_slow(rng: &mut DeterministicRng, bound: u32, threshold: u32) -> usize {
        loop {
            let mut w = rng.next_u64();
            for _ in 0..4 {
                let prod = ((w & 0xFFFF) as u32) * bound;
                if (prod & 0xFFFF) >= threshold {
                    return (prod >> 16) as usize;
                }
                w >>= 16;
            }
        }
    }

    #[cold]
    fn sample_32_slow(rng: &mut DeterministicRng, bound: u64, threshold: u32) -> usize {
        loop {
            let w = rng.next_u64();
            let p0 = (w & 0xFFFF_FFFF) * bound;
            if (p0 as u32) >= threshold {
                return (p0 >> 32) as usize;
            }
            let p1 = (w >> 32) * bound;
            if (p1 as u32) >= threshold {
                return (p1 >> 32) as usize;
            }
        }
    }

    #[inline]
    pub(crate) fn next(&mut self, rng: &mut DeterministicRng) -> usize {
        if self.bound <= 1 {
            return 0;
        }
        if self.bound <= 1024 {
            let bound = self.bound as u32;
            let threshold = self.threshold;
            let prod = ((rng.next_u64() & 0xFFFF) as u32) * bound;
            if (prod & 0xFFFF) >= threshold {
                return (prod >> 16) as usize;
            }
            return Self::sample_16_slow(rng, bound, threshold);
        }
        if u32::try_from(self.bound).is_ok() {
            let bound = self.bound as u64;
            let threshold = self.threshold;
            let prod = (rng.next_u64() >> 32) * bound;
            if (prod as u32) >= threshold {
                return (prod >> 32) as usize;
            }
            return Self::sample_32_slow(rng, bound, threshold);
        }
        rng.next_index(self.bound)
    }

    /// Fill `out[k] = pool[idx_k] * sqrt_mu[k]` for every cell using 4-wide
    /// 16-bit Lemire (1 `next_u64()` per 4 cells, 0 refill branches, parallel
    /// multiplies) for `bound <= 1024`, or 2-wide 32-bit Lemire for larger pools.
    #[inline]
    pub(crate) fn fill_weighted_deltas(
        &self,
        pool: &[f64],
        sqrt_mu: &[f64],
        rng: &mut DeterministicRng,
        out: &mut [f64],
    ) {
        debug_assert_eq!(out.len(), sqrt_mu.len());
        if self.bound <= 1 {
            let r0 = pool[0];
            for (dst, &sm) in out.iter_mut().zip(sqrt_mu.iter()) {
                *dst = r0 * sm;
            }
            return;
        }
        if self.bound <= 1024 {
            let bound = self.bound as u32;
            let threshold = self.threshold;
            let (out_chunks, out_rem) = out.as_chunks_mut::<4>();
            let (sm_chunks, sm_rem) = sqrt_mu.as_chunks::<4>();
            for (oc, sc) in out_chunks.iter_mut().zip(sm_chunks.iter()) {
                let w = rng.next_u64();
                let p0 = ((w & 0xFFFF) as u32) * bound;
                let p1 = (((w >> 16) & 0xFFFF) as u32) * bound;
                let p2 = (((w >> 32) & 0xFFFF) as u32) * bound;
                let p3 = ((w >> 48) as u32) * bound;
                let l0 = p0 & 0xFFFF;
                let l1 = p1 & 0xFFFF;
                let l2 = p2 & 0xFFFF;
                let l3 = p3 & 0xFFFF;
                if l0.min(l1).min(l2).min(l3) >= threshold {
                    oc[0] = pool[(p0 >> 16) as usize] * sc[0];
                    oc[1] = pool[(p1 >> 16) as usize] * sc[1];
                    oc[2] = pool[(p2 >> 16) as usize] * sc[2];
                    oc[3] = pool[(p3 >> 16) as usize] * sc[3];
                } else {
                    let i0 = if l0 >= threshold {
                        (p0 >> 16) as usize
                    } else {
                        Self::sample_16_slow(rng, bound, threshold)
                    };
                    let i1 = if l1 >= threshold {
                        (p1 >> 16) as usize
                    } else {
                        Self::sample_16_slow(rng, bound, threshold)
                    };
                    let i2 = if l2 >= threshold {
                        (p2 >> 16) as usize
                    } else {
                        Self::sample_16_slow(rng, bound, threshold)
                    };
                    let i3 = if l3 >= threshold {
                        (p3 >> 16) as usize
                    } else {
                        Self::sample_16_slow(rng, bound, threshold)
                    };
                    oc[0] = pool[i0] * sc[0];
                    oc[1] = pool[i1] * sc[1];
                    oc[2] = pool[i2] * sc[2];
                    oc[3] = pool[i3] * sc[3];
                }
            }
            if !out_rem.is_empty() {
                let mut w = rng.next_u64();
                for (dst, &sm) in out_rem.iter_mut().zip(sm_rem.iter()) {
                    let p = ((w & 0xFFFF) as u32) * bound;
                    w >>= 16;
                    let idx = if (p & 0xFFFF) >= threshold {
                        (p >> 16) as usize
                    } else {
                        Self::sample_16_slow(rng, bound, threshold)
                    };
                    *dst = pool[idx] * sm;
                }
            }
            return;
        }
        if u32::try_from(self.bound).is_ok() {
            let bound = self.bound as u64;
            let threshold = self.threshold;
            let (out_chunks, out_rem) = out.as_chunks_mut::<2>();
            let (sm_chunks, sm_rem) = sqrt_mu.as_chunks::<2>();
            for (oc, sc) in out_chunks.iter_mut().zip(sm_chunks.iter()) {
                let w = rng.next_u64();
                let p0 = (w & 0xFFFF_FFFF) * bound;
                let p1 = (w >> 32) * bound;
                let l0 = p0 as u32;
                let l1 = p1 as u32;
                if l0.min(l1) >= threshold {
                    oc[0] = pool[(p0 >> 32) as usize] * sc[0];
                    oc[1] = pool[(p1 >> 32) as usize] * sc[1];
                } else {
                    let i0 = if l0 >= threshold {
                        (p0 >> 32) as usize
                    } else {
                        Self::sample_32_slow(rng, bound, threshold)
                    };
                    let i1 = if l1 >= threshold {
                        (p1 >> 32) as usize
                    } else {
                        Self::sample_32_slow(rng, bound, threshold)
                    };
                    oc[0] = pool[i0] * sc[0];
                    oc[1] = pool[i1] * sc[1];
                }
            }
            if let (Some(dst), Some(&sm)) = (out_rem.first_mut(), sm_rem.first()) {
                let p = (rng.next_u64() >> 32) * bound;
                let idx = if (p as u32) >= threshold {
                    (p >> 32) as usize
                } else {
                    Self::sample_32_slow(rng, bound, threshold)
                };
                *dst = pool[idx] * sm;
            }
            return;
        }
        for (dst, &sm) in out.iter_mut().zip(sqrt_mu.iter()) {
            *dst = pool[rng.next_index(self.bound)] * sm;
        }
    }
}

/// Lanczos approximation to `ln Γ(z)` for `z > 0` (accuracy ~1e-14).
pub(crate) fn ln_gamma(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = z - 1.0;
    let mut sum = COEFFS[0];
    for (idx, &coeff) in COEFFS.iter().enumerate().skip(1) {
        sum += coeff / (x + idx as f64);
    }
    let t = x + 7.5;
    0.5 * TAU.ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

// ===========================================================================
// ODP GLM Fit (Exact O(n^2) Tridiagonal Green's Function + O(n^3) Fallback)
// ===========================================================================

struct OdpSolvedDiagnostics {
    residuals: Vec<OdpCellResidual>,
    n_observed: usize,
    rank: usize,
    degrees_of_freedom: usize,
    dispersion: f64,
    origin_reserves: Vec<f64>,
    total_reserve: f64,
    analytic_origin_process_se: Vec<f64>,
    analytic_origin_param_se: Vec<f64>,
    analytic_origin_prediction_error: Vec<f64>,
    analytic_total_process_se: f64,
    analytic_total_param_se: f64,
    analytic_total_prediction_error: f64,
}

/// Exact `O(n^2)` time, `O(n)` auxiliary-space solver for full strictly-positive
/// right triangles (`μ_{i,j} = α_i · β_j > 0` for all `i + j < n`).
///
/// In cumulative development coordinates `T_k = sum_{j=1}^k β_j v_j` (`T_0 = 0`),
/// the Schur complement of `X^T W X` reduces to an `(n - 1) × (n - 1)` symmetric
/// positive-definite tridiagonal matrix `K`. Its bidiagonal `L D L^T` factorization
/// takes `O(n)` time, its Green's function yields every hat-matrix leverage `h_{i,j}`
/// in `O(1)` (`O(n^2)` total), and every analytic ODP parameter variance collapses to
/// a 2-term closed form (`O(n)` total across all origins plus portfolio total).
///
/// Returns `None` if any cell has non-positive fitted mass or ill-conditioned pivot,
/// signaling the caller to fall back to the general `O(n^3)` SPD-inverse path.
fn try_solve_odp_tridiagonal(
    n: usize,
    ldfs: &[f64],
    tail: f64,
    latest: &[f64],
    fitted_cumulative: &[f64],
    fitted_incremental: &[f64],
    observed_incremental: &[f64],
) -> Option<OdpSolvedDiagnostics> {
    let m = n - 1;
    if latest.iter().any(|&v| !(v > 0.0) || !v.is_finite())
        || ldfs.iter().any(|&f| !(f > 1.0) || !f.is_finite())
    {
        return None;
    }

    // Rank-1 factorization μ_{i,j} = α_i · β_j with sum_j β_j = 1.
    let mut cdfs_to_last = vec![0.0; n];
    cdfs_from_ldfs(ldfs, 1.0, &mut cdfs_to_last);
    let mut b_cum = vec![0.0; n];
    let mut beta = vec![0.0; n];
    for j in 0..n {
        let inv_cdf = 1.0 / cdfs_to_last[j];
        if !inv_cdf.is_finite() || !(inv_cdf > 0.0) {
            return None;
        }
        b_cum[j] = inv_cdf;
        beta[j] = if j == 0 {
            inv_cdf
        } else {
            inv_cdf - b_cum[j - 1]
        };
        if !(beta[j] > 0.0) || !beta[j].is_finite() {
            return None;
        }
    }

    let mut alpha = vec![0.0; n];
    for i in 0..n {
        let a = fitted_cumulative[i * n + m];
        if !(a > 0.0) || !a.is_finite() {
            return None;
        }
        alpha[i] = a;
    }

    // A[k] = sum_{i=0}^{n-1-k} α_i.
    let mut a_sum = vec![0.0; n];
    let mut acc = 0.0;
    for k in (0..n).rev() {
        acc += alpha[m - k];
        a_sum[k] = acc;
    }

    // Assemble symmetric tridiagonal K of dimension m = n - 1 (indices 0..m correspond to k = 1..n).
    let mut d = vec![0.0; m];
    let mut e = vec![0.0; m.saturating_sub(1)];
    for k in 1..n {
        let idx = k - 1;
        if k < m {
            let off = a_sum[k + 1] / beta[k + 1];
            d[idx] = a_sum[k] / beta[k] + off - alpha[m - k] / b_cum[k];
            e[idx] = -off;
        } else {
            d[idx] = a_sum[k] / beta[k] - alpha[0] / b_cum[k];
        }
    }

    // Forward L D L^T sweep in O(n): L_{idx+1, idx} = -gamma[idx].
    let mut d_tilde = vec![0.0; m];
    let mut gamma = vec![0.0; m.saturating_sub(1)];
    if !d[0].is_finite() || d[0] <= 0.0 {
        return None;
    }
    d_tilde[0] = d[0];
    for idx in 0..(m - 1) {
        let g = -e[idx] / d_tilde[idx];
        let next_dt = d[idx + 1] + e[idx] * g;
        if !g.is_finite() || !next_dt.is_finite() || next_dt <= 1e-14 * d[idx + 1].abs().max(1e-30)
        {
            return None;
        }
        gamma[idx] = g;
        d_tilde[idx + 1] = next_dt;
    }

    // Backward diagonal recurrence for G = K^{-1} in O(n) (sum of positive squares).
    let mut g_diag = vec![0.0; m];
    g_diag[m - 1] = 1.0 / d_tilde[m - 1];
    for idx in (0..(m - 1)).rev() {
        g_diag[idx] = 1.0 / d_tilde[idx] + gamma[idx] * gamma[idx] * g_diag[idx + 1];
    }

    // Precompute w_j^T G w_j for j = 1..n in O(n).
    let mut w_g_w = vec![0.0; n];
    w_g_w[1] = g_diag[0] / (beta[1] * beta[1]);
    for j in 2..n {
        let g_jm1_j = gamma[j - 2] * g_diag[j - 1];
        w_g_w[j] = (g_diag[j - 1] - 2.0 * g_jm1_j + g_diag[j - 2]) / (beta[j] * beta[j]);
    }

    // Compute all N = n(n+1)/2 residuals & exact leverages in row-major order in O(n^2).
    let n_observed = n * (n + 1) / 2;
    let rank = 2 * n - 1;
    let degrees_of_freedom = n_observed - rank;
    let mut residuals = Vec::with_capacity(n_observed);
    let mut pearson_sq_sum = 0.0;
    let mut g_row = vec![0.0; n];

    for origin in 0..n {
        let ki = m - origin;
        if ki >= 1 {
            g_row[ki] = g_diag[ki - 1];
            for j in (1..ki).rev() {
                g_row[j] = gamma[j - 1] * g_row[j + 1];
            }
        }
        let inv_b_ki = 1.0 / b_cum[ki];
        let base_phi = inv_b_ki / alpha[origin];
        let z_g_z = if ki >= 1 {
            g_diag[ki - 1] * inv_b_ki * inv_b_ki
        } else {
            0.0
        };
        let row_off = origin * n;

        for dev in 0..=ki {
            let obs = observed_incremental[row_off + dev];
            let mu = fitted_incremental[row_off + dev];
            if !(mu > 0.0) || !mu.is_finite() {
                return None;
            }
            let raw_residual = (obs - mu) / mu.sqrt();
            pearson_sq_sum += raw_residual * raw_residual;

            let y_g_y = if dev == 0 {
                z_g_z
            } else if dev == 1 {
                let w_g_z = -g_row[1] * inv_b_ki / beta[1];
                w_g_w[1] + 2.0 * w_g_z + z_g_z
            } else {
                let w_g_z = -(g_row[dev] - g_row[dev - 1]) * inv_b_ki / beta[dev];
                w_g_w[dev] + 2.0 * w_g_z + z_g_z
            };
            let leverage = (mu * (base_phi + y_g_y)).clamp(0.0, 1.0);
            let saturated = (1.0 - leverage) <= 1e-10;

            residuals.push(OdpCellResidual {
                origin,
                dev,
                observed_incremental: obs,
                fitted_incremental: mu,
                raw_pearson_residual: raw_residual,
                leverage,
                saturated,
            });
        }
    }

    let dispersion = pearson_sq_sum / degrees_of_freedom as f64;

    // Precompute column G[k, m-1] for k = 0..m in O(n).
    let mut g_last = vec![0.0; m];
    g_last[m - 1] = g_diag[m - 1];
    for idx in (0..(m - 1)).rev() {
        g_last[idx] = gamma[idx] * g_last[idx + 1];
    }

    // Closed-form O(n) origin reserves & analytic prediction errors.
    let mut origin_reserves = vec![0.0; n];
    let mut analytic_origin_process_se = vec![0.0; n];
    let mut analytic_origin_param_se = vec![0.0; n];
    let mut analytic_origin_prediction_error = vec![0.0; n];
    let mut pre_tail_total_reserve = 0.0;
    let mut r_diag_sum = 0.0;
    let g_mm = g_diag[m - 1];

    for origin in 0..n {
        let ki = m - origin;
        let ai = alpha[origin];
        let r_pre_tail = if origin == 0 {
            0.0
        } else {
            ai * (1.0 - b_cum[ki])
        };
        pre_tail_total_reserve += r_pre_tail;
        let tail_increment = ai * (tail - 1.0);
        origin_reserves[origin] = r_pre_tail + tail_increment;

        let param_quad = if origin == 0 {
            0.0
        } else if origin < m {
            let inv_b = 1.0 / b_cum[ki];
            let diag_term = (r_pre_tail * r_pre_tail) * inv_b / ai;
            r_diag_sum += diag_term;
            diag_term
                + ai * ai * (g_diag[ki - 1] * inv_b * inv_b - 2.0 * g_last[ki - 1] * inv_b + g_mm)
        } else {
            let diag_term = (r_pre_tail * r_pre_tail) / (ai * b_cum[0]);
            r_diag_sum += diag_term;
            diag_term + ai * ai * g_mm
        };
        let param_var = (dispersion * param_quad).max(0.0);
        let proc_var = (dispersion * r_pre_tail).max(0.0);
        analytic_origin_process_se[origin] = proc_var.sqrt();
        analytic_origin_param_se[origin] = param_var.sqrt();
        analytic_origin_prediction_error[origin] = (proc_var + param_var).sqrt();
    }

    // Portfolio total parameter variance via one O(n) forward pass: y^T K^{-1} y = z^T D_tilde^{-1} z.
    let mut total_quad = r_diag_sum;
    let mut z_prev = 0.0;
    for idx in 0..m {
        let k = idx + 1;
        let y_k = if k < m {
            -alpha[m - k] / b_cum[k]
        } else {
            a_sum[0] - alpha[0]
        };
        let z_k = if idx == 0 {
            y_k
        } else {
            y_k + gamma[idx - 1] * z_prev
        };
        total_quad += (z_k * z_k) / d_tilde[idx];
        z_prev = z_k;
    }

    let total_reserve: f64 = origin_reserves.iter().sum();
    let total_param_var = (dispersion * total_quad).max(0.0);
    let total_proc_var = (dispersion * pre_tail_total_reserve).max(0.0);

    Some(OdpSolvedDiagnostics {
        residuals,
        n_observed,
        rank,
        degrees_of_freedom,
        dispersion,
        origin_reserves,
        total_reserve,
        analytic_origin_process_se,
        analytic_origin_param_se,
        analytic_origin_prediction_error,
        analytic_total_process_se: total_proc_var.sqrt(),
        analytic_total_param_se: total_param_var.sqrt(),
        analytic_total_prediction_error: (total_proc_var + total_param_var).sqrt(),
    })
}

/// General `O(n^3)` fallback for triangles with zero-mass fitted cells (e.g., an
/// LDF equal to `1.0` or a zero origin row). Uses `invert_spd` once in `O(p^3)`
/// and computes each hat-matrix leverage in `O(1)` via 3-sparse lookups (`O(N)`
/// total rather than `O(N p^2)`).
fn solve_odp_general(
    n: usize,
    tail: f64,
    fitted_cumulative: &[f64],
    fitted_incremental: &[f64],
    observed_incremental: &[f64],
) -> Result<OdpSolvedDiagnostics, ValidationError> {
    let mut active_origins = vec![false; n];
    let mut active_devs = vec![false; n];
    let mut active_cells: Vec<(usize, usize)> = Vec::with_capacity(n * (n + 1) / 2);
    for origin in 0..n {
        for dev in 0..(n - origin) {
            let mu = fitted_incremental[origin * n + dev];
            if mu.is_finite() && mu > 0.0 {
                active_origins[origin] = true;
                active_devs[dev] = true;
                active_cells.push((origin, dev));
            }
        }
    }

    let mut origin_col = vec![None; n];
    let mut dev_col = vec![None; n];
    let mut param_count = 1usize;
    let mut first_origin_skipped = false;
    for origin in 0..n {
        if active_origins[origin] {
            if first_origin_skipped {
                origin_col[origin] = Some(param_count);
                param_count += 1;
            } else {
                first_origin_skipped = true;
            }
        }
    }
    let mut first_dev_skipped = false;
    for dev in 0..n {
        if active_devs[dev] {
            if first_dev_skipped {
                dev_col[dev] = Some(param_count);
                param_count += 1;
            } else {
                first_dev_skipped = true;
            }
        }
    }

    let n_observed = active_cells.len();
    if n_observed <= param_count {
        return Err(ValidationError::InvalidParameter(
            "ODP model has non-positive residual degrees of freedom",
        ));
    }
    let degrees_of_freedom = n_observed - param_count;

    // Assemble Fisher information matrix M = X^T W X in O(N).
    let mut info = vec![0.0; param_count * param_count];
    for &(origin, dev) in &active_cells {
        let w = fitted_incremental[origin * n + dev];
        let mut cols = [0usize; 3];
        let mut ncols = 1usize;
        if let Some(c) = origin_col[origin] {
            cols[ncols] = c;
            ncols += 1;
        }
        if let Some(c) = dev_col[dev] {
            cols[ncols] = c;
            ncols += 1;
        }
        for a in 0..ncols {
            for b in 0..ncols {
                info[cols[a] * param_count + cols[b]] += w;
            }
        }
    }

    let chol = cholesky_lower(&info, param_count)?;
    let info_inv = invert_spd(&info, param_count)?;

    // Exact hat-matrix leverages in O(1) per cell via 3-sparse quadratic form x^T M^{-1} x.
    let mut residuals = Vec::with_capacity(n_observed);
    let mut pearson_sq_sum = 0.0;
    for &(origin, dev) in &active_cells {
        let obs = observed_incremental[origin * n + dev];
        let mu = fitted_incremental[origin * n + dev];
        let raw_residual = (obs - mu) / mu.sqrt();
        pearson_sq_sum += raw_residual * raw_residual;

        let mut cols = [0usize; 3];
        let mut ncols = 1usize;
        if let Some(c) = origin_col[origin] {
            cols[ncols] = c;
            ncols += 1;
        }
        if let Some(c) = dev_col[dev] {
            cols[ncols] = c;
            ncols += 1;
        }
        let mut qf = 0.0;
        for a in 0..ncols {
            let ca = cols[a];
            qf += info_inv[ca * param_count + ca];
            for b in (a + 1)..ncols {
                qf += 2.0 * info_inv[ca * param_count + cols[b]];
            }
        }
        let leverage = (mu * qf).clamp(0.0, 1.0);
        let saturated = (1.0 - leverage) <= 1e-10;

        residuals.push(OdpCellResidual {
            origin,
            dev,
            observed_incremental: obs,
            fitted_incremental: mu,
            raw_pearson_residual: raw_residual,
            leverage,
            saturated,
        });
    }

    let dispersion = pearson_sq_sum / degrees_of_freedom as f64;

    let mut origin_reserves = vec![0.0; n];
    let mut analytic_origin_process_se = vec![0.0; n];
    let mut analytic_origin_param_se = vec![0.0; n];
    let mut analytic_origin_prediction_error = vec![0.0; n];
    let mut g_total = vec![0.0; param_count];
    let mut g_origin = vec![0.0; param_count];
    let mut y_buf = vec![0.0; param_count];
    let mut pre_tail_total_reserve = 0.0;

    for origin in 0..n {
        g_origin.fill(0.0);
        let mut r_pre_tail = 0.0;
        for dev in (n - origin)..n {
            let mu = fitted_incremental[origin * n + dev];
            r_pre_tail += mu;
            g_origin[0] += mu;
            g_total[0] += mu;
            if let Some(c) = origin_col[origin] {
                g_origin[c] += mu;
                g_total[c] += mu;
            }
            if let Some(c) = dev_col[dev] {
                g_origin[c] += mu;
                g_total[c] += mu;
            }
        }
        pre_tail_total_reserve += r_pre_tail;
        let tail_increment = fitted_cumulative[origin * n + (n - 1)] * (tail - 1.0);
        origin_reserves[origin] = r_pre_tail + tail_increment;

        solve_lower_into(&chol, param_count, &g_origin, &mut y_buf);
        let param_var = (dispersion * y_buf.iter().map(|v| v * v).sum::<f64>()).max(0.0);
        let proc_var = (dispersion * r_pre_tail).max(0.0);
        analytic_origin_process_se[origin] = proc_var.sqrt();
        analytic_origin_param_se[origin] = param_var.sqrt();
        analytic_origin_prediction_error[origin] = (proc_var + param_var).sqrt();
    }

    let total_reserve: f64 = origin_reserves.iter().sum();
    solve_lower_into(&chol, param_count, &g_total, &mut y_buf);
    let total_param_var = (dispersion * y_buf.iter().map(|v| v * v).sum::<f64>()).max(0.0);
    let total_proc_var = (dispersion * pre_tail_total_reserve).max(0.0);

    Ok(OdpSolvedDiagnostics {
        residuals,
        n_observed,
        rank: param_count,
        degrees_of_freedom,
        dispersion,
        origin_reserves,
        total_reserve,
        analytic_origin_process_se,
        analytic_origin_param_se,
        analytic_origin_prediction_error,
        analytic_total_process_se: total_proc_var.sqrt(),
        analytic_total_param_se: total_param_var.sqrt(),
        analytic_total_prediction_error: (total_proc_var + total_param_var).sqrt(),
    })
}

/// Fit the Over-Dispersed Poisson GLM (`log μ_{i,j} = c + α_i + β_j`) to a
/// cumulative loss triangle.
///
/// Produces fitted expectations, unscaled Pearson residuals, exact GLM hat-
/// matrix diagonal leverages `h_{ii}`, Pearson dispersion `φ`, and England &
/// Verrall (1999, 2002) analytic prediction errors.
pub fn fit_odp(
    triangle: &TriangleView<'_>,
    tail: f64,
    policy: MonotonicityPolicy,
) -> Result<OdpFit, ValidationError> {
    validate_triangle(triangle)?;
    if policy == MonotonicityPolicy::Strict {
        validate_monotone_cumulative(triangle)?;
    }
    validate_tail(tail)?;

    let n = triangle.n_origins;
    if n < 3 {
        return Err(ValidationError::InvalidParameter(
            "ODP model requires at least a 3x3 triangle for positive degrees of freedom",
        ));
    }

    // Single-pass volume-weighted LDFs without duplicating triangle validation.
    let mut ldfs = vec![0.0; n - 1];
    volume_weighted_ldfs(triangle.data, n, n, &mut ldfs);
    for &f in &ldfs {
        if !f.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if f <= 0.0 {
            return Err(ValidationError::DivideByZero("development denominator"));
        }
        if f < 1.0 {
            return Err(ValidationError::InvalidParameter(
                "ODP GLM requires non-negative fitted expectations (all volume-weighted LDFs >= 1.0)",
            ));
        }
    }

    let observed_incremental = cumulative_to_incremental(triangle);
    let latest = latest_diagonal(triangle);

    // Exact ODP GLM MLE via backward + forward chain-ladder recursion in O(n^2).
    let mut fitted_cumulative = vec![0.0; n * n];
    for origin in 0..n {
        let d = n - 1 - origin;
        let row_off = origin * n;
        fitted_cumulative[row_off + d] = latest[origin];
        for dev in (0..d).rev() {
            fitted_cumulative[row_off + dev] = fitted_cumulative[row_off + dev + 1] / ldfs[dev];
        }
        for dev in (d + 1)..n {
            fitted_cumulative[row_off + dev] = fitted_cumulative[row_off + dev - 1] * ldfs[dev - 1];
        }
    }

    let mut fitted_incremental = vec![0.0; n * n];
    for origin in 0..n {
        let row_off = origin * n;
        fitted_incremental[row_off] = fitted_cumulative[row_off];
        for dev in 1..n {
            fitted_incremental[row_off + dev] =
                fitted_cumulative[row_off + dev] - fitted_cumulative[row_off + dev - 1];
        }
    }

    let solved = match try_solve_odp_tridiagonal(
        n,
        &ldfs,
        tail,
        &latest,
        &fitted_cumulative,
        &fitted_incremental,
        &observed_incremental,
    ) {
        Some(fast) => fast,
        None => solve_odp_general(
            n,
            tail,
            &fitted_cumulative,
            &fitted_incremental,
            &observed_incremental,
        )?,
    };

    Ok(OdpFit {
        n_origins: n,
        n_devs: n,
        ldfs,
        tail,
        fitted_cumulative,
        fitted_incremental,
        observed_incremental,
        residuals: solved.residuals,
        n_observed: solved.n_observed,
        rank: solved.rank,
        degrees_of_freedom: solved.degrees_of_freedom,
        dispersion: solved.dispersion,
        origin_reserves: solved.origin_reserves,
        total_reserve: solved.total_reserve,
        analytic_origin_process_se: solved.analytic_origin_process_se,
        analytic_origin_param_se: solved.analytic_origin_param_se,
        analytic_origin_prediction_error: solved.analytic_origin_prediction_error,
        analytic_total_process_se: solved.analytic_total_process_se,
        analytic_total_param_se: solved.analytic_total_param_se,
        analytic_total_prediction_error: solved.analytic_total_prediction_error,
    })
}

/// Build the adjusted residual pool for resampling along with its pre-centering mean.
pub(crate) fn build_adjusted_residual_pool(
    fit: &OdpFit,
    adjustment: ResidualAdjustment,
    center: bool,
) -> Result<(Vec<f64>, f64), ValidationError> {
    let mut pool = Vec::with_capacity(fit.residuals.len());
    match adjustment {
        ResidualAdjustment::DegreesOfFreedom => {
            let scale = (fit.n_observed as f64 / fit.degrees_of_freedom as f64).sqrt();
            for cell in &fit.residuals {
                pool.push(cell.raw_pearson_residual * scale);
            }
        }
        ResidualAdjustment::HatMatrixLeverage => {
            for cell in &fit.residuals {
                if !cell.saturated {
                    pool.push(cell.raw_pearson_residual / (1.0 - cell.leverage).sqrt());
                }
            }
        }
    }
    if pool.is_empty() {
        return Err(ValidationError::InvalidParameter(
            "no eligible residuals available for ODP bootstrap resampling",
        ));
    }
    let pool_mean = pool.iter().sum::<f64>() / pool.len() as f64;
    if center {
        for r in &mut pool {
            *r -= pool_mean;
        }
    }
    Ok((pool, pool_mean))
}

/// Compact reusable scratch buffers for zero-allocation, single-pass ODP
/// bootstrap iterations.
///
/// Packs only the `N = n(n+1)/2` upper-triangle `sqrt(mu)` values (half the
/// memory of an `n × n` grid), precomputes baseline column denominators and
/// latest-diagonal sums so the hot loop accumulates only zero-mean FMA deltas,
/// and eliminates the `col_num` array via `col_num[k] = col_den[k+1] + latest_pseudo[m-1-k]`.
pub(crate) struct OdpBootstrapScratch {
    n: usize,
    all_observed_positive: bool,
    pub(crate) inv_phi: f64,
    packed_sqrt_mu: Vec<f64>,
    cell_deltas: Vec<f64>,
    base_col_den: Vec<f64>,
    base_latest: Vec<f64>,
    delta_col_den: Vec<f64>,
    latest_pseudo: Vec<f64>,
    pub(crate) pseudo_ldfs: Vec<f64>,
    origin_reserves: Vec<f64>,
}

impl OdpBootstrapScratch {
    pub(crate) fn for_fit(fit: &OdpFit) -> Self {
        let n = fit.n_origins;
        let m = n.saturating_sub(1);
        let upper_len = n * (n + 1) / 2;
        let mut packed_sqrt_mu = Vec::with_capacity(upper_len);
        let mut base_col_den = vec![0.0; m];
        let mut base_latest = vec![0.0; n];

        for origin in 0..n {
            let row_off = origin * n;
            let last_dev = n - 1 - origin;
            for dev in 0..=last_dev {
                let mu = fit.fitted_incremental[row_off + dev];
                packed_sqrt_mu.push(if mu > 0.0 { mu.sqrt() } else { 0.0 });
                if dev < last_dev {
                    base_col_den[dev] += fit.fitted_cumulative[row_off + dev];
                }
            }
            base_latest[origin] = fit.fitted_cumulative[row_off + last_dev];
        }

        let inv_phi = if fit.dispersion > 0.0 && fit.dispersion.is_finite() {
            1.0 / fit.dispersion
        } else {
            0.0
        };

        Self {
            n,
            all_observed_positive: fit.n_observed == upper_len,
            inv_phi,
            packed_sqrt_mu,
            cell_deltas: vec![0.0; upper_len],
            base_col_den,
            base_latest,
            delta_col_den: vec![0.0; m],
            latest_pseudo: vec![0.0; n],
            pseudo_ldfs: vec![0.0; m],
            origin_reserves: vec![0.0; n],
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct RefitOutcome {
    pub(crate) valid: bool,
    all_ldfs_ge_one: bool,
}

/// Complete volume-weighted LDF computation from `delta_col_den` + `latest_pseudo`
/// in a single right-to-left `O(n)` pass using `col_num[k] = col_den[k+1] + latest_pseudo[m-1-k]`.
#[inline]
fn finalize_pseudo_ldfs(scratch: &mut OdpBootstrapScratch) -> RefitOutcome {
    let m = scratch.n.saturating_sub(1);
    let mut all_ge_one = true;
    let mut next_den = 0.0;
    for k in (0..m).rev() {
        let den = scratch.base_col_den[k] + scratch.delta_col_den[k];
        if !(den > 0.0) || !den.is_finite() {
            return RefitOutcome {
                valid: false,
                all_ldfs_ge_one: false,
            };
        }
        let num = next_den + scratch.latest_pseudo[m - 1 - k];
        let f = num / den;
        if !f.is_finite() || f <= 0.0 {
            return RefitOutcome {
                valid: false,
                all_ldfs_ge_one: false,
            };
        }
        if f < 1.0 {
            all_ge_one = false;
        }
        scratch.pseudo_ldfs[k] = f;
        next_den = den;
    }
    RefitOutcome {
        valid: true,
        all_ldfs_ge_one: all_ge_one,
    }
}

/// Accumulate zero-mean cell perturbations `cell_deltas[0..N]` into
/// `delta_col_den[0..m]` and `latest_pseudo[0..n]` using paired origin rows
/// (breaking the serial `running += d` FP latency chain into 2 parallel pipes
/// and halving `delta_den` store traffic), then finalize volume-weighted LDFs.
#[inline]
fn accumulate_cell_deltas_and_refit(scratch: &mut OdpBootstrapScratch) -> RefitOutcome {
    let n = scratch.n;
    let m = n.saturating_sub(1);
    let deltas = &scratch.cell_deltas;
    let delta_den = &mut scratch.delta_col_den;
    let latest_out = &mut scratch.latest_pseudo;
    let base_latest = &scratch.base_latest;

    // Pair origins (0, 1) to overwrite delta_den[0..m] directly (no fill(0.0)).
    // Note: fit_odp guarantees n >= 3 so m >= 2.
    let mut row_start = 0usize;
    let mut origin = 0usize;
    if m >= 2 {
        let r0 = &deltas[0..n];
        let r1 = &deltas[n..2 * n - 1];
        row_start = 2 * n - 1;
        let mut run0 = 0.0;
        let mut run1 = 0.0;
        for (den_slot, (&d0, &d1)) in delta_den[..m - 1]
            .iter_mut()
            .zip(r0[..m - 1].iter().zip(r1[..m - 1].iter()))
        {
            run0 += d0;
            run1 += d1;
            *den_slot = run0 + run1;
        }
        run0 += r0[m - 1];
        delta_den[m - 1] = run0;
        latest_out[0] = base_latest[0] + run0 + r0[m];
        latest_out[1] = base_latest[1] + run1 + r1[m - 1];
        origin = 2;
    } else if m == 1 {
        let d0 = deltas[0];
        delta_den[0] = d0;
        latest_out[0] = base_latest[0] + d0 + deltas[1];
        row_start = 2;
        origin = 1;
    }

    // Remaining origin pairs (origin, origin + 1) while origin + 1 < m.
    while origin + 1 < m {
        let len_a = n - origin; // last_dev_a + 1 = m - origin + 1
        let len_b = len_a - 1; // last_dev_b + 1 = m - origin
        let ra = &deltas[row_start..row_start + len_a];
        let rb = &deltas[row_start + len_a..row_start + len_a + len_b];
        row_start += len_a + len_b;

        let last_a = len_a - 1;
        let last_b = len_b - 1;
        let mut run_a = 0.0;
        let mut run_b = 0.0;
        for (den_slot, (&da, &db)) in delta_den[..last_b]
            .iter_mut()
            .zip(ra[..last_b].iter().zip(rb[..last_b].iter()))
        {
            run_a += da;
            run_b += db;
            *den_slot += run_a + run_b;
        }
        run_a += ra[last_b];
        delta_den[last_b] += run_a;
        latest_out[origin] = base_latest[origin] + run_a + ra[last_a];
        latest_out[origin + 1] = base_latest[origin + 1] + run_b + rb[last_b];
        origin += 2;
    }

    if origin < m {
        let last_dev = m - origin;
        let row = &deltas[row_start..=(row_start + last_dev)];
        row_start += last_dev + 1;
        let mut running = 0.0;
        for (den_slot, &d) in delta_den[..last_dev].iter_mut().zip(&row[..last_dev]) {
            running += d;
            *den_slot += running;
        }
        latest_out[origin] = base_latest[origin] + running + row[last_dev];
    }
    latest_out[m] = base_latest[m] + deltas[row_start];

    finalize_pseudo_ldfs(scratch)
}

/// Fused 4-wide batched resampling (`pool[idx] * sqrt_mu`) + bounds-check-free
/// delta accumulation + LDF refit.
#[inline]
pub(crate) fn refit_pseudo_ldfs_from_pool(
    pool: &[f64],
    sampler: &mut FastIndexSampler,
    rng: &mut DeterministicRng,
    scratch: &mut OdpBootstrapScratch,
) -> RefitOutcome {
    if scratch.all_observed_positive {
        sampler.fill_weighted_deltas(pool, &scratch.packed_sqrt_mu, rng, &mut scratch.cell_deltas);
        return accumulate_cell_deltas_and_refit(scratch);
    }

    // General path when some observed cells have zero fitted mass (`sm == 0.0`).
    for (dst, &sm) in scratch
        .cell_deltas
        .iter_mut()
        .zip(scratch.packed_sqrt_mu.iter())
    {
        *dst = if sm > 0.0 {
            pool[sampler.next(rng)] * sm
        } else {
            0.0
        };
    }
    accumulate_cell_deltas_and_refit(scratch)
}

/// Fused single-pass pseudo-triangle delta accumulation + LDF refit from an
/// externally supplied residual slice (used by synchronized multi-line bootstrap).
#[inline]
fn refit_pseudo_ldfs_from_slice(
    sampled_residuals: &[f64],
    scratch: &mut OdpBootstrapScratch,
) -> RefitOutcome {
    if scratch.all_observed_positive {
        for ((dst, &r), &sm) in scratch
            .cell_deltas
            .iter_mut()
            .zip(sampled_residuals.iter())
            .zip(scratch.packed_sqrt_mu.iter())
        {
            *dst = r * sm;
        }
        return accumulate_cell_deltas_and_refit(scratch);
    }

    let mut r_idx = 0usize;
    for (dst, &sm) in scratch
        .cell_deltas
        .iter_mut()
        .zip(scratch.packed_sqrt_mu.iter())
    {
        if sm > 0.0 {
            *dst = sampled_residuals[r_idx] * sm;
            r_idx += 1;
        } else {
            *dst = 0.0;
        }
    }
    accumulate_cell_deltas_and_refit(scratch)
}

#[inline]
fn sample_positive_process(
    mag: f64,
    phi: f64,
    inv_phi: f64,
    dist: ProcessDistribution,
    rng: &mut DeterministicRng,
) -> f64 {
    if !(phi > 0.0) || !(mag > 0.0) || !mag.is_finite() {
        return mag;
    }
    match dist {
        ProcessDistribution::Gamma => rng.sample_gamma(mag * inv_phi, phi),
        ProcessDistribution::OverDispersedPoisson => phi * rng.sample_poisson(mag * inv_phi),
        ProcessDistribution::None => mag,
    }
}

#[inline]
pub(crate) fn sample_process_increment_fast(
    future_mu: f64,
    phi: f64,
    inv_phi: f64,
    dist: ProcessDistribution,
    rng: &mut DeterministicRng,
) -> f64 {
    if !(phi > 0.0) || future_mu == 0.0 || !future_mu.is_finite() {
        return future_mu;
    }
    // Per Shapland (2016) §3.5.3, preserve process variance on negative pseudo-
    // incremental expectations by reflecting the draw: -Sample(|mu|, phi).
    let sign = if future_mu < 0.0 { -1.0 } else { 1.0 };
    let mag = future_mu.abs();
    let draw = match dist {
        ProcessDistribution::Gamma => rng.sample_gamma(mag * inv_phi, phi),
        ProcessDistribution::OverDispersedPoisson => phi * rng.sample_poisson(mag * inv_phi),
        ProcessDistribution::None => mag,
    };
    sign * draw
}

/// Project per-origin reserves and sample exact process variance in `O(n)`.
///
/// Exploits the reproductive additivity of Gamma(`shape_j`, `φ`) and
/// `φ · Poisson(λ_j)` under common dispersion `φ` so each open origin requires
/// 1 draw (2 when negative pseudo-LDFs create both positive and negative future
/// increments) instead of `O(n)` cell-level draws (`O(n^2)` total).
///
/// Writes `scratch.origin_reserves[0..n]` and returns `Some(total_reserve)`.
#[inline]
fn project_origin_reserves(
    tail: f64,
    phi: f64,
    all_ldfs_ge_one: bool,
    process_distribution: ProcessDistribution,
    rng: &mut DeterministicRng,
    scratch: &mut OdpBootstrapScratch,
) -> Option<f64> {
    let n = scratch.n;
    let inv_phi = scratch.inv_phi;
    let tail_inc = tail - 1.0;

    if all_ldfs_ge_one || process_distribution == ProcessDistribution::None {
        // Walking origin = 0..n traverses latest_dev = n-1 down to 0, so a single
        // register `cdf` compounds pseudo-LDFs right-to-left with zero arrays.
        let mut cdf = tail;
        let l0 = scratch.latest_pseudo[0];
        if !l0.is_finite() {
            return None;
        }
        let r0 = if tail_inc > 0.0 {
            sample_process_increment_fast(l0 * tail_inc, phi, inv_phi, process_distribution, rng)
        } else {
            0.0
        };
        if !r0.is_finite() {
            return None;
        }
        scratch.origin_reserves[0] = r0;
        let mut total = r0;

        match process_distribution {
            ProcessDistribution::Gamma if phi > 0.0 => {
                for origin in 1..n {
                    cdf *= scratch.pseudo_ldfs[n - 1 - origin];
                    let latest_i = scratch.latest_pseudo[origin];
                    let expected_res = latest_i * (cdf - 1.0);
                    let res_i = if expected_res > 0.0 && expected_res.is_finite() {
                        let shape = expected_res * inv_phi;
                        if shape >= 1.0 {
                            rng.sample_gamma_ge1(shape) * phi
                        } else {
                            rng.sample_gamma(shape, phi)
                        }
                    } else if expected_res < 0.0 && expected_res.is_finite() {
                        -rng.sample_gamma(-expected_res * inv_phi, phi)
                    } else if expected_res == 0.0 {
                        0.0
                    } else {
                        return None;
                    };
                    scratch.origin_reserves[origin] = res_i;
                    total += res_i;
                }
            }
            ProcessDistribution::OverDispersedPoisson if phi > 0.0 => {
                for origin in 1..n {
                    cdf *= scratch.pseudo_ldfs[n - 1 - origin];
                    let latest_i = scratch.latest_pseudo[origin];
                    if !latest_i.is_finite() {
                        return None;
                    }
                    let expected_res = latest_i * (cdf - 1.0);
                    let res_i = sample_process_increment_fast(
                        expected_res,
                        phi,
                        inv_phi,
                        ProcessDistribution::OverDispersedPoisson,
                        rng,
                    );
                    if !res_i.is_finite() {
                        return None;
                    }
                    scratch.origin_reserves[origin] = res_i;
                    total += res_i;
                }
            }
            _ => {
                for origin in 1..n {
                    cdf *= scratch.pseudo_ldfs[n - 1 - origin];
                    let latest_i = scratch.latest_pseudo[origin];
                    let expected_res = latest_i * (cdf - 1.0);
                    if !expected_res.is_finite() {
                        return None;
                    }
                    scratch.origin_reserves[origin] = expected_res;
                    total += expected_res;
                }
            }
        }
        return total.is_finite().then_some(total);
    }

    // General path when any pseudo-LDF < 1.0: track exact positive and negative
    // suffix increment factors in two registers (exact equivalence to Shapland
    // cell-by-cell reflection because every valid pseudo-LDF > 0 preserves signs).
    let mut pos_factor = tail_inc;
    let mut neg_factor = 0.0;

    let l0 = scratch.latest_pseudo[0];
    if !l0.is_finite() {
        return None;
    }
    let r0 = if tail_inc > 0.0 {
        sample_process_increment_fast(l0 * tail_inc, phi, inv_phi, process_distribution, rng)
    } else {
        0.0
    };
    if !r0.is_finite() {
        return None;
    }
    scratch.origin_reserves[0] = r0;
    let mut total = r0;

    for origin in 1..n {
        let f = scratch.pseudo_ldfs[n - 1 - origin];
        let step = f - 1.0;
        if step >= 0.0 {
            pos_factor = step + f * pos_factor;
            neg_factor *= f;
        } else {
            pos_factor *= f;
            neg_factor = (-step) + f * neg_factor;
        }

        let latest_i = scratch.latest_pseudo[origin];
        if !latest_i.is_finite() {
            return None;
        }
        let (pos_mu, neg_mu) = if latest_i >= 0.0 {
            (latest_i * pos_factor, latest_i * neg_factor)
        } else {
            (-latest_i * neg_factor, -latest_i * pos_factor)
        };
        let pos_draw = if pos_mu > 0.0 {
            sample_positive_process(pos_mu, phi, inv_phi, process_distribution, rng)
        } else {
            0.0
        };
        let neg_draw = if neg_mu > 0.0 {
            sample_positive_process(neg_mu, phi, inv_phi, process_distribution, rng)
        } else {
            0.0
        };
        let res_i = pos_draw - neg_draw;
        if !res_i.is_finite() {
            return None;
        }
        scratch.origin_reserves[origin] = res_i;
        total += res_i;
    }

    total.is_finite().then_some(total)
}

/// Execute one ODP bootstrap iteration into reusable `scratch` buffers using an
/// externally supplied residual slice (used by synchronized multi-line bootstrap).
///
/// Returns `Some(&scratch.origin_reserves)` on valid refit or `None` if the
/// pseudo-triangle has a degenerate non-positive column sum.
pub(crate) fn run_odp_iteration_with_scratch<'a>(
    fit: &OdpFit,
    sampled_residuals: &[f64],
    process_distribution: ProcessDistribution,
    rng: &mut DeterministicRng,
    scratch: &'a mut OdpBootstrapScratch,
) -> Option<&'a [f64]> {
    let outcome = refit_pseudo_ldfs_from_slice(sampled_residuals, scratch);
    if !outcome.valid {
        return None;
    }
    project_origin_reserves(
        fit.tail,
        fit.dispersion,
        outcome.all_ldfs_ge_one,
        process_distribution,
        rng,
        scratch,
    )?;
    Some(&scratch.origin_reserves)
}

/// Extract 4 independent unbiased 16-bit Lemire indices from a caller-supplied
/// `u64` word, falling back to `rng` only on the rare `< threshold` rejection.
#[inline]
fn decode_4_indices_16(
    w: u64,
    bound: u32,
    threshold: u32,
    rng: &mut DeterministicRng,
) -> [usize; 4] {
    if bound <= 1 {
        return [0; 4];
    }
    let p0 = ((w & 0xFFFF) as u32) * bound;
    let p1 = (((w >> 16) & 0xFFFF) as u32) * bound;
    let p2 = (((w >> 32) & 0xFFFF) as u32) * bound;
    let p3 = ((w >> 48) as u32) * bound;
    let l0 = p0 & 0xFFFF;
    let l1 = p1 & 0xFFFF;
    let l2 = p2 & 0xFFFF;
    let l3 = p3 & 0xFFFF;
    if l0.min(l1).min(l2).min(l3) >= threshold {
        return [
            (p0 >> 16) as usize,
            (p1 >> 16) as usize,
            (p2 >> 16) as usize,
            (p3 >> 16) as usize,
        ];
    }
    [
        if l0 >= threshold {
            (p0 >> 16) as usize
        } else {
            FastIndexSampler::sample_16_slow(rng, bound, threshold)
        },
        if l1 >= threshold {
            (p1 >> 16) as usize
        } else {
            FastIndexSampler::sample_16_slow(rng, bound, threshold)
        },
        if l2 >= threshold {
            (p2 >> 16) as usize
        } else {
            FastIndexSampler::sample_16_slow(rng, bound, threshold)
        },
        if l3 >= threshold {
            (p3 >> 16) as usize
        } else {
            FastIndexSampler::sample_16_slow(rng, bound, threshold)
        },
    ]
}

/// 4-wide SoA lockstep ODP bootstrap fast path: fuses 16-bit Lemire residual
/// resampling (fed by 4-wide `DeterministicRng4` words into a bounds-check-free
/// packed pool) directly into `[f64; 4]` row/column accumulators, refits 4 sets
/// of pseudo-LDFs in parallel, and samples 4-wide vectorized Marsaglia–Tsang
/// Gamma process variance.
#[allow(clippy::too_many_arguments)]
fn run_odp_bootstrap_x4(
    batches: usize,
    fit: &OdpFit,
    pool: &[f64],
    sampler: &FastIndexSampler,
    rng: &mut DeterministicRng,
    scratch: &mut OdpBootstrapScratch,
    simulated_totals: &mut Vec<f64>,
    simulated_by_origin: &mut [Vec<f64>],
    failed_iterations: &mut usize,
) {
    let n = scratch.n;
    let m = n - 1;
    let phi = fit.dispersion;
    let inv_phi = scratch.inv_phi;
    let tail = fit.tail;
    let tail_inc = tail - 1.0;
    let bound = sampler.bound as u32;
    let threshold = sampler.threshold;

    // Copy pool into a fixed 1024-entry table so `pool_buf[idx & 0x3FF]` is
    // provably in-bounds at compile time (zero bounds checks in the inner loop).
    let mut pool_buf = [0.0f64; 1024];
    pool_buf[..pool.len()].copy_from_slice(pool);

    let mut rng4 = DeterministicRng4::from_rng(rng);
    let mut delta_col_den_stack = [[0.0f64; 4]; 16];
    let mut latest_pseudo_stack = [[0.0f64; 4]; 16];
    let mut pseudo_ldfs_stack = [[0.0f64; 4]; 16];
    let mut origin_res_stack = [[0.0f64; 4]; 16];

    let mut delta_col_den_heap = if n > 16 {
        vec![[0.0f64; 4]; m]
    } else {
        Vec::new()
    };
    let mut latest_pseudo_heap = if n > 16 {
        vec![[0.0f64; 4]; n]
    } else {
        Vec::new()
    };
    let mut pseudo_ldfs_heap = if n > 16 {
        vec![[0.0f64; 4]; m]
    } else {
        Vec::new()
    };
    let mut origin_res_heap = if n > 16 {
        vec![[0.0f64; 4]; n]
    } else {
        Vec::new()
    };

    let delta_col_den: &mut [[f64; 4]] = if n <= 16 {
        &mut delta_col_den_stack[..m]
    } else {
        &mut delta_col_den_heap[..m]
    };
    let latest_pseudo: &mut [[f64; 4]] = if n <= 16 {
        &mut latest_pseudo_stack[..n]
    } else {
        &mut latest_pseudo_heap[..n]
    };
    let pseudo_ldfs: &mut [[f64; 4]] = if n <= 16 {
        &mut pseudo_ldfs_stack[..m]
    } else {
        &mut pseudo_ldfs_heap[..m]
    };
    let origin_res: &mut [[f64; 4]] = if n <= 16 {
        &mut origin_res_stack[..n]
    } else {
        &mut origin_res_heap[..n]
    };

    let mut words = [0u64; 4];
    let mut word_pos = 4usize;

    for _ in 0..batches {
        // 1. Fused 4-wide resample + row/col accumulation (zero cell_deltas buffer).
        // Origin 0 overwrites delta_col_den[0..m] directly (no fill(0.0)).
        let sqrt_mu = &scratch.packed_sqrt_mu;
        let base_col_den = &scratch.base_col_den;
        let base_latest = &scratch.base_latest;

        macro_rules! next_idx4 {
            () => {{
                if word_pos == 4 {
                    words = rng4.next_u64x4();
                    word_pos = 0;
                }
                let w = words[word_pos];
                word_pos += 1;
                decode_4_indices_16(w, bound, threshold, rng)
            }};
        }

        {
            let mut run = [0.0f64; 4];
            for (den_slot, &sm) in delta_col_den.iter_mut().zip(&sqrt_mu[..m]) {
                let idx = next_idx4!();
                run[0] += pool_buf[idx[0] & 0x3FF] * sm;
                run[1] += pool_buf[idx[1] & 0x3FF] * sm;
                run[2] += pool_buf[idx[2] & 0x3FF] * sm;
                run[3] += pool_buf[idx[3] & 0x3FF] * sm;
                *den_slot = run;
            }
            let sm = sqrt_mu[m];
            let idx = next_idx4!();
            let bl = base_latest[0];
            latest_pseudo[0] = [
                bl + run[0] + pool_buf[idx[0] & 0x3FF] * sm,
                bl + run[1] + pool_buf[idx[1] & 0x3FF] * sm,
                bl + run[2] + pool_buf[idx[2] & 0x3FF] * sm,
                bl + run[3] + pool_buf[idx[3] & 0x3FF] * sm,
            ];
        }

        let mut cell_idx = m + 1;

        // Origins 1..m
        for origin in 1..m {
            let last_dev = m - origin;
            let row_sm = &sqrt_mu[cell_idx..=(cell_idx + last_dev)];
            cell_idx += last_dev + 1;
            let mut run = [0.0f64; 4];
            for (den_slot, &sm) in delta_col_den[..last_dev]
                .iter_mut()
                .zip(&row_sm[..last_dev])
            {
                let idx = next_idx4!();
                run[0] += pool_buf[idx[0] & 0x3FF] * sm;
                run[1] += pool_buf[idx[1] & 0x3FF] * sm;
                run[2] += pool_buf[idx[2] & 0x3FF] * sm;
                run[3] += pool_buf[idx[3] & 0x3FF] * sm;
                den_slot[0] += run[0];
                den_slot[1] += run[1];
                den_slot[2] += run[2];
                den_slot[3] += run[3];
            }
            let sm = row_sm[last_dev];
            let idx = next_idx4!();
            let bl = base_latest[origin];
            latest_pseudo[origin] = [
                bl + run[0] + pool_buf[idx[0] & 0x3FF] * sm,
                bl + run[1] + pool_buf[idx[1] & 0x3FF] * sm,
                bl + run[2] + pool_buf[idx[2] & 0x3FF] * sm,
                bl + run[3] + pool_buf[idx[3] & 0x3FF] * sm,
            ];
        }

        // Final origin row (m, single cell at dev = 0)
        {
            let sm = sqrt_mu[cell_idx];
            let idx = next_idx4!();
            let bl = base_latest[m];
            latest_pseudo[m] = [
                bl + pool_buf[idx[0] & 0x3FF] * sm,
                bl + pool_buf[idx[1] & 0x3FF] * sm,
                bl + pool_buf[idx[2] & 0x3FF] * sm,
                bl + pool_buf[idx[3] & 0x3FF] * sm,
            ];
        }

        // 2. Finalize 4 sets of volume-weighted LDFs right-to-left.
        let mut lane_valid = [true; 4];
        let mut lane_all_ge_one = [true; 4];
        let mut next_den = [0.0f64; 4];
        for k in (0..m).rev() {
            let bd = base_col_den[k];
            let d = delta_col_den[k];
            let lp = latest_pseudo[m - 1 - k];
            let den = [bd + d[0], bd + d[1], bd + d[2], bd + d[3]];
            let num = [
                next_den[0] + lp[0],
                next_den[1] + lp[1],
                next_den[2] + lp[2],
                next_den[3] + lp[3],
            ];
            let f = [
                num[0] / den[0],
                num[1] / den[1],
                num[2] / den[2],
                num[3] / den[3],
            ];
            for j in 0..4 {
                if !(den[j] > 0.0) || !den[j].is_finite() || !f[j].is_finite() || f[j] <= 0.0 {
                    lane_valid[j] = false;
                } else if f[j] < 1.0 {
                    lane_all_ge_one[j] = false;
                }
            }
            pseudo_ldfs[k] = f;
            next_den = den;
        }

        let all_fast = lane_valid[0]
            & lane_valid[1]
            & lane_valid[2]
            & lane_valid[3]
            & lane_all_ge_one[0]
            & lane_all_ge_one[1]
            & lane_all_ge_one[2]
            & lane_all_ge_one[3];

        if all_fast {
            let mut cdf = [tail; 4];
            let l0 = latest_pseudo[0];
            let mut batch_ok =
                l0[0].is_finite() & l0[1].is_finite() & l0[2].is_finite() & l0[3].is_finite();
            let r0 = if batch_ok && tail_inc > 0.0 {
                let shapes = [
                    l0[0] * tail_inc * inv_phi,
                    l0[1] * tail_inc * inv_phi,
                    l0[2] * tail_inc * inv_phi,
                    l0[3] * tail_inc * inv_phi,
                ];
                let fast_shapes = shapes[0] >= 1.0
                    && shapes[1] >= 1.0
                    && shapes[2] >= 1.0
                    && shapes[3] >= 1.0
                    && shapes[0].is_finite()
                    && shapes[1].is_finite()
                    && shapes[2].is_finite()
                    && shapes[3].is_finite();
                if fast_shapes {
                    let g = rng4.sample_gamma_ge1_x4(shapes, rng);
                    [g[0] * phi, g[1] * phi, g[2] * phi, g[3] * phi]
                } else {
                    [
                        sample_process_increment_fast(
                            l0[0] * tail_inc,
                            phi,
                            inv_phi,
                            ProcessDistribution::Gamma,
                            rng,
                        ),
                        sample_process_increment_fast(
                            l0[1] * tail_inc,
                            phi,
                            inv_phi,
                            ProcessDistribution::Gamma,
                            rng,
                        ),
                        sample_process_increment_fast(
                            l0[2] * tail_inc,
                            phi,
                            inv_phi,
                            ProcessDistribution::Gamma,
                            rng,
                        ),
                        sample_process_increment_fast(
                            l0[3] * tail_inc,
                            phi,
                            inv_phi,
                            ProcessDistribution::Gamma,
                            rng,
                        ),
                    ]
                }
            } else {
                [0.0; 4]
            };
            origin_res[0] = r0;
            let mut totals = r0;

            if batch_ok {
                for origin in 1..n {
                    let f = pseudo_ldfs[n - 1 - origin];
                    cdf[0] *= f[0];
                    cdf[1] *= f[1];
                    cdf[2] *= f[2];
                    cdf[3] *= f[3];
                    let li = latest_pseudo[origin];
                    let exp_res = [
                        li[0] * (cdf[0] - 1.0),
                        li[1] * (cdf[1] - 1.0),
                        li[2] * (cdf[2] - 1.0),
                        li[3] * (cdf[3] - 1.0),
                    ];
                    let shapes = [
                        exp_res[0] * inv_phi,
                        exp_res[1] * inv_phi,
                        exp_res[2] * inv_phi,
                        exp_res[3] * inv_phi,
                    ];
                    let fast_shapes = shapes[0] >= 1.0
                        && shapes[1] >= 1.0
                        && shapes[2] >= 1.0
                        && shapes[3] >= 1.0
                        && shapes[0].is_finite()
                        && shapes[1].is_finite()
                        && shapes[2].is_finite()
                        && shapes[3].is_finite();
                    let res_i = if fast_shapes {
                        let g = rng4.sample_gamma_ge1_x4(shapes, rng);
                        [g[0] * phi, g[1] * phi, g[2] * phi, g[3] * phi]
                    } else {
                        let mut r = [0.0f64; 4];
                        for j in 0..4 {
                            let er = exp_res[j];
                            if !er.is_finite() {
                                batch_ok = false;
                                break;
                            }
                            r[j] = if er > 0.0 {
                                let sh = shapes[j];
                                if sh >= 1.0 {
                                    rng.sample_gamma_ge1(sh) * phi
                                } else {
                                    rng.sample_gamma(sh, phi)
                                }
                            } else if er < 0.0 {
                                -rng.sample_gamma(-er * inv_phi, phi)
                            } else {
                                0.0
                            };
                        }
                        if !batch_ok {
                            break;
                        }
                        r
                    };
                    origin_res[origin] = res_i;
                    totals[0] += res_i[0];
                    totals[1] += res_i[1];
                    totals[2] += res_i[2];
                    totals[3] += res_i[3];
                }
            }

            if batch_ok
                & totals[0].is_finite()
                & totals[1].is_finite()
                & totals[2].is_finite()
                & totals[3].is_finite()
            {
                simulated_totals.extend_from_slice(&totals);
                if tail_inc > 0.0 {
                    for (origin_vec, r4) in simulated_by_origin.iter_mut().zip(origin_res.iter()) {
                        origin_vec.extend_from_slice(r4);
                    }
                } else {
                    // Origin 0 is identically 0.0 when tail == 1.0; filled once after the loop.
                    for (origin_vec, r4) in simulated_by_origin[1..]
                        .iter_mut()
                        .zip(origin_res[1..].iter())
                    {
                        origin_vec.extend_from_slice(r4);
                    }
                }
                continue;
            }
        }

        // Scalar fallback for any batch containing invalid lanes, LDF < 1, or non-finite values.
        for j in 0..4 {
            if !lane_valid[j] {
                *failed_iterations += 1;
                continue;
            }
            for origin in 0..n {
                scratch.latest_pseudo[origin] = latest_pseudo[origin][j];
            }
            for k in 0..m {
                scratch.pseudo_ldfs[k] = pseudo_ldfs[k][j];
            }
            if let Some(total) = project_origin_reserves(
                tail,
                phi,
                lane_all_ge_one[j],
                ProcessDistribution::Gamma,
                rng,
                scratch,
            ) {
                simulated_totals.push(total);
                if tail_inc > 0.0 {
                    for (origin_vec, &r) in simulated_by_origin
                        .iter_mut()
                        .zip(scratch.origin_reserves.iter())
                    {
                        origin_vec.push(r);
                    }
                } else {
                    for (origin_vec, &r) in simulated_by_origin[1..]
                        .iter_mut()
                        .zip(scratch.origin_reserves[1..].iter())
                    {
                        origin_vec.push(r);
                    }
                }
            } else {
                *failed_iterations += 1;
            }
        }
    }

    if tail_inc <= 0.0 {
        simulated_by_origin[0].resize(simulated_totals.len(), 0.0);
    }
}

/// Run the full England & Verrall (1999, 2002) / Shapland (2016) ODP bootstrap.
///
/// # Example
/// ```rust
/// use reserving_kernels::datasets::genins;
/// use reserving_kernels::prelude::*;
///
/// let tri = genins();
/// let opts = OdpBootstrapOptions {
///     iterations: 200,
///     seed: 42,
///     ..OdpBootstrapOptions::default()
/// };
/// let boot = bootstrap_odp(&tri.view(), 1.0, opts)?;
/// assert_eq!(boot.simulated_totals.len(), 200);
/// assert!(boot.base_total_reserve > 0.0);
/// # Ok::<(), ValidationError>(())
/// ```
pub fn bootstrap_odp(
    triangle: &TriangleView<'_>,
    tail: f64,
    options: OdpBootstrapOptions,
) -> Result<OdpBootstrapResult, ValidationError> {
    if options.iterations == 0 {
        return Err(ValidationError::InvalidIterations);
    }

    let fit = fit_odp(triangle, tail, options.monotonicity_policy)?;
    let (pool, pool_mean) =
        build_adjusted_residual_pool(&fit, options.residual_adjustment, options.center_residuals)?;

    let mut rng = DeterministicRng::from_seed(options.seed);
    let mut sampler = FastIndexSampler::new(pool.len());
    let n = fit.n_origins;
    let mut simulated_totals = Vec::with_capacity(options.iterations);
    let mut simulated_by_origin: Vec<Vec<f64>> = (0..n)
        .map(|_| Vec::with_capacity(options.iterations))
        .collect();
    let mut scratch = OdpBootstrapScratch::for_fit(&fit);
    let mut failed_iterations = 0usize;

    let remaining = if options.iterations >= 4
        && scratch.all_observed_positive
        && pool.len() <= 1024
        && options.process_distribution == ProcessDistribution::Gamma
        && fit.dispersion > 0.0
        && n >= 3
    {
        let batches = options.iterations / 4;
        run_odp_bootstrap_x4(
            batches,
            &fit,
            &pool,
            &sampler,
            &mut rng,
            &mut scratch,
            &mut simulated_totals,
            &mut simulated_by_origin,
            &mut failed_iterations,
        );
        options.iterations % 4
    } else {
        options.iterations
    };

    for _ in 0..remaining {
        let outcome = refit_pseudo_ldfs_from_pool(&pool, &mut sampler, &mut rng, &mut scratch);
        if !outcome.valid {
            failed_iterations += 1;
            continue;
        }
        if let Some(total) = project_origin_reserves(
            fit.tail,
            fit.dispersion,
            outcome.all_ldfs_ge_one,
            options.process_distribution,
            &mut rng,
            &mut scratch,
        ) {
            simulated_totals.push(total);
            for (origin_vec, &r) in simulated_by_origin
                .iter_mut()
                .zip(scratch.origin_reserves.iter())
            {
                origin_vec.push(r);
            }
        } else {
            failed_iterations += 1;
        }
    }

    if simulated_totals.is_empty() {
        return Err(ValidationError::NotFinite);
    }

    // Reuse a single scratch buffer across portfolio + all n origin summaries
    // and skip redundant finiteness scans (every pushed value is verified finite).
    let mut summary_buf = Vec::with_capacity(simulated_totals.len());
    let summary = summarize_with_scratch(&simulated_totals, &mut summary_buf, true)?;
    let mut origin_summaries = Vec::with_capacity(n);
    for series in &simulated_by_origin {
        origin_summaries.push(summarize_with_scratch(series, &mut summary_buf, true)?);
    }

    Ok(OdpBootstrapResult {
        base_total_reserve: fit.total_reserve,
        fit,
        options,
        simulated_totals,
        simulated_by_origin,
        origin_summaries,
        summary,
        failed_iterations,
        residual_pool_mean_before_centering: pool_mean,
    })
}

/// Run a deterministic ODP bootstrap using the classic England & Verrall
/// (1999, 2002) specification (`DegreesOfFreedom` scaling + Gamma process variance).
///
/// Because this compact entry point only returns portfolio totals, it skips
/// allocating per-origin simulation series (`O(B)` space instead of `O(n B)`),
/// skips `n` per-origin summaries, and collapses process variance to 1–2 exact
/// Gamma draws per iteration.
pub fn bootstrap(
    triangle: &TriangleView<'_>,
    tail: f64,
    config: BootstrapConfig,
) -> Result<BootstrapResult, ValidationError> {
    if config.iterations == 0 {
        return Err(ValidationError::ShapeMismatch);
    }

    let options = OdpBootstrapOptions::england_verrall(config.iterations, config.seed);
    let fit = fit_odp(triangle, tail, options.monotonicity_policy)?;
    let (pool, _) =
        build_adjusted_residual_pool(&fit, options.residual_adjustment, options.center_residuals)?;

    let mut rng = DeterministicRng::from_seed(options.seed);
    let mut sampler = FastIndexSampler::new(pool.len());
    let n = fit.n_origins;
    let phi = fit.dispersion;
    let mut scratch = OdpBootstrapScratch::for_fit(&fit);
    let inv_phi = scratch.inv_phi;
    let tail_inc = fit.tail - 1.0;
    let mut simulated_totals = Vec::with_capacity(config.iterations);

    for _ in 0..config.iterations {
        let outcome = refit_pseudo_ldfs_from_pool(&pool, &mut sampler, &mut rng, &mut scratch);
        if !outcome.valid {
            continue;
        }
        let mut ok = true;
        let total = if outcome.all_ldfs_ge_one {
            let mut cdf = fit.tail;
            let l0 = scratch.latest_pseudo[0];
            if !l0.is_finite() {
                ok = false;
            }
            let mut pos_sum = if tail_inc > 0.0 && l0 > 0.0 {
                l0 * tail_inc
            } else {
                0.0
            };
            let mut neg_sum = if tail_inc > 0.0 && l0 < 0.0 {
                -l0 * tail_inc
            } else {
                0.0
            };
            for origin in 1..n {
                cdf *= scratch.pseudo_ldfs[n - 1 - origin];
                let li = scratch.latest_pseudo[origin];
                if !li.is_finite() {
                    ok = false;
                    break;
                }
                let expected = li * (cdf - 1.0);
                if expected >= 0.0 {
                    pos_sum += expected;
                } else {
                    neg_sum -= expected;
                }
            }
            if ok {
                let p_draw = if pos_sum > 0.0 {
                    sample_positive_process(
                        pos_sum,
                        phi,
                        inv_phi,
                        options.process_distribution,
                        &mut rng,
                    )
                } else {
                    0.0
                };
                let n_draw = if neg_sum > 0.0 {
                    sample_positive_process(
                        neg_sum,
                        phi,
                        inv_phi,
                        options.process_distribution,
                        &mut rng,
                    )
                } else {
                    0.0
                };
                p_draw - n_draw
            } else {
                f64::NAN
            }
        } else {
            let mut pos_factor = tail_inc;
            let mut neg_factor = 0.0;
            let l0 = scratch.latest_pseudo[0];
            if !l0.is_finite() {
                ok = false;
            }
            let mut pos_sum = if tail_inc > 0.0 && l0 > 0.0 {
                l0 * tail_inc
            } else {
                0.0
            };
            let mut neg_sum = if tail_inc > 0.0 && l0 < 0.0 {
                -l0 * tail_inc
            } else {
                0.0
            };
            for origin in 1..n {
                let f = scratch.pseudo_ldfs[n - 1 - origin];
                let step = f - 1.0;
                if step >= 0.0 {
                    pos_factor = step + f * pos_factor;
                    neg_factor *= f;
                } else {
                    pos_factor *= f;
                    neg_factor = (-step) + f * neg_factor;
                }
                let li = scratch.latest_pseudo[origin];
                if !li.is_finite() {
                    ok = false;
                    break;
                }
                if li >= 0.0 {
                    pos_sum += li * pos_factor;
                    neg_sum += li * neg_factor;
                } else {
                    pos_sum += -li * neg_factor;
                    neg_sum += -li * pos_factor;
                }
            }
            if ok {
                let p_draw = if pos_sum > 0.0 {
                    sample_positive_process(
                        pos_sum,
                        phi,
                        inv_phi,
                        options.process_distribution,
                        &mut rng,
                    )
                } else {
                    0.0
                };
                let n_draw = if neg_sum > 0.0 {
                    sample_positive_process(
                        neg_sum,
                        phi,
                        inv_phi,
                        options.process_distribution,
                        &mut rng,
                    )
                } else {
                    0.0
                };
                p_draw - n_draw
            } else {
                f64::NAN
            }
        };
        if ok && total.is_finite() {
            simulated_totals.push(total);
        }
    }

    if simulated_totals.is_empty() {
        return Err(ValidationError::NotFinite);
    }

    let mut summary_buf = Vec::with_capacity(simulated_totals.len());
    let summary = summarize_with_scratch(&simulated_totals, &mut summary_buf, true)?;
    Ok(BootstrapResult {
        base_total_reserve: fit.total_reserve,
        simulated_totals,
        summary,
    })
}

/// Reserve-only convenience wrapper around [`bootstrap`].
pub fn bootstrap_summary(
    triangle: &TriangleView<'_>,
    tail: f64,
    config: BootstrapConfig,
) -> Result<BootstrapSummary, ValidationError> {
    Ok(bootstrap(triangle, tail, config)?.summary)
}

/// Output of the 1-year Solvency II Claims Development Result (CDR)
/// re-reserving ODP bootstrap (England & Verrall / Merz & Wüthrich;
/// mirrors `mages/ChainLadder`'s `CDR(BootChainLadder)`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OdpCdrBootstrapResult {
    /// Initial deterministic reserve $R^{(0)}$ summed across origins.
    pub base_total_reserve: f64,
    /// Initial deterministic reserve $R_i^{(0)}$ per origin.
    pub base_origin_reserves: Vec<f64>,
    /// Simulated total 1-year CDR samples: $R^{(0)} - (Y_{\text{next}} + R^{(1)})$.
    /// Negative values represent adverse 1-year development.
    pub simulated_total_cdr: Vec<f64>,
    /// Simulated 1-year CDR samples per origin (`n_origins` vectors).
    pub simulated_cdr_by_origin: Vec<Vec<f64>>,
    /// Empirical summary statistics of total 1-year CDR.
    pub summary: BootstrapSummary,
    /// Empirical summary statistics of 1-year CDR per origin.
    pub origin_summaries: Vec<BootstrapSummary>,
    /// Count of skipped iterations due to degenerate pseudo-triangles.
    pub failed_iterations: usize,
}

/// Run a 1-year Solvency II Claims Development Result (CDR) re-reserving
/// bootstrap on top of the ODP bootstrap model.
///
/// For each simulation $b$:
/// 1. Resample scaled Pearson residuals and refit pseudo-LDFs $\hat{f}_k^{(b)}$.
/// 2. Simulate the next calendar-year incremental payments $Y_{i, n-i}^{(b)}$
///    (with process variance from `options.process_distribution`) for each open
///    origin $i \ge 1$.
/// 3. Append the simulated diagonal $C_{i, n-i}^{(b)} = C_{i, n-1-i} + Y_{i, n-i}^{(b)}$
///    to the observed triangle, re-estimate volume-weighted chain-ladder LDFs
///    on the extended 1-year-later triangle, and compute the updated reserve $R_i^{(1,b)}$.
/// 4. Record $\text{CDR}_i^{(b)}(1) = R_i^{(0)} - (Y_{i, n-i}^{(b)} + R_i^{(1,b)})$.
pub fn bootstrap_odp_cdr(
    triangle: &TriangleView<'_>,
    tail: f64,
    options: OdpBootstrapOptions,
) -> Result<OdpCdrBootstrapResult, ValidationError> {
    if options.iterations == 0 {
        return Err(ValidationError::InvalidIterations);
    }

    let fit = fit_odp(triangle, tail, options.monotonicity_policy)?;
    let (pool, _) =
        build_adjusted_residual_pool(&fit, options.residual_adjustment, options.center_residuals)?;

    let n = fit.n_origins;
    let m = n.saturating_sub(1);
    let mut rng = DeterministicRng::from_seed(options.seed);
    let mut sampler = FastIndexSampler::new(pool.len());

    let mut simulated_total_cdr = Vec::with_capacity(options.iterations);
    let mut simulated_cdr_by_origin: Vec<Vec<f64>> = (0..n)
        .map(|_| Vec::with_capacity(options.iterations))
        .collect();
    let mut scratch = OdpBootstrapScratch::for_fit(&fit);
    let inv_phi = scratch.inv_phi;
    let mut ext_ldfs = vec![1.0; m];
    let mut ext_cdfs = vec![tail; n];
    let mut next_pay = vec![0.0; n];
    let mut next_cum_diag = vec![0.0; n];
    let mut failed_iterations = 0usize;

    // Precompute observed column sums once: each 1-year CDR draw only appends a
    // single simulated diagonal cell (i = n - 1 - k, j = k + 1) to link k.
    let mut obs_den = vec![0.0; m];
    let mut obs_num = vec![0.0; m];
    for i in 0..m {
        let row = &triangle.data[i * n..i * n + (n - i)];
        for k in 0..(n - 1 - i) {
            let c0 = row[k];
            let c1 = row[k + 1];
            if c0.is_finite() && c1.is_finite() {
                obs_den[k] += c0;
                obs_num[k] += c1;
            }
        }
    }
    let obs_diag: Vec<f64> = (0..n).map(|i| triangle.get(i, n - 1 - i)).collect();
    let r1_origin0 = obs_diag[0] * (tail - 1.0);

    for _ in 0..options.iterations {
        // 1. Fused single-pass resampling + delta accumulation + LDF refit.
        let outcome = refit_pseudo_ldfs_from_pool(&pool, &mut sampler, &mut rng, &mut scratch);
        if !outcome.valid {
            failed_iterations += 1;
            continue;
        }

        // 2. Simulate next calendar diagonal (i + j == n) and update extended LDFs in O(n).
        let mut valid_ext = true;
        for i in 1..n {
            let k = n - 1 - i; // link k -> k + 1 (j_next = k + 1 = n - i)
            let c_prev = obs_diag[i];
            let expected_inc = c_prev * (scratch.pseudo_ldfs[k] - 1.0);
            let sim_inc = sample_process_increment_fast(
                expected_inc,
                fit.dispersion,
                inv_phi,
                options.process_distribution,
                &mut rng,
            );
            let c_next = c_prev + sim_inc;
            next_pay[i] = sim_inc;
            next_cum_diag[i] = c_next;

            let den = obs_den[k] + c_prev;
            let num = obs_num[k] + c_next;
            if !den.is_finite() || den <= 0.0 || !num.is_finite() || num <= 0.0 {
                valid_ext = false;
                break;
            }
            ext_ldfs[k] = num / den;
        }
        if !valid_ext {
            failed_iterations += 1;
            continue;
        }

        // 3. Compound extended CDFs right-to-left in O(n).
        ext_cdfs[m] = tail;
        for k in (0..m).rev() {
            ext_cdfs[k] = ext_ldfs[k] * ext_cdfs[k + 1];
        }

        // 4. Compute updated reserve R^{(1)} and 1-year CDR = R^{(0)} - (Y_next + R^{(1)}) in O(n).
        let cdr0 = fit.origin_reserves[0] - r1_origin0;
        if !cdr0.is_finite() {
            failed_iterations += 1;
            continue;
        }
        scratch.origin_reserves[0] = cdr0;
        let mut total_cdr = cdr0;
        let mut ok = true;
        for i in 1..n {
            let j_next = n - i;
            let c_next = next_cum_diag[i];
            let r1 = c_next * (ext_cdfs[j_next] - 1.0);
            let cdr_i = fit.origin_reserves[i] - (next_pay[i] + r1);
            if !cdr_i.is_finite() {
                ok = false;
                break;
            }
            scratch.origin_reserves[i] = cdr_i;
            total_cdr += cdr_i;
        }
        if !ok || !total_cdr.is_finite() {
            failed_iterations += 1;
            continue;
        }
        simulated_total_cdr.push(total_cdr);
        for (origin_vec, &r) in simulated_cdr_by_origin
            .iter_mut()
            .zip(scratch.origin_reserves.iter())
        {
            origin_vec.push(r);
        }
    }

    if simulated_total_cdr.is_empty() {
        return Err(ValidationError::NotFinite);
    }

    let mut summary_buf = Vec::with_capacity(simulated_total_cdr.len());
    let summary = summarize_with_scratch(&simulated_total_cdr, &mut summary_buf, true)?;
    let mut origin_summaries = Vec::with_capacity(n);
    for series in &simulated_cdr_by_origin {
        origin_summaries.push(summarize_with_scratch(series, &mut summary_buf, true)?);
    }

    Ok(OdpCdrBootstrapResult {
        base_total_reserve: fit.total_reserve,
        base_origin_reserves: fit.origin_reserves,
        simulated_total_cdr,
        simulated_cdr_by_origin,
        summary,
        origin_summaries,
        failed_iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::genins;
    use crate::deterministic::chain_ladder::ChainLadderEngine;
    use crate::deterministic::ldf::DevelopmentFactorEngine;
    use crate::types::Triangle;

    fn triangle_4x4() -> Triangle {
        #[rustfmt::skip]
        let data = vec![
            1000.0, 1500.0, 1700.0, 1800.0,
            1200.0, 1800.0, 2000.0, f64::NAN,
            1500.0, 2100.0, f64::NAN, f64::NAN,
            1800.0, f64::NAN, f64::NAN, f64::NAN,
        ];
        Triangle::new(4, 4, data)
    }

    #[test]
    fn bootstrap_is_deterministic_for_same_seed() {
        let triangle = triangle_4x4();
        let config = BootstrapConfig {
            iterations: 50,
            seed: 42,
        };

        let first = bootstrap(&triangle.view(), 1.0, config).expect("bootstrap");
        let second = bootstrap(&triangle.view(), 1.0, config).expect("bootstrap");

        assert_eq!(first.simulated_totals, second.simulated_totals);
        assert_eq!(first.summary, second.summary);
    }

    #[test]
    fn bootstrap_returns_ordered_quantiles() {
        let triangle = triangle_4x4();
        let summary = bootstrap_summary(
            &triangle.view(),
            1.0,
            BootstrapConfig {
                iterations: 100,
                seed: 7,
            },
        )
        .expect("summary");

        assert!(summary.mean >= 0.0);
        assert!(summary.std_dev >= 0.0);
        assert!(summary.p50 <= summary.p75);
        assert!(summary.p75 <= summary.p90);
        assert!(summary.p90 <= summary.p95);
        assert!(summary.p95 <= summary.p99);
    }

    #[test]
    fn bootstrap_rejects_zero_iterations() {
        let triangle = triangle_4x4();
        assert_eq!(
            bootstrap(
                &triangle.view(),
                1.0,
                BootstrapConfig {
                    iterations: 0,
                    seed: 1
                }
            ),
            Err(ValidationError::ShapeMismatch)
        );
    }

    #[test]
    fn bootstrap_base_reserve_matches_chain_ladder() {
        let triangle = triangle_4x4();
        let result = bootstrap(
            &triangle.view(),
            1.0,
            BootstrapConfig {
                iterations: 10,
                seed: 9,
            },
        )
        .expect("bootstrap");

        let factors =
            DevelopmentFactorEngine::volume_weighted(&triangle.view(), 1.0).expect("factors");
        let projection =
            ChainLadderEngine::project(&triangle.view(), &factors).expect("projection");
        assert!((result.base_total_reserve - projection.total_reserve).abs() < 1e-9);
    }

    #[test]
    fn odp_fit_leverages_sum_to_rank_and_saturate_corners() {
        let triangle = triangle_4x4();
        let fit = fit_odp(&triangle.view(), 1.0, MonotonicityPolicy::Strict).expect("fit");

        assert_eq!(fit.n_observed, 10);
        assert_eq!(fit.rank, 7);
        assert_eq!(fit.degrees_of_freedom, 3);
        let h_sum: f64 = fit.residuals.iter().map(|r| r.leverage).sum();
        assert!((h_sum - 7.0).abs() < 1e-10, "sum(h_ii)={h_sum}");
        let saturated_corners: Vec<(usize, usize)> = fit
            .residuals
            .iter()
            .filter(|r| r.saturated)
            .map(|r| (r.origin, r.dev))
            .collect();
        assert_eq!(saturated_corners, vec![(0, 3), (3, 0)]);
    }

    #[test]
    fn odp_fit_tridiagonal_matches_general_solver_to_machine_precision() {
        let tri = genins();
        let fit = fit_odp(&tri.view(), 1.05, MonotonicityPolicy::Strict).expect("fast fit");
        let general = solve_odp_general(
            fit.n_origins,
            fit.tail,
            &fit.fitted_cumulative,
            &fit.fitted_incremental,
            &fit.observed_incremental,
        )
        .expect("general fit");

        assert_eq!(fit.residuals.len(), general.residuals.len());
        for (a, b) in fit.residuals.iter().zip(general.residuals.iter()) {
            assert_eq!((a.origin, a.dev), (b.origin, b.dev));
            assert!((a.leverage - b.leverage).abs() < 1e-12);
            assert!((a.raw_pearson_residual - b.raw_pearson_residual).abs() < 1e-12);
        }
        for i in 0..fit.n_origins {
            let tol = 1e-10 * (1.0 + general.analytic_origin_prediction_error[i]);
            assert!(
                (fit.analytic_origin_param_se[i] - general.analytic_origin_param_se[i]).abs() < tol
            );
            assert!(
                (fit.analytic_origin_prediction_error[i]
                    - general.analytic_origin_prediction_error[i])
                    .abs()
                    < tol
            );
        }
        assert!(
            (fit.analytic_total_param_se - general.analytic_total_param_se).abs()
                < 1e-10 * general.analytic_total_param_se
        );
    }

    #[test]
    fn odp_fit_handles_flat_ldf_via_general_fallback() {
        // Last development link has LDF == 1.0 (zero incremental fitted mass in col 3).
        #[rustfmt::skip]
        let tri = Triangle::new(4, 4, vec![
            1000.0, 1500.0, 1700.0, 1700.0,
            1200.0, 1800.0, 2000.0, f64::NAN,
            1500.0, 2100.0, f64::NAN, f64::NAN,
            1800.0, f64::NAN, f64::NAN, f64::NAN,
        ]);
        let fit = fit_odp(&tri.view(), 1.0, MonotonicityPolicy::Strict).expect("fallback fit");
        assert_eq!(fit.n_observed, 9);
        assert_eq!(fit.rank, 6);
        assert_eq!(fit.degrees_of_freedom, 3);
        let h_sum: f64 = fit.residuals.iter().map(|r| r.leverage).sum();
        assert!((h_sum - 6.0).abs() < 1e-10, "sum(h_ii)={h_sum}");
    }

    #[test]
    fn ziggurat_normal_and_fast_index_samplers_match_theoretical_distributions() {
        let mut rng = DeterministicRng::from_seed(1234567);
        let n = 200_000usize;
        let mut sum = 0.0;
        let mut s2 = 0.0;
        let mut s3 = 0.0;
        let mut s4 = 0.0;
        let mut tail_hits = 0usize;
        for _ in 0..n {
            let z = rng.next_standard_normal();
            sum += z;
            let z2 = z * z;
            s2 += z2;
            s3 += z2 * z;
            s4 += z2 * z2;
            if z.abs() >= ZIG_NORM_R {
                tail_hits += 1;
            }
        }
        let nf = n as f64;
        let mean = sum / nf;
        let var = s2 / nf - mean * mean;
        let skew = s3 / nf;
        let kurt = s4 / nf;
        assert!(mean.abs() < 0.01, "mean={mean}");
        assert!((var - 1.0).abs() < 0.015, "var={var}");
        assert!(skew.abs() < 0.03, "skew={skew}");
        assert!((kurt - 3.0).abs() < 0.08, "kurt={kurt}");
        // P(|Z| >= 3.654153) ≈ 2.58e-4 => ~52 hits in 200k.
        assert!((20..=100).contains(&tail_hits), "tail_hits={tail_hits}");

        // FastIndexSampler uniformity on bound = 53 (16-bit Lemire) and 1500 (32-bit Lemire).
        let mut counts53 = [0usize; 53];
        let mut s53 = FastIndexSampler::new(53);
        for _ in 0..53_000 {
            counts53[s53.next(&mut rng)] += 1;
        }
        let chi2: f64 = counts53
            .iter()
            .map(|&c| {
                let d = c as f64 - 1000.0;
                d * d / 1000.0
            })
            .sum();
        assert!(chi2 < 100.0, "chi2(df=52)={chi2}");

        let mut s1500 = FastIndexSampler::new(1500);
        for _ in 0..5_000 {
            assert!(s1500.next(&mut rng) < 1500);
        }
    }

    #[test]
    fn gamma_and_poisson_samplers_match_theoretical_moments() {
        let mut rng = DeterministicRng::from_seed(42069);
        let n = 40_000usize;

        for &(shape, scale) in &[(0.4, 2.5), (3.5, 10.0), (25.0, 50.0)] {
            let samples: Vec<f64> = (0..n).map(|_| rng.sample_gamma(shape, scale)).collect();
            let mean = samples.iter().sum::<f64>() / n as f64;
            let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            let expected_mean = shape * scale;
            let expected_var = shape * scale * scale;
            assert!(
                (mean - expected_mean).abs() / expected_mean < 0.03,
                "gamma({shape},{scale}) mean={mean} expected={expected_mean}"
            );
            assert!(
                (var - expected_var).abs() / expected_var < 0.08,
                "gamma({shape},{scale}) var={var} expected={expected_var}"
            );
        }

        for &lambda in &[0.7, 9.5, 10.0, 12.0, 20.0, 45.0, 300.0] {
            let samples: Vec<f64> = (0..n).map(|_| rng.sample_poisson(lambda)).collect();
            let mean = samples.iter().sum::<f64>() / n as f64;
            let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            assert!(
                (mean - lambda).abs() / lambda < 0.03,
                "poisson({lambda}) mean={mean}"
            );
            assert!(
                (var - lambda).abs() / lambda < 0.06,
                "poisson({lambda}) var={var}"
            );
        }
    }
}
