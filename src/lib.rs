#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "wasm"), forbid(unsafe_code))]
#![warn(missing_docs)]
#![allow(clippy::needless_range_loop)]
// Intentional NaN-rejecting numeric guards such as `!(x > 0.0)`.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod capital;
pub mod datasets;
pub mod deterministic;
pub mod diagnostics;
pub mod excess;
pub(crate) mod linalg;
pub mod preprocessing;
pub mod stochastic;
pub mod subannual;
mod triangle_ops;
pub mod types;
pub mod validate;

#[cfg(feature = "wasm")]
#[doc(hidden)]
pub mod wasm_exports;

pub use types::{Triangle, TriangleView};

/// Declarative constructor for a validated square cumulative [`Triangle`] from
/// upper-triangle rows (coercing integer or float literals to `f64`).
///
/// Returns `Result<Triangle, ValidationError>`.
///
/// # Example
/// ```rust
/// use reserving_kernels::{prelude::*, triangle};
///
/// let tri = triangle![
///     [100, 160, 195, 205],
///     [120, 190, 232],
///     [135, 218],
///     [150],
/// ]?;
/// assert_eq!(tri.n(), 4);
/// assert_eq!(tri.latest_diagonal(), vec![205.0, 232.0, 218.0, 150.0]);
/// # Ok::<(), ValidationError>(())
/// ```
#[macro_export]
macro_rules! triangle {
    ( $( [ $( $val:expr ),+ $(,)? ] ),+ $(,)? ) => {{
        $crate::types::Triangle::try_from_upper_rows(&[
            $( &[ $( ($val) as f64 ),+ ] as &[f64] ),+
        ])
    }};
}

/// Convenient re-exports of the most commonly used reserving types and engines.
pub mod prelude {
    pub use crate::capital::aggregate_loss::{
        AggregateLossDistribution, FrequencyDistribution, fft_aggregate_loss, panjer_recursion,
    };
    pub use crate::capital::capital_allocation::{
        EulerAllocationResult, MyersReadAllocationResult, euler_rvar_allocation,
        euler_sd_allocation, euler_tvar_allocation, myers_read_allocation,
        myers_read_from_default_ratio,
    };
    pub use crate::capital::dependence::{
        CopulaSampleResult, CorrelationType, GumbelHac, GumbelHacCluster, ImanConoverResult,
        apply_copula_reordering, clayton_copula_sample, clayton_lower_tail_dependence,
        clayton_theta_from_tau, gumbel_copula_sample, gumbel_theta_from_tau,
        gumbel_upper_tail_dependence, iman_conover, repair_correlation_matrix, student_t_cdf,
        student_t_copula_sample, student_t_tail_dependence,
    };
    pub use crate::capital::reserve_distribution::{
        crps_empirical, interval_score, pit_empirical, rvar, tvar_coherent, var_least,
    };
    pub use crate::datasets::{FriedlandAutoDataset, PaidIncurredDataset};
    pub use crate::deterministic::benktander::{BenktanderEngine, BenktanderProjection};
    pub use crate::deterministic::bf::{BornhuetterFergusonEngine, BornhuetterFergusonProjection};
    pub use crate::deterministic::cape_cod::{CapeCodEngine, CapeCodProjection};
    pub use crate::deterministic::case_outstanding::{
        CaseOutstandingOptions, CaseOutstandingProjection, CaseRatioAverage,
        project_case_outstanding, project_from_paid_incurred,
    };
    pub use crate::deterministic::chain_ladder::{ChainLadderEngine, ChainLadderProjection};
    pub use crate::deterministic::credibility_cl::{
        CollectiveCredibilityParams, CredibilityClPrior, CredibilityClResult,
    };
    pub use crate::deterministic::double_chain_ladder::DoubleChainLadderResult;
    pub use crate::deterministic::freq_sev::{
        AdlerKlineOptions, AdlerKlineResult, CountProjectionMode, WrightOperationalTimeOptions,
        WrightOperationalTimeResult, adler_kline_reserve, wright_operational_time_reserve,
    };
    pub use crate::deterministic::incremental_additive::{
        IncrementalAdditiveOptions, IncrementalAdditiveProjection, IncrementalRatioAverage,
        project_incremental_additive,
    };
    pub use crate::deterministic::lae::{
        AlaeRatioTriangleResult, AlaeRatioWeighting, ClaimCountYear, ClarkAlaeOptions,
        ClarkAlaeResult, JohnsonUlaeResult, JohnsonUlaeWeights, UlaeOptions, UlaeRatioResult,
        UlaeRatioSelection, alae_ratio_triangle, clark_alae_mle, expected_calendar_emergence,
        ulae_johnson_count_based, ulae_kittel, ulae_mango_allen, ulae_paid_to_paid,
    };
    pub use crate::deterministic::ldf::{DevelopmentFactorEngine, DevelopmentFactors};
    pub use crate::deterministic::murphy::{MurphyLinkFit, MurphyOptions, MurphyResult};
    pub use crate::deterministic::schnieper::SchnieperResult;
    pub use crate::deterministic::separation::{
        InflationProjection, SeparationMethod, SeparationResult,
    };
    pub use crate::diagnostics::triangle_diagnostics::TriangleDiagnostics;
    pub use crate::excess::excess_development::{
        ExcessDevelopmentFactors, ExcessProjection, LayerDevelopmentFactors,
    };
    pub use crate::excess::severity::SeverityDistribution;
    pub use crate::stochastic::bayesian_reserving::{
        BayesianOdpOptions, BayesianOdpResult, BivariateCclResult, MeyersModel, MeyersOptions,
        MeyersResult, bayesian_odp_laplace, meyers_bivariate_ccl, meyers_laplace_snis,
    };
    pub use crate::stochastic::bootstrap_odp::{
        OdpBootstrapOptions, OdpBootstrapResult, OdpCdrBootstrapResult, OdpFit,
        ProcessDistribution, ResidualAdjustment, bootstrap_odp, bootstrap_odp_cdr, fit_odp,
    };
    pub use crate::stochastic::clark_mle::{ClarkMleResult, ClarkOptions, GrowthCurve};
    pub use crate::stochastic::glm_reserving::{
        GlmDesign, GlmFamily, GlmOptions, GlmReservingResult, TweediePowerSelection, fit_glm,
        select_tweedie_power,
    };
    pub use crate::stochastic::hertig::{HertigOptions, HertigResult};
    pub use crate::stochastic::kalman_reserving::{KalmanReservingOptions, KalmanReservingResult};
    pub use crate::stochastic::mack::{Mack, MackOptions, MackTailSpec};
    pub use crate::stochastic::mack_bf::{BfPayoutEstimator, MackBfOptions, MackBfResult};
    pub use crate::stochastic::merz_wuthrich::MerzWuthrichResult;
    pub use crate::stochastic::micro_reserving::{
        ClaimPayment, ClaimRecord, MicroReservingModel, MicroReservingResult,
    };
    pub use crate::stochastic::multivariate_cl::{
        MultivariateClOptions, MultivariateClResult, MultivariateModel, analyze_multivariate_cl,
    };
    pub use crate::stochastic::multivariate_mack::MultivariateMackResult;
    pub use crate::stochastic::paid_incurred_chain::{PaidIncurredChainResult, PicOptions};
    pub use crate::stochastic::ptf_regression::{
        AccidentSpec, CalendarSpec, DevelopmentSpec, LogNormalMeanMode, PtfOptions, PtfResult,
    };
    pub use crate::stochastic::reserve_runoff::{
        ChainLadderRollForward, MackInnovation, MackRunoffOptions, OneYearRunoffResult,
        RollForwardStep, mack_one_year_runoff, mack_one_year_runoff_with_options,
        odp_one_year_runoff,
    };
    pub use crate::subannual::{
        AsymmetricAgeGrid, AsymmetricChainLadderProjection, AsymmetricOdpBootstrapResult,
        AsymmetricOdpFit, CalendarDiagonalMap, ClarkExactAgeOptions, ClarkExactAgeResult,
        EquilibriumIbnrCheckResult, InterimBfProjection, InterimCapeCodProjection,
        InterpolatedPattern, InversePowerInterpResult, StubExposureScaleResult,
        StubExtrapolationResult, SubannualInterpMethod, SubannualProductParityResult,
        asymmetric_age_grid, asymmetric_chain_ladder, asymmetric_odp_bootstrap,
        calendar_diagonal_index, clark_mle_exact_age, equilibrium_ibnr_check, fit_asymmetric_odp,
        interim_bf, interim_cape_cod, interp_exponential, interp_inverse_power, interp_ivf12,
        interp_linear_cdf, interp_linear_pct_reported, interp_spliced_weibull, interp_swimon,
        interpolate_pattern, stub_average_age, stub_exposure_scale, stub_natural_log, stub_plus12,
        stub_power_ratio, subannual_product_parity,
    };
    pub use crate::types::{Triangle, TriangleView};
    pub use crate::validate::{MonotonicityPolicy, ValidationError};
}
