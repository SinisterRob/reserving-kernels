#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "wasm"), forbid(unsafe_code))]
#![warn(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::neg_cmp_op_on_partial_ord)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::doc_overindented_list_items)]

//! Pure-Rust base kernel set for actuarial loss-reserving, stochastic
//! variance decomposition, portfolio dependence, and severity layering.
//!
//! Inputs are loss-development triangles laid out row-major as `f64` slices
//! (cumulative paid or incurred), with `NaN` marking unobserved cells. All
//! compute kernels operate on `f64` in hot paths and assume validated inputs
//! (see [`validate`]).
//!
//! Core modules:
//!
//! * [`ldf`] & [`tail`] — age-to-age development factors and parametric tail
//!   extrapolation (Bondy, Generalized Bondy, Exponential, Sherman Inverse
//!   Power, Weibull, and Mack 1999 tail variance).
//! * [`chain_ladder`], [`bf`], [`benktander`], [`cape_cod`],
//!   [`generalized_cape_cod`], [`berquist_sherman`], [`brosius`], and
//!   [`munich_chain_ladder`] — deterministic point reserving and operational
//!   adjustment methods.
//! * [`mack`], [`mack_bf`], [`merz_wuthrich`], [`paid_incurred_chain`],
//!   [`clark_mle`], [`ptf_regression`], [`glm_reserving`], and
//!   [`bootstrap_odp`] — stochastic reserving, 1-year Solvency II CDR MSEP,
//!   parametric MLE, log-linear trend families, IRLS GLMs, and ODP bootstrap.
//! * [`dependence`], [`reserve_distribution`], [`aggregate_loss`], and
//!   [`capital_allocation`] — Iman–Conover (Mildenhall 2006) rank reordering,
//!   synchronized multi-line ODP bootstrap, Panjer/FFT compound distributions,
//!   CAS Monograph No. 15 coherent risk measures, Euler allocation, and
//!   Myers–Read default-put allocation.
//! * [`severity`] & [`excess_layers`] — parametric LEV/ILFs, lognormal layer
//!   expectations, and Monte Carlo layer simulation.
//! * [`diagnostics`], [`selection`], [`reserve_rollforward`], & [`exhibit`] —
//!   Mack (1994) assumption tests, holdout backtesting, roll-forwards, and
//!   serializable exhibits.

pub mod aggregate;
pub mod aggregate_loss;
pub mod apriori;
pub mod benktander;
pub mod berquist_sherman;
pub mod bf;
pub mod bootstrap_odp;
pub mod brosius;
pub mod cape_cod;
pub mod capital_allocation;
pub mod case_outstanding;
pub mod chain_ladder;
pub mod clark_mle;
pub mod count_development;
pub mod datasets;
pub mod dependence;
pub mod diagnostics;
pub mod excess_layers;
pub mod exhibit;
pub mod generalized_cape_cod;
pub mod glm_reserving;
pub mod incremental_additive;
pub mod large_loss;
pub mod ldf;
pub(crate) mod linalg;
pub mod mack;
pub mod mack_bf;
pub mod merz_wuthrich;
pub mod multivariate_cl;
pub mod munich_chain_ladder;
pub mod onlevel;
pub mod paid_incurred_chain;
pub mod ptf_regression;
pub mod ragged;
pub mod reserve_distribution;
pub mod reserve_rollforward;
pub mod selection;
pub mod severity;
pub mod tail;
pub mod trend;
pub mod triangle;
pub mod triangle_algebra;
pub mod types;
pub mod validate;

#[cfg(feature = "wasm")]
pub mod wasm_exports;

pub use types::{Triangle, TriangleView};

/// Convenient re-exports of the most commonly used reserving types and engines.
pub mod prelude {
    pub use crate::aggregate_loss::{
        AggregateLossDistribution, FrequencyDistribution, fft_aggregate_loss, panjer_recursion,
    };
    pub use crate::benktander::{BenktanderEngine, BenktanderProjection};
    pub use crate::bf::{BornhuetterFergusonEngine, BornhuetterFergusonProjection};
    pub use crate::bootstrap_odp::{
        OdpBootstrapOptions, OdpBootstrapResult, OdpCdrBootstrapResult, OdpFit,
        ProcessDistribution, ResidualAdjustment, bootstrap_odp, bootstrap_odp_cdr, fit_odp,
    };
    pub use crate::cape_cod::{CapeCodEngine, CapeCodProjection};
    pub use crate::capital_allocation::{
        EulerAllocationResult, MyersReadAllocationResult, euler_rvar_allocation,
        euler_sd_allocation, euler_tvar_allocation, myers_read_allocation,
        myers_read_from_default_ratio,
    };
    pub use crate::case_outstanding::{
        CaseOutstandingOptions, CaseOutstandingProjection, CaseRatioAverage,
        project_case_outstanding, project_from_paid_incurred,
    };
    pub use crate::chain_ladder::{ChainLadderEngine, ChainLadderProjection};
    pub use crate::clark_mle::{ClarkMleResult, ClarkOptions, GrowthCurve};
    pub use crate::datasets::{FriedlandAutoDataset, PaidIncurredDataset};
    pub use crate::dependence::{CorrelationType, ImanConoverResult, iman_conover};
    pub use crate::diagnostics::triangle_diagnostics::TriangleDiagnostics;
    pub use crate::glm_reserving::{
        GlmDesign, GlmFamily, GlmOptions, GlmReservingResult, TweediePowerSelection, fit_glm,
        select_tweedie_power,
    };
    pub use crate::incremental_additive::{
        IncrementalAdditiveOptions, IncrementalAdditiveProjection, IncrementalRatioAverage,
        project_incremental_additive,
    };
    pub use crate::ldf::{DevelopmentFactorEngine, DevelopmentFactors};
    pub use crate::mack::{Mack, MackOptions, MackTailSpec};
    pub use crate::mack_bf::{BfPayoutEstimator, MackBfOptions, MackBfResult};
    pub use crate::merz_wuthrich::MerzWuthrichResult;
    pub use crate::multivariate_cl::{
        MultivariateClOptions, MultivariateClResult, MultivariateModel, analyze_multivariate_cl,
    };
    pub use crate::paid_incurred_chain::{PaidIncurredChainResult, PicOptions};
    pub use crate::ptf_regression::{
        AccidentSpec, CalendarSpec, DevelopmentSpec, LogNormalMeanMode, PtfOptions, PtfResult,
    };
    pub use crate::reserve_distribution::{rvar, tvar_coherent, var_least};
    pub use crate::severity::SeverityDistribution;
    pub use crate::types::{Triangle, TriangleView};
    pub use crate::validate::{MonotonicityPolicy, ValidationError};
}
