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
//! * [`chain_ladder`], [`bf`], [`benktander`], [`cape_cod`], [`brosius`], and
//!   [`munich_chain_ladder`] — deterministic point reserving methods.
//! * [`mack`] & [`bootstrap_odp`] — Mack (1993, 1999) distribution-free
//!   standard errors and England–Verrall / Shapland ODP GLM + bootstrap.
//! * [`dependence`] & [`reserve_distribution`] — Iman–Conover (Mildenhall
//!   2006) rank reordering, synchronized multi-line ODP bootstrap, and CAS
//!   Monograph No. 15 coherent VaR/TVaR/RVaR risk measures.
//! * [`severity`] & [`excess_layers`] — parametric LEV/ILFs, lognormal layer
//!   expectations, and Monte Carlo layer simulation.
//! * [`diagnostics`], [`selection`], [`reserve_rollforward`], & [`exhibit`] —
//!   Mack (1994) assumption tests, holdout backtesting, roll-forwards, and
//!   serializable exhibits.

pub mod aggregate;
pub mod apriori;
pub mod benktander;
pub mod bf;
pub mod bootstrap_odp;
pub mod brosius;
pub mod cape_cod;
pub mod chain_ladder;
pub mod count_development;
pub mod dependence;
pub mod diagnostics;
pub mod excess_layers;
pub mod exhibit;
pub mod large_loss;
pub mod ldf;
pub mod mack;
pub mod munich_chain_ladder;
pub mod onlevel;
pub mod reserve_distribution;
pub mod reserve_rollforward;
pub mod selection;
pub mod severity;
pub mod tail;
pub mod trend;
pub mod triangle;
pub mod types;
pub mod validate;

#[cfg(feature = "wasm")]
pub mod wasm_exports;

pub use types::{Triangle, TriangleView};
