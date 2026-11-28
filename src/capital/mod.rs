//! Multi-line dependence, empirical reserve risk measures, aggregate loss convolution, and capital allocation.
//!
//! Module dependency direction inside the dependence layer is strictly
//! one-way: `correlation` (shared helpers) <- `copulas` <- `dependence`.

pub mod aggregate_loss;
pub mod capital_allocation;
pub mod copulas;
pub mod correlation;
pub mod dependence;
pub mod reserve_distribution;
