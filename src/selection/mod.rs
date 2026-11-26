//! Selection and backtesting kernels for deterministic factor choices.

pub mod backtest;
pub mod scoring;

/// Candidate LDF selector family for backtesting and scoring.
#[derive(Debug, Clone, PartialEq)]
pub enum CandidateSelector {
    /// Volume-weighted all-year link ratios.
    VolumeWeighted,
    /// Simple-average all-year link ratios.
    SimpleAverage,
    /// Median all-year link ratios.
    Median,
    /// Ex-high-low trimmed-average all-year link ratios.
    ExHiLo,
    /// Volume-weighted using the most recent `n` origins.
    VolumeWeightedLastN(usize),
    /// Simple-average using the most recent `n` origins.
    SimpleAverageLastN(usize),
    /// Caller-supplied factor vector.
    Manual(Vec<f64>),
}
