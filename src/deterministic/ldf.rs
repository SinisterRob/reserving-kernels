//! Age-to-age development factors and CDF construction.
//!
//! This module is the Section 2 bridge:
//!
//! ```text
//! cumulative triangle -> selected age-to-age LDFs -> CDFs -> maturity %
//! ```
//!
//! Three layers, all in this file:
//!
//! * **L0 columnar kernels** — operate on flat row-major slices and
//!   caller-provided output buffers (e.g. [`volume_weighted_ldfs`],
//!   [`cdfs_from_ldfs`]). Used by the WASM hot path and the columnar Mack /
//!   Brosius engines.
//! * **L1 view convenience** — take a [`TriangleView`] and return owned
//!   `Vec<f64>` (e.g. [`volume_weighted`], [`simple_average`],
//!   [`cdf_from_ldf`]). Used by tests, the unchecked WASM path, and the
//!   internal raw projection helpers.
//! * **L2 service / engine** — validated entry points returning a typed
//!   [`DevelopmentFactors`] result (see [`DevelopmentFactorEngine`]).
//!
//! Input is a row-major cumulative triangle; unobserved cells are `NaN`.
//! Each LDF vector has length `n − 1`; the trailing tail factor (dev `n − 1`
//! → ultimate) is supplied by the caller.

use crate::deterministic::averaging;
use crate::error::ReservingError;
use crate::types::TriangleView;
use crate::validate::{
    MonotonicityPolicy, validate_monotone_cumulative, validate_positive_finite_slice,
    validate_tail_with_policy, validate_triangle,
};

// ===========================================================================
// Types
// ===========================================================================

/// Selected development factor family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LdfSelectionMethod {
    /// Volume-weighted link ratios.
    VolumeWeighted,
    /// Simple average link ratios.
    SimpleAverage,
    /// Median link ratios.
    Median,
    /// Exclude the single highest and lowest individual ratio when possible.
    ExHiLo,
    /// Volume-weighted link ratios restricted to the most recent `last_n` origins.
    LastNVolumeWeighted {
        /// Number of most recent origins to include.
        last_n: usize,
    },
    /// Simple-average link ratios restricted to the most recent `last_n` origins.
    LastNSimpleAverage {
        /// Number of most recent origins to include.
        last_n: usize,
    },
    /// Geometric average of positive individual link ratios.
    Geometric,
    /// Caller-selected link ratios. Carries no factor values, so computed
    /// selectors (e.g. [`DevelopmentFactorEngine::select_with_policy`]) reject
    /// it; supply explicit factors via [`DevelopmentFactorEngine::from_selected_ldfs`].
    Selected,
}

/// Selected age-to-age factors and cumulative development factors.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct DevelopmentFactors {
    /// Selected age-to-age LDFs, length `n_dev - 1`.
    pub ldfs: Vec<f64>,
    /// CDFs by development age, length `n_dev`.
    pub cdfs: Vec<f64>,
    /// Tail factor from the final development age to ultimate.
    pub tail: f64,
}

impl DevelopmentFactors {
    /// Percent mature/reported at each development age, `1 / cdf[j]`.
    pub fn maturity_percentages(&self) -> Vec<f64> {
        self.cdfs
            .iter()
            .map(|&cdf| if cdf > 0.0 { 1.0 / cdf } else { f64::NAN })
            .collect()
    }
}

/// Numerator/denominator trail for one selected link ratio.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct LinkRatioDiagnostic {
    /// Development age `j` for the `j -> j + 1` link.
    pub dev: usize,
    /// Sum of observed `C[i, j + 1]`.
    pub numerator: f64,
    /// Sum of observed `C[i, j]`.
    pub denominator: f64,
    /// Selected LDF, `numerator / denominator`.
    pub selected_ldf: f64,
    /// Number of origin rows included in the link.
    pub included_rows: usize,
}

// ===========================================================================
// L0 — columnar kernels (flat slices + output buffers)
// ===========================================================================

/// Return whether cell `(i, j)` is observed under right-triangle geometry.
#[inline]
pub const fn is_observed_right_triangle(i: usize, j: usize, n_dev: usize) -> bool {
    i + j < n_dev
}

/// Volume-weighted age-to-age LDF for one development age.
///
/// Only origin rows where both `dev` and `dev + 1` are observed are included.
/// Returns `ShapeMismatch` when `cum.len() != n_origin * n_dev` or
/// `dev + 1 >= n_dev`. A zero denominator yields `Ok(NaN)`.
#[inline]
pub fn volume_weighted_ldf(
    cum: &[f64],
    n_origin: usize,
    n_dev: usize,
    dev: usize,
) -> Result<f64, ReservingError> {
    volume_weighted_link_diagnostic(cum, n_origin, n_dev, dev).map(|d| d.selected_ldf)
}

fn diagnostic_from_sums(
    dev: usize,
    numerator: f64,
    denominator: f64,
    included_rows: usize,
) -> LinkRatioDiagnostic {
    let selected_ldf = if denominator > 0.0 {
        numerator / denominator
    } else {
        f64::NAN
    };
    LinkRatioDiagnostic {
        dev,
        numerator,
        denominator,
        selected_ldf,
        included_rows,
    }
}

/// Diagnostic numerator/denominator trail for one volume-weighted LDF.
///
/// Returns `ShapeMismatch` when `cum.len() != n_origin * n_dev` or
/// `dev + 1 >= n_dev`.
pub fn volume_weighted_link_diagnostic(
    cum: &[f64],
    n_origin: usize,
    n_dev: usize,
    dev: usize,
) -> Result<LinkRatioDiagnostic, ReservingError> {
    if cum.len() != n_origin * n_dev || dev + 1 >= n_dev {
        return Err(ReservingError::ShapeMismatch);
    }

    let mut numerator = 0.0;
    let mut denominator = 0.0;
    let mut included_rows = 0;

    for i in 0..n_origin {
        if is_observed_right_triangle(i, dev + 1, n_dev) {
            denominator += cum[i * n_dev + dev];
            numerator += cum[i * n_dev + dev + 1];
            included_rows += 1;
        }
    }

    Ok(diagnostic_from_sums(
        dev,
        numerator,
        denominator,
        included_rows,
    ))
}

/// Volume-weighted LDFs for all development links (row-major cache-friendly accumulation).
///
/// Panic-free raw kernel: on a shape mismatch (`cum.len() != n_origin * n_dev`
/// or `out.len() != n_dev - 1`) every element of `out` is set to `NaN`. Use
/// [`try_volume_weighted_ldfs`] to receive a typed error instead.
pub fn volume_weighted_ldfs(cum: &[f64], n_origin: usize, n_dev: usize, out: &mut [f64]) {
    if try_volume_weighted_ldfs(cum, n_origin, n_dev, out).is_err() {
        out.fill(f64::NAN);
    }
}

/// Checked variant of [`volume_weighted_ldfs`]; returns `ShapeMismatch` on
/// inconsistent slice lengths. Zero-denominator links are written as `NaN`.
pub fn try_volume_weighted_ldfs(
    cum: &[f64],
    n_origin: usize,
    n_dev: usize,
    out: &mut [f64],
) -> Result<(), ReservingError> {
    let m = n_dev.saturating_sub(1);
    if cum.len() != n_origin * n_dev || out.len() != m {
        return Err(ReservingError::ShapeMismatch);
    }
    if m == 0 {
        return Ok(());
    }
    if n_origin == 0 {
        out.fill(f64::NAN);
        return Ok(());
    }

    // Use a stack buffer for common triangle sizes (<= 64 links) to avoid heap allocation.
    let mut stack_den = [0.0_f64; 64];
    let mut heap_den;
    let den: &mut [f64] = if m <= stack_den.len() {
        &mut stack_den[..m]
    } else {
        heap_den = vec![0.0_f64; m];
        &mut heap_den[..]
    };
    out.fill(0.0);

    let max_rows = n_origin.min(m);
    for i in 0..max_rows {
        let upper = m - i;
        let row = &cum[(i * n_dev)..=(i * n_dev + upper)];
        for dev in 0..upper {
            den[dev] += row[dev];
            out[dev] += row[dev + 1];
        }
    }

    for dev in 0..m {
        out[dev] = if den[dev] > 0.0 {
            out[dev] / den[dev]
        } else {
            f64::NAN
        };
    }
    Ok(())
}

fn observed_counts_shape_ok(
    cum: &[f64],
    n_origin: usize,
    n_dev: usize,
    observed_counts: &[usize],
) -> bool {
    cum.len() == n_origin * n_dev && observed_counts.len() == n_origin
}

/// Volume-weighted age-to-age LDF for arbitrary observed row widths.
///
/// `observed_counts[i]` is the number of observed development periods in row `i`.
/// Returns `ShapeMismatch` on inconsistent slice lengths or `dev + 1 >= n_dev`.
pub fn volume_weighted_ldf_observed_counts(
    cum: &[f64],
    n_origin: usize,
    n_dev: usize,
    observed_counts: &[usize],
    dev: usize,
) -> Result<f64, ReservingError> {
    volume_weighted_link_diagnostic_observed_counts(cum, n_origin, n_dev, observed_counts, dev)
        .map(|d| d.selected_ldf)
}

/// Diagnostic numerator/denominator trail for arbitrary observed row widths.
///
/// Returns `ShapeMismatch` on inconsistent slice lengths or `dev + 1 >= n_dev`.
pub fn volume_weighted_link_diagnostic_observed_counts(
    cum: &[f64],
    n_origin: usize,
    n_dev: usize,
    observed_counts: &[usize],
    dev: usize,
) -> Result<LinkRatioDiagnostic, ReservingError> {
    if !observed_counts_shape_ok(cum, n_origin, n_dev, observed_counts) || dev + 1 >= n_dev {
        return Err(ReservingError::ShapeMismatch);
    }
    let pairs = averaging::link_pairs_observed_counts(cum, n_origin, n_dev, observed_counts, dev);
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for pair in &pairs {
        denominator += pair.current;
        numerator += pair.next;
    }
    Ok(diagnostic_from_sums(
        dev,
        numerator,
        denominator,
        pairs.len(),
    ))
}

/// Volume-weighted LDFs for arbitrary observed row widths.
///
/// Panic-free raw kernel: on a shape mismatch every element of `out` is set to
/// `NaN`. Use [`try_volume_weighted_ldfs_observed_counts`] for a typed error.
pub fn volume_weighted_ldfs_observed_counts(
    cum: &[f64],
    n_origin: usize,
    n_dev: usize,
    observed_counts: &[usize],
    out: &mut [f64],
) {
    if try_volume_weighted_ldfs_observed_counts(cum, n_origin, n_dev, observed_counts, out).is_err()
    {
        out.fill(f64::NAN);
    }
}

/// Checked variant of [`volume_weighted_ldfs_observed_counts`].
pub fn try_volume_weighted_ldfs_observed_counts(
    cum: &[f64],
    n_origin: usize,
    n_dev: usize,
    observed_counts: &[usize],
    out: &mut [f64],
) -> Result<(), ReservingError> {
    if !observed_counts_shape_ok(cum, n_origin, n_dev, observed_counts)
        || out.len() != n_dev.saturating_sub(1)
    {
        return Err(ReservingError::ShapeMismatch);
    }
    for (dev, slot) in out.iter_mut().enumerate() {
        *slot = averaging::volume_weighted(&averaging::link_pairs_observed_counts(
            cum,
            n_origin,
            n_dev,
            observed_counts,
            dev,
        ));
    }
    Ok(())
}

/// CDFs from selected LDFs using the convention:
///
/// ```text
/// cdfs[last_dev] = tail
/// cdfs[j] = ldfs[j] * cdfs[j + 1]
/// ```
///
/// Panic-free raw kernel: if `out.len() != ldfs.len() + 1` every element of
/// `out` is set to `NaN`. Use [`try_cdfs_from_ldfs`] for a typed error.
pub fn cdfs_from_ldfs(ldfs: &[f64], tail: f64, out: &mut [f64]) {
    if try_cdfs_from_ldfs(ldfs, tail, out).is_err() {
        out.fill(f64::NAN);
    }
}

/// Checked variant of [`cdfs_from_ldfs`]; returns `ShapeMismatch` when
/// `out.len() != ldfs.len() + 1`.
pub fn try_cdfs_from_ldfs(ldfs: &[f64], tail: f64, out: &mut [f64]) -> Result<(), ReservingError> {
    if out.len() != ldfs.len() + 1 {
        return Err(ReservingError::ShapeMismatch);
    }
    let last = ldfs.len();
    out[last] = tail;
    for j in (0..last).rev() {
        out[j] = ldfs[j] * out[j + 1];
    }
    Ok(())
}

/// Latest maturity percentage by origin under right-triangle geometry.
///
/// Returns `ShapeMismatch` when `cdfs.len() != n_dev` or `out.len() != n_origin`.
pub fn latest_maturity_percentages(
    n_origin: usize,
    n_dev: usize,
    cdfs: &[f64],
    out: &mut [f64],
) -> Result<(), ReservingError> {
    if cdfs.len() != n_dev || out.len() != n_origin {
        return Err(ReservingError::ShapeMismatch);
    }

    for (i, slot) in out.iter_mut().enumerate() {
        if i >= n_dev {
            *slot = f64::NAN;
            continue;
        }
        let cdf = cdfs[n_dev - 1 - i];
        *slot = if cdf > 0.0 { 1.0 / cdf } else { f64::NAN };
    }
    Ok(())
}

// ===========================================================================
// L1 — view-based selection methods (returns Vec<f64>)
// ===========================================================================

/// Apply one shared averaging kernel to every development link of `t`.
fn per_link(t: &TriangleView<'_>, average: impl Fn(&[averaging::LinkPair]) -> f64) -> Vec<f64> {
    let n = t.n_origins.min(t.n_devs);
    if n < 2 {
        return Vec::new();
    }
    (0..n - 1)
        .map(|j| average(&averaging::link_pairs(t, j)))
        .collect()
}

/// Volume-weighted all-year LDF: `f_j = Σ_i C_{i, j+1} / Σ_i C_{i, j}`
/// summing over origins where both cells are observed (`i ≤ n − j − 2`).
///
/// This is the classical chain-ladder factor. Returns a vector of length
/// `n − 1`. Links whose denominator is zero emit `NaN`; a malformed view
/// (`data.len() < n_origins * n_devs`) emits all-`NaN`.
pub fn volume_weighted(t: &TriangleView<'_>) -> Vec<f64> {
    let n = t.n_origins.min(t.n_devs);
    if n < 2 {
        return Vec::new();
    }
    let m = n - 1;
    if t.data.len() < t.n_origins * t.n_devs {
        return vec![f64::NAN; m];
    }
    let mut out = vec![0.0; m];
    let mut stack_den = [0.0_f64; 64];
    let mut heap_den;
    let den: &mut [f64] = if m <= stack_den.len() {
        &mut stack_den[..m]
    } else {
        heap_den = vec![0.0_f64; m];
        &mut heap_den[..]
    };
    for i in 0..m {
        let upper = m - i;
        let row = &t.data[(i * t.n_devs)..=(i * t.n_devs + upper)];
        for j in 0..upper {
            den[j] += row[j];
            out[j] += row[j + 1];
        }
    }
    for j in 0..m {
        out[j] = if den[j] > 0.0 {
            out[j] / den[j]
        } else {
            f64::NAN
        };
    }
    out
}

/// Simple (straight) average of individual age-to-age ratios:
/// `f_j = mean_i( C_{i, j+1} / C_{i, j} )`. Each origin contributes equal
/// weight regardless of loss size. Links with no defined ratio emit `NaN`.
pub fn simple_average(t: &TriangleView<'_>) -> Vec<f64> {
    per_link(t, averaging::simple_average)
}

/// Median of individual age-to-age ratios for each development link
/// (`NaN` for links with no defined ratio).
pub fn median(t: &TriangleView<'_>) -> Vec<f64> {
    per_link(t, averaging::median)
}

/// Ex-hi / ex-lo (trimmed mean) of individual age-to-age ratios:
/// `f_j = mean of the individual ratios after dropping the single max and
/// single min`. When fewer than 3 ratios are available for a link, falls
/// back to [`simple_average`] on that link (`NaN` with no defined ratio).
pub fn ex_hi_lo(t: &TriangleView<'_>) -> Vec<f64> {
    per_link(t, averaging::ex_hi_lo)
}

/// Volume-weighted average restricted to the last `last_n` origin years.
///
/// `f_j = Σ C_{i, j+1} / Σ C_{i, j}` where `i` runs over the most-recent
/// `last_n` observed origins at development `j`. If `last_n == 0` or the
/// link has fewer observations, falls back to all available origins.
pub fn volume_weighted_last_n(t: &TriangleView<'_>, last_n: usize) -> Vec<f64> {
    per_link(t, |pairs| {
        averaging::volume_weighted(averaging::recent(pairs, last_n))
    })
}

/// Simple-average LDF restricted to the last `last_n` origin years
/// (`last_n == 0` uses all origins). The window is taken over observed rows
/// before undefined ratios are skipped.
pub fn simple_average_last_n(t: &TriangleView<'_>, last_n: usize) -> Vec<f64> {
    per_link(t, |pairs| {
        averaging::simple_average(averaging::recent(pairs, last_n))
    })
}

/// Geometric average of positive observed individual link ratios at each
/// development period. Emits `NaN` for links with no strictly positive pair
/// so the checked engine rejects them instead of silently selecting `1.0`.
pub fn geometric_average(t: &TriangleView<'_>) -> Vec<f64> {
    per_link(t, averaging::geometric)
}

/// Convert age-to-age LDFs into cumulative development factors (CDFs).
///
/// `cdf[j] = f[j] × f[j+1] × … × f[n−2] × tail`. CDF at the last development
/// age (`j = n − 1`) is simply `tail`. The returned vector has length
/// `ldf.len() + 1 == n`.
pub fn cdf_from_ldf(ldf: &[f64], tail: f64) -> Vec<f64> {
    let m = ldf.len();
    let mut out = vec![0.0; m + 1];
    out[m] = tail;
    let mut acc = tail;
    for j in (0..m).rev() {
        acc *= ldf[j];
        out[j] = acc;
    }
    out
}

/// Percent-reported vector from CDFs: `pct[j] = 1 / cdf[j]`. Length matches
/// `cdf`. Emits `NaN` where `cdf[j]` is non-positive.
pub fn pct_reported(cdf: &[f64]) -> Vec<f64> {
    cdf.iter()
        .map(|&x| if x > 0.0 { 1.0 / x } else { f64::NAN })
        .collect()
}

/// Volume-weighted LDFs with an explicit row-major cell weight/exclusion mask
/// of length `n_origins * n_devs`. Setting `weights[i * n_devs + j] = 0.0`
/// excludes origin `i` from development link `j -> j + 1`.
pub fn weighted_volume_ldfs(
    t: &TriangleView<'_>,
    weights: &[f64],
) -> Result<Vec<f64>, ReservingError> {
    if t.data.len() != t.n_origins * t.n_devs || weights.len() != t.n_origins * t.n_devs {
        return Err(ReservingError::ShapeMismatch);
    }
    for &w in weights {
        if !w.is_finite() {
            return Err(ReservingError::NotFinite);
        }
        if w < 0.0 {
            return Err(ReservingError::Negative);
        }
    }
    let m = t.n_devs.saturating_sub(1);
    let mut out = vec![1.0; m];
    for j in 0..m {
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..t.n_origins {
            if !t.is_scheduled_observed(i, j + 1) {
                continue;
            }
            let w = weights[i * t.n_devs + j];
            if w <= 0.0 {
                continue;
            }
            let a = t.get(i, j);
            let b = t.get(i, j + 1);
            if a.is_finite() && b.is_finite() && a >= 0.0 {
                num += w * b;
                den += w * a;
            }
        }
        if den <= 0.0 {
            return Err(ReservingError::DivideByZero(
                "weighted development denominator",
            ));
        }
        out[j] = num / den;
    }
    Ok(out)
}

// ===========================================================================
// L2 — validated engine
// ===========================================================================

/// Extrapolate missing or final-column development variances via Mack's (1993) min-rule:
/// `sigma_k^2 = min(sigma_{k-1}^4 / sigma_{k-2}^2, sigma_{k-1}^2, sigma_{k-2}^2)`.
pub(crate) fn extrapolate_mack_min_rule(raw: &[Option<f64>]) -> Vec<f64> {
    let mut out = vec![0.0; raw.len()];
    for k in 0..raw.len() {
        if let Some(v) = raw[k] {
            out[k] = v.max(0.0);
        } else if k >= 2 {
            let a = out[k - 1];
            let b = out[k - 2];
            out[k] = if a > 0.0 && b > 0.0 {
                (a * a / b).min(a).min(b).max(0.0)
            } else {
                0.0
            };
        } else if k == 1 {
            out[k] = out[0].max(0.0);
        } else {
            out[k] = 0.0;
        }
    }
    out
}

/// Mack (1993) `σ²_k` by development link for a square triangle with
/// selected `ldfs`, using the shared `n_k − 1` estimator
/// ([`averaging::mack_sigma_sq`]) and the Mack min-rule for links with fewer
/// than two defined ratios.
///
/// This is the single crate-internal deterministic definition shared by
/// Munich Chain Ladder and the actual-vs-expected residual scale.
pub(crate) fn mack_sigma_sq_by_link(t: &TriangleView<'_>, ldfs: &[f64]) -> Vec<f64> {
    let raw: Vec<Option<f64>> = mack_sigma_sq_raw_by_link(t, ldfs)
        .into_iter()
        .map(|s| s.is_finite().then_some(s))
        .collect();
    extrapolate_mack_min_rule(&raw)
}

/// Unextrapolated Mack `σ²_k` by link (`NaN` where fewer than two defined
/// ratios exist). See [`mack_sigma_sq_by_link`] for the min-rule-filled form.
pub(crate) fn mack_sigma_sq_raw_by_link(t: &TriangleView<'_>, ldfs: &[f64]) -> Vec<f64> {
    ldfs.iter()
        .enumerate()
        .map(|(k, &f)| averaging::mack_sigma_sq(&averaging::link_pairs(t, k), f))
        .collect()
}

/// Section 2 development-factor service.
///
/// Every selection method has a strict entry point (gross paid triangles:
/// monotone cumulative rows, `tail >= 1`) and a `_with_policy` variant. Under
/// [`MonotonicityPolicy::AllowNegativeIncrements`] decreasing cumulative rows
/// and positive tails below `1.0` are accepted (incurred / net triangles).
#[derive(Debug, Clone, Copy, Default)]
pub struct DevelopmentFactorEngine;

impl DevelopmentFactorEngine {
    fn factors_from_validated_ldfs(
        n_devs: usize,
        tail: f64,
        ldfs: Vec<f64>,
    ) -> Result<DevelopmentFactors, ReservingError> {
        validate_positive_finite_slice(&ldfs, "development denominator")?;

        let mut cdfs = vec![0.0; n_devs];
        try_cdfs_from_ldfs(&ldfs, tail, &mut cdfs)?;
        if cdfs.iter().any(|cdf| !cdf.is_finite()) {
            return Err(ReservingError::NotFinite);
        }

        Ok(DevelopmentFactors { ldfs, cdfs, tail })
    }

    fn select_validated(
        tri: &TriangleView<'_>,
        tail: f64,
        policy: MonotonicityPolicy,
        select: impl FnOnce(&TriangleView<'_>) -> Vec<f64>,
    ) -> Result<DevelopmentFactors, ReservingError> {
        validate_development_inputs_with_policy(tri, tail, policy)?;
        Self::factors_from_validated_ldfs(tri.n_devs, tail, select(tri))
    }

    /// Select factors for any computed [`LdfSelectionMethod`] under an explicit
    /// [`MonotonicityPolicy`].
    ///
    /// [`LdfSelectionMethod::Selected`] carries no factor values and is rejected
    /// with `InvalidParameter`; use [`Self::from_selected_ldfs`] for
    /// caller-selected factors. `last_n == 0` on the last-N variants uses all
    /// origins.
    pub fn select_with_policy(
        tri: &TriangleView<'_>,
        tail: f64,
        method: LdfSelectionMethod,
        policy: MonotonicityPolicy,
    ) -> Result<DevelopmentFactors, ReservingError> {
        match method {
            LdfSelectionMethod::VolumeWeighted => {
                Self::volume_weighted_with_policy(tri, tail, policy)
            }
            LdfSelectionMethod::SimpleAverage => {
                Self::select_validated(tri, tail, policy, simple_average)
            }
            LdfSelectionMethod::Median => Self::select_validated(tri, tail, policy, median),
            LdfSelectionMethod::ExHiLo => Self::select_validated(tri, tail, policy, ex_hi_lo),
            LdfSelectionMethod::Geometric => {
                Self::select_validated(tri, tail, policy, geometric_average)
            }
            LdfSelectionMethod::LastNVolumeWeighted { last_n } => {
                Self::select_validated(tri, tail, policy, |t| volume_weighted_last_n(t, last_n))
            }
            LdfSelectionMethod::LastNSimpleAverage { last_n } => {
                Self::select_validated(tri, tail, policy, |t| simple_average_last_n(t, last_n))
            }
            LdfSelectionMethod::Selected => Err(ReservingError::InvalidParameter(
                "LdfSelectionMethod::Selected requires explicit factors; use DevelopmentFactorEngine::from_selected_ldfs",
            )),
        }
    }

    /// Build validated factors from caller-selected age-to-age LDFs
    /// (`ldfs.len() == n_devs - 1 >= 1`), all finite and strictly positive.
    pub fn from_selected_ldfs(
        ldfs: &[f64],
        tail: f64,
        policy: MonotonicityPolicy,
    ) -> Result<DevelopmentFactors, ReservingError> {
        if ldfs.is_empty() {
            return Err(ReservingError::ShapeMismatch);
        }
        validate_tail_with_policy(tail, policy)?;
        Self::factors_from_validated_ldfs(ldfs.len() + 1, tail, ldfs.to_vec())
    }

    /// Select volume-weighted factors and CDFs from a validated cumulative triangle.
    pub fn volume_weighted(
        tri: &TriangleView<'_>,
        tail: f64,
    ) -> Result<DevelopmentFactors, ReservingError> {
        Self::volume_weighted_with_policy(tri, tail, MonotonicityPolicy::Strict)
    }

    /// Select volume-weighted factors and CDFs under an explicit [`MonotonicityPolicy`].
    pub fn volume_weighted_with_policy(
        tri: &TriangleView<'_>,
        tail: f64,
        policy: MonotonicityPolicy,
    ) -> Result<DevelopmentFactors, ReservingError> {
        validate_development_inputs_with_policy(tri, tail, policy)?;
        let mut ldfs = vec![0.0; tri.n_devs - 1];
        try_volume_weighted_ldfs(tri.data, tri.n_origins, tri.n_devs, &mut ldfs)?;
        Self::factors_from_validated_ldfs(tri.n_devs, tail, ldfs)
    }

    /// Select simple-average factors and CDFs from a validated cumulative triangle.
    pub fn simple_average(
        tri: &TriangleView<'_>,
        tail: f64,
    ) -> Result<DevelopmentFactors, ReservingError> {
        Self::simple_average_with_policy(tri, tail, MonotonicityPolicy::Strict)
    }

    /// Simple-average factors under an explicit [`MonotonicityPolicy`].
    pub fn simple_average_with_policy(
        tri: &TriangleView<'_>,
        tail: f64,
        policy: MonotonicityPolicy,
    ) -> Result<DevelopmentFactors, ReservingError> {
        Self::select_with_policy(tri, tail, LdfSelectionMethod::SimpleAverage, policy)
    }

    /// Select median factors and CDFs from a validated cumulative triangle.
    pub fn median(tri: &TriangleView<'_>, tail: f64) -> Result<DevelopmentFactors, ReservingError> {
        Self::median_with_policy(tri, tail, MonotonicityPolicy::Strict)
    }

    /// Median factors under an explicit [`MonotonicityPolicy`].
    pub fn median_with_policy(
        tri: &TriangleView<'_>,
        tail: f64,
        policy: MonotonicityPolicy,
    ) -> Result<DevelopmentFactors, ReservingError> {
        Self::select_with_policy(tri, tail, LdfSelectionMethod::Median, policy)
    }

    /// Select ex-hi-lo factors and CDFs from a validated cumulative triangle.
    pub fn ex_hi_lo(
        tri: &TriangleView<'_>,
        tail: f64,
    ) -> Result<DevelopmentFactors, ReservingError> {
        Self::ex_hi_lo_with_policy(tri, tail, MonotonicityPolicy::Strict)
    }

    /// Ex-hi-lo factors under an explicit [`MonotonicityPolicy`].
    pub fn ex_hi_lo_with_policy(
        tri: &TriangleView<'_>,
        tail: f64,
        policy: MonotonicityPolicy,
    ) -> Result<DevelopmentFactors, ReservingError> {
        Self::select_with_policy(tri, tail, LdfSelectionMethod::ExHiLo, policy)
    }

    /// Select volume-weighted last-N factors and CDFs from a validated cumulative triangle.
    pub fn volume_weighted_last_n(
        tri: &TriangleView<'_>,
        tail: f64,
        last_n: usize,
    ) -> Result<DevelopmentFactors, ReservingError> {
        Self::volume_weighted_last_n_with_policy(tri, tail, last_n, MonotonicityPolicy::Strict)
    }

    /// Volume-weighted last-N factors under an explicit [`MonotonicityPolicy`].
    pub fn volume_weighted_last_n_with_policy(
        tri: &TriangleView<'_>,
        tail: f64,
        last_n: usize,
        policy: MonotonicityPolicy,
    ) -> Result<DevelopmentFactors, ReservingError> {
        Self::select_with_policy(
            tri,
            tail,
            LdfSelectionMethod::LastNVolumeWeighted { last_n },
            policy,
        )
    }

    /// Select simple-average last-N factors and CDFs from a validated cumulative triangle.
    pub fn simple_average_last_n(
        tri: &TriangleView<'_>,
        tail: f64,
        last_n: usize,
    ) -> Result<DevelopmentFactors, ReservingError> {
        Self::simple_average_last_n_with_policy(tri, tail, last_n, MonotonicityPolicy::Strict)
    }

    /// Simple-average last-N factors under an explicit [`MonotonicityPolicy`].
    pub fn simple_average_last_n_with_policy(
        tri: &TriangleView<'_>,
        tail: f64,
        last_n: usize,
        policy: MonotonicityPolicy,
    ) -> Result<DevelopmentFactors, ReservingError> {
        Self::select_with_policy(
            tri,
            tail,
            LdfSelectionMethod::LastNSimpleAverage { last_n },
            policy,
        )
    }

    /// Select geometric-average factors and CDFs from a validated cumulative triangle.
    ///
    /// Links with no strictly positive `(C_{i,j}, C_{i,j+1})` pair are rejected
    /// with `NotFinite`.
    pub fn geometric_average(
        tri: &TriangleView<'_>,
        tail: f64,
    ) -> Result<DevelopmentFactors, ReservingError> {
        Self::geometric_average_with_policy(tri, tail, MonotonicityPolicy::Strict)
    }

    /// Geometric-average factors under an explicit [`MonotonicityPolicy`].
    pub fn geometric_average_with_policy(
        tri: &TriangleView<'_>,
        tail: f64,
        policy: MonotonicityPolicy,
    ) -> Result<DevelopmentFactors, ReservingError> {
        Self::select_validated(tri, tail, policy, geometric_average)
    }

    /// Select volume-weighted factors with a per-cell weight / outlier-exclusion
    /// mask (`weights[i * n_devs + j] == 0.0` excludes the link `j -> j+1` for
    /// origin `i`).
    pub fn volume_weighted_masked(
        tri: &TriangleView<'_>,
        weights: &[f64],
        tail: f64,
        policy: MonotonicityPolicy,
    ) -> Result<DevelopmentFactors, ReservingError> {
        validate_development_inputs_with_policy(tri, tail, policy)?;
        let ldfs = weighted_volume_ldfs(tri, weights)?;
        Self::factors_from_validated_ldfs(tri.n_devs, tail, ldfs)
    }

    /// Diagnostics for volume-weighted link selections.
    pub fn volume_weighted_diagnostics(
        tri: &TriangleView<'_>,
        tail: f64,
    ) -> Result<Vec<LinkRatioDiagnostic>, ReservingError> {
        Self::volume_weighted_diagnostics_with_policy(tri, tail, MonotonicityPolicy::Strict)
    }

    /// Volume-weighted link diagnostics under an explicit [`MonotonicityPolicy`].
    pub fn volume_weighted_diagnostics_with_policy(
        tri: &TriangleView<'_>,
        tail: f64,
        policy: MonotonicityPolicy,
    ) -> Result<Vec<LinkRatioDiagnostic>, ReservingError> {
        validate_development_inputs_with_policy(tri, tail, policy)?;

        let mut diagnostics = Vec::with_capacity(tri.n_devs - 1);
        for dev in 0..tri.n_devs - 1 {
            let diagnostic =
                volume_weighted_link_diagnostic(tri.data, tri.n_origins, tri.n_devs, dev)?;
            if !diagnostic.selected_ldf.is_finite() {
                return Err(ReservingError::DivideByZero("development denominator"));
            }
            diagnostics.push(diagnostic);
        }
        Ok(diagnostics)
    }
}

fn validate_development_inputs_with_policy(
    tri: &TriangleView<'_>,
    tail: f64,
    policy: MonotonicityPolicy,
) -> Result<(), ReservingError> {
    validate_triangle(tri)?;
    if tri.n_devs < 2 {
        return Err(ReservingError::ShapeMismatch);
    }
    validate_tail_with_policy(tail, policy)?;
    if policy == MonotonicityPolicy::Strict {
        validate_monotone_cumulative(tri)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Triangle;

    fn triangle_3x3() -> Triangle {
        #[rustfmt::skip]
        let data = vec![
            100.0, 150.0, 180.0,
            120.0, 180.0, f64::NAN,
            140.0, f64::NAN, f64::NAN,
        ];
        Triangle::new(3, 3, data)
    }

    fn triangle_5x5() -> Triangle {
        #[rustfmt::skip]
        let data = vec![
            100.0, 150.0, 180.0, 198.0, 207.9,
            120.0, 180.0, 216.0, 237.6, f64::NAN,
            140.0, 210.0, 252.0, f64::NAN, f64::NAN,
            160.0, 240.0, f64::NAN, f64::NAN, f64::NAN,
            180.0, f64::NAN, f64::NAN, f64::NAN, f64::NAN,
        ];
        Triangle::new(5, 5, data)
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * (1.0 + b.abs())
    }

    #[test]
    fn volume_weighted_single_ldf_uses_only_observed_pairs() {
        let tri = triangle_3x3();

        let f0 = volume_weighted_ldf(tri.data(), 3, 3, 0).expect("f0");
        let f1 = volume_weighted_ldf(tri.data(), 3, 3, 1).expect("f1");

        assert!(close(f0, (150.0 + 180.0) / (100.0 + 120.0)));
        assert!(close(f1, 180.0 / 150.0));
    }

    #[test]
    fn volume_weighted_all_ldfs_3x3() {
        let tri = triangle_3x3();
        let mut out = [0.0; 2];

        volume_weighted_ldfs(tri.data(), 3, 3, &mut out);

        assert!(close(out[0], 1.5));
        assert!(close(out[1], 1.2));
    }

    #[test]
    fn volume_weighted_all_ldfs_5x5() {
        let tri = triangle_5x5();
        let mut out = [0.0; 4];

        volume_weighted_ldfs(tri.data(), 5, 5, &mut out);

        assert!(close(out[0], 1.5));
        assert!(close(out[1], 1.2));
        assert!(close(out[2], 1.1));
        assert!(close(out[3], 1.05));
    }

    #[test]
    fn observed_counts_ldfs_handle_semiannual_rectangle() {
        #[rustfmt::skip]
        let cum = [
            100.0, 150.0, 180.0, 198.0,
            120.0, 180.0, f64::NAN, f64::NAN,
            140.0, f64::NAN, f64::NAN, f64::NAN,
        ];
        let observed_counts = [4, 2, 1];
        let mut out = [0.0; 3];

        volume_weighted_ldfs_observed_counts(&cum, 3, 4, &observed_counts, &mut out);

        assert!(close(out[0], (150.0 + 180.0) / (100.0 + 120.0)));
        assert!(close(out[1], 180.0 / 150.0));
        assert!(close(out[2], 198.0 / 180.0));
    }

    #[test]
    fn cdf_kernel_uses_tail_convention() {
        let ldfs = [1.5, 1.2, 1.1];
        let mut cdfs = [0.0; 4];

        cdfs_from_ldfs(&ldfs, 1.0, &mut cdfs);

        assert!(close(cdfs[0], 1.98));
        assert!(close(cdfs[1], 1.32));
        assert!(close(cdfs[2], 1.10));
        assert!(close(cdfs[3], 1.00));
    }

    #[test]
    fn engine_returns_factors_cdfs_and_maturity() {
        let tri = triangle_5x5();
        let factors = DevelopmentFactorEngine::volume_weighted(&tri.view(), 1.0).expect("factors");

        assert_eq!(factors.ldfs.len(), 4);
        assert_eq!(factors.cdfs.len(), 5);
        assert!(close(factors.cdfs[4], 1.0));

        let maturity = factors.maturity_percentages();
        assert!(close(maturity[4], 1.0));
        assert!(maturity[0] < maturity[1]);
    }

    #[test]
    fn engine_supports_non_volume_weighted_selection_methods() {
        let tri = triangle_5x5();

        let simple =
            DevelopmentFactorEngine::simple_average(&tri.view(), 1.0).expect("simple average");
        assert_eq!(simple.ldfs, simple_average(&tri.view()));

        let median_factors =
            DevelopmentFactorEngine::median(&tri.view(), 1.0).expect("median factors");
        assert_eq!(median_factors.ldfs, median(&tri.view()));

        let ex_hi_lo_factors =
            DevelopmentFactorEngine::ex_hi_lo(&tri.view(), 1.0).expect("ex-hi-lo factors");
        assert_eq!(ex_hi_lo_factors.ldfs, ex_hi_lo(&tri.view()));

        let last_n = DevelopmentFactorEngine::volume_weighted_last_n(&tri.view(), 1.0, 2)
            .expect("last-n volume weighted factors");
        assert_eq!(last_n.ldfs, volume_weighted_last_n(&tri.view(), 2));

        let last_n_simple = DevelopmentFactorEngine::simple_average_last_n(&tri.view(), 1.0, 2)
            .expect("last-n simple average factors");
        assert_eq!(last_n_simple.ldfs, simple_average_last_n(&tri.view(), 2));
    }

    #[test]
    fn latest_maturity_percentages_follow_latest_diagonal_age() {
        let cdfs = [1.98, 1.32, 1.10, 1.0];
        let mut out = [0.0; 4];

        latest_maturity_percentages(4, 4, &cdfs, &mut out).expect("maturity");

        assert!(close(out[0], 1.0));
        assert!(close(out[1], 1.0 / 1.10));
        assert!(close(out[2], 1.0 / 1.32));
        assert!(close(out[3], 1.0 / 1.98));
    }

    #[test]
    fn diagnostics_include_numerator_denominator_and_rows() {
        let tri = triangle_3x3();
        let diagnostics = DevelopmentFactorEngine::volume_weighted_diagnostics(&tri.view(), 1.0)
            .expect("diagnostics");

        assert_eq!(diagnostics.len(), 2);
        assert_eq!(diagnostics[0].dev, 0);
        assert!(close(diagnostics[0].numerator, 330.0));
        assert!(close(diagnostics[0].denominator, 220.0));
        assert_eq!(diagnostics[0].included_rows, 2);
        assert!(close(diagnostics[0].selected_ldf, 1.5));
    }

    #[test]
    fn engine_rejects_tail_below_one() {
        let tri = triangle_3x3();

        assert_eq!(
            DevelopmentFactorEngine::volume_weighted(&tri.view(), 0.99),
            Err(ReservingError::InvalidTail { tail: 0.99 })
        );
    }

    #[test]
    fn engine_rejects_non_cumulative_rows() {
        #[rustfmt::skip]
        let tri = Triangle::new(3, 3, vec![
            100.0, 90.0, 180.0,
            120.0, 180.0, f64::NAN,
            140.0, f64::NAN, f64::NAN,
        ]);

        assert_eq!(
            DevelopmentFactorEngine::volume_weighted(&tri.view(), 1.0),
            Err(ReservingError::NonMonotoneCumulative {
                origin: 0,
                dev: 1,
                previous: 100.0,
                value: 90.0,
            })
        );
    }
}
