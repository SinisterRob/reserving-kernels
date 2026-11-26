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

use crate::triangle::age_to_age;
use crate::types::TriangleView;
use crate::validate::{
    MonotonicityPolicy, ValidationError, validate_monotone_cumulative,
    validate_positive_finite_slice, validate_triangle,
};

// ===========================================================================
// Types
// ===========================================================================

/// Selected development factor family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// Caller-selected link ratios.
    Selected,
}

/// Selected age-to-age factors and cumulative development factors.
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, Copy, PartialEq)]
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
#[inline]
pub fn volume_weighted_ldf(cum: &[f64], n_origin: usize, n_dev: usize, dev: usize) -> f64 {
    volume_weighted_link_diagnostic(cum, n_origin, n_dev, dev).selected_ldf
}

/// Diagnostic numerator/denominator trail for one volume-weighted LDF.
pub fn volume_weighted_link_diagnostic(
    cum: &[f64],
    n_origin: usize,
    n_dev: usize,
    dev: usize,
) -> LinkRatioDiagnostic {
    debug_assert_eq!(cum.len(), n_origin * n_dev);
    debug_assert!(dev + 1 < n_dev);

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

/// Volume-weighted LDFs for all development links.
pub fn volume_weighted_ldfs(cum: &[f64], n_origin: usize, n_dev: usize, out: &mut [f64]) {
    debug_assert_eq!(cum.len(), n_origin * n_dev);
    debug_assert_eq!(out.len(), n_dev.saturating_sub(1));

    for dev in 0..n_dev.saturating_sub(1) {
        out[dev] = volume_weighted_ldf(cum, n_origin, n_dev, dev);
    }
}

/// Volume-weighted age-to-age LDF for arbitrary observed row widths.
///
/// `observed_counts[i]` is the number of observed development periods in row `i`.
pub fn volume_weighted_ldf_observed_counts(
    cum: &[f64],
    n_origin: usize,
    n_dev: usize,
    observed_counts: &[usize],
    dev: usize,
) -> f64 {
    volume_weighted_link_diagnostic_observed_counts(cum, n_origin, n_dev, observed_counts, dev)
        .selected_ldf
}

/// Diagnostic numerator/denominator trail for arbitrary observed row widths.
pub fn volume_weighted_link_diagnostic_observed_counts(
    cum: &[f64],
    n_origin: usize,
    n_dev: usize,
    observed_counts: &[usize],
    dev: usize,
) -> LinkRatioDiagnostic {
    debug_assert_eq!(cum.len(), n_origin * n_dev);
    debug_assert_eq!(observed_counts.len(), n_origin);
    debug_assert!(dev + 1 < n_dev);

    let mut numerator = 0.0;
    let mut denominator = 0.0;
    let mut included_rows = 0;

    for i in 0..n_origin {
        if observed_counts[i] > dev + 1 {
            denominator += cum[i * n_dev + dev];
            numerator += cum[i * n_dev + dev + 1];
            included_rows += 1;
        }
    }

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

/// Volume-weighted LDFs for arbitrary observed row widths.
pub fn volume_weighted_ldfs_observed_counts(
    cum: &[f64],
    n_origin: usize,
    n_dev: usize,
    observed_counts: &[usize],
    out: &mut [f64],
) {
    debug_assert_eq!(cum.len(), n_origin * n_dev);
    debug_assert_eq!(observed_counts.len(), n_origin);
    debug_assert_eq!(out.len(), n_dev.saturating_sub(1));

    for dev in 0..n_dev.saturating_sub(1) {
        out[dev] = volume_weighted_ldf_observed_counts(cum, n_origin, n_dev, observed_counts, dev);
    }
}

/// CDFs from selected LDFs using the convention:
///
/// ```text
/// cdfs[last_dev] = tail
/// cdfs[j] = ldfs[j] * cdfs[j + 1]
/// ```
pub fn cdfs_from_ldfs(ldfs: &[f64], tail: f64, out: &mut [f64]) {
    debug_assert_eq!(out.len(), ldfs.len() + 1);
    if out.is_empty() {
        return;
    }

    let last = out.len() - 1;
    out[last] = tail;
    for j in (0..ldfs.len()).rev() {
        out[j] = ldfs[j] * out[j + 1];
    }
}

/// Latest maturity percentage by origin under right-triangle geometry.
pub fn latest_maturity_percentages(n_origin: usize, n_dev: usize, cdfs: &[f64], out: &mut [f64]) {
    debug_assert_eq!(cdfs.len(), n_dev);
    debug_assert_eq!(out.len(), n_origin);

    for i in 0..n_origin {
        if i >= n_dev {
            out[i] = f64::NAN;
            continue;
        }
        let latest_dev = n_dev - 1 - i;
        let cdf = cdfs[latest_dev];
        out[i] = if cdf > 0.0 { 1.0 / cdf } else { f64::NAN };
    }
}

// ===========================================================================
// L1 — view-based selection methods (returns Vec<f64>)
// ===========================================================================

/// Volume-weighted all-year LDF: `f_j = Σ_i C_{i, j+1} / Σ_i C_{i, j}`
/// summing over origins where both cells are observed (`i ≤ n − j − 2`).
///
/// This is the classical chain-ladder factor. Returns a vector of length
/// `n − 1`. Cells where the denominator is zero emit `NaN`.
pub fn volume_weighted(t: &TriangleView<'_>) -> Vec<f64> {
    let n = t.n_origins.min(t.n_devs);
    if n < 2 {
        return Vec::new();
    }
    let mut out = vec![0.0; n - 1];
    for j in 0..n - 1 {
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..n - j - 1 {
            num += t.get(i, j + 1);
            den += t.get(i, j);
        }
        out[j] = if den > 0.0 { num / den } else { f64::NAN };
    }
    out
}

/// Simple (straight) average of individual age-to-age ratios:
/// `f_j = mean_i( C_{i, j+1} / C_{i, j} )`. Each origin contributes equal
/// weight regardless of loss size. Returns a vector of length `n − 1`.
pub fn simple_average(t: &TriangleView<'_>) -> Vec<f64> {
    let n = t.n_origins.min(t.n_devs);
    if n < 2 {
        return Vec::new();
    }
    let mut out = vec![0.0; n - 1];
    for j in 0..n - 1 {
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..n - j - 1 {
            let r = age_to_age(t, i, j);
            if r.is_finite() {
                sum += r;
                count += 1;
            }
        }
        out[j] = if count > 0 {
            sum / (count as f64)
        } else {
            f64::NAN
        };
    }
    out
}

/// Median of individual age-to-age ratios for each development link.
pub fn median(t: &TriangleView<'_>) -> Vec<f64> {
    let n = t.n_origins.min(t.n_devs);
    if n < 2 {
        return Vec::new();
    }
    let mut out = vec![0.0; n - 1];
    let mut scratch: Vec<f64> = Vec::with_capacity(n);
    for j in 0..n - 1 {
        scratch.clear();
        for i in 0..n - j - 1 {
            let r = age_to_age(t, i, j);
            if r.is_finite() {
                scratch.push(r);
            }
        }
        scratch.sort_by(f64::total_cmp);
        let k = scratch.len();
        out[j] = if k == 0 {
            f64::NAN
        } else if k % 2 == 1 {
            scratch[k / 2]
        } else {
            (scratch[k / 2 - 1] + scratch[k / 2]) / 2.0
        };
    }
    out
}

/// Ex-hi / ex-lo (trimmed mean) of individual age-to-age ratios:
/// `f_j = mean of the individual ratios after dropping the single max and
/// single min`. When fewer than 3 ratios are available for a link, falls
/// back to [`simple_average`] on that link.
pub fn ex_hi_lo(t: &TriangleView<'_>) -> Vec<f64> {
    let n = t.n_origins.min(t.n_devs);
    if n < 2 {
        return Vec::new();
    }
    let mut out = vec![0.0; n - 1];
    let mut scratch: Vec<f64> = Vec::with_capacity(n);
    for j in 0..n - 1 {
        scratch.clear();
        for i in 0..n - j - 1 {
            let r = age_to_age(t, i, j);
            if r.is_finite() {
                scratch.push(r);
            }
        }
        let k = scratch.len();
        out[j] = if k >= 3 {
            let mut mn = scratch[0];
            let mut mx = scratch[0];
            let mut sum = 0.0;
            for &v in &scratch {
                if v < mn {
                    mn = v;
                }
                if v > mx {
                    mx = v;
                }
                sum += v;
            }
            (sum - mn - mx) / ((k - 2) as f64)
        } else if k > 0 {
            scratch.iter().sum::<f64>() / (k as f64)
        } else {
            f64::NAN
        };
    }
    out
}

/// Volume-weighted average restricted to the last `last_n` origin years.
///
/// `f_j = Σ C_{i, j+1} / Σ C_{i, j}` where `i` runs over the most-recent
/// `last_n` observed origins at development `j`. If `last_n == 0` or the
/// link has fewer observations, falls back to all available origins.
pub fn volume_weighted_last_n(t: &TriangleView<'_>, last_n: usize) -> Vec<f64> {
    let n = t.n_origins.min(t.n_devs);
    if n < 2 {
        return Vec::new();
    }
    let mut out = vec![0.0; n - 1];
    for j in 0..n - 1 {
        let available = n - j - 1;
        let use_k = if last_n == 0 {
            available
        } else {
            available.min(last_n)
        };
        // Most-recent origins are the largest `i` in `0..available`.
        let start = available - use_k;
        let mut num = 0.0;
        let mut den = 0.0;
        for i in start..available {
            num += t.get(i, j + 1);
            den += t.get(i, j);
        }
        out[j] = if den > 0.0 { num / den } else { f64::NAN };
    }
    out
}

/// Simple-average LDF restricted to the last `last_n` origin years.
pub fn simple_average_last_n(t: &TriangleView<'_>, last_n: usize) -> Vec<f64> {
    let n = t.n_origins.min(t.n_devs);
    if n < 2 {
        return Vec::new();
    }
    let mut out = vec![0.0; n - 1];
    for j in 0..n - 1 {
        let available = n - j - 1;
        let use_k = if last_n == 0 {
            available
        } else {
            available.min(last_n)
        };
        let start = available - use_k;
        let mut sum = 0.0;
        let mut count = 0;
        for i in start..available {
            let r = age_to_age(t, i, j);
            if r.is_finite() {
                sum += r;
                count += 1;
            }
        }
        out[j] = if count > 0 {
            sum / (count as f64)
        } else {
            f64::NAN
        };
    }
    out
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

// ===========================================================================
// L2 — validated engine
// ===========================================================================

/// Section 2 development-factor service.
pub struct DevelopmentFactorEngine;

impl DevelopmentFactorEngine {
    fn factors_from_selected_ldfs_with_policy(
        tri: &TriangleView<'_>,
        tail: f64,
        ldfs: Vec<f64>,
        policy: MonotonicityPolicy,
    ) -> Result<DevelopmentFactors, ValidationError> {
        validate_development_inputs_with_policy(tri, tail, policy)?;
        validate_positive_finite_slice(&ldfs, "development denominator")?;

        let mut cdfs = vec![0.0; tri.n_devs];
        cdfs_from_ldfs(&ldfs, tail, &mut cdfs);
        if cdfs.iter().any(|cdf| !cdf.is_finite()) {
            return Err(ValidationError::NotFinite);
        }

        Ok(DevelopmentFactors { ldfs, cdfs, tail })
    }

    fn factors_from_selected_ldfs(
        tri: &TriangleView<'_>,
        tail: f64,
        ldfs: Vec<f64>,
    ) -> Result<DevelopmentFactors, ValidationError> {
        Self::factors_from_selected_ldfs_with_policy(tri, tail, ldfs, MonotonicityPolicy::Strict)
    }

    /// Select volume-weighted factors and CDFs from a validated cumulative triangle.
    pub fn volume_weighted(
        tri: &TriangleView<'_>,
        tail: f64,
    ) -> Result<DevelopmentFactors, ValidationError> {
        Self::volume_weighted_with_policy(tri, tail, MonotonicityPolicy::Strict)
    }

    /// Select volume-weighted factors and CDFs under an explicit [`MonotonicityPolicy`].
    pub fn volume_weighted_with_policy(
        tri: &TriangleView<'_>,
        tail: f64,
        policy: MonotonicityPolicy,
    ) -> Result<DevelopmentFactors, ValidationError> {
        validate_development_inputs_with_policy(tri, tail, policy)?;
        let mut ldfs = vec![0.0; tri.n_devs - 1];
        volume_weighted_ldfs(tri.data, tri.n_origins, tri.n_devs, &mut ldfs);
        Self::factors_from_selected_ldfs_with_policy(tri, tail, ldfs, policy)
    }

    /// Select simple-average factors and CDFs from a validated cumulative triangle.
    pub fn simple_average(
        tri: &TriangleView<'_>,
        tail: f64,
    ) -> Result<DevelopmentFactors, ValidationError> {
        Self::factors_from_selected_ldfs(tri, tail, simple_average(tri))
    }

    /// Select median factors and CDFs from a validated cumulative triangle.
    pub fn median(
        tri: &TriangleView<'_>,
        tail: f64,
    ) -> Result<DevelopmentFactors, ValidationError> {
        Self::factors_from_selected_ldfs(tri, tail, median(tri))
    }

    /// Select ex-hi-lo factors and CDFs from a validated cumulative triangle.
    pub fn ex_hi_lo(
        tri: &TriangleView<'_>,
        tail: f64,
    ) -> Result<DevelopmentFactors, ValidationError> {
        Self::factors_from_selected_ldfs(tri, tail, ex_hi_lo(tri))
    }

    /// Select volume-weighted last-N factors and CDFs from a validated cumulative triangle.
    pub fn volume_weighted_last_n(
        tri: &TriangleView<'_>,
        tail: f64,
        last_n: usize,
    ) -> Result<DevelopmentFactors, ValidationError> {
        Self::factors_from_selected_ldfs(tri, tail, volume_weighted_last_n(tri, last_n))
    }

    /// Select simple-average last-N factors and CDFs from a validated cumulative triangle.
    pub fn simple_average_last_n(
        tri: &TriangleView<'_>,
        tail: f64,
        last_n: usize,
    ) -> Result<DevelopmentFactors, ValidationError> {
        Self::factors_from_selected_ldfs(tri, tail, simple_average_last_n(tri, last_n))
    }

    /// Diagnostics for volume-weighted link selections.
    pub fn volume_weighted_diagnostics(
        tri: &TriangleView<'_>,
        tail: f64,
    ) -> Result<Vec<LinkRatioDiagnostic>, ValidationError> {
        validate_development_inputs(tri, tail)?;

        let mut diagnostics = Vec::with_capacity(tri.n_devs - 1);
        for dev in 0..tri.n_devs - 1 {
            let diagnostic =
                volume_weighted_link_diagnostic(tri.data, tri.n_origins, tri.n_devs, dev);
            if !diagnostic.selected_ldf.is_finite() {
                return Err(ValidationError::DivideByZero("development denominator"));
            }
            diagnostics.push(diagnostic);
        }
        Ok(diagnostics)
    }
}

fn validate_development_inputs(tri: &TriangleView<'_>, tail: f64) -> Result<(), ValidationError> {
    validate_development_inputs_with_policy(tri, tail, MonotonicityPolicy::Strict)
}

fn validate_development_inputs_with_policy(
    tri: &TriangleView<'_>,
    tail: f64,
    policy: MonotonicityPolicy,
) -> Result<(), ValidationError> {
    validate_triangle(tri)?;
    if tri.n_devs < 2 {
        return Err(ValidationError::ShapeMismatch);
    }
    if !tail.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if tail < 1.0 {
        return Err(ValidationError::InvalidTail { tail });
    }
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

        let f0 = volume_weighted_ldf(&tri.data, 3, 3, 0);
        let f1 = volume_weighted_ldf(&tri.data, 3, 3, 1);

        assert!(close(f0, (150.0 + 180.0) / (100.0 + 120.0)));
        assert!(close(f1, 180.0 / 150.0));
    }

    #[test]
    fn volume_weighted_all_ldfs_3x3() {
        let tri = triangle_3x3();
        let mut out = [0.0; 2];

        volume_weighted_ldfs(&tri.data, 3, 3, &mut out);

        assert!(close(out[0], 1.5));
        assert!(close(out[1], 1.2));
    }

    #[test]
    fn volume_weighted_all_ldfs_5x5() {
        let tri = triangle_5x5();
        let mut out = [0.0; 4];

        volume_weighted_ldfs(&tri.data, 5, 5, &mut out);

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

        latest_maturity_percentages(4, 4, &cdfs, &mut out);

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
            Err(ValidationError::InvalidTail { tail: 0.99 })
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
            Err(ValidationError::NonMonotoneCumulative {
                origin: 0,
                dev: 1,
                previous: 100.0,
                value: 90.0,
            })
        );
    }
}
