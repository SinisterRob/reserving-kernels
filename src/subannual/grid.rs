//! Asymmetric grid geometry, valuation age mapping, stub average claim age,
//! and calendar-diagonal alignment.
//!
//! Implements:
//!
//! * **Asymmetric Age Grid (`asymmetric_age_grid`)** — Friedland (2010),
//!   *Estimating Unpaid Claims Using Basic Techniques*, Ch. 7: maps
//!   `(origin_idx, dev_idx)` on rectangular grids (annual origin × semiannual,
//!   quarterly, or monthly development; stride ratios 2:1, 4:1, 12:1, or general
//!   positive ratios) to exact valuation ages in months, supporting interim
//!   valuations with a partial stub latest origin.
//! * **Stub Average Age (`stub_average_age`)** — Clark (2003), *"LDF
//!   Curve-Fitting and Stochastic Reserving"*, CAS Fall Forum; Bloom (2015),
//!   *"Interpolation and Extrapolation of Loss Development Factors for Interim
//!   Valuations"*: computes the effective average claim age for complete or
//!   incomplete/stub origin periods under uniform or custom exposure profiles.
//! * **Calendar Diagonal Alignment (`calendar_diagonal_index`)** — Shapland
//!   (2016), *Using the ODP Bootstrap Model: A Practitioner's Guide*, CAS
//!   Monograph No. 4: maps cells on non-square grids to their true calendar
//!   valuation diagonals so calendar-effect diagnostics and ODP bootstrap
//!   residuals align across asymmetric triangles.

use serde::Deserialize;

use crate::validate::ValidationError;

/// Deserialize a `Vec<f64>` where JSON `null` values map back to `f64::NAN`
/// (enabling lossless `serde_json` roundtrips for grids containing `NaN` sentinels).
fn deserialize_vec_f64_null_as_nan<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opts: Vec<Option<f64>> = Vec::deserialize(deserializer)?;
    Ok(opts.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
}

/// Exact age and observation geometry for an asymmetric (rectangular) loss
/// development triangle.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AsymmetricAgeGrid {
    /// Number of origin (accident / policy) periods.
    pub n_origins: usize,
    /// Number of development columns.
    pub n_devs: usize,
    /// Length of a full origin period in months (e.g., `12.0` for annual).
    pub origin_months: f64,
    /// Step between consecutive development columns in months (e.g., `6.0`, `3.0`, `1.0`).
    pub dev_months: f64,
    /// Stride ratio `origin_months / dev_months` (e.g., `2.0`, `4.0`, `12.0`).
    pub stride_ratio: f64,
    /// Valuation age (in months from origin start) of each development column
    /// (length `n_devs`).
    pub dev_ages_months: Vec<f64>,
    /// Start month of each origin period relative to origin 0 (`i * origin_months`).
    pub origin_start_months: Vec<f64>,
    /// Earned exposure duration in months for each origin period at the
    /// valuation date (`origin_months` for full origins, `< origin_months` for
    /// a partial stub origin).
    pub origin_durations_months: Vec<f64>,
    /// Valuation date in months measured from the start of origin 0.
    pub valuation_month: f64,
    /// Number of observed development columns for each origin row (length `n_origins`).
    pub observed_counts: Vec<usize>,
    /// Row-major (`n_origins * n_devs`) exact valuation age in months from
    /// origin start for each cell (`NaN` for unobserved cells).
    #[serde(deserialize_with = "deserialize_vec_f64_null_as_nan")]
    pub cell_ages_months: Vec<f64>,
    /// Row-major (`n_origins * n_devs`) effective average claim age in months
    /// (accounting for uniform exposure over the origin's earned duration; `NaN`
    /// for unobserved cells).
    #[serde(deserialize_with = "deserialize_vec_f64_null_as_nan")]
    pub cell_average_ages_months: Vec<f64>,
}

impl AsymmetricAgeGrid {
    /// Valuation age in months at `(origin, dev)`, or `NaN` if unobserved.
    #[inline]
    pub fn cell_age(&self, origin: usize, dev: usize) -> f64 {
        self.cell_ages_months[origin * self.n_devs + dev]
    }

    /// Effective average claim age in months at `(origin, dev)`, or `NaN` if unobserved.
    #[inline]
    pub fn cell_average_age(&self, origin: usize, dev: usize) -> f64 {
        self.cell_average_ages_months[origin * self.n_devs + dev]
    }

    /// `true` iff cell `(origin, dev)` is observed at the grid's valuation date.
    #[inline]
    pub fn is_observed(&self, origin: usize, dev: usize) -> bool {
        origin < self.n_origins && dev < self.observed_counts[origin]
    }

    /// Latest observed development column index for each origin (`observed_counts[i] - 1`).
    pub fn latest_dev_indices(&self) -> Vec<usize> {
        self.observed_counts
            .iter()
            .map(|&c| c.saturating_sub(1))
            .collect()
    }

    /// Exact valuation age in months of the latest observed cell for each origin.
    pub fn latest_ages_months(&self) -> Vec<f64> {
        (0..self.n_origins)
            .map(|i| {
                let c = self.observed_counts[i];
                if c == 0 {
                    f64::NAN
                } else {
                    self.cell_ages_months[i * self.n_devs + (c - 1)]
                }
            })
            .collect()
    }

    /// Earned exposure fraction `origin_durations_months[i] / origin_months` for each origin.
    pub fn exposure_fractions(&self) -> Vec<f64> {
        self.origin_durations_months
            .iter()
            .map(|&d| d / self.origin_months)
            .collect()
    }
}

/// Build an [`AsymmetricAgeGrid`] mapping `(origin_idx, dev_idx)` on non-square
/// grids to exact valuation ages and effective average claim ages in months
/// (Friedland 2010).
///
/// # Parameters
/// * `n_origins` — number of origin periods (`>= 1`).
/// * `n_devs` — number of development columns (`>= 1`).
/// * `origin_months` — full length of one origin period in months (e.g., `12.0`).
/// * `dev_months` — step between adjacent development columns in months (e.g., `6.0`, `3.0`, `1.0`).
/// * `first_dev_age_months` — valuation age in months of column `0` (typically `dev_months`).
/// * `stub_months` — optional duration in months of the most recent origin at the
///   valuation date (`0 < stub_months <= origin_months`). When `None`, the most
///   recent origin is valued at `origin_months` (or `first_dev_age_months` if
///   `n_origins == 1` and `n_devs == 1`). When `Some(s)`, the valuation date is
///   set to `(n_origins - 1) * origin_months + s`, creating an interim valuation
///   where origin `n_origins - 1` is an `s`-month stub.
pub fn asymmetric_age_grid(
    n_origins: usize,
    n_devs: usize,
    origin_months: f64,
    dev_months: f64,
    first_dev_age_months: f64,
    stub_months: Option<f64>,
) -> Result<AsymmetricAgeGrid, ValidationError> {
    if n_origins == 0 || n_devs == 0 {
        return Err(ValidationError::ShapeMismatch);
    }
    if !origin_months.is_finite() || !dev_months.is_finite() || !first_dev_age_months.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if origin_months <= 0.0 || dev_months <= 0.0 || first_dev_age_months <= 0.0 {
        return Err(ValidationError::InvalidParameter(
            "origin_months, dev_months, and first_dev_age_months must be positive",
        ));
    }
    let latest_duration = match stub_months {
        Some(s) => {
            if !s.is_finite() {
                return Err(ValidationError::NotFinite);
            }
            if s <= 0.0 || s > origin_months + 1e-9 {
                return Err(ValidationError::InvalidParameter(
                    "stub_months must be in (0, origin_months]",
                ));
            }
            s.min(origin_months)
        }
        None => origin_months,
    };

    let stride_ratio = origin_months / dev_months;
    let dev_ages_months: Vec<f64> = (0..n_devs)
        .map(|j| first_dev_age_months + (j as f64) * dev_months)
        .collect();
    let origin_start_months: Vec<f64> =
        (0..n_origins).map(|i| (i as f64) * origin_months).collect();
    let valuation_month = ((n_origins - 1) as f64) * origin_months + latest_duration;

    let mut origin_durations_months = Vec::with_capacity(n_origins);
    let mut observed_counts = Vec::with_capacity(n_origins);
    let mut cell_ages_months = vec![f64::NAN; n_origins * n_devs];
    let mut cell_average_ages_months = vec![f64::NAN; n_origins * n_devs];

    let tol = 1e-9 * origin_months.max(dev_months).max(1.0);
    for i in 0..n_origins {
        let max_age_i = valuation_month - origin_start_months[i];
        let dur_i = max_age_i.min(origin_months).max(0.0);
        origin_durations_months.push(dur_i);

        let mut count = 0usize;
        for j in 0..n_devs {
            let age = dev_ages_months[j];
            if age <= max_age_i + tol {
                cell_ages_months[i * n_devs + j] = age;
                cell_average_ages_months[i * n_devs + j] = stub_average_age(age, dur_i)?;
                count += 1;
            }
        }
        observed_counts.push(count);
    }

    Ok(AsymmetricAgeGrid {
        n_origins,
        n_devs,
        origin_months,
        dev_months,
        stride_ratio,
        dev_ages_months,
        origin_start_months,
        origin_durations_months,
        valuation_month,
        observed_counts,
        cell_ages_months,
        cell_average_ages_months,
    })
}

/// Compute the effective average claim age (in months) for a complete or
/// incomplete/stub origin period under uniform exposure (Clark 2003; Bloom 2015).
///
/// For an origin period with earned exposure over `[0, d]` where
/// `d = min(valuation_age_months, origin_duration_months)`, the average accident
/// date under uniform exposure is `d / 2`, giving an effective average claim age
/// of:
///
/// ```text
/// avg_age = valuation_age_months - 0.5 * min(valuation_age_months, origin_duration_months)
/// ```
///
/// Examples:
/// * A 6-month stub valued at 6 months (`valuation_age = 6`, `origin_duration = 6` or `12`)
///   has average claim age `3.0` months.
/// * A full 12-month accident year valued at 12 months has average claim age `6.0` months.
/// * A full 12-month accident year valued at 18 months (interim 6m valuation) has
///   average claim age `12.0` months.
pub fn stub_average_age(
    valuation_age_months: f64,
    origin_duration_months: f64,
) -> Result<f64, ValidationError> {
    if !valuation_age_months.is_finite() || !origin_duration_months.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if valuation_age_months <= 0.0 || origin_duration_months <= 0.0 {
        return Err(ValidationError::InvalidParameter(
            "valuation_age_months and origin_duration_months must be positive",
        ));
    }
    let effective_duration = valuation_age_months.min(origin_duration_months);
    Ok(valuation_age_months - 0.5 * effective_duration)
}

/// Compute the effective average claim age (in months) under a non-uniform
/// piecewise-constant sub-period exposure profile (e.g., seasonal monthly weights).
///
/// `exposure_weights` gives non-negative exposure weights over consecutive equal
/// slices of width `slice_months` starting at month `0`. Only exposure earned up
/// to `valuation_age_months` contributes (partial last slice is pro-rated linearly).
pub fn stub_average_age_weighted(
    valuation_age_months: f64,
    slice_months: f64,
    exposure_weights: &[f64],
) -> Result<f64, ValidationError> {
    if !valuation_age_months.is_finite() || !slice_months.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if valuation_age_months <= 0.0 || slice_months <= 0.0 {
        return Err(ValidationError::InvalidParameter(
            "valuation_age_months and slice_months must be positive",
        ));
    }
    if exposure_weights.is_empty() {
        return Err(ValidationError::ShapeMismatch);
    }

    let mut weighted_accident_time = 0.0;
    let mut total_weight = 0.0;
    for (k, &w) in exposure_weights.iter().enumerate() {
        if !w.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if w < 0.0 {
            return Err(ValidationError::Negative);
        }
        let t_start = (k as f64) * slice_months;
        if t_start >= valuation_age_months {
            break;
        }
        let t_end = ((k + 1) as f64 * slice_months).min(valuation_age_months);
        let frac = (t_end - t_start) / slice_months;
        let eff_w = w * frac;
        let mid_t = f64::midpoint(t_start, t_end);
        weighted_accident_time += eff_w * mid_t;
        total_weight += eff_w;
    }

    if total_weight <= 0.0 {
        return Err(ValidationError::DivideByZero("exposure_weights"));
    }
    let avg_accident_time = weighted_accident_time / total_weight;
    Ok(valuation_age_months - avg_accident_time)
}

/// Calendar-diagonal metadata for a single observed cell `(origin, dev)`.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CalendarDiagonalCell {
    /// Origin period index.
    pub origin: usize,
    /// Development column index.
    pub dev: usize,
    /// Exact calendar valuation time in months from origin 0 start.
    pub calendar_month: f64,
    /// Chronological diagonal bucket index (`0..n_diagonals`).
    pub diagonal_index: usize,
    /// Calendar major-period (e.g., calendar year) index `floor((calendar_month - eps) / origin_months)`.
    pub calendar_period_index: usize,
}

/// Complete calendar-diagonal mapping across an asymmetric triangle (Shapland 2016).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CalendarDiagonalMap {
    /// Number of origin periods.
    pub n_origins: usize,
    /// Number of development columns.
    pub n_devs: usize,
    /// Distinct sorted calendar valuation months for each diagonal bucket.
    pub diagonal_months: Vec<f64>,
    /// Row-major (`n_origins * n_devs`) diagonal bucket index (`None` for unobserved cells).
    pub cell_diagonal: Vec<Option<usize>>,
    /// Row-major (`n_origins * n_devs`) exact calendar valuation month (`NaN` for unobserved cells).
    #[serde(deserialize_with = "deserialize_vec_f64_null_as_nan")]
    pub cell_calendar_months: Vec<f64>,
    /// Cells grouped by chronological diagonal bucket (`diagonals[d]` lists all
    /// observed cells valued at `diagonal_months[d]`).
    pub diagonals: Vec<Vec<CalendarDiagonalCell>>,
}

impl CalendarDiagonalMap {
    /// Number of distinct observed calendar diagonals.
    #[inline]
    pub fn n_diagonals(&self) -> usize {
        self.diagonal_months.len()
    }

    /// Diagonal bucket index for cell `(origin, dev)`, if observed.
    #[inline]
    pub fn diagonal_of(&self, origin: usize, dev: usize) -> Option<usize> {
        self.cell_diagonal[origin * self.n_devs + dev]
    }
}

/// Align calendar-year / valuation-period diagonals across an asymmetric
/// [`AsymmetricAgeGrid`] (Shapland 2016).
///
/// Unlike square triangles where calendar diagonals follow `origin + dev`, an
/// asymmetric grid with stride ratio `s = origin_months / dev_months` aligns
/// cells with equal `origin_start_months[i] + dev_ages_months[j]` (e.g., on an
/// annual × semiannual grid, `(0, 2)` and `(1, 0)` share calendar month `18.0`).
pub fn calendar_diagonal_index(
    grid: &AsymmetricAgeGrid,
) -> Result<CalendarDiagonalMap, ValidationError> {
    if grid.n_origins == 0
        || grid.n_devs == 0
        || grid.observed_counts.len() != grid.n_origins
        || grid.origin_start_months.len() != grid.n_origins
        || grid.n_origins.checked_mul(grid.n_devs) != Some(grid.cell_ages_months.len())
    {
        return Err(ValidationError::ShapeMismatch);
    }
    if !grid.origin_months.is_finite() || !grid.dev_months.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if grid.origin_months <= 0.0 || grid.dev_months <= 0.0 {
        return Err(ValidationError::NonPositive("grid period months"));
    }
    let tol = 1e-6 * grid.dev_months.max(1.0);

    // Collect all observed (origin, dev, calendar_month) triples.
    let mut observed: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..grid.n_origins {
        let origin_start = grid.origin_start_months[i];
        if !origin_start.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        let count = grid.observed_counts[i].min(grid.n_devs);
        for j in 0..count {
            let age = grid.cell_ages_months[i * grid.n_devs + j];
            if age.is_finite() {
                let cal_m = origin_start + age;
                if !cal_m.is_finite() {
                    return Err(ValidationError::NotFinite);
                }
                observed.push((i, j, cal_m));
            }
        }
    }

    // Extract distinct sorted calendar valuation months within tolerance.
    let mut sorted_months: Vec<f64> = observed.iter().map(|&(_, _, m)| m).collect();
    sorted_months.sort_by(f64::total_cmp);
    let mut diagonal_months: Vec<f64> = Vec::new();
    for m in sorted_months {
        match diagonal_months.last() {
            Some(&prev) if (m - prev).abs() <= tol => {}
            _ => diagonal_months.push(m),
        }
    }

    let n_diag = diagonal_months.len();
    let mut cell_diagonal = vec![None; grid.n_origins * grid.n_devs];
    let mut cell_calendar_months = vec![f64::NAN; grid.n_origins * grid.n_devs];
    let mut diagonals: Vec<Vec<CalendarDiagonalCell>> = vec![Vec::new(); n_diag];

    for (origin, dev, cal_m) in observed {
        // Find matching diagonal bucket.
        let diag_idx = diagonal_months
            .iter()
            .position(|&dm| (dm - cal_m).abs() <= tol)
            .ok_or(ValidationError::InvalidParameter(
                "calendar month failed to match any diagonal bucket",
            ))?;
        let cal_period = ((cal_m - 1e-9).max(0.0) / grid.origin_months).floor() as usize;

        let flat = origin * grid.n_devs + dev;
        cell_diagonal[flat] = Some(diag_idx);
        cell_calendar_months[flat] = cal_m;
        diagonals[diag_idx].push(CalendarDiagonalCell {
            origin,
            dev,
            calendar_month: cal_m,
            diagonal_index: diag_idx,
            calendar_period_index: cal_period,
        });
    }

    Ok(CalendarDiagonalMap {
        n_origins: grid.n_origins,
        n_devs: grid.n_devs,
        diagonal_months,
        cell_diagonal,
        cell_calendar_months,
        diagonals,
    })
}

/// Summary of mean residuals grouped by true calendar diagonal on an asymmetric grid.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AsymmetricCalendarDiagnostics {
    /// Calendar valuation month for each diagonal bucket.
    pub diagonal_months: Vec<f64>,
    /// Mean residual on each calendar diagonal (`NaN` if no finite residuals on that diagonal).
    #[serde(deserialize_with = "deserialize_vec_f64_null_as_nan")]
    pub diagonal_means: Vec<f64>,
    /// Count of finite residuals on each calendar diagonal.
    pub diagonal_counts: Vec<usize>,
    /// Diagonal bucket index with the largest absolute mean residual.
    pub strongest_diagonal: Option<usize>,
    /// Root-mean-square diagonal bias across populated diagonals.
    pub rms_diagonal_bias: f64,
}

/// Aggregate a row-major residual grid (`n_origins * n_devs`, `NaN` for excluded/unobserved)
/// along the true asymmetric calendar diagonals from [`calendar_diagonal_index`].
pub fn diagnose_asymmetric_calendar_effects(
    map: &CalendarDiagonalMap,
    residuals: &[f64],
) -> Result<AsymmetricCalendarDiagnostics, ValidationError> {
    if residuals.len() != map.n_origins * map.n_devs {
        return Err(ValidationError::ShapeMismatch);
    }
    let n_diag = map.n_diagonals();
    let mut diagonal_means = Vec::with_capacity(n_diag);
    let mut diagonal_counts = Vec::with_capacity(n_diag);

    for bucket in &map.diagonals {
        let mut sum = 0.0;
        let mut cnt = 0usize;
        for cell in bucket {
            let r = residuals[cell.origin * map.n_devs + cell.dev];
            if r.is_finite() {
                sum += r;
                cnt += 1;
            }
        }
        diagonal_counts.push(cnt);
        diagonal_means.push(if cnt > 0 { sum / cnt as f64 } else { f64::NAN });
    }

    let strongest_diagonal = diagonal_means
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
        .map(|(idx, _)| idx);

    let mut sq_sum = 0.0;
    let mut finite_cnt = 0usize;
    for &m in &diagonal_means {
        if m.is_finite() {
            sq_sum += m * m;
            finite_cnt += 1;
        }
    }
    let rms_diagonal_bias = if finite_cnt > 0 {
        (sq_sum / finite_cnt as f64).sqrt()
    } else {
        0.0
    };

    Ok(AsymmetricCalendarDiagnostics {
        diagonal_months: map.diagonal_months.clone(),
        diagonal_means,
        diagonal_counts,
        strongest_diagonal,
        rms_diagonal_bias,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annual_by_semiannual_grid_with_6m_stub() {
        // 4 annual origins × 8 semiannual devs, valued at 6 months into AY 3 (month 42).
        let grid = asymmetric_age_grid(4, 8, 12.0, 6.0, 6.0, Some(6.0)).expect("grid");
        assert!((grid.stride_ratio - 2.0).abs() < 1e-12);
        assert_eq!(grid.observed_counts, vec![7, 5, 3, 1]);
        assert!((grid.valuation_month - 42.0).abs() < 1e-12);
        // Latest origin is a 6m stub -> age 6m, avg claim age 3m.
        assert!((grid.cell_age(3, 0) - 6.0).abs() < 1e-12);
        assert!((grid.cell_average_age(3, 0) - 3.0).abs() < 1e-12);
        // AY 2 at latest valuation (18m) -> full 12m origin -> avg claim age 12m.
        assert!((grid.cell_age(2, 2) - 18.0).abs() < 1e-12);
        assert!((grid.cell_average_age(2, 2) - 12.0).abs() < 1e-12);
    }

    #[test]
    fn calendar_diagonals_align_stride_2_to_1() {
        let grid = asymmetric_age_grid(3, 6, 12.0, 6.0, 6.0, None).expect("grid");
        let map = calendar_diagonal_index(&grid).expect("cal map");
        // AY 0 at 18m (dev 2) and AY 1 at 6m (dev 0) are both valued at calendar month 18.
        assert_eq!(map.diagonal_of(0, 2), map.diagonal_of(1, 0));
        // AY 0 at 30m (dev 4), AY 1 at 18m (dev 2), AY 2 at 6m (dev 0) share month 30.
        assert_eq!(map.diagonal_of(0, 4), map.diagonal_of(1, 2));
        assert_eq!(map.diagonal_of(1, 2), map.diagonal_of(2, 0));
    }

    #[test]
    fn stub_average_age_uniform_and_weighted() {
        assert!((stub_average_age(6.0, 6.0).unwrap() - 3.0).abs() < 1e-12);
        assert!((stub_average_age(6.0, 12.0).unwrap() - 3.0).abs() < 1e-12);
        assert!((stub_average_age(12.0, 12.0).unwrap() - 6.0).abs() < 1e-12);
        assert!((stub_average_age(18.0, 12.0).unwrap() - 12.0).abs() < 1e-12);

        // Uniform weights across 6 monthly slices should match uniform stub_average_age(6, 6) = 3.0.
        let w = [1.0; 6];
        let avg = stub_average_age_weighted(6.0, 1.0, &w).unwrap();
        assert!((avg - 3.0).abs() < 1e-12);
    }
}
