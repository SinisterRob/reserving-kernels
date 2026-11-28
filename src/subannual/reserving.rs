//! Asymmetric chain-ladder and interim-valuation Bornhuetter–Ferguson / Cape Cod
//! reserving kernels (Friedland 2010).
//!
//! * [`asymmetric_chain_ladder`] — computes link ratios and ultimates directly
//!   on rectangular `n_origins × n_devs` triangles with ragged/staggered
//!   diagonals (e.g., annual origins × semiannual/quarterly development at an
//!   interim valuation date).
//! * [`interim_bf`] — Bornhuetter–Ferguson using interpolated sub-annual CDFs
//!   and pro-rated stub-period expected ultimate / exposure.
//! * [`interim_cape_cod`] — Cape Cod using interpolated sub-annual CDFs and
//!   pro-rated stub-period earned premium.

use crate::deterministic::ldf::{
    LdfSelectionMethod, cdfs_from_ldfs, volume_weighted_ldfs_observed_counts,
};
use crate::preprocessing::ragged::{
    ex_hi_lo_observed_counts, simple_average_observed_counts,
    volume_weighted_last_n_observed_counts,
};
use crate::subannual::grid::AsymmetricAgeGrid;
use crate::validate::{
    MonotonicityPolicy, ValidationError, validate_nonnegative_finite_slice,
    validate_positive_finite_slice, validate_tail,
};

/// Result of [`asymmetric_chain_ladder`] on a rectangular/staggered triangle.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AsymmetricChainLadderProjection {
    /// Selected step-to-step LDFs (length `n_devs - 1`).
    pub ldfs: Vec<f64>,
    /// Cumulative development factors to ultimate by development column (length `n_devs`).
    pub cdfs: Vec<f64>,
    /// Applied tail factor (`>= 1.0`).
    pub tail: f64,
    /// Latest observed cumulative loss by origin.
    pub latest: Vec<f64>,
    /// Development column index of each origin's latest observed cell.
    pub latest_dev_indices: Vec<usize>,
    /// Valuation age in months of each origin's latest observed cell.
    pub latest_ages_months: Vec<f64>,
    /// Applied CDF-to-ultimate for each origin (`cdfs[latest_dev_indices[i]]`).
    pub origin_cdfs: Vec<f64>,
    /// Projected ultimate loss by origin.
    pub ultimates: Vec<f64>,
    /// Projected IBNR reserve (`ultimate - latest`) by origin.
    pub ibnr: Vec<f64>,
    /// Total latest observed cumulative loss across origins.
    pub total_latest: f64,
    /// Total projected ultimate loss across origins.
    pub total_ultimate: f64,
    /// Total projected IBNR reserve across origins.
    pub total_ibnr: f64,
}

/// Validate a rectangular cumulative buffer against an [`AsymmetricAgeGrid`].
pub(crate) fn validate_asymmetric_cumulative(
    cum: &[f64],
    grid: &AsymmetricAgeGrid,
    policy: MonotonicityPolicy,
) -> Result<(), ValidationError> {
    if grid.n_origins == 0 || grid.n_devs < 2 {
        return Err(ValidationError::ShapeMismatch);
    }
    if cum.len() != grid.n_origins * grid.n_devs
        || grid.observed_counts.len() != grid.n_origins
        || grid.cell_ages_months.len() != grid.n_origins * grid.n_devs
    {
        return Err(ValidationError::ShapeMismatch);
    }
    for i in 0..grid.n_origins {
        let cnt = grid.observed_counts[i];
        if cnt == 0 || cnt > grid.n_devs {
            return Err(ValidationError::ShapeMismatch);
        }
        let mut prev = None;
        for j in 0..cnt {
            let v = cum[i * grid.n_devs + j];
            if !v.is_finite() {
                return Err(ValidationError::NotFinite);
            }
            if v < 0.0 {
                return Err(ValidationError::Negative);
            }
            if policy == MonotonicityPolicy::Strict
                && let Some(p) = prev
                && v < p
            {
                return Err(ValidationError::NonMonotoneCumulative {
                    origin: i,
                    dev: j,
                    previous: p,
                    value: v,
                });
            }
            prev = Some(v);
        }
    }
    Ok(())
}

/// Compute median link ratios across arbitrary observed row widths.
fn median_observed_counts(cum: &[f64], n_o: usize, n_d: usize, counts: &[usize], out: &mut [f64]) {
    let n_links = n_d.saturating_sub(1);
    let mut buf = Vec::with_capacity(n_o);
    for j in 0..n_links {
        buf.clear();
        for i in 0..n_o {
            if counts[i] <= j + 1 {
                continue;
            }
            let x = cum[i * n_d + j];
            let y = cum[i * n_d + j + 1];
            if x > 0.0 && x.is_finite() && y.is_finite() {
                buf.push(y / x);
            }
        }
        if buf.is_empty() {
            out[j] = f64::NAN;
        } else {
            buf.sort_by(f64::total_cmp);
            let mid = buf.len() / 2;
            out[j] = if buf.len() % 2 == 1 {
                buf[mid]
            } else {
                f64::midpoint(buf[mid - 1], buf[mid])
            };
        }
    }
}

/// Compute link ratios and chain-ladder ultimates directly on an asymmetric
/// (e.g., annual-origin × semiannual/quarterly-dev) triangle with staggered or
/// ragged diagonals at interim valuation dates (Friedland 2010).
///
/// Only links `j -> j + 1` with at least one eligible origin pair (`counts[i] > j + 1`)
/// are estimated from data; any unobserved trailing columns beyond `max(observed_counts)`
/// are assigned neutral factor `1.0`.
pub fn asymmetric_chain_ladder(
    cum: &[f64],
    grid: &AsymmetricAgeGrid,
    method: LdfSelectionMethod,
    tail: f64,
    policy: MonotonicityPolicy,
) -> Result<AsymmetricChainLadderProjection, ValidationError> {
    validate_asymmetric_cumulative(cum, grid, policy)?;
    validate_tail(tail)?;

    let n_o = grid.n_origins;
    let n_d = grid.n_devs;
    let n_links = n_d - 1;
    let counts = &grid.observed_counts;
    let max_obs = counts.iter().copied().max().unwrap_or(0);
    if max_obs < 2 {
        return Err(ValidationError::InvalidParameter(
            "asymmetric_chain_ladder requires at least one origin with >= 2 observed cells",
        ));
    }

    let mut ldfs = vec![1.0; n_links];
    match method {
        LdfSelectionMethod::VolumeWeighted => {
            volume_weighted_ldfs_observed_counts(cum, n_o, n_d, counts, &mut ldfs);
        }
        LdfSelectionMethod::SimpleAverage => {
            simple_average_observed_counts(cum, n_o, n_d, counts, None, &mut ldfs);
        }
        LdfSelectionMethod::ExHiLo => {
            ex_hi_lo_observed_counts(cum, n_o, n_d, counts, &mut ldfs);
        }
        LdfSelectionMethod::Median => {
            median_observed_counts(cum, n_o, n_d, counts, &mut ldfs);
        }
        LdfSelectionMethod::LastNVolumeWeighted { last_n } => {
            if last_n == 0 {
                return Err(ValidationError::InvalidParameter("last_n must be >= 1"));
            }
            volume_weighted_last_n_observed_counts(cum, n_o, n_d, counts, last_n, &mut ldfs);
        }
        LdfSelectionMethod::LastNSimpleAverage { last_n } => {
            if last_n == 0 {
                return Err(ValidationError::InvalidParameter("last_n must be >= 1"));
            }
            simple_average_observed_counts(cum, n_o, n_d, counts, Some(last_n), &mut ldfs);
        }
        LdfSelectionMethod::Selected => {
            return Err(ValidationError::InvalidParameter(
                "Selected method requires explicit LDFs; use a computed LdfSelectionMethod",
            ));
        }
    }

    // Trailing unobserved columns beyond max_obs - 1 default to neutral 1.0;
    // observed links 0..(max_obs - 1) must be finite and strictly positive.
    for j in (max_obs - 1)..n_links {
        if !ldfs[j].is_finite() {
            ldfs[j] = 1.0;
        }
    }
    validate_positive_finite_slice(&ldfs, "asymmetric LDF")?;

    let mut cdfs = vec![0.0; n_d];
    cdfs_from_ldfs(&ldfs, tail, &mut cdfs);

    let mut latest = Vec::with_capacity(n_o);
    let mut latest_dev_indices = Vec::with_capacity(n_o);
    let mut latest_ages_months = Vec::with_capacity(n_o);
    let mut origin_cdfs = Vec::with_capacity(n_o);
    let mut ultimates = Vec::with_capacity(n_o);
    let mut ibnr = Vec::with_capacity(n_o);

    for i in 0..n_o {
        let d_i = counts[i] - 1;
        let lat = cum[i * n_d + d_i];
        let cdf = cdfs[d_i];
        let ult = lat * cdf;
        latest.push(lat);
        latest_dev_indices.push(d_i);
        latest_ages_months.push(grid.cell_age(i, d_i));
        origin_cdfs.push(cdf);
        ultimates.push(ult);
        ibnr.push(ult - lat);
    }

    let total_latest = latest.iter().sum();
    let total_ultimate = ultimates.iter().sum();
    let total_ibnr = ibnr.iter().sum();

    Ok(AsymmetricChainLadderProjection {
        ldfs,
        cdfs,
        tail,
        latest,
        latest_dev_indices,
        latest_ages_months,
        origin_cdfs,
        ultimates,
        ibnr,
        total_latest,
        total_ultimate,
        total_ibnr,
    })
}

/// Output from [`interim_bf`] (interim-valuation Bornhuetter–Ferguson with
/// pro-rated stub exposure and sub-annual CDFs).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InterimBfProjection {
    /// Latest observed cumulative losses by origin.
    pub latest: Vec<f64>,
    /// Applied sub-annual CDF-to-ultimate for each origin.
    pub cdfs: Vec<f64>,
    /// Earned exposure fraction `e_i in (0, 1]` for each origin.
    pub exposure_fractions: Vec<f64>,
    /// Pro-rated earned expected ultimate (`full_period_expected_ultimate[i] * e_i`).
    pub earned_expected_ultimate: Vec<f64>,
    /// Implied percent reported (`1 / cdfs[i]`).
    pub pct_reported: Vec<f64>,
    /// Projected IBNR reserve (`earned_expected_ultimate[i] * (1 - 1 / cdfs[i])`).
    pub ibnr: Vec<f64>,
    /// Projected earned ultimate (`latest[i] + ibnr[i]`).
    pub ultimates: Vec<f64>,
    /// Total latest observed cumulative losses.
    pub total_latest: f64,
    /// Total projected IBNR reserve.
    pub total_ibnr: f64,
    /// Total projected earned ultimate losses.
    pub total_ultimate: f64,
}

/// Run Bornhuetter–Ferguson at an interim valuation date using interpolated
/// sub-annual CDFs and pro-rated stub-period expected ultimate / exposure
/// (Friedland 2010).
///
/// Reduces identically to standard Bornhuetter–Ferguson when all
/// `exposure_fractions == 1.0`.
pub fn interim_bf(
    latest: &[f64],
    cdfs_at_origin: &[f64],
    full_period_expected_ultimate: &[f64],
    exposure_fractions: &[f64],
) -> Result<InterimBfProjection, ValidationError> {
    let n = latest.len();
    if n == 0
        || cdfs_at_origin.len() != n
        || full_period_expected_ultimate.len() != n
        || exposure_fractions.len() != n
    {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_nonnegative_finite_slice(latest)?;
    validate_positive_finite_slice(cdfs_at_origin, "origin CDF")?;
    validate_nonnegative_finite_slice(full_period_expected_ultimate)?;
    validate_positive_finite_slice(exposure_fractions, "exposure_fractions")?;
    if exposure_fractions.iter().any(|&e| e > 1.0 + 1e-9) {
        return Err(ValidationError::InvalidParameter(
            "exposure_fractions must lie in (0, 1]",
        ));
    }

    let mut earned_expected_ultimate = Vec::with_capacity(n);
    let mut pct_reported = Vec::with_capacity(n);
    let mut ibnr = Vec::with_capacity(n);
    let mut ultimates = Vec::with_capacity(n);

    for i in 0..n {
        let earned_exp = full_period_expected_ultimate[i] * exposure_fractions[i];
        let p = 1.0 / cdfs_at_origin[i];
        let r = earned_exp * (1.0 - p);
        let u = latest[i] + r;
        earned_expected_ultimate.push(earned_exp);
        pct_reported.push(p);
        ibnr.push(r);
        ultimates.push(u);
    }

    Ok(InterimBfProjection {
        latest: latest.to_vec(),
        cdfs: cdfs_at_origin.to_vec(),
        exposure_fractions: exposure_fractions.to_vec(),
        earned_expected_ultimate,
        pct_reported,
        ibnr: ibnr.clone(),
        ultimates: ultimates.clone(),
        total_latest: latest.iter().sum(),
        total_ibnr: ibnr.iter().sum(),
        total_ultimate: ultimates.iter().sum(),
    })
}

/// Output from [`interim_cape_cod`] (interim-valuation Cape Cod with pro-rated
/// stub earned premium and sub-annual CDFs).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InterimCapeCodProjection {
    /// Latest observed cumulative losses by origin.
    pub latest: Vec<f64>,
    /// Applied sub-annual CDF-to-ultimate for each origin.
    pub cdfs: Vec<f64>,
    /// Earned exposure fraction `e_i in (0, 1]` for each origin.
    pub exposure_fractions: Vec<f64>,
    /// Pro-rated earned premium (`full_period_premium[i] * e_i`).
    pub earned_premium: Vec<f64>,
    /// Used-up earned premium (`earned_premium[i] / cdfs[i]`).
    pub used_up_premium: Vec<f64>,
    /// Pooled Cape Cod expected loss ratio `sum(latest) / sum(used_up_premium)`.
    pub elr: f64,
    /// Projected IBNR reserve (`earned_premium[i] * elr * (1 - 1 / cdfs[i])`).
    pub ibnr: Vec<f64>,
    /// Projected earned ultimate (`latest[i] + ibnr[i]`).
    pub ultimates: Vec<f64>,
    /// Total latest observed cumulative losses.
    pub total_latest: f64,
    /// Total projected IBNR reserve.
    pub total_ibnr: f64,
    /// Total projected earned ultimate losses.
    pub total_ultimate: f64,
}

/// Run Cape Cod at an interim valuation date using interpolated sub-annual CDFs
/// and pro-rated stub-period earned premium (Friedland 2010).
///
/// Reduces identically to standard Cape Cod when all `exposure_fractions == 1.0`.
pub fn interim_cape_cod(
    latest: &[f64],
    cdfs_at_origin: &[f64],
    full_period_premium: &[f64],
    exposure_fractions: &[f64],
) -> Result<InterimCapeCodProjection, ValidationError> {
    let n = latest.len();
    if n == 0
        || cdfs_at_origin.len() != n
        || full_period_premium.len() != n
        || exposure_fractions.len() != n
    {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_nonnegative_finite_slice(latest)?;
    validate_positive_finite_slice(cdfs_at_origin, "origin CDF")?;
    validate_positive_finite_slice(full_period_premium, "premium")?;
    validate_positive_finite_slice(exposure_fractions, "exposure_fractions")?;
    if exposure_fractions.iter().any(|&e| e > 1.0 + 1e-9) {
        return Err(ValidationError::InvalidParameter(
            "exposure_fractions must lie in (0, 1]",
        ));
    }

    let mut earned_premium = Vec::with_capacity(n);
    let mut used_up_premium = Vec::with_capacity(n);
    for i in 0..n {
        let ep = full_period_premium[i] * exposure_fractions[i];
        let uup = ep / cdfs_at_origin[i];
        earned_premium.push(ep);
        used_up_premium.push(uup);
    }

    let total_latest: f64 = latest.iter().sum();
    let total_uup: f64 = used_up_premium.iter().sum();
    if !(total_uup > 0.0) || !total_uup.is_finite() {
        return Err(ValidationError::DivideByZero("used-up premium"));
    }
    let elr = total_latest / total_uup;

    let mut ibnr = Vec::with_capacity(n);
    let mut ultimates = Vec::with_capacity(n);
    for i in 0..n {
        let r = earned_premium[i] * elr * (1.0 - 1.0 / cdfs_at_origin[i]);
        let u = latest[i] + r;
        ibnr.push(r);
        ultimates.push(u);
    }

    Ok(InterimCapeCodProjection {
        latest: latest.to_vec(),
        cdfs: cdfs_at_origin.to_vec(),
        exposure_fractions: exposure_fractions.to_vec(),
        earned_premium,
        used_up_premium,
        elr,
        ibnr: ibnr.clone(),
        ultimates: ultimates.clone(),
        total_latest,
        total_ibnr: ibnr.iter().sum(),
        total_ultimate: ultimates.iter().sum(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subannual::grid::asymmetric_age_grid;

    #[test]
    fn asymmetric_cl_recovers_exact_synthetic_pattern() {
        // 4 annual origins × 6 semiannual devs, valued at full year 4 (48m) -> counts = [6, 6, 4, 2].
        let grid = asymmetric_age_grid(4, 6, 12.0, 6.0, 6.0, None).unwrap();
        assert_eq!(grid.observed_counts, vec![6, 6, 4, 2]);
        let true_ldfs = [1.5, 1.3, 1.15, 1.08, 1.03];
        let base_first = [100.0, 120.0, 140.0, 160.0];
        let mut cum = vec![f64::NAN; 4 * 6];
        for i in 0..4 {
            let mut val = base_first[i];
            for j in 0..grid.observed_counts[i] {
                if j > 0 {
                    val *= true_ldfs[j - 1];
                }
                cum[i * 6 + j] = val;
            }
        }

        let res = asymmetric_chain_ladder(
            &cum,
            &grid,
            LdfSelectionMethod::VolumeWeighted,
            1.0,
            MonotonicityPolicy::Strict,
        )
        .unwrap();
        for (got, expected) in res.ldfs.iter().zip(true_ldfs.iter()) {
            assert!((got - expected).abs() < 1e-12);
        }
        assert!(res.total_ibnr > 0.0);
    }

    #[test]
    fn interim_bf_and_cape_cod_pro_rate_stub_row() {
        let latest = [1000.0, 800.0, 300.0];
        let cdfs = [1.0, 1.25, 2.0];
        let full_exp = [1000.0, 1000.0, 1000.0];
        let fracs = [1.0, 1.0, 0.5];

        let bf = interim_bf(&latest, &cdfs, &full_exp, &fracs).unwrap();
        // Origin 2: earned expected = 500, pct unreported = 0.5 -> IBNR = 250, Ult = 550.
        assert!((bf.earned_expected_ultimate[2] - 500.0).abs() < 1e-12);
        assert!((bf.ibnr[2] - 250.0).abs() < 1e-12);
        assert!((bf.ultimates[2] - 550.0).abs() < 1e-12);

        let cc = interim_cape_cod(&latest, &cdfs, &full_exp, &fracs).unwrap();
        // Used-up premiums: 1000/1 + 1000/1.25 + 500/2 = 1000 + 800 + 250 = 2050.
        // ELR = 2100 / 2050.
        assert!((cc.elr - (2100.0 / 2050.0)).abs() < 1e-12);
        assert!((cc.ibnr[2] - 500.0 * cc.elr * 0.5).abs() < 1e-12);
    }
}
