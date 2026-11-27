//! Pre-12-month stub extrapolation and exposure-scaling kernels.
//!
//! Implements pre-12-month stub CDF/LDF extrapolation and exposure-vs-reporting
//! separation from:
//!
//! * **Bloom (2015)**, *"Interpolation and Extrapolation of Loss Development
//!   Factors for Interim Valuations"*, Casualty Actuarial Society:
//!   * [`stub_plus12`] — shifted power-law (`+12` month axis shift).
//!   * [`stub_power_ratio`] — early annual factor ratio raised to fractional stub power.
//!   * [`stub_natural_log`] — conservative log-linear early-age extrapolation.
//! * **Friedland (2010)**, *Estimating Unpaid Claims Using Basic Techniques*,
//!   Ch. 7–10, & **Bloom (2015)**:
//!   * [`stub_exposure_scale`] — separates partial-year earned exposure growth
//!     from pure claim reporting lag on stub rows.

use crate::subannual::grid::stub_average_age;
use crate::validate::ValidationError;

/// Result of extrapolating a pre-12-month stub CDF and stub-to-12-month LDF.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StubExtrapolationResult {
    /// Target stub valuation age in months (`0 < stub_age_months <= first_annual_age`).
    pub stub_age_months: f64,
    /// Extrapolated age-to-ultimate CDF at `stub_age_months`.
    pub stub_cdf: f64,
    /// Implied stub-to-first-annual LDF (`stub_cdf / cdf_12`).
    pub stub_to_12_ldf: f64,
    /// Implied percent reported at `stub_age_months` (`1 / stub_cdf`).
    pub pct_reported: f64,
    /// Effective power exponent used by the extrapolation rule.
    pub exponent: f64,
}

/// Bloom (2015) **"Plus 12"** pre-12-month stub extrapolation.
///
/// Shifts the development age axis by `+12` months before applying power-law
/// scaling through `(12, cdf_12)` and `(24, cdf_24)`:
///
/// ```text
/// exponent = ln(24 / (s + 12)) / ln(36 / 24)
/// CDF(s)   = cdf_12 * (cdf_12 / cdf_24)^exponent
/// ```
///
/// Shifting by `+12` keeps `CDF(s)` finite as `s -> 0+` while matching `cdf_12`
/// identically at `s = 12`.
pub fn stub_plus12(
    stub_age_months: f64,
    cdf_12: f64,
    cdf_24: f64,
) -> Result<StubExtrapolationResult, ValidationError> {
    if !stub_age_months.is_finite() || !cdf_12.is_finite() || !cdf_24.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if stub_age_months <= 0.0 || stub_age_months > 12.0 + 1e-9 {
        return Err(ValidationError::InvalidParameter(
            "stub_age_months must be in (0, 12]",
        ));
    }
    if cdf_12 <= 0.0 || cdf_24 <= 0.0 {
        return Err(ValidationError::DivideByZero("annual CDF"));
    }

    let s = stub_age_months.min(12.0);
    let f12_24 = cdf_12 / cdf_24;
    let exponent = (24.0 / (s + 12.0)).ln() / (36.0_f64 / 24.0_f64).ln();
    let stub_to_12_ldf = f12_24.powf(exponent);
    let stub_cdf = cdf_12 * stub_to_12_ldf;

    Ok(StubExtrapolationResult {
        stub_age_months: s,
        stub_cdf,
        stub_to_12_ldf,
        pct_reported: 1.0 / stub_cdf,
        exponent,
    })
}

/// Bloom (2015) **Power Ratio** pre-12-month stub extrapolation.
///
/// Estimates the stub-to-12-month LDF from the ratio of the first two annual
/// development factors (`ldf_12_24` and `ldf_24_36`) raised to the remaining
/// fractional stub duration `u = (12 - s) / 12`:
///
/// ```text
/// f_0          = ldf_12_24 * (ldf_12_24 / ldf_24_36)
/// LDF(s -> 12) = f_0^((12 - s) / 12)
/// CDF(s)       = cdf_12 * LDF(s -> 12)
/// ```
///
/// Reduces identically to constant-rate exponential extrapolation when
/// `ldf_12_24 == ldf_24_36`, and produces steeper early-age development when
/// `ldf_12_24 > ldf_24_36`.
pub fn stub_power_ratio(
    stub_age_months: f64,
    cdf_12: f64,
    ldf_12_24: f64,
    ldf_24_36: f64,
) -> Result<StubExtrapolationResult, ValidationError> {
    if !stub_age_months.is_finite()
        || !cdf_12.is_finite()
        || !ldf_12_24.is_finite()
        || !ldf_24_36.is_finite()
    {
        return Err(ValidationError::NotFinite);
    }
    if stub_age_months <= 0.0 || stub_age_months > 12.0 + 1e-9 {
        return Err(ValidationError::InvalidParameter(
            "stub_age_months must be in (0, 12]",
        ));
    }
    if cdf_12 <= 0.0 || ldf_12_24 <= 0.0 || ldf_24_36 <= 0.0 {
        return Err(ValidationError::DivideByZero("CDF or LDF"));
    }

    let s = stub_age_months.min(12.0);
    let exponent = (12.0 - s) / 12.0;
    let base_factor = ldf_12_24 * (ldf_12_24 / ldf_24_36);
    let stub_to_12_ldf = base_factor.powf(exponent);
    let stub_cdf = cdf_12 * stub_to_12_ldf;

    Ok(StubExtrapolationResult {
        stub_age_months: s,
        stub_cdf,
        stub_to_12_ldf,
        pct_reported: 1.0 / stub_cdf,
        exponent,
    })
}

/// Bloom (2015) **Natural Log** pre-12-month stub extrapolation.
///
/// Fits a log-linear relationship `ln(CDF(t)) = a + b · ln(t)` through
/// `(12, cdf_12)` and `(24, cdf_24)`:
///
/// ```text
/// exponent = ln(12 / s) / ln(2)
/// CDF(s)   = cdf_12 * (cdf_12 / cdf_24)^exponent
/// ```
///
/// Produces more conservative (higher) pre-12-month stub LDFs than
/// [`stub_plus12`] for all `s in (0, 12)` whenever `cdf_12 > cdf_24`.
pub fn stub_natural_log(
    stub_age_months: f64,
    cdf_12: f64,
    cdf_24: f64,
) -> Result<StubExtrapolationResult, ValidationError> {
    if !stub_age_months.is_finite() || !cdf_12.is_finite() || !cdf_24.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if stub_age_months <= 0.0 || stub_age_months > 12.0 + 1e-9 {
        return Err(ValidationError::InvalidParameter(
            "stub_age_months must be in (0, 12]",
        ));
    }
    if cdf_12 <= 0.0 || cdf_24 <= 0.0 {
        return Err(ValidationError::DivideByZero("annual CDF"));
    }

    let s = stub_age_months.min(12.0);
    let f12_24 = cdf_12 / cdf_24;
    let exponent = (12.0 / s).ln() / 2.0_f64.ln();
    let stub_to_12_ldf = f12_24.powf(exponent);
    let stub_cdf = cdf_12 * stub_to_12_ldf;

    Ok(StubExtrapolationResult {
        stub_age_months: s,
        stub_cdf,
        stub_to_12_ldf,
        pct_reported: 1.0 / stub_cdf,
        exponent,
    })
}

/// Output from [`stub_exposure_scale`] separating partial-year earned exposure
/// growth from pure claim reporting lag.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StubExposureScaleResult {
    /// Earned exposure fraction `e = stub_duration_months / full_period_months` in `(0, 1]`.
    pub exposure_fraction: f64,
    /// Pro-rated expected ultimate loss on earned-to-date exposure (`full_period_expected_loss * e`).
    pub earned_expected_loss: f64,
    /// Pure claim reporting-lag CDF developing earned-to-date losses to their earned ultimate.
    pub pure_reporting_cdf: f64,
    /// Combined full-year CDF (`pure_reporting_cdf / e`) developing stub observed losses
    /// to a full-year ultimate (incorporating both reporting lag and unearned exposure growth).
    pub full_year_combined_cdf: f64,
    /// Percent of earned-to-date ultimate already reported (`1 / pure_reporting_cdf`).
    pub earned_percent_reported: f64,
    /// Effective average claim age in months for the stub under uniform exposure (`stub_duration / 2`).
    pub stub_average_age_months: f64,
}

/// Separate partial-year earned exposure growth from pure claim reporting lag
/// for an interim stub origin period (Friedland 2010; Bloom 2015).
///
/// # Parameters
/// * `full_period_expected_loss` — a priori expected ultimate loss for a full
///   origin period (e.g., 12 months).
/// * `pure_reporting_cdf` — reporting-lag CDF at the stub's effective maturity
///   (`>= 1.0`), developing earned-to-date claims to their ultimate value.
/// * `stub_duration_months` — elapsed earned months in the stub period (`0 < d <= full_period_months`).
/// * `full_period_months` — full origin period length in months (e.g., `12.0`).
///
/// If you instead have an empirical **combined** full-year stub CDF (which mixes
/// unearned exposure growth with reporting lag), pass `combined_cdf * (d / full_period_months)`
/// as `pure_reporting_cdf` (or use [`decompose_combined_stub_cdf`]).
pub fn stub_exposure_scale(
    full_period_expected_loss: f64,
    pure_reporting_cdf: f64,
    stub_duration_months: f64,
    full_period_months: f64,
) -> Result<StubExposureScaleResult, ValidationError> {
    if !full_period_expected_loss.is_finite()
        || !pure_reporting_cdf.is_finite()
        || !stub_duration_months.is_finite()
        || !full_period_months.is_finite()
    {
        return Err(ValidationError::NotFinite);
    }
    if full_period_expected_loss < 0.0 {
        return Err(ValidationError::Negative);
    }
    if pure_reporting_cdf <= 0.0 {
        return Err(ValidationError::DivideByZero("pure_reporting_cdf"));
    }
    if full_period_months <= 0.0
        || stub_duration_months <= 0.0
        || stub_duration_months > full_period_months + 1e-9
    {
        return Err(ValidationError::InvalidParameter(
            "stub_duration_months must be in (0, full_period_months]",
        ));
    }

    let d = stub_duration_months.min(full_period_months);
    let exposure_fraction = d / full_period_months;
    let earned_expected_loss = full_period_expected_loss * exposure_fraction;
    let full_year_combined_cdf = pure_reporting_cdf / exposure_fraction;
    let avg_age = stub_average_age(d, full_period_months)?;

    Ok(StubExposureScaleResult {
        exposure_fraction,
        earned_expected_loss,
        pure_reporting_cdf,
        full_year_combined_cdf,
        earned_percent_reported: 1.0 / pure_reporting_cdf,
        stub_average_age_months: avg_age,
    })
}

/// Decompose an empirical full-year combined stub CDF into its pure reporting-lag
/// component (`combined_stub_cdf * exposure_fraction`) and earned-exposure scaling.
pub fn decompose_combined_stub_cdf(
    full_period_expected_loss: f64,
    combined_stub_cdf: f64,
    stub_duration_months: f64,
    full_period_months: f64,
) -> Result<StubExposureScaleResult, ValidationError> {
    if !combined_stub_cdf.is_finite() || !stub_duration_months.is_finite() || !full_period_months.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if combined_stub_cdf <= 0.0 || full_period_months <= 0.0 || stub_duration_months <= 0.0 {
        return Err(ValidationError::InvalidParameter(
            "combined_stub_cdf, stub_duration_months, and full_period_months must be positive",
        ));
    }
    let exposure_fraction = (stub_duration_months.min(full_period_months)) / full_period_months;
    let pure_reporting_cdf = combined_stub_cdf * exposure_fraction;
    stub_exposure_scale(
        full_period_expected_loss,
        pure_reporting_cdf,
        stub_duration_months,
        full_period_months,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_methods_match_cdf12_at_12_months() {
        let cdf_12 = 2.40;
        let cdf_24 = 1.50;
        let ldf_12_24 = cdf_12 / cdf_24;
        let ldf_24_36 = 1.25;

        let p12 = stub_plus12(12.0, cdf_12, cdf_24).unwrap();
        let pr = stub_power_ratio(12.0, cdf_12, ldf_12_24, ldf_24_36).unwrap();
        let nl = stub_natural_log(12.0, cdf_12, cdf_24).unwrap();

        assert!((p12.stub_cdf - cdf_12).abs() < 1e-12);
        assert!((pr.stub_cdf - cdf_12).abs() < 1e-12);
        assert!((nl.stub_cdf - cdf_12).abs() < 1e-12);
    }

    #[test]
    fn natural_log_is_more_conservative_than_plus12() {
        let cdf_12 = 2.40;
        let cdf_24 = 1.50;
        for s in [3.0, 6.0, 9.0] {
            let p12 = stub_plus12(s, cdf_12, cdf_24).unwrap();
            let nl = stub_natural_log(s, cdf_12, cdf_24).unwrap();
            assert!(
                nl.stub_cdf > p12.stub_cdf,
                "at {s}m: natural_log ({}) should exceed plus12 ({})",
                nl.stub_cdf,
                p12.stub_cdf
            );
        }
    }

    #[test]
    fn stub_exposure_scale_separates_exposure_and_reporting() {
        let res = stub_exposure_scale(1_000.0, 2.0, 6.0, 12.0).unwrap();
        assert!((res.exposure_fraction - 0.5).abs() < 1e-12);
        assert!((res.earned_expected_loss - 500.0).abs() < 1e-12);
        assert!((res.full_year_combined_cdf - 4.0).abs() < 1e-12);
        assert!((res.stub_average_age_months - 3.0).abs() < 1e-12);

        let roundtrip = decompose_combined_stub_cdf(1_000.0, 4.0, 6.0, 12.0).unwrap();
        assert!((roundtrip.pure_reporting_cdf - 2.0).abs() < 1e-12);
    }
}
