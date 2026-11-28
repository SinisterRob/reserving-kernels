//! Shared a priori expected-ultimate builders for BF-style methods.

use crate::error::ReservingError;
use crate::validate::{validate_nonnegative_finite_slice, validate_nonnegative_finite_vector};

/// Provenance for a built a priori vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AprioriSource {
    /// One ELR applied to all origins.
    ScalarElr,
    /// ELR supplied by origin.
    ByOriginElr,
    /// Expected ultimate supplied directly.
    ExpectedUltimate,
    /// Pure premium multiplied by exposures.
    PurePremium,
    /// Exposure × frequency × severity by origin.
    FrequencySeverity,
}

/// Borrowed a priori input forms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AprioriInput<'a> {
    /// Same expected loss ratio for every origin.
    ScalarElr(f64),
    /// Expected loss ratio by origin.
    ByOriginElr(&'a [f64]),
    /// Expected ultimate by origin.
    ExpectedUltimate(&'a [f64]),
    /// Exposure vector with one scalar pure premium.
    PurePremium {
        /// Exposure by origin.
        exposures: &'a [f64],
        /// Pure premium assumption.
        pure_premium: f64,
    },
    /// Exposure × frequency × severity by origin.
    FrequencySeverity {
        /// Exposure by origin.
        exposures: &'a [f64],
        /// Frequency by origin.
        frequency: &'a [f64],
        /// Severity by origin.
        severity: &'a [f64],
    },
}

/// Built expected-ultimate vector plus provenance.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AprioriVector {
    /// Expected ultimate loss by origin.
    pub expected_ultimate: Vec<f64>,
    /// Source type used to build it.
    pub source: AprioriSource,
}

fn validate_scalar_nonnegative(value: f64) -> Result<(), ReservingError> {
    if !value.is_finite() {
        return Err(ReservingError::NotFinite);
    }
    if value < 0.0 {
        return Err(ReservingError::Negative);
    }
    Ok(())
}

fn multiply_two(a: &[f64], b: &[f64], expected_len: usize) -> Result<Vec<f64>, ReservingError> {
    validate_nonnegative_finite_vector(a, expected_len)?;
    validate_nonnegative_finite_vector(b, expected_len)?;
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).collect())
}

fn multiply_three(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    expected_len: usize,
) -> Result<Vec<f64>, ReservingError> {
    validate_nonnegative_finite_vector(a, expected_len)?;
    validate_nonnegative_finite_vector(b, expected_len)?;
    validate_nonnegative_finite_vector(c, expected_len)?;
    Ok(a.iter()
        .zip(b.iter())
        .zip(c.iter())
        .map(|((x, y), z)| x * y * z)
        .collect())
}

/// Build an expected-ultimate vector from one of the supported a priori forms.
pub fn build_apriori(
    premium_or_exposure: &[f64],
    input: AprioriInput<'_>,
) -> Result<AprioriVector, ReservingError> {
    let n = premium_or_exposure.len();
    validate_nonnegative_finite_vector(premium_or_exposure, n)?;

    let (expected_ultimate, source) = match input {
        AprioriInput::ScalarElr(elr) => {
            validate_scalar_nonnegative(elr)?;
            (
                premium_or_exposure
                    .iter()
                    .map(|value| value * elr)
                    .collect(),
                AprioriSource::ScalarElr,
            )
        }
        AprioriInput::ByOriginElr(elr) => (
            multiply_two(premium_or_exposure, elr, n)?,
            AprioriSource::ByOriginElr,
        ),
        AprioriInput::ExpectedUltimate(expected) => {
            validate_nonnegative_finite_vector(expected, n)?;
            (expected.to_vec(), AprioriSource::ExpectedUltimate)
        }
        AprioriInput::PurePremium {
            exposures,
            pure_premium,
        } => {
            validate_scalar_nonnegative(pure_premium)?;
            validate_nonnegative_finite_vector(exposures, n)?;
            (
                exposures.iter().map(|value| value * pure_premium).collect(),
                AprioriSource::PurePremium,
            )
        }
        AprioriInput::FrequencySeverity {
            exposures,
            frequency,
            severity,
        } => (
            multiply_three(exposures, frequency, severity, n)?,
            AprioriSource::FrequencySeverity,
        ),
    };

    validate_nonnegative_finite_slice(&expected_ultimate)?;
    Ok(AprioriVector {
        expected_ultimate,
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_scalar_elr_apriori() {
        let premiums = [100.0, 200.0, 300.0];
        let apriori = build_apriori(&premiums, AprioriInput::ScalarElr(0.6)).expect("apriori");

        assert_eq!(apriori.source, AprioriSource::ScalarElr);
        assert_eq!(apriori.expected_ultimate, vec![60.0, 120.0, 180.0]);
    }

    #[test]
    fn builds_by_origin_elr_apriori() {
        let premiums = [100.0, 200.0, 300.0];
        let elr = [0.5, 0.6, 0.7];
        let apriori = build_apriori(&premiums, AprioriInput::ByOriginElr(&elr)).expect("apriori");

        assert_eq!(apriori.source, AprioriSource::ByOriginElr);
        assert_eq!(apriori.expected_ultimate, vec![50.0, 120.0, 210.0]);
    }

    #[test]
    fn builds_expected_ultimate_passthrough() {
        let basis = [1.0, 1.0, 1.0];
        let expected = [80.0, 90.0, 100.0];
        let apriori =
            build_apriori(&basis, AprioriInput::ExpectedUltimate(&expected)).expect("apriori");

        assert_eq!(apriori.source, AprioriSource::ExpectedUltimate);
        assert_eq!(apriori.expected_ultimate, expected);
    }

    #[test]
    fn builds_pure_premium_apriori() {
        let basis = [1.0, 1.0, 1.0];
        let exposures = [10.0, 12.0, 14.0];
        let apriori = build_apriori(
            &basis,
            AprioriInput::PurePremium {
                exposures: &exposures,
                pure_premium: 25.0,
            },
        )
        .expect("apriori");

        assert_eq!(apriori.source, AprioriSource::PurePremium);
        assert_eq!(apriori.expected_ultimate, vec![250.0, 300.0, 350.0]);
    }

    #[test]
    fn builds_frequency_severity_apriori() {
        let basis = [1.0, 1.0, 1.0];
        let exposures = [100.0, 120.0, 140.0];
        let frequency = [0.10, 0.11, 0.12];
        let severity = [50.0, 55.0, 60.0];
        let apriori = build_apriori(
            &basis,
            AprioriInput::FrequencySeverity {
                exposures: &exposures,
                frequency: &frequency,
                severity: &severity,
            },
        )
        .expect("apriori");

        assert_eq!(apriori.source, AprioriSource::FrequencySeverity);
        assert_eq!(apriori.expected_ultimate, vec![500.0, 726.0, 1008.0]);
    }

    #[test]
    fn rejects_negative_inputs() {
        let premiums = [100.0, 200.0, 300.0];
        assert_eq!(
            build_apriori(&premiums, AprioriInput::ScalarElr(-0.1)),
            Err(ReservingError::Negative)
        );
    }
}
