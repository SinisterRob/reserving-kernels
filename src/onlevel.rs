//! Premium on-level transforms.

use crate::validate::{ValidationError, validate_nonnegative_finite_slice};

fn validate_rate_change(rate_change: f64) -> Result<(), ValidationError> {
    if !rate_change.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if rate_change <= -1.0 {
        return Err(ValidationError::Negative);
    }
    Ok(())
}

/// Multiply premium by origin-specific on-level factors.
pub fn onlevel_premium(
    premium: &[f64],
    onlevel_factors: &[f64],
) -> Result<Vec<f64>, ValidationError> {
    if premium.len() != onlevel_factors.len() {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_nonnegative_finite_slice(premium)?;
    validate_nonnegative_finite_slice(onlevel_factors)?;
    if onlevel_factors.iter().any(|factor| *factor <= 0.0) {
        return Err(ValidationError::DivideByZero("on-level factor"));
    }

    Ok(premium
        .iter()
        .zip(onlevel_factors.iter())
        .map(|(&value, &factor)| value * factor)
        .collect())
}

/// Chain sequential percentage rate changes into one cumulative rate-level factor.
pub fn rate_change_chain(rate_changes: &[f64]) -> Result<f64, ValidationError> {
    let mut factor = 1.0;
    for &rate_change in rate_changes {
        validate_rate_change(rate_change)?;
        factor *= 1.0 + rate_change;
    }
    Ok(factor)
}

/// Parallelogram-style on-level factor from earned weights and sequential rate changes.
///
/// `earned_weights` should sum to roughly `1.0`; the function uses the supplied
/// weights as-is and returns the factor to current rate level.
pub fn parallelogram_onlevel_factor(
    rate_changes: &[f64],
    earned_weights: &[f64],
) -> Result<f64, ValidationError> {
    if rate_changes.len() != earned_weights.len() {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_nonnegative_finite_slice(earned_weights)?;
    let current_level = rate_change_chain(rate_changes)?;

    let mut prior_level = 1.0;
    let mut average_level = 0.0;
    for (&weight, &rate_change) in earned_weights.iter().zip(rate_changes.iter()) {
        validate_rate_change(rate_change)?;
        average_level += weight * prior_level;
        prior_level *= 1.0 + rate_change;
    }

    if average_level <= 0.0 {
        return Err(ValidationError::DivideByZero("average rate level"));
    }
    Ok(current_level / average_level)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * (1.0 + b.abs())
    }

    #[test]
    fn chains_rate_changes() {
        let factor = rate_change_chain(&[0.10, 0.05]).expect("factor");
        assert!(close(factor, 1.155));
    }

    #[test]
    fn onlevels_premium_originwise() {
        let adjusted = onlevel_premium(&[100.0, 200.0], &[1.1, 0.9]).expect("premium");
        assert!(close(adjusted[0], 110.0));
        assert!(close(adjusted[1], 180.0));
    }

    #[test]
    fn parallelogram_factor_exceeds_one_for_positive_changes() {
        let factor =
            parallelogram_onlevel_factor(&[0.10, 0.05], &[0.5, 0.5]).expect("on-level factor");
        assert!(factor > 1.0);
    }
}
