//! Premium on-level transforms.

use crate::error::ReservingError;
use crate::validate::validate_nonnegative_finite_slice;

fn validate_rate_change(rate_change: f64) -> Result<(), ReservingError> {
    if !rate_change.is_finite() {
        return Err(ReservingError::NotFinite);
    }
    if rate_change <= -1.0 {
        return Err(ReservingError::Negative);
    }
    Ok(())
}

/// Multiply premium by origin-specific on-level factors.
pub fn onlevel_premium(
    premium: &[f64],
    onlevel_factors: &[f64],
) -> Result<Vec<f64>, ReservingError> {
    if premium.len() != onlevel_factors.len() {
        return Err(ReservingError::ShapeMismatch);
    }
    validate_nonnegative_finite_slice(premium)?;
    validate_nonnegative_finite_slice(onlevel_factors)?;
    if onlevel_factors.iter().any(|factor| *factor <= 0.0) {
        return Err(ReservingError::DivideByZero("on-level factor"));
    }

    Ok(premium
        .iter()
        .zip(onlevel_factors.iter())
        .map(|(&value, &factor)| value * factor)
        .collect())
}

/// Chain sequential percentage rate changes into one cumulative rate-level factor.
pub fn rate_change_chain(rate_changes: &[f64]) -> Result<f64, ReservingError> {
    let mut factor = 1.0;
    for &rate_change in rate_changes {
        validate_rate_change(rate_change)?;
        factor *= 1.0 + rate_change;
    }
    Ok(factor)
}

/// Parallelogram-style on-level factor from sequential rate changes and earned
/// exposure weights across each rate-level regime.
///
/// For `k` sequential rate changes `r_1, ..., r_k`, there are `k + 1` rate-level
/// regimes: `L_0 = 1.0`, `L_1 = 1 + r_1`, ..., `L_k = prod_{j=1}^k (1 + r_j)`.
/// Pass `earned_weights` of length `k + 1` corresponding to those `k + 1` regimes
/// (weights are automatically normalized by their sum so any non-negative
/// exposure/area weights work). Returns `L_k / sum_j (w_j L_j)`.
pub fn parallelogram_onlevel_factor(
    rate_changes: &[f64],
    earned_weights: &[f64],
) -> Result<f64, ReservingError> {
    if earned_weights.len() != rate_changes.len() + 1 {
        return Err(ReservingError::ShapeMismatch);
    }
    validate_nonnegative_finite_slice(earned_weights)?;

    let mut level = 1.0;
    let mut weighted_level = earned_weights[0] * level;
    let mut weight_sum = earned_weights[0];

    for (&rate_change, &weight) in rate_changes.iter().zip(earned_weights[1..].iter()) {
        validate_rate_change(rate_change)?;
        level *= 1.0 + rate_change;
        weighted_level += weight * level;
        weight_sum += weight;
    }

    if !level.is_finite() || !weighted_level.is_finite() || !weight_sum.is_finite() {
        return Err(ReservingError::NotFinite);
    }
    if weight_sum <= 0.0 || weighted_level <= 0.0 {
        return Err(ReservingError::DivideByZero("average rate level"));
    }
    let average_level = weighted_level / weight_sum;
    Ok(level / average_level)
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
    fn parallelogram_factor_matches_exact_weighted_regime_levels() {
        // Regimes: L0 = 1.0 (weight 0.25), L1 = 1.10 (weight 0.50), L2 = 1.155 (weight 0.25)
        // Average level = 0.25*1.0 + 0.50*1.10 + 0.25*1.155 = 1.08875
        // On-level factor = 1.155 / 1.08875
        let factor = parallelogram_onlevel_factor(&[0.10, 0.05], &[0.25, 0.50, 0.25])
            .expect("on-level factor");
        assert!(close(factor, 1.155 / 1.08875));
        // No rate changes -> factor == 1.0.
        assert!(close(
            parallelogram_onlevel_factor(&[], &[1.0]).expect("no change"),
            1.0
        ));
    }
}
