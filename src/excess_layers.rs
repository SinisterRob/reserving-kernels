//! Lognormal severity layering, Limited Expected Values (LEV), and Increased Limit Factors (ILFs).
//!
//! Provides exact closed-form lognormal limited expected value $E[X \wedge L]$
//! alongside a deterministic SplitMix64 + Box–Muller Monte Carlo engine for
//! primary retention and excess layer pricing.

use std::f64::consts::TAU;

/// Default ground-up expected severity ($100,000).
pub const GROUND_UP_MEAN: f64 = 100_000.0;
/// Default primary retention / excess attachment point ($100,000).
pub const ATTACHMENT: f64 = 100_000.0;
const SAMPLE_SIZE: u32 = 65_536;
const SEED: u64 = 2026;
const INV_SQRT_TAU: f64 = 0.3989422804014327;

/// Closed-form expected loss in either the retained primary layer (`layer == 0`, $0$ to `ATTACHMENT`)
/// or the excess layer (`layer == 1`, `ATTACHMENT` to `top`).
pub fn expected_layer(tail: u32, top: f64, layer: u32) -> f64 {
    let sigma = log_sd(tail);
    let lower = limited_mean(ATTACHMENT, sigma);
    if layer == 0 {
        lower
    } else if top <= ATTACHMENT {
        0.0
    } else {
        (limited_mean(top, sigma) - lower).max(0.0)
    }
}

/// Deterministic Monte Carlo estimate (fixed seed `2026`, $N = 65{,}536$ synchronized draws)
/// of layer expected loss using SplitMix64 + Box–Muller normal transformation.
pub fn simulated_layer(tail: u32, top: f64, layer: u32) -> f64 {
    if layer != 0 && top <= ATTACHMENT {
        return 0.0;
    }
    let sigma = log_sd(tail);
    let log_mean = GROUND_UP_MEAN.ln() - sigma * sigma / 2.0;
    let mut state = SEED;
    let mut total = 0.0;
    let pairs = SAMPLE_SIZE / 2;
    for _ in 0..pairs {
        let radius = (-2.0 * uniform(&mut state).ln()).sqrt();
        let angle = TAU * uniform(&mut state);
        let (sin_a, cos_a) = angle.sin_cos();
        for z in [radius * cos_a, radius * sin_a] {
            let severity = (log_mean + sigma * z).exp();
            let lower = severity.min(ATTACHMENT);
            total += if layer == 0 {
                lower
            } else {
                (severity.min(top) - lower).max(0.0)
            };
        }
    }
    total / f64::from(SAMPLE_SIZE)
}

/// Probability that a single ground-up loss exceeds `threshold`, $P(X > \text{threshold})$.
pub fn exceedance_prob(tail: u32, threshold: f64) -> f64 {
    if threshold <= 0.0 {
        return 1.0;
    }
    let sigma = log_sd(tail);
    let log_mean = GROUND_UP_MEAN.ln() - sigma * sigma / 2.0;
    let z = (threshold.ln() - log_mean) / sigma;
    crate::severity::normal_cdf_exact(-z)
}

/// Lognormal probability density function $f_X(x)$ at loss amount `x`.
pub fn severity_pdf(tail: u32, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let sigma = log_sd(tail);
    let log_mean = GROUND_UP_MEAN.ln() - sigma * sigma / 2.0;
    let z = (x.ln() - log_mean) / sigma;
    (INV_SQRT_TAU / (x * sigma)) * (-0.5 * z * z).exp()
}

/// Lognormal Limited Expected Value $E[X \wedge L]$ for ground-up mean `GROUND_UP_MEAN` and log-SD `sigma`.
pub fn limited_mean(limit: f64, sigma: f64) -> f64 {
    if !limit.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
        return f64::NAN;
    }
    if limit <= 0.0 {
        return 0.0;
    }
    let log_mean = GROUND_UP_MEAN.ln() - sigma * sigma / 2.0;
    let z = (limit.ln() - log_mean) / sigma;
    GROUND_UP_MEAN * crate::severity::normal_cdf_exact(z - sigma)
        + limit * crate::severity::normal_cdf_exact(-z)
}

fn log_sd(tail: u32) -> f64 {
    match tail {
        0 => 0.55,
        2 => 1.55,
        _ => 1.0,
    }
}

fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut bits = *state;
    bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
    bits ^= bits >> 31;
    ((bits >> 12) as f64 + 0.5) / 4_503_599_627_370_496.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_layer_matches_reference_values() {
        for (tail, reference) in [78_331.62, 61_707.51, 43_833.97].into_iter().enumerate() {
            assert!((expected_layer(tail as u32, 350_000.0, 0) - reference).abs() < 0.03);
        }
    }

    #[test]
    fn layer_width_and_tail_change_relativities() {
        let mut previous_ilf = 1.0;
        for tail in 0..=2 {
            let lower = expected_layer(tail, 350_000.0, 0);
            assert_eq!(expected_layer(tail, ATTACHMENT, 1), 0.0);
            let mut previous_upper = 0.0;
            for top in (150_000..=750_000).step_by(25_000) {
                let upper = expected_layer(tail, f64::from(top), 1);
                assert!(upper > previous_upper);
                assert!(lower + upper <= GROUND_UP_MEAN);
                previous_upper = upper;
            }
            let ilf = 1.0 + expected_layer(tail, 350_000.0, 1) / lower;
            assert!(ilf > previous_ilf);
            previous_ilf = ilf;
        }
    }

    #[test]
    fn fixed_seed_simulation_agrees_with_limited_expectations() {
        for tail in 0..=2 {
            for top in (150_000..=750_000).step_by(25_000) {
                for layer in 0..=1 {
                    let exact = expected_layer(tail, f64::from(top), layer);
                    let simulated = simulated_layer(tail, f64::from(top), layer);
                    assert!(
                        (exact - simulated).abs() < 1_000.0,
                        "tail {tail}, top {top}, layer {layer}: {exact} vs. {simulated}"
                    );
                }
            }
        }
        assert_eq!(
            simulated_layer(1, 350_000.0, 1),
            simulated_layer(1, 350_000.0, 1)
        );
    }

    #[test]
    fn exceedance_and_pdf_are_consistent() {
        for tail in 0..=2 {
            let p_att = exceedance_prob(tail, ATTACHMENT);
            let p_top = exceedance_prob(tail, 350_000.0);
            assert!(p_att > 0.0 && p_att < 1.0);
            assert!(p_top > 0.0 && p_top < p_att);
            assert_eq!(exceedance_prob(tail, 0.0), 1.0);
            assert_eq!(severity_pdf(tail, 0.0), 0.0);
            assert!(severity_pdf(tail, 50_000.0) > 0.0);
            let dx = 1_000.0;
            let mut area = 0.0;
            for i in 0..2_000 {
                let x = (i as f64 + 0.5) * dx;
                area += severity_pdf(tail, x) * dx;
            }
            assert!(area > 0.90 && area <= 1.001, "tail {tail} area={area}");
        }
    }
}
