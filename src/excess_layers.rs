//! Lognormal severity layering and Limited Expected Values (LEV).
//!
//! Provides exact closed-form lognormal limited expected value $E[X \wedge L]$
//! alongside a deterministic SplitMix64 + antithetic Marsaglia polar Monte Carlo
//! engine with sorted prefix-sum $O(\log N)$ layer evaluation.

use std::sync::OnceLock;

/// Default ground-up expected severity ($100,000).
pub const GROUND_UP_MEAN: f64 = 100_000.0;
/// Default primary retention / excess attachment point ($100,000).
pub const ATTACHMENT: f64 = 100_000.0;
const LN_GROUND_UP_MEAN: f64 = 11.512925464970229;
const SAMPLE_SIZE: usize = 65_536;
const HALF_SAMPLE: usize = SAMPLE_SIZE / 2;
const INV_SAMPLE_SIZE: f64 = 1.0 / (SAMPLE_SIZE as f64);
const SEED: u64 = 2026;
const INV_SQRT_TAU: f64 = 0.3989422804014327;

#[derive(Clone, Copy)]
struct TailParams {
    sigma: f64,
    inv_sigma: f64,
    log_mean: f64,
    pdf_scale: f64,
}

const fn make_tail(sigma: f64) -> TailParams {
    TailParams {
        sigma,
        inv_sigma: 1.0 / sigma,
        log_mean: LN_GROUND_UP_MEAN - 0.5 * sigma * sigma,
        pdf_scale: INV_SQRT_TAU / sigma,
    }
}

const TAIL_PARAMS: [TailParams; 3] = [make_tail(0.55), make_tail(1.00), make_tail(1.55)];

#[inline]
const fn tail_index(tail: u32) -> usize {
    match tail {
        0 => 0,
        2 => 2,
        _ => 1,
    }
}

static SORTED_NORMALS: OnceLock<Vec<f64>> = OnceLock::new();
static TAIL_PREFIX_SUMS: [OnceLock<Vec<f64>>; 3] =
    [OnceLock::new(), OnceLock::new(), OnceLock::new()];

fn sorted_normals() -> &'static [f64] {
    SORTED_NORMALS.get_or_init(|| {
        let mut state = SEED;
        let mut pos = Vec::with_capacity(HALF_SAMPLE);
        while pos.len() < HALF_SAMPLE {
            let u = 2.0 * uniform(&mut state) - 1.0;
            let v = 2.0 * uniform(&mut state) - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let scale = (-2.0 * s.ln() / s).sqrt();
                pos.push((u * scale).abs());
                pos.push((v * scale).abs());
            }
        }
        pos.sort_unstable_by(f64::total_cmp);
        let mut all = Vec::with_capacity(SAMPLE_SIZE);
        for &z in pos.iter().rev() {
            all.push(-z);
        }
        all.extend_from_slice(&pos);
        all
    })
}

fn tail_prefix(t: usize) -> &'static [f64] {
    TAIL_PREFIX_SUMS[t].get_or_init(|| {
        let normals = sorted_normals();
        let p = TAIL_PARAMS[t];
        let mut prefix = Vec::with_capacity(SAMPLE_SIZE + 1);
        let mut acc = 0.0;
        prefix.push(0.0);
        for &z in normals {
            acc += (p.log_mean + p.sigma * z).exp();
            prefix.push(acc);
        }
        prefix
    })
}

#[inline]
fn simulated_lev(tail: u32, limit: f64) -> f64 {
    if limit <= 0.0 || limit.is_nan() {
        return 0.0;
    }
    let t = tail_index(tail);
    let normals = sorted_normals();
    let prefix = tail_prefix(t);
    if !limit.is_finite() {
        return prefix[SAMPLE_SIZE] * INV_SAMPLE_SIZE;
    }
    let p = TAIL_PARAMS[t];
    let z_thresh = (limit.ln() - p.log_mean) * p.inv_sigma;
    let k = normals.partition_point(|&z| z < z_thresh);
    (prefix[k] + ((SAMPLE_SIZE - k) as f64) * limit) * INV_SAMPLE_SIZE
}

/// Closed-form expected loss in either the retained primary layer (`layer == 0`, $0$ to `ATTACHMENT`)
/// or the excess layer (`layer == 1`, `ATTACHMENT` to `top`).
pub fn expected_layer(tail: u32, top: f64, layer: u32) -> f64 {
    let p = TAIL_PARAMS[tail_index(tail)];
    let lower = limited_mean_params(ATTACHMENT, p);
    if layer == 0 {
        lower
    } else if top <= ATTACHMENT {
        0.0
    } else {
        (limited_mean_params(top, p) - lower).max(0.0)
    }
}

/// Deterministic Monte Carlo estimate (fixed seed `2026`, $N = 65{,}536$ synchronized
/// antithetic draws) evaluated in $O(\log N)$ via cached sorted prefix sums.
pub fn simulated_layer(tail: u32, top: f64, layer: u32) -> f64 {
    if layer == 0 {
        simulated_lev(tail, ATTACHMENT)
    } else if top <= ATTACHMENT {
        0.0
    } else {
        (simulated_lev(tail, top) - simulated_lev(tail, ATTACHMENT)).max(0.0)
    }
}

/// Probability that a single ground-up loss exceeds `threshold`, $P(X > \text{threshold})$.
pub fn exceedance_prob(tail: u32, threshold: f64) -> f64 {
    if threshold <= 0.0 {
        return 1.0;
    }
    let p = TAIL_PARAMS[tail_index(tail)];
    let z = (threshold.ln() - p.log_mean) * p.inv_sigma;
    crate::severity::normal_cdf_exact(-z)
}

/// Lognormal probability density function $f_X(x)$ at loss amount `x`.
pub fn severity_pdf(tail: u32, x: f64) -> f64 {
    if x <= 0.0 || !x.is_finite() {
        return 0.0;
    }
    let p = TAIL_PARAMS[tail_index(tail)];
    let z = (x.ln() - p.log_mean) * p.inv_sigma;
    (p.pdf_scale / x) * (-0.5 * z * z).exp()
}

#[inline]
fn limited_mean_params(limit: f64, p: TailParams) -> f64 {
    if !limit.is_finite() {
        return f64::NAN;
    }
    if limit <= 0.0 {
        return 0.0;
    }
    let z = (limit.ln() - p.log_mean) * p.inv_sigma;
    GROUND_UP_MEAN * crate::severity::normal_cdf_exact(z - p.sigma)
        + limit * crate::severity::normal_cdf_exact(-z)
}

/// Lognormal Limited Expected Value $E[X \wedge L]$ for ground-up mean `GROUND_UP_MEAN` and log-SD `sigma`.
pub fn limited_mean(limit: f64, sigma: f64) -> f64 {
    if !limit.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
        return f64::NAN;
    }
    if limit <= 0.0 {
        return 0.0;
    }
    let log_mean = LN_GROUND_UP_MEAN - 0.5 * sigma * sigma;
    let z = (limit.ln() - log_mean) / sigma;
    GROUND_UP_MEAN * crate::severity::normal_cdf_exact(z - sigma)
        + limit * crate::severity::normal_cdf_exact(-z)
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
                let x = (f64::from(i) + 0.5) * dx;
                area += severity_pdf(tail, x) * dx;
            }
            assert!(area > 0.90 && area <= 1.001, "tail {tail} area={area}");
        }
    }
}
