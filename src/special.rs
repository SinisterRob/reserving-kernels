//! Neutral special-function and rank-statistic helpers.
//!
//! This module sits below every actuarial layer (`deterministic`,
//! `diagnostics`, `stochastic`, `capital`, `excess`) so diagnostics can use
//! normal / chi-square tail probabilities and rank correlations without
//! importing sibling-layer internals.
//!
//! All functions are pure, allocation-light, and panic-free.

use std::f64::consts::{SQRT_2, TAU};

/// Standard normal CDF `Φ(z)` via the Cody (1969) rational `erfc`
/// approximation (max absolute error `< 1e-15`).
pub(crate) fn normal_cdf(z: f64) -> f64 {
    0.5 * erfc(-z / SQRT_2)
}

/// Two-sided normal p-value `2 · (1 − Φ(|z|))`, clamped to `[0, 1]`.
/// Returns `NaN` for `NaN` input.
pub(crate) fn two_sided_normal_p_value(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    (2.0 * normal_cdf(-z.abs())).clamp(0.0, 1.0)
}

/// Complementary error function `erfc(x)` accurate to ~1e-15 (Cody 1969).
pub(crate) fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    if x < 0.46875 {
        const A: [f64; 5] = [
            3.161_123_743_870_565_6e0,
            1.138_641_541_510_501_6e2,
            3.774_852_376_853_02e2,
            3.209_377_589_138_469_5e3,
            1.857_777_061_846_031_5e-1,
        ];
        const B: [f64; 4] = [
            2.360_129_095_234_412_1e1,
            2.440_246_379_344_441_7e2,
            1.282_616_526_077_372_3e3,
            2.844_236_833_439_171e3,
        ];
        let y = x * x;
        let num = (((A[4] * y + A[0]) * y + A[1]) * y + A[2]) * y + A[3];
        let den = (((y + B[0]) * y + B[1]) * y + B[2]) * y + B[3];
        1.0 - x * num / den
    } else if x <= 4.0 {
        const C: [f64; 9] = [
            5.641_884_969_886_701e-1,
            8.883_149_794_388_375,
            6.611_919_063_714_163e1,
            2.986_351_381_974_001e2,
            8.819_522_212_417_69e2,
            1.712_047_612_634_070_6e3,
            2.051_078_377_826_071_5e3,
            1.230_339_354_797_997_2e3,
            2.153_115_354_744_038_5e-8,
        ];
        const D: [f64; 8] = [
            1.574_492_611_070_983_5e1,
            1.176_939_508_913_125e2,
            5.371_811_018_620_099e2,
            1.621_389_574_566_690_2e3,
            3.290_799_235_733_459_6e3,
            4.362_619_090_143_247e3,
            3.439_367_674_143_721_6e3,
            1.230_339_354_803_749_4e3,
        ];
        // Horner form: (((C8 x + C0) x + C1) x + ... ) x + C7.
        let mut num = C[8] * x + C[0];
        for &c in &C[1..8] {
            num = num * x + c;
        }
        // Horner form: ((x + D0) x + D1) x + ... + D7.
        let mut den = x + D[0];
        for &d in &D[1..8] {
            den = den * x + d;
        }
        (-x * x).exp() * num / den
    } else if x < 26.5 {
        const P: [f64; 6] = [
            3.053_266_349_612_323_4e-1,
            3.603_448_999_498_044_4e-1,
            1.257_817_261_112_292_5e-1,
            1.608_378_514_874_228e-2,
            6.587_491_615_298_378e-4,
            1.631_538_713_730_209_8e-2,
        ];
        const Q: [f64; 5] = [
            2.568_520_192_289_822,
            1.872_952_849_923_460_5e0,
            5.279_051_029_514_285e-1,
            6.051_834_131_244_132e-2,
            2.335_204_976_268_691_9e-3,
        ];
        let y = 1.0 / (x * x);
        let num = ((((P[5] * y + P[0]) * y + P[1]) * y + P[2]) * y + P[3]) * y + P[4];
        let den = ((((y + Q[0]) * y + Q[1]) * y + Q[2]) * y + Q[3]) * y + Q[4];
        let inv_sqrt_pi = std::f64::consts::FRAC_2_SQRT_PI * 0.5;
        (-x * x).exp() / x * (inv_sqrt_pi - y * num / den)
    } else {
        0.0
    }
}

/// Natural log of the gamma function for `z > 0` (Lanczos, `g = 7`, `n = 9`).
pub(crate) fn ln_gamma(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = z - 1.0;
    let mut sum = COEFFS[0];
    for (idx, &coeff) in COEFFS.iter().enumerate().skip(1) {
        sum += coeff / (x + idx as f64);
    }
    let t = x + 7.5;
    0.5 * TAU.ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// Regularized upper incomplete gamma `Q(a, x) = Γ(a, x) / Γ(a)` for
/// `a > 0`, `x >= 0` (series for `x < a + 1`, Lentz continued fraction
/// otherwise; Numerical Recipes §6.2). Returns `NaN` on invalid input.
pub(crate) fn regularized_upper_gamma(a: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 500;
    const EPS: f64 = 1e-15;
    const TINY: f64 = 1e-300;
    if !(a > 0.0) || !(x >= 0.0) || !a.is_finite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x.is_infinite() {
        return 0.0;
    }
    let log_prefactor = -x + a * x.ln() - ln_gamma(a);
    if x < a + 1.0 {
        let mut ap = a;
        let mut del = 1.0 / a;
        let mut sum = del;
        for _ in 0..MAX_ITER {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * EPS {
                break;
            }
        }
        (1.0 - sum * log_prefactor.exp()).clamp(0.0, 1.0)
    } else {
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=MAX_ITER {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < TINY {
                d = TINY;
            }
            c = b + an / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < EPS {
                break;
            }
        }
        (log_prefactor.exp() * h).clamp(0.0, 1.0)
    }
}

/// Upper-tail chi-square probability `P(χ²_k > x)`.
pub(crate) fn chi_square_survival(x: f64, degrees_of_freedom: usize) -> f64 {
    if degrees_of_freedom == 0 {
        return f64::NAN;
    }
    regularized_upper_gamma(0.5 * degrees_of_freedom as f64, 0.5 * x)
}

/// 1-based average (mid) ranks with ties sharing their mean rank.
pub(crate) fn average_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| values[a].total_cmp(&values[b]));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && values[order[j]] == values[order[i]] {
            j += 1;
        }
        let avg_rank = f64::midpoint((i + 1) as f64, j as f64);
        for &idx in &order[i..j] {
            ranks[idx] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Sample Pearson correlation; `NaN` when undefined.
pub(crate) fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 || n != y.len() {
        return f64::NAN;
    }
    let nf = n as f64;
    let mx = x.iter().sum::<f64>() / nf;
    let my = y.iter().sum::<f64>() / nf;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for (&a, &b) in x.iter().zip(y.iter()) {
        let dx = a - mx;
        let dy = b - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let denom = (sxx * syy).sqrt();
    if denom > 0.0 {
        (sxy / denom).clamp(-1.0, 1.0)
    } else {
        f64::NAN
    }
}

/// Tie-adjusted Spearman rank correlation; `NaN` when undefined.
pub(crate) fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() < 2 || x.len() != y.len() {
        return f64::NAN;
    }
    if x.iter().any(|v| v.is_nan()) || y.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    pearson_correlation(&average_ranks(x), &average_ranks(y))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_cdf_matches_reference_points() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-15);
        assert!((normal_cdf(1.959_963_984_540_054) - 0.975).abs() < 1e-12);
        assert!((normal_cdf(-1.0) - 0.158_655_253_931_457_05).abs() < 1e-13);
        assert!((normal_cdf(3.0) - 0.998_650_101_968_369_9).abs() < 1e-13);
        assert!((two_sided_normal_p_value(1.959_963_984_540_054) - 0.05).abs() < 1e-12);
        assert!(two_sided_normal_p_value(f64::NAN).is_nan());
    }

    #[test]
    fn ln_gamma_matches_factorials() {
        assert!(ln_gamma(1.0).abs() < 1e-12);
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-12);
        assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-12);
    }

    #[test]
    fn chi_square_survival_matches_tables() {
        // χ²_1 > 3.841459 has probability 0.05; χ²_2 survival is exp(-x/2).
        assert!((chi_square_survival(3.841_458_820_694_124, 1) - 0.05).abs() < 1e-10);
        assert!((chi_square_survival(5.0, 2) - (-2.5_f64).exp()).abs() < 1e-12);
        // χ²_10 > 18.307038 has probability 0.05 (both series & CF branches).
        assert!((chi_square_survival(18.307_038_053_275_14, 10) - 0.05).abs() < 1e-9);
        assert!((chi_square_survival(3.940_299_136_119_247, 10) - 0.95).abs() < 1e-9);
        assert!((chi_square_survival(0.0, 3) - 1.0).abs() < 1e-15);
        assert!(chi_square_survival(1.0, 0).is_nan());
    }

    #[test]
    fn spearman_handles_ties_and_monotone_maps() {
        assert!((spearman_correlation(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]) - 1.0).abs() < 1e-15);
        assert!((spearman_correlation(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]) + 1.0).abs() < 1e-15);
        assert_eq!(average_ranks(&[5.0, 1.0, 5.0]), vec![2.5, 1.0, 2.5]);
        assert!(spearman_correlation(&[1.0], &[1.0]).is_nan());
    }
}
