//! Formal non-parametric statistical tests for Mack (1993) chain-ladder
//! model assumptions.
//!
//! Implements both diagnostic tests from Mack (1994), *Measuring the
//! Variability of Chain Ladder Reserve Estimates*, Casualty Actuarial Society
//! Spring Forum 1994, pp. 101–182:
//!
//! * **Appendix G — Adjacent Development Factor Correlation Test**:
//!   Tests whether subsequent individual age-to-age factors `F_{i,k}` and
//!   `F_{i,k+1}` are uncorrelated via weighted Spearman rank correlations `T_k`.
//! * **Appendix H — Calendar-Year (Diagonal) Effect Test**:
//!   Classifies individual age-to-age factors within each column as Small (`S`)
//!   or Large (`L`) relative to the column median and tests whether diagonals
//!   exhibit systematic high/low clustering using exact binomial moments of
//!   `Z_d = min(S_d, L_d)`.

use crate::dependence::spearman_correlation;
use crate::severity::normal_cdf_exact;
use crate::triangle::age_to_age;
use crate::types::TriangleView;

/// Spearman rank-correlation record for one adjacent development-link pair `(k, k + 1)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdjacentLinkCorrelation {
    /// Leading development link index `k` (correlating link `k -> k+1` with `k+1 -> k+2`).
    pub link: usize,
    /// Number of common origin years observed across both links (`m_k`).
    pub n_pairs: usize,
    /// Spearman rank correlation `T_k`.
    pub spearman_t: f64,
    /// Mack weight `w_k = m_k - 1` (inverse null variance of `T_k`).
    pub weight: f64,
}

/// Result of Mack (1994, Appendix G) adjacent development-factor correlation test.
#[derive(Debug, Clone, PartialEq)]
pub struct MackCorrelationTest {
    /// Per-pair Spearman rank correlation records.
    pub by_link: Vec<AdjacentLinkCorrelation>,
    /// Weighted aggregate test statistic `T = Σ w_k T_k / Σ w_k`.
    pub statistic_t: f64,
    /// Null variance `Var(T) = 1 / Σ w_k`.
    pub null_variance: f64,
    /// Standardized normal test score `z = T / sqrt(Var(T))`.
    pub z_score: f64,
    /// Two-sided normal p-value `2 · (1 - Φ(|z|))`.
    pub two_sided_p_value: f64,
    /// Mack's 50% confidence half-width `0.6745 · sqrt(Var(T))`.
    pub bound_50_pct: f64,
    /// 95% confidence half-width `1.96 · sqrt(Var(T))`.
    pub bound_95_pct: f64,
}

/// Per-calendar-diagonal counts and binomial moments for Mack (1994, Appendix H).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalendarDiagonalTestRow {
    /// Calendar diagonal index `d = origin + dev + 1` (target cell calendar period).
    pub calendar_diagonal: usize,
    /// Count of individual link ratios strictly below their column median (`S_d`).
    pub small_count: usize,
    /// Count of individual link ratios strictly above their column median (`L_d`).
    pub large_count: usize,
    /// Test contribution `Z_d = min(S_d, L_d)`.
    pub z_d: f64,
    /// Exact null expectation `E[Z_d]` under `Binomial(S_d + L_d, 1/2)`.
    pub expected_z_d: f64,
    /// Exact null variance `Var(Z_d)` under `Binomial(S_d + L_d, 1/2)`.
    pub variance_z_d: f64,
}

/// Result of Mack (1994, Appendix H) calendar-year effect test.
#[derive(Debug, Clone, PartialEq)]
pub struct MackCalendarYearTest {
    /// Per-diagonal Small/Large counts and exact binomial moments.
    pub diagonals: Vec<CalendarDiagonalTestRow>,
    /// Aggregate statistic `Z = Σ Z_d`.
    pub statistic_z: f64,
    /// Exact null expectation `E[Z] = Σ E[Z_d]`.
    pub expected_z: f64,
    /// Exact null variance `Var(Z) = Σ Var(Z_d)`.
    pub variance_z: f64,
    /// Standardized score `(Z - E[Z]) / sqrt(Var(Z))`. Negative values indicate
    /// diagonal clustering (calendar-year effects).
    pub z_score: f64,
    /// Lower-tail p-value `Φ(z)` (small values indicate significant calendar-year effect).
    pub lower_tail_p_value: f64,
    /// Two-sided p-value `2 · (1 - Φ(|z|))`.
    pub two_sided_p_value: f64,
}

/// Run Mack (1994, Appendix G) Spearman rank correlation test for independence
/// of adjacent development factors.
pub fn adjacent_factor_correlation_test(t: &TriangleView<'_>) -> MackCorrelationTest {
    let n = t.n_origins.min(t.n_devs);
    let mut by_link = Vec::new();
    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;

    if n >= 4 {
        for k in 0..(n - 3) {
            let mut left = Vec::new();
            let mut right = Vec::new();
            for i in 0..(n - k - 2) {
                let r1 = age_to_age(t, i, k);
                let r2 = age_to_age(t, i, k + 1);
                if r1.is_finite() && r2.is_finite() {
                    left.push(r1);
                    right.push(r2);
                }
            }
            let m = left.len();
            if m >= 2 {
                let tk = spearman_correlation(&left, &right);
                if tk.is_finite() {
                    let w = (m - 1) as f64;
                    weighted_sum += w * tk;
                    total_weight += w;
                    by_link.push(AdjacentLinkCorrelation {
                        link: k,
                        n_pairs: m,
                        spearman_t: tk,
                        weight: w,
                    });
                }
            }
        }
    }

    if total_weight <= 0.0 {
        return MackCorrelationTest {
            by_link,
            statistic_t: f64::NAN,
            null_variance: f64::NAN,
            z_score: f64::NAN,
            two_sided_p_value: f64::NAN,
            bound_50_pct: f64::NAN,
            bound_95_pct: f64::NAN,
        };
    }

    let statistic_t = weighted_sum / total_weight;
    let null_variance = 1.0 / total_weight;
    let sd = null_variance.sqrt();
    let z_score = statistic_t / sd;
    let two_sided_p_value = (2.0 * normal_cdf_exact(-z_score.abs())).clamp(0.0, 1.0);

    MackCorrelationTest {
        by_link,
        statistic_t,
        null_variance,
        z_score,
        two_sided_p_value,
        bound_50_pct: 0.674_489_750_196_081_7 * sd,
        bound_95_pct: 1.959_963_984_540_054 * sd,
    }
}

/// Exact expectation and variance of `Z = min(S, n - S)` for `S ~ Binomial(n, 1/2)`
/// (Mack 1994, Appendix H).
fn exact_min_binomial_moments(n: usize) -> (f64, f64) {
    if n <= 1 {
        return (0.0, 0.0);
    }
    let nf = n as f64;
    let m = (n - 1) / 2;
    // Compute C(n - 1, m) / 2^n safely in log space to avoid overflow for large triangles.
    let log_comb = crate::bootstrap_odp::ln_gamma(nf)
        - crate::bootstrap_odp::ln_gamma((m + 1) as f64)
        - crate::bootstrap_odp::ln_gamma((n - m) as f64);
    let comb_over_2n = (log_comb - nf * std::f64::consts::LN_2).exp();
    let expected = 0.5 * nf - comb_over_2n * nf;
    let variance = (0.25 * nf * (nf - 1.0) - comb_over_2n * nf * (nf - 1.0) + expected
        - expected * expected)
        .max(0.0);
    (expected, variance)
}

/// Run Mack (1994, Appendix H) calendar-year (diagonal) effect test.
pub fn calendar_year_effect_test(t: &TriangleView<'_>) -> MackCalendarYearTest {
    let n = t.n_origins.min(t.n_devs);
    if n < 3 {
        return MackCalendarYearTest {
            diagonals: Vec::new(),
            statistic_z: f64::NAN,
            expected_z: f64::NAN,
            variance_z: f64::NAN,
            z_score: f64::NAN,
            lower_tail_p_value: f64::NAN,
            two_sided_p_value: f64::NAN,
        };
    }

    let mut small_by_diag = vec![0usize; n];
    let mut large_by_diag = vec![0usize; n];

    for k in 0..(n - 1) {
        let mut pairs: Vec<(usize, f64)> = Vec::new();
        for i in 0..(n - k - 1) {
            let r = age_to_age(t, i, k);
            if r.is_finite() {
                pairs.push((i, r));
            }
        }
        if pairs.len() < 2 {
            continue;
        }
        let mut sorted_vals: Vec<f64> = pairs.iter().map(|&(_, r)| r).collect();
        sorted_vals.sort_by(f64::total_cmp);
        let len = sorted_vals.len();
        let median = if len % 2 == 1 {
            sorted_vals[len / 2]
        } else {
            f64::midpoint(sorted_vals[len / 2 - 1], sorted_vals[len / 2])
        };

        for (i, r) in pairs {
            let diag = i + k + 1;
            if diag < n {
                if r < median {
                    small_by_diag[diag] += 1;
                } else if r > median {
                    large_by_diag[diag] += 1;
                }
            }
        }
    }

    let mut diagonals = Vec::new();
    let mut statistic_z = 0.0;
    let mut expected_z = 0.0;
    let mut variance_z = 0.0;

    for diag in 1..n {
        let s = small_by_diag[diag];
        let l = large_by_diag[diag];
        let nd = s + l;
        if nd == 0 {
            continue;
        }
        let z_d = s.min(l) as f64;
        let (e_d, v_d) = exact_min_binomial_moments(nd);
        statistic_z += z_d;
        expected_z += e_d;
        variance_z += v_d;
        diagonals.push(CalendarDiagonalTestRow {
            calendar_diagonal: diag,
            small_count: s,
            large_count: l,
            z_d,
            expected_z_d: e_d,
            variance_z_d: v_d,
        });
    }

    if !(variance_z > 0.0) {
        return MackCalendarYearTest {
            diagonals,
            statistic_z,
            expected_z,
            variance_z,
            z_score: f64::NAN,
            lower_tail_p_value: f64::NAN,
            two_sided_p_value: f64::NAN,
        };
    }

    let z_score = (statistic_z - expected_z) / variance_z.sqrt();
    let lower_tail_p_value = normal_cdf_exact(z_score).clamp(0.0, 1.0);
    let two_sided_p_value = (2.0 * normal_cdf_exact(-z_score.abs())).clamp(0.0, 1.0);

    MackCalendarYearTest {
        diagonals,
        statistic_z,
        expected_z,
        variance_z,
        z_score,
        lower_tail_p_value,
        two_sided_p_value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Triangle;

    #[test]
    fn exact_min_binomial_moments_match_hand_calculations() {
        assert_eq!(exact_min_binomial_moments(1), (0.0, 0.0));
        let (e2, v2) = exact_min_binomial_moments(2);
        assert!((e2 - 0.5).abs() < 1e-12 && (v2 - 0.25).abs() < 1e-12);
        let (e3, v3) = exact_min_binomial_moments(3);
        assert!((e3 - 0.75).abs() < 1e-12 && (v3 - 0.1875).abs() < 1e-12);
        let (e4, v4) = exact_min_binomial_moments(4);
        assert!((e4 - 1.25).abs() < 1e-12 && (v4 - 0.4375).abs() < 1e-12);
    }

    #[test]
    fn calendar_test_detects_strong_calendar_year_inflation_shock() {
        // Construct an 8x8 triangle with a strong calendar-year inflation shock
        // on the latest two diagonals (every factor on those diagonals is high).
        let n = 8usize;
        let base_ldfs = [2.0, 1.5, 1.25, 1.15, 1.10, 1.05, 1.02];
        let mut data = vec![f64::NAN; n * n];
        for i in 0..n {
            let mut c = 1000.0 + 50.0 * i as f64;
            data[i * n] = c;
            for j in 1..(n - i) {
                let cal = i + j;
                let shock = if cal >= n - 2 {
                    1.25
                } else if cal <= 3 {
                    0.90
                } else {
                    1.0
                };
                c *= base_ldfs[j - 1] * shock;
                data[i * n + j] = c;
            }
        }
        let tri = Triangle::new(n, n, data);
        let cal_test = calendar_year_effect_test(&tri.view());
        assert!(cal_test.z_score < -1.96, "z={}", cal_test.z_score);
        assert!(cal_test.lower_tail_p_value < 0.025);
    }

    #[test]
    fn correlation_test_computes_exact_null_variance() {
        let n = 6usize;
        let mut data = vec![f64::NAN; n * n];
        for i in 0..n {
            let mut c = 100.0 * (i + 1) as f64;
            data[i * n] = c;
            for j in 1..(n - i) {
                let jitter = 1.0 + 0.02 * ((i * 3 + j) % 5) as f64;
                c *= (1.4 - 0.05 * j as f64) * jitter;
                data[i * n + j] = c;
            }
        }
        let tri = Triangle::new(n, n, data);
        let res = adjacent_factor_correlation_test(&tri.view());
        // Total weight for n=6: (n-2)(n-3)/2 = 4*3/2 = 6 -> Var(T) = 1/6.
        assert!((res.null_variance - 1.0 / 6.0).abs() < 1e-12);
        assert!(res.two_sided_p_value >= 0.0 && res.two_sided_p_value <= 1.0);
    }
}
