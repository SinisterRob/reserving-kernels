//! Shared summary math, risk measures, and proper predictive-scoring rules for
//! reserve-simulation output.
//!
//! Includes both the legacy upper-rank empirical quantile / TCE helpers
//! ([`quantile`], [`tvar`]), the exact discrete risk-measure algorithms from
//! CAS Monograph No. 15 (Mildenhall & Major, *Introduction to Capital Modeling
//! and Portfolio Management*, §3.2–§3.4), and stateless proper scoring rules
//! (Gneiting & Raftery 2007) for out-of-sample predictive validation:
//!
//! * [`var_least`] — Algorithm 3.4.2 (least quantile `F^-(p)`).
//! * [`tvar_coherent`] — Algorithm 3.4.3 (exact integral Tail Value at Risk
//!   `TVaR_p(X) = (1 / (1 - p)) ∫_p^1 F^-(s) ds`, guaranteed subadditive even
//!   on discrete / tied empirical distributions).
//! * [`rvar`] — Range Value at Risk `RVaR_{p_low, p_high}(X)` (Cont et al.
//!   2010; CAS Monograph No. 15, §3.3; note general RVaR is not subadditive).
//! * [`crps_empirical`], [`pit_empirical`], [`interval_score`] — exact
//!   empirical Continuous Ranked Probability Score, Probability Integral
//!   Transform, and Winkler interval score for evaluating forecast draws
//!   against realized outcomes.

use crate::validate::{ValidationError, validate_finite_slice};

/// Summary statistics for a reserve sample distribution.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ReserveDistributionSummary {
    /// Arithmetic mean.
    pub mean: f64,
    /// Sample variance.
    pub variance: f64,
    /// Sample standard deviation.
    pub std_dev: f64,
    /// 50th percentile.
    pub p50: f64,
    /// 75th percentile.
    pub p75: f64,
    /// 90th percentile.
    pub p90: f64,
    /// 95th percentile.
    pub p95: f64,
    /// 99th percentile.
    pub p99: f64,
}

fn validate_prob(p: f64) -> Result<(), ValidationError> {
    if !p.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if !(0.0..=1.0).contains(&p) {
        return Err(ValidationError::InvalidProbability { value: p });
    }
    Ok(())
}

fn sorted(values: &[f64]) -> Result<Vec<f64>, ValidationError> {
    if values.is_empty() {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_finite_slice(values)?;
    let mut out = values.to_vec();
    out.sort_unstable_by(f64::total_cmp);
    Ok(out)
}

#[inline]
fn quantile_rank_index(len: usize, p: f64) -> usize {
    if p <= 0.0 {
        return 0;
    }
    if p >= 1.0 {
        return len - 1;
    }
    // Snap near-integer p * N within 1e-12 to avoid IEEE-754 representation jumps
    // (e.g. 0.07 * 100.0 == 7.000000000000001).
    let pn = p * len as f64;
    let rounded = pn.round();
    let rank = if (pn - rounded).abs() <= 1e-12 {
        rounded as usize
    } else {
        pn.ceil() as usize
    };
    rank.saturating_sub(1).min(len - 1)
}

#[inline]
fn select_quantile_in_place(buf: &mut [f64], p: f64) -> f64 {
    let idx = quantile_rank_index(buf.len(), p);
    let (_, nth, _) = buf.select_nth_unstable_by(idx, f64::total_cmp);
    *nth
}

/// Arithmetic mean of a reserve sample (`O(n)` without sorting).
pub fn mean(values: &[f64]) -> Result<f64, ValidationError> {
    if values.is_empty() {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_finite_slice(values)?;
    Ok(values.iter().sum::<f64>() / values.len() as f64)
}

/// Sample variance of a reserve sample.
pub fn variance(values: &[f64]) -> Result<f64, ValidationError> {
    let avg = mean(values)?;
    if values.len() <= 1 {
        return Ok(0.0);
    }
    Ok(values
        .iter()
        .map(|value| (value - avg).powi(2))
        .sum::<f64>()
        / (values.len() as f64 - 1.0))
}

/// Sample standard deviation of a reserve sample.
pub fn std_dev(values: &[f64]) -> Result<f64, ValidationError> {
    Ok(variance(values)?.sqrt())
}

/// Empirical quantile with upper-rank selection (`ceil(p * N) - 1`) in `O(n)`.
pub fn quantile(values: &[f64], p: f64) -> Result<f64, ValidationError> {
    validate_prob(p)?;
    if values.is_empty() {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_finite_slice(values)?;
    let mut buf = values.to_vec();
    Ok(select_quantile_in_place(&mut buf, p))
}

/// Discrete least-quantile Value at Risk `VaR_p(X) = F^-(p) = inf { x : F(x) >= p }`
/// (CAS Monograph No. 15, Algorithm 3.4.2): returns `ordered[ceil(p * N) - 1]`
/// with IEEE-754 near-integer boundary snapping in `O(n)`.
pub fn var_least(values: &[f64], p: f64) -> Result<f64, ValidationError> {
    quantile(values, p)
}

/// Tail Conditional Expectation (average of observations `>= quantile(values, p)`).
///
/// Note: on discrete distributions with ties or non-integer `(1 - p) * N`,
/// this TCE convention can differ from the exact subadditive integral TVaR.
/// Use [`tvar_coherent`] for the CAS Monograph No. 15 coherent TVaR.
pub fn tvar(values: &[f64], p: f64) -> Result<f64, ValidationError> {
    let threshold = quantile(values, p)?;
    let mut sum = 0.0;
    let mut count = 0usize;
    for &value in values {
        if value >= threshold {
            sum += value;
            count += 1;
        }
    }
    if count == 0 {
        return Ok(threshold);
    }
    Ok(sum / count as f64)
}

/// Exact coherent Tail Value at Risk `TVaR_p(X) = (1 / (1 - p)) ∫_p^1 F^-(s) ds`
/// for equally likely discrete outcomes (CAS Monograph No. 15, Algorithm 3.4.3).
///
/// Guaranteed subadditive on all finite discrete distributions, including
/// distributions with ties or non-integer tail counts `N · (1 - p)`.
pub fn tvar_coherent(values: &[f64], p: f64) -> Result<f64, ValidationError> {
    validate_prob(p)?;
    if values.is_empty() {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_finite_slice(values)?;
    let n_len = values.len();
    if p <= 0.0 {
        return Ok(values.iter().sum::<f64>() / n_len as f64);
    }
    if p >= 1.0 {
        return Ok(values.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    }
    let pn = p * n_len as f64;
    let n_idx = (pn.floor() as usize).min(n_len - 1);
    if n_idx + 1 >= n_len {
        return Ok(values.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    }
    // O(n) partial selection: buf[n_idx] == ordered[n_idx] and buf[n_idx+1..] is
    // the exact upper-tail multiset (sum is order-invariant).
    let mut buf = values.to_vec();
    let (_, nth, upper) = buf.select_nth_unstable_by(n_idx, f64::total_cmp);
    let t1: f64 = upper.iter().sum();
    let t2 = ((n_idx + 1) as f64 - pn) * *nth;
    Ok((t1 + t2) / ((1.0 - p) * n_len as f64))
}

/// Range Value at Risk `RVaR_{p_low, p_high}(X)` (Cont et al. 2010; CAS
/// Monograph No. 15, §3.3): exact average of the quantile function `F^-(s)`
/// between `p_low` and `p_high` (`0 <= p_low < p_high <= 1`).
pub fn rvar(values: &[f64], p_low: f64, p_high: f64) -> Result<f64, ValidationError> {
    validate_prob(p_low)?;
    validate_prob(p_high)?;
    if !(p_low < p_high) {
        return Err(ValidationError::InvalidParameter(
            "rvar requires p_low < p_high",
        ));
    }
    validate_finite_slice(values)?;
    if values.is_empty() {
        return Err(ValidationError::ShapeMismatch);
    }
    let n_len = values.len();
    let n = n_len as f64;
    let lo_idx = ((p_low * n).floor() as usize).min(n_len - 1);
    let hi_idx = ((p_high * n).ceil() as usize)
        .saturating_sub(1)
        .min(n_len - 1);
    let mut work = values.to_vec();
    // Cascading O(n) partial selection isolates the exact [lo_idx..=hi_idx] band.
    let (_, hi_val_ref, _) = work.select_nth_unstable_by(hi_idx, f64::total_cmp);
    let hi_val = *hi_val_ref;
    let (lo_val, mid_slice) = if lo_idx == hi_idx {
        (hi_val, &work[0..0])
    } else {
        let (_, lo_val_ref, right) = work[..hi_idx].select_nth_unstable_by(lo_idx, f64::total_cmp);
        (*lo_val_ref, &*right)
    };
    let mut integral = 0.0;
    let lo_w = ((lo_idx + 1) as f64 / n).min(p_high) - (lo_idx as f64 / n).max(p_low);
    if lo_w > 0.0 {
        integral += lo_w * lo_val;
    }
    if hi_idx > lo_idx {
        let hi_w = ((hi_idx + 1) as f64 / n).min(p_high) - (hi_idx as f64 / n).max(p_low);
        if hi_w > 0.0 {
            integral += hi_w * hi_val;
        }
        let inv_n = 1.0 / n;
        for &x in mid_slice {
            integral += inv_n * x;
        }
    }
    Ok(integral / (p_high - p_low))
}

/// Empirical probability that reserve exceeds `threshold`.
pub fn probability_exceeding(values: &[f64], threshold: f64) -> Result<f64, ValidationError> {
    if !threshold.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    validate_finite_slice(values)?;
    if values.is_empty() {
        return Err(ValidationError::ShapeMismatch);
    }
    let count = values.iter().filter(|value| **value > threshold).count();
    Ok(count as f64 / values.len() as f64)
}

/// Build a standard reserve-distribution summary bundle in `O(n)` via cascading
/// partial selection (no full `O(n log n)` sort required).
pub fn summarize(values: &[f64]) -> Result<ReserveDistributionSummary, ValidationError> {
    let mut scratch = Vec::with_capacity(values.len());
    summarize_with_scratch(values, &mut scratch, false)
}

/// Allocation-reusing internal variant of [`summarize`] that shares a caller-
/// provided scratch buffer across repeated summary calls (e.g., portfolio total
/// plus every origin year) and optionally skips redundant finiteness validation
/// when the caller has already enforced finiteness on every element.
pub(crate) fn summarize_with_scratch(
    values: &[f64],
    buf: &mut Vec<f64>,
    already_finite: bool,
) -> Result<ReserveDistributionSummary, ValidationError> {
    if values.is_empty() {
        return Err(ValidationError::ShapeMismatch);
    }
    let len = values.len();
    let first = values[0];

    // Fast O(1)-probed constant-series shortcut (e.g., fully developed origin 0
    // when tail == 1.0, or zero-dispersion fixtures): skips variance, copy, and
    // all five partial selections.
    if first.is_finite()
        && first == values[len - 1]
        && first == values[len / 2]
        && values.iter().all(|&v| v == first)
    {
        return Ok(ReserveDistributionSummary {
            mean: first,
            variance: 0.0,
            std_dev: 0.0,
            p50: first,
            p75: first,
            p90: first,
            p95: first,
            p99: first,
        });
    }

    // 4-way unrolled Neumaier compensated sum: auto-vectorizes while eliminating
    // floating-point drift on N >= 10^6 simulation draws. Any NaN/Inf poisons
    // the sum irreversibly so a single post-loop finiteness check suffices.
    let mut s0 = 0.0;
    let mut c0 = 0.0;
    let mut s1 = 0.0;
    let mut c1 = 0.0;
    let mut s2 = 0.0;
    let mut c2 = 0.0;
    let mut s3 = 0.0;
    let mut c3 = 0.0;
    let neumaier = |sum: &mut f64, comp: &mut f64, v: f64| {
        let t = *sum + v;
        if sum.abs() >= v.abs() {
            *comp += (*sum - t) + v;
        } else {
            *comp += (v - t) + *sum;
        }
        *sum = t;
    };
    let (chunks, rem) = values.as_chunks::<4>();
    for chunk in chunks {
        neumaier(&mut s0, &mut c0, chunk[0]);
        neumaier(&mut s1, &mut c1, chunk[1]);
        neumaier(&mut s2, &mut c2, chunk[2]);
        neumaier(&mut s3, &mut c3, chunk[3]);
    }
    for &v in rem {
        neumaier(&mut s0, &mut c0, v);
    }
    let total_sum = ((s0 + c0) + (s1 + c1)) + ((s2 + c2) + (s3 + c3));
    if !already_finite && !total_sum.is_finite() {
        return Err(ValidationError::NotFinite);
    }

    let n = len as f64;
    let avg = total_sum / n;
    let var = if len <= 1 {
        0.0
    } else {
        let mut v0 = 0.0;
        let mut v1 = 0.0;
        let mut v2 = 0.0;
        let mut v3 = 0.0;
        let (chunks, rem) = values.as_chunks::<4>();
        for chunk in chunks {
            let d0 = chunk[0] - avg;
            let d1 = chunk[1] - avg;
            let d2 = chunk[2] - avg;
            let d3 = chunk[3] - avg;
            v0 += d0 * d0;
            v1 += d1 * d1;
            v2 += d2 * d2;
            v3 += d3 * d3;
        }
        for &val in rem {
            let d = val - avg;
            v0 += d * d;
        }
        ((v0 + v1) + (v2 + v3)) / (n - 1.0)
    };

    buf.clear();
    buf.extend_from_slice(values);
    let i50 = quantile_rank_index(len, 0.50);
    let i75 = quantile_rank_index(len, 0.75);
    let i90 = quantile_rank_index(len, 0.90);
    let i95 = quantile_rank_index(len, 0.95);
    let i99 = quantile_rank_index(len, 0.99);

    // Every element is verified finite above, so a direct `<`/`>` comparator
    // lowers to a single `ucomisd` instead of `f64::total_cmp` bit manipulation.
    let cmp_finite = |a: &f64, b: &f64| {
        if *a < *b {
            std::cmp::Ordering::Less
        } else if *a > *b {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    };

    let (_, p50_ref, rest) = buf.select_nth_unstable_by(i50, cmp_finite);
    let p50 = *p50_ref;
    let (p75, rest) = if i75 == i50 {
        (p50, rest)
    } else {
        let (_, r, next) = rest.select_nth_unstable_by(i75 - i50 - 1, cmp_finite);
        (*r, next)
    };
    let (p90, rest) = if i90 == i75 {
        (p75, rest)
    } else {
        let (_, r, next) = rest.select_nth_unstable_by(i90 - i75 - 1, cmp_finite);
        (*r, next)
    };
    let (p95, rest) = if i95 == i90 {
        (p90, rest)
    } else {
        let (_, r, next) = rest.select_nth_unstable_by(i95 - i90 - 1, cmp_finite);
        (*r, next)
    };
    let p99 = if i99 == i95 {
        p95
    } else {
        let (_, r, _) = rest.select_nth_unstable_by(i99 - i95 - 1, cmp_finite);
        *r
    };

    Ok(ReserveDistributionSummary {
        mean: avg,
        variance: var,
        std_dev: var.sqrt(),
        p50,
        p75,
        p90,
        p95,
        p99,
    })
}

/// Exact empirical Continuous Ranked Probability Score (Gneiting & Raftery 2007)
/// for a predictive sample `draws` evaluated at realized outcome `observed`:
///
/// ```text
/// CRPS(F_hat, y) = (1 / N) sum_i |x_i - y| - (1 / (2 N^2)) sum_{i,j} |x_i - x_j|
/// ```
///
/// Computed in `O(N log N)` via the sorted order-statistic identity
/// `sum_{i,j} |x_i - x_j| = 2 sum_{k=0}^{N-1} (2k - N + 1) x_{(k)}`.
/// Lower is better; reduces to `|x - y|` for a degenerate point forecast.
pub fn crps_empirical(draws: &[f64], observed: f64) -> Result<f64, ValidationError> {
    if !observed.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    let xs = sorted(draws)?;
    let n = xs.len() as f64;
    let term1 = xs.iter().map(|&x| (x - observed).abs()).sum::<f64>() / n;
    let shift = xs[0];
    let spread = xs
        .iter()
        .enumerate()
        .map(|(k, &x)| (2.0 * k as f64 - n + 1.0) * (x - shift))
        .sum::<f64>()
        / (n * n);
    Ok((term1 - spread).max(0.0))
}

/// Empirical Probability Integral Transform `F_hat(observed) = (1 / N) #{x_i <= observed}`
/// in `[0, 1]` for evaluating predictive calibration of `draws` against `observed`.
pub fn pit_empirical(draws: &[f64], observed: f64) -> Result<f64, ValidationError> {
    if !observed.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if draws.is_empty() {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_finite_slice(draws)?;
    let count = draws.iter().filter(|&&x| x <= observed).count();
    Ok(count as f64 / draws.len() as f64)
}

/// Proper central `(1 - alpha)` Winkler / Gneiting–Raftery interval score
/// evaluated at `observed` using empirical quantiles `[q_{alpha/2}, q_{1 - alpha/2}]`:
///
/// ```text
/// S_alpha(l, u; y) = (u - l) + (2 / alpha) * max(l - y, 0) + (2 / alpha) * max(y - u, 0)
/// ```
///
/// Requires `0 < alpha < 1`. Lower is better.
pub fn interval_score(draws: &[f64], observed: f64, alpha: f64) -> Result<f64, ValidationError> {
    if !observed.is_finite() || !alpha.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(ValidationError::InvalidProbability { value: alpha });
    }
    validate_finite_slice(draws)?;
    if draws.is_empty() {
        return Err(ValidationError::ShapeMismatch);
    }
    let n = draws.len();
    let idx_lo = quantile_rank_index(n, 0.5 * alpha);
    let idx_hi = quantile_rank_index(n, 1.0 - 0.5 * alpha);
    let mut work = draws.to_vec();
    let (_, upper_ref, _) = work.select_nth_unstable_by(idx_hi, f64::total_cmp);
    let upper = *upper_ref;
    let lower = if idx_lo == idx_hi {
        upper
    } else {
        *work[..idx_hi]
            .select_nth_unstable_by(idx_lo, f64::total_cmp)
            .1
    };
    let width = upper - lower;
    let penalty = (2.0 / alpha) * ((lower - observed).max(0.0) + (observed - upper).max(0.0));
    Ok(width + penalty)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * (1.0 + b.abs())
    }

    #[test]
    fn computes_moments_and_quantiles() {
        let values = [10.0, 20.0, 30.0, 40.0, 50.0];
        let summary = summarize(&values).expect("summary");

        assert!(close(summary.mean, 30.0));
        assert!(close(summary.variance, 250.0));
        assert!(close(summary.std_dev, 250.0_f64.sqrt()));
        assert!(close(summary.p50, 30.0));
        assert!(close(summary.p95, 50.0));
    }

    #[test]
    fn computes_tvar_and_probability_exceeding() {
        let values = [10.0, 20.0, 30.0, 40.0, 50.0];

        assert!(close(tvar(&values, 0.8).expect("tvar"), 45.0));
        assert!(close(
            probability_exceeding(&values, 25.0).expect("probability"),
            3.0 / 5.0
        ));
    }

    #[test]
    fn cas_monograph_15_coherent_tvar_is_subadditive_on_discrete_example() {
        // CAS Monograph 15 §3.2 discrete example (20 equiprobable slots representing
        // probs 0.90, 0.05, 0.05): Unit 1 = (0x18, 0, 10), Unit 2 = (0x18, 10, 0),
        // Total = (0x18, 10, 10).
        let mut u1 = vec![0.0; 20];
        u1[19] = 10.0;
        let mut u2 = vec![0.0; 20];
        u2[18] = 10.0;
        let tot: Vec<f64> = u1.iter().zip(u2.iter()).map(|(a, b)| a + b).collect();

        // At p = 0.90: VaR fails subadditivity (0 + 0 < 10), while coherent TVaR
        // satisfies subadditivity (5 + 5 = 10 == TVaR(total)).
        assert!(close(tvar_coherent(&u1, 0.90).unwrap(), 5.0));
        assert!(close(tvar_coherent(&u2, 0.90).unwrap(), 5.0));
        assert!(close(tvar_coherent(&tot, 0.90).unwrap(), 10.0));

        // Coherent TVaR matches RVaR(p, 1.0) identically.
        let sample = [10.0, 20.0, 30.0, 40.0, 50.0];
        assert!(close(
            tvar_coherent(&sample, 0.75).unwrap(),
            rvar(&sample, 0.75, 1.0).unwrap()
        ));
        assert!(close(tvar_coherent(&sample, 0.0).unwrap(), 30.0));
        assert!(close(tvar_coherent(&sample, 1.0).unwrap(), 50.0));
    }

    #[test]
    fn rejects_invalid_quantile_probability() {
        let values = [1.0, 2.0, 3.0];
        assert_eq!(
            quantile(&values, 1.1),
            Err(ValidationError::InvalidProbability { value: 1.1 })
        );
    }

    #[test]
    fn empirical_scoring_rules_satisfy_exact_identities() {
        // Point forecast: CRPS reduces to |x - y|.
        let point = [100.0; 10];
        assert!(close(crps_empirical(&point, 115.0).unwrap(), 15.0));
        assert!(close(crps_empirical(&point, 100.0).unwrap(), 0.0));

        // Direct O(N^2) verification on an unsorted sample.
        let draws: [f64; 5] = [14.0, 2.0, 9.0, 20.0, 5.0];
        let y = 11.0_f64;
        let n = draws.len() as f64;
        let term1: f64 = draws.iter().map(|&x| (x - y).abs()).sum::<f64>() / n;
        let mut pair_sum = 0.0;
        for &a in &draws {
            for &b in &draws {
                pair_sum += (a - b).abs();
            }
        }
        let expected_crps = term1 - pair_sum / (2.0 * n * n);
        assert!(close(crps_empirical(&draws, y).unwrap(), expected_crps));

        // PIT: 3 of 5 draws (2, 5, 9) are <= 11.0.
        assert!(close(pit_empirical(&draws, y).unwrap(), 0.6));
        assert!(close(pit_empirical(&draws, 1.0).unwrap(), 0.0));
        assert!(close(pit_empirical(&draws, 25.0).unwrap(), 1.0));

        // Winkler interval score: sorted draws [2, 5, 9, 14, 20], alpha = 0.2 -> [q_0.1=2, q_0.9=20].
        assert!(close(interval_score(&draws, 11.0, 0.2).unwrap(), 18.0));
        assert!(close(
            interval_score(&draws, 25.0, 0.2).unwrap(),
            18.0 + (2.0 / 0.2) * 5.0
        ));
        assert_eq!(
            interval_score(&draws, 11.0, 0.0),
            Err(ValidationError::InvalidProbability { value: 0.0 })
        );
    }
}
