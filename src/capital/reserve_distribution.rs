//! Shared summary math and coherent risk measures for reserve-simulation output.
//!
//! Includes both the legacy upper-rank empirical quantile / TCE helpers
//! ([`quantile`], [`tvar`]) and the exact subadditive discrete risk-measure
//! algorithms from CAS Monograph No. 15 (Mildenhall & Major, *Introduction to
//! Capital Modeling and Portfolio Management*, §3.2–§3.4):
//!
//! * [`var_least`] — Algorithm 3.4.2 (least quantile `F^-(p)`).
//! * [`tvar_coherent`] — Algorithm 3.4.3 (exact integral Tail Value at Risk
//!   `TVaR_p(X) = (1 / (1 - p)) ∫_p^1 F^-(s) ds`, guaranteed subadditive even
//!   on discrete / tied empirical distributions).
//! * [`rvar`] — Range Value at Risk `RVaR_{p_low, p_high}(X)` (Cont et al.
//!   2010; CAS Monograph No. 15, §3.3).

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
    let ordered = sorted(values)?;
    let n = ordered.len() as f64;
    // Exact integral ∫_{p_low}^{p_high} F^-(s) ds over piecewise-constant step function
    // where F^-(s) = ordered[j] on [j/N, (j+1)/N).
    let mut integral = 0.0;
    for (j, &x) in ordered.iter().enumerate() {
        let lo = (j as f64 / n).max(p_low);
        let hi = ((j + 1) as f64 / n).min(p_high);
        if hi > lo {
            integral += (hi - lo) * x;
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

    // 4-way unrolled sum (auto-vectorizes cleanly). By IEEE-754 arithmetic, any
    // NaN or ±inf element poisons the accumulator irreversibly to non-finite, so
    // a single post-loop check validates finiteness with zero per-element branches.
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let (chunks, rem) = values.as_chunks::<4>();
    for chunk in chunks {
        s0 += chunk[0];
        s1 += chunk[1];
        s2 += chunk[2];
        s3 += chunk[3];
    }
    for &v in rem {
        s0 += v;
    }
    let total_sum = (s0 + s1) + (s2 + s3);
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

    let (_, p50_ref, rest) = buf.select_nth_unstable_by(i50, f64::total_cmp);
    let p50 = *p50_ref;
    let (p75, rest) = if i75 == i50 {
        (p50, rest)
    } else {
        let (_, r, next) = rest.select_nth_unstable_by(i75 - i50 - 1, f64::total_cmp);
        (*r, next)
    };
    let (p90, rest) = if i90 == i75 {
        (p75, rest)
    } else {
        let (_, r, next) = rest.select_nth_unstable_by(i90 - i75 - 1, f64::total_cmp);
        (*r, next)
    };
    let (p95, rest) = if i95 == i90 {
        (p90, rest)
    } else {
        let (_, r, next) = rest.select_nth_unstable_by(i95 - i90 - 1, f64::total_cmp);
        (*r, next)
    };
    let p99 = if i99 == i95 {
        p95
    } else {
        let (_, r, _) = rest.select_nth_unstable_by(i99 - i95 - 1, f64::total_cmp);
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
}
