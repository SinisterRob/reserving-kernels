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
#[derive(Debug, Clone, PartialEq)]
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
        return Err(ValidationError::InvalidMaturity { index: 0, value: p });
    }
    Ok(())
}

fn sorted(values: &[f64]) -> Result<Vec<f64>, ValidationError> {
    if values.is_empty() {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_finite_slice(values)?;
    let mut out = values.to_vec();
    out.sort_by(f64::total_cmp);
    Ok(out)
}

#[inline]
fn quantile_sorted(ordered: &[f64], p: f64) -> f64 {
    if p <= 0.0 {
        return ordered[0];
    }
    if p >= 1.0 {
        return *ordered.last().expect("non-empty");
    }
    let idx = ((p * ordered.len() as f64).ceil() as usize).saturating_sub(1);
    ordered[idx.min(ordered.len() - 1)]
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

/// Empirical quantile with upper-rank selection (`ceil(p * N) - 1`).
pub fn quantile(values: &[f64], p: f64) -> Result<f64, ValidationError> {
    validate_prob(p)?;
    let ordered = sorted(values)?;
    Ok(quantile_sorted(&ordered, p))
}

/// Discrete least-quantile Value at Risk `VaR_p(X)` (CAS Monograph No. 15,
/// Algorithm 3.4.2): returns `ordered[floor(p * N)]` (clamped at `N - 1`).
pub fn var_least(values: &[f64], p: f64) -> Result<f64, ValidationError> {
    validate_prob(p)?;
    let ordered = sorted(values)?;
    let n = ordered.len();
    if p >= 1.0 {
        return Ok(ordered[n - 1]);
    }
    let idx = ((p * n as f64).floor() as usize).min(n - 1);
    Ok(ordered[idx])
}

/// Tail Conditional Expectation (average of observations `>= quantile(values, p)`).
///
/// Note: on discrete distributions with ties or non-integer `(1 - p) * N`,
/// this TCE convention can differ from the exact subadditive integral TVaR.
/// Use [`tvar_coherent`] for the CAS Monograph No. 15 coherent TVaR.
pub fn tvar(values: &[f64], p: f64) -> Result<f64, ValidationError> {
    let threshold = quantile(values, p)?;
    let tail: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| *value >= threshold)
        .collect();
    if tail.is_empty() {
        return Ok(threshold);
    }
    Ok(tail.iter().sum::<f64>() / tail.len() as f64)
}

/// Exact coherent Tail Value at Risk `TVaR_p(X) = (1 / (1 - p)) ∫_p^1 F^-(s) ds`
/// for equally likely discrete outcomes (CAS Monograph No. 15, Algorithm 3.4.3).
///
/// Guaranteed subadditive on all finite discrete distributions, including
/// distributions with ties or non-integer tail counts `N · (1 - p)`.
pub fn tvar_coherent(values: &[f64], p: f64) -> Result<f64, ValidationError> {
    validate_prob(p)?;
    let ordered = sorted(values)?;
    let n_len = ordered.len();
    if p <= 0.0 {
        return Ok(ordered.iter().sum::<f64>() / n_len as f64);
    }
    if p >= 1.0 {
        return Ok(ordered[n_len - 1]);
    }
    let pn = p * n_len as f64;
    let n_idx = (pn.floor() as usize).min(n_len - 1);
    if n_idx + 1 >= n_len {
        return Ok(ordered[n_len - 1]);
    }
    let t1: f64 = ordered[(n_idx + 1)..].iter().sum();
    let t2 = ((n_idx + 1) as f64 - pn) * ordered[n_idx];
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

/// Build a standard reserve-distribution summary bundle (single sort pass).
pub fn summarize(values: &[f64]) -> Result<ReserveDistributionSummary, ValidationError> {
    let ordered = sorted(values)?;
    let n = ordered.len() as f64;
    let avg = ordered.iter().sum::<f64>() / n;
    let var = if ordered.len() <= 1 {
        0.0
    } else {
        ordered.iter().map(|v| (v - avg).powi(2)).sum::<f64>() / (n - 1.0)
    };
    Ok(ReserveDistributionSummary {
        mean: avg,
        variance: var,
        std_dev: var.sqrt(),
        p50: quantile_sorted(&ordered, 0.50),
        p75: quantile_sorted(&ordered, 0.75),
        p90: quantile_sorted(&ordered, 0.90),
        p95: quantile_sorted(&ordered, 0.95),
        p99: quantile_sorted(&ordered, 0.99),
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
            Err(ValidationError::InvalidMaturity {
                index: 0,
                value: 1.1
            })
        );
    }
}
