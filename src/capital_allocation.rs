//! Coherent capital allocation via Tasche's (2008) Euler principle and
//! Myers & Read's (2001) marginal default-put allocation.
//!
//! Implements:
//!
//! * D. Tasche, *"Capital Allocation to Business Units and Sub-Portfolios: the
//!   Euler Principle"*, in *Pillar II in the New Basel Accord* (2008),
//!   pp. 423–453.
//! * S. C. Myers & J. A. Read, *"Capital Allocation for Insurance Companies"*,
//!   *Journal of Risk and Insurance*, Vol. 68, No. 4 (2001), pp. 545–580.
//! * S. J. Mildenhall & J. A. Major, *Introduction to Capital Modeling and
//!   Portfolio Management*, CAS Monograph No. 15, §5.
//!
//! ## Exact Conservation Identities
//!
//! For aligned multi-line scenario samples `X_0, ..., X_{d-1}` (each of length
//! `N`) with aggregate portfolio loss `S_k = sum_{i=0}^{d-1} X_{i, k}`:
//!
//! * **Euler TVaR (`euler_tvar_allocation`)**:
//!   `AC_i(TVaR_alpha) = E[X_i | S >= VaR_alpha(S)]` using exact tie-symmetric
//!   quantile-integral weights matching [`crate::reserve_distribution::tvar_coherent`]
//!   (CAS Monograph No. 15, Algorithm 3.4.3). Guarantees exact full allocation
//!   `sum_i AC_i == tvar_coherent(S, alpha)` even on discrete distributions
//!   with ties or non-integer `(1 - alpha) * N`.
//! * **Euler RVaR (`euler_rvar_allocation`)**:
//!   exact tie-symmetric allocation for Range VaR `RVaR_{p_low, p_high}(S)`,
//!   conserving `sum_i AC_i == rvar(S, p_low, p_high)`.
//! * **Euler Standard Deviation (`euler_sd_allocation`)**:
//!   `AC_i = E[X_i] + kappa * Cov(X_i, S) / sigma(S)`, conserving
//!   `sum_i AC_i == E[S] + kappa * sigma(S)`.
//! * **Myers & Read (2001) (`myers_read_allocation`)**:
//!   marginal default-value allocation equalizing `dD / dL_i = D / L` at asset
//!   level `A = L + Q`:
//!   `Q_i = L_i * (Q / L) + (E[X_i | S > A] - (L_i / L) * E[S | S > A])`,
//!   conserving `sum_i Q_i == Q` and `sum_i (L_i + Q_i) == A` identically.

use crate::reserve_distribution::{mean, rvar, tvar_coherent, var_least};
use crate::validate::{ValidationError, validate_finite_slice};

/// Output of an Euler coherent risk-measure capital allocation across `d` lines.
#[derive(Debug, Clone, PartialEq)]
pub struct EulerAllocationResult {
    /// Portfolio aggregate risk measure `rho(S)`.
    pub portfolio_risk_measure: f64,
    /// Portfolio expected loss `E[S]`.
    pub portfolio_mean: f64,
    /// Portfolio risk capital above mean `rho(S) - E[S]`.
    pub portfolio_risk_capital: f64,
    /// Standalone expected loss `E[X_i]` per line (length `d`).
    pub standalone_means: Vec<f64>,
    /// Standalone risk measure `rho(X_i)` per line (length `d`).
    pub standalone_risk_measures: Vec<f64>,
    /// Euler allocated total risk measure `AC_i` per line (`sum_i AC_i == rho(S)`).
    pub allocated_total: Vec<f64>,
    /// Euler allocated risk capital above mean `AC_i - E[X_i]` per line.
    pub allocated_capital: Vec<f64>,
    /// Allocation shares `AC_i / rho(S)` per line (summing to `1.0` when `rho(S) != 0`).
    pub allocation_shares: Vec<f64>,
    /// Diversification credit per line: `standalone_risk_measures[i] - allocated_total[i]`.
    pub diversification_credit: Vec<f64>,
    /// Portfolio diversification ratio `rho(S) / sum_i rho(X_i)`.
    pub diversification_ratio: f64,
}

/// Output of Myers & Read (2001) marginal default-put surplus allocation.
#[derive(Debug, Clone, PartialEq)]
pub struct MyersReadAllocationResult {
    /// Total portfolio expected liabilities `L = E[S]`.
    pub portfolio_liability: f64,
    /// Total portfolio assets `A = L + Q`.
    pub portfolio_assets: f64,
    /// Total portfolio surplus `Q = A - L`.
    pub portfolio_surplus: f64,
    /// Portfolio insolvency probability `P(S > A)`.
    pub default_probability: f64,
    /// Insolvency default put value `D(A) = E[(S - A)_+]`.
    pub default_put_value: f64,
    /// Portfolio default-value-to-liability ratio `d = D(A) / L`.
    pub default_value_ratio: f64,
    /// Expected liability `L_i = E[X_i]` per line (length `d`).
    pub line_liabilities: Vec<f64>,
    /// Conditional tail expectation `E[X_i | S > A]` per line (length `d`).
    pub conditional_tail_losses: Vec<f64>,
    /// Allocated assets `A_i = L_i + Q_i` per line (`sum_i A_i == A`).
    pub allocated_assets: Vec<f64>,
    /// Allocated surplus `Q_i` per line (`sum_i Q_i == Q`).
    pub allocated_surplus: Vec<f64>,
    /// Allocated surplus-to-liability ratio `s_i = Q_i / L_i` per line.
    pub surplus_ratios: Vec<f64>,
    /// Portfolio aggregate surplus-to-liability ratio `s = Q / L`.
    pub portfolio_surplus_ratio: f64,
}

/// Compute exact coherent Euler allocation of `TVaR_alpha(S)` across `lines`
/// (Tasche 2008; CAS Monograph No. 15, §5).
///
/// Scenario weights match [`crate::reserve_distribution::tvar_coherent`]
/// (Algorithm 3.4.3) and distribute boundary-tie mass symmetrically across all
/// tied scenarios, ensuring exact permutation invariance and
/// `sum_i allocated_total[i] == tvar_coherent(S, alpha)`.
pub fn euler_tvar_allocation(
    lines: &[&[f64]],
    alpha: f64,
) -> Result<EulerAllocationResult, ValidationError> {
    if !alpha.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if !(0.0..=1.0).contains(&alpha) {
        return Err(ValidationError::InvalidMaturity {
            index: 0,
            value: alpha,
        });
    }
    let (d, n, total) = validate_and_sum_lines(lines)?;
    let weights = coherent_interval_weights(&total, alpha, 1.0);

    let portfolio_risk_measure = tvar_coherent(&total, alpha)?;
    let standalone_risk_measures = lines
        .iter()
        .map(|col| tvar_coherent(col, alpha))
        .collect::<Result<Vec<_>, _>>()?;

    build_euler_result(
        lines,
        &total,
        &weights,
        d,
        n,
        portfolio_risk_measure,
        standalone_risk_measures,
    )
}

/// Compute exact tie-symmetric Euler allocation of Range VaR
/// `RVaR_{p_low, p_high}(S)` across `lines` (Cont et al. 2010; CAS Monograph 15).
pub fn euler_rvar_allocation(
    lines: &[&[f64]],
    p_low: f64,
    p_high: f64,
) -> Result<EulerAllocationResult, ValidationError> {
    if !p_low.is_finite() || !p_high.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if !(0.0..=1.0).contains(&p_low) {
        return Err(ValidationError::InvalidMaturity {
            index: 0,
            value: p_low,
        });
    }
    if !(0.0..=1.0).contains(&p_high) {
        return Err(ValidationError::InvalidMaturity {
            index: 1,
            value: p_high,
        });
    }
    if !(p_low < p_high) {
        return Err(ValidationError::InvalidParameter(
            "euler_rvar_allocation requires p_low < p_high",
        ));
    }
    let (d, n, total) = validate_and_sum_lines(lines)?;
    let weights = coherent_interval_weights(&total, p_low, p_high);

    let portfolio_risk_measure = rvar(&total, p_low, p_high)?;
    let standalone_risk_measures = lines
        .iter()
        .map(|col| rvar(col, p_low, p_high))
        .collect::<Result<Vec<_>, _>>()?;

    build_euler_result(
        lines,
        &total,
        &weights,
        d,
        n,
        portfolio_risk_measure,
        standalone_risk_measures,
    )
}

/// Compute exact Euler allocation under the Standard Deviation principle
/// `rho_kappa(S) = E[S] + kappa * sigma(S)` (Tasche 2008, Example 2.3):
///
/// `AC_i = E[X_i] + kappa * Cov(X_i, S) / sigma(S)`.
pub fn euler_sd_allocation(
    lines: &[&[f64]],
    kappa: f64,
) -> Result<EulerAllocationResult, ValidationError> {
    if !kappa.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if kappa < 0.0 {
        return Err(ValidationError::Negative);
    }
    let (d, n, total) = validate_and_sum_lines(lines)?;
    let nf = n as f64;
    let denom = if n > 1 { nf - 1.0 } else { 1.0 };

    let portfolio_mean = mean(&total)?;
    let total_var = if n > 1 {
        total
            .iter()
            .map(|&s| (s - portfolio_mean).powi(2))
            .sum::<f64>()
            / denom
    } else {
        0.0
    };
    let total_sd = total_var.sqrt();
    let portfolio_risk_measure = portfolio_mean + kappa * total_sd;

    let mut standalone_means = Vec::with_capacity(d);
    let mut standalone_risk_measures = Vec::with_capacity(d);
    let mut allocated_total = Vec::with_capacity(d);
    let mut allocated_capital = Vec::with_capacity(d);

    for &col in lines {
        let mu_i = mean(col)?;
        let var_i = if n > 1 {
            col.iter().map(|&x| (x - mu_i).powi(2)).sum::<f64>() / denom
        } else {
            0.0
        };
        let cov_is = if n > 1 {
            col.iter()
                .zip(total.iter())
                .map(|(&x, &s)| (x - mu_i) * (s - portfolio_mean))
                .sum::<f64>()
                / denom
        } else {
            0.0
        };
        let cap_i = if total_sd > 0.0 {
            kappa * cov_is / total_sd
        } else {
            0.0
        };
        standalone_means.push(mu_i);
        standalone_risk_measures.push(mu_i + kappa * var_i.sqrt());
        allocated_capital.push(cap_i);
        allocated_total.push(mu_i + cap_i);
    }

    Ok(finalize_euler_struct(
        portfolio_risk_measure,
        portfolio_mean,
        standalone_means,
        standalone_risk_measures,
        allocated_total,
        allocated_capital,
    ))
}

/// Compute Myers & Read (2001) marginal default-put surplus allocation across
/// `lines` at explicit total portfolio assets `total_assets` (`A = L + Q`).
///
/// Requires `total_assets < max(S)` so that the insolvency event `{S > A}` has
/// strictly positive empirical probability (`P(S > A) > 0`).
pub fn myers_read_allocation(
    lines: &[&[f64]],
    total_assets: f64,
) -> Result<MyersReadAllocationResult, ValidationError> {
    if !total_assets.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    let (d, n, total) = validate_and_sum_lines(lines)?;
    let nf = n as f64;

    let portfolio_liability: f64 = total.iter().sum::<f64>() / nf;
    if !(portfolio_liability > 0.0) {
        return Err(ValidationError::InvalidParameter(
            "Myers & Read allocation requires strictly positive expected portfolio liability",
        ));
    }
    let line_liabilities: Vec<f64> = lines
        .iter()
        .map(|col| col.iter().sum::<f64>() / nf)
        .collect();

    let mut tail_indices = Vec::new();
    let mut default_put_sum = 0.0;
    let mut tail_total_sum = 0.0;
    for (k, &sk) in total.iter().enumerate() {
        if sk > total_assets {
            tail_indices.push(k);
            default_put_sum += sk - total_assets;
            tail_total_sum += sk;
        }
    }
    if tail_indices.is_empty() {
        return Err(ValidationError::InvalidParameter(
            "Myers & Read allocation requires total_assets < max(S) for positive default probability",
        ));
    }

    let n_tail = tail_indices.len() as f64;
    let default_probability = n_tail / nf;
    let default_put_value = default_put_sum / nf;
    let default_value_ratio = default_put_value / portfolio_liability;
    let portfolio_surplus = total_assets - portfolio_liability;
    let portfolio_surplus_ratio = portfolio_surplus / portfolio_liability;
    let cond_tail_total = tail_total_sum / n_tail;

    let mut conditional_tail_losses = Vec::with_capacity(d);
    let mut allocated_surplus = Vec::with_capacity(d);
    let mut allocated_assets = Vec::with_capacity(d);
    let mut surplus_ratios = Vec::with_capacity(d);

    for (i, &col) in lines.iter().enumerate() {
        let mut line_tail_sum = 0.0;
        for &idx in &tail_indices {
            line_tail_sum += col[idx];
        }
        let e_xi_tail = line_tail_sum / n_tail;
        let li = line_liabilities[i];
        let li_share = li / portfolio_liability;

        // Myers & Read (2001) Eq. (11):
        // Q_i = L_i * (Q / L) + (E[X_i | S > A] - (L_i / L) * E[S | S > A]).
        let qi = li * portfolio_surplus_ratio + (e_xi_tail - li_share * cond_tail_total);
        let ai = li + qi;
        conditional_tail_losses.push(e_xi_tail);
        allocated_surplus.push(qi);
        allocated_assets.push(ai);
        surplus_ratios.push(if li.abs() > 0.0 { qi / li } else { 0.0 });
    }

    Ok(MyersReadAllocationResult {
        portfolio_liability,
        portfolio_assets: total_assets,
        portfolio_surplus,
        default_probability,
        default_put_value,
        default_value_ratio,
        line_liabilities,
        conditional_tail_losses,
        allocated_assets,
        allocated_surplus,
        surplus_ratios,
        portfolio_surplus_ratio,
    })
}

/// Solve for the portfolio asset level `A` matching a target default-value-to-
/// liability ratio `target_default_ratio = E[(S - A)_+] / E[S]` and compute
/// the resulting Myers & Read (2001) surplus allocation.
pub fn myers_read_from_default_ratio(
    lines: &[&[f64]],
    target_default_ratio: f64,
) -> Result<MyersReadAllocationResult, ValidationError> {
    if !target_default_ratio.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if !(target_default_ratio > 0.0 && target_default_ratio < 1.0) {
        return Err(ValidationError::InvalidParameter(
            "target_default_ratio must lie strictly inside (0, 1)",
        ));
    }
    let (_d, n, total) = validate_and_sum_lines(lines)?;
    let nf = n as f64;
    let l = total.iter().sum::<f64>() / nf;
    if !(l > 0.0) {
        return Err(ValidationError::InvalidParameter(
            "Myers & Read allocation requires positive expected liability",
        ));
    }
    let target_put = target_default_ratio * l;
    let s_min = total.iter().copied().fold(f64::INFINITY, f64::min);
    let s_max = total.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let eval_put = |a: f64| -> f64 { total.iter().map(|&s| (s - a).max(0.0)).sum::<f64>() / nf };

    let mut lo = s_min.min(0.0);
    while eval_put(lo) < target_put {
        lo -= l.max(1.0);
    }
    let mut hi = s_max;
    if eval_put(lo) <= target_put {
        return Err(ValidationError::InvalidParameter(
            "target_default_ratio cannot be achieved on this loss sample",
        ));
    }

    // D(A) is strictly decreasing and convex on [s_min, s_max]; bisection converges to 1e-13.
    for _ in 0..90 {
        let mid = 0.5 * (lo + hi);
        if eval_put(mid) > target_put {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let assets = 0.5 * (lo + hi);
    // Guard against landing on s_max numerically.
    let safe_assets = if assets >= s_max {
        var_least(&total, 1.0 - 1.0 / nf)?
    } else {
        assets
    };
    myers_read_allocation(lines, safe_assets)
}

fn validate_and_sum_lines(lines: &[&[f64]]) -> Result<(usize, usize, Vec<f64>), ValidationError> {
    if lines.is_empty() {
        return Err(ValidationError::ShapeMismatch);
    }
    let n = lines[0].len();
    if n == 0 {
        return Err(ValidationError::ShapeMismatch);
    }
    for &col in lines {
        if col.len() != n {
            return Err(ValidationError::ShapeMismatch);
        }
        validate_finite_slice(col)?;
    }
    let mut total = vec![0.0; n];
    for &col in lines {
        for k in 0..n {
            total[k] += col[k];
        }
    }
    Ok((lines.len(), n, total))
}

/// Exact tie-symmetric scenario weights `w_s` (`sum_s w_s == 1`) such that
/// `sum_s w_s * total[s] == (1 / (p_high - p_low)) * int_{p_low}^{p_high} F^-(u) du`.
fn coherent_interval_weights(total: &[f64], p_low: f64, p_high: f64) -> Vec<f64> {
    let n = total.len();
    let nf = n as f64;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| total[a].total_cmp(&total[b]));

    let mut weights = vec![0.0; n];
    if p_low >= 1.0 {
        // Put equal weight on all scenarios tied at the maximum.
        let max_val = total[order[n - 1]];
        let mut lo = n - 1;
        while lo > 0 && total[order[lo - 1]] == max_val {
            lo -= 1;
        }
        let count = (n - lo) as f64;
        for &idx in &order[lo..n] {
            weights[idx] = 1.0 / count;
        }
        return weights;
    }

    let width = p_high - p_low;
    // First assign raw quantile-overlap mass to each sorted rank r in [0, n).
    let mut rank_mass = vec![0.0; n];
    for r in 0..n {
        let lo = (r as f64 / nf).max(p_low);
        let hi = ((r + 1) as f64 / nf).min(p_high);
        if hi > lo {
            rank_mass[r] = (hi - lo) / width;
        }
    }

    // Symmetrize weights across every group of tied total-loss values so that
    // the allocation is strictly permutation-invariant while conserving exact sum.
    let mut r = 0usize;
    while r < n {
        let val = total[order[r]];
        let mut end = r + 1;
        while end < n && total[order[end]] == val {
            end += 1;
        }
        let group_mass: f64 = rank_mass[r..end].iter().sum();
        let per_member = group_mass / (end - r) as f64;
        for &idx in &order[r..end] {
            weights[idx] = per_member;
        }
        r = end;
    }
    weights
}

fn build_euler_result(
    lines: &[&[f64]],
    total: &[f64],
    weights: &[f64],
    d: usize,
    n: usize,
    portfolio_risk_measure: f64,
    standalone_risk_measures: Vec<f64>,
) -> Result<EulerAllocationResult, ValidationError> {
    let portfolio_mean = mean(total)?;
    let mut standalone_means = Vec::with_capacity(d);
    let mut allocated_total = Vec::with_capacity(d);
    let mut allocated_capital = Vec::with_capacity(d);

    for &col in lines {
        let mu_i = mean(col)?;
        let mut ac_i = 0.0;
        for k in 0..n {
            ac_i += weights[k] * col[k];
        }
        standalone_means.push(mu_i);
        allocated_total.push(ac_i);
        allocated_capital.push(ac_i - mu_i);
    }

    Ok(finalize_euler_struct(
        portfolio_risk_measure,
        portfolio_mean,
        standalone_means,
        standalone_risk_measures,
        allocated_total,
        allocated_capital,
    ))
}

fn finalize_euler_struct(
    portfolio_risk_measure: f64,
    portfolio_mean: f64,
    standalone_means: Vec<f64>,
    standalone_risk_measures: Vec<f64>,
    allocated_total: Vec<f64>,
    allocated_capital: Vec<f64>,
) -> EulerAllocationResult {
    let portfolio_risk_capital = portfolio_risk_measure - portfolio_mean;
    let allocation_shares: Vec<f64> = allocated_total
        .iter()
        .map(|&ac| {
            if portfolio_risk_measure.abs() > 0.0 {
                ac / portfolio_risk_measure
            } else {
                0.0
            }
        })
        .collect();
    let diversification_credit: Vec<f64> = standalone_risk_measures
        .iter()
        .zip(allocated_total.iter())
        .map(|(&s, &a)| s - a)
        .collect();
    let sum_standalone: f64 = standalone_risk_measures.iter().sum();
    let diversification_ratio = if sum_standalone.abs() > 0.0 {
        portfolio_risk_measure / sum_standalone
    } else {
        1.0
    };

    EulerAllocationResult {
        portfolio_risk_measure,
        portfolio_mean,
        portfolio_risk_capital,
        standalone_means,
        standalone_risk_measures,
        allocated_total,
        allocated_capital,
        allocation_shares,
        diversification_credit,
        diversification_ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euler_tvar_conserves_sum_exactly_with_ties_and_non_integer_tail() {
        // 10 scenarios with boundary ties and non-integer tail at alpha = 0.73.
        let line_a = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 70.0, 90.0, 120.0];
        let line_b = [15.0, 10.0, 25.0, 30.0, 35.0, 40.0, 50.0, 50.0, 60.0, 110.0];
        let line_c = [5.0, 12.0, 18.0, 22.0, 28.0, 30.0, 30.0, 30.0, 40.0, 70.0];

        let res = euler_tvar_allocation(&[&line_a, &line_b, &line_c], 0.73).expect("euler tvar");
        let sum_ac: f64 = res.allocated_total.iter().sum();
        let sum_cap: f64 = res.allocated_capital.iter().sum();

        assert!((sum_ac - res.portfolio_risk_measure).abs() < 1e-12);
        assert!((sum_cap - res.portfolio_risk_capital).abs() < 1e-12);
        assert!(res.diversification_ratio <= 1.0 + 1e-12);
        for i in 0..3 {
            assert!(res.allocated_total[i] <= res.standalone_risk_measures[i] + 1e-12);
        }
    }

    #[test]
    fn myers_read_conserves_surplus_and_assets_exactly() {
        let line_a = [80.0, 90.0, 95.0, 100.0, 105.0, 115.0, 130.0, 180.0];
        let line_b = [40.0, 45.0, 50.0, 52.0, 55.0, 60.0, 70.0, 140.0];
        let assets = 220.0;

        let mr = myers_read_allocation(&[&line_a, &line_b], assets).expect("myers read");
        let sum_surplus: f64 = mr.allocated_surplus.iter().sum();
        let sum_assets: f64 = mr.allocated_assets.iter().sum();

        assert!((sum_surplus - mr.portfolio_surplus).abs() < 1e-12);
        assert!((sum_assets - mr.portfolio_assets).abs() < 1e-12);
        assert!(mr.default_probability > 0.0);
        assert!(mr.default_put_value > 0.0);

        // Round-trip via target default ratio recovers the same allocation.
        let mr2 = myers_read_from_default_ratio(&[&line_a, &line_b], mr.default_value_ratio)
            .expect("from ratio");
        assert!((mr2.portfolio_assets - assets).abs() < 1e-6);
    }
}
