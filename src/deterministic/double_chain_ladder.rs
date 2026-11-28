//! Martínez-Miranda, Nielsen & Verrall (2012) Double Chain Ladder (DCL).
//!
//! Implements:
//!
//! * M. D. Martínez-Miranda, J. P. Nielsen & R. J. Verrall (2012), *Double Chain
//!   Ladder*, ASTIN Bulletin, Vol. 42, No. 1, pp. 59–76.
//!
//! Combines a **cumulative paid-loss triangle** $X$ and a **cumulative reported
//! claim-count triangle** $N$ to separate:
//!
//! * **Reporting delay distribution** $\hat{\beta}_j$ ($\sum_j \hat{\beta}_j = 1$)
//!   from the chain ladder on reported counts $N$.
//! * **Payment / settlement delay distribution** $\hat{\pi}_l$ solved by exact
//!   forward deconvolution of the paid development proportions $\tilde{\beta}_j$:
//!   $$\tilde{\beta}_j = \sum_{l=0}^{j} \hat{\beta}_{j-l} \hat{\pi}_l$$
//! * **Underwriting-year severity inflation** $\hat{\gamma}_i = \tilde{\alpha}_i / \hat{\alpha}_i$
//!   (ratio of paid CL ultimate to reported-count CL ultimate).
//! * **RBNS** (Reported But Not Settled) vs. **IBNR** (Incurred But Not Reported)
//!   reserve decomposition, both within the run-off square ($j \le n - 1$) and
//!   including the DCL-implied tail ($j \ge n$).

use crate::deterministic::chain_ladder::ChainLadderEngine;
use crate::deterministic::ldf::DevelopmentFactorEngine;
use crate::preprocessing::triangle::{cumulative_to_incremental, latest_diagonal};
use crate::types::TriangleView;
use crate::validate::{
    MonotonicityPolicy, ValidationError, validate_monotone_cumulative, validate_triangle,
};

/// Result of Martínez-Miranda, Nielsen & Verrall (2012) Double Chain Ladder.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct DoubleChainLadderResult {
    /// Ultimate reported claim counts $\hat{\alpha}_i$ from count CL, length $n$.
    pub count_ultimate: Vec<f64>,
    /// Reporting delay distribution $\hat{\beta}_j$ ($\sum \hat{\beta}_j = 1$), length $n$.
    pub reporting_delay: Vec<f64>,
    /// Ultimate paid losses $\tilde{\alpha}_i$ from paid CL, length $n$.
    pub paid_cl_ultimate: Vec<f64>,
    /// Paid development proportions $\tilde{\beta}_j$ ($\sum \tilde{\beta}_j = 1$), length $n$.
    pub paid_delay: Vec<f64>,
    /// Deconvolved payment delay parameters $\hat{\pi}_l$ ($l = 0..n-1$), length $n$.
    pub payment_delay: Vec<f64>,
    /// Underwriting-year mean severity $\hat{\gamma}_i = \tilde{\alpha}_i / \hat{\alpha}_i$, length $n$.
    pub severity_inflation: Vec<f64>,
    /// Normalized severity inflation index $\hat{\iota}_i = \hat{\gamma}_i / \hat{\gamma}_0$, length $n$.
    pub normalized_inflation: Vec<f64>,
    /// Latest observed cumulative paid loss per origin, length $n$.
    pub latest_paid: Vec<f64>,
    /// In-triangle ($j \le n - 1$) RBNS reserve per origin (using observed reported counts), length $n$.
    pub rbns_reserve: Vec<f64>,
    /// In-triangle ($j \le n - 1$) IBNR reserve per origin (future reported counts), length $n$.
    pub ibnr_reserve: Vec<f64>,
    /// DCL-implied tail ($j \ge n$) RBNS reserve per origin, length $n$.
    pub rbns_tail: Vec<f64>,
    /// DCL-implied tail ($j \ge n$) IBNR reserve per origin, length $n$.
    pub ibnr_tail: Vec<f64>,
    /// Total in-triangle DCL reserve per origin (`rbns_reserve + ibnr_reserve`), length $n$.
    pub total_in_triangle_by_origin: Vec<f64>,
    /// Total DCL reserve per origin including tail (`rbns + ibnr + rbns_tail + ibnr_tail`), length $n$.
    pub total_with_tail_by_origin: Vec<f64>,
    /// Total in-triangle RBNS reserve across all origins.
    pub total_rbns: f64,
    /// Total in-triangle IBNR reserve across all origins.
    pub total_ibnr: f64,
    /// Total in-triangle DCL reserve (`total_rbns + total_ibnr`).
    pub total_in_triangle_reserve: f64,
    /// Total DCL tail reserve (`sum(rbns_tail + ibnr_tail)`).
    pub total_tail_reserve: f64,
    /// Total DCL reserve including tail.
    pub total_reserve_with_tail: f64,
    /// Reference standard paid Chain Ladder total reserve (for reconciliation).
    pub paid_cl_total_reserve: f64,
}

impl std::fmt::Display for DoubleChainLadderResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DoubleChainLadder(rbns={:.2}, ibnr={:.2}, total={:.2}, tail={:.2})",
            self.total_rbns,
            self.total_ibnr,
            self.total_in_triangle_reserve,
            self.total_tail_reserve
        )
    }
}

fn delay_proportions_from_cdfs(cdfs: &[f64]) -> Vec<f64> {
    let n = cdfs.len();
    let mut beta = vec![0.0; n];
    let mut prev_pct = 0.0;
    for j in 0..n {
        let pct = 1.0 / cdfs[j];
        beta[j] = pct - prev_pct;
        prev_pct = pct;
    }
    beta
}

/// Fit Double Chain Ladder (Martínez-Miranda, Nielsen & Verrall 2012) to
/// cumulative paid-loss and cumulative reported-count triangles under strict
/// monotonicity.
pub fn analyze(
    paid_cumulative: &TriangleView<'_>,
    counts_cumulative: &TriangleView<'_>,
) -> Result<DoubleChainLadderResult, ValidationError> {
    analyze_with_policy(
        paid_cumulative,
        counts_cumulative,
        MonotonicityPolicy::Strict,
    )
}

/// Fit Double Chain Ladder with an explicit [`MonotonicityPolicy`] and optional
/// custom severity inflation override (if `None`, uses empirical CL severity
/// $\hat{\gamma}_i = \tilde{\alpha}_i / \hat{\alpha}_i$).
pub fn analyze_with_policy(
    paid_cumulative: &TriangleView<'_>,
    counts_cumulative: &TriangleView<'_>,
    policy: MonotonicityPolicy,
) -> Result<DoubleChainLadderResult, ValidationError> {
    analyze_with_inflation(paid_cumulative, counts_cumulative, None, policy)
}

/// Fit Double Chain Ladder with an optional explicit severity inflation vector
/// $\gamma_i$ (length $n$) and [`MonotonicityPolicy`].
pub fn analyze_with_inflation(
    paid_cumulative: &TriangleView<'_>,
    counts_cumulative: &TriangleView<'_>,
    custom_severity: Option<&[f64]>,
    policy: MonotonicityPolicy,
) -> Result<DoubleChainLadderResult, ValidationError> {
    validate_triangle(paid_cumulative)?;
    validate_triangle(counts_cumulative)?;
    let n = paid_cumulative.n_origins;
    if n < 2 {
        return Err(ValidationError::InvalidParameter(
            "triangles must be at least 2x2",
        ));
    }
    if counts_cumulative.n_origins != n || counts_cumulative.n_devs != n {
        return Err(ValidationError::ShapeMismatch);
    }
    if matches!(policy, MonotonicityPolicy::Strict) {
        validate_monotone_cumulative(paid_cumulative)?;
        validate_monotone_cumulative(counts_cumulative)?;
    }

    let paid_factors =
        DevelopmentFactorEngine::volume_weighted_with_policy(paid_cumulative, 1.0, policy)?;
    let count_factors =
        DevelopmentFactorEngine::volume_weighted_with_policy(counts_cumulative, 1.0, policy)?;
    let paid_cl = ChainLadderEngine::project_with_policy(paid_cumulative, &paid_factors, policy)?;
    let count_cl =
        ChainLadderEngine::project_with_policy(counts_cumulative, &count_factors, policy)?;

    let reporting_delay = delay_proportions_from_cdfs(&count_factors.cdfs);
    let paid_delay = delay_proportions_from_cdfs(&paid_factors.cdfs);

    if !reporting_delay[0].is_finite() || reporting_delay[0] <= 0.0 {
        return Err(ValidationError::DivideByZero(
            "initial reporting delay proportion beta_0",
        ));
    }

    // Exact forward deconvolution: paid_delay[j] = sum_{l=0}^j reporting_delay[j-l] * pi[l].
    let mut payment_delay = vec![0.0; n];
    for l in 0..n {
        let mut rhs = paid_delay[l];
        for m in 0..l {
            rhs -= reporting_delay[l - m] * payment_delay[m];
        }
        payment_delay[l] = rhs / reporting_delay[0];
    }

    // Severity inflation gamma_i = paid_cl_ultimate[i] / count_ultimate[i].
    let severity_inflation = if let Some(custom) = custom_severity {
        if custom.len() != n {
            return Err(ValidationError::ShapeMismatch);
        }
        for &g in custom {
            if !g.is_finite() || g < 0.0 {
                return Err(ValidationError::InvalidParameter(
                    "custom severity inflation entries must be finite and non-negative",
                ));
            }
        }
        custom.to_vec()
    } else {
        let mut gamma = Vec::with_capacity(n);
        for i in 0..n {
            let alpha_i = count_cl.ultimates[i];
            if alpha_i <= 0.0 {
                return Err(ValidationError::DivideByZero("ultimate reported count"));
            }
            gamma.push(paid_cl.ultimates[i] / alpha_i);
        }
        gamma
    };

    let base_gamma = severity_inflation[0];
    let normalized_inflation: Vec<f64> = if base_gamma > 0.0 {
        severity_inflation.iter().map(|&g| g / base_gamma).collect()
    } else {
        vec![f64::NAN; n]
    };

    let inc_counts = cumulative_to_incremental(counts_cumulative);
    let latest_paid = latest_diagonal(paid_cumulative);

    // Prefix sums of payment_delay: pay_prefix[m] = sum_{l=0}^{m-1} payment_delay[l].
    // Collapses both RBNS and IBNR (in-triangle and tail) from O(n^3) to O(n^2).
    let mut pay_prefix = vec![0.0; n + 1];
    for l in 0..n {
        pay_prefix[l + 1] = pay_prefix[l] + payment_delay[l];
    }
    let pay_total = pay_prefix[n];

    let mut rbns_reserve = vec![0.0; n];
    let mut ibnr_reserve = vec![0.0; n];
    let mut rbns_tail = vec![0.0; n];
    let mut ibnr_tail = vec![0.0; n];

    for i in 0..n {
        let last_obs = n - 1 - i;
        let gamma_i = severity_inflation[i];
        let alpha_i = count_cl.ultimates[i];
        let row_base = i * n;

        // RBNS: claims already reported at k in 0..=last_obs.
        //   In-triangle (j in last_obs+1..n): lag l = j - k in (last_obs + 1 - k)..(n - k).
        //   Tail (j in n..k+n):               lag l = j - k in (n - k)..n.
        let mut rbns_in = 0.0;
        let mut rbns_tl = 0.0;
        for k in 0..=last_obs {
            let cnt = inc_counts[row_base + k];
            let p_nk = pay_prefix[n - k];
            rbns_in += cnt * (p_nk - pay_prefix[last_obs + 1 - k]);
            rbns_tl += cnt * (pay_total - p_nk);
        }
        rbns_reserve[i] = gamma_i * rbns_in;
        rbns_tail[i] = gamma_i * rbns_tl;

        // IBNR: future claims reported at k in (last_obs + 1)..n.
        //   In-triangle (j in k..n): lag l = j - k in 0..(n - k).
        //   Tail (j in n..k+n):      lag l = j - k in (n - k)..n.
        let mut ibnr_in = 0.0;
        let mut ibnr_tl = 0.0;
        for k in (last_obs + 1)..n {
            let rep = reporting_delay[k];
            let p_nk = pay_prefix[n - k];
            ibnr_in += rep * p_nk;
            ibnr_tl += rep * (pay_total - p_nk);
        }
        let scale = gamma_i * alpha_i;
        ibnr_reserve[i] = scale * ibnr_in;
        ibnr_tail[i] = scale * ibnr_tl;
    }

    let total_in_triangle_by_origin: Vec<f64> =
        (0..n).map(|i| rbns_reserve[i] + ibnr_reserve[i]).collect();
    let total_with_tail_by_origin: Vec<f64> = (0..n)
        .map(|i| rbns_reserve[i] + ibnr_reserve[i] + rbns_tail[i] + ibnr_tail[i])
        .collect();

    let total_rbns: f64 = rbns_reserve.iter().sum();
    let total_ibnr: f64 = ibnr_reserve.iter().sum();
    let total_in_triangle_reserve = total_rbns + total_ibnr;
    let total_tail_reserve: f64 = (0..n).map(|i| rbns_tail[i] + ibnr_tail[i]).sum();
    let total_reserve_with_tail = total_in_triangle_reserve + total_tail_reserve;
    let paid_cl_total_reserve: f64 = paid_cl.reserves.iter().sum();

    Ok(DoubleChainLadderResult {
        count_ultimate: count_cl.ultimates,
        reporting_delay,
        paid_cl_ultimate: paid_cl.ultimates,
        paid_delay,
        payment_delay,
        severity_inflation,
        normalized_inflation,
        latest_paid,
        rbns_reserve,
        ibnr_reserve,
        rbns_tail,
        ibnr_tail,
        total_in_triangle_by_origin,
        total_with_tail_by_origin,
        total_rbns,
        total_ibnr,
        total_in_triangle_reserve,
        total_tail_reserve,
        total_reserve_with_tail,
        paid_cl_total_reserve,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triangle;

    #[test]
    fn exact_deconvolution_and_cl_reproduction_on_noiseless_portfolio() {
        // Synthetic portfolio with exact reporting delay, payment delay, and severity inflation.
        let alpha = [100.0, 110.0, 120.0, 130.0, 140.0];
        let rep_delay = [0.60, 0.25, 0.10, 0.04, 0.01];
        let pay_delay = [0.50, 0.30, 0.15, 0.04, 0.01];
        let gamma = [1000.0, 1050.0, 1102.5, 1157.625, 1215.50625];
        let n = 5;

        let mut count_rows = Vec::new();
        let mut paid_rows = Vec::new();
        for i in 0..n {
            let mut c_row = Vec::new();
            let mut p_row = Vec::new();
            for j in 0..(n - i) {
                c_row.push(alpha[i] * rep_delay[j]);
                let mut paid_ij = 0.0;
                for k in 0..=j {
                    paid_ij += alpha[i] * rep_delay[k] * pay_delay[j - k];
                }
                p_row.push(gamma[i] * paid_ij);
            }
            count_rows.push(c_row);
            paid_rows.push(p_row);
        }

        let c_slices: Vec<&[f64]> = count_rows.iter().map(std::vec::Vec::as_slice).collect();
        let p_slices: Vec<&[f64]> = paid_rows.iter().map(std::vec::Vec::as_slice).collect();
        let counts = Triangle::try_from_incremental_rows(&c_slices).unwrap();
        let paid = Triangle::try_from_incremental_rows(&p_slices).unwrap();

        let dcl = analyze(&paid.view(), &counts.view()).unwrap();

        // Exact reproduction of standard paid CL total reserve in the triangle square!
        assert!((dcl.total_in_triangle_reserve - dcl.paid_cl_total_reserve).abs() < 1e-6);

        // Exact recovery of reporting delay and payment delay proportions (up to tail normalization).
        for j in 0..n {
            assert!((dcl.reporting_delay[j] - rep_delay[j]).abs() < 1e-10);
        }
        assert!(dcl.total_rbns > 0.0);
        assert!(dcl.total_ibnr > 0.0);
        // Normalized inflation recovers 5% annual trend.
        for i in 1..n {
            let ratio = dcl.normalized_inflation[i] / dcl.normalized_inflation[i - 1];
            assert!((ratio - 1.05).abs() < 1e-6);
        }
    }
}
