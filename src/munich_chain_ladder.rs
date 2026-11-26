//! Munich Chain Ladder (MCL) bivariate paid/incurred reserving method.
//!
//! Implements:
//!
//! * Quarg & Mack (2004), *Munich Chain Ladder: A Reserving Method that Reduces
//!   the Gap between IBNR Projections Based on Paid Losses and IBNR Projections
//!   Based on Incurred Losses*, Blätter der DGVFM, Vol. 26, No. 4, pp. 597–630;
//!   reprinted in English in *Variance*, Vol. 2, No. 2 (2008), pp. 266–299.
//!
//! ## Overview
//!
//! Classical univariate Chain Ladder applied separately to paid (`P`) and
//! incurred (`I`) triangles often yields materially divergent ultimates when
//! case-reserve adequacy shifts across origin years. Munich Chain Ladder
//! links the two triangles by regressing standardized paid and incurred
//! development-factor residuals on prior-age `I/P` and `P/I` ratio residuals
//! through the origin (`λ^P` and `λ^I`), then projecting the lower triangles
//! jointly:
//!
//! ```text
//! P̂_{i, k+1} = P̂_{i, k} · [ f_k^P + λ^P · (σ_k^P / τ_k^P) · (Î_{i,k}/P̂_{i,k} - q_k^{-1}) ]
//! Î_{i, k+1} = Î_{i, k} · [ f_k^I + λ^I · (σ_k^I / τ_k^I) · (P̂_{i,k}/Î_{i,k} - q_k) ]
//! ```

use crate::ldf::volume_weighted;
use crate::triangle::latest_diagonal;
use crate::types::TriangleView;
use crate::validate::{ValidationError, validate_tail, validate_triangle};

/// Full bivariate Munich Chain Ladder result.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MunichChainLadder {
    /// Traditional univariate volume-weighted paid LDFs (length `n - 1`).
    pub paid_ldfs: Vec<f64>,
    /// Traditional univariate volume-weighted incurred LDFs (length `n - 1`).
    pub incurred_ldfs: Vec<f64>,
    /// Mack conditional standard deviations `σ_k^P` for paid links (length `n - 1`).
    pub sigma_paid: Vec<f64>,
    /// Mack conditional standard deviations `σ_k^I` for incurred links (length `n - 1`).
    pub sigma_incurred: Vec<f64>,
    /// Volume-weighted `P/I` ratios `q_k = Σ P_{i,k} / Σ I_{i,k}` (length `n`).
    pub q_paid_to_incurred: Vec<f64>,
    /// Volume-weighted `I/P` ratios `q_k^{-1} = Σ I_{i,k} / Σ P_{i,k}` (length `n`).
    pub q_incurred_to_paid: Vec<f64>,
    /// Conditional standard deviations `τ_k^P` of `I/P` around `q_k^{-1}` (length `n`).
    pub tau_paid: Vec<f64>,
    /// Conditional standard deviations `τ_k^I` of `P/I` around `q_k` (length `n`).
    pub tau_incurred: Vec<f64>,
    /// Regression slope `λ^P` linking paid link residuals to prior `I/P` residuals.
    pub lambda_paid: f64,
    /// Regression slope `λ^I` linking incurred link residuals to prior `P/I` residuals.
    pub lambda_incurred: f64,
    /// Completed `n × n` paid cumulative triangle (row-major).
    pub full_paid: Vec<f64>,
    /// Completed `n × n` incurred cumulative triangle (row-major).
    pub full_incurred: Vec<f64>,
    /// Munich Chain Ladder paid ultimates by origin (including tail).
    pub paid_ultimates: Vec<f64>,
    /// Munich Chain Ladder incurred ultimates by origin (including tail).
    pub incurred_ultimates: Vec<f64>,
    /// Munich Chain Ladder paid reserves (`paid_ultimates - paid_latest`).
    pub paid_reserves: Vec<f64>,
    /// Munich Chain Ladder incurred reserves (`incurred_ultimates - incurred_latest`).
    pub incurred_reserves: Vec<f64>,
    /// Traditional univariate chain-ladder paid ultimates for gap comparison.
    pub traditional_paid_ultimates: Vec<f64>,
    /// Traditional univariate chain-ladder incurred ultimates for gap comparison.
    pub traditional_incurred_ultimates: Vec<f64>,
    /// Total Munich paid reserve.
    pub total_paid_reserve: f64,
    /// Total Munich incurred reserve.
    pub total_incurred_reserve: f64,
}

impl MunichChainLadder {
    /// Aggregate absolute gap between paid and incurred ultimates under Munich CL.
    pub fn munich_total_ultimate_gap(&self) -> f64 {
        let p: f64 = self.paid_ultimates.iter().sum();
        let i: f64 = self.incurred_ultimates.iter().sum();
        (p - i).abs()
    }

    /// Aggregate absolute gap between paid and incurred ultimates under traditional CL.
    pub fn traditional_total_ultimate_gap(&self) -> f64 {
        let p: f64 = self.traditional_paid_ultimates.iter().sum();
        let i: f64 = self.traditional_incurred_ultimates.iter().sum();
        (p - i).abs()
    }
}

fn mack_sigma_sq(t: &TriangleView<'_>, ldfs: &[f64]) -> Vec<f64> {
    let n = t.n_origins;
    let mut s = vec![0.0; n - 1];
    for k in 0..(n - 1) {
        let nk = n - k - 1;
        if nk >= 2 {
            let mut acc = 0.0;
            for i in 0..nk {
                let c_k = t.get(i, k);
                let c_kp1 = t.get(i, k + 1);
                let ratio = c_kp1 / c_k;
                acc += c_k * (ratio - ldfs[k]).powi(2);
            }
            s[k] = acc / (nk as f64 - 1.0);
        } else {
            s[k] = f64::NAN;
        }
    }
    extrapolate_last_sigma_sq(&mut s);
    s
}

fn extrapolate_last_sigma_sq(s: &mut [f64]) {
    let m = s.len();
    if m == 0 {
        return;
    }
    let last = m - 1;
    if !s[last].is_finite() {
        s[last] = if m >= 3 {
            let a = s[last - 1];
            let b = s[last - 2];
            if a > 0.0 && b > 0.0 {
                (a * a / b).min(a).min(b).max(0.0)
            } else {
                0.0
            }
        } else if m == 2 {
            s[0].max(0.0)
        } else {
            0.0
        };
    }
}

/// Fit and project the Quarg & Mack (2004) Munich Chain Ladder model from
/// aligned cumulative paid and incurred triangles.
///
/// Both triangles must share square `n × n` geometry (`n >= 3`) with strictly
/// positive observed cells so `P/I` and `I/P` ratios are well defined.
/// Non-monotone cumulative incurred rows (case-reserve releases) are permitted
/// natively as in Quarg & Mack (2004).
pub fn project(
    paid: &TriangleView<'_>,
    incurred: &TriangleView<'_>,
    paid_tail: f64,
    incurred_tail: f64,
) -> Result<MunichChainLadder, ValidationError> {
    validate_triangle(paid)?;
    validate_triangle(incurred)?;
    validate_tail(paid_tail)?;
    validate_tail(incurred_tail)?;

    if paid.n_origins != incurred.n_origins || paid.n_devs != incurred.n_devs {
        return Err(ValidationError::ShapeMismatch);
    }
    let n = paid.n_origins;
    if n < 3 {
        return Err(ValidationError::InvalidParameter(
            "Munich Chain Ladder requires at least a 3x3 triangle",
        ));
    }

    for i in 0..n {
        for j in 0..(n - i) {
            if paid.get(i, j) <= 0.0 {
                return Err(ValidationError::DivideByZero("paid cell"));
            }
            if incurred.get(i, j) <= 0.0 {
                return Err(ValidationError::DivideByZero("incurred cell"));
            }
        }
    }

    let paid_ldfs = volume_weighted(paid);
    let incurred_ldfs = volume_weighted(incurred);
    if paid_ldfs.iter().any(|f| !f.is_finite() || *f <= 0.0)
        || incurred_ldfs.iter().any(|f| !f.is_finite() || *f <= 0.0)
    {
        return Err(ValidationError::DivideByZero("univariate LDF"));
    }

    let sigma_paid: Vec<f64> = mack_sigma_sq(paid, &paid_ldfs)
        .into_iter()
        .map(|v| v.max(0.0).sqrt())
        .collect();
    let sigma_incurred: Vec<f64> = mack_sigma_sq(incurred, &incurred_ldfs)
        .into_iter()
        .map(|v| v.max(0.0).sqrt())
        .collect();

    // Paid-to-incurred ratios q_k and incurred-to-paid ratios q_k^{-1} = 1 / q_k,
    // plus conditional variance scales tau_k^P and tau_k^I (Quarg & Mack 2004 §4.2).
    let mut q_paid_to_incurred = vec![0.0; n];
    let mut q_incurred_to_paid = vec![0.0; n];
    let mut tau_paid_sq = vec![f64::NAN; n];
    let mut tau_incurred_sq = vec![f64::NAN; n];

    for k in 0..n {
        let mk = n - k;
        let mut sum_p = 0.0;
        let mut sum_i = 0.0;
        for i in 0..mk {
            sum_p += paid.get(i, k);
            sum_i += incurred.get(i, k);
        }
        let qk = sum_p / sum_i;
        let qk_inv = sum_i / sum_p;
        q_paid_to_incurred[k] = qk;
        q_incurred_to_paid[k] = qk_inv;

        if mk >= 2 {
            let mut acc_p = 0.0;
            let mut acc_i = 0.0;
            for i in 0..mk {
                let p = paid.get(i, k);
                let inc = incurred.get(i, k);
                acc_p += p * (inc / p - qk_inv).powi(2);
                acc_i += inc * (p / inc - qk).powi(2);
            }
            tau_paid_sq[k] = acc_p / (mk as f64 - 1.0);
            tau_incurred_sq[k] = acc_i / (mk as f64 - 1.0);
        }
    }
    extrapolate_last_sigma_sq(&mut tau_paid_sq);
    extrapolate_last_sigma_sq(&mut tau_incurred_sq);

    let tau_paid: Vec<f64> = tau_paid_sq.into_iter().map(|v| v.max(0.0).sqrt()).collect();
    let tau_incurred: Vec<f64> = tau_incurred_sq
        .into_iter()
        .map(|v| v.max(0.0).sqrt())
        .collect();

    // Estimate lambda^P and lambda^I via OLS through the origin over observed
    // links with >= 2 pairs (k = 0..n-2; at k = n-2 only 1 pair exists so link residual is 0).
    let mut num_p = 0.0;
    let mut den_p = 0.0;
    let mut num_i = 0.0;
    let mut den_i = 0.0;

    for k in 0..(n - 2) {
        let nk = n - k - 1;
        let sp = sigma_paid[k];
        let si = sigma_incurred[k];
        let tp = tau_paid[k];
        let ti = tau_incurred[k];

        for i in 0..nk {
            let p_k = paid.get(i, k);
            let p_kp1 = paid.get(i, k + 1);
            let i_k = incurred.get(i, k);
            let i_kp1 = incurred.get(i, k + 1);

            if sp > 0.0 && tp > 0.0 {
                let r_p = (p_kp1 / p_k - paid_ldfs[k]) / sp * p_k.sqrt();
                let r_qinv = (i_k / p_k - q_incurred_to_paid[k]) / tp * p_k.sqrt();
                num_p += r_qinv * r_p;
                den_p += r_qinv * r_qinv;
            }
            if si > 0.0 && ti > 0.0 {
                let r_i = (i_kp1 / i_k - incurred_ldfs[k]) / si * i_k.sqrt();
                let r_q = (p_k / i_k - q_paid_to_incurred[k]) / ti * i_k.sqrt();
                num_i += r_q * r_i;
                den_i += r_q * r_q;
            }
        }
    }

    let lambda_paid = if den_p > 0.0 { num_p / den_p } else { 0.0 };
    let lambda_incurred = if den_i > 0.0 { num_i / den_i } else { 0.0 };

    // Complete lower triangles jointly step-by-step across development columns.
    let mut full_paid = vec![0.0; n * n];
    let mut full_incurred = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..(n - i) {
            full_paid[i * n + j] = paid.get(i, j);
            full_incurred[i * n + j] = incurred.get(i, j);
        }
    }

    for k in 0..(n - 1) {
        let coeff_p = if tau_paid[k] > 0.0 {
            lambda_paid * sigma_paid[k] / tau_paid[k]
        } else {
            0.0
        };
        let coeff_i = if tau_incurred[k] > 0.0 {
            lambda_incurred * sigma_incurred[k] / tau_incurred[k]
        } else {
            0.0
        };

        for i in (n - 1 - k)..n {
            let p_prev = full_paid[i * n + k];
            let i_prev = full_incurred[i * n + k];
            if !(p_prev > 0.0) || !(i_prev > 0.0) {
                return Err(ValidationError::NotFinite);
            }
            let f_p_adj = paid_ldfs[k] + coeff_p * (i_prev / p_prev - q_incurred_to_paid[k]);
            let f_i_adj = incurred_ldfs[k] + coeff_i * (p_prev / i_prev - q_paid_to_incurred[k]);
            full_paid[i * n + k + 1] = p_prev * f_p_adj;
            full_incurred[i * n + k + 1] = i_prev * f_i_adj;
        }
    }

    let paid_latest = latest_diagonal(paid);
    let incurred_latest = latest_diagonal(incurred);
    let mut paid_ultimates = vec![0.0; n];
    let mut incurred_ultimates = vec![0.0; n];
    let mut paid_reserves = vec![0.0; n];
    let mut incurred_reserves = vec![0.0; n];

    for i in 0..n {
        let p_end = full_paid[i * n + (n - 1)];
        let i_end = full_incurred[i * n + (n - 1)];
        if !p_end.is_finite() || !i_end.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if p_end < 0.0 || i_end < 0.0 {
            return Err(ValidationError::Negative);
        }
        paid_ultimates[i] = p_end * paid_tail;
        incurred_ultimates[i] = i_end * incurred_tail;
        paid_reserves[i] = paid_ultimates[i] - paid_latest[i];
        incurred_reserves[i] = incurred_ultimates[i] - incurred_latest[i];
    }

    let paid_cdfs = crate::ldf::cdf_from_ldf(&paid_ldfs, paid_tail);
    let incurred_cdfs = crate::ldf::cdf_from_ldf(&incurred_ldfs, incurred_tail);
    let mut traditional_paid_ultimates = vec![0.0; n];
    let mut traditional_incurred_ultimates = vec![0.0; n];
    for i in 0..n {
        let d = n - 1 - i;
        traditional_paid_ultimates[i] = paid_latest[i] * paid_cdfs[d];
        traditional_incurred_ultimates[i] = incurred_latest[i] * incurred_cdfs[d];
    }

    let total_paid_reserve: f64 = paid_reserves.iter().sum();
    let total_incurred_reserve: f64 = incurred_reserves.iter().sum();

    Ok(MunichChainLadder {
        paid_ldfs,
        incurred_ldfs,
        sigma_paid,
        sigma_incurred,
        q_paid_to_incurred,
        q_incurred_to_paid,
        tau_paid,
        tau_incurred,
        lambda_paid,
        lambda_incurred,
        full_paid,
        full_incurred,
        paid_ultimates,
        incurred_ultimates,
        paid_reserves,
        incurred_reserves,
        traditional_paid_ultimates,
        traditional_incurred_ultimates,
        total_paid_reserve,
        total_incurred_reserve,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Triangle;

    /// Quarg & Mack (2004) / Variance (2008) 7×7 benchmark paid and incurred triangles.
    fn quarg_mack_2004_triangles() -> (Triangle, Triangle) {
        const NAN: f64 = f64::NAN;
        #[rustfmt::skip]
        let paid = vec![
            576.0, 1804.0, 1970.0, 2024.0, 2074.0, 2102.0, 2131.0,
            866.0, 1948.0, 2162.0, 2232.0, 2284.0, 2348.0,    NAN,
           1412.0, 3758.0, 4252.0, 4416.0, 4494.0,    NAN,    NAN,
           2286.0, 5292.0, 5724.0, 5850.0,    NAN,    NAN,    NAN,
           1868.0, 3778.0, 4648.0,    NAN,    NAN,    NAN,    NAN,
           1442.0, 4010.0,    NAN,    NAN,    NAN,    NAN,    NAN,
           2044.0,    NAN,    NAN,    NAN,    NAN,    NAN,    NAN,
        ];
        #[rustfmt::skip]
        let incurred = vec![
            978.0, 2104.0, 2134.0, 2144.0, 2174.0, 2182.0, 2174.0,
           1844.0, 2552.0, 2466.0, 2480.0, 2508.0, 2454.0,    NAN,
           2904.0, 4354.0, 4698.0, 4600.0, 4644.0,    NAN,    NAN,
           3502.0, 5958.0, 6070.0, 6142.0,    NAN,    NAN,    NAN,
           2812.0, 4882.0, 4852.0,    NAN,    NAN,    NAN,    NAN,
           2642.0, 4406.0,    NAN,    NAN,    NAN,    NAN,    NAN,
           5022.0,    NAN,    NAN,    NAN,    NAN,    NAN,    NAN,
        ];
        (Triangle::new(7, 7, paid), Triangle::new(7, 7, incurred))
    }

    #[test]
    fn munich_chain_ladder_reduces_paid_incurred_gap_on_quarg_mack_2004() {
        let (paid, incurred) = quarg_mack_2004_triangles();
        let mcl = project(&paid.view(), &incurred.view(), 1.0, 1.0).expect("mcl");

        // In Quarg & Mack (2004), both lambda^P and lambda^I are positive on this
        // dataset: high I/P predicts faster subsequent paid development (lambda^P > 0),
        // and high P/I (thin case reserves) predicts upward subsequent incurred
        // development (lambda^I > 0).
        assert!(
            (mcl.lambda_paid - 0.636_021).abs() < 1e-4,
            "lambda_paid={}",
            mcl.lambda_paid
        );
        assert!(
            (mcl.lambda_incurred - 0.436_187).abs() < 1e-4,
            "lambda_incurred={}",
            mcl.lambda_incurred
        );
        // The core property of MCL: dramatically shrinks the paid/incurred ultimate gap,
        // especially on the immature youngest origin year (2300.5 -> 150.8, >90% reduction).
        let trad_youngest_gap =
            (mcl.traditional_paid_ultimates[6] - mcl.traditional_incurred_ultimates[6]).abs();
        let mcl_youngest_gap = (mcl.paid_ultimates[6] - mcl.incurred_ultimates[6]).abs();
        assert!(
            mcl_youngest_gap < trad_youngest_gap * 0.10,
            "youngest gap: munich={mcl_youngest_gap} vs trad={trad_youngest_gap}"
        );
        assert!(
            mcl.munich_total_ultimate_gap() < mcl.traditional_total_ultimate_gap() * 0.45,
            "total gap: munich={} vs traditional={}",
            mcl.munich_total_ultimate_gap(),
            mcl.traditional_total_ultimate_gap()
        );
    }
}
