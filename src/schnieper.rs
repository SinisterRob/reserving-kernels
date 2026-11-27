//! Schnieper (1991) & Liu–Verrall (2009) True IBNR / IBNER Separation.
//!
//! Implements:
//!
//! * R. Schnieper (1991), *Separating True IBNR and IBNER Claims*, ASTIN
//!   Bulletin, Vol. 21, No. 1, pp. 111–127.
//! * H. Liu & R. Verrall (2009), *A Stochastic Model for Separating True IBNR
//!   and IBNER Claims*, ASTIN Bulletin, Vol. 39, No. 1, pp. 199–225.
//!
//! Separates cumulative incurred development into two distinct stochastic
//! components:
//!
//! $$C_{i,j} = C_{i,j-1} + N_{i,j} + D_{i,j}, \qquad j \ge 1$$
//!
//! where:
//! * $N_{i,j}$ is the incremental incurred amount from **newly reported claims**
//!   (true IBNR emergence), with $E[N_{i,j} \mid \mathcal{D}_{j-1}] = E_i \lambda_j$
//!   and $\mathrm{Var}(N_{i,j} \mid \mathcal{D}_{j-1}) = E_i \sigma_j^2$.
//! * $D_{i,j}$ is the signed incremental **development on previously known claims**
//!   (IBNER, positive for upward case strengthening and negative for releases),
//!   with $E[D_{i,j} \mid \mathcal{D}_{j-1}] = C_{i,j-1} \delta_j$ and
//!   $\mathrm{Var}(D_{i,j} \mid \mathcal{D}_{j-1}) = C_{i,j-1} \tau_j^2$.
//!
//! Unrolling the recursion yields the exact additive reserve decomposition:
//!
//! $$\hat{R}_i^{\text{IBNR}} + \hat{R}_i^{\text{IBNER}} = \hat{R}_i^{\text{Total}}$$
//!
//! together with Mack-style analytical process variance, parameter estimation
//! variance, and cross-origin covariance (Liu & Verrall 2009, §3). When
//! $N_{i,j} = 0$ for all $j \ge 1$, the model reduces identically to Mack (1993).

use crate::types::TriangleView;
use crate::validate::{ValidationError, validate_positive_finite_slice};

/// Result of Schnieper (1991) / Liu & Verrall (2009) True IBNR & IBNER separation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SchnieperResult {
    /// True IBNR emergence factors $\hat{\lambda}_j$ per development $j = 1..n-1$, length $n - 1$.
    pub lambda: Vec<f64>,
    /// IBNER development factors $\hat{\delta}_j$ per development $j = 1..n-1$, length $n - 1$.
    pub delta: Vec<f64>,
    /// True IBNR variance parameters $\hat{\sigma}_j^2$ per development $j = 1..n-1$, length $n - 1$.
    pub sigma_sq: Vec<f64>,
    /// IBNER variance parameters $\hat{\tau}_j^2$ per development $j = 1..n-1$, length $n - 1$.
    pub tau_sq: Vec<f64>,
    /// Latest observed cumulative incurred loss per origin, length $n$.
    pub latest_incurred: Vec<f64>,
    /// Projected True IBNR reserve per origin (future new claims + their development), length $n$.
    pub ibnr_reserve: Vec<f64>,
    /// Projected IBNER reserve per origin (future development on already-known claims), length $n$.
    pub ibner_reserve: Vec<f64>,
    /// Projected total reserve per origin (`ibnr_reserve + ibner_reserve`), length $n$.
    pub total_reserve_by_origin: Vec<f64>,
    /// Projected ultimate incurred per origin, length $n$.
    pub ultimate: Vec<f64>,
    /// Analytical process variance per origin, length $n$.
    pub process_variance: Vec<f64>,
    /// Analytical parameter estimation variance per origin, length $n$.
    pub parameter_variance: Vec<f64>,
    /// Analytical standard error per origin, length $n$.
    pub standard_error: Vec<f64>,
    /// Total True IBNR reserve across all origins.
    pub total_ibnr: f64,
    /// Total IBNER reserve across all origins.
    pub total_ibner: f64,
    /// Total reserve across all origins (`total_ibnr + total_ibner`).
    pub total_reserve: f64,
    /// Total analytical MSEP across all origins (including cross-origin covariance).
    pub total_mse: f64,
    /// Total analytical standard error.
    pub total_standard_error: f64,
    /// Total coefficient of variation.
    pub total_cv: f64,
}

impl std::fmt::Display for SchnieperResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Schnieper(ibnr={:.2}, ibner={:.2}, total={:.2}, se={:.2})",
            self.total_ibnr, self.total_ibner, self.total_reserve, self.total_standard_error
        )
    }
}

fn validate_square_signed_triangle(t: &TriangleView<'_>) -> Result<(), ValidationError> {
    if t.data.len() != t.n_origins * t.n_devs {
        return Err(ValidationError::ShapeMismatch);
    }
    if t.n_origins != t.n_devs {
        return Err(ValidationError::NotSquare);
    }
    let n = t.n_origins;
    for i in 0..n {
        for j in 0..(n - i) {
            if !t.get(i, j).is_finite() {
                return Err(ValidationError::NotFinite);
            }
        }
    }
    Ok(())
}

fn extrapolate_mack_min_rule(raw: &[Option<f64>]) -> Vec<f64> {
    let mut out = vec![0.0; raw.len()];
    for k in 0..raw.len() {
        if let Some(v) = raw[k] {
            out[k] = v.max(0.0);
        } else if k >= 2 {
            let a = out[k - 1];
            let b = out[k - 2];
            out[k] = if a > 0.0 && b > 0.0 {
                (a * a / b).min(a).min(b).max(0.0)
            } else {
                0.0
            };
        } else if k == 1 {
            out[k] = out[0].max(0.0);
        } else {
            out[k] = 0.0;
        }
    }
    out
}

/// Fit Schnieper (1991) / Liu & Verrall (2009) True IBNR & IBNER separation.
///
/// Arguments:
/// * `new_claims`: $n \times n$ incremental triangle $N_{i,j} \ge 0$ of newly
///   reported claim amounts ($N_{i,0}$ is initial incurred at age 0).
/// * `known_development`: $n \times n$ signed incremental triangle $D_{i,j}$
///   of development on previously known claims (column 0 is ignored; $D_{i,j}$
///   for $j \ge 1$ may be positive or negative).
/// * `exposures`: per-origin exposure vector $E_i > 0$ (length $n$).
pub fn analyze(
    new_claims: &TriangleView<'_>,
    known_development: &TriangleView<'_>,
    exposures: &[f64],
) -> Result<SchnieperResult, ValidationError> {
    validate_square_signed_triangle(new_claims)?;
    validate_square_signed_triangle(known_development)?;
    let n = new_claims.n_origins;
    if n < 2 {
        return Err(ValidationError::InvalidParameter(
            "triangles must be at least 2x2",
        ));
    }
    if known_development.n_origins != n || exposures.len() != n {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_positive_finite_slice(exposures, "exposures")?;

    // Reconstruct observed cumulative incurred triangle C_{i,j} = C_{i,j-1} + N_{i,j} + D_{i,j}.
    let mut cum = vec![f64::NAN; n * n];
    for i in 0..n {
        let n0 = new_claims.get(i, 0);
        if n0 < 0.0 {
            return Err(ValidationError::Negative);
        }
        cum[i * n] = n0;
        for j in 1..(n - i) {
            let nij = new_claims.get(i, j);
            if nij < 0.0 {
                return Err(ValidationError::Negative);
            }
            let dij = known_development.get(i, j);
            let next_c = cum[i * n + (j - 1)] + nij + dij;
            if next_c < 0.0 {
                return Err(ValidationError::Negative);
            }
            cum[i * n + j] = next_c;
        }
    }

    let n_links = n - 1;
    let mut lambda = vec![0.0; n_links];
    let mut delta = vec![0.0; n_links];
    let mut sum_e = vec![0.0; n_links];
    let mut sum_c = vec![0.0; n_links];
    let mut raw_sigma_sq: Vec<Option<f64>> = vec![None; n_links];
    let mut raw_tau_sq: Vec<Option<f64>> = vec![None; n_links];

    // Development index j = 1..n-1 corresponds to link index l = j - 1 in 0..n-1.
    for j in 1..n {
        let l = j - 1;
        let nk = n - j; // observed rows i = 0..n-1-j

        let mut num_n = 0.0;
        let mut den_e = 0.0;
        let mut num_d = 0.0;
        let mut den_c = 0.0;

        for i in 0..nk {
            num_n += new_claims.get(i, j);
            den_e += exposures[i];
            num_d += known_development.get(i, j);
            den_c += cum[i * n + (j - 1)];
        }
        if den_e <= 0.0 || den_c <= 0.0 {
            return Err(ValidationError::DivideByZero(
                "exposure or prior cumulative sum",
            ));
        }

        let lam_j = num_n / den_e;
        let del_j = num_d / den_c;
        lambda[l] = lam_j;
        delta[l] = del_j;
        sum_e[l] = den_e;
        sum_c[l] = den_c;

        if nk >= 2 {
            let mut acc_s = 0.0;
            let mut acc_t = 0.0;
            let mut valid_c = 0usize;
            for i in 0..nk {
                let ei = exposures[i];
                let nij = new_claims.get(i, j);
                acc_s += ei * (nij / ei - lam_j).powi(2);

                let c_prev = cum[i * n + (j - 1)];
                if c_prev > 0.0 {
                    let dij = known_development.get(i, j);
                    acc_t += c_prev * (dij / c_prev - del_j).powi(2);
                    valid_c += 1;
                }
            }
            raw_sigma_sq[l] = Some(acc_s / (nk as f64 - 1.0));
            if valid_c >= 2 {
                raw_tau_sq[l] = Some(acc_t / (valid_c as f64 - 1.0));
            }
        }
    }

    let sigma_sq = extrapolate_mack_min_rule(&raw_sigma_sq);
    let tau_sq = extrapolate_mack_min_rule(&raw_tau_sq);

    // Precompute IBNER cumulative development factors G_k = prod_{m=k+1}^{n-1} (1 + delta_m).
    let mut g_factor = vec![1.0; n];
    for k in (0..(n - 1)).rev() {
        g_factor[k] = g_factor[k + 1] * (1.0 + delta[k]);
    }

    // Project full cumulative triangle, True IBNR, IBNER, and analytical variances.
    let mut proj = cum.clone();
    let mut latest_incurred = vec![0.0; n];
    let mut ibnr_reserve = vec![0.0; n];
    let mut ibner_reserve = vec![0.0; n];
    let mut total_reserve_by_origin = vec![0.0; n];
    let mut ultimate = vec![0.0; n];
    let mut process_variance = vec![0.0; n];
    let mut parameter_variance = vec![0.0; n];
    let mut standard_error = vec![0.0; n];

    for i in 0..n {
        let last_obs = n - 1 - i;
        let lat = cum[i * n + last_obs];
        latest_incurred[i] = lat;

        // Exact additive decomposition:
        // IBNER = C_{i, last_obs} * (G_{last_obs} - 1)
        // True IBNR = sum_{j = last_obs + 1}^{n - 1} E_i * lambda_j * G_j
        let r_ibner = lat * (g_factor[last_obs] - 1.0);
        let mut r_ibnr = 0.0;
        let mut pv = 0.0;
        let mut ev = 0.0;

        for j in (last_obs + 1)..n {
            let l = j - 1;
            let c_prev = proj[i * n + (j - 1)];
            let factor = 1.0 + delta[l];
            let f2 = factor * factor;

            let n_hat = exposures[i] * lambda[l];
            r_ibnr += n_hat * g_factor[j];

            pv = f2 * pv + exposures[i] * sigma_sq[l] + c_prev.max(0.0) * tau_sq[l];
            ev = f2 * ev
                + exposures[i] * exposures[i] * (sigma_sq[l] / sum_e[l])
                + c_prev * c_prev * (tau_sq[l] / sum_c[l]);

            proj[i * n + j] = c_prev * factor + n_hat;
        }

        ibnr_reserve[i] = r_ibnr;
        ibner_reserve[i] = r_ibner;
        total_reserve_by_origin[i] = r_ibnr + r_ibner;
        ultimate[i] = proj[i * n + (n - 1)];
        process_variance[i] = pv.max(0.0);
        parameter_variance[i] = ev.max(0.0);
        standard_error[i] = (process_variance[i] + parameter_variance[i]).sqrt();
    }

    // Cross-origin parameter covariance recursion (Liu & Verrall 2009 §3).
    let mut total_mse: f64 = (0..n)
        .map(|i| process_variance[i] + parameter_variance[i])
        .sum();

    for i in 1..n {
        let start_j = n - i;
        for m in (i + 1)..n {
            let mut cov_im = 0.0;
            for j in start_j..n {
                let l = j - 1;
                let factor = 1.0 + delta[l];
                let ci_prev = proj[i * n + (j - 1)];
                let cm_prev = proj[m * n + (j - 1)];
                let step_cov = exposures[i] * exposures[m] * (sigma_sq[l] / sum_e[l])
                    + ci_prev * cm_prev * (tau_sq[l] / sum_c[l]);
                cov_im = factor * factor * cov_im + step_cov;
            }
            total_mse += 2.0 * cov_im;
        }
    }
    total_mse = total_mse.max(0.0);
    let total_standard_error = total_mse.sqrt();
    let total_ibnr: f64 = ibnr_reserve.iter().sum();
    let total_ibner: f64 = ibner_reserve.iter().sum();
    let total_reserve = total_ibnr + total_ibner;
    let total_cv = if total_reserve.abs() > 0.0 {
        total_standard_error / total_reserve.abs()
    } else {
        0.0
    };

    Ok(SchnieperResult {
        lambda,
        delta,
        sigma_sq,
        tau_sq,
        latest_incurred,
        ibnr_reserve,
        ibner_reserve,
        total_reserve_by_origin,
        ultimate,
        process_variance,
        parameter_variance,
        standard_error,
        total_ibnr,
        total_ibner,
        total_reserve,
        total_mse,
        total_standard_error,
        total_cv,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triangle;
    use crate::mack;

    #[test]
    fn exact_ibnr_plus_ibner_equals_total_reserve() {
        let new_claims = Triangle::try_from_upper_rows(&[
            &[100.0, 30.0, 10.0, 3.0],
            &[110.0, 35.0, 12.0],
            &[120.0, 38.0],
            &[130.0],
        ])
        .unwrap();
        // Signed development on prior known claims (can include releases).
        let dev_data = vec![
            0.0,
            15.0,
            -5.0,
            -2.0,
            0.0,
            18.0,
            -6.0,
            f64::NAN,
            0.0,
            20.0,
            f64::NAN,
            f64::NAN,
            0.0,
            f64::NAN,
            f64::NAN,
            f64::NAN,
        ];
        let known_dev = Triangle::new(4, 4, dev_data);
        let exposures = [1000.0, 1100.0, 1200.0, 1300.0];

        let res = analyze(&new_claims.view(), &known_dev.view(), &exposures).unwrap();

        for i in 0..4 {
            let expected_res = res.ultimate[i] - res.latest_incurred[i];
            assert!(((res.ibnr_reserve[i] + res.ibner_reserve[i]) - expected_res).abs() < 1e-10);
        }
        assert!(((res.total_ibnr + res.total_ibner) - res.total_reserve).abs() < 1e-10);
        assert!(res.total_ibnr > 0.0);
        assert!(res.total_standard_error > 0.0);
    }

    #[test]
    fn zero_new_claims_after_age_0_reduces_identically_to_mack() {
        let cum = Triangle::try_from_upper_rows(&[
            &[100.0, 165.0, 200.0, 212.0, 218.0],
            &[120.0, 190.0, 238.0, 250.0],
            &[135.0, 222.0, 268.0],
            &[150.0, 240.0],
            &[160.0],
        ])
        .unwrap();
        let n = 5;
        let mut n_data = vec![f64::NAN; n * n];
        let mut d_data = vec![f64::NAN; n * n];
        for i in 0..n {
            n_data[i * n] = cum.get(i, 0);
            d_data[i * n] = 0.0;
            for j in 1..(n - i) {
                n_data[i * n + j] = 0.0;
                d_data[i * n + j] = cum.get(i, j) - cum.get(i, j - 1);
            }
        }
        let n_tri = Triangle::new(n, n, n_data);
        let d_tri = Triangle::new(n, n, d_data);
        let exposures = [100.0; 5];

        let m = mack::analyze(&cum.view(), 1.0).unwrap();
        let sc = analyze(&n_tri.view(), &d_tri.view(), &exposures).unwrap();

        assert!(sc.total_ibnr.abs() < 1e-12);
        assert!((sc.total_reserve - m.total_reserve).abs() < 1e-10);
        assert!((sc.total_standard_error - m.total_standard_error).abs() < 1e-9);
    }
}
