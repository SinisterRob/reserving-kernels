//! Taylor (1977) Arithmetic and Geometric Separation Methods.
//!
//! Implements:
//!
//! * G. C. Taylor (1977), *Separation of Inflation and Other Effects from the
//!   Distribution of Non-Life Insurance Claim Delays*, ASTIN Bulletin, Vol. 9,
//!   No. 1–2, pp. 219–230.
//!
//! Factors incremental claim costs per unit exposure $S_{i,j} = X_{i,j} / n_i$
//! into development delay proportions $r_j$ ($\sum_{j=0}^{n-1} r_j = 1$) and
//! exogenous calendar-year inflation indices $\lambda_{i+j}$:
//!
//! $$S_{i,j} = r_j \cdot \lambda_{i+j}$$
//!
//! Supports both:
//! * **Arithmetic Separation** ([`SeparationMethod::Arithmetic`]): exact
//!   backward diagonal/column recursion matching Taylor (1977) §2.
//! * **Geometric Separation** ([`SeparationMethod::Geometric`]): exact
//!   log-linear OLS via Householder QR ($\ln S_{i,j} = \alpha_j + \beta_{i+j}$)
//!   normalized to $\sum_j r_j = 1$, matching Taylor (1977) §3.

use crate::linalg::weighted_least_squares_qr_beta_only;
use crate::preprocessing::triangle::{cumulative_to_incremental, latest_diagonal};
use crate::types::TriangleView;
use crate::validate::{ValidationError, validate_positive_finite_slice, validate_triangle};

/// Variant of Taylor's (1977) Separation Method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SeparationMethod {
    /// Non-parametric arithmetic recursion over calendar diagonals and development columns.
    Arithmetic,
    /// Log-linear geometric least-squares separation ($\ln S_{i,j} = \ln r_j + \ln \lambda_{i+j}$).
    Geometric,
}

/// Strategy for projecting future calendar-year inflation indices $\lambda_n, \dots, \lambda_{2n-2}$.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum InflationProjection {
    /// Extrapolate from $\hat{\lambda}_{n-1}$ at a constant annual inflation rate `rate`
    /// ($\lambda_{n-1+h} = \hat{\lambda}_{n-1} (1 + \text{rate})^h$).
    ConstantRate(f64),
    /// Extrapolate using the geometric mean historical inflation rate
    /// $(\hat{\lambda}_{n-1} / \hat{\lambda}_0)^{1/(n-1)} - 1$.
    HistoricalGeometricMean,
    /// Use explicit caller-supplied future calendar-year indices
    /// $\lambda_n, \dots, \lambda_{2n-2}$ (length `n - 1`).
    ExplicitFutureIndices(Vec<f64>),
}

/// Result of Taylor's (1977) Separation Method.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct SeparationResult {
    /// Separation variant used.
    pub method: SeparationMethod,
    /// Estimated development delay proportions $r_j$ ($\sum r_j = 1$), length $n$.
    pub delay_proportions: Vec<f64>,
    /// Estimated historical calendar-year inflation indices $\lambda_0, \dots, \lambda_{n-1}$, length $n$.
    pub calendar_indices: Vec<f64>,
    /// Projected future calendar-year inflation indices $\lambda_n, \dots, \lambda_{2n-2}$, length $n - 1$.
    pub future_calendar_indices: Vec<f64>,
    /// Implied historical period-to-period calendar inflation rates $\lambda_k / \lambda_{k-1} - 1$, length $n - 1$.
    pub implied_inflation_rates: Vec<f64>,
    /// Latest observed cumulative loss per origin, length $n$.
    pub latest: Vec<f64>,
    /// Projected reserve per origin, length $n$.
    pub reserve: Vec<f64>,
    /// Projected ultimate per origin (`latest + reserve`), length $n$.
    pub ultimate: Vec<f64>,
    /// Total reserve across all origins.
    pub total_reserve: f64,
}

impl std::fmt::Display for SeparationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Separation({:?}, total_reserve={:.2})",
            self.method, self.total_reserve
        )
    }
}

/// Fit Taylor's (1977) Separation Method to a square cumulative loss triangle
/// and per-origin exposures (e.g. claim counts or earned exposure $n_i > 0$).
pub fn analyze(
    cumulative: &TriangleView<'_>,
    exposures: &[f64],
    method: SeparationMethod,
    projection: &InflationProjection,
) -> Result<SeparationResult, ValidationError> {
    validate_triangle(cumulative)?;
    let n = cumulative.n_origins;
    if n < 2 {
        return Err(ValidationError::InvalidParameter(
            "triangle must be at least 2x2",
        ));
    }
    if exposures.len() != n {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_positive_finite_slice(exposures, "exposures")?;

    let inc = cumulative_to_incremental(cumulative);
    let mut s = vec![f64::NAN; n * n];
    for i in 0..n {
        for j in 0..(n - i) {
            let x = inc[i * n + j];
            if !x.is_finite() {
                return Err(ValidationError::NotFinite);
            }
            s[i * n + j] = x / exposures[i];
        }
    }

    let (delay_proportions, calendar_indices) = match method {
        SeparationMethod::Arithmetic => fit_arithmetic(&s, n)?,
        SeparationMethod::Geometric => fit_geometric(&s, n)?,
    };

    let mut implied_inflation_rates = Vec::with_capacity(n - 1);
    for k in 1..n {
        let prev = calendar_indices[k - 1];
        let curr = calendar_indices[k];
        if prev.abs() > 0.0 {
            implied_inflation_rates.push(curr / prev - 1.0);
        } else {
            implied_inflation_rates.push(f64::NAN);
        }
    }

    let future_calendar_indices = match projection {
        InflationProjection::ConstantRate(rate) => {
            if !rate.is_finite() || *rate <= -1.0 {
                return Err(ValidationError::InvalidParameter(
                    "inflation rate must be finite and > -1.0",
                ));
            }
            let base = calendar_indices[n - 1];
            (1..n).map(|h| base * (1.0 + rate).powi(h as i32)).collect()
        }
        InflationProjection::HistoricalGeometricMean => {
            let first = calendar_indices[0];
            let last = calendar_indices[n - 1];
            if first <= 0.0 || last <= 0.0 {
                return Err(ValidationError::InvalidParameter(
                    "geometric mean inflation requires positive first and last calendar indices",
                ));
            }
            let g = (last / first).powf(1.0 / (n as f64 - 1.0));
            (1..n).map(|h| last * g.powi(h as i32)).collect()
        }
        InflationProjection::ExplicitFutureIndices(indices) => {
            if indices.len() != n - 1 {
                return Err(ValidationError::ShapeMismatch);
            }
            for &v in indices {
                if !v.is_finite() || v < 0.0 {
                    return Err(ValidationError::InvalidParameter(
                        "future calendar indices must be finite and non-negative",
                    ));
                }
            }
            indices.clone()
        }
    };

    let latest = latest_diagonal(cumulative);
    let mut reserve = vec![0.0; n];
    let mut ultimate = vec![0.0; n];
    for i in 0..n {
        let mut r_i = 0.0;
        for j in (n - i)..n {
            let cal_idx = i + j; // in n..=2n-2
            let lam = future_calendar_indices[cal_idx - n];
            r_i += exposures[i] * delay_proportions[j] * lam;
        }
        reserve[i] = r_i;
        ultimate[i] = latest[i] + r_i;
    }
    let total_reserve: f64 = reserve.iter().sum();

    Ok(SeparationResult {
        method,
        delay_proportions,
        calendar_indices,
        future_calendar_indices,
        implied_inflation_rates,
        latest,
        reserve,
        ultimate,
        total_reserve,
    })
}

fn fit_arithmetic(s: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), ValidationError> {
    let mut diag_sums = vec![0.0; n];
    let mut col_sums = vec![0.0; n];

    for i in 0..n {
        for j in 0..(n - i) {
            let v = s[i * n + j];
            diag_sums[i + j] += v;
            col_sums[j] += v;
        }
    }

    let mut r = vec![0.0; n];
    let mut lam = vec![0.0; n];

    let mut sum_r_tail = 0.0_f64;
    let mut sum_lam_tail = 0.0_f64;

    for k in (0..n).rev() {
        let rem_r = 1.0_f64 - sum_r_tail;
        if rem_r.abs() <= 1e-14 {
            return Err(ValidationError::DivideByZero(
                "arithmetic separation remaining delay proportion",
            ));
        }
        lam[k] = diag_sums[k] / rem_r;
        sum_lam_tail += lam[k];
        if sum_lam_tail.abs() <= 1e-14 {
            return Err(ValidationError::DivideByZero(
                "arithmetic separation cumulative calendar index",
            ));
        }
        r[k] = col_sums[k] / sum_lam_tail;
        sum_r_tail += r[k];
    }

    Ok((r, lam))
}

fn fit_geometric(s: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), ValidationError> {
    // Log-linear OLS: ln(S_{i,j}) = alpha_j + beta_{i+j}, with beta_{n-1} = 0.
    // Parameters: [alpha_0..alpha_{n-1}, beta_0..beta_{n-2}] (length 2n - 1).
    let p = 2 * n - 1;
    let n_obs = n * (n + 1) / 2;
    let mut x_rows = vec![0.0; n_obs * p];
    let mut y_vals = Vec::with_capacity(n_obs);
    let mut row_idx = 0usize;

    for i in 0..n {
        for j in 0..(n - i) {
            let v = s[i * n + j];
            if v <= 0.0 {
                return Err(ValidationError::InvalidParameter(
                    "geometric separation requires strictly positive incremental losses",
                ));
            }
            let base = row_idx * p;
            x_rows[base + j] = 1.0;
            let k = i + j;
            if k < n - 1 {
                x_rows[base + n + k] = 1.0;
            }
            y_vals.push(v.ln());
            row_idx += 1;
        }
    }

    let weights = vec![1.0; n_obs];
    let beta = weighted_least_squares_qr_beta_only(&x_rows, &y_vals, &weights, n_obs, p)?;

    let raw_r: Vec<f64> = (0..n).map(|j| beta[j].exp()).collect();
    let sum_r: f64 = raw_r.iter().sum();
    if !sum_r.is_finite() || sum_r <= 0.0 {
        return Err(ValidationError::DivideByZero(
            "geometric separation delay normalization",
        ));
    }

    let r: Vec<f64> = raw_r.into_iter().map(|v| v / sum_r).collect();
    let mut lam = Vec::with_capacity(n);
    for k in 0..(n - 1) {
        let l = beta[n + k].exp() * sum_r;
        if !l.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        lam.push(l);
    }
    lam.push(sum_r);

    Ok((r, lam))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triangle;

    #[test]
    fn exact_recovery_on_synthetic_stepped_inflation_triangle() {
        let true_r = [0.40, 0.30, 0.18, 0.08, 0.04];
        let true_lam = [100.0, 108.0, 120.0, 140.0, 165.0];
        let exposures = [50.0, 55.0, 60.0, 62.0, 70.0];
        let n = 5;

        let mut inc_rows: Vec<Vec<f64>> = Vec::new();
        for i in 0..n {
            let mut row = Vec::new();
            for j in 0..(n - i) {
                row.push(exposures[i] * true_r[j] * true_lam[i + j]);
            }
            inc_rows.push(row);
        }
        let slices: Vec<&[f64]> = inc_rows.iter().map(std::vec::Vec::as_slice).collect();
        let tri = Triangle::try_from_incremental_rows(&slices).unwrap();

        for method in [SeparationMethod::Arithmetic, SeparationMethod::Geometric] {
            let res = analyze(
                &tri.view(),
                &exposures,
                method,
                &InflationProjection::ConstantRate(0.10),
            )
            .unwrap();

            let sum_r: f64 = res.delay_proportions.iter().sum();
            assert!((sum_r - 1.0).abs() < 1e-12);
            for j in 0..n {
                assert!(
                    (res.delay_proportions[j] - true_r[j]).abs() < 1e-10,
                    "{method:?} r[{j}] mismatch"
                );
                assert!(
                    (res.calendar_indices[j] - true_lam[j]).abs() < 1e-9,
                    "{method:?} lam[{j}] mismatch"
                );
            }
            assert!(res.total_reserve > 0.0);
        }
    }
}
