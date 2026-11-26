//! Brosius least-squares development.
//!
//! For each development link `j → j+1`, Brosius fits the best linear
//! predictor of `C_{i, j+1}` given `C_{i, j}`:
//!
//! `C_{i, j+1} ≈ a_j + b_j × C_{i, j}`.
//!
//! OLS gives `b_j = cov(x, y) / var(x)` and `a_j = mean(y) − b_j × mean(x)`,
//! where `(x, y) = (C_{·,j}, C_{·,j+1})` over origins `i = 0..n−j−2`.
//!
//! Brosius is a credibility-weighted estimator: `b_j = 0` recovers a pure
//! budget method (expected next value is constant), `b_j = f_j` with
//! `a_j = 0` recovers chain-ladder. Real triangles typically fall between.
//!
//! Returns `(a, b, next_projection)` per link, and per-origin ultimates by
//! chaining projections forward from each origin's latest observation.

use crate::triangle::latest_diagonal;
use crate::types::TriangleView;
use crate::validate::{
    MonotonicityPolicy, ValidationError, validate_finite_slice, validate_monotone_cumulative,
    validate_observed_count_inputs, validate_tail, validate_triangle,
};

/// Brosius projection result.
#[derive(Debug, Clone)]
pub struct Brosius {
    /// Intercept per link: `a_j`, length `n − 1`.
    pub intercept: Vec<f64>,
    /// Slope per link: `b_j`, length `n − 1`.
    pub slope: Vec<f64>,
    /// Estimated ultimate per origin.
    pub ultimate: Vec<f64>,
    /// Unpaid reserve per origin.
    pub reserve: Vec<f64>,
    /// Latest observed cumulative value per origin.
    pub latest: Vec<f64>,
    /// OLS fit diagnostics per development link.
    pub diagnostics: Vec<BrosiusLinkDiagnostic>,
}

impl Brosius {
    /// Credibility weight `Z_j = b_j * mean_x / mean_y` per link (Brosius 1993).
    ///
    /// Expresses the OLS predictor `a_j + b_j * x` in credibility form
    /// `Z_j * (f_j * x) + (1 - Z_j) * mean_y` where `f_j = mean_y / mean_x`
    /// is the volume-weighted link ratio.
    pub fn credibility_weights(&self) -> Vec<f64> {
        self.diagnostics
            .iter()
            .map(|d| {
                if d.mean_y.is_finite() && d.mean_y > 0.0 && d.mean_x.is_finite() {
                    d.slope * d.mean_x / d.mean_y
                } else {
                    f64::NAN
                }
            })
            .collect()
    }
}

/// Brosius OLS fit diagnostics for one development link.
#[derive(Debug, Clone)]
pub struct BrosiusLinkDiagnostic {
    /// Development link start index.
    pub dev: usize,
    /// Number of observed pairs used.
    pub n: usize,
    /// Mean prior cumulative value.
    pub mean_x: f64,
    /// Mean next cumulative value.
    pub mean_y: f64,
    /// Population variance of prior cumulative values.
    pub var_x: f64,
    /// Population covariance between prior and next cumulative values.
    pub cov_xy: f64,
    /// Fitted intercept.
    pub intercept: f64,
    /// Fitted slope.
    pub slope: f64,
    /// Whether the single-pair chain-ladder-ratio fallback was used.
    pub fallback_used: bool,
}

/// Fit Brosius OLS parameters per link and project ultimates without validation.
///
/// `tail` is the multiplicative tail factor applied at development age
/// `n − 1` (pass `1.0` for fully-developed).
///
/// Prefer [`project`] at external ingress boundaries.
pub fn project_unchecked(t: &TriangleView<'_>, tail: f64) -> Brosius {
    let n = t.n_origins.min(t.n_devs);
    let latest = latest_diagonal(t);
    let mut intercept = vec![0.0; n.saturating_sub(1)];
    let mut slope = vec![0.0; n.saturating_sub(1)];
    let mut diagnostics = Vec::with_capacity(n.saturating_sub(1));

    for j in 0..n.saturating_sub(1) {
        let k = n - j - 1; // number of origins with both (i, j) and (i, j+1)
        let mut sx = 0.0;
        let mut sy = 0.0;
        for i in 0..k {
            sx += t.get(i, j);
            sy += t.get(i, j + 1);
        }
        let kf = k as f64;
        let mut mean_x = f64::NAN;
        let mut mean_y = f64::NAN;
        let mut var_x = f64::NAN;
        let mut cov_xy = f64::NAN;
        let mut fallback_used = false;
        if k >= 2 {
            mean_x = sx / kf;
            mean_y = sy / kf;
            // Two-pass centered sums prevent catastrophic cancellation on large loss amounts.
            let mut sum_dx2 = 0.0;
            let mut sum_dxdy = 0.0;
            for i in 0..k {
                let dx = t.get(i, j) - mean_x;
                let dy = t.get(i, j + 1) - mean_y;
                sum_dx2 += dx * dx;
                sum_dxdy += dx * dy;
            }
            var_x = sum_dx2 / kf;
            cov_xy = sum_dxdy / kf;
            let b = if var_x > 0.0 { cov_xy / var_x } else { 0.0 };
            let a = mean_y - b * mean_x;
            intercept[j] = a;
            slope[j] = b;
        } else if k == 1 {
            // Degenerate: single pair, use chain-ladder ratio as slope, no
            // intercept (a = 0).
            let x = t.get(0, j);
            let y = t.get(0, j + 1);
            intercept[j] = 0.0;
            slope[j] = if x > 0.0 { y / x } else { f64::NAN };
            mean_x = x;
            mean_y = y;
            var_x = 0.0;
            cov_xy = 0.0;
            fallback_used = true;
        } else {
            intercept[j] = f64::NAN;
            slope[j] = f64::NAN;
        }
        diagnostics.push(BrosiusLinkDiagnostic {
            dev: j,
            n: k,
            mean_x,
            mean_y,
            var_x,
            cov_xy,
            intercept: intercept[j],
            slope: slope[j],
            fallback_used,
        });
    }

    let mut ultimate = vec![0.0; n];
    let mut reserve = vec![0.0; n];
    for i in 0..n {
        let mut c = latest[i];
        let start = n - 1 - i;
        for j in start..n.saturating_sub(1) {
            c = intercept[j] + slope[j] * c;
        }
        c *= tail;
        ultimate[i] = c;
        reserve[i] = c - latest[i];
    }
    Brosius {
        intercept,
        slope,
        ultimate,
        reserve,
        latest,
        diagnostics,
    }
}

/// Fit Brosius OLS parameters per link and project ultimates from observed row widths.
///
/// `observed_counts[i]` is the number of observed development periods in row
/// `i`. Each link fit uses rows where both `dev` and `dev + 1` are observed.
pub fn project_observed_counts(
    t: &TriangleView<'_>,
    observed_counts: &[usize],
    tail: f64,
) -> Result<Brosius, ValidationError> {
    validate_observed_count_inputs(t, observed_counts, tail)?;
    let n_origin = t.n_origins;
    let n_dev = t.n_devs;
    let mut latest = vec![0.0; n_origin];
    for i in 0..n_origin {
        latest[i] = t.get(i, observed_counts[i] - 1);
    }

    let mut intercept = vec![0.0; n_dev.saturating_sub(1)];
    let mut slope = vec![0.0; n_dev.saturating_sub(1)];
    let mut diagnostics = Vec::with_capacity(n_dev.saturating_sub(1));
    for dev in 0..n_dev.saturating_sub(1) {
        let pairs = observed_counts
            .iter()
            .enumerate()
            .filter_map(|(origin, &count)| (count > dev + 1).then_some(origin))
            .collect::<Vec<_>>();
        let k = pairs.len();
        let mut sx = 0.0;
        let mut sy = 0.0;
        for &origin in &pairs {
            sx += t.get(origin, dev);
            sy += t.get(origin, dev + 1);
        }

        let kf = k as f64;
        let mut mean_x = f64::NAN;
        let mut mean_y = f64::NAN;
        let mut var_x = f64::NAN;
        let mut cov_xy = f64::NAN;
        let mut fallback_used = false;
        if k >= 2 {
            mean_x = sx / kf;
            mean_y = sy / kf;
            let mut sum_dx2 = 0.0;
            let mut sum_dxdy = 0.0;
            for &origin in &pairs {
                let dx = t.get(origin, dev) - mean_x;
                let dy = t.get(origin, dev + 1) - mean_y;
                sum_dx2 += dx * dx;
                sum_dxdy += dx * dy;
            }
            var_x = sum_dx2 / kf;
            cov_xy = sum_dxdy / kf;
            let b = if var_x > 0.0 { cov_xy / var_x } else { 0.0 };
            let a = mean_y - b * mean_x;
            intercept[dev] = a;
            slope[dev] = b;
        } else if k == 1 {
            let origin = pairs[0];
            let x = t.get(origin, dev);
            let y = t.get(origin, dev + 1);
            intercept[dev] = 0.0;
            slope[dev] = if x > 0.0 { y / x } else { f64::NAN };
            mean_x = x;
            mean_y = y;
            var_x = 0.0;
            cov_xy = 0.0;
            fallback_used = true;
        } else {
            intercept[dev] = f64::NAN;
            slope[dev] = f64::NAN;
        }
        diagnostics.push(BrosiusLinkDiagnostic {
            dev,
            n: k,
            mean_x,
            mean_y,
            var_x,
            cov_xy,
            intercept: intercept[dev],
            slope: slope[dev],
            fallback_used,
        });
    }

    let mut ultimate = vec![0.0; n_origin];
    let mut reserve = vec![0.0; n_origin];
    for i in 0..n_origin {
        let mut c = latest[i];
        for dev in (observed_counts[i] - 1)..n_dev.saturating_sub(1) {
            c = intercept[dev] + slope[dev] * c;
        }
        c *= tail;
        ultimate[i] = c;
        reserve[i] = c - latest[i];
    }

    let projection = Brosius {
        intercept,
        slope,
        ultimate,
        reserve,
        latest,
        diagnostics,
    };
    validate_finite_slice(&projection.intercept)?;
    validate_finite_slice(&projection.slope)?;
    validate_finite_slice(&projection.ultimate)?;
    validate_finite_slice(&projection.reserve)?;
    Ok(projection)
}

/// Fit Brosius OLS parameters per link and project ultimates after validation.
///
/// The checked path enforces square-triangle shape, finite non-negative
/// observed cells, monotone cumulative rows, `tail >= 1`, and finite projected
/// parameters and outputs.
pub fn project(t: &TriangleView<'_>, tail: f64) -> Result<Brosius, ValidationError> {
    project_with_policy(t, tail, MonotonicityPolicy::Strict)
}

/// Fit Brosius OLS parameters per link and project ultimates under an explicit
/// [`MonotonicityPolicy`].
pub fn project_with_policy(
    t: &TriangleView<'_>,
    tail: f64,
    policy: MonotonicityPolicy,
) -> Result<Brosius, ValidationError> {
    validate_triangle(t)?;
    if policy == MonotonicityPolicy::Strict {
        validate_monotone_cumulative(t)?;
    }
    validate_tail(tail)?;

    let projection = project_unchecked(t, tail);
    validate_finite_slice(&projection.intercept)?;
    validate_finite_slice(&projection.slope)?;
    validate_finite_slice(&projection.latest)?;
    validate_finite_slice(&projection.ultimate)?;
    validate_finite_slice(&projection.reserve)?;
    for diagnostic in &projection.diagnostics {
        if !diagnostic.mean_x.is_finite()
            || !diagnostic.mean_y.is_finite()
            || !diagnostic.var_x.is_finite()
            || !diagnostic.cov_xy.is_finite()
            || !diagnostic.intercept.is_finite()
            || !diagnostic.slope.is_finite()
        {
            return Err(ValidationError::NotFinite);
        }
    }
    Ok(projection)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Triangle;

    #[test]
    fn project_rejects_non_cumulative_triangle() {
        #[rustfmt::skip]
        let triangle = Triangle::new(3, 3, vec![
            100.0, 150.0, 180.0,
            120.0, 110.0, f64::NAN,
            140.0, f64::NAN, f64::NAN,
        ]);

        assert!(matches!(
            project(&triangle.view(), 1.0),
            Err(ValidationError::NonMonotoneCumulative {
                origin: 1,
                dev: 1,
                previous: 120.0,
                value: 110.0,
            })
        ));
    }

    #[test]
    fn project_rejects_invalid_tail() {
        #[rustfmt::skip]
        let triangle = Triangle::new(3, 3, vec![
            100.0, 150.0, 180.0,
            120.0, 180.0, f64::NAN,
            140.0, f64::NAN, f64::NAN,
        ]);

        assert!(matches!(
            project(&triangle.view(), 0.99),
            Err(ValidationError::InvalidTail { tail: 0.99 })
        ));
    }

    #[test]
    fn project_returns_link_diagnostics() {
        #[rustfmt::skip]
        let triangle = Triangle::new(3, 3, vec![
            100.0, 150.0, 180.0,
            120.0, 180.0, f64::NAN,
            140.0, f64::NAN, f64::NAN,
        ]);

        let projection = project(&triangle.view(), 1.0).expect("brosius");

        assert_eq!(projection.diagnostics.len(), 2);
        assert_eq!(projection.diagnostics[0].n, 2);
        assert!(!projection.diagnostics[0].fallback_used);
        assert_eq!(projection.diagnostics[1].n, 1);
        assert!(projection.diagnostics[1].fallback_used);
    }

    #[test]
    fn project_observed_counts_projects_rectangular_triangle() {
        #[rustfmt::skip]
        let data = vec![
            100.0, 150.0, 180.0, 198.0,
            120.0, 180.0, 216.0, f64::NAN,
            140.0, 210.0, f64::NAN, f64::NAN,
        ];
        let view = TriangleView {
            n_origins: 3,
            n_devs: 4,
            data: &data,
        };

        let projection =
            project_observed_counts(&view, &[4, 3, 2], 1.0).expect("brosius rectangular");

        assert_eq!(projection.diagnostics.len(), 3);
        assert_eq!(projection.reserve.len(), 3);
        assert!(projection.ultimate[0].is_finite());
        assert!(projection.ultimate[2] > projection.latest[2]);
        assert!(projection.diagnostics[2].fallback_used);
    }
}
