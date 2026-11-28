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
//! ## Fallbacks (Brosius 1993, §4)
//!
//! Under the default [`BrosiusFitRule::Brosius1993`] the raw OLS fit is
//! replaced when it leaves the credibility range `0 <= Z_j <= 1`:
//!
//! * `b_j < 0` (negative correlation; `Z_j < 0`) → **budget method**:
//!   `b_j = 0`, `a_j = mean(y)`.
//! * `a_j < 0` (`Z_j > 1`) → **link-ratio method**: `a_j = 0`,
//!   `b_j = Σ y / Σ x` (volume-weighted chain-ladder link ratio).
//! * Exactly one pair → link-ratio method (`a_j = 0`, `b_j = y / x`).
//! * Zero variance in `x` → budget method (slope unidentifiable).
//!
//! The applied fallback is recorded per link in
//! [`BrosiusLinkDiagnostic::fallback`]. [`BrosiusFitRule::RawOls`] keeps the
//! unconstrained OLS line (still applying the single-pair and zero-variance
//! rules, which are needed for identifiability); it reproduces
//! `murphy::MurphyOptions::brosius_equivalent` exactly.
//!
//! Returns `(a, b, next_projection)` per link, and per-origin ultimates by
//! chaining projections forward from each origin's latest observation.

use crate::types::TriangleView;
use crate::validate::{
    MonotonicityPolicy, ValidationError, validate_finite_slice, validate_monotone_cumulative,
    validate_observed_count_inputs, validate_tail, validate_triangle,
};

/// Fallback rule applied to a single development link.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum BrosiusFallback {
    /// Raw OLS line used (`a_j >= 0`, `b_j >= 0`, or [`BrosiusFitRule::RawOls`]).
    #[default]
    None,
    /// Only one observed pair: link ratio `b = y / x`, `a = 0`.
    SinglePairLinkRatio,
    /// Zero variance in prior values: budget method `b = 0`, `a = mean(y)`.
    ZeroVarianceBudget,
    /// OLS intercept `a < 0`: link-ratio method `b = Σy / Σx`, `a = 0`.
    NegativeInterceptLinkRatio,
    /// OLS slope `b < 0`: budget method `b = 0`, `a = mean(y)`.
    NegativeSlopeBudget,
    /// No observed pairs at this link (unchecked path only; parameters are `NaN`).
    NoData,
}

/// Fitting rule for Brosius link parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum BrosiusFitRule {
    /// Brosius (1993) fallbacks: link ratio when `a < 0`, budget when `b < 0`.
    #[default]
    Brosius1993,
    /// Unconstrained OLS (negative intercepts/slopes kept).
    RawOls,
}

/// Brosius projection result.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    /// Raw OLS intercept before any fallback (`NaN` when the OLS line is not
    /// identifiable: fewer than two pairs or zero variance in `x`).
    #[serde(default = "nan")]
    pub ols_intercept: f64,
    /// Raw OLS slope before any fallback (`NaN` when not identifiable).
    #[serde(default = "nan")]
    pub ols_slope: f64,
    /// Selected intercept used for projection.
    pub intercept: f64,
    /// Selected slope used for projection.
    pub slope: f64,
    /// Whether any fallback replaced the raw OLS line (`fallback != None`).
    pub fallback_used: bool,
    /// Which fallback (if any) was applied at this link.
    #[serde(default)]
    pub fallback: BrosiusFallback,
}

const fn nan() -> f64 {
    f64::NAN
}

/// Fit one link from observed `(x, y) = (C_{i,j}, C_{i,j+1})` pairs.
fn fit_link(dev: usize, pairs: &[(f64, f64)], rule: BrosiusFitRule) -> BrosiusLinkDiagnostic {
    let k = pairs.len();
    let mut diag = BrosiusLinkDiagnostic {
        dev,
        n: k,
        mean_x: f64::NAN,
        mean_y: f64::NAN,
        var_x: f64::NAN,
        cov_xy: f64::NAN,
        ols_intercept: f64::NAN,
        ols_slope: f64::NAN,
        intercept: f64::NAN,
        slope: f64::NAN,
        fallback_used: true,
        fallback: BrosiusFallback::NoData,
    };
    if k == 0 {
        return diag;
    }
    let (sx, sy) = pairs
        .iter()
        .fold((0.0, 0.0), |(ax, ay), &(x, y)| (ax + x, ay + y));
    if k == 1 {
        let (x, y) = pairs[0];
        diag.mean_x = x;
        diag.mean_y = y;
        diag.var_x = 0.0;
        diag.cov_xy = 0.0;
        diag.intercept = 0.0;
        diag.slope = if x > 0.0 { y / x } else { f64::NAN };
        diag.fallback = BrosiusFallback::SinglePairLinkRatio;
        return diag;
    }
    let kf = k as f64;
    let mean_x = sx / kf;
    let mean_y = sy / kf;
    // Two-pass centered sums prevent catastrophic cancellation on large loss amounts.
    let (sum_dx2, sum_dxdy) = pairs.iter().fold((0.0, 0.0), |(a2, axy), &(x, y)| {
        let dx = x - mean_x;
        (a2 + dx * dx, axy + dx * (y - mean_y))
    });
    let var_x = sum_dx2 / kf;
    let cov_xy = sum_dxdy / kf;
    diag.mean_x = mean_x;
    diag.mean_y = mean_y;
    diag.var_x = var_x;
    diag.cov_xy = cov_xy;

    let var_tol = 1e-12 * mean_x * mean_x;
    if !(var_x > var_tol) {
        diag.intercept = mean_y;
        diag.slope = 0.0;
        diag.fallback = BrosiusFallback::ZeroVarianceBudget;
        return diag;
    }
    let b = cov_xy / var_x;
    let a = mean_y - b * mean_x;
    diag.ols_intercept = a;
    diag.ols_slope = b;
    // With non-negative means, `b < 0` implies `a > mean_y >= 0`, so the two
    // fallback conditions are mutually exclusive; test the slope first.
    let (a_sel, b_sel, fallback) = match rule {
        BrosiusFitRule::Brosius1993 if b < 0.0 => {
            (mean_y, 0.0, BrosiusFallback::NegativeSlopeBudget)
        }
        BrosiusFitRule::Brosius1993 if a < 0.0 => {
            let f = if sx > 0.0 { sy / sx } else { f64::NAN };
            (0.0, f, BrosiusFallback::NegativeInterceptLinkRatio)
        }
        BrosiusFitRule::Brosius1993 | BrosiusFitRule::RawOls => (a, b, BrosiusFallback::None),
    };
    diag.intercept = a_sel;
    diag.slope = b_sel;
    diag.fallback = fallback;
    diag.fallback_used = fallback != BrosiusFallback::None;
    diag
}

/// Shared fit-and-project core. `observed_counts[i]` is the number of observed
/// development periods in row `i` (callers guarantee `1 <= count <= n_devs`).
fn fit_and_project(
    t: &TriangleView<'_>,
    n_devs: usize,
    observed_counts: &[usize],
    tail: f64,
    rule: BrosiusFitRule,
) -> Brosius {
    let n_links = n_devs.saturating_sub(1);
    let mut intercept = Vec::with_capacity(n_links);
    let mut slope = Vec::with_capacity(n_links);
    let mut diagnostics = Vec::with_capacity(n_links);
    let mut pairs = Vec::with_capacity(observed_counts.len());
    for dev in 0..n_links {
        pairs.clear();
        pairs.extend(
            observed_counts
                .iter()
                .enumerate()
                .filter(|&(_, &count)| count > dev + 1)
                .map(|(origin, _)| (t.get(origin, dev), t.get(origin, dev + 1))),
        );
        let diag = fit_link(dev, &pairs, rule);
        intercept.push(diag.intercept);
        slope.push(diag.slope);
        diagnostics.push(diag);
    }

    let n_origin = observed_counts.len();
    let mut latest = Vec::with_capacity(n_origin);
    let mut ultimate = Vec::with_capacity(n_origin);
    let mut reserve = Vec::with_capacity(n_origin);
    for (i, &count) in observed_counts.iter().enumerate() {
        let last = count.saturating_sub(1);
        let obs = t.get(i, last);
        let mut c = obs;
        for dev in last..n_links {
            c = intercept[dev] + slope[dev] * c;
        }
        c *= tail;
        latest.push(obs);
        ultimate.push(c);
        reserve.push(c - obs);
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

fn validate_projection(projection: &Brosius) -> Result<(), ValidationError> {
    validate_finite_slice(&projection.intercept)?;
    validate_finite_slice(&projection.slope)?;
    validate_finite_slice(&projection.latest)?;
    validate_finite_slice(&projection.ultimate)?;
    validate_finite_slice(&projection.reserve)?;
    for d in &projection.diagnostics {
        if !d.mean_x.is_finite()
            || !d.mean_y.is_finite()
            || !d.var_x.is_finite()
            || !d.cov_xy.is_finite()
            || !d.intercept.is_finite()
            || !d.slope.is_finite()
        {
            return Err(ValidationError::NotFinite);
        }
    }
    Ok(())
}

/// Fit Brosius parameters per link (Brosius 1993 fallbacks) and project
/// ultimates without validation.
///
/// `tail` is the multiplicative tail factor applied at development age
/// `n − 1` (pass `1.0` for fully-developed). Uses the leading
/// `min(n_origins, n_devs)` square block; unestimable links yield `NaN`.
///
/// Prefer [`project`] at external ingress boundaries.
pub fn project_unchecked(t: &TriangleView<'_>, tail: f64) -> Brosius {
    let n = t.n_origins.min(t.n_devs);
    let counts: Vec<usize> = (0..n).map(|i| n - i).collect();
    fit_and_project(t, n, &counts, tail, BrosiusFitRule::Brosius1993)
}

/// Fit Brosius parameters per link (Brosius 1993 fallbacks) and project
/// ultimates from observed row widths.
///
/// `observed_counts[i]` is the number of observed development periods in row
/// `i`. Each link fit uses rows where both `dev` and `dev + 1` are observed.
pub fn project_observed_counts(
    t: &TriangleView<'_>,
    observed_counts: &[usize],
    tail: f64,
) -> Result<Brosius, ValidationError> {
    validate_observed_count_inputs(t, observed_counts, tail)?;
    let projection = fit_and_project(
        t,
        t.n_devs,
        observed_counts,
        tail,
        BrosiusFitRule::Brosius1993,
    );
    validate_finite_slice(&projection.intercept)?;
    validate_finite_slice(&projection.slope)?;
    validate_finite_slice(&projection.ultimate)?;
    validate_finite_slice(&projection.reserve)?;
    Ok(projection)
}

/// Fit Brosius parameters per link and project ultimates after validation.
///
/// The checked path enforces square-triangle shape, finite non-negative
/// observed cells, monotone cumulative rows, `tail >= 1`, and finite projected
/// parameters and outputs. Uses [`BrosiusFitRule::Brosius1993`] fallbacks.
pub fn project(t: &TriangleView<'_>, tail: f64) -> Result<Brosius, ValidationError> {
    project_with_policy(t, tail, MonotonicityPolicy::Strict)
}

/// Fit Brosius parameters per link and project ultimates under an explicit
/// [`MonotonicityPolicy`] (Brosius 1993 fallbacks).
pub fn project_with_policy(
    t: &TriangleView<'_>,
    tail: f64,
    policy: MonotonicityPolicy,
) -> Result<Brosius, ValidationError> {
    project_with_fit_rule(t, tail, policy, BrosiusFitRule::Brosius1993)
}

/// Fit Brosius parameters per link under an explicit [`BrosiusFitRule`] and
/// [`MonotonicityPolicy`], then project ultimates after validation.
pub fn project_with_fit_rule(
    t: &TriangleView<'_>,
    tail: f64,
    policy: MonotonicityPolicy,
    rule: BrosiusFitRule,
) -> Result<Brosius, ValidationError> {
    validate_triangle(t)?;
    if policy == MonotonicityPolicy::Strict {
        validate_monotone_cumulative(t)?;
    }
    validate_tail(tail)?;
    let n = t.n_origins;
    let counts: Vec<usize> = (0..n).map(|i| n - i).collect();
    let projection = fit_and_project(t, n, &counts, tail, rule);
    validate_projection(&projection)?;
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

    #[test]
    fn fit_link_records_each_fallback() {
        let rule = BrosiusFitRule::Brosius1993;
        assert_eq!(fit_link(0, &[], rule).fallback, BrosiusFallback::NoData);
        assert_eq!(
            fit_link(0, &[(100.0, 150.0)], rule).fallback,
            BrosiusFallback::SinglePairLinkRatio
        );
        let flat = fit_link(0, &[(100.0, 150.0), (100.0, 170.0)], rule);
        assert_eq!(flat.fallback, BrosiusFallback::ZeroVarianceBudget);
        assert!((flat.intercept - 160.0).abs() < 1e-12);
        let neg_b = fit_link(0, &[(100.0, 200.0), (200.0, 180.0)], rule);
        assert_eq!(neg_b.fallback, BrosiusFallback::NegativeSlopeBudget);
        assert!((neg_b.intercept - 190.0).abs() < 1e-12 && neg_b.slope == 0.0);
        let neg_a = fit_link(0, &[(100.0, 100.0), (200.0, 300.0)], rule);
        assert_eq!(neg_a.fallback, BrosiusFallback::NegativeInterceptLinkRatio);
        assert!((neg_a.slope - 400.0 / 300.0).abs() < 1e-12 && neg_a.intercept == 0.0);
        let raw = fit_link(0, &[(100.0, 100.0), (200.0, 300.0)], BrosiusFitRule::RawOls);
        assert_eq!(raw.fallback, BrosiusFallback::None);
        assert!((raw.intercept + 100.0).abs() < 1e-12 && (raw.slope - 2.0).abs() < 1e-12);
    }
}
