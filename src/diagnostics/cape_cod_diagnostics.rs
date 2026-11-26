//! Cape Cod denominator and ELR stability diagnostics.

use crate::cape_cod::{CapeCodEngine, CapeCodProjection};
use crate::ldf::DevelopmentFactors;
use crate::types::TriangleView;
use crate::validate::ValidationError;

/// Diagnostic decomposition of a Cape Cod ELR fit.
#[derive(Debug, Clone, PartialEq)]
pub struct CapeCodDiagnostics {
    /// Fitted Cape Cod ELR using all origins.
    pub elr: f64,
    /// Latest observed cumulative value by origin.
    pub latest: Vec<f64>,
    /// Latest observed development age by origin.
    pub latest_dev: Vec<usize>,
    /// Percent reported at each origin's latest development age.
    pub reported_percentage: Vec<f64>,
    /// Premium consumed in the Cape Cod denominator by origin.
    pub used_premium: Vec<f64>,
    /// Fitted expected ultimate by origin.
    pub expected_ultimate: Vec<f64>,
    /// Per-origin stand-alone implied ELR, `latest / used_premium`.
    pub elr_by_origin: Vec<f64>,
    /// Leave-one-origin-out ELR values.
    pub leave_one_out_elr: Vec<f64>,
    /// Coefficient of variation of leave-one-out ELRs.
    pub leave_one_out_coeff_var: f64,
    /// Bounded stability score in `(0, 1]`, with higher meaning less fragile.
    pub stability_score: f64,
    /// Concentration-weighted credibility score in `(0, 1]`.
    pub credibility_score: f64,
    /// Origin with the largest leave-one-out ELR delta.
    pub high_leverage_origin: Option<usize>,
    /// Total latest observed loss.
    pub total_latest: f64,
    /// Total used premium.
    pub total_used_premium: f64,
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_std_dev(values: &[f64], mean_value: f64) -> f64 {
    if values.len() <= 1 || !mean_value.is_finite() {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|value| (value - mean_value).powi(2))
        .sum::<f64>()
        / (values.len() as f64 - 1.0);
    variance.sqrt()
}

/// Compute the Cape Cod denominator contribution by origin, `premium * reported_percentage`.
pub fn used_up_premium(
    premium: &[f64],
    reported_percentage: &[f64],
) -> Result<Vec<f64>, ValidationError> {
    if premium.len() != reported_percentage.len() {
        return Err(ValidationError::ShapeMismatch);
    }

    let mut out = Vec::with_capacity(premium.len());
    for (&premium_value, &reported) in premium.iter().zip(reported_percentage.iter()) {
        if !premium_value.is_finite() || !reported.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if premium_value <= 0.0 {
            return Err(ValidationError::DivideByZero("premium"));
        }
        if !(0.0..=1.0).contains(&reported) {
            return Err(ValidationError::InvalidMaturity {
                index: out.len(),
                value: reported,
            });
        }
        out.push(premium_value * reported);
    }
    Ok(out)
}

/// Compute stand-alone implied ELR by origin, `latest / used_premium`.
pub fn cape_cod_elr_by_origin(
    latest: &[f64],
    used_premium: &[f64],
) -> Result<Vec<f64>, ValidationError> {
    if latest.len() != used_premium.len() {
        return Err(ValidationError::ShapeMismatch);
    }

    let mut out = Vec::with_capacity(latest.len());
    for (&latest_value, &used) in latest.iter().zip(used_premium.iter()) {
        if !latest_value.is_finite() || !used.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if used <= 0.0 {
            return Err(ValidationError::DivideByZero("Cape Cod used premium"));
        }
        out.push(latest_value / used);
    }
    Ok(out)
}

/// Recompute the Cape Cod ELR leaving out each origin once.
pub fn cape_cod_elr_leave_one_out(
    latest: &[f64],
    used_premium: &[f64],
) -> Result<Vec<f64>, ValidationError> {
    if latest.len() != used_premium.len() {
        return Err(ValidationError::ShapeMismatch);
    }
    for (&latest_value, &used) in latest.iter().zip(used_premium.iter()) {
        if !latest_value.is_finite() || !used.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if used < 0.0 {
            return Err(ValidationError::Negative);
        }
    }

    let n = latest.len();
    let mut out = Vec::with_capacity(n);
    for skip in 0..n {
        let mut leave_num = 0.0;
        let mut leave_den = 0.0;
        for i in 0..n {
            if i != skip {
                leave_num += latest[i];
                leave_den += used_premium[i];
            }
        }
        if leave_den <= 0.0 {
            return Err(ValidationError::DivideByZero("Cape Cod used premium"));
        }
        out.push(leave_num / leave_den);
    }
    Ok(out)
}

/// Convert leave-one-out ELR dispersion into a bounded stability score.
pub fn cape_cod_elr_stability_score(
    elr: f64,
    leave_one_out_elr: &[f64],
) -> Result<f64, ValidationError> {
    if !elr.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if elr < 0.0 {
        return Err(ValidationError::Negative);
    }
    if leave_one_out_elr.is_empty() {
        return Ok(1.0);
    }
    if leave_one_out_elr.iter().any(|value| !value.is_finite()) {
        return Err(ValidationError::NotFinite);
    }

    let mean_value = mean(leave_one_out_elr);
    let std_dev = sample_std_dev(leave_one_out_elr, mean_value);
    let coeff_var = if mean_value.abs() > 0.0 {
        std_dev / mean_value.abs()
    } else {
        0.0
    };
    Ok(1.0 / (1.0 + coeff_var.max(0.0)))
}

/// Concentration-weighted credibility score from used-premium share balance and stability.
pub fn cape_cod_elr_credibility_score(
    used_premium: &[f64],
    stability_score: f64,
) -> Result<f64, ValidationError> {
    if !stability_score.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if !(0.0..=1.0).contains(&stability_score) {
        return Err(ValidationError::InvalidMaturity {
            index: 0,
            value: stability_score,
        });
    }
    if used_premium.is_empty() {
        return Ok(stability_score);
    }
    for &value in used_premium {
        if !value.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if value < 0.0 {
            return Err(ValidationError::Negative);
        }
    }

    let total_used: f64 = used_premium.iter().sum();
    if total_used <= 0.0 {
        return Err(ValidationError::DivideByZero("Cape Cod used premium"));
    }

    let herfindahl: f64 = used_premium
        .iter()
        .map(|value| {
            let share = value / total_used;
            share * share
        })
        .sum();
    let concentration_score = if herfindahl > 0.0 {
        1.0 / (used_premium.len() as f64 * herfindahl)
    } else {
        0.0
    };
    Ok(stability_score * concentration_score.clamp(0.0, 1.0))
}

fn from_projection(projection: CapeCodProjection) -> Result<CapeCodDiagnostics, ValidationError> {
    let elr_by_origin = cape_cod_elr_by_origin(&projection.latest, &projection.used_premium)?;
    let leave_one_out_elr =
        cape_cod_elr_leave_one_out(&projection.latest, &projection.used_premium)?;
    let leave_one_out_mean = mean(&leave_one_out_elr);
    let leave_one_out_std_dev = sample_std_dev(&leave_one_out_elr, leave_one_out_mean);
    let leave_one_out_coeff_var = if leave_one_out_mean.abs() > 0.0 {
        leave_one_out_std_dev / leave_one_out_mean.abs()
    } else {
        0.0
    };
    let stability_score = cape_cod_elr_stability_score(projection.elr, &leave_one_out_elr)?;
    let credibility_score =
        cape_cod_elr_credibility_score(&projection.used_premium, stability_score)?;

    let mut high_leverage_origin = None;
    let mut max_delta = f64::NEG_INFINITY;
    for (origin, leave_one_out) in leave_one_out_elr.iter().copied().enumerate() {
        let delta = (projection.elr - leave_one_out).abs();
        if delta.is_finite() && delta > max_delta {
            max_delta = delta;
            high_leverage_origin = Some(origin);
        }
    }

    Ok(CapeCodDiagnostics {
        elr: projection.elr,
        latest: projection.latest,
        latest_dev: projection.latest_dev,
        reported_percentage: projection.reported_percentage,
        used_premium: projection.used_premium,
        expected_ultimate: projection.expected_ultimate,
        elr_by_origin,
        leave_one_out_elr,
        leave_one_out_coeff_var,
        stability_score,
        credibility_score,
        high_leverage_origin,
        total_latest: projection.total_latest,
        total_used_premium: projection.total_used_premium,
    })
}

/// Diagnose Cape Cod using caller-supplied development factors and premium.
pub fn diagnose(
    cumulative: &TriangleView<'_>,
    factors: &DevelopmentFactors,
    premium: &[f64],
) -> Result<CapeCodDiagnostics, ValidationError> {
    let projection = CapeCodEngine::project(cumulative, factors, premium)?;
    from_projection(projection)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ldf::DevelopmentFactorEngine;
    use crate::types::Triangle;

    fn triangle_4x4() -> Triangle {
        #[rustfmt::skip]
        let data = vec![
            1000.0, 1500.0, 1700.0, 1800.0,
            1200.0, 1800.0, 2000.0, f64::NAN,
            1500.0, 2100.0, f64::NAN, f64::NAN,
            1800.0, f64::NAN, f64::NAN, f64::NAN,
        ];
        Triangle::new(4, 4, data)
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * (1.0 + b.abs())
    }

    #[test]
    fn used_up_premium_multiplies_originwise() {
        let premium = [1000.0, 1200.0, 1400.0];
        let reported = [1.0, 0.8, 0.5];
        let used = used_up_premium(&premium, &reported).expect("used premium");

        assert_eq!(used, vec![1000.0, 960.0, 700.0]);
    }

    #[test]
    fn leave_one_out_elr_recomputes_totals() {
        let latest = [180.0, 120.0, 100.0];
        let used = [90.0, 60.0, 50.0];
        let leave_one_out = cape_cod_elr_leave_one_out(&latest, &used).expect("leave one out");

        assert_eq!(leave_one_out.len(), 3);
        assert!(close(leave_one_out[0], (220.0_f64) / 110.0));
        assert!(close(leave_one_out[1], (280.0_f64) / 140.0));
        assert!(close(leave_one_out[2], (300.0_f64) / 150.0));
    }

    #[test]
    fn credibility_score_penalizes_used_premium_concentration() {
        let balanced =
            cape_cod_elr_credibility_score(&[100.0, 100.0, 100.0], 0.9).expect("balanced");
        let concentrated =
            cape_cod_elr_credibility_score(&[290.0, 5.0, 5.0], 0.9).expect("concentrated");

        assert!(balanced <= 0.9);
        assert!(concentrated < balanced);
    }

    #[test]
    fn diagnose_returns_elr_breakdown_and_stability() {
        let triangle = triangle_4x4();
        let premium = [1000.0, 1100.0, 1200.0, 1300.0];
        let factors =
            DevelopmentFactorEngine::volume_weighted(&triangle.view(), 1.0).expect("factors");

        let diagnostics = diagnose(&triangle.view(), &factors, &premium).expect("diagnostics");

        assert_eq!(diagnostics.latest.len(), 4);
        assert_eq!(diagnostics.used_premium.len(), 4);
        assert_eq!(diagnostics.elr_by_origin.len(), 4);
        assert_eq!(diagnostics.leave_one_out_elr.len(), 4);
        assert!(diagnostics.elr.is_finite());
        assert!(diagnostics.stability_score.is_finite());
        assert!(diagnostics.credibility_score.is_finite());
        assert!(diagnostics.stability_score > 0.0);
        assert!(diagnostics.stability_score <= 1.0);
        assert!(diagnostics.credibility_score > 0.0);
        assert!(diagnostics.credibility_score <= diagnostics.stability_score);
        assert!(close(
            diagnostics.total_used_premium,
            diagnostics.used_premium.iter().sum()
        ));
        assert!(close(
            diagnostics.expected_ultimate[0],
            premium[0] * diagnostics.elr
        ));
        assert!(diagnostics.high_leverage_origin.is_some());
    }

    #[test]
    fn rejects_invalid_reported_percentage() {
        let premium = [1000.0];
        let reported = [1.2];

        assert_eq!(
            used_up_premium(&premium, &reported),
            Err(ValidationError::InvalidMaturity {
                index: 0,
                value: 1.2
            })
        );
    }
}
