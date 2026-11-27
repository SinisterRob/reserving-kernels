//! Case-reserve triangle diagnostics.

use crate::diagnostics::paid_incurred_reconciliation::case_triangle;
use crate::types::{Triangle, TriangleView};
use crate::validate::{ValidationError, validate_triangle};

/// Case-reserve diagnostic bundle by origin and development.
#[derive(Debug, Clone, PartialEq)]
pub struct CaseReserveDiagnostics {
    /// Latest case reserve by origin.
    pub latest_case_reserve: Vec<f64>,
    /// Latest case-reserve ratio by origin, `case / incurred`.
    pub latest_case_reserve_ratio: Vec<f64>,
    /// Mean case reserve by development age over observed cells.
    pub case_decay_pattern: Vec<f64>,
}

fn validate_compatible(
    paid: &TriangleView<'_>,
    incurred: &TriangleView<'_>,
) -> Result<(), ValidationError> {
    validate_triangle(paid)?;
    validate_triangle(incurred)?;
    if paid.n_origins != incurred.n_origins || paid.n_devs != incurred.n_devs {
        return Err(ValidationError::ShapeMismatch);
    }
    Ok(())
}

/// Case-reserve ratio triangle `(incurred - paid) / incurred`.
pub fn case_reserve_ratio_triangle(
    paid: &TriangleView<'_>,
    incurred: &TriangleView<'_>,
) -> Result<Triangle, ValidationError> {
    validate_compatible(paid, incurred)?;
    let mut data = vec![f64::NAN; paid.n_origins * paid.n_devs];
    for origin in 0..paid.n_origins {
        for dev in 0..paid.n_devs {
            let paid_value = paid.get(origin, dev);
            let incurred_value = incurred.get(origin, dev);
            if paid_value.is_finite() && incurred_value.is_finite() && incurred_value > 0.0 {
                data[origin * paid.n_devs + dev] = (incurred_value - paid_value) / incurred_value;
            }
        }
    }
    Ok(Triangle::new(paid.n_origins, paid.n_devs, data))
}

/// Alias for the case-reserve emergence surface `incurred - paid`.
pub fn incurred_minus_paid_emergence(
    paid: &TriangleView<'_>,
    incurred: &TriangleView<'_>,
) -> Result<Triangle, ValidationError> {
    case_triangle(paid, incurred)
}

/// Mean case reserve by development age across observed cells.
pub fn case_decay_pattern(
    paid: &TriangleView<'_>,
    incurred: &TriangleView<'_>,
) -> Result<Vec<f64>, ValidationError> {
    validate_compatible(paid, incurred)?;
    let mut out = vec![f64::NAN; paid.n_devs];
    for dev in 0..paid.n_devs {
        let mut sum = 0.0;
        let mut count = 0;
        for origin in 0..paid.n_origins {
            if !paid.is_scheduled_observed(origin, dev) {
                continue;
            }
            let paid_value = paid.get(origin, dev);
            let incurred_value = incurred.get(origin, dev);
            if paid_value.is_finite() && incurred_value.is_finite() {
                sum += incurred_value - paid_value;
                count += 1;
            }
        }
        if count > 0 {
            out[dev] = sum / f64::from(count);
        }
    }
    Ok(out)
}

/// Bundle latest case-reserve signals and average decay by development age.
pub fn diagnose(
    paid: &TriangleView<'_>,
    incurred: &TriangleView<'_>,
) -> Result<CaseReserveDiagnostics, ValidationError> {
    validate_compatible(paid, incurred)?;
    let mut latest_case_reserve = Vec::with_capacity(paid.n_origins);
    let mut latest_case_reserve_ratio = Vec::with_capacity(paid.n_origins);
    for origin in 0..paid.n_origins {
        let dev = paid.n_devs - 1 - origin;
        let paid_latest = paid.get(origin, dev);
        let incurred_latest = incurred.get(origin, dev);
        let case_latest = incurred_latest - paid_latest;
        latest_case_reserve.push(case_latest);
        latest_case_reserve_ratio.push(if incurred_latest > 0.0 {
            case_latest / incurred_latest
        } else {
            f64::NAN
        });
    }
    Ok(CaseReserveDiagnostics {
        latest_case_reserve,
        latest_case_reserve_ratio,
        case_decay_pattern: case_decay_pattern(paid, incurred)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paid_triangle() -> Triangle {
        #[rustfmt::skip]
        let data = vec![
            1000.0, 1500.0, 1700.0, 1800.0,
            1200.0, 1800.0, 2000.0, f64::NAN,
            1500.0, 2100.0, f64::NAN, f64::NAN,
            1800.0, f64::NAN, f64::NAN, f64::NAN,
        ];
        Triangle::new(4, 4, data)
    }

    fn incurred_triangle() -> Triangle {
        #[rustfmt::skip]
        let data = vec![
            1200.0, 1650.0, 1785.0, 1840.0,
            1400.0, 1950.0, 2120.0, f64::NAN,
            1700.0, 2280.0, f64::NAN, f64::NAN,
            2000.0, f64::NAN, f64::NAN, f64::NAN,
        ];
        Triangle::new(4, 4, data)
    }

    #[test]
    fn builds_case_reserve_ratio_and_decay_pattern() {
        let ratio =
            case_reserve_ratio_triangle(&paid_triangle().view(), &incurred_triangle().view())
                .expect("ratio");
        let decay = case_decay_pattern(&paid_triangle().view(), &incurred_triangle().view())
            .expect("decay");

        assert!(ratio.data()[0].is_finite());
        assert_eq!(decay.len(), 4);
        assert!(decay[0].is_finite());
    }

    #[test]
    fn diagnosis_returns_latest_signals() {
        let diagnostics =
            diagnose(&paid_triangle().view(), &incurred_triangle().view()).expect("diagnostics");
        assert_eq!(diagnostics.latest_case_reserve.len(), 4);
        assert_eq!(diagnostics.latest_case_reserve_ratio.len(), 4);
        assert_eq!(diagnostics.case_decay_pattern.len(), 4);
    }
}
