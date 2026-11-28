//! Deterministic trend transforms for vectors and triangles.

use crate::error::ReservingError;
use crate::types::{Triangle, TriangleView};
use crate::validate::{validate_nonnegative_finite_slice, validate_triangle};

fn validate_years(years: &[f64]) -> Result<(), ReservingError> {
    for &value in years {
        if !value.is_finite() {
            return Err(ReservingError::NotFinite);
        }
    }
    Ok(())
}

fn validate_rate(rate: f64) -> Result<(), ReservingError> {
    if !rate.is_finite() {
        return Err(ReservingError::NotFinite);
    }
    if rate <= -1.0 {
        return Err(ReservingError::Negative);
    }
    Ok(())
}

fn trend_factor(rate: f64, years: f64) -> f64 {
    (1.0 + rate).powf(years)
}

fn apply_trend(values: &[f64], years: &[f64], rate: f64) -> Result<Vec<f64>, ReservingError> {
    if values.len() != years.len() {
        return Err(ReservingError::ShapeMismatch);
    }
    validate_nonnegative_finite_slice(values)?;
    validate_years(years)?;
    validate_rate(rate)?;

    let mut out = Vec::with_capacity(values.len());
    for (&value, &elapsed_years) in values.iter().zip(years.iter()) {
        let trended = value * trend_factor(rate, elapsed_years);
        if !trended.is_finite() {
            return Err(ReservingError::NotFinite);
        }
        out.push(trended);
    }
    Ok(out)
}

/// Apply one annual loss-trend rate to a vector using years-to-level by item.
pub fn apply_loss_trend(
    losses: &[f64],
    years_to_level: &[f64],
    annual_rate: f64,
) -> Result<Vec<f64>, ReservingError> {
    apply_trend(losses, years_to_level, annual_rate)
}

/// Apply one annual premium-trend rate to a vector using years-to-level by item.
pub fn apply_premium_trend(
    premium: &[f64],
    years_to_level: &[f64],
    annual_rate: f64,
) -> Result<Vec<f64>, ReservingError> {
    apply_trend(premium, years_to_level, annual_rate)
}

/// Trend values to a common level using the supplied annual rate and item-specific years.
pub fn trend_to_common_level(
    values: &[f64],
    years_to_level: &[f64],
    annual_rate: f64,
) -> Result<Vec<f64>, ReservingError> {
    apply_trend(values, years_to_level, annual_rate)
}

/// Detrend each origin row of a triangle by dividing through the supplied origin factor.
pub fn detrend_triangle(
    triangle: &TriangleView<'_>,
    origin_factors: &[f64],
) -> Result<Triangle, ReservingError> {
    validate_triangle(triangle)?;
    if triangle.n_origins != origin_factors.len() {
        return Err(ReservingError::ShapeMismatch);
    }
    validate_nonnegative_finite_slice(origin_factors)?;

    let mut data = vec![f64::NAN; triangle.n_origins * triangle.n_devs];
    for origin in 0..triangle.n_origins {
        let factor = origin_factors[origin];
        if factor <= 0.0 {
            return Err(ReservingError::DivideByZero("trend factor"));
        }
        for dev in 0..triangle.n_devs {
            if !triangle.is_scheduled_observed(origin, dev) {
                continue;
            }
            let value = triangle.get(origin, dev);
            if value.is_finite() {
                let scaled = value / factor;
                if !scaled.is_finite() {
                    return Err(ReservingError::NotFinite);
                }
                data[origin * triangle.n_devs + dev] = scaled;
            }
        }
    }
    Ok(Triangle::new(triangle.n_origins, triangle.n_devs, data))
}

/// Retrend each origin row of a triangle by multiplying through the supplied origin factor.
pub fn retrend_triangle(
    triangle: &TriangleView<'_>,
    origin_factors: &[f64],
) -> Result<Triangle, ReservingError> {
    validate_triangle(triangle)?;
    if triangle.n_origins != origin_factors.len() {
        return Err(ReservingError::ShapeMismatch);
    }
    validate_nonnegative_finite_slice(origin_factors)?;

    let mut data = vec![f64::NAN; triangle.n_origins * triangle.n_devs];
    for origin in 0..triangle.n_origins {
        let factor = origin_factors[origin];
        if factor <= 0.0 {
            return Err(ReservingError::DivideByZero("trend factor"));
        }
        for dev in 0..triangle.n_devs {
            if !triangle.is_scheduled_observed(origin, dev) {
                continue;
            }
            let value = triangle.get(origin, dev);
            if value.is_finite() {
                let scaled = value * factor;
                if !scaled.is_finite() {
                    return Err(ReservingError::NotFinite);
                }
                data[origin * triangle.n_devs + dev] = scaled;
            }
        }
    }
    Ok(Triangle::new(triangle.n_origins, triangle.n_devs, data))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_3x3() -> Triangle {
        #[rustfmt::skip]
        let data = vec![
            100.0, 150.0, 180.0,
            120.0, 180.0, f64::NAN,
            140.0, f64::NAN, f64::NAN,
        ];
        Triangle::new(3, 3, data)
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * (1.0 + b.abs())
    }

    #[test]
    fn applies_loss_trend_by_years() {
        let trended = apply_loss_trend(&[100.0, 200.0], &[1.0, 2.0], 0.05).expect("trend");

        assert!(close(trended[0], 105.0));
        assert!(close(trended[1], 200.0 * 1.05_f64.powi(2)));
    }

    #[test]
    fn detrend_and_retrend_triangle_round_trip() {
        let triangle = triangle_3x3();
        let factors = [1.10, 1.05, 1.00];

        let detrended = detrend_triangle(&triangle.view(), &factors).expect("detrend");
        let retrended = retrend_triangle(&detrended.view(), &factors).expect("retrend");

        for (left, right) in triangle.data().iter().zip(retrended.data().iter()) {
            if left.is_nan() {
                assert!(right.is_nan());
            } else {
                assert!(close(*left, *right));
            }
        }
    }
}
