//! Large-loss triangle transforms.

use crate::types::{Triangle, TriangleView};
use crate::validate::{ValidationError, validate_triangle};

fn validate_nonnegative_scalar(
    value: f64,
    zero_context: &'static str,
) -> Result<(), ValidationError> {
    if !value.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if value < 0.0 {
        return Err(ValidationError::Negative);
    }
    if value == 0.0 {
        return Err(ValidationError::DivideByZero(zero_context));
    }
    Ok(())
}

fn transform_triangle<F>(triangle: &TriangleView<'_>, mut f: F) -> Result<Triangle, ValidationError>
where
    F: FnMut(f64) -> f64,
{
    validate_triangle(triangle)?;
    let mut data = vec![f64::NAN; triangle.n_origins * triangle.n_devs];
    for origin in 0..triangle.n_origins {
        for dev in 0..triangle.n_devs {
            let value = triangle.get(origin, dev);
            if value.is_finite() {
                data[origin * triangle.n_devs + dev] = f(value);
            }
        }
    }
    Ok(Triangle::new(triangle.n_origins, triangle.n_devs, data))
}

fn combine_triangles<F>(
    left: &TriangleView<'_>,
    right: &TriangleView<'_>,
    mut f: F,
) -> Result<Triangle, ValidationError>
where
    F: FnMut(f64, f64) -> f64,
{
    validate_triangle(left)?;
    validate_triangle(right)?;
    if left.n_origins != right.n_origins || left.n_devs != right.n_devs {
        return Err(ValidationError::ShapeMismatch);
    }

    let mut data = vec![f64::NAN; left.n_origins * left.n_devs];
    for origin in 0..left.n_origins {
        for dev in 0..left.n_devs {
            let a = left.get(origin, dev);
            let b = right.get(origin, dev);
            if a.is_finite() && b.is_finite() {
                data[origin * left.n_devs + dev] = f(a, b);
            }
        }
    }
    Ok(Triangle::new(left.n_origins, left.n_devs, data))
}

/// Cap every observed triangle cell at `cap`.
pub fn cap_triangle(triangle: &TriangleView<'_>, cap: f64) -> Result<Triangle, ValidationError> {
    validate_nonnegative_scalar(cap, "cap")?;
    transform_triangle(triangle, |value| value.min(cap))
}

/// Keep only the portion of each observed cell above `attachment`.
pub fn excess_triangle(
    triangle: &TriangleView<'_>,
    attachment: f64,
) -> Result<Triangle, ValidationError> {
    if !attachment.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if attachment < 0.0 {
        return Err(ValidationError::Negative);
    }
    transform_triangle(triangle, |value| (value - attachment).max(0.0))
}

/// Keep only the layer portion `min(max(x - attachment, 0), limit)`.
pub fn layer_triangle(
    triangle: &TriangleView<'_>,
    attachment: f64,
    limit: f64,
) -> Result<Triangle, ValidationError> {
    if !attachment.is_finite() || !limit.is_finite() {
        return Err(ValidationError::NotFinite);
    }
    if attachment < 0.0 || limit < 0.0 {
        return Err(ValidationError::Negative);
    }
    if limit == 0.0 {
        return Err(ValidationError::DivideByZero("limit"));
    }
    transform_triangle(triangle, |value| (value - attachment).max(0.0).min(limit))
}

/// Remove large-loss content by capping each observed cell at `threshold`.
pub fn large_loss_removed_triangle(
    triangle: &TriangleView<'_>,
    threshold: f64,
) -> Result<Triangle, ValidationError> {
    cap_triangle(triangle, threshold)
}

/// Add a large-loss addback triangle back onto a base triangle cellwise.
pub fn large_loss_addback(
    base_triangle: &TriangleView<'_>,
    addback_triangle: &TriangleView<'_>,
) -> Result<Triangle, ValidationError> {
    combine_triangles(base_triangle, addback_triangle, |base, addback| {
        base + addback
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_3x3() -> Triangle {
        #[rustfmt::skip]
        let data = vec![
            100.0, 250.0, 500.0,
            120.0, 400.0, f64::NAN,
            140.0, f64::NAN, f64::NAN,
        ];
        Triangle::new(3, 3, data)
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * (1.0 + b.abs())
    }

    #[test]
    fn caps_triangle_cells() {
        let triangle = triangle_3x3();
        let capped = cap_triangle(&triangle.view(), 300.0).expect("cap");

        assert!(close(capped.data[0], 100.0));
        assert!(close(capped.data[1], 250.0));
        assert!(close(capped.data[2], 300.0));
        assert!(close(capped.data[4], 300.0));
    }

    #[test]
    fn builds_excess_and_layer_triangles() {
        let triangle = triangle_3x3();
        let excess = excess_triangle(&triangle.view(), 200.0).expect("excess");
        let layer = layer_triangle(&triangle.view(), 200.0, 150.0).expect("layer");

        assert!(close(excess.data[0], 0.0));
        assert!(close(excess.data[2], 300.0));
        assert!(close(layer.data[1], 50.0));
        assert!(close(layer.data[2], 150.0));
    }

    #[test]
    fn addback_combines_triangles_cellwise() {
        let triangle = triangle_3x3();
        let removed = large_loss_removed_triangle(&triangle.view(), 300.0).expect("removed");
        let excess = excess_triangle(&triangle.view(), 300.0).expect("excess");
        let restored = large_loss_addback(&removed.view(), &excess.view()).expect("restored");

        for (left, right) in triangle.data.iter().zip(restored.data.iter()) {
            if left.is_nan() {
                assert!(right.is_nan());
            } else {
                assert!(close(*left, *right));
            }
        }
    }
}
