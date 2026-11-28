//! Large-loss triangle transforms.
//!
//! ## Semantics: per-claim versus aggregate triangles
//!
//! Capping a cell at a per-claim limit `L` is only meaningful when the cell
//! holds **one claim's** cumulative loss. An accident-year *aggregate*
//! triangle cell is the sum over many claims, so `min(cell, L)` caps the
//! whole accident year rather than removing the large claims inside it.
//!
//! * [`cap_triangle`], [`excess_triangle`], [`layer_triangle`] and
//!   [`large_loss_removed_triangle`] are **cellwise** transforms. Apply them to
//!   individual-claim development triangles (or to any triangle where a cell
//!   is a single risk), never to an aggregate accident-year triangle.
//! * For aggregate triangles use the Friedland (2010, Ch. 6/7) approach:
//!   - build limited and excess aggregate triangles from claim-level
//!     histories with [`aggregate_claims_split`] (each claim is limited at
//!     `L` *before* summing), or
//!   - subtract a separately supplied large-loss (excess) aggregate triangle
//!     with [`remove_large_loss_excess`], and add it back with
//!     [`large_loss_addback`].

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
            if !triangle.is_scheduled_observed(origin, dev) {
                continue;
            }
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
            if !left.is_scheduled_observed(origin, dev) {
                continue;
            }
            let a = left.get(origin, dev);
            let b = right.get(origin, dev);
            if a.is_finite() && b.is_finite() {
                let out = f(a, b);
                if !out.is_finite() {
                    return Err(ValidationError::NotFinite);
                }
                data[origin * left.n_devs + dev] = out;
            }
        }
    }
    Ok(Triangle::new(left.n_origins, left.n_devs, data))
}

/// Cap every observed triangle cell at `cap` (cellwise).
///
/// Only valid when each cell is a single claim's cumulative loss; see the
/// module docs. For aggregate triangles use [`aggregate_claims_split`].
pub fn cap_triangle(triangle: &TriangleView<'_>, cap: f64) -> Result<Triangle, ValidationError> {
    validate_nonnegative_scalar(cap, "cap")?;
    transform_triangle(triangle, |value| value.min(cap))
}

/// Keep only the portion of each observed cell above `attachment` (cellwise,
/// per-claim semantics; see the module docs).
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

/// Keep only the layer portion `min(max(x - attachment, 0), limit)` (cellwise,
/// per-claim semantics; see the module docs).
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

/// Cap each observed cell of a **single-claim** triangle at `threshold`.
///
/// This is identical to [`cap_triangle`] and is *not* a large-loss removal
/// for accident-year aggregate triangles (it would cap whole accident years).
/// For aggregate data use [`aggregate_claims_split`] (claim-level limiting)
/// or [`remove_large_loss_excess`] (subtract a supplied excess triangle).
#[deprecated(
    since = "0.4.1",
    note = "cellwise cap is per-claim only; use aggregate_claims_split or remove_large_loss_excess for aggregate triangles"
)]
pub fn large_loss_removed_triangle(
    triangle: &TriangleView<'_>,
    threshold: f64,
) -> Result<Triangle, ValidationError> {
    cap_triangle(triangle, threshold)
}

/// Remove large-loss content from an **aggregate** triangle by subtracting a
/// separately supplied large-loss excess triangle cellwise (Friedland 2010):
/// `limited[i, j] = aggregate[i, j] − excess[i, j]`.
///
/// Both triangles must share shape and observed geometry. Returns `Negative`
/// if any excess cell exceeds its aggregate cell (by more than a relative
/// `1e-12` rounding tolerance; such cells are floored at zero).
pub fn remove_large_loss_excess(
    aggregate: &TriangleView<'_>,
    excess: &TriangleView<'_>,
) -> Result<Triangle, ValidationError> {
    let mut violation = false;
    let out = combine_triangles(aggregate, excess, |total, large| {
        let diff = total - large;
        if diff < -1e-12 * total.abs().max(1.0) {
            violation = true;
        }
        diff.max(0.0)
    })?;
    if violation {
        return Err(ValidationError::Negative);
    }
    Ok(out)
}

/// One claim's cumulative development history for claim-level aggregation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClaimHistory<'a> {
    /// Origin (accident-period) row index of the claim.
    pub origin: usize,
    /// Cumulative loss by development age for every observed age of the
    /// origin row (length `n − origin`; zeros before the claim is reported).
    pub cumulative: &'a [f64],
}

/// Aggregate limited and excess triangles built from claim-level histories.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct LargeLossSplit {
    /// Aggregate of `min(claim cell, limit)` — the large-loss-limited triangle.
    pub limited: Triangle,
    /// Aggregate of `max(claim cell − limit, 0)` — the excess (large-loss) triangle.
    pub excess: Triangle,
    /// Number of claims whose latest cumulative value exceeds `limit`.
    pub n_claims_exceeding: usize,
}

/// Build aggregate `n × n` limited and excess triangles from per-claim
/// cumulative histories, limiting **each claim** at `limit` before summing
/// (Friedland 2010 large-loss treatment).
///
/// `limited + excess` reproduces the ground-up aggregate triangle exactly.
/// Validation: `n >= 1`, `limit` finite and `> 0`, every claim origin `< n`,
/// `cumulative.len() == n − origin`, and finite non-negative claim cells.
pub fn aggregate_claims_split(
    claims: &[ClaimHistory<'_>],
    n: usize,
    limit: f64,
) -> Result<LargeLossSplit, ValidationError> {
    validate_nonnegative_scalar(limit, "limit")?;
    if n == 0 {
        return Err(ValidationError::ShapeMismatch);
    }
    let mut limited = vec![f64::NAN; n * n];
    let mut excess = vec![f64::NAN; n * n];
    for i in 0..n {
        for j in 0..(n - i) {
            limited[i * n + j] = 0.0;
            excess[i * n + j] = 0.0;
        }
    }
    let mut n_claims_exceeding = 0_usize;
    for claim in claims {
        if claim.origin >= n || claim.cumulative.len() != n - claim.origin {
            return Err(ValidationError::ShapeMismatch);
        }
        for (j, &value) in claim.cumulative.iter().enumerate() {
            if !value.is_finite() {
                return Err(ValidationError::NotFinite);
            }
            if value < 0.0 {
                return Err(ValidationError::Negative);
            }
            let cell = claim.origin * n + j;
            limited[cell] += value.min(limit);
            excess[cell] += (value - limit).max(0.0);
        }
        if claim.cumulative.last().is_some_and(|&v| v > limit) {
            n_claims_exceeding += 1;
        }
    }
    Ok(LargeLossSplit {
        limited: Triangle::new(n, n, limited),
        excess: Triangle::new(n, n, excess),
        n_claims_exceeding,
    })
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

        assert!(close(capped.data()[0], 100.0));
        assert!(close(capped.data()[1], 250.0));
        assert!(close(capped.data()[2], 300.0));
        assert!(close(capped.data()[4], 300.0));
    }

    #[test]
    fn builds_excess_and_layer_triangles() {
        let triangle = triangle_3x3();
        let excess = excess_triangle(&triangle.view(), 200.0).expect("excess");
        let layer = layer_triangle(&triangle.view(), 200.0, 150.0).expect("layer");

        assert!(close(excess.data()[0], 0.0));
        assert!(close(excess.data()[2], 300.0));
        assert!(close(layer.data()[1], 50.0));
        assert!(close(layer.data()[2], 150.0));
    }

    #[test]
    #[allow(deprecated)]
    fn addback_combines_triangles_cellwise() {
        let triangle = triangle_3x3();
        let removed = large_loss_removed_triangle(&triangle.view(), 300.0).expect("removed");
        let excess = excess_triangle(&triangle.view(), 300.0).expect("excess");
        let restored = large_loss_addback(&removed.view(), &excess.view()).expect("restored");

        for (left, right) in triangle.data().iter().zip(restored.data().iter()) {
            if left.is_nan() {
                assert!(right.is_nan());
            } else {
                assert!(close(*left, *right));
            }
        }
    }

    /// Two origins, three claims. AY0: claims A (big) and B (small); AY1: C.
    fn claims() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        (vec![200.0, 900.0], vec![50.0, 80.0], vec![120.0])
    }

    #[test]
    fn claim_level_split_limits_each_claim_before_aggregation() {
        let (a, b, c) = claims();
        let histories = [
            ClaimHistory {
                origin: 0,
                cumulative: &a,
            },
            ClaimHistory {
                origin: 0,
                cumulative: &b,
            },
            ClaimHistory {
                origin: 1,
                cumulative: &c,
            },
        ];
        let split = aggregate_claims_split(&histories, 2, 500.0).expect("split");
        // AY0 dev1 limited = min(900, 500) + min(80, 500) = 580; excess = 400.
        assert!(close(split.limited.data()[0], 250.0));
        assert!(close(split.limited.data()[1], 580.0));
        assert!(close(split.excess.data()[1], 400.0));
        assert!(close(split.limited.data()[2], 120.0));
        assert!(split.limited.data()[3].is_nan());
        assert_eq!(split.n_claims_exceeding, 1);

        // The naive aggregate cap would wrongly cap AY0 at 500 (losing 80 of
        // non-large loss): ground-up aggregate is [250, 980; 120].
        let aggregate = Triangle::new(2, 2, vec![250.0, 980.0, 120.0, f64::NAN]);
        let naive = cap_triangle(&aggregate.view(), 500.0).expect("naive");
        assert!(close(naive.data()[1], 500.0));
        assert!(naive.data()[1] < split.limited.data()[1]);

        // Friedland aggregate API: aggregate − excess == claim-level limited,
        // and limited + excess reproduces the ground-up aggregate.
        let removed =
            remove_large_loss_excess(&aggregate.view(), &split.excess.view()).expect("removed");
        let restored = large_loss_addback(&removed.view(), &split.excess.view()).expect("add");
        for k in [0, 1, 2] {
            assert!(close(removed.data()[k], split.limited.data()[k]));
            assert!(close(restored.data()[k], aggregate.data()[k]));
        }
    }

    #[test]
    fn excess_larger_than_aggregate_is_rejected() {
        let aggregate = Triangle::new(2, 2, vec![100.0, 150.0, 80.0, f64::NAN]);
        let excess = Triangle::new(2, 2, vec![0.0, 200.0, 0.0, f64::NAN]);
        assert_eq!(
            remove_large_loss_excess(&aggregate.view(), &excess.view()),
            Err(ValidationError::Negative)
        );
    }

    #[test]
    fn claim_split_validates_shapes_and_limit() {
        let bad = [1.0, 2.0, 3.0];
        let histories = [ClaimHistory {
            origin: 0,
            cumulative: &bad,
        }];
        assert_eq!(
            aggregate_claims_split(&histories, 2, 10.0),
            Err(ValidationError::ShapeMismatch)
        );
        let ok = [1.0];
        let histories = [ClaimHistory {
            origin: 2,
            cumulative: &ok,
        }];
        assert_eq!(
            aggregate_claims_split(&histories, 2, 10.0),
            Err(ValidationError::ShapeMismatch)
        );
        assert_eq!(
            aggregate_claims_split(&[], 2, 0.0),
            Err(ValidationError::DivideByZero("limit"))
        );
    }
}
