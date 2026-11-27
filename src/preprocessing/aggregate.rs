//! Portfolio-level slice reducers.

use crate::types::TriangleView;

/// Sum of latest observed cumulative losses (total "paid to date" across
/// origin years) without intermediate heap allocation.
#[inline]
pub fn total_paid(t: &TriangleView<'_>) -> f64 {
    let n = t.n_origins.min(t.n_devs);
    let mut sum = 0.0;
    for i in 0..n {
        sum += t.get(i, n - 1 - i);
    }
    sum
}

/// Sum of a per-origin ultimate vector.
#[inline]
pub fn total_ultimate(ultimate: &[f64]) -> f64 {
    ultimate.iter().sum()
}

/// Sum of a per-origin reserve vector.
#[inline]
pub fn total_reserve(reserve: &[f64]) -> f64 {
    reserve.iter().sum()
}

/// Portfolio paid ratio: total paid / total ultimate.
#[inline]
pub fn paid_ratio(t: &TriangleView<'_>, ultimate: &[f64]) -> f64 {
    let u = total_ultimate(ultimate);
    if u > 0.0 { total_paid(t) / u } else { f64::NAN }
}

/// Portfolio loss ratio: total ultimate / total premium.
/// Returns `NaN` if slice lengths differ or total premium is non-positive.
#[inline]
pub fn portfolio_loss_ratio(ultimate: &[f64], premium: &[f64]) -> f64 {
    if ultimate.len() != premium.len() {
        return f64::NAN;
    }
    let mut num = 0.0;
    let mut den = 0.0;
    for (u, p) in ultimate.iter().zip(premium.iter()) {
        num += *u;
        den += *p;
    }
    if den > 0.0 { num / den } else { f64::NAN }
}
