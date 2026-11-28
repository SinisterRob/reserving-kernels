//! Reserve roll-forward kernels.

use crate::error::ReservingError;
use crate::validate::validate_nonnegative_finite_vector;

/// Basic reserve movement decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct ReserveRollForward {
    /// Prior ultimate indication.
    pub old_ultimate: f64,
    /// Current ultimate indication.
    pub new_ultimate: f64,
    /// Prior latest observed/reported amount.
    pub old_latest: f64,
    /// Current latest observed/reported amount.
    pub new_latest: f64,
    /// Change in latest observed/reported amount.
    pub paid_or_reported_activity: f64,
    /// Change in ultimate indication.
    pub change_in_ultimate: f64,
    /// Prior reserve, `old_ultimate - old_latest`.
    pub old_reserve: f64,
    /// Current reserve, `new_ultimate - new_latest`.
    pub new_reserve: f64,
    /// Change in reserve, `new_reserve - old_reserve`.
    pub change_in_reserve: f64,
}

fn validate_scalar(value: f64) -> Result<(), ReservingError> {
    if !value.is_finite() {
        return Err(ReservingError::NotFinite);
    }
    if value < 0.0 {
        return Err(ReservingError::Negative);
    }
    Ok(())
}

fn build_one(
    old_ultimate: f64,
    new_ultimate: f64,
    old_latest: f64,
    new_latest: f64,
) -> ReserveRollForward {
    let paid_or_reported_activity = new_latest - old_latest;
    let change_in_ultimate = new_ultimate - old_ultimate;
    let old_reserve = old_ultimate - old_latest;
    let new_reserve = new_ultimate - new_latest;
    let change_in_reserve = new_reserve - old_reserve;

    ReserveRollForward {
        old_ultimate,
        new_ultimate,
        old_latest,
        new_latest,
        paid_or_reported_activity,
        change_in_ultimate,
        old_reserve,
        new_reserve,
        change_in_reserve,
    }
}

/// Build one reserve roll-forward decomposition from scalar inputs.
pub fn rollforward(
    old_ultimate: f64,
    new_ultimate: f64,
    old_latest: f64,
    new_latest: f64,
) -> Result<ReserveRollForward, ReservingError> {
    validate_scalar(old_ultimate)?;
    validate_scalar(new_ultimate)?;
    validate_scalar(old_latest)?;
    validate_scalar(new_latest)?;
    Ok(build_one(
        old_ultimate,
        new_ultimate,
        old_latest,
        new_latest,
    ))
}

/// Build reserve roll-forwards by origin from vector inputs.
pub fn rollforward_by_origin(
    old_ultimate: &[f64],
    new_ultimate: &[f64],
    old_latest: &[f64],
    new_latest: &[f64],
) -> Result<Vec<ReserveRollForward>, ReservingError> {
    let n = old_ultimate.len();
    validate_nonnegative_finite_vector(old_ultimate, n)?;
    validate_nonnegative_finite_vector(new_ultimate, n)?;
    validate_nonnegative_finite_vector(old_latest, n)?;
    validate_nonnegative_finite_vector(new_latest, n)?;

    Ok((0..n)
        .map(|index| {
            build_one(
                old_ultimate[index],
                new_ultimate[index],
                old_latest[index],
                new_latest[index],
            )
        })
        .collect())
}

/// Aggregate many origin-level roll-forwards into one total movement row.
pub fn aggregate_rollforward(
    rows: &[ReserveRollForward],
) -> Result<ReserveRollForward, ReservingError> {
    let mut old_ultimate = 0.0;
    let mut new_ultimate = 0.0;
    let mut old_latest = 0.0;
    let mut new_latest = 0.0;

    for row in rows {
        validate_scalar(row.old_ultimate)?;
        validate_scalar(row.new_ultimate)?;
        validate_scalar(row.old_latest)?;
        validate_scalar(row.new_latest)?;
        old_ultimate += row.old_ultimate;
        new_ultimate += row.new_ultimate;
        old_latest += row.old_latest;
        new_latest += row.new_latest;
    }
    if !old_ultimate.is_finite()
        || !new_ultimate.is_finite()
        || !old_latest.is_finite()
        || !new_latest.is_finite()
    {
        return Err(ReservingError::NotFinite);
    }

    Ok(build_one(
        old_ultimate,
        new_ultimate,
        old_latest,
        new_latest,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * (1.0 + b.abs())
    }

    #[test]
    fn scalar_rollforward_balances() {
        let row = rollforward(1000.0, 1100.0, 600.0, 700.0).expect("rollforward");

        assert!(close(row.paid_or_reported_activity, 100.0));
        assert!(close(row.change_in_ultimate, 100.0));
        assert!(close(row.old_reserve, 400.0));
        assert!(close(row.new_reserve, 400.0));
        assert!(close(row.change_in_reserve, 0.0));
    }

    #[test]
    fn origin_and_aggregate_rollforward_match() {
        let rows = rollforward_by_origin(
            &[1000.0, 800.0],
            &[1100.0, 760.0],
            &[600.0, 500.0],
            &[700.0, 540.0],
        )
        .expect("rows");
        let total = aggregate_rollforward(&rows).expect("aggregate");

        assert_eq!(rows.len(), 2);
        assert!(close(total.old_ultimate, 1800.0));
        assert!(close(total.new_ultimate, 1860.0));
        assert!(close(total.old_reserve, 700.0));
        assert!(close(total.new_reserve, 620.0));
        assert!(close(total.change_in_reserve, -80.0));
    }

    #[test]
    fn rollforward_rejects_shape_mismatch() {
        assert_eq!(
            rollforward_by_origin(&[1.0, 2.0], &[1.0], &[1.0, 2.0], &[1.0, 2.0]),
            Err(ReservingError::ShapeMismatch)
        );
    }
}
