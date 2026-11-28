//! Triangle holdout and selector backtesting kernels.

use crate::preprocessing::selection::CandidateSelector;
use crate::types::{Triangle, TriangleView};

/// One held-out triangle cell and its actual value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoldoutCell {
    /// Origin row index.
    pub origin: usize,
    /// Development column index.
    pub dev: usize,
    /// Actual cumulative value held out from the training triangle.
    pub actual: f64,
}

/// Training triangle plus the held-out cells removed from it.
#[derive(Debug, Clone, PartialEq)]
pub struct HoldoutSet {
    /// Triangle after replacing held-out cells with `NaN`.
    pub training: Triangle,
    /// Cells held out for scoring.
    pub held_out: Vec<HoldoutCell>,
}

/// Backtest metrics for one selector on one holdout scheme.
#[derive(Debug, Clone, PartialEq)]
pub struct BacktestScore {
    /// Selector being evaluated.
    pub selector: CandidateSelector,
    /// Number of held-out cells scored.
    pub n_held_out: usize,
    /// Mean absolute percentage error.
    pub mape: f64,
    /// Root mean squared error.
    pub rmse: f64,
    /// Mean signed error.
    pub mean_error: f64,
    /// Predicted cumulative values for held-out cells in holdout order.
    pub predictions: Vec<f64>,
}

fn select_ldfs(t: &TriangleView<'_>, selector: &CandidateSelector) -> Vec<f64> {
    if let CandidateSelector::Manual(ldfs) = selector {
        return ldfs.clone();
    }

    let n = t.n_origins.min(t.n_devs);
    if n < 2 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(n - 1);
    let mut ratios = Vec::with_capacity(n);
    let mut weighted_pairs = Vec::with_capacity(n);
    for dev in 0..n - 1 {
        ratios.clear();
        weighted_pairs.clear();
        for origin in 0..n - dev - 1 {
            let current = t.get(origin, dev);
            let next = t.get(origin, dev + 1);
            if current.is_finite() && next.is_finite() && current > 0.0 {
                ratios.push(next / current);
                weighted_pairs.push((current, next));
            }
        }

        let factor = match selector {
            CandidateSelector::VolumeWeighted => {
                let den: f64 = weighted_pairs.iter().map(|(current, _)| *current).sum();
                let num: f64 = weighted_pairs.iter().map(|(_, next)| *next).sum();
                if den > 0.0 { num / den } else { f64::NAN }
            }
            CandidateSelector::SimpleAverage => {
                if ratios.is_empty() {
                    f64::NAN
                } else {
                    ratios.iter().sum::<f64>() / ratios.len() as f64
                }
            }
            CandidateSelector::Median => {
                ratios.sort_by(f64::total_cmp);
                let len = ratios.len();
                if len == 0 {
                    f64::NAN
                } else if len % 2 == 1 {
                    ratios[len / 2]
                } else {
                    f64::midpoint(ratios[len / 2 - 1], ratios[len / 2])
                }
            }
            CandidateSelector::ExHiLo => {
                ratios.sort_by(f64::total_cmp);
                if ratios.len() >= 3 {
                    let trimmed = &ratios[1..ratios.len() - 1];
                    trimmed.iter().sum::<f64>() / trimmed.len() as f64
                } else if ratios.is_empty() {
                    f64::NAN
                } else {
                    ratios.iter().sum::<f64>() / ratios.len() as f64
                }
            }
            CandidateSelector::VolumeWeightedLastN(last_n) => {
                let take = if *last_n == 0 {
                    weighted_pairs.len()
                } else {
                    weighted_pairs.len().min(*last_n)
                };
                let pairs = &weighted_pairs[weighted_pairs.len().saturating_sub(take)..];
                let den: f64 = pairs.iter().map(|(current, _)| *current).sum();
                let num: f64 = pairs.iter().map(|(_, next)| *next).sum();
                if den > 0.0 { num / den } else { f64::NAN }
            }
            CandidateSelector::SimpleAverageLastN(last_n) => {
                let take = if *last_n == 0 {
                    ratios.len()
                } else {
                    ratios.len().min(*last_n)
                };
                let subset = &ratios[ratios.len().saturating_sub(take)..];
                if subset.is_empty() {
                    f64::NAN
                } else {
                    subset.iter().sum::<f64>() / subset.len() as f64
                }
            }
            CandidateSelector::Manual(_) => unreachable!(),
        };
        out.push(factor);
    }
    out
}

fn training_view_with_holdout(t: &TriangleView<'_>, held_out: &[HoldoutCell]) -> Triangle {
    let mut data = t.data.to_vec();
    for cell in held_out {
        data[cell.origin * t.n_devs + cell.dev] = f64::NAN;
    }
    Triangle::new(t.n_origins, t.n_devs, data)
}

/// Hold out the latest observed cumulative value in each origin row except fully developed rows.
pub fn holdout_latest_diagonal(t: &TriangleView<'_>) -> HoldoutSet {
    let n = t.n_origins.min(t.n_devs);
    let mut held_out = Vec::new();
    for origin in 0..n {
        let dev = n - 1 - origin;
        if dev == n - 1 || dev == 0 {
            continue;
        }
        let actual = t.get(origin, dev);
        if actual.is_finite() {
            held_out.push(HoldoutCell {
                origin,
                dev,
                actual,
            });
        }
    }

    HoldoutSet {
        training: training_view_with_holdout(t, &held_out),
        held_out,
    }
}

/// Hold out the last `k` observed diagonals under square-triangle geometry.
pub fn holdout_last_k_diagonals(t: &TriangleView<'_>, k: usize) -> HoldoutSet {
    let n = t.n_origins.min(t.n_devs);
    let mut held_out = Vec::new();
    let k = k.min(n);
    for offset in 0..k {
        let diagonal = n - 1 - offset;
        for origin in 0..=diagonal {
            let dev = diagonal - origin;
            if dev >= t.n_devs || dev == 0 {
                continue;
            }
            let actual = t.get(origin, dev);
            if actual.is_finite() {
                held_out.push(HoldoutCell {
                    origin,
                    dev,
                    actual,
                });
            }
        }
    }

    HoldoutSet {
        training: training_view_with_holdout(t, &held_out),
        held_out,
    }
}

fn predict_holdout_cells(
    training: &TriangleView<'_>,
    ldfs: &[f64],
    _tail: f64,
    held_out: &[HoldoutCell],
) -> Vec<f64> {
    held_out
        .iter()
        .map(|cell| {
            let mut latest_dev = None;
            let mut latest = f64::NAN;
            for dev in 0..=cell.dev {
                let value = training.get(cell.origin, dev);
                if training.is_scheduled_observed(cell.origin, dev) && value.is_finite() {
                    latest_dev = Some(dev);
                    latest = value;
                }
            }

            let Some(latest_dev) = latest_dev else {
                return f64::NAN;
            };

            if !latest.is_finite() || cell.dev < latest_dev {
                return f64::NAN;
            }
            let mut predicted = latest;
            for link in latest_dev..cell.dev {
                if link >= ldfs.len() {
                    return f64::NAN;
                }
                predicted *= ldfs[link];
            }
            predicted
        })
        .collect()
}

/// Fit a selector on the training triangle and score predictions on the held-out cells.
pub fn fit_predict_score(
    holdout: &HoldoutSet,
    selector: CandidateSelector,
    tail: f64,
) -> BacktestScore {
    let training_view = holdout.training.view();
    let ldfs = select_ldfs(&training_view, &selector);
    let predictions = predict_holdout_cells(&training_view, &ldfs, tail, &holdout.held_out);

    let mut abs_pct_sum = 0.0;
    let mut sq_sum = 0.0;
    let mut err_sum = 0.0;
    let mut count = 0usize;
    let mut mape_count = 0usize;

    for (prediction, cell) in predictions.iter().zip(holdout.held_out.iter()) {
        if !prediction.is_finite() || !cell.actual.is_finite() {
            continue;
        }
        let error = prediction - cell.actual;
        if cell.actual != 0.0 {
            abs_pct_sum += (error / cell.actual).abs();
            mape_count += 1;
        }
        sq_sum += error * error;
        err_sum += error;
        count += 1;
    }

    BacktestScore {
        selector,
        n_held_out: count,
        mape: if mape_count > 0 {
            abs_pct_sum / mape_count as f64
        } else {
            f64::NAN
        },
        rmse: if count > 0 {
            (sq_sum / count as f64).sqrt()
        } else {
            f64::NAN
        },
        mean_error: if count > 0 {
            err_sum / count as f64
        } else {
            f64::NAN
        },
        predictions,
    }
}

/// Backtest a selector family against a latest-diagonal holdout.
pub fn backtest_selector_family(
    t: &TriangleView<'_>,
    selectors: &[CandidateSelector],
    tail: f64,
) -> Vec<BacktestScore> {
    let holdout = holdout_latest_diagonal(t);
    selectors
        .iter()
        .cloned()
        .map(|selector| fit_predict_score(&holdout, selector, tail))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn latest_diagonal_holdout_removes_recent_cells() {
        let tri = triangle_4x4();
        let holdout = holdout_latest_diagonal(&tri.view());

        assert_eq!(holdout.held_out.len(), 2);
        assert!(holdout.training.data()[4 + 2].is_nan());
        assert!(holdout.training.data()[2 * 4 + 1].is_nan());
        assert!(holdout.training.data()[3 * 4].is_finite());
    }

    #[test]
    fn fit_predict_score_returns_numeric_metrics() {
        let tri = triangle_4x4();
        let holdout = holdout_latest_diagonal(&tri.view());
        let score = fit_predict_score(&holdout, CandidateSelector::VolumeWeighted, 1.0);

        assert_eq!(score.n_held_out, 2);
        assert_eq!(score.predictions.len(), 2);
        assert!(score.mape.is_finite());
        assert!(score.rmse.is_finite());
    }

    #[test]
    fn family_backtest_scores_each_selector() {
        let tri = triangle_4x4();
        let scores = backtest_selector_family(
            &tri.view(),
            &[
                CandidateSelector::VolumeWeighted,
                CandidateSelector::SimpleAverage,
            ],
            1.0,
        );

        assert_eq!(scores.len(), 2);
        assert!(scores.iter().all(|score| score.n_held_out == 2));
    }
}
