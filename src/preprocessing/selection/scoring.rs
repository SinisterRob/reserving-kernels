//! Numeric scoring helpers for deterministic factor selectors.

use crate::deterministic::ldf;
use crate::diagnostics::link_ratio_diagnostics::{LinkRatioDiagnostics, diagnose_all};
use crate::preprocessing::selection::CandidateSelector;
use crate::preprocessing::selection::backtest::BacktestScore;
use crate::types::TriangleView;

/// Scored selector candidate with its factor vector.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectorScore {
    /// Candidate selector being scored.
    pub selector: CandidateSelector,
    /// Weighted total score.
    pub total_score: f64,
    /// Stability contribution from link-ratio CV.
    pub stability_score: f64,
    /// Holdout prediction accuracy contribution.
    pub backtest_score: f64,
    /// Smoothness contribution from adjacent-factor continuity.
    pub smoothness_score: f64,
    /// Outlier-sensitivity penalty from leave-one-out deltas.
    pub outlier_penalty: f64,
    /// Selected factor vector.
    pub selected_factors: Vec<f64>,
}

fn selector_ldfs(t: &TriangleView<'_>, selector: &CandidateSelector) -> Vec<f64> {
    match selector {
        CandidateSelector::VolumeWeighted => ldf::volume_weighted(t),
        CandidateSelector::SimpleAverage => ldf::simple_average(t),
        CandidateSelector::Median => ldf::median(t),
        CandidateSelector::ExHiLo => ldf::ex_hi_lo(t),
        CandidateSelector::VolumeWeightedLastN(n) => ldf::volume_weighted_last_n(t, *n),
        CandidateSelector::SimpleAverageLastN(n) => ldf::simple_average_last_n(t, *n),
        CandidateSelector::Manual(values) => values.clone(),
    }
}

fn stability_score(diagnostics: &[LinkRatioDiagnostics]) -> f64 {
    let finite: Vec<f64> = diagnostics
        .iter()
        .map(|diag| diag.coeff_var)
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return 0.0;
    }
    let avg_cv = finite.iter().sum::<f64>() / finite.len() as f64;
    1.0 / (1.0 + avg_cv)
}

fn smoothness_score(ldfs: &[f64]) -> f64 {
    if ldfs.len() <= 1 {
        return 1.0;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for pair in ldfs.windows(2) {
        if pair[0].is_finite() && pair[1].is_finite() {
            sum += (pair[1] - pair[0]).abs();
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }
    1.0 / (1.0 + sum / count as f64)
}

fn outlier_penalty(diagnostics: &[LinkRatioDiagnostics]) -> f64 {
    let mut deltas = Vec::new();
    for diag in diagnostics {
        for loo in &diag.leave_one_out_factors {
            if loo.is_finite() && diag.selected_factor.is_finite() {
                deltas.push((diag.selected_factor - loo).abs());
            }
        }
    }
    if deltas.is_empty() {
        return 0.0;
    }
    deltas.iter().sum::<f64>() / deltas.len() as f64
}

fn backtest_component(score: &BacktestScore) -> f64 {
    if !score.mape.is_finite() {
        return 0.0;
    }
    1.0 / (1.0 + score.mape)
}

fn to_ldf_method(selector: &CandidateSelector) -> crate::deterministic::ldf::LdfSelectionMethod {
    match selector {
        CandidateSelector::VolumeWeighted => {
            crate::deterministic::ldf::LdfSelectionMethod::VolumeWeighted
        }
        CandidateSelector::SimpleAverage => {
            crate::deterministic::ldf::LdfSelectionMethod::SimpleAverage
        }
        CandidateSelector::Median => crate::deterministic::ldf::LdfSelectionMethod::Median,
        CandidateSelector::ExHiLo => crate::deterministic::ldf::LdfSelectionMethod::ExHiLo,
        CandidateSelector::VolumeWeightedLastN(last_n) => {
            crate::deterministic::ldf::LdfSelectionMethod::LastNVolumeWeighted { last_n: *last_n }
        }
        CandidateSelector::SimpleAverageLastN(last_n) => {
            crate::deterministic::ldf::LdfSelectionMethod::LastNSimpleAverage { last_n: *last_n }
        }
        CandidateSelector::Manual(_) => crate::deterministic::ldf::LdfSelectionMethod::Selected,
    }
}

/// Score one selector using stability, holdout accuracy, smoothness, and leave-one-out sensitivity.
pub fn score_selector(
    t: &TriangleView<'_>,
    selector: CandidateSelector,
    tail: f64,
) -> SelectorScore {
    let holdout = crate::preprocessing::selection::backtest::holdout_latest_diagonal(t);
    score_selector_with_holdout(t, selector, tail, &holdout)
}

fn score_selector_with_holdout(
    t: &TriangleView<'_>,
    selector: CandidateSelector,
    tail: f64,
    holdout: &crate::preprocessing::selection::backtest::HoldoutSet,
) -> SelectorScore {
    let ldfs = selector_ldfs(t, &selector);
    let diagnostics = diagnose_all(t, to_ldf_method(&selector));
    let backtest = crate::preprocessing::selection::backtest::fit_predict_score(
        holdout,
        selector.clone(),
        tail,
    );

    let stability = stability_score(&diagnostics);
    let backtest_score = backtest_component(&backtest);
    let smoothness = smoothness_score(&ldfs);
    // Manual factors are fixed by the actuary and have zero leave-one-out sensitivity.
    let outlier_penalty = if matches!(selector, CandidateSelector::Manual(_)) {
        0.0
    } else {
        outlier_penalty(&diagnostics)
    };
    let total_score = stability + backtest_score + smoothness - outlier_penalty;

    SelectorScore {
        selector,
        total_score,
        stability_score: stability,
        backtest_score,
        smoothness_score: smoothness,
        outlier_penalty,
        selected_factors: ldfs,
    }
}

/// Score a family of selector candidates (reusing a single holdout split).
pub fn score_selector_family(
    t: &TriangleView<'_>,
    selectors: &[CandidateSelector],
    tail: f64,
) -> Vec<SelectorScore> {
    let holdout = crate::preprocessing::selection::backtest::holdout_latest_diagonal(t);
    selectors
        .iter()
        .cloned()
        .map(|selector| score_selector_with_holdout(t, selector, tail, &holdout))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn score_selector_returns_component_scores() {
        let tri = triangle_4x4();
        let score = score_selector(&tri.view(), CandidateSelector::VolumeWeighted, 1.0);

        assert_eq!(score.selected_factors.len(), 3);
        assert!(score.stability_score.is_finite());
        assert!(score.backtest_score.is_finite());
        assert!(score.smoothness_score.is_finite());
        assert!(score.outlier_penalty.is_finite());
        assert!(score.total_score.is_finite());
    }

    #[test]
    fn score_selector_family_scores_each_candidate() {
        let tri = triangle_4x4();
        let scores = score_selector_family(
            &tri.view(),
            &[
                CandidateSelector::VolumeWeighted,
                CandidateSelector::SimpleAverage,
                CandidateSelector::ExHiLo,
            ],
            1.0,
        );

        assert_eq!(scores.len(), 3);
        assert!(scores.iter().all(|score| score.total_score.is_finite()));
    }
}
