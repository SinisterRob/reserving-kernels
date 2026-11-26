//! Typed actuarial exhibit DTOs over the reserving kernels.
//!
//! This module reshapes checked kernel outputs into stable, serializable
//! structures for tables, heatmaps, and method-comparison charts. It does not
//! reimplement reserving math.

use serde::Serialize;

use crate::bf::{BornhuetterFergusonEngine, BornhuetterFergusonProjection};
use crate::brosius::{self, Brosius};
use crate::cape_cod::{CapeCodEngine, CapeCodProjection};
use crate::chain_ladder::ChainLadderEngine;
use crate::ldf::{DevelopmentFactorEngine, DevelopmentFactors};
use crate::mack::{self, Mack};
use crate::triangle::cumulative_to_incremental;
use crate::types::TriangleView;
use crate::validate::{ValidationError, validate_monotone_cumulative, validate_triangle};

/// Full exhibit payload for a visualization workbench.
#[derive(Debug, Clone, Serialize)]
pub struct ExhibitBundle {
    /// Cumulative triangle heatmap/table exhibit.
    pub triangle: TriangleExhibit,
    /// Incremental triangle heatmap/table exhibit.
    pub incremental_triangle: TriangleExhibit,
    /// Selected factor calculation trail.
    pub factors: FactorExhibit,
    /// Per-origin method comparison table.
    pub projections: ProjectionComparisonExhibit,
    /// Validation and governance messages for display.
    pub validation: ValidationExhibit,
    /// Mack uncertainty exhibit, when Mack calculation succeeds.
    pub mack: Option<MackExhibit>,
    /// Brosius regression exhibit, when Brosius calculation succeeds.
    pub brosius: Option<BrosiusExhibit>,
}

/// Validation status and display messages.
#[derive(Debug, Clone, Serialize)]
pub struct ValidationExhibit {
    /// Whether the exhibit builder accepted the inputs.
    pub ok: bool,
    /// Validation errors suitable for display.
    pub errors: Vec<String>,
    /// Non-blocking governance warnings.
    pub warnings: Vec<String>,
}

/// One triangle cell for heatmaps and dense tables.
#[derive(Debug, Clone, Serialize)]
pub struct TriangleCellDto {
    /// Origin row index.
    pub origin: usize,
    /// Development column index.
    pub dev: usize,
    /// Cell value; `None` represents unobserved/missing values.
    pub value: Option<f64>,
    /// Whether the cell is scheduled observed under square right-triangle geometry.
    pub observed: bool,
    /// Whether the cell is on the latest diagonal.
    pub latest: bool,
    /// Whether this value is a projection rather than observed data.
    pub projected: bool,
}

/// Triangle display payload.
#[derive(Debug, Clone, Serialize)]
pub struct TriangleExhibit {
    /// Origin row count.
    pub n_origins: usize,
    /// Development column count.
    pub n_devs: usize,
    /// Row-major display cells.
    pub cells: Vec<TriangleCellDto>,
}

/// One factor row tying factor selection, CDF, maturity, and credibility.
#[derive(Debug, Clone, Serialize)]
pub struct FactorExhibitRow {
    /// Development age `j` for the `j -> j + 1` link.
    pub dev: usize,
    /// Sum of observed `C[i, j + 1]`.
    pub numerator: f64,
    /// Sum of observed `C[i, j]`.
    pub denominator: f64,
    /// Selected LDF.
    pub selected_ldf: f64,
    /// CDF at development age `j`.
    pub cdf: f64,
    /// Percent mature/reported at development age `j`.
    pub maturity: f64,
    /// Number of rows included in the link ratio.
    pub included_rows: usize,
}

/// Factor exhibit payload.
#[derive(Debug, Clone, Serialize)]
pub struct FactorExhibit {
    /// Factor rows by development age.
    pub rows: Vec<FactorExhibitRow>,
    /// Tail factor used in the CDFs.
    pub tail: f64,
}

/// One origin row comparing reserving methods.
#[derive(Debug, Clone, Serialize)]
pub struct ProjectionRow {
    /// Origin row index.
    pub origin: usize,
    /// Latest observed development index.
    pub latest_dev: usize,
    /// Latest observed cumulative value.
    pub latest: f64,
    /// Percent mature/reported at latest observed development.
    pub maturity: Option<f64>,
    /// Chain-ladder ultimate.
    pub chain_ladder_ultimate: Option<f64>,
    /// Chain-ladder reserve.
    pub chain_ladder_reserve: Option<f64>,
    /// BF expected ultimate.
    pub bf_expected: Option<f64>,
    /// BF ultimate.
    pub bf_ultimate: Option<f64>,
    /// BF reserve.
    pub bf_reserve: Option<f64>,
    /// Cape Cod expected ultimate.
    pub cape_cod_expected: Option<f64>,
    /// Cape Cod ultimate.
    pub cape_cod_ultimate: Option<f64>,
    /// Cape Cod reserve.
    pub cape_cod_reserve: Option<f64>,
    /// Mack ultimate.
    pub mack_ultimate: Option<f64>,
    /// Mack reserve.
    pub mack_reserve: Option<f64>,
    /// Mack standard error.
    pub mack_se: Option<f64>,
    /// Mack coefficient of variation.
    pub mack_cv: Option<f64>,
    /// Brosius ultimate.
    pub brosius_ultimate: Option<f64>,
    /// Brosius reserve.
    pub brosius_reserve: Option<f64>,
}

/// Projection totals across methods.
#[derive(Debug, Clone, Serialize)]
pub struct ProjectionTotals {
    /// Total latest observed loss.
    pub latest: f64,
    /// Chain-ladder reserve total.
    pub chain_ladder_reserve: Option<f64>,
    /// BF reserve total.
    pub bf_reserve: Option<f64>,
    /// Cape Cod reserve total.
    pub cape_cod_reserve: Option<f64>,
    /// Mack reserve total.
    pub mack_reserve: Option<f64>,
    /// Brosius reserve total.
    pub brosius_reserve: Option<f64>,
}

/// Method comparison display payload.
#[derive(Debug, Clone, Serialize)]
pub struct ProjectionComparisonExhibit {
    /// Per-origin method rows.
    pub rows: Vec<ProjectionRow>,
    /// Portfolio totals.
    pub totals: ProjectionTotals,
}

/// Mack uncertainty display payload.
#[derive(Debug, Clone, Serialize)]
pub struct MackExhibit {
    /// Origin rows with reserve uncertainty.
    pub rows: Vec<MackOriginRiskRow>,
    /// Sigma squared by development link.
    pub sigma_rows: Vec<MackSigmaRow>,
    /// Total reserve.
    pub total_reserve: f64,
    /// Total standard error.
    pub total_standard_error: f64,
    /// Total coefficient of variation.
    pub total_cv: f64,
    /// Normal-approximation portfolio quantiles.
    pub quantiles: MackQuantiles,
    /// Tail uncertainty disclosure for display.
    pub tail_disclosure: String,
}

/// Per-origin Mack uncertainty row.
#[derive(Debug, Clone, Serialize)]
pub struct MackOriginRiskRow {
    /// Origin row index.
    pub origin: usize,
    /// Mack reserve.
    pub reserve: f64,
    /// Mack standard error.
    pub standard_error: f64,
    /// Mack coefficient of variation.
    pub cv: f64,
    /// Reserve minus one standard error.
    pub reserve_minus_1se: f64,
    /// Reserve plus one standard error.
    pub reserve_plus_1se: f64,
}

/// Mack sigma-squared row by development link.
#[derive(Debug, Clone, Serialize)]
pub struct MackSigmaRow {
    /// Development link start index.
    pub dev: usize,
    /// Mack sigma squared.
    pub sigma_sq: f64,
}

/// Normal-approximation Mack reserve quantiles.
#[derive(Debug, Clone, Serialize)]
pub struct MackQuantiles {
    /// 50th percentile.
    pub p50: f64,
    /// 75th percentile.
    pub p75: f64,
    /// 90th percentile.
    pub p90: f64,
    /// 95th percentile.
    pub p95: f64,
}

/// Brosius regression display payload.
#[derive(Debug, Clone, Serialize)]
pub struct BrosiusExhibit {
    /// Link fit diagnostics.
    pub links: Vec<BrosiusLinkExhibitRow>,
    /// Scatterplot source points by link.
    pub scatter_points: Vec<BrosiusScatterPoint>,
}

/// One Brosius link fit row.
#[derive(Debug, Clone, Serialize)]
pub struct BrosiusLinkExhibitRow {
    /// Development link start index.
    pub dev: usize,
    /// Number of observed pairs used.
    pub n: usize,
    /// Fitted intercept.
    pub intercept: f64,
    /// Fitted slope.
    pub slope: f64,
    /// Chain-ladder LDF for comparison.
    pub chain_ladder_ldf: Option<f64>,
    /// Whether the single-pair fallback was used.
    pub fallback_used: bool,
}

/// One observed point for a Brosius scatter/fitted-line chart.
#[derive(Debug, Clone, Serialize)]
pub struct BrosiusScatterPoint {
    /// Development link start index.
    pub dev: usize,
    /// Origin row index.
    pub origin: usize,
    /// Prior cumulative value.
    pub x: f64,
    /// Next cumulative value.
    pub y: f64,
    /// Fitted next value.
    pub fitted_y: f64,
}

/// Build a full exhibit bundle from a triangle and optional method inputs.
///
/// `premium` enables Cape Cod. `expected_ultimate` enables BF. Chain ladder,
/// Mack, and Brosius are always attempted after shared validation succeeds.
pub fn build_exhibit_bundle(
    triangle: &TriangleView<'_>,
    premium: Option<&[f64]>,
    expected_ultimate: Option<&[f64]>,
    tail: f64,
) -> Result<ExhibitBundle, ValidationError> {
    validate_triangle(triangle)?;
    validate_monotone_cumulative(triangle)?;

    let factors = DevelopmentFactorEngine::volume_weighted(triangle, tail)?;
    let factor_diagnostics = DevelopmentFactorEngine::volume_weighted_diagnostics(triangle, tail)?;
    let chain_ladder = ChainLadderEngine::project(triangle, &factors)?;
    let bf = match expected_ultimate {
        Some(expected) => Some(BornhuetterFergusonEngine::project_from_expected_ultimate(
            triangle, &factors, expected,
        )?),
        None => None,
    };
    let cape_cod = match premium {
        Some(premium) => Some(CapeCodEngine::project(triangle, &factors, premium)?),
        None => None,
    };
    let mack = mack::analyze(triangle, tail)?;
    let brosius = brosius::project(triangle, tail)?;

    let triangle_exhibit = triangle_exhibit(triangle);
    let incremental_triangle = incremental_triangle_exhibit(triangle);
    let factor_exhibit = factor_exhibit(&factors, &factor_diagnostics);
    let projections = projection_comparison_exhibit(
        &chain_ladder,
        bf.as_ref(),
        cape_cod.as_ref(),
        &mack,
        &brosius,
    );
    let mack_exhibit = mack_exhibit(&mack, tail);
    let brosius_exhibit = brosius_exhibit(triangle, &factors, &brosius);

    let mut warnings = Vec::new();
    if tail > 1.0 {
        warnings.push(
            "Tail factor exceeds 1.0; Mack standard errors do not include tail uncertainty."
                .to_string(),
        );
    }
    if premium.is_none() {
        warnings.push("Cape Cod exhibit omitted because premium was not supplied.".to_string());
    }
    if expected_ultimate.is_none() {
        warnings.push("BF exhibit omitted because expected ultimate was not supplied.".to_string());
    }

    Ok(ExhibitBundle {
        triangle: triangle_exhibit,
        incremental_triangle,
        factors: factor_exhibit,
        projections,
        validation: ValidationExhibit {
            ok: true,
            errors: Vec::new(),
            warnings,
        },
        mack: Some(mack_exhibit),
        brosius: Some(brosius_exhibit),
    })
}

fn triangle_exhibit(triangle: &TriangleView<'_>) -> TriangleExhibit {
    triangle_exhibit_from_data(triangle.n_origins, triangle.n_devs, triangle.data, false)
}

fn incremental_triangle_exhibit(triangle: &TriangleView<'_>) -> TriangleExhibit {
    let incremental = cumulative_to_incremental(triangle);
    triangle_exhibit_from_data(triangle.n_origins, triangle.n_devs, &incremental, false)
}

fn triangle_exhibit_from_data(
    n_origins: usize,
    n_devs: usize,
    data: &[f64],
    projected: bool,
) -> TriangleExhibit {
    let mut cells = Vec::with_capacity(n_origins * n_devs);
    for origin in 0..n_origins {
        for dev in 0..n_devs {
            let observed = origin + dev <= n_devs.saturating_sub(1);
            let latest = observed && origin + dev == n_devs.saturating_sub(1);
            let raw = data[origin * n_devs + dev];
            cells.push(TriangleCellDto {
                origin,
                dev,
                value: raw.is_finite().then_some(raw),
                observed,
                latest,
                projected,
            });
        }
    }

    TriangleExhibit {
        n_origins,
        n_devs,
        cells,
    }
}

fn factor_exhibit(
    factors: &DevelopmentFactors,
    diagnostics: &[crate::ldf::LinkRatioDiagnostic],
) -> FactorExhibit {
    let maturity = factors.maturity_percentages();
    let rows = diagnostics
        .iter()
        .map(|diagnostic| FactorExhibitRow {
            dev: diagnostic.dev,
            numerator: diagnostic.numerator,
            denominator: diagnostic.denominator,
            selected_ldf: diagnostic.selected_ldf,
            cdf: factors.cdfs[diagnostic.dev],
            maturity: maturity[diagnostic.dev],
            included_rows: diagnostic.included_rows,
        })
        .collect();

    FactorExhibit {
        rows,
        tail: factors.tail,
    }
}

fn projection_comparison_exhibit(
    chain_ladder: &crate::chain_ladder::ChainLadderProjection,
    bf: Option<&BornhuetterFergusonProjection>,
    cape_cod: Option<&CapeCodProjection>,
    mack: &Mack,
    brosius: &Brosius,
) -> ProjectionComparisonExhibit {
    let rows = chain_ladder
        .latest
        .iter()
        .enumerate()
        .map(|(origin, &latest)| ProjectionRow {
            origin,
            latest_dev: chain_ladder.latest_dev[origin],
            latest,
            maturity: bf
                .map(|projection| projection.maturity[origin])
                .or_else(|| cape_cod.map(|projection| projection.reported_percentage[origin])),
            chain_ladder_ultimate: Some(chain_ladder.ultimates[origin]),
            chain_ladder_reserve: Some(chain_ladder.reserves[origin]),
            bf_expected: bf.map(|projection| projection.expected_ultimate[origin]),
            bf_ultimate: bf.map(|projection| projection.ultimates[origin]),
            bf_reserve: bf.map(|projection| projection.reserves[origin]),
            cape_cod_expected: cape_cod.map(|projection| projection.expected_ultimate[origin]),
            cape_cod_ultimate: cape_cod
                .map(|projection| projection.bf_projection.ultimates[origin]),
            cape_cod_reserve: cape_cod.map(|projection| projection.bf_projection.reserves[origin]),
            mack_ultimate: Some(mack.ultimate[origin]),
            mack_reserve: Some(mack.reserve[origin]),
            mack_se: Some(mack.standard_error[origin]),
            mack_cv: Some(mack.cv[origin]),
            brosius_ultimate: Some(brosius.ultimate[origin]),
            brosius_reserve: Some(brosius.reserve[origin]),
        })
        .collect();

    ProjectionComparisonExhibit {
        rows,
        totals: ProjectionTotals {
            latest: chain_ladder.total_latest,
            chain_ladder_reserve: Some(chain_ladder.total_reserve),
            bf_reserve: bf.map(|projection| projection.total_reserve),
            cape_cod_reserve: cape_cod.map(|projection| projection.bf_projection.total_reserve),
            mack_reserve: Some(mack.total_reserve),
            brosius_reserve: Some(brosius.reserve.iter().sum()),
        },
    }
}

fn mack_exhibit(mack: &Mack, tail: f64) -> MackExhibit {
    let rows = mack
        .reserve
        .iter()
        .enumerate()
        .map(|(origin, &reserve)| {
            let standard_error = mack.standard_error[origin];
            MackOriginRiskRow {
                origin,
                reserve,
                standard_error,
                cv: mack.cv[origin],
                reserve_minus_1se: reserve - standard_error,
                reserve_plus_1se: reserve + standard_error,
            }
        })
        .collect();
    let sigma_rows = mack
        .sigma_sq
        .iter()
        .enumerate()
        .map(|(dev, &sigma_sq)| MackSigmaRow { dev, sigma_sq })
        .collect();

    MackExhibit {
        rows,
        sigma_rows,
        total_reserve: mack.total_reserve,
        total_standard_error: mack.total_standard_error,
        total_cv: mack.total_cv,
        quantiles: MackQuantiles {
            p50: mack.total_reserve,
            p75: mack.total_reserve + 0.674 * mack.total_standard_error,
            p90: mack.total_reserve + 1.282 * mack.total_standard_error,
            p95: mack.total_reserve + 1.645 * mack.total_standard_error,
        },
        tail_disclosure: if tail > 1.0 {
            "Mack normal approximation; tail uncertainty is not modeled.".to_string()
        } else {
            "Mack normal approximation.".to_string()
        },
    }
}

fn brosius_exhibit(
    triangle: &TriangleView<'_>,
    factors: &DevelopmentFactors,
    brosius: &Brosius,
) -> BrosiusExhibit {
    let links = brosius
        .diagnostics
        .iter()
        .map(|diagnostic| BrosiusLinkExhibitRow {
            dev: diagnostic.dev,
            n: diagnostic.n,
            intercept: diagnostic.intercept,
            slope: diagnostic.slope,
            chain_ladder_ldf: factors.ldfs.get(diagnostic.dev).copied(),
            fallback_used: diagnostic.fallback_used,
        })
        .collect();

    let mut scatter_points = Vec::new();
    for diagnostic in &brosius.diagnostics {
        for origin in 0..diagnostic.n {
            let x = triangle.get(origin, diagnostic.dev);
            let y = triangle.get(origin, diagnostic.dev + 1);
            scatter_points.push(BrosiusScatterPoint {
                dev: diagnostic.dev,
                origin,
                x,
                y,
                fitted_y: diagnostic.intercept + diagnostic.slope * x,
            });
        }
    }

    BrosiusExhibit {
        links,
        scatter_points,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Triangle;

    fn sample_triangle() -> Triangle {
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
    fn bundle_contains_core_exhibits() {
        let triangle = sample_triangle();
        let premium = [4000.0, 4500.0, 5000.0, 5500.0];
        let expected = [2500.0, 2750.0, 3000.0, 3250.0];

        let bundle = build_exhibit_bundle(&triangle.view(), Some(&premium), Some(&expected), 1.0)
            .expect("bundle");

        assert!(bundle.validation.ok);
        assert_eq!(bundle.triangle.cells.len(), 16);
        assert_eq!(bundle.incremental_triangle.cells.len(), 16);
        assert_eq!(bundle.factors.rows.len(), 3);
        assert_eq!(bundle.projections.rows.len(), 4);
        assert!(bundle.projections.totals.chain_ladder_reserve.is_some());
        assert!(bundle.projections.totals.bf_reserve.is_some());
        assert!(bundle.projections.totals.cape_cod_reserve.is_some());
        assert!(bundle.mack.is_some());
        assert!(bundle.brosius.is_some());
    }

    #[test]
    fn triangle_exhibit_marks_latest_and_unobserved_cells() {
        let triangle = sample_triangle();

        let exhibit = triangle_exhibit(&triangle.view());

        let latest_cells = exhibit.cells.iter().filter(|cell| cell.latest).count();
        let unobserved_cells = exhibit.cells.iter().filter(|cell| !cell.observed).count();
        assert_eq!(latest_cells, 4);
        assert_eq!(unobserved_cells, 6);
        assert_eq!(exhibit.cells[7].value, None);
    }

    #[test]
    fn bundle_allows_optional_bf_and_cape_cod_inputs() {
        let triangle = sample_triangle();

        let bundle = build_exhibit_bundle(&triangle.view(), None, None, 1.0).expect("bundle");

        assert!(
            bundle
                .projections
                .rows
                .iter()
                .all(|row| row.bf_reserve.is_none())
        );
        assert!(
            bundle
                .projections
                .rows
                .iter()
                .all(|row| row.cape_cod_reserve.is_none())
        );
        assert_eq!(bundle.validation.warnings.len(), 2);
    }
}
