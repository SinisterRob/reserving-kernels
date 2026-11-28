//! Integration tests for actuarial reserving kernels.
//!
//! Canonical test triangle (4 × 4, cumulative losses):
//!
//! ```text
//! i\j   0      1      2      3
//! 0   1000   1500   1700   1800
//! 1   1200   1800   2000     ·
//! 2   1500   2100     ·      ·
//! 3   1800     ·      ·      ·
//! ```

use reserving_kernels::{
    deterministic::bf, deterministic::brosius, deterministic::cape_cod,
    deterministic::cape_cod::CapeCodEngine, deterministic::chain_ladder, deterministic::ldf,
    deterministic::ldf::DevelopmentFactorEngine, preprocessing::triangle, stochastic::mack,
    types::Triangle, validate::ValidationError,
};

const NAN: f64 = f64::NAN;

fn test_triangle() -> Triangle {
    #[rustfmt::skip]
    let data = vec![
        1000.0, 1500.0, 1700.0, 1800.0,
        1200.0, 1800.0, 2000.0,    NAN,
        1500.0, 2100.0,    NAN,    NAN,
        1800.0,    NAN,    NAN,    NAN,
    ];
    Triangle::try_new(4, 4, data).expect("valid triangle")
}

fn close(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps * (1.0 + b.abs())
}

// ---- Triangle helpers ------------------------------------------------------

#[test]
fn latest_diagonal_is_anti_diagonal() {
    let t = test_triangle();
    let d = triangle::latest_diagonal(&t.view());
    assert_eq!(d, vec![1800.0, 2000.0, 2100.0, 1800.0]);
}

#[test]
fn cum_incr_roundtrip() {
    let t = test_triangle();
    let incr = triangle::cumulative_to_incremental(&t.view());
    let back = triangle::incremental_to_cumulative(&reserving_kernels::types::TriangleView {
        n_origins: 4,
        n_devs: 4,
        data: &incr,
    });
    for (a, b) in t.data().iter().zip(back.iter()) {
        if a.is_finite() {
            assert!(close(*a, *b, 1e-12));
        }
    }
}

// ---- LDF methods -----------------------------------------------------------

#[test]
fn volume_weighted_ldfs() {
    let t = test_triangle();
    let f = ldf::volume_weighted(&t.view());
    assert!(close(f[0], 5400.0 / 3700.0, 1e-12));
    assert!(close(f[1], 3700.0 / 3300.0, 1e-12));
    assert!(close(f[2], 1800.0 / 1700.0, 1e-12));
}

#[test]
fn simple_average_ldfs() {
    let t = test_triangle();
    let f = ldf::simple_average(&t.view());
    assert!(close(f[0], (1.5 + 1.5 + 1.4) / 3.0, 1e-12));
    assert!(close(
        f[1],
        f64::midpoint(1700.0 / 1500.0, 2000.0 / 1800.0),
        1e-12
    ));
    assert!(close(f[2], 1800.0 / 1700.0, 1e-12));
}

#[test]
fn ex_hi_lo_drops_max_and_min() {
    let t = test_triangle();
    let f = ldf::ex_hi_lo(&t.view());
    // Three ratios at link 0→1: 1.5, 1.5, 1.4. Drop 1.5 (max) and 1.4 (min),
    // leaving the other 1.5 → mean 1.5.
    assert!(close(f[0], 1.5, 1e-12));
    // Fewer than 3 ratios at links 1→2 and 2→3 → fallback to simple avg.
    assert!(close(
        f[1],
        f64::midpoint(1700.0 / 1500.0, 2000.0 / 1800.0),
        1e-12
    ));
}

#[test]
fn volume_weighted_last_n_restricts_origins() {
    let t = test_triangle();
    // last_n = 2 at link 0→1: origins 1 and 2 only → (1800+2100)/(1200+1500).
    let f = ldf::volume_weighted_last_n(&t.view(), 2);
    assert!(close(f[0], 3900.0 / 2700.0, 1e-12));
}

#[test]
fn cdf_from_ldf_compounds_right_to_left() {
    let ldf = [1.459459459459459_f64, 1.121212121212121, 1.058823529411765];
    let cdf = ldf::cdf_from_ldf(&ldf, 1.0);
    assert!(close(cdf[3], 1.0, 1e-12));
    assert!(close(cdf[2], 1.058823529411765, 1e-12));
    assert!(close(cdf[1], 1.121212121212121 * 1.058823529411765, 1e-12));
    assert!(close(
        cdf[0],
        1.459459459459459 * 1.121212121212121 * 1.058823529411765,
        1e-12
    ));
}

// ---- Chain ladder ----------------------------------------------------------

#[test]
fn chain_ladder_projects_latest_times_cdf() {
    let t = test_triangle();
    let f = ldf::volume_weighted(&t.view());
    let cl = chain_ladder::project_unchecked(&t.view(), &f, 1.0);
    // Origin 0 is fully developed.
    assert!(close(cl.ultimates[0], 1800.0, 1e-9));
    // Origin 1: 2000 × f[2].
    assert!(close(cl.ultimates[1], 2000.0 * 1800.0 / 1700.0, 1e-9));
    // Origin 2: 2100 × f[1] × f[2].
    assert!(close(
        cl.ultimates[2],
        2100.0 * (3700.0 / 3300.0) * (1800.0 / 1700.0),
        1e-9
    ));
    // Origin 3: 1800 × f[0] × f[1] × f[2].
    assert!(close(
        cl.ultimates[3],
        1800.0 * (5400.0 / 3700.0) * (3700.0 / 3300.0) * (1800.0 / 1700.0),
        1e-9
    ));
    // Reserve = ultimate − latest.
    assert!(close(cl.reserves[0], 0.0, 1e-9));
    assert!(close(cl.reserves[1], cl.ultimates[1] - 2000.0, 1e-9));
}

// ---- BF --------------------------------------------------------------------

#[test]
fn bf_reduces_to_chain_ladder_when_expected_equals_cl() {
    let t = test_triangle();
    let f = ldf::volume_weighted(&t.view());
    let cl = chain_ladder::project_unchecked(&t.view(), &f, 1.0);
    // Feed CL ultimates as the BF expected.
    let r = bf::project_unchecked(&t.view(), &f, 1.0, &cl.ultimates);
    for i in 0..4 {
        assert!(close(r.ultimates[i], cl.ultimates[i], 1e-9));
        assert!(close(r.reserves[i], cl.reserves[i], 1e-9));
    }
}

#[test]
fn bf_reserve_zero_when_fully_reported() {
    let t = test_triangle();
    let f = ldf::volume_weighted(&t.view());
    let expected = vec![5000.0, 5000.0, 5000.0, 5000.0];
    let r = bf::project_unchecked(&t.view(), &f, 1.0, &expected);
    // Origin 0 is fully reported (pct_reported = 1) → reserve = 0.
    assert!(close(r.reserves[0], 0.0, 1e-9));
    // Youngest origin has small pct_reported → reserve close to expected.
    assert!(r.reserves[3] > r.reserves[1]);
}

// ---- Cape Cod -------------------------------------------------------------

#[test]
fn cape_cod_elr_is_total_paid_over_used_premium() {
    let t = test_triangle();
    let f = ldf::volume_weighted(&t.view());
    let premium = vec![4000.0, 4500.0, 5000.0, 5500.0];
    let e = cape_cod::elr(&t.view(), &f, 1.0, &premium);
    assert!(e.is_finite());
    // Latest paid: 1800 + 2000 + 2100 + 1800 = 7700.
    // pct_reported at each origin's latest age (ages 3,2,1,0):
    //   p3 = 1 (tail=1, already at ultimate)
    //   p2 = 1 / 1.058823 ≈ 0.9444
    //   p1 = 1 / (1.121212 × 1.058823) ≈ 0.8424
    //   p0 = 1 / (1.459459 × 1.121212 × 1.058823) ≈ 0.5772
    // Used premium ≈ 4000×1 + 4500×0.9444 + 5000×0.8424 + 5500×0.5772
    //              ≈ 4000 + 4250 + 4212 + 3175 ≈ 15637
    // ELR ≈ 7700 / 15637 ≈ 0.4925
    assert!(e > 0.4 && e < 0.6);
}

#[test]
fn cape_cod_engine_returns_used_premium_exhibit() {
    let t = test_triangle();
    let factors = DevelopmentFactorEngine::volume_weighted(&t.view(), 1.0).expect("factors");
    let premium = vec![4000.0, 4500.0, 5000.0, 5500.0];

    let projected = CapeCodEngine::project(&t.view(), &factors, &premium).expect("cape cod");

    assert!(close(projected.total_latest, 7700.0, 1e-12));
    assert_eq!(projected.latest_dev, vec![3, 2, 1, 0]);
    assert!(close(projected.reported_percentage[0], 1.0, 1e-12));
    assert!(close(projected.used_premium[0], 4000.0, 1e-12));
    assert!(close(
        projected.elr,
        projected.total_latest / projected.total_used_premium,
        1e-12
    ));
    for (i, &premium_value) in premium.iter().enumerate() {
        assert!(close(
            projected.expected_ultimate[i],
            premium_value * projected.elr,
            1e-12
        ));
    }
    assert!(close(
        projected.bf_projection.total_expected_ultimate,
        projected.expected_ultimate.iter().sum(),
        1e-12
    ));
}

#[test]
fn cape_cod_engine_rejects_zero_premium() {
    let t = test_triangle();
    let factors = DevelopmentFactorEngine::volume_weighted(&t.view(), 1.0).expect("factors");
    let premium = vec![4000.0, 4500.0, 0.0, 5500.0];

    assert_eq!(
        CapeCodEngine::project(&t.view(), &factors, &premium),
        Err(ValidationError::NonPositive("premium"))
    );
}

// ---- Brosius --------------------------------------------------------------

#[test]
fn brosius_fits_ols_per_link() {
    let t = test_triangle();
    let r = brosius::project_unchecked(&t.view(), 1.0);
    // Link 0→1: x=[1000,1200,1500], y=[1500,1800,2100].
    // mean_x = 1233.333, mean_y = 1800
    // var_x  = 42222.22, cov = 50000 → b ≈ 1.18421, a ≈ 339.47.
    assert!(close(r.slope[0], 1.1842105263157894, 1e-6));
    assert!(close(r.intercept[0], 339.4736842105263, 1e-6));
    // Fitted values reproduce the column sum.
    let fitted_sum: f64 = (0..3)
        .map(|i| {
            let x = [1000.0, 1200.0, 1500.0][i];
            r.intercept[0] + r.slope[0] * x
        })
        .sum();
    assert!(close(fitted_sum, 5400.0, 1e-6));
}

// ---- Mack -----------------------------------------------------------------

#[test]
fn mack_ldfs_match_volume_weighted() {
    let t = test_triangle();
    let m = mack::analyze(&t.view(), 1.0).expect("mack");
    let vw = ldf::volume_weighted(&t.view());
    for (a, b) in m.ldf.iter().zip(vw.iter()) {
        assert!(close(*a, *b, 1e-12));
    }
}

#[test]
fn mack_standard_errors_are_non_negative_and_finite() {
    let t = test_triangle();
    let m = mack::analyze(&t.view(), 1.0).expect("mack");
    assert!(m.total_standard_error.is_finite());
    assert!(m.total_standard_error >= 0.0);
    for se in &m.standard_error {
        assert!(se.is_finite());
        assert!(*se >= 0.0);
    }
    // Origin 0 is fully developed → its SE is zero.
    assert!(close(m.standard_error[0], 0.0, 1e-12));
    // Youngest origin should have the largest SE.
    assert!(m.standard_error[3] >= m.standard_error[1]);
    assert!(m.standard_error[3] >= m.standard_error[2]);
}

#[test]
fn mack_checked_rejects_non_cumulative_triangle_under_strict_policy() {
    #[rustfmt::skip]
    let t = Triangle::try_new(4, 4, vec![
        1000.0, 1500.0, 1700.0, 1800.0,
        1200.0, 1100.0, 2000.0,    NAN,
        1500.0, 2100.0,    NAN,    NAN,
        1800.0,    NAN,    NAN,    NAN,
    ])
.expect("valid triangle");

    let result = mack::analyze_with_policy(
        &t.view(),
        1.0,
        reserving_kernels::validate::MonotonicityPolicy::Strict,
    );
    assert!(matches!(
        result,
        Err(ValidationError::NonMonotoneCumulative {
            origin: 1,
            dev: 1,
            previous: 1200.0,
            value: 1100.0,
        })
    ));
}

#[test]
fn mack_and_mw_accept_raa_by_default_and_reject_2x2() {
    let raa = reserving_kernels::datasets::raa();
    let m = raa
        .mack(1.0)
        .expect("RAA should succeed with default Mack policy");
    assert!(m.total_standard_error > 0.0);
    let cdr = raa
        .merz_wuthrich()
        .expect("RAA should succeed with default MW policy");
    assert!(cdr.total_cdr_standard_error > 0.0);
    assert!(
        raa.merz_wuthrich_with_policy(reserving_kernels::validate::MonotonicityPolicy::Strict)
            .is_err()
    );

    let tiny = Triangle::try_from_upper_rows(&[&[100.0, 150.0], &[120.0]]).expect("2x2");
    assert!(matches!(
        tiny.mack(1.0),
        Err(ValidationError::InvalidParameter(_))
    ));
    let unchecked = mack::analyze_unchecked(&tiny.view(), 1.0);
    assert!(unchecked.sigma_sq[0].is_nan());
}

#[test]
fn panjer_rejects_defective_pmf_and_bootstrap_rejects_zero_iterations() {
    use reserving_kernels::capital::aggregate_loss::{FrequencyDistribution, panjer_recursion};
    use reserving_kernels::stochastic::bootstrap_odp::{OdpBootstrapOptions, bootstrap_odp};

    let freq = FrequencyDistribution::Poisson { lambda: 3.0 };
    assert!(matches!(
        panjer_recursion(freq, &[0.0, 0.5, 0.3], 100.0, 64),
        Err(ValidationError::InvalidParameter(_))
    ));

    let t = test_triangle();
    let opts = OdpBootstrapOptions {
        iterations: 0,
        ..OdpBootstrapOptions::default()
    };
    assert_eq!(
        bootstrap_odp(&t.view(), 1.0, opts),
        Err(ValidationError::InvalidIterations)
    );
}

#[test]
fn mack_cv_is_bounded() {
    let t = test_triangle();
    let m = mack::analyze(&t.view(), 1.0).expect("mack");
    // Each CV finite; total CV positive and reasonably bounded.
    assert!(m.total_cv.is_finite());
    assert!(m.total_cv >= 0.0);
}

// ---- Full-square sanity ---------------------------------------------------

#[test]
fn larger_triangle_chain_ladder_consistency() {
    // 6 × 6 stylised triangle; CL ultimate ≥ latest for every origin,
    // strict for unfinished origins.
    #[rustfmt::skip]
    let data = vec![
        100.0, 160.0, 190.0, 205.0, 210.0, 212.0,
        120.0, 200.0, 240.0, 260.0, 265.0,   NAN,
        150.0, 250.0, 300.0, 325.0,   NAN,   NAN,
        180.0, 310.0, 370.0,   NAN,   NAN,   NAN,
        200.0, 350.0,   NAN,   NAN,   NAN,   NAN,
        220.0,   NAN,   NAN,   NAN,   NAN,   NAN,
    ];
    let t = Triangle::try_new(6, 6, data).expect("valid triangle");
    let f = ldf::volume_weighted(&t.view());
    let cl = chain_ladder::project_unchecked(&t.view(), &f, 1.0);
    let latest = triangle::latest_diagonal(&t.view());
    for (i, &latest_value) in latest.iter().enumerate() {
        assert!(cl.ultimates[i] + 1e-9 >= latest_value);
        if i > 0 {
            assert!(cl.ultimates[i] > latest_value);
        }
    }
}

// ---- Ergonomics & Sprint Regression Tests ---------------------------------

#[test]
fn ergonomic_triangle_constructors_and_fluent_methods() {
    use reserving_kernels::prelude::*;

    let from_upper = Triangle::try_from_upper_rows(&[
        &[100.0, 160.0, 195.0, 205.0],
        &[120.0, 190.0, 232.0],
        &[135.0, 218.0],
        &[150.0],
    ])
    .expect("upper rows");

    let from_inc = Triangle::try_from_incremental_rows(&[
        &[100.0, 60.0, 35.0, 10.0],
        &[120.0, 70.0, 42.0],
        &[135.0, 83.0],
        &[150.0],
    ])
    .expect("incremental rows");

    assert_eq!(from_upper.n(), 4);
    assert_eq!(from_upper.latest_diagonal(), from_inc.latest_diagonal());
    assert!(from_upper.to_string().contains("205.00"));
    assert!(from_upper.to_string().contains('-'));

    let cl = from_upper.chain_ladder(1.0).expect("cl");
    let m = from_upper.mack(1.0).expect("mack");
    let cdr = from_upper.merz_wuthrich().expect("mw");
    assert!(close(cl.total_reserve, m.total_reserve, 1e-9));
    assert!(cdr.total_cdr_standard_error <= m.total_standard_error + 1e-9);

    let prem = [400.0, 450.0, 500.0, 550.0];
    let cc = from_upper.cape_cod(&prem, 1.0).expect("cape cod");
    assert!(cc.bf_projection.total_reserve > 0.0);

    // Owned Vec<Vec<f64>> rows, origin_row, Display, fluent options, and compound shorthand.
    let owned_rows = vec![
        vec![100.0, 160.0, 195.0, 205.0],
        vec![120.0, 190.0, 232.0],
        vec![135.0, 218.0],
        vec![150.0],
    ];
    let from_vecs = Triangle::try_from_rows(&owned_rows).expect("from vecs");
    assert_eq!(from_vecs.origin_row(1), &[120.0, 190.0, 232.0]);
    assert!(m.to_string().contains("Mack("));
    assert!(cdr.to_string().contains("MerzWuthrich1Yr("));

    let glm = from_vecs.glm(&GlmOptions::odp()).expect("glm odp");
    assert!(close(glm.total_reserve, cl.total_reserve, 1e-6));

    let sev_pmf = SeverityDistribution::exponential(100.0)
        .discretize_mean_preserving(25.0, 16)
        .expect("disc");
    let agg = FrequencyDistribution::poisson(2.0)
        .compound_fft(&sev_pmf, 25.0, 64, None)
        .expect("fft");
    assert!(agg.mean > 0.0);

    // Serde JSON round-trip on core options and results.
    let mack_json = serde_json::to_string(&m).expect("serialize mack");
    let mack_rt: Mack = serde_json::from_str(&mack_json).expect("deserialize mack");
    assert!(close(mack_rt.total_reserve, m.total_reserve, 1e-12));
    let glm_opts = GlmOptions::tweedie(1.5).with_design(GlmDesign::Hoerl {
        per_origin_intercepts: true,
    });
    let opts_rt: GlmOptions =
        serde_json::from_str(&serde_json::to_string(&glm_opts).expect("ser")).expect("de");
    assert_eq!(opts_rt, glm_opts);
}

#[test]
fn regression_var_least_and_myers_read_exact_solver() {
    use reserving_kernels::capital::capital_allocation::myers_read_from_default_ratio;
    use reserving_kernels::capital::reserve_distribution::var_least;

    // Least quantile F^-(0.5) on 1..=10 is 5.0 (F(5)=0.5 >= 0.5), not 6.0.
    let xs: Vec<f64> = (1..=10).map(f64::from).collect();
    assert_eq!(var_least(&xs, 0.50).expect("var"), 5.0);

    // IEEE-754 boundary check: 0.28 * 100 and 0.29 * 100 must yield 28.0 and 29.0.
    let xs100: Vec<f64> = (1..=100).map(f64::from).collect();
    assert_eq!(var_least(&xs100, 0.28).expect("p28"), 28.0);
    assert_eq!(var_least(&xs100, 0.29).expect("p29"), 29.0);

    // Power-of-2 sample with tied maximum and tiny target default ratio must succeed.
    let line_a: Vec<f64> = (0..16)
        .map(|i| {
            if i >= 14 {
                500.0
            } else {
                10.0 * (f64::from(i) + 1.0)
            }
        })
        .collect();
    let line_b: Vec<f64> = (0..16)
        .map(|i| {
            if i >= 14 {
                500.0
            } else {
                5.0 * (f64::from(i) + 1.0)
            }
        })
        .collect();
    let mr = myers_read_from_default_ratio(&[&line_a, &line_b], 1e-6).expect("myers read");
    let sum_q: f64 = mr.allocated_surplus.iter().sum();
    assert!(close(sum_q, mr.portfolio_surplus, 1e-9));
}

#[test]
fn regression_severity_overflow_cancellation_and_excess_layers() {
    use reserving_kernels::excess::excess_layers::{LognormalLayerModel, LognormalLayerSampler};
    use reserving_kernels::excess::severity::SeverityDistribution;

    // Pareto PDF at realistic dollar scale must not overflow theta^alpha to NaN.
    let pareto = SeverityDistribution::Pareto {
        alpha: 62.0,
        theta: 100_000.0,
    };
    let pdf = pareto.pdf(5_000.0);
    assert!(pdf.is_finite() && pdf > 0.0);

    // GPD near xi -> 0 must match Exponential smoothly without cancellation.
    let gpd = SeverityDistribution::GeneralizedPareto {
        xi: 1e-11,
        beta: 50_000.0,
    };
    let exp = SeverityDistribution::Exponential { mean: 50_000.0 };
    assert!(
        (gpd.limited_expected_value(10_000.0).expect("gpd lev")
            - exp.limited_expected_value(10_000.0).expect("exp lev"))
        .abs()
            < 1e-6
    );

    // NaN must propagate through Lognormal survival.
    let ln = SeverityDistribution::Lognormal {
        mu: 10.0,
        sigma: 1.0,
    };
    assert!(ln.survival(f64::NAN).is_nan());

    // Excess layer with top <= attachment must return 0.0, never negative.
    let model = LognormalLayerModel::try_new(100_000.0, 1.0).expect("model");
    let sampler = LognormalLayerSampler::try_new(model, 4_096, 2026).expect("sampler");
    assert_eq!(model.exact_layer(100_000.0, 50_000.0), 0.0);
    assert_eq!(sampler.simulated_layer(100_000.0, 50_000.0), 0.0);
}

#[test]
fn regression_mack_zero_cell_and_mack_bf_unbiased_process_variance() {
    use reserving_kernels::stochastic::mack_bf::{self, BfPayoutEstimator, MackBfOptions};
    use reserving_kernels::validate::ValidationError;

    // Mack must return DivideByZero when c_k == 0 && c_{k+1} > 0 instead of masking NaN SE to 0.
    let zero_tri = Triangle::try_from_upper_rows(&[&[0.0, 50.0, 100.0], &[100.0, 150.0], &[120.0]])
        .expect("tri");
    assert!(matches!(
        zero_tri.mack(1.0),
        Err(ValidationError::DivideByZero(_))
    ));

    // Noiseless proportional triangle with 1.2x biased prior must still yield s_sq == 0.0.
    let clean = Triangle::try_from_upper_rows(&[
        &[40.0, 75.0, 95.0, 100.0],
        &[80.0, 150.0, 190.0],
        &[120.0, 225.0],
        &[160.0],
    ])
    .expect("clean");
    let biased_prior = [120.0, 240.0, 360.0, 480.0];
    let prior_se = [10.0, 20.0, 30.0, 40.0];
    let bf_res = mack_bf::analyze_from_triangle(
        &clean.view(),
        &biased_prior,
        &prior_se,
        MackBfOptions {
            payout_estimator: BfPayoutEstimator::DirectIncremental,
            ..Default::default()
        },
    )
    .expect("mack bf");
    assert!(bf_res.process_sigma_sq.iter().all(|&s| s.abs() < 1e-12));
}

#[test]
fn regression_ldf_tail_onlevel_dependence_and_diagnostics() {
    use reserving_kernels::capital::aggregate_loss::{FrequencyDistribution, panjer_recursion};
    use reserving_kernels::capital::dependence::spearman_correlation;
    use reserving_kernels::deterministic::ldf::{DevelopmentFactorEngine, ex_hi_lo};
    use reserving_kernels::deterministic::tail;
    use reserving_kernels::diagnostics::{calendar_effect, triangle_diagnostics};
    use reserving_kernels::preprocessing::onlevel::parallelogram_onlevel_factor;
    use reserving_kernels::types::TriangleView;
    use reserving_kernels::validate::ValidationError;

    // Malformed view must return ShapeMismatch, not panic OOB.
    let bad = TriangleView {
        n_origins: 4,
        n_devs: 4,
        data: &[1.0, 2.0],
    };
    assert_eq!(
        DevelopmentFactorEngine::simple_average(&bad, 1.0),
        Err(ValidationError::ShapeMismatch)
    );

    // ex_hi_lo must neutralize an extreme 1e16 outlier without catastrophic cancellation.
    let outlier_tri = Triangle::try_from_upper_rows(&[
        &[1e-14, 100.0, 110.0, 115.0],
        &[100.0, 150.0, 165.0],
        &[100.0, 150.0],
        &[100.0],
    ])
    .expect("outlier");
    let trimmed = ex_hi_lo(&outlier_tri.view());
    assert!((trimmed[0] - 1.5).abs() < 1e-12);

    // Non-decaying LDFs must be rejected by generalized_bondy instead of returning +inf.
    assert!(tail::generalized_bondy(&[1.1, 1.2, 1.3], 0, 3).is_err());

    // Parallelogram on-leveling includes the final rate regime.
    let ol = parallelogram_onlevel_factor(&[0.10], &[0.5, 0.5]).expect("onlevel");
    assert!(close(ol, 1.10 / 1.05, 1e-12));

    // Spearman correlation propagates NaN.
    assert!(spearman_correlation(&[1.0, 2.0, f64::NAN], &[1.0, 2.0, 3.0]).is_nan());

    // Truncated aggregate distribution with total_mass <= alpha must reject TVaR.
    let dist = panjer_recursion(
        FrequencyDistribution::Poisson { lambda: 10.0 },
        &[0.0, 1.0],
        1.0,
        5,
    )
    .expect("panjer");
    assert!(dist.tvar(0.95).is_err());

    // Empty triangle calendar_effect & rectangular negative_increment_flags must not panic.
    let empty = TriangleView {
        n_origins: 0,
        n_devs: 0,
        data: &[],
    };
    let cal = calendar_effect::diagnose(&empty, &[]);
    assert!(cal.strongest_calendar.is_none());

    let rect_data = vec![100.0; 15];
    let rect = TriangleView {
        n_origins: 3,
        n_devs: 5,
        data: &rect_data,
    };
    assert_eq!(
        triangle_diagnostics::negative_increment_flags(&rect),
        Vec::new()
    );
}
