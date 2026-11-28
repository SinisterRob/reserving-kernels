//! Source-document benchmark and validation suite.
//!
//! Every test in this file verifies `reserving-kernels` against verbatim
//! published actuarial datasets, closed-form identities, or worked examples
//! from primary literature (see `docs/SOURCES.md` for full citations).

use reserving_kernels::{
    capital::aggregate_loss::{
        FrequencyDistribution, discretize_mean_preserving, fft_aggregate_loss, panjer_recursion,
    },
    capital::capital_allocation::{
        euler_rvar_allocation, euler_sd_allocation, euler_tvar_allocation, myers_read_allocation,
        myers_read_from_default_ratio,
    },
    capital::dependence::{CorrelationType, iman_conover, synchronized_odp_bootstrap},
    capital::reserve_distribution::{rvar, tvar_coherent, var_least},
    deterministic::benktander::BenktanderEngine,
    deterministic::berquist_sherman::{
        CaseReserveTrend, DisposalInterpolation, adjust_case_reserves, adjust_paid_disposal_rate,
    },
    deterministic::bf::BornhuetterFergusonEngine,
    deterministic::cape_cod::CapeCodEngine,
    deterministic::chain_ladder::ChainLadderEngine,
    deterministic::generalized_cape_cod::GeneralizedCapeCodEngine,
    deterministic::ldf::DevelopmentFactorEngine,
    deterministic::munich_chain_ladder,
    deterministic::tail,
    diagnostics::{
        cape_cod_diagnostics, mack_diagnostics,
        mack_tests::{adjacent_factor_correlation_test, calendar_year_effect_test},
        paid_incurred_reconciliation,
    },
    excess::severity::SeverityDistribution,
    stochastic::bootstrap_odp::{
        OdpBootstrapOptions, ProcessDistribution, ResidualAdjustment, bootstrap_odp, fit_odp,
    },
    stochastic::clark_mle::{ClarkOptions, GrowthCurve, fit_clark_cape_cod, fit_clark_ldf},
    stochastic::glm_reserving::{GlmDesign, GlmFamily, GlmOptions, fit_glm},
    stochastic::mack::{self, MackTailSpec},
    stochastic::mack_bf::{self, BfPayoutEstimator, MackBfOptions},
    stochastic::merz_wuthrich,
    stochastic::paid_incurred_chain::{self, PicOptions},
    stochastic::ptf_regression::{self, AccidentSpec, CalendarSpec, DevelopmentSpec, PtfOptions},
    types::Triangle,
    validate::MonotonicityPolicy,
};

const NAN: f64 = f64::NAN;

fn close_rel(a: f64, b: f64, rel_tol: f64) -> bool {
    (a - b).abs() <= rel_tol * (1.0 + b.abs())
}

// ===========================================================================
// Canonical Published Triangles
// ===========================================================================

/// Reinsurance Association of America (RAA, 1991) 10x10 Automatic Facultative
/// General Liability incurred triangle from Mack (1993), ASTIN Bulletin 23(2).
fn raa_triangle() -> Triangle {
    #[rustfmt::skip]
    let data = vec![
        5012.0,  8269.0, 10907.0, 11805.0, 13539.0, 16181.0, 18009.0, 18608.0, 18662.0, 18834.0,
         106.0,  4285.0,  5396.0, 10666.0, 13782.0, 15599.0, 15496.0, 16169.0, 16704.0,     NAN,
        3410.0,  8992.0, 13873.0, 16141.0, 18735.0, 22214.0, 22863.0, 23466.0,     NAN,     NAN,
        5655.0, 11555.0, 15766.0, 21266.0, 23425.0, 26083.0, 27067.0,     NAN,     NAN,     NAN,
        1092.0,  9565.0, 15836.0, 22169.0, 25955.0, 26180.0,     NAN,     NAN,     NAN,     NAN,
        1513.0,  6445.0, 11702.0, 12935.0, 15852.0,     NAN,     NAN,     NAN,     NAN,     NAN,
         557.0,  4020.0, 10946.0, 12314.0,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,
        1351.0,  6947.0, 13112.0,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,
        3133.0,  5395.0,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,
        2063.0,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,     NAN,
    ];
    Triangle::try_new(10, 10, data).expect("valid triangle")
}

/// Taylor & Ashe (1983) / England & Verrall (1999, 2002) `GenIns` 10x10 cumulative paid triangle.
fn genins_triangle() -> Triangle {
    #[rustfmt::skip]
    let data = vec![
        357848.0, 1124788.0, 1735330.0, 2218270.0, 2745596.0, 3319994.0, 3466336.0, 3606286.0, 3833515.0, 3901463.0,
        352118.0, 1236139.0, 2170033.0, 3353322.0, 3799067.0, 4120063.0, 4647867.0, 4914039.0, 5339085.0,       NAN,
        290507.0, 1292306.0, 2218525.0, 3235179.0, 3985995.0, 4132918.0, 4628910.0, 4909315.0,       NAN,       NAN,
        310608.0, 1418858.0, 2195047.0, 3757447.0, 4029929.0, 4381982.0, 4588268.0,       NAN,       NAN,       NAN,
        443160.0, 1136350.0, 2128333.0, 2897821.0, 3402672.0, 3873311.0,       NAN,       NAN,       NAN,       NAN,
        396132.0, 1333217.0, 2180715.0, 2985752.0, 3691712.0,       NAN,       NAN,       NAN,       NAN,       NAN,
        440832.0, 1288463.0, 2419861.0, 3483130.0,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,
        359480.0, 1421128.0, 2864498.0,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,
        376686.0, 1363294.0,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,
        344014.0,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,
    ];
    Triangle::try_new(10, 10, data).expect("valid triangle")
}

/// Friedland (2010), *Estimating Unpaid Claims Using Basic Techniques* (CAS),
/// US Industry Auto 10x10 Paid Claims, Reported Claims, and Earned Premium.
fn friedland_us_industry_auto() -> (Triangle, Triangle, Vec<f64>) {
    #[rustfmt::skip]
    let paid = vec![
        18539254.0, 33231039.0, 40062008.0, 43892039.0, 45896535.0, 46765422.0, 47221322.0, 47446877.0, 47555456.0, 47644187.0,
        20410193.0, 36090684.0, 43259402.0, 47159241.0, 49208532.0, 50162043.0, 50625757.0, 50878808.0, 51000534.0,        NAN,
        22120843.0, 38976014.0, 46389282.0, 50562385.0, 52735280.0, 53740101.0, 54284334.0, 54533225.0,        NAN,        NAN,
        22992259.0, 40096198.0, 47767835.0, 52093916.0, 54363436.0, 55378801.0, 55878421.0,        NAN,        NAN,        NAN,
        24092782.0, 41795313.0, 49903803.0, 54352884.0, 56754376.0, 57807215.0,        NAN,        NAN,        NAN,        NAN,
        24084451.0, 41399612.0, 49070332.0, 53584201.0, 55930654.0,        NAN,        NAN,        NAN,        NAN,        NAN,
        24369770.0, 41489863.0, 49236678.0, 53774672.0,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,
        25100697.0, 42702229.0, 50644994.0,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,
        25608776.0, 43606497.0,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,
        27229969.0,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,
    ];
    #[rustfmt::skip]
    let reported = vec![
        37017487.0, 43169009.0, 45568919.0, 46784558.0, 47337318.0, 47533264.0, 47634419.0, 47689655.0, 47724678.0, 47742304.0,
        38954484.0, 46045718.0, 48882924.0, 50219672.0, 50729292.0, 50926779.0, 51069285.0, 51163540.0, 51185767.0,        NAN,
        41155776.0, 49371478.0, 52358476.0, 53780322.0, 54303086.0, 54582950.0, 54742188.0, 54837929.0,        NAN,        NAN,
        42394069.0, 50584112.0, 53704296.0, 55150118.0, 55895583.0, 56156727.0, 56299562.0,        NAN,        NAN,        NAN,
        44755243.0, 52971643.0, 56102312.0, 57703851.0, 58363564.0, 58592712.0,        NAN,        NAN,        NAN,        NAN,
        45163102.0, 52497731.0, 55468551.0, 57015411.0, 57565344.0,        NAN,        NAN,        NAN,        NAN,        NAN,
        45417309.0, 52640322.0, 55553673.0, 56976657.0,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,
        46360869.0, 53790061.0, 56786410.0,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,
        46582684.0, 54641339.0,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,
        48853563.0,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,        NAN,
    ];
    let earned_premium = vec![
        68574209.0, 68544981.0, 68907977.0, 72544955.0, 79228887.0, 86643542.0, 91763523.0,
        94115312.0, 95272279.0, 95176240.0,
    ];
    (
        Triangle::try_new(10, 10, paid).expect("valid triangle"),
        Triangle::try_new(10, 10, reported).expect("valid triangle"),
        earned_premium,
    )
}

/// Deterministic 12x12 multiplicative ODP golden-test triangle (`Y[i,j] = U[i] * pi[j]`).
fn deterministic_odp_golden_triangle() -> (Triangle, f64) {
    const PI: [f64; 12] = [
        0.30, 0.23, 0.16, 0.11, 0.07, 0.045, 0.030, 0.020, 0.015, 0.010, 0.006, 0.004,
    ];
    let n = 12usize;
    let mut data = vec![NAN; n * n];
    let mut true_reserve = 0.0;
    for i in 0..n {
        let u_i = 1_000_000.0 * 1.04_f64.powi(i as i32);
        let mut cum = 0.0;
        for j in 0..n {
            let inc = u_i * PI[j];
            if i + j < n {
                cum += inc;
                data[i * n + j] = cum;
            } else {
                true_reserve += inc;
            }
        }
    }
    (
        Triangle::try_new(n, n, data).expect("valid triangle"),
        true_reserve,
    )
}

// ===========================================================================
// 1. Mack (1993) ASTIN Bulletin — RAA 10x10 Benchmark
// ===========================================================================

#[test]
fn mack_1993_raa_matches_published_reserves_and_standard_errors() {
    let raa = raa_triangle();
    let m = mack::analyze_with_policy(
        &raa.view(),
        1.0,
        MonotonicityPolicy::AllowNegativeIncrements,
    )
    .expect("mack RAA");

    // Published Mack (1993) Table 2 values (RAA in $000s).
    let expected_reserves = [
        0.0, 154.0, 617.4, 1636.1, 2746.7, 3649.1, 5435.3, 10907.2, 10650.0, 16339.4,
    ];
    let expected_se = [
        0.0, 206.2, 623.4, 747.2, 1469.5, 2001.9, 2209.2, 5357.9, 6333.2, 24566.3,
    ];

    assert!(
        (m.total_reserve - 52_135.23).abs() < 0.05,
        "R={}",
        m.total_reserve
    );
    assert!(
        (m.total_standard_error - 26_909.01).abs() < 0.10,
        "SE={}",
        m.total_standard_error
    );

    for i in 0..10 {
        assert!(
            (m.reserve[i] - expected_reserves[i]).abs() < 0.15,
            "origin {i}: reserve={} expected={}",
            m.reserve[i],
            expected_reserves[i]
        );
        assert!(
            (m.standard_error[i] - expected_se[i]).abs() < 0.15,
            "origin {i}: se={} expected={}",
            m.standard_error[i],
            expected_se[i]
        );
    }

    // Mack (1999) tail uncertainty: adding tail sigma_sq and tail_se strictly increases SE,
    // including for origin 0 which has tail exposure when tail > 1.
    let fixed_tail = mack::analyze_with_policy(
        &raa.view(),
        1.02,
        MonotonicityPolicy::AllowNegativeIncrements,
    )
    .expect("fixed tail");
    let stochastic_tail = mack::analyze_with_tail_uncertainty_and_policy(
        &raa.view(),
        MackTailSpec {
            tail_factor: 1.02,
            tail_sigma_sq: 1.34,
            tail_se: 0.008,
        },
        MonotonicityPolicy::AllowNegativeIncrements,
    )
    .expect("stochastic tail");

    assert!(close_rel(
        fixed_tail.total_standard_error,
        1.02 * m.total_standard_error,
        1e-12
    ));
    assert!(stochastic_tail.standard_error[0] > 0.0);
    assert!(stochastic_tail.total_standard_error > fixed_tail.total_standard_error);

    // Mack (1994) Appendix G & H assumption diagnostics run cleanly on RAA.
    let corr_test = adjacent_factor_correlation_test(&raa.view());
    let cal_test = calendar_year_effect_test(&raa.view());
    assert!(corr_test.statistic_t.is_finite());
    assert!(corr_test.two_sided_p_value >= 0.0 && corr_test.two_sided_p_value <= 1.0);
    assert!(cal_test.statistic_z.is_finite());
    assert!(cal_test.two_sided_p_value >= 0.0 && cal_test.two_sided_p_value <= 1.0);
}

// ===========================================================================
// 2. Taylor & Ashe (1983) / England & Verrall (1999, 2002) / Shapland (2016)
// ===========================================================================

#[test]
fn genins_matches_mack_1993_england_verrall_1999_2002_and_shapland_2016() {
    let genins = genins_triangle();

    // --- Mack (1993) on GenIns ---
    let m = mack::analyze(&genins.view(), 1.0).expect("mack GenIns");
    assert!(
        (m.total_reserve - 18_680_855.61).abs() < 1.0,
        "CL reserve={}",
        m.total_reserve
    );
    assert!(
        (m.total_standard_error - 2_447_094.86).abs() < 1.0,
        "Mack SE={}",
        m.total_standard_error
    );
    // Square observed-count path matches standard square Mack path identically.
    let counts = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    let m_obs = mack::analyze_observed_counts(&genins.view(), &counts, 1.0).expect("mack obs");
    assert!(close_rel(
        m.total_standard_error,
        m_obs.total_standard_error,
        1e-12
    ));
    let m_diag = mack_diagnostics::diagnose(&genins.view(), 1.0).expect("mack diag");
    assert!(close_rel(m_diag.total_variance, m.total_mse, 1e-12));

    // --- ODP GLM Analytic Fit (England & Verrall 1999, Table 2; 2002, Table 2) ---
    let fit = fit_odp(&genins.view(), 1.0, MonotonicityPolicy::Strict).expect("odp fit");
    assert_eq!(fit.n_observed, 55);
    assert_eq!(fit.rank, 19);
    assert_eq!(fit.degrees_of_freedom, 36);
    assert!((fit.total_reserve - 18_680_855.61).abs() < 1.0);
    assert!(
        (fit.dispersion - 52_601.361_5).abs() < 0.05,
        "phi={}",
        fit.dispersion
    );
    assert!(
        (fit.analytic_total_prediction_error - 2_945_646.0).abs() < 5.0,
        "analytic ODP PE={}",
        fit.analytic_total_prediction_error
    );
    let h_sum: f64 = fit.residuals.iter().map(|r| r.leverage).sum();
    assert!((h_sum - 19.0).abs() < 1e-10, "sum(h_ii)={h_sum}");

    // --- England & Verrall (1999, 2002) ODP Bootstrap (DegreesOfFreedom + Gamma) ---
    let ev_boot = bootstrap_odp(
        &genins.view(),
        1.0,
        OdpBootstrapOptions::england_verrall(4_000, 42069),
    )
    .expect("EV bootstrap");
    assert_eq!(ev_boot.failed_iterations, 0);
    assert!(
        (ev_boot.summary.mean - 18_680_856.0).abs() / 18_680_856.0 < 0.02,
        "EV boot mean={}",
        ev_boot.summary.mean
    );
    let ev_cov = ev_boot.summary.std_dev / ev_boot.summary.mean;
    // Published England & Verrall (1999, 2002) prediction error is ~15.8% - 16.5% CoV.
    assert!(
        (0.140..=0.185).contains(&ev_cov),
        "EV boot CoV={ev_cov:.4} (sd={:.0})",
        ev_boot.summary.std_dev
    );

    // --- Shapland (2016, CAS Monograph 4) Hat-Matrix Leverage Bootstrap ---
    let shap_boot = bootstrap_odp(
        &genins.view(),
        1.0,
        OdpBootstrapOptions::shapland(4_000, 42069),
    )
    .expect("Shapland bootstrap");
    assert_eq!(shap_boot.failed_iterations, 0);
    assert!(
        (shap_boot.summary.mean - 18_680_856.0).abs() / 18_680_856.0 < 0.02,
        "Shapland boot mean={}",
        shap_boot.summary.mean
    );
    let shap_cov = shap_boot.summary.std_dev / shap_boot.summary.mean;
    assert!(
        (0.145..=0.195).contains(&shap_cov),
        "Shapland boot CoV={shap_cov:.4}"
    );

    // --- Scaled ODP Process Variance matches Gamma first two moments ---
    let mut odp_proc_opts = OdpBootstrapOptions::england_verrall(2_000, 42069);
    odp_proc_opts.process_distribution = ProcessDistribution::OverDispersedPoisson;
    let odp_proc = bootstrap_odp(&genins.view(), 1.0, odp_proc_opts).expect("ODP proc boot");
    assert!((odp_proc.summary.mean - ev_boot.summary.mean).abs() / ev_boot.summary.mean < 0.03);

    // --- Parameter-only bootstrap has strictly lower SD than full predictive bootstrap ---
    let mut param_only_opts = OdpBootstrapOptions::england_verrall(2_000, 42069);
    param_only_opts.process_distribution = ProcessDistribution::None;
    let param_only = bootstrap_odp(&genins.view(), 1.0, param_only_opts).expect("param-only");
    assert!(param_only.summary.std_dev < ev_boot.summary.std_dev);
}

// ===========================================================================
// 3. Deterministic ODP Golden-Test Fixture Exact Mean Recovery
// ===========================================================================

#[test]
fn odp_golden_deterministic_recovers_exact_means_and_zero_dispersion() {
    let (tri, true_reserve) = deterministic_odp_golden_triangle();
    assert!((true_reserve - 2_912_230.612_368_711_3).abs() < 1e-6);

    let fit = fit_odp(&tri.view(), 1.0, MonotonicityPolicy::Strict).expect("golden fit");
    assert_eq!(fit.n_observed, 78);
    assert_eq!(fit.rank, 23);
    assert_eq!(fit.degrees_of_freedom, 55);
    assert!((fit.total_reserve - true_reserve).abs() < 1e-6);
    assert!(fit.dispersion < 1e-20, "dispersion={}", fit.dispersion);
    assert!(fit.analytic_total_prediction_error < 1e-8);

    let boot = bootstrap_odp(
        &tri.view(),
        1.0,
        OdpBootstrapOptions {
            iterations: 50,
            seed: 42069,
            residual_adjustment: ResidualAdjustment::HatMatrixLeverage,
            process_distribution: ProcessDistribution::Gamma,
            center_residuals: true,
            monotonicity_policy: MonotonicityPolicy::Strict,
        },
    )
    .expect("golden boot");
    assert!((boot.summary.mean - true_reserve).abs() < 1e-6);
    assert!(boot.summary.std_dev < 1e-6);
}

// ===========================================================================
// 4. Friedland (2010) US Industry Auto — CL, BF, Benktander, Cape Cod, Recon
// ===========================================================================

#[test]
fn friedland_2010_us_industry_auto_cross_method_consistency() {
    let (paid, reported, premium) = friedland_us_industry_auto();
    let paid_factors = DevelopmentFactorEngine::volume_weighted(&paid.view(), 1.0).expect("paid");
    let rep_factors =
        DevelopmentFactorEngine::volume_weighted(&reported.view(), 1.0).expect("reported");

    let cl_paid = ChainLadderEngine::project(&paid.view(), &paid_factors).expect("CL paid");
    let cl_rep = ChainLadderEngine::project(&reported.view(), &rep_factors).expect("CL rep");

    // On stable US Industry Auto, paid and reported CL ultimates agree closely (< 1% apart).
    let paid_rep_rel_diff =
        (cl_paid.total_ultimate - cl_rep.total_ultimate).abs() / cl_rep.total_ultimate;
    assert!(
        paid_rep_rel_diff < 0.01,
        "paid_ult={} rep_ult={}",
        cl_paid.total_ultimate,
        cl_rep.total_ultimate
    );

    // Cape Cod ELR ~ 68% - 75% for US Industry Auto 1998-2007.
    let cc = CapeCodEngine::project(&reported.view(), &rep_factors, &premium).expect("Cape Cod");
    assert!(cc.elr > 0.60 && cc.elr < 0.80, "Cape Cod ELR={}", cc.elr);
    let cc_diag =
        cape_cod_diagnostics::diagnose(&reported.view(), &rep_factors, &premium).expect("cc diag");
    assert!(cc_diag.stability_score > 0.90);

    // BF and Benktander using Cape Cod ELR as a priori: Benktander (m=2) lies
    // strictly between BF (m=1) and CL for every immature origin.
    let bf = BornhuetterFergusonEngine::project_from_scalar_elr(
        &reported.view(),
        &rep_factors,
        &premium,
        cc.elr,
    )
    .expect("BF");
    let bk2 = BenktanderEngine::project_from_scalar_elr(
        &reported.view(),
        &rep_factors,
        &premium,
        cc.elr,
        2,
    )
    .expect("Benktander");

    for i in 1..10 {
        let lo = bf.ultimates[i].min(cl_rep.ultimates[i]);
        let hi = bf.ultimates[i].max(cl_rep.ultimates[i]);
        assert!(
            bk2.ultimates[i] >= lo - 1e-6 && bk2.ultimates[i] <= hi + 1e-6,
            "origin {i}: bk2={} not in [{lo}, {hi}]",
            bk2.ultimates[i]
        );
    }

    // Paid-incurred reconciliation confirms positive case reserves across all origins.
    let recon = paid_incurred_reconciliation::reconcile_volume_weighted(
        &paid.view(),
        &reported.view(),
        1.0,
    )
    .expect("recon");
    assert!(recon.case_reserve_latest.iter().all(|&c| c > 0.0));
}

// ===========================================================================
// 5. Quarg & Mack (2004) Munich Chain Ladder + Sherman (1984) Tail Curves
// ===========================================================================

#[test]
fn quarg_mack_2004_and_sherman_1984_tail_benchmarks() {
    // Published Quarg & Mack (2004) 7×7 regression slopes and gap reduction.
    let qm = reserving_kernels::datasets::quarg_mack_2004();
    let qm_mcl =
        munich_chain_ladder::project(&qm.paid.view(), &qm.incurred.view(), 1.0, 1.0).expect("qm");
    assert!((qm_mcl.lambda_paid - 0.636_021).abs() < 1e-4);
    assert!((qm_mcl.lambda_incurred - 0.436_187).abs() < 1e-4);
    let trad_gap =
        (qm_mcl.traditional_paid_ultimates[6] - qm_mcl.traditional_incurred_ultimates[6]).abs();
    let mcl_gap = (qm_mcl.paid_ultimates[6] - qm_mcl.incurred_ultimates[6]).abs();
    assert!(mcl_gap < 0.1 * trad_gap);

    // Smoke check on Friedland US Industry Auto 10×10.
    let (paid, reported, _) = friedland_us_industry_auto();
    let mcl = munich_chain_ladder::project(&paid.view(), &reported.view(), 1.0, 1.0).expect("mcl");
    assert!(mcl.total_paid_reserve > 0.0);
    assert!(mcl.total_incurred_reserve > 0.0);

    // RAA tail curves (Bondy vs Generalized Bondy vs Exponential vs Inverse Power).
    let raa = raa_triangle();
    let raa_factors = DevelopmentFactorEngine::volume_weighted_with_policy(
        &raa.view(),
        1.0,
        MonotonicityPolicy::AllowNegativeIncrements,
    )
    .expect("raa ldfs");
    let bondy = tail::bondy(&raa_factors.ldfs).expect("bondy");
    let gbondy = tail::generalized_bondy(&raa_factors.ldfs, 0, 5).expect("gen bondy");
    let exp_tail = tail::fit_exponential_decay(&raa_factors.ldfs, 0, 100).expect("exp");
    let inv_tail = tail::fit_inverse_power(&raa_factors.ldfs, 0, 100).expect("inv");

    assert!((bondy.tail_factor - 1.009_217).abs() < 1e-5);
    assert!((gbondy.tail_factor - 1.003_14).abs() < 5e-4);
    assert!(exp_tail.tail_factor > 1.0 && inv_tail.tail_factor > exp_tail.tail_factor);
}

// ===========================================================================
// 6. Severity Layering, Mildenhall (2006) Dependence & CAS Monograph 15 Risk Measures
// ===========================================================================

#[test]
fn severity_layering_dependence_and_cas_monograph_15_identities() {
    // Mixed Exponential ISO-style curve: E[X] = sum(w_i * m_i), layer sum conservation.
    let mixed = SeverityDistribution::MixedExponential {
        weights: vec![0.5, 0.35, 0.15],
        means: vec![25_000.0, 150_000.0, 800_000.0],
    };
    let mean = mixed.unbounded_mean().unwrap();
    assert!(close_rel(mean, 185_000.0, 1e-12));
    let l1 = mixed.layer_expected_loss(0.0, 100_000.0).unwrap();
    let l2 = mixed.layer_expected_loss(100_000.0, 400_000.0).unwrap();
    let l3 = mixed.layer_expected_loss(500_000.0, 500_000.0).unwrap();
    let lev_1m = mixed.limited_expected_value(1_000_000.0).unwrap();
    assert!(close_rel(l1 + l2 + l3, lev_1m, 1e-12));
    let xs_1m = mixed.excess_ratio(1_000_000.0).unwrap();
    assert!(close_rel(lev_1m + xs_1m * mean, mean, 1e-12));

    // Mildenhall (2006) Iman-Conover + Kirschner et al. (2002) synchronized ODP bootstrap
    // + CAS Monograph 15 coherent TVaR subadditivity check on portfolio reserves.
    let genins = genins_triangle();
    let (paid, reported, _) = friedland_us_industry_auto();
    let b1 =
        bootstrap_odp(&genins.view(), 1.0, OdpBootstrapOptions::shapland(500, 101)).expect("b1");
    let b2 = bootstrap_odp(&paid.view(), 1.0, OdpBootstrapOptions::shapland(500, 202)).expect("b2");

    let ic = iman_conover(
        &[&b1.simulated_totals, &b2.simulated_totals],
        &[1.0, 0.50, 0.50, 1.0],
        CorrelationType::SpearmanTarget,
        42069,
    )
    .expect("ic");
    assert!((ic.achieved_spearman[1] - 0.50).abs() < 0.05);

    // CAS Monograph 15 Algorithm 3.4.3 subadditivity: TVaR(X + Y) <= TVaR(X) + TVaR(Y).
    let port: Vec<f64> = ic.reordered_columns[0]
        .iter()
        .zip(ic.reordered_columns[1].iter())
        .map(|(a, b)| a + b)
        .collect();
    let tvar_p = tvar_coherent(&port, 0.90).unwrap();
    let tvar_1 = tvar_coherent(&ic.reordered_columns[0], 0.90).unwrap();
    let tvar_2 = tvar_coherent(&ic.reordered_columns[1], 0.90).unwrap();
    assert!(tvar_p <= tvar_1 + tvar_2 + 1e-6);
    assert!(tvar_p >= var_least(&port, 0.90).unwrap() - 1e-6);
    assert!(rvar(&port, 0.85, 0.95).unwrap() <= tvar_p + 1e-6);

    // Synchronized multi-line bootstrap across aligned paid & reported triangles.
    let sync = synchronized_odp_bootstrap(
        &[paid.view(), reported.view()],
        &[1.0, 1.0],
        OdpBootstrapOptions::shapland(200, 42069),
    )
    .expect("sync");
    assert_eq!(sync.failed_iterations, 0);
    assert!(sync.aggregate_summary.mean > 0.0);
}

// ===========================================================================
// 7. Berquist & Sherman (1977) & Gluck (1997) Generalized Cape Cod
// ===========================================================================

#[test]
fn berquist_sherman_1977_and_gluck_1997_operational_benchmarks() {
    // --- Gluck (1997) / Friedland (2010) Ch. 10 on US Industry Auto 10x10 ---
    let (_paid, reported, premium) = friedland_us_industry_auto();
    let rep_factors =
        DevelopmentFactorEngine::volume_weighted(&reported.view(), 1.0).expect("rep ldfs");
    let cc = CapeCodEngine::project(&reported.view(), &rep_factors, &premium).expect("cc");
    let cl = ChainLadderEngine::project(&reported.view(), &rep_factors).expect("cl");

    // delta = 1.0 matches standard Stanard-Buhlmann Cape Cod identically.
    let gcc_1 = GeneralizedCapeCodEngine::project(&reported.view(), &rep_factors, &premium, 1.0)
        .expect("gcc delta=1");
    for i in 0..10 {
        assert!(close_rel(gcc_1.elr_by_origin[i], cc.elr, 1e-12));
        assert!(close_rel(
            gcc_1.bf_projection.reserves[i],
            cc.bf_projection.reserves[i],
            1e-12
        ));
    }

    // delta -> 0+ converges to Chain Ladder identically.
    let gcc_0 = GeneralizedCapeCodEngine::project(&reported.view(), &rep_factors, &premium, 1e-12)
        .expect("gcc delta->0");
    assert!(close_rel(
        gcc_0.bf_projection.total_reserve,
        cl.total_reserve,
        1e-9
    ));

    // Intermediate decay (delta = 0.75) yields origin-varying ELRs responsive to recent years.
    let gcc_mid = GeneralizedCapeCodEngine::project(&reported.view(), &rep_factors, &premium, 0.75)
        .expect("gcc delta=0.75");
    assert!(gcc_mid.bf_projection.total_reserve > 0.0);
    assert!(
        (gcc_mid.elr_by_origin[0] - gcc_mid.elr_by_origin[9]).abs() > 1e-4,
        "expected distinct origin ELRs under decay 0.75"
    );
}

#[test]
fn berquist_sherman_1977_operational_benchmark() {
    // --- Berquist & Sherman (1977) / Friedland (2010) Ch. 13 ---
    // Case-reserve strengthening + settlement-speedup benchmark with known true ultimates.
    let n = 6usize;
    let r_sev = 0.06_f64;
    let base_closed = [0.22, 0.50, 0.72, 0.86, 0.95, 1.00];
    let fast_closed = [0.30, 0.58, 0.78, 0.90, 0.97, 1.00];
    let ult_counts = [500.0, 520.0, 540.0, 560.0, 580.0, 600.0];

    let mut paid_data = vec![NAN; n * n];
    let mut rep_data = vec![NAN; n * n];
    let mut open_data = vec![NAN; n * n];
    let mut closed_data = vec![NAN; n * n];
    let mut true_ult = vec![0.0; n];

    for i in 0..n {
        let infl = (r_sev * i as f64).exp();
        let n_i = ult_counts[i];
        // Convex severity vs disposal rate: larger/more complex claims settle later.
        let paid_curve = |d: f64| n_i * infl * (800.0 * d + 400.0 * d * d);
        true_ult[i] = paid_curve(1.0);
        for j in 0..(n - i) {
            let idx = i * n + j;
            let disp = if i + j == n - 1 {
                fast_closed[j]
            } else {
                base_closed[j]
            };
            let closed = n_i * disp;
            let open = n_i * (1.0 - disp);
            let p = paid_curve(disp);
            // Case adequacy jumps 30% on the latest diagonal.
            let adequacy = if i + j == n - 1 { 1.30 } else { 1.00 };
            let avg_case = 950.0 * infl * adequacy;
            paid_data[idx] = p;
            closed_data[idx] = closed;
            open_data[idx] = open;
            rep_data[idx] = p + open * avg_case;
        }
    }
    let paid_t = Triangle::try_new(n, n, paid_data).expect("valid triangle");
    let rep_t = Triangle::try_new(n, n, rep_data).expect("valid triangle");
    let open_t = Triangle::try_new(n, n, open_data).expect("valid triangle");
    let closed_t = Triangle::try_new(n, n, closed_data).expect("valid triangle");

    let bs_case = adjust_case_reserves(
        &rep_t.view(),
        &paid_t.view(),
        &open_t.view(),
        CaseReserveTrend::ExponentialRate(r_sev),
        1.0,
        MonotonicityPolicy::AllowNegativeIncrements,
    )
    .expect("bs case");
    let bs_paid = adjust_paid_disposal_rate(
        &paid_t.view(),
        &closed_t.view(),
        &ult_counts,
        None,
        DisposalInterpolation::PiecewiseLinear,
        1.0,
    )
    .expect("bs paid");

    let true_total: f64 = true_ult.iter().sum();
    let naive_paid_f = DevelopmentFactorEngine::volume_weighted(&paid_t.view(), 1.0).unwrap();
    let naive_paid_cl = ChainLadderEngine::project(&paid_t.view(), &naive_paid_f).unwrap();
    assert!(
        (bs_paid.total_ultimate - true_total).abs()
            < (naive_paid_cl.total_ultimate - true_total).abs() * 0.25,
        "BS paid adjustment should remove settlement-speedup bias: bs={} naive={} true={}",
        bs_paid.total_ultimate,
        naive_paid_cl.total_ultimate,
        true_total
    );

    let naive_rep_f = DevelopmentFactorEngine::volume_weighted(&rep_t.view(), 1.0).unwrap();
    let naive_rep_cl = ChainLadderEngine::project(&rep_t.view(), &naive_rep_f).unwrap();
    assert!(
        (bs_case.total_ultimate - true_total).abs()
            < (naive_rep_cl.total_ultimate - true_total).abs() * 0.35,
        "BS case adjustment should remove case-strengthening bias: bs={} naive={} true={}",
        bs_case.total_ultimate,
        naive_rep_cl.total_ultimate,
        true_total
    );
}

// ===========================================================================
// 8. Clark (2003) GenIns MLE + Mack (2008) BF MSEP
// ===========================================================================

#[test]
fn clark_2003_genins_and_mack_2008_bf_msep_benchmarks() {
    let genins = genins_triangle();

    // --- Clark (2003) LDF on GenIns 10x10 (Loglogistic & Weibull) ---
    let clark_ll = fit_clark_ldf(&genins.view(), &ClarkOptions::new(GrowthCurve::Loglogistic))
        .expect("Clark Loglogistic");
    let clark_wb = fit_clark_ldf(&genins.view(), &ClarkOptions::new(GrowthCurve::Weibull))
        .expect("Clark Weibull");

    // Verify MLE stationarity, degrees of freedom (55 - 12 = 43), and variance identity.
    assert_eq!(clark_ll.n_observed, 55);
    assert_eq!(clark_ll.n_params, 12);
    assert_eq!(clark_ll.degrees_of_freedom, 43);
    assert!(
        clark_ll.max_grad_norm < 1e-6,
        "grad={}",
        clark_ll.max_grad_norm
    );
    assert!(
        clark_wb.max_grad_norm < 1e-6,
        "grad={}",
        clark_wb.max_grad_norm
    );

    // Loglogistic carries a heavier tail than Weibull on GenIns, and both exceed CL pre-tail 18.68M.
    assert!(
        clark_ll.total_reserve > clark_wb.total_reserve && clark_wb.total_reserve > 18_680_000.0,
        "LL={} WB={}",
        clark_ll.total_reserve,
        clark_wb.total_reserve
    );
    assert!(close_rel(
        clark_ll.total_se_aggregate.powi(2),
        clark_ll.total_process_se.powi(2) + clark_ll.total_parameter_se.powi(2),
        1e-12
    ));

    // Scale invariance: multiplying ages by 12 (years -> months) leaves reserves & SEs invariant.
    let mut month_opts = ClarkOptions::new(GrowthCurve::Weibull);
    month_opts.ages = Some((0..10).map(|j| 6.0 + 12.0 * f64::from(j)).collect());
    let clark_wb_months = fit_clark_ldf(&genins.view(), &month_opts).expect("Clark months");
    assert!(close_rel(
        clark_wb.total_reserve,
        clark_wb_months.total_reserve,
        1e-6
    ));
    assert!(close_rel(
        clark_wb.total_se_aggregate,
        clark_wb_months.total_se_aggregate,
        1e-6
    ));
    assert!(close_rel(
        clark_wb.theta * 12.0,
        clark_wb_months.theta,
        1e-5
    ));

    // Clark (2003) Cape Cod on GenIns with constant exposure has 3 parameters (dof = 52)
    // and strictly tighter parameter SE than 12-parameter LDF.
    let unit_prem = vec![10_000_000.0; 10];
    let clark_cc = fit_clark_cape_cod(
        &genins.view(),
        &unit_prem,
        &ClarkOptions::new(GrowthCurve::Weibull),
    )
    .expect("Clark Cape Cod");
    assert_eq!(clark_cc.n_params, 3);
    assert_eq!(clark_cc.degrees_of_freedom, 52);
    assert!(clark_cc.total_parameter_se < clark_wb.total_parameter_se);
}

#[test]
fn mack_2008_bf_msep_benchmark() {
    // --- Mack (2008) Analytical BF MSEP on RAA 10x10 ---
    let raa = raa_triangle();
    let m_cl = mack::analyze_with_policy(
        &raa.view(),
        1.0,
        MonotonicityPolicy::AllowNegativeIncrements,
    )
    .expect("mack RAA");
    let prior_ult = vec![20_000.0; 10];
    let prior_se = vec![2_000.0; 10]; // 10% a priori uncertainty
    let m_bf = mack_bf::analyze_from_triangle(
        &raa.view(),
        &prior_ult,
        &prior_se,
        MackBfOptions {
            payout_estimator: BfPayoutEstimator::ChainLadderImplied,
            prior_correlation: 0.0,
            monotonicity_policy: MonotonicityPolicy::AllowNegativeIncrements,
        },
    )
    .expect("mack BF RAA");

    // On RAA's youngest origin (origin 9), Mack (1993) CL SE = 24,566.3 explodes,
    // while Mack (2008) BF prediction error is dramatically smaller.
    assert!(
        m_bf.standard_error[9] < m_cl.standard_error[9] * 0.5,
        "BF SE[9]={} vs CL SE[9]={}",
        m_bf.standard_error[9],
        m_cl.standard_error[9]
    );
    assert!(close_rel(
        m_bf.total_msep,
        m_bf.total_process_variance + m_bf.total_estimation_variance,
        1e-12
    ));
}

// ===========================================================================
// 9. Merz & Wuthrich (2008) 1-Year CDR & Merz & Wuthrich (2010) PIC
// ===========================================================================

#[test]
fn merz_wuthrich_2008_cdr_and_2010_paid_incurred_chain_benchmarks() {
    // --- Merz & Wuthrich (2008) on GenIns & RAA 10x10 ---
    let genins = genins_triangle();
    let mw_genins = merz_wuthrich::analyze(&genins.view()).expect("MW GenIns");
    let decomp_genins =
        merz_wuthrich::runoff_decomposition(&genins.view(), MonotonicityPolicy::Strict)
            .expect("MW decomp GenIns");

    assert!(close_rel(
        mw_genins.total_ultimate_standard_error,
        2_447_094.86,
        1e-5
    ));
    assert!(
        mw_genins.total_emergence_ratio > 0.50 && mw_genins.total_emergence_ratio < 0.95,
        "emergence={}",
        mw_genins.total_emergence_ratio
    );
    assert!(mw_genins.total_covariance > 0.0);
    // Exact multi-year telescoping identity: sum_{t=1}^{9} msep(CDR_t) == msep_Mack(Ultimate).
    assert!(close_rel(
        decomp_genins.cumulative_cdr_msep,
        decomp_genins.mack_ultimate_msep,
        1e-12
    ));

    let raa = raa_triangle();
    let decomp_raa = merz_wuthrich::runoff_decomposition(
        &raa.view(),
        MonotonicityPolicy::AllowNegativeIncrements,
    )
    .expect("MW decomp RAA");
    assert!(close_rel(
        decomp_raa.cumulative_cdr_msep,
        decomp_raa.mack_ultimate_msep,
        1e-12
    ));
}

#[test]
fn merz_wuthrich_2010_paid_incurred_chain_benchmark() {
    // --- Merz & Wuthrich (2010) Paid-Incurred Chain on Friedland US Industry Auto 10x10 ---
    let (paid, reported, _) = friedland_us_industry_auto();
    let pic = paid_incurred_chain::analyze(&paid.view(), &reported.view(), &PicOptions::default())
        .expect("PIC Friedland");
    let paid_f = DevelopmentFactorEngine::volume_weighted(&paid.view(), 1.0).unwrap();
    let cl_paid = ChainLadderEngine::project(&paid.view(), &paid_f).unwrap();

    // On stable US Auto, PIC total ultimate closely aligns with paid/incurred CL (within 3%)
    // while strictly blending paid and incurred information (0 < w_i < 1 for all immature origins).
    assert!(
        (pic.total_ultimate - cl_paid.total_ultimate).abs() / cl_paid.total_ultimate < 0.03,
        "PIC ult={} vs CL paid ult={}",
        pic.total_ultimate,
        cl_paid.total_ultimate
    );
    for i in 1..10 {
        assert!(pic.incurred_weights[i] > 0.0 && pic.incurred_weights[i] < 1.0);
        assert!(pic.standard_error[i] > 0.0);
    }
    assert!(pic.total_standard_error > 0.0 && pic.total_cv < 0.15);
}

// ===========================================================================
// 10. Barnett & Zehnwirth (2000) PTF & Taylor & McGuire (2016) GLM IRLS
// ===========================================================================

#[test]
fn barnett_zehnwirth_2000_ptf_and_taylor_mcguire_2016_glm_benchmarks() {
    let genins = genins_triangle();

    // --- Barnett & Zehnwirth (2000) PTF on GenIns 10x10 ---
    let ptf_genins = ptf_regression::fit_ptf(
        &genins.view(),
        &PtfOptions {
            accident: AccidentSpec::Constant,
            development: DevelopmentSpec::PerPeriod,
            calendar: CalendarSpec::ConstantRate,
            ..PtfOptions::default()
        },
    )
    .expect("PTF GenIns");
    assert_eq!(ptf_genins.n_observed, 55);
    assert_eq!(ptf_genins.n_params, 11); // intercept + 9 dev increments + 1 calendar rate
    assert!(ptf_genins.r_squared > 0.75, "R2={}", ptf_genins.r_squared);
    assert!(ptf_genins.total_reserve > 12_000_000.0 && ptf_genins.total_reserve < 30_000_000.0);
    assert!(ptf_genins.total_standard_error > 0.0);
}

#[test]
fn taylor_mcguire_2016_glm_irls_benchmark() {
    let genins = genins_triangle();
    // --- Taylor & McGuire (2016, CAS Monograph No. 3) IRLS GLMs on GenIns ---
    // Ch. 3: Cross-classified ODP IRLS reproduces exact fit_odp & Chain Ladder.
    let odp_exact = fit_odp(&genins.view(), 1.0, MonotonicityPolicy::Strict).expect("fit_odp");
    let glm_odp = fit_glm(
        &genins.view(),
        &GlmOptions::new(GlmFamily::Odp, GlmDesign::CrossClassified),
    )
    .expect("GLM ODP");
    assert!(glm_odp.converged);
    assert!(close_rel(glm_odp.total_reserve, 18_680_855.61, 1e-7));
    assert!(close_rel(
        glm_odp.pearson_dispersion,
        odp_exact.dispersion,
        1e-7
    ));
    assert!(close_rel(
        glm_odp.total_prediction_error,
        odp_exact.analytic_total_prediction_error,
        1e-6
    ));
    let h_sum: f64 = glm_odp.leverages.iter().sum();
    assert!((h_sum - 19.0).abs() < 1e-9);

    // Ch. 4–5: Hoerl curve & Hoerl + calendar-year spline across ODP, Tweedie(1.5), Gamma, LogNormal.
    for family in [
        GlmFamily::Odp,
        GlmFamily::Tweedie { power: 1.5 },
        GlmFamily::Gamma,
        GlmFamily::LogNormal,
    ] {
        let hoerl = fit_glm(
            &genins.view(),
            &GlmOptions::new(
                family,
                GlmDesign::Hoerl {
                    per_origin_intercepts: true,
                },
            ),
        )
        .expect("Hoerl GLM");
        assert!(hoerl.converged);
        assert_eq!(hoerl.n_params, 12);
        assert!(hoerl.total_reserve > 12_000_000.0 && hoerl.total_reserve < 30_000_000.0);
        let h_tot: f64 = hoerl.leverages.iter().sum();
        assert!((h_tot - 12.0).abs() < 1e-8);
    }

    let spline_glm = fit_glm(
        &genins.view(),
        &GlmOptions::new(
            GlmFamily::Gamma,
            GlmDesign::HoerlWithCalendarSplines {
                per_origin_intercepts: false,
                calendar_knots: vec![4.0, 7.0],
            },
        ),
    )
    .expect("Hoerl calendar spline GLM");
    assert!(spline_glm.converged);
    assert_eq!(spline_glm.n_params, 6); // intercept + 2 Hoerl + base cal + 2 hinges
    assert!(spline_glm.total_reserve > 0.0 && spline_glm.total_prediction_error > 0.0);
}

// ===========================================================================
// 11. Panjer (1981) / Robertson (1992) & Tasche (2008) / Myers & Read (2001)
// ===========================================================================

#[test]
fn panjer_1981_robertson_1992_and_tasche_2008_myers_read_2001_benchmarks() {
    // --- Klugman, Panjer & Willmot Ch. 9: Compound Poisson & NegBin Panjer vs FFT ---
    let sev_discrete = [0.0, 0.25, 0.35, 0.25, 0.15];
    let step = 50.0;

    for freq in [
        FrequencyDistribution::Poisson { lambda: 4.0 },
        FrequencyDistribution::NegativeBinomial { r: 3.0, beta: 1.5 },
        FrequencyDistribution::Binomial { m: 12, q: 0.30 },
    ] {
        let p = panjer_recursion(freq, &sev_discrete, step, 512).expect("panjer");
        let f = fft_aggregate_loss(freq, &sev_discrete, step, 512, None).expect("fft");

        // Exact theoretical compound moments E[S] = E[N] E[X] and Var(S).
        assert!(close_rel(p.mean, p.exact_theoretical_mean, 1e-10));
        assert!(close_rel(p.variance, p.exact_theoretical_variance, 1e-9));
        // L_inf agreement between Panjer recursion and tilted radix-2 FFT.
        let l_inf = p
            .pmf
            .iter()
            .zip(f.pmf.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(l_inf < 1e-9, "L_inf={l_inf} for {freq:?}");
    }

    // Verify exponential tilting eliminates circular wrap-around aliasing on a tight grid (N = 32).
    let tight_freq = FrequencyDistribution::Poisson { lambda: 10.0 };
    let panjer_tight = panjer_recursion(tight_freq, &sev_discrete, step, 32).expect("p tight");
    let fft_untilted =
        fft_aggregate_loss(tight_freq, &sev_discrete, step, 32, Some(0.0)).expect("u tight");
    let fft_tilted =
        fft_aggregate_loss(tight_freq, &sev_discrete, step, 32, Some(0.4)).expect("t tight");
    let err_untilted = panjer_tight
        .pmf
        .iter()
        .zip(fft_untilted.pmf.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let err_tilted = panjer_tight
        .pmf
        .iter()
        .zip(fft_tilted.pmf.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        err_tilted < err_untilted * 1e-3 && err_tilted < 1e-6,
        "tilted={err_tilted} untilted={err_untilted}"
    );

    // Large-grid O(N log N) tilted FFT on continuous Lognormal severity (N = 2^13 = 8192).
    let sev_dist = SeverityDistribution::Lognormal {
        mu: 8.5,
        sigma: 0.75,
    };
    let grid_step = 500.0;
    let sev_pmf = discretize_mean_preserving(&sev_dist, grid_step, 512).expect("disc");
    let big_freq = FrequencyDistribution::Poisson { lambda: 25.0 };
    let big_agg = fft_aggregate_loss(big_freq, &sev_pmf, grid_step, 8192, None).expect("big fft");
    assert!(close_rel(
        big_agg.mean,
        big_agg.exact_theoretical_mean,
        1e-5
    ));
    assert!(big_agg.tvar(0.95).unwrap() >= big_agg.var(0.95).unwrap());
    assert!(big_agg.stop_loss_premium(big_agg.mean).unwrap() > 0.0);
}

#[test]
fn tasche_2008_euler_and_myers_read_2001_allocation_benchmark() {
    // --- Tasche (2008) Euler & Myers & Read (2001) Multi-Line Portfolio Allocation ---
    let genins = genins_triangle();
    let (paid, reported, _) = friedland_us_industry_auto();
    let b1 = bootstrap_odp(&genins.view(), 1.0, OdpBootstrapOptions::shapland(600, 111)).unwrap();
    let b2 = bootstrap_odp(&paid.view(), 1.0, OdpBootstrapOptions::shapland(600, 222)).unwrap();
    let b3 = bootstrap_odp(
        &reported.view(),
        1.0,
        OdpBootstrapOptions::shapland(600, 333),
    )
    .unwrap();

    // Correlate the 3 lines via Iman-Conover (Mildenhall 2006).
    let corr = [1.0, 0.35, 0.25, 0.35, 1.0, 0.65, 0.25, 0.65, 1.0];
    let ic = iman_conover(
        &[
            &b1.simulated_totals,
            &b2.simulated_totals,
            &b3.simulated_totals,
        ],
        &corr,
        CorrelationType::SpearmanTarget,
        42069,
    )
    .expect("ic 3-line");
    let lines: Vec<&[f64]> = ic
        .reordered_columns
        .iter()
        .map(std::vec::Vec::as_slice)
        .collect();

    // Exact Euler TVaR full-allocation conservation & subadditivity.
    let euler_tvar = euler_tvar_allocation(&lines, 0.90).expect("euler tvar");
    let sum_tvar: f64 = euler_tvar.allocated_total.iter().sum();
    assert!(close_rel(
        sum_tvar,
        euler_tvar.portfolio_risk_measure,
        1e-12
    ));
    assert!(euler_tvar.diversification_ratio < 1.0);
    for i in 0..3 {
        assert!(euler_tvar.allocated_total[i] <= euler_tvar.standalone_risk_measures[i] + 1e-6);
    }

    // Exact Euler RVaR and Standard Deviation conservation.
    let euler_rvar = euler_rvar_allocation(&lines, 0.80, 0.95).expect("euler rvar");
    let sum_rvar: f64 = euler_rvar.allocated_total.iter().sum();
    assert!(close_rel(
        sum_rvar,
        euler_rvar.portfolio_risk_measure,
        1e-12
    ));

    let euler_sd = euler_sd_allocation(&lines, 2.0).expect("euler sd");
    let sum_sd: f64 = euler_sd.allocated_total.iter().sum();
    assert!(close_rel(sum_sd, euler_sd.portfolio_risk_measure, 1e-12));

    // Myers & Read (2001) marginal default-put allocation at 0.5% target default-value ratio.
    let mr = myers_read_from_default_ratio(&lines, 0.005).expect("myers read");
    let mr_surplus_sum: f64 = mr.allocated_surplus.iter().sum();
    let mr_assets_sum: f64 = mr.allocated_assets.iter().sum();
    assert!(close_rel(mr_surplus_sum, mr.portfolio_surplus, 1e-12));
    assert!(close_rel(mr_assets_sum, mr.portfolio_assets, 1e-12));
    assert!((mr.default_value_ratio - 0.005).abs() < 1e-4);

    // Direct asset-level Myers & Read matches target-ratio solution identically.
    let mr_direct = myers_read_allocation(&lines, mr.portfolio_assets).expect("mr direct");
    for i in 0..3 {
        assert!(close_rel(
            mr.allocated_surplus[i],
            mr_direct.allocated_surplus[i],
            1e-12
        ));
    }
}

// ===========================================================================
// 11. Extended Reserving Transformers, Multivariate CL & 1-Year CDR Bootstrap
// ===========================================================================

#[test]
fn extended_reserving_kernels_and_datasets_benchmarks() {
    use reserving_kernels::{
        datasets,
        deterministic::case_outstanding::{CaseOutstandingOptions, project_from_paid_incurred},
        deterministic::incremental_additive::{
            IncrementalAdditiveOptions, project_incremental_additive,
        },
        deterministic::ldf,
        stochastic::bootstrap_odp::bootstrap_odp_cdr,
        stochastic::glm_reserving::select_tweedie_power,
        stochastic::mack::MackOptions,
        stochastic::multivariate_cl::{
            MultivariateClOptions, MultivariateModel, analyze_multivariate_cl,
        },
    };

    // 1. Canonical built-in datasets match literature dimensions.
    let raa = datasets::raa();
    let genins = datasets::genins();
    let qm = datasets::quarg_mack_2004();
    let auto = datasets::friedland_us_industry_auto();
    assert_eq!((raa.n_origins(), raa.n_devs()), (10, 10));
    assert_eq!((genins.n_origins(), genins.n_devs()), (10, 10));
    assert_eq!((qm.paid.n_origins(), qm.paid.n_devs()), (7, 7));
    assert_eq!(auto.earned_premium.len(), 10);

    // 2. Case Outstanding (Friedland Ch. 11) on US Industry Auto & Quarg-Mack.
    let co = project_from_paid_incurred(
        &auto.paid.view(),
        &auto.reported.view(),
        &CaseOutstandingOptions::default(),
    )
    .expect("case outstanding auto");
    assert!(co.total_reserve > 0.0 && co.total_reserve.is_finite());

    let co_qm = project_from_paid_incurred(
        &qm.paid.view(),
        &qm.incurred.view(),
        &CaseOutstandingOptions::default(),
    )
    .expect("case outstanding qm");
    assert!(co_qm.total_reserve > 0.0 && co_qm.total_reserve.is_finite());

    // 3. Incremental Additive on Friedland US Industry Auto with earned premium.
    let ia = project_incremental_additive(
        &auto.paid.view(),
        &auto.earned_premium,
        &IncrementalAdditiveOptions::default(),
    )
    .expect("incremental additive");
    assert!(ia.total_reserve > 0.0 && ia.total_reserve.is_finite());

    // 4. Generalized Mack alpha (0, 1, 2) and cell weight mask on RAA.
    let m1 = mack::analyze_with_policy(
        &raa.view(),
        1.0,
        MonotonicityPolicy::AllowNegativeIncrements,
    )
    .expect("mack alpha=1");
    let m_opt1 = mack::analyze_with_options(
        &raa.view(),
        &MackOptions::default()
            .with_alpha(1.0)
            .with_policy(MonotonicityPolicy::AllowNegativeIncrements),
    )
    .expect("mack opt alpha=1");
    assert!(close_rel(
        m1.total_standard_error,
        m_opt1.total_standard_error,
        1e-12
    ));

    let m_alpha0 = mack::analyze_with_options(
        &raa.view(),
        &MackOptions::default()
            .with_alpha(0.0)
            .with_policy(MonotonicityPolicy::AllowNegativeIncrements),
    )
    .expect("mack alpha=0");
    let m_alpha2 = mack::analyze_with_options(
        &raa.view(),
        &MackOptions::default()
            .with_alpha(2.0)
            .with_policy(MonotonicityPolicy::AllowNegativeIncrements),
    )
    .expect("mack alpha=2");
    assert!(m_alpha0.total_standard_error > 0.0 && m_alpha0.total_standard_error.is_finite());
    assert!(m_alpha2.total_standard_error > 0.0 && m_alpha2.total_standard_error.is_finite());

    // Masking out a single outlier cell (0, 0) changes LDF[0] cleanly.
    let mut mask = vec![1.0; 100];
    mask[0] = 0.0;
    let masked_ldfs = ldf::weighted_volume_ldfs(&genins.view(), &mask).expect("masked ldfs");
    let full_ldfs = ldf::volume_weighted(&genins.view());
    assert!((masked_ldfs[0] - full_ldfs[0]).abs() > 1e-4);
    assert!(close_rel(masked_ldfs[1], full_ldfs[1], 1e-12));

    let geo_ldfs = ldf::geometric_average(&genins.view());
    assert_eq!(geo_ldfs.len(), 9);
    assert!(geo_ldfs.iter().all(|&x| x.is_finite() && x > 0.0));

    // 5. Clark tail -> MackTailSpec integration.
    let clark_spec =
        tail::clark_tail(&genins.view(), GrowthCurve::Loglogistic).expect("clark tail");
    assert!(clark_spec.tail_factor > 1.0 && clark_spec.tail_se >= 0.0);
    let m_clark_tail = mack::analyze_with_tail_uncertainty(&genins.view(), clark_spec)
        .expect("mack with clark tail");
    assert!(
        m_clark_tail.total_standard_error
            > mack::analyze(&genins.view(), 1.0)
                .unwrap()
                .total_standard_error
    );

    // 6. 1-Year CDR Re-Reserving ODP Bootstrap on GenIns:
    //    1-year CDR SE must be strictly less than ultimate ODP bootstrap SE.
    let cdr_boot = bootstrap_odp_cdr(
        &genins.view(),
        1.0,
        OdpBootstrapOptions::shapland(1_000, 42069),
    )
    .expect("cdr bootstrap");
    let ult_boot = bootstrap_odp(
        &genins.view(),
        1.0,
        OdpBootstrapOptions::shapland(1_000, 42069),
    )
    .expect("ult bootstrap");
    assert_eq!(cdr_boot.failed_iterations, 0);
    assert!(
        cdr_boot.summary.std_dev > 0.0 && cdr_boot.summary.std_dev < ult_boot.summary.std_dev,
        "1yr CDR SD={} should be < ultimate SD={}",
        cdr_boot.summary.std_dev,
        ult_boot.summary.std_dev
    );

    // 7. Multivariate Chain Ladder (Diagonal MCL & GMCL) across 2 lines.
    let mcl_2 = analyze_multivariate_cl(
        &[genins.view(), auto.paid.view()],
        &MultivariateClOptions {
            model: MultivariateModel::DiagonalMcl,
            policy: MonotonicityPolicy::Strict,
        },
    )
    .expect("2-line diagonal MCL");
    let m_gen = mack::analyze(&genins.view(), 1.0).unwrap();
    let m_auto = mack::analyze(&auto.paid.view(), 1.0).unwrap();
    assert!(close_rel(
        mcl_2.line_standard_errors[0],
        m_gen.total_standard_error,
        1e-10
    ));
    assert!(close_rel(
        mcl_2.line_standard_errors[1],
        m_auto.total_standard_error,
        1e-10
    ));

    let gmcl_2 = analyze_multivariate_cl(
        &[qm.paid.view(), qm.incurred.view()],
        &MultivariateClOptions {
            model: MultivariateModel::GeneralGmcl,
            policy: MonotonicityPolicy::AllowNegativeIncrements,
        },
    )
    .expect("2-line GMCL");
    assert!(gmcl_2.portfolio_total_reserve.is_finite() && gmcl_2.portfolio_standard_error > 0.0);

    // 8. Tweedie power profile-likelihood selection on GenIns.
    let tw = select_tweedie_power(
        &genins.view(),
        GlmDesign::CrossClassified,
        &[1.0, 1.25, 1.5, 1.75, 2.0],
    )
    .expect("tweedie power");
    assert_eq!(tw.candidates.len(), 5);
    assert!((1.0..=2.0).contains(&tw.best_power));
}

#[test]
fn pure_reserving_extension_kernels_benchmarks() {
    use reserving_kernels::{
        deterministic::brosius, deterministic::credibility_cl, deterministic::double_chain_ladder,
        deterministic::murphy, deterministic::schnieper, deterministic::separation,
        excess::excess_development, excess::severity::SeverityDistribution, stochastic::hertig,
        stochastic::kalman_reserving, stochastic::mack, stochastic::micro_reserving,
        stochastic::multivariate_mack,
    };

    let genins = genins_triangle();
    let raa = raa_triangle();
    let auto = reserving_kernels::datasets::friedland_us_industry_auto();

    // 1. Murphy (1994): exact equivalence to Mack (1993) at delta=1, alpha=0 on GenIns & RAA,
    //    and exact equivalence to Brosius (1993) at delta=0 with intercept.
    let m_gen = mack::analyze(&genins.view(), 1.0).unwrap();
    let mu_gen = murphy::analyze(&genins.view(), murphy::MurphyOptions::mack_equivalent()).unwrap();
    assert!(close_rel(mu_gen.total_reserve, m_gen.total_reserve, 1e-12));
    assert!(close_rel(
        mu_gen.total_standard_error,
        m_gen.total_standard_error,
        1e-12
    ));

    let br_gen = brosius::project_with_fit_rule(
        &genins.view(),
        1.0,
        MonotonicityPolicy::Strict,
        brosius::BrosiusFitRule::RawOls,
    )
    .unwrap();
    let mu_br =
        murphy::analyze(&genins.view(), murphy::MurphyOptions::brosius_equivalent()).unwrap();
    for i in 0..10 {
        assert!(close_rel(mu_br.ultimate[i], br_gen.ultimate[i], 1e-10));
    }

    // 2. Taylor (1977) Arithmetic & Geometric Separation on exact stepped-inflation triangle.
    let true_r = [0.35, 0.25, 0.20, 0.12, 0.08];
    let true_lam = [1.0, 1.05, 1.12, 1.25, 1.40];
    let exp = [100.0, 110.0, 120.0, 125.0, 130.0];
    let mut sep_rows = Vec::new();
    for i in 0..5 {
        let mut r = Vec::new();
        for j in 0..(5 - i) {
            r.push(exp[i] * true_r[j] * true_lam[i + j]);
        }
        sep_rows.push(r);
    }
    let sep_slices: Vec<&[f64]> = sep_rows.iter().map(std::vec::Vec::as_slice).collect();
    let sep_tri = Triangle::try_from_incremental_rows(&sep_slices).unwrap();
    for method in [
        separation::SeparationMethod::Arithmetic,
        separation::SeparationMethod::Geometric,
    ] {
        let sres = separation::analyze(
            &sep_tri.view(),
            &exp,
            method,
            &separation::InflationProjection::ConstantRate(0.08),
        )
        .unwrap();
        for j in 0..5 {
            assert!((sres.delay_proportions[j] - true_r[j]).abs() < 1e-10);
            assert!((sres.calendar_indices[j] - true_lam[j]).abs() < 1e-9);
        }
    }

    // 3. Gisler & Wuthrich (2008) Credibility CL on GenIns: limits recover Mack & pure prior.
    let cred_mack = credibility_cl::analyze(
        &genins.view(),
        &credibility_cl::CredibilityClPrior {
            prior_ldf: vec![1.5; 9],
            tau_sq: vec![f64::INFINITY; 9],
            sigma_sq_override: None,
            tail: 1.0,
            policy: MonotonicityPolicy::Strict,
        },
    )
    .unwrap();
    assert!(close_rel(
        cred_mack.total_reserve,
        m_gen.total_reserve,
        1e-12
    ));
    assert!(close_rel(
        cred_mack.total_standard_error,
        m_gen.total_standard_error,
        1e-12
    ));

    let cred_shrink = credibility_cl::analyze(
        &genins.view(),
        &credibility_cl::CredibilityClPrior {
            prior_ldf: m_gen.ldf.clone(),
            tau_sq: vec![0.05; 9],
            sigma_sq_override: None,
            tail: 1.0,
            policy: MonotonicityPolicy::Strict,
        },
    )
    .unwrap();
    assert!(cred_shrink.total_standard_error < m_gen.total_standard_error);

    // 4. Braun (2004) / Merz & Wuthrich (2008) Multivariate Mack across GenIns & Friedland Auto Paid.
    let mv_solo = multivariate_mack::analyze(&[genins.view()], 1.0).unwrap();
    assert!(close_rel(mv_solo.total_mse, m_gen.total_mse, 1e-12));

    let mv_pair = multivariate_mack::analyze(&[genins.view(), auto.paid.view()], 1.0).unwrap();
    assert_eq!(mv_pair.n_triangles, 2);
    assert!(mv_pair.total_standard_error <= mv_pair.undiversified_se + 1e-8);
    assert!(mv_pair.reserve_correlation_matrix[1].abs() <= 1.0 + 1e-12);

    // 5. Schnieper (1991) / Liu & Verrall (2009) True IBNR / IBNER separation:
    //    exact reduction to Mack when N_{i,j} = 0 for j >= 1.
    let mut n_data = vec![f64::NAN; 100];
    let mut d_data = vec![f64::NAN; 100];
    for i in 0..10 {
        n_data[i * 10] = genins.get(i, 0);
        d_data[i * 10] = 0.0;
        for j in 1..(10 - i) {
            n_data[i * 10 + j] = 0.0;
            d_data[i * 10 + j] = genins.get(i, j) - genins.get(i, j - 1);
        }
    }
    let sc = schnieper::analyze(
        &Triangle::try_new(10, 10, n_data)
            .expect("valid triangle")
            .view(),
        &Triangle::try_new(10, 10, d_data)
            .expect("valid triangle")
            .view(),
        &[1000.0; 10],
    )
    .unwrap();
    assert!(sc.total_ibnr.abs() < 1e-10);
    assert!(close_rel(sc.total_reserve, m_gen.total_reserve, 1e-10));
    assert!(close_rel(
        sc.total_standard_error,
        m_gen.total_standard_error,
        1e-9
    ));

    // 6. Double Chain Ladder (Martinez-Miranda, Nielsen & Verrall 2012) exact CL reproduction.
    let alpha = [200.0, 220.0, 240.0, 260.0];
    let rep = [0.65, 0.20, 0.10, 0.05];
    let pay = [0.55, 0.25, 0.15, 0.05];
    let sev = [500.0, 520.0, 540.0, 560.0];
    let mut c_rows = Vec::new();
    let mut p_rows = Vec::new();
    for i in 0..4 {
        let mut cr = Vec::new();
        let mut pr = Vec::new();
        for j in 0..(4 - i) {
            cr.push(alpha[i] * rep[j]);
            let mut paid = 0.0;
            for k in 0..=j {
                paid += alpha[i] * rep[k] * pay[j - k];
            }
            pr.push(sev[i] * paid);
        }
        c_rows.push(cr);
        p_rows.push(pr);
    }
    let c_sl: Vec<&[f64]> = c_rows.iter().map(std::vec::Vec::as_slice).collect();
    let p_sl: Vec<&[f64]> = p_rows.iter().map(std::vec::Vec::as_slice).collect();
    let dcl = double_chain_ladder::analyze(
        &Triangle::try_from_incremental_rows(&p_sl).unwrap().view(),
        &Triangle::try_from_incremental_rows(&c_sl).unwrap().view(),
    )
    .unwrap();
    assert!(close_rel(
        dcl.total_in_triangle_reserve,
        dcl.paid_cl_total_reserve,
        1e-9
    ));
    assert!(dcl.total_rbns > 0.0 && dcl.total_ibnr > 0.0);

    // 7. Kalman Reserving (De Jong & Zehnwirth 1983 / Verrall 1989) on GenIns.
    let kf = kalman_reserving::analyze(
        &genins.view(),
        kalman_reserving::KalmanReservingOptions::static_log_linear(),
    )
    .unwrap();
    assert!(kf.total_reserve > 1e7 && kf.total_standard_error > 0.0);

    // 8. Hertig (1985) closed-form log-normal link-ratio model on RAA & GenIns.
    let h_raa = hertig::analyze(&raa.view(), hertig::HertigOptions::default()).unwrap();
    assert!(h_raa.total_reserve > 0.0 && h_raa.total_standard_error > 0.0);
    let q50 = h_raa.aggregate_reserve_quantile(0.50).unwrap();
    let q95 = h_raa.aggregate_reserve_quantile(0.95).unwrap();
    assert!(q95 > q50);

    // 9. Pinto & Gogol (1987) / Siewert (1996) Excess Loss Development conservation identity.
    let gu_cdfs = genins.ldfs(1.0).unwrap().cdfs;
    let dists: Vec<SeverityDistribution> = (0..10)
        .map(|j| SeverityDistribution::Lognormal {
            mu: 9.5 + 0.08 * f64::from(j),
            sigma: 1.15 + 0.02 * f64::from(j),
        })
        .collect();
    let xs_dev = excess_development::from_severity_schedule(
        100_000.0,
        &gu_cdfs,
        &dists,
        dists.last().unwrap(),
    )
    .unwrap();
    let lat_gu = genins.latest_diagonal();
    let lat_lim: Vec<f64> = (0..10)
        .map(|i| lat_gu[i] * (1.0 - xs_dev.excess_ratio[9 - i]))
        .collect();
    let lat_xs: Vec<f64> = (0..10)
        .map(|i| lat_gu[i] * xs_dev.excess_ratio[9 - i])
        .collect();
    let xs_proj = excess_development::project_split_losses(&lat_lim, &lat_xs, &xs_dev).unwrap();
    assert!(xs_proj.max_conservation_error < 1e-6);

    // 10. Antonio & Plat (2014) Micro-Level Reserving synthetic event-log recovery.
    let micro_model = micro_reserving::MicroReservingModel {
        occurrence_intensities: vec![180.0; 4],
        reporting_delay_rate: 2.5,
        interim_payment_hazard: 0.7,
        settlement_hazard: 1.3,
        interim_severity_mu: 6.8,
        interim_severity_sigma: 0.35,
        settlement_payment_prob: 0.85,
        settlement_severity_mu: 7.8,
        settlement_severity_sigma: 0.45,
    };
    let (claims, true_rbns, true_ibnr) =
        micro_reserving::simulate_synthetic_portfolio(&micro_model, 4.0, 2026).unwrap();
    let fitted = micro_reserving::fit_from_claims(&claims, 4.0, 4).unwrap();
    let sim = micro_reserving::simulate_reserves(&claims, &fitted, 4.0, 1_500, 77).unwrap();
    let true_total = true_rbns + true_ibnr;
    assert!((sim.mean_total - true_total).abs() / true_total < 0.20);
}

#[test]
fn england_verrall_wuthrich_2019_one_year_re_reserving_genins_table_2_benchmark() {
    use reserving_kernels::datasets::genins;
    use reserving_kernels::stochastic::bootstrap_odp::OdpBootstrapOptions;
    use reserving_kernels::stochastic::reserve_runoff::{
        MackInnovation, MackRunoffOptions, mack_one_year_runoff, mack_one_year_runoff_with_options,
        odp_one_year_runoff,
    };

    let tri = genins();

    // 1. England, Verrall & Wüthrich (2019), Table 2 on GenIns:
    //    Opening reserve: 18,680,856
    //    1-year CDR RMSEP: 1,778,968 (9.52%)
    //    Lifetime RMSEP:   2,447,095 (13.10%)
    let mack_normal = mack_one_year_runoff_with_options(
        &tri.view(),
        1.0,
        MackRunoffOptions::new(20_000, 20_190_111).with_innovation(MackInnovation::Normal),
    )
    .expect("mack normal runoff");
    assert_eq!(mack_normal.failed_iterations, 0);
    assert!((mack_normal.opening_total_reserve - 18_680_855.61).abs() < 1.0);
    // Verify negligible mean bias alongside SD so simulated SD ≈ simulated RMSE.
    assert!(
        mack_normal.adverse_development_summary.mean.abs() / mack_normal.opening_total_reserve
            < 0.005,
        "Normal Mack 1yr mean bias={}",
        mack_normal.adverse_development_summary.mean
    );
    assert!(
        close_rel(
            mack_normal.adverse_development_summary.std_dev,
            1_778_968.0,
            0.03
        ),
        "Normal Mack 1yr SD={} vs 1,778,968",
        mack_normal.adverse_development_summary.std_dev
    );
    assert!(
        close_rel(
            mack_normal.lifetime_reserve_summary.std_dev,
            2_447_095.0,
            0.03
        ),
        "Normal Mack lifetime SD={} vs 2,447,095",
        mack_normal.lifetime_reserve_summary.std_dev
    );

    // Non-parametric Mack bootstrap residuals (default).
    let mack_boot = mack_one_year_runoff(&tri.view(), 20_000, 20_190_111).expect("mack boot");
    assert_eq!(mack_boot.failed_iterations, 0);
    assert!(
        mack_boot.adverse_development_summary.mean.abs() / mack_boot.opening_total_reserve < 0.005,
        "Bootstrap Mack 1yr mean bias={}",
        mack_boot.adverse_development_summary.mean
    );
    assert!(
        close_rel(
            mack_boot.adverse_development_summary.std_dev,
            1_778_968.0,
            0.08
        ),
        "Bootstrap Mack 1yr SD={} vs 1,778,968",
        mack_boot.adverse_development_summary.std_dev
    );
    assert!(
        close_rel(
            mack_boot.lifetime_reserve_summary.std_dev,
            2_447_095.0,
            0.08
        ),
        "Bootstrap Mack lifetime SD={} vs 2,447,095",
        mack_boot.lifetime_reserve_summary.std_dev
    );

    // Exact pathwise accounting identity L_{t+1} = P_{t+1} + R_{t+1} - R_t on every scenario.
    for s in 0..mack_boot.total_adverse_development.len() {
        let lhs = mack_boot.total_adverse_development[s];
        let rhs = mack_boot.next_year_total_payments[s] + mack_boot.closing_total_reserves[s]
            - mack_boot.opening_total_reserve;
        assert!((lhs - rhs).abs() <= 1e-7);
    }

    // Coherent tail capital measures on one-year adverse development.
    let var995 = mack_boot.var_adverse_development(0.995).expect("var995");
    let tvar995 = mack_boot.tvar_adverse_development(0.995).expect("tvar995");
    assert!(tvar995 >= var995 && var995 > 0.0);

    // 2. ODP re-reserving reusing the same ChainLadderRollForward engine.
    let odp = odp_one_year_runoff(
        &tri.view(),
        1.0,
        OdpBootstrapOptions::shapland(5_000, 20_190_111),
    )
    .expect("odp runoff");
    assert_eq!(odp.failed_iterations, 0);
    assert!((odp.opening_total_reserve - 18_680_855.61).abs() < 1.0);
    assert!(odp.adverse_development_summary.std_dev > 0.0);
    assert!(odp.adverse_development_summary.std_dev < odp.lifetime_reserve_summary.std_dev);
}
