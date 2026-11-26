//! Source-document benchmark and validation suite.
//!
//! Every test in this file verifies `reserving-kernels` against verbatim
//! published actuarial datasets, closed-form identities, or worked examples
//! from primary literature (see `docs/SOURCES.md` for full citations).

use reserving_kernels::{
    benktander::BenktanderEngine,
    bf::BornhuetterFergusonEngine,
    bootstrap_odp::{
        OdpBootstrapOptions, ProcessDistribution, ResidualAdjustment, bootstrap_odp, fit_odp,
    },
    cape_cod::CapeCodEngine,
    chain_ladder::ChainLadderEngine,
    dependence::{CorrelationType, iman_conover, synchronized_odp_bootstrap},
    diagnostics::{
        cape_cod_diagnostics, mack_diagnostics,
        mack_tests::{adjacent_factor_correlation_test, calendar_year_effect_test},
        paid_incurred_reconciliation,
    },
    ldf::DevelopmentFactorEngine,
    mack::{self, MackTailSpec},
    munich_chain_ladder,
    reserve_distribution::{rvar, tvar_coherent, var_least},
    severity::SeverityDistribution,
    tail,
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
    Triangle::new(10, 10, data)
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
    Triangle::new(10, 10, data)
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
        Triangle::new(10, 10, paid),
        Triangle::new(10, 10, reported),
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
    (Triangle::new(n, n, data), true_reserve)
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
// 6. Severity ILFs, Mildenhall (2006) Dependence & CAS Monograph 15 Risk Measures
// ===========================================================================

#[test]
fn severity_ilf_dependence_and_cas_monograph_15_identities() {
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
