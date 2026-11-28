//! Regression tests for deterministic reserving-method fixes:
//! Brosius (1993) fallbacks, Case Outstanding / Incremental Additive
//! unestimable-ratio errors, Clark ALAE average-accident-date ages,
//! Berquist–Sherman custom-target consistency and severity trend, and
//! hand-derived Adler–Kline goldens.
//!
//! Every expected value below is derived by hand in the accompanying comments.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use reserving_kernels::{
    ReservingError, Triangle, datasets,
    deterministic::{
        berquist_sherman::{
            CaseReserveTrend, DisposalInterpolation, adjust_case_reserves,
            adjust_paid_disposal_rate, fit_severity_trend,
        },
        brosius::{self, BrosiusFallback, BrosiusFitRule},
        case_outstanding::{
            CaseOutstandingOptions, CaseRatioAverage, project_case_outstanding,
            project_from_paid_incurred,
        },
        freq_sev::{AdlerKlineOptions, CountProjectionMode, adler_kline_reserve},
        incremental_additive::{
            IncrementalAdditiveOptions, IncrementalRatioAverage, project_incremental_additive,
        },
        lae::{ClarkAlaeOptions, clark_alae_mle},
        murphy::{self, MurphyOptions},
    },
    stochastic::clark_mle::{ClarkOptions, GrowthCurve, fit_clark_cape_cod},
    validate::MonotonicityPolicy,
};

const NAN: f64 = f64::NAN;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

fn tri(rows: &[&[f64]]) -> Triangle {
    Triangle::try_from_upper_rows(rows).expect("triangle")
}

// ===========================================================================
// Brosius (1993) fallbacks
// ===========================================================================

#[test]
fn brosius_negative_slope_falls_back_to_budget_method() {
    // Link 0 pairs (100,400), (200,380), (300,360):
    //   mean_x = 200, mean_y = 380, dx = (-100, 0, 100), dy = (20, 0, -20)
    //   cov = -4000/3, var = 20000/3  =>  b = -0.2 < 0  =>  budget: a = 380, b = 0.
    // Link 1 pairs (400,440), (380,420): b = 20/20 = 1, a = 430 - 390 = 40 (no fallback).
    // Link 2 single pair (440,450): link ratio 450/440, a = 0.
    let t = tri(&[
        &[100.0, 400.0, 440.0, 450.0],
        &[200.0, 380.0, 420.0],
        &[300.0, 360.0],
        &[250.0],
    ]);
    let br = brosius::project(&t.view(), 1.0).expect("brosius");
    let d = &br.diagnostics;
    assert_eq!(d[0].fallback, BrosiusFallback::NegativeSlopeBudget);
    assert!(d[0].fallback_used);
    assert!(close(d[0].ols_slope, -0.2, 1e-12));
    assert!(close(d[0].ols_intercept, 420.0, 1e-12));
    assert!(close(br.intercept[0], 380.0, 1e-12));
    assert!(br.slope[0].abs() < 1e-15);

    assert_eq!(d[1].fallback, BrosiusFallback::None);
    assert!(!d[1].fallback_used);
    assert!(close(br.intercept[1], 40.0, 1e-12));
    assert!(close(br.slope[1], 1.0, 1e-12));

    assert_eq!(d[2].fallback, BrosiusFallback::SinglePairLinkRatio);
    let f2 = 450.0 / 440.0;

    // Origin 3: 250 -> 380 -> 420 -> 420 * f2.
    assert!(close(br.ultimate[3], 420.0 * f2, 1e-12));
    // Origin 2: 360 -> 400 -> 400 * f2.
    assert!(close(br.ultimate[2], 400.0 * f2, 1e-12));
    // Origin 1: 420 * f2; origin 0: 450.
    assert!(close(br.ultimate[1], 420.0 * f2, 1e-12));
    assert!(close(br.ultimate[0], 450.0, 1e-12));
    assert!(close(br.reserve[3], 420.0 * f2 - 250.0, 1e-12));
}

#[test]
fn brosius_negative_intercept_falls_back_to_link_ratio() {
    // Link 0 pairs (100,150), (200,350), (300,500):
    //   mean_x = 200, mean_y = 1000/3; cov = 35000/3, var = 20000/3
    //   => b = 1.75, a = 1000/3 - 350 = -50/3 < 0  =>  link ratio Σy/Σx = 1000/600 = 5/3, a = 0.
    // Link 1 pairs (150,165), (350,380): b = 215/200 = 1.075, a = 272.5 - 268.75 = 3.75.
    // Link 2 single pair (165,170): 170/165.
    let t = tri(&[
        &[100.0, 150.0, 165.0, 170.0],
        &[200.0, 350.0, 380.0],
        &[300.0, 500.0],
        &[250.0],
    ]);
    let br = brosius::project(&t.view(), 1.0).expect("brosius");
    let d = &br.diagnostics;
    assert_eq!(d[0].fallback, BrosiusFallback::NegativeInterceptLinkRatio);
    assert!(close(d[0].ols_intercept, -50.0 / 3.0, 1e-12));
    assert!(close(d[0].ols_slope, 1.75, 1e-12));
    assert!(br.intercept[0].abs() < 1e-15);
    assert!(close(br.slope[0], 5.0 / 3.0, 1e-12));
    assert_eq!(d[1].fallback, BrosiusFallback::None);
    assert!(close(br.intercept[1], 3.75, 1e-12));
    assert!(close(br.slope[1], 1.075, 1e-12));

    let f2 = 170.0 / 165.0;
    // Origin 3: 250 * 5/3 = 1250/3; 3.75 + 1.075 * 1250/3; then * f2.
    let expected_3 = (3.75 + 1.075 * (1250.0 / 3.0)) * f2;
    assert!(close(br.ultimate[3], expected_3, 1e-12));
    // Origin 2: (3.75 + 1.075 * 500) * f2.
    assert!(close(br.ultimate[2], (3.75 + 537.5) * f2, 1e-12));

    // The raw OLS rule keeps the negative intercept.
    let raw = brosius::project_with_fit_rule(
        &t.view(),
        1.0,
        MonotonicityPolicy::Strict,
        BrosiusFitRule::RawOls,
    )
    .expect("raw ols");
    assert!(close(raw.intercept[0], -50.0 / 3.0, 1e-12));
    assert!(close(raw.slope[0], 1.75, 1e-12));
    assert_eq!(raw.diagnostics[0].fallback, BrosiusFallback::None);
    let raw_3 = (3.75 + 1.075 * (-50.0 / 3.0 + 1.75 * 250.0)) * f2;
    assert!(close(raw.ultimate[3], raw_3, 1e-12));
}

#[test]
fn brosius_unchecked_and_observed_counts_agree_on_square_triangle() {
    let t = datasets::genins();
    let checked = brosius::project(&t.view(), 1.0).expect("checked");
    let unchecked = brosius::project_unchecked(&t.view(), 1.0);
    let counts: Vec<usize> = (0..10).map(|i| 10 - i).collect();
    let by_counts = brosius::project_observed_counts(&t.view(), &counts, 1.0).expect("counts");
    for i in 0..10 {
        assert_eq!(
            checked.ultimate[i].to_bits(),
            unchecked.ultimate[i].to_bits()
        );
        assert_eq!(
            checked.ultimate[i].to_bits(),
            by_counts.ultimate[i].to_bits()
        );
    }
    // GenIns link 0 has a negative OLS slope and links 1, 2, 5, 6, 7 negative intercepts.
    assert_eq!(
        checked.diagnostics[0].fallback,
        BrosiusFallback::NegativeSlopeBudget
    );
    for j in [1, 2, 5, 6, 7] {
        assert_eq!(
            checked.diagnostics[j].fallback,
            BrosiusFallback::NegativeInterceptLinkRatio,
            "link {j}"
        );
    }
    for j in [3, 4] {
        assert_eq!(checked.diagnostics[j].fallback, BrosiusFallback::None);
    }
}

#[test]
fn brosius_raw_ols_matches_murphy_brosius_equivalent() {
    let t = datasets::genins();
    let raw = brosius::project_with_fit_rule(
        &t.view(),
        1.0,
        MonotonicityPolicy::Strict,
        BrosiusFitRule::RawOls,
    )
    .expect("raw");
    let mu = murphy::analyze(&t.view(), MurphyOptions::brosius_equivalent()).expect("murphy");
    for i in 0..10 {
        assert!(close(mu.ultimate[i], raw.ultimate[i], 1e-10));
    }
}

// ===========================================================================
// Case Outstanding
// ===========================================================================

fn co_paid() -> Triangle {
    tri(&[&[100.0, 160.0, 190.0], &[120.0, 180.0], &[130.0]])
}

fn co_case() -> Triangle {
    tri(&[&[80.0, 40.0, 10.0], &[100.0, 45.0], &[90.0]])
}

#[test]
fn case_outstanding_hand_golden_volume_weighted() {
    // j=1: incremental paid (60, 60), prior case (80, 100), case (40, 45)
    //      r_p = 120/180 = 2/3, r_c = 85/180 = 17/36.
    // j=2: incremental paid 30, prior case 40, case 10  =>  r_p = 0.75, r_c = 0.25.
    // Origin 1: paid 180 + 45*0.75 = 213.75, case 11.25 => ult 225.
    // Origin 2: paid 130 + 90*2/3 = 190, case 42.5; paid 190 + 42.5*0.75 = 221.875,
    //           case 10.625 => ult 232.5.
    let res = project_case_outstanding(
        &co_paid().view(),
        &co_case().view(),
        &CaseOutstandingOptions::default(),
    )
    .expect("co");
    assert!(close(res.paid_to_case_ratios[0], 2.0 / 3.0, 1e-12));
    assert!(close(res.case_to_case_ratios[0], 17.0 / 36.0, 1e-12));
    assert!(close(res.paid_to_case_ratios[1], 0.75, 1e-12));
    assert!(close(res.case_to_case_ratios[1], 0.25, 1e-12));
    let expected_ult = [200.0, 225.0, 232.5];
    let expected_res = [10.0, 45.0, 102.5];
    let expected_ibnr = [0.0, 0.0, 12.5];
    for i in 0..3 {
        assert!(close(res.ultimate[i], expected_ult[i], 1e-12), "ult {i}");
        assert!(close(res.reserve[i], expected_res[i], 1e-12), "res {i}");
        assert!((res.ibnr[i] - expected_ibnr[i]).abs() < 1e-10, "ibnr {i}");
    }
    assert!(close(res.total_reserve, 157.5, 1e-12));
    assert!(close(res.total_ibnr, 12.5, 1e-12));
}

#[test]
fn case_outstanding_simple_and_last_n_averages() {
    // Simple: r_p1 = (60/80 + 60/100)/2 = 0.675, r_c1 = (0.5 + 0.45)/2 = 0.475.
    let simple = project_case_outstanding(
        &co_paid().view(),
        &co_case().view(),
        &CaseOutstandingOptions {
            average: CaseRatioAverage::Simple,
            ..CaseOutstandingOptions::default()
        },
    )
    .expect("simple");
    assert!(close(simple.paid_to_case_ratios[0], 0.675, 1e-12));
    assert!(close(simple.case_to_case_ratios[0], 0.475, 1e-12));
    // Origin 2: paid 130 + 90*0.675 = 190.75, case 42.75; + 42.75*0.75 = 222.8125,
    // case 10.6875 => ult 233.5.
    assert!(close(simple.ultimate[2], 233.5, 1e-12));

    // last_n = 1 keeps only the latest origin (i = 1): r_p1 = 0.6, r_c1 = 0.45.
    let last1 = project_case_outstanding(
        &co_paid().view(),
        &co_case().view(),
        &CaseOutstandingOptions {
            last_n: Some(1),
            ..CaseOutstandingOptions::default()
        },
    )
    .expect("last_n");
    assert!(close(last1.paid_to_case_ratios[0], 0.6, 1e-12));
    assert!(close(last1.case_to_case_ratios[0], 0.45, 1e-12));

    // Excluding the terminal case drops end_case from the ultimate.
    let no_term = project_case_outstanding(
        &co_paid().view(),
        &co_case().view(),
        &CaseOutstandingOptions {
            include_terminal_case: false,
            ..CaseOutstandingOptions::default()
        },
    )
    .expect("no terminal");
    assert!(close(no_term.ultimate[2], 221.875, 1e-12));
}

#[test]
fn case_outstanding_unestimable_ratio_with_open_case_is_error() {
    // Every observed prior case at j=0 is zero, so the j=1 ratios are unestimable,
    // yet origin 2 still carries a case reserve of 50 into j=1. The old kernel
    // returned ratio 0.0 and dropped the 50 (ultimate 120 < reported 170).
    let paid = tri(&[&[100.0, 150.0, 160.0], &[110.0, 170.0], &[120.0]]);
    let case = tri(&[&[0.0, 0.0, 0.0], &[0.0, 0.0], &[50.0]]);
    let err = project_case_outstanding(
        &paid.view(),
        &case.view(),
        &CaseOutstandingOptions::default(),
    )
    .expect_err("unestimable ratio must error");
    assert!(matches!(err, ReservingError::InvalidParameter(_)));
}

#[test]
fn case_outstanding_unneeded_unestimable_ratio_is_nan_and_ultimate_is_paid() {
    // No case reserves anywhere: ratios are unestimable but never needed.
    let paid = tri(&[&[100.0, 150.0, 160.0], &[110.0, 170.0], &[120.0]]);
    let case = tri(&[&[0.0, 0.0, 0.0], &[0.0, 0.0], &[0.0]]);
    let res = project_case_outstanding(
        &paid.view(),
        &case.view(),
        &CaseOutstandingOptions::default(),
    )
    .expect("all-closed triangle");
    assert!(res.paid_to_case_ratios.iter().all(|r| r.is_nan()));
    assert!(res.case_to_case_ratios.iter().all(|r| r.is_nan()));
    assert_eq!(res.ultimate, vec![160.0, 170.0, 120.0]);
    assert!(res.total_reserve.abs() < 1e-15);
}

#[test]
fn case_outstanding_one_to_one_settlement_reproduces_reported() {
    // Case pays out exactly 1:1 at the next age (r_p = 1, r_c = 0):
    // ultimate == latest reported (paid + case) for every origin.
    let paid = tri(&[&[100.0, 150.0, 150.0], &[110.0, 170.0], &[120.0]]);
    let case = tri(&[&[50.0, 0.0, 0.0], &[60.0, 0.0], &[70.0]]);
    let res = project_case_outstanding(
        &paid.view(),
        &case.view(),
        &CaseOutstandingOptions::default(),
    )
    .expect("1:1");
    let reported_latest = [150.0, 170.0, 190.0];
    for i in 0..3 {
        assert!(close(res.ultimate[i], reported_latest[i], 1e-12));
        assert!(res.ultimate[i] >= reported_latest[i] - 1e-9);
        assert!(res.ibnr[i].abs() < 1e-9);
    }
}

#[test]
fn case_outstanding_real_datasets_ultimate_at_least_paid() {
    let auto = datasets::friedland_us_industry_auto();
    let qm = datasets::quarg_mack_2004();
    for (paid, inc) in [(&auto.paid, &auto.reported), (&qm.paid, &qm.incurred)] {
        let res = project_from_paid_incurred(
            &paid.view(),
            &inc.view(),
            &CaseOutstandingOptions::default(),
        )
        .expect("co dataset");
        let n = paid.n_origins();
        for i in 0..n {
            let latest_paid = paid.view().get(i, n - 1 - i);
            assert!(res.ultimate[i].is_finite());
            assert!(res.ultimate[i] >= latest_paid, "origin {i}");
        }
    }
}

// ===========================================================================
// Incremental Additive
// ===========================================================================

#[test]
fn incremental_additive_hand_golden() {
    // Incrementals: r0 (10, 5, 2), r1 (24, 12), r2 (40); exposures (100, 200, 400).
    // Exposure-weighted: zeta0 = 74/700, zeta1 = 17/300, zeta2 = 2/100.
    // Reserves: r1 = 200*0.02 = 4; r2 = 400*17/300 + 400*0.02 = 22.6667 + 8.
    let t = tri(&[&[10.0, 15.0, 17.0], &[24.0, 36.0], &[40.0]]);
    let exposures = [100.0, 200.0, 400.0];
    let vw = project_incremental_additive(
        &t.view(),
        &exposures,
        &IncrementalAdditiveOptions::default(),
    )
    .expect("vw");
    assert!(close(vw.zeta[0], 74.0 / 700.0, 1e-12));
    assert!(close(vw.zeta[1], 17.0 / 300.0, 1e-12));
    assert!(close(vw.zeta[2], 0.02, 1e-12));
    assert!(vw.reserve[0].abs() < 1e-12);
    assert!(close(vw.reserve[1], 4.0, 1e-12));
    assert!(close(vw.reserve[2], 400.0 * 17.0 / 300.0 + 8.0, 1e-12));
    assert!(close(
        vw.ultimate[2],
        40.0 + 400.0 * 17.0 / 300.0 + 8.0,
        1e-12
    ));

    // Simple average: zeta1 = (0.05 + 0.06)/2 = 0.055 => r2 reserve = 22 + 8 = 30.
    let simple = project_incremental_additive(
        &t.view(),
        &exposures,
        &IncrementalAdditiveOptions {
            average: IncrementalRatioAverage::Simple,
            ..IncrementalAdditiveOptions::default()
        },
    )
    .expect("simple");
    assert!(close(simple.zeta[1], 0.055, 1e-12));
    assert!(close(simple.total_reserve, 34.0, 1e-12));

    // last_n = 1 at age 1 keeps only origin 1: zeta1 = 0.06 => r2 reserve = 24 + 8.
    let last1 = project_incremental_additive(
        &t.view(),
        &exposures,
        &IncrementalAdditiveOptions {
            last_n: Some(1),
            ..IncrementalAdditiveOptions::default()
        },
    )
    .expect("last1");
    assert!(close(last1.zeta[1], 0.06, 1e-12));
    assert!(close(last1.reserve[2], 32.0, 1e-12));
    // last_n = 1 at age 0 keeps only origin 2: zeta0 = 40/400.
    assert!(close(last1.zeta[0], 0.1, 1e-12));
}

#[test]
fn incremental_additive_rejects_invalid_inputs() {
    let t = tri(&[&[10.0, 15.0, 17.0], &[24.0, 36.0], &[40.0]]);
    for bad in [
        [100.0, 0.0, 400.0],
        [100.0, -1.0, 400.0],
        [100.0, NAN, 400.0],
    ] {
        assert!(matches!(
            project_incremental_additive(&t.view(), &bad, &IncrementalAdditiveOptions::default()),
            Err(ReservingError::InvalidParameter(_))
        ));
    }
    assert!(
        project_incremental_additive(
            &t.view(),
            &[1.0, 2.0],
            &IncrementalAdditiveOptions::default()
        )
        .is_err(),
        "exposure length mismatch"
    );
    assert!(
        project_incremental_additive(
            &t.view(),
            &[100.0, 200.0, 400.0],
            &IncrementalAdditiveOptions {
                trend: -1.0,
                ..IncrementalAdditiveOptions::default()
            }
        )
        .is_err()
    );
}

// ===========================================================================
// Clark ALAE vs Clark Cape Cod
// ===========================================================================

/// Deterministic, monotone ALAE triangle from a Loglogistic curve at
/// midpoint ages with a fixed multiplicative perturbation (no RNG).
fn clark_alae_fixture(n: usize) -> (Triangle, Vec<f64>) {
    let ults: Vec<f64> = (0..n).map(|i| 10_000.0 + 750.0 * i as f64).collect();
    let mut data = vec![NAN; n * n];
    for i in 0..n {
        let mut cum = 0.0;
        let mut prev_g = 0.0;
        for j in 0..(n - i) {
            let g = GrowthCurve::Loglogistic.eval(j as f64 + 0.5, 1.7, 2.3);
            let bump = 1.0 + 0.08 * (((i * 7 + j * 3) % 5) as f64 - 2.0) / 2.0;
            cum += ults[i] * 0.11 * (g - prev_g) * bump;
            prev_g = g;
            data[i * n + j] = cum;
        }
    }
    (Triangle::try_new(n, n, data).expect("valid triangle"), ults)
}

#[test]
fn clark_alae_default_offset_reproduces_clark_cape_cod() {
    let (alae, ults) = clark_alae_fixture(8);
    let alae_fit =
        clark_alae_mle(&alae.view(), &ults, ClarkAlaeOptions::default()).expect("clark alae");
    let cc = fit_clark_cape_cod(
        &alae.view(),
        &ults,
        &ClarkOptions::new(GrowthCurve::Loglogistic),
    )
    .expect("clark cape cod");
    assert!(alae_fit.converged);
    assert!(
        close(alae_fit.omega, cc.omega, 1e-6),
        "omega {} vs {}",
        alae_fit.omega,
        cc.omega
    );
    assert!(
        close(alae_fit.theta, cc.theta, 1e-6),
        "theta {} vs {}",
        alae_fit.theta,
        cc.theta
    );
    assert!(close(alae_fit.elr_alae, cc.elr.expect("elr"), 1e-6));
    for (a, b) in alae_fit.ages.iter().zip(cc.ages.iter()) {
        assert!((a - b).abs() < 1e-15);
    }
    for i in 0..8 {
        assert!(
            close(alae_fit.reserves[i], cc.reserves[i], 1e-5),
            "reserve {i}"
        );
    }
}

#[test]
fn clark_alae_max_age_truncation_matches_clark_cape_cod() {
    let (alae, ults) = clark_alae_fixture(7);
    let max_age = 10.0;
    let opts = ClarkAlaeOptions {
        max_age: Some(max_age),
        ..ClarkAlaeOptions::default()
    };
    let alae_fit = clark_alae_mle(&alae.view(), &ults, opts).expect("clark alae");
    let cc = fit_clark_cape_cod(
        &alae.view(),
        &ults,
        &ClarkOptions::new(GrowthCurve::Loglogistic).with_max_age(max_age),
    )
    .expect("clark cc");
    assert!(close(
        alae_fit.ultimate_emergence,
        GrowthCurve::Loglogistic.eval(max_age, alae_fit.omega, alae_fit.theta),
        1e-15
    ));
    assert!(alae_fit.ultimate_emergence < 1.0);
    assert!(close(alae_fit.total_reserve, cc.total_reserve, 1e-5));
    let untruncated =
        clark_alae_mle(&alae.view(), &ults, ClarkAlaeOptions::default()).expect("untruncated");
    assert!(alae_fit.total_reserve < untruncated.total_reserve);
}

#[test]
fn clark_alae_zero_offset_uses_integer_ages_and_validates_options() {
    let (alae, ults) = clark_alae_fixture(6);
    let fit = clark_alae_mle(
        &alae.view(),
        &ults,
        ClarkAlaeOptions {
            origin_offset: 0.0,
            ..ClarkAlaeOptions::default()
        },
    )
    .expect("offset 0");
    assert_eq!(fit.ages, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    for bad in [1.0, 1.5, NAN] {
        assert!(matches!(
            clark_alae_mle(
                &alae.view(),
                &ults,
                ClarkAlaeOptions {
                    origin_offset: bad,
                    ..ClarkAlaeOptions::default()
                }
            ),
            Err(ReservingError::InvalidParameter(_))
        ));
    }
    // Largest default age is 5.5; max_age below it is rejected.
    assert!(matches!(
        clark_alae_mle(
            &alae.view(),
            &ults,
            ClarkAlaeOptions {
                max_age: Some(5.0),
                ..ClarkAlaeOptions::default()
            }
        ),
        Err(ReservingError::InvalidParameter(_))
    ));
}

// ===========================================================================
// Berquist–Sherman
// ===========================================================================

/// Paid linear in closed counts (`P = k * C`), varying historical disposal.
fn bs_linear_fixture() -> (Triangle, Triangle, Vec<f64>, f64) {
    let n = 5usize;
    let k = 1_000.0;
    let ult_counts = vec![100.0, 110.0, 120.0, 130.0, 140.0];
    let disp = [
        [0.20, 0.45, 0.70, 0.88, 1.00],
        [0.22, 0.48, 0.72, 0.90, NAN],
        [0.25, 0.52, 0.76, NAN, NAN],
        [0.28, 0.57, NAN, NAN, NAN],
        [0.31, NAN, NAN, NAN, NAN],
    ];
    let mut paid = vec![NAN; n * n];
    let mut closed = vec![NAN; n * n];
    for i in 0..n {
        for j in 0..(n - i) {
            let c = ult_counts[i] * disp[i][j];
            closed[i * n + j] = c;
            paid[i * n + j] = k * c;
        }
    }
    (
        Triangle::try_new(n, n, paid).expect("valid triangle"),
        Triangle::try_new(n, n, closed).expect("valid triangle"),
        ult_counts,
        k,
    )
}

#[test]
fn berquist_sherman_custom_targets_project_adjusted_diagonal() {
    // With P = k*C, the adjusted triangle is Paid*_{i,j} = k*N_i*d*_j, so the
    // adjusted LDFs are d*_{j+1}/d*_j and developing the adjusted latest diagonal
    // gives k*N_i*d*_{n-1} = k*N_i exactly (d*_{n-1} = 1). Applying the same
    // LDFs to the actual latest paid (old behaviour) would give
    // k*N_i*d_{i,n-1-i}/d*_{n-1-i} * ... != k*N_i.
    let (paid, closed, ult_counts, k) = bs_linear_fixture();
    let targets = [0.30, 0.55, 0.75, 0.90, 1.00];
    let res = adjust_paid_disposal_rate(
        &paid.view(),
        &closed.view(),
        &ult_counts,
        Some(&targets),
        DisposalInterpolation::PiecewiseLinear,
        1.0,
    )
    .expect("bs custom");
    let n = 5;
    for i in 0..n {
        let latest_adj = res.adjusted_paid.view().get(i, n - 1 - i);
        assert!(close(
            latest_adj,
            k * ult_counts[i] * targets[n - 1 - i],
            1e-12
        ));
        assert!(close(res.ultimates[i], k * ult_counts[i], 1e-10), "ult {i}");
        let actual_paid = paid.view().get(i, n - 1 - i);
        assert!(close(
            res.reserves[i],
            res.ultimates[i] - actual_paid,
            1e-12
        ));
    }
    let total_res: f64 = res.reserves.iter().sum();
    assert!(close(res.total_reserve, total_res, 1e-12));
}

#[test]
fn berquist_sherman_latest_diagonal_targets_match_default() {
    let (paid, closed, ult_counts, _) = bs_linear_fixture();
    let default = adjust_paid_disposal_rate(
        &paid.view(),
        &closed.view(),
        &ult_counts,
        None,
        DisposalInterpolation::PiecewiseLinear,
        1.0,
    )
    .expect("default");
    let explicit = adjust_paid_disposal_rate(
        &paid.view(),
        &closed.view(),
        &ult_counts,
        Some(&default.target_disposal_rates),
        DisposalInterpolation::PiecewiseLinear,
        1.0,
    )
    .expect("explicit");
    for i in 0..5 {
        assert!(close(explicit.ultimates[i], default.ultimates[i], 1e-10));
        assert!(close(explicit.reserves[i], default.reserves[i], 1e-10));
    }
}

#[test]
fn berquist_sherman_rejects_decreasing_custom_targets() {
    let (paid, closed, ult_counts, _) = bs_linear_fixture();
    assert!(matches!(
        adjust_paid_disposal_rate(
            &paid.view(),
            &closed.view(),
            &ult_counts,
            Some(&[0.3, 0.6, 0.5, 0.9, 1.0]),
            DisposalInterpolation::PiecewiseLinear,
            1.0,
        ),
        Err(ReservingError::InvalidParameter(_))
    ));
}

#[test]
fn berquist_sherman_paid_severity_trend_avoids_case_strengthening_bias() {
    // Paid-on-closed severity grows at r = 0.07 per accident year exactly;
    // average case is strengthened +40% on the latest diagonal.
    let n = 5usize;
    let r = 0.07_f64;
    let base_closed = [30.0, 60.0, 80.0, 95.0, 100.0];
    let base_paid_sev = [500.0, 800.0, 1000.0, 1100.0, 1150.0];
    let base_open = [50.0, 30.0, 15.0, 5.0, 0.0];
    let mut paid = vec![NAN; n * n];
    let mut closed = vec![NAN; n * n];
    let mut open = vec![NAN; n * n];
    let mut reported = vec![NAN; n * n];
    for i in 0..n {
        let infl = (r * i as f64).exp();
        for j in 0..(n - i) {
            let idx = i * n + j;
            let p = base_closed[j] * base_paid_sev[j] * infl;
            let adequacy = if i + j == n - 1 { 1.4 } else { 1.0 };
            paid[idx] = p;
            closed[idx] = base_closed[j];
            open[idx] = base_open[j];
            reported[idx] = p + base_open[j] * 2_000.0 * infl * adequacy;
        }
    }
    let paid = Triangle::try_new(n, n, paid).expect("valid triangle");
    let closed = Triangle::try_new(n, n, closed).expect("valid triangle");
    let open = Triangle::try_new(n, n, open).expect("valid triangle");
    let reported = Triangle::try_new(n, n, reported).expect("valid triangle");

    let fitted = fit_severity_trend(&paid.view(), &closed.view()).expect("paid trend");
    assert!(close(fitted, r, 1e-12), "fitted={fitted}");

    // The circular average-case fit absorbs the strengthening and overstates r.
    let circular = adjust_case_reserves(
        &reported.view(),
        &paid.view(),
        &open.view(),
        CaseReserveTrend::FitFromAverageCase,
        1.0,
        MonotonicityPolicy::AllowNegativeIncrements,
    )
    .expect("circular");
    assert!(circular.exponential_trend_rate > r + 0.01);

    let adj = adjust_case_reserves(
        &reported.view(),
        &paid.view(),
        &open.view(),
        CaseReserveTrend::ExponentialRate(fitted),
        1.0,
        MonotonicityPolicy::AllowNegativeIncrements,
    )
    .expect("paid-trend adj");
    assert!(close(adj.exponential_trend_rate, r, 1e-12));
}

#[test]
fn berquist_sherman_case_with_zero_open_counts_is_error() {
    let reported = tri(&[&[150.0, 200.0], &[160.0]]);
    let paid = tri(&[&[100.0, 180.0], &[120.0]]);
    let open = tri(&[&[0.0, 0.0], &[5.0]]);
    assert!(matches!(
        adjust_case_reserves(
            &reported.view(),
            &paid.view(),
            &open.view(),
            CaseReserveTrend::ExponentialRate(0.0),
            1.0,
            MonotonicityPolicy::AllowNegativeIncrements,
        ),
        Err(ReservingError::InvalidParameter(_))
    ));
}

#[test]
fn berquist_sherman_undefined_anchor_average_case_is_error() {
    // Column 0 anchor (origin 1) has zero open claims and zero case while origin 0
    // still has open claims at age 0: the restated average case is undefined.
    let reported = tri(&[&[150.0, 200.0], &[120.0]]);
    let paid = tri(&[&[100.0, 200.0], &[120.0]]);
    let open = tri(&[&[5.0, 0.0], &[0.0]]);
    assert!(matches!(
        adjust_case_reserves(
            &reported.view(),
            &paid.view(),
            &open.view(),
            CaseReserveTrend::ExponentialRate(0.0),
            1.0,
            MonotonicityPolicy::AllowNegativeIncrements,
        ),
        Err(ReservingError::InvalidParameter(_))
    ));
}

// ===========================================================================
// Adler–Kline (Friedland Method 2) hand-computed goldens
// ===========================================================================

fn adler_kline_fixture() -> (Triangle, Triangle, [f64; 3]) {
    // Closed counts (cumulative) and incremental paid; N_hat = 100 for every origin.
    let closed = tri(&[&[40.0, 70.0, 90.0], &[50.0, 80.0], &[45.0]]);
    let inc_paid = Triangle::try_new(
        3,
        3,
        vec![
            4_000.0, 6_000.0, 5_000.0, 5_500.0, 6_600.0, NAN, 4_950.0, NAN, NAN,
        ],
    )
    .expect("valid triangle");
    (closed, inc_paid, [100.0, 100.0, 100.0])
}

#[test]
fn adler_kline_anchored_and_conserving_hand_golden() {
    // Disposal rates: d0 = (0.40 + 0.50 + 0.45)/3 = 0.45, d1 = (0.70 + 0.80)/2 = 0.75, d2 = 0.90.
    // Severities trended 10% to AY 2:
    //   lag 0: (100*1.21, 110*1.1, 110) = (121, 121, 110) => 352/3
    //   lag 1: (200*1.21, 220*1.1) = (242, 242) => 242
    //   lag 2: 250*1.21 = 302.5 (also the default tail severity).
    // AnchoredToActual (detrend 1.1^-(2-i)):
    //   AY0: tail 100*(1-0.9) = 10 claims * 302.5/1.21 = 2500
    //   AY1: lag 2 (0.9-0.8)*100 = 10 * 302.5/1.1 = 2750; tail 10 * 275 = 2750 => 5500
    //   AY2: lag 1 30*242 = 7260; lag 2 15*302.5 = 4537.5; tail 10*302.5 = 3025 => 14822.5
    // ExactConservation:
    //   AY0: remaining 10 to the tail => 2500
    //   AY1: scale = 20/(100*0.25) = 0.8; lag 2 12 claims * 275 = 3300; tail 8 * 275 = 2200
    //   AY2: scale = 55/55 = 1 => identical to anchored: 14822.5
    let (closed, inc_paid, ults) = adler_kline_fixture();
    for (mode, expected) in [
        (
            CountProjectionMode::AnchoredToActual,
            [2500.0, 5500.0, 14_822.5],
        ),
        (
            CountProjectionMode::ExactConservation,
            [2500.0, 5500.0, 14_822.5],
        ),
    ] {
        let res = adler_kline_reserve(
            &closed.view(),
            &inc_paid.view(),
            &ults,
            AdlerKlineOptions {
                severity_trend: 0.10,
                count_projection_mode: mode,
                ..AdlerKlineOptions::default()
            },
        )
        .expect("adler-kline");
        assert!(close(res.selected_disposal_rates[0], 0.45, 1e-12));
        assert!(close(res.selected_disposal_rates[1], 0.75, 1e-12));
        assert!(close(res.selected_disposal_rates[2], 0.90, 1e-12));
        assert!(close(
            res.selected_trended_severities[0],
            352.0 / 3.0,
            1e-12
        ));
        assert!(close(res.selected_trended_severities[1], 242.0, 1e-12));
        assert!(close(res.selected_trended_severities[2], 302.5, 1e-12));
        assert!(close(res.tail_severity, 302.5, 1e-12));
        for i in 0..3 {
            assert!(close(res.reserves[i], expected[i], 1e-10), "{mode:?} AY{i}");
        }
        assert!(close(res.total_reserve, 22_822.5, 1e-10));
    }
    // Exact conservation splits AY1's 20 open claims into 12 (lag 2) + 8 (tail).
    let exact = adler_kline_reserve(
        &closed.view(),
        &inc_paid.view(),
        &ults,
        AdlerKlineOptions {
            severity_trend: 0.10,
            ..AdlerKlineOptions::default()
        },
    )
    .expect("exact");
    assert!(close(exact.projected_closed_counts[5], 12.0, 1e-12));
    assert!(close(exact.tail_closed_counts[1], 8.0, 1e-12));
}

#[test]
fn adler_kline_without_any_closings_is_error() {
    let closed = tri(&[&[0.0, 0.0], &[0.0]]);
    let inc_paid = Triangle::try_new(2, 2, vec![10.0, 5.0, 12.0, NAN]).expect("valid triangle");
    assert!(matches!(
        adler_kline_reserve(
            &closed.view(),
            &inc_paid.view(),
            &[10.0, 10.0],
            AdlerKlineOptions::default()
        ),
        Err(ReservingError::InvalidParameter(_))
    ));
}

#[test]
fn adler_kline_lag_without_closings_carries_prior_severity() {
    // Lag 1 has no incremental closings (counts flat), so it inherits lag 0's
    // severity: (100 + 100)/2 = 100. AY1 projects 50 future claims at lag 1
    // (selected d1 = 0.5 = d0, so all 50 open claims go to the tail at 100).
    let closed = tri(&[&[50.0, 50.0], &[50.0]]);
    let inc_paid =
        Triangle::try_new(2, 2, vec![5_000.0, 0.0, 5_000.0, NAN]).expect("valid triangle");
    let res = adler_kline_reserve(
        &closed.view(),
        &inc_paid.view(),
        &[100.0, 100.0],
        AdlerKlineOptions::default(),
    )
    .expect("carry-forward");
    assert!(close(res.selected_trended_severities[1], 100.0, 1e-12));
    // AY0 tail: 50 * 100; AY1: 50 remaining * 100.
    assert!(close(res.total_reserve, 10_000.0, 1e-12));
}
