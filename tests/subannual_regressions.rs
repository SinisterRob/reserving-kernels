//! Regression tests for the `subannual` module review findings:
//!
//! * hand-derived golden values (interpolators, quarterly chain ladder, IVF12),
//! * structural invariants (knots, monotonicity, `CDF >= 1`),
//! * edge cases (stub boundaries, single-period grids, `NaN` cells),
//! * malformed / hand-built JSON and mutated structs returning `Err` without
//!   panicking (`catch_unwind`),
//! * Clark exact-age MLE cross-checked against the square `clark_mle` fit on
//!   an annual-equivalent asymmetric grid.

use std::panic::{AssertUnwindSafe, catch_unwind};

use reserving_kernels::ReservingError;
use reserving_kernels::datasets::genins;
use reserving_kernels::deterministic::ldf::LdfSelectionMethod;
use reserving_kernels::stochastic::bootstrap_odp::OdpBootstrapOptions;
use reserving_kernels::stochastic::clark_mle::{
    ClarkCovariance, ClarkOptions, fit_clark_cape_cod, fit_clark_ldf,
};
use reserving_kernels::subannual::*;
use reserving_kernels::validate::MonotonicityPolicy;

const ANNUAL_AGES: [f64; 5] = [12.0, 24.0, 36.0, 48.0, 60.0];
const ANNUAL_CDFS: [f64; 5] = [2.50, 1.45, 1.18, 1.06, 1.00];

fn assert_close(got: f64, expected: f64, tol: f64, what: &str) {
    assert!(
        (got - expected).abs() <= tol,
        "{what}: got {got:.15}, expected {expected:.15}"
    );
}

// ===========================================================================
// 1. Hand-derived golden values
// ===========================================================================

#[test]
fn simple_interpolators_match_hand_computed_midpoints() {
    // Midpoint of the first annual window, t = 18 (w = 0.5):
    // * Linear CDF:          (2.50 + 1.45) / 2                       = 1.975
    // * Linear % reported:   p = (1/2.5 + 1/1.45) / 2 = (2/5 + 20/29) / 2
    //                          = 79/145  ->  CDF = 145/79             = 1.835443037974684
    // * Exponential:         sqrt(2.50 * 1.45) = sqrt(3.625)          = 1.903943276465977
    let t = [18.0];
    let lin = interp_linear_cdf(&ANNUAL_AGES, &ANNUAL_CDFS, &t).unwrap();
    let pct = interp_linear_pct_reported(&ANNUAL_AGES, &ANNUAL_CDFS, &t).unwrap();
    let ex = interp_exponential(&ANNUAL_AGES, &ANNUAL_CDFS, &t).unwrap();
    assert_close(lin.cdfs[0], 1.975, 1e-14, "linear CDF @18");
    assert_close(pct.cdfs[0], 145.0 / 79.0, 1e-14, "linear pct @18");
    assert_close(ex.cdfs[0], 3.625_f64.sqrt(), 1e-14, "exponential @18");
}

#[test]
fn ivf12_matches_hand_derived_pchip_golden_values() {
    // phi_k = ln(CDF_k / CDF_{k+1}), phi_4 = 0:
    //   phi = [0.544727175441672, 0.206049117954910, 0.107245530353598,
    //          0.058268908123976, 0.0]
    // Secants d_k = (phi_{k+1} - phi_k) / 12 (all negative). PCHIP tangents:
    //   m_0 = (3 d_0 - d_1) / 2                (non-centred 3-point, same sign as d_0)
    //   m_k = 2 / (1/d_{k-1} + 1/d_k)          (harmonic mean, k = 1..3)
    //   m_4 = 0                                (pinned)
    //   m  = [-0.038217941035791, -0.012748194694659, -0.005457503399155,
    //         -0.004435016687892, 0.0]
    //   (Fritsch-Carlson alpha^2 + beta^2 <= 9 holds on every segment: no rescale.)
    // A cubic Hermite at its midpoint is (y_0 + y_1)/2 + h (m_0 - m_1) / 8, so
    //   phi(54) = 0.026960637838318,  phi(42) = 0.080792710061893,
    //   phi(30) = 0.145925017318466,  phi(18) = 0.337016418720217.
    // ln CDF(t) = ln CDF(60) + sum of phi at t, t+12, ... below 60:
    let golden = [
        (18.0, 1.797_865_688_214_107_3),
        (30.0, 1.283_276_670_003_840_5),
        (42.0, 1.109_273_634_958_527_7),
        (54.0, 1.022_736_552_156_053_2),
    ];
    let targets: Vec<f64> = golden.iter().map(|g| g.0).collect();
    let pat = interp_ivf12(&ANNUAL_AGES, &ANNUAL_CDFS, &targets).unwrap();
    for (&(t, expected), &got) in golden.iter().zip(pat.cdfs.iter()) {
        assert_close(got, expected, 1e-12, &format!("IVF12 @{t}"));
    }

    // Exponential gives sqrt(2.5 * 1.45) = 1.9039 at 18 and sqrt(1.06) = 1.0296 at 54;
    // IVF12 is materially different (the former implementation matched to 4e-16).
    let ex = interp_exponential(&ANNUAL_AGES, &ANNUAL_CDFS, &targets).unwrap();
    assert!((pat.cdfs[0] - ex.cdfs[0]).abs() > 0.1);
    assert!((pat.cdfs[3] - ex.cdfs[3]).abs() > 5e-3);
}

#[test]
fn quarterly_chain_ladder_matches_independent_hand_computation() {
    // 3 annual origins × 6 quarterly devs (ages 3..18m), valued 6 months into
    // origin 2 (valuation month 30): origin 0 & 1 see all 6 columns
    // (max ages 30, 18), origin 2 is a 6-month stub seeing 2 columns.
    let grid = asymmetric_age_grid(3, 6, 12.0, 3.0, 3.0, Some(6.0)).unwrap();
    assert_eq!(grid.observed_counts, vec![6, 6, 2]);
    assert_eq!(grid.latest_ages_months().unwrap(), vec![18.0, 18.0, 6.0]);
    assert_close(grid.exposure_fractions()[2], 0.5, 0.0, "stub exposure");

    let nan = f64::NAN;
    #[rustfmt::skip]
    let cum = [
        100.0, 150.0, 180.0, 200.0, 210.0, 215.0,
        120.0, 168.0, 210.0, 228.0, 240.0, 246.0,
         90.0, 126.0,   nan,   nan,   nan,   nan,
    ];
    let res = asymmetric_chain_ladder(
        &cum,
        &grid,
        LdfSelectionMethod::VolumeWeighted,
        1.0,
        MonotonicityPolicy::Strict,
    )
    .unwrap();

    // Volume-weighted links computed by hand:
    //   f0 = (150 + 168 + 126) / (100 + 120 + 90) = 444 / 310
    //   f1 = (180 + 210) / (150 + 168)            = 390 / 318
    //   f2 = (200 + 228) / (180 + 210)            = 428 / 390
    //   f3 = (210 + 240) / (200 + 228)            = 450 / 428
    //   f4 = (215 + 246) / (210 + 240)            = 461 / 450
    let expected_ldfs = [
        444.0 / 310.0,
        390.0 / 318.0,
        428.0 / 390.0,
        450.0 / 428.0,
        461.0 / 450.0,
    ];
    for (k, (&got, &exp)) in res.ldfs.iter().zip(expected_ldfs.iter()).enumerate() {
        assert_close(got, exp, 1e-14, &format!("ldf {k}"));
    }
    // CDF from column 1 telescopes: f1 f2 f3 f4 = 461 / 318.
    assert_close(res.cdfs[1], 461.0 / 318.0, 1e-14, "cdf[1]");
    assert_close(res.cdfs[5], 1.0, 0.0, "cdf[5]");
    // Only the stub origin carries IBNR: 126 * 461/318 - 126.
    let ibnr_2 = 126.0 * 461.0 / 318.0 - 126.0;
    assert_close(res.ibnr[0], 0.0, 1e-12, "ibnr 0");
    assert_close(res.ibnr[1], 0.0, 1e-12, "ibnr 1");
    assert_close(res.ibnr[2], ibnr_2, 1e-12, "ibnr 2");
    assert_close(res.total_ibnr, ibnr_2, 1e-12, "total ibnr");
    assert_eq!(res.latest_ages_months, vec![18.0, 18.0, 6.0]);
}

// ===========================================================================
// 2. Invariants
// ===========================================================================

#[test]
fn ivf12_invariants_knots_monotone_cdf_at_least_one() {
    let monthly: Vec<f64> = (12..=60).map(f64::from).collect();
    let pat = interp_ivf12(&ANNUAL_AGES, &ANNUAL_CDFS, &monthly).unwrap();
    let ex = interp_exponential(&ANNUAL_AGES, &ANNUAL_CDFS, &monthly).unwrap();
    for (k, &c) in ANNUAL_CDFS.iter().enumerate() {
        assert_close(pat.cdfs[12 * k], c, 1e-12, &format!("knot {k}"));
    }
    for w in pat.cdfs.windows(2) {
        assert!(w[0] >= w[1], "non-monotone: {} < {}", w[0], w[1]);
    }
    assert!(pat.cdfs.iter().all(|&c| c >= 1.0));
    // Differs from exponential at every off-knot month.
    for (i, (&a, &b)) in pat.cdfs.iter().zip(ex.cdfs.iter()).enumerate() {
        if i % 12 != 0 {
            assert!((a - b).abs() > 1e-6, "month {}: {a} vs {b}", monthly[i]);
        }
    }
    // Smooth emergence: ln CDF is C^1, so second differences of ln CDF on a
    // monthly grid stay small everywhere (no kinks at the annual knots).
    let ln: Vec<f64> = pat.cdfs.iter().map(|c| c.ln()).collect();
    let slopes: Vec<f64> = ln.windows(2).map(|w| w[1] - w[0]).collect();
    for k in (12..48).step_by(12) {
        let jump = (slopes[k] - slopes[k - 1]).abs();
        let scale = slopes[k].abs().max(slopes[k - 1].abs());
        assert!(jump < 0.25 * scale, "kink at month {}: {jump}", 12 + k);
    }
}

#[test]
fn every_interpolator_preserves_knots_on_monthly_grid_and_stays_at_least_one() {
    let monthly: Vec<f64> = (12..=60).map(f64::from).collect();
    for method in [
        SubannualInterpMethod::LinearCdf,
        SubannualInterpMethod::LinearPctReported,
        SubannualInterpMethod::Exponential,
        SubannualInterpMethod::InversePower,
        SubannualInterpMethod::Swimon,
        SubannualInterpMethod::SplicedWeibull,
        SubannualInterpMethod::Ivf12,
    ] {
        let pat = interpolate_pattern(method, &ANNUAL_AGES, &ANNUAL_CDFS, &monthly).unwrap();
        assert_eq!(pat.method, method);
        assert_eq!(pat.ldfs.len(), monthly.len() - 1);
        for (k, &c) in ANNUAL_CDFS.iter().enumerate() {
            assert_close(pat.cdfs[12 * k], c, 1e-10, &format!("{method:?} knot {k}"));
        }
        assert!(pat.cdfs.iter().all(|&c| c >= 1.0 - 1e-12), "{method:?}");
        for (p, c) in pat.pct_reported.iter().zip(pat.cdfs.iter()) {
            assert_close(p * c, 1.0, 1e-14, "pct * cdf");
        }
    }
}

#[test]
fn ivf12_single_window_and_flat_factor_edge_cases() {
    // Two knots (one window): phi runs from ln 1.5 to 0 along the quadratic
    // with zero end slope, i.e. Hermite tangents m_0 = 2 d_0, m_1 = 0 where
    // d_0 = -ln 1.5 / 12. Midpoint: phi(18) = ln1.5 / 2 + 12 (2 d_0) / 8
    // = ln1.5 (1/2 - 1/4) = ln1.5 / 4, so CDF(18) = 1.5^(1/4).
    let pat = interp_ivf12(&[12.0, 24.0], &[1.5, 1.0], &[12.0, 18.0, 24.0]).unwrap();
    assert_close(pat.cdfs[0], 1.5, 1e-14, "knot 12");
    assert_close(pat.cdfs[1], 1.5_f64.powf(0.25), 1e-14, "mid 18");
    assert_close(pat.cdfs[2], 1.0, 1e-14, "knot 24");

    // Fully developed pattern: every CDF identical -> flat output.
    let flat = interp_ivf12(&[12.0, 24.0, 36.0], &[1.0, 1.0, 1.0], &[15.0, 27.0]).unwrap();
    assert!(flat.cdfs.iter().all(|&c| (c - 1.0).abs() < 1e-15));

    // Unequal spacing and out-of-range targets are rejected.
    assert!(interp_ivf12(&[12.0, 24.0, 48.0], &[2.0, 1.5, 1.0], &[18.0]).is_err());
    assert!(interp_ivf12(&ANNUAL_AGES, &ANNUAL_CDFS, &[6.0]).is_err());
    assert!(interp_ivf12(&ANNUAL_AGES, &ANNUAL_CDFS, &[61.0]).is_err());
}

// ===========================================================================
// 3. Edge cases: stubs at boundaries, single period, NaN cells
// ===========================================================================

#[test]
fn stub_boundaries_and_single_period_grids() {
    // A full-length stub is identical to a full valuation.
    let full = asymmetric_age_grid(3, 8, 12.0, 6.0, 6.0, None).unwrap();
    let stub12 = asymmetric_age_grid(3, 8, 12.0, 6.0, 6.0, Some(12.0)).unwrap();
    // (Compared via JSON because NaN sentinels make `PartialEq` false.)
    assert_eq!(
        serde_json::to_string(&full).unwrap(),
        serde_json::to_string(&stub12).unwrap()
    );
    // Out-of-range stubs are rejected.
    for bad in [0.0, -1.0, 12.5, f64::NAN, f64::INFINITY] {
        assert!(asymmetric_age_grid(3, 8, 12.0, 6.0, 6.0, Some(bad)).is_err());
    }
    // A stub shorter than the first dev age leaves the latest origin with no
    // observed cells; `latest_ages_months` reports NaN instead of panicking.
    let tiny = asymmetric_age_grid(2, 4, 12.0, 6.0, 6.0, Some(1.0)).unwrap();
    assert_eq!(tiny.observed_counts, vec![2, 0]);
    let latest = tiny.latest_ages_months().unwrap();
    assert_eq!(latest[0], 12.0);
    assert!(latest[1].is_nan());
    assert_eq!(tiny.cell_age(1, 0), None);
    // ... and the reserving kernels refuse the empty row with an error.
    let cum = [
        100.0,
        150.0,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
    ];
    assert!(
        asymmetric_chain_ladder(
            &cum,
            &tiny,
            LdfSelectionMethod::VolumeWeighted,
            1.0,
            MonotonicityPolicy::Strict
        )
        .is_err()
    );

    // Single period / single cell.
    let one = asymmetric_age_grid(1, 1, 12.0, 12.0, 12.0, None).unwrap();
    assert_eq!(one.observed_counts, vec![1]);
    assert_eq!(one.cell_age(0, 0), Some(12.0));
    assert_eq!(one.cell_average_age(0, 0), Some(6.0));
    let map = calendar_diagonal_index(&one).unwrap();
    assert_eq!(map.n_diagonals(), 1);
    assert_eq!(map.diagonal_of(0, 0), Some(0));
    assert!(
        asymmetric_chain_ladder(
            &[100.0],
            &one,
            LdfSelectionMethod::VolumeWeighted,
            1.0,
            MonotonicityPolicy::Strict
        )
        .is_err()
    );
    let diag = diagnose_asymmetric_calendar_effects(&map, &[0.5]).unwrap();
    assert_eq!(diag.strongest_diagonal, Some(0));
    assert_close(diag.rms_diagonal_bias, 0.5, 0.0, "rms");

    // Overflowing dimensions return an error instead of panicking/aborting.
    assert_eq!(
        asymmetric_age_grid(usize::MAX, 2, 12.0, 6.0, 6.0, None),
        Err(ReservingError::ShapeMismatch)
    );
}

#[test]
fn nan_cells_are_rejected_when_observed_and_ignored_when_unobserved() {
    let grid = asymmetric_age_grid(2, 4, 12.0, 6.0, 6.0, None).unwrap();
    assert_eq!(grid.observed_counts, vec![4, 2]);
    let nan = f64::NAN;
    let ok = [100.0, 150.0, 170.0, 175.0, 110.0, 160.0, nan, nan];
    assert!(
        asymmetric_chain_ladder(
            &ok,
            &grid,
            LdfSelectionMethod::VolumeWeighted,
            1.0,
            MonotonicityPolicy::Strict
        )
        .is_ok()
    );
    let bad = [100.0, nan, 170.0, 175.0, 110.0, 160.0, nan, nan];
    assert_eq!(
        asymmetric_chain_ladder(
            &bad,
            &grid,
            LdfSelectionMethod::VolumeWeighted,
            1.0,
            MonotonicityPolicy::Strict
        ),
        Err(ReservingError::NotFinite)
    );

    // NaN residuals are skipped (counted out) by the calendar diagnostics.
    let map = calendar_diagonal_index(&grid).unwrap();
    let mut residuals = vec![1.0; 8];
    residuals[0] = nan;
    let diag = diagnose_asymmetric_calendar_effects(&map, &residuals).unwrap();
    let d0 = map.diagonal_of(0, 0).unwrap();
    assert_eq!(diag.diagonal_counts[d0], 0);
    assert!(diag.diagonal_means[d0].is_nan());

    // NaN anchors / targets in interpolation are rejected.
    assert!(interp_ivf12(&ANNUAL_AGES, &[2.5, nan, 1.18, 1.06, 1.0], &[18.0]).is_err());
    assert!(interp_swimon(&ANNUAL_AGES, &ANNUAL_CDFS, &[nan]).is_err());
}

// ===========================================================================
// 4. Malformed JSON and mutated structs -> Err, never panic
// ===========================================================================

#[test]
fn malformed_grid_json_is_rejected_without_panic() {
    let grid = asymmetric_age_grid(3, 4, 12.0, 6.0, 6.0, None).unwrap();
    let json = serde_json::to_string(&grid).unwrap();
    // Lossless roundtrip (NaN <-> null) still works.
    let back: AsymmetricAgeGrid = serde_json::from_str(&json).unwrap();
    assert_eq!(back.observed_counts, grid.observed_counts);
    assert_eq!(back.latest_ages_months(), grid.latest_ages_months());

    let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
    let mut short = value.clone();
    short["cell_ages_months"] = serde_json::json!([6.0, 12.0]);
    let mut big_count = value.clone();
    big_count["observed_counts"] = serde_json::json!([4, 9, 1]);
    let mut zero_dims = value.clone();
    zero_dims["n_devs"] = serde_json::json!(0);
    value["n_origins"] = serde_json::json!(usize::MAX);

    for bad in [short, big_count, zero_dims, value] {
        let s = bad.to_string();
        let res = catch_unwind(|| serde_json::from_str::<AsymmetricAgeGrid>(&s));
        assert!(matches!(res, Ok(Err(_))), "expected Err for {s}");
    }
}

#[test]
fn hand_built_calendar_map_json_with_out_of_range_cell_is_rejected() {
    // 1 origin × 2 devs; a diagonal cell claims origin = 5.
    let json = r#"{
        "n_origins": 1,
        "n_devs": 2,
        "diagonal_months": [12.0],
        "cell_diagonal": [0, null],
        "cell_calendar_months": [12.0, null],
        "diagonals": [[{"origin": 5, "dev": 0, "calendar_month": 12.0,
                        "diagonal_index": 0, "calendar_period_index": 0}]]
    }"#;
    let res = catch_unwind(|| serde_json::from_str::<CalendarDiagonalMap>(json));
    assert!(matches!(res, Ok(Err(_))));

    // dev == n_devs would alias into the next row under naive indexing.
    let alias = json
        .replace("\"origin\": 5", "\"origin\": 0")
        .replace("\"dev\": 0", "\"dev\": 2");
    assert!(serde_json::from_str::<CalendarDiagonalMap>(&alias).is_err());

    // Out-of-range bucket reference.
    let bad_bucket = json
        .replace("\"origin\": 5", "\"origin\": 0")
        .replace("[0, null]", "[3, null]");
    assert!(serde_json::from_str::<CalendarDiagonalMap>(&bad_bucket).is_err());

    // The valid version parses and diagnoses fine.
    let good = json.replace("\"origin\": 5", "\"origin\": 0");
    let map: CalendarDiagonalMap = serde_json::from_str(&good).unwrap();
    let diag = diagnose_asymmetric_calendar_effects(&map, &[2.0, f64::NAN]).unwrap();
    assert_close(diag.diagonal_means[0], 2.0, 0.0, "diag mean");
}

#[test]
fn mutated_public_fields_yield_errors_not_panics() {
    let grid = asymmetric_age_grid(3, 4, 12.0, 6.0, 6.0, None).unwrap();
    let map = calendar_diagonal_index(&grid).unwrap();

    // Short cell ages: accessors return None / Err.
    let mut g = grid.clone();
    g.cell_ages_months.truncate(2);
    let res = catch_unwind(AssertUnwindSafe(|| {
        (
            g.cell_age(2, 0),
            g.cell_average_age(2, 3),
            g.is_observed(7, 0),
            g.latest_ages_months(),
            calendar_diagonal_index(&g).is_err(),
        )
    }));
    let (a, b, c, latest, cal_err) = res.expect("no panic");
    assert_eq!(a, None);
    assert_eq!(b, None);
    assert!(!c);
    assert_eq!(latest, Err(ReservingError::ShapeMismatch));
    assert!(cal_err);

    // observed_counts exceeding n_devs.
    let mut g2 = grid.clone();
    g2.observed_counts[0] = 99;
    let cum = vec![100.0; 12];
    let res = catch_unwind(AssertUnwindSafe(|| {
        (
            g2.latest_ages_months().is_err(),
            asymmetric_chain_ladder(
                &cum,
                &g2,
                LdfSelectionMethod::VolumeWeighted,
                1.0,
                MonotonicityPolicy::Strict,
            )
            .is_err(),
            fit_asymmetric_odp(&cum, &g2, 1.0, MonotonicityPolicy::Strict).is_err(),
        )
    }));
    assert_eq!(res.expect("no panic"), (true, true, true));

    // Calendar map with an out-of-range cell injected after construction.
    let mut m = map.clone();
    let mut cell = m.diagonals[0][0];
    cell.origin = 5;
    m.diagonals[0].push(cell);
    let residuals = vec![0.0; 12];
    let res = catch_unwind(AssertUnwindSafe(|| {
        diagnose_asymmetric_calendar_effects(&m, &residuals)
    }));
    assert!(matches!(res, Ok(Err(_))));
    let res = catch_unwind(AssertUnwindSafe(|| m.diagonal_of(5, 0)));
    assert_eq!(res.expect("no panic"), None);
}

// ===========================================================================
// 5. Clark exact-age vs square stochastic Clark MLE
// ===========================================================================

#[test]
fn clark_exact_age_on_annual_grid_matches_square_clark_mle() {
    // On an annual × annual grid the average claim ages are 6, 18, 30, ... months
    // = 12 * (j + 0.5) years, i.e. Clark's default midpoint ages rescaled by 12.
    // omega and reserves are scale-invariant; theta scales by exactly 12.
    let tri = genins();
    let n = tri.n();
    let grid = asymmetric_age_grid(n, n, 12.0, 12.0, 12.0, None).unwrap();
    for j in 0..n {
        assert_eq!(grid.cell_average_age(0, j), Some(12.0 * (j as f64 + 0.5)));
    }

    let sq = fit_clark_ldf(
        &tri.view(),
        &ClarkOptions::loglogistic().with_covariance(ClarkCovariance::Expected),
    )
    .unwrap();
    let ex = fit_clark_ldf_exact_age(
        tri.data(),
        &grid.cell_average_ages_months,
        grid.n_origins,
        grid.n_devs,
        &grid.observed_counts,
        &ClarkExactAgeOptions::loglogistic(),
    )
    .unwrap();
    assert_close(ex.omega, sq.omega, 1e-4, "LDF omega");
    assert_close(
        ex.theta / 12.0,
        sq.theta,
        1e-4 * sq.theta,
        "LDF theta (years)",
    );
    assert_close(
        ex.total_reserve / sq.total_reserve,
        1.0,
        1e-5,
        "LDF total reserve ratio",
    );
    assert_close(
        ex.total_se_aggregate / sq.total_se_aggregate,
        1.0,
        1e-4,
        "LDF total SE ratio",
    );

    let prem = vec![10_000_000.0; n];
    let sq_cc = fit_clark_cape_cod(
        &tri.view(),
        &prem,
        &ClarkOptions::weibull().with_covariance(ClarkCovariance::Expected),
    )
    .unwrap();
    let ex_cc = fit_clark_cape_cod_exact_age(
        tri.data(),
        &grid.cell_average_ages_months,
        grid.n_origins,
        grid.n_devs,
        &grid.observed_counts,
        &prem,
        &ClarkExactAgeOptions::weibull(),
    )
    .unwrap();
    assert_close(ex_cc.omega, sq_cc.omega, 1e-4, "CC omega");
    assert_close(
        ex_cc.total_reserve / sq_cc.total_reserve,
        1.0,
        1e-5,
        "CC total reserve ratio",
    );
}

#[test]
fn clark_exact_age_rejects_overflowing_dimensions() {
    let res = catch_unwind(|| {
        clark_mle_exact_age(
            &[1.0],
            &[1.0],
            usize::MAX,
            2,
            &[1],
            None,
            &ClarkExactAgeOptions::default(),
        )
    });
    assert!(matches!(res, Ok(Err(ReservingError::ShapeMismatch))));
}

#[test]
fn asymmetric_odp_bootstrap_is_reproducible_with_house_seed() {
    let grid = asymmetric_age_grid(4, 8, 12.0, 6.0, 6.0, Some(6.0)).unwrap();
    assert_eq!(grid.observed_counts, vec![7, 5, 3, 1]);
    let ldfs = [1.6, 1.3, 1.15, 1.08, 1.04, 1.02, 1.01];
    let mut cum = vec![f64::NAN; 4 * 8];
    for i in 0..4 {
        let mut v = 400.0 + 50.0 * i as f64;
        for j in 0..grid.observed_counts[i] {
            if j > 0 {
                let jitter = 1.0 + 0.02 * (((i * 3 + j * 5) % 5) as f64 - 2.0);
                v *= (ldfs[j - 1] * jitter).max(1.001);
            }
            cum[i * 8 + j] = v;
        }
    }
    let opts = OdpBootstrapOptions::shapland(200, 42069);
    let a = asymmetric_odp_bootstrap(&cum, &grid, 1.0, opts).unwrap();
    let b = asymmetric_odp_bootstrap(&cum, &grid, 1.0, opts).unwrap();
    assert_eq!(a.simulated_totals, b.simulated_totals);
    assert_eq!(a.simulated_totals.len() + a.failed_iterations, 200);
}
