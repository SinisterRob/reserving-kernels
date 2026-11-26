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
    bf, brosius, cape_cod, cape_cod::CapeCodEngine, chain_ladder, ldf,
    ldf::DevelopmentFactorEngine, mack, triangle, types::Triangle, validate::ValidationError,
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
    Triangle::new(4, 4, data)
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
    for (a, b) in t.data.iter().zip(back.iter()) {
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
        (1700.0 / 1500.0 + 2000.0 / 1800.0) / 2.0,
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
        (1700.0 / 1500.0 + 2000.0 / 1800.0) / 2.0,
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
        Err(ValidationError::DivideByZero("premium"))
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
fn mack_checked_rejects_non_cumulative_triangle() {
    #[rustfmt::skip]
    let t = Triangle::new(4, 4, vec![
        1000.0, 1500.0, 1700.0, 1800.0,
        1200.0, 1100.0, 2000.0,    NAN,
        1500.0, 2100.0,    NAN,    NAN,
        1800.0,    NAN,    NAN,    NAN,
    ]);

    let result = mack::analyze(&t.view(), 1.0);
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
    let t = Triangle::new(6, 6, data);
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
