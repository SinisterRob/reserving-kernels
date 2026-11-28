//! Deterministic-layer regression tests.
//!
//! * Friedland (2010) US Industry Auto goldens for chain ladder, Cape Cod,
//!   Bornhuetter–Ferguson and Benktander. The crate's dataset reproduces the
//!   Friedland triangles, but the book's exhibits use judgmentally selected
//!   LDFs and tails, so the goldens below are **independently derived** with
//!   exact rational arithmetic (Python `fractions.Fraction`) from the
//!   all-year volume-weighted definition with `tail = 1.0`:
//!
//!   ```text
//!   f_j   = Σ_{i ≤ n−j−2} C_{i,j+1} / Σ_{i ≤ n−j−2} C_{i,j}
//!   CDF_j = Π_{k ≥ j} f_k,      p_i = 1 / CDF_{n−1−i}
//!   CL:     U_i = C_{i,n−1−i} · CDF_{n−1−i}
//!   CC ELR: Σ_i C_{i,n−1−i} / Σ_i P_i p_i
//!   BF:     U_i = C_{i,n−1−i} + (1 − p_i) · P_i · ELR
//!   BK(2):  U_i = C_{i,n−1−i} + (1 − p_i) · U_i^{BF}
//!   ```
//!
//!   then rounded to `f64` once. Tolerance is relative `1e-11`.
//! * Policy overrides and sub-unity tails for every LDF selection method.
//! * Zero-denominator / no-data links rejected with typed errors.
//! * Volume-weighted parity between the LDF engine and link diagnostics.
//! * Large-loss semantics, Mack σ (`n − 1`) consolidation, panic-free kernels.
//! * Separation (hand-computed 3×3), Double Chain Ladder limiting identity,
//!   Munich Chain Ladder structural identities.

use reserving_kernels::ReservingError;
use reserving_kernels::Triangle;
use reserving_kernels::datasets::friedland_us_industry_auto;
use reserving_kernels::deterministic::benktander::BenktanderEngine;
use reserving_kernels::deterministic::bf::{self, BornhuetterFergusonEngine};
use reserving_kernels::deterministic::cape_cod::CapeCodEngine;
use reserving_kernels::deterministic::chain_ladder::{self, ChainLadderEngine};
use reserving_kernels::deterministic::ldf::{
    self, DevelopmentFactorEngine, DevelopmentFactors, LdfSelectionMethod,
};
use reserving_kernels::deterministic::separation::{self, InflationProjection, SeparationMethod};
use reserving_kernels::deterministic::{double_chain_ladder, munich_chain_ladder};
use reserving_kernels::diagnostics::{actual_expected, link_ratio_diagnostics};
use reserving_kernels::preprocessing::large_loss::{self, ClaimHistory};
use reserving_kernels::stochastic::mack;
use reserving_kernels::types::TriangleView;
use reserving_kernels::validate::MonotonicityPolicy;

const STRICT: MonotonicityPolicy = MonotonicityPolicy::Strict;
const ALLOW: MonotonicityPolicy = MonotonicityPolicy::AllowNegativeIncrements;

fn rel_close(actual: f64, expected: f64, rel: f64) -> bool {
    (actual - expected).abs() <= rel * expected.abs().max(1.0)
}

fn assert_vec_close(actual: &[f64], expected: &[f64], rel: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (k, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(rel_close(a, e, rel), "{what}[{k}]: {a} vs {e}");
    }
}

// ===========================================================================
// Friedland (2010) US Industry Auto goldens
// ===========================================================================

const GOLDEN_REL: f64 = 1e-11;

const FRIEDLAND_PAID_LDFS: [f64; 9] = [
    1.733_499_610_081_612_2,
    1.191_757_550_974_765_5,
    1.091_283_300_828_943_3,
    1.043_906_451_838_27,
    1.018_904_301_061_238_2,
    1.009_529_248_336_613_4,
    1.004_782_030_125_494_2,
    1.002_342_266_926_490_3,
    1.001_865_842_691_11,
];

const FRIEDLAND_PAID_CL_ULT: [f64; 10] = [
    47_644_187.0,
    51_095_692.973_606_6,
    54_762_945.115_243_115,
    56_382_145.645_384_274,
    58_884_152.169_994_69,
    58_049_658.976_143_286,
    58_262_501.479_328_48,
    59_880_496.196_972_03,
    61_445_202.386_218_34,
    66_513_174.175_348_72,
];

const FRIEDLAND_REPORTED_LDFS: [f64; 9] = [
    1.175_116_582_918_348_8,
    1.058_232_933_629_448_5,
    1.027_177_295_924_067_7,
    1.011_040_734_719_573,
    1.004_364_077_745_332_2,
    1.002_608_674_619_641,
    1.001_598_165_951_552_5,
    1.000_579_141_625_114,
    1.000_369_326_745_379,
];

const FRIEDLAND_REPORTED_CL_ULT: [f64; 10] = [
    47_742_304.0,
    51_204_671.272_735_834,
    54_889_952.770_581_1,
    56_443_033.797_635_07,
    58_895_266.415_834_07,
    58_115_110.265_397_7,
    58_155_873.034_459_94,
    59_536_930.555_043_93,
    60_624_006.361_067_72,
    63_694_289.638_746_28,
];

const FRIEDLAND_CAPE_COD_ELR: f64 = 0.695_448_447_956_837_5;

const FRIEDLAND_BF_ULT: [f64; 10] = [
    47_742_304.0,
    51_203_366.121_703_185,
    54_883_348.611_072_6,
    56_427_803.429_023_28,
    58_875_767.502_947_1,
    58_135_364.088_654_734,
    58_270_658.611_323_08,
    59_810_214.316_659_324,
    61_179_893.068_544_485,
    64_275_827.708_899_006,
];

const FRIEDLAND_BK2_ULT: [f64; 10] = [
    47_742_304.0,
    51_204_670.790_886_61,
    54_889_946.511_270_21,
    56_442_995.083_763_465,
    58_895_166.246_795_76,
    58_115_301.865_632_49,
    58_158_200.520_674_124,
    59_549_555.871_745_26,
    60_678_863.924_282_28,
    63_829_787.617_891_82,
];

#[test]
fn friedland_chain_ladder_goldens() {
    let auto = friedland_us_industry_auto();
    let paid = DevelopmentFactorEngine::volume_weighted(&auto.paid.view(), 1.0).expect("paid");
    let rep =
        DevelopmentFactorEngine::volume_weighted(&auto.reported.view(), 1.0).expect("reported");
    assert_vec_close(&paid.ldfs, &FRIEDLAND_PAID_LDFS, GOLDEN_REL, "paid ldf");
    assert_vec_close(&rep.ldfs, &FRIEDLAND_REPORTED_LDFS, GOLDEN_REL, "rep ldf");

    let cl_paid = ChainLadderEngine::project(&auto.paid.view(), &paid).expect("cl paid");
    let cl_rep = ChainLadderEngine::project(&auto.reported.view(), &rep).expect("cl rep");
    assert_vec_close(
        &cl_paid.ultimates,
        &FRIEDLAND_PAID_CL_ULT,
        GOLDEN_REL,
        "paid ult",
    );
    assert_vec_close(
        &cl_rep.ultimates,
        &FRIEDLAND_REPORTED_CL_ULT,
        GOLDEN_REL,
        "rep ult",
    );
    assert!(rel_close(
        cl_paid.total_reserve,
        74_869_788.118_239_54,
        GOLDEN_REL
    ));
    assert!(rel_close(
        cl_rep.total_reserve,
        25_819_851.111_501_623,
        GOLDEN_REL
    ));
}

#[test]
fn friedland_cape_cod_bf_benktander_goldens() {
    let auto = friedland_us_industry_auto();
    let view = auto.reported.view();
    let factors = DevelopmentFactorEngine::volume_weighted(&view, 1.0).expect("factors");

    let cc = CapeCodEngine::project(&view, &factors, &auto.earned_premium).expect("cape cod");
    assert!(rel_close(cc.elr, FRIEDLAND_CAPE_COD_ELR, GOLDEN_REL));
    assert_vec_close(
        &cc.bf_projection.ultimates,
        &FRIEDLAND_BF_ULT,
        GOLDEN_REL,
        "cape cod ult",
    );

    let bf = BornhuetterFergusonEngine::project_from_scalar_elr(
        &view,
        &factors,
        &auto.earned_premium,
        FRIEDLAND_CAPE_COD_ELR,
    )
    .expect("bf");
    assert_vec_close(&bf.ultimates, &FRIEDLAND_BF_ULT, GOLDEN_REL, "bf ult");
    assert!(rel_close(
        bf.total_ultimate,
        570_804_547.458_826_8,
        GOLDEN_REL
    ));

    let bk = BenktanderEngine::project_from_scalar_elr(
        &view,
        &factors,
        &auto.earned_premium,
        FRIEDLAND_CAPE_COD_ELR,
        2,
    )
    .expect("benktander");
    assert_vec_close(&bk.ultimates, &FRIEDLAND_BK2_ULT, GOLDEN_REL, "bk2 ult");
}

// ===========================================================================
// Policy overrides and sub-unity tails for every LDF method
// ===========================================================================

/// 4×4 incurred-style triangle with a downward case-reserve revision in origin 0.
fn incurred_with_decrease() -> Triangle {
    #[rustfmt::skip]
    let data = vec![
        1000.0, 1500.0, 1450.0, 1460.0,
        1200.0, 1700.0, 1690.0, f64::NAN,
        1400.0, 2000.0, f64::NAN, f64::NAN,
        1500.0, f64::NAN, f64::NAN, f64::NAN,
    ];
    Triangle::try_new(4, 4, data).expect("valid triangle")
}

fn computed_methods() -> Vec<LdfSelectionMethod> {
    vec![
        LdfSelectionMethod::VolumeWeighted,
        LdfSelectionMethod::SimpleAverage,
        LdfSelectionMethod::Median,
        LdfSelectionMethod::ExHiLo,
        LdfSelectionMethod::LastNVolumeWeighted { last_n: 2 },
        LdfSelectionMethod::LastNSimpleAverage { last_n: 2 },
        LdfSelectionMethod::Geometric,
    ]
}

fn l1_ldfs(t: &TriangleView<'_>, method: LdfSelectionMethod) -> Vec<f64> {
    match method {
        LdfSelectionMethod::VolumeWeighted => ldf::volume_weighted(t),
        LdfSelectionMethod::SimpleAverage => ldf::simple_average(t),
        LdfSelectionMethod::Median => ldf::median(t),
        LdfSelectionMethod::ExHiLo => ldf::ex_hi_lo(t),
        LdfSelectionMethod::Geometric => ldf::geometric_average(t),
        LdfSelectionMethod::LastNVolumeWeighted { last_n } => {
            ldf::volume_weighted_last_n(t, last_n)
        }
        LdfSelectionMethod::LastNSimpleAverage { last_n } => ldf::simple_average_last_n(t, last_n),
        LdfSelectionMethod::Selected => Vec::new(),
    }
}

type PolicyFn =
    fn(&TriangleView<'_>, f64, MonotonicityPolicy) -> Result<DevelopmentFactors, ReservingError>;

fn named_policy_entry_points() -> Vec<(&'static str, PolicyFn)> {
    vec![
        (
            "volume_weighted",
            DevelopmentFactorEngine::volume_weighted_with_policy,
        ),
        (
            "simple_average",
            DevelopmentFactorEngine::simple_average_with_policy,
        ),
        ("median", DevelopmentFactorEngine::median_with_policy),
        ("ex_hi_lo", DevelopmentFactorEngine::ex_hi_lo_with_policy),
        ("vw_last_2", |t, tail, p| {
            DevelopmentFactorEngine::volume_weighted_last_n_with_policy(t, tail, 2, p)
        }),
        ("sa_last_2", |t, tail, p| {
            DevelopmentFactorEngine::simple_average_last_n_with_policy(t, tail, 2, p)
        }),
        (
            "geometric",
            DevelopmentFactorEngine::geometric_average_with_policy,
        ),
    ]
}

#[test]
fn every_ldf_method_honours_policy_override() {
    let tri = incurred_with_decrease();
    let view = tri.view();
    let non_monotone = ReservingError::NonMonotoneCumulative {
        origin: 0,
        dev: 2,
        previous: 1500.0,
        value: 1450.0,
    };

    for method in computed_methods() {
        assert_eq!(
            DevelopmentFactorEngine::select_with_policy(&view, 1.0, method, STRICT),
            Err(non_monotone),
            "{method:?} strict"
        );
        let allowed = DevelopmentFactorEngine::select_with_policy(&view, 1.0, method, ALLOW)
            .unwrap_or_else(|e| panic!("{method:?} allow: {e}"));
        assert_eq!(allowed.ldfs, l1_ldfs(&view, method), "{method:?} ldfs");
    }

    for (name, entry) in named_policy_entry_points() {
        assert_eq!(
            entry(&view, 1.0, STRICT),
            Err(non_monotone),
            "{name} strict"
        );
        let f = entry(&view, 1.0, ALLOW).unwrap_or_else(|e| panic!("{name} allow: {e}"));
        assert!(f.ldfs[1] < 1.0, "{name}: downward link must be preserved");
    }
    // Strict convenience wrappers keep their historical behaviour.
    assert_eq!(
        DevelopmentFactorEngine::geometric_average(&view, 1.0),
        Err(non_monotone)
    );
    assert_eq!(
        DevelopmentFactorEngine::volume_weighted_diagnostics(&view, 1.0),
        Err(non_monotone)
    );
    assert!(
        DevelopmentFactorEngine::volume_weighted_diagnostics_with_policy(&view, 1.0, ALLOW).is_ok()
    );
}

#[test]
fn sub_unity_tails_accepted_only_under_allow_negative_increments() {
    let tri = incurred_with_decrease();
    let view = tri.view();
    for (name, entry) in named_policy_entry_points() {
        let f = entry(&view, 0.97, ALLOW).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(f.tail, 0.97);
        assert_eq!(f.cdfs[3], 0.97);
        for bad in [0.0, -1.0] {
            assert_eq!(
                entry(&view, bad, ALLOW),
                Err(ReservingError::InvalidTail { tail: bad }),
                "{name} tail {bad}"
            );
        }
        assert_eq!(
            entry(&view, f64::NAN, ALLOW),
            Err(ReservingError::NotFinite),
            "{name} NaN tail"
        );
    }
    // Strict monotone triangle: tail < 1 rejected under Strict, accepted under Allow.
    let auto = friedland_us_industry_auto();
    assert_eq!(
        DevelopmentFactorEngine::median_with_policy(&auto.paid.view(), 0.99, STRICT),
        Err(ReservingError::InvalidTail { tail: 0.99 })
    );
    let f =
        DevelopmentFactorEngine::median_with_policy(&auto.paid.view(), 0.99, ALLOW).expect("allow");
    let cl = ChainLadderEngine::project_with_policy(&auto.paid.view(), &f, ALLOW).expect("cl");
    assert!(rel_close(cl.ultimates[0], 47_644_187.0 * 0.99, 1e-14));
}

#[test]
fn selected_method_requires_explicit_factors() {
    let tri = incurred_with_decrease();
    assert!(matches!(
        DevelopmentFactorEngine::select_with_policy(
            &tri.view(),
            1.0,
            LdfSelectionMethod::Selected,
            ALLOW
        ),
        Err(ReservingError::InvalidParameter(_))
    ));
    let f = DevelopmentFactorEngine::from_selected_ldfs(&[1.4, 1.0, 1.01], 1.02, STRICT)
        .expect("selected");
    assert_vec_close(
        &f.cdfs,
        &[1.4 * 1.0302, 1.0302, 1.0302, 1.02],
        1e-14,
        "cdfs",
    );
    assert_eq!(
        DevelopmentFactorEngine::from_selected_ldfs(&[1.4, 0.0], 1.0, ALLOW),
        Err(ReservingError::DivideByZero("development denominator"))
    );
    assert_eq!(
        DevelopmentFactorEngine::from_selected_ldfs(&[], 1.0, ALLOW),
        Err(ReservingError::ShapeMismatch)
    );
    assert!(matches!(
        link_ratio_diagnostics::diagnose_all(&tri.view(), LdfSelectionMethod::Selected),
        Err(ReservingError::InvalidParameter(_))
    ));
}

// ===========================================================================
// Zero-denominator and no-data links
// ===========================================================================

/// First development column is all zero on every observed link row.
fn zero_first_column() -> Triangle {
    #[rustfmt::skip]
    let data = vec![
        0.0, 10.0, 12.0,
        0.0,  8.0, f64::NAN,
        5.0, f64::NAN, f64::NAN,
    ];
    Triangle::try_new(3, 3, data).expect("valid triangle")
}

#[test]
fn zero_denominator_links_error_for_every_method() {
    let tri = zero_first_column();
    let view = tri.view();
    for (name, entry) in named_policy_entry_points() {
        assert!(entry(&view, 1.0, STRICT).is_err(), "{name} must reject");
    }
    for method in computed_methods() {
        assert!(
            l1_ldfs(&view, method)[0].is_nan(),
            "{method:?} L1 should emit NaN"
        );
    }
    assert_eq!(
        DevelopmentFactorEngine::volume_weighted_diagnostics(&view, 1.0),
        Err(ReservingError::DivideByZero("development denominator"))
    );
}

#[test]
fn geometric_average_without_positive_pairs_is_nan_and_engine_errors() {
    let tri = zero_first_column();
    let geo = ldf::geometric_average(&tri.view());
    assert!(geo[0].is_nan(), "no positive pair must not default to 1.0");
    assert!((geo[1] - 1.2).abs() < 1e-15);
    assert_eq!(
        DevelopmentFactorEngine::geometric_average(&tri.view(), 1.0),
        Err(ReservingError::NotFinite)
    );
}

// ===========================================================================
// Volume-weighted parity: engine vs link diagnostics
// ===========================================================================

#[test]
fn volume_weighted_engine_and_diagnostics_are_bit_identical() {
    #[rustfmt::skip]
    let zero_row = Triangle::try_new(4, 4, vec![
        100.0, 150.0, 170.0, 180.0,
        120.0, 180.0, 200.0, f64::NAN,
          0.0,  50.0, f64::NAN, f64::NAN,
        140.0, f64::NAN, f64::NAN, f64::NAN,
    ]).expect("valid triangle");
    let auto = friedland_us_industry_auto();
    for tri in [&zero_row, &auto.paid, &auto.reported] {
        let engine = DevelopmentFactorEngine::volume_weighted(&tri.view(), 1.0).expect("engine");
        let diag =
            link_ratio_diagnostics::diagnose_all(&tri.view(), LdfSelectionMethod::VolumeWeighted)
                .expect("diag");
        assert_eq!(diag.len(), engine.ldfs.len());
        for (d, &f) in diag.iter().zip(&engine.ldfs) {
            assert_eq!(d.volume_weighted.to_bits(), f.to_bits());
        }
    }
    // The 0 -> 50 row: 380/220 (engine definition), not 330/220.
    let diag = link_ratio_diagnostics::diagnose_link(
        &zero_row.view(),
        0,
        LdfSelectionMethod::VolumeWeighted,
    )
    .expect("diag");
    assert!((diag.volume_weighted - 380.0 / 220.0).abs() < 1e-15);
}

// ===========================================================================
// Mack σ (n − 1) consolidation
// ===========================================================================

#[test]
fn residual_scale_matches_mack_sigma_where_estimable() {
    let auto = friedland_us_industry_auto();
    let view = auto.paid.view();
    let ldfs = ldf::volume_weighted(&view);
    let sigma = actual_expected::residual_scale_by_dev(&view, &ldfs);
    let mack = mack::analyze(&view, 1.0).expect("mack");
    // Links 0..n-3 have >= 2 pairs: identical to Mack (1993) σ_k.
    for k in 0..ldfs.len() - 1 {
        assert!(
            rel_close(sigma[k] * sigma[k], mack.sigma_sq[k], 1e-10),
            "link {k}: {} vs {}",
            sigma[k] * sigma[k],
            mack.sigma_sq[k]
        );
    }
    // Final link has one pair: undefined (no silent n-denominator estimate).
    assert!(sigma[ldfs.len() - 1].is_nan());
}

#[test]
fn munich_sigma_uses_shared_mack_estimator() {
    let qm = reserving_kernels::datasets::quarg_mack_2004();
    let mcl =
        munich_chain_ladder::project(&qm.paid.view(), &qm.incurred.view(), 1.0, 1.0).expect("mcl");
    let sigma = actual_expected::residual_scale_by_dev(&qm.paid.view(), &mcl.paid_ldfs);
    for k in 0..sigma.len() - 1 {
        assert!(rel_close(mcl.sigma_paid[k], sigma[k], 1e-12), "link {k}");
    }
    // Last link: Mack min-rule extrapolation, finite and non-negative.
    let last = mcl.sigma_paid[sigma.len() - 1];
    assert!(last.is_finite() && last >= 0.0);
}

// ===========================================================================
// Large-loss semantics
// ===========================================================================

#[test]
fn large_loss_claim_level_limiting_differs_from_aggregate_cap() {
    // AY0: one 900k claim + one 80k claim; AY1: one 120k claim; limit 500k.
    let a = [200.0, 900.0];
    let b = [50.0, 80.0];
    let c = [120.0];
    let claims = [
        ClaimHistory {
            origin: 0,
            cumulative: &a,
        },
        ClaimHistory {
            origin: 0,
            cumulative: &b,
        },
        ClaimHistory {
            origin: 1,
            cumulative: &c,
        },
    ];
    let split = large_loss::aggregate_claims_split(&claims, 2, 500.0).expect("split");
    let aggregate =
        Triangle::try_new(2, 2, vec![250.0, 980.0, 120.0, f64::NAN]).expect("valid triangle");
    // Claim-level limiting keeps the small claim's 80 in the limited triangle.
    assert_eq!(split.limited.data()[1], 580.0);
    assert_eq!(split.excess.data()[1], 400.0);
    // Aggregate-safe removal: aggregate − supplied excess.
    let removed =
        large_loss::remove_large_loss_excess(&aggregate.view(), &split.excess.view()).expect("rm");
    assert_eq!(removed.data()[1], 580.0);
    // Naive cellwise cap on an aggregate triangle caps the accident year instead.
    let naive = large_loss::cap_triangle(&aggregate.view(), 500.0).expect("cap");
    assert_eq!(naive.data()[1], 500.0);
}

// ===========================================================================
// Panic-free L0 kernels
// ===========================================================================

#[test]
fn l0_kernels_return_errors_instead_of_panicking() {
    let mut out2 = [0.0; 2];
    let mut out3 = [0.0; 3];
    assert_eq!(
        chain_ladder::project_ultimates(&[1.0, 2.0], &[0, 9], &[1.0, 1.0], &mut out2),
        Err(ReservingError::ShapeMismatch)
    );
    assert_eq!(
        chain_ladder::reserves(&[1.0, 2.0], &[1.0], &mut out2),
        Err(ReservingError::ShapeMismatch)
    );
    let mut dev = [0_usize; 3];
    assert_eq!(
        chain_ladder::latest_observed_by_origin(&[0.0; 4], 2, 2, &mut out3, &mut dev),
        Err(ReservingError::ShapeMismatch)
    );
    assert_eq!(
        ldf::try_cdfs_from_ldfs(&[1.1, 1.2], 1.0, &mut out2),
        Err(ReservingError::ShapeMismatch)
    );
    assert_eq!(
        ldf::try_volume_weighted_ldfs(&[1.0; 8], 3, 3, &mut out2),
        Err(ReservingError::ShapeMismatch)
    );
    assert_eq!(
        ldf::try_volume_weighted_ldfs_observed_counts(&[1.0; 9], 3, 3, &[3, 2], &mut out2),
        Err(ReservingError::ShapeMismatch)
    );
    assert_eq!(
        ldf::volume_weighted_ldf(&[1.0; 9], 3, 3, 2),
        Err(ReservingError::ShapeMismatch)
    );
    assert_eq!(
        ldf::latest_maturity_percentages(3, 3, &[1.0; 2], &mut out3),
        Err(ReservingError::ShapeMismatch)
    );
    assert_eq!(
        bf::bf_reserves(&[1.0, 2.0], &[0.5], &mut out2),
        Err(ReservingError::ShapeMismatch)
    );
    assert_eq!(
        bf::bf_ultimates(&[1.0], &[0.5], &mut out2),
        Err(ReservingError::ShapeMismatch)
    );
    assert_eq!(
        bf::expected_ultimate_from_elr(&[1.0, 2.0], &[0.5, 0.6], &mut out3),
        Err(ReservingError::ShapeMismatch)
    );

    // Legacy unit-returning kernels fill NaN on shape mismatch.
    ldf::cdfs_from_ldfs(&[1.1, 1.2], 1.0, &mut out2);
    assert!(out2.iter().all(|v| v.is_nan()));
    ldf::volume_weighted_ldfs(&[1.0; 8], 3, 3, &mut out2);
    assert!(out2.iter().all(|v| v.is_nan()));

    // Unchecked L1 projections degrade to NaN instead of indexing out of bounds.
    let tri = Triangle::try_new(
        3,
        3,
        vec![
            100.0,
            150.0,
            180.0,
            120.0,
            180.0,
            f64::NAN,
            140.0,
            f64::NAN,
            f64::NAN,
        ],
    )
    .expect("valid triangle");
    let bf_short = bf::project_unchecked(&tri.view(), &[1.5], 1.0, &[200.0]);
    assert!(bf_short.ultimates[0].is_nan());
    assert!(
        reserving_kernels::deterministic::cape_cod::elr(&tri.view(), &[1.5], 1.0, &[1.0; 3])
            .is_nan()
    );
}

// ===========================================================================
// Separation — hand-computed 3×3 (Taylor 1977 arithmetic recursion)
// ===========================================================================

#[test]
fn arithmetic_separation_hand_computed_3x3() {
    // Incremental S (exposure 1): [50, 30, 20], [60, 40], [70].
    // Diagonal sums d = [50, 90, 130]; column sums v = [180, 70, 20].
    // k=2: λ2 = 130/1 = 130;              r2 = 20/130 = 2/13
    // k=1: λ1 = 90/(1 − 2/13) = 1170/11;  r1 = 70/(130 + 1170/11) = 77/260
    // k=0: λ0 = 50/(1 − 117/260) = 1000/11; r0 = 180/(3600/11) = 11/20
    // Zero future inflation (λ3 = λ4 = 130):
    //   R1 = r2·λ3 = 20;  R2 = r1·λ3 + r2·λ4 = 38.5 + 20 = 58.5; total 78.5.
    let tri = Triangle::try_from_incremental_rows(&[&[50.0, 30.0, 20.0], &[60.0, 40.0], &[70.0]])
        .expect("triangle");
    let res = separation::analyze(
        &tri.view(),
        &[1.0, 1.0, 1.0],
        SeparationMethod::Arithmetic,
        &InflationProjection::ConstantRate(0.0),
    )
    .expect("separation");
    assert_vec_close(
        &res.delay_proportions,
        &[11.0 / 20.0, 77.0 / 260.0, 2.0 / 13.0],
        1e-14,
        "r",
    );
    assert_vec_close(
        &res.calendar_indices,
        &[1000.0 / 11.0, 1170.0 / 11.0, 130.0],
        1e-14,
        "lambda",
    );
    assert_vec_close(
        &res.future_calendar_indices,
        &[130.0, 130.0],
        1e-14,
        "future",
    );
    assert_vec_close(&res.reserve, &[0.0, 20.0, 58.5], 1e-13, "reserve");
    assert!((res.total_reserve - 78.5).abs() < 1e-12);
    assert_vec_close(&res.ultimate, &[100.0, 120.0, 128.5], 1e-13, "ultimate");
    assert_vec_close(
        &res.implied_inflation_rates,
        &[1170.0 / 1000.0 - 1.0, 130.0 * 11.0 / 1170.0 - 1.0],
        1e-13,
        "inflation",
    );
}

#[test]
fn separation_rejects_bad_inputs() {
    let tri = Triangle::try_from_incremental_rows(&[&[50.0, 30.0, 20.0], &[60.0, 40.0], &[70.0]])
        .expect("triangle");
    let view = tri.view();
    let arith = SeparationMethod::Arithmetic;
    assert_eq!(
        separation::analyze(
            &view,
            &[1.0, 1.0],
            arith,
            &InflationProjection::ConstantRate(0.0)
        )
        .map(|r| r.total_reserve),
        Err(ReservingError::ShapeMismatch)
    );
    assert!(
        separation::analyze(
            &view,
            &[1.0, 0.0, 1.0],
            arith,
            &InflationProjection::ConstantRate(0.0)
        )
        .is_err()
    );
    assert!(
        separation::analyze(
            &view,
            &[1.0; 3],
            arith,
            &InflationProjection::ConstantRate(-1.0)
        )
        .is_err()
    );
    assert_eq!(
        separation::analyze(
            &view,
            &[1.0; 3],
            arith,
            &InflationProjection::ExplicitFutureIndices(vec![1.0])
        )
        .map(|r| r.total_reserve),
        Err(ReservingError::ShapeMismatch)
    );
    // Explicit indices are applied verbatim: R2 = r1·λ3 + r2·λ4.
    let explicit = separation::analyze(
        &view,
        &[1.0; 3],
        arith,
        &InflationProjection::ExplicitFutureIndices(vec![140.0, 150.0]),
    )
    .expect("explicit");
    let r2 = 77.0 / 260.0 * 140.0 + 2.0 / 13.0 * 150.0;
    assert!((explicit.reserve[2] - r2).abs() < 1e-12);
}

// ===========================================================================
// Double Chain Ladder — limiting identity vs chain ladder
// ===========================================================================

#[test]
fn dcl_with_immediate_reporting_reproduces_paid_chain_ladder() {
    // With every claim reported at development 0 (constant cumulative counts),
    // β = (1, 0, …, 0), the deconvolved payment delay equals the paid
    // development proportions, IBNR vanishes, and RBNS = paid CL reserve
    // (Martínez-Miranda, Nielsen & Verrall 2012, §3 limiting case).
    let auto = friedland_us_industry_auto();
    let n = 10;
    let mut counts = vec![f64::NAN; n * n];
    for i in 0..n {
        for j in 0..(n - i) {
            counts[i * n + j] = 1_000.0 + 10.0 * i as f64;
        }
    }
    let counts = Triangle::try_new(n, n, counts).expect("valid triangle");
    let dcl = double_chain_ladder::analyze(&auto.paid.view(), &counts.view()).expect("dcl");
    let cl_factors = DevelopmentFactorEngine::volume_weighted(&auto.paid.view(), 1.0).expect("f");
    let cl = ChainLadderEngine::project(&auto.paid.view(), &cl_factors).expect("cl");

    assert!((dcl.reporting_delay[0] - 1.0).abs() < 1e-15);
    assert!(dcl.reporting_delay[1..].iter().all(|b| b.abs() < 1e-15));
    assert_vec_close(&dcl.payment_delay, &dcl.paid_delay, 1e-12, "pi == beta~");
    assert!(dcl.total_ibnr.abs() < 1e-6);
    assert!(dcl.total_tail_reserve.abs() < 1e-6);
    assert_vec_close(&dcl.rbns_reserve, &cl.reserves, 1e-10, "rbns == cl reserve");
    assert!(rel_close(
        dcl.total_in_triangle_reserve,
        cl.total_reserve,
        1e-10
    ));
    assert!(rel_close(
        dcl.paid_cl_total_reserve,
        cl.total_reserve,
        1e-12
    ));
}

#[test]
fn dcl_rejects_mismatched_and_non_monotone_inputs() {
    let auto = friedland_us_industry_auto();
    let small = Triangle::try_new(2, 2, vec![1.0, 2.0, 3.0, f64::NAN]).expect("valid triangle");
    assert_eq!(
        double_chain_ladder::analyze(&auto.paid.view(), &small.view()).map(|d| d.total_rbns),
        Err(ReservingError::ShapeMismatch)
    );
    let tri = incurred_with_decrease();
    let counts = Triangle::try_new(
        4,
        4,
        vec![
            10.0,
            12.0,
            13.0,
            13.0,
            11.0,
            13.0,
            14.0,
            f64::NAN,
            12.0,
            14.0,
            f64::NAN,
            f64::NAN,
            13.0,
            f64::NAN,
            f64::NAN,
            f64::NAN,
        ],
    )
    .expect("valid triangle");
    assert!(matches!(
        double_chain_ladder::analyze(&tri.view(), &counts.view()),
        Err(ReservingError::NonMonotoneCumulative { .. })
    ));
    assert!(double_chain_ladder::analyze_with_policy(&tri.view(), &counts.view(), ALLOW).is_ok());
}

// ===========================================================================
// Munich Chain Ladder — structural identities
// ===========================================================================

#[test]
fn munich_with_proportional_paid_incurred_collapses_to_chain_ladder() {
    // P = 0.5 · I cellwise (exact binary scaling): every P/I residual is zero, so λ^P and λ^I have
    // zero regression denominators (λ = 0) and MCL equals univariate CL.
    let auto = friedland_us_industry_auto();
    let inc = &auto.reported;
    let paid_data: Vec<f64> = inc.data().iter().map(|v| 0.5 * v).collect();
    let paid = Triangle::try_new(10, 10, paid_data).expect("valid triangle");
    let mcl = munich_chain_ladder::project(&paid.view(), &inc.view(), 1.0, 1.0).expect("mcl");
    assert_eq!(mcl.lambda_paid, 0.0);
    assert_eq!(mcl.lambda_incurred, 0.0);
    assert_vec_close(
        &mcl.paid_ultimates,
        &mcl.traditional_paid_ultimates,
        1e-12,
        "paid",
    );
    assert_vec_close(
        &mcl.incurred_ultimates,
        &mcl.traditional_incurred_ultimates,
        1e-12,
        "incurred",
    );
    assert_vec_close(
        &mcl.incurred_ultimates,
        &FRIEDLAND_REPORTED_CL_ULT,
        1e-11,
        "golden",
    );
}

#[test]
fn munich_tail_scales_ultimates_and_validates_inputs() {
    let qm = reserving_kernels::datasets::quarg_mack_2004();
    let base =
        munich_chain_ladder::project(&qm.paid.view(), &qm.incurred.view(), 1.0, 1.0).expect("base");
    let tailed = munich_chain_ladder::project(&qm.paid.view(), &qm.incurred.view(), 1.05, 1.02)
        .expect("tailed");
    for i in 0..7 {
        assert!(rel_close(
            tailed.paid_ultimates[i],
            base.paid_ultimates[i] * 1.05,
            1e-14
        ));
        assert!(rel_close(
            tailed.incurred_ultimates[i],
            base.incurred_ultimates[i] * 1.02,
            1e-14
        ));
    }
    assert_eq!(tailed.lambda_paid, base.lambda_paid);

    let small = Triangle::try_new(2, 2, vec![1.0, 2.0, 3.0, f64::NAN]).expect("valid triangle");
    assert!(matches!(
        munich_chain_ladder::project(&small.view(), &small.view(), 1.0, 1.0),
        Err(ReservingError::InvalidParameter(_))
    ));
    assert_eq!(
        munich_chain_ladder::project(&qm.paid.view(), &small.view(), 1.0, 1.0)
            .map(|m| m.lambda_paid),
        Err(ReservingError::ShapeMismatch)
    );
    let mut zero_paid = qm.paid.data().to_vec();
    zero_paid[0] = 0.0;
    let zero_paid = Triangle::try_new(7, 7, zero_paid).expect("valid triangle");
    assert_eq!(
        munich_chain_ladder::project(&zero_paid.view(), &qm.incurred.view(), 1.0, 1.0)
            .map(|m| m.lambda_paid),
        Err(ReservingError::DivideByZero("paid cell"))
    );
    assert_eq!(
        munich_chain_ladder::project(&qm.paid.view(), &qm.incurred.view(), 0.9, 1.0)
            .map(|m| m.lambda_paid),
        Err(ReservingError::InvalidTail { tail: 0.9 })
    );
}

#[test]
fn geometric_selection_variant_matches_engine_and_diagnostics() {
    use reserving_kernels::deterministic::ldf::{DevelopmentFactorEngine, LdfSelectionMethod};
    use reserving_kernels::diagnostics::link_ratio_diagnostics::diagnose_all;
    let tri = reserving_kernels::datasets::raa();
    let direct = DevelopmentFactorEngine::geometric_average_with_policy(
        &tri.view(),
        1.0,
        reserving_kernels::validate::MonotonicityPolicy::AllowNegativeIncrements,
    )
    .expect("geometric");
    let via_enum = DevelopmentFactorEngine::select_with_policy(
        &tri.view(),
        1.0,
        LdfSelectionMethod::Geometric,
        reserving_kernels::validate::MonotonicityPolicy::AllowNegativeIncrements,
    )
    .expect("enum geometric");
    assert_eq!(direct.ldfs, via_enum.ldfs);
    let diag = diagnose_all(&tri.view(), LdfSelectionMethod::Geometric).expect("diag");
    for (d, f) in diag.iter().zip(direct.ldfs.iter()) {
        assert!(
            (d.selected_factor - f).abs() <= 1e-12 * f.abs(),
            "{} vs {f}",
            d.selected_factor
        );
    }
}
