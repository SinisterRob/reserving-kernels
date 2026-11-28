//! Regression tests for stochastic-reserving review findings.
//!
//! Each test pins a behaviour fixed in the stochastic-module review pass
//! (σ² extrapolation, observed-count policy, Mack BF pattern SE, Braun
//! parity, Clark covariance, Merz-Wüthrich CDR pins, GLM / ODP bootstrap
//! fixes). Slow seeded Monte Carlo checks are `#[ignore]`d — run them with
//! `cargo test --test stochastic_regressions -- --include-ignored`.

use reserving_kernels::{
    ReservingError,
    stochastic::bootstrap_odp::{OdpBootstrapOptions, OdpNumerics},
    stochastic::clark_mle::{ClarkCovariance, ClarkOptions, fit_clark_cape_cod, fit_clark_ldf},
    stochastic::mack::{self, MackOptions},
    stochastic::mack_bf::{self, BfPayoutEstimator, MackBfOptions},
    stochastic::merz_wuthrich,
    stochastic::multivariate_cl::{
        MultivariateClOptions, MultivariateModel, analyze_multivariate_cl,
    },
    stochastic::multivariate_mack,
    stochastic::reserve_runoff::{
        MackRunoffOptions, mack_one_year_runoff_with_options, odp_one_year_runoff_with_numerics,
    },
    types::Triangle,
    validate::MonotonicityPolicy,
};

const NAN: f64 = f64::NAN;

fn close_rel(a: f64, b: f64, rel_tol: f64) -> bool {
    (a - b).abs() <= rel_tol * (1.0 + b.abs())
}

/// RAA (Mack 1993) 10x10 incurred triangle.
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
    Triangle::try_new(10, 10, data).expect("triangle")
}

/// Taylor & Ashe (1983) `GenIns` 10x10 cumulative paid triangle.
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
    Triangle::try_new(10, 10, data).expect("triangle")
}

fn upper_counts(n: usize) -> Vec<usize> {
    (0..n).map(|i| n - i).collect()
}

// ===========================================================================
// Mack: analyze_with_options σ² extrapolation & observed-count policy
// ===========================================================================

#[test]
fn mack_options_unit_weights_extrapolate_sigma_like_analyze_on_3x3() {
    let tri = Triangle::try_from_upper_rows(&[&[100.0, 160.0, 195.0], &[120.0, 190.0], &[135.0]])
        .expect("tri");
    let base = mack::analyze(&tri.view(), 1.0).expect("mack");
    let opts = mack::analyze_with_options(
        &tri.view(),
        &MackOptions::default().with_weights(vec![1.0; 9]),
    )
    .expect("mack options");
    assert_eq!(base.sigma_sq.len(), 2);
    assert!(base.sigma_sq[1] > 0.0);
    for k in 0..2 {
        assert!(close_rel(opts.sigma_sq[k], base.sigma_sq[k], 1e-12));
    }
    assert!(close_rel(
        opts.total_standard_error,
        base.total_standard_error,
        1e-10
    ));
}

#[test]
fn mack_options_masked_cell_extrapolates_all_thin_links_on_raa() {
    let raa = raa_triangle();
    let mut w = vec![1.0; 100];
    // Mask link 7 for origin 0 -> link 7 retains a single active pair.
    w[7] = 0.0;
    let res = mack::analyze_with_options(&raa.view(), &MackOptions::default().with_weights(w))
        .expect("masked");
    assert!(res.sigma_sq[7] > 0.0, "sigma7={}", res.sigma_sq[7]);
    assert!(res.sigma_sq[8] > 0.0, "sigma8={}", res.sigma_sq[8]);
    let s5 = res.sigma_sq[5];
    let s6 = res.sigma_sq[6];
    let expect7 = (s6 * s6 / s5).min(s5).min(s6);
    assert!(close_rel(res.sigma_sq[7], expect7, 1e-12));
    let expect8 = (expect7 * expect7 / s6).min(s6).min(expect7);
    assert!(close_rel(res.sigma_sq[8], expect8, 1e-12));
    // Unmasked unit weights reproduce the classical analysis exactly.
    let unit = mack::analyze_with_options(
        &raa.view(),
        &MackOptions::default().with_weights(vec![1.0; 100]),
    )
    .expect("unit");
    let base = mack::analyze(&raa.view(), 1.0).expect("base");
    assert!(close_rel(
        unit.total_standard_error,
        base.total_standard_error,
        1e-10
    ));
}

#[test]
fn mack_observed_counts_default_allows_negative_increments_on_raa() {
    let raa = raa_triangle();
    let counts = upper_counts(10);
    let obs = mack::analyze_observed_counts(&raa.view(), &counts, 1.0).expect("relaxed");
    let base = mack::analyze(&raa.view(), 1.0).expect("base");
    assert!(close_rel(obs.total_reserve, base.total_reserve, 1e-10));
    assert!(close_rel(
        obs.total_standard_error,
        base.total_standard_error,
        1e-10
    ));
    let strict = mack::analyze_observed_counts_with_policy(
        &raa.view(),
        &counts,
        1.0,
        MonotonicityPolicy::Strict,
    );
    assert!(matches!(
        strict,
        Err(ReservingError::NonMonotoneCumulative { .. })
    ));
}

// ===========================================================================
// Merz-Wüthrich (2008) one-year CDR published-value pins on GenIns
// ===========================================================================

#[test]
fn merz_wuthrich_genins_cdr_matches_chainladder_published_values() {
    let g = genins_triangle();
    let mw = merz_wuthrich::analyze(&g.view()).expect("mw");
    // `ChainLadder::CDR(MackChainLadder(GenIns))` (Gesmann et al.): total
    // CDR(1) S.E. 1,778,967.66 and Mack S.E. 2,447,094.86; origin 2 (0-based 1)
    // CDR S.E. = Mack S.E. = 75,535.04.
    assert!(close_rel(mw.total_cdr_standard_error, 1_778_967.66, 1e-8));
    assert!(close_rel(
        mw.total_ultimate_standard_error,
        2_447_094.86,
        1e-8
    ));
    assert!(close_rel(mw.cdr_standard_error[1], 75_535.04, 1e-7));
    // Remaining per-origin CDR S.E. pinned from this implementation after the
    // total matched the published value (regression guard).
    let pinned = [
        0.0,
        75_535.040_757_488,
        105_309.302_864_920,
        79_846.170_894_337,
        235_115.114_384_219,
        318_427.187_659_863,
        361_089.310_886_177,
        629_681.031_934_885,
        588_661.901_625_091,
        1_029_924.990_976_374,
    ];
    for (i, (&got, &want)) in mw.cdr_standard_error.iter().zip(&pinned).enumerate() {
        assert!(close_rel(got, want, 1e-9), "origin {i}: {got} vs {want}");
    }
}

// ===========================================================================
// Mack (2008) BF pattern SE min-rule and CL-implied consistency
// ===========================================================================

#[test]
fn mack_bf_direct_pattern_se_uses_mack_2008_min_rule() {
    let raa = raa_triangle();
    let prior = vec![20_000.0; 10];
    let prior_se = vec![2_000.0; 10];
    let res = mack_bf::analyze_from_triangle(
        &raa.view(),
        &prior,
        &prior_se,
        MackBfOptions {
            payout_estimator: BfPayoutEstimator::DirectIncremental,
            prior_correlation: 0.0,
            monotonicity_policy: MonotonicityPolicy::AllowNegativeIncrements,
        },
    )
    .expect("bf");
    let n = 10;
    // Reconstruct s²(ŷ_j) = s_j² / V_j with V_j = Σ_{i ≤ n-1-j} E_i.
    let var_y: Vec<f64> = (0..n)
        .map(|j| res.process_sigma_sq[j] / (prior[0] * (n - j) as f64))
        .collect();
    let total: f64 = var_y.iter().sum();
    let mut lower = 0.0;
    for k in 0..(n - 1) {
        lower += var_y[k];
        let expect = lower.min(total - lower).sqrt();
        assert!(
            close_rel(res.payout_se[k], expect, 1e-12),
            "k={k}: {} vs {expect}",
            res.payout_se[k]
        );
    }
    assert_eq!(res.payout_se[n - 1], 0.0);
    // Early ages use the (smaller) lower sum: strictly below the upper-only rule.
    let upper0 = (total - var_y[0]).sqrt();
    assert!(res.payout_se[0] < upper0);
}

#[test]
fn mack_bf_chain_ladder_pattern_se_is_delta_method_of_cl_pattern() {
    let genins = genins_triangle();
    let prior = vec![5_000_000.0; 10];
    let prior_se = vec![500_000.0; 10];
    let res = mack_bf::analyze_from_triangle(
        &genins.view(),
        &prior,
        &prior_se,
        MackBfOptions {
            payout_estimator: BfPayoutEstimator::ChainLadderImplied,
            prior_correlation: 0.0,
            monotonicity_policy: MonotonicityPolicy::Strict,
        },
    )
    .expect("bf");
    let m = mack::analyze(&genins.view(), 1.0).expect("mack");
    let n = 10;
    let mut acc = 0.0;
    for k in (0..(n - 1)).rev() {
        let s_k: f64 = (0..(n - 1 - k)).map(|i| genins.get(i, k)).sum();
        acc += m.sigma_sq[k] / (m.ldf[k] * m.ldf[k] * s_k);
        let expect = res.payout_z[k] * acc.sqrt();
        assert!(close_rel(res.payout_se[k], expect, 1e-12), "k={k}");
    }
    assert_eq!(res.payout_se[n - 1], 0.0);
}

/// Second line for multi-line tests: GenIns scaled with deterministic jitter.
fn genins_jittered() -> Triangle {
    let g = genins_triangle();
    let n = 10;
    let mut data = vec![NAN; n * n];
    for i in 0..n {
        for j in 0..(n - i) {
            let jitter = 1.0 + 0.03 * (((i * 7 + j * 13) % 5) as f64 - 2.0);
            data[i * n + j] = g.get(i, j) * 0.8 * jitter * (1.0 + 0.01 * j as f64);
        }
    }
    Triangle::try_new(n, n, data).expect("triangle")
}

// ===========================================================================
// Braun (2004) cross-covariance parity: multivariate Mack vs diagonal MCL
// ===========================================================================

#[test]
fn braun_link_covariance_matches_between_multivariate_mack_and_diagonal_mcl() {
    let a = genins_triangle();
    let b = genins_jittered();
    let views = [a.view(), b.view()];
    let mm = multivariate_mack::analyze_with_policy(
        &views,
        1.0,
        MonotonicityPolicy::AllowNegativeIncrements,
    )
    .expect("multivariate mack");
    let mcl = analyze_multivariate_cl(
        &views,
        &MultivariateClOptions {
            model: MultivariateModel::DiagonalMcl,
            policy: MonotonicityPolicy::AllowNegativeIncrements,
        },
    )
    .expect("mcl");
    // Links with >= 2 pairs (k <= n - 3) use the estimator directly.
    // Links k <= 6 (>= 3 pairs); at k = 7 multivariate Mack additionally
    // clamps to the Cauchy-Schwarz bound sqrt(sigma_r^2 sigma_s^2).
    for k in 0..7 {
        let cov_mm = mm.link_covariances[k][1];
        let cov_mcl = mcl.sigma_matrices[k][1];
        assert!(
            close_rel(cov_mm, cov_mcl, 1e-10),
            "k={k}: mack={cov_mm} mcl={cov_mcl}"
        );
        // Diagonals reduce to Mack's n_k - 1 estimator in both.
        assert!(close_rel(
            mm.marginal_mack[0].sigma_sq[k],
            mcl.sigma_matrices[k][0],
            1e-10
        ));
    }
}

// ===========================================================================
// Clark (2003): observed vs expected information covariance
// ===========================================================================

#[test]
fn clark_observed_information_is_default_and_close_to_expected_on_genins() {
    let g = genins_triangle();
    let obs = fit_clark_ldf(&g.view(), &ClarkOptions::loglogistic()).expect("observed");
    let exp = fit_clark_ldf(
        &g.view(),
        &ClarkOptions::loglogistic().with_covariance(ClarkCovariance::Expected),
    )
    .expect("expected");
    assert_eq!(obs.covariance, ClarkCovariance::Observed);
    assert!(!obs.covariance_fallback);
    assert_eq!(exp.covariance, ClarkCovariance::Expected);
    // Same MLE point estimates; only the parameter covariance differs.
    assert!(close_rel(obs.total_reserve, exp.total_reserve, 1e-12));
    assert!(close_rel(obs.total_process_se, exp.total_process_se, 1e-12));
    assert!((obs.total_parameter_se - exp.total_parameter_se).abs() > 0.0);
    let ratio = obs.total_parameter_se / exp.total_parameter_se;
    assert!(
        (0.7..1.4).contains(&ratio),
        "obs/exp param SE ratio {ratio}"
    );
    // Observed covariance is symmetric positive on its diagonal.
    let p = obs.n_params;
    for i in 0..p {
        assert!(obs.parameter_covariance[i * p + i] > 0.0);
        for j in 0..p {
            assert!(close_rel(
                obs.parameter_covariance[i * p + j],
                obs.parameter_covariance[j * p + i],
                1e-9
            ));
        }
    }
    let cc = fit_clark_cape_cod(&g.view(), &[10_000_000.0; 10], &ClarkOptions::weibull())
        .expect("cape cod observed");
    assert_eq!(cc.covariance, ClarkCovariance::Observed);
    assert!(cc.total_parameter_se > 0.0);
}

// ===========================================================================
// Reserve run-off: explicit failed-scenario policy
// ===========================================================================

#[test]
fn runoff_failed_fraction_threshold_is_validated_and_clean_runs_pass() {
    let g = genins_triangle();
    let ok = mack_one_year_runoff_with_options(
        &g.view(),
        1.0,
        MackRunoffOptions::new(500, 42069).with_max_failed_fraction(0.0),
    )
    .expect("no failures on GenIns");
    assert_eq!(ok.failed_iterations, 0);
    let bad = mack_one_year_runoff_with_options(
        &g.view(),
        1.0,
        MackRunoffOptions::new(10, 42069).with_max_failed_fraction(1.5),
    );
    assert!(matches!(
        bad,
        Err(ReservingError::InvalidProbability { .. })
    ));
    let odp = odp_one_year_runoff_with_numerics(
        &g.view(),
        1.0,
        OdpBootstrapOptions::england_verrall(200, 42069),
        OdpNumerics {
            max_failed_fraction: 0.0,
            ..OdpNumerics::default()
        },
    )
    .expect("odp runoff");
    assert_eq!(odp.failed_iterations, 0);
}
