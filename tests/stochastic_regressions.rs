//! Regression tests for stochastic-reserving review findings.
//!
//! Each test pins a behaviour fixed in the stochastic-module review pass
//! (σ² extrapolation, observed-count policy, Mack BF pattern SE, Braun
//! parity, Clark covariance, Merz-Wüthrich CDR pins, GLM / ODP bootstrap
//! fixes). Slow seeded Monte Carlo checks are `#[ignore]`d — run them with
//! `cargo test --test stochastic_regressions -- --include-ignored`.

use reserving_kernels::{
    ReservingError,
    stochastic::bootstrap_odp::{
        OdpBootstrapOptions, OdpNumerics, ProcessDistribution, bootstrap, bootstrap_odp,
        bootstrap_odp_with_numerics, fit_odp,
    },
    stochastic::clark_mle::{ClarkCovariance, ClarkOptions, fit_clark_cape_cod, fit_clark_ldf},
    stochastic::glm_reserving::{GlmDesign, GlmFamily, GlmOptions, fit_glm, select_tweedie_power},
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

// ===========================================================================
// S2: GLM log-normal bias correction & consistent MSEP
// ===========================================================================

/// Minimal deterministic SplitMix64 + Box-Muller for test-side simulation.
struct TestRng(u64);

impl TestRng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// True log-linear increments `exp(a_i + b_j)` for an `n × n` model.
fn log_linear_eta(n: usize) -> Vec<f64> {
    let mut eta = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let a = 8.0 + 0.05 * i as f64;
            let b = if j == 0 { 0.0 } else { 0.6 - 0.45 * j as f64 };
            eta[i * n + j] = a + b;
        }
    }
    eta
}

fn simulate_log_normal_triangle(n: usize, sigma: f64, rng: &mut TestRng) -> (Triangle, f64) {
    let eta = log_linear_eta(n);
    let mut data = vec![NAN; n * n];
    let mut future = 0.0;
    for i in 0..n {
        let mut cum = 0.0;
        for j in 0..n {
            let y = (eta[i * n + j] + sigma * rng.normal()).exp();
            if i + j < n {
                cum += y;
                data[i * n + j] = cum;
            } else {
                future += y;
            }
        }
    }
    (Triangle::try_new(n, n, data).expect("triangle"), future)
}

#[test]
fn log_normal_glm_uses_bias_corrected_means_and_consistent_msep() {
    let mut rng = TestRng(42069);
    let (tri, _) = simulate_log_normal_triangle(8, 0.3, &mut rng);
    let fit = fit_glm(&tri.view(), &GlmOptions::log_normal()).expect("log-normal");
    let n = 8;
    let sigma_sq = fit.pearson_dispersion;
    // Process variance is exactly Σ_future μ̂²(e^{σ̂²} − 1).
    let mut proc = 0.0;
    let mut reserve = 0.0;
    for i in 0..n {
        for j in (n - i)..n {
            let mu = fit.fitted_incremental[i * n + j];
            proc += mu * mu * sigma_sq.exp_m1();
            reserve += mu;
        }
    }
    assert!(close_rel(fit.total_process_se.powi(2), proc, 1e-10));
    assert!(close_rel(fit.total_reserve, reserve, 1e-12));
    assert!(close_rel(
        fit.total_prediction_error.powi(2),
        fit.total_process_se.powi(2) + fit.total_parameter_se.powi(2),
        1e-12
    ));
    // Bias correction: each future mean sits below the naive exp(η̂ + σ̂²/2)
    // by exactly exp(-v/2) with v = Var(η̂) > 0; reconstruct the naive mean
    // from the coefficients (corner coding: intercept + row + column).
    let c = &fit.coefficients;
    let eta_hat = |i: usize, j: usize| -> f64 {
        let mut e = c[0];
        if i > 0 {
            e += c[i];
        }
        if j > 0 {
            e += c[n - 1 + j];
        }
        e
    };
    for i in 1..n {
        let j = n - 1;
        let naive = (eta_hat(i, j) + 0.5 * sigma_sq).exp();
        assert!(fit.fitted_incremental[i * n + j] < naive);
    }
    assert_eq!(fit.eta_clamp_hits, 0);
}

/// Seeded Monte Carlo: bias of the reserve estimator < 2% and realised MSEP
/// ≈ reported MSEP. Slow: run with `cargo test -- --include-ignored`.
#[test]
#[ignore = "slow seeded Monte Carlo; run with --include-ignored"]
fn log_normal_glm_bias_and_msep_ratio_seeded_monte_carlo() {
    let n = 10;
    let sigma = 0.3;
    let eta = log_linear_eta(n);
    let true_mean: f64 = (0..n)
        .flat_map(|i| ((n - i)..n).map(move |j| (i, j)))
        .map(|(i, j)| (eta[i * n + j] + 0.5 * sigma * sigma).exp())
        .sum();
    let mut rng = TestRng(42069);
    let sims = 2000;
    let mut sum_hat = 0.0;
    let mut sum_sq_err = 0.0;
    let mut sum_msep = 0.0;
    for _ in 0..sims {
        let (tri, future) = simulate_log_normal_triangle(n, sigma, &mut rng);
        let fit = fit_glm(&tri.view(), &GlmOptions::log_normal()).expect("fit");
        sum_hat += fit.total_reserve;
        sum_sq_err += (future - fit.total_reserve).powi(2);
        sum_msep += fit.total_prediction_error.powi(2);
    }
    let bias = sum_hat / f64::from(sims) / true_mean - 1.0;
    assert!(bias.abs() < 0.02, "relative bias {bias}");
    let ratio = sum_sq_err / sum_msep;
    assert!(
        (0.85..1.20).contains(&ratio),
        "realised/reported MSEP {ratio}"
    );
}

// ===========================================================================
// S2: Tweedie p = 2 zero handling, EQL selection, ODP deviance consistency
// ===========================================================================

fn genins_with_zero_increment() -> Triangle {
    let g = genins_triangle();
    let n = 10;
    let mut data = g.data().to_vec();
    // Make the (1, 8) increment exactly zero: C[1,8] = C[1,7].
    data[n + 8] = data[n + 7];
    Triangle::try_new(n, n, data).expect("triangle")
}

#[test]
fn tweedie_power_two_rejects_zero_increments_like_gamma() {
    let tri = genins_with_zero_increment();
    assert!(fit_glm(&tri.view(), &GlmOptions::tweedie(2.0)).is_err());
    assert!(fit_glm(&tri.view(), &GlmOptions::gamma()).is_err());
    assert!(fit_glm(&tri.view(), &GlmOptions::tweedie(1.5)).is_ok());
    let sel = select_tweedie_power(&tri.view(), GlmDesign::CrossClassified, &[1.2, 1.5, 2.0])
        .expect("selection");
    assert!(sel.candidates.iter().all(|c| c.power < 2.0));
    assert!(sel.best_power < 2.0);
    assert!(sel.best_fit.converged);
}

#[test]
fn odp_glm_deviance_with_negative_increments_uses_abs_y_convention() {
    let raa = raa_triangle();
    let mut opts = GlmOptions::odp();
    opts.monotonicity_policy = MonotonicityPolicy::AllowNegativeIncrements;
    let fit = fit_glm(&raa.view(), &opts).expect("odp raa");
    let n = 10;
    let mut dev = 0.0;
    let mut saw_negative = false;
    for i in 0..n {
        for j in 0..(n - i) {
            let y = if j == 0 {
                raa.get(i, 0)
            } else {
                raa.get(i, j) - raa.get(i, j - 1)
            };
            let mu = fit.fitted_incremental[i * n + j];
            let term = if y > 0.0 {
                y * (y / mu).ln()
            } else if y < 0.0 {
                saw_negative = true;
                y * (-y / mu).ln()
            } else {
                0.0
            };
            dev += 2.0 * (term - (y - mu));
        }
    }
    assert!(saw_negative);
    assert!(
        close_rel(fit.deviance, dev, 1e-9),
        "{} vs {dev}",
        fit.deviance
    );
    assert!(close_rel(
        fit.deviance_dispersion,
        dev / fit.degrees_of_freedom as f64,
        1e-9
    ));
}

#[test]
fn glm_options_expose_numerical_tunables() {
    let g = genins_triangle();
    let base = fit_glm(&g.view(), &GlmOptions::tweedie(1.5)).expect("base");
    let mut opts = GlmOptions::tweedie(1.5);
    opts.max_step_halvings = 40;
    opts.y_floor_relative = 1e-3;
    opts.eta_clamp = 60.0;
    let alt = fit_glm(&g.view(), &opts).expect("alt");
    assert_eq!(base.eta_clamp_hits, 0);
    assert_eq!(alt.eta_clamp_hits, 0);
    assert!(close_rel(base.total_reserve, alt.total_reserve, 1e-6));
    assert_eq!(GlmFamily::Odp, GlmOptions::default().family);
}

// ===========================================================================
// S2: ODP bootstrap — tail in analytic PE, bootstrap == bootstrap_odp,
// reflection branches, zero-residual exclusion, golden stream
// ===========================================================================

#[test]
fn odp_analytic_prediction_error_includes_fixed_tail() {
    let g = genins_triangle();
    let f1 = fit_odp(&g.view(), 1.0, MonotonicityPolicy::Strict).expect("fit");
    let ft = fit_odp(&g.view(), 1.10, MonotonicityPolicy::Strict).expect("fit tail");
    // Parameter SE scales by tail; process SE uses the tail-inclusive reserve.
    assert!(close_rel(
        ft.analytic_total_param_se,
        1.10 * f1.analytic_total_param_se,
        1e-10
    ));
    assert!(close_rel(
        ft.analytic_total_process_se.powi(2),
        ft.dispersion * ft.total_reserve,
        1e-10
    ));
    // Analytic PE is consistent with the bootstrap SD (E&V 2002 pool).
    let boot = bootstrap_odp(
        &g.view(),
        1.10,
        OdpBootstrapOptions::england_verrall(4000, 42069),
    )
    .expect("boot");
    let sd = boot.summary.std_dev;
    let ratio = sd / ft.analytic_total_prediction_error;
    assert!(
        (0.93..1.07).contains(&ratio),
        "bootstrap SD {sd} vs analytic {}",
        ft.analytic_total_prediction_error
    );
}

#[test]
fn legacy_bootstrap_is_identical_projection_of_bootstrap_odp() {
    let g = genins_triangle();
    for opts in [
        OdpBootstrapOptions::england_verrall(203, 42069),
        OdpBootstrapOptions::shapland(203, 42069),
        OdpBootstrapOptions::england_verrall(50, 7)
            .with_process_distribution(ProcessDistribution::OverDispersedPoisson),
    ] {
        let full = bootstrap_odp(&g.view(), 1.05, opts).expect("full");
        let compact = bootstrap(&g.view(), 1.05, opts).expect("compact");
        assert_eq!(full.simulated_totals, compact.simulated_totals);
        assert_eq!(full.failed_iterations, compact.failed_iterations);
        assert_eq!(compact.failed_iterations, 0);
    }
    let zero = bootstrap(&g.view(), 1.0, OdpBootstrapOptions::england_verrall(0, 1));
    assert_eq!(zero.map(|_| ()), Err(ReservingError::InvalidIterations));
}

#[test]
fn negative_pseudo_increments_are_reflected_on_x4_and_scalar_paths() {
    // RAA-like late development makes pseudo-LDFs < 1 frequent.
    let g = genins_triangle();
    // x4 vectorised path: Gamma, iterations >= 4; scalar path: ODP process.
    for (opts, label) in [
        (
            OdpBootstrapOptions::england_verrall(4001, 42069),
            "x4+scalar",
        ),
        (
            OdpBootstrapOptions::england_verrall(1000, 42069)
                .with_process_distribution(ProcessDistribution::OverDispersedPoisson),
            "scalar",
        ),
    ] {
        let boot = bootstrap_odp(&g.view(), 1.0, opts).expect(label);
        let origin1 = &boot.simulated_by_origin[1];
        assert!(origin1.iter().all(|v| v.is_finite()), "{label}");
        assert!(
            origin1.iter().any(|&v| v < 0.0),
            "{label}: reflection should produce some negative origin-1 reserves"
        );
        let mean = boot.summary.mean;
        assert!(close_rel(mean, boot.base_total_reserve, 0.05), "{label}");
    }
}

#[test]
fn zero_residual_exclusion_option_changes_degrees_of_freedom_pool_only() {
    let g = genins_triangle();
    let opts = OdpBootstrapOptions::england_verrall(500, 42069);
    let keep = bootstrap_odp(&g.view(), 1.0, opts).expect("keep");
    let drop = bootstrap_odp_with_numerics(
        &g.view(),
        1.0,
        opts,
        OdpNumerics {
            exclude_zero_residuals: true,
            ..OdpNumerics::default()
        },
    )
    .expect("drop");
    assert_ne!(keep.simulated_totals, drop.simulated_totals);
    assert!(close_rel(keep.summary.mean, drop.summary.mean, 0.05));
    // Hat-matrix pool already excludes saturated cells: option is a no-op there.
    let sh = OdpBootstrapOptions::shapland(200, 42069);
    let a = bootstrap_odp(&g.view(), 1.0, sh).expect("a");
    let b = bootstrap_odp_with_numerics(
        &g.view(),
        1.0,
        sh,
        OdpNumerics {
            exclude_zero_residuals: true,
            ..OdpNumerics::default()
        },
    )
    .expect("b");
    assert_eq!(a.simulated_totals, b.simulated_totals);
}

/// Golden pin of the first bootstrap draws for seed 42069 (E&V 2002 spec,
/// GenIns, tail 1.0). Bit-exact only per platform: restricted to
/// linux / x86_64 where it was recorded, to detect RNG / stream changes.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn bootstrap_odp_first_draws_golden_seed_42069() {
    let g = genins_triangle();
    let boot = bootstrap_odp(
        &g.view(),
        1.0,
        OdpBootstrapOptions::england_verrall(8, 42069),
    )
    .expect("boot");
    let golden: [f64; 8] = [
        21_214_431.632_405_326,
        17_721_598.399_976_738,
        18_909_734.996_420_678,
        19_719_445.977_566_436,
        21_994_147.204_176_97,
        17_408_756.868_490_62,
        17_110_393.182_354_525,
        14_339_983.178_005_299,
    ];
    let got: Vec<u64> = boot.simulated_totals.iter().map(|v| v.to_bits()).collect();
    let want: Vec<u64> = golden.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, want, "draws: {:?}", boot.simulated_totals);
}
