//! Regression & adversarial property tests locking in zero-panic, finite-output,
//! and numerical-stability invariants across the crate.

use reserving_kernels::{
    ReservingError, Triangle, TriangleView,
    capital::{
        aggregate_loss::{FrequencyDistribution, fft_aggregate_loss},
        capital_allocation::{euler_rvar_allocation, euler_tvar_allocation},
        copulas::{apply_copula_reordering, student_t_cdf},
        reserve_distribution::{crps_empirical, interval_score, rvar},
    },
    datasets::genins,
    deterministic::{
        benktander::BenktanderEngine,
        bf::BornhuetterFergusonEngine,
        case_outstanding::{
            CaseOutstandingOptions, project_case_outstanding, project_from_paid_incurred,
        },
        double_chain_ladder,
        freq_sev::{AdlerKlineOptions, CountProjectionMode, adler_kline_reserve},
        ldf::DevelopmentFactorEngine,
    },
    excess::{
        excess_development::{from_excess_ratios, project_split_losses},
        excess_layers::LognormalLayerModel,
        severity::SeverityDistribution,
    },
    stochastic::{
        bayesian_reserving::{MeyersModel, MeyersOptions, meyers_bivariate_ccl},
        glm_reserving::{GlmDesign, GlmOptions, select_tweedie_power},
        kalman_reserving::{self, KalmanReservingOptions},
        micro_reserving::{MicroReservingModel, simulate_synthetic_portfolio},
        multivariate_cl::{MultivariateClOptions, MultivariateModel, analyze_multivariate_cl},
        reserve_runoff::ChainLadderRollForward,
    },
    subannual::{
        ClarkExactAgeOptions, asymmetric_age_grid, calendar_diagonal_index, clark_mle_exact_age,
        interp_swimon, subannual_product_parity,
    },
    validate::MonotonicityPolicy,
};

#[test]
fn swimon_does_not_panic_on_tiny_concave_rounding_slack() {
    // Linear percent-reported anchors: 1/3, 2/3, 1.0 have equal secants up to 1e-16 rounding.
    let ages = [12.0, 24.0, 36.0];
    let cdfs = [3.0, 1.5, 1.0];
    let targets = [6.0, 12.0, 18.0, 24.0, 30.0, 36.0];
    let pat = interp_swimon(&ages, &cdfs, &targets).expect("swimon must not panic");
    assert_eq!(pat.cdfs.len(), targets.len());
    assert!(pat.cdfs.iter().all(|v| v.is_finite() && *v >= 1.0));
    assert!((pat.cdfs[1] - 3.0).abs() < 1e-12);
    assert!((pat.cdfs[3] - 1.5).abs() < 1e-12);
    assert!((pat.cdfs[5] - 1.0).abs() < 1e-12);
}

#[test]
fn malformed_views_and_grids_return_err_without_panicking() {
    let short_view = TriangleView {
        n_origins: 3,
        n_devs: 3,
        data: &[100.0, 150.0, 200.0],
    };
    assert!(select_tweedie_power(&short_view, GlmDesign::CrossClassified, &[1.5]).is_err());
    assert!(ChainLadderRollForward::from_triangle(&short_view, 1.0, vec![0.0; 3]).is_err());

    let valid = Triangle::try_from_upper_rows(&[&[100.0, 150.0], &[120.0]]).expect("valid");
    assert!(
        project_case_outstanding(
            &valid.view(),
            &short_view,
            &CaseOutstandingOptions::default()
        )
        .is_err()
    );
    assert!(
        project_from_paid_incurred(
            &valid.view(),
            &short_view,
            &CaseOutstandingOptions::default()
        )
        .is_err()
    );

    let mut bad_factors =
        from_excess_ratios(100.0, &[2.0, 1.0], &[0.2, 0.3], 0.35).expect("factors");
    bad_factors.limited_cdf.truncate(1);
    assert!(project_split_losses(&[50.0, 60.0], &[10.0, 20.0], &bad_factors).is_err());

    let mut bad_grid = asymmetric_age_grid(2, 2, 12.0, 12.0, 12.0, None).expect("grid");
    bad_grid.dev_months = f64::NAN;
    bad_grid.cell_ages_months.truncate(1);
    assert!(calendar_diagonal_index(&bad_grid).is_err());
}

#[test]
fn double_chain_ladder_and_benktander_respect_allow_negative_increments() {
    // Paid triangle with a legitimate downward case/salvage revision at (0, 2).
    let paid = Triangle::try_from_upper_rows(&[&[100.0, 180.0, 175.0], &[110.0, 195.0], &[120.0]])
        .expect("paid");
    let counts = Triangle::try_from_upper_rows(&[&[10.0, 15.0, 16.0], &[11.0, 16.0], &[12.0]])
        .expect("counts");

    let dcl = double_chain_ladder::analyze_with_inflation(
        &paid.view(),
        &counts.view(),
        None,
        MonotonicityPolicy::AllowNegativeIncrements,
    )
    .expect("DCL should accept AllowNegativeIncrements");
    assert!(dcl.rbns_reserve.iter().all(|v| v.is_finite()));
    assert!(dcl.ibnr_reserve.iter().all(|v| v.is_finite()));

    // Benktander m = 1 must match BF even when CDF < 1.0 yields a negative ultimate.
    let factors = DevelopmentFactorEngine::volume_weighted_with_policy(
        &paid.view(),
        1.0,
        MonotonicityPolicy::AllowNegativeIncrements,
    )
    .expect("factors");
    let expected_ult = [175.0, 20_000.0, 20_000.0];
    let bf = BornhuetterFergusonEngine::project_from_expected_ultimate_with_policy(
        &paid.view(),
        &factors,
        &expected_ult,
        MonotonicityPolicy::AllowNegativeIncrements,
    )
    .expect("bf");
    let bk = BenktanderEngine::project_with_policy(
        &paid.view(),
        &factors,
        &expected_ult,
        1,
        MonotonicityPolicy::AllowNegativeIncrements,
    )
    .expect("benktander m=1 must match bf under AllowNegativeIncrements");
    for (u_bf, u_bk) in bf.ultimates.iter().zip(bk.ultimates.iter()) {
        assert!((u_bf - u_bk).abs() < 1e-9);
    }
}

#[test]
fn adler_kline_exact_conservation_routes_remaining_claims_to_tail_when_disposal_hits_one() {
    // With disposal_averaging_window = 1:
    //   j = 0 uses latest origin (origin 2: 100/100 = 1.0) => selected_disposal_rates[0] = 1.0
    //   j = 1 is monotonically clamped >= d_hat[0] => selected_disposal_rates[1] = 1.0
    // Origin 1 only closed 80 of 100 ultimate claims by j = 1, so ExactConservation must
    // route the 20 remaining open claims to tail_closed_counts[1] instead of dropping them.
    let closed = Triangle::try_from_upper_rows(&[&[100.0, 100.0, 100.0], &[80.0, 80.0], &[100.0]])
        .expect("closed");
    let inc_paid =
        Triangle::try_from_upper_rows(&[&[100_000.0, 0.0, 0.0], &[80_000.0, 0.0], &[100_000.0]])
            .expect("inc_paid");
    let ult_counts = [100.0, 100.0, 100.0];
    let opts = AdlerKlineOptions {
        count_projection_mode: CountProjectionMode::ExactConservation,
        disposal_averaging_window: Some(1),
        ..AdlerKlineOptions::default()
    };

    let res = adler_kline_reserve(&closed.view(), &inc_paid.view(), &ult_counts, opts)
        .expect("adler kline");
    assert!((res.tail_closed_counts[1] - 20.0).abs() < 1e-9);
    assert!(res.reserves[1] > 0.0);
}

#[test]
fn lognormal_extreme_sigma_lev_and_tiny_x_pdf_stay_finite() {
    let dist = SeverityDistribution::Lognormal {
        mu: 10.0,
        sigma: 50.0,
    };
    assert!(dist.unbounded_mean().is_none());
    let lev = dist.limited_expected_value(1_000_000.0).expect("lev");
    assert!(lev.is_finite() && lev > 0.0 && lev <= 1_000_000.0);

    let tiny_pdf = dist.pdf(1e-320);
    assert!(tiny_pdf.is_finite() && tiny_pdf >= 0.0);
    assert!(dist.pdf(f64::NAN).is_nan());

    let layer_model = LognormalLayerModel::try_new(50_000.0, 1.5).expect("model");
    assert!(layer_model.pdf(f64::NAN).is_nan());
    assert!(layer_model.pdf(1e-320).is_finite());
}

#[test]
fn numerical_stability_identities_crps_student_t_rvar_and_fft() {
    // CRPS shift-invariance at large offset (1e9).
    let base = [100.0, 120.0, 150.0, 190.0, 250.0];
    let shift = 1.0e9;
    let shifted: Vec<f64> = base.iter().map(|&x| x + shift).collect();
    let c1 = crps_empirical(&base, 140.0).expect("crps base");
    let c2 = crps_empirical(&shifted, 140.0 + shift).expect("crps shifted");
    assert!((c1 - c2).abs() < 1e-7, "c1={c1} c2={c2}");

    // Student-t CDF at huge nu converges to standard normal Phi(1.0) ≈ 0.841344746.
    let cdf_hi_nu = student_t_cdf(1.0, 1.0e18);
    assert!((cdf_hi_nu - 0.841_344_746_068_542_9).abs() < 1e-7);

    // O(n) rvar and interval_score agree with exact definitions on tied samples.
    let vals = [10.0, 20.0, 20.0, 30.0, 50.0, 80.0];
    let rv = rvar(&vals, 0.25, 0.75).expect("rvar");
    assert!(rv.is_finite() && rv > 20.0 && rv < 40.0);
    let is = interval_score(&vals, 25.0, 0.2).expect("interval_score");
    assert!(is.is_finite() && is > 0.0);

    // Capital allocation invalid probability returns InvalidProbability.
    assert_eq!(
        euler_tvar_allocation(&[&vals], 1.5),
        Err(ReservingError::InvalidProbability { value: 1.5 })
    );
    assert_eq!(
        euler_rvar_allocation(&[&vals], -0.1, 0.9),
        Err(ReservingError::InvalidProbability { value: -0.1 })
    );

    // apply_copula_reordering rejects NaN uniforms.
    assert_eq!(
        apply_copula_reordering(&[&[1.0, 2.0]], &[vec![0.5, f64::NAN]]),
        Err(ReservingError::NotFinite)
    );

    // FFT aggregate loss handles extreme NB parameters without panicking.
    let nb = FrequencyDistribution::NegativeBinomial { r: 2.0, beta: 1e10 };
    let _ = fft_aggregate_loss(nb, &[0.5, 0.5], 1.0, 64, Some(0.0));
}

#[test]
fn micro_reserving_guards_extreme_hazard_ratio_against_infinite_loop() {
    let model = MicroReservingModel {
        occurrence_intensities: vec![10.0, 10.0],
        reporting_delay_rate: 2.0,
        interim_payment_hazard: 1.0e20,
        settlement_hazard: 1.0,
        interim_severity_mu: 8.0,
        interim_severity_sigma: 0.5,
        settlement_payment_prob: 0.5,
        settlement_severity_mu: 9.0,
        settlement_severity_sigma: 0.5,
    };
    assert!(simulate_synthetic_portfolio(&model, 2.0, 42).is_err());
}

#[test]
fn meyers_bivariate_ccl_propagates_parameter_and_correlation_uncertainty() {
    let g = genins();
    // Construct a correlated second line by scaling genins with small deterministic jitter.
    let n = g.n_origins();
    let mut data2 = vec![f64::NAN; n * n];
    for i in 0..n {
        for j in 0..(n - i) {
            let jitter = 1.0 + 0.02 * (((i * 7 + j * 13) % 5) as f64 - 2.0);
            data2[i * n + j] = g.get(i, j) * 1.15 * jitter;
        }
    }
    let line2 = Triangle::try_new(n, n, data2).expect("valid triangle");
    let opts = MeyersOptions {
        model: MeyersModel::Ccl,
        n_samples: 400,
        seed: 77,
        ..MeyersOptions::default()
    };
    let biv = meyers_bivariate_ccl(&g.view(), &line2.view(), &opts).expect("bivariate ccl");
    assert!(biv.cross_line_rho > 0.5);
    assert!(biv.portfolio_mean_reserve.is_finite() && biv.portfolio_mean_reserve > 0.0);
    assert!(
        biv.portfolio_sd_reserve
            >= biv.line1.total_sd_reserve.max(biv.line2.total_sd_reserve) * 0.85,
        "joint SD ({}) should reflect full parameter + correlated process uncertainty vs standalone ({}, {})",
        biv.portfolio_sd_reserve,
        biv.line1.total_sd_reserve,
        biv.line2.total_sd_reserve
    );
}

#[test]
fn general_gmcl_exact_sandwich_msep_is_finite_and_positive() {
    let g = genins();
    let n = g.n_origins();
    let mut data2 = vec![f64::NAN; n * n];
    for i in 0..n {
        for j in 0..(n - i) {
            let factor = 0.8 + 0.05 * (i as f64) + 0.03 * (j as f64);
            data2[i * n + j] = g.get(i, j) * factor;
        }
    }
    let line2 = Triangle::try_new(n, n, data2).expect("valid triangle");
    let gmcl = analyze_multivariate_cl(
        &[g.view(), line2.view()],
        &MultivariateClOptions {
            model: MultivariateModel::GeneralGmcl,
            policy: MonotonicityPolicy::AllowNegativeIncrements,
        },
    )
    .expect("gmcl");
    assert!(gmcl.total_msep_matrix.iter().all(|v| v.is_finite()));
    assert!(gmcl.portfolio_standard_error.is_finite() && gmcl.portfolio_standard_error > 0.0);
}

#[test]
fn subannual_product_parity_rejects_unsorted_ages_and_clark_exact_converges() {
    assert!(
        subannual_product_parity(
            &[12.0, 24.0, 36.0],
            &[1.5, 1.2],
            &[12.0, 36.0, 24.0],
            &[1.5, 1.2],
            1e-6,
        )
        .is_err()
    );

    let grid = asymmetric_age_grid(4, 4, 12.0, 12.0, 12.0, None).expect("grid");
    let cum = vec![
        100.0,
        180.0,
        220.0,
        240.0,
        110.0,
        195.0,
        238.0,
        f64::NAN,
        120.0,
        210.0,
        f64::NAN,
        f64::NAN,
        130.0,
        f64::NAN,
        f64::NAN,
        f64::NAN,
    ];
    let res = clark_mle_exact_age(
        &cum,
        &grid.cell_ages_months,
        grid.n_origins,
        grid.n_devs,
        &grid.observed_counts,
        None,
        &ClarkExactAgeOptions::default(),
    )
    .expect("clark exact");
    assert!(res.total_reserve.is_finite() && res.total_reserve > 0.0);
    assert!(res.total_se.iter().all(|v| v.is_finite() && *v >= 0.0));
}

#[test]
fn adversarial_randomized_triangles_never_panic_or_return_nonfinite_ok() {
    // Deterministic SplitMix64 generator (zero external dependencies).
    let mut state = 0x5EED_CAFE_BABE_1234_u64;
    let mut next_u64 = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut next_f64 = || (next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));

    for case in 0..80 {
        let n = match case % 5 {
            0 => 1,
            1 => 2,
            2 => 3,
            3 => 4,
            _ => 6,
        };
        let scale = match case % 4 {
            0 => 1.0,
            1 => 1.0e6,
            2 => 1.0e-3,
            _ => 1.0e3,
        };
        let mut data = vec![f64::NAN; n * n];
        for i in 0..n {
            let mut acc = if case % 7 == 0 && i == 0 {
                0.0
            } else {
                (10.0 + 90.0 * next_f64()) * scale
            };
            for j in 0..(n - i) {
                let delta = if case % 6 == 0 {
                    (next_f64() - 0.25) * 40.0 * scale
                } else {
                    next_f64() * 50.0 * scale
                };
                acc = (acc + delta).max(0.0);
                data[i * n + j] = acc;
            }
            // Inject stray finite values into the unobserved lower triangle on some cases
            // to ensure kernels strictly respect upper-triangle geometry.
            if case % 3 == 0 {
                for j in (n - i)..n {
                    data[i * n + j] = 999_999.0;
                }
            }
        }
        let view = TriangleView {
            n_origins: n,
            n_devs: n,
            data: &data,
        };
        let prem: Vec<f64> = (0..n)
            .map(|_| (500.0 + 500.0 * next_f64()) * scale)
            .collect();
        let exp_ult: Vec<f64> = prem.iter().map(|p| p * 0.65).collect();

        if let Ok(cl) = view.chain_ladder(1.0) {
            assert!(cl.total_reserve.is_finite());
        }
        if let Ok(m) = view.mack_with_policy(1.0, MonotonicityPolicy::AllowNegativeIncrements) {
            assert!(m.total_reserve.is_finite() && m.total_standard_error.is_finite());
        }
        if let Ok(mw) = view.merz_wuthrich_with_policy(MonotonicityPolicy::AllowNegativeIncrements)
        {
            assert!(mw.total_cdr_msep.is_finite());
        }
        if let Ok(bf) = view.bf(&exp_ult, 1.0) {
            assert!(bf.total_reserve.is_finite());
        }
        if let Ok(cc) = view.cape_cod(&prem, 1.0) {
            assert!(cc.bf_projection.total_reserve.is_finite());
        }
        if let Ok(br) = view.brosius(1.0) {
            assert!(br.reserve.iter().all(|v| v.is_finite()));
        }
        if let Ok(g) = view.glm(&GlmOptions::odp()) {
            assert!(g.total_reserve.is_finite() && g.total_prediction_error.is_finite());
        }
        if let Ok(kf) = kalman_reserving::analyze(&view, KalmanReservingOptions::default()) {
            assert!(kf.total_reserve.is_finite() && kf.total_standard_error.is_finite());
        }
        if let Ok(ex) = view.exhibit(Some(&prem), Some(&exp_ult), 1.0) {
            assert_eq!(ex.projections.rows.len(), n);
        }
    }
}

#[test]
fn layer_spec_unifies_attachment_limit_and_attachment_top_conventions() {
    use reserving_kernels::excess::excess_layers::LayerSpec;

    let from_limit = LayerSpec::from_attachment_limit(100_000.0, 400_000.0).expect("from_limit");
    let from_top = LayerSpec::from_attachment_top(100_000.0, 500_000.0).expect("from_top");
    assert_eq!(from_limit, from_top);
    assert!((from_limit.top() - 500_000.0).abs() < 1e-9);

    let model = LognormalLayerModel::try_new(50_000.0, 1.5).expect("model");
    let l1 = model.exact_layer_spec(from_limit);
    let l2 = model.exact_layer(100_000.0, 500_000.0);
    // Relative tolerance: Miri injects libm (ln/exp/erfc) imprecision, so the
    // two construction paths need not be bit-identical there.
    assert!((l1 - l2).abs() <= 1e-12 * l1.abs().max(1.0), "{l1} vs {l2}");

    let sev = SeverityDistribution::Lognormal {
        mu: 50_000.0_f64.ln() - 0.5 * 1.5 * 1.5,
        sigma: 1.5,
    };
    let s1 = sev.layer_expected_loss_spec(from_top).expect("sev layer");
    assert!((s1 - l1).abs() < 1e-6);
}

#[test]
fn neumaier_compensated_summary_and_truncated_var_tvar_coherence() {
    use reserving_kernels::capital::{
        aggregate_loss::panjer_recursion, reserve_distribution::summarize,
    };

    // High-offset series where naive summation loses low-order digits.
    let mut vals = vec![0.25; 20_001];
    vals[0] = 1.0e12;
    let summary = summarize(&vals).expect("summary");
    let expected_mean = (1.0e12 + 5_000.0) / 20_001.0;
    assert!((summary.mean - expected_mean).abs() < 1e-6);

    // Truncated Panjer grid: both var and tvar reject alpha > total_mass and satisfy tvar >= var.
    let agg = panjer_recursion(
        FrequencyDistribution::Poisson { lambda: 5.0 },
        &[0.5, 0.5],
        1.0,
        5,
    )
    .expect("panjer");
    assert!(agg.total_mass < 0.99);
    assert!(agg.var(0.999).is_err());
    assert!(agg.tvar(0.999).is_err());
    let v50 = agg.var(0.50).expect("var 50");
    // ~11% of the mass lies beyond the grid: the default 1%-of-tail budget rejects
    // TVaR instead of silently renormalizing; an explicit loose budget yields a
    // lower bound that still dominates VaR.
    assert!(agg.tvar(0.50).is_err());
    let tv50 = agg.tvar_with_tolerance(0.50, 1.0).expect("tvar 50");
    assert!(tv50 >= v50 - 1e-12);
}

#[cfg(feature = "wasm")]
#[test]
fn wasm_c_abi_exports_roundtrip_and_clear_buffers_on_invalid_input() {
    use reserving_kernels::wasm_exports::{
        rk_ldf_ptr, rk_mack, rk_mack_total_reserve, rk_res_ptr, rk_se_ptr, rk_tri_ptr,
        rk_tri_stride, rk_ult_ptr,
    };

    let stride = rk_tri_stride() as usize;
    let tri_ptr = rk_tri_ptr();
    let rows: [&[f64]; 4] = [
        &[100.0, 160.0, 195.0, 205.0],
        &[120.0, 190.0, 232.0],
        &[135.0, 218.0],
        &[150.0],
    ];
    // SAFETY: Single test thread writing within the N_MAX x N_MAX static WASM exchange buffer.
    unsafe {
        for (i, row) in rows.iter().enumerate() {
            for j in 0..4 {
                *tri_ptr.add(i * stride + j) = row.get(j).copied().unwrap_or(f64::NAN);
            }
        }
    }
    let se = rk_mack(4, 1.0);
    let total_res = rk_mack_total_reserve(4);
    assert!(se.is_finite() && se > 0.0);
    assert!(total_res.is_finite() && total_res > 0.0);

    // SAFETY: Reading back N_MAX-bounded static output buffers after rk_mack.
    unsafe {
        assert!((*rk_ldf_ptr()).is_finite());
        assert!((*rk_ult_ptr()).is_finite());
        assert!((*rk_res_ptr().add(3)).is_finite());
        assert!((*rk_se_ptr().add(3)).is_finite());
    }

    // Invalid tail (< 1.0) must return NaN and clear stale output buffers to NaN.
    assert!(rk_mack(4, 0.5).is_nan());
    assert!(rk_mack_total_reserve(4).is_nan());
    // SAFETY: Reading cleared static output buffer.
    unsafe {
        assert!((*rk_ult_ptr()).is_nan());
        assert!((*rk_res_ptr()).is_nan());
    }
}

#[test]
fn serde_deserialization_fuzz_never_panics_on_malformed_json() {
    let payloads = [
        r#"{"n_origins":0,"n_devs":0,"data":[]}"#,
        r#"{"n_origins":3,"n_devs":2,"data":[1.0,2.0,3.0,4.0,5.0,6.0]}"#,
        r#"{"n_origins":2,"n_devs":2,"data":[1.0,-5.0,3.0,null]}"#,
        r#"{"n_origins":2,"n_devs":2,"data":[null,2.0,3.0,null]}"#,
        r#"{"n_origins":1000000,"n_devs":1000000,"data":[1.0]}"#,
        r#"{"n_origins":3,"n_devs":3,"data":[1.0,2.0,3.0,4.0,5.0,null,6.0,null,null]}"#,
        "{}",
        "null",
        "[]",
    ];
    for raw in payloads {
        if let Ok(tri) = serde_json::from_str::<Triangle>(raw) {
            let _ = tri.chain_ladder(1.0);
            let _ = tri.mack(1.0);
            let _ = tri.diagnostics();
        }
        if let Ok(grid) =
            serde_json::from_str::<reserving_kernels::subannual::AsymmetricAgeGrid>(raw)
        {
            let _ = calendar_diagonal_index(&grid);
        }
    }
}
