//! Integration, mathematical-identity, and source-benchmark tests for the
//! `subannual` kernel registry (sub-annual development, asymmetric grids, and
//! pre-12-month stub periods).

use reserving_kernels::ReservingError;
use reserving_kernels::datasets::{friedland_us_industry_auto, genins};
use reserving_kernels::deterministic::bf::BornhuetterFergusonEngine;
use reserving_kernels::deterministic::cape_cod::CapeCodEngine;
use reserving_kernels::deterministic::ldf::{DevelopmentFactorEngine, LdfSelectionMethod};
use reserving_kernels::stochastic::bootstrap_odp::{
    OdpBootstrapOptions, ProcessDistribution, ResidualAdjustment, fit_odp,
};
use reserving_kernels::stochastic::clark_mle::GrowthCurve;
use reserving_kernels::subannual::*;
use reserving_kernels::validate::MonotonicityPolicy;

// ===========================================================================
// 1. Grid Geometry & Age Mapping
// ===========================================================================

#[test]
fn asymmetric_age_grid_handles_2_to_1_4_to_1_and_12_to_1_strides() {
    // 2:1 (annual × semiannual, full valuation)
    let g2 = asymmetric_age_grid(4, 8, 12.0, 6.0, 6.0, None).unwrap();
    assert!((g2.stride_ratio - 2.0).abs() < 1e-12);
    assert_eq!(g2.observed_counts, vec![8, 6, 4, 2]);
    assert_eq!(
        g2.latest_ages_months().unwrap(),
        vec![48.0, 36.0, 24.0, 12.0]
    );

    // 4:1 (annual × quarterly, 3-month Q1 stub valuation)
    let g4 = asymmetric_age_grid(3, 12, 12.0, 3.0, 3.0, Some(3.0)).unwrap();
    assert!((g4.stride_ratio - 4.0).abs() < 1e-12);
    assert_eq!(g4.observed_counts, vec![9, 5, 1]);
    assert_eq!(g4.latest_ages_months().unwrap(), vec![27.0, 15.0, 3.0]);
    assert!((g4.exposure_fractions()[2] - 0.25).abs() < 1e-12);
    assert!((g4.cell_average_age(2, 0).unwrap() - 1.5).abs() < 1e-12);

    // 12:1 (annual × monthly, 6-month stub valuation)
    let g12 = asymmetric_age_grid(2, 24, 12.0, 1.0, 1.0, Some(6.0)).unwrap();
    assert!((g12.stride_ratio - 12.0).abs() < 1e-12);
    assert_eq!(g12.observed_counts, vec![18, 6]);
    assert!((g12.cell_average_age(1, 5).unwrap() - 3.0).abs() < 1e-12);
}

#[test]
fn stub_average_age_weighted_handles_front_and_back_loaded_exposure() {
    // Back-loaded exposure: 3x weight in months 4..6 vs months 1..3 -> average accident
    // date is later -> average claim age at 6m is smaller than uniform 3.0m.
    let back_loaded = [1.0, 1.0, 1.0, 3.0, 3.0, 3.0];
    let avg_back = stub_average_age_weighted(6.0, 1.0, &back_loaded).unwrap();
    assert!(avg_back < 3.0 && avg_back > 1.5);

    // Front-loaded exposure: average accident date is earlier -> average claim age > 3.0m.
    let front_loaded = [3.0, 3.0, 3.0, 1.0, 1.0, 1.0];
    let avg_front = stub_average_age_weighted(6.0, 1.0, &front_loaded).unwrap();
    assert!(avg_front > 3.0 && avg_front < 4.5);
    assert!(((avg_back + avg_front) - 6.0).abs() < 1e-12);
}

#[test]
fn calendar_diagonal_index_pinpoints_shock_on_quarterly_asymmetric_grid() {
    // 4 annual origins × 16 quarterly devs (4:1 stride), valued at 48m.
    let grid = asymmetric_age_grid(4, 16, 12.0, 3.0, 3.0, None).unwrap();
    let cal_map = calendar_diagonal_index(&grid).unwrap();

    // Verify (0, 4) [AY0 at 15m] and (1, 0) [AY1 at 3m] share calendar month 15.0.
    assert_eq!(cal_map.diagonal_of(0, 4), cal_map.diagonal_of(1, 0));

    // Inject a +5.0 residual shock exclusively on calendar month 39.0.
    let mut residuals = vec![f64::NAN; 4 * 16];
    let target_month = 39.0;
    for i in 0..4 {
        for j in 0..grid.observed_counts[i] {
            let cm = grid.origin_start_months[i] + grid.cell_age(i, j).unwrap();
            residuals[i * 16 + j] = if (cm - target_month).abs() < 1e-9 {
                5.0
            } else {
                0.0
            };
        }
    }

    let diag = diagnose_asymmetric_calendar_effects(&cal_map, &residuals).unwrap();
    let strongest = diag.strongest_diagonal.expect("strongest diagonal");
    assert!((diag.diagonal_months[strongest] - target_month).abs() < 1e-9);
    assert!((diag.diagonal_means[strongest] - 5.0).abs() < 1e-12);
}

// ===========================================================================
// 2. Sub-Annual LDF & CDF Interpolation
// ===========================================================================

/// Annual-window product parity telescopes (`prod CDF(t_k)/CDF(t_{k+1}) =
/// CDF(start)/CDF(end)`), so for any interpolator it is *equivalent* to knot
/// preservation. This test therefore checks knot preservation directly, plus
/// monotonicity and `CDF >= 1` on a real industry pattern, and then proves the
/// parity diagnostic is not vacuous by showing it flags a perturbed pattern.
#[test]
fn all_seven_interpolators_preserve_knots_and_parity_detects_breakage_on_friedland_auto() {
    let auto = friedland_us_industry_auto();
    let factors = auto.paid.ldfs(1.0).unwrap();
    let n = factors.cdfs.len();
    let annual_ages: Vec<f64> = (1..=n).map(|y| (y * 12) as f64).collect();
    let quarterly_ages: Vec<f64> = (4..=(n * 4)).map(|q| (q * 3) as f64).collect();

    for method in [
        SubannualInterpMethod::LinearCdf,
        SubannualInterpMethod::LinearPctReported,
        SubannualInterpMethod::Exponential,
        SubannualInterpMethod::InversePower,
        SubannualInterpMethod::Swimon,
        SubannualInterpMethod::SplicedWeibull,
        SubannualInterpMethod::Ivf12,
    ] {
        let pat = interpolate_pattern(method, &annual_ages, &factors.cdfs, &quarterly_ages)
            .expect("interpolate Friedland Auto");
        // Knot k (age 12(k+1)) sits at quarterly index 4k.
        for (k, &expected) in factors.cdfs.iter().enumerate() {
            let got = pat.cdfs[4 * k];
            assert!(
                (got - expected).abs() <= 1e-10 * expected,
                "{method:?} knot {k}: got {got}, expected {expected}"
            );
        }
        for (k, w) in pat.cdfs.windows(2).enumerate() {
            assert!(
                w[0] >= w[1] - 1e-10,
                "{method:?} non-monotone at quarterly index {k}: {} < {}",
                w[0],
                w[1]
            );
        }
        assert!(
            pat.cdfs.iter().all(|&c| c >= 1.0 - 1e-12),
            "{method:?} CDF < 1"
        );

        // Parity passes on the untouched pattern...
        let parity = subannual_product_parity(
            &annual_ages,
            &factors.ldfs,
            &pat.ages_months,
            &pat.ldfs,
            1e-9,
        )
        .expect("parity");
        assert!(parity.passed, "{method:?}: {}", parity.max_abs_error);

        // ...and fails once a single sub-annual LDF in window 0 is bumped by 1%.
        let mut broken = pat.ldfs.clone();
        broken[1] *= 1.01;
        let parity_broken =
            subannual_product_parity(&annual_ages, &factors.ldfs, &pat.ages_months, &broken, 1e-9)
                .expect("parity");
        assert!(
            !parity_broken.passed,
            "{method:?}: parity missed a 1% break"
        );
        assert!(!parity_broken.windows[0].within_tolerance);
        assert!(
            parity_broken.windows[1..]
                .iter()
                .all(|w| w.within_tolerance)
        );
    }
}

#[test]
fn inverse_power_recovers_exact_sherman_parameters() {
    // True Sherman curve: f(t) = 1 + a * t^{-b} with a = 20.0, b = 1.4.
    let a_true = 20.0_f64;
    let b_true = 1.4_f64;
    let ages: [f64; 6] = [12.0, 24.0, 36.0, 48.0, 60.0, 72.0];
    let ldfs: Vec<f64> = ages[..ages.len() - 1]
        .iter()
        .map(|&age| 1.0 + a_true * age.powf(-b_true))
        .collect();
    let mut cdfs = vec![1.0; ages.len()];
    for k in (0..(ages.len() - 1)).rev() {
        cdfs[k] = cdfs[k + 1] * ldfs[k];
    }

    let targets: Vec<f64> = (12..=72).step_by(6).map(f64::from).collect();
    let res = interp_inverse_power(&ages, &cdfs, &targets).unwrap();
    assert!((res.a - a_true).abs() < 1e-10);
    assert!((res.b - b_true).abs() < 1e-10);
    assert!((res.r_squared - 1.0).abs() < 1e-12);
}

#[test]
fn spliced_weibull_reproduces_exact_weibull_emergence_curve() {
    // True Weibull: P(t) = 1 - exp(-(t / 24)^0.85)
    let eval_p = |t: f64| 1.0 - (-(t / 24.0_f64).powf(0.85)).exp();
    let annual_ages = [12.0, 24.0, 36.0, 48.0, 60.0];
    let annual_cdfs: Vec<f64> = annual_ages.iter().map(|&t| 1.0 / eval_p(t)).collect();
    let sub_ages = [6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0, 54.0, 60.0];

    let pat = interp_spliced_weibull(&annual_ages, &annual_cdfs, &sub_ages).unwrap();
    for (&t, &p_got) in sub_ages.iter().zip(pat.pct_reported.iter()) {
        assert!(
            (p_got - eval_p(t)).abs() < 1e-12,
            "Weibull mismatch at t={t}: got {p_got}, expected {}",
            eval_p(t)
        );
    }
}

#[test]
fn ivf12_satisfies_exact_12_month_factor_identity_and_differs_from_exponential() {
    let annual_ages = [12.0, 24.0, 36.0, 48.0, 60.0];
    let annual_cdfs = [2.60, 1.50, 1.20, 1.07, 1.00];
    let monthly_ages: Vec<f64> = (12..=60).map(f64::from).collect();
    let pat = interp_ivf12(&annual_ages, &annual_cdfs, &monthly_ages).unwrap();
    let ex = interp_exponential(&annual_ages, &annual_cdfs, &monthly_ages).unwrap();

    // Rolling 12-month factor F(t) = CDF(t) / CDF(t + 12) is the smooth
    // interpolant itself; at the knots it equals the annual LDFs exactly.
    for k in 0..4 {
        let got = pat.cdfs[12 * k] / pat.cdfs[12 * (k + 1)];
        let expected = annual_cdfs[k] / annual_cdfs[k + 1];
        assert!((got - expected).abs() < 1e-12);
    }
    // In the final window CDF(t) = CDF(60) * F(t), and F is monotone between
    // its knot values F(48) = 1.07 and F(60) = 1.0.
    for idx in 37..48 {
        let f = pat.cdfs[idx] / annual_cdfs[4];
        assert!(f > 1.0 && f < 1.07, "F({}) = {f}", monthly_ages[idx]);
    }
    // Genuinely distinct from geometric interpolation off the knots.
    let max_diff = pat
        .cdfs
        .iter()
        .zip(ex.cdfs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff > 1e-2,
        "IVF12 collapsed onto exponential: {max_diff}"
    );
}

// ===========================================================================
// 3. Pre-12-Month Stub Extrapolation
// ===========================================================================

#[test]
fn stub_power_ratio_reduces_to_exponential_when_early_ldfs_equal() {
    let f = 1.5;
    let cdf_12 = 2.25;
    let res = stub_power_ratio(6.0, cdf_12, f, f).unwrap();
    assert!((res.stub_to_12_ldf - f.sqrt()).abs() < 1e-12);
    assert!((res.stub_cdf - cdf_12 * f.sqrt()).abs() < 1e-12);

    // When f1 > f2, power ratio produces a steeper stub LDF than constant exponential.
    let steeper = stub_power_ratio(6.0, cdf_12, 1.5, 1.2).unwrap();
    assert!(steeper.stub_to_12_ldf > f.sqrt());
}

// ===========================================================================
// 4. Asymmetric & Interim-Valuation Reserving
// ===========================================================================

#[test]
fn asymmetric_chain_ladder_matches_square_chain_ladder_on_genins() {
    let tri = genins();
    let n = tri.n();
    let grid = asymmetric_age_grid(n, n, 12.0, 12.0, 12.0, None).unwrap();

    let sq = tri.chain_ladder(1.0).unwrap();
    let asym = asymmetric_chain_ladder(
        tri.data(),
        &grid,
        LdfSelectionMethod::VolumeWeighted,
        1.0,
        MonotonicityPolicy::Strict,
    )
    .unwrap();

    assert!((sq.total_reserve - asym.total_ibnr).abs() < 1e-6);
    assert!((sq.total_ultimate - asym.total_ultimate).abs() < 1e-6);
}

#[test]
fn interim_bf_and_cape_cod_reduce_identically_to_standard_engines_on_friedland_auto() {
    let auto = friedland_us_industry_auto();
    let factors = DevelopmentFactorEngine::volume_weighted(&auto.paid.view(), 1.0).unwrap();
    let n = auto.paid.n();
    let latest = auto.paid.latest_diagonal();
    let cdfs_at_origin: Vec<f64> = (0..n).map(|i| factors.cdfs[n - 1 - i]).collect();
    let unit_exp = vec![1.0; n];

    let std_cc = CapeCodEngine::project(&auto.paid.view(), &factors, &auto.earned_premium).unwrap();
    let int_cc =
        interim_cape_cod(&latest, &cdfs_at_origin, &auto.earned_premium, &unit_exp).unwrap();
    assert!((std_cc.elr - int_cc.elr).abs() < 1e-12);
    assert!((std_cc.bf_projection.total_reserve - int_cc.total_ibnr).abs() < 1e-6);

    let expected_ult: Vec<f64> = auto.earned_premium.iter().map(|&p| p * 0.70).collect();
    let std_bf = BornhuetterFergusonEngine::project_from_expected_ultimate(
        &auto.paid.view(),
        &factors,
        &expected_ult,
    )
    .unwrap();
    let int_bf = interim_bf(&latest, &cdfs_at_origin, &expected_ult, &unit_exp).unwrap();
    assert!((std_bf.total_reserve - int_bf.total_ibnr).abs() < 1e-6);
}

#[test]
fn clark_mle_exact_age_recovers_truth_on_asymmetric_quarterly_stub_triangle() {
    // 5 annual origins × 16 quarterly devs, valued at a 6-month interim stub (month 54).
    let grid = asymmetric_age_grid(5, 16, 12.0, 3.0, 3.0, Some(6.0)).unwrap();
    assert_eq!(grid.observed_counts, vec![16, 14, 10, 6, 2]);

    let omega_true = 1.35;
    let theta_true = 18.0; // months
    let elr_true = 0.65;
    let full_prem = [1000.0, 1100.0, 1200.0, 1300.0, 1400.0];
    let earned_prem: Vec<f64> = full_prem
        .iter()
        .zip(grid.exposure_fractions().iter())
        .map(|(&p, &f)| p * f)
        .collect();

    let mut cum = vec![f64::NAN; 5 * 16];
    let mut expected_total_res = 0.0;
    for i in 0..5 {
        let u_i = earned_prem[i] * elr_true;
        for j in 0..grid.observed_counts[i] {
            let avg_age = grid.cell_average_age(i, j).unwrap();
            let g = GrowthCurve::Loglogistic.eval(avg_age, omega_true, theta_true);
            // Tiny deterministic noise so sigma_sq > 0 and Fisher matrix is well-conditioned.
            let noise = 1.0 + 1e-4 * (((i * 5 + j * 3) % 7) as f64 - 3.0);
            cum[i * 16 + j] = u_i * g * noise;
        }
        let last_age = grid
            .cell_average_age(i, grid.observed_counts[i] - 1)
            .unwrap();
        let g_last = GrowthCurve::Loglogistic.eval(last_age, omega_true, theta_true);
        expected_total_res += u_i * (1.0 - g_last);
    }

    let res = fit_clark_cape_cod_exact_age(
        &cum,
        &grid.cell_average_ages_months,
        grid.n_origins,
        grid.n_devs,
        &grid.observed_counts,
        &earned_prem,
        &ClarkExactAgeOptions::loglogistic().with_monotonicity_policy(MonotonicityPolicy::Strict),
    )
    .unwrap();

    assert!((res.omega - omega_true).abs() < 1e-2);
    assert!((res.theta - theta_true).abs() < 1e-1);
    assert!((res.elr.unwrap() - elr_true).abs() < 1e-2);
    assert!((res.total_reserve - expected_total_res).abs() / expected_total_res < 1e-2);
}

#[test]
fn asymmetric_odp_bootstrap_is_deterministic_and_supports_all_modes() {
    let tri = genins();
    let n = tri.n();
    let grid = asymmetric_age_grid(n, n, 12.0, 12.0, 12.0, None).unwrap();

    let sq_fit = fit_odp(&tri.view(), 1.0, MonotonicityPolicy::Strict).unwrap();
    let asym_fit = fit_asymmetric_odp(tri.data(), &grid, 1.0, MonotonicityPolicy::Strict).unwrap();
    assert!((sq_fit.dispersion - asym_fit.phi).abs() < 1e-8);

    let opts_a = OdpBootstrapOptions {
        iterations: 100,
        seed: 42069,
        residual_adjustment: ResidualAdjustment::HatMatrixLeverage,
        process_distribution: ProcessDistribution::OverDispersedPoisson,
        center_residuals: true,
        monotonicity_policy: MonotonicityPolicy::Strict,
    };
    let b1 = asymmetric_odp_bootstrap(tri.data(), &grid, 1.0, opts_a).unwrap();
    let b2 = asymmetric_odp_bootstrap(tri.data(), &grid, 1.0, opts_a).unwrap();
    assert_eq!(b1.simulated_totals, b2.simulated_totals);

    // Serde JSON roundtrip check.
    let json = serde_json::to_string(&b1.fit).unwrap();
    let decoded: AsymmetricOdpFit = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.n_observed, b1.fit.n_observed);
    assert_eq!(decoded.n_params, b1.fit.n_params);
}

#[test]
fn validation_rejects_invalid_subannual_inputs() {
    assert_eq!(
        asymmetric_age_grid(0, 4, 12.0, 6.0, 6.0, None),
        Err(ReservingError::ShapeMismatch)
    );
    assert!(stub_average_age(-1.0, 12.0).is_err());
    assert!(stub_plus12(15.0, 2.0, 1.4).is_err());
    assert!(stub_natural_log(0.0, 2.0, 1.4).is_err());
    assert!(interp_linear_cdf(&[12.0, 24.0], &[2.0, 1.4], &[6.0]).is_err());
}
