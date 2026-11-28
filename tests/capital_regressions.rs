//! Regression tests for capital / excess numerical review findings:
//! truncated-grid TVaR, TVaR monotonicity, high-theta Archimedean samplers,
//! Student-t calibration, high-layer precision, FFT aliasing detection,
//! Panjer scaling-and-squaring, Euler / Myers–Read marginal conditions and
//! Higham (2002) nearest-correlation repair.
//!
//! All stochastic fixtures use seed 42069.

use reserving_kernels::capital::aggregate_loss::{
    FftOptions, FrequencyDistribution, discretize_rounding, fft_aggregate_loss,
    fft_aggregate_loss_with_options, panjer_recursion,
};
use reserving_kernels::capital::capital_allocation::{
    euler_sd_allocation, euler_tvar_allocation, myers_read_allocation,
};
use reserving_kernels::capital::copulas::{
    clayton_copula_sample, gumbel_copula_sample, student_t_copula_sample,
};
use reserving_kernels::capital::correlation::{
    CorrelationType, HighamOptions, nearest_correlation_matrix,
};
use reserving_kernels::capital::reserve_distribution::tvar_coherent;
use reserving_kernels::excess::severity::SeverityDistribution;

const SEED: u64 = 42069;

/// Test-local SplitMix64 + Box–Muller generator (fixture data only).
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
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn ks_uniform(vals: &[f64]) -> f64 {
    let mut sorted = vals.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len() as f64;
    sorted
        .iter()
        .enumerate()
        .map(|(i, &u)| ((i + 1) as f64 / n - u).abs().max((u - i as f64 / n).abs()))
        .fold(0.0, f64::max)
}

/// O(n^2) Kendall tau-a on the first `m` pairs.
fn kendall_tau(x: &[f64], y: &[f64], m: usize) -> f64 {
    let m = m.min(x.len());
    let mut s = 0i64;
    for i in 0..m {
        for j in (i + 1)..m {
            let p = (x[i] - x[j]) * (y[i] - y[j]);
            if p > 0.0 {
                s += 1;
            } else if p < 0.0 {
                s -= 1;
            }
        }
    }
    s as f64 / (m * (m - 1) / 2) as f64
}

fn l_inf(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------
// Findings 1 & 2: truncated-grid TVaR and monotonicity incl. alpha = 1
// ---------------------------------------------------------------------------

#[test]
fn truncated_heavy_tail_grid_rejects_tvar_instead_of_renormalizing() {
    let sev = SeverityDistribution::pareto(1.5, 50_000.0);
    let step = 10_000.0;
    let n = 2048usize;
    let pmf = discretize_rounding(&sev, step, n - 1, true).expect("disc");
    let agg = panjer_recursion(FrequencyDistribution::poisson(20.0), &pmf, step, n).expect("p");
    assert!(agg.missing_mass() > 1e-4, "missing={}", agg.missing_mass());
    assert!(agg.tvar(0.99).is_err());
    assert!(agg.tvar(1.0).is_err());
    assert!(agg.var(1.0).is_err());
    // A loose explicit budget gives a lower bound that dominates VaR.
    let tv = agg.tvar_with_tolerance(0.99, 1.0).expect("loose");
    assert!(tv >= agg.var(0.99).expect("var"));
}

#[test]
fn tvar_is_monotone_in_alpha_including_one_and_matches_mean_at_zero() {
    let sev = [0.0, 0.25, 0.35, 0.25, 0.15];
    let agg = panjer_recursion(FrequencyDistribution::poisson(3.0), &sev, 100.0, 256).expect("p");
    let alphas = [
        0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999, 1.0,
    ];
    let mut prev = f64::NEG_INFINITY;
    for &a in &alphas {
        let t = agg.tvar(a).expect("tvar");
        assert!(t >= prev - 1e-9, "alpha={a} tvar={t} prev={prev}");
        assert!(t >= agg.var(a).expect("var") - 1e-9);
        prev = t;
    }
    let top = agg.tvar(1.0).expect("tvar1");
    assert_eq!(top, agg.var(1.0).expect("var1"));
    let k_top = agg.pmf.iter().rposition(|&p| p > 0.0).expect("support");
    assert_eq!(top, k_top as f64 * agg.step);
    assert!((agg.tvar(0.0).expect("tvar0") - agg.mean).abs() < 1e-9 * agg.mean);
}

// ---------------------------------------------------------------------------
// Finding 3: Clayton / Gumbel samplers at high theta
// ---------------------------------------------------------------------------

#[test]
fn clayton_marginals_and_kendall_tau_hold_up_to_theta_500() {
    let n = 20_000usize;
    let ks_crit = 1.63 / (n as f64).sqrt();
    for theta in [2.0, 20.0, 200.0, 500.0] {
        let s = clayton_copula_sample(2, theta, n, SEED).expect("clayton");
        for col in &s.columns {
            let ks = ks_uniform(col);
            assert!(ks < ks_crit, "theta={theta} ks={ks}");
        }
        let tau = kendall_tau(&s.columns[0], &s.columns[1], 2_000);
        let expected = theta / (theta + 2.0);
        let tol = if theta > 100.0 { 0.003 } else { 0.03 };
        assert!(
            (tau - expected).abs() < tol,
            "theta={theta} tau={tau} expected={expected}"
        );
    }
}

#[test]
fn gumbel_marginals_and_kendall_tau_hold_at_high_theta() {
    let n = 20_000usize;
    let ks_crit = 1.63 / (n as f64).sqrt();
    for theta in [2.0, 10.0, 50.0, 200.0] {
        let s = gumbel_copula_sample(2, theta, n, SEED).expect("gumbel");
        for col in &s.columns {
            let ks = ks_uniform(col);
            assert!(ks < ks_crit, "theta={theta} ks={ks}");
        }
        let tau = kendall_tau(&s.columns[0], &s.columns[1], 2_000);
        let expected = 1.0 - 1.0 / theta;
        let tol = if theta > 20.0 { 0.005 } else { 0.03 };
        assert!(
            (tau - expected).abs() < tol,
            "theta={theta} tau={tau} expected={expected}"
        );
    }
}

// ---------------------------------------------------------------------------
// Finding 4: Student-t calibration
// ---------------------------------------------------------------------------

#[test]
fn student_t_kendall_calibration_is_exact_and_spearman_is_rejected() {
    let target = [1.0, 0.5, 0.5, 1.0];
    for nu in [2.0, 5.0] {
        let s =
            student_t_copula_sample(&target, 2, nu, CorrelationType::KendallTarget, 20_000, SEED)
                .expect("t kendall");
        let tau = kendall_tau(&s.columns[0], &s.columns[1], 2_500);
        assert!((tau - 0.5).abs() < 0.03, "nu={nu} tau={tau}");
    }
    let err = student_t_copula_sample(&target, 2, 2.0, CorrelationType::SpearmanTarget, 100, SEED);
    assert!(err.is_err());
}

// ---------------------------------------------------------------------------
// Finding 5: high-layer precision
// ---------------------------------------------------------------------------

#[test]
fn exponential_high_layers_match_analytic_to_1e_10() {
    let theta = 1_000.0;
    let dist = SeverityDistribution::exponential(theta);
    for mult in [30.0, 34.0, 37.0] {
        let a = mult * theta;
        let l = theta;
        let analytic = theta * (-a / theta).exp() * (1.0 - (-l / theta).exp());
        let layer = dist.layer_expected_loss(a, l).expect("layer");
        let rel = (layer - analytic).abs() / analytic;
        assert!(
            rel < 1e-10,
            "a={a} layer={layer} analytic={analytic} rel={rel}"
        );

        let xs = dist.excess_expected_value(a).expect("excess");
        let xs_exact = theta * (-a / theta).exp();
        assert!((xs - xs_exact).abs() / xs_exact < 1e-12);
        let ratio = dist.excess_ratio(a).expect("ratio");
        assert!((ratio - (-mult).exp()).abs() / (-mult).exp() < 1e-12);
    }
}

/// Composite Gauss–Legendre (8-point) on `[lo, hi]` with `panels` panels in log-x.
fn integrate_survival(dist: &SeverityDistribution, lo: f64, hi: f64, panels: usize) -> f64 {
    const X: [f64; 4] = [
        0.183_434_642_495_649_8,
        0.525_532_409_916_329,
        0.796_666_477_413_626_7,
        0.960_289_856_497_536_3,
    ];
    const W: [f64; 4] = [
        0.362_683_783_378_362,
        0.313_706_645_877_887_3,
        0.222_381_034_453_374_5,
        0.101_228_536_290_376_3,
    ];
    let (ll, lh) = (lo.ln(), hi.ln());
    let h = (lh - ll) / panels as f64;
    let mut acc = 0.0;
    for p in 0..panels {
        let mid = ll + (p as f64 + 0.5) * h;
        for (&x, &w) in X.iter().zip(W.iter()) {
            for t in [mid - 0.5 * h * x, mid + 0.5 * h * x] {
                let xv = t.exp();
                acc += 0.5 * h * w * dist.survival(xv) * xv;
            }
        }
    }
    acc
}

#[test]
fn heavy_tail_high_layers_match_quadrature() {
    let dists = [
        SeverityDistribution::lognormal(10.0, 1.2),
        SeverityDistribution::pareto(2.5, 150_000.0),
        SeverityDistribution::single_parameter_pareto(2.0, 50_000.0),
        SeverityDistribution::gpd(0.35, 80_000.0),
        SeverityDistribution::mixed_exponential(vec![0.7, 0.3], vec![40_000.0, 240_000.0]),
    ];
    for dist in &dists {
        for (a, l) in [(5.0e6, 1.0e6), (2.0e7, 5.0e6), (1.0e5, 1.0e3)] {
            let exact = dist.layer_expected_loss(a, l).expect("layer");
            let quad = integrate_survival(dist, a, a + l, 64);
            assert!(
                (exact - quad).abs() <= 1e-9 * quad,
                "dist={dist:?} a={a} l={l} exact={exact} quad={quad}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Findings 6 & 7: FFT aliasing detection and Panjer scaling-and-squaring
// ---------------------------------------------------------------------------

#[test]
fn fft_matches_panjer_on_heavy_tailed_pareto_and_flags_undersized_grid() {
    let sev = SeverityDistribution::pareto(1.5, 50_000.0);
    let step = 10_000.0;
    let pmf = discretize_rounding(&sev, step, 1024, true).expect("disc");
    let freq = FrequencyDistribution::poisson(20.0);
    let p = panjer_recursion(freq, &pmf, step, 4096).expect("panjer");
    let f = fft_aggregate_loss_with_options(freq, &pmf, step, 4096, &FftOptions::default())
        .expect("fft");
    let diff = l_inf(&p.pmf, &f.pmf);
    assert!(diff < 1e-10, "L_inf={diff}");
    assert!(f.aliasing_error_estimate < 1e-6);
    assert!((p.tvar(0.99).expect("p") - f.tvar(0.99).expect("f")).abs() < 1e-6 * p.mean);

    // Undersized, untilted grid: wrap-around keeps total_mass ~ 1 but is detected.
    let small = discretize_rounding(&sev, step, 63, true).expect("disc small");
    let aliased = fft_aggregate_loss(freq, &small, step, 64, Some(0.0)).expect("legacy");
    assert!((aliased.total_mass - 1.0).abs() < 1e-9);
    assert!(
        aliased.aliasing_error_estimate > 1e-3,
        "est={}",
        aliased.aliasing_error_estimate
    );
    assert!(aliased.relative_mean_error > 0.01);
    assert!(aliased.tvar(0.99).is_err());
    assert!(aliased.var(0.99).is_err());
    let strict = FftOptions {
        tilt: Some(0.0),
        ..FftOptions::default()
    };
    assert!(fft_aggregate_loss_with_options(freq, &small, step, 64, &strict).is_err());
}

#[test]
fn panjer_poisson_lambda_1000_uses_scaling_and_squaring_and_matches_fft() {
    let sev = [0.0, 0.25, 0.35, 0.25, 0.15];
    let freq = FrequencyDistribution::poisson(1000.0);
    let p = panjer_recursion(freq, &sev, 1.0, 4096).expect("panjer lambda 1000");
    assert!(p.scaling_squaring_steps >= 1);
    assert!((p.total_mass - 1.0).abs() < 1e-10);
    assert!((p.mean - p.exact_theoretical_mean).abs() / p.exact_theoretical_mean < 1e-10);
    assert!(
        (p.variance - p.exact_theoretical_variance).abs() / p.exact_theoretical_variance < 1e-8
    );
    let f = fft_aggregate_loss(freq, &sev, 1.0, 4096, None).expect("fft");
    let diff = l_inf(&p.pmf, &f.pmf);
    assert!(diff < 1e-10, "L_inf={diff}");

    // Negative Binomial with large r is also split.
    let nb = FrequencyDistribution::negative_binomial(800.0, 0.5);
    let pn = panjer_recursion(nb, &sev, 1.0, 4096).expect("nb");
    assert!(pn.scaling_squaring_steps >= 1);
    let fnb = fft_aggregate_loss(nb, &sev, 1.0, 4096, None).expect("fft nb");
    assert!(l_inf(&pn.pmf, &fnb.pmf) < 1e-10);

    // Binomial reports clipped negative mass (zero or tiny) explicitly.
    let b = panjer_recursion(FrequencyDistribution::binomial(40, 0.3), &sev, 1.0, 256)
        .expect("binomial");
    assert!(b.clipped_negative_mass >= 0.0 && b.clipped_negative_mass < 1e-6);
}

// ---------------------------------------------------------------------------
// Euler and Myers–Read marginal conditions
// ---------------------------------------------------------------------------

fn fixture_lines(n: usize) -> Vec<Vec<f64>> {
    let mut rng = TestRng(SEED);
    let mut lines = vec![
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    for _ in 0..n {
        let common = rng.normal();
        let z0 = 0.6 * common + 0.8 * rng.normal();
        let z1 = 0.3 * common + (1.0_f64 - 0.09).sqrt() * rng.normal();
        let z2 = rng.normal();
        lines[0].push((10.0 + 0.8 * z0).exp());
        lines[1].push((11.0 + 1.1 * z1).exp());
        lines[2].push(50_000.0 + 8_000.0 * z2);
    }
    lines
}

#[test]
fn euler_tvar_and_sd_allocations_match_brute_force_directional_derivatives() {
    let lines = fixture_lines(2_000);
    let refs: Vec<&[f64]> = lines.iter().map(Vec::as_slice).collect();
    let n = lines[0].len();
    let total: Vec<f64> = (0..n).map(|k| lines.iter().map(|l| l[k]).sum()).collect();

    for alpha in [0.9, 0.95, 0.9537, 0.99] {
        let euler = euler_tvar_allocation(&refs, alpha).expect("euler");
        for (i, line) in lines.iter().enumerate() {
            let h = 1e-7;
            let up: Vec<f64> = total.iter().zip(line).map(|(s, x)| s + h * x).collect();
            let dn: Vec<f64> = total.iter().zip(line).map(|(s, x)| s - h * x).collect();
            let fd = (tvar_coherent(&up, alpha).expect("up")
                - tvar_coherent(&dn, alpha).expect("dn"))
                / (2.0 * h);
            let ac = euler.allocated_total[i];
            assert!(
                (fd - ac).abs() <= 1e-5 * ac.abs().max(1.0),
                "alpha={alpha} line={i} fd={fd} euler={ac}"
            );
        }
    }

    let kappa = 2.0;
    let sd_alloc = euler_sd_allocation(&refs, kappa).expect("sd");
    let rho = |s: &[f64]| {
        let nf = s.len() as f64;
        let m = s.iter().sum::<f64>() / nf;
        let v = s.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (nf - 1.0);
        m + kappa * v.sqrt()
    };
    for (i, line) in lines.iter().enumerate() {
        let h = 1e-6;
        let up: Vec<f64> = total.iter().zip(line).map(|(s, x)| s + h * x).collect();
        let dn: Vec<f64> = total.iter().zip(line).map(|(s, x)| s - h * x).collect();
        let fd = (rho(&up) - rho(&dn)) / (2.0 * h);
        let ac = sd_alloc.allocated_total[i];
        assert!(
            (fd - ac).abs() <= 1e-6 * ac.abs(),
            "line={i} fd={fd} euler_sd={ac}"
        );
    }
}

#[test]
fn myers_read_allocation_satisfies_marginal_default_value_condition() {
    let lines = fixture_lines(3_000);
    let refs: Vec<&[f64]> = lines.iter().map(Vec::as_slice).collect();
    let n = lines[0].len();
    let total: Vec<f64> = (0..n).map(|k| lines.iter().map(|l| l[k]).sum()).collect();
    let mut sorted = total.clone();
    sorted.sort_by(f64::total_cmp);
    let assets = sorted[(0.98 * n as f64) as usize] + 0.5;

    let mr = myers_read_allocation(&refs, assets).expect("myers read");
    let nf = n as f64;
    // D(eps) / L(eps) when line i grows by (1 + eps) and brings eps * a_i of assets.
    let ratio = |i: usize, a_i: f64, eps: f64| {
        let d: f64 = (0..n)
            .map(|k| (total[k] + eps * (lines[i][k] - a_i) - assets).max(0.0))
            .sum::<f64>()
            / nf;
        let l = mr.portfolio_liability + eps * mr.line_liabilities[i];
        d / l
    };
    let base = mr.default_value_ratio;
    for i in 0..lines.len() {
        let a_i = mr.allocated_assets[i];
        let eps = 1e-6;
        let deriv = (ratio(i, a_i, eps) - ratio(i, a_i, -eps)) / (2.0 * eps);
        assert!(
            deriv.abs() <= 1e-6 * base,
            "line={i} d(D/L)/d eps={deriv} base={base}"
        );
        // A mis-allocated asset share violates the marginal condition.
        let bumped = a_i + 0.05 * mr.line_liabilities[i];
        let deriv_bad = (ratio(i, bumped, eps) - ratio(i, bumped, -eps)) / (2.0 * eps);
        assert!(deriv_bad.abs() > 1e-3 * base, "line={i} bad={deriv_bad}");
    }
}

// ---------------------------------------------------------------------------
// Finding 9: Higham (2002) nearest correlation matrix
// ---------------------------------------------------------------------------

#[test]
fn higham_2002_three_by_three_example() {
    #[rustfmt::skip]
    let a = [
        1.0, 1.0, 0.0,
        1.0, 1.0, 1.0,
        0.0, 1.0, 1.0,
    ];
    let opts = HighamOptions {
        min_eigenvalue: 0.0,
        ..HighamOptions::default()
    };
    let res = nearest_correlation_matrix(&a, 3, &opts).expect("higham");
    #[rustfmt::skip]
    let expected = [
        1.0,    0.7607, 0.1573,
        0.7607, 1.0,    0.7607,
        0.1573, 0.7607, 1.0,
    ];
    for (got, want) in res.matrix.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-4, "got={got} want={want}");
    }
    let dist_expected = res
        .matrix
        .iter()
        .zip(a.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt();
    assert!((res.frobenius_distance - dist_expected).abs() < 1e-12);

    // Deterministic: identical output on a repeated call.
    let again = nearest_correlation_matrix(&a, 3, &opts).expect("again");
    assert_eq!(res, again);

    // Default (strictly PD floor) output admits a Cholesky factor, i.e. is
    // accepted by the Student-t sampler under LatentGaussian semantics.
    let pd = nearest_correlation_matrix(&a, 3, &HighamOptions::default()).expect("pd");
    assert!(
        student_t_copula_sample(
            &pd.matrix,
            3,
            4.0,
            CorrelationType::LatentGaussian,
            10,
            SEED
        )
        .is_ok()
    );
}
