use std::hint::black_box;
use std::time::Duration;

use criterion::{
    BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};
use reserving_kernels::{
    capital::aggregate_loss::{FrequencyDistribution, fft_aggregate_loss, panjer_recursion},
    capital::capital_allocation::{euler_tvar_allocation, myers_read_from_default_ratio},
    capital::dependence::{CorrelationType, iman_conover, synchronized_odp_bootstrap},
    capital::reserve_distribution,
    datasets::{
        friedland_us_industry_auto, genins as genins_triangle, quarg_mack_2004, raa as raa_triangle,
    },
    deterministic::double_chain_ladder,
    deterministic::ldf,
    deterministic::munich_chain_ladder,
    excess::excess_layers,
    stochastic::bootstrap_odp::{OdpBootstrapOptions, bootstrap_odp, bootstrap_odp_cdr, fit_odp},
    stochastic::clark_mle::{ClarkOptions, fit_clark_cape_cod, fit_clark_ldf},
    stochastic::glm_reserving::{GlmOptions, fit_glm},
    stochastic::kalman_reserving::{self, KalmanReservingOptions},
    stochastic::mack,
    stochastic::mack_bf::{self, MackBfOptions},
    stochastic::merz_wuthrich,
    stochastic::micro_reserving::{ClaimRecord, MicroReservingModel, simulate_reserves},
    stochastic::multivariate_cl::{
        MultivariateClOptions, MultivariateModel, analyze_multivariate_cl,
    },
    stochastic::multivariate_mack,
    stochastic::paid_incurred_chain::{self, PicOptions},
    stochastic::ptf_regression::{self, PtfOptions},
    stochastic::reserve_runoff::{mack_one_year_runoff, odp_one_year_runoff},
    subannual::interp::interp_swimon,
    types::Triangle,
    validate::MonotonicityPolicy,
};

fn synthetic_cumulative_triangle(n: usize) -> Triangle {
    let mut data = vec![f64::NAN; n * n];
    for i in 0..n {
        let mut acc = 1_000.0 + (i as f64 * 17.0);
        for j in 0..(n - i) {
            acc += 10.0 + (j as f64 * 0.25) + ((i + j) % 3) as f64 * 0.7;
            data[i * n + j] = acc;
        }
    }
    Triangle::try_new(n, n, data).expect("valid triangle")
}

fn bench_ldf_and_deterministic(c: &mut Criterion) {
    let mut group = c.benchmark_group("deterministic");
    let tri40 = synthetic_cumulative_triangle(40);
    let view40 = tri40.view();
    let mut out40 = vec![0.0; 39];

    group.throughput(Throughput::Elements((40 * 41 / 2) as u64));
    group.bench_function("volume_weighted_ldfs_40x40", |b| {
        b.iter(|| ldf::volume_weighted_ldfs(black_box(tri40.data()), 40, 40, &mut out40));
    });

    let raa = raa_triangle();
    let raa_view = raa.view();
    group.bench_function("mack_raa_10x10", |b| {
        b.iter(|| {
            mack::analyze_with_policy(
                black_box(&raa_view),
                1.0,
                MonotonicityPolicy::AllowNegativeIncrements,
            )
            .expect("mack raa")
        });
    });

    group.bench_function("mack_40x40", |b| {
        b.iter(|| mack::analyze(black_box(&view40), 1.0).expect("mack 40"));
    });

    group.bench_function("merz_wuthrich_raa_10x10", |b| {
        b.iter(|| {
            merz_wuthrich::analyze_with_policy(
                black_box(&raa_view),
                MonotonicityPolicy::AllowNegativeIncrements,
            )
            .expect("mw raa")
        });
    });

    group.bench_function("merz_wuthrich_40x40", |b| {
        b.iter(|| merz_wuthrich::analyze(black_box(&view40)).expect("mw 40"));
    });
    group.finish();
}

fn bench_mle_and_glm(c: &mut Criterion) {
    let mut group = c.benchmark_group("mle_glm");
    let genins = genins_triangle();
    let view = genins.view();

    let clark_opts = ClarkOptions::loglogistic();
    group.bench_function("clark_loglogistic_genins", |b| {
        b.iter(|| fit_clark_ldf(black_box(&view), black_box(&clark_opts)).expect("clark"));
    });

    let clark_w = ClarkOptions::weibull();
    group.bench_function("clark_weibull_genins", |b| {
        b.iter(|| fit_clark_ldf(black_box(&view), black_box(&clark_w)).expect("clark weibull"));
    });

    let odp_opts = GlmOptions::odp();
    group.bench_function("glm_odp_genins", |b| {
        b.iter(|| fit_glm(black_box(&view), black_box(&odp_opts)).expect("glm odp"));
    });

    let gamma_opts = GlmOptions::gamma();
    group.bench_function("glm_gamma_genins", |b| {
        b.iter(|| fit_glm(black_box(&view), black_box(&gamma_opts)).expect("glm gamma"));
    });

    group.bench_function("odp_fit_genins", |b| {
        b.iter(|| fit_odp(black_box(&view), 1.0, MonotonicityPolicy::Strict).expect("odp fit"));
    });
    group.finish();
}

fn bench_bootstrap(c: &mut Criterion) {
    let mut group = c.benchmark_group("bootstrap");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(30);
    let genins = genins_triangle();
    let view = genins.view();

    for iters in [1_000usize, 10_000usize] {
        let opts = OdpBootstrapOptions::shapland(iters, 42069);
        group.throughput(Throughput::Elements(iters as u64));
        group.bench_with_input(BenchmarkId::new("odp_genins", iters), &opts, |b, &o| {
            b.iter(|| bootstrap_odp(black_box(&view), 1.0, o).expect("boot"));
        });
    }

    let cdr_opts = OdpBootstrapOptions::shapland(1_000, 42069);
    group.throughput(Throughput::Elements(1_000));
    group.bench_function("odp_cdr_genins_1000", |b| {
        b.iter(|| bootstrap_odp_cdr(black_box(&view), 1.0, cdr_opts).expect("cdr"));
    });

    let sync_opts = OdpBootstrapOptions::shapland(500, 42069);
    let genins2 = Triangle::try_new(
        genins.n_origins(),
        genins.n_devs(),
        genins
            .data()
            .iter()
            .map(|&v| if v.is_finite() { v * 1.35 } else { f64::NAN })
            .collect(),
    )
    .expect("valid triangle");
    let views = [genins.view(), genins2.view()];
    group.throughput(Throughput::Elements(500));
    group.bench_function("synchronized_odp_2line_500", |b| {
        b.iter(|| {
            synchronized_odp_bootstrap(black_box(&views), &[1.0, 1.0], sync_opts).expect("sync")
        });
    });
    group.finish();
}

fn bench_dependence_and_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("dependence_dist");
    let n = 2_000usize;
    let x: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 1.13).collect();
    let y: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % n) as f64 + 1.0).collect();
    let z: Vec<f64> = (0..n).map(|i| ((i * 13 + 5) % n) as f64 + 1.0).collect();
    let corr = [1.0, 0.5, 0.2, 0.5, 1.0, 0.3, 0.2, 0.3, 1.0];

    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("iman_conover_2000x3", |b| {
        b.iter(|| {
            iman_conover(
                black_box(&[&x[..], &y[..], &z[..]]),
                black_box(&corr),
                CorrelationType::SpearmanTarget,
                42069,
            )
            .expect("ic")
        });
    });

    let samples: Vec<f64> = (0..10_000)
        .map(|i| ((i * 2_654_435_761usize) % 10_000) as f64 * 123.45)
        .collect();
    group.throughput(Throughput::Elements(samples.len() as u64));
    group.bench_function("reserve_dist_summarize_10k", |b| {
        b.iter(|| reserve_distribution::summarize(black_box(&samples)).expect("sum"));
    });
    group.bench_function("reserve_dist_tvar_coherent_10k", |b| {
        b.iter(|| reserve_distribution::tvar_coherent(black_box(&samples), 0.95).expect("tvar"));
    });
    group.finish();
}

fn bench_aggregate_and_excess(c: &mut Criterion) {
    let mut group = c.benchmark_group("aggregate_excess");
    let freq = FrequencyDistribution::Poisson { lambda: 5.0 };
    let mut sev = vec![0.0; 64];
    let mut sum = 0.0;
    for (k, slot) in sev.iter_mut().enumerate().skip(1) {
        let w = (-(k as f64) / 8.0).exp();
        *slot = w;
        sum += w;
    }
    for slot in &mut sev {
        *slot /= sum;
    }

    group.bench_function("panjer_recursion_512", |b| {
        b.iter(|| panjer_recursion(freq, black_box(&sev), 100.0, 512).expect("panjer"));
    });
    group.bench_function("fft_aggregate_loss_4096", |b| {
        b.iter(|| fft_aggregate_loss(freq, black_box(&sev), 100.0, 4096, None).expect("fft"));
    });

    let layer_model = excess_layers::LognormalLayerModel::try_new(100_000.0, 1.0).expect("model");
    let layer_sampler =
        excess_layers::LognormalLayerSampler::try_new(layer_model, 65_536, 2026).expect("sampler");
    group.bench_function("excess_layers_lookup_65k", |b| {
        b.iter(|| layer_sampler.simulated_layer(black_box(100_000.0), black_box(350_000.0)));
    });
    group.bench_function("excess_layers_build_65k", |b| {
        b.iter(|| {
            excess_layers::LognormalLayerSampler::try_new(
                black_box(layer_model),
                65_536,
                black_box(2026),
            )
            .expect("sampler")
        });
    });
    group.finish();
}

fn bench_advanced_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("advanced");
    let genins = genins_triangle();
    let genins_view = genins.view();
    let auto = friedland_us_industry_auto();
    let paid_view = auto.paid.view();
    let inc_view = auto.reported.view();

    let kalman_opts = KalmanReservingOptions::default();
    group.bench_function("kalman_genins_10x10", |b| {
        b.iter(|| {
            kalman_reserving::analyze(black_box(&genins_view), black_box(kalman_opts))
                .expect("kalman")
        });
    });

    let ptf_opts = PtfOptions::default();
    group.bench_function("ptf_genins_10x10", |b| {
        b.iter(|| {
            ptf_regression::fit_ptf(black_box(&genins_view), black_box(&ptf_opts)).expect("ptf")
        });
    });

    let pic_opts = PicOptions::default();
    group.bench_function("pic_friedland_10x10", |b| {
        b.iter(|| {
            paid_incurred_chain::analyze(
                black_box(&paid_view),
                black_box(&inc_view),
                black_box(&pic_opts),
            )
            .expect("pic")
        });
    });

    let tri40_a = synthetic_cumulative_triangle(40);
    let mut tri40_b_data = vec![f64::NAN; 40 * 40];
    for i in 0..40 {
        let mut acc = 1_300.0 + (i as f64 * 23.0);
        for j in 0..(40 - i) {
            acc += 14.0 + (j as f64 * 0.35) + ((i * 3 + j) % 5) as f64 * 0.9;
            tri40_b_data[i * 40 + j] = acc;
        }
    }
    let tri40_b = Triangle::try_new(40, 40, tri40_b_data).expect("valid triangle");
    let mv_views_40 = [tri40_a.view(), tri40_b.view()];
    group.bench_function("multivariate_mack_2line_40x40", |b| {
        b.iter(|| multivariate_mack::analyze(black_box(&mv_views_40), 1.0).expect("mv mack"));
    });

    let mv_views_10 = [paid_view, inc_view];
    let gmcl_opts = MultivariateClOptions {
        model: MultivariateModel::GeneralGmcl,
        policy: MonotonicityPolicy::AllowNegativeIncrements,
    };
    group.bench_function("multivariate_cl_gmcl_2line_10x10", |b| {
        b.iter(|| {
            analyze_multivariate_cl(black_box(&mv_views_10), black_box(&gmcl_opts)).expect("gmcl")
        });
    });

    group.bench_function("dcl_40x40", |b| {
        b.iter(|| {
            double_chain_ladder::analyze(black_box(&mv_views_40[0]), black_box(&mv_views_40[1]))
                .expect("dcl")
        });
    });

    let n_scen = 10_000usize;
    let lines_owned: Vec<Vec<f64>> = (0..5)
        .map(|l| {
            (0..n_scen)
                .map(|i| {
                    (((i * (2_654_435_761usize + l * 97)) % n_scen) as f64 + 10.0) * (l + 1) as f64
                })
                .collect()
        })
        .collect();
    let line_slices: Vec<&[f64]> = lines_owned.iter().map(Vec::as_slice).collect();
    group.bench_function("euler_tvar_10k_x_5", |b| {
        b.iter(|| euler_tvar_allocation(black_box(&line_slices), 0.95).expect("euler tvar"));
    });
    group.bench_function("myers_read_ratio_10k_x_5", |b| {
        b.iter(|| {
            myers_read_from_default_ratio(black_box(&line_slices), 0.01).expect("myers read")
        });
    });

    let micro_model = MicroReservingModel {
        occurrence_intensities: vec![40.0; 5],
        reporting_delay_rate: 2.5,
        interim_payment_hazard: 1.2,
        settlement_hazard: 0.8,
        interim_severity_mu: 7.0,
        interim_severity_sigma: 0.6,
        settlement_payment_prob: 0.75,
        settlement_severity_mu: 8.0,
        settlement_severity_sigma: 0.7,
    };
    let open_claims: Vec<ClaimRecord> = (0..30)
        .map(|id| ClaimRecord {
            claim_id: id,
            occurrence_time: 4.0 + (id as f64) * 0.02,
            report_time: 4.1 + (id as f64) * 0.02,
            settlement_time: None,
            payments: Vec::new(),
        })
        .collect();
    group.bench_function("micro_simulate_1000", |b| {
        b.iter(|| {
            simulate_reserves(
                black_box(&open_claims),
                black_box(&micro_model),
                5.0,
                1_000,
                42069,
            )
            .expect("micro")
        });
    });

    // High-value reserving, 1-year run-off, and sub-annual kernels
    let qm = quarg_mack_2004();
    let qm_paid = qm.paid.view();
    let qm_inc = qm.incurred.view();
    group.bench_function("munich_cl_quarg_mack_7x7", |b| {
        b.iter(|| {
            munich_chain_ladder::project(black_box(&qm_paid), black_box(&qm_inc), 1.0, 1.0)
                .expect("munich cl")
        });
    });

    let friedland = friedland_us_industry_auto();
    let friedland_rep = friedland.reported.view();
    let expected_ult: Vec<f64> = friedland.earned_premium.iter().map(|&p| p * 0.72).collect();
    group.bench_function("bf_friedland_10x10", |b| {
        b.iter(|| friedland_rep.bf(black_box(&expected_ult), 1.0).expect("bf"));
    });
    group.bench_function("cape_cod_friedland_10x10", |b| {
        b.iter(|| {
            friedland_rep
                .cape_cod(black_box(&friedland.earned_premium), 1.0)
                .expect("cape cod")
        });
    });
    let clark_opts = ClarkOptions::default();
    group.bench_function("clark_cape_cod_friedland_10x10", |b| {
        b.iter(|| {
            fit_clark_cape_cod(
                black_box(&friedland_rep),
                black_box(&friedland.earned_premium),
                black_box(&clark_opts),
            )
            .expect("clark cape cod")
        });
    });

    let genins_prior = vec![5_000_000.0; 10];
    let genins_prior_se = vec![500_000.0; 10];
    group.bench_function("mack_bf_genins_10x10", |b| {
        b.iter(|| {
            mack_bf::analyze_from_triangle(
                black_box(&genins_view),
                black_box(&genins_prior),
                black_box(&genins_prior_se),
                MackBfOptions::default(),
            )
            .expect("mack bf")
        });
    });

    group.bench_function("mack_one_year_runoff_genins_1000", |b| {
        b.iter(|| {
            mack_one_year_runoff(black_box(&genins_view), 1_000, 42069).expect("mack runoff")
        });
    });

    let odp_runoff_opts = OdpBootstrapOptions {
        iterations: 1_000,
        seed: 42069,
        ..OdpBootstrapOptions::default()
    };
    group.bench_function("odp_one_year_runoff_genins_1000", |b| {
        b.iter(|| {
            odp_one_year_runoff(black_box(&genins_view), 1.0, odp_runoff_opts).expect("odp runoff")
        });
    });

    let anchor_ages = [12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 96.0, 108.0, 120.0];
    let anchor_cdfs = [3.49, 1.75, 1.35, 1.18, 1.09, 1.04, 1.02, 1.01, 1.003, 1.0];
    let target_ages: Vec<f64> = (1..=120).map(f64::from).collect();
    group.bench_function("subannual_swimon_120m", |b| {
        b.iter(|| {
            interp_swimon(
                black_box(&anchor_ages),
                black_box(&anchor_cdfs),
                black_box(&target_ages),
            )
            .expect("swimon")
        });
    });

    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(600))
        .measurement_time(Duration::from_secs(2))
        .noise_threshold(0.02)
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets =
        bench_ldf_and_deterministic,
        bench_mle_and_glm,
        bench_bootstrap,
        bench_dependence_and_distribution,
        bench_aggregate_and_excess,
        bench_advanced_kernels
}
criterion_main!(benches);
