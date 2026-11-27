use std::time::Duration;

use criterion::{
    BenchmarkId, Criterion, SamplingMode, Throughput, black_box, criterion_group, criterion_main,
};
use reserving_kernels::{
    aggregate_loss::{FrequencyDistribution, fft_aggregate_loss, panjer_recursion},
    bootstrap_odp::{OdpBootstrapOptions, bootstrap_odp, bootstrap_odp_cdr, fit_odp},
    clark_mle::{ClarkOptions, fit_clark_ldf},
    datasets::{genins as genins_triangle, raa as raa_triangle},
    dependence::{CorrelationType, iman_conover, synchronized_odp_bootstrap},
    excess_layers,
    glm_reserving::{GlmOptions, fit_glm},
    ldf, mack, merz_wuthrich, reserve_distribution,
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
    Triangle::new(n, n, data)
}

fn bench_ldf_and_deterministic(c: &mut Criterion) {
    let mut group = c.benchmark_group("deterministic");
    let tri40 = synthetic_cumulative_triangle(40);
    let view40 = tri40.view();
    let mut out40 = vec![0.0; 39];

    group.throughput(Throughput::Elements((40 * 41 / 2) as u64));
    group.bench_function("volume_weighted_ldfs_40x40", |b| {
        b.iter(|| ldf::volume_weighted_ldfs(black_box(&tri40.data), 40, 40, &mut out40));
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
    let genins2 = Triangle::new(
        genins.n_origins,
        genins.n_devs,
        genins
            .data
            .iter()
            .map(|&v| if v.is_finite() { v * 1.35 } else { f64::NAN })
            .collect(),
    );
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

    group.bench_function("excess_layers_simulated_65k", |b| {
        b.iter(|| excess_layers::simulated_layer(black_box(1), black_box(350_000.0), black_box(1)));
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
        bench_aggregate_and_excess
}
criterion_main!(benches);
