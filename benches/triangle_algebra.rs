use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use reserving_kernels::{
    capital::dependence::{CorrelationType, iman_conover},
    preprocessing::triangle,
    stochastic::bootstrap_odp::{OdpBootstrapOptions, bootstrap_odp, fit_odp},
    stochastic::mack,
    types::Triangle,
    validate::MonotonicityPolicy,
};

fn right_cumulative_triangle(n: usize) -> Triangle {
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

fn bench_latest_diagonal(c: &mut Criterion) {
    let mut group = c.benchmark_group("triangle/latest_diagonal");
    for n in [8_usize, 32, 128, 512] {
        let tri = right_cumulative_triangle(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &tri, |b, tri| {
            let view = tri.view();
            b.iter(|| triangle::latest_diagonal(black_box(&view)));
        });
    }
    group.finish();
}

fn bench_cumulative_to_incremental(c: &mut Criterion) {
    let mut group = c.benchmark_group("triangle/cumulative_to_incremental");
    for n in [8_usize, 32, 128, 512] {
        let tri = right_cumulative_triangle(n);
        group.throughput(Throughput::Elements(((n * (n + 1)) / 2) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &tri, |b, tri| {
            let view = tri.view();
            b.iter(|| triangle::cumulative_to_incremental(black_box(&view)));
        });
    }
    group.finish();
}

fn bench_incremental_to_cumulative(c: &mut Criterion) {
    let mut group = c.benchmark_group("triangle/incremental_to_cumulative");
    for n in [8_usize, 32, 128, 512] {
        let tri = right_cumulative_triangle(n);
        let incremental = triangle::cumulative_to_incremental(&tri.view());
        let incr = Triangle::new(n, n, incremental);
        group.throughput(Throughput::Elements(((n * (n + 1)) / 2) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &incr, |b, incr| {
            let view = incr.view();
            b.iter(|| triangle::incremental_to_cumulative(black_box(&view)));
        });
    }
    group.finish();
}

fn bench_stochastic_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("stochastic/kernels");
    let tri10 = right_cumulative_triangle(10);
    let view10 = tri10.view();

    group.bench_function("mack_10x10", |b| {
        b.iter(|| mack::analyze(black_box(&view10), 1.0).expect("mack"));
    });

    group.bench_function("odp_fit_10x10", |b| {
        b.iter(|| fit_odp(black_box(&view10), 1.0, MonotonicityPolicy::Strict).expect("odp fit"));
    });

    let opts = OdpBootstrapOptions::shapland(200, 42069);
    group.bench_function("odp_bootstrap_10x10_200iter", |b| {
        b.iter(|| bootstrap_odp(black_box(&view10), 1.0, opts).expect("odp boot"));
    });

    let x: Vec<f64> = (0..1000).map(|i| (f64::from(i) + 1.0) * 1.13).collect();
    let y: Vec<f64> = (0..1000)
        .map(|i| f64::from((i * 7 + 3) % 1000) + 1.0)
        .collect();
    let z: Vec<f64> = (0..1000)
        .map(|i| f64::from((i * 13 + 5) % 1000) + 1.0)
        .collect();
    let corr = [1.0, 0.5, 0.2, 0.5, 1.0, 0.3, 0.2, 0.3, 1.0];
    group.bench_function("iman_conover_1000x3", |b| {
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

    group.finish();
}

criterion_group!(
    benches,
    bench_latest_diagonal,
    bench_cumulative_to_incremental,
    bench_incremental_to_cumulative,
    bench_stochastic_kernels
);
criterion_main!(benches);
