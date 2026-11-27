<p align="center">
  <img src="assets/banner.png" alt="reserving-kernels — Actuarial computing in Rust" width="100%" />
</p>

# reserving-kernels

**Deterministic, zero-unsafe Rust numeric kernels for P&C actuarial loss reserving, 1-year Solvency II risk, GLMs, compound aggregate loss convolution, excess layer pricing, and coherent capital allocation.**

`Rust 1.85+` · `MIT OR Apache-2.0` · `#![forbid(unsafe_code)]` · `serde`/`serde_json` only · `wasm32` ready

Every statistical method is derived from primary peer-reviewed actuarial literature and numerically verified against published benchmark triangles and closed-form identities in [`tests/source_benchmarks.rs`](tests/source_benchmarks.rs). See [`docs/SOURCES.md`](docs/SOURCES.md) for the complete bibliography and equation traceability matrix.

## Why `reserving-kernels`

- **Pure-Rust, zero heavy dependencies** — `#![forbid(unsafe_code)]` by default, no BLAS/LAPACK/FFTW/ndarray required; Householder QR, Cholesky, IRLS, and tilted radix-2 Cooley–Tukey FFT are built in and compile cleanly to `wasm32-unknown-unknown` (`--features wasm`).
- **Branch-light row-major `f64` triangles** — observed cumulative or incremental values are finite `f64`, unobserved lower-right cells use `NaN` sentinels, and ingress validation (`validate`) enforces explicit `MonotonicityPolicy` (`Strict` vs. `AllowNegativeIncrements` for incurred/reinsurance triangles).
- **Deterministic & reproducible** — fixed-seed SplitMix64 + Xoshiro256** PRNG streams with Lemire bounded indexing, Marsaglia polar normals, Marsaglia–Tsang Gamma, and Knuth/Hörmann PTRD Poisson samplers across ODP bootstrap, synchronized multi-line bootstrap, Iman–Conover rank reordering, and Monte Carlo layering.

## Quick Start

Add to `Cargo.toml`:

```toml
[dependencies]
reserving-kernels = { git = "https://github.com/SinisterRob/reserving-kernels" }
```

### Example: Mack Ultimate SE, 1-Year Solvency II CDR, and Panjer/FFT Convolution

```rust
use reserving_kernels::prelude::*;

// Construct directly from upper-triangle rows (no manual NaN padding required)
let triangle = Triangle::try_from_upper_rows(&[
    &[100.0, 160.0, 195.0, 205.0],
    &[120.0, 190.0, 232.0],
    &[135.0, 218.0],
    &[150.0],
])?;

// Mack (1993) ultimate chain-ladder reserve & standard error
let m = triangle.mack(1.0)?;

// Merz & Wuthrich (2008) 1-year Claims Development Result (Solvency II horizon)
let cdr = triangle.merz_wuthrich()?;
assert!(cdr.total_cdr_standard_error <= m.total_standard_error);

// Panjer (1981) recursion vs. Robertson (1992) exponentially tilted radix-2 FFT
let freq = FrequencyDistribution::Poisson { lambda: 3.0 };
let severity_pmf = [0.0, 0.5, 0.3, 0.2];
let panjer = panjer_recursion(freq, &severity_pmf, 100.0, 128)?;
let fft = fft_aggregate_loss(freq, &severity_pmf, 100.0, 128, None)?;
assert!((panjer.mean - fft.mean).abs() < 1e-8);
# Ok::<(), ValidationError>(())
```

## Implemented Modules

### 1. Loss Development, Point Estimators & Operational Adjustments

| Module | Literature / Method | Capabilities |
| --- | --- | --- |
| `ldf` | Link-ratio selectors | Volume-weighted, simple, geometric, median, ex-hi-lo trimmed, last-N, cell-weighted/masked LDFs, CDFs, percent reported |
| `tail` | Bondy (1966), Sherman (1984), Clark (2003), CAS TFWP (2013), Mack (1999) | Bondy, Generalized Bondy, Exponential, Inverse Power, Weibull, Clark growth-curve tail, Mack tail variance |
| `chain_ladder` | Classical Chain Ladder | Unchecked hot-path kernels and checked `ChainLadderEngine` |
| `bf` | Bornhuetter & Ferguson (1972) | Reserve indications `R = (1 - 1/CDF) · E` with scalar, vector, or `apriori` inputs |
| `benktander` | Benktander (1976), Hovinen (1981), Mack (2000) | Closed-form m-iteration credibility recursion and MSE-optimal credibility weights |
| `cape_cod` | Stanard (1985), Bühlmann (1983) | Data-implied ELR via used-up premium weighting |
| `generalized_cape_cod` | Gluck (1997), Friedland Ch. 10 | Decay-weighted Cape Cod (`w_{i,k} = delta^|i-k|`), reducing to Cape Cod (`delta = 1`) and CL (`delta -> 0+`) |
| `case_outstanding` | Friedland (2010, Ch. 11) | Paid-to-case and case-to-case development projections from aligned paid and case-reserve triangles |
| `incremental_additive` | Schmidt (2006), Friedland | Exposure-weighted incremental loss ratios with optional origin trend projection |
| `berquist_sherman` | Berquist & Sherman (1977), Friedland Ch. 13 | Exponential case-reserve adequacy restatement and paid disposal-rate interpolation |
| `brosius` | Brosius (1993) | Two-pass centered OLS credibility development (`y = a + b·x`) |
| `murphy` | Murphy (1994) | Generalized weighted-regression CL (`Var ∝ C^δ`, `δ ∈ {0, 1, 2}`) with optional intercept & exact analytical MSE |
| `separation` | Taylor (1977) | Non-parametric Arithmetic & Geometric Separation of delay proportions and calendar inflation |
| `credibility_cl` | Gisler & Wüthrich (2008) | Bühlmann–Straub credibility shrinkage of LDFs toward external/collective priors with exact analytical MSEP |
| `munich_chain_ladder` | Quarg & Mack (2004 / 2008) | Bivariate paid/incurred chain ladder closing the paid-vs-incurred ultimate gap |
| `schnieper` | Schnieper (1991), Liu & Verrall (2009) | Two-triangle True IBNR vs. IBNER separation with exact additive decomposition and Mack-style MSEP |
| `double_chain_ladder` | Martínez-Miranda, Nielsen & Verrall (2012) | Paid + reported-count deconvolution into reporting delay, payment delay, severity inflation, and RBNS/IBNR reserves |
| `datasets` | Published benchmark triangles | Zero-I/O static constructors for `RAA`, `GenIns`, `Quarg & Mack (2004)`, and `Friedland US Industry Auto` |
| `ragged`, `triangle_algebra` | Low-level flat kernels | Rectangular/ragged observed-count link selectors and allocation-free slice transforms |

### 2. Stochastic Reserving, 1-Year Solvency & GLMs

| Module | Literature / Method | Capabilities |
| --- | --- | --- |
| `mack` | Mack (1993, 1999), Murphy (1994) | Distribution-free origin & aggregate CL SEs, generalized `alpha` weights (`0, 1, 2`), cell weight masks, and stochastic tail (`MackTailSpec`) |
| `multivariate_mack` | Braun (2004), Merz & Wüthrich (2008) | Distribution-free analytical MSEP across `N` correlated run-off triangles with cross-line covariance & diversification |
| `mack_bf` | Mack (2008) | Analytical BF MSEP decomposing process, prior `s(E_i)`, pattern `s(z_k)`, interaction, and correlated-prior covariance |
| `merz_wuthrich` | Merz & Wüthrich (2008) | 1-year Solvency II CDR analytical MSEP with cross-origin covariance and exact multi-year run-off decomposition |
| `multivariate_cl` | Braun (2004), Pröhl & Schmidt (2005), Zhang (2010) | Joint $K$-line Diagonal MCL and General GMCL development with full $K \times K$ cross-line MSEP covariance |
| `paid_incurred_chain` | Merz & Wüthrich (2010) | Bayesian log-normal joint paid/incurred model with closed-form conjugate posterior mean and analytical MSEP |
| `hertig` | Hertig (1985) | Exact closed-form log-normal link-ratio model yielding analytical mean, variance, skewness, and quantiles |
| `kalman_reserving` | De Jong & Zehnwirth (1983), Verrall (1989) | Recursive Joseph-form Kalman filter over calendar diagonals tracking dynamic inflation shifts |
| `micro_reserving` | Antonio & Plat (2014) | Individual-claim reserving: truncation-corrected NHPP, reporting delay, and multi-state RBNS + IBNR simulation |
| `clark_mle` | Clark (2003) | Loglogistic & Weibull growth-curve ODP MLE (LDF & Cape Cod) with inverse Fisher information Cramér–Rao SEs |
| `ptf_regression` | Barnett & Zehnwirth (2000) | Probabilistic Trend Family log-linear regression isolating accident, development, and calendar inflation shocks via QR |
| `glm_reserving` | Taylor & McGuire (2016, CAS Mono. 3) | Householder QR IRLS engine for ODP, Gamma, Tweedie (`p` in `[1, 2]`), Log-normal, and profile EQL Tweedie power selection |
| `bootstrap_odp` | England & Verrall (1999, 2002), Shapland (2016, CAS Mono. 4) | Exact ODP GLM fit, hat-matrix leverages `h_ii`, DF vs. leverage residual scaling, Gamma & scaled-ODP process variance, and 1-year CDR re-reserving bootstrap |

### 3. Dependence, Aggregate Loss Convolution & Capital Allocation

| Module | Literature / Method | Capabilities |
| --- | --- | --- |
| `dependence` | Iman & Conover (1982), Mildenhall (2006), Kirschner et al. (2002) | Marginal-preserving rank reordering (Gaussian/Spearman/Kendall) and synchronized multi-line ODP bootstrap |
| `reserve_distribution` | CAS Monograph No. 15 (§3.2–§3.4) | Empirical moments/quantiles, least-quantile VaR (Alg. 3.4.2), exact subadditive coherent TVaR (Alg. 3.4.3), and RVaR |
| `aggregate_loss` | Panjer (1981), Robertson (1992), KPW Ch. 9 | Exact `(a, b, 0)` Panjer recursion (Poisson/Binomial/NegBin), exponentially tilted radix-2 FFT, rounding & mean-preserving LEV discretization |
| `capital_allocation` | Tasche (2008), Myers & Read (2001), CAS Mono. 15 §5 | Tie-symmetric Euler allocation for TVaR, RVaR, and SD principles plus marginal default-put surplus allocation |

### 4. Excess Layer Pricing, Severity & Diagnostics

| Module | Literature / Method | Capabilities |
| --- | --- | --- |
| `severity` | KPW App. A, Miccolis (1977), Lee (1988), Keatinge (1999) | Closed-form LEV `E[X ∧ L]`, survival/PDF, layer losses, excess ratios, and ILFs for Lognormal, Pareto II, Single-Param Pareto, Exponential, Mixed Exponential, and GPD |
| `excess_layers` | Parametric & Monte Carlo layering | Lognormal LEV, attachment/exhaustion exceedance probabilities, ILFs, and cached sorted antithetic Marsaglia polar $O(\log N)$ simulation |
| `excess_development` | Pinto & Gogol (1987), Siewert (1996) | Excess-of-`L`, limited-to-`L`, and bounded-layer CDFs/LDFs from ground-up CDFs & age-dependent excess ratios with exact conservation |
| `diagnostics` | Mack (1994 App. G & H), A-vs-E, LOO | Adjacent-factor Spearman test, calendar-year binomial test, leave-one-out fragility, Cape Cod leverage, paid/incurred reconciliation |
| `selection`, `reserve_rollforward`, `exhibit` | Backtesting & reporting | Holdout LDF backtesting, valuation roll-forward decomposition, and serializable JSON exhibit DTOs |
| `large_loss`, `trend`, `onlevel`, `count_development`, `apriori`, `aggregate` | Preprocessing utilities | Large-loss capping/excess splits, exponential trending, parallelogram on-leveling, count development, a priori builders |

## Performance & Benchmarking

Steady-state, single-threaded Criterion timings (`cargo bench --bench kernels`, 600 ms warmup, 2 s measurement window, 2% noise threshold) on identical triangles (`RAA` 10×10, `GenIns` 10×10, synthetic 40×40) comparing **`reserving-kernels`** (Rust `--release`) against **`mages/ChainLadder` 0.2.21** (R) and **`casact/chainladder-python` 0.10.1** (Python) on an Intel Core i9-14900K:

<p align="center">
  <img src="assets/benchmarks.svg" alt="reserving-kernels vs R ChainLadder and chainladder-python benchmark comparison" width="100%" />
</p>

| Benchmark | `reserving-kernels` (Rust) | `mages/ChainLadder` (R) | `chainladder-python` (Py) | Speedup vs. R | Speedup vs. Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| Volume-Weighted LDFs (40×40) | **0.28 µs** | 840 µs | 5,710 µs | 3,040× | 20,680× |
| Mack Chain Ladder (`RAA` 10×10) | **0.29 µs** | 5,840 µs | 25,446 µs | 20,280× | 88,350× |
| Mack Chain Ladder (40×40) | **4.09 µs** | 24,650 µs | 30,077 µs | 6,025× | 7,350× |
| Merz–Wüthrich 1-Yr CDR (`RAA`) | **0.41 µs** | 9,420 µs | — | 23,050× | — |
| Clark Loglogistic MLE (`GenIns`) | **11.9 µs** | 8,300 µs | 12,026 µs | 695× | 1,007× |
| ODP GLM Reserve (`GenIns`) | **42.6 µs** | 5,300 µs | 219,562 µs | 124× | 5,155× |
| ODP Bootstrap (`GenIns`, 1k sims) | **0.80 ms** | 216.5 ms | 76.8 ms | 270× | 96× |
| ODP Bootstrap (`GenIns`, 10k sims) | **8.20 ms** | 2,223 ms | 416.2 ms | 271× | 51× |

> *Context:* `reserving-kernels` operates on validated flat `f64` slices with reusable scratch workspaces (`~0.80 µs` per ODP bootstrap draw, `9.8 µs` for 10k-sample coherent TVaR, `35 ns` for cached 65k-draw excess-layer MC), whereas R `ChainLadder` and `chainladder-python` include DataFrame/S3–S4 metadata layers. All point estimates and Mack/Merz–Wüthrich SEs were verified for numerical parity prior to timing.

### Benchmarking & Profiling Methodology

- **Suites:** `benches/kernels.rs` covers 20 canonical reserving, MLE/GLM, bootstrap, dependence, distribution, and convolution workloads (`RAA`, `GenIns`, 40×40); `benches/triangle_algebra.rs` sweeps flat triangle scaling (`8..512`).
- **Regression baselines:** Use Criterion's statistical diffing (`--save-baseline` / `--baseline`) to verify changes at `p < 0.05` with confidence intervals and throughput (`elem/s`).
- **CPU profiling:** Pair `cargo bench --bench kernels -- --profile-time 5` with `samply record` or `cargo flamegraph` to inspect hot loops under release codegen.

## Verification & Literature Traceability

All kernels are continuously tested against published datasets and exact mathematical identities (208 automated unit, integration, doctest, and source-benchmark tests):

- **Published Triangles:** Mack (1993) `RAA` 10×10, Taylor & Ashe (1983) / England & Verrall (1999, 2002) `GenIns` 10×10, Friedland (2010) US Industry Auto 10×10, and Quarg & Mack (2004) 7×7.
- **Exact Identities:** Merz–Wüthrich multi-year CDR run-off summing identically to Mack ultimate MSEP; IRLS cross-classified ODP matching exact `fit_odp` leverages and dispersion; Panjer vs. tilted radix-2 FFT L∞ agreement; tie-invariant Euler TVaR and Myers–Read exact full-allocation conservation.
- See [`docs/SOURCES.md`](docs/SOURCES.md) for full citations and [`tests/source_benchmarks.rs`](tests/source_benchmarks.rs) for the executable benchmark suite.

## Building & Testing

```bash
# Run full unit, integration, doctest, and source-benchmark suite
cargo test --all-targets --all-features
cargo test --doc

# Run Criterion benchmarks (save / compare statistical baselines)
cargo bench --bench kernels -- --save-baseline main
cargo bench --bench kernels -- --baseline main

# Profile a specific kernel under release optimization
samply record cargo bench --bench kernels -- glm_odp_genins --profile-time 5

# Build WebAssembly target
cargo build --release --target wasm32-unknown-unknown --features wasm
```

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
