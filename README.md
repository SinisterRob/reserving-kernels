<p align="center">
  <img src="https://raw.githubusercontent.com/SinisterRob/reserving-kernels/main/assets/banner.png" alt="reserving-kernels — Actuarial computing in Rust" width="100%" />
</p>

<p align="center">
  <strong>Pure Rust numeric kernels for P&amp;C actuarial reserving and reserve-risk capital modeling.</strong>
</p>

<p align="center">
  <a href="https://crates.io/crates/reserving-kernels"><img src="https://img.shields.io/crates/v/reserving-kernels?style=flat" alt="Crates.io" /></a>
  <a href="https://docs.rs/reserving-kernels"><img src="https://img.shields.io/docsrs/reserving-kernels?style=flat" alt="docs.rs" /></a>
  <a href="https://github.com/SinisterRob/reserving-kernels/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/SinisterRob/reserving-kernels/ci.yml?branch=main&label=CI&style=flat" alt="CI" /></a>
  <a href="https://github.com/SinisterRob/reserving-kernels/blob/main/Cargo.toml"><img src="https://img.shields.io/badge/Rust-1.97%2B-f97316?style=flat" alt="Rust 1.97+" /></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT%20%2F%20Apache--2.0-30363d?style=flat" alt="License: MIT or Apache-2.0" /></a>
  <a href="https://github.com/SinisterRob/reserving-kernels/blob/main/src/lib.rs"><img src="https://img.shields.io/badge/unsafe-forbidden-30363d?style=flat" alt="No unsafe code" /></a>
  <a href="https://github.com/SinisterRob/reserving-kernels/blob/main/Cargo.toml"><img src="https://img.shields.io/badge/deps-serde%20only-30363d?style=flat" alt="Dependencies: serde only" /></a>
  <a href="#building-and-testing"><img src="https://img.shields.io/badge/target-wasm32-30363d?style=flat" alt="WebAssembly target" /></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &nbsp;·&nbsp;
  <a href="https://github.com/SinisterRob/reserving-kernels/blob/main/docs/examples/mack-chain-ladder-raa.md">Worked Example</a> &nbsp;·&nbsp;
  <a href="#implemented-modules">Modules</a> &nbsp;·&nbsp;
  <a href="#performance">Performance</a> &nbsp;·&nbsp;
  <a href="#verification">Verification</a> &nbsp;·&nbsp;
  <a href="#building-and-testing">Build &amp; Test</a> &nbsp;·&nbsp;
  <a href="https://github.com/SinisterRob/reserving-kernels/blob/main/docs/SOURCES.md">Literature</a>
</p>

---

`reserving-kernels` covers retrospective loss development, stochastic reserving, and the **reserve risk** module of capital modeling (1-year CDR, multi-line dependence, and capital allocation). Every method is implemented from peer-reviewed actuarial literature ([`docs/SOURCES.md`](https://github.com/SinisterRob/reserving-kernels/blob/main/docs/SOURCES.md)) and verified against published benchmark triangles ([`tests/source_benchmarks.rs`](https://github.com/SinisterRob/reserving-kernels/blob/main/tests/source_benchmarks.rs)).

## Why `reserving-kernels`?

- **Zero heavy dependencies:** Pure Rust with no BLAS, LAPACK, FFTW, or `ndarray`. Linear algebra, IRLS, and FFT routines are built in and compile cleanly to `wasm32` (`--features wasm`).
- **Validated flat triangles:** Row-major `f64` grids using `NaN` for unobserved cells, with explicit monotonicity validation and support for negative development in incurred or reinsurance triangles.
- **Deterministic by design:** Fixed-seed PRNG streams and custom samplers keep ODP bootstraps, multi-line correlations, and Monte Carlo runs deterministic and reproducible on each target.

## Quick Start

Add `reserving-kernels` to your `Cargo.toml`:

```toml
[dependencies]
reserving-kernels = "0.3"
```

Basic triangle reserving and aggregate loss convolution:

```rust
use reserving_kernels::prelude::*;

fn main() -> Result<(), ValidationError> {
    // Build from upper-triangle rows (no manual NaN padding needed)
    let triangle = Triangle::try_from_upper_rows(&[
        &[100.0, 160.0, 195.0, 205.0],
        &[120.0, 190.0, 232.0],
        &[135.0, 218.0],
        &[150.0],
    ])?;

    // Mack ultimate reserve & standard error
    let mack = triangle.mack(1.0)?;
    println!(
        "Mack reserve: {:.2} (SE: {:.2}, CV: {:.1}%)",
        mack.total_reserve,
        mack.total_standard_error,
        mack.total_cv * 100.0,
    );

    // Merz–Wüthrich 1-year Claims Development Result (Solvency II)
    let cdr = triangle.merz_wuthrich()?;
    println!("1-yr CDR SE:  {:.2}", cdr.total_cdr_standard_error);
    assert!(cdr.total_cdr_standard_error <= mack.total_standard_error);

    // Aggregate loss: Panjer recursion vs. FFT
    let freq = FrequencyDistribution::Poisson { lambda: 3.0 };
    let severity_pmf = [0.0, 0.5, 0.3, 0.2];
    let panjer = panjer_recursion(freq, &severity_pmf, 100.0, 128)?;
    let fft = fft_aggregate_loss(freq, &severity_pmf, 100.0, 128, None)?;
    assert!((panjer.mean - fft.mean).abs() < 1e-8);

    Ok(())
}
```

## Implemented Modules

`reserving-kernels` is organized into four areas covering 54 public modules (plus optional `wasm_exports` behind `--features wasm`).

| Area | Highlights |
| --- | --- |
| **1. Loss Development & Point Estimators** | Chain Ladder, Bornhuetter–Ferguson, Benktander, Cape Cod (standard & generalized), Berquist–Sherman, Munich & Double Chain Ladder, Separation, sub-annual grids |
| **2. Stochastic Reserving, 1-Year Risk & GLMs** | Mack (uni/multivariate & BF), Merz–Wüthrich 1-year CDR, ODP/Gamma/Tweedie GLMs, ODP Bootstrap, Paid–Incurred Chain, Kalman & individual-claim reserving |
| **3. Dependence, Aggregate Loss & Capital Allocation** | Iman–Conover rank correlation, synchronized multi-line bootstrap, Panjer & FFT aggregate loss convolution, VaR/TVaR/RVaR, Euler & Myers–Read allocation |
| **4. Excess Layering, Severity & Diagnostics** | Parametric severity & LEV/excess ratios, excess-layer development, Mack diagnostic tests, backtesting, roll-forwards, and preprocessing utilities |

<details>
<summary><strong>Full module directory (54 public modules)</strong></summary>

### 1. Loss Development & Point Estimators

| Module | Capabilities |
| --- | --- |
| `types`, `triangle`, `validate` | Flat row-major `Triangle` / `TriangleView`, monotonicity policies, and validation errors |
| `ldf` | Volume-weighted, simple, geometric, median, trimmed, last-N, and masked LDFs/CDFs |
| `tail` | Bondy, Generalized Bondy, Exponential, Inverse Power, Weibull, Clark growth-curve, and Mack tail variance |
| `chain_ladder` | Unchecked hot-path kernels and checked `ChainLadderEngine` |
| `bf` | Bornhuetter–Ferguson reserve indications with scalar, vector, or a priori inputs |
| `benktander` | Closed-form m-iteration credibility recursion and MSE-optimal weights |
| `cape_cod`, `generalized_cape_cod` | Standard and decay-weighted Cape Cod ELR estimation |
| `case_outstanding` | Paid-to-case and case-to-case development from aligned triangles |
| `incremental_additive` | Exposure-weighted incremental loss ratios with optional trend |
| `berquist_sherman` | Case-reserve adequacy restatement and paid disposal-rate interpolation |
| `brosius`, `murphy` | Centered OLS credibility and generalized weighted-regression CL with exact MSE |
| `separation` | Arithmetic and geometric separation of delay patterns and calendar inflation |
| `credibility_cl` | Bühlmann–Straub shrinkage of LDFs toward collective priors with analytical MSEP |
| `munich_chain_ladder` | Bivariate paid/incurred CL closing the paid-vs-incurred gap |
| `schnieper` | Two-triangle True IBNR vs. IBNER separation with Mack-style MSEP |
| `double_chain_ladder` | Paid + count deconvolution into reporting/payment delays, severity inflation, and RBNS/IBNR |
| `subannual` | Asymmetric grids (2:1/4:1/12:1), 7 CDF/LDF interpolators, Bloom stubs, asymmetric CL/BF/Cape Cod/Clark/ODP, and parity checks |
| `datasets` | Zero-I/O benchmark triangles (RAA, GenIns, Quarg & Mack, Friedland Auto) |
| `ragged`, `triangle_algebra` | Low-level ragged link selectors and allocation-free slice transforms |

### 2. Stochastic Reserving, 1-Year Risk & GLMs

| Module | Capabilities |
| --- | --- |
| `mack`, `multivariate_mack` | Distribution-free CL standard errors (single & correlated multi-line) with stochastic tails |
| `mack_bf` | Analytical BF MSEP decomposing process, prior, pattern, and interaction variance |
| `merz_wuthrich` | 1-year Solvency II CDR analytical MSEP and multi-year run-off decomposition |
| `multivariate_cl` | Joint K-line Diagonal MCL and General GMCL with full cross-line covariance |
| `paid_incurred_chain` | Bayesian log-normal joint paid/incurred model with closed-form posterior & MSEP |
| `hertig` | Exact closed-form log-normal link-ratio moments and quantiles |
| `kalman_reserving` | Recursive Kalman filter over calendar diagonals for dynamic inflation shifts |
| `micro_reserving` | Individual-claim reserving (NHPP, reporting delay, multi-state RBNS + IBNR simulation) |
| `clark_mle` | Loglogistic & Weibull growth-curve ODP MLE with Cramér–Rao standard errors |
| `ptf_regression` | Probabilistic Trend Family log-linear regression for accident, dev, and calendar trends |
| `glm_reserving` | Built-in QR IRLS engine for ODP, Gamma, Tweedie, and Log-normal GLMs |
| `bootstrap_odp` | ODP GLM bootstrap with hat-matrix leverages, process variance, and 1-year CDR re-reserving |

### 3. Dependence, Aggregate Loss & Capital Allocation

| Module | Capabilities |
| --- | --- |
| `dependence` | Iman–Conover rank reordering and synchronized multi-line ODP bootstrap |
| `reserve_distribution` | Empirical moments, VaR, coherent TVaR, and RVaR |
| `aggregate_loss` | Panjer (a, b, 0) recursion, exponentially tilted FFT, and severity discretization |
| `capital_allocation` | Euler allocation (TVaR, RVaR, SD) and Myers–Read marginal default-put allocation |

### 4. Excess Layering, Severity & Diagnostics

| Module | Capabilities |
| --- | --- |
| `severity` | Closed-form LEV, survival/PDF, layer losses, and excess ratios (Lognormal, Pareto, Exponential, Mixed Exp, GPD) |
| `excess_layers` | Parametric layering, exceedance probabilities, and antithetic Monte Carlo simulation |
| `excess_development` | Excess-of-L, limited-to-L, and bounded-layer CDFs/LDFs with exact conservation |
| `diagnostics` | Mack Spearman & calendar-year tests, actual vs. expected, leave-one-out fragility, and paid/incurred reconciliation |
| `selection`, `reserve_rollforward`, `exhibit` | Holdout LDF backtesting, valuation roll-forwards, and JSON exhibit DTOs |
| `large_loss`, `trend`, `onlevel` | Large-loss capping, loss trending, and parallelogram rate on-leveling |
| `count_development`, `apriori`, `aggregate` | Claim-count development, a priori loss-ratio builders, and triangle aggregation |

</details>

## Performance

Single-threaded benchmarks on an Intel Core i9-14900K comparing `reserving-kernels` (`--release`) against `mages/ChainLadder` 0.2.21 (R) and `casact/chainladder-python` 0.10.1 (Python) on identical triangles:

<p align="center">
  <img src="https://raw.githubusercontent.com/SinisterRob/reserving-kernels/main/assets/benchmarks.svg" alt="reserving-kernels vs R ChainLadder and chainladder-python benchmark comparison" width="100%" />
</p>

| Benchmark | `reserving-kernels` | R `ChainLadder` | `chainladder-python` | vs. R | vs. Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| Volume-Weighted LDFs (40×40) | **0.30 µs** | 840 µs | 5,710 µs | 2,830× | 19,250× |
| Mack Chain Ladder (RAA 10×10) | **0.27 µs** | 5,840 µs | 25,446 µs | 21,620× | 94,200× |
| Mack Chain Ladder (40×40) | **2.62 µs** | 24,650 µs | 30,077 µs | 9,410× | 11,480× |
| Merz–Wüthrich 1-Yr CDR (RAA) | **0.38 µs** | 9,420 µs | — | 24,580× | — |
| Clark Loglogistic MLE (GenIns) | **11.7 µs** | 8,300 µs | 12,026 µs | 711× | 1,030× |
| ODP GLM Reserve (GenIns) | **1.20 µs** | 5,300 µs | 219.6 ms | 4,410× | 182,860× |
| ODP Bootstrap (GenIns, 1k sims) | **0.19 ms** | 216.5 ms | 76.8 ms | 1,160× | 410× |
| ODP Bootstrap (GenIns, 10k sims) | **1.90 ms** | 2,223 ms | 416.2 ms | 1,170× | 220× |

*Note: `reserving-kernels` operates directly on validated `f64` slices with reusable scratch space, whereas the R and Python libraries carry DataFrame and S3/S4 object overhead. All point estimates and Mack / Merz–Wüthrich standard errors were verified for numerical parity before timing.*

## Verification

Kernels are tested against published actuarial datasets and closed-form mathematical identities:

- **Worked example:** [Mack Chain-Ladder in Rust: A Worked Example](https://github.com/SinisterRob/reserving-kernels/blob/main/docs/examples/mack-chain-ladder-raa.md) reproduces Mack (1993) Table 2 on the RAA 10×10 triangle cell-for-cell (`cargo run --example mack_raa`).
- **Benchmark triangles:** Mack (1993) RAA, Taylor & Ashe / England & Verrall GenIns, Friedland US Industry Auto, and Quarg & Mack (2004).
- **Exact identities:** Merz–Wüthrich multi-year run-off summing to Mack ultimate MSEP; ODP GLM matching exact analytical leverages and dispersion; Panjer vs. FFT aggregate loss agreement; and full-allocation conservation for Euler TVaR and Myers–Read.

Full citations and equation mappings live in [`docs/SOURCES.md`](https://github.com/SinisterRob/reserving-kernels/blob/main/docs/SOURCES.md); parity tests live in [`tests/source_benchmarks.rs`](https://github.com/SinisterRob/reserving-kernels/blob/main/tests/source_benchmarks.rs).

## Building and Testing

```bash
# Run unit, integration, doctest, and benchmark-parity tests
cargo test --all-targets --all-features
cargo test --doc --all-features

# Run the Mack (1993) RAA worked example or JSON exhibit example
cargo run --example mack_raa
cargo run --example export_exhibit

# Run Criterion benchmarks (benches/kernels.rs & benches/triangle_algebra.rs)
cargo bench --bench kernels -- --save-baseline main
cargo bench --bench kernels -- --baseline main

# Profile a specific kernel under release optimization (requires samply)
samply record cargo bench --bench kernels -- glm_odp_genins --profile-time 5

# Build for WebAssembly
cargo build --release --target wasm32-unknown-unknown --features wasm
```

## License

Licensed under either of [MIT](https://github.com/SinisterRob/reserving-kernels/blob/main/LICENSE-MIT) or [Apache-2.0](https://github.com/SinisterRob/reserving-kernels/blob/main/LICENSE-APACHE) at your option.
