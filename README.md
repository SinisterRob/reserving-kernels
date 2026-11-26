# reserving-kernels

Deterministic, zero-unsafe Rust numeric kernels for property & casualty actuarial loss reserving, diagnostics, stochastic bootstrapping, portfolio dependence, and excess layer pricing.

Every statistical method is grounded in primary actuarial literature and verified against published benchmark triangles and closed-form identities in [`tests/source_benchmarks.rs`](tests/source_benchmarks.rs). See [`docs/SOURCES.md`](docs/SOURCES.md) for the complete bibliography and equation traceability matrix.

## Overview

`reserving-kernels` provides a branch-light, row-major `f64` compute substrate for loss-development triangles, portfolio dependence, and severity layering:

- **Pure Rust hot path** — `#![forbid(unsafe_code)]` by default, zero heavy linear-algebra dependencies (`serde` and `serde_json` only), optional `wasm32-unknown-unknown` target (`--features wasm`).
- **Row-major `f64` triangle representation** — observed cumulative or incremental cells are finite `f64`; unobserved lower-right cells use `NaN` sentinels with upfront boundary validation (`validate`) and explicit `MonotonicityPolicy` (`Strict` vs. `AllowNegativeIncrements` for incurred/reinsurance triangles).
- **Deterministic stochastic engines** — explicit fixed-seed SplitMix64 + Xoshiro256** PRNG streams with exact Marsaglia–Tsang Gamma and Knuth/Hörmann PTRD Poisson samplers for ODP residual bootstrapping, synchronized multi-line bootstrapping, Iman–Conover rank reordering, and lognormal Monte Carlo layering.

## Implemented Methods

### 1. Loss Development, Tail Extrapolation & Point Estimators
- **`ldf`** — Age-to-age link-ratio selectors: volume-weighted all-year, simple average, median, ex-hi-lo trimmed, last-$N$ volume-weighted, and last-$N$ simple average, plus CDF and percent-reported construction.
- **`tail`** — Parametric and heuristic tail-factor fitting: Bondy (1966), Generalized Bondy (exact profile least squares), Exponential decay, Sherman (1984) Inverse Power Curve, Weibull, and Mack (1999) tail variance extrapolation.
- **`chain_ladder`** — Classical chain-ladder projection and checked `ChainLadderEngine`.
- **`bf`** — Bornhuetter–Ferguson (1972) reserve indications (`R = (1 - 1 / CDF) * Expected Ultimate`).
- **`benktander`** — Benktander (1976) / Hovinen (1981) / Mack (2000) iterated BF credibility projection and MSE-optimal credibility weights.
- **`cape_cod`** — Stanard (1985) / Bühlmann (1983) Cape Cod generalized expected loss ratio estimation with used-up premium weighting.
- **`generalized_cape_cod`** — Gluck (1997) decay-weighted Cape Cod (`w_{i,k} = delta^{|i-k|}`) interpolating smoothly between standard Cape Cod (`delta = 1`) and Chain Ladder (`delta -> 0+`).
- **`berquist_sherman`** — Berquist & Sherman (1977) operational adjustments: exponential case-reserve adequacy restatement and paid claim disposal-rate interpolation.
- **`brosius`** — Brosius (1993) two-pass centered least-squares credibility development (`y = a + b x`) with link credibility weights.
- **`munich_chain_ladder`** — Quarg & Mack (2004 / 2008) bivariate paid/incurred chain ladder closing the paid-vs-incurred ultimate gap.

### 2. Diagnostics, Holdout Backtesting & Reconciliation
- **`diagnostics`** — Leave-one-out link-ratio fragility, actual-vs-expected standardized residuals, Mack (1994) Appendix G Spearman adjacent-factor correlation test and Appendix H calendar-year binomial test, Cape Cod ELR stability, paid-to-incurred reconciliation, and case-reserve adequacy ratios.
- **`selection`** — Latest-diagonal and multi-diagonal holdout backtesting with composite stability/smoothness/accuracy scoring across LDF selector families.
- **`reserve_rollforward`** — Valuation-to-valuation reserve movement decomposition (actual vs. expected emergence and true ultimate change).
- **`exhibit`** — Serializable JSON exhibit DTOs for downstream tables, heatmaps, and method-comparison charts.

### 3. Stochastic Reserving, 1-Year Solvency & GLMs
- **`mack`** — Mack (1993) distribution-free chain-ladder standard errors (verified against published RAA and GenIns totals) plus Mack (1999) recursive standard errors with stochastic tail-factor variance (`MackTailSpec`).
- **`mack_bf`** — Mack (2008) distribution-free analytical MSEP for Bornhuetter–Ferguson decomposing process, prior ultimate, payout-pattern, interaction, and cross-origin covariance risk.
- **`merz_wuthrich`** — Merz & Wüthrich (2008) 1-year Claims Development Result (`CDR`) analytical MSEP for Solvency II horizons plus exact multi-year run-off decomposition summing to Mack's ultimate MSEP.
- **`paid_incurred_chain`** — Merz & Wüthrich (2010) Bayesian log-normal Paid-Incurred Chain (PIC) model with closed-form conjugate posterior mean and analytical MSEP.
- **`clark_mle`** — Clark (2003) Loglogistic and Weibull growth-curve ODP MLE (LDF & Cape Cod) with inverse Fisher information Cramér–Rao parameter standard errors.
- **`ptf_regression`** — Barnett & Zehnwirth (2000) Probabilistic Trend Family (PTF) log-linear regression isolating accident, development, and calendar-year inflation shocks via Householder QR weighted OLS.
- **`glm_reserving`** — Taylor & McGuire (2016, CAS Monograph No. 3) Householder QR IRLS engine supporting ODP, Gamma, Tweedie (`p in [1, 2]`), and Log-normal families with Cross-Classified, Hoerl curve, and calendar-year spline designs.
- **`bootstrap_odp`** — England & Verrall (1999, 2002) and Shapland (2016, CAS Monograph No. 4) Over-Dispersed Poisson GLM fit (`fit_odp`) and residual bootstrap (`bootstrap_odp`): exact hat-matrix leverages (`h_ii`), mutually exclusive degrees-of-freedom vs. leverage scaling, analytic ODP prediction error, Gamma and scaled-ODP process variance, and explicit failed-iteration accounting.

### 4. Dependence, Aggregate Loss Convolution & Capital Allocation
- **`dependence`** — Mildenhall (2006) / Iman–Conover (1982) whole-scenario batch rank reordering (preserving exact empirical marginals with Gaussian/Spearman/Kendall targets) and Kirschner, Kerley & Isaacs (2002) synchronized multi-line ODP bootstrapping.
- **`reserve_distribution`** — Single-pass empirical moments, quantiles, exceedance probabilities, and CAS Monograph No. 15 discrete risk measures: least-quantile VaR (Algorithm 3.4.2), exact subadditive coherent TVaR (Algorithm 3.4.3), and Range VaR (RVaR).
- **`aggregate_loss`** — Panjer (1981) `(a, b, 0)` exact recursion and Robertson (1992) pure-Rust radix-2 Cooley–Tukey FFT compound distribution inversion with exponential tilting, plus rounding and mean-preserving LEV discretization.
- **`capital_allocation`** — Tasche (2008) tie-symmetric Euler allocation for coherent TVaR, RVaR, and Standard Deviation principles plus Myers & Read (2001) marginal default-put surplus allocation.

### 5. Excess / Layer Pricing & Severity
- **`severity`** — Claim-count severity helpers plus exact closed-form Limited Expected Values `E[X ∧ L]`, survival/PDF, layer expected losses, excess ratios, and ILFs (Klugman–Panjer–Willmot; Miccolis 1977; Lee 1988; Keatinge 1999) for Lognormal, Pareto Type II (Lomax), Single-Parameter Pareto, Exponential, Mixed Exponential, and Generalized Pareto distributions.
- **`excess_layers`** — Closed-form lognormal Limited Expected Value `E[X ∧ L]`, attachment and exhaustion exceedance probabilities, Increased Limit Factors (ILFs), and synchronized SplitMix64 + Box–Muller Monte Carlo layer simulation.
- **`large_loss`**, **`trend`**, **`onlevel`**, **`count_development`**, **`apriori`**, **`aggregate`** — Triangle capping/excess partitioning, exponential trend transforms, parallelogram on-level premium factors, claim-count development, shared a priori builders, and portfolio reducers.

## Quick Start

```bash
cargo test --all-targets --all-features
cargo run --example export_exhibit
cargo bench
```

### WebAssembly Build

```bash
cargo build --release --target wasm32-unknown-unknown --features wasm
```

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
