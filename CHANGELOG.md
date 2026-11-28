# Changelog

All notable changes to `reserving-kernels` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 0.5.0

### Fixed
- **CI:** The Miri job passed four positional test names to `cargo miri test`, which accepts a single `TESTNAME`, so the main CI run was red. It now passes them to libtest after `--` with `--exact`. Miri also runs the new WASM status-code regression test.
- **WASM exports:** `n > rk_n_max()` used to be silently clamped to 20, returning results for a truncated triangle. It is now rejected with `0` (counts), `NaN` (scalars), or `-1` (`rk_ldf_*`), and stale output buffers are cleared to `NaN`.
- **WASM docs:** Removed the claim that `catch_unwind` maps panics to `0`/`NaN` on `wasm32-unknown-unknown`. That target aborts on panic and traps the instance; status mapping applies only to native and Miri builds.
- **`Triangle::try_new`:** Untrusted dimensions are now checked with `checked_mul`. Before this, a `2^32 × 2^32` header with empty `data` could wrap `n_origins * n_devs` to `0`, which panicked in debug builds and slipped past the shape check in release builds.
- **Parity scripts:** `scripts/parity_r.R` and `scripts/parity_python.py` now check every parity value against reference values within explicit tolerances and exit non-zero on a mismatch. They exit `2` when the packages are missing. The R `bench_us` helper re-evaluates its expression on every repetition; the old `force()` evaluated it only once.
- **Subannual:** `interp_ivf12` was numerically identical to `interp_exponential`. It now applies PCHIP interpolation to the log 12-month factor and chains the result back to a smooth CDF. Knots are preserved, the CDF is monotone and ≥ 1, and the result differs from exponential between knots.
- **Subannual:** Grid and map accessors and `diagnose_asymmetric_calendar_effects` could panic on malformed input, and deserializing `AsymmetricAgeGrid` and `CalendarDiagonalMap` did not validate anything. Both are fixed; `n_origins * n_devs` is overflow-checked.
- **LDF engines:** `geometric_average` returned `1.0` for links with no positive pairs. It now returns `NaN`, and the checked engine returns `NotFinite`.
- **Link-ratio diagnostics:** The volume-weighted factor dropped `C_ij = 0` rows. It is now bit-identical to the LDF engine.
- **Diagnostics:** `residual_scale_by_dev` uses the unbiased Mack `n − 1` estimator.
- **Shape validation:** `validate_triangle` and the observed-count validators use `checked_mul`.
- **Brosius:** Now applies the Brosius (1993) fallbacks: negative slope uses the budget method, negative intercept uses the link-ratio method. Each link reports its fallback in a `BrosiusFallback` diagnostic.
- **Case Outstanding / Incremental Additive:** When a ratio or factor that a projection needs cannot be estimated, these now return an error. Before, a silent `0.0` dropped known case reserves.
- **Clark ALAE:** Uses the `clark_mle` age convention (`origin_offset`, default 0.5) and supports `max_age` truncation.
- **Berquist–Sherman:** Disposal-rate projections develop the adjusted latest diagonal (Friedland Ch. 13). Custom target rates must be non-decreasing.
- **Adler–Kline:** Lags with no closings carry the prior severity forward. If no closings exist at all, it returns an error.
- **Aggregate loss TVaR/VaR:** `AggregateLossDistribution::tvar` no longer renormalises truncated grids. It returns an error when missing or aliased mass exceeds 1% of the tail. `var(1)` and `tvar(1)` return the top of the support, and TVaR is monotone in alpha.
- **Clayton copula:** The sampler is correct for large θ (log-space Gamma frailty). The t-copula draws its chi-square in log space.
- **Severity layers:** Layer and excess-ratio values use direct closed forms, so they no longer lose precision through cancellation at high attachments.
- **Jacobi eigensolver:** Returns an error instead of silently returning an unconverged result.
- **Mack:** `analyze_with_options` now extrapolates σ² for every link with fewer than 2 active pairs, using the Mack min-rule. Before, it extrapolated only an exact-zero last link, which understated the SE on masked or 3×3 triangles. The `alpha` doc had 0 and 2 swapped: 0 is the simple average and 2 is OLS through the origin.
- **Mack-BF:** The pattern SE uses the Mack (2008) min-rule over the lower and upper sums. The chain-ladder-implied pattern SE is now consistent with the z actually used (delta method).
- **Meyers LCL/CCL:** Importance sampling collapsed on real 10×10 triangles: on GenIns CCL the effective sample size was 1, and `converged` was still `true`. The sampler now integrates (α, β) exactly and runs a tempered adaptive t-mixture proposal. Effective sample size is now about 1,800–6,000 at M = 4096. Below `min_ess_fraction` the fit returns an error, or with `fallback_to_laplace` it returns the Laplace result flagged `laplace_fallback = true`. Bivariate CCL runs Newton polish and reports `converged`.
- **Log-normal GLM:** The mean is bias-corrected as `exp(η̂ + (σ̂² − v)/2)` (Verrall 1991); before, the reserve was overstated by 4–17% on small triangles. The diagonal and off-diagonal MSEP terms are now consistent, and the "exact MSEP" doc claim is removed.
- **ODP analytic prediction error:** Now includes the tail factor. Before, with a tail of 1.10 on GenIns, the reserve rose 28% while the prediction error was unchanged.
- **Tweedie:** `power >= 2` rejects zero increments (it previously reported a meaningless Gamma deviance). `select_tweedie_power` skips fits that did not converge. Its doc now says it uses EQL.
- **ODP deviance:** With negative increments, the reported deviance now equals the objective being optimised.
- **`bootstrap()`:** Now a projection of `bootstrap_odp`, so draws are identical for the same seed. It reports `failed_iterations`, and zero iterations returns `InvalidIterations`.
- **Braun (2004):** One shared cross-covariance estimator for `multivariate_mack` and the diagonal MCL; the two previously used different denominators.

### Changed (Breaking)
- **`Triangle::new` is now `pub(crate)`.** It skipped validation and relied on a `debug_assert`, so malformed buffers could panic in release builds inside `latest_diagonal` and `diagnostics`. External callers must use `Triangle::try_new`, `try_from_rows`, `try_from_upper_rows`, `try_from_incremental_rows`, or the `triangle!` macro.
- **WASM `rk_ldf_*` exports now return `i32`:** the number of LDFs written (`0` is a valid result for `n = 1`), or `RK_STATUS_ERR` (`-1`) on invalid input. They returned `u32` before, so an error and `n = 1` both returned `0`.
- **Error type:** `validate::ValidationError` is replaced by `error::ReservingError`, re-exported as `reserving_kernels::{ReservingError, Result}`. There is no alias (refactor 4cde5c4).
- **Selection:** `LdfSelectionMethod::Selected` now returns `InvalidParameter` from computed selectors instead of silently using volume-weighted factors. Use `DevelopmentFactorEngine::from_selected_ldfs` instead.
- **Selection:** New `LdfSelectionMethod::Geometric` variant.
- **L0 kernels:** `project_ultimates`, `reserves`, `latest_observed_by_origin*`, the BF kernels, the `triangle_algebra` helpers, `volume_weighted_ldf*` and `volume_weighted_link_diagnostic*` return `Result` instead of panicking. `try_volume_weighted_ldfs`, `try_cdfs_from_ldfs` and related functions are added.
- **Diagnostics:** `diagnose_link` and `diagnose_all` return `Result`. `CalendarEffectDiagnostics::p_value_like_score` is replaced by `chi_square_statistic`, `degrees_of_freedom` and `p_value`.
- **Subannual:** `AsymmetricAgeGrid::cell_age` and `cell_average_age` return `Option`, and `latest_ages_months` returns `Result`. `#[non_exhaustive]` is added to public `subannual` structs; build `ClarkExactAgeOptions` with its new builder methods.
- **`#[non_exhaustive]`:** Added to public deterministic, diagnostics and preprocessing result structs.
- **Student-t copula:** Rejects `SpearmanTarget` (use `KendallTarget`) and rejects non-PD targets by default. Repair is opt-in through `*_with_repair`, which reports `repair_frobenius_distance`.
- **Behaviour changes:**
  - `brosius::project` gives different results on triangles with links where a < 0 or b < 0, such as GenIns. `BrosiusFitRule::RawOls` restores the previous fit.
  - The Clark ALAE default ages move from `j+1` to `j+0.5`.
  - `LognormalLayerSampler` uses the crate RNG, so the same seed now gives different draws.
- **Mack observed counts:** `mack::analyze_observed_counts` defaults to `AllowNegativeIncrements`, the same as `analyze`. `analyze_observed_counts_with_policy` is added.
- **Clark:** Parameter covariance uses the observed information by default (`ClarkCovariance::Observed`, as in Clark 2003). If that matrix is not positive definite it falls back to the expected information and sets `covariance_fallback`. Parameter SEs change.
- **Meyers:** The default priors follow CAS Monograph 1 (`MeyersPriors::monograph_2015`); the old Gaussian priors are available as a preset. New result fields.
- **Log-normal GLM:** Reserves and prediction errors change because of the bias correction.
- **Run-off:** `MackRunoffOptions` adds `max_failed_fraction` (default 1%) and no longer derives `Eq`. A run-off returns an error when the failed fraction exceeds it.

### Changed
- **Release workflow:** Removed automatic deletion of older GitHub Releases and automatic `cargo yank` of every older crates.io version. Yanking is now a manual step, used only for broken releases.
- **Release workflow:** Runs the full CI workflow (reused through `workflow_call`) as a required gate before verifying, building, or publishing a tag. The release clippy gate now also denies `clippy::expect_used`, matching CI.
- **README:** States that the ODP GLM benchmark row uses the closed-form chain-ladder fast path, while R and Python use IRLS. States that the R and Python ratios include interpreter overhead. Matches the chainladder-python version to the parity script (`>= 0.10`, timings on 0.10.1). Narrows the determinism claim to bit-identical results per target/platform, because samplers use the platform `libm`. Documents the WASM status convention.
- **Monotonicity policy:** Every `DevelopmentFactorEngine` method has a `_with_policy` variant, plus `select_with_policy`. Positive tails below 1.0 are accepted under `AllowNegativeIncrements`.
- **Large loss:** The module documents that it operates on per-claim triangles. `large_loss_removed_triangle` is deprecated in favour of `remove_large_loss_excess` and `aggregate_claims_split`.
- **Internal:** Added a shared `deterministic::averaging` kernel module and a neutral `special` module (removing the lateral dependency from `diagnostics` on `capital`/`stochastic`). Refactored the tail-curve and trend clones. New `capital::correlation` module gives `copulas` and `dependence` a one-way dependency. One lognormal LEV implementation.
- **Toolchain:** Added `rust-toolchain.toml`, pinned to 1.100.1 with rustfmt, clippy, rust-analyzer, rust-src and the wasm32 target. The CI MSRV job runs `cargo +1.97.0` explicitly.
- **Configurable numerics:**
  - `GlmOptions` adds `eta_clamp`, `y_floor_relative` and `max_step_halvings`. `GlmReservingResult::eta_clamp_hits` reports how many cells the clamp bound.
  - New `OdpNumerics` (`exclude_zero_residuals`, `leverage_saturation_tol`, `max_failed_fraction`), with `fit_odp_with_numerics`, `bootstrap_odp_with_numerics` and `odp_one_year_runoff_with_numerics`.
- **Internal:** The tridiagonal ODP solver now lives in one module (`stochastic::odp_tridiagonal`), shared by the bootstrap and the GLM. One log-normal MSEP helper.
- **Determinism:** The stochastic docs claim bit-reproducibility only within the same platform.

### Added
- `tests/infra_regressions.rs` adds a seeded (seed `42069`) mutation fuzz over `Triangle` JSON. It covers truncation, byte flips, NaN/null/huge/negative values, shape and overflow mismatches, and structural damage, and checks that nothing panics and that every accepted triangle meets the invariants. The file also tests the `Triangle` construction invariants and the WASM status codes.
- `wasm_exports::RK_STATUS_ERR` constant. `scripts/wasm-smoke.mjs` now covers the status convention.
- Unit test asserting every bundled dataset passes `Triangle::try_new`.
- Brosius: `BrosiusFitRule`, `project_with_fit_rule`. Berquist–Sherman: `fit_severity_trend` (paid-on-closed severity). Clark ALAE: `ages` and `ultimate_emergence` in the result.
- Aggregate loss: `tvar_with_tolerance`, `var_with_tolerance`, `missing_mass`, `FftOptions` / `fft_aggregate_loss_with_options` with an aliasing bound, and `PanjerOptions` with scaling-and-squaring for Poisson/NB at large λ and clipped-mass reporting.
- Higham (2002) `nearest_correlation_matrix` / `HighamOptions`, now used by `repair_correlation_matrix`.
- Severity: `excess_expected_value`, `layer_ratio`.
- New regression suites: `tests/deterministic_regressions.rs` (Friedland Auto goldens derived with exact rational arithmetic, policy coverage, panic-free kernels), `tests/methods_regressions.rs`, `tests/subannual_regressions.rs` and `tests/capital_regressions.rs` (Clayton/Gumbel KS and Kendall τ, high-layer precision, Panjer vs FFT, Higham example, Euler and Myers–Read checks against finite differences).
- **Stochastic tests:** pins for Merz–Wüthrich GenIns CDR (total SE 1,778,967.66), the 4-wide/scalar bootstrap paths with negative pseudo-LDFs, bootstrap and `bootstrap_odp` giving equal draws, ODP analytic PE with a tail against the bootstrap, Clark observed vs expected covariance, Braun parity, Mack-BF min-rule, and a linux-x86_64 golden test of the first bootstrap draws.
- **Slow test:** a seeded log-normal GLM bias/MSEP Monte Carlo, marked `#[ignore]`; run it with `--include-ignored`.

## [0.4.0] - 2026-11-26

### Changed (Breaking)
- **Phase 2 domain architecture:** Reorganized about 50 modules into the `deterministic`, `stochastic`, `capital`, `excess`, `preprocessing`, `diagnostics`, and `subannual` domains. Method facades moved out of `types.rs` into `triangle_ops.rs`.

### Added
- `triangle!` macro for validated literal triangles.
- `stochastic::reserve_runoff`: Mack and ODP one-year re-reserving simulation, with Merz & Wüthrich (2014) traceability.
- Mack (1993) RAA worked example (`docs/examples/mack-chain-ladder-raa.md`, `cargo run --example mack_raa`).
- Eight new Criterion benchmarks and more than 20 published kernel latencies in the README.

### Changed
- Updated `criterion` to 0.8. Benchmarks use `std::hint::black_box`.
- Hardened the pre-commit and pre-push hooks, added a `cargo-deny` and CI security gate, and fixed `nullglob` in the script syntax check.
- The release workflow yanked previous crates.io versions automatically. This is reverted in 0.5.0.
- Removed the tracked `__pycache__`, then removed `CITATION.cff`.

## [0.3.0] - 2026-11-26

### Changed (Breaking)
- **Encapsulated `Triangle` fields:** `n_origins`, `n_devs`, and `data` are now private with read-only accessors (`n_origins()`, `n_devs()`, `data()`) so invariants validated by `try_new` and `Deserialize` cannot be bypassed.
- **Parameterized `excess_layers`:** Replaced hardcoded demo constants, magic `u32` preset codes, and global `OnceLock` caches with `LognormalLayerModel` (closed-form LEV/layer/exceedance/PDF) and caller-owned `LognormalLayerSampler` (deterministic antithetic Monte Carlo with $O(\log N)$ sorted prefix-sum queries).
- **SemVer hardening:** Marked headline result structs (`Mack`, `MerzWuthrichResult`, `ChainLadderProjection`, `BornhuetterFergusonProjection`, `CapeCodProjection`, `BenktanderProjection`, `OdpFit`, `OdpBootstrapResult`, `GlmReservingResult`, `ClarkMleResult`, `AggregateLossDistribution`, `DevelopmentFactors`) as `#[non_exhaustive]`.

### Added
- Reproducible R (`scripts/parity_r.R`) and Python (`scripts/parity_python.py`) parity and timing scripts.
- Dedicated unit tests for `cape_cod`, `validate`, `ragged`, and `triangle`, and split multi-source benchmarks in `tests/source_benchmarks.rs`.
- Refreshed i9-14900K benchmark baseline (`benches/baseline-i9-14900k.json`), `assets/benchmarks.svg`, and `README.md` performance table.

## [0.2.3] - 2026-11-26

### Fixed
- `Triangle` JSON deserialization now maps `null` to `NaN` and validates shape/values via `Triangle::try_new`, preventing invalid triangles from being deserialized.
- Implemented `PartialEq` for `Triangle` and `TriangleView` treating `NaN == NaN` so `tri == tri.clone()` holds for triangles with unobserved cells.
- Mack and Merz–Wüthrich now reject `< 3x3` triangles instead of silently returning `sigma_sq = [0.0]` / `SE = 0` (`analyze_unchecked` returns `NaN` for unestimable `sigma_sq`).
- Panjer and FFT compound aggregate loss now require the severity PMF to sum to `1.0` within `1e-6` instead of accepting defective PMFs.
- Added dedicated `ValidationError` variants (`InvalidProbability`, `InvalidIterations`, `NonPositive`) for out-of-range probabilities, zero bootstrap iterations, and non-positive premiums.

### Changed
- Mack (`mack::analyze`, `mack::analyze_with_tail_uncertainty`, `MackOptions::default`) and Merz–Wüthrich (`merz_wuthrich::analyze`) now default to `MonotonicityPolicy::AllowNegativeIncrements` so canonical triangles with downward revisions (such as Mack's RAA dataset) work out of the box.
- Added `Triangle::merz_wuthrich_with_policy` and `TriangleView::merz_wuthrich_with_policy`.
- Narrowed the README determinism claim to per-target reproducibility.

## [0.2.2] - 2026-11-26

### Changed
- Moved `serde_json` from `[dependencies]` to `[dev-dependencies]` so the library has `serde` as its sole runtime dependency.
- Marked `ValidationError` as `#[non_exhaustive]` and derived `Copy`.
- Derived `Debug`, `Clone`, `Copy`, and `Default` on the checked reserving engine unit structs (`DevelopmentFactorEngine`, `ChainLadderEngine`, `BornhuetterFergusonEngine`, `BenktanderEngine`, `CapeCodEngine`, `GeneralizedCapeCodEngine`).
- Derived `PartialEq` and implemented `Display` for `TriangleView`.
- Marked `wasm_exports` as `#[doc(hidden)]` and documented its single-threaded `wasm32` C-ABI contract.
- Updated `README.md` for crates.io and docs.rs (crates.io install snippet, absolute image and repository links, badges, and full 54-module directory).
- Added `[package.metadata.docs.rs]` and `documentation` metadata in `Cargo.toml`.

## [0.2.1] - 2026-11-26

### Changed
- Decoupled ILF pricing calculations to `pricing-kernels` and focused crate scope on actuarial reserving and reserve-risk capital modeling.
- Optimized GLM/linalg kernels, enabled strict Clippy pedantic/nursery lints, and updated MSRV to Rust 1.97.
- Upgraded benchmark SVG report and CI workflows.

## [0.2.0] - 2026-11-26

### Added
- Sub-annual development, asymmetric grids (2:1/4:1/12:1), stub-period extrapolations, 7 sub-annual CDF/LDF interpolators, and asymmetric CL/BF/Cape Cod/Clark MLE/ODP bootstrap kernels (`subannual`).
- O(n²) `fit_odp` and optimized ODP bootstrap, reserving, and capital allocation kernels.

## [0.1.0] - 2026-11-26

### Added
- Initial release of pure-Rust P&C actuarial reserving, stochastic variance, dependence, aggregate loss convolution, excess layering, and capital allocation kernels with benchmark parity tests and WASM C-ABI exports.
