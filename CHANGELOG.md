# Changelog

All notable changes to `reserving-kernels` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
