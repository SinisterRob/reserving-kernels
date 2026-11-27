//! Sub-annual development (semiannual/quarterly/monthly), asymmetric grids,
//! and pre-12-month stub-period reserving kernels.
//!
//! Organizes five cohesive kernel groups:
//!
//! 1. **Grid Geometry & Age Mapping** ([`grid`]):
//!    * [`asymmetric_age_grid`] — Friedland (2010) non-square valuation age grid.
//!    * [`stub_average_age`] & [`stub_average_age_weighted`] — Clark (2003), Bloom (2015)
//!      effective average claim age for stub/incomplete origins.
//!    * [`calendar_diagonal_index`] & [`diagnose_asymmetric_calendar_effects`] — Shapland (2016)
//!      true calendar-diagonal alignment across asymmetric grids.
//! 2. **Sub-Annual LDF & CDF Interpolation** ([`interp`]):
//!    * [`interp_linear_cdf`], [`interp_linear_pct_reported`], [`interp_exponential`] (Bloom 2015),
//!      [`interp_inverse_power`] (Sherman 1984), [`interp_swimon`] (Robbin 2018),
//!      [`interp_spliced_weibull`] (Robbin 2018), and [`interp_ivf12`] (Robbin 2018).
//! 3. **Pre-12-Month Stub Extrapolation** ([`stub`]):
//!    * [`stub_plus12`], [`stub_power_ratio`], [`stub_natural_log`] (Bloom 2015), and
//!      [`stub_exposure_scale`] / [`decompose_combined_stub_cdf`] (Friedland 2010, Bloom 2015).
//! 4. **Asymmetric & Interim-Valuation Reserving** ([`reserving`], [`clark_exact`], [`odp_asymmetric`]):
//!    * [`asymmetric_chain_ladder`], [`interim_bf`], [`interim_cape_cod`] (Friedland 2010),
//!      [`clark_mle_exact_age`] (Clark 2003), and [`asymmetric_odp_bootstrap`] (Shapland 2016).
//! 5. **Diagnostics & Consistency Checks** ([`diagnostics`]):
//!    * [`equilibrium_ibnr_check`] (Robbin 2018) and [`subannual_product_parity`].

pub mod clark_exact;
pub mod diagnostics;
pub mod grid;
pub mod interp;
pub mod odp_asymmetric;
pub mod reserving;
pub mod stub;

pub use clark_exact::{
    ClarkExactAgeOptions, ClarkExactAgeResult, clark_mle_exact_age, fit_clark_cape_cod_exact_age,
    fit_clark_ldf_exact_age,
};
pub use diagnostics::{
    AnnualWindowParity, EquilibriumIbnrCheckResult, SubannualProductParityResult,
    equilibrium_ibnr_check, subannual_product_parity,
};
pub use grid::{
    AsymmetricAgeGrid, AsymmetricCalendarDiagnostics, CalendarDiagonalCell, CalendarDiagonalMap,
    asymmetric_age_grid, calendar_diagonal_index, diagnose_asymmetric_calendar_effects,
    stub_average_age, stub_average_age_weighted,
};
pub use interp::{
    InterpolatedPattern, InversePowerInterpResult, SubannualInterpMethod, interp_exponential,
    interp_inverse_power, interp_ivf12, interp_linear_cdf, interp_linear_pct_reported,
    interp_spliced_weibull, interp_swimon, interpolate_pattern,
};
pub use odp_asymmetric::{
    AsymmetricOdpBootstrapResult, AsymmetricOdpFit, asymmetric_odp_bootstrap, fit_asymmetric_odp,
};
pub use reserving::{
    AsymmetricChainLadderProjection, InterimBfProjection, InterimCapeCodProjection,
    asymmetric_chain_ladder, interim_bf, interim_cape_cod,
};
pub use stub::{
    StubExposureScaleResult, StubExtrapolationResult, decompose_combined_stub_cdf,
    stub_exposure_scale, stub_natural_log, stub_plus12, stub_power_ratio,
};
