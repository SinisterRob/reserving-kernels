//! Ergonomic reserving method facades on [`Triangle`] and [`TriangleView`].
//!
//! Kept separate from [`crate::types`] so `types` remains a pure data/geometry
//! layer with zero upward dependencies on reserving methods.

use crate::deterministic::benktander::{BenktanderEngine, BenktanderProjection};
use crate::deterministic::bf::{BornhuetterFergusonEngine, BornhuetterFergusonProjection};
use crate::deterministic::brosius::{self, Brosius};
use crate::deterministic::cape_cod::{CapeCodEngine, CapeCodProjection};
use crate::deterministic::chain_ladder::{ChainLadderEngine, ChainLadderProjection};
use crate::deterministic::generalized_cape_cod::{
    GeneralizedCapeCodEngine, GeneralizedCapeCodProjection,
};
use crate::deterministic::ldf::{DevelopmentFactorEngine, DevelopmentFactors};
use crate::diagnostics::triangle_diagnostics::{self, TriangleDiagnostics};
use crate::error::ReservingError;
use crate::preprocessing::exhibit::{ExhibitBundle, build_exhibit_bundle};
use crate::preprocessing::triangle::{cumulative_to_incremental, latest_diagonal};
use crate::stochastic::bootstrap_odp::{
    OdpBootstrapOptions, OdpBootstrapResult, OdpFit, bootstrap_odp, fit_odp,
};
use crate::stochastic::clark_mle::{self, ClarkMleResult, ClarkOptions};
use crate::stochastic::glm_reserving::{self, GlmOptions, GlmReservingResult};
use crate::stochastic::mack::{self, Mack};
use crate::stochastic::merz_wuthrich::{self, MerzWuthrichResult};
use crate::stochastic::ptf_regression::{self, PtfOptions, PtfResult};
use crate::types::{Triangle, TriangleView};
use crate::validate::MonotonicityPolicy;

impl Triangle {
    /// Extract the latest-observed cumulative anti-diagonal.
    pub fn latest_diagonal(&self) -> Vec<f64> {
        latest_diagonal(&self.view())
    }

    /// Convert this cumulative triangle to an incremental [`Triangle`].
    pub fn to_incremental(&self) -> Self {
        let n = self.n();
        // `cumulative_to_incremental` returns an `n × n` buffer, so the shape
        // invariant of the crate-internal constructor holds. Incremental cells
        // may legitimately be negative, so `try_new` is not applicable here.
        Self::new(n, n, cumulative_to_incremental(&self.view()))
    }

    /// Volume-weighted development factors and CDFs (`tail = 1.0` for fully developed).
    pub fn ldfs(&self, tail: f64) -> Result<DevelopmentFactors, ReservingError> {
        self.view().ldfs(tail)
    }

    /// Volume-weighted Chain-Ladder projection (`tail = 1.0` for fully developed).
    pub fn chain_ladder(&self, tail: f64) -> Result<ChainLadderProjection, ReservingError> {
        self.view().chain_ladder(tail)
    }

    /// Mack (1993) distribution-free chain-ladder reserve & standard error analysis.
    pub fn mack(&self, tail: f64) -> Result<Mack, ReservingError> {
        self.view().mack(tail)
    }

    /// Mack (1993) analysis under an explicit [`MonotonicityPolicy`].
    pub fn mack_with_policy(
        &self,
        tail: f64,
        policy: MonotonicityPolicy,
    ) -> Result<Mack, ReservingError> {
        self.view().mack_with_policy(tail, policy)
    }

    /// Merz & Wüthrich (2008) 1-year Solvency II Claims Development Result (CDR) MSEP.
    pub fn merz_wuthrich(&self) -> Result<MerzWuthrichResult, ReservingError> {
        self.view().merz_wuthrich()
    }

    /// Merz & Wüthrich (2008) 1-year CDR MSEP under an explicit [`MonotonicityPolicy`].
    pub fn merz_wuthrich_with_policy(
        &self,
        policy: MonotonicityPolicy,
    ) -> Result<MerzWuthrichResult, ReservingError> {
        self.view().merz_wuthrich_with_policy(policy)
    }

    /// Bornhuetter–Ferguson projection using volume-weighted LDFs.
    pub fn bf(
        &self,
        expected_ultimate: &[f64],
        tail: f64,
    ) -> Result<BornhuetterFergusonProjection, ReservingError> {
        self.view().bf(expected_ultimate, tail)
    }

    /// Cape Cod projection using volume-weighted LDFs.
    pub fn cape_cod(
        &self,
        premium: &[f64],
        tail: f64,
    ) -> Result<CapeCodProjection, ReservingError> {
        self.view().cape_cod(premium, tail)
    }

    /// Run England & Verrall / Shapland ODP bootstrap reserve simulation.
    pub fn bootstrap_odp(
        &self,
        tail: f64,
        options: OdpBootstrapOptions,
    ) -> Result<OdpBootstrapResult, ReservingError> {
        self.view().bootstrap_odp(tail, options)
    }

    /// Build a complete JSON-serializable reserving exhibit bundle.
    pub fn exhibit(
        &self,
        premium: Option<&[f64]>,
        expected_ultimate: Option<&[f64]>,
        tail: f64,
    ) -> Result<ExhibitBundle, ReservingError> {
        self.view().exhibit(premium, expected_ultimate, tail)
    }

    /// Benktander (1976) / Hovinen credibility projection using volume-weighted LDFs.
    pub fn benktander(
        &self,
        expected_ultimate: &[f64],
        iterations: usize,
        tail: f64,
    ) -> Result<BenktanderProjection, ReservingError> {
        self.view().benktander(expected_ultimate, iterations, tail)
    }

    /// Generalized Cape Cod (Gluck 1997) decay-weighted projection.
    pub fn generalized_cape_cod(
        &self,
        premium: &[f64],
        decay: f64,
        tail: f64,
    ) -> Result<GeneralizedCapeCodProjection, ReservingError> {
        self.view().generalized_cape_cod(premium, decay, tail)
    }

    /// Brosius (1993) OLS credibility development projection.
    pub fn brosius(&self, tail: f64) -> Result<Brosius, ReservingError> {
        self.view().brosius(tail)
    }

    /// Fit the Over-Dispersed Poisson GLM (`fit_odp`) with strict monotonicity.
    pub fn fit_odp(&self, tail: f64) -> Result<OdpFit, ReservingError> {
        self.view().fit_odp(tail)
    }

    /// Fit a reserving GLM (`Odp`, `Gamma`, `Tweedie`, or `LogNormal`) via Householder QR IRLS.
    pub fn glm(&self, options: &GlmOptions) -> Result<GlmReservingResult, ReservingError> {
        self.view().glm(options)
    }

    /// Fit Clark (2003) LDF parametric growth-curve MLE.
    pub fn clark_ldf(&self, options: &ClarkOptions) -> Result<ClarkMleResult, ReservingError> {
        self.view().clark_ldf(options)
    }

    /// Fit Clark (2003) Cape Cod parametric growth-curve MLE.
    pub fn clark_cape_cod(
        &self,
        premium: &[f64],
        options: &ClarkOptions,
    ) -> Result<ClarkMleResult, ReservingError> {
        self.view().clark_cape_cod(premium, options)
    }

    /// Fit Barnett & Zehnwirth (2000) Probabilistic Trend Family log-linear regression.
    pub fn ptf(&self, options: &PtfOptions) -> Result<PtfResult, ReservingError> {
        self.view().ptf(options)
    }

    /// Structural triangle diagnostics (observed/missing counts, monotonicity, zero latest).
    pub fn diagnostics(&self) -> TriangleDiagnostics {
        self.view().diagnostics()
    }
}

impl TriangleView<'_> {
    /// Volume-weighted development factors and CDFs (`tail = 1.0` for fully developed).
    pub fn ldfs(&self, tail: f64) -> Result<DevelopmentFactors, ReservingError> {
        DevelopmentFactorEngine::volume_weighted(self, tail)
    }

    /// Volume-weighted Chain-Ladder projection (`tail = 1.0` for fully developed).
    pub fn chain_ladder(&self, tail: f64) -> Result<ChainLadderProjection, ReservingError> {
        let factors = DevelopmentFactorEngine::volume_weighted(self, tail)?;
        ChainLadderEngine::project(self, &factors)
    }

    /// Mack (1993) distribution-free chain-ladder reserve & standard error analysis.
    pub fn mack(&self, tail: f64) -> Result<Mack, ReservingError> {
        mack::analyze(self, tail)
    }

    /// Mack (1993) analysis under an explicit [`MonotonicityPolicy`].
    pub fn mack_with_policy(
        &self,
        tail: f64,
        policy: MonotonicityPolicy,
    ) -> Result<Mack, ReservingError> {
        mack::analyze_with_policy(self, tail, policy)
    }

    /// Merz & Wüthrich (2008) 1-year Solvency II Claims Development Result (CDR) MSEP.
    pub fn merz_wuthrich(&self) -> Result<MerzWuthrichResult, ReservingError> {
        merz_wuthrich::analyze(self)
    }

    /// Merz & Wüthrich (2008) 1-year CDR MSEP under an explicit [`MonotonicityPolicy`].
    pub fn merz_wuthrich_with_policy(
        &self,
        policy: MonotonicityPolicy,
    ) -> Result<MerzWuthrichResult, ReservingError> {
        merz_wuthrich::analyze_with_policy(self, policy)
    }

    /// Bornhuetter–Ferguson projection using volume-weighted LDFs.
    pub fn bf(
        &self,
        expected_ultimate: &[f64],
        tail: f64,
    ) -> Result<BornhuetterFergusonProjection, ReservingError> {
        let factors = DevelopmentFactorEngine::volume_weighted(self, tail)?;
        BornhuetterFergusonEngine::project_from_expected_ultimate(self, &factors, expected_ultimate)
    }

    /// Cape Cod projection using volume-weighted LDFs.
    pub fn cape_cod(
        &self,
        premium: &[f64],
        tail: f64,
    ) -> Result<CapeCodProjection, ReservingError> {
        let factors = DevelopmentFactorEngine::volume_weighted(self, tail)?;
        CapeCodEngine::project(self, &factors, premium)
    }

    /// Run England & Verrall / Shapland ODP bootstrap reserve simulation.
    pub fn bootstrap_odp(
        &self,
        tail: f64,
        options: OdpBootstrapOptions,
    ) -> Result<OdpBootstrapResult, ReservingError> {
        bootstrap_odp(self, tail, options)
    }

    /// Build a complete JSON-serializable reserving exhibit bundle.
    pub fn exhibit(
        &self,
        premium: Option<&[f64]>,
        expected_ultimate: Option<&[f64]>,
        tail: f64,
    ) -> Result<ExhibitBundle, ReservingError> {
        build_exhibit_bundle(self, premium, expected_ultimate, tail)
    }

    /// Benktander (1976) / Hovinen credibility projection using volume-weighted LDFs.
    pub fn benktander(
        &self,
        expected_ultimate: &[f64],
        iterations: usize,
        tail: f64,
    ) -> Result<BenktanderProjection, ReservingError> {
        let factors = DevelopmentFactorEngine::volume_weighted(self, tail)?;
        BenktanderEngine::project_from_expected_ultimate(
            self,
            &factors,
            expected_ultimate,
            iterations,
        )
    }

    /// Generalized Cape Cod (Gluck 1997) decay-weighted projection.
    pub fn generalized_cape_cod(
        &self,
        premium: &[f64],
        decay: f64,
        tail: f64,
    ) -> Result<GeneralizedCapeCodProjection, ReservingError> {
        let factors = DevelopmentFactorEngine::volume_weighted(self, tail)?;
        GeneralizedCapeCodEngine::project(self, &factors, premium, decay)
    }

    /// Brosius (1993) OLS credibility development projection.
    pub fn brosius(&self, tail: f64) -> Result<Brosius, ReservingError> {
        brosius::project(self, tail)
    }

    /// Fit the Over-Dispersed Poisson GLM (`fit_odp`) with strict monotonicity.
    pub fn fit_odp(&self, tail: f64) -> Result<OdpFit, ReservingError> {
        fit_odp(self, tail, MonotonicityPolicy::Strict)
    }

    /// Fit a reserving GLM (`Odp`, `Gamma`, `Tweedie`, or `LogNormal`) via Householder QR IRLS.
    pub fn glm(&self, options: &GlmOptions) -> Result<GlmReservingResult, ReservingError> {
        glm_reserving::fit_glm(self, options)
    }

    /// Fit Clark (2003) LDF parametric growth-curve MLE.
    pub fn clark_ldf(&self, options: &ClarkOptions) -> Result<ClarkMleResult, ReservingError> {
        clark_mle::fit_clark_ldf(self, options)
    }

    /// Fit Clark (2003) Cape Cod parametric growth-curve MLE.
    pub fn clark_cape_cod(
        &self,
        premium: &[f64],
        options: &ClarkOptions,
    ) -> Result<ClarkMleResult, ReservingError> {
        clark_mle::fit_clark_cape_cod(self, premium, options)
    }

    /// Fit Barnett & Zehnwirth (2000) Probabilistic Trend Family log-linear regression.
    pub fn ptf(&self, options: &PtfOptions) -> Result<PtfResult, ReservingError> {
        ptf_regression::fit_ptf(self, options)
    }

    /// Structural triangle diagnostics (observed/missing counts, monotonicity, zero latest).
    pub fn diagnostics(&self) -> TriangleDiagnostics {
        triangle_diagnostics::analyze(self)
    }
}
