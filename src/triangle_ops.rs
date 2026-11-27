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
use crate::validate::{MonotonicityPolicy, ValidationError};

impl Triangle {
    /// Extract the latest-observed cumulative anti-diagonal.
    pub fn latest_diagonal(&self) -> Vec<f64> {
        latest_diagonal(&self.view())
    }

    /// Convert this cumulative triangle to an incremental [`Triangle`].
    pub fn to_incremental(&self) -> Self {
        let n = self.n();
        Self::new(n, n, cumulative_to_incremental(&self.view()))
    }

    /// Volume-weighted development factors and CDFs (`tail = 1.0` for fully developed).
    pub fn ldfs(&self, tail: f64) -> Result<DevelopmentFactors, ValidationError> {
        self.view().ldfs(tail)
    }

    /// Volume-weighted Chain-Ladder projection (`tail = 1.0` for fully developed).
    pub fn chain_ladder(&self, tail: f64) -> Result<ChainLadderProjection, ValidationError> {
        self.view().chain_ladder(tail)
    }

    /// Mack (1993) distribution-free chain-ladder reserve & standard error analysis.
    pub fn mack(&self, tail: f64) -> Result<Mack, ValidationError> {
        self.view().mack(tail)
    }

    /// Mack (1993) analysis under an explicit [`MonotonicityPolicy`].
    pub fn mack_with_policy(
        &self,
        tail: f64,
        policy: MonotonicityPolicy,
    ) -> Result<Mack, ValidationError> {
        self.view().mack_with_policy(tail, policy)
    }

    /// Merz & Wüthrich (2008) 1-year Solvency II Claims Development Result (CDR) MSEP.
    pub fn merz_wuthrich(&self) -> Result<MerzWuthrichResult, ValidationError> {
        self.view().merz_wuthrich()
    }

    /// Merz & Wüthrich (2008) 1-year CDR MSEP under an explicit [`MonotonicityPolicy`].
    pub fn merz_wuthrich_with_policy(
        &self,
        policy: MonotonicityPolicy,
    ) -> Result<MerzWuthrichResult, ValidationError> {
        self.view().merz_wuthrich_with_policy(policy)
    }

    /// Bornhuetter–Ferguson projection using volume-weighted LDFs.
    pub fn bf(
        &self,
        expected_ultimate: &[f64],
        tail: f64,
    ) -> Result<BornhuetterFergusonProjection, ValidationError> {
        self.view().bf(expected_ultimate, tail)
    }

    /// Cape Cod projection using volume-weighted LDFs.
    pub fn cape_cod(
        &self,
        premium: &[f64],
        tail: f64,
    ) -> Result<CapeCodProjection, ValidationError> {
        self.view().cape_cod(premium, tail)
    }

    /// Run England & Verrall / Shapland ODP bootstrap reserve simulation.
    pub fn bootstrap_odp(
        &self,
        tail: f64,
        options: OdpBootstrapOptions,
    ) -> Result<OdpBootstrapResult, ValidationError> {
        self.view().bootstrap_odp(tail, options)
    }

    /// Build a complete JSON-serializable reserving exhibit bundle.
    pub fn exhibit(
        &self,
        premium: Option<&[f64]>,
        expected_ultimate: Option<&[f64]>,
        tail: f64,
    ) -> Result<ExhibitBundle, ValidationError> {
        self.view().exhibit(premium, expected_ultimate, tail)
    }

    /// Benktander (1976) / Hovinen credibility projection using volume-weighted LDFs.
    pub fn benktander(
        &self,
        expected_ultimate: &[f64],
        iterations: usize,
        tail: f64,
    ) -> Result<BenktanderProjection, ValidationError> {
        self.view().benktander(expected_ultimate, iterations, tail)
    }

    /// Generalized Cape Cod (Gluck 1997) decay-weighted projection.
    pub fn generalized_cape_cod(
        &self,
        premium: &[f64],
        decay: f64,
        tail: f64,
    ) -> Result<GeneralizedCapeCodProjection, ValidationError> {
        self.view().generalized_cape_cod(premium, decay, tail)
    }

    /// Brosius (1993) OLS credibility development projection.
    pub fn brosius(&self, tail: f64) -> Result<Brosius, ValidationError> {
        self.view().brosius(tail)
    }

    /// Fit the Over-Dispersed Poisson GLM (`fit_odp`) with strict monotonicity.
    pub fn fit_odp(&self, tail: f64) -> Result<OdpFit, ValidationError> {
        self.view().fit_odp(tail)
    }

    /// Fit a reserving GLM (`Odp`, `Gamma`, `Tweedie`, or `LogNormal`) via Householder QR IRLS.
    pub fn glm(&self, options: &GlmOptions) -> Result<GlmReservingResult, ValidationError> {
        self.view().glm(options)
    }

    /// Fit Clark (2003) LDF parametric growth-curve MLE.
    pub fn clark_ldf(&self, options: &ClarkOptions) -> Result<ClarkMleResult, ValidationError> {
        self.view().clark_ldf(options)
    }

    /// Fit Clark (2003) Cape Cod parametric growth-curve MLE.
    pub fn clark_cape_cod(
        &self,
        premium: &[f64],
        options: &ClarkOptions,
    ) -> Result<ClarkMleResult, ValidationError> {
        self.view().clark_cape_cod(premium, options)
    }

    /// Fit Barnett & Zehnwirth (2000) Probabilistic Trend Family log-linear regression.
    pub fn ptf(&self, options: &PtfOptions) -> Result<PtfResult, ValidationError> {
        self.view().ptf(options)
    }

    /// Structural triangle diagnostics (observed/missing counts, monotonicity, zero latest).
    pub fn diagnostics(&self) -> TriangleDiagnostics {
        self.view().diagnostics()
    }
}

impl TriangleView<'_> {
    /// Volume-weighted development factors and CDFs (`tail = 1.0` for fully developed).
    pub fn ldfs(&self, tail: f64) -> Result<DevelopmentFactors, ValidationError> {
        DevelopmentFactorEngine::volume_weighted(self, tail)
    }

    /// Volume-weighted Chain-Ladder projection (`tail = 1.0` for fully developed).
    pub fn chain_ladder(&self, tail: f64) -> Result<ChainLadderProjection, ValidationError> {
        let factors = DevelopmentFactorEngine::volume_weighted(self, tail)?;
        ChainLadderEngine::project(self, &factors)
    }

    /// Mack (1993) distribution-free chain-ladder reserve & standard error analysis.
    pub fn mack(&self, tail: f64) -> Result<Mack, ValidationError> {
        mack::analyze(self, tail)
    }

    /// Mack (1993) analysis under an explicit [`MonotonicityPolicy`].
    pub fn mack_with_policy(
        &self,
        tail: f64,
        policy: MonotonicityPolicy,
    ) -> Result<Mack, ValidationError> {
        mack::analyze_with_policy(self, tail, policy)
    }

    /// Merz & Wüthrich (2008) 1-year Solvency II Claims Development Result (CDR) MSEP.
    pub fn merz_wuthrich(&self) -> Result<MerzWuthrichResult, ValidationError> {
        merz_wuthrich::analyze(self)
    }

    /// Merz & Wüthrich (2008) 1-year CDR MSEP under an explicit [`MonotonicityPolicy`].
    pub fn merz_wuthrich_with_policy(
        &self,
        policy: MonotonicityPolicy,
    ) -> Result<MerzWuthrichResult, ValidationError> {
        merz_wuthrich::analyze_with_policy(self, policy)
    }

    /// Bornhuetter–Ferguson projection using volume-weighted LDFs.
    pub fn bf(
        &self,
        expected_ultimate: &[f64],
        tail: f64,
    ) -> Result<BornhuetterFergusonProjection, ValidationError> {
        let factors = DevelopmentFactorEngine::volume_weighted(self, tail)?;
        BornhuetterFergusonEngine::project_from_expected_ultimate(self, &factors, expected_ultimate)
    }

    /// Cape Cod projection using volume-weighted LDFs.
    pub fn cape_cod(
        &self,
        premium: &[f64],
        tail: f64,
    ) -> Result<CapeCodProjection, ValidationError> {
        let factors = DevelopmentFactorEngine::volume_weighted(self, tail)?;
        CapeCodEngine::project(self, &factors, premium)
    }

    /// Run England & Verrall / Shapland ODP bootstrap reserve simulation.
    pub fn bootstrap_odp(
        &self,
        tail: f64,
        options: OdpBootstrapOptions,
    ) -> Result<OdpBootstrapResult, ValidationError> {
        bootstrap_odp(self, tail, options)
    }

    /// Build a complete JSON-serializable reserving exhibit bundle.
    pub fn exhibit(
        &self,
        premium: Option<&[f64]>,
        expected_ultimate: Option<&[f64]>,
        tail: f64,
    ) -> Result<ExhibitBundle, ValidationError> {
        build_exhibit_bundle(self, premium, expected_ultimate, tail)
    }

    /// Benktander (1976) / Hovinen credibility projection using volume-weighted LDFs.
    pub fn benktander(
        &self,
        expected_ultimate: &[f64],
        iterations: usize,
        tail: f64,
    ) -> Result<BenktanderProjection, ValidationError> {
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
    ) -> Result<GeneralizedCapeCodProjection, ValidationError> {
        let factors = DevelopmentFactorEngine::volume_weighted(self, tail)?;
        GeneralizedCapeCodEngine::project(self, &factors, premium, decay)
    }

    /// Brosius (1993) OLS credibility development projection.
    pub fn brosius(&self, tail: f64) -> Result<Brosius, ValidationError> {
        brosius::project(self, tail)
    }

    /// Fit the Over-Dispersed Poisson GLM (`fit_odp`) with strict monotonicity.
    pub fn fit_odp(&self, tail: f64) -> Result<OdpFit, ValidationError> {
        fit_odp(self, tail, MonotonicityPolicy::Strict)
    }

    /// Fit a reserving GLM (`Odp`, `Gamma`, `Tweedie`, or `LogNormal`) via Householder QR IRLS.
    pub fn glm(&self, options: &GlmOptions) -> Result<GlmReservingResult, ValidationError> {
        glm_reserving::fit_glm(self, options)
    }

    /// Fit Clark (2003) LDF parametric growth-curve MLE.
    pub fn clark_ldf(&self, options: &ClarkOptions) -> Result<ClarkMleResult, ValidationError> {
        clark_mle::fit_clark_ldf(self, options)
    }

    /// Fit Clark (2003) Cape Cod parametric growth-curve MLE.
    pub fn clark_cape_cod(
        &self,
        premium: &[f64],
        options: &ClarkOptions,
    ) -> Result<ClarkMleResult, ValidationError> {
        clark_mle::fit_clark_cape_cod(self, premium, options)
    }

    /// Fit Barnett & Zehnwirth (2000) Probabilistic Trend Family log-linear regression.
    pub fn ptf(&self, options: &PtfOptions) -> Result<PtfResult, ValidationError> {
        ptf_regression::fit_ptf(self, options)
    }

    /// Structural triangle diagnostics (observed/missing counts, monotonicity, zero latest).
    pub fn diagnostics(&self) -> TriangleDiagnostics {
        triangle_diagnostics::analyze(self)
    }
}
