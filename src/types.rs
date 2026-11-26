//! Prevalidated columnar triangle inputs.
//!
//! A loss-development triangle is an `n_origins × n_devs` grid of cumulative
//! amounts (paid or incurred). Cell `(i, j)` is observed iff `i + j ≤ n - 1`
//! for a square triangle. Unobserved cells are represented as `NaN` so the
//! compute kernels stay branch-light — null-handling lives at the ingress
//! boundary (see [`crate::validate`]).

use std::fmt;

use crate::benktander::{BenktanderEngine, BenktanderProjection};
use crate::bf::{BornhuetterFergusonEngine, BornhuetterFergusonProjection};
use crate::bootstrap_odp::{OdpBootstrapOptions, OdpBootstrapResult, OdpFit, bootstrap_odp, fit_odp};
use crate::brosius::{self, Brosius};
use crate::cape_cod::{CapeCodEngine, CapeCodProjection};
use crate::chain_ladder::{ChainLadderEngine, ChainLadderProjection};
use crate::clark_mle::{self, ClarkMleResult, ClarkOptions};
use crate::diagnostics::triangle_diagnostics::{self, TriangleDiagnostics};
use crate::exhibit::{ExhibitBundle, build_exhibit_bundle};
use crate::generalized_cape_cod::{GeneralizedCapeCodEngine, GeneralizedCapeCodProjection};
use crate::glm_reserving::{self, GlmOptions, GlmReservingResult};
use crate::ldf::{DevelopmentFactorEngine, DevelopmentFactors};
use crate::mack::{self, Mack};
use crate::merz_wuthrich::{self, MerzWuthrichResult};
use crate::ptf_regression::{self, PtfOptions, PtfResult};
use crate::triangle::{cumulative_to_incremental, latest_diagonal};
use crate::validate::{MonotonicityPolicy, ValidationError, validate_triangle};

/// An `n_origins × n_devs` cumulative loss triangle in row-major layout.
///
/// `data[i * n_devs + j]` is the cumulative loss for origin `i` at
/// development age `j`. Unobserved cells are `NaN`. The kernels assume the
/// triangle is *square* (`n_origins == n_devs`) and that cell `(i, j)` is
/// observed for `i + j ≤ n - 1`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Triangle {
    /// Number of origin (accident / policy) periods.
    pub n_origins: usize,
    /// Number of development periods.
    pub n_devs: usize,
    /// Row-major buffer of length `n_origins * n_devs`.
    pub data: Vec<f64>,
}

impl Triangle {
    /// Construct a triangle from a row-major slice without validation.
    ///
    /// Callers must ensure `data.len() == n_origins * n_devs`. Use
    /// [`Self::try_new`] at external ingress boundaries when full validation
    /// (square, finite, non-negative observed cells) is required.
    #[inline]
    pub fn new(n_origins: usize, n_devs: usize, data: Vec<f64>) -> Self {
        debug_assert_eq!(data.len(), n_origins * n_devs);
        Self {
            n_origins,
            n_devs,
            data,
        }
    }

    /// Construct and validate a square cumulative triangle.
    ///
    /// This enforces the same shape, finiteness, and non-negative observed-cell
    /// contract used by the checked reserving engines. Row monotonicity is
    /// still method-specific and is validated by those engines.
    pub fn try_new(
        n_origins: usize,
        n_devs: usize,
        data: Vec<f64>,
    ) -> Result<Self, ValidationError> {
        let triangle = Self {
            n_origins,
            n_devs,
            data,
        };
        validate_triangle(&triangle.view())?;
        Ok(triangle)
    }

    /// Construct and validate a square cumulative triangle directly from
    /// ragged upper-triangle rows (no manual `f64::NAN` padding required).
    ///
    /// Row `i` must have length `n - i` where `n = rows.len() > 0`.
    /// For owned `Vec<Vec<f64>>` inputs, see [`Self::try_from_rows`].
    ///
    /// # Example
    /// ```rust
    /// use reserving_kernels::Triangle;
    ///
    /// let tri = Triangle::try_from_upper_rows(&[
    ///     &[100.0, 160.0, 195.0],
    ///     &[120.0, 190.0],
    ///     &[135.0],
    /// ])?;
    /// assert_eq!(tri.n(), 3);
    /// assert_eq!(tri.latest_diagonal(), vec![195.0, 190.0, 135.0]);
    /// # Ok::<(), reserving_kernels::validate::ValidationError>(())
    /// ```
    pub fn try_from_upper_rows(rows: &[&[f64]]) -> Result<Self, ValidationError> {
        Self::try_from_rows(rows)
    }

    /// Generic variant of [`Self::try_from_upper_rows`] accepting any slice of
    /// row buffers (`&[Vec<f64>]`, `&[&[f64]]`, etc.).
    pub fn try_from_rows<R: AsRef<[f64]>>(rows: &[R]) -> Result<Self, ValidationError> {
        let n = rows.len();
        if n == 0 {
            return Err(ValidationError::ShapeMismatch);
        }
        let mut data = vec![f64::NAN; n * n];
        for (i, row) in rows.iter().enumerate() {
            let row = row.as_ref();
            if row.len() != n - i {
                return Err(ValidationError::ShapeMismatch);
            }
            data[i * n..i * n + row.len()].copy_from_slice(row);
        }
        Self::try_new(n, n, data)
    }

    /// Construct and validate a square cumulative triangle by cumulating
    /// incremental upper-triangle rows along each origin period.
    ///
    /// Row `i` must have length `n - i` where `n = rows.len() > 0`.
    pub fn try_from_incremental_rows(rows: &[&[f64]]) -> Result<Self, ValidationError> {
        let n = rows.len();
        if n == 0 {
            return Err(ValidationError::ShapeMismatch);
        }
        let mut data = vec![f64::NAN; n * n];
        for (i, row) in rows.iter().enumerate() {
            if row.len() != n - i {
                return Err(ValidationError::ShapeMismatch);
            }
            let mut acc = 0.0;
            for (j, &inc) in row.iter().enumerate() {
                if !inc.is_finite() {
                    return Err(ValidationError::NotFinite);
                }
                acc += inc;
                data[i * n + j] = acc;
            }
        }
        Self::try_new(n, n, data)
    }

    /// Cumulative value at origin `i`, development `j`. `NaN` if unobserved.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n_devs + j]
    }

    /// Observed cumulative slice for origin `i` (length `n - i`, excluding trailing `NaN`s).
    #[inline]
    pub fn origin_row(&self, i: usize) -> &[f64] {
        let n = self.n();
        let len = n.saturating_sub(i);
        let start = i * self.n_devs;
        &self.data[start..start + len]
    }

    /// Square triangle dimension `min(n_origins, n_devs)`.
    #[inline]
    pub fn n(&self) -> usize {
        self.n_origins.min(self.n_devs)
    }

    /// `true` iff the triangle has zero origins or zero development periods.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_origins == 0 || self.n_devs == 0
    }

    /// Borrow as a [`TriangleView`].
    #[inline]
    pub fn view(&self) -> TriangleView<'_> {
        TriangleView {
            n_origins: self.n_origins,
            n_devs: self.n_devs,
            data: &self.data,
        }
    }

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

impl fmt::Display for Triangle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.n_origins {
            if i > 0 {
                writeln!(f)?;
            }
            for j in 0..self.n_devs {
                if j > 0 {
                    write!(f, "\t")?;
                }
                let v = self.data[i * self.n_devs + j];
                if v.is_finite() {
                    write!(f, "{v:.2}")?;
                } else {
                    write!(f, "-")?;
                }
            }
        }
        Ok(())
    }
}

/// A non-owning view over a triangle's row-major buffer.
#[derive(Debug, Clone, Copy)]
pub struct TriangleView<'a> {
    /// Number of origin periods.
    pub n_origins: usize,
    /// Number of development periods.
    pub n_devs: usize,
    /// Row-major buffer.
    pub data: &'a [f64],
}

impl<'a> TriangleView<'a> {
    /// Cumulative value at origin `i`, development `j`. `NaN` if unobserved.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n_devs + j]
    }

    /// `true` iff cell `(i, j)` is scheduled to be observed under the
    /// upper-triangular square-triangle convention (`i + j < n`).
    #[inline]
    pub fn is_scheduled_observed(&self, i: usize, j: usize) -> bool {
        let n = self.n_origins.min(self.n_devs);
        i < n && j < n && i + j < n
    }

    /// `true` iff cell `(i, j)` is scheduled observed and holds a finite value.
    #[inline]
    pub fn is_valid_observed(&self, i: usize, j: usize) -> bool {
        self.is_scheduled_observed(i, j) && self.get(i, j).is_finite()
    }

    /// `true` iff cell `(i, j)` is scheduled observed and holds a finite value.
    ///
    /// This preserves the original hot-path predicate. Use
    /// [`Self::is_scheduled_observed`] when validation needs to distinguish
    /// triangle geometry from numeric validity.
    #[inline]
    pub fn is_observed(&self, i: usize, j: usize) -> bool {
        self.is_valid_observed(i, j)
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::ValidationError;

    #[test]
    fn try_new_rejects_shape_mismatch() {
        assert_eq!(
            Triangle::try_new(2, 2, vec![1.0, 2.0, 3.0]),
            Err(ValidationError::ShapeMismatch)
        );
    }

    #[test]
    fn try_new_accepts_valid_square_triangle() {
        let triangle = Triangle::try_new(2, 2, vec![1.0, 2.0, 3.0, f64::NAN]).expect("triangle");

        assert_eq!(triangle.n_origins, 2);
        assert_eq!(triangle.n_devs, 2);
    }
}
