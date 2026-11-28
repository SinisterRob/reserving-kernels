//! Parameterized lognormal severity layering and Limited Expected Values (LEV).
//!
//! Provides exact closed-form lognormal limited expected value $E[X \wedge L]$
//! and layer expectations via [`LognormalLayerModel`], alongside a
//! deterministic SplitMix64 + antithetic Marsaglia polar Monte Carlo engine
//! ([`LognormalLayerSampler`]) with sorted prefix-sum $O(\log N)$ layer queries.

use crate::validate::ValidationError;

const INV_SQRT_TAU: f64 = 0.398_942_280_401_432_7;

/// Strongly-typed excess-of-loss layer specification `(attachment, limit)`
/// that eliminates ambiguity between `(attachment, top)` and `(attachment, limit)`.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LayerSpec {
    /// Attachment point `A >= 0`.
    pub attachment: f64,
    /// Layer width (limit) `L >= 0`.
    pub limit: f64,
}

impl LayerSpec {
    /// Construct a validated layer from `(attachment, limit)` where `top = attachment + limit`.
    pub fn from_attachment_limit(attachment: f64, limit: f64) -> Result<Self, ValidationError> {
        if !attachment.is_finite() || !limit.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if attachment < 0.0 || limit < 0.0 {
            return Err(ValidationError::Negative);
        }
        if !(attachment + limit).is_finite() {
            return Err(ValidationError::NotFinite);
        }
        Ok(Self { attachment, limit })
    }

    /// Construct a validated layer from `(attachment, top)` where `limit = (top - attachment).max(0.0)`.
    pub fn from_attachment_top(attachment: f64, top: f64) -> Result<Self, ValidationError> {
        if !attachment.is_finite() || !top.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if attachment < 0.0 || top < 0.0 {
            return Err(ValidationError::Negative);
        }
        Ok(Self {
            attachment,
            limit: (top - attachment).max(0.0),
        })
    }

    /// Upper exhaustion point `attachment + limit`.
    #[inline]
    pub fn top(self) -> f64 {
        self.attachment + self.limit
    }
}

/// Closed-form lognormal severity model parameterized by ground-up `mean > 0`
/// and log-scale standard deviation `sigma > 0`.
///
/// The underlying lognormal parameters satisfy
/// $\mu = \ln(\text{mean}) - \tfrac{1}{2}\sigma^2$ so `E[X] = mean`.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LognormalLayerModel {
    mean: f64,
    sigma: f64,
    inv_sigma: f64,
    log_mean: f64,
    pdf_scale: f64,
}

impl LognormalLayerModel {
    /// Construct a validated lognormal layer model with ground-up `mean > 0`
    /// and log-scale standard deviation `sigma > 0`.
    pub fn try_new(mean: f64, sigma: f64) -> Result<Self, ValidationError> {
        if !mean.is_finite() || !sigma.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        if mean <= 0.0 {
            return Err(ValidationError::NonPositive("ground-up mean"));
        }
        if sigma <= 0.0 {
            return Err(ValidationError::NonPositive("lognormal sigma"));
        }
        let log_mean = mean.ln() - 0.5 * sigma * sigma;
        if !log_mean.is_finite() {
            return Err(ValidationError::NotFinite);
        }
        Ok(Self {
            mean,
            sigma,
            inv_sigma: 1.0 / sigma,
            log_mean,
            pdf_scale: INV_SQRT_TAU / sigma,
        })
    }

    /// Ground-up expected severity `E[X]`.
    #[inline]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Log-scale standard deviation $\sigma$.
    #[inline]
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Closed-form Limited Expected Value $E[X \wedge L]$.
    ///
    /// Returns `0.0` for `limit <= 0.0`, `mean` for `+∞`, and `NaN` for `NaN`.
    #[inline]
    pub fn exact_lev(&self, limit: f64) -> f64 {
        if limit.is_nan() {
            return f64::NAN;
        }
        if limit <= 0.0 {
            return 0.0;
        }
        if !limit.is_finite() {
            return self.mean;
        }
        let z = (limit.ln() - self.log_mean) * self.inv_sigma;
        self.mean * crate::excess::severity::normal_cdf_exact(z - self.sigma)
            + limit * crate::excess::severity::normal_cdf_exact(-z)
    }

    /// Closed-form expected loss in the layer `(attachment, top]`:
    /// $E[\min(X, \text{top}) - \min(X, \text{attachment})]$.
    ///
    /// **Parameter convention:** `top` is the upper exhaustion point
    /// (`attachment + limit`), unlike [`crate::excess::severity::SeverityDistribution::layer_expected_loss`]
    /// which takes `(attachment, limit)`.
    ///
    /// Returns `0.0` when `top <= attachment`.
    #[inline]
    pub fn exact_layer(&self, attachment: f64, top: f64) -> f64 {
        if attachment.is_nan() || top.is_nan() {
            return f64::NAN;
        }
        let att = attachment.max(0.0);
        if top <= att {
            return 0.0;
        }
        (self.exact_lev(top) - self.exact_lev(att)).max(0.0)
    }

    /// Closed-form expected loss in the typed [`LayerSpec`] layer.
    #[inline]
    pub fn exact_layer_spec(&self, layer: LayerSpec) -> f64 {
        self.exact_layer(layer.attachment, layer.top())
    }

    /// Exceedance probability $P(X > \text{threshold})$.
    #[inline]
    pub fn exceedance_prob(&self, threshold: f64) -> f64 {
        if threshold.is_nan() {
            return f64::NAN;
        }
        if threshold <= 0.0 {
            return 1.0;
        }
        if !threshold.is_finite() {
            return 0.0;
        }
        let z = (threshold.ln() - self.log_mean) * self.inv_sigma;
        crate::excess::severity::normal_cdf_exact(-z)
    }

    /// Lognormal probability density function $f_X(x)$ at loss amount `x`.
    #[inline]
    pub fn pdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x <= 0.0 || !x.is_finite() {
            return 0.0;
        }
        let ln_x = x.ln();
        let z = (ln_x - self.log_mean) * self.inv_sigma;
        (self.pdf_scale.ln() - ln_x - 0.5 * z * z).exp()
    }
}

/// Deterministic antithetic Monte Carlo lognormal layer sampler with sorted
/// prefix sums for $O(\log N)$ repeated layer evaluations.
///
/// No process-global caches are used; callers own the sampler lifetime.
#[derive(Debug, Clone)]
pub struct LognormalLayerSampler {
    model: LognormalLayerModel,
    inv_n: f64,
    sorted_normals: Vec<f64>,
    prefix_sums: Vec<f64>,
}

impl LognormalLayerSampler {
    /// Generate `n_samples` (rounded up to an even count for exact antithetic
    /// pairing) deterministic SplitMix64 + Marsaglia polar draws and build the
    /// sorted prefix-sum table.
    pub fn try_new(
        model: LognormalLayerModel,
        n_samples: usize,
        seed: u64,
    ) -> Result<Self, ValidationError> {
        if n_samples < 2 {
            return Err(ValidationError::InvalidParameter(
                "LognormalLayerSampler requires at least 2 samples",
            ));
        }
        let half = n_samples.div_ceil(2);
        let total = half.checked_mul(2).ok_or(ValidationError::ShapeMismatch)?;
        let mut state = seed;
        let mut pos = Vec::with_capacity(half);
        while pos.len() < half {
            let u = 2.0 * uniform(&mut state) - 1.0;
            let v = 2.0 * uniform(&mut state) - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let scale = (-2.0 * s.ln() / s).sqrt();
                pos.push((u * scale).abs());
                if pos.len() < half {
                    pos.push((v * scale).abs());
                }
            }
        }
        pos.sort_unstable_by(f64::total_cmp);

        let mut sorted_normals = Vec::with_capacity(total);
        for &z in pos.iter().rev() {
            sorted_normals.push(-z);
        }
        sorted_normals.extend_from_slice(&pos);

        let mut prefix_sums = Vec::with_capacity(total + 1);
        let mut acc = 0.0;
        prefix_sums.push(0.0);
        for &z in &sorted_normals {
            acc += (model.log_mean + model.sigma * z).exp();
            prefix_sums.push(acc);
        }
        if !acc.is_finite() {
            return Err(ValidationError::NotFinite);
        }

        Ok(Self {
            model,
            inv_n: 1.0 / (total as f64),
            sorted_normals,
            prefix_sums,
        })
    }

    /// Underlying analytical [`LognormalLayerModel`].
    #[inline]
    pub fn model(&self) -> LognormalLayerModel {
        self.model
    }

    /// Number of antithetic draws held by this sampler.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.sorted_normals.len()
    }

    /// Simulated Limited Expected Value $E[X \wedge L]$ in $O(\log N)$.
    #[inline]
    pub fn simulated_lev(&self, limit: f64) -> f64 {
        if limit.is_nan() {
            return f64::NAN;
        }
        if limit <= 0.0 {
            return 0.0;
        }
        let n = self.sorted_normals.len();
        if !limit.is_finite() {
            return self.prefix_sums[n] * self.inv_n;
        }
        let z_thresh = (limit.ln() - self.model.log_mean) * self.model.inv_sigma;
        let k = self.sorted_normals.partition_point(|&z| z < z_thresh);
        (self.prefix_sums[k] + ((n - k) as f64) * limit) * self.inv_n
    }

    /// Simulated expected loss in the layer `(attachment, top]` in $O(\log N)$.
    #[inline]
    pub fn simulated_layer(&self, attachment: f64, top: f64) -> f64 {
        if attachment.is_nan() || top.is_nan() {
            return f64::NAN;
        }
        let att = attachment.max(0.0);
        if top <= att {
            return 0.0;
        }
        (self.simulated_lev(top) - self.simulated_lev(att)).max(0.0)
    }
}

fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut bits = *state;
    bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    bits ^= bits >> 31;
    ((bits >> 12) as f64 + 0.5) / 4_503_599_627_370_496.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_layer_matches_reference_values() {
        for (sigma, reference) in [(0.55, 78_331.62), (1.00, 61_707.51), (1.55, 43_833.97)] {
            let model = LognormalLayerModel::try_new(100_000.0, sigma).expect("model");
            assert!((model.exact_layer(0.0, 100_000.0) - reference).abs() < 0.03);
        }
    }

    #[test]
    fn layer_width_and_tail_change_relativities() {
        let mut previous_ratio = 1.0;
        for sigma in [0.55, 1.00, 1.55] {
            let model = LognormalLayerModel::try_new(100_000.0, sigma).expect("model");
            let lower = model.exact_layer(0.0, 100_000.0);
            assert_eq!(model.exact_layer(100_000.0, 100_000.0), 0.0);
            assert_eq!(model.exact_layer(100_000.0, 50_000.0), 0.0);
            let mut previous_upper = 0.0;
            for top in (150_000..=750_000).step_by(25_000) {
                let upper = model.exact_layer(100_000.0, f64::from(top));
                assert!(upper > previous_upper);
                assert!(lower + upper <= model.mean());
                previous_upper = upper;
            }
            let ratio = 1.0 + model.exact_layer(100_000.0, 350_000.0) / lower;
            assert!(ratio > previous_ratio);
            previous_ratio = ratio;
        }
    }

    #[test]
    fn fixed_seed_simulation_agrees_with_limited_expectations() {
        for sigma in [0.55, 1.00, 1.55] {
            let model = LognormalLayerModel::try_new(100_000.0, sigma).expect("model");
            let sampler = LognormalLayerSampler::try_new(model, 65_536, 2026).expect("sampler");
            for top in (150_000..=750_000).step_by(50_000) {
                let exact = model.exact_layer(100_000.0, f64::from(top));
                let simulated = sampler.simulated_layer(100_000.0, f64::from(top));
                assert!(
                    (exact - simulated).abs() < 1_000.0,
                    "sigma {sigma}, top {top}: {exact} vs. {simulated}"
                );
            }
        }
    }

    #[test]
    fn exceedance_and_pdf_are_consistent() {
        let model = LognormalLayerModel::try_new(100_000.0, 1.0).expect("model");
        let p_att = model.exceedance_prob(100_000.0);
        let p_top = model.exceedance_prob(350_000.0);
        assert!(p_att > 0.0 && p_att < 1.0);
        assert!(p_top > 0.0 && p_top < p_att);
        assert_eq!(model.exceedance_prob(0.0), 1.0);
        assert_eq!(model.pdf(0.0), 0.0);
        assert!(model.pdf(50_000.0) > 0.0);
        let dx = 1_000.0;
        let area: f64 = (0..2_000)
            .map(|i| model.pdf((f64::from(i) + 0.5) * dx) * dx)
            .sum();
        assert!(area > 0.95 && area <= 1.001, "area={area}");
    }

    #[test]
    fn validates_model_and_sampler_inputs() {
        assert_eq!(
            LognormalLayerModel::try_new(0.0, 1.0),
            Err(ValidationError::NonPositive("ground-up mean"))
        );
        assert_eq!(
            LognormalLayerModel::try_new(100_000.0, -0.5),
            Err(ValidationError::NonPositive("lognormal sigma"))
        );
        let model = LognormalLayerModel::try_new(100_000.0, 1.0).expect("model");
        assert!(LognormalLayerSampler::try_new(model, 1, 42).is_err());
    }
}
