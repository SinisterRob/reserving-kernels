//! Excess-of-loss and limited-loss development factors.
//!
//! Implements:
//!
//! * Pinto & Gogol (1987), *An Analysis of Excess Loss Development*, Proceedings
//!   of the Casualty Actuarial Society (PCAS), Vol. LXXIV, pp. 227–255.
//! * Siewert (1996), *A Model for Reserving Workers Compensation High
//!   Deductibles*, CAS Forum (1996), pp. 217–244.
//!
//! Bridges ground-up loss development ([`crate::deterministic::ldf`]) and parametric severity
//! excess ratios ([`crate::excess::severity`], [`crate::excess::excess_layers`]).
//!
//! Given ground-up cumulative development factors $\text{CDF}_j^{\text{GU}}$ and
//! age-dependent excess ratios $R_j(L) = \frac{E[(X_j - L)_+]}{E\[X_j\]}$ (with
//! ultimate excess ratio $R_\infty(L)$):
//!
//! $$\text{CDF}_j^{\text{xs}}(L) = \text{CDF}_j^{\text{GU}} \cdot \frac{R_\infty(L)}{R_j(L)}$$
//!
//! $$\text{CDF}_j^{\text{lim}}(L) = \text{CDF}_j^{\text{GU}} \cdot \frac{1 - R_\infty(L)}{1 - R_j(L)}$$
//!
//! For any retention $L$ and maturity $j$, these satisfy the exact conservation
//! identity:
//!
//! $$C_j^{\text{lim}}(L) \cdot \text{CDF}_j^{\text{lim}}(L) + C_j^{\text{xs}}(L) \cdot \text{CDF}_j^{\text{xs}}(L) = C_j^{\text{GU}} \cdot \text{CDF}_j^{\text{GU}}$$
//!
//! whenever $C_j^{\text{lim}}(L) = (1 - R_j(L)) C_j^{\text{GU}}$ and
//! $C_j^{\text{xs}}(L) = R_j(L) C_j^{\text{GU}}$.

use crate::excess::severity::SeverityDistribution;
use crate::validate::{ValidationError, validate_positive_finite_slice};

/// Excess and limited development factors derived at a single retention $L$.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExcessDevelopmentFactors {
    /// Retention / limit threshold $L$.
    pub limit: f64,
    /// Ground-up CDFs per development age, length `m`.
    pub ground_up_cdf: Vec<f64>,
    /// Excess ratios $R_j(L) \in (0, 1)$ per development age, length `m`.
    pub excess_ratio: Vec<f64>,
    /// Ultimate excess ratio $R_\infty(L) \in (0, 1)$.
    pub ultimate_excess_ratio: f64,
    /// Excess-of-$L$ CDFs $\text{CDF}_j^{\text{xs}}(L)$, length `m`.
    pub excess_cdf: Vec<f64>,
    /// Limited-to-$L$ CDFs $\text{CDF}_j^{\text{lim}}(L)$, length `m`.
    pub limited_cdf: Vec<f64>,
    /// Implied age-to-age excess LDFs (`excess_cdf[j] / excess_cdf[j+1]`), length `m - 1`.
    pub excess_ldf: Vec<f64>,
    /// Implied age-to-age limited LDFs (`limited_cdf[j] / limited_cdf[j+1]`), length `m - 1`.
    pub limited_ldf: Vec<f64>,
    /// Excess-to-ground-up relativity $R_\infty(L) / R_j(L)$ per age, length `m`.
    pub excess_relativity: Vec<f64>,
}

/// Bounded layer $(A, A + W]$ development factors derived from two excess-ratio schedules.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LayerDevelopmentFactors {
    /// Lower attachment point $A$.
    pub attachment: f64,
    /// Layer width $W > 0$ (exhaustion $E = A + W$).
    pub limit: f64,
    /// Ground-up CDFs per development age, length `m`.
    pub ground_up_cdf: Vec<f64>,
    /// Layer loss share $R_j(A) - R_j(A+W)$ per development age, length `m`.
    pub layer_share: Vec<f64>,
    /// Ultimate layer loss share $R_\infty(A) - R_\infty(A+W)$.
    pub ultimate_layer_share: f64,
    /// Layer CDFs per development age, length `m`.
    pub layer_cdf: Vec<f64>,
    /// Implied age-to-age layer LDFs, length `m - 1`.
    pub layer_ldf: Vec<f64>,
}

/// Projected ground-up, limited, and excess ultimates across origins.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExcessProjection {
    /// Ground-up ultimate per origin.
    pub ground_up_ultimate: Vec<f64>,
    /// Limited-to-$L$ ultimate per origin.
    pub limited_ultimate: Vec<f64>,
    /// Excess-of-$L$ ultimate per origin.
    pub excess_ultimate: Vec<f64>,
    /// Limited-to-$L$ reserve per origin.
    pub limited_reserve: Vec<f64>,
    /// Excess-of-$L$ reserve per origin.
    pub excess_reserve: Vec<f64>,
    /// Maximum conservation discrepancy `|(lim_ult + xs_ult) - gu_ult|` across origins.
    pub max_conservation_error: f64,
}

fn cdfs_to_ldfs(cdfs: &[f64]) -> Vec<f64> {
    if cdfs.len() < 2 {
        return Vec::new();
    }
    (0..cdfs.len() - 1).map(|j| cdfs[j] / cdfs[j + 1]).collect()
}

/// Derive excess-of-$L$ and limited-to-$L$ CDFs and LDFs from ground-up CDFs
/// and age-dependent excess ratios (Siewert 1996 / Pinto & Gogol 1987).
///
/// `ground_up_cdf[j]` is the ground-up CDF at maturity index `j`,
/// `excess_ratios[j]` is $R_j(L) \in (0, 1)$, and `ultimate_excess_ratio` is
/// $R_\infty(L) \in (0, 1)$.
pub fn from_excess_ratios(
    limit: f64,
    ground_up_cdf: &[f64],
    excess_ratios: &[f64],
    ultimate_excess_ratio: f64,
) -> Result<ExcessDevelopmentFactors, ValidationError> {
    if !limit.is_finite() || limit < 0.0 {
        return Err(ValidationError::InvalidParameter(
            "limit must be finite and non-negative",
        ));
    }
    if ground_up_cdf.is_empty() || ground_up_cdf.len() != excess_ratios.len() {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_positive_finite_slice(ground_up_cdf, "ground_up_cdf")?;
    if !ultimate_excess_ratio.is_finite()
        || ultimate_excess_ratio <= 0.0
        || ultimate_excess_ratio >= 1.0
    {
        return Err(ValidationError::InvalidParameter(
            "ultimate_excess_ratio must be strictly in (0, 1)",
        ));
    }
    for &r in excess_ratios {
        if !r.is_finite() || r <= 0.0 || r >= 1.0 {
            return Err(ValidationError::InvalidParameter(
                "each excess ratio must be strictly in (0, 1)",
            ));
        }
    }

    let m = ground_up_cdf.len();
    let mut excess_cdf = Vec::with_capacity(m);
    let mut limited_cdf = Vec::with_capacity(m);
    let mut excess_relativity = Vec::with_capacity(m);

    for j in 0..m {
        let rel = ultimate_excess_ratio / excess_ratios[j];
        let lim_rel = (1.0 - ultimate_excess_ratio) / (1.0 - excess_ratios[j]);
        excess_relativity.push(rel);
        excess_cdf.push(ground_up_cdf[j] * rel);
        limited_cdf.push(ground_up_cdf[j] * lim_rel);
    }

    let excess_ldf = cdfs_to_ldfs(&excess_cdf);
    let limited_ldf = cdfs_to_ldfs(&limited_cdf);

    Ok(ExcessDevelopmentFactors {
        limit,
        ground_up_cdf: ground_up_cdf.to_vec(),
        excess_ratio: excess_ratios.to_vec(),
        ultimate_excess_ratio,
        excess_cdf,
        limited_cdf,
        excess_ldf,
        limited_ldf,
        excess_relativity,
    })
}

/// Derive excess-of-$L$ and limited-to-$L$ development factors directly from
/// age-indexed parametric [`SeverityDistribution`] models.
///
/// Computes $R_j(L) = \text{dist}_j.\text{excess\_ratio}(L)$ and
/// $R_\infty(L) = \text{ultimate\_dist}.\text{excess\_ratio}(L)$.
pub fn from_severity_schedule(
    limit: f64,
    ground_up_cdf: &[f64],
    age_severities: &[SeverityDistribution],
    ultimate_severity: &SeverityDistribution,
) -> Result<ExcessDevelopmentFactors, ValidationError> {
    if ground_up_cdf.len() != age_severities.len() {
        return Err(ValidationError::ShapeMismatch);
    }
    let mut ratios = Vec::with_capacity(age_severities.len());
    for dist in age_severities {
        ratios.push(dist.excess_ratio(limit)?);
    }
    let ult_ratio = ultimate_severity.excess_ratio(limit)?;
    from_excess_ratios(limit, ground_up_cdf, &ratios, ult_ratio)
}

/// Derive development factors for a bounded excess layer $(A, A + W]$ from
/// age-dependent parametric [`SeverityDistribution`] models.
pub fn layer_from_severity_schedule(
    attachment: f64,
    limit: f64,
    ground_up_cdf: &[f64],
    age_severities: &[SeverityDistribution],
    ultimate_severity: &SeverityDistribution,
) -> Result<LayerDevelopmentFactors, ValidationError> {
    if !attachment.is_finite() || attachment < 0.0 || !limit.is_finite() || limit <= 0.0 {
        return Err(ValidationError::InvalidParameter(
            "attachment must be >= 0 and limit must be > 0",
        ));
    }
    if ground_up_cdf.is_empty() || ground_up_cdf.len() != age_severities.len() {
        return Err(ValidationError::ShapeMismatch);
    }
    validate_positive_finite_slice(ground_up_cdf, "ground_up_cdf")?;

    if !(attachment + limit).is_finite() {
        return Err(ValidationError::NotFinite);
    }
    // Layer shares R(A) - R(A + W) are evaluated directly as E[layer] / E[X]
    // (no difference of two nearly equal excess ratios for high layers).
    let ult_share = ultimate_severity.layer_ratio(attachment, limit)?;
    if !ult_share.is_finite() || ult_share <= 0.0 {
        return Err(ValidationError::InvalidParameter(
            "ultimate layer share must be strictly positive",
        ));
    }

    let mut layer_share = Vec::with_capacity(age_severities.len());
    let mut layer_cdf = Vec::with_capacity(age_severities.len());
    for (j, dist) in age_severities.iter().enumerate() {
        let share = dist.layer_ratio(attachment, limit)?;
        if !share.is_finite() || share <= 0.0 {
            return Err(ValidationError::InvalidParameter(
                "age layer share must be strictly positive",
            ));
        }
        layer_share.push(share);
        layer_cdf.push(ground_up_cdf[j] * (ult_share / share));
    }
    let layer_ldf = cdfs_to_ldfs(&layer_cdf);

    Ok(LayerDevelopmentFactors {
        attachment,
        limit,
        ground_up_cdf: ground_up_cdf.to_vec(),
        layer_share,
        ultimate_layer_share: ult_share,
        layer_cdf,
        layer_ldf,
    })
}

/// Project limited and excess ultimates from origin-level latest limited and
/// excess losses using [`ExcessDevelopmentFactors`].
///
/// `latest_limited[i]` and `latest_excess[i]` correspond to origin `i` whose
/// maturity index is `n - 1 - i` (standard square-triangle convention where
/// origin 0 is the oldest / most mature).
pub fn project_split_losses(
    latest_limited: &[f64],
    latest_excess: &[f64],
    factors: &ExcessDevelopmentFactors,
) -> Result<ExcessProjection, ValidationError> {
    let n = latest_limited.len();
    if n == 0
        || latest_excess.len() != n
        || factors.ground_up_cdf.len() != n
        || factors.limited_cdf.len() != n
        || factors.excess_cdf.len() != n
    {
        return Err(ValidationError::ShapeMismatch);
    }
    if factors.ground_up_cdf.iter().any(|v| !v.is_finite())
        || factors.limited_cdf.iter().any(|v| !v.is_finite())
        || factors.excess_cdf.iter().any(|v| !v.is_finite())
    {
        return Err(ValidationError::NotFinite);
    }

    let mut ground_up_ultimate = Vec::with_capacity(n);
    let mut limited_ultimate = Vec::with_capacity(n);
    let mut excess_ultimate = Vec::with_capacity(n);
    let mut limited_reserve = Vec::with_capacity(n);
    let mut excess_reserve = Vec::with_capacity(n);
    let mut max_err = 0.0_f64;

    for i in 0..n {
        let age = n - 1 - i;
        let lim_lat = latest_limited[i];
        let xs_lat = latest_excess[i];
        if !lim_lat.is_finite() || !xs_lat.is_finite() || lim_lat < 0.0 || xs_lat < 0.0 {
            return Err(ValidationError::InvalidParameter(
                "latest losses must be finite and non-negative",
            ));
        }
        let gu_lat = lim_lat + xs_lat;
        let gu_ult = gu_lat * factors.ground_up_cdf[age];
        let lim_ult = lim_lat * factors.limited_cdf[age];
        let xs_ult = xs_lat * factors.excess_cdf[age];
        max_err = max_err.max(((lim_ult + xs_ult) - gu_ult).abs());

        ground_up_ultimate.push(gu_ult);
        limited_ultimate.push(lim_ult);
        excess_ultimate.push(xs_ult);
        limited_reserve.push(lim_ult - lim_lat);
        excess_reserve.push(xs_ult - xs_lat);
    }

    Ok(ExcessProjection {
        ground_up_ultimate,
        limited_ultimate,
        excess_ultimate,
        limited_reserve,
        excess_reserve,
        max_conservation_error: max_err,
    })
}

/// Linear development-portion leverage scaling of ground-up LDFs to excess
/// LDFs, in the spirit of Pinto & Gogol (1987):
///
/// $$\text{LDF}_j^{\text{xs}} = 1 + (f_j^{\text{GU}} - 1) \cdot \text{leverage}_j$$
///
/// for ground-up LDFs $f_j^{\text{GU}} \ge 1$ and caller-supplied,
/// age-specific leverage multipliers $\text{leverage}_j \ge 0$ (`> 1` means the
/// excess layer develops proportionally faster than ground-up). The scaling is
/// **linear in the development portion** `f − 1`; it is *not* a power law
/// `f^γ` despite the historical function name, which is retained for API
/// compatibility. Returns the `m + 1` excess CDFs accumulated backward from
/// `excess_tail` (`cdfs[m] = excess_tail`).
pub fn pinto_gogol_power_scale(
    ground_up_ldf: &[f64],
    leverage: &[f64],
    excess_tail: f64,
) -> Result<Vec<f64>, ValidationError> {
    if ground_up_ldf.len() != leverage.len() {
        return Err(ValidationError::ShapeMismatch);
    }
    if !excess_tail.is_finite() || excess_tail < 1.0 {
        return Err(ValidationError::InvalidTail { tail: excess_tail });
    }
    let m = ground_up_ldf.len();
    let mut excess_ldf = Vec::with_capacity(m);
    for j in 0..m {
        let f = ground_up_ldf[j];
        let lev = leverage[j];
        if !f.is_finite() || f < 1.0 || !lev.is_finite() || lev < 0.0 {
            return Err(ValidationError::InvalidParameter(
                "ground_up_ldf must be >= 1 and leverage must be >= 0",
            ));
        }
        excess_ldf.push(1.0 + (f - 1.0) * lev);
    }
    let mut cdfs = vec![excess_tail; m + 1];
    for j in (0..m).rev() {
        cdfs[j] = cdfs[j + 1] * excess_ldf[j];
    }
    Ok(cdfs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_conservation_identity_across_all_maturities() {
        let gu_cdfs = [2.40, 1.65, 1.25, 1.08, 1.00];
        let excess_ratios = [0.15, 0.22, 0.28, 0.32, 0.35];
        let ult_ratio = 0.35;
        let limit = 250_000.0;

        let factors = from_excess_ratios(limit, &gu_cdfs, &excess_ratios, ult_ratio).unwrap();

        // Excess CDFs must exceed ground-up CDFs when excess ratio increases with age.
        for j in 0..4 {
            assert!(factors.excess_cdf[j] > factors.ground_up_cdf[j]);
            assert!(factors.limited_cdf[j] < factors.ground_up_cdf[j]);
        }

        // Exact conservation: Lim_Ult + XS_Ult == GU_Ult at every origin.
        let gu_latest = [1_000_000.0, 950_000.0, 900_000.0, 800_000.0, 700_000.0];
        let mut lim_latest = Vec::new();
        let mut xs_latest = Vec::new();
        for i in 0..5 {
            let age = 4 - i;
            let r = excess_ratios[age];
            lim_latest.push(gu_latest[i] * (1.0 - r));
            xs_latest.push(gu_latest[i] * r);
        }

        let proj = project_split_losses(&lim_latest, &xs_latest, &factors).unwrap();
        assert!(proj.max_conservation_error < 1e-8);
        for i in 0..5 {
            let sum = proj.limited_ultimate[i] + proj.excess_ultimate[i];
            assert!((sum - proj.ground_up_ultimate[i]).abs() < 1e-8);
        }
    }

    #[test]
    fn severity_schedule_and_layer_cdfs_are_consistent() {
        let gu_cdfs = [2.0, 1.4, 1.1, 1.0];
        // Severity grows with maturity (typical WC/liability claim-size emergence).
        let dists = [
            SeverityDistribution::Lognormal {
                mu: 9.8,
                sigma: 1.2,
            },
            SeverityDistribution::Lognormal {
                mu: 10.1,
                sigma: 1.25,
            },
            SeverityDistribution::Lognormal {
                mu: 10.3,
                sigma: 1.28,
            },
            SeverityDistribution::Lognormal {
                mu: 10.4,
                sigma: 1.3,
            },
        ];
        let ult = &dists[3];

        let xs = from_severity_schedule(100_000.0, &gu_cdfs, &dists, ult).unwrap();
        assert!(xs.excess_cdf[0] > gu_cdfs[0]);
        assert!((xs.excess_cdf[3] - 1.0).abs() < 1e-12);

        let layer =
            layer_from_severity_schedule(100_000.0, 400_000.0, &gu_cdfs, &dists, ult).unwrap();
        assert!(layer.layer_cdf[0] > gu_cdfs[0]);
        assert!((layer.layer_cdf[3] - 1.0).abs() < 1e-12);
    }
}
