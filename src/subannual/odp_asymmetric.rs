//! Asymmetric Over-Dispersed Poisson (ODP) GLM fit and residual bootstrap
//! (Shapland 2016).
//!
//! Adapts ODP GLM fitting, exact hat-matrix degrees of freedom / leverages,
//! calendar-diagonal alignment, and residual resampling to non-square
//! `n_origins × n_devs` triangles with staggered or partial stub diagonals.
//!
//! For any monotone staircase of observed counts
//! (`observed_counts[0] >= observed_counts[1] >= ... >= observed_counts[n_o - 1] >= 1`),
//! backward asymmetric volume-weighted chain-ladder recursion yields the exact
//! ODP GLM maximum likelihood expectations (`log μ_{i,j} = c + α_i + β_j`).

use crate::bootstrap_odp::{
    DeterministicRng, FastIndexSampler, OdpBootstrapOptions, OdpCellResidual, ProcessDistribution,
    ResidualAdjustment,
};
use crate::ldf::{cdfs_from_ldfs, volume_weighted_ldfs_observed_counts};
use crate::linalg::invert_spd;
use crate::reserve_distribution::{ReserveDistributionSummary, summarize_with_scratch};
use crate::subannual::grid::{AsymmetricAgeGrid, CalendarDiagonalMap, calendar_diagonal_index};
use crate::subannual::reserving::validate_asymmetric_cumulative;
use crate::validate::{MonotonicityPolicy, ValidationError, validate_tail};

/// Asymmetric ODP GLM fit diagnostics, exact hat-matrix leverages, and analytic
/// prediction errors.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AsymmetricOdpFit {
    /// Number of origin periods.
    pub n_origins: usize,
    /// Number of development columns.
    pub n_devs: usize,
    /// Observed development count per origin.
    pub observed_counts: Vec<usize>,
    /// Applied tail factor (`>= 1.0`).
    pub tail: f64,
    /// Volume-weighted asymmetric link ratios (length `n_devs - 1`).
    pub ldfs: Vec<f64>,
    /// Cumulative development factors to ultimate (length `n_devs`).
    pub cdfs: Vec<f64>,
    /// Row-major (`n_origins * n_devs`) fitted incremental expectations `μ_{i,j}`.
    pub fitted_incremental: Vec<f64>,
    /// Per-cell observed/fitted increments, Pearson residuals, and exact hat-matrix leverages.
    pub residuals: Vec<OdpCellResidual>,
    /// Number of active observed cells `N`.
    pub n_observed: usize,
    /// Number of active GLM parameters `p`.
    pub n_params: usize,
    /// Residual degrees of freedom `N - p`.
    pub degrees_of_freedom: usize,
    /// Pearson over-dispersion scale `φ = sum(r^2) / (N - p)`.
    pub phi: f64,
    /// Sum of hat-matrix diagonal leverages (equals `n_params` identically).
    pub leverage_sum: f64,
    /// Latest observed cumulative losses by origin.
    pub latest: Vec<f64>,
    /// Point-estimate reserves by origin.
    pub origin_reserves: Vec<f64>,
    /// Aggregate point-estimate reserve.
    pub total_reserve: f64,
    /// Analytic ODP aggregate process standard error `sqrt(φ · R)`.
    pub analytic_total_process_se: f64,
    /// True calendar-diagonal alignment map for the asymmetric grid.
    pub calendar_map: CalendarDiagonalMap,
}

/// Full output of [`asymmetric_odp_bootstrap`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AsymmetricOdpBootstrapResult {
    /// Underlying asymmetric ODP GLM fit.
    pub fit: AsymmetricOdpFit,
    /// Bootstrap configuration options used.
    pub options: OdpBootstrapOptions,
    /// Base point-estimate total reserve.
    pub base_total_reserve: f64,
    /// Simulated aggregate reserves for successful iterations.
    pub simulated_totals: Vec<f64>,
    /// Simulated reserves per origin: `simulated_by_origin[origin][iter]`.
    pub simulated_by_origin: Vec<Vec<f64>>,
    /// Distribution summary per origin.
    pub origin_summaries: Vec<ReserveDistributionSummary>,
    /// Distribution summary for aggregate portfolio reserve.
    pub summary: ReserveDistributionSummary,
    /// Number of bootstrap draws rejected due to degenerate pseudo-triangles.
    pub failed_iterations: usize,
    /// Arithmetic mean of adjusted residual pool prior to optional centering.
    pub residual_pool_mean_before_centering: f64,
}

/// Fit the Over-Dispersed Poisson GLM (`log μ_{i,j} = c + α_i + β_j`) to an
/// asymmetric `n_origins × n_devs` cumulative triangle (Shapland 2016).
pub fn fit_asymmetric_odp(
    cum: &[f64],
    grid: &AsymmetricAgeGrid,
    tail: f64,
    policy: MonotonicityPolicy,
) -> Result<AsymmetricOdpFit, ValidationError> {
    validate_asymmetric_cumulative(cum, grid, policy)?;
    validate_tail(tail)?;

    let n_o = grid.n_origins;
    let n_d = grid.n_devs;
    let counts = &grid.observed_counts;

    // Require monotone staircase observed counts so backward chain-ladder recursion
    // produces exact ODP GLM sufficient-statistic expectations.
    for i in 1..n_o {
        if counts[i] > counts[i - 1] {
            return Err(ValidationError::InvalidParameter(
                "asymmetric ODP requires non-increasing observed_counts across origins",
            ));
        }
    }
    let max_obs = counts[0];
    if max_obs < 2 {
        return Err(ValidationError::InvalidParameter(
            "asymmetric ODP requires at least 2 observed development columns",
        ));
    }

    let mut ldfs = vec![1.0; n_d - 1];
    volume_weighted_ldfs_observed_counts(cum, n_o, n_d, counts, &mut ldfs);
    for j in (max_obs - 1)..(n_d - 1) {
        if !ldfs[j].is_finite() {
            ldfs[j] = 1.0;
        }
    }
    if ldfs.iter().any(|&f| !f.is_finite() || f < 1.0) {
        return Err(ValidationError::InvalidParameter(
            "asymmetric ODP GLM requires all volume-weighted LDFs >= 1.0",
        ));
    }

    let mut cdfs = vec![0.0; n_d];
    cdfs_from_ldfs(&ldfs, tail, &mut cdfs);

    // Observed incremental losses and latest diagonal.
    let mut obs_inc = vec![f64::NAN; n_o * n_d];
    let mut latest = Vec::with_capacity(n_o);
    for i in 0..n_o {
        let cnt = counts[i];
        latest.push(cum[i * n_d + cnt - 1]);
        obs_inc[i * n_d] = cum[i * n_d];
        for j in 1..cnt {
            obs_inc[i * n_d + j] = cum[i * n_d + j] - cum[i * n_d + j - 1];
        }
    }

    // Exact ODP GLM MLE via backward + forward chain-ladder recursion.
    let mut fitted_cum = vec![0.0; n_o * n_d];
    for i in 0..n_o {
        let d = counts[i] - 1;
        fitted_cum[i * n_d + d] = latest[i];
        for j in (0..d).rev() {
            fitted_cum[i * n_d + j] = fitted_cum[i * n_d + j + 1] / ldfs[j];
        }
        for j in (d + 1)..n_d {
            fitted_cum[i * n_d + j] = fitted_cum[i * n_d + j - 1] * ldfs[j - 1];
        }
    }

    let mut fitted_inc = vec![0.0; n_o * n_d];
    for i in 0..n_o {
        fitted_inc[i * n_d] = fitted_cum[i * n_d];
        for j in 1..n_d {
            fitted_inc[i * n_d + j] = fitted_cum[i * n_d + j] - fitted_cum[i * n_d + j - 1];
        }
    }

    // Identify active rows, columns, and observed cells with strictly positive fitted mass.
    let mut active_origins = vec![false; n_o];
    let mut active_devs = vec![false; n_d];
    let mut active_cells: Vec<(usize, usize)> = Vec::new();
    for i in 0..n_o {
        for j in 0..counts[i] {
            let mu = fitted_inc[i * n_d + j];
            if mu.is_finite() && mu > 0.0 {
                active_origins[i] = true;
                active_devs[j] = true;
                active_cells.push((i, j));
            }
        }
    }

    let mut origin_col = vec![None; n_o];
    let mut dev_col = vec![None; n_d];
    let mut param_count = 1usize;
    let mut first_o = false;
    for i in 0..n_o {
        if active_origins[i] {
            if first_o {
                origin_col[i] = Some(param_count);
                param_count += 1;
            } else {
                first_o = true;
            }
        }
    }
    let mut first_d = false;
    for j in 0..n_d {
        if active_devs[j] {
            if first_d {
                dev_col[j] = Some(param_count);
                param_count += 1;
            } else {
                first_d = true;
            }
        }
    }

    let n_observed = active_cells.len();
    if n_observed <= param_count {
        return Err(ValidationError::InvalidParameter(
            "asymmetric ODP model has non-positive residual degrees of freedom",
        ));
    }
    let degrees_of_freedom = n_observed - param_count;

    // Fisher information matrix X^T W X.
    let mut info = vec![0.0; param_count * param_count];
    for &(i, j) in &active_cells {
        let w = fitted_inc[i * n_d + j];
        let mut cols = [0usize; 3];
        let mut ncols = 1usize;
        if let Some(c) = origin_col[i] {
            cols[ncols] = c;
            ncols += 1;
        }
        if let Some(c) = dev_col[j] {
            cols[ncols] = c;
            ncols += 1;
        }
        for a in 0..ncols {
            for b in 0..ncols {
                info[cols[a] * param_count + cols[b]] += w;
            }
        }
    }

    // Exact inverse M^{-1} in O(p^3), then O(1) 3-sparse leverage evaluation per cell
    // (O(p^3 + N) total instead of O(N p^2)).
    let info_inv = invert_spd(&info, param_count)?;

    let mut residuals = Vec::with_capacity(n_observed);
    let mut pearson_sq_sum = 0.0;
    let mut leverage_sum = 0.0;

    for &(i, j) in &active_cells {
        let obs = obs_inc[i * n_d + j];
        let mu = fitted_inc[i * n_d + j];
        let raw_residual = (obs - mu) / mu.sqrt();
        pearson_sq_sum += raw_residual * raw_residual;

        let mut cols = [0usize; 3];
        let mut ncols = 1usize;
        if let Some(c) = origin_col[i] {
            cols[ncols] = c;
            ncols += 1;
        }
        if let Some(c) = dev_col[j] {
            cols[ncols] = c;
            ncols += 1;
        }
        let mut qf = 0.0;
        for a in 0..ncols {
            let ca = cols[a];
            qf += info_inv[ca * param_count + ca];
            for b in (a + 1)..ncols {
                qf += 2.0 * info_inv[ca * param_count + cols[b]];
            }
        }
        let leverage = (mu * qf).clamp(0.0, 1.0);
        leverage_sum += leverage;
        let saturated = (1.0 - leverage) <= 1e-10;

        residuals.push(OdpCellResidual {
            origin: i,
            dev: j,
            observed_incremental: obs,
            fitted_incremental: mu,
            raw_pearson_residual: raw_residual,
            leverage,
            saturated,
        });
    }

    let phi = pearson_sq_sum / degrees_of_freedom as f64;
    let mut origin_reserves = Vec::with_capacity(n_o);
    for i in 0..n_o {
        let d = counts[i] - 1;
        origin_reserves.push(latest[i] * (cdfs[d] - 1.0));
    }
    let total_reserve: f64 = origin_reserves.iter().sum();
    let analytic_total_process_se = (phi * total_reserve).max(0.0).sqrt();
    let calendar_map = calendar_diagonal_index(grid)?;

    Ok(AsymmetricOdpFit {
        n_origins: n_o,
        n_devs: n_d,
        observed_counts: counts.clone(),
        tail,
        ldfs,
        cdfs,
        fitted_incremental: fitted_inc,
        residuals,
        n_observed,
        n_params: param_count,
        degrees_of_freedom,
        phi,
        leverage_sum,
        latest,
        origin_reserves,
        total_reserve,
        analytic_total_process_se,
        calendar_map,
    })
}

/// Run Shapland (2016) ODP residual bootstrap on an asymmetric `n_origins × n_devs`
/// triangle with partial stub diagonals.
pub fn asymmetric_odp_bootstrap(
    cum: &[f64],
    grid: &AsymmetricAgeGrid,
    tail: f64,
    options: OdpBootstrapOptions,
) -> Result<AsymmetricOdpBootstrapResult, ValidationError> {
    if options.iterations == 0 {
        return Err(ValidationError::InvalidParameter(
            "bootstrap iterations must be >= 1",
        ));
    }
    let fit = fit_asymmetric_odp(cum, grid, tail, options.monotonicity_policy)?;

    // Build adjusted residual pool (mutually exclusive DoF vs. Shapland hat-matrix leverage).
    let dof_scale = (fit.n_observed as f64 / fit.degrees_of_freedom as f64).sqrt();
    let mut pool: Vec<f64> = Vec::with_capacity(fit.residuals.len());
    for cell in &fit.residuals {
        match options.residual_adjustment {
            ResidualAdjustment::DegreesOfFreedom => {
                pool.push(cell.raw_pearson_residual * dof_scale);
            }
            ResidualAdjustment::HatMatrixLeverage => {
                if !cell.saturated {
                    let denom = (1.0 - cell.leverage).sqrt();
                    pool.push(cell.raw_pearson_residual / denom);
                }
            }
        }
    }
    if pool.is_empty() {
        return Err(ValidationError::InvalidParameter(
            "no eligible non-saturated residuals available for asymmetric ODP bootstrap",
        ));
    }

    let pool_mean = pool.iter().sum::<f64>() / pool.len() as f64;
    if options.center_residuals {
        for r in &mut pool {
            *r -= pool_mean;
        }
    }

    let n_o = fit.n_origins;
    let n_d = fit.n_devs;
    let counts = &fit.observed_counts;
    let max_obs = counts[0];
    let n_links = n_d - 1;
    let phi = fit.phi;
    let inv_phi = if phi > 0.0 && phi.is_finite() {
        1.0 / phi
    } else {
        0.0
    };

    // Precompute packed observed sqrt(mu), baseline column sums, and latest diagonal once.
    let total_obs_cells: usize = counts.iter().sum();
    let mut packed_sqrt_mu = Vec::with_capacity(total_obs_cells);
    let mut base_col_den = vec![0.0; n_links];
    let mut base_col_num = vec![0.0; n_links];
    let mut base_latest = vec![0.0; n_o];
    for i in 0..n_o {
        let cnt = counts[i];
        let row_off = i * n_d;
        let mut acc = 0.0;
        for j in 0..cnt {
            let mu = fit.fitted_incremental[row_off + j];
            packed_sqrt_mu.push(if mu > 0.0 { mu.sqrt() } else { 0.0 });
            acc += mu;
            if j + 1 < cnt {
                base_col_den[j] += acc;
            }
            if j > 0 {
                base_col_num[j - 1] += acc;
            }
        }
        base_latest[i] = acc;
    }

    let mut rng = DeterministicRng::from_seed(options.seed);
    let sampler = FastIndexSampler::new(pool.len());
    let mut simulated_totals = Vec::with_capacity(options.iterations);
    let mut simulated_by_origin: Vec<Vec<f64>> = (0..n_o)
        .map(|_| Vec::with_capacity(options.iterations))
        .collect();
    let mut failed_iterations = 0usize;

    let mut cell_deltas = vec![0.0; total_obs_cells];
    let mut delta_den = vec![0.0; n_links];
    let mut delta_num = vec![0.0; n_links];
    let mut latest_pseudo = vec![0.0; n_o];
    let mut pseudo_ldfs = vec![1.0; n_links];
    let mut pos_factors = vec![0.0; n_d];
    let mut neg_factors = vec![0.0; n_d];
    let mut iter_origins = vec![0.0; n_o];
    let active_links = max_obs.saturating_sub(1);
    let tail_inc = tail - 1.0;

    for _ in 0..options.iterations {
        // 1. Batched 4-wide delta resampling + asymmetric volume-weighted LDF sums.
        sampler.fill_weighted_deltas(&pool, &packed_sqrt_mu, &mut rng, &mut cell_deltas);
        delta_den[..active_links].fill(0.0);
        delta_num[..active_links].fill(0.0);
        let mut cell = 0usize;
        for i in 0..n_o {
            let cnt = counts[i];
            let mut delta = 0.0;
            for j in 0..cnt {
                delta += cell_deltas[cell];
                cell += 1;
                if j + 1 < cnt {
                    delta_den[j] += delta;
                }
                if j > 0 {
                    delta_num[j - 1] += delta;
                }
            }
            latest_pseudo[i] = base_latest[i] + delta;
        }

        // 2. Compute pseudo-LDFs on active links; unobserved tail links stay 1.0.
        let mut valid_ldfs = true;
        let mut all_ge_one = true;
        for j in 0..active_links {
            let den = base_col_den[j] + delta_den[j];
            let num = base_col_num[j] + delta_num[j];
            if !(den > 0.0) || !den.is_finite() {
                valid_ldfs = false;
                break;
            }
            let f = num / den;
            if !f.is_finite() || f <= 0.0 {
                valid_ldfs = false;
                break;
            }
            if f < 1.0 {
                all_ge_one = false;
            }
            pseudo_ldfs[j] = f;
        }
        if !valid_ldfs {
            failed_iterations += 1;
            continue;
        }
        for slot in &mut pseudo_ldfs[active_links..n_links] {
            *slot = 1.0;
        }

        // 3. Compound suffix reserve factors right-to-left in O(n_d) and draw
        //    exact origin-level process variance in O(n_o) via Gamma/Poisson additivity.
        pos_factors[n_d - 1] = tail_inc;
        neg_factors[n_d - 1] = 0.0;
        if all_ge_one {
            let mut cdf = tail;
            for j in (0..n_links).rev() {
                cdf *= pseudo_ldfs[j];
                pos_factors[j] = cdf - 1.0;
                neg_factors[j] = 0.0;
            }
        } else {
            for j in (0..n_links).rev() {
                let f = pseudo_ldfs[j];
                let step = f - 1.0;
                if step >= 0.0 {
                    pos_factors[j] = step + f * pos_factors[j + 1];
                    neg_factors[j] = f * neg_factors[j + 1];
                } else {
                    pos_factors[j] = f * pos_factors[j + 1];
                    neg_factors[j] = (-step) + f * neg_factors[j + 1];
                }
            }
        }

        let mut iter_total = 0.0;
        let mut valid_iter = true;
        for i in 0..n_o {
            let d = counts[i] - 1;
            let prev_cum = latest_pseudo[i];
            if !prev_cum.is_finite() {
                valid_iter = false;
                break;
            }
            let (pos_mu, neg_mu) = if prev_cum >= 0.0 {
                (prev_cum * pos_factors[d], prev_cum * neg_factors[d])
            } else {
                (-prev_cum * neg_factors[d], -prev_cum * pos_factors[d])
            };
            let pos_draw = if pos_mu > 0.0 {
                sample_process_fast(pos_mu, phi, inv_phi, options.process_distribution, &mut rng)
            } else {
                0.0
            };
            // Negative future_mu preserves the asymmetric non-reflected convention
            // (`mu <= 0.0` returns `mu` deterministically).
            let res_i = pos_draw - neg_mu;
            if !res_i.is_finite() {
                valid_iter = false;
                break;
            }
            iter_origins[i] = res_i;
            iter_total += res_i;
        }

        if !valid_iter || !iter_total.is_finite() {
            failed_iterations += 1;
            continue;
        }

        simulated_totals.push(iter_total);
        for (origin_vec, &r) in simulated_by_origin.iter_mut().zip(iter_origins.iter()) {
            origin_vec.push(r);
        }
    }

    if simulated_totals.is_empty() {
        return Err(ValidationError::InvalidParameter(
            "all asymmetric ODP bootstrap iterations failed",
        ));
    }

    let mut summary_buf = Vec::with_capacity(simulated_totals.len());
    let summary = summarize_with_scratch(&simulated_totals, &mut summary_buf, true)?;
    let mut origin_summaries = Vec::with_capacity(n_o);
    for vals in &simulated_by_origin {
        origin_summaries.push(summarize_with_scratch(vals, &mut summary_buf, true)?);
    }

    Ok(AsymmetricOdpBootstrapResult {
        base_total_reserve: fit.total_reserve,
        fit,
        options,
        simulated_totals,
        simulated_by_origin,
        origin_summaries,
        summary,
        failed_iterations,
        residual_pool_mean_before_centering: pool_mean,
    })
}

#[inline]
fn sample_process_fast(
    mu: f64,
    phi: f64,
    inv_phi: f64,
    dist: ProcessDistribution,
    rng: &mut DeterministicRng,
) -> f64 {
    if mu <= 0.0 || phi <= 0.0 {
        return mu;
    }
    match dist {
        ProcessDistribution::None => mu,
        ProcessDistribution::Gamma => rng.sample_gamma(mu * inv_phi, phi),
        ProcessDistribution::OverDispersedPoisson => phi * rng.sample_poisson(mu * inv_phi),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bootstrap_odp::fit_odp;
    use crate::datasets::genins;
    use crate::subannual::grid::asymmetric_age_grid;

    #[test]
    fn asymmetric_odp_matches_square_fit_odp_on_genins() {
        let tri = genins();
        let n = tri.n();
        let grid = asymmetric_age_grid(n, n, 12.0, 12.0, 12.0, None).unwrap();
        let sq = fit_odp(&tri.view(), 1.0, MonotonicityPolicy::Strict).unwrap();
        let asym = fit_asymmetric_odp(&tri.data, &grid, 1.0, MonotonicityPolicy::Strict).unwrap();

        assert_eq!(sq.n_observed, asym.n_observed);
        assert_eq!(sq.rank, asym.n_params);
        assert_eq!(sq.degrees_of_freedom, asym.degrees_of_freedom);
        assert!((sq.dispersion - asym.phi).abs() < 1e-8);
        assert!((sq.total_reserve - asym.total_reserve).abs() < 1e-6);
        assert!((asym.leverage_sum - asym.n_params as f64).abs() < 1e-8);
    }

    #[test]
    fn asymmetric_odp_bootstrap_runs_on_annual_by_semiannual_with_stub() {
        // 5 annual origins × 10 semiannual devs, valued at 6m stub -> counts = [9, 7, 5, 3, 1].
        let grid = asymmetric_age_grid(5, 10, 12.0, 6.0, 6.0, Some(6.0)).unwrap();
        assert_eq!(grid.observed_counts, vec![9, 7, 5, 3, 1]);

        let true_ldfs = [1.55, 1.32, 1.18, 1.11, 1.07, 1.04, 1.025, 1.015, 1.01];
        let mut cum = vec![f64::NAN; 5 * 10];
        for i in 0..5 {
            let mut val = 500.0 + (i as f64) * 60.0;
            for j in 0..grid.observed_counts[i] {
                if j > 0 {
                    // Add deterministic small noise so phi > 0.
                    let jitter = 1.0 + 0.015 * (((i * 7 + j * 13) % 5) as f64 - 2.0);
                    val *= (true_ldfs[j - 1] * jitter).max(1.001);
                }
                cum[i * 10 + j] = val;
            }
        }

        let opts = OdpBootstrapOptions::shapland(250, 12345);
        let boot = asymmetric_odp_bootstrap(&cum, &grid, 1.0, opts).unwrap();
        assert!((boot.fit.leverage_sum - boot.fit.n_params as f64).abs() < 1e-8);
        assert_eq!(boot.simulated_totals.len() + boot.failed_iterations, 250);
        assert!(boot.summary.mean > 0.0);
        assert!(boot.summary.std_dev > 0.0);
    }
}
