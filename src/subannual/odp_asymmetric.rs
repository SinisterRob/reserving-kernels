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
    DeterministicRng, OdpBootstrapOptions, OdpCellResidual, ProcessDistribution,
    ResidualAdjustment,
};
use crate::ldf::{cdfs_from_ldfs, volume_weighted_ldfs_observed_counts};
use crate::linalg::{cholesky_lower, solve_lower_into};
use crate::reserve_distribution::{ReserveDistributionSummary, summarize as summarize_distribution};
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
            if !first_o {
                first_o = true;
            } else {
                origin_col[i] = Some(param_count);
                param_count += 1;
            }
        }
    }
    let mut first_d = false;
    for j in 0..n_d {
        if active_devs[j] {
            if !first_d {
                first_d = true;
            } else {
                dev_col[j] = Some(param_count);
                param_count += 1;
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

    let chol = cholesky_lower(&info, param_count)?;

    let mut residuals = Vec::with_capacity(n_observed);
    let mut pearson_sq_sum = 0.0;
    let mut leverage_sum = 0.0;
    let mut x_vec = vec![0.0; param_count];
    let mut y_buf = vec![0.0; param_count];

    for &(i, j) in &active_cells {
        let obs = obs_inc[i * n_d + j];
        let mu = fitted_inc[i * n_d + j];
        let raw_residual = (obs - mu) / mu.sqrt();
        pearson_sq_sum += raw_residual * raw_residual;

        x_vec.fill(0.0);
        x_vec[0] = 1.0;
        if let Some(c) = origin_col[i] {
            x_vec[c] = 1.0;
        }
        if let Some(c) = dev_col[j] {
            x_vec[c] = 1.0;
        }
        solve_lower_into(&chol, param_count, &x_vec, &mut y_buf);
        let leverage = (mu * y_buf.iter().map(|v| v * v).sum::<f64>()).clamp(0.0, 1.0);
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

    let mut rng = DeterministicRng::from_seed(options.seed);
    let mut simulated_totals = Vec::with_capacity(options.iterations);
    let mut simulated_by_origin = vec![Vec::with_capacity(options.iterations); n_o];
    let mut failed_iterations = 0usize;

    let mut pseudo_cum = vec![f64::NAN; n_o * n_d];
    let mut pseudo_ldfs = vec![1.0; n_links];

    for _ in 0..options.iterations {
        // 1. Resample pseudo-observed cumulative triangle on the asymmetric observed region.
        for i in 0..n_o {
            let mut acc = 0.0;
            for j in 0..counts[i] {
                let mu = fit.fitted_incremental[i * n_d + j];
                let r = pool[rng.next_index(pool.len())];
                let q_star = mu + r * mu.sqrt();
                acc += q_star;
                pseudo_cum[i * n_d + j] = acc;
            }
        }

        // 2. Refit asymmetric volume-weighted LDFs on the pseudo-triangle.
        volume_weighted_ldfs_observed_counts(&pseudo_cum, n_o, n_d, counts, &mut pseudo_ldfs);
        for j in (max_obs - 1)..n_links {
            if !pseudo_ldfs[j].is_finite() {
                pseudo_ldfs[j] = 1.0;
            }
        }
        if pseudo_ldfs.iter().any(|&f| !f.is_finite() || f <= 0.0) {
            failed_iterations += 1;
            continue;
        }

        // 3. Project future cells (unobserved region j >= counts[i]) + optional tail
        //    and apply process variance.
        let mut iter_total = 0.0;
        let mut iter_origins = vec![0.0; n_o];
        let mut valid_iter = true;

        for i in 0..n_o {
            let d = counts[i] - 1;
            let mut prev_cum = pseudo_cum[i * n_d + d];
            if !prev_cum.is_finite() {
                valid_iter = false;
                break;
            }
            let mut res_i = 0.0;
            for j in (d + 1)..n_d {
                let next_cum = prev_cum * pseudo_ldfs[j - 1];
                let mu_fut = next_cum - prev_cum;
                prev_cum = next_cum;
                res_i += sample_process(mu_fut, phi, options.process_distribution, &mut rng);
            }
            if tail > 1.0 {
                let tail_mu = prev_cum * (tail - 1.0);
                res_i += sample_process(tail_mu, phi, options.process_distribution, &mut rng);
            }
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
        for i in 0..n_o {
            simulated_by_origin[i].push(iter_origins[i]);
        }
    }

    if simulated_totals.is_empty() {
        return Err(ValidationError::InvalidParameter(
            "all asymmetric ODP bootstrap iterations failed",
        ));
    }

    let summary = summarize_distribution(&simulated_totals)?;
    let origin_summaries: Vec<ReserveDistributionSummary> = simulated_by_origin
        .iter()
        .map(|vals| summarize_distribution(vals))
        .collect::<Result<Vec<_>, _>>()?;

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
fn sample_process(
    mu: f64,
    phi: f64,
    dist: ProcessDistribution,
    rng: &mut DeterministicRng,
) -> f64 {
    if mu <= 0.0 || phi <= 0.0 {
        return mu;
    }
    match dist {
        ProcessDistribution::None => mu,
        ProcessDistribution::Gamma => rng.sample_gamma(mu / phi, phi),
        ProcessDistribution::OverDispersedPoisson => phi * rng.sample_poisson(mu / phi),
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
