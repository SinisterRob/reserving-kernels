#![cfg(all(target_os = "windows", target_arch = "x86_64"))]

mod excel;

use std::panic::catch_unwind;

use excel::{XLERR_NUM, XLERR_VALUE, Xloper12};
use reserving_kernels::Triangle;
use reserving_kernels::deterministic::chain_ladder::ChainLadderEngine;
use reserving_kernels::deterministic::ldf::DevelopmentFactorEngine;
use reserving_kernels::stochastic::mack;
use reserving_kernels::validate::MonotonicityPolicy;

type Projection<const COLS: usize> = fn(&Triangle, f64) -> Result<Vec<[f64; COLS]>, i32>;

#[unsafe(no_mangle)]
unsafe extern "system" fn rk_xll_chain_ladder(
    triangle: *const Xloper12,
    tail: *const Xloper12,
) -> *mut Xloper12 {
    // SAFETY: Excel registers both arguments as borrowed XLOPER12 values (type Q).
    unsafe { worksheet_function(triangle, tail, chain_ladder) }
}

#[unsafe(no_mangle)]
unsafe extern "system" fn rk_xll_mack(
    triangle: *const Xloper12,
    tail: *const Xloper12,
) -> *mut Xloper12 {
    // SAFETY: Excel registers both arguments as borrowed XLOPER12 values (type Q).
    unsafe { worksheet_function(triangle, tail, mack) }
}

unsafe fn worksheet_function<const COLS: usize>(
    triangle: *const Xloper12,
    tail: *const Xloper12,
    project: Projection<COLS>,
) -> *mut Xloper12 {
    let projected = catch_unwind(|| {
        // SAFETY: Both arguments and any Excel array elements live for the duration of this call.
        let (triangle, tail) = unsafe { (excel::triangle(triangle)?, excel::tail_factor(tail)?) };
        let rows = project(&triangle, tail)?;
        if !rows.iter().flatten().all(|value| value.is_finite()) {
            return Err(XLERR_NUM);
        }
        Ok(rows)
    });
    match projected {
        Ok(Ok(rows)) => Xloper12::return_array(rows),
        Ok(Err(error)) => Xloper12::return_error(error),
        Err(_) => Xloper12::return_error(XLERR_VALUE),
    }
}

fn chain_ladder(triangle: &Triangle, tail: f64) -> Result<Vec<[f64; 2]>, i32> {
    let triangle = triangle.view();
    let policy = MonotonicityPolicy::AllowNegativeIncrements;
    let factors = DevelopmentFactorEngine::volume_weighted_with_policy(&triangle, tail, policy)
        .map_err(|_| XLERR_NUM)?;
    let projection = ChainLadderEngine::project_with_policy(&triangle, &factors, policy)
        .map_err(|_| XLERR_NUM)?;
    let mut rows: Vec<_> = projection
        .ultimates
        .into_iter()
        .zip(projection.reserves)
        .map(|(ultimate, reserve)| [ultimate, reserve])
        .collect();
    rows.push([projection.total_ultimate, projection.total_reserve]);
    Ok(rows)
}

fn mack(triangle: &Triangle, tail: f64) -> Result<Vec<[f64; 3]>, i32> {
    let projection = mack::analyze(&triangle.view(), tail).map_err(|_| XLERR_NUM)?;
    let total_ultimate = projection.ultimate.iter().sum();
    let mut rows: Vec<_> = projection
        .ultimate
        .into_iter()
        .zip(projection.reserve)
        .zip(projection.standard_error)
        .map(|((ultimate, reserve), standard_error)| [ultimate, reserve, standard_error])
        .collect();
    rows.push([
        total_ultimate,
        projection.total_reserve,
        projection.total_standard_error,
    ]);
    Ok(rows)
}
