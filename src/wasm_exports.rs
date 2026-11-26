//! C-ABI exports for the browser WASM demo.
//!
//! Built only with `--features wasm`. All exports take/return `f64` (or
//! `u32` for counts/sizes) so JavaScript can call them directly through
//! `WebAssembly.Instance.exports` with no bindgen glue.
//!
//! Data is exchanged through static buffers:
//!
//! * `rk_tri_ptr`  — cumulative triangle, row-major, `N_MAX × N_MAX`.
//!                   Write `NaN` for unobserved cells.
//! * `rk_prem_ptr` — per-origin premium vector, length `N_MAX`.
//! * `rk_ldf_ptr`  — output LDFs, length `N_MAX − 1` meaningful.
//! * `rk_cdf_ptr`  — output CDFs (cumulative dev factors), length `N_MAX`.
//! * `rk_ult_ptr`  — output ultimates, length `N_MAX`.
//! * `rk_res_ptr`  — output reserves, length `N_MAX`.
//! * `rk_exp_ptr`  — BF input: expected ultimate per origin, length `N_MAX`.
//! * `rk_se_ptr`   — Mack output: standard error per origin, length `N_MAX`.
//! * `rk_sig_ptr`  — Mack output: σ² per development link, length `N_MAX`.
//! * `rk_int_ptr`  — Brosius output: intercept per link.
//! * `rk_slp_ptr`  — Brosius output: slope per link.

#![allow(unsafe_code)]
#![allow(missing_docs)]

use crate::types::TriangleView;
use crate::{bf, brosius, cape_cod, chain_ladder, ldf, mack};

const N_MAX: usize = 20;
const TRI_CAP: usize = N_MAX * N_MAX;

static mut TRI: [f64; TRI_CAP] = [0.0; TRI_CAP];
static mut PREM: [f64; N_MAX] = [0.0; N_MAX];
static mut LDF: [f64; N_MAX] = [0.0; N_MAX];
static mut CDF: [f64; N_MAX] = [0.0; N_MAX];
static mut ULT: [f64; N_MAX] = [0.0; N_MAX];
static mut RES: [f64; N_MAX] = [0.0; N_MAX];
static mut EXP: [f64; N_MAX] = [0.0; N_MAX];
static mut SE: [f64; N_MAX] = [0.0; N_MAX];
static mut SIG: [f64; N_MAX] = [0.0; N_MAX];
static mut INT: [f64; N_MAX] = [0.0; N_MAX];
static mut SLP: [f64; N_MAX] = [0.0; N_MAX];
static mut MACK_TOTAL_RESERVE: f64 = 0.0;

fn clamp_n(n: u32) -> usize {
    (n as usize).min(N_MAX)
}

unsafe fn read_buf(ptr: *const f64, len: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push(unsafe { *ptr.add(i) });
    }
    out
}

unsafe fn write_buf(dst: *mut f64, src: &[f64]) {
    let take = src.len().min(N_MAX);
    for i in 0..take {
        unsafe { *dst.add(i) = src[i] };
    }
    for i in take..N_MAX {
        unsafe { *dst.add(i) = f64::NAN };
    }
}

fn valid_tail(tail: f64) -> bool {
    tail.is_finite() && tail >= 1.0
}

fn tri_packed(n: usize) -> Vec<f64> {
    // The TRI buffer is laid out N_MAX × N_MAX row-major so JS can write
    // any cell at a stable stride. Pack into a fresh n × n Vec per call —
    // simple, allocation is cheap and eliminates aliasing concerns.
    let mut out = vec![0.0; n * n];
    unsafe {
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] = TRI[i * N_MAX + j];
            }
        }
    }
    out
}

fn view_of<'a>(data: &'a [f64], n: usize) -> TriangleView<'a> {
    TriangleView {
        n_origins: n,
        n_devs: n,
        data,
    }
}

// ---- Buffer accessors ------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn rk_n_max() -> u32 {
    N_MAX as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn rk_tri_ptr() -> *mut f64 {
    core::ptr::addr_of_mut!(TRI).cast()
}
#[unsafe(no_mangle)]
pub extern "C" fn rk_tri_stride() -> u32 {
    N_MAX as u32
}
#[unsafe(no_mangle)]
pub extern "C" fn rk_prem_ptr() -> *mut f64 {
    core::ptr::addr_of_mut!(PREM).cast()
}
#[unsafe(no_mangle)]
pub extern "C" fn rk_ldf_ptr() -> *mut f64 {
    core::ptr::addr_of_mut!(LDF).cast()
}
#[unsafe(no_mangle)]
pub extern "C" fn rk_cdf_ptr() -> *mut f64 {
    core::ptr::addr_of_mut!(CDF).cast()
}
#[unsafe(no_mangle)]
pub extern "C" fn rk_ult_ptr() -> *mut f64 {
    core::ptr::addr_of_mut!(ULT).cast()
}
#[unsafe(no_mangle)]
pub extern "C" fn rk_res_ptr() -> *mut f64 {
    core::ptr::addr_of_mut!(RES).cast()
}
#[unsafe(no_mangle)]
pub extern "C" fn rk_exp_ptr() -> *mut f64 {
    core::ptr::addr_of_mut!(EXP).cast()
}
#[unsafe(no_mangle)]
pub extern "C" fn rk_se_ptr() -> *mut f64 {
    core::ptr::addr_of_mut!(SE).cast()
}
#[unsafe(no_mangle)]
pub extern "C" fn rk_sig_ptr() -> *mut f64 {
    core::ptr::addr_of_mut!(SIG).cast()
}
#[unsafe(no_mangle)]
pub extern "C" fn rk_int_ptr() -> *mut f64 {
    core::ptr::addr_of_mut!(INT).cast()
}
#[unsafe(no_mangle)]
pub extern "C" fn rk_slp_ptr() -> *mut f64 {
    core::ptr::addr_of_mut!(SLP).cast()
}

// ---- Helpers to write into static output buffers --------------------------

unsafe fn write_ldf(src: &[f64]) {
    unsafe {
        for (i, v) in src.iter().enumerate().take(N_MAX) {
            LDF[i] = *v;
        }
        for i in src.len()..N_MAX {
            LDF[i] = f64::NAN;
        }
    }
}
unsafe fn write_cdf(src: &[f64]) {
    unsafe {
        for (i, v) in src.iter().enumerate().take(N_MAX) {
            CDF[i] = *v;
        }
        for i in src.len()..N_MAX {
            CDF[i] = f64::NAN;
        }
    }
}
unsafe fn write_ult(src: &[f64]) {
    unsafe {
        for (i, v) in src.iter().enumerate().take(N_MAX) {
            ULT[i] = *v;
        }
        for i in src.len()..N_MAX {
            ULT[i] = f64::NAN;
        }
    }
}
unsafe fn write_res(src: &[f64]) {
    unsafe {
        for (i, v) in src.iter().enumerate().take(N_MAX) {
            RES[i] = *v;
        }
        for i in src.len()..N_MAX {
            RES[i] = f64::NAN;
        }
    }
}

// ---- LDF selection methods ------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn rk_ldf_volume_weighted(n: u32) -> u32 {
    let n = clamp_n(n);
    let packed = tri_packed(n);
    let v = ldf::volume_weighted(&view_of(&packed, n));
    unsafe { write_ldf(&v) }
    v.len() as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn rk_ldf_simple_average(n: u32) -> u32 {
    let n = clamp_n(n);
    let packed = tri_packed(n);
    let v = ldf::simple_average(&view_of(&packed, n));
    unsafe { write_ldf(&v) }
    v.len() as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn rk_ldf_ex_hi_lo(n: u32) -> u32 {
    let n = clamp_n(n);
    let packed = tri_packed(n);
    let v = ldf::ex_hi_lo(&view_of(&packed, n));
    unsafe { write_ldf(&v) }
    v.len() as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn rk_ldf_volume_weighted_last_n(n: u32, last_n: u32) -> u32 {
    let n = clamp_n(n);
    let packed = tri_packed(n);
    let v = ldf::volume_weighted_last_n(&view_of(&packed, n), last_n as usize);
    unsafe { write_ldf(&v) }
    v.len() as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn rk_ldf_simple_average_last_n(n: u32, last_n: u32) -> u32 {
    let n = clamp_n(n);
    let packed = tri_packed(n);
    let v = ldf::simple_average_last_n(&view_of(&packed, n), last_n as usize);
    unsafe { write_ldf(&v) }
    v.len() as u32
}

/// Compute CDFs from the currently-loaded LDF buffer. Writes into the CDF
/// buffer. Returns the count written (`n`).
#[unsafe(no_mangle)]
pub extern "C" fn rk_cdf_compute(n: u32, tail: f64) -> u32 {
    let n = clamp_n(n);
    if n == 0 || !valid_tail(tail) {
        return 0;
    }
    if n < 2 {
        unsafe { write_cdf(&[tail]) };
        return 1;
    }
    let ldf_slice = unsafe { read_buf(core::ptr::addr_of!(LDF).cast(), n - 1) };
    let cdf = ldf::cdf_from_ldf(&ldf_slice, tail);
    unsafe { write_cdf(&cdf) }
    cdf.len() as u32
}

// ---- Chain-ladder projection (reads LDF buffer) ---------------------------

#[unsafe(no_mangle)]
pub extern "C" fn rk_chain_ladder(n: u32, tail: f64) -> u32 {
    let n = clamp_n(n);
    if n == 0 || !valid_tail(tail) {
        return 0;
    }
    let ldf_slice: Vec<f64> = if n >= 2 {
        unsafe { read_buf(core::ptr::addr_of!(LDF).cast(), n - 1) }
    } else {
        Vec::new()
    };
    let packed = tri_packed(n);
    let cl = chain_ladder::project_unchecked(&view_of(&packed, n), &ldf_slice, tail);
    let cdf = ldf::cdf_from_ldf(&ldf_slice, tail);
    unsafe {
        write_ult(&cl.ultimates);
        write_res(&cl.reserves);
        write_cdf(&cdf);
    }
    n as u32
}

// ---- Bornhuetter–Ferguson (reads LDF + EXP buffers) -----------------------

#[unsafe(no_mangle)]
pub extern "C" fn rk_bf(n: u32, tail: f64) -> u32 {
    let n = clamp_n(n);
    if n == 0 || !valid_tail(tail) {
        return 0;
    }
    let ldf_slice: Vec<f64> = if n >= 2 {
        unsafe { read_buf(core::ptr::addr_of!(LDF).cast(), n - 1) }
    } else {
        Vec::new()
    };
    let exp_slice = unsafe { read_buf(core::ptr::addr_of!(EXP).cast(), n) };
    let packed = tri_packed(n);
    let r = bf::project_unchecked(&view_of(&packed, n), &ldf_slice, tail, &exp_slice);
    let cdf = ldf::cdf_from_ldf(&ldf_slice, tail);
    unsafe {
        write_ult(&r.ultimates);
        write_res(&r.reserves);
        write_cdf(&cdf);
    }
    n as u32
}

/// Fill the EXP buffer from `premium × elr`.
#[unsafe(no_mangle)]
pub extern "C" fn rk_bf_expected_from_premium(n: u32, elr: f64) -> u32 {
    let n = clamp_n(n);
    let prem = unsafe { read_buf(core::ptr::addr_of!(PREM).cast(), n) };
    let exp: Vec<f64> = prem.into_iter().map(|p| p * elr).collect();
    unsafe { write_buf(core::ptr::addr_of_mut!(EXP).cast(), &exp) };
    n as u32
}

// ---- Cape Cod (reads LDF + PREM buffers) ----------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn rk_cape_cod_elr(n: u32, tail: f64) -> f64 {
    let n = clamp_n(n);
    if n == 0 || !valid_tail(tail) {
        return f64::NAN;
    }
    let ldf_slice: Vec<f64> = if n >= 2 {
        unsafe { read_buf(core::ptr::addr_of!(LDF).cast(), n - 1) }
    } else {
        Vec::new()
    };
    let prem_slice = unsafe { read_buf(core::ptr::addr_of!(PREM).cast(), n) };
    let packed = tri_packed(n);
    cape_cod::elr(&view_of(&packed, n), &ldf_slice, tail, &prem_slice)
}

#[unsafe(no_mangle)]
pub extern "C" fn rk_cape_cod(n: u32, tail: f64) -> f64 {
    let n = clamp_n(n);
    if n == 0 || !valid_tail(tail) {
        return f64::NAN;
    }
    let ldf_slice: Vec<f64> = if n >= 2 {
        unsafe { read_buf(core::ptr::addr_of!(LDF).cast(), n - 1) }
    } else {
        Vec::new()
    };
    let prem_slice = unsafe { read_buf(core::ptr::addr_of!(PREM).cast(), n) };
    let packed = tri_packed(n);
    let (elr, r) = cape_cod::project_unchecked(&view_of(&packed, n), &ldf_slice, tail, &prem_slice);
    let cdf = ldf::cdf_from_ldf(&ldf_slice, tail);
    unsafe {
        write_ult(&r.ultimates);
        write_res(&r.reserves);
        write_cdf(&cdf);
        write_buf(core::ptr::addr_of_mut!(EXP).cast(), &r.expected_ultimate);
    }
    elr
}

// ---- Brosius -------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn rk_brosius(n: u32, tail: f64) -> u32 {
    let n = clamp_n(n);
    if n == 0 || !valid_tail(tail) {
        return 0;
    }
    let packed = tri_packed(n);
    let r = brosius::project_unchecked(&view_of(&packed, n), tail);
    unsafe {
        write_buf(core::ptr::addr_of_mut!(INT).cast(), &r.intercept);
        write_buf(core::ptr::addr_of_mut!(SLP).cast(), &r.slope);
        write_ult(&r.ultimate);
        write_res(&r.reserve);
    }
    n as u32
}

// ---- Mack ----------------------------------------------------------------

/// Runs the Mack analysis. Writes LDF, CDF, ULT, RES, SE, SIG buffers.
/// Returns the total standard error across the portfolio reserve.
#[unsafe(no_mangle)]
pub extern "C" fn rk_mack(n: u32, tail: f64) -> f64 {
    let n = clamp_n(n);
    if n == 0 || !valid_tail(tail) {
        return f64::NAN;
    }
    let packed = tri_packed(n);
    let r = mack::analyze_unchecked(&view_of(&packed, n), tail);
    unsafe {
        write_ldf(&r.ldf);
        write_ult(&r.ultimate);
        write_res(&r.reserve);
        write_buf(core::ptr::addr_of_mut!(SE).cast(), &r.standard_error);
        write_buf(core::ptr::addr_of_mut!(SIG).cast(), &r.sigma_sq);
        let cdf = ldf::cdf_from_ldf(&r.ldf, tail);
        write_cdf(&cdf);
        MACK_TOTAL_RESERVE = r.total_reserve;
    }
    r.total_standard_error
}

/// Returns the total reserve after the last Mack call.
#[unsafe(no_mangle)]
pub extern "C" fn rk_mack_total_reserve(_n: u32) -> f64 {
    unsafe { *core::ptr::addr_of!(MACK_TOTAL_RESERVE) }
}
