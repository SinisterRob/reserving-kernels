//! C-ABI exports for the browser WASM demo.
//!
//! Built only with `--features wasm`. All exports take/return `f64` (or
//! `u32` for counts/sizes) so JavaScript can call them directly through
//! `WebAssembly.Instance.exports` with no bindgen glue.
//!
//! # Safety
//!
//! These exports exchange data through process-global `static mut` scratch
//! buffers and are **strictly for single-threaded `wasm32` hosts**. Do not
//! invoke them concurrently from multiple threads on native targets.
//!
//! Data is exchanged through static buffers:
//!
//! * `rk_tri_ptr`  — cumulative triangle, row-major, `N_MAX × N_MAX`
//!   (write `NaN` for unobserved cells).
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
use crate::{
    deterministic::bf, deterministic::brosius, deterministic::cape_cod,
    deterministic::chain_ladder, deterministic::ldf, stochastic::mack,
};

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

#[cfg(not(target_arch = "wasm32"))]
static NATIVE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(not(target_arch = "wasm32"))]
fn with_wasm_guard<T>(f: impl FnOnce() -> T) -> T {
    let _guard = NATIVE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    f()
}

#[cfg(target_arch = "wasm32")]
#[inline]
fn with_wasm_guard<T>(f: impl FnOnce() -> T) -> T {
    f()
}

fn clamp_n(n: u32) -> usize {
    (n as usize).min(N_MAX)
}

unsafe fn read_buf(ptr: *const f64, len: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        // SAFETY: Caller guarantees `ptr` points to at least `len` initialized `f64` elements
        // in a static WASM exchange buffer with exclusive single-threaded access.
        out.push(unsafe { *ptr.add(i) });
    }
    out
}

unsafe fn write_buf(dst: *mut f64, src: &[f64]) {
    let take = src.len().min(N_MAX);
    for i in 0..take {
        // SAFETY: Caller guarantees `dst` points to a static `N_MAX`-element `f64` buffer
        // with exclusive single-threaded access, and `i < take <= N_MAX`.
        unsafe { *dst.add(i) = src[i] };
    }
    for i in take..N_MAX {
        // SAFETY: Same preconditions as above; `i < N_MAX` stays within the static buffer.
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
    let tri_ptr: *const f64 = core::ptr::addr_of!(TRI).cast();
    // SAFETY: WASM execution is single-threaded with no concurrent mutable alias to `TRI`;
    // callers pass `n <= N_MAX` (via `clamp_n`), so `i * N_MAX + j < TRI_CAP`.
    unsafe {
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] = *tri_ptr.add(i * N_MAX + j);
            }
        }
    }
    out
}

fn view_of(data: &[f64], n: usize) -> TriangleView<'_> {
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
    // SAFETY: Single-threaded WASM C-ABI boundary with exclusive access to static `LDF`;
    // `write_buf` bounds all writes by `N_MAX`.
    unsafe { write_buf(core::ptr::addr_of_mut!(LDF).cast(), src) };
}
unsafe fn write_cdf(src: &[f64]) {
    // SAFETY: Single-threaded WASM C-ABI boundary with exclusive access to static `CDF`;
    // `write_buf` bounds all writes by `N_MAX`.
    unsafe { write_buf(core::ptr::addr_of_mut!(CDF).cast(), src) };
}
unsafe fn write_ult(src: &[f64]) {
    // SAFETY: Single-threaded WASM C-ABI boundary with exclusive access to static `ULT`;
    // `write_buf` bounds all writes by `N_MAX`.
    unsafe { write_buf(core::ptr::addr_of_mut!(ULT).cast(), src) };
}
unsafe fn write_res(src: &[f64]) {
    // SAFETY: Single-threaded WASM C-ABI boundary with exclusive access to static `RES`;
    // `write_buf` bounds all writes by `N_MAX`.
    unsafe { write_buf(core::ptr::addr_of_mut!(RES).cast(), src) };
}

// ---- LDF selection methods ------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn rk_ldf_volume_weighted(n: u32) -> u32 {
    std::panic::catch_unwind(|| {
        with_wasm_guard(|| {
            let n = clamp_n(n);
            let packed = tri_packed(n);
            let v = ldf::volume_weighted(&view_of(&packed, n));
            // SAFETY: Guarded exclusive access to static output buffer `LDF`.
            unsafe { write_ldf(&v) }
            v.len() as u32
        })
    })
    .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn rk_ldf_simple_average(n: u32) -> u32 {
    std::panic::catch_unwind(|| {
        with_wasm_guard(|| {
            let n = clamp_n(n);
            let packed = tri_packed(n);
            let v = ldf::simple_average(&view_of(&packed, n));
            // SAFETY: Guarded exclusive access to static output buffer `LDF`.
            unsafe { write_ldf(&v) }
            v.len() as u32
        })
    })
    .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn rk_ldf_ex_hi_lo(n: u32) -> u32 {
    std::panic::catch_unwind(|| {
        with_wasm_guard(|| {
            let n = clamp_n(n);
            let packed = tri_packed(n);
            let v = ldf::ex_hi_lo(&view_of(&packed, n));
            // SAFETY: Guarded exclusive access to static output buffer `LDF`.
            unsafe { write_ldf(&v) }
            v.len() as u32
        })
    })
    .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn rk_ldf_volume_weighted_last_n(n: u32, last_n: u32) -> u32 {
    std::panic::catch_unwind(|| {
        with_wasm_guard(|| {
            let n = clamp_n(n);
            let packed = tri_packed(n);
            let v = ldf::volume_weighted_last_n(&view_of(&packed, n), last_n as usize);
            // SAFETY: Guarded exclusive access to static output buffer `LDF`.
            unsafe { write_ldf(&v) }
            v.len() as u32
        })
    })
    .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn rk_ldf_simple_average_last_n(n: u32, last_n: u32) -> u32 {
    std::panic::catch_unwind(|| {
        with_wasm_guard(|| {
            let n = clamp_n(n);
            let packed = tri_packed(n);
            let v = ldf::simple_average_last_n(&view_of(&packed, n), last_n as usize);
            // SAFETY: Guarded exclusive access to static output buffer `LDF`.
            unsafe { write_ldf(&v) }
            v.len() as u32
        })
    })
    .unwrap_or(0)
}

/// Compute CDFs from the currently-loaded LDF buffer. Writes into the CDF
/// buffer. Returns the count written (`n`).
#[unsafe(no_mangle)]
pub extern "C" fn rk_cdf_compute(n: u32, tail: f64) -> u32 {
    std::panic::catch_unwind(|| {
        with_wasm_guard(|| {
            let n = clamp_n(n);
            if n == 0 || !valid_tail(tail) {
                // SAFETY: Guarded exclusive write; clears stale CDF output on invalid input.
                unsafe { write_cdf(&[]) };
                return 0;
            }
            if n < 2 {
                // SAFETY: Guarded exclusive access to static output buffer `CDF`.
                unsafe { write_cdf(&[tail]) };
                return 1;
            }
            // SAFETY: `LDF` has length `N_MAX` and `n - 1 < N_MAX`.
            let ldf_slice = unsafe { read_buf(core::ptr::addr_of!(LDF).cast(), n - 1) };
            let cdf = ldf::cdf_from_ldf(&ldf_slice, tail);
            // SAFETY: Guarded exclusive access to static output buffer `CDF`.
            unsafe { write_cdf(&cdf) }
            cdf.len() as u32
        })
    })
    .unwrap_or(0)
}

// ---- Chain-ladder projection (reads LDF buffer) ---------------------------

#[unsafe(no_mangle)]
pub extern "C" fn rk_chain_ladder(n: u32, tail: f64) -> u32 {
    std::panic::catch_unwind(|| {
        with_wasm_guard(|| {
            let n = clamp_n(n);
            if n == 0 || !valid_tail(tail) {
                // SAFETY: Guarded exclusive write; clears stale outputs on invalid input.
                unsafe {
                    write_ult(&[]);
                    write_res(&[]);
                    write_cdf(&[]);
                }
                return 0;
            }
            let ldf_slice: Vec<f64> = if n >= 2 {
                // SAFETY: `LDF` has length `N_MAX` and `n - 1 < N_MAX`.
                unsafe { read_buf(core::ptr::addr_of!(LDF).cast(), n - 1) }
            } else {
                Vec::new()
            };
            let packed = tri_packed(n);
            let cl = chain_ladder::project_unchecked(&view_of(&packed, n), &ldf_slice, tail);
            let cdf = ldf::cdf_from_ldf(&ldf_slice, tail);
            // SAFETY: Guarded exclusive access to static output buffers.
            unsafe {
                write_ult(&cl.ultimates);
                write_res(&cl.reserves);
                write_cdf(&cdf);
            }
            n as u32
        })
    })
    .unwrap_or(0)
}

// ---- Bornhuetter–Ferguson (reads LDF + EXP buffers) -----------------------

#[unsafe(no_mangle)]
pub extern "C" fn rk_bf(n: u32, tail: f64) -> u32 {
    std::panic::catch_unwind(|| {
        with_wasm_guard(|| {
            let n = clamp_n(n);
            if n == 0 || !valid_tail(tail) {
                // SAFETY: Guarded exclusive write; clears stale outputs on invalid input.
                unsafe {
                    write_ult(&[]);
                    write_res(&[]);
                    write_cdf(&[]);
                }
                return 0;
            }
            let ldf_slice: Vec<f64> = if n >= 2 {
                // SAFETY: `LDF` has length `N_MAX` and `n - 1 < N_MAX`.
                unsafe { read_buf(core::ptr::addr_of!(LDF).cast(), n - 1) }
            } else {
                Vec::new()
            };
            // SAFETY: `EXP` has length `N_MAX` and `n <= N_MAX`.
            let exp_slice = unsafe { read_buf(core::ptr::addr_of!(EXP).cast(), n) };
            let packed = tri_packed(n);
            let r = bf::project_unchecked(&view_of(&packed, n), &ldf_slice, tail, &exp_slice);
            let cdf = ldf::cdf_from_ldf(&ldf_slice, tail);
            // SAFETY: Guarded exclusive access to static output buffers.
            unsafe {
                write_ult(&r.ultimates);
                write_res(&r.reserves);
                write_cdf(&cdf);
            }
            n as u32
        })
    })
    .unwrap_or(0)
}

/// Fill the EXP buffer from `premium × elr`.
#[unsafe(no_mangle)]
pub extern "C" fn rk_bf_expected_from_premium(n: u32, elr: f64) -> u32 {
    std::panic::catch_unwind(|| {
        with_wasm_guard(|| {
            let n = clamp_n(n);
            if !elr.is_finite() || elr < 0.0 {
                // SAFETY: Guarded exclusive write; clears stale EXP output on invalid input.
                unsafe { write_buf(core::ptr::addr_of_mut!(EXP).cast(), &[]) };
                return 0;
            }
            // SAFETY: `PREM` has length `N_MAX` and `n <= N_MAX`.
            let prem = unsafe { read_buf(core::ptr::addr_of!(PREM).cast(), n) };
            let exp: Vec<f64> = prem.into_iter().map(|p| p * elr).collect();
            // SAFETY: `EXP` is a static `N_MAX`-element buffer under exclusive guard.
            unsafe { write_buf(core::ptr::addr_of_mut!(EXP).cast(), &exp) };
            n as u32
        })
    })
    .unwrap_or(0)
}

// ---- Cape Cod (reads LDF + PREM buffers) ----------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn rk_cape_cod_elr(n: u32, tail: f64) -> f64 {
    std::panic::catch_unwind(|| {
        with_wasm_guard(|| {
            let n = clamp_n(n);
            if n == 0 || !valid_tail(tail) {
                return f64::NAN;
            }
            let ldf_slice: Vec<f64> = if n >= 2 {
                // SAFETY: `LDF` has length `N_MAX` and `n - 1 < N_MAX`.
                unsafe { read_buf(core::ptr::addr_of!(LDF).cast(), n - 1) }
            } else {
                Vec::new()
            };
            // SAFETY: `PREM` has length `N_MAX` and `n <= N_MAX`.
            let prem_slice = unsafe { read_buf(core::ptr::addr_of!(PREM).cast(), n) };
            let packed = tri_packed(n);
            cape_cod::elr(&view_of(&packed, n), &ldf_slice, tail, &prem_slice)
        })
    })
    .unwrap_or(f64::NAN)
}

#[unsafe(no_mangle)]
pub extern "C" fn rk_cape_cod(n: u32, tail: f64) -> f64 {
    std::panic::catch_unwind(|| {
        with_wasm_guard(|| {
            let n = clamp_n(n);
            if n == 0 || !valid_tail(tail) {
                // SAFETY: Guarded exclusive write; clears stale outputs on invalid input.
                unsafe {
                    write_ult(&[]);
                    write_res(&[]);
                    write_cdf(&[]);
                    write_buf(core::ptr::addr_of_mut!(EXP).cast(), &[]);
                }
                return f64::NAN;
            }
            let ldf_slice: Vec<f64> = if n >= 2 {
                // SAFETY: `LDF` has length `N_MAX` and `n - 1 < N_MAX`.
                unsafe { read_buf(core::ptr::addr_of!(LDF).cast(), n - 1) }
            } else {
                Vec::new()
            };
            // SAFETY: `PREM` has length `N_MAX` and `n <= N_MAX`.
            let prem_slice = unsafe { read_buf(core::ptr::addr_of!(PREM).cast(), n) };
            let packed = tri_packed(n);
            let (elr, r) =
                cape_cod::project_unchecked(&view_of(&packed, n), &ldf_slice, tail, &prem_slice);
            let cdf = ldf::cdf_from_ldf(&ldf_slice, tail);
            // SAFETY: Guarded exclusive access to static output buffers.
            unsafe {
                write_ult(&r.ultimates);
                write_res(&r.reserves);
                write_cdf(&cdf);
                write_buf(core::ptr::addr_of_mut!(EXP).cast(), &r.expected_ultimate);
            }
            elr
        })
    })
    .unwrap_or(f64::NAN)
}

// ---- Brosius -------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn rk_brosius(n: u32, tail: f64) -> u32 {
    std::panic::catch_unwind(|| {
        with_wasm_guard(|| {
            let n = clamp_n(n);
            if n == 0 || !valid_tail(tail) {
                // SAFETY: Guarded exclusive write; clears stale outputs on invalid input.
                unsafe {
                    write_buf(core::ptr::addr_of_mut!(INT).cast(), &[]);
                    write_buf(core::ptr::addr_of_mut!(SLP).cast(), &[]);
                    write_ult(&[]);
                    write_res(&[]);
                }
                return 0;
            }
            let packed = tri_packed(n);
            let r = brosius::project_unchecked(&view_of(&packed, n), tail);
            // SAFETY: Guarded exclusive access to static output buffers.
            unsafe {
                write_buf(core::ptr::addr_of_mut!(INT).cast(), &r.intercept);
                write_buf(core::ptr::addr_of_mut!(SLP).cast(), &r.slope);
                write_ult(&r.ultimate);
                write_res(&r.reserve);
            }
            n as u32
        })
    })
    .unwrap_or(0)
}

// ---- Mack ----------------------------------------------------------------

/// Runs the Mack analysis. Writes LDF, CDF, ULT, RES, SE, SIG buffers.
/// Returns the total standard error across the portfolio reserve.
#[unsafe(no_mangle)]
pub extern "C" fn rk_mack(n: u32, tail: f64) -> f64 {
    std::panic::catch_unwind(|| {
        with_wasm_guard(|| {
            let n = clamp_n(n);
            if n == 0 || !valid_tail(tail) {
                // SAFETY: Guarded exclusive write; clears stale outputs on invalid input.
                unsafe {
                    write_ldf(&[]);
                    write_ult(&[]);
                    write_res(&[]);
                    write_buf(core::ptr::addr_of_mut!(SE).cast(), &[]);
                    write_buf(core::ptr::addr_of_mut!(SIG).cast(), &[]);
                    write_cdf(&[]);
                    *core::ptr::addr_of_mut!(MACK_TOTAL_RESERVE) = f64::NAN;
                }
                return f64::NAN;
            }
            let packed = tri_packed(n);
            let r = mack::analyze_unchecked(&view_of(&packed, n), tail);
            // SAFETY: Guarded exclusive access to static output buffers and scalar.
            unsafe {
                write_ldf(&r.ldf);
                write_ult(&r.ultimate);
                write_res(&r.reserve);
                write_buf(core::ptr::addr_of_mut!(SE).cast(), &r.standard_error);
                write_buf(core::ptr::addr_of_mut!(SIG).cast(), &r.sigma_sq);
                let cdf = ldf::cdf_from_ldf(&r.ldf, tail);
                write_cdf(&cdf);
                *core::ptr::addr_of_mut!(MACK_TOTAL_RESERVE) = r.total_reserve;
            }
            r.total_standard_error
        })
    })
    .unwrap_or(f64::NAN)
}

/// Returns the total reserve after the last Mack call.
#[unsafe(no_mangle)]
pub extern "C" fn rk_mack_total_reserve(_n: u32) -> f64 {
    with_wasm_guard(|| {
        // SAFETY: Guarded read of initialized static scalar `MACK_TOTAL_RESERVE`.
        unsafe { *core::ptr::addr_of!(MACK_TOTAL_RESERVE) }
    })
}
