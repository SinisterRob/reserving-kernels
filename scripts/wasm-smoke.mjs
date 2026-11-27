#!/usr/bin/env node
/**
 * Runtime smoke test for the compiled wasm32-unknown-unknown C-ABI binary.
 *
 * Usage:
 *   node scripts/wasm-smoke.mjs [path/to/reserving_kernels.wasm]
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import zlib from "node:zlib";

const wasmPath =
  process.argv[2] ?? "target/wasm32-unknown-unknown/release/reserving_kernels.wasm";

const bytes = fs.readFileSync(wasmPath);
const gzipBytes = zlib.gzipSync(bytes, { level: 9 }).length;

const { instance } = await WebAssembly.instantiate(bytes);
const e = instance.exports;

const requiredExports = [
  "memory",
  "rk_n_max",
  "rk_tri_ptr",
  "rk_tri_stride",
  "rk_prem_ptr",
  "rk_ldf_ptr",
  "rk_cdf_ptr",
  "rk_ult_ptr",
  "rk_res_ptr",
  "rk_exp_ptr",
  "rk_se_ptr",
  "rk_sig_ptr",
  "rk_int_ptr",
  "rk_slp_ptr",
  "rk_ldf_volume_weighted",
  "rk_ldf_simple_average",
  "rk_ldf_ex_hi_lo",
  "rk_ldf_volume_weighted_last_n",
  "rk_ldf_simple_average_last_n",
  "rk_cdf_compute",
  "rk_chain_ladder",
  "rk_bf",
  "rk_bf_expected_from_premium",
  "rk_cape_cod_elr",
  "rk_cape_cod",
  "rk_brosius",
  "rk_mack",
  "rk_mack_total_reserve",
];

for (const name of requiredExports) {
  assert.ok(name in e, `Missing WASM export: ${name}`);
}

const f64View = (ptr, len) => new Float64Array(e.memory.buffer, ptr, len);
const assertClose = (actual, expected, tol, label) => {
  assert.ok(
    Number.isFinite(actual) && Math.abs(actual - expected) <= tol,
    `${label}: expected ${expected} ± ${tol}, got ${actual}`,
  );
};

const nMax = e.rk_n_max();
const stride = e.rk_tri_stride();
assert.ok(nMax >= 4 && stride === nMax, `Unexpected n_max=${nMax}, stride=${stride}`);

// Write Quick Start 4x4 upper triangle into static TRI buffer.
const n = 4;
const rows = [
  [100.0, 160.0, 195.0, 205.0],
  [120.0, 190.0, 232.0, Number.NaN],
  [135.0, 218.0, Number.NaN, Number.NaN],
  [150.0, Number.NaN, Number.NaN, Number.NaN],
];
const tri = f64View(e.rk_tri_ptr(), nMax * stride);
tri.fill(Number.NaN);
for (let i = 0; i < n; i++) {
  for (let j = 0; j < n; j++) {
    tri[i * stride + j] = rows[i][j];
  }
}

const prem = f64View(e.rk_prem_ptr(), nMax);
prem.set([300.0, 340.0, 380.0, 420.0]);

// 1. Volume-weighted LDFs & CDFs
assert.equal(e.rk_ldf_volume_weighted(n), n - 1);
const ldfs = f64View(e.rk_ldf_ptr(), n - 1);
assertClose(ldfs[0], 568.0 / 355.0, 1e-12, "LDF[0]");
assertClose(ldfs[1], 427.0 / 350.0, 1e-12, "LDF[1]");
assertClose(ldfs[2], 205.0 / 195.0, 1e-12, "LDF[2]");

assert.equal(e.rk_cdf_compute(n, 1.0), n);
const cdfs = f64View(e.rk_cdf_ptr(), n);
assertClose(cdfs[3], 1.0, 1e-12, "CDF[3]");
assertClose(cdfs[0], ldfs[0] * ldfs[1] * ldfs[2], 1e-12, "CDF[0]");

// 2. Chain Ladder
assert.equal(e.rk_chain_ladder(n, 1.0), n);
const clUlt = Array.from(f64View(e.rk_ult_ptr(), n));
const clRes = Array.from(f64View(e.rk_res_ptr(), n));
assertClose(clUlt[0], 205.0, 1e-12, "CL ult[0]");
assertClose(clRes[0], 0.0, 1e-12, "CL res[0]");
const clTotalReserve = clRes.reduce((a, b) => a + b, 0);
assert.ok(clTotalReserve > 0 && Number.isFinite(clTotalReserve), "CL total reserve positive");

// 3. Bornhuetter-Ferguson
assert.equal(e.rk_bf_expected_from_premium(n, 0.7), n);
assert.equal(e.rk_bf(n, 1.0), n);
const bfRes = Array.from(f64View(e.rk_res_ptr(), n));
assertClose(bfRes[0], 0.0, 1e-12, "BF res[0]");
assert.ok(bfRes.slice(1).every((v) => Number.isFinite(v) && v > 0), "BF reserves positive");

// 4. Cape Cod
const ccElrStandalone = e.rk_cape_cod_elr(n, 1.0);
const ccElr = e.rk_cape_cod(n, 1.0);
assertClose(ccElr, ccElrStandalone, 1e-12, "Cape Cod ELR parity");
assert.ok(ccElr > 0 && ccElr < 2.0, `Cape Cod ELR out of range: ${ccElr}`);

// 5. Brosius
assert.equal(e.rk_brosius(n, 1.0), n);
const broRes = Array.from(f64View(e.rk_res_ptr(), n));
assert.ok(broRes.every((v) => Number.isFinite(v)), "Brosius reserves finite");

// 6. Mack (cross-checked against Chain Ladder total reserve)
const mackSe = e.rk_mack(n, 1.0);
const mackTotalReserve = e.rk_mack_total_reserve(n);
assertClose(mackTotalReserve, clTotalReserve, 1e-9, "Mack vs CL total reserve");
assert.ok(Number.isFinite(mackSe) && mackSe > 0, `Mack SE invalid: ${mackSe}`);

// 7. Invalid tail rejection
assert.equal(e.rk_chain_ladder(n, 0.5), 0, "Invalid tail (< 1.0) should return 0");
assert.ok(Number.isNaN(e.rk_mack(n, Number.NaN)), "NaN tail should return NaN");

console.log(
  `WASM smoke test passed (${wasmPath}: ${bytes.length.toLocaleString()} bytes raw, ${gzipBytes.toLocaleString()} bytes gzip)`,
);
console.log(
  `  Mack total reserve = ${mackTotalReserve.toFixed(2)}, SE = ${mackSe.toFixed(2)}, Cape Cod ELR = ${(ccElr * 100).toFixed(2)}%`,
);
