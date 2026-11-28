#!/usr/bin/env python3
"""Reproducible chainladder-python parity & timing harness for reserving-kernels.

Requires: `chainladder` (>= 0.10; README timings use 0.10.1) and `numpy`.

Every parity value is asserted against the reserving-kernels reference below
within an explicit tolerance; the script exits non-zero on any mismatch.
Timings are informational only and include Python interpreter / pandas
overhead.

Usage:
    python3 scripts/parity_python.py [--skip-timings]
"""

import statistics
import sys
import time

MIN_CHAINLADDER = (0, 10)

# Reference values produced by reserving-kernels (and asserted in
# tests/source_benchmarks.rs). Tolerances are relative.
REFERENCE = {
    # Mack (1993) RAA: reserve 52,135.23, total SE 26,909.01.
    "mack_raa_reserve": (52_135.23, 1e-6),
    "mack_raa_se": (26_909.01, 1e-5),
    # Taylor & Ashe GenIns: reserve 18,680,855.61, total Mack SE 2,447,094.86.
    "mack_genins_reserve": (18_680_855.61, 1e-6),
    "mack_genins_se": (2_447_094.86, 1e-5),
    # Clark (2003) loglogistic LDF on GenIns (no truncation); crate-derived.
    "clark_ll_genins_reserve": (35_534_710.25, 1e-3),
}


def bench_us(fn, reps: int = 20) -> float:
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(times)


def check(failures: list, key: str, actual: float) -> None:
    expected, rel_tol = REFERENCE[key]
    rel_err = abs(actual - expected) / abs(expected)
    status = "OK  " if rel_err <= rel_tol else "FAIL"
    print(f"[{status}] {key:<26} actual={actual:,.2f} expected={expected:,.2f} rel_err={rel_err:.2e} tol={rel_tol:.0e}")
    if rel_err > rel_tol:
        failures.append(key)


def main() -> int:
    try:
        import chainladder as cl
    except ImportError:
        print("ERROR: the `chainladder` package is not installed (pip install 'chainladder>=0.10').", file=sys.stderr)
        return 2

    version = tuple(int(p) for p in cl.__version__.split(".")[:2] if p.isdigit())
    print(f"chainladder-python {cl.__version__}")
    if version < MIN_CHAINLADDER:
        print(f"ERROR: chainladder >= {'.'.join(map(str, MIN_CHAINLADDER))} required", file=sys.stderr)
        return 2

    raa = cl.load_sample("raa")
    genins = cl.load_sample("genins")
    failures: list = []

    print("=== Numerical Parity (casact/chainladder-python) ===")
    mack_raa = cl.MackChainladder().fit(raa)
    check(failures, "mack_raa_reserve", float(mack_raa.ibnr_.sum()))
    check(failures, "mack_raa_se", float(mack_raa.total_mack_std_err_.values[0, 0, 0, -1]))

    mack_gen = cl.MackChainladder().fit(genins)
    check(failures, "mack_genins_reserve", float(mack_gen.ibnr_.sum()))
    check(failures, "mack_genins_se", float(mack_gen.total_mack_std_err_.values[0, 0, 0, -1]))

    clark_gen = cl.ClarkLDF(growth="loglogistic").fit(genins)
    check(failures, "clark_ll_genins_reserve", float(clark_gen.ibnr_.sum()))

    if "--skip-timings" not in sys.argv[1:]:
        print("\n=== Median Single-Threaded Timings (include interpreter overhead) ===")
        print(f"Mack Chain Ladder (RAA 10x10):   {bench_us(lambda: cl.MackChainladder().fit(raa), 20):.1f} us")
        print(f"Clark Loglogistic MLE (GenIns):  {bench_us(lambda: cl.ClarkLDF(growth='loglogistic').fit(genins), 20):.1f} us")
        print(
            "ODP Bootstrap (GenIns, 1k sims): "
            f"{bench_us(lambda: cl.BootstrapODPSample(n_sims=1000, random_state=42).fit_transform(genins), 5) / 1e3:.2f} ms"
        )

    if failures:
        print(f"\nPARITY FAILED: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("\nAll parity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
