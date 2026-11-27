#!/usr/bin/env python3
"""Reproducible chainladder-python parity & timing harness for reserving-kernels.

Requires: `chainladder` (>= 0.8) and `numpy`.

Usage:
    python3 scripts/parity_python.py
"""

import statistics
import time

import chainladder as cl


def bench_us(fn, reps: int = 20) -> float:
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(times)


def main() -> None:
    raa = cl.load_sample("raa")
    genins = cl.load_sample("genins")

    print("=== Numerical Parity (casact/chainladder-python) ===")
    mack_raa = cl.MackChainladder().fit(raa)
    print(
        f"Mack RAA:    reserve = {float(mack_raa.ibnr_.sum()):.2f}, "
        f"SE = {float(mack_raa.total_mack_std_err_.values[0, 0, 0, -1]):.2f}"
    )

    mack_gen = cl.MackChainladder().fit(genins)
    print(
        f"Mack GenIns: reserve = {float(mack_gen.ibnr_.sum()):.2f}, "
        f"SE = {float(mack_gen.total_mack_std_err_.values[0, 0, 0, -1]):.2f}"
    )

    clark_gen = cl.ClarkLDF(growth="loglogistic").fit(genins)
    print(f"Clark LL GenIns: reserve = {float(clark_gen.ibnr_.sum()):.2f}")

    print("\n=== Median Single-Threaded Timings ===")
    print(f"Mack Chain Ladder (RAA 10x10):   {bench_us(lambda: cl.MackChainladder().fit(raa), 20):.1f} us")
    print(f"Clark Loglogistic MLE (GenIns):  {bench_us(lambda: cl.ClarkLDF(growth='loglogistic').fit(genins), 20):.1f} us")
    print(
        "ODP Bootstrap (GenIns, 1k sims): "
        f"{bench_us(lambda: cl.BootstrapODPSample(n_sims=1000, random_state=42).fit_transform(genins), 5) / 1e3:.2f} ms"
    )


if __name__ == "__main__":
    main()
