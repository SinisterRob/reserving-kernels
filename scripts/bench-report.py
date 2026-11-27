#!/usr/bin/env python3
"""Parse Criterion benchmark results, check regressions, and refresh README / SVG assets.

Uses only the Python 3 standard library so it runs cleanly on any mesh node or CI runner.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


def load_criterion(criterion_dir: Path) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    pattern = str(criterion_dir / "**" / "new" / "estimates.json")
    for path_str in sorted(glob.glob(pattern, recursive=True)):
        path = Path(path_str)
        rel = path.relative_to(criterion_dir).as_posix()
        bench_id = rel[: -len("/new/estimates.json")]
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        results[bench_id] = {
            "median_ns": round(float(data["median"]["point_estimate"]), 2),
            "mean_ns": round(float(data["mean"]["point_estimate"]), 2),
        }
    return results


def fmt_time_table(ns: float) -> str:
    if ns < 1_000.0:
        us = ns / 1_000.0
        return f"{us:.2f} µs"
    if ns < 10_000.0:
        us = ns / 1_000.0
        return f"{us:.2f} µs"
    if ns < 100_000.0:
        us = ns / 1_000.0
        return f"{us:.1f} µs"
    if ns < 1_000_000.0:
        ms = ns / 1_000_000.0
        return f"{ms:.2f} ms"
    if ns < 10_000_000.0:
        ms = ns / 1_000_000.0
        return f"{ms:.2f} ms"
    ms = ns / 1_000_000.0
    if ms < 1_000.0:
        return f"{ms:.1f} ms"
    return f"{ms:,.0f} ms"


def fmt_time_ref_table(ns: float | None, prefer_ms: bool = False) -> str:
    if ns is None:
        return "—"
    us = ns / 1_000.0
    ms = ns / 1_000_000.0
    if prefer_ms or ms >= 50.0:
        if ms >= 1_000.0:
            return f"{ms:,.0f} ms"
        return f"{ms:.1f} ms"
    if us >= 1_000.0:
        return f"{us:,.0f} µs"
    return f"{us:.0f} µs"


def fmt_time_svg(ns: float) -> str:
    us = ns / 1_000.0
    if us < 10.0:
        return f"{us:.2f} µs"
    if us < 100.0:
        return f"{us:.1f} µs"
    if us < 1_000.0:
        return f"{us:.0f} µs"
    ms = us / 1_000.0
    if ms < 10.0:
        return f"{ms:.2f} ms"
    if ms < 100.0:
        return f"{ms:.1f} ms"
    if ms < 1_000.0:
        return f"{ms:.0f} ms"
    return f"{ms:,.0f} ms"


def fmt_time_svg_rust(ns: float) -> str:
    us = ns / 1_000.0
    if us < 10.0:
        return f"{us:.2f} µs"
    if us < 100.0:
        return f"{us:.1f} µs"
    ms = us / 1_000.0
    if ms < 10.0:
        return f"{ms:.2f} ms"
    return f"{ms:.1f} ms"


def fmt_ratio(other_ns: float | None, rust_ns: float) -> str:
    if other_ns is None or rust_ns <= 0.0:
        return "—"
    ratio = other_ns / rust_ns
    if ratio >= 1_000.0:
        rounded = round(ratio, -1)
        return f"{rounded:,.0f}×"
    return f"{ratio:,.0f}×"


def fmt_ns_compact(ns: float) -> str:
    if ns < 1_000.0:
        return f"{ns:.1f} ns"
    if ns < 1_000_000.0:
        return f"{ns / 1_000.0:.2f} µs"
    return f"{ns / 1_000_000.0:.2f} ms"


def detect_rustc_version() -> str:
    try:
        out = subprocess.check_output(["rustc", "--version"], text=True).strip()
        parts = out.split()
        return parts[1] if len(parts) >= 2 else out
    except Exception:
        return "unknown"


def detect_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def render_svg(baseline: dict, out_path: Path) -> None:
    meta = baseline["metadata"]
    benches = baseline["benchmarks"]
    comp = baseline["comparison"]

    x0 = 248.0
    px_per_decade = 71.5
    min_log_us = -1.0  # 0.1 µs

    def bar_width(ns: float) -> float:
        us = max(ns / 1_000.0, 0.1)
        return max(4.0, (math.log10(us) - min_log_us) * px_per_decade)

    row_step = 54
    base_y = 88
    height = 76 + len(comp) * row_step + 68

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 {height}" width="900" height="{height}" role="img" aria-label="reserving-kernels benchmark latency comparison">',
        "  <defs>",
        "    <style>",
        "      .bg { fill: #0d1117; stroke: #30363d; stroke-width: 1; }",
        '      .title { font: 600 17px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #f0f6fc; }',
        '      .sub { font: 400 12px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #8b949e; }',
        '      .label { font: 500 12.5px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #e6edf3; }',
        "      .val { font: 600 10.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }",
        "      .tick { font: 400 11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; fill: #8b949e; }",
        "      .grid { stroke: #21262d; stroke-width: 1; }",
        "      .axis { stroke: #30363d; stroke-width: 1; }",
        '      .legend { font: 500 11.5px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #c9d1d9; }',
        "    </style>",
        "  </defs>",
        f'  <rect x="0.5" y="0.5" width="899" height="{height - 1}" rx="10" class="bg"/>',
        '  <text x="24" y="34" class="title">Steady-State Kernel Latency (Log Scale — Lower is Faster)</text>',
        f'  <text x="24" y="54" class="sub">Single-threaded in-process timings on {meta["hardware"]} · rustc {meta["rustc"]} · {meta["reference_r"]} (R) · {meta["reference_python"]}</text>',
        "  <!-- Legend -->",
        '  <rect x="545" y="21" width="11" height="11" rx="2" fill="#f97316"/>',
        '  <text x="562" y="31" class="legend">reserving-kernels (Rust)</text>',
        '  <rect x="708" y="21" width="11" height="11" rx="2" fill="#38bdf8"/>',
        '  <text x="725" y="31" class="legend">R</text>',
        '  <rect x="752" y="21" width="11" height="11" rx="2" fill="#a78bfa"/>',
        '  <text x="769" y="31" class="legend">Python</text>',
        "",
    ]

    grid_bottom = 76 + len(comp) * row_step + 6
    ticks = [
        "0.1 µs",
        "1 µs",
        "10 µs",
        "100 µs",
        "1 ms",
        "10 ms",
        "100 ms",
        "1 s",
        "10 s",
    ]
    for i, tick in enumerate(ticks):
        tx = x0 + i * px_per_decade
        lines.append(f'  <line x1="{tx:.1f}" y1="76" x2="{tx:.1f}" y2="{grid_bottom}" class="grid"/>')
        lines.append(
            f'  <text x="{tx:.1f}" y="{grid_bottom + 18}" text-anchor="middle" class="tick">{tick}</text>'
        )
    lines.append(f'  <line x1="{int(x0)}" y1="76" x2="{int(x0)}" y2="{grid_bottom}" class="axis"/>')

    for idx, entry in enumerate(comp):
        y = base_y + idx * row_step
        if idx % 2 == 1:
            lines.append(
                f'  <rect x="14" y="{y - 6}" width="872" height="50" rx="5" fill="#161b22" opacity="0.55"/>'
            )
        lines.append(
            f'  <text x="236" y="{y + 23}" text-anchor="end" class="label">{entry["label"]}</text>'
        )

        rust_ns = benches[entry["id"]]["median_ns"]
        rw = bar_width(rust_ns)
        lines.append(
            f'  <rect x="{int(x0)}" y="{y}" width="{rw:.1f}" height="10" rx="2.5" fill="#f97316"/>'
        )
        lines.append(
            f'  <text x="{x0 + rw + 6:.1f}" y="{y + 9}" class="val" fill="#fdba74">{fmt_time_svg_rust(rust_ns)}</text>'
        )

        r_ns = entry.get("r_ns")
        if r_ns is not None:
            w = bar_width(r_ns)
            lines.append(
                f'  <rect x="{int(x0)}" y="{y + 13}" width="{w:.1f}" height="10" rx="2.5" fill="#38bdf8"/>'
            )
            lines.append(
                f'  <text x="{x0 + w + 6:.1f}" y="{y + 22}" class="val" fill="#7dd3fc">{fmt_time_svg(r_ns)}</text>'
            )
        else:
            lines.append(
                f'  <text x="{int(x0) + 6}" y="{y + 22}" class="val" fill="#6e7681">n/a</text>'
            )

        py_ns = entry.get("python_ns")
        if py_ns is not None:
            w = bar_width(py_ns)
            lines.append(
                f'  <rect x="{int(x0)}" y="{y + 26}" width="{w:.1f}" height="10" rx="2.5" fill="#a78bfa"/>'
            )
            lines.append(
                f'  <text x="{x0 + w + 6:.1f}" y="{y + 35}" class="val" fill="#c4b5fd">{fmt_time_svg(py_ns)}</text>'
            )
        else:
            lines.append(
                f'  <text x="{int(x0) + 6}" y="{y + 35}" class="val" fill="#6e7681">n/a</text>'
            )

    lines.append(
        f'  <text x="24" y="{height - 12}" class="sub">Note: reserving-kernels operates on flat f64 slices; R ChainLadder and chainladder-python include S3/S4 or 4D DataFrame metadata overhead.</text>'
    )
    lines.append("</svg>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_readme_table(baseline: dict, readme_path: Path) -> None:
    benches = baseline["benchmarks"]
    comp = baseline["comparison"]
    rows = [
        "| Benchmark | `reserving-kernels` | R `ChainLadder` | `chainladder-python` | vs. R | vs. Python |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in comp:
        rust_ns = benches[entry["id"]]["median_ns"]
        prefer_ms = rust_ns >= 100_000.0
        r_ns = entry.get("r_ns")
        py_ns = entry.get("python_ns")
        rows.append(
            f"| {entry['label']} | **{fmt_time_table(rust_ns)}** | "
            f"{fmt_time_ref_table(r_ns, prefer_ms=prefer_ms)} | "
            f"{fmt_time_ref_table(py_ns, prefer_ms=prefer_ms)} | "
            f"{fmt_ratio(r_ns, rust_ns)} | {fmt_ratio(py_ns, rust_ns)} |"
        )
    new_table = "\n".join(rows)

    text = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"\| Benchmark \| `reserving-kernels` \| R `ChainLadder` \| `chainladder-python` \| vs\. R \| vs\. Python \|\n"
        r"\| --- \| ---: \| ---: \| ---: \| ---: \| ---: \|\n"
        r"(?:\|.*\n)+",
        re.MULTILINE,
    )
    updated, count = pattern.subn(new_table + "\n", text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not locate performance table in {readme_path}")
    readme_path.write_text(updated, encoding="utf-8")


def compare_and_report(
    current: dict[str, dict[str, float]],
    baseline: dict,
    threshold_pct: float,
) -> tuple[str, list[str], dict]:
    base_benches: dict[str, dict[str, float]] = baseline.get("benchmarks", {})
    meta = baseline.get("metadata", {})

    lines = [
        f"## Criterion Benchmark Report (`{meta.get('hardware', 'unknown')}`)",
        "",
        f"- **Baseline:** `{meta.get('commit', 'unknown')}` ({meta.get('updated_at', '')}, rustc `{meta.get('rustc', '')}`)",
        f"- **Current:** `{detect_git_sha()}` (rustc `{detect_rustc_version()}`)",
        f"- **Regression threshold:** `+{threshold_pct:.1f}%` (median)",
        "",
        "| Benchmark | Baseline | Current | Delta | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]

    regressions: list[str] = []
    summary_rows: list[dict] = []

    for bench_id in sorted(current.keys()):
        cur_ns = current[bench_id]["median_ns"]
        base_entry = base_benches.get(bench_id)
        if base_entry is None:
            lines.append(
                f"| `{bench_id}` | — | {fmt_ns_compact(cur_ns)} | new | 🆕 new |"
            )
            summary_rows.append(
                {"id": bench_id, "baseline_ns": None, "current_ns": cur_ns, "delta_pct": None}
            )
            continue

        base_ns = float(base_entry["median_ns"])
        delta_pct = ((cur_ns - base_ns) / base_ns) * 100.0 if base_ns > 0 else 0.0
        if delta_pct > threshold_pct:
            status = "🔴 regression"
            regressions.append(f"{bench_id}: +{delta_pct:.1f}% ({fmt_ns_compact(base_ns)} -> {fmt_ns_compact(cur_ns)})")
        elif delta_pct > 5.0:
            status = "🟡 slower"
        elif delta_pct < -5.0:
            status = "🟢 faster"
        else:
            status = "⚪ steady"

        lines.append(
            f"| `{bench_id}` | {fmt_ns_compact(base_ns)} | {fmt_ns_compact(cur_ns)} | "
            f"{delta_pct:+.1f}% | {status} |"
        )
        summary_rows.append(
            {
                "id": bench_id,
                "baseline_ns": base_ns,
                "current_ns": cur_ns,
                "delta_pct": round(delta_pct, 2),
                "status": status,
            }
        )

    if regressions:
        lines.extend(
            [
                "",
                f"### Regressions exceeding +{threshold_pct:.1f}%",
                *[f"- `{r}`" for r in regressions],
            ]
        )
    else:
        lines.extend(["", f"No regressions exceeding `+{threshold_pct:.1f}%` detected across {len(current)} benchmarks."])

    report_md = "\n".join(lines) + "\n"
    summary_json = {
        "commit": detect_git_sha(),
        "rustc": detect_rustc_version(),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "threshold_pct": threshold_pct,
        "regressions": regressions,
        "benchmarks": summary_rows,
    }
    return report_md, regressions, summary_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--criterion-dir",
        type=Path,
        default=Path(os.environ.get("CARGO_TARGET_DIR", "target")) / "criterion",
        help="Path to Criterion output directory",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("benches/baseline-i9-14900k.json"),
        help="Path to checked-in baseline JSON",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=15.0,
        help="Regression percentage threshold on median latency (default: 15.0)",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if any benchmark regresses beyond --threshold",
    )
    parser.add_argument(
        "--github-summary",
        action="store_true",
        help="Append Markdown report to $GITHUB_STEP_SUMMARY if set",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write structured JSON benchmark report",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Merge current Criterion measurements into --baseline JSON",
    )
    parser.add_argument(
        "--render-svg",
        type=Path,
        default=None,
        help="Regenerate benchmark comparison SVG at the given path",
    )
    parser.add_argument(
        "--update-readme",
        type=Path,
        default=None,
        help="Update the Performance table in the given README.md path",
    )
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    current = load_criterion(args.criterion_dir) if args.criterion_dir.exists() else {}

    if not current and not (args.render_svg or args.update_readme):
        print(f"error: no Criterion estimates found in {args.criterion_dir}", file=sys.stderr)
        return 1

    if current:
        report_md, regressions, summary_json = compare_and_report(
            current, baseline, args.threshold
        )
        print(report_md, end="")

        if args.github_summary:
            summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
            if summary_file:
                with open(summary_file, "a", encoding="utf-8") as fh:
                    fh.write(report_md)

        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(summary_json, indent=2) + "\n", encoding="utf-8"
            )

        if args.update_baseline:
            baseline["benchmarks"].update(current)
            baseline["metadata"]["rustc"] = detect_rustc_version()
            baseline["metadata"]["commit"] = detect_git_sha()
            baseline["metadata"]["updated_at"] = dt.date.today().isoformat()
            args.baseline.write_text(
                json.dumps(baseline, indent=2, sort_keys=False) + "\n",
                encoding="utf-8",
            )
            print(f"Updated baseline: {args.baseline}")
    else:
        regressions = []

    if args.render_svg is not None:
        render_svg(baseline, args.render_svg)
        print(f"Rendered SVG: {args.render_svg}")

    if args.update_readme is not None:
        update_readme_table(baseline, args.update_readme)
        print(f"Updated README table: {args.update_readme}")

    if args.fail_on_regression and regressions:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
