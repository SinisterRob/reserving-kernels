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
    return f"{ms / 1_000.0:.2f} s"


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


def fmt_speedup_sig3(ratio: float) -> str:
    if ratio >= 1_000.0:
        mag = 10 ** (int(math.floor(math.log10(ratio))) - 2)
        val = int(round(ratio / mag) * mag)
        return f"{val:,}×"
    return f"{int(round(ratio)):,}×"


def fmt_speedup_range(rust_ns: float, r_ns: float | None, py_ns: float | None) -> str:
    ratios = [x / rust_ns for x in (r_ns, py_ns) if x is not None and rust_ns > 0.0]
    if not ratios:
        return "—"
    lo, hi = min(ratios), max(ratios)
    if len(ratios) == 1:
        return fmt_speedup_sig3(hi)
    return f"{fmt_speedup_sig3(lo)}–{fmt_speedup_sig3(hi)}"


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

    width = 920
    row_step = 54
    first_y = 96
    height = first_y + len(comp) * row_step + 38

    bar_x = 258.0
    bar_max_w = 424.0
    speedup_x = 892.0

    c_rust = "#f97316"
    c_r = "#38bdf8"
    c_r_txt = "#7dd3fc"
    c_py = "#a78bfa"
    c_py_txt = "#c4b5fd"

    hw_short = meta["hardware"].replace("Intel Core ", "")
    rustc_short = ".".join(str(meta["rustc"]).split(".")[:2])

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="reserving-kernels linear benchmark latency comparison">',
        "  <defs>",
        "    <style>",
        "      .bg { fill: #0d1117; stroke: #262c36; stroke-width: 1; }",
        '      .title { font: 600 17px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #f0f6fc; letter-spacing: -0.2px; }',
        '      .sub { font: 400 11.5px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #8b949e; }',
        '      .label { font: 500 12.5px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #e6edf3; }',
        "      .val { font: 500 10.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }",
        "      .val-rust { font: 600 10.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; fill: #fdba74; }",
        "      .speedup { font: 700 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; fill: #fb923c; }",
        '      .hdr { font: 600 10px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #6e7681; letter-spacing: 0.6px; }',
        '      .legend { font: 500 11.5px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #c9d1d9; }',
        '      .foot { font: 400 10px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #6e7681; }',
        "    </style>",
        "  </defs>",
        f'  <rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="12" class="bg"/>',
        '  <text x="26" y="36" class="title">Steady-State Kernel Latency</text>',
        '  <text x="26" y="55" class="sub">Linear scale per benchmark · lower is faster</text>',
        "  <!-- Legend (right-anchored flush with SPEEDUP column) -->",
        f'  <text x="{speedup_x:.1f}" y="35" text-anchor="end" class="legend">Python</text>',
        f'  <rect x="{speedup_x - 53:.1f}" y="26" width="10" height="10" rx="2.5" fill="{c_py}"/>',
        f'  <text x="{speedup_x - 71:.1f}" y="35" text-anchor="end" class="legend">R</text>',
        f'  <rect x="{speedup_x - 95:.1f}" y="26" width="10" height="10" rx="2.5" fill="{c_r}"/>',
        f'  <text x="{speedup_x - 113:.1f}" y="35" text-anchor="end" class="legend">reserving-kernels (Rust)</text>',
        f'  <rect x="{speedup_x - 252:.1f}" y="26" width="10" height="10" rx="2.5" fill="{c_rust}"/>',
        f'  <line x1="26" y1="72" x2="{width - 26}" y2="72" stroke="#21262d" stroke-width="1"/>',
        f'  <text x="{bar_x:.1f}" y="88" class="hdr">RELATIVE TIME (LINEAR)</text>',
        f'  <text x="{speedup_x:.1f}" y="88" text-anchor="end" class="hdr">SPEEDUP</text>',
        f'  <line x1="{bar_x:.1f}" y1="{first_y + 4}" x2="{bar_x:.1f}" y2="{first_y + len(comp) * row_step - 4}" stroke="#30363d" stroke-width="1"/>',
    ]

    for idx, entry in enumerate(comp):
        y0 = first_y + idx * row_step
        yc = y0 + row_step / 2.0

        if idx % 2 == 1:
            lines.append(
                f'  <rect x="18" y="{y0 + 2:.1f}" width="{width - 36}" height="{row_step - 4:.1f}" rx="6" fill="#161b22" opacity="0.55"/>'
            )

        lines.append(f'  <text x="26" y="{yc + 4:.1f}" class="label">{entry["label"]}</text>')

        rust_ns = float(benches[entry["id"]]["median_ns"])
        r_ns = entry.get("r_ns")
        py_ns = entry.get("python_ns")
        baselines = [float(v) for v in (r_ns, py_ns) if v is not None]
        max_ns = max(baselines) if baselines else rust_ns

        w_rust = max(2.5, (rust_ns / max_ns) * bar_max_w)
        w_r = max(2.5, (float(r_ns) / max_ns) * bar_max_w) if r_ns is not None else None
        w_py = max(2.5, (float(py_ns) / max_ns) * bar_max_w) if py_ns is not None else None

        if r_ns is not None and py_ns is not None:
            y_rust, y_r, y_py = yc - 16.0, yc - 4.0, yc + 8.0
        elif r_ns is not None:
            y_rust, y_r, y_py = yc - 10.0, yc + 2.0, None
        elif py_ns is not None:
            y_rust, y_r, y_py = yc - 10.0, None, yc + 2.0
        else:
            y_rust, y_r, y_py = yc - 4.0, None, None

        lines.append(
            f'  <rect x="{bar_x:.1f}" y="{y_rust:.1f}" width="{w_rust:.1f}" height="8" rx="1.5" fill="{c_rust}"/>'
        )
        lines.append(
            f'  <text x="{bar_x + w_rust + 7:.1f}" y="{y_rust + 7.5:.1f}" class="val-rust">{fmt_time_svg_rust(rust_ns)}</text>'
        )

        if y_r is not None and w_r is not None and r_ns is not None:
            lines.append(
                f'  <rect x="{bar_x:.1f}" y="{y_r:.1f}" width="{w_r:.1f}" height="8" rx="2" fill="{c_r}" opacity="0.88"/>'
            )
            lines.append(
                f'  <text x="{bar_x + w_r + 7:.1f}" y="{y_r + 7.5:.1f}" class="val" fill="{c_r_txt}">{fmt_time_svg(float(r_ns))}</text>'
            )

        if y_py is not None and w_py is not None and py_ns is not None:
            lines.append(
                f'  <rect x="{bar_x:.1f}" y="{y_py:.1f}" width="{w_py:.1f}" height="8" rx="2" fill="{c_py}" opacity="0.88"/>'
            )
            lines.append(
                f'  <text x="{bar_x + w_py + 7:.1f}" y="{y_py + 7.5:.1f}" class="val" fill="{c_py_txt}">{fmt_time_svg(float(py_ns))}</text>'
            )

        sp = fmt_speedup_range(rust_ns, float(r_ns) if r_ns is not None else None, float(py_ns) if py_ns is not None else None)
        lines.append(
            f'  <text x="{speedup_x:.1f}" y="{yc + 4:.1f}" text-anchor="end" class="speedup">{sp}</text>'
        )

    lines.append(
        f'  <text x="26" y="{height - 14}" class="foot">{hw_short} · rustc {rustc_short} · {meta["reference_r"]} (R) · {meta["reference_python"]} · flat f64 slices vs. S3/S4 &amp; DataFrame overhead</text>'
    )
    lines.append("</svg>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_readme_table(baseline: dict, readme_path: Path) -> None:
    benches = baseline["benchmarks"]
    comp = baseline["comparison"]
    rows = [
        "| Benchmark | `reserving-kernels` | R `ChainLadder` | `chainladder-python` | vs. R | vs. Python |",
        "| --- | :---: | :---: | :---: | :---: | :---: |",
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
        r"\| --- \| (?:---:| :---: |:---:) \| (?:---:| :---: |:---:) \| (?:---:| :---: |:---:) \| (?:---:| :---: |:---:) \| (?:---:| :---: |:---:) \|\n"
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
