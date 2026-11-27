#!/usr/bin/env bash
# Run Criterion benchmarks on robbit-theliomira (i9-14900K) locally or over the Tailscale mesh,
# check against benches/baseline-i9-14900k.json, and optionally refresh baseline + SVG + README.
#
# Usage:
#   ./scripts/bench-theliomira.sh                      # Run full kernels suite & compare vs baseline
#   ./scripts/bench-theliomira.sh --filter odp         # Run matching benchmarks only
#   ./scripts/bench-theliomira.sh --refresh-assets     # Update baseline JSON, assets/benchmarks.svg, README.md
#   ./scripts/bench-theliomira.sh --fail-on-regression # Exit non-zero if any benchmark regresses > threshold
#   ./scripts/bench-theliomira.sh --local              # Force local execution (skip SSH hop)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THELIOMIRA_HOST="${THELIOMIRA_HOST:-robbit-theliomira}"
THELIOMIRA_REPO="${THELIOMIRA_REPO:-~/Projects/reserving-kernels}"

FILTER=""
THRESHOLD="15"
REFRESH_ASSETS=false
FAIL_ON_REGRESSION=false
FORCE_LOCAL=false
BENCH_TARGET="kernels"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --filter)
      FILTER="${2:?missing value for --filter}"
      shift 2
      ;;
    --bench)
      BENCH_TARGET="${2:?missing value for --bench}"
      shift 2
      ;;
    --threshold)
      THRESHOLD="${2:?missing value for --threshold}"
      shift 2
      ;;
    --refresh-assets)
      REFRESH_ASSETS=true
      shift
      ;;
    --fail-on-regression)
      FAIL_ON_REGRESSION=true
      shift
      ;;
    --local)
      FORCE_LOCAL=true
      shift
      ;;
    -h|--help)
      sed -n '2,11p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

CURRENT_HOST="$(hostname -s 2>/dev/null || hostname)"

if [[ "$FORCE_LOCAL" == false && "$CURRENT_HOST" != "robbit-theliomira" && "$CURRENT_HOST" != "theliomira" ]]; then
  echo "==> Delegating benchmark run to ${THELIOMIRA_HOST} (${THELIOMIRA_REPO}) over Tailscale..."
  REMOTE_ARGS=("--local" "--bench" "$BENCH_TARGET" "--threshold" "$THRESHOLD")
  [[ -n "$FILTER" ]] && REMOTE_ARGS+=("--filter" "$FILTER")
  [[ "$REFRESH_ASSETS" == true ]] && REMOTE_ARGS+=("--refresh-assets")
  [[ "$FAIL_ON_REGRESSION" == true ]] && REMOTE_ARGS+=("--fail-on-regression")

  QUOTED_ARGS="$(printf ' %q' "${REMOTE_ARGS[@]}")"
  ssh -o BatchMode=yes "$THELIOMIRA_HOST" "bash -lc 'cd ${THELIOMIRA_REPO} && git pull --ff-only && ./scripts/bench-theliomira.sh${QUOTED_ARGS}'"

  if [[ "$REFRESH_ASSETS" == true ]]; then
    echo "==> Copying refreshed benchmark assets back to ${REPO_ROOT}..."
    scp -q "${THELIOMIRA_HOST}:${THELIOMIRA_REPO}/benches/baseline-i9-14900k.json" "${REPO_ROOT}/benches/baseline-i9-14900k.json"
    scp -q "${THELIOMIRA_HOST}:${THELIOMIRA_REPO}/assets/benchmarks.svg" "${REPO_ROOT}/assets/benchmarks.svg"
    scp -q "${THELIOMIRA_HOST}:${THELIOMIRA_REPO}/README.md" "${REPO_ROOT}/README.md"
  fi
  exit 0
fi

cd "$REPO_ROOT"
echo "==> Running cargo bench --bench ${BENCH_TARGET} on ${CURRENT_HOST} ($(rustc --version))..."
BENCH_CMD=(cargo bench --bench "$BENCH_TARGET" -- --noplot)
if [[ -n "$FILTER" ]]; then
  BENCH_CMD+=("$FILTER")
fi
"${BENCH_CMD[@]}"

REPORT_CMD=(
  python3 scripts/bench-report.py
  --criterion-dir "${CARGO_TARGET_DIR:-target}/criterion"
  --baseline benches/baseline-i9-14900k.json
  --threshold "$THRESHOLD"
)
if [[ "$FAIL_ON_REGRESSION" == true ]]; then
  REPORT_CMD+=(--fail-on-regression)
fi
if [[ "$REFRESH_ASSETS" == true ]]; then
  REPORT_CMD+=(
    --update-baseline
    --render-svg assets/benchmarks.svg
    --update-readme README.md
  )
fi

"${REPORT_CMD[@]}"
