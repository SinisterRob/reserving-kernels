#!/usr/bin/env bash
# Enable repo-tracked .githooks on this checkout.
#
# Usage:
#   ./scripts/install-hooks.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

chmod +x "$REPO_ROOT"/.githooks/* "$REPO_ROOT"/scripts/*.sh "$REPO_ROOT"/scripts/*.py "$REPO_ROOT"/scripts/*.mjs 2>/dev/null || true
git -C "$REPO_ROOT" config core.hooksPath .githooks
echo "core.hooksPath -> $(git -C "$REPO_ROOT" config --get core.hooksPath)"
