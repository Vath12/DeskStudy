#!/usr/bin/env bash
# Write the printable calibration charts (US Letter and A4 PDFs) to calibration_chart/.
# Optional: --square-mm N for a different square size (default 36).
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

uv run desk-view chart "$@"
echo
echo "Print one of the PDFs above at 100% scale (\"Actual size\", not \"Fit to page\")."
