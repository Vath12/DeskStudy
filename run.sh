#!/usr/bin/env bash
# Start Desk View. Installs/updates dependencies automatically on first run.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

exec uv run desk-view "$@"
