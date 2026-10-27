#!/usr/bin/env bash
# Start Desk View in the background (output goes to .logs/desk-view.log).
# Pass -d to run in the foreground attached to this terminal instead.
# Installs/updates dependencies automatically on first run.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

foreground=0
args=()
for arg in "$@"; do
  if [[ "$arg" == "-d" ]]; then
    foreground=1
  else
    args+=("$arg")
  fi
done

if (( foreground )); then
  exec uv run desk-view ${args[@]+"${args[@]}"}
fi

mkdir -p .logs
log=.logs/desk-view.log
nohup uv run desk-view ${args[@]+"${args[@]}"} >"$log" 2>&1 &
echo "Desk View started in the background (pid $!). Logs: $log"
