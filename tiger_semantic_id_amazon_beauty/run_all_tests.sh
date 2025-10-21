#!/usr/bin/env bash
set -euo pipefail

# Try to activate a virtual environment if present
if [ -d "venv" ]; then
  source "venv/bin/activate" || true
elif [ -d ".venv" ]; then
  source ".venv/bin/activate" || true
fi

SCOPE="tiger"
if [ "${1:-}" = "--all" ]; then
  SCOPE="all"
fi

if [ "$SCOPE" = "tiger" ]; then
  echo "Running TIGER tests..."
  python3 -m pytest -q tests
else
  echo "Running all tests under this project..."
  python3 -m pytest -q
fi
