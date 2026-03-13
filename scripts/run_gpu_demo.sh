#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

uv sync
./scripts/build_lightgbm_cuda.sh
uv run --no-sync train-lightgbm
