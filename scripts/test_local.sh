#!/usr/bin/env bash
set -euo pipefail

# Canonical local test command for this repo.
# Uses the user's venv and avoids non-writable defaults on host.
VENV_PY="/home/ajmalrasi/jetson/bin/python"

export SAVE_DIR="${SAVE_DIR:-/tmp}"

"$VENV_PY" -m pytest -q -o cache_dir=/tmp/pytest-cache-jetson-yolo-alert "$@"
