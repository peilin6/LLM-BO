#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s CONFIG.yaml\n' "$0" >&2
  exit 2
fi

CONFIG=$1
ROOT=/home/nice/qinbin/stu/lpl/code/dibo
PYTHON=/home/nice/qinbin/stu/lpl/miniconda3/envs/dibo/bin/python
DIBO=/home/nice/qinbin/stu/lpl/miniconda3/envs/dibo/bin/dibo

cd "$ROOT"
"$PYTHON" scripts/gpu_preflight.py --config "$CONFIG" --output runs/gpu_preflight.json
"$DIBO" env-check --config "$CONFIG"
"$DIBO" tune --config "$CONFIG"