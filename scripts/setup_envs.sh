#!/usr/bin/env bash
# Create the two Python environments scripts/reproduce.sh needs, then print the command
# that runs it. Run from the repository root, in bash (Git Bash on Windows).
#
#   bash scripts/setup_envs.sh
#
# Interpreters and the PyTorch wheel index can be overridden:
#   PY_CPU=python3.10 PY_GPU=python3.13 TORCH_INDEX=https://download.pytorch.org/whl/cu130 \
#     bash scripts/setup_envs.sh
#   DRY_RUN=1 bash scripts/setup_envs.sh  # only print what would run
#
# .venv      Python 3.10 + requirements.txt      normalisation, blocking, LightGBM, decisions
# .venv-gpu  Python 3.13 + PyTorch (CUDA) + requirements.txt + requirements-gpu.txt (its pins win)
#            bi-encoder and cross-encoder training and scoring (the pipeline calls it)
set -euo pipefail
WIN=
if command -v py >/dev/null 2>&1; then  # Windows launcher
  WIN=1
  PY_CPU=${PY_CPU:-py -3.10}
  PY_GPU=${PY_GPU:-py -3.13}
else
  PY_CPU=${PY_CPU:-python3.10}
  PY_GPU=${PY_GPU:-python3.13}
fi
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
run() { echo "+ $*"; [ -n "${DRY_RUN:-}" ] || eval "$@"; }
bin() { if [ -n "$WIN" ]; then echo "$1/Scripts/python.exe"; else echo "$1/bin/python"; fi; }
for f in requirements.txt requirements-gpu.txt; do
  [ -f "$f" ] || { echo "run this from the folder holding $f and src/"; exit 1; }
done

run "$PY_CPU -m venv .venv"
CPU=$(bin .venv)
run "$CPU -m pip install --upgrade pip"
run "$CPU -m pip install -r requirements.txt"

run "$PY_GPU -m venv .venv-gpu"
GPU=$(bin .venv-gpu)
run "$GPU -m pip install --upgrade pip"
run "$GPU -m pip install torch==2.11.0 --index-url $TORCH_INDEX"
run "$GPU -m pip install -r requirements.txt"      # the GPU steps import the same package
run "$GPU -m pip install -r requirements-gpu.txt"  # last, so its pins are the ones kept
[ -n "${DRY_RUN:-}" ] || "$GPU" -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"

echo
echo "Environments ready. Reproduce the submitted outputs with:"
echo "  CPU=$CPU GPU=$GPU bash scripts/reproduce.sh <DATA_DIR>"
echo "(<DATA_DIR> holds train/ and test/; the outputs land in output/.)"
