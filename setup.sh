#!/usr/bin/env bash
# One-time setup: Python venv for the trainer (./venv) + the UI's Node packages.
#   ./setup.sh
# Options (environment variables):
#   PYTHON=python3.12        interpreter used to create the venv (3.10+)
#   TORCH_INDEX=https://download.pytorch.org/whl/cu126   PyTorch build for older drivers
#   FAST_KERNELS=0           skip flash-linear-attention (faster MOSS caption encoding)
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
export PIP_NO_CACHE_DIR=1  # do not keep a second copy of every wheel on disk

if [ ! -x venv/bin/python ]; then
    echo "Creating venv with $($PY --version)..."
    "$PY" -m venv venv
fi
venv/bin/python -m pip install --upgrade pip
echo "Installing PyTorch from $TORCH_INDEX ..."
venv/bin/pip install torch --index-url "$TORCH_INDEX"
venv/bin/pip install -r trainer/requirements.txt
if [ "${FAST_KERNELS:-1}" = "1" ]; then
    venv/bin/pip install flash-linear-attention || echo "(optional) flash-linear-attention failed to install; MOSS caption caching will use the slower PyTorch path."
fi

venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
PY

if command -v npm >/dev/null; then
    npm install
else
    echo "Node.js/npm not found: install Node 18+ for the web UI."
fi
du -sh venv 2>/dev/null | awk '{print "venv size: " $1}'
echo "Done. Start the UI with ./start.sh"
