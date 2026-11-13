#!/usr/bin/env bash
# One-time setup: Python venv for the trainer (./venv) + the UI's Node packages.
#   ./setup.sh
# Options (environment variables):
#   PYTHON=python3.12        interpreter used to create the venv (3.11+ for PyTorch 2.15)
#   TORCH_VERSION=2.15.0 TORCH_INDEX=https://download.pytorch.org/whl/cu134   a specific PyTorch build
# With an NVIDIA driver for CUDA 13 (580+) and Python 3.11+, PyTorch 2.15 (CUDA 13.4) is installed:
# compiled MOSS-Image blocks then keep their gradient checkpoint inside the compiled graph, far less
# memory (Fizgig #221). Otherwise the CUDA 12.8 build, keeping the PyTorch version already installed.
#   FAST_KERNELS=0           skip flash-linear-attention (faster MOSS caption encoding)
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
TORCH_VERSION=${TORCH_VERSION:-}
TORCH_INDEX=${TORCH_INDEX:-}
export PIP_NO_CACHE_DIR=1  # do not keep a second copy of every wheel on disk

if [ ! -x venv/bin/python ]; then
    echo "Creating venv with $($PY --version)..."
    "$PY" -m venv venv
fi
venv/bin/python -m pip install --upgrade pip
if [ -z "$TORCH_VERSION$TORCH_INDEX" ]; then
    DRIVER=$( (nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || true) | head -n1 | cut -d. -f1)
    PY311=$(venv/bin/python -c 'import sys; print(int(sys.version_info >= (3, 11)))')
    if [[ "$DRIVER" =~ ^[0-9]+$ ]] && [ "$DRIVER" -ge 580 ] && [ "$PY311" = 1 ]; then
        TORCH_VERSION=2.15.0
        TORCH_INDEX=https://download.pytorch.org/whl/cu134
    else
        [ "$PY311" = 1 ] || echo "The venv's Python is older than 3.11, which PyTorch 2.15 needs: keeping the CUDA 12.8 build. To upgrade: rm -rf venv && PYTHON=python3.12 ./setup.sh"
        if ! [[ "$DRIVER" =~ ^[0-9]+$ ]]; then
            echo "No NVIDIA driver found: installing the CUDA 12.8 build of PyTorch."
        elif [ "$DRIVER" -lt 580 ]; then
            echo "NVIDIA driver $DRIVER is older than 580 (CUDA 13): keeping the CUDA 12.8 build of PyTorch."
        fi
    fi
fi
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
# torchvision comes from the same index, matched to the torch installed, so
# the tagger's timm cannot pull a mismatched build (pip would replace torch)
if [ -z "$TORCH_VERSION" ]; then
    TORCH_VERSION=$(venv/bin/python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || true)
fi
echo "Installing PyTorch ${TORCH_VERSION:-(latest)} from $TORCH_INDEX ..."
if [ -n "$TORCH_VERSION" ]; then
    venv/bin/pip install torchvision "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
else
    venv/bin/pip install torch torchvision --index-url "$TORCH_INDEX"
fi
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
