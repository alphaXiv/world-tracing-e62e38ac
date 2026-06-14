#!/usr/bin/env bash
# Minimal proof-of-concept for World Tracing (paper 2606.13652).
#
# Installs the released `wt` inference package and runs the r75b object
# model (1.7B params, 504x504, 6 layers) on the shipped object test images,
# dumping quantitative evidence of the paper's core multilayer-geometry
# claim into .openresearch/artifacts/ (EVAL.md + metrics.json).
#
# Self-contained: needs only a single CUDA GPU. The frozen MoGe encoder is
# baked into the released checkpoint (no separate download), and attention
# falls back to PyTorch SDPA when flash-attn is absent.
set -euo pipefail

cd "$(dirname "$0")"

echo "[run] python: $(python --version 2>&1), torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not yet installed')"

# Install the inference package (base deps only: torch, numpy, opencv,
# einops, safetensors, huggingface_hub, structlog, beartype, jaxtyping).
echo "[run] installing wt (editable) ..."
pip install -q -e . 2>&1 | tail -5 || pip install -e . 2>&1 | tail -20

python -c "import torch; print('[run] CUDA available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo "[run] launching PoC ..."
python poc.py

echo "[run] artifacts:"
ls -la .openresearch/artifacts/
echo "[run] ===== EVAL.md ====="
cat .openresearch/artifacts/EVAL.md
