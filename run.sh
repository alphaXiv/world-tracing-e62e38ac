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

# wt imports torch.utils.checkpoint.CheckpointPolicy at module load (added in
# torch 2.5). The base image ships torch 2.4.1, and `torch>=2.2` in pyproject
# does not force an upgrade, so pin a torch that has the symbol first.
echo "[run] ensuring torch >= 2.5 (CheckpointPolicy) ..."
python - <<'PY' || pip install -q "torch==2.6.0"
import sys
from packaging.version import parse
import torch
sys.exit(0 if parse(torch.__version__.split("+")[0]) >= parse("2.5") else 1)
PY

# Install the inference package (base deps only: torch, numpy, opencv,
# einops, safetensors, huggingface_hub, structlog, beartype, jaxtyping).
echo "[run] installing wt (editable) ..."
pip install -q -e . 2>&1 | tail -5 || pip install -e . 2>&1 | tail -20

python -c "import torch; print('[run] CUDA available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

# The released r75b checkpoint is gated on Hugging Face; downloading it needs a
# token authorized on haoz19/object-model-6layer. huggingface_hub picks up
# HF_TOKEN / HUGGING_FACE_HUB_TOKEN / a cached ~/.cache/huggingface/token.
python - <<'PY' || true
import os
tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not tok:
    try:
        from huggingface_hub import get_token  # modern API
        tok = get_token()
    except Exception:
        tok = None
if not tok:
    print("[run] HF token: NONE visible in run environment")
else:
    try:
        from huggingface_hub import whoami
        print(f"[run] HF token: present (user={whoami(token=tok).get('name')})")
    except Exception as e:
        print(f"[run] HF token: present but whoami failed: {e}")
PY

echo "[run] launching PoC ..."
python poc.py

echo "[run] artifacts:"
ls -la .openresearch/artifacts/
echo "[run] ===== EVAL.md ====="
cat .openresearch/artifacts/EVAL.md
