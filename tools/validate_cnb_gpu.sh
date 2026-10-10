#!/usr/bin/env bash
# CNB GPU smoke entry: validates the actual runner before any GPU evidence is recorded.
set -euo pipefail

python tools/environment_preflight.py --gpu --packages torch triton
nvidia-smi
python - <<'PY'
import torch

assert torch.cuda.is_available(), "CNB GPU runner has no CUDA-visible device"
device = torch.device("cuda")
x = torch.ones(1024, device=device)
print({"device": torch.cuda.get_device_name(device), "sum": float(x.sum().cpu())})
PY
