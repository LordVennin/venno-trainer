"""Store frozen Linear weights as float8 (e4m3) with a per-output-row scale and
dequantize on the fly. Halves the base model's VRAM; the LoRA stays full
precision. Works on any CUDA GPU (the matmul itself runs in bf16)."""

import torch
import torch.nn.functional as F
from torch import nn

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max


def _fp8_forward(self, x):
    w = self.weight.to(self.compute_dtype) * self.fp8_scale.to(self.compute_dtype)
    bias = self.bias.to(self.compute_dtype) if self.bias is not None else None
    return F.linear(x.to(self.compute_dtype), w, bias)


@torch.no_grad()
def quantize_linear(module: nn.Linear, compute_dtype=torch.bfloat16):
    w = module.weight.data.float()
    scale = (w.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
    module.weight = nn.Parameter((w / scale).to(FP8), requires_grad=False)
    module.register_buffer("fp8_scale", scale.to(torch.float32))
    module.compute_dtype = compute_dtype
    module.forward = _fp8_forward.__get__(module, nn.Linear)


def quantize_modules(root: nn.Module, include, compute_dtype=torch.bfloat16) -> int:
    """Quantize every nn.Linear whose qualified name starts with one of `include`."""
    count = 0
    for name, module in root.named_modules():
        if isinstance(module, nn.Linear) and any(name.startswith(p) for p in include):
            quantize_linear(module, compute_dtype)
            count += 1
    return count
