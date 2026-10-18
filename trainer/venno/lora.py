"""LoRA modules injected into frozen nn.Linear layers (kohya-style math:
out = W x + up(down(x)) * multiplier * alpha / rank, down kaiming-init, up zero).

Modules are keyed by an internal name chosen by the model adapter; the adapter
converts to and from the on-disk key layout ComfyUI expects."""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class LoRAModule(nn.Module):
    def __init__(self, name: str, org: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.name = name
        self.rank = rank
        self.dropout = dropout
        self.multiplier = 1.0
        self.lora_down = nn.Linear(org.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, org.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.register_buffer("alpha", torch.tensor(float(alpha)))
        self.scale = float(alpha) / rank
        # kept outside the module tree so the base weights are not LoRA state
        self._org = [org]
        self._org_forward = org.forward

    def apply(self):
        self._org[0].forward = self.forward

    def remove(self):
        self._org[0].forward = self._org_forward

    def forward(self, x):
        out = self._org_forward(x)
        if self.multiplier == 0:
            return out
        h = self.lora_down(x.to(self.lora_down.weight.dtype) if not torch.is_autocast_enabled() else x)
        if self.dropout > 0 and self.training:
            h = F.dropout(h, p=self.dropout)
        h = self.lora_up(h)
        return out + (h * (self.multiplier * self.scale)).to(out.dtype)


class LoRANetwork(nn.Module):
    def __init__(self, targets: List[Tuple[str, nn.Linear]], rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.loras = nn.ModuleDict()
        self.names = []
        for name, module in targets:
            key = name.replace(".", "__")  # ModuleDict keys cannot contain dots
            self.loras[key] = LoRAModule(name, module, rank, alpha, dropout)
            self.names.append(name)

    def apply_to(self):
        for lora in self.loras.values():
            lora.apply()

    def remove(self):
        for lora in self.loras.values():
            lora.remove()

    def set_multiplier(self, m: float):
        for lora in self.loras.values():
            lora.multiplier = m

    def internal_state_dict(self) -> Dict[str, torch.Tensor]:
        """{name.lora_down.weight, name.lora_up.weight, name.alpha} with dotted model paths."""
        out = {}
        for lora in self.loras.values():
            out[f"{lora.name}.lora_down.weight"] = lora.lora_down.weight.detach()
            out[f"{lora.name}.lora_up.weight"] = lora.lora_up.weight.detach()
            out[f"{lora.name}.alpha"] = lora.alpha.detach()
        return out

    def load_internal_state_dict(self, sd: Dict[str, torch.Tensor], strict: bool = True):
        missing = []
        for lora in self.loras.values():
            down = sd.get(f"{lora.name}.lora_down.weight")
            up = sd.get(f"{lora.name}.lora_up.weight")
            if down is None or up is None:
                missing.append(lora.name)
                continue
            if down.shape != lora.lora_down.weight.shape or up.shape != lora.lora_up.weight.shape:
                raise ValueError(
                    f"LoRA shape mismatch for {lora.name}: file rank {down.shape[0]}, network rank {lora.rank}"
                )
            with torch.no_grad():
                lora.lora_down.weight.copy_(down)
                lora.lora_up.weight.copy_(up)
        if strict and missing:
            raise ValueError(f"LoRA weights missing for {len(missing)} modules, e.g. {missing[:3]}")
        return missing

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
