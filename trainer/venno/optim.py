"""Optimizers and learning-rate schedules."""

import ast
import math

import torch


def parse_optimizer_args(text):
    """'betas=0.9,0.99 decouple=True eps=1e-8' -> {'betas': (0.9, 0.99), ...}"""
    out = {}
    for part in str(text or "").split():
        if "=" not in part:
            continue
        key, raw = part.split("=", 1)
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        out[key] = value
    return out


OPTIMIZERS = ["AdamW8bit", "AdamW", "Adafactor", "Prodigy", "PagedAdamW8bit"]


def create_optimizer(name, params, lr, weight_decay, extra_args=""):
    kwargs = {"weight_decay": float(weight_decay or 0)}
    kwargs.update(parse_optimizer_args(extra_args))
    key = name.lower()
    if key == "adamw":
        return torch.optim.AdamW(params, lr=lr, **kwargs)
    if key in ("adamw8bit", "pagedadamw8bit"):
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise RuntimeError(f"{name} needs bitsandbytes (pip install bitsandbytes)") from e
        cls = bnb.optim.PagedAdamW8bit if key.startswith("paged") else bnb.optim.AdamW8bit
        return cls(params, lr=lr, **kwargs)
    if key == "adafactor":
        from transformers.optimization import Adafactor

        defaults = {"relative_step": False, "scale_parameter": False, "warmup_init": False}
        defaults.update(kwargs)
        return Adafactor(params, lr=lr, **defaults)
    if key == "prodigy":
        try:
            from prodigyopt import Prodigy
        except ImportError as e:
            raise RuntimeError("Prodigy needs prodigyopt (pip install prodigyopt)") from e
        return Prodigy(params, lr=lr, **kwargs)
    raise ValueError(f"Unknown optimizer {name}; choose one of {', '.join(OPTIMIZERS)}")


def create_scheduler(name, optimizer, total_steps, warmup_steps=0, cycles=1, power=1.0):
    total_steps = max(1, int(total_steps))
    warmup = max(0, int(warmup_steps))

    def warm(step):
        return (step + 1) / (warmup + 1) if step < warmup else None

    def factor(step):
        if name == "constant":
            return 1.0
        w = warm(step)
        if w is not None:
            return w
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(1.0, max(0.0, progress))
        if name == "constant_with_warmup":
            return 1.0
        if name == "linear":
            return max(0.0, 1.0 - progress)
        if name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if name == "cosine_with_restarts":
            if progress >= 1.0:
                return 0.0
            return 0.5 * (1.0 + math.cos(math.pi * ((progress * max(1, int(cycles))) % 1.0)))
        if name == "polynomial":
            return (1.0 - progress) ** power
        raise ValueError(f"Unknown LR scheduler {name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
