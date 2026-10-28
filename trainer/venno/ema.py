"""Exponential moving average of the LoRA weights. Checkpoints and samples
are made from the average; training always runs on the raw weights. The
average is the centre of the zigzag a high learning rate takes around the
good solution, so saved epochs are smoother and less distorted.

The decay ramps in as min(decay, (1 + n) / (10 + n)) so the first steps track
the weights instead of anchoring to the zero init. After Fizgig's EMAWeights
(github.com/shootthesound/Fizgig, training/ema.py, Apache-2.0), which measured
decay 0.98 above EMA-off on late epochs, with half the epoch-to-epoch spread;
0.995 lagged the run."""

from contextlib import contextmanager

import torch


class EMAWeights:
    def __init__(self, params, decay=0.98, ramp=10):
        self.decay = float(decay)
        self.ramp = int(ramp)
        self.n = 0
        self.params = [p for p in params if p.requires_grad]
        self.shadow = [p.detach().clone().float() for p in self.params]

    @torch.no_grad()
    def update(self):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (self.ramp + self.n))
        for s, p in zip(self.shadow, self.params):
            s.mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    @contextmanager
    def applied(self):
        """The averaged weights in place of the raw ones for the block (a save or samples)."""
        backup = [p.detach().clone() for p in self.params]
        with torch.no_grad():
            for s, p in zip(self.shadow, self.params):
                p.copy_(s.to(p.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for b, p in zip(backup, self.params):
                    p.copy_(b)

    def state_dict(self):
        return {"n": self.n, "decay": self.decay, "shadow": [s.cpu() for s in self.shadow]}

    def load_state_dict(self, st):
        if len(st["shadow"]) != len(self.shadow):
            raise ValueError("EMA state does not match this LoRA")
        self.n = int(st["n"])
        for s, v in zip(self.shadow, st["shadow"]):
            s.copy_(v.to(s.device))
