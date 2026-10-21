"""Interface every model adapter implements, plus loading helpers."""

import contextlib
import json
import os
from typing import Dict, List, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn


class ModelAdapter:
    # identifies cache entries; bump the tag if the encoding ever changes
    cache_key = ""
    text_cache_tag = ""
    pixel_multiple = 16
    caption_first_line_only = False
    # text encoder small enough to keep loaded and encode captions every step
    supports_live_text = False
    default_sample_shift = 1.0
    text_chunk = 16  # captions per encode_text call while caching

    def __init__(self, run, device, dtype):
        self.run = run
        self.job = run["job"]
        self.paths = run.get("model_paths", {}) or {}
        self.device = device
        self.dtype = dtype

    def fetch_models(self):
        """Download missing model files before caching starts (no-op by default)."""

    # text ---------------------------------------------------------------
    def load_text_encoder(self): ...
    def unload_text_encoder(self): ...
    def encode_text(self, texts: List[str]) -> List[Dict[str, torch.Tensor]]: ...
    def collate_text(self, conds: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]: ...

    def text_sort_key(self, text: str):
        return len(text)

    # vae ----------------------------------------------------------------
    def load_vae(self): ...
    def unload_vae(self): ...
    def encode_images(self, pixels: torch.Tensor) -> torch.Tensor: ...
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor: ...

    # dit ----------------------------------------------------------------
    def load_dit(self) -> nn.Module: ...
    def lora_targets(self, ignore: List[str]) -> List[Tuple[str, nn.Linear]]: ...
    def fp8_prefixes(self) -> List[str]: ...
    def enable_gradient_checkpointing(self, skip_last: int = 0) -> int: ...
    def compile_blocks(self) -> int: ...
    def predict(self, noisy: torch.Tensor, sigmas: torch.Tensor, text: Dict[str, torch.Tensor]) -> torch.Tensor: ...
    def sample(self, cond, uncond, width, height, steps, cfg, seed, shift): ...

    # lora files ---------------------------------------------------------
    def export_lora(self, internal: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]: ...
    def import_lora(self, sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]: ...


LORA_SUFFIXES = ("lora_down.weight", "lora_up.weight", "alpha")


def split_lora_key(key):
    """'a.b.c.lora_down.weight' -> ('a.b.c', 'lora_down.weight')"""
    for suffix in LORA_SUFFIXES:
        if key.endswith("." + suffix):
            return key[: -len(suffix) - 1], suffix
    raise KeyError(f"Not a LoRA key: {key}")


def free_memory():
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def strip_prefixes(sd, prefixes):
    out = {}
    for k, v in sd.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
                break
        out[k] = v
    return out


def load_safetensors_dir(path, key_filter=None) -> Dict[str, torch.Tensor]:
    """One .safetensors file, or every shard in a folder (sharded index aware)."""
    files = []
    if os.path.isfile(path):
        files = [path]
    else:
        index = [n for n in os.listdir(path) if n.endswith(".safetensors.index.json")]
        if index:
            with open(os.path.join(path, index[0])) as f:
                files = sorted({os.path.join(path, v) for v in json.load(f)["weight_map"].values()})
        else:
            files = sorted(os.path.join(path, n) for n in os.listdir(path) if n.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors weights in {path}")
    sd = {}
    for file in files:
        if key_filter is None:
            sd.update(load_file(file))
            continue
        with safe_open(file, "pt") as f:
            for k in f.keys():
                if key_filter(k):
                    sd[k] = f.get_tensor(k)
    return sd


@contextlib.contextmanager
def meta_device():
    with torch.device("meta"):
        yield


def rebuild_rotary(model, config):
    """Rotary inv_freq buffers are computed in __init__ (not saved), so they
    are still on the meta device after a meta construction + assign load."""
    rotary = getattr(model, "rotary_emb", None)
    if rotary is not None and any(b.is_meta for b in rotary.buffers()):
        with torch.device("cpu"):
            model.rotary_emb = type(rotary)(config=config)


def assert_materialized(module: nn.Module, what: str):
    meta = [n for n, t in list(module.named_parameters()) + list(module.named_buffers()) if t.is_meta]
    if meta:
        raise RuntimeError(f"{what}: weights missing from the checkpoint, e.g. {meta[:5]}")
