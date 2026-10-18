"""On-disk latent and text-embedding caches.

Both live under one cache root (shared by every job) and are keyed by content,
so a dataset is only encoded once per model and editing an image or caption
simply produces a new entry:
  <root>/<model>/latents/<aa>/<sha1(path|mtime|size)>_<w>x<h>.safetensors
  <root>/<model>/text/<aa>/<sha1(tag|caption)>.safetensors
"""

import hashlib
import os
from typing import Dict, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _atomic_save(tensors, path, metadata=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    save_file({k: v.contiguous() for k, v in tensors.items()}, tmp, metadata=metadata)
    os.replace(tmp, path)


class LatentCache:
    def __init__(self, root: str, model_key: str):
        self.dir = os.path.join(root, model_key, "latents")

    def path(self, image: str, bucket) -> str:
        st = os.stat(image)
        key = _sha1(f"{os.path.abspath(image)}|{st.st_mtime_ns}|{st.st_size}")
        return os.path.join(self.dir, key[:2], f"{key}_{bucket[0]}x{bucket[1]}.safetensors")

    def has(self, image: str, bucket, need_flip: bool) -> bool:
        p = self.path(image, bucket)
        if not os.path.exists(p):
            return False
        if not need_flip:
            return True
        try:
            with safe_open(p, "pt") as f:
                return "latent_flip" in f.keys()
        except Exception:
            return False

    def save(self, image: str, bucket, latent: torch.Tensor, latent_flip: Optional[torch.Tensor] = None):
        tensors = {"latent": latent.to(torch.bfloat16)}
        if latent_flip is not None:
            tensors["latent_flip"] = latent_flip.to(torch.bfloat16)
        _atomic_save(tensors, self.path(image, bucket))

    def load(self, image: str, bucket, flip: bool = False) -> torch.Tensor:
        with safe_open(self.path(image, bucket), "pt") as f:
            key = "latent_flip" if flip and "latent_flip" in f.keys() else "latent"
            return f.get_tensor(key)


class TextCache:
    def __init__(self, root: str, model_key: str, tag: str):
        self.dir = os.path.join(root, model_key, "text")
        self.tag = tag

    def path(self, text: str) -> str:
        key = _sha1(f"{self.tag}|{text}")
        return os.path.join(self.dir, key[:2], f"{key}.safetensors")

    def has(self, text: str) -> bool:
        return os.path.exists(self.path(text))

    def save(self, text: str, tensors: Dict[str, torch.Tensor]):
        _atomic_save(tensors, self.path(text), metadata={"caption": text[:4000]})

    def load(self, text: str) -> Dict[str, torch.Tensor]:
        with safe_open(self.path(text), "pt") as f:
            return {k: f.get_tensor(k) for k in f.keys()}
