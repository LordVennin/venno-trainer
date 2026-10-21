"""Auto-tagging models. Each returns raw predictions per image as
(tag, probability, category) tuples; thresholds, exclusions and formatting
are applied by the UI so they can be changed without re-running a model.

- WD v3 (SmilingWolf, Danbooru tags): timm weights, white-padded square
  input, BGR channel order, as the reference wd-tagger code feeds them.
- JTP-3 Hydra (Project RedRocket, e621 tags): SigLIP2 NaFlex ViT with the
  Hydra head, aspect-preserving patch input (see jtp3/)."""

import csv
import json
import os
import threading
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

WD_CATEGORIES = {0: "general", 4: "character", 9: "rating"}
E621_CATEGORIES = {0: "general", 1: "artist", 3: "copyright", 4: "character", 5: "species", 6: "invalid", 7: "meta", 8: "lore"}

MODELS = {
    "wd-eva02-large-v3": {"label": "WD EVA02-Large v3 (Danbooru, best WD)", "repo": "SmilingWolf/wd-eva02-large-tagger-v3", "kind": "wd", "gb": 1.3},
    "wd-vit-large-v3": {"label": "WD ViT-Large v3 (Danbooru)", "repo": "SmilingWolf/wd-vit-large-tagger-v3", "kind": "wd", "gb": 1.3},
    "wd-swinv2-v3": {"label": "WD SwinV2 v3 (Danbooru)", "repo": "SmilingWolf/wd-swinv2-tagger-v3", "kind": "wd", "gb": 0.4},
    "wd-vit-v3": {"label": "WD ViT v3 (Danbooru, fastest)", "repo": "SmilingWolf/wd-vit-tagger-v3", "kind": "wd", "gb": 0.4},
    "jtp-3-hydra": {"label": "JTP-3 Hydra (e621, furry)", "repo": "RedRocket/Hydra", "kind": "jtp3", "gb": 1.0},
}

WD_FILES = ["config.json", "model.safetensors", "selected_tags.csv"]
JTP3_FILES = ["models/jtp-3-hydra.safetensors", "data/jtp-3-hydra-tags.csv"]


def model_dir(models_dir, model_id):
    return os.path.join(models_dir, MODELS[model_id]["repo"].replace("/", "__"))


def is_downloaded(models_dir, model_id):
    files = WD_FILES if MODELS[model_id]["kind"] == "wd" else JTP3_FILES
    return all(os.path.isfile(os.path.join(model_dir(models_dir, model_id), f)) for f in files)


def fetch(models_dir, model_id, token=None, status: Callable[[str], None] = print):
    """Download the model's files into models_dir once; later loads stay offline."""
    target = model_dir(models_dir, model_id)
    if is_downloaded(models_dir, model_id):
        return target
    from huggingface_hub import hf_hub_download

    info = MODELS[model_id]
    files = WD_FILES if info["kind"] == "wd" else JTP3_FILES
    status(f"Downloading {info['label']} (~{info['gb']} GB, once)...")
    done = threading.Event()

    def report():
        while not done.wait(5.0):
            size = sum(os.path.getsize(os.path.join(d, n)) for d, _, ns in os.walk(target) for n in ns if os.path.exists(os.path.join(d, n)))
            status(f"Downloading {info['label']}: {size / 1e9:.2f} of ~{info['gb']} GB")

    threading.Thread(target=report, daemon=True).start()
    try:
        for f in files:
            hf_hub_download(info["repo"], f, local_dir=target, token=token or None)
    finally:
        done.set()
    return target


def load_rgb(path, background=(255, 255, 255)):
    """RGB image with EXIF rotation applied and transparency composited."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            canvas = Image.new("RGB", im.size, background)
            canvas.paste(im, mask=im.getchannel("A"))
            return canvas
        return im.convert("RGB")


class WDTagger:
    def __init__(self, root, device, dtype):
        import timm
        from safetensors.torch import load_file

        with open(os.path.join(root, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        self.size = int(cfg.get("pretrained_cfg", {}).get("input_size", [3, 448, 448])[-1])
        model = timm.create_model(cfg["architecture"], pretrained=False, num_classes=cfg["num_classes"], **cfg.get("model_args", {}))
        model.load_state_dict(load_file(os.path.join(root, "model.safetensors")))
        self.model = model.eval().requires_grad_(False).to(device=device, dtype=dtype)
        self.device, self.dtype = device, dtype
        self.tags: List[Tuple[str, str]] = []
        with open(os.path.join(root, "selected_tags.csv"), encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                self.tags.append((row["name"], WD_CATEGORIES.get(int(row["category"]), "general")))

    def preprocess(self, path):
        im = load_rgb(path)
        side = max(im.size)
        square = Image.new("RGB", (side, side), (255, 255, 255))
        square.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
        square = square.resize((self.size, self.size), Image.BICUBIC)
        arr = np.asarray(square, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr[:, :, ::-1].copy()).permute(2, 0, 1)  # RGB -> BGR, CHW

    @torch.no_grad()
    def predict(self, batch: List[torch.Tensor]):
        x = torch.stack(batch).to(device=self.device, dtype=self.dtype)
        return torch.sigmoid(self.model(x).float()).cpu()


class JTP3Tagger:
    PATCH, SEQ = 16, 1024

    def __init__(self, root, device, dtype):
        from .jtp3 import model as jm

        self.jm = jm
        model, labels = jm.load_model(os.path.join(root, "models", "jtp-3-hydra.safetensors"), device=device, dtype=dtype)
        self.model = model.requires_grad_(False)
        self.device, self.dtype = device, dtype
        cats = {}
        with open(os.path.join(root, "data", "jtp-3-hydra-tags.csv"), encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                cats[row["tag"]] = E621_CATEGORIES.get(int(row["category"] or 0), "general")
        # e621 files ratings under meta; report them as ratings like WD does
        self.tags = [(t, "rating" if t in ("safe", "questionable", "explicit") else cats.get(t, "general")) for t in labels]

    def preprocess(self, path):
        with Image.open(path) as im:
            processed = self.jm.process_image(im, self.PATCH, self.SEQ)
            return self.jm.patchify_image(processed, self.PATCH, self.SEQ)

    @torch.no_grad()
    def predict(self, batch):
        patches = torch.stack([b[0] for b in batch]).to(self.device)
        coords = torch.stack([b[1] for b in batch]).to(self.device)
        valid = torch.stack([b[2] for b in batch]).to(self.device)
        patches = patches.to(self.dtype).div_(127.5).sub_(1.0)
        logits = self.model(patches, patch_coord=coords.to(torch.int32), patch_valid=valid)
        return torch.sigmoid(logits.float()).cpu()


def load(model_id, models_dir, device, token=None, status=print):
    if model_id not in MODELS:
        raise ValueError(f"Unknown tagger model: {model_id}")
    try:
        import timm  # noqa: F401
    except ImportError as e:
        raise RuntimeError("The tagger needs the 'timm' package: run ./setup.sh again to install it.") from e
    root = fetch(models_dir, model_id, token, status)
    status(f"Loading {MODELS[model_id]['label']} on {device}...")
    dev = torch.device(device)
    if dev.type == "cuda":
        dtype = torch.bfloat16 if MODELS[model_id]["kind"] == "jtp3" else torch.float16
    else:
        dtype = torch.float32
    cls = WDTagger if MODELS[model_id]["kind"] == "wd" else JTP3Tagger
    return cls(root, dev, dtype)


def run(tagger, paths: List[str], floor=0.1, top=400, batch_size=4) -> List[Dict]:
    """Predictions for each path, best first; failures are reported per image."""
    out: List[Optional[Dict]] = [None] * len(paths)
    ready = []
    for i, p in enumerate(paths):
        try:
            ready.append((i, tagger.preprocess(p)))
        except Exception as e:  # unreadable image
            out[i] = {"path": p, "error": f"{type(e).__name__}: {e}"}
    for k in range(0, len(ready), batch_size):
        chunk = ready[k:k + batch_size]
        probs = tagger.predict([x for _, x in chunk])
        for (i, _), row in zip(chunk, probs):
            keep = torch.nonzero(row >= floor).flatten()
            order = keep[torch.argsort(row[keep], descending=True)][:top]
            out[i] = {
                "path": paths[i],
                "preds": [[tagger.tags[j][0], round(float(row[j]), 4), tagger.tags[j][1]] for j in order.tolist()],
            }
    return out
