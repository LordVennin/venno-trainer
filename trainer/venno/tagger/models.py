"""Auto-tagging models. Each returns predictions per image as
(tag, probability, category) tuples; thresholds, exclusions and formatting
are applied by the UI so they can be changed without re-running a model.
Hydra 3.5 picks its own tags (per-tag thresholds calibrated on its
validation data, then tag implications), so its tuples carry a fourth item:
whether the tag was picked.

- WD v3 (SmilingWolf, Danbooru tags): timm weights, white-padded square
  input, BGR channel order, as the reference wd-tagger code feeds them.
- JTP-3 Hydra (Project RedRocket, e621 tags): SigLIP2 NaFlex ViT with the
  Hydra head, aspect-preserving patch input (see jtp3/).
- Hydra 3.5 (Project RedRocket, e621 tags): the successor of JTP-3, run with
  the vendor's own model, image loading and calibration code (hydra35/)."""

import csv
import json
import os
import sys
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
    "hydra-3.5": {"label": "Hydra 3.5 (e621, furry, calibrated)", "repo": "RedRocket/Hydra", "kind": "hydra35", "gb": 1.1},
    "jtp-3-hydra": {"label": "JTP-3 Hydra (e621, furry)", "repo": "RedRocket/Hydra", "kind": "jtp3", "gb": 1.0},
}

FILES = {
    "wd": ["config.json", "model.safetensors", "selected_tags.csv"],
    "jtp3": ["models/jtp-3-hydra.safetensors", "data/jtp-3-hydra-tags.csv"],
    "hydra35": ["models/hydra-3.5.safetensors", "data/aliases-2024-04-07.txt"],
}


def model_dir(models_dir, model_id):
    return os.path.join(models_dir, MODELS[model_id]["repo"].replace("/", "__"))


def is_downloaded(models_dir, model_id):
    files = FILES[MODELS[model_id]["kind"]]
    return all(os.path.isfile(os.path.join(model_dir(models_dir, model_id), f)) for f in files)


def fetch(models_dir, model_id, token=None, status: Callable[[str], None] = print):
    """Download the model's files into models_dir once; later loads stay offline."""
    target = model_dir(models_dir, model_id)
    if is_downloaded(models_dir, model_id):
        return target
    from huggingface_hub import hf_hub_download

    info = MODELS[model_id]
    files = FILES[info["kind"]]
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


class Hydra35Tagger:
    """Hydra 3.5 through the vendor code. Tags are picked per image with
    thresholds calibrated for a confidence slider (0 = speculative, 0.5 =
    balanced F1, 1 = confident; minimum precision 0.1), then implications."""

    def __init__(self, root, device, dtype):
        if sys.version_info < (3, 11):
            raise RuntimeError(f"Hydra 3.5 needs Python 3.11 or newer (this venv has {sys.version.split()[0]}); WD and JTP-3 work on any version.")
        try:
            import pyvips  # noqa: F401
        except (ImportError, OSError) as e:
            raise RuntimeError("Hydra 3.5 needs pyvips: run ./setup.sh again to install it.") from e
        from .hydra35 import image as himage
        from .hydra35.classification import simple_slider
        from .hydra35.label import parse_aliases
        from .hydra35.model import load_model

        self.himage, self.simple_slider = himage, simple_slider
        model = load_model(os.path.join(root, "models", "hydra-3.5.safetensors"))
        self.model = model.to(device=device, dtype=dtype)
        self.device, self.dtype = device, dtype
        self.seqlen = model.image_config.max_seqlen
        self.tags = []
        for label in model.labels:
            sub = label.subcategory
            self.tags.append((label.label, sub if sub in ("rating", "color") else label.category))
        with open(os.path.join(root, "data", "aliases-2024-04-07.txt"), encoding="utf-8") as f:
            self.aliases = dict(parse_aliases(f.read()))
        self.calibrations = {}

    def preprocess(self, path):
        return self.himage.as_tensor(self.model.open_image(path))

    @torch.no_grad()
    def predict(self, batch):
        patches, sizes = self.himage.stack(batch, 16, self.seqlen, device=self.device)
        return self.model.forward(self.model.from_srgb(patches), sizes).float().cpu()

    def calibration(self, confidence):
        key = round(min(1.0, max(0.0, float(confidence))), 2)
        if key not in self.calibrations:
            self.calibrations[key] = self.model.calibrate(self.simple_slider(key))
        return self.calibrations[key]

    def classify(self, probs, opts):
        """The tags Hydra picks for one image: {tag: probability}."""
        return self.calibration(opts.get("confidence", 0.5)).classify_output(
            probs,
            implications=opts.get("implications") or "inherit",
            exclude_categories=set(opts.get("exclude_categories") or ()),
            exclude_labels=set(opts.get("exclude_tags") or ()),
            sort=False,
        )


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
    kind = MODELS[model_id]["kind"]
    if kind == "hydra35":
        dtype = torch.bfloat16  # its calibrated thresholds are snapped to bf16 outputs
    elif dev.type == "cuda":
        dtype = torch.float16 if kind == "wd" else torch.bfloat16
    else:
        dtype = torch.float32
    cls = {"wd": WDTagger, "jtp3": JTP3Tagger, "hydra35": Hydra35Tagger}[kind]
    return cls(root, dev, dtype)


def run(tagger, paths: List[str], floor=0.1, top=400, batch_size=4, options=None) -> List[Dict]:
    """Predictions for each path, best first; failures are reported per image.
    Taggers that pick their own tags (classify) also mark the picked ones,
    which are always included."""
    options = options or {}
    classify = getattr(tagger, "classify", None)
    aliases = getattr(tagger, "aliases", {}) if options.get("aliases") else {}
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
            if classify is None:
                preds = [[tagger.tags[j][0], round(float(row[j]), 4), tagger.tags[j][1]] for j in order.tolist()]
            else:
                picked = classify(row, options)
                cats = dict(tagger.tags)
                preds = [[t, round(float(p), 4), cats[t], True] for t, p in picked.items()]
                preds += [[tagger.tags[j][0], round(float(row[j]), 4), tagger.tags[j][1], False]
                          for j in order.tolist() if tagger.tags[j][0] not in picked]
                preds.sort(key=lambda p: -p[1])
                preds = [[aliases.get(p[0], p[0]), *p[1:]] for p in preds]
            out[i] = {"path": paths[i], "preds": preds}
    return out
