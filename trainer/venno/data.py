"""Dataset folders -> items -> aspect buckets -> per-epoch batches.

Caption processing follows sd-scripts: prefix, whole-caption dropout (rate or
every N epochs), then shuffle with keep_tokens and per-tag dropout on
comma-separated tags."""

import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl"}


@dataclass
class Folder:
    path: str
    repeats: int = 1
    loss_weight: float = 1.0
    keep_tokens: int = 0
    caption_prefix: str = ""
    caption_dropout: float = 0.0
    tag_dropout: float = 0.0
    dropout_every_n_epochs: int = 0
    shuffle: bool = False
    flip: bool = False
    is_reg: bool = False

    @property
    def has_tag_ops(self):
        return self.shuffle or self.tag_dropout > 0

    @classmethod
    def from_job(cls, f):
        return cls(
            path=str(f.get("path", "")).strip(),
            repeats=max(1, int(f.get("repeats") or 1)),
            loss_weight=float(f.get("loss_weight") if f.get("loss_weight") not in (None, "") else 1.0),
            keep_tokens=max(0, int(f.get("keep_tokens") or 0)),
            caption_prefix=str(f.get("caption_prefix") or ""),
            caption_dropout=float(f.get("caption_dropout") or 0),
            tag_dropout=float(f.get("tag_dropout") or 0),
            dropout_every_n_epochs=max(0, int(f.get("dropout_every_n_epochs") or 0)),
            shuffle=bool(f.get("shuffle")),
            flip=bool(f.get("flip")),
            is_reg=bool(f.get("is_reg")),
        )


@dataclass
class Item:
    image: str
    caption: str  # raw caption text from file (prefix not applied)
    folder: Folder
    size: Tuple[int, int] = (0, 0)
    # (width, height) bucket per resolution index
    buckets: Dict[int, Tuple[int, int]] = field(default_factory=dict)


def read_caption(image_path, caption_ext, first_line_only):
    ext = caption_ext if caption_ext.startswith(".") else "." + caption_ext
    cap = os.path.splitext(image_path)[0] + ext
    if not os.path.isfile(cap):
        return None
    with open(cap, "r", encoding="utf-8") as f:
        text = f.read()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    return lines[0] if first_line_only else " ".join(lines)


def scan_folders(folders: List[Folder], caption_ext=".txt", first_line_only=False):
    items, missing = [], []
    for folder in folders:
        if not os.path.isdir(folder.path):
            raise FileNotFoundError(f"Dataset folder not found: {folder.path}")
        names = sorted(n for n in os.listdir(folder.path) if os.path.splitext(n)[1].lower() in IMAGE_EXTS)
        for name in names:
            path = os.path.join(folder.path, name)
            caption = read_caption(path, caption_ext, first_line_only)
            if caption is None:
                missing.append(path)
                caption = ""
            items.append(Item(image=path, caption=caption, folder=folder))
    return items, missing


def read_sizes(items: List[Item]):
    for item in items:
        with Image.open(item.image) as im:
            item.size = im.size


# ------------------------------------------------------------------ captions

def join_prefix(prefix, caption):
    if not prefix:
        return caption
    if not caption:
        return prefix.strip().rstrip(",")
    return prefix + " " + caption  # sd-scripts: prefix + " " + caption


def tag_process(caption, folder: Folder, rng: random.Random):
    """Shuffle (keeping the first keep_tokens tags) and drop tags."""
    if not folder.has_tag_ops:
        return caption
    tokens = [t.strip() for t in caption.strip().split(",")]
    fixed, flex = tokens[: folder.keep_tokens], tokens[folder.keep_tokens:]
    if folder.shuffle:
        rng.shuffle(flex)
    if folder.tag_dropout > 0:
        flex = [t for t in flex if rng.random() >= folder.tag_dropout]
    return ", ".join(t for t in fixed + flex if t)


def caption_variant(item: Item, k: int, num_variants: int, seed: int):
    """Deterministic caption variant k for caching (tag ops applied when the
    folder uses them, otherwise the single plain caption)."""
    base = join_prefix(item.folder.caption_prefix, item.caption)
    if not item.folder.has_tag_ops or num_variants <= 1:
        return base
    rng = random.Random(f"{seed}:{item.image}:{k}")
    return tag_process(base, item.folder, rng)


def caption_live(item: Item, epoch: int, rng: random.Random):
    """Fully random caption for live text encoding (sd-scripts order)."""
    f = item.folder
    if is_dropped(f, epoch, rng):
        return ""
    return tag_process(join_prefix(f.caption_prefix, item.caption), f, rng)


def is_dropped(folder: Folder, epoch: int, rng: random.Random):
    """Whether this sample trains with an empty caption. `epoch` is the 0-based
    epoch index; "every N epochs" counts from 1 like sd-scripts (epoch + 1), so
    N=3 drops the captions of every image in the folder on epochs 3, 6, 9... The
    images themselves always train."""
    if folder.caption_dropout > 0 and rng.random() < folder.caption_dropout:
        return True
    n = folder.dropout_every_n_epochs
    return n > 0 and (epoch + 1) % n == 0


# ------------------------------------------------------------------- buckets

def make_buckets(reso, step, min_side, max_side):
    area = reso * reso
    buckets = set()
    w = min_side
    while w <= max_side:
        h = min(max_side, (area // w) // step * step)
        if h >= min_side:
            buckets.add((w, h))
            buckets.add((h, w))
        w += step
    return sorted(buckets)


def choose_bucket(size, reso, step, buckets, enable_bucket, no_upscale):
    w, h = size
    if not enable_bucket:
        return (reso, reso)
    if no_upscale and w * h <= reso * reso:
        # small image: its own size rounded down to the step, aspect preserved
        bw = max(step, int(w // step * step))
        bh = max(step, int(h // step * step))
        return (bw, bh)
    ar = w / h
    return min(buckets, key=lambda b: abs(b[0] / b[1] - ar))


def assign_buckets(items: List[Item], resolutions, step, min_side, max_side, enable_bucket, no_upscale):
    tables = {}
    for ri, reso in enumerate(resolutions):
        lo = min(min_side, reso)
        hi = max(max_side, reso)
        tables[ri] = make_buckets(reso, step, lo // step * step or step, hi // step * step)
    for item in items:
        for ri, reso in enumerate(resolutions):
            item.buckets[ri] = choose_bucket(item.size, reso, step, tables[ri], enable_bucket, no_upscale)


def load_bucketed_image(path, bucket):
    """Resize to cover the bucket, center crop. Returns RGB PIL image."""
    bw, bh = bucket
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = max(bw / w, bh / h)
        nw, nh = max(bw, round(w * scale)), max(bh, round(h * scale))
        if (nw, nh) != (w, h):
            im = im.resize((nw, nh), Image.LANCZOS)
        left, top = (nw - bw) // 2, (nh - bh) // 2
        return im.crop((left, top, left + bw, top + bh))


# ------------------------------------------------------------------- batches

@dataclass
class Sample:
    item: Item
    res_index: int

    @property
    def bucket(self):
        return self.item.buckets[self.res_index]


def epoch_batches(items: List[Item], batch_sizes: List[int], num_res: int, rng: random.Random):
    """All batches of one epoch: repeats applied, regularization images matched
    to the number of training samples (sd-scripts), grouped by bucket."""
    batches = []
    for ri in range(num_res):
        bs = batch_sizes[ri] if ri < len(batch_sizes) else batch_sizes[-1]
        train = [Sample(i, ri) for i in items if not i.folder.is_reg for _ in range(i.folder.repeats)]
        reg_pool = [Sample(i, ri) for i in items if i.folder.is_reg for _ in range(i.folder.repeats)]
        reg = []
        if reg_pool and train:
            rng.shuffle(reg_pool)
            reg = [reg_pool[k % len(reg_pool)] for k in range(len(train))]
        by_bucket: Dict[Tuple[int, int], List[Sample]] = {}
        for s in train + reg:
            by_bucket.setdefault(s.bucket, []).append(s)
        for samples in by_bucket.values():
            rng.shuffle(samples)
            for k in range(0, len(samples), bs):
                batches.append(samples[k:k + bs])
    rng.shuffle(batches)
    return batches
