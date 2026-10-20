"""Venno trainer entry point:  python -m venno.train <run.json>

run.json is written by the UI server: {"job": <job.json>, "model_paths": {...},
"output_dir", "cache_dir", "models_dir", "hf_token"}.

Phases: scan dataset -> cache latents (VAE only) -> cache text embeddings
(text encoder only) -> load DiT + LoRA -> train, saving checkpoints/states
and sample images along the way."""

import json
import math
import os
import random
import re
import shutil
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# fewer fragmentation OOMs with changing bucket shapes; must be set before CUDA starts
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL.PngImagePlugin import PngInfo
from safetensors.torch import load_file, save_file

from . import data as D
from .cache import LatentCache, TextCache
from .fp8 import quantize_modules
from .log import Progress, log
from .lora import LoRANetwork
from .models.base import free_memory
from .optim import create_optimizer, create_scheduler

ADAPTERS = {
    "anima": ("venno.models.anima.adapter", "AnimaAdapter"),
    "moss_image": ("venno.models.moss.adapter", "MossAdapter"),
}

_stop_requested = False


class StopTraining(Exception):
    pass


def _request_stop(signum, frame):
    global _stop_requested
    if not _stop_requested:
        log("Stop requested: finishing the current step and saving...")
    _stop_requested = True


def check_stop():
    if _stop_requested:
        raise StopTraining()


def num(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def intv(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def int_list(text, default):
    vals = [int(x) for x in re.split(r"[,\s]+", str(text or "")) if x.strip().isdigit() and int(x) > 0]
    return vals or default


def make_adapter(run, device, dtype):
    import importlib

    module, cls = ADAPTERS[run["job"]["model"]]
    return getattr(importlib.import_module(module), cls)(run, device, dtype)


class Trainer:
    def __init__(self, run):
        self.run = run
        self.job = run["job"]
        self.t = self.job["training"]
        self.n = self.job["network"]
        self.d = self.job["dataset"]
        self.s = self.job["sampling"]
        dev = os.environ.get("VENNO_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        if self.device.type == "cpu":
            log("WARNING: no CUDA GPU found, training on CPU (only useful for tests).")
        else:
            # TF32 tensor cores for the fp32 parts (MOSS VAE, loss); cuDNN picks the
            # fastest conv kernels per bucket shape (VAE caching)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        self.dtype = torch.bfloat16 if self.device.type == "cuda" and self.t.get("mixed_precision") != "fp32" else torch.float32
        if os.environ.get("VENNO_DTYPE") == "bf16":  # CPU smoke tests with real-size models
            self.dtype = torch.bfloat16
        self.adapter = make_adapter(run, self.device, self.dtype)
        self.output_dir = run["output_dir"]
        self.name = self.t.get("output_name") or "lora"
        self.samples_dir = os.path.join(self.output_dir, "samples")
        os.makedirs(self.samples_dir, exist_ok=True)
        cache_root = run.get("cache_dir") or os.path.join(self.output_dir, "cache")
        self.latents = LatentCache(cache_root, self.adapter.cache_key)
        self.texts = TextCache(cache_root, self.adapter.cache_key, self.adapter.text_cache_tag)
        self.seed = intv(self.t.get("seed"), 42)
        self.live_text = self.adapter.supports_live_text and not self.t.get("cache_te", True)
        self.variants = max(1, intv(self.t.get("te_variants"), 8))

    # ------------------------------------------------------------- dataset
    def prepare_dataset(self):
        folders = [D.Folder.from_job(f) for f in self.d["folders"] if f.get("enabled", True) and str(f.get("path", "")).strip()]
        if not folders:
            raise ValueError("No enabled dataset folders.")
        self.items, missing = D.scan_folders(folders, self.d.get("caption_ext") or ".txt", self.adapter.caption_first_line_only)
        if not self.items:
            raise ValueError("The dataset folders contain no images.")
        log(f"Dataset: {len(self.items)} images in {len(folders)} folder(s)" + (f", {len(missing)} without captions" if missing else ""))
        if missing:
            os.makedirs(self.output_dir, exist_ok=True)
            listing = os.path.join(self.output_dir, "uncaptioned.txt")
            with open(listing, "w", encoding="utf-8") as f:
                f.write("\n".join(missing) + "\n")
            for p in missing[:10]:
                log(f"  no caption: {p}")
            more = f"{len(missing) - 10} more; " if len(missing) > 10 else ""
            log(f"  ({more}full list in {listing}; they train with an empty caption. "
                "Dataset tab → click 'uncaptioned' on a folder to caption them.)")
        D.read_sizes(self.items)
        self.resolutions = int_list(self.d.get("resolutions"), [1024])
        self.batch_sizes = int_list(self.d.get("batch_sizes"), [1])
        mult = self.adapter.pixel_multiple
        step = max(mult, intv(self.d.get("bucket_steps"), 64) // mult * mult)
        D.assign_buckets(
            self.items, self.resolutions, step,
            intv(self.d.get("min_bucket"), 512), intv(self.d.get("max_bucket"), 1536),
            self.d.get("enable_bucket", True), self.d.get("bucket_no_upscale", True),
        )
        buckets = {b for i in self.items for b in i.buckets.values()}
        log(f"Resolutions {self.resolutions}, batch sizes {self.batch_sizes}, {len(buckets)} buckets")
        self.batches_per_epoch = len(D.epoch_batches(self.items, self.batch_sizes, len(self.resolutions), random.Random(0)))
        accum = max(1, intv(self.t.get("grad_accum"), 1))
        self.accum = accum
        self.steps_per_epoch = max(1, math.ceil(self.batches_per_epoch / accum))
        if self.t.get("duration_unit") == "steps":
            self.total_steps = max(1, intv(self.t.get("max_steps"), 1000))
            self.total_epochs = math.ceil(self.total_steps / self.steps_per_epoch)
        else:
            self.total_epochs = max(1, intv(self.t.get("max_epochs"), 10))
            self.total_steps = self.total_epochs * self.steps_per_epoch
        log(f"{self.batches_per_epoch} batches/epoch, {self.steps_per_epoch} steps/epoch, {self.total_steps} total steps ({self.total_epochs} epochs)")

    # --------------------------------------------------------------- cache
    def cache_latents(self):
        todo = []
        for item in self.items:
            for bucket in set(item.buckets.values()):
                if not self.latents.has(item.image, bucket, item.folder.flip):
                    todo.append((item, bucket))
        if not todo:
            log("Latents: all cached")
            return
        log(f"Caching latents for {len(todo)} image/bucket pairs...")
        self.adapter.load_vae()
        bs = max(1, intv(self.t.get("vae_batch"), 2))
        groups = {}
        for item, bucket in todo:
            groups.setdefault(bucket, []).append(item)
        jobs = [(bucket, items[k:k + bs]) for bucket, items in groups.items() for k in range(0, len(items), bs)]
        progress = Progress(len(todo), desc="caching latents")
        done = 0

        def load(job):
            bucket, items = job
            imgs = [D.load_bucketed_image(i.image, bucket) for i in items]
            arr = np.stack([np.asarray(im, dtype=np.float32) for im in imgs]) / 127.5 - 1.0
            return bucket, items, torch.from_numpy(arr).permute(0, 3, 1, 2)

        with ThreadPoolExecutor(max_workers=max(1, intv(self.t.get("workers"), 4))) as pool:
            for bucket, items, pixels in pool.map(load, jobs):
                check_stop()
                lat = self.adapter.encode_images(pixels)
                flip_lat = None
                if any(i.folder.flip for i in items):
                    flip_lat = self.adapter.encode_images(torch.flip(pixels, dims=[3]))
                for k, item in enumerate(items):
                    self.latents.save(item.image, bucket, lat[k], flip_lat[k] if flip_lat is not None and item.folder.flip else None)
                done += len(items)
                progress.update(done)
        progress.close()
        self.adapter.unload_vae()

    def caption_texts(self, item):
        """Every caption string training may use for this item (cached mode)."""
        n = self.variants if item.folder.has_tag_ops else 1
        return [D.caption_variant(item, k, n, self.seed) for k in range(n)]

    def sample_prompts(self):
        prompts = [p for p in self.s.get("prompts", []) if p.get("enabled", True) and str(p.get("text", "")).strip()]
        return prompts if self.s.get("enabled", True) else []

    def cache_text(self):
        needed = {""}  # caption dropout
        if not self.live_text:
            for item in self.items:
                needed.update(self.caption_texts(item))
        for p in self.sample_prompts():
            needed.add(p["text"].strip())
        needed.add((self.s.get("negative") or "").strip())
        todo = sorted(t for t in needed if not self.texts.has(t))
        if self.live_text:
            self.adapter.load_text_encoder()
        elif hasattr(self.adapter, "load_tokenizers"):
            self.adapter.load_tokenizers()
        if not todo:
            log("Text embeddings: all cached")
            return
        log(f"Caching {len(todo)} text embeddings...")
        if not self.live_text:
            self.adapter.load_text_encoder()
        # similar lengths together: less padding (Anima), equal-length batches (MOSS)
        todo.sort(key=self.adapter.text_sort_key)
        progress = Progress(len(todo), desc="caching text")
        bs = self.adapter.text_chunk
        for k in range(0, len(todo), bs):
            check_stop()
            chunk = todo[k:k + bs]
            for text, cond in zip(chunk, self.adapter.encode_text(chunk)):
                self.texts.save(text, cond)
            progress.update(min(len(todo), k + bs))
        progress.close()
        if not self.live_text:
            self.adapter.unload_text_encoder()

    # ---------------------------------------------------------------- model
    def setup_model(self):
        log("Loading DiT...")
        dit = self.adapter.load_dit()
        if self.t.get("fp8_base"):
            if self.device.type != "cuda":
                log("fp8 base weights need CUDA; skipped")
            else:
                n = quantize_modules(dit, self.adapter.fp8_prefixes(), self.dtype)
                log(f"Stored {n} base Linear weights as fp8")
                free_memory()
        ignore = [p for p in re.split(r"[,\s]+", self.n.get("ignore_layers") or "") if p]
        targets = self.adapter.lora_targets(ignore)
        rank = max(1, intv(self.n.get("dim"), 16))
        alpha = num(self.n.get("alpha"), rank)
        self.network = LoRANetwork(targets, rank, alpha, num(self.n.get("dropout"), 0.0)).to(self.device)
        self.network.apply_to()
        log(f"LoRA: {len(targets)} modules, rank {rank}, alpha {alpha:g}, "
            f"{sum(p.numel() for p in self.network.parameters()):,} trainable parameters")
        init = (self.n.get("init_weights") or "").strip()
        if init:
            sd = self.adapter.import_lora(load_file(init))
            missing = self.network.load_internal_state_dict(sd, strict=False)
            log(f"Initialized LoRA from {init}" + (f" ({len(missing)} modules not in file)" if missing else ""))
        if self.t.get("gradient_checkpointing", True):
            skip = max(0, intv(self.t.get("ckpt_skip_blocks"), 0))
            n = self.adapter.enable_gradient_checkpointing(skip)
            if skip:
                log(f"Gradient checkpointing on {n} blocks ({skip} run without it)")
        if self.t.get("compile"):
            self.compile_blocks()
        params = self.network.trainable_parameters()
        self.optimizer = create_optimizer(
            self.t.get("optimizer", "AdamW8bit"), params, num(self.t.get("lr"), 1e-4),
            self.t.get("weight_decay"), self.t.get("optimizer_args", ""),
        )
        self.scheduler = create_scheduler(
            self.t.get("lr_scheduler", "constant"), self.optimizer, self.total_steps,
            intv(self.t.get("lr_warmup_steps"), 0), intv(self.t.get("lr_cycles"), 1),
        )

    def compile_blocks(self):
        try:
            import torch._dynamo as dynamo

            dynamo.config.suppress_errors = True  # fall back to eager instead of crashing
            for name in ("cache_size_limit", "recompile_limit"):
                if hasattr(dynamo.config, name):
                    setattr(dynamo.config, name, 64)  # one graph per bucket shape
            n = self.adapter.compile_blocks()
            log(f"torch.compile enabled for {n} blocks (the first steps of each new bucket shape are slow)")
        except Exception as e:
            log(f"torch.compile unavailable ({e}); continuing without it")

    # ------------------------------------------------------------- batches
    def iter_groups(self, batches, epoch, rng):
        """Gradient-accumulation groups of loaded batches; the next group is read
        from the cache in a background thread while the GPU trains on this one."""
        groups = [batches[k:k + self.accum] for k in range(0, len(batches), self.accum)]

        def load(group):
            return [self.load_batch(samples, epoch, rng) for samples in group]

        if not groups:
            return
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(load, groups[0])
            for i in range(len(groups)):
                loaded = pending.result()
                if i + 1 < len(groups):
                    pending = pool.submit(load, groups[i + 1])
                yield loaded

    def vram_note(self):
        if self.device.type != "cuda":
            return None
        return torch.cuda.max_memory_allocated() / 1e9

    def load_batch(self, samples, epoch, rng):
        latents, conds, weights = [], [], []
        for s in samples:
            item = s.item
            flip = item.folder.flip and rng.random() < 0.5
            latents.append(self.latents.load(item.image, s.bucket, flip))
            if self.live_text:
                conds.append(D.caption_live(item, epoch, rng))
            else:
                if D.is_dropped(item.folder, epoch, rng):
                    text = ""
                else:
                    texts = self.caption_texts(item)
                    text = texts[rng.randrange(len(texts))]
                conds.append(self.texts.load(text))
            weights.append(item.folder.loss_weight)
        if self.live_text:
            conds = self.adapter.encode_text(conds)
        x = torch.stack(latents)
        if self.device.type == "cuda":
            x = x.pin_memory()
        x = x.to(self.device, torch.float32, non_blocking=True)
        return x, self.adapter.collate_text(conds), torch.tensor(weights, device=self.device)

    def sample_sigmas(self, n, rng_t):
        method = self.t.get("timestep", "logit_normal")
        if method == "logit_normal":
            t = torch.sigmoid(torch.randn(n, generator=rng_t))
        else:
            t = torch.rand(n, generator=rng_t)
        shift = num(self.t.get("flow_shift"), 1.0)
        if shift != 1.0:
            t = shift * t / (1 + (shift - 1) * t)
        return t.clamp(1e-5, 1 - 1e-5)

    def train_step(self, x, text, weights, rng_t):
        noise = torch.randn(x.shape, generator=rng_t).to(x.device)
        sig = self.sample_sigmas(x.shape[0], rng_t).to(x.device)
        noisy = (1 - sig.view(-1, 1, 1, 1)) * x + sig.view(-1, 1, 1, 1) * noise
        target = noise - x
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            pred = self.adapter.predict(noisy.to(self.dtype), sig.to(self.dtype), text)
        if self.t.get("loss_type") == "mae":
            loss = (pred.float() - target).abs()
        else:
            loss = (pred.float() - target) ** 2
        loss = loss.mean(dim=list(range(1, loss.ndim))) * weights
        return loss.mean()

    # ---------------------------------------------------------- save/load
    def save_lora(self, path):
        dtype = {"fp16": torch.float16, "float16": torch.float16, "fp32": torch.float32, "float": torch.float32,
                 "float32": torch.float32}.get(self.t.get("save_precision"), torch.bfloat16)
        sd = {k: v.to("cpu", dtype if v.ndim else torch.float32).contiguous() for k, v in self.adapter.export_lora(self.network.internal_state_dict()).items()}
        meta = {
            "ss_network_dim": str(self.n.get("dim")), "ss_network_alpha": str(self.n.get("alpha")),
            "ss_base_model_version": self.adapter.cache_key, "ss_output_name": self.name,
            "venno_trainer": "1", "modelspec.architecture": self.adapter.cache_key + "/lora",
        }
        tmp = path + ".tmp"
        save_file(sd, tmp, metadata=meta)
        os.replace(tmp, path)
        log(f"Saved {os.path.basename(path)}")

    def state_dir(self, tag):
        return os.path.join(self.output_dir, f"{self.name}-{tag}-state")

    def save_state(self, tag, step, epoch, rng, rng_t):
        d = self.state_dir(tag)
        tmp = d + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        save_file({k: v.cpu().contiguous() for k, v in self.network.internal_state_dict().items()}, os.path.join(tmp, "lora.safetensors"))
        torch.save({"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                    "rng": rng.getstate(), "rng_t": rng_t.get_state()}, os.path.join(tmp, "state.pt"))
        with open(os.path.join(tmp, "train_state.json"), "w") as f:
            json.dump({"step": step, "epoch": epoch, "total_steps": self.total_steps, "time": time.time()}, f)
        shutil.rmtree(d, ignore_errors=True)
        os.replace(tmp, d)
        keep = intv(self.t.get("keep_last_n_states"), 1)
        if keep > 0:
            states = sorted(self.list_states(), key=os.path.getmtime)
            for old in states[:-keep]:
                shutil.rmtree(old, ignore_errors=True)

    def list_states(self):
        if not os.path.isdir(self.output_dir):
            return []
        return [os.path.join(self.output_dir, n) for n in os.listdir(self.output_dir)
                if n.startswith(self.name + "-") and n.endswith("-state")
                and os.path.isfile(os.path.join(self.output_dir, n, "train_state.json"))]

    def find_resume(self):
        explicit = (self.n.get("resume_path") or "").strip()
        if explicit:
            return explicit
        if self.n.get("auto_resume"):
            states = sorted(self.list_states(), key=os.path.getmtime)
            return states[-1] if states else None
        return None

    def load_state(self, d, rng, rng_t):
        self.network.load_internal_state_dict(load_file(os.path.join(d, "lora.safetensors")))
        st = torch.load(os.path.join(d, "state.pt"), map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(st["optimizer"])
        self.scheduler.load_state_dict(st["scheduler"])
        # a learning rate edited before resuming takes effect
        lr = num(self.t.get("lr"), 1e-4)
        self.scheduler.base_lrs = [lr for _ in self.scheduler.base_lrs]
        for group in self.optimizer.param_groups:
            group["initial_lr"] = lr
        rng.setstate(st["rng"])
        rng_t.set_state(st["rng_t"])
        with open(os.path.join(d, "train_state.json")) as f:
            info = json.load(f)
        log(f"Resumed from {d} at step {info['step']} (epoch {info['epoch']})")
        return info["step"], info["epoch"]

    def prune_saves(self, pattern):
        keep = intv(self.t.get("keep_last_n"), 0)
        if keep <= 0:
            return
        rx = re.compile(re.escape(self.name) + pattern + r"\.safetensors$")
        files = sorted((f for f in os.listdir(self.output_dir) if rx.match(f)),
                       key=lambda f: os.path.getmtime(os.path.join(self.output_dir, f)))
        for f in files[:-keep]:
            os.remove(os.path.join(self.output_dir, f))

    # -------------------------------------------------------------- samples
    def generate_samples(self, tag):
        prompts = self.sample_prompts()
        if not prompts:
            return
        log(f"Generating {len(prompts)} sample(s)...")
        self.adapter.load_vae()
        neg_text = (self.s.get("negative") or "").strip()
        uncond = self.texts.load(neg_text)
        was_training = self.adapter.dit.training
        self.adapter.dit.eval()
        self.network.eval()
        shift = num(self.s.get("shift"), self.adapter.default_sample_shift)
        for i, p in enumerate(prompts):
            text = p["text"].strip()
            w = intv(p.get("width"), intv(self.s.get("width"), 1024))
            h = intv(p.get("height"), intv(self.s.get("height"), 1024))
            steps = intv(p.get("steps"), intv(self.s.get("steps"), 28))
            cfg = num(p.get("cfg"), num(self.s.get("cfg"), 4.0))
            seed = intv(p.get("seed"), intv(self.s.get("seed"), 42))
            img = self.adapter.sample(self.texts.load(text), uncond, w, h, steps, cfg, seed, shift)
            info = PngInfo()
            info.add_text("parameters", f"{text}\nNegative prompt: {neg_text}\nSteps: {steps}, CFG scale: {cfg}, Seed: {seed}, Size: {w}x{h}, Shift: {shift}")
            img.save(os.path.join(self.samples_dir, f"{self.name}_{tag}_{i:02d}_{seed}.png"), pnginfo=info)
        self.adapter.unload_vae()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        if was_training:
            self.adapter.dit.train()
            self.network.train()
        free_memory()

    # ------------------------------------------------------------------ run
    def train(self):
        self.prepare_dataset()
        self.cache_latents()
        self.cache_text()
        self.setup_model()
        rng = random.Random(self.seed)
        rng_t = torch.Generator("cpu").manual_seed(self.seed)
        torch.manual_seed(self.seed)
        step, start_epoch = 0, 0
        resume = self.find_resume()
        if resume:
            step, start_epoch = self.load_state(resume, rng, rng_t)

        by_epochs = self.t.get("duration_unit") != "steps"
        save_every = max(1, intv(self.t.get("save_every_epochs" if by_epochs else "save_every_steps"), 1))
        sample_every = max(1, intv(self.s.get("every_epochs" if by_epochs else "every_steps"), 1))
        max_norm = num(self.t.get("max_grad_norm"), 1.0)

        if step == 0 and self.s.get("sample_first"):
            self.generate_samples("000000")

        self.adapter.dit.train()
        self.network.train()
        start_step = step
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        progress = Progress(self.total_steps, start=step)
        recent = []
        epoch = start_epoch
        params = self.network.trainable_parameters()
        while step < self.total_steps and not _stop_requested:
            batches = D.epoch_batches(self.items, self.batch_sizes, len(self.resolutions), rng)
            # resumed mid-epoch: skip the batches this epoch already used
            skip = (step - epoch * self.steps_per_epoch) * self.accum if step > epoch * self.steps_per_epoch else 0
            batches = batches[skip:]
            groups = self.iter_groups(batches, epoch, rng)
            for group in groups:
                total_loss = 0.0
                for x, text, weights in group:
                    loss = self.train_step(x, text, weights, rng_t) / len(group)
                    loss.backward()
                    total_loss += loss.detach().item()
                if not math.isfinite(total_loss):
                    raise RuntimeError(f"Loss became {total_loss} at step {step + 1}; lower the learning rate.")
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, max_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                step += 1
                recent = (recent + [total_loss])[-50:]
                progress.update(step, sum(recent) / len(recent), self.vram_note())
                if step == start_step + 2 and self.device.type == "cuda":
                    total = torch.cuda.get_device_properties(0).total_memory / 1e9
                    log(f"Peak VRAM while training: {self.vram_note():.1f} of {total:.1f} GB")

                if not by_epochs:
                    if step % save_every == 0 and step < self.total_steps:
                        self.save_lora(os.path.join(self.output_dir, f"{self.name}-step{step:08d}.safetensors"))
                        self.prune_saves(r"-step\d{8}")
                        self.save_state(f"step{step:08d}", step, epoch, rng, rng_t)
                    if self.s.get("enabled", True) and step % sample_every == 0 and step < self.total_steps:
                        self.generate_samples(f"{step:06d}")
                if step >= self.total_steps or _stop_requested:
                    groups.close()
                    break
            else:
                epoch += 1  # only a fully used epoch counts, so a resume picks up mid-epoch
            if step >= self.total_steps or _stop_requested:
                if by_epochs and step >= self.total_steps and not _stop_requested:
                    epoch = self.total_epochs
                break
            if by_epochs and epoch < self.total_epochs:
                if epoch % save_every == 0:
                    self.save_lora(os.path.join(self.output_dir, f"{self.name}-{epoch:06d}.safetensors"))
                    self.prune_saves(r"-\d{6}")
                    self.save_state(f"{epoch:06d}", step, epoch, rng, rng_t)
                if self.s.get("enabled", True) and epoch % sample_every == 0:
                    self.generate_samples(f"e{epoch:06d}")
        progress.close()

        if _stop_requested:
            self.save_state(f"step{step:08d}", step, epoch, rng, rng_t)
            self.save_lora(os.path.join(self.output_dir, f"{self.name}-step{step:08d}.safetensors"))
            log("Stopped early; resume with 'Auto-resume' to continue.")
            return
        self.save_lora(os.path.join(self.output_dir, f"{self.name}.safetensors"))
        self.save_state("final", step, epoch, rng, rng_t)
        if self.s.get("enabled", True):
            self.generate_samples("final")
        log("Training finished.")


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m venno.train <run.json>")
        return 2
    with open(argv[0], encoding="utf-8") as f:
        run = json.load(f)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    try:
        Trainer(run).train()
    except StopTraining:
        log("Stopped before training started; cached data is kept for the next run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
