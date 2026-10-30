"""MOSS-Image (OpenMOSS-Team/MOSS-Image-SFT / -Base): 42-block single-stream
DiT, Qwen3.5-9B text encoder (final hidden state of the chat-templated prompt
with thinking enabled, one unpadded forward per prompt), FLUX.2 VAE packed 2x2
to 128 latent channels and BatchNorm-normalized. Follows the vendor pipeline
and ai-toolkit's moss_image: timestep = sigma * 1000, target = noise - clean,
CFG negative branch skips DiT block 22, LoRA saved in ComfyUI's
diffusion_model.layers.N.proj.{q,k,v,o,...} layout."""

import contextlib
import json
import os
import signal
import threading
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image

from .. import base
from ..base import split_lora_key
from .transformer import MossImageParams, MossImageTransformer
from .vae import AutoEncoder, convert_diffusers_state_dict

VARIANT_REPOS = {
    "sft": "OpenMOSS-Team/MOSS-Image-SFT",
    "base": "OpenMOSS-Team/MOSS-Image-Base",
}
DOWNLOAD_PATTERNS = ["model_index.json", "transformer/*", "text_encoder/*", "tokenizer/*", "vae/*"]
MAX_TEXT_TOKENS = 4096
UNCOND_SKIP_LAYERS = (22,)
NUM_TRAIN_TIMESTEPS = 1000


def moss_sigmas(num_steps, shift=5.0, training_shift=1.0, total=NUM_TRAIN_TIMESTEPS):
    """The vendor MOSSImageScheduler grid: shifted, snapped to the nearest
    training sigma, duplicates removed, terminal 0 appended."""

    def grid(count, s):
        lower = s / (total + s - 1)
        values = torch.linspace(1, lower, count)
        return s * values / (1 + (s - 1) * values)

    sigmas = grid(num_steps, shift)
    training = grid(total, training_shift).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    sigmas = torch.unique_consecutive(training[nearest])
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def resolve_model_root(paths, variant, models_dir, hf_token=None):
    """A local diffusers-layout folder, a HF repo id, or (blank) the variant's
    vendor repo, downloaded once into models_dir."""
    root = (paths.get("model_path") or "").strip()
    if root and os.path.isdir(root):
        return root
    repo = root or VARIANT_REPOS.get(variant, VARIANT_REPOS["sft"])
    if repo.count("/") != 1:
        raise FileNotFoundError(f"MOSS-Image model folder not found: {repo}")
    target = os.path.join(models_dir, repo.replace("/", "__"))
    if download_complete(target):
        print(f"Model files: {target}", flush=True)
        return target
    from huggingface_hub import snapshot_download

    print(f"Downloading {repo} into {target} (~27 GB the first time; resumes if interrupted)", flush=True)
    with _download_monitor(target), _exit_on_stop():
        root = snapshot_download(repo, allow_patterns=DOWNLOAD_PATTERNS, local_dir=target, token=hf_token or None)
    print("Download complete", flush=True)
    return root


def download_complete(root):
    """Whether a diffusers-layout MOSS folder has every file the trainer reads.
    The Hub client writes to a hidden *.incomplete file and renames it when
    done, so a shard under its final name is a finished shard."""
    need = ["model_index.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]
    for part in ("transformer", "text_encoder", "vae"):
        folder = os.path.join(root, part)
        need.append(f"{part}/config.json")
        if not os.path.isdir(folder):
            return False
        index = [n for n in os.listdir(folder) if n.endswith(".safetensors.index.json")]
        if index:
            try:
                with open(os.path.join(folder, index[0]), encoding="utf-8") as f:
                    shards = set(json.load(f)["weight_map"].values())
            except (OSError, ValueError, KeyError):
                return False
            need += [f"{part}/{s}" for s in shards]
        elif not any(n.endswith(".safetensors") for n in os.listdir(folder)):
            return False
    return all(os.path.isfile(os.path.join(root, n)) for n in need)


def _folder_bytes(root):
    total = 0
    for dirpath, _, names in os.walk(root):
        for n in names:
            try:
                total += os.path.getsize(os.path.join(dirpath, n))
            except OSError:
                pass
    return total


@contextlib.contextmanager
def _download_monitor(target, every=20.0):
    """Log how much has arrived every `every` seconds, so a long download (or
    one waiting on the network) does not look like a hang."""
    stop = threading.Event()

    def report():
        while not stop.wait(every):
            print(f"Downloading... {_folder_bytes(target) / 1e9:.1f} GB in {target}", flush=True)

    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()


@contextlib.contextmanager
def _exit_on_stop():
    """Stop during a download exits at once: nothing needs saving, and the
    partial files resume on the next run. (The trainer's normal stop handler
    only sets a flag, which a download never checks.)"""

    def handler(signum, frame):
        print("Stopped during the model download; it resumes on the next run.", flush=True)
        os._exit(130)

    try:
        previous = {s: signal.signal(s, handler) for s in (signal.SIGINT, signal.SIGTERM)}
    except ValueError:  # not the main thread
        previous = {}
    try:
        yield
    finally:
        for s, h in previous.items():
            signal.signal(s, h)


class MossAdapter(base.ModelAdapter):
    cache_key = "moss_image"
    text_cache_tag = "moss-qwen35-chat-thinking-v1"
    pixel_multiple = 16  # f8 VAE + 2x2 packing
    caption_first_line_only = False
    supports_live_text = False  # the 9B text encoder cannot share the GPU with the DiT
    default_sample_shift = 5.0

    def __init__(self, run, device, dtype):
        super().__init__(run, device, dtype)
        self.root = None
        self.tokenizer = self.text_encoder = None
        self.vae = None
        self.dit = None
        self.vae_dtype = torch.float32  # the vendor decodes in fp32

    def fetch_models(self):
        self.model_root()

    def model_root(self):
        if self.root is None:
            self.root = resolve_model_root(
                self.paths, self.job.get("variant", "sft"),
                self.run.get("models_dir") or "models", self.run.get("hf_token"),
            )
        return self.root

    def _override(self, key):
        p = (self.paths.get(key) or "").strip()
        if p and not os.path.exists(p):
            raise FileNotFoundError(f"MOSS-Image {key} not found: {p}")
        return p

    # text ---------------------------------------------------------------
    def load_text_encoder(self):
        from transformers import AutoConfig, AutoTokenizer, Qwen3_5TextModel

        root = self.model_root()
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(root, "tokenizer"))
        te_path = self._override("te_path") or os.path.join(root, "text_encoder")
        config_dir = te_path if os.path.isdir(te_path) else os.path.join(root, "text_encoder")
        config = AutoConfig.from_pretrained(config_dir)
        config = getattr(config, "text_config", config)
        # language tower only: drop the vision tower, lm_head and mtp head
        dropped = ("visual.", "model.visual.", "lm_head.", "mtp.")
        sd = base.load_safetensors_dir(te_path, key_filter=lambda k: not k.startswith(dropped))
        sd = base.strip_prefixes(sd, ("model.language_model.", "language_model.", "model."))
        with base.meta_device():
            model = Qwen3_5TextModel(config)
        model.load_state_dict({k: v.to(self.dtype) for k, v in sd.items()}, strict=False, assign=True)
        base.rebuild_rotary(model, config)
        base.assert_materialized(model, "Qwen3.5 text encoder")
        self.text_encoder = model.eval().requires_grad_(False).to(self.device)

    def unload_text_encoder(self):
        self.text_encoder = None
        base.free_memory()

    text_chunk = 64  # with length-sorted captions, most chunks hold equal-length runs

    def _token_ids(self, text):
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True,
        )
        return self.tokenizer(prompt, truncation=True, max_length=MAX_TEXT_TOKENS)["input_ids"]

    def text_sort_key(self, text):
        return len(self._token_ids(text))

    @torch.no_grad()
    def encode_text(self, texts, max_batch=16, max_tokens=8192):
        """Unpadded forwards (padding leaks into the gated-delta recurrence), but
        prompts with the same token count share one batched forward."""
        ids = [self._token_ids(t) for t in texts]
        by_len = {}
        for i, seq in enumerate(ids):
            by_len.setdefault(len(seq), []).append(i)
        out = [None] * len(texts)
        for length, idx in by_len.items():
            per = max(1, min(max_batch, max_tokens // max(1, length)))
            for k in range(0, len(idx), per):
                chunk = idx[k:k + per]
                batch = torch.tensor([ids[i] for i in chunk], device=self.device)
                with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
                    states = self.text_encoder(input_ids=batch, use_cache=False).last_hidden_state
                for j, i in enumerate(chunk):
                    out[i] = {"embeds": states[j].to(torch.bfloat16).cpu()}
        return out

    def collate_text(self, conds):
        length = max(c["embeds"].shape[0] for c in conds)
        embeds = torch.stack([F.pad(c["embeds"].to(self.dtype), (0, 0, 0, length - c["embeds"].shape[0])) for c in conds])
        mask = torch.stack([torch.arange(length) < c["embeds"].shape[0] for c in conds])
        return {"embeds": embeds.to(self.device), "mask": mask.to(self.device)}

    # vae ----------------------------------------------------------------
    def load_vae(self):
        path = self._override("vae_path") or os.path.join(self.model_root(), "vae")
        sd = base.load_safetensors_dir(path)
        if any(k.startswith("encoder.down_blocks.") for k in sd):
            sd = convert_diffusers_state_dict(sd)
        params = AutoEncoder.aitk_config_from_state_dict(sd)
        with base.meta_device():
            vae = AutoEncoder(params)
        vae.load_state_dict(sd, assign=True, strict=False)
        if vae.bn.num_batches_tracked.is_meta:  # bookkeeping counter, not always saved
            vae.bn.num_batches_tracked = torch.tensor(0, dtype=torch.long)
        base.assert_materialized(vae, "FLUX.2 VAE")
        self.vae = vae.eval().requires_grad_(False).to(self.device, dtype=self.vae_dtype)

    def unload_vae(self):
        self.vae = None
        base.free_memory()

    @torch.no_grad()
    def encode_images(self, pixels):
        return self.vae.encode(pixels.to(self.device, self.vae_dtype)).float().cpu()

    @torch.no_grad()
    def decode_latents(self, latents):
        return self.vae.decode(latents.to(self.device, self.vae_dtype)).float()

    # dit ----------------------------------------------------------------
    def load_dit(self):
        path = self._override("transformer_path") or os.path.join(self.model_root(), "transformer")
        sd = base.load_safetensors_dir(path)
        sd = base.strip_prefixes(sd, ("model.diffusion_model.", "diffusion_model."))
        sd = MossImageTransformer.convert_state_dict_on_load(sd)
        params = MossImageTransformer.aitk_config_from_state_dict(sd)
        with base.meta_device():
            dit = MossImageTransformer(params)
        dit.load_state_dict({k: v.to(self.dtype) for k, v in sd.items()}, strict=True, assign=True)
        self.dit = dit.requires_grad_(False).to(self.device)
        return self.dit

    def fp8_prefixes(self):
        return ["layers."]

    def lora_targets(self, ignore):
        targets = []
        for i, layer in enumerate(self.dit.layers):
            for child in ("qkv", "o", "ffn_up_gate", "ffn_down"):
                name = f"layers.{i}.proj.{child}"
                if not any(p in name + "." for p in ignore):
                    targets.append((name, layer.proj[child]))
        return targets

    def enable_gradient_checkpointing(self, skip_last=0):
        """Checkpoint every block except the last `skip_last` (spare VRAM buys speed)."""
        self.dit.enable_gradient_checkpointing()
        n = len(self.dit.layers)
        self.dit.ckpt_skip = set(range(n - min(skip_last, n), n))
        return n - len(self.dit.ckpt_skip)

    def compile_blocks(self):
        blocks = list(self.dit.layers) + list(self.dit.text_embedder.blocks)
        for blk in blocks:
            blk.compile()
        return len(blocks)

    def predict(self, noisy, sigmas, text):
        return self.dit(noisy, sigmas * NUM_TRAIN_TIMESTEPS, text["embeds"], text["mask"])

    @torch.no_grad()
    def sample(self, cond, uncond, width, height, steps, cfg, seed, shift, uncond_skip=UNCOND_SKIP_LAYERS):
        width, height = width // 16 * 16, height // 16 * 16
        c = self.collate_text([cond])
        do_cfg = uncond is not None and cfg != 1.0
        if do_cfg:
            u = self.collate_text([uncond])
        g = torch.Generator("cpu").manual_seed(int(seed))
        # fp32 noise and Euler updates, as the vendor pipeline
        latents = torch.randn(1, self.dit.params.latent_channels, height // 16, width // 16, generator=g).to(self.device)
        sig = moss_sigmas(steps, shift=shift or self.default_sample_shift).to(self.device)
        for i in range(len(sig) - 1):
            t = (sig[i] * NUM_TRAIN_TIMESTEPS).reshape(1)
            pred = self.dit(latents, t, c["embeds"], c["mask"]).float()
            if do_cfg:
                pu = self.dit(latents, t, u["embeds"], u["mask"], skip_layers=uncond_skip).float()
                pred = pu + cfg * (pred - pu)
            latents = latents + (sig[i + 1] - sig[i]) * pred
        pixels = self.decode_latents(latents)
        img = ((pixels[0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        return Image.fromarray(img.permute(1, 2, 0).cpu().numpy())

    # lora files ---------------------------------------------------------
    # The module fuses q/k/v; ComfyUI's MOSS LoRA convention (and ai-toolkit's
    # saves) address q, k and v separately: up rows split, down/alpha shared.
    def export_lora(self, internal):
        out = {}
        for key, v in internal.items():
            name, suffix = split_lora_key(key)
            peft = {"lora_down.weight": "lora_A.weight", "lora_up.weight": "lora_B.weight", "alpha": "alpha"}[suffix]
            if name.endswith(".proj.qkv"):
                stem = name[: -len("qkv")]
                parts = v.chunk(3, dim=0) if suffix == "lora_up.weight" else (v, v, v)
                for p, t in zip("qkv", parts):
                    out[f"diffusion_model.{stem}{p}.{peft}"] = t.contiguous().clone()
            else:
                out[f"diffusion_model.{name}.{peft}"] = v
        return out

    def import_lora(self, sd):
        sd = base.strip_prefixes(sd, ("diffusion_model.", "transformer."))
        sd = {k.replace(".lora_A.weight", ".lora_down.weight").replace(".lora_B.weight", ".lora_up.weight"): v for k, v in sd.items()}
        out = dict(sd)
        for key in [k for k in sd if k.endswith(".proj.q.lora_down.weight")]:
            stem = key[: -len("q.lora_down.weight")]
            downs = [sd.get(f"{stem}{p}.lora_down.weight") for p in "qkv"]
            ups = [sd.get(f"{stem}{p}.lora_up.weight") for p in "qkv"]
            if any(t is None for t in downs + ups):
                continue
            if not all(torch.equal(downs[0], d) for d in downs[1:]):
                raise ValueError("LoRA has independent q/k/v down projections; only shared-down (our/ai-toolkit) files can be loaded")
            for p in "qkv":
                for s in ("lora_down.weight", "lora_up.weight", "alpha"):
                    out.pop(f"{stem}{p}.{s}", None)
            out[f"{stem}qkv.lora_down.weight"] = downs[0]
            out[f"{stem}qkv.lora_up.weight"] = torch.cat(ups, 0)
            alpha = sd.get(f"{stem}q.alpha")
            if alpha is not None:
                out[f"{stem}qkv.alpha"] = alpha
        return out
