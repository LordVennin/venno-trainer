"""Anima: Cosmos-Predict2 DiT + LLM adapter, Qwen3-0.6B text encoder (+ T5
token ids for the adapter), Wan 2.1 VAE. Mirrors anima_train_network.py from
the Anima Standalone Trainer: rectified flow, timestep = sigma in [0, 1],
captions padded to 512 Qwen3 / T5 tokens, kohya "lora_unet_*" LoRA keys."""

import os
from typing import Dict, List

import torch
from PIL import Image
from safetensors.torch import load_file

from .. import base
from ..base import split_lora_key
from . import dit as D
from .vae import WanVAE_

HERE = os.path.dirname(__file__)
QWEN3_CONFIG_DIR = os.path.join(HERE, "configs", "qwen3_06b")
T5_TOKENIZER = os.path.join(HERE, "configs", "t5_old", "tokenizer.json")
MAX_TOKENS = 512
VAE_CONFIG = dict(
    dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
    temperal_downsample=[False, True, True], dropout=0.0,
)


class AnimaAdapter(base.ModelAdapter):
    cache_key = "anima"
    text_cache_tag = "anima-qwen3-t5-v1"
    pixel_multiple = 16  # VAE /8 and 2x2 patches
    caption_first_line_only = True  # sd-scripts reads the first caption line
    supports_live_text = True
    default_sample_shift = 1.0

    def __init__(self, run, device, dtype):
        super().__init__(run, device, dtype)
        self.text_encoder = self.qwen_tok = self.t5_tok = None
        self.vae = self.vae_scale = None
        self.dit = None
        self.vae_dtype = dtype

    def _path(self, key, label):
        p = (self.paths.get(key) or "").strip()
        if not p:
            raise ValueError(f"Set the Anima {label} path in Global Settings.")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Anima {label} not found: {p}")
        return p

    # text ---------------------------------------------------------------
    def load_tokenizers(self):
        from transformers import AutoTokenizer, PreTrainedTokenizerFast

        if self.qwen_tok is None:
            self.qwen_tok = AutoTokenizer.from_pretrained(QWEN3_CONFIG_DIR, local_files_only=True)
            if self.qwen_tok.pad_token is None:
                self.qwen_tok.pad_token = self.qwen_tok.eos_token
            # same ids as the trainer's T5TokenizerFast(spiece.model, tokenizer.json)
            self.t5_tok = PreTrainedTokenizerFast(
                tokenizer_file=T5_TOKENIZER, eos_token="</s>", pad_token="<pad>", unk_token="<unk>"
            )

    def load_text_encoder(self):
        from transformers import Qwen3Config, Qwen3Model

        self.load_tokenizers()
        path = self._path("qwen3_path", "Qwen3 text encoder")
        if os.path.isdir(path):
            config = Qwen3Config.from_pretrained(path)
        else:
            config = Qwen3Config.from_pretrained(QWEN3_CONFIG_DIR)
        sd = base.load_safetensors_dir(path, key_filter=lambda k: not k.startswith("lm_head."))
        sd = base.strip_prefixes(sd, ("model.",))
        with base.meta_device():
            model = Qwen3Model(config)
        model.load_state_dict({k: v.to(self.dtype) for k, v in sd.items()}, strict=False, assign=True)
        base.rebuild_rotary(model, config)
        base.assert_materialized(model, "Qwen3 text encoder")
        model.config.use_cache = False
        self.text_encoder = model.eval().requires_grad_(False).to(self.device)

    def unload_text_encoder(self):
        self.text_encoder = None
        base.free_memory()

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> List[Dict[str, torch.Tensor]]:
        enc = self.qwen_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS)
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        if ids.shape[1] == 0:  # every caption empty
            hidden = torch.zeros(len(texts), 0, self.text_encoder.config.hidden_size)
        else:
            with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
                hidden = self.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        out = []
        for i, text in enumerate(texts):
            n = int(mask[i].sum())
            t5 = self.t5_tok(text, truncation=True, max_length=MAX_TOKENS)["input_ids"]
            out.append({
                # right padding + causal attention: the valid states equal an unpadded encode
                "qwen": hidden[i, :n].to(torch.bfloat16).cpu(),
                "t5": torch.tensor(t5, dtype=torch.int32),
            })
        return out

    def collate_text(self, conds):
        # always padded to 512 like the original trainer: padded T5 positions
        # are zero vectors the DiT still attends to
        b = len(conds)
        qwen = torch.zeros(b, MAX_TOKENS, 1024, dtype=self.dtype)
        qmask = torch.zeros(b, MAX_TOKENS, dtype=torch.int64)
        t5 = torch.zeros(b, MAX_TOKENS, dtype=torch.int64)
        t5mask = torch.zeros(b, MAX_TOKENS, dtype=torch.int64)
        for i, c in enumerate(conds):
            n = c["qwen"].shape[0]
            if n:
                qwen[i, :n] = c["qwen"].to(self.dtype)
                qmask[i, :n] = 1
            m = c["t5"].shape[0]
            t5[i, :m] = c["t5"].long()
            t5mask[i, :m] = 1
        dev = self.device
        return {"qwen": qwen.to(dev), "qmask": qmask.to(dev), "t5": t5.to(dev), "t5mask": t5mask.to(dev)}

    # vae ----------------------------------------------------------------
    def load_vae(self):
        path = self._path("vae_path", "VAE")
        sd = load_file(path) if path.endswith(".safetensors") else torch.load(path, map_location="cpu", weights_only=True)
        with base.meta_device():
            vae = WanVAE_(**VAE_CONFIG)
        vae.load_state_dict(sd, assign=True)
        self.vae = vae.eval().requires_grad_(False).to(self.device, dtype=self.vae_dtype)
        mean = torch.tensor(D.ANIMA_VAE_MEAN, device=self.device, dtype=self.vae_dtype)
        std = torch.tensor(D.ANIMA_VAE_STD, device=self.device, dtype=self.vae_dtype)
        self.vae_scale = [mean, 1.0 / std]

    def unload_vae(self):
        self.vae = None
        base.free_memory()

    @torch.no_grad()
    def encode_images(self, pixels):
        x = pixels.to(self.device, self.vae_dtype).unsqueeze(2)
        return self.vae.encode(x, self.vae_scale)[:, :, 0].float().cpu()

    @torch.no_grad()
    def decode_latents(self, latents):
        z = latents.to(self.device, self.vae_dtype).unsqueeze(2)
        return self.vae.decode(z, self.vae_scale)[:, :, 0].float()

    # dit ----------------------------------------------------------------
    def load_dit(self):
        path = self._path("dit_path", "DiT")
        sd = base.strip_prefixes(load_file(path), ("net.", "model.diffusion_model."))
        config = D.get_dit_config(sd)
        config["use_llm_adapter"] = "llm_adapter.out_proj.weight" in sd
        if not config["use_llm_adapter"]:
            raise ValueError("This Anima checkpoint has no LLM adapter weights (llm_adapter.*)")
        with base.meta_device():
            dit = D.MiniTrainDIT(**config)
        dit.load_state_dict({k: v.to(self.dtype) for k, v in sd.items()}, strict=False, assign=True)
        _rebuild_dit_buffers(dit)
        base.assert_materialized(dit, "Anima DiT")
        if self.job["training"].get("flash_attn") and D.FLASH_ATTN_AVAILABLE:
            dit.set_flash_attn(True)
        self.dit = dit.requires_grad_(False).to(self.device)
        return self.dit

    def fp8_prefixes(self):
        return ["blocks."]

    def lora_targets(self, ignore):
        targets = []
        for i, block in enumerate(self.dit.blocks):
            for child, module in block.named_modules():
                if isinstance(module, torch.nn.Linear):
                    name = f"blocks.{i}.{child}"
                    if not any(p in name + "." for p in ignore):
                        targets.append((name, module))
        return targets

    def enable_gradient_checkpointing(self):
        self.dit.enable_gradient_checkpointing()

    def predict(self, noisy, sigmas, text):
        b, _, h, w = noisy.shape
        padding = torch.zeros(b, 1, h, w, dtype=noisy.dtype, device=noisy.device)
        out = self.dit(
            noisy.unsqueeze(2), sigmas, text["qwen"],
            padding_mask=padding, source_attention_mask=text["qmask"],
            t5_input_ids=text["t5"], t5_attn_mask=text["t5mask"],
        )
        return out[:, :, 0]

    @torch.no_grad()
    def sample(self, cond, uncond, width, height, steps, cfg, seed, shift):
        """Euler sampling as in anima_train_utils.do_sample, plus an optional
        sigma shift (1.0 = the original trainer's linear schedule)."""
        dit = self.dit
        width, height = max(64, width - width % 16), max(64, height - height % 16)
        c = self.collate_text([cond])
        emb = self._adapter_emb(c)
        use_cfg = cfg > 1.0 and uncond is not None
        if use_cfg:
            emb = torch.cat([emb, self._adapter_emb(self.collate_text([uncond]))])
        g = torch.Generator("cpu").manual_seed(int(seed))
        x = torch.randn(1, 16, 1, height // 8, width // 8, generator=g, dtype=torch.float32).to(self.device, self.dtype)
        sig = torch.linspace(1.0, 0.0, steps + 1)
        if shift and shift != 1.0:
            sig = shift * sig / (1 + (shift - 1) * sig)
        sig = sig.to(self.device, self.dtype)
        n = 2 if use_cfg else 1
        pad = torch.zeros(n, 1, height // 8, width // 8, device=self.device, dtype=self.dtype)
        for i in range(steps):
            t = sig[i].expand(n)
            out = dit(x.expand(n, -1, -1, -1, -1), t, emb, padding_mask=pad)
            if use_cfg:
                pos, neg = out.chunk(2)
                out = neg + cfg * (pos - neg)
            x = x + out * (sig[i + 1] - sig[i])
        return to_pil(self.decode_latents(x[:, :, 0]))

    def _adapter_emb(self, c):
        emb = self.dit.llm_adapter(
            source_hidden_states=c["qwen"], target_input_ids=c["t5"],
            target_attention_mask=c["t5mask"], source_attention_mask=c["qmask"],
        )
        emb[~c["t5mask"].bool()] = 0
        return emb

    # lora files ---------------------------------------------------------
    def export_lora(self, internal):
        # "blocks.0.self_attn.q_proj.lora_down.weight" -> "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight"
        out = {}
        for key, v in internal.items():
            name, suffix = split_lora_key(key)
            out[f"lora_unet_{name.replace('.', '_')}.{suffix}"] = v
        return out

    def import_lora(self, sd):
        lookup = {f"lora_unet_{n.replace('.', '_')}": n for n, _ in self.lora_targets([])}
        out = {}
        for key, v in sd.items():
            prefix, _, suffix = key.partition(".")
            if prefix in lookup:
                out[f"{lookup[prefix]}.{suffix}"] = v
        return out


def _rebuild_dit_buffers(dit):
    """Buffers computed in __init__ (RoPE ranges, adapter inv_freq) are not in
    the checkpoint; recompute them after the meta-device construction."""
    pe = dit.pos_embedder
    pe.to_empty(device="cpu", recurse=False)
    pe.reset_parameters()
    if getattr(dit, "extra_pos_embedder", None) is not None:
        dit.extra_pos_embedder.to_empty(device="cpu")
        dit.extra_pos_embedder.reset_parameters()
    if hasattr(dit, "llm_adapter"):
        rot = dit.llm_adapter.rotary_emb
        head_dim = rot.inv_freq.shape[0] * 2
        inv = 1.0 / (rot.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        rot.register_buffer("inv_freq", inv, persistent=False)


def to_pil(pixels):
    img = ((pixels[0].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
