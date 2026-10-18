"""Tiny random models saved in the same on-disk layouts as the real
checkpoints, plus small image datasets, for CPU tests."""

import json
import os
import shutil

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

TRAINER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOSS_TOKENIZER_REPO = "OpenMOSS-Team/MOSS-Image-SFT"


def make_images(folder, n, size=(64, 64), caption=lambda i: f"tag{i}, red, blue, green", seed=0):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        w, h = size if i % 2 == 0 else (size[0] + 32, size[1])
        arr = (rng.random((h, w, 3)) * 255).astype(np.uint8)
        arr[: h // 2] = [200, 40 + 20 * i, 40]
        Image.fromarray(arr).save(os.path.join(folder, f"img{i:03d}.png"))
        if caption is not None:
            with open(os.path.join(folder, f"img{i:03d}.txt"), "w") as f:
                f.write(caption(i))


def _save(sd, path, dtype=torch.bfloat16):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file({k: (v.to(dtype) if v.is_floating_point() else v).contiguous() for k, v in sd.items()}, path)


def make_anima(root):
    """net.-prefixed DiT (2048 channels, 1 block), Qwen3 folder (1 layer), Wan VAE."""
    from transformers import Qwen3Config, Qwen3Model

    from venno.models.anima import dit as D
    from venno.models.anima.adapter import QWEN3_CONFIG_DIR, VAE_CONFIG
    from venno.models.anima.vae import WanVAE_

    torch.manual_seed(0)
    cfg = dict(
        max_img_h=512, max_img_w=512, max_frames=128, in_channels=16, out_channels=16, patch_spatial=2,
        patch_temporal=1, concat_padding_mask=True, model_channels=2048, num_blocks=1, num_heads=16,
        crossattn_emb_channels=1024, pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
        min_fps=1, max_fps=30, use_adaln_lora=True, adaln_lora_dim=256, rope_h_extrapolation_ratio=4.0,
        rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0, extra_per_block_abs_pos_emb=False,
        rope_enable_fps_modulation=False, use_llm_adapter=True,
    )
    dit = D.MiniTrainDIT(**cfg)
    with torch.no_grad():
        for blk in dit.blocks:  # zero-init gates would make the block a no-op
            for seq in (blk.adaln_modulation_self_attn, blk.adaln_modulation_cross_attn, blk.adaln_modulation_mlp):
                seq[2].weight.normal_(0, 0.02)
    sd = {"net." + k: v for k, v in dit.state_dict().items() if "pos_embedder" not in k}
    _save(sd, os.path.join(root, "dit.safetensors"))

    qcfg = Qwen3Config.from_pretrained(QWEN3_CONFIG_DIR)
    qcfg.num_hidden_layers = 1
    qcfg.layer_types = qcfg.layer_types[:1]
    qcfg.intermediate_size = 512
    te = Qwen3Model(qcfg)
    te_dir = os.path.join(root, "qwen3")
    os.makedirs(te_dir, exist_ok=True)
    qcfg.save_pretrained(te_dir)
    _save({"model." + k: v for k, v in te.state_dict().items()}, os.path.join(te_dir, "model.safetensors"))

    vae = WanVAE_(**VAE_CONFIG)
    _save(vae.state_dict(), os.path.join(root, "vae.safetensors"), torch.float32)
    return {
        "dit_path": os.path.join(root, "dit.safetensors"),
        "qwen3_path": te_dir,
        "vae_path": os.path.join(root, "vae.safetensors"),
    }


def moss_text_config():
    from transformers import AutoConfig

    path = os.path.join(TRAINER, "tests", "fixtures", "moss_text_encoder_config.json")
    with open(path) as f:
        return json.load(f)


def fetch_moss_tokenizer(dest):
    """Real MOSS tokenizer (chat template + 248k vocab) from the Hub."""
    from huggingface_hub import snapshot_download

    p = snapshot_download(MOSS_TOKENIZER_REPO, allow_patterns=["tokenizer/*"], local_dir=dest)
    return os.path.join(p, "tokenizer")


def make_moss(root, tokenizer_dir, num_layers=3, text_dim=64):
    """Vendor diffusers layout: transformer/ with split q/k/v, text_encoder/
    with a multimodal config + language_model.* keys, tokenizer/, vae/."""
    from transformers import AutoConfig, Qwen3_5TextModel

    from venno.models.moss.transformer import MossImageParams, MossImageTransformer
    from venno.models.moss.vae import AutoEncoder, AutoEncoderSmallDecoderParams

    torch.manual_seed(0)
    params = MossImageParams(dim=128, ffn_dim=256, num_heads=4, num_layers=num_layers, text_dim=text_dim, freq_dim=64)
    dit = MossImageTransformer(params)
    sd = MossImageTransformer.convert_state_dict_on_save(dit.state_dict())
    for k in [k for k in sd if k.endswith(".proj.qkv.weight")]:  # the vendor keeps q/k/v split
        q, kk, v = sd.pop(k).chunk(3, 0)
        base = k[: -len("qkv.weight")]
        sd[base + "q.weight"], sd[base + "k.weight"], sd[base + "v.weight"] = q, kk, v
    keys = sorted(sd)
    half = len(keys) // 2
    shards = {"diffusion_pytorch_model-00001-of-00002.safetensors": keys[:half],
              "diffusion_pytorch_model-00002-of-00002.safetensors": keys[half:]}
    tdir = os.path.join(root, "transformer")
    for name, ks in shards.items():
        _save({k: sd[k] for k in ks}, os.path.join(tdir, name))
    with open(os.path.join(tdir, "diffusion_pytorch_model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {k: n for n, ks in shards.items() for k in ks}}, f)

    full = moss_text_config()
    tc = full["text_config"]
    tc.update(hidden_size=text_dim, intermediate_size=128, num_hidden_layers=4,
              layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=2,
              num_key_value_heads=1, head_dim=32, linear_key_head_dim=16, linear_value_head_dim=16,
              linear_num_key_heads=2, linear_num_value_heads=4)
    edir = os.path.join(root, "text_encoder")
    os.makedirs(edir, exist_ok=True)
    with open(os.path.join(edir, "config.json"), "w") as f:
        json.dump(full, f)
    config = AutoConfig.from_pretrained(edir)
    te = Qwen3_5TextModel(config.text_config)
    tsd = {"language_model." + k: v for k, v in te.state_dict().items()}
    tsd["visual.patch_embed.proj.weight"] = torch.zeros(4, 4)  # dropped on load
    _save(tsd, os.path.join(edir, "model.safetensors"))

    shutil.copytree(tokenizer_dir, os.path.join(root, "tokenizer"), dirs_exist_ok=True)

    vae = AutoEncoder(AutoEncoderSmallDecoderParams())
    with torch.no_grad():
        vae.bn.running_var.fill_(0.5)
    _save(vae.state_dict(), os.path.join(root, "vae", "diffusion_pytorch_model.safetensors"), torch.float32)
    return {"model_path": root}


def base_job(model, dataset_folders, **training):
    import subprocess

    out = subprocess.run(
        ["node", "-e", f"const S=require('{os.path.join(os.path.dirname(TRAINER), 'public', 'js', 'schema.js')}');"
         f"process.stdout.write(JSON.stringify(S.defaultJob('{model}')))"],
        capture_output=True, text=True, check=True,
    )
    job = json.loads(out.stdout)
    job["dataset"]["folders"] = [dict(job["dataset"]["folders"][0], **f) for f in dataset_folders]
    job["training"].update(training)
    return job
