"""JoyCaption Beta One (fancyfeast/llama-joycaption-beta-one-hf-llava):
LLaVA with a SigLIP2 vision tower and Llama 3.1 8B, loaded with transformers'
LlavaForConditionalGeneration + AutoProcessor. Prompts are built by the UI
(public/js/captions.js); this module only downloads, loads and generates.

Generation follows the model card: system "You are a helpful image
captioner.", the chat template with add_generation_prompt=True (it inserts
the image token itself), pixel values in bf16, sampling, and only the new
tokens decoded. Several images run as one left-padded batch."""

import os
import threading
from typing import Callable, Dict, List

import torch
from PIL import Image

from .models import load_rgb

SYSTEM_PROMPT = "You are a helpful image captioner."

CAPTIONERS = {
    "joycaption-beta-one": {
        "label": "JoyCaption Beta One",
        "repo": "fancyfeast/llama-joycaption-beta-one-hf-llava",
        "gb": 17.0,
    },
}

NEEDED = ["config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "chat_template.json"]
# NF4 for the language model only: quantizing the vision tower and projector costs accuracy
SKIP_QUANT = ["vision_tower", "multi_modal_projector", "lm_head"]


def model_dir(models_dir, model_id):
    return os.path.join(models_dir, CAPTIONERS[model_id]["repo"].replace("/", "__"))


def is_downloaded(models_dir, model_id):
    import json

    root = model_dir(models_dir, model_id)
    if not all(os.path.isfile(os.path.join(root, f)) for f in NEEDED):
        return False
    index = os.path.join(root, "model.safetensors.index.json")
    if not os.path.isfile(index):
        return os.path.isfile(os.path.join(root, "model.safetensors"))
    try:
        with open(index, encoding="utf-8") as f:
            shards = set(json.load(f)["weight_map"].values())
    except (OSError, ValueError, KeyError):
        return False
    return all(os.path.isfile(os.path.join(root, s)) for s in shards)


def fetch(models_dir, model_id, token=None, status: Callable[[str], None] = print):
    """Download once into models_dir; a finished download stays offline."""
    root = model_dir(models_dir, model_id)
    if is_downloaded(models_dir, model_id):
        return root
    from huggingface_hub import snapshot_download

    info = CAPTIONERS[model_id]
    status(f"Downloading {info['label']} (~{info['gb']:.0f} GB, once; resumes if interrupted)...")
    done = threading.Event()

    def report():
        while not done.wait(10.0):
            size = sum(os.path.getsize(os.path.join(d, n)) for d, _, ns in os.walk(root) for n in ns if os.path.exists(os.path.join(d, n)))
            status(f"Downloading {info['label']}: {size / 1e9:.1f} of ~{info['gb']:.0f} GB")

    threading.Thread(target=report, daemon=True).start()
    try:
        snapshot_download(info["repo"], local_dir=root, token=token or None,
                          allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
    finally:
        done.set()
    return root


class JoyCaptioner:
    def __init__(self, root, device, quant="bf16"):
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(root)
        tok = self.processor.tokenizer
        tok.padding_side = "left"  # prompts must end where generation starts
        if tok.pad_token is None:
            vocab = tok.get_vocab()
            tok.pad_token = "<|finetune_right_pad_id|>" if "<|finetune_right_pad_id|>" in vocab else tok.eos_token
        kwargs = {"dtype": torch.bfloat16}
        if quant == "nf4":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=SKIP_QUANT,
            )
            kwargs["device_map"] = {"": str(self.device)}
        elif self.device.type == "cuda":
            kwargs["device_map"] = {"": str(self.device)}
        model = LlavaForConditionalGeneration.from_pretrained(root, **kwargs)
        if "device_map" not in kwargs:
            model = model.to(self.device)
        self.model = model.eval().requires_grad_(False)
        self.quant = quant

    def chat(self, prompt):
        convo = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        return self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def generate(self, images: List[Image.Image], prompts: List[str], temperature=0.6, top_p=0.9,
                 max_new_tokens=512, do_sample=True) -> List[str]:
        inputs = self.processor(text=[self.chat(p) for p in prompts], images=images, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        sample = bool(do_sample) and float(temperature) > 0
        tok = self.processor.tokenizer
        gen = dict(max_new_tokens=int(max_new_tokens), do_sample=sample, use_cache=True, pad_token_id=tok.pad_token_id,
                   suppress_tokens=None)
        if sample:
            gen.update(temperature=float(temperature), top_p=float(top_p), top_k=None)
        else:
            gen.update(temperature=None, top_p=None, top_k=None)
        out = self.model.generate(**inputs, **gen)
        new = out[:, inputs["input_ids"].shape[1]:]  # left padding: every prompt ends at the same column
        return [tok.decode(row, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip() for row in new]


def load(model_id, models_dir, device, quant="bf16", token=None, status=print):
    if model_id not in CAPTIONERS:
        raise ValueError(f"Unknown captioner: {model_id}")
    if quant not in ("bf16", "nf4"):
        raise ValueError(f"Unknown load mode: {quant}")
    root = fetch(models_dir, model_id, token, status)
    status(f"Loading {CAPTIONERS[model_id]['label']} ({'4-bit NF4' if quant == 'nf4' else 'bf16'}) on {device}...")
    return JoyCaptioner(root, device, quant)


def run(captioner, paths: List[str], prompts: List[str], batch_size=1, **gen) -> List[Dict]:
    """A caption (or error) for each path, in order."""
    out: List[Dict] = [None] * len(paths)
    ready = []
    for i, p in enumerate(paths):
        try:
            ready.append((i, load_rgb(p)))
        except Exception as e:  # unreadable image
            out[i] = {"path": p, "error": f"{type(e).__name__}: {e}"}
    for k in range(0, len(ready), max(1, batch_size)):
        chunk = ready[k:k + max(1, batch_size)]
        try:
            texts = captioner.generate([im for _, im in chunk], [prompts[i] for i, _ in chunk], **gen)
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError("Out of GPU memory: lower the batch size or use the 4-bit load mode.")
        for (i, _), text in zip(chunk, texts):
            out[i] = {"path": paths[i], "text": text}
    return out
