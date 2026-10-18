# Venno Trainer

A personal LoRA trainer with a web UI, laid out like the
[Anima Standalone Trainer](https://github.com/gazingstars123/Anima-Standalone-Trainer) UI
(job sidebar, Training / Dataset / Network / Prompts / Samples / Console tabs, collapsible
per-folder dataset cards). It trains:

| Model | Variants | What is trained |
|---|---|---|
| **Anima** | the model you point it at (base, aesthetic, preview…) | LoRA on the DiT blocks |
| **MOSS-Image** | SFT, Base | LoRA on the 42 DiT blocks (block 22 skipped by default) |

Training runs in the bundled Python trainer (`trainer/venno`): **one Python environment** for
both models. You do not need the Anima trainer, ai-toolkit or their environments.

## Setup (Linux, NVIDIA GPU)

Requirements: Python 3.10+, Node.js 18+, an NVIDIA driver for CUDA 12.8 (R570+).

```bash
./setup.sh      # creates ./venv (PyTorch CUDA 12.8 + trainer packages), installs the UI
./start.sh      # http://localhost:3100
```

`setup.sh` options: `TORCH_INDEX=https://download.pytorch.org/whl/cu126 ./setup.sh` for an
older driver, `PYTHON=python3.12` to choose the interpreter, `FAST_KERNELS=0` to skip
`flash-linear-attention`. The venv is about 6 GB, almost all of it PyTorch + CUDA. pip's
download cache is disabled during setup so wheels are not stored a second time.

Then open **⚙️ Global Settings**:
- **Trainer**: press *Check environment*. The venv defaults to `./venv`, the cache to
  `./cache` and model downloads to `./models`; change them to put large data on another disk.
- **Anima**: paths to the DiT, `qwen_3_06b_base.safetensors` and `qwen_image_vae.safetensors`
  (from [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) `split_files/`;
  the files your Anima trainer already uses work as they are).
- **MOSS-Image**: leave the model folder blank to download the selected variant
  ([SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) or
  [Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base), about 27 GB) into the models folder
  on first use, or point it at a folder you already downloaded. Optional overrides take a
  fine-tuned transformer, another text encoder or the full-decoder FLUX.2 VAE.

## Memory on a 24 GB card

Caching runs first, one model at a time: the VAE encodes every image, then the text encoder
encodes every caption, and both are unloaded before the DiT loads. So MOSS's 9B Qwen3.5
encoder (~18 GB in bf16) never shares the GPU with its 5.1B DiT (~10 GB in bf16). LoRA training
with gradient checkpointing should then fit on a 4090 at 1024². If it does not (bigger batches or
resolutions), enable **fp8 Base Weights**: it stores the frozen DiT in 8 bits (about half the
VRAM) while the LoRA trains in full precision. Anima (2B DiT) fits with room to spare.

## Speed on a single 4090

Always on: content-keyed caches (each dataset is encoded once per model), bf16 with PyTorch's
flash-attention kernels, TF32 for the fp32 parts (MOSS VAE), cuDNN autotuning for VAE caching,
length-sorted caption encoding (MOSS encodes equal-length captions in one batch instead of one at
a time), the next batch loaded in the background while the GPU trains, fused AdamW and an
allocator setting that avoids fragmentation OOMs. The console shows peak VRAM, and the progress
area tells you when VRAM is going unused.

Options to try, in this order:
1. **Blocks Without Checkpointing**: skips recomputing the last N blocks. Raise it until the peak
   VRAM shown in the console is around 21–22 GB. Anima, with its small DiT, has the most room.
2. **Batch size**: larger batches use the GPU better, if VRAM allows.
3. **torch.compile** (experimental): compiles the transformer blocks. The first step at each new
   bucket shape is slow, the rest are faster. It falls back to normal mode if compiling fails.

**fp8 base weights** save memory but are slightly slower, so only use them when a setting would
not otherwise fit. `setup.sh` installs `flash-linear-attention`, which speeds up the MOSS text
encoder while captions are cached.

## Datasets

- **One card per folder** with its own repeats, loss weight, caption dropout, tag dropout,
  dropout every N epochs, keep tokens, caption prefix/trigger, shuffle, flip and regularization
  flag. Cards can be enabled/disabled, reordered, duplicated, collapsed and filtered.
- **+ Add Subfolders…** adds every subfolder of a parent folder at once; kohya-style names such
  as `10_character` set the repeats.
- **Bulk Edit…** sets any of those fields on the selected folders (or all of them).
- **Live stats**: images, uncaptioned files, samples and steps per epoch, total steps, and each
  folder's share of an epoch.
- Captions are processed in sd-scripts order: prefix → caption dropout (per folder, or whole
  epochs) → shuffle with keep tokens → tag dropout on comma-separated tags. Anima reads the first
  caption line (as sd-scripts does); MOSS uses the whole file.
- **Dropout with cached captions.** Caption dropout uses a cached empty caption. For folders with
  tag dropout or shuffling, *Caption variants per image* (default 8) randomized captions are cached
  and one is picked each step. For Anima you can instead turn caching off: the small Qwen3 encoder
  stays loaded and every step gets a fresh random caption.
- Multi-resolution: list several resolutions (e.g. `512, 768, 1024`) with one batch size each.
  Regularization folders are sampled to match the number of training images, as sd-scripts does.

Caches (latents and caption embeddings) are keyed by image and caption content and shared by every
job, so a dataset is encoded once per model. *Global Settings → Trainer* shows their size and
can clear them.

## Training, saving, resuming

- Epoch or step schedules, AdamW / AdamW8bit / PagedAdamW8bit / Adafactor / Prodigy, constant /
  warmup / cosine / cosine-with-restarts / linear / polynomial schedules, gradient accumulation,
  logit-normal or uniform timesteps with a flow shift (defaults: Anima logit-normal, shift 3, as in
  the Anima trainer; MOSS uniform, shift 5, as ai-toolkit's MOSS preset).
- LoRAs are saved in the key layout ComfyUI expects: `lora_unet_blocks_N_…` for Anima (as the
  Anima trainer writes them) and `diffusion_model.layers.N.proj.q/k/v/o…` for MOSS (as ai-toolkit
  writes them).
- Every save also writes a training state (`<name>-…-state`). **Stop** finishes the current step
  and saves one too. With *Auto-resume* the next run continues from the newest state; a learning
  rate changed in the meantime takes effect.
- Sample images during training use each model's own sampler: Euler with CFG for Anima, and for
  MOSS the vendor's snapped shift-5 schedule with block 22 skipped on the negative branch. The
  Prompts tab sets size, steps, CFG, seed and shift per prompt.
- Console with live progress, ETA and a loss chart; samples gallery and checkpoint list.
- **Clone to another model** keeps the dataset folders and prompts and takes that model's defaults
  for everything else.
- Each run writes `jobs/<name>/run.json` and `launch_command.sh`, so a job can be rerun from a
  terminal: `PYTHONPATH=trainer venv/bin/python -m venno.train jobs/<name>/run.json`.

## Not included yet

Multi-GPU training, full finetuning, Anima's Muon optimizer, block swapping and MOSS edit
(reference-image) training.

## Layout

```
server.js, lib/        Node UI server (jobs, launching, logs, dataset scans)
public/                web UI; public/js/schema.js is the job schema shared with the server
trainer/venno/         Python trainer
  train.py             entry point: caching, training loop, saves, samples, resume
  data.py, cache.py    datasets, captions, buckets, latent/text caches
  lora.py, fp8.py      LoRA modules, fp8 base-weight storage
  models/anima/        Anima DiT + Wan VAE (Apache-2.0, from the Anima Standalone Trainer)
  models/moss/         MOSS-Image DiT + FLUX.2 VAE (MIT, from ai-toolkit)
trainer/tests/         CPU tests on tiny models in the real checkpoint layouts
```

## Development

```bash
npm test                                                     # UI/server tests
cd trainer && ../venv/bin/python -m unittest discover -s tests -v   # trainer tests (CPU)
```

The MOSS test downloads the real MOSS tokenizer from the Hub (and is skipped when offline);
`tests/test_models.py` checks the ported MOSS transformer against the vendor's reference code.
