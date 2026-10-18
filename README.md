# Venno Trainer

A personal LoRA training UI for several models, laid out like the
[Anima Standalone Trainer](https://github.com/gazingstars123/Anima-Standalone-Trainer) UI
(job sidebar, Training / Dataset / Network / Prompts / Samples / Console tabs, collapsible
per-folder dataset cards).

Venno Trainer is only the UI. It writes each trainer's own config format and launches the
trainer you already installed:

| Model | Trainer used | What gets generated |
|---|---|---|
| **Anima** | Anima Standalone Trainer (sd-scripts fork) | `config.toml`, `dataset.toml`, `sample_prompts.txt` → `accelerate launch anima_train_network.py` (or `anima_train.py` for a full finetune) |
| **MOSS-Image** (SFT, Base, plus edit-training variants) | [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) (`moss_image` arch) | `config.yaml` → `python run.py config.yaml` |

## Dataset handling

- **One card per folder.** Each card has its own repeats, caption dropout, tag dropout, keep
  tokens, caption prefix / trigger, shuffle, flip and regularization flag. Cards can be turned
  on/off, reordered, duplicated or collapsed.
- **Loss weight per folder** (MOSS / ai-toolkit `loss_multiplier`). sd-scripts has no per-folder
  loss weight, so for Anima use repeats.
- **+ Add Subfolders…** scans a parent folder and adds every subfolder as a dataset in one go.
  Kohya-style names such as `10_character` set the repeats.
- **Bulk Edit…** sets any field (dropout, repeats, weight, prefix, flags) on the selected folders,
  or on all of them.
- **Live stats** for each folder and in total: image count, uncaptioned images, samples per epoch,
  steps per epoch, total steps, and each folder's share of an epoch.
- **Checks before launch.** It flags combinations the trainers reject or silently ignore. For
  example, tag dropout and caption shuffling only work with text-encoder caching off, and with
  caching on Anima uses one global caption dropout (the highest folder rate).

## Other features

- Epoch or step schedules. ai-toolkit only trains by steps, so epochs are converted to steps
  using the dataset scan.
- Sample prompts with per-prompt size, steps, CFG and seed, a shared negative prompt, and
  "Paste Many".
- Live console, step/ETA progress and a loss chart, read from either trainer's progress bar.
- Samples gallery (lightbox, multi-select delete) and a checkpoint list.
- **Clone to another model**: the same dataset folders and prompts, with the other model's
  defaults for everything else.
- **Advanced overrides**: raw TOML (Anima) or YAML (ai-toolkit) merged over the generated
  config, and a preview of exactly which files and command Train will produce.
- Every launch writes `launch_command.sh` / `.bat` in the job folder, so you can rerun a job
  outside the UI.

## Setup

1. Install the trainers you need with their own instructions:
   - Anima: clone Anima-Standalone-Trainer and run its `setup_env.sh` / `setup_env.bat`.
   - MOSS-Image: clone ai-toolkit and install it into a venv (see its README). The model, the
     Qwen3.5-9B text encoder and the FLUX.2 VAE download on first use (set a HuggingFace token
     if needed).
2. Install Node.js 18+ and start the UI:
   ```bash
   ./start.sh          # Linux / macOS
   start.bat           # Windows
   ```
   Open <http://localhost:3100>. Options: `--port=3100`, `--host=0.0.0.0` (the default only
   listens on 127.0.0.1, because the UI launches processes).
3. Open **⚙️ Global Settings**:
   - **Trainers**: each install folder, plus its venv (leave blank to use `<install>/venv`).
   - **Anima**: DiT, Qwen3 text encoder and VAE paths.
   - **MOSS-Image**: optional transformer / text encoder / VAE paths. Leave them blank and
     ai-toolkit uses the variant's HF repo and downloads what it needs.

Jobs are stored in `jobs/<name>/` (`job.json`, generated configs, `output/`, `logs/`).
Settings are stored in `settings.json`. Both are git-ignored.

## MOSS-Image defaults

These follow ai-toolkit's own MOSS-Image presets: transformer and Qwen3.5-9B text encoder
quantized to `convrot8`, low-VRAM loading, `shift` timesteps, 1024² samples at 30 steps / CFG 3,
and DiT block 22 skipped on the CFG negative branch (`uncond_skip_layers: [22]`). Because of
that skip, the LoRA is kept off block 22 by default (`ignore_if_contains: layers.22.`). The edit
variants add a control-image folder to each dataset and a control image to each prompt, and
turn off text-embedding caching (the prompt embeds include the reference images).

## Development

```bash
npm install
npm test       # config builder tests (node:test)
```

To add a model: add it to `MODELS` / `MODEL_DEFAULTS` in `public/js/schema.js` and point it at an
existing backend in `lib/backends/` (or add a new backend with `build()` and `checkSettings()`).
Fields marked `data-backend="…"` in `public/index.html` only show for that backend.
