// Shared job schema: model registry, option lists and job defaults.
// Loaded by the browser (window.VennoSchema) and by the server (require).
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.VennoSchema = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ------------------------------------------------------------------
  // Backends: an installed trainer the UI writes configs for and launches.
  // ------------------------------------------------------------------
  const BACKENDS = {
    anima: {
      label: "Anima Standalone Trainer (sd-scripts)",
      repo: "https://github.com/gazingstars123/Anima-Standalone-Trainer",
    },
    aitk: {
      label: "ostris ai-toolkit",
      repo: "https://github.com/ostris/ai-toolkit",
    },
  };

  // ------------------------------------------------------------------
  // Models. `paths` are the global model path fields shown in Settings.
  // ------------------------------------------------------------------
  const MODELS = {
    anima: {
      label: "Anima",
      backend: "anima",
      variants: {
        default: { label: "Anima (DiT LoRA / finetune)" },
      },
      paths: {
        dit_path: { label: "DiT Model Path", placeholder: "/models/anima/anima.safetensors" },
        qwen3_path: { label: "Qwen3 Text Encoder Path", placeholder: "/models/anima/qwen_3_06b_base.safetensors" },
        vae_path: { label: "VAE Path", placeholder: "/models/anima/qwen_image_vae.safetensors" },
      },
    },
    moss_image: {
      label: "MOSS-Image",
      backend: "aitk",
      variants: {
        sft: { label: "MOSS-Image SFT", repo: "OpenMOSS-Team/MOSS-Image-SFT", edit: false },
        base: { label: "MOSS-Image Base", repo: "OpenMOSS-Team/MOSS-Image-Base", edit: false },
        edit: { label: "MOSS-Image SFT [Edit training]", repo: "OpenMOSS-Team/MOSS-Image-SFT", edit: true },
        base_edit: { label: "MOSS-Image Base [Edit training]", repo: "OpenMOSS-Team/MOSS-Image-Base", edit: true },
      },
      paths: {
        name_or_path: {
          label: "Transformer (HF repo, local diffusers dir or .safetensors)",
          placeholder: "Leave blank to use the variant's HF repo",
        },
        te_path: {
          label: "Qwen3.5-9B Text Encoder (optional)",
          placeholder: "Leave blank to let ai-toolkit pick/download it",
        },
        vae_path: {
          label: "FLUX.2 VAE (optional)",
          placeholder: "Leave blank to let ai-toolkit pick/download it",
        },
      },
    },
  };

  // Option lists differ per backend; the UI fills <select data-options="x">.
  const OPTIONS = {
    optimizer: {
      anima: [
        "AdamW8bit", "AdamW", "PagedAdamW8bit", "Adafactor", "Lion8bit",
        "Prodigy", "AdamWScheduleFree", "RAdamScheduleFree",
      ],
      aitk: [
        "adamw8bit", "adamw", "adam", "adafactor", "prodigyopt", "prodigy8bit",
        "automagic", "automagic2", "automagic3",
      ],
    },
    lr_scheduler: {
      anima: ["cosine", "constant", "constant_with_warmup", "linear", "cosine_with_restarts", "polynomial"],
      aitk: ["constant", "cosine", "cosine_with_restarts", "linear", "constant_with_warmup"],
    },
    timestep: {
      anima: ["logit_normal", "uniform"],
      aitk: ["shift", "sigmoid", "linear", "weighted"],
    },
    network_type: {
      anima: [["lora", "LoRA"], ["full_finetune", "Full finetune"]],
      aitk: [["lora", "LoRA"], ["lokr", "LoKr"]],
    },
    qtype: {
      anima: [],
      aitk: ["convrot8", "qfloat8", "float8", "convrot4", "nvfp4", "uint6", "uint4"],
    },
    mixed_precision: {
      anima: [["bf16", "BFloat16"], ["fp16", "Float16"], ["no", "Float32"]],
      aitk: [["bf16", "BFloat16"], ["fp16", "Float16"], ["fp32", "Float32"]],
    },
    save_precision: {
      anima: [["bf16", "BFloat16"], ["fp16", "Float16"], ["float", "Float32"]],
      aitk: [["bf16", "BFloat16"], ["float16", "Float16"], ["float32", "Float32"]],
    },
  };

  function defaultFolder() {
    return {
      path: "",
      enabled: true,
      repeats: 1,
      loss_weight: 1.0,
      keep_tokens: 1,
      caption_prefix: "",
      caption_dropout: 0.05,
      tag_dropout: 0.0,
      dropout_every_n_epochs: 0,
      shuffle: false,
      flip: false,
      is_reg: false,
      control_path: "",
      collapsed: false,
    };
  }

  function baseJob() {
    return {
      version: 1,
      model: "anima",
      variant: "default",
      training: {
        lr: "1e-4",
        te_lr: "5e-5",
        optimizer: "AdamW8bit",
        weight_decay: 0.01,
        optimizer_args: "",
        lr_scheduler: "cosine",
        lr_warmup_steps: 0,
        lr_cycles: 1,
        seed: 42,
        duration_unit: "epochs",
        max_epochs: 15,
        max_steps: 2000,
        save_every_epochs: 1,
        save_every_steps: 250,
        keep_last_n: 0,
        keep_last_n_states: 1,
        output_name: "my_lora",
        mixed_precision: "bf16",
        save_precision: "bf16",
        workers: 4,
        gradient_checkpointing: true,
        grad_accum: 1,
        max_grad_norm: 1.0,
        flash_attn: false,
        blocks_to_swap: 0,
        cache_latents: true,
        cache_te: true,
        timestep: "logit_normal",
        flow_shift: 3.0,
        timestep_bias: "balanced",
        loss_type: "mse",
        quantize: true,
        qtype: "convrot8",
        quantize_te: true,
        qtype_te: "convrot8",
        low_vram: true,
        layer_offloading: false,
        gpu_ids: "0",
      },
      network: {
        type: "lora",
        dim: 16,
        alpha: 16,
        dropout: 0,
        network_args: "",
        init_weights: "",
        train_te: false,
        ignore_layers: "",
        auto_resume: false,
        resume_path: "",
      },
      dataset: {
        resolutions: "1024",
        batch_sizes: "1",
        caption_ext: ".txt",
        enable_bucket: true,
        bucket_no_upscale: true,
        min_bucket: 512,
        max_bucket: 1536,
        bucket_steps: 64,
        folders: [defaultFolder()],
      },
      sampling: {
        enabled: true,
        every_epochs: 1,
        every_steps: 250,
        sample_first: false,
        negative: "",
        width: 1024,
        height: 1024,
        steps: 28,
        cfg: 4.0,
        seed: 42,
        prompts: [],
      },
      advanced: {
        overrides: "",
      },
    };
  }

  // Per-model default overrides applied on top of baseJob().
  const MODEL_DEFAULTS = {
    anima: {
      variant: "default",
      training: {
        lr: "1e-4", optimizer: "AdamW8bit", lr_scheduler: "cosine",
        timestep: "logit_normal", flow_shift: 3.0, output_name: "my_anima_lora",
        mixed_precision: "bf16", save_precision: "bf16",
      },
      network: { type: "lora", dim: 16, alpha: 16 },
      dataset: { resolutions: "1024", batch_sizes: "2", max_bucket: 1536 },
      sampling: { width: 832, height: 1216, steps: 28, cfg: 4.0 },
    },
    moss_image: {
      variant: "sft",
      training: {
        lr: "1e-4", optimizer: "adamw8bit", lr_scheduler: "constant",
        duration_unit: "steps", max_steps: 3000, save_every_steps: 250, keep_last_n: 4,
        timestep: "shift", timestep_bias: "balanced", output_name: "my_moss_lora",
        mixed_precision: "bf16", save_precision: "bf16", weight_decay: 0.0001,
        quantize: true, qtype: "convrot8", quantize_te: true, qtype_te: "convrot8",
        low_vram: true, cache_te: true, cache_latents: true,
      },
      // The vendor sampler skips DiT block 22 on the CFG negative branch, so
      // the LoRA stays off that block (ai-toolkit's own MOSS default).
      network: { type: "lora", dim: 32, alpha: 32, ignore_layers: "layers.22." },
      dataset: { resolutions: "512, 768, 1024", batch_sizes: "1" },
      sampling: { width: 1024, height: 1024, steps: 30, cfg: 3.0, every_steps: 250 },
    },
  };

  function isPlainObject(v) {
    return v !== null && typeof v === "object" && !Array.isArray(v);
  }

  function deepMerge(target, src) {
    for (const [k, v] of Object.entries(src || {})) {
      if (isPlainObject(v) && isPlainObject(target[k])) deepMerge(target[k], v);
      else target[k] = Array.isArray(v) ? JSON.parse(JSON.stringify(v)) : v;
    }
    return target;
  }

  function defaultJob(model) {
    const job = baseJob();
    const id = MODELS[model] ? model : "anima";
    job.model = id;
    deepMerge(job, MODEL_DEFAULTS[id]);
    return job;
  }

  // Fill missing keys of a stored job from its model defaults (forward compat).
  function normalizeJob(job) {
    const out = defaultJob(job && job.model);
    deepMerge(out, job || {});
    out.dataset.folders = (out.dataset.folders || []).map((f) =>
      Object.assign(defaultFolder(), f),
    );
    if (out.dataset.folders.length === 0) out.dataset.folders.push(defaultFolder());
    if (!MODELS[out.model].variants[out.variant]) {
      out.variant = Object.keys(MODELS[out.model].variants)[0];
    }
    return out;
  }

  function backendOf(job) {
    return MODELS[job.model].backend;
  }

  function variantOf(job) {
    return MODELS[job.model].variants[job.variant] || {};
  }

  // "512, 1024" -> [512, 1024]
  function parseIntList(str, fallback) {
    const list = String(str ?? "")
      .split(/[,\s]+/)
      .map((s) => parseInt(s, 10))
      .filter((n) => Number.isFinite(n) && n > 0);
    return list.length ? list : fallback;
  }

  // kohya-style folder names ("10_character") carry their repeat count.
  function repeatsFromFolderName(name) {
    const m = /^(\d+)_/.exec(name || "");
    return m ? parseInt(m[1], 10) : null;
  }

  // Problems that would make the trainer refuse to start or silently ignore
  // settings. level "error" blocks launching, "warn" is advisory.
  function validateJob(job) {
    const issues = [];
    const backend = backendOf(job);
    const folders = job.dataset.folders.filter((f) => f.enabled);
    if (folders.length === 0) issues.push({ level: "error", msg: "No enabled dataset folders." });
    folders.forEach((f, i) => {
      if (!f.path.trim()) issues.push({ level: "error", msg: `Dataset ${i + 1} has no image directory.` });
    });
    const paths = folders.map((f) => f.path.trim()).filter(Boolean);
    if (new Set(paths).size !== paths.length) {
      issues.push({ level: "error", msg: "The same image directory is used by more than one dataset." });
    }
    const tagOps = folders.some((f) => Number(f.tag_dropout) > 0 || f.shuffle);
    if (job.training.cache_te && tagOps) {
      issues.push({
        level: backend === "anima" ? "error" : "warn",
        msg: backend === "anima"
          ? "Tag dropout / shuffle captions need 'Cache Text Encoder Outputs' off (sd-scripts refuses to start otherwise)."
          : "Tag dropout / shuffle captions are ignored by ai-toolkit while text embeddings are cached.",
      });
    }
    if (backend === "anima" && job.training.cache_te) {
      const rates = new Set(folders.map((f) => Number(f.caption_dropout)));
      if (rates.size > 1) {
        issues.push({
          level: "warn",
          msg: "With cached text encoder outputs Anima applies one global caption dropout (the highest folder rate).",
        });
      }
    }
    if (backend === "anima" && job.network.type === "lora" && job.network.train_te && job.training.cache_te) {
      issues.push({ level: "error", msg: "Training the text encoder requires 'Cache Text Encoder Outputs' off." });
    }
    if (variantOf(job).edit && folders.some((f) => !f.control_path.trim())) {
      issues.push({ level: "error", msg: "Edit training needs a control images folder on every dataset." });
    }
    return issues;
  }

  return {
    BACKENDS,
    MODELS,
    OPTIONS,
    MODEL_DEFAULTS,
    defaultFolder,
    defaultJob,
    normalizeJob,
    deepMerge,
    backendOf,
    variantOf,
    parseIntList,
    repeatsFromFolderName,
    validateJob,
  };
});
