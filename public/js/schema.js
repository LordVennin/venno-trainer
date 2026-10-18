// Shared job schema: model registry, option lists and job defaults.
// Loaded by the browser (window.VennoSchema) and by the server (require).
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.VennoSchema = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ------------------------------------------------------------------
  // Models. Training runs in the bundled Python trainer (trainer/venno);
  // `paths` are the global model path fields shown in Settings.
  // ------------------------------------------------------------------
  const MODELS = {
    anima: {
      label: "Anima",
      variants: {
        default: { label: "Anima (DiT LoRA)" },
      },
      paths: {
        dit_path: { label: "DiT Model Path", placeholder: "/models/anima/split_files/diffusion_models/anima-base-v1.0.safetensors" },
        qwen3_path: { label: "Qwen3 Text Encoder Path", placeholder: "/models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors" },
        vae_path: { label: "VAE Path", placeholder: "/models/anima/split_files/vae/qwen_image_vae.safetensors" },
      },
    },
    moss_image: {
      label: "MOSS-Image",
      variants: {
        sft: { label: "MOSS-Image SFT", repo: "OpenMOSS-Team/MOSS-Image-SFT" },
        base: { label: "MOSS-Image Base", repo: "OpenMOSS-Team/MOSS-Image-Base" },
      },
      paths: {
        model_path: {
          label: "Model folder or HF repo (diffusers layout)",
          placeholder: "Blank = download the variant's repo into the models folder (~27 GB)",
        },
        transformer_path: { label: "Transformer override (.safetensors, optional)", placeholder: "e.g. a fine-tuned MOSS DiT" },
        te_path: { label: "Text encoder override (folder or .safetensors, optional)", placeholder: "Qwen3.5-9B weights" },
        vae_path: { label: "VAE override (.safetensors, optional)", placeholder: "e.g. the full-decoder flux2-vae.safetensors" },
      },
    },
  };

  // [value, label] lists for <select data-options="x">.
  const OPTIONS = {
    optimizer: ["AdamW8bit", "AdamW", "Adafactor", "Prodigy", "PagedAdamW8bit"],
    lr_scheduler: ["constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear", "polynomial"],
    timestep: [["logit_normal", "Logit normal"], ["uniform", "Uniform"]],
    mixed_precision: [["bf16", "BFloat16"], ["fp32", "Float32 (slow, testing)"]],
    save_precision: [["bf16", "BFloat16"], ["fp16", "Float16"], ["fp32", "Float32"]],
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
      collapsed: false,
    };
  }

  function baseJob() {
    return {
      version: 2,
      model: "anima",
      variant: "default",
      training: {
        lr: "1e-4",
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
        gradient_checkpointing: true,
        grad_accum: 1,
        max_grad_norm: 1.0,
        flash_attn: false,
        fp8_base: false,
        cache_te: true,
        te_variants: 8,
        vae_batch: 2,
        workers: 4,
        timestep: "logit_normal",
        flow_shift: 3.0,
        loss_type: "mse",
        gpu_ids: "0",
      },
      network: {
        dim: 16,
        alpha: 16,
        dropout: 0,
        init_weights: "",
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
        shift: 1.0,
        prompts: [],
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
      },
      network: { dim: 16, alpha: 16 },
      dataset: { resolutions: "1024", batch_sizes: "2", max_bucket: 1536 },
      // the Anima trainer previews with a linear (unshifted) Euler schedule
      sampling: { width: 832, height: 1216, steps: 28, cfg: 4.0, shift: 1.0 },
    },
    moss_image: {
      variant: "sft",
      training: {
        lr: "1e-4", optimizer: "AdamW8bit", lr_scheduler: "constant",
        duration_unit: "steps", max_steps: 3000, save_every_steps: 250, keep_last_n: 4,
        // uniform sigmas shifted by 5 = ai-toolkit's "shift" timesteps for MOSS
        timestep: "uniform", flow_shift: 5.0, output_name: "my_moss_lora",
        weight_decay: 0.0001, cache_te: true,
      },
      // The vendor sampler skips DiT block 22 on the CFG negative branch, so
      // the LoRA stays off that block (ai-toolkit's MOSS default).
      network: { dim: 32, alpha: 32, ignore_layers: "layers.22." },
      dataset: { resolutions: "512, 768, 1024", batch_sizes: "1" },
      sampling: { width: 1024, height: 1024, steps: 30, cfg: 3.0, shift: 5.0, every_steps: 250 },
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

  // Problems that would make training fail or behave unexpectedly.
  // level "error" blocks launching, "warn" is advisory.
  function validateJob(job) {
    const issues = [];
    const folders = job.dataset.folders.filter((f) => f.enabled);
    if (folders.length === 0) issues.push({ level: "error", msg: "No enabled dataset folders." });
    folders.forEach((f, i) => {
      if (!f.path.trim()) issues.push({ level: "error", msg: `Dataset ${i + 1} has no image directory.` });
    });
    const paths = folders.map((f) => f.path.trim()).filter(Boolean);
    if (new Set(paths).size !== paths.length) {
      issues.push({ level: "error", msg: "The same image directory is used by more than one dataset." });
    }
    if (!String(job.training.output_name || "").trim()) {
      issues.push({ level: "error", msg: "Set an output name." });
    }
    const tagOps = folders.some((f) => Number(f.tag_dropout) > 0 || f.shuffle);
    if (tagOps && textCached(job) && Number(job.training.te_variants) < 2) {
      issues.push({
        level: "warn",
        msg: "Tag dropout / shuffle need at least 2 cached caption variants (Training → Caching) to have any effect.",
      });
    }
    return issues;
  }

  // MOSS-Image's 9B text encoder cannot stay loaded next to the DiT, so its
  // captions are always cached.
  function textCached(job) {
    return job.model === "moss_image" || !!job.training.cache_te;
  }

  return {
    MODELS,
    OPTIONS,
    MODEL_DEFAULTS,
    defaultFolder,
    defaultJob,
    normalizeJob,
    deepMerge,
    variantOf,
    textCached,
    parseIntList,
    repeatsFromFolderName,
    validateJob,
  };
});
