// Anima Standalone Trainer (sd-scripts fork) backend.
// Writes config.toml / dataset.toml / sample_prompts.txt in the format the
// Anima trainer's own UI produces, then launches anima_train_network.py
// (or anima_train.py for full finetunes) through accelerate.
const path = require("path");
const TOML = require("@iarna/toml");
const S = require("../../public/js/schema");
const { num, int, dropUndefined, parseOverrides, findLastStateDir } = require("./util");

function buildDataset(job) {
  const d = job.dataset;
  const resolutions = S.parseIntList(d.resolutions, [1024]);
  const batches = S.parseIntList(d.batch_sizes, [1]);
  const folders = d.folders.filter((f) => f.enabled && f.path.trim());

  const subsets = folders.map((f) =>
    dropUndefined({
      image_dir: f.path.trim(),
      num_repeats: Math.max(1, int(f.repeats, 1)),
      keep_tokens: int(f.keep_tokens, 0),
      caption_prefix: f.caption_prefix ? f.caption_prefix : undefined,
      caption_dropout_rate: num(f.caption_dropout, 0),
      caption_tag_dropout_rate: num(f.tag_dropout, 0),
      caption_dropout_every_n_epochs: int(f.dropout_every_n_epochs, 0),
      shuffle_caption: !!f.shuffle,
      flip_aug: !!f.flip,
      cache_info: true,
      is_reg: f.is_reg ? true : undefined,
    }),
  );

  return {
    general: {
      enable_bucket: !!d.enable_bucket,
      bucket_no_upscale: !!d.bucket_no_upscale,
      min_bucket_reso: int(d.min_bucket, 512),
      max_bucket_reso: int(d.max_bucket, 1536),
      bucket_reso_steps: int(d.bucket_steps, 64),
    },
    // One dataset block per resolution (multi-resolution training); each may
    // have its own batch size, the last batch size repeats.
    datasets: resolutions.map((r, i) => ({
      resolution: [r, r],
      batch_size: batches[i] ?? batches[batches.length - 1],
      caption_extension: d.caption_ext || ".txt",
      subsets,
    })),
  };
}

// sd-scripts prompt file: one prompt per line with --w/--h/--s/--l/--d/--n.
function buildPrompts(job) {
  const s = job.sampling;
  const neg = (s.negative || "").replace(/[\r\n]+/g, " ").trim();
  return s.prompts
    .filter((p) => p.enabled !== false && (p.text || "").trim())
    .map((p) => {
      const text = p.text.replace(/[\r\n]+/g, " ").trim();
      let line =
        `${text} --w ${int(p.width ?? s.width, 1024)} --h ${int(p.height ?? s.height, 1024)}` +
        ` --s ${int(p.steps ?? s.steps, 28)} --l ${num(p.cfg ?? s.cfg, 4)} --d ${int(p.seed ?? s.seed, 42)}`;
      if (neg) line += ` --n ${neg}`;
      return line;
    })
    .join("\n");
}

function build(job, ctx) {
  const { jobDir, settings } = ctx;
  const t = job.training;
  const n = job.network;
  const s = job.sampling;
  const paths = (settings.models && settings.models.anima) || {};
  const isEpochs = t.duration_unit !== "steps";
  const isLora = n.type !== "full_finetune";

  const outputDir = path.join(jobDir, "output");
  const logsDir = path.join(jobDir, "logs");
  const datasetPath = path.join(jobDir, "dataset.toml");
  const promptsPath = path.join(jobDir, "sample_prompts.txt");
  const prompts = buildPrompts(job);

  const optimizerArgs = [];
  if (t.weight_decay !== "" && t.weight_decay != null) optimizerArgs.push(`weight_decay=${num(t.weight_decay, 0)}`);
  for (const a of String(t.optimizer_args || "").split(/\s+/).filter(Boolean)) optimizerArgs.push(a);

  const keepStates = int(t.keep_last_n_states, 1);
  const saveEverySteps = int(t.save_every_steps, 250);

  const training = dropUndefined({
    output_dir: outputDir,
    output_name: t.output_name || "lora",
    save_model_as: "safetensors",
    logging_dir: logsDir,
    log_with: "tensorboard",
    learning_rate: num(t.lr, 1e-4),
    text_encoder_lr: isLora && n.train_te ? num(t.te_lr, 5e-5) : undefined,
    optimizer_type: t.optimizer,
    optimizer_args: optimizerArgs.length ? optimizerArgs : undefined,
    lr_scheduler: t.lr_scheduler,
    lr_scheduler_num_cycles: t.lr_scheduler === "cosine_with_restarts" ? int(t.lr_cycles, 1) : undefined,
    lr_warmup_steps: int(t.lr_warmup_steps, 0),
    mixed_precision: t.mixed_precision,
    save_precision: t.save_precision,
    max_data_loader_n_workers: int(t.workers, 4),
    persistent_data_loader_workers: int(t.workers, 4) > 0,
    gradient_accumulation_steps: int(t.grad_accum, 1),
    max_grad_norm: num(t.max_grad_norm, 1.0),
    gradient_checkpointing: !!t.gradient_checkpointing,
    flash_attn: !!t.flash_attn,
    blocks_to_swap: int(t.blocks_to_swap, 0) > 0 ? int(t.blocks_to_swap, 0) : undefined,
    seed: int(t.seed, 42),
    max_train_epochs: isEpochs ? int(t.max_epochs, 10) : undefined,
    save_every_n_epochs: isEpochs ? int(t.save_every_epochs, 1) : undefined,
    save_last_n_epochs: isEpochs && int(t.keep_last_n, 0) > 0 ? int(t.keep_last_n, 0) : undefined,
    save_last_n_epochs_state: isEpochs && keepStates > 0 ? keepStates : undefined,
    max_train_steps: !isEpochs ? int(t.max_steps, 2000) : undefined,
    save_every_n_steps: !isEpochs ? saveEverySteps : undefined,
    // sd-scripts counts these in steps; the UI means "keep N saves"
    save_last_n_steps: !isEpochs && int(t.keep_last_n, 0) > 0 ? int(t.keep_last_n, 0) * saveEverySteps - 1 : undefined,
    save_last_n_steps_state: !isEpochs && keepStates > 0 ? keepStates * saveEverySteps - 1 : undefined,
    save_state: true,
  });

  if (n.auto_resume) {
    const last = findLastStateDir(outputDir);
    if (last) training.resume = last;
  } else if (n.resume_path && n.resume_path.trim()) {
    training.resume = n.resume_path.trim();
  }

  const config = {
    model_arguments: {
      dit_path: paths.dit_path || "",
      qwen3_path: paths.qwen3_path || "",
      vae_path: paths.vae_path || "",
    },
    dataset_arguments: {
      dataset_config: datasetPath,
      cache_latents: !!t.cache_latents,
      cache_latents_to_disk: !!t.cache_latents,
      cache_text_encoder_outputs: !!t.cache_te,
      cache_text_encoder_outputs_to_disk: !!t.cache_te,
    },
    training_arguments: training,
    anima_arguments: {
      timestep_sample_method: t.timestep,
      discrete_flow_shift: num(t.flow_shift, 3.0),
      weighting_scheme: "logit_normal",
    },
  };

  if (isLora) {
    config.network_arguments = dropUndefined({
      network_module: "networks.lora_anima",
      network_dim: int(n.dim, 16),
      network_alpha: num(n.alpha, 16),
      network_train_unet_only: !n.train_te,
      network_dropout: num(n.dropout, 0) > 0 ? num(n.dropout, 0) : undefined,
      network_args: n.network_args && n.network_args.trim() ? n.network_args.trim().split(/\s+/) : undefined,
      network_weights: n.init_weights && n.init_weights.trim() ? n.init_weights.trim() : undefined,
    });
  }

  if (s.enabled && prompts) {
    config.sample_arguments = { sample_prompts: promptsPath };
    if (isEpochs) config.sample_arguments.sample_every_n_epochs = int(s.every_epochs, 1);
    else config.sample_arguments.sample_every_n_steps = int(s.every_steps, 250);
    if (s.sample_first) config.sample_arguments.sample_at_first = true;
  }

  S.deepMerge(config, parseOverrides(job.advanced.overrides, "toml"));

  const root = settings.backends.anima.root;
  const gpuList = String(t.gpu_ids || "").split(",").map((x) => x.trim()).filter(Boolean);
  const script = isLora ? "anima_train_network.py" : "anima_train.py";
  const configPath = path.join(jobDir, "config.toml");
  const accelerate = ["-m", "accelerate.commands.launch", "--num_cpu_threads_per_process", "1"];
  if (gpuList.length > 1) {
    // plain DDP; FSDP/DeepSpeed/TP stay in the Anima trainer's own UI
    accelerate.push("--multi_gpu", "--num_processes", String(gpuList.length), "--mixed_precision", t.mixed_precision);
  }

  return {
    files: [
      { path: configPath, content: TOML.stringify(config) },
      { path: datasetPath, content: TOML.stringify(buildDataset(job)) },
      { path: promptsPath, content: prompts + (prompts ? "\n" : "") },
    ],
    args: [...accelerate, path.join(root, script), `--config_file=${configPath}`],
    cwd: root,
    env: Object.assign(
      gpuList.length ? { CUDA_VISIBLE_DEVICES: gpuList.join(",") } : {},
      gpuList.length > 1 && process.platform === "win32" ? { USE_LIBUV: "0" } : {},
    ),
    outputDir,
    samplesDir: path.join(outputDir, "sample"),
    logsDir,
    configPath,
  };
}

function checkSettings(settings) {
  const errs = [];
  if (!settings.backends.anima.root) errs.push("Set the Anima trainer folder in Global Settings.");
  const p = (settings.models && settings.models.anima) || {};
  for (const [k, def] of Object.entries(S.MODELS.anima.paths)) {
    if (!p[k]) errs.push(`Set "${def.label}" in Global Settings → Anima.`);
  }
  return errs;
}

module.exports = { build, buildDataset, buildPrompts, checkSettings };
