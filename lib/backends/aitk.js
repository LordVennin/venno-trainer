// ostris ai-toolkit backend (used for MOSS-Image).
// Writes one YAML job config and launches `python run.py <config>`.
// ai-toolkit trains by steps; an epoch count is converted with the dataset
// scan (images x repeats / batch size).
const path = require("path");
const YAML = require("js-yaml");
const S = require("../../public/js/schema");
const { num, int, dropUndefined, parseOverrides } = require("./util");

// Steps for N epochs, given the effective sample count of the scan.
function epochsToSteps(job, epochs, effectiveImages) {
  const batch = S.parseIntList(job.dataset.batch_sizes, [1])[0];
  const accum = Math.max(1, int(job.training.grad_accum, 1));
  const perEpoch = Math.max(1, Math.ceil(effectiveImages / (batch * accum)));
  return perEpoch * Math.max(1, epochs);
}

function schedulerParams(t, steps) {
  const warmup = int(t.lr_warmup_steps, 0);
  switch (t.lr_scheduler) {
    case "constant_with_warmup":
      return { num_warmup_steps: warmup };
    case "cosine_with_restarts":
      return { total_iters: Math.max(1, Math.floor(steps / Math.max(1, int(t.lr_cycles, 1)))) };
    case "linear":
      // torch LinearLR defaults ramp *up*; make it a decay
      return { start_factor: 1.0, end_factor: 0.01 };
    default:
      return undefined;
  }
}

function buildDatasets(job, variant) {
  const d = job.dataset;
  const t = job.training;
  const resolutions = S.parseIntList(d.resolutions, [1024]);
  return d.folders
    .filter((f) => f.enabled && f.path.trim())
    .map((f) =>
      dropUndefined({
        folder_path: f.path.trim(),
        caption_ext: (d.caption_ext || ".txt").replace(/^\./, ""),
        num_repeats: Math.max(1, int(f.repeats, 1)),
        loss_multiplier: num(f.loss_weight, 1.0) !== 1 ? num(f.loss_weight, 1.0) : undefined,
        caption_dropout_rate: num(f.caption_dropout, 0),
        token_dropout_rate: num(f.tag_dropout, 0) > 0 ? num(f.tag_dropout, 0) : undefined,
        shuffle_tokens: f.shuffle ? true : undefined,
        keep_tokens: int(f.keep_tokens, 0) > 0 ? int(f.keep_tokens, 0) : undefined,
        // ai-toolkit prepends the trigger only when the caption lacks it
        trigger_word: f.caption_prefix && f.caption_prefix.trim() ? f.caption_prefix.trim() : undefined,
        flip_x: f.flip ? true : undefined,
        is_reg: f.is_reg ? true : undefined,
        control_path: variant.edit && f.control_path.trim() ? f.control_path.trim() : undefined,
        resolution: resolutions,
        buckets: !!d.enable_bucket,
        cache_latents_to_disk: !!t.cache_latents,
      }),
    );
}

function buildSamples(job, variant) {
  const s = job.sampling;
  return s.prompts
    .filter((p) => p.enabled !== false && (p.text || "").trim())
    .map((p) =>
      dropUndefined({
        prompt: p.text.replace(/[\r\n]+/g, " ").trim(),
        width: int(p.width ?? s.width, 1024),
        height: int(p.height ?? s.height, 1024),
        sample_steps: int(p.steps ?? s.steps, 30),
        guidance_scale: num(p.cfg ?? s.cfg, 3),
        seed: int(p.seed ?? s.seed, 42),
        ctrl_img: variant.edit && p.ctrl_img ? p.ctrl_img : undefined,
      }),
    );
}

function build(job, ctx) {
  const { jobDir, settings, effectiveImages } = ctx;
  const t = job.training;
  const n = job.network;
  const s = job.sampling;
  const variant = S.variantOf(job);
  const paths = (settings.models && settings.models[job.model]) || {};
  const name = t.output_name || "lora";
  const trainingFolder = path.join(jobDir, "output");

  let steps = int(t.max_steps, 2000);
  let saveEvery = int(t.save_every_steps, 250);
  let sampleEvery = int(s.every_steps, 250);
  if (t.duration_unit === "epochs") {
    const images = effectiveImages || 0;
    if (!images) throw new Error("Epoch mode needs a dataset scan; no images were found in the dataset folders.");
    steps = epochsToSteps(job, int(t.max_epochs, 10), images);
    const perEpoch = epochsToSteps(job, 1, images);
    saveEvery = perEpoch * Math.max(1, int(t.save_every_epochs, 1));
    sampleEvery = perEpoch * Math.max(1, int(s.every_epochs, 1));
  }

  const ignore = String(n.ignore_layers || "").split(/[,\s]+/).filter(Boolean);
  const network = dropUndefined({
    type: n.type === "lokr" ? "lokr" : "lora",
    linear: int(n.dim, 32),
    linear_alpha: num(n.alpha, 32),
    dropout: num(n.dropout, 0) > 0 ? num(n.dropout, 0) : undefined,
    pretrained_lora_path: n.init_weights && n.init_weights.trim() ? n.init_weights.trim() : undefined,
    network_kwargs: ignore.length ? { ignore_if_contains: ignore } : undefined,
  });
  if (network.type === "lokr") {
    network.lokr_full_rank = true;
    network.lokr_factor = -1;
  }

  const modelKwargs = { uncond_skip_layers: [22] };
  if (variant.edit) Object.assign(modelKwargs, { edit: true, kv_cache: true, match_target_res: true });

  const model = dropUndefined({
    name_or_path: (paths.name_or_path && paths.name_or_path.trim()) || variant.repo,
    arch: "moss_image",
    quantize: !!t.quantize,
    qtype: t.quantize ? t.qtype : undefined,
    quantize_te: !!t.quantize_te,
    qtype_te: t.quantize_te ? t.qtype_te : undefined,
    low_vram: !!t.low_vram,
    layer_offloading: t.layer_offloading ? true : undefined,
    te_name_or_path: paths.te_path && paths.te_path.trim() ? paths.te_path.trim() : undefined,
    vae_path: paths.vae_path && paths.vae_path.trim() ? paths.vae_path.trim() : undefined,
    model_kwargs: modelKwargs,
  });

  const optimizerParams = {};
  if (t.weight_decay !== "" && t.weight_decay != null) optimizerParams.weight_decay = num(t.weight_decay, 0);
  for (const kv of String(t.optimizer_args || "").split(/\s+/).filter(Boolean)) {
    const [k, v] = kv.split("=");
    if (k && v !== undefined) optimizerParams[k] = isNaN(Number(v)) ? v : Number(v);
  }

  const train = dropUndefined({
    batch_size: S.parseIntList(job.dataset.batch_sizes, [1])[0],
    steps,
    gradient_accumulation: Math.max(1, int(t.grad_accum, 1)),
    train_unet: true,
    train_text_encoder: false,
    gradient_checkpointing: !!t.gradient_checkpointing,
    noise_scheduler: "flowmatch",
    timestep_type: t.timestep,
    content_or_style: t.timestep_bias,
    loss_type: t.loss_type,
    optimizer: t.optimizer,
    optimizer_params: optimizerParams,
    lr: num(t.lr, 1e-4),
    lr_scheduler: t.lr_scheduler,
    lr_scheduler_params: schedulerParams(t, steps),
    max_grad_norm: num(t.max_grad_norm, 1.0),
    dtype: t.mixed_precision,
    // edit prompts embed the control images, so they cannot be cached
    cache_text_embeddings: variant.edit ? false : !!t.cache_te,
    unload_text_encoder: false,
    skip_first_sample: !s.sample_first,
    disable_sampling: !s.enabled || buildSamples(job, variant).length === 0,
    ema_config: { use_ema: false },
  });

  const keep = int(t.keep_last_n, 0);
  const process0 = {
    type: "sd_trainer",
    training_folder: trainingFolder,
    device: "cuda:0",
    network,
    save: {
      dtype: t.save_precision,
      save_every: saveEvery,
      max_step_saves_to_keep: keep > 0 ? keep : 1000,
      save_format: "safetensors",
    },
    datasets: buildDatasets(job, variant),
    train,
    logging: { log_every: 1 },
    model,
    sample: {
      sampler: "flowmatch",
      sample_every: sampleEvery,
      width: int(s.width, 1024),
      height: int(s.height, 1024),
      samples: buildSamples(job, variant),
      neg: (s.negative || "").replace(/[\r\n]+/g, " ").trim(),
      seed: int(s.seed, 42),
      walk_seed: false,
      guidance_scale: num(s.cfg, 3),
      sample_steps: int(s.steps, 30),
    },
  };

  S.deepMerge(process0, parseOverrides(job.advanced.overrides, "yaml"));

  const config = {
    job: "extension",
    config: { name, process: [process0] },
    meta: { name: "[name]", version: "1.0" },
  };

  const root = settings.backends.aitk.root;
  const configPath = path.join(jobDir, "config.yaml");
  const gpuList = String(t.gpu_ids || "").split(",").map((x) => x.trim()).filter(Boolean);
  const env = {};
  // ai-toolkit trains on one device; pick it with CUDA_VISIBLE_DEVICES
  if (gpuList.length) env.CUDA_VISIBLE_DEVICES = gpuList[0];
  if (settings.hf_token) env.HF_TOKEN = settings.hf_token;

  const saveRoot = path.join(trainingFolder, name);
  return {
    files: [{ path: configPath, content: YAML.dump(config, { lineWidth: 200, noRefs: true }) }],
    args: [path.join(root, "run.py"), configPath],
    cwd: root,
    env,
    outputDir: saveRoot,
    samplesDir: path.join(saveRoot, "samples"),
    logsDir: path.join(jobDir, "logs"),
    configPath,
    steps,
  };
}

function checkSettings(settings) {
  return settings.backends.aitk.root ? [] : ["Set the ai-toolkit folder in Global Settings."];
}

module.exports = { build, buildDatasets, buildSamples, epochsToSteps, checkSettings };
