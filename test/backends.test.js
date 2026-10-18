const test = require("node:test");
const assert = require("node:assert");
const path = require("path");
const TOML = require("@iarna/toml");
const YAML = require("js-yaml");
const S = require("../public/js/schema");
const anima = require("../lib/backends/anima");
const aitk = require("../lib/backends/aitk");

const settings = {
  backends: { anima: { root: "/opt/anima" }, aitk: { root: "/opt/aitk" } },
  models: {
    anima: { dit_path: "/m/dit.safetensors", qwen3_path: "/m/te.safetensors", vae_path: "/m/vae.safetensors" },
    moss_image: { name_or_path: "", te_path: "", vae_path: "" },
  },
  hf_token: "hf_x",
};

function file(result, name) {
  return result.files.find((f) => path.basename(f.path) === name).content;
}

function folders(job) {
  job.dataset.folders = [
    Object.assign(S.defaultFolder(), { path: "/data/char", repeats: 10, caption_dropout: 0.1, loss_weight: 2 }),
    Object.assign(S.defaultFolder(), { path: "/data/style", repeats: 2, tag_dropout: 0.2, keep_tokens: 2 }),
    Object.assign(S.defaultFolder(), { path: "/data/off", enabled: false }),
  ];
  return job;
}

test("anima: dataset.toml has one block per resolution and only enabled folders", () => {
  const job = folders(S.defaultJob("anima"));
  job.dataset.resolutions = "768, 1024";
  job.dataset.batch_sizes = "4, 2";
  const ds = TOML.parse(file(anima.build(job, { jobDir: "/j", settings }), "dataset.toml"));
  assert.deepStrictEqual(ds.datasets.map((d) => d.resolution), [[768, 768], [1024, 1024]]);
  assert.deepStrictEqual(ds.datasets.map((d) => d.batch_size), [4, 2]);
  const subs = ds.datasets[0].subsets;
  assert.strictEqual(subs.length, 2);
  assert.strictEqual(subs[0].num_repeats, 10);
  assert.strictEqual(subs[0].caption_dropout_rate, 0.1);
  assert.strictEqual(subs[1].caption_tag_dropout_rate, 0.2);
  assert.strictEqual(subs[1].keep_tokens, 2);
});

test("anima: config.toml wires model paths, network and sampling", () => {
  const job = S.defaultJob("anima");
  job.sampling.prompts = [{ text: "1girl, solo" }, { text: "skip me", enabled: false }];
  job.sampling.negative = "lowres";
  const r = anima.build(job, { jobDir: "/j", settings });
  const cfg = TOML.parse(file(r, "config.toml"));
  assert.strictEqual(cfg.model_arguments.dit_path, "/m/dit.safetensors");
  assert.strictEqual(cfg.network_arguments.network_module, "networks.lora_anima");
  assert.strictEqual(cfg.training_arguments.max_train_epochs, 15);
  assert.strictEqual(cfg.sample_arguments.sample_every_n_epochs, 1);
  assert.deepStrictEqual(cfg.training_arguments.optimizer_args, ["weight_decay=0.01"]);
  const prompts = file(r, "sample_prompts.txt").trim().split("\n");
  assert.strictEqual(prompts.length, 1);
  assert.match(prompts[0], /^1girl, solo --w 832 --h 1216 --s 28 --l 4 --d 42 --n lowres$/);
  assert.ok(r.args.includes(path.join("/opt/anima", "anima_train_network.py")));
  assert.ok(!r.args.includes("--multi_gpu"));
});

test("anima: steps mode converts keep-last-N into sd-scripts step counts", () => {
  const job = S.defaultJob("anima");
  Object.assign(job.training, { duration_unit: "steps", max_steps: 1000, save_every_steps: 100, keep_last_n: 3, keep_last_n_states: 2 });
  const cfg = TOML.parse(file(anima.build(job, { jobDir: "/j", settings }), "config.toml"));
  assert.strictEqual(cfg.training_arguments.max_train_steps, 1000);
  assert.strictEqual(cfg.training_arguments.save_last_n_steps, 299);
  assert.strictEqual(cfg.training_arguments.save_last_n_steps_state, 199);
  assert.strictEqual(cfg.training_arguments.max_train_epochs, undefined);
});

test("anima: full finetune drops network args and uses anima_train.py; multi-GPU uses DDP", () => {
  const job = S.defaultJob("anima");
  job.network.type = "full_finetune";
  job.training.gpu_ids = "0,1";
  const r = anima.build(job, { jobDir: "/j", settings });
  const cfg = TOML.parse(file(r, "config.toml"));
  assert.strictEqual(cfg.network_arguments, undefined);
  assert.ok(r.args.includes(path.join("/opt/anima", "anima_train.py")));
  assert.ok(r.args.includes("--multi_gpu"));
  assert.strictEqual(r.env.CUDA_VISIBLE_DEVICES, "0,1");
});

test("anima: advanced TOML overrides are merged", () => {
  const job = S.defaultJob("anima");
  job.advanced.overrides = "[training_arguments]\nnoise_offset = 0.05\n";
  const cfg = TOML.parse(file(anima.build(job, { jobDir: "/j", settings }), "config.toml"));
  assert.strictEqual(cfg.training_arguments.noise_offset, 0.05);
  job.advanced.overrides = "not = [valid";
  assert.throws(() => anima.build(job, { jobDir: "/j", settings }), /not valid TOML/);
});

test("moss: YAML targets ai-toolkit moss_image with per-folder weights and dropout", () => {
  const job = folders(S.defaultJob("moss_image"));
  job.sampling.prompts = [{ text: "a cat", cfg: 3.5 }];
  const r = aitk.build(job, { jobDir: "/j", settings });
  const cfg = YAML.load(file(r, "config.yaml"));
  const p = cfg.config.process[0];
  assert.strictEqual(p.type, "sd_trainer");
  assert.strictEqual(p.model.arch, "moss_image");
  assert.strictEqual(p.model.name_or_path, "OpenMOSS-Team/MOSS-Image-SFT");
  assert.deepStrictEqual(p.model.model_kwargs, { uncond_skip_layers: [22] });
  assert.deepStrictEqual(p.network.network_kwargs.ignore_if_contains, ["layers.22."]);
  assert.strictEqual(p.datasets.length, 2);
  assert.strictEqual(p.datasets[0].num_repeats, 10);
  assert.strictEqual(p.datasets[0].loss_multiplier, 2);
  assert.strictEqual(p.datasets[1].token_dropout_rate, 0.2);
  assert.deepStrictEqual(p.datasets[0].resolution, [512, 768, 1024]);
  assert.strictEqual(p.train.steps, 3000);
  assert.strictEqual(p.train.cache_text_embeddings, true);
  assert.strictEqual(p.sample.samples[0].guidance_scale, 3.5);
  assert.strictEqual(p.save.max_step_saves_to_keep, 4);
  assert.deepStrictEqual(r.args, [path.join("/opt/aitk", "run.py"), path.join("/j", "config.yaml")]);
  assert.strictEqual(r.env.HF_TOKEN, "hf_x");
});

test("moss: epoch mode converts to steps from the dataset scan", () => {
  const job = S.defaultJob("moss_image");
  job.dataset.folders[0].path = "/d";
  Object.assign(job.training, { duration_unit: "epochs", max_epochs: 5, save_every_epochs: 2 });
  job.dataset.batch_sizes = "2";
  const r = aitk.build(job, { jobDir: "/j", settings, effectiveImages: 101 });
  const p = YAML.load(file(r, "config.yaml")).config.process[0];
  assert.strictEqual(p.train.steps, 51 * 5);
  assert.strictEqual(p.save.save_every, 102);
  assert.throws(() => aitk.build(job, { jobDir: "/j", settings, effectiveImages: 0 }), /dataset scan/);
});

test("moss edit variant: control folders, no TE caching, edit kwargs", () => {
  const job = S.defaultJob("moss_image");
  job.variant = "base_edit";
  Object.assign(job.dataset.folders[0], { path: "/d", control_path: "/c" });
  const p = YAML.load(file(aitk.build(job, { jobDir: "/j", settings }), "config.yaml")).config.process[0];
  assert.strictEqual(p.model.name_or_path, "OpenMOSS-Team/MOSS-Image-Base");
  assert.strictEqual(p.model.model_kwargs.edit, true);
  assert.strictEqual(p.datasets[0].control_path, "/c");
  assert.strictEqual(p.train.cache_text_embeddings, false);
});

test("validation flags tag dropout with cached TE and duplicate folders", () => {
  const job = folders(S.defaultJob("anima"));
  const issues = S.validateJob(job);
  assert.ok(issues.some((i) => i.level === "error" && /Tag dropout/.test(i.msg)));
  assert.ok(issues.some((i) => i.level === "warn" && /global caption dropout/.test(i.msg)));
  job.dataset.folders[1].path = "/data/char";
  assert.ok(S.validateJob(job).some((i) => /more than one dataset/.test(i.msg)));
  const moss = folders(S.defaultJob("moss_image"));
  assert.ok(S.validateJob(moss).every((i) => i.level !== "error"));
});

test("normalizeJob fills defaults and kohya folder names give repeats", () => {
  const job = S.normalizeJob({ model: "moss_image", dataset: { folders: [{ path: "/x" }] } });
  assert.strictEqual(job.variant, "sft");
  assert.strictEqual(job.dataset.folders[0].repeats, 1);
  assert.strictEqual(S.repeatsFromFolderName("12_char"), 12);
  assert.strictEqual(S.repeatsFromFolderName("char"), null);
});
