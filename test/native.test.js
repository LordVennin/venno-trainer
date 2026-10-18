const test = require("node:test");
const assert = require("node:assert");
const path = require("path");
const S = require("../public/js/schema");
const native = require("../lib/native");
const { parseProgress } = require("../lib/runner");

const settings = {
  python: "",
  cache_dir: "/cache",
  models_dir: "",
  hf_token: "hf_x",
  models: { anima: { dit_path: "/m/dit", qwen3_path: "/m/te", vae_path: "/m/vae" }, moss_image: { model_path: "" } },
};

test("run.json carries the normalized job, model paths and dirs", () => {
  const job = S.defaultJob("anima");
  job.training.gpu_ids = "1,0";
  const r = native.buildRun(job, settings, "/jobs/a");
  assert.strictEqual(r.run.job.model, "anima");
  assert.deepStrictEqual(r.run.model_paths, settings.models.anima);
  assert.strictEqual(r.run.output_dir, path.join("/jobs/a", "output"));
  assert.strictEqual(r.run.cache_dir, "/cache");
  assert.strictEqual(r.run.models_dir, path.join(native.REPO_ROOT, "models"));
  assert.deepStrictEqual(r.args, ["-m", "venno.train", path.join("/jobs/a", "run.json")]);
  assert.strictEqual(r.cwd, native.TRAINER_DIR);
  assert.strictEqual(r.env.CUDA_VISIBLE_DEVICES, "1");
  assert.strictEqual(r.env.HF_TOKEN, "hf_x");
  assert.strictEqual(r.samplesDir, path.join("/jobs/a", "output", "samples"));
});

test("Anima needs its three model paths; MOSS can download", () => {
  assert.deepStrictEqual(native.checkSettings(S.defaultJob("anima"), settings), []);
  const missing = native.checkSettings(S.defaultJob("anima"), { models: { anima: { dit_path: "/x" } } });
  assert.strictEqual(missing.length, 2);
  assert.deepStrictEqual(native.checkSettings(S.defaultJob("moss_image"), { models: {} }), []);
});

test("model defaults follow the original trainers", () => {
  const a = S.defaultJob("anima");
  assert.strictEqual(a.training.timestep, "logit_normal");
  assert.strictEqual(a.training.flow_shift, 3);
  assert.strictEqual(a.sampling.shift, 1);
  const m = S.defaultJob("moss_image");
  assert.strictEqual(m.variant, "sft");
  assert.strictEqual(m.training.flow_shift, 5);
  assert.strictEqual(m.network.ignore_layers, "layers.22.");
  assert.strictEqual(m.sampling.cfg, 3);
  assert.ok(S.textCached(m));
});

test("validation", () => {
  const job = S.defaultJob("anima");
  job.dataset.folders = [
    Object.assign(S.defaultFolder(), { path: "/d/a", tag_dropout: 0.2 }),
    Object.assign(S.defaultFolder(), { path: "/d/a" }),
  ];
  assert.ok(S.validateJob(job).some((i) => i.level === "error" && /more than one dataset/.test(i.msg)));
  job.dataset.folders[1].path = "/d/b";
  assert.ok(S.validateJob(job).every((i) => i.level !== "error"));
  job.training.te_variants = 1;
  assert.ok(S.validateJob(job).some((i) => i.level === "warn" && /caption variants/.test(i.msg)));
  job.training.cache_te = false; // live encoding re-randomizes every step
  assert.ok(S.validateJob(job).every((i) => i.level !== "warn"));
});

test("old v1 jobs normalize (removed edit variant falls back)", () => {
  const job = S.normalizeJob({ model: "moss_image", variant: "edit", dataset: { folders: [{ path: "/x" }] } });
  assert.strictEqual(job.variant, "sft");
  assert.strictEqual(job.dataset.folders[0].repeats, 1);
});

test("progress line from the trainer is parsed", () => {
  const p = parseProgress("steps:  12%|██  | 120/1000 [01:23<10:00, 1.44it/s, avr_loss=0.0912]");
  assert.deepStrictEqual(p, { step: 120, total: 1000, elapsed: "01:23", eta: "10:00", loss: 0.0912 });
  assert.strictEqual(parseProgress("caching latents: 50%| | 2/4 [00:01<00:01, 2.0it/s, avr_loss=0]").step, 2);
});

test("peak VRAM is read from the training progress line", () => {
  const p = parseProgress("steps:  10%|██ | 30/300 [00:30<04:30, 1.00it/s, avr_loss=0.05, vram=13.4G]");
  assert.strictEqual(p.vram, 13.4);
  assert.strictEqual(p.loss, 0.05);
  const r = native.buildRun(S.defaultJob("moss_image"), settings, "/j");
  assert.strictEqual(r.env.PYTORCH_CUDA_ALLOC_CONF, "expandable_segments:True");
});
