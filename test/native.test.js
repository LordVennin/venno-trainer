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
  assert.strictEqual(m.network.ignore_layers, ""); // all 42 blocks (layers.22. was a mistake)
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

test("captions are written next to images, never over existing ones", () => {
  const fs = require("fs");
  const os = require("os");
  const { scanFolder, writeCaption, imageInFolder } = require("../lib/datasets");
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "venno-cap-"));
  for (const n of ["a.png", "b.jpg", "c.webp"]) fs.writeFileSync(path.join(dir, n), "x");
  fs.writeFileSync(path.join(dir, "a.txt"), "has one");
  assert.deepStrictEqual(scanFolder(dir, ".txt", Infinity).missing, ["b.jpg", "c.webp"]);
  writeCaption(dir, "b.jpg", "txt", "  sks, 1girl  ");
  assert.strictEqual(fs.readFileSync(path.join(dir, "b.txt"), "utf8"), "sks, 1girl\n");
  assert.throws(() => writeCaption(dir, "a.png", ".txt", "new"), /already exists/);
  assert.throws(() => imageInFolder(dir, "../etc/passwd"), /Not an image/);
  assert.throws(() => imageInFolder(dir, "a.txt"), /Not an image/);
  assert.deepStrictEqual(scanFolder(dir, ".txt", Infinity).missing, ["c.webp"]);
  fs.rmSync(dir, { recursive: true });
});

test("model defaults: MOSS trains all blocks with its report timesteps; Anima skips adaLN; EMA on", () => {
  const m = S.defaultJob("moss_image");
  assert.strictEqual(m.network.ignore_layers, "");
  assert.strictEqual(m.training.timestep, "moss_report");
  assert.strictEqual(m.training.clean_end_share, "60:7:30:8:3");
  assert.strictEqual(m.training.ema_decay, 0.98);
  const a = S.defaultJob("anima");
  assert.strictEqual(a.network.ignore_layers, "adaln_modulation");
  assert.strictEqual(a.training.ema_decay, 0.98);
  assert.deepStrictEqual(S.validateJob({ ...m, dataset: { ...m.dataset, folders: [{ ...m.dataset.folders[0], path: "/d" }] } }), []);
});

test("older MOSS jobs are warned about block 22 and shift-5 timesteps; bad settings are errors", () => {
  const folders = [{ ...S.defaultFolder(), path: "/d" }];
  const old = S.normalizeJob({ model: "moss_image", training: { timestep: "uniform", flow_shift: 5 }, network: { ignore_layers: "layers.22." }, dataset: { folders } });
  const msgs = S.validateJob(old).map((i) => `${i.level}: ${i.msg}`);
  assert.strictEqual(msgs.length, 2);
  assert.match(msgs[0], /^warn: Block 22 is left out/);
  assert.match(msgs[1], /^warn: This job uses AI-Toolkit's shift-5/);
  assert.strictEqual(S.validateJob(S.normalizeJob({ model: "moss_image", network: { ignore_layers: "layers.2." }, dataset: { folders } })).length, 0, "layers.2. is not block 22");
  const bad = S.normalizeJob({ model: "moss_image", training: { clean_end_share: "60:7", ema_decay: 1.5 }, dataset: { folders } });
  assert.deepStrictEqual(S.validateJob(bad).map((i) => i.level), ["error", "error"]);
  const anima = S.normalizeJob({ model: "anima", training: { timestep: "moss_report" }, dataset: { folders } });
  assert.match(S.validateJob(anima)[0].msg, /MOSS-Image jobs only/);
});

test("caption mix: blank or three tags:caption:both weights", () => {
  assert.strictEqual(S.defaultFolder().caption_mix, "");
  for (const ok of ["", "25:25:50", "1:1:2", "0:50:50%"]) assert.ok(S.captionMixValid(ok), ok);
  for (const bad of ["25:75", "a:b:c", "0:0:0", "-1:1:1", "25::50"]) assert.ok(!S.captionMixValid(bad), bad);
  const job = S.normalizeJob({ model: "moss_image", dataset: { folders: [{ ...S.defaultFolder(), path: "/data/fox", caption_mix: "25:75" }] } });
  const [issue] = S.validateJob(job);
  assert.strictEqual(issue.level, "error");
  assert.match(issue.msg, /Caption mix "25:75" \(fox\)/);
});
