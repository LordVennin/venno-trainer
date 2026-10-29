const test = require("node:test");
const assert = require("node:assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const T = require("../public/js/tags");
const tagger = require("../lib/tagger");

function tmpDataset(captions) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "venno_tag_"));
  for (const [name, cap] of Object.entries(captions)) {
    fs.writeFileSync(path.join(dir, name), "x");
    if (cap != null) fs.writeFileSync(path.join(dir, name.replace(/\.\w+$/, ".txt")), cap + "\n");
  }
  return dir;
}

test("tags match across spellings and count once per image", () => {
  assert.strictEqual(T.tagKey("Long_Hair"), "long hair");
  assert.strictEqual(T.tagKey("star \\(sky\\)"), "star (sky)");
  const counts = T.countTags(["fox, long_hair, long hair", "Fox, red fur", null, "red fur"]);
  assert.deepStrictEqual(counts.map((c) => [c.tag, c.count]), [["red fur", 2], ["fox", 2], ["long_hair", 1]].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])));
  assert.strictEqual(counts.find((c) => c.key === "long hair").variants, 2);
});

test("predictions become tags: thresholds per category, exclusions, prefix, rating", () => {
  const preds = [
    ["general", 0.9, "rating"], ["sensitive", 0.2, "rating"],
    ["long_hair", 0.8, "general"], ["^_^", 0.7, "general"], ["star_(sky)", 0.5, "general"], ["blurry", 0.6, "general"],
    ["low_conf", 0.2, "general"], ["some_character", 0.9, "character"], ["weak_character", 0.6, "character"],
  ];
  const base = { general: 0.35, character: 0.85, include: { general: true, character: true, rating: false }, exclude: "blurry", prefix: "sks" };
  assert.deepStrictEqual(T.selectPredictions(preds, base), ["sks", "some character", "long hair", "^_^", "star (sky)"]);
  const withRating = T.selectPredictions(preds, { ...base, include: { ...base.include, rating: true }, escape: true, underscores: false, max_tags: 2 });
  assert.deepStrictEqual(withRating, ["sks", "some_character", "long_hair", "general"]);
  assert.deepStrictEqual(T.selectPredictions([["fox", 0.9, "species"], ["hi_res", 0.99, "meta"]], { general: 0.5, include: { meta: false } }), ["fox"]);
  assert.deepStrictEqual(T.selectPredictions([["a_(b)", 0.9, "general"]], { escape: true }), ["a \\(b\\)"]);
});

test("merge modes and bulk operations", () => {
  assert.strictEqual(T.mergeCaption(null, ["a", "b"], "skip"), "a, b");
  assert.strictEqual(T.mergeCaption("x", ["a"], "skip"), null);
  assert.strictEqual(T.mergeCaption("x, a", ["a", "b"], "append"), "x, a, b");
  assert.strictEqual(T.mergeCaption("x", ["a"], "overwrite"), "a");
  assert.strictEqual(T.applyOp("fox, red fur", { type: "remove", tag: "red_fur" }), "fox");
  assert.strictEqual(T.applyOp("fox, red fur", { type: "remove", tag: "wolf" }), null);
  assert.strictEqual(T.applyOp("fox, vulpine", { type: "rename", tag: "vulpine", to: "fox" }), "fox");
  assert.strictEqual(T.applyOp("fox", { type: "add", tag: "sks", position: "start" }), "sks, fox");
  assert.strictEqual(T.applyOp("fox", { type: "add", tag: "tail", position: "end" }), "fox, tail");
  assert.strictEqual(T.applyOp("fox, tail", { type: "add", tag: "Tail" }), null);
});

test("a caption line after the tags is kept by every tag operation", () => {
  const cap = "fox, red fur\nA red fox sits in tall grass, looking at the camera.";
  assert.deepStrictEqual(T.splitCaption(cap), { tags: "fox, red fur", prose: "A red fox sits in tall grass, looking at the camera." });
  assert.deepStrictEqual(T.captionTags(cap), ["fox", "red fur"]);
  assert.deepStrictEqual(T.captionTags("fox\nred fur\ntail"), ["fox", "red fur", "tail"], "one tag per line is still all tags");
  assert.strictEqual(T.applyOp(cap, { type: "remove", tag: "red_fur" }), "fox\nA red fox sits in tall grass, looking at the camera.");
  assert.strictEqual(T.applyOp(cap, { type: "add", tag: "sks", position: "start" }), "sks, fox, red fur\nA red fox sits in tall grass, looking at the camera.");
  assert.strictEqual(T.applyOp(cap, { type: "remove", tag: "looking at the camera" }), null, "prose is not a tag");
  assert.strictEqual(T.mergeCaption(cap, ["tail"], "append"), "fox, red fur, tail\nA red fox sits in tall grass, looking at the camera.");
  assert.strictEqual(T.mergeCaption(cap, ["wolf"], "overwrite"), "wolf\nA red fox sits in tall grass, looking at the camera.");
  assert.deepStrictEqual(T.countTags([cap]).map((c) => c.tag), ["fox", "red fur"]);
  assert.ok(!T.matchesFilter({ name: "a.png", caption: cap }, "*camera*"));
});

test("filter expressions", () => {
  const it = { name: "img_01.png", caption: "fox, red fur, solo" };
  assert.ok(T.matchesFilter(it, "fox, red_fur"));
  assert.ok(!T.matchesFilter(it, "fox, -solo"));
  assert.ok(T.matchesFilter(it, "*fur*"));
  assert.ok(T.matchesFilter(it, "name:01"));
  assert.ok(!T.matchesFilter({ name: "b.png", caption: null }, "fox"));
  assert.ok(T.matchesFilter({ name: "b.png", caption: null }, "-fox"));
});

test("listing, bulk edit and undo on caption files", () => {
  const dir = tmpDataset({ "a.png": "fox, red fur", "b.jpg": "wolf", "c.webp": null });
  const items = tagger.listFolder(dir, ".txt");
  assert.deepStrictEqual(items.map((i) => [i.name, i.caption]), [["a.png", "fox, red fur"], ["b.jpg", "wolf"], ["c.webp", null]]);
  const undo = new tagger.UndoStore(() => path.join(dir, ".undo.json"));
  assert.strictEqual(tagger.bulkEdit(items, ".txt", { type: "add", tag: "sks", position: "start" }, undo, "add sks"), 3);
  assert.strictEqual(fs.readFileSync(path.join(dir, "c.txt"), "utf8"), "sks\n");
  assert.strictEqual(tagger.bulkEdit(items, ".txt", { type: "remove", tag: "red_fur" }, undo, "rm"), 1);
  assert.deepStrictEqual(undo.list().map((e) => [e.label, e.files]), [["add sks", 3], ["rm", 1]]);
  undo.undo();
  assert.strictEqual(fs.readFileSync(path.join(dir, "a.txt"), "utf8"), "sks, fox, red fur\n");
  undo.undo();
  assert.strictEqual(fs.readFileSync(path.join(dir, "a.txt"), "utf8"), "fox, red fur\n");
  assert.ok(!fs.existsSync(path.join(dir, "c.txt")), "a caption created by the edit is removed again");
  assert.strictEqual(undo.undo(), null);
});

test("an auto-tag run writes captions by mode and can be undone", async () => {
  const dir = tmpDataset({ "a.png": "keep me", "b.png": null, "bad.png": null });
  const fake = {
    calls: 0,
    async tag(paths) {
      this.calls++;
      return {
        device: "cpu",
        results: paths.map((p) => (p.endsWith("bad.png") ? { path: p, error: "broken" } : { path: p, preds: [["fox", 0.9, "general"], ["tail", 0.2, "general"]] })),
      };
    },
  };
  const undo = new tagger.UndoStore(() => path.join(dir, ".undo.json"));
  const items = tagger.listFolder(dir, ".txt");
  const opts = { items, captionExt: ".txt", model: "m", device: "cpu", batch_size: 1, options: { general: 0.35, prefix: "sks" } };
  let state = await new tagger.TagRun(fake, undo, { ...opts, mode: "skip" }).run();
  assert.strictEqual(state.state, "done");
  assert.strictEqual(state.written, 1);
  assert.strictEqual(state.skipped, 1);
  assert.strictEqual(state.errors.length, 1);
  assert.strictEqual(fs.readFileSync(path.join(dir, "a.txt"), "utf8"), "keep me\n");
  assert.strictEqual(fs.readFileSync(path.join(dir, "b.txt"), "utf8"), "sks, fox\n");
  state = await new tagger.TagRun(fake, undo, { ...opts, mode: "append" }).run();
  assert.strictEqual(fs.readFileSync(path.join(dir, "a.txt"), "utf8"), "keep me, sks, fox\n");
  assert.strictEqual(fs.readFileSync(path.join(dir, "b.txt"), "utf8"), "sks, fox\n");
  undo.undo();
  assert.strictEqual(fs.readFileSync(path.join(dir, "a.txt"), "utf8"), "keep me\n");
  const run = new tagger.TagRun(fake, undo, { ...opts, mode: "overwrite" });
  run.cancel();
  assert.strictEqual((await run.run()).state, "cancelled");
});

test("a model that picks its own tags (Hydra 3.5) overrides the thresholds", () => {
  const preds = [["fox", 0.4, "species", true], ["red_fur", 0.9, "color", false], ["hi_res", 0.95, "meta", true], ["tail", 0.6, "general", true]];
  assert.deepStrictEqual(T.selectPredictions(preds, { general: 0.5, include: { meta: false } }), ["fox", "tail"]);
});

test("an auto-tag run hands the model options to the tagger", async () => {
  const dir = tmpDataset({ "a.png": null });
  let seen;
  const fake = {
    async tag(paths, args) {
      seen = args.options;
      return { device: "cpu", results: paths.map((p) => ({ path: p, preds: [["fox", 0.3, "species", true], ["cat", 0.99, "species", false]] })) };
    },
  };
  const undo = new tagger.UndoStore(() => path.join(dir, ".undo.json"));
  const run = new tagger.TagRun(fake, undo, {
    items: tagger.listFolder(dir, ".txt"), captionExt: ".txt", model: "hydra-3.5", device: "cpu", batch_size: 1, mode: "skip",
    options: { model_options: { confidence: 0.7 } },
  });
  await run.run();
  assert.deepStrictEqual(seen, { confidence: 0.7 });
  assert.strictEqual(fs.readFileSync(path.join(dir, "a.txt"), "utf8"), "fox\n");
});
