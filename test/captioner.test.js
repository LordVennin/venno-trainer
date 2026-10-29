const test = require("node:test");
const assert = require("node:assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const C = require("../public/js/captions");
const tagger = require("../lib/tagger");
const captioner = require("../lib/captioner");
const { scanFolder } = require("../lib/datasets");

function tmpDataset(captions) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "venno_cap_"));
  for (const [name, cap] of Object.entries(captions)) {
    fs.writeFileSync(path.join(dir, name), "x");
    if (cap != null) fs.writeFileSync(path.join(dir, name.replace(/\.\w+$/, ".txt")), cap + "\n");
  }
  return dir;
}
const read = (dir, f) => (fs.existsSync(path.join(dir, f)) ? fs.readFileSync(path.join(dir, f), "utf8").trim() : null);

// A stand-in for the tagger process: records calls, answers with the prompt it got.
function fakeService({ fail = [], preds } = {}) {
  return {
    calls: [],
    async tag(paths, args) {
      this.calls.push(["tag", paths.length, args]);
      return { device: "cpu", results: paths.map((p) => ({ path: p, preds: preds || [["red_fox", 0.9, "species"], ["some_artist", 0.99, "artist"], ["hi_res", 0.95, "meta"], ["tail", 0.2, "general"]] })) };
    },
    async caption(paths, prompts, args) {
      this.calls.push(["caption", paths.length, args]);
      return {
        device: "cuda",
        results: paths.map((p, i) =>
          fail.some((f) => p.endsWith(f)) ? { path: p, error: "broken image" } : { path: p, text: `This image shows a fox.\n\nPrompt was: ${prompts[i]}` },
        ),
      };
    },
  };
}

function stores(dir) {
  return { undo: new tagger.UndoStore(() => path.join(dir, ".state", "undo.json")), manifest: new captioner.Manifest(() => path.join(dir, ".state", "done.json")) };
}

const OPTS = { model: "joycaption-beta-one", quant: "bf16", batch_size: 2, template: C.DEFAULT_TEMPLATE, fallback_template: C.DEFAULT_FALLBACK, length: "long", hint_source: "caption", max_tags: 60, gen: { temperature: 0.6 } };

test("template filling, fallback without tags, and mixed lengths", () => {
  const p = C.buildPrompt(["red fox", "tail"], { template: "Write a {length} caption. Tags: {tags}. Again {tags}.", length: "short" });
  assert.strictEqual(p.prompt, "Write a short caption. Tags: red fox, tail. Again red fox, tail.");
  assert.strictEqual(C.buildPrompt([], { length: "medium" }).prompt, C.DEFAULT_FALLBACK.replace("{length}", "medium-length"));
  assert.ok(C.buildPrompt(["a"], { length: "long" }).prompt.includes("These tags were written by a person and are accurate: a."));
  const seen = new Set([0, 0.4, 0.9].map((r) => C.buildPrompt(["a"], { length: "mixed" }, () => r).length));
  assert.deepStrictEqual([...seen].sort(), ["long", "medium-length", "short"]);
});

test("tag hints: comma or space separated, cleaned, capped; prose gives none", () => {
  assert.deepStrictEqual(C.parseTagText("red_fox solo ^_^"), ["red_fox", "solo", "^_^"]);
  assert.deepStrictEqual(C.parseTagText("fox, red fur\nsolo"), ["fox", "red fur", "solo"]);
  assert.deepStrictEqual(C.parseTagText("A red fox sits on a mossy rock, looking up at the sky."), []);
  const hints = C.cleanHints(["red_fox", "star_\\(sky\\)", "^_^", "Red Fox", "watermark", "solo"], { exclude: "watermark", underscores: true });
  assert.deepStrictEqual(hints, ["red fox", "star (sky)", "^_^", "solo"]);
  assert.deepStrictEqual(C.cleanHints(["a_b"], { underscores: false }), ["a_b"]);
  assert.strictEqual(C.cleanHints(Array.from({ length: 100 }, (_, i) => `t${i}`), { max_tags: 60 }).length, 60);
  // from a tagger: its thresholds, and artist/meta left out of the hints
  const preds = [["red_fox", 0.9, "species"], ["some_artist", 0.99, "artist"], ["hi_res", 0.95, "meta"], ["tail", 0.2, "general"]];
  assert.deepStrictEqual(C.hintsFromPredictions(preds, { general: 0.35, include: { artist: true, meta: true } }, { include: { artist: false, meta: false } }), ["red fox"]);
  assert.deepStrictEqual(C.hintsFromPredictions([["fox", 0.1, "species", true], ["cat", 0.9, "species", false]], {}, {}), ["fox"]);
});

test("captions are one line without filler openers", () => {
  assert.strictEqual(C.cleanCaption("  This image shows a red fox.\n\nIt sits on a rock.  "), "A red fox. It sits on a rock.");
  assert.strictEqual(C.cleanCaption("In this image, the image depicts: a wolf"), "A wolf");
  assert.strictEqual(C.cleanCaption("Behold a wolf", { strip: "Behold\n" }), "A wolf");
  assert.strictEqual(C.cleanCaption("This image shows a fox", { strip: "" }), "This image shows a fox");
});

test("backups use an extension the scanner, trainer and Tagger ignore", () => {
  assert.strictEqual(C.backupPath("/d/img.01.png"), "/d/img.01.tags");
  const dir = tmpDataset({ "a.png": "fox" });
  fs.writeFileSync(path.join(dir, "a.tags"), "fox");
  fs.writeFileSync(path.join(dir, "b.tags"), "wolf");
  assert.deepStrictEqual(scanFolder(dir, ".txt"), { exists: true, images: 1, captioned: 1, missing: [] });
  assert.deepStrictEqual(tagger.listFolder(dir, ".txt").map((i) => i.name), ["a.png"]);
});

test("a run captions uncaptioned images with their hints, one line each", async () => {
  const dir = tmpDataset({ "a.png": "red_fox, solo", "b.png": null, "c.png": null });
  fs.writeFileSync(path.join(dir, "c.tags"), "grey_wolf"); // a backup gives hints for an uncaptioned image too
  const { undo, manifest } = stores(dir);
  const svc = fakeService();
  const run = new captioner.CaptionRun(svc, undo, manifest, { items: tagger.listFolder(dir, ".txt"), captionExt: ".txt", caption: OPTS, mode: "skip" });
  const st = await run.run();
  assert.strictEqual(st.state, "done");
  assert.deepStrictEqual([st.written, st.skipped, st.total], [2, 1, 2]);
  assert.strictEqual(read(dir, "a.txt"), "red_fox, solo");
  const b = read(dir, "b.txt");
  assert.ok(b.startsWith("A fox. Prompt was: Write a long descriptive caption") && !b.includes("\n"));
  assert.ok(b.includes(C.DEFAULT_FALLBACK.replace("{length}", "long")), "no tags: the fallback template");
  assert.ok(read(dir, "c.txt").includes("accurate: grey wolf."));
  assert.deepStrictEqual(svc.calls.map((c) => c[0]), ["caption"], "batch size 2: one model call");
  assert.ok(manifest.has(path.join(dir, "b.txt")) && manifest.has(path.join(dir, "c.txt")));
});

test("replace with backups, resume skips converted images, undo restores everything", async () => {
  const dir = tmpDataset({ "a.png": "red_fox, solo", "b.png": "wolf", "c.png": "cat" });
  const { undo, manifest } = stores(dir);
  const items = tagger.listFolder(dir, ".txt");
  // first run stops after the first batch
  const svc = fakeService();
  const run1 = new captioner.CaptionRun(svc, undo, manifest, { items, captionExt: ".txt", caption: { ...OPTS, batch_size: 1 }, mode: "replace", backup: true, skipConverted: true });
  const orig = svc.caption.bind(svc);
  svc.caption = async (...a) => {
    const r = await orig(...a);
    run1.cancel();
    return r;
  };
  assert.strictEqual((await run1.run()).state, "cancelled");
  assert.strictEqual(read(dir, "a.tags"), "red_fox, solo");
  assert.ok(read(dir, "a.txt").includes("accurate: red fox, solo."));
  assert.strictEqual(read(dir, "b.tags"), null);
  // resume: a is skipped, b and c are captioned
  const svc2 = fakeService();
  const st = await new captioner.CaptionRun(svc2, undo, manifest, { items, captionExt: ".txt", caption: OPTS, mode: "replace", backup: true, skipConverted: true }).run();
  assert.deepStrictEqual([st.written, st.skipped], [2, 1]);
  assert.strictEqual(read(dir, "b.tags"), "wolf");
  assert.ok(read(dir, "c.txt").startsWith("A fox."));
  // re-captioning a converted image takes its hints from the backup, not the prose caption
  const svc3 = fakeService();
  await new captioner.CaptionRun(svc3, undo, manifest, { items: items.slice(0, 1), captionExt: ".txt", caption: OPTS, mode: "replace", backup: true }).run();
  assert.ok(read(dir, "a.txt").includes("accurate: red fox, solo."));
  // undo the three runs, newest first
  for (let i = 0; i < 3; i++) {
    const r = undo.undo();
    if (r.extra && r.extra.joycaption) manifest.remove(r.extra.joycaption);
  }
  assert.deepStrictEqual(["a", "b", "c"].map((n) => read(dir, n + ".txt")), ["red_fox, solo", "wolf", "cat"]);
  assert.deepStrictEqual(["a", "b", "c"].map((n) => read(dir, n + ".tags")), [null, null, null]);
  assert.ok(!manifest.has(path.join(dir, "a.txt")) && !manifest.has(path.join(dir, "c.txt")));
});

test("tags + caption layout: the original tags on line 1, the caption on line 2", async () => {
  const dir = tmpDataset({ "a.png": "red_fox, solo", "b.png": "A wolf in the snow, howling at the moon.", "c.png": null });
  fs.writeFileSync(path.join(dir, "b.tags"), "grey_wolf, snow"); // converted earlier, tags kept
  const { undo, manifest } = stores(dir);
  const items = tagger.listFolder(dir, ".txt");
  const opts = { items, captionExt: ".txt", caption: OPTS, mode: "replace", layout: "tags_caption" };
  const st = await new captioner.CaptionRun(fakeService(), undo, manifest, opts).run();
  assert.strictEqual(st.written, 3);
  const lines = (f) => read(dir, f).split("\n");
  const a = lines("a.txt");
  assert.strictEqual(a.length, 2);
  assert.strictEqual(a[0], "red_fox, solo", "tags as written");
  assert.ok(a[1].startsWith("A fox.") && a[1].includes("accurate: red fox, solo."));
  assert.strictEqual(lines("b.txt")[0], "grey_wolf, snow", "a converted caption gets its tags back from the backup");
  assert.strictEqual(lines("c.txt").length, 1, "no tags on file: the caption alone");
  // a second run takes its hints and tags from line 1 only
  await new captioner.CaptionRun(fakeService(), undo, manifest, { ...opts, items: items.slice(0, 1) }).run();
  const again = lines("a.txt");
  assert.strictEqual(again.length, 2);
  assert.strictEqual(again[0], "red_fox, solo");
  assert.ok(again[1].includes("accurate: red fox, solo.") && !again[1].includes("accurate: red fox, solo, A fox"));
  // the preview shows the file text; the tagger's hints stand in when the file has no tags
  const pv = await new captioner.CaptionRun(fakeService(), undo, manifest, {
    ...opts, items: items.slice(2), caption: { ...OPTS, hint_source: "tagger" }, tagger: { model: "wd", options: {} }, preview: 5,
  }).run();
  const [tagLine, capLine] = pv.results[0].text.split("\n");
  assert.ok(pv.results[0].tags.length && tagLine === pv.results[0].tags.join(", "), pv.results[0].text);
  assert.strictEqual(capLine, pv.results[0].caption);
  assert.strictEqual(read(dir, "c.txt").split("\n").length, 1, "a preview writes nothing");
  for (let i = 0; i < 2; i++) undo.undo();
  assert.deepStrictEqual(["a", "b"].map((n) => read(dir, n + ".txt")), ["red_fox, solo", "A wolf in the snow, howling at the moon."]);
  assert.strictEqual(read(dir, "c.txt"), null);
});

test("tagger hints: every image is tagged before captioning starts (one model swap)", async () => {
  const dir = tmpDataset({ "a.png": null, "b.png": null, "c.png": null, "d.png": null, "e.png": null });
  const { undo, manifest } = stores(dir);
  const svc = fakeService();
  const st = await new captioner.CaptionRun(svc, undo, manifest, {
    items: tagger.listFolder(dir, ".txt"), captionExt: ".txt", caption: { ...OPTS, hint_source: "tagger", hint_include: { artist: false, meta: false } },
    tagger: { model: "hydra-3.5", device: "cuda", batch_size: 1, options: { general: 0.35, include: { artist: true, meta: true }, model_options: { confidence: 0.5 } } },
    mode: "skip",
  }).run();
  assert.strictEqual(st.written, 5);
  const kinds = svc.calls.map((c) => c[0]);
  assert.deepStrictEqual(kinds, ["tag", "tag", "caption", "caption", "caption"]);
  assert.ok(kinds.lastIndexOf("tag") < kinds.indexOf("caption"), "all tagging happens before captioning");
  assert.deepStrictEqual(svc.calls[0][2].options, { confidence: 0.5 });
  assert.ok(read(dir, "a.txt").includes("accurate: red fox."), "artist and meta tags are not hints");
});

test("preview captions a sample and writes nothing; failures are reported", async () => {
  const dir = tmpDataset(Object.fromEntries(Array.from({ length: 10 }, (_, i) => [`i${i}.png`, `tag${i}`])));
  const { undo, manifest } = stores(dir);
  const st = await new captioner.CaptionRun(fakeService({ fail: ["i3.png"] }), undo, manifest, {
    items: tagger.listFolder(dir, ".txt"), captionExt: ".txt", caption: OPTS, mode: "replace", backup: true, preview: 4, rng: () => 0.3,
  }).run();
  assert.strictEqual(st.total, 4);
  assert.strictEqual(st.written, 0);
  assert.strictEqual(st.results.length + st.errors.length, 4);
  for (const r of st.results) {
    assert.deepStrictEqual(r.tags, [read(dir, r.name.replace(".png", ".txt"))]);
    assert.ok(r.prompt.includes(`accurate: ${r.tags[0]}.`) && r.caption.startsWith("A fox."));
  }
  assert.deepStrictEqual(fs.readdirSync(dir).filter((f) => f.endsWith(".tags")), []);
  assert.strictEqual(undo.list().length, 0);
  // a failing image in a real run: reported, its caption untouched
  const st2 = await new captioner.CaptionRun(fakeService({ fail: ["i3.png"] }), undo, manifest, {
    items: tagger.listFolder(dir, ".txt"), captionExt: ".txt", caption: OPTS, mode: "replace",
  }).run();
  assert.deepStrictEqual([st2.written, st2.errors], [9, ["i3.png: broken image"]]);
  assert.strictEqual(read(dir, "i3.txt"), "tag3");
});

test("JoyCaption refuses to start while training runs unless Device is GPU", async () => {
  assert.strictEqual(captioner.captionDeviceError({ device: "auto", training: false }), null);
  assert.strictEqual(captioner.captionDeviceError({ device: "cuda", training: true }), null);
  assert.match(captioner.captionDeviceError({ device: "auto", training: true }), /training run is using the GPU/);

  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "venno_srv_"));
  process.env.VENNO_JOBS_DIR = path.join(tmp, "jobs");
  process.env.VENNO_SETTINGS = path.join(tmp, "settings.json");
  const { app, runner } = require("../server");
  const srv = app.listen(0);
  const port = srv.address().port;
  const dir = tmpDataset({ "a.png": "fox" });
  const post = (url, body) =>
    fetch(`http://127.0.0.1:${port}${url}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(async (r) => [r.status, await r.json()]);
  try {
    runner.runs.set("some job", { pid: 0 });
    const item = { dir, name: "a.png" };
    let [status, body] = await post("/api/tagger/caption-run", { items: [item], options: { device: "auto" } });
    assert.strictEqual(status, 409);
    assert.match(body.error, /training run is using the GPU/);
    [status, body] = await post("/api/tagger/joycaption", { ...item, options: { device: "auto" } });
    assert.strictEqual(status, 409);
    [status, body] = await post("/api/tagger/caption-run", { items: [item], caption_ext: ".tags", options: { device: "cuda" } });
    assert.strictEqual(status, 400, "the backup extension cannot be the caption extension");
    assert.strictEqual(read(dir, "a.txt"), "fox");
  } finally {
    runner.runs.clear();
    srv.close();
  }
});

test("the browser scripts share one global scope: no top-level name is declared twice", () => {
  const seen = new Map();
  for (const f of ["app.js", "tagger.js", "captioner.js"]) {
    const src = fs.readFileSync(path.join(__dirname, "..", "public", "js", f), "utf8");
    for (const m of src.matchAll(/^(?:const|let|var|function|async function|class) ([A-Za-z_$][\w$]*)/gm)) {
      assert.ok(!seen.has(m[1]), `${m[1]} is declared in both ${seen.get(m[1])} and ${f}`);
      seen.set(m[1], f);
    }
  }
});
