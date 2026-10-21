// JoyCaption runs: natural-language captions for many images, using each
// image's tags as hints. Phase 1 gathers the hints (from the caption file, or
// by running the selected tagger on every image first, so the models swap
// once per run, not per image); phase 2 captions. Captions are written as
// each batch finishes, recorded for Undo and in a manifest so a stopped run
// can resume. A preview run captions a random sample and writes nothing.
const EventEmitter = require("events");
const fs = require("fs");
const path = require("path");

const C = require("../public/js/captions");
const { imageInFolder } = require("./datasets");
const { captionPath, readCaption, writeCaptionFile } = require("./tagger");

const TRAINING_REFUSAL =
  "A training run is using the GPU. JoyCaption needs about 17 GB in bf16 (about 7 GB in 4-bit) and is far too slow on the CPU, " +
  "so it does not start while training runs. Set the captioner's Device to GPU to run it alongside training anyway.";

// JoyCaption only runs on the GPU: with Device "auto" it refuses while a
// training run holds the card instead of falling back to the CPU.
function captionDeviceError({ device, training }) {
  if (training && device !== "cuda") return TRAINING_REFUSAL;
  return null;
}

// Caption files JoyCaption wrote (so a resumed run can skip them).
class Manifest {
  constructor(file) {
    this.file = file; // () => path
    this.data = null;
  }

  _load() {
    if (this.data) return this.data;
    try {
      this.data = JSON.parse(fs.readFileSync(this.file(), "utf8"));
    } catch (e) {
      this.data = {};
    }
    return this.data;
  }

  has(cap) {
    return Object.prototype.hasOwnProperty.call(this._load(), cap);
  }

  add(caps) {
    const d = this._load();
    for (const c of caps) d[c] = Date.now();
  }

  remove(caps) {
    const d = this._load();
    for (const c of caps) delete d[c];
    this.save();
  }

  save() {
    fs.mkdirSync(path.dirname(this.file()), { recursive: true });
    fs.writeFileSync(this.file(), JSON.stringify(this._load()));
  }
}

// Converted already: JoyCaption wrote the current caption file, or a run in
// backup mode moved the original tags to the .tags file.
function isConverted(manifest, cap, backup) {
  return fs.existsSync(cap) && (manifest.has(cap) || fs.existsSync(backup));
}

// Tag hints from the image's files: the .tags backup if a run kept one
// (the caption may already be prose), else the caption.
function fileHints(cap, backup, opts) {
  const text = fs.existsSync(backup) ? readCaption(backup) : readCaption(cap);
  return C.cleanHints(C.parseTagText(text), opts);
}

function sample(list, n, rng) {
  const a = [...list];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a.slice(0, n);
}

class CaptionRun extends EventEmitter {
  // caption: {model, quant, batch_size, gen, template, fallback_template, length,
  //           strip, hint_source, hint_include, max_tags, exclude, underscores}
  // tagger:  {model, device, batch_size, options} for tagger hints
  constructor(service, undo, manifest, { items, captionExt, models_dir, hf_token, caption, tagger, mode = "skip", backup = false, skipConverted = false, preview = 0, rng = Math.random }) {
    super();
    Object.assign(this, { service, undo, manifest, items, captionExt, models_dir, hf_token, caption, tagger, mode, backup, skipConverted, preview, rng });
    this.state = {
      state: "running", phase: "preparing", preview: preview > 0, done: 0, total: 0, written: 0, skipped: 0,
      hint_done: 0, hint_total: 0, errors: [], results: [], mode, backup, started: Date.now(), phase_started: Date.now(),
    };
    this.cancelled = false;
    this.before = {};
    this.writtenCaps = [];
  }

  cancel() {
    this.cancelled = true;
  }

  _emit() {
    this.emit("progress", this.state);
  }

  _error(it, msg) {
    this.state.errors.push(`${it.name}: ${msg}`);
  }

  _candidates() {
    let todo = this.items.map((it) => {
      const image = imageInFolder(it.dir, it.name);
      return { ...it, image, cap: captionPath(image, this.captionExt), backup: C.backupPath(image) };
    });
    const total = todo.length;
    if (this.mode === "skip") todo = todo.filter((t) => !fs.existsSync(t.cap));
    if (this.skipConverted) todo = todo.filter((t) => !isConverted(this.manifest, t.cap, t.backup));
    this.state.skipped = total - todo.length;
    if (this.preview > 0) todo = sample(todo, this.preview, this.rng);
    return todo;
  }

  async _hints(todo) {
    const opts = this.caption;
    if (opts.hint_source !== "tagger" || !this.tagger) {
      for (const t of todo) t.tags = fileHints(t.cap, t.backup, opts);
      return todo;
    }
    this.state.phase = "hints";
    this.state.phase_started = Date.now();
    this.state.hint_total = todo.length;
    const chunk = Math.max(4, (this.tagger.batch_size || 4) * 4);
    const ok = [];
    for (let k = 0; k < todo.length && !this.cancelled; k += chunk) {
      const part = todo.slice(k, k + chunk);
      const r = await this.service.tag(part.map((t) => t.image), {
        model: this.tagger.model, device: this.tagger.device, models_dir: this.models_dir, hf_token: this.hf_token,
        batch_size: this.tagger.batch_size, options: (this.tagger.options || {}).model_options,
      });
      r.results.forEach((res, i) => {
        if (res.error) return this._error(part[i], res.error);
        part[i].tags = C.hintsFromPredictions(res.preds, this.tagger.options || {}, { ...opts, include: opts.hint_include });
        ok.push(part[i]);
      });
      this.state.hint_done = Math.min(todo.length, k + chunk);
      this._emit();
    }
    return ok;
  }

  _write(t, caption) {
    const old = readCaption(t.cap);
    if (this.backup && old != null && old.trim() && !fs.existsSync(t.backup)) {
      this.before[t.backup] = null;
      writeCaptionFile(t.backup, old);
    }
    if (!(t.cap in this.before)) this.before[t.cap] = old;
    writeCaptionFile(t.cap, caption);
    this.writtenCaps.push(t.cap);
    this.state.written++;
  }

  async run() {
    const opts = this.caption;
    try {
      let todo = this._candidates();
      this.state.total = todo.length;
      this._emit();
      todo = await this._hints(todo);
      this.state.phase = "captions";
      this.state.phase_started = Date.now();
      this.state.total = todo.length;
      const bs = Math.max(1, parseInt(opts.batch_size, 10) || 1);
      for (let k = 0; k < todo.length && !this.cancelled; k += bs) {
        const part = todo.slice(k, k + bs);
        const prompts = part.map((t) => {
          const p = C.buildPrompt(t.tags, opts, this.rng);
          t.prompt = p.prompt;
          return p.prompt;
        });
        const r = await this.service.caption(part.map((t) => t.image), prompts, {
          model: opts.model, quant: opts.quant, models_dir: this.models_dir, hf_token: this.hf_token, batch_size: bs, gen: opts.gen,
        });
        this.state.device = r.device;
        r.results.forEach((res, i) => {
          const t = part[i];
          const caption = res.error ? "" : C.cleanCaption(res.text, opts);
          if (!caption) return this._error(t, res.error || "the model returned an empty caption");
          if (this.preview > 0) {
            this.state.results.push({ dir: t.dir, name: t.name, tags: t.tags, prompt: t.prompt, caption });
          } else this._write(t, caption);
        });
        if (this.preview <= 0) {
          this.manifest.add(this.writtenCaps.slice(-part.length));
          this.manifest.save();
        }
        this.state.done = Math.min(todo.length, k + bs);
        this._emit();
      }
      this.state.state = this.cancelled ? "cancelled" : "done";
    } catch (e) {
      this.state.state = "error";
      this.state.error = e.message;
    } finally {
      if (this.preview <= 0) {
        this.manifest.add(this.writtenCaps);
        this.manifest.save();
        this.undo.push(`Auto-caption ${this.state.written} image(s) with JoyCaption`, this.before, { joycaption: this.writtenCaps });
      }
      this.state.finished = Date.now();
      this._emit();
    }
    return this.state;
  }
}

module.exports = { CaptionRun, Manifest, captionDeviceError, isConverted, fileHints, TRAINING_REFUSAL };
