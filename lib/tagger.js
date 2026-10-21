// Tagger back end: the long-running Python tagger process (venno.tagger),
// cached thumbnails, folder auto-tag runs and an undo history for every
// change made to caption files in bulk.
const crypto = require("crypto");
const EventEmitter = require("events");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");

const T = require("../public/js/tags");
const { IMAGE_EXTS, imageInFolder } = require("./datasets");

function captionPath(image, captionExt) {
  const ext = captionExt && captionExt.startsWith(".") ? captionExt : "." + (captionExt || "txt");
  return image.slice(0, -path.extname(image).length) + ext;
}

function readCaption(file) {
  try {
    return fs.readFileSync(file, "utf8");
  } catch (e) {
    return null;
  }
}

// Writing an empty caption removes the file, so the image shows as uncaptioned.
function writeCaptionFile(file, text) {
  if (text == null || !String(text).trim()) {
    if (fs.existsSync(file)) fs.unlinkSync(file);
    return;
  }
  fs.writeFileSync(file, String(text).trim() + "\n", "utf8");
}

// Every image in each folder with its caption text (null = no caption file).
function listFolder(dir, captionExt) {
  if (!dir || !fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) return null;
  const names = fs.readdirSync(dir).filter((n) => IMAGE_EXTS.has(path.extname(n).toLowerCase()));
  names.sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  return names.map((name) => {
    const full = path.join(dir, name);
    const cap = readCaption(captionPath(full, captionExt));
    return { dir, name, caption: cap == null ? null : cap.trim() };
  });
}

class TaggerService extends EventEmitter {
  constructor({ python, cwd, env }) {
    super();
    this.python = python; // () => python executable
    this.cwd = cwd;
    this.env = env; // () => extra env
    this.proc = null;
    this.pending = new Map();
    this.nextId = 1;
    this.status = "";
    this.loaded = null;
  }

  get running() {
    return !!this.proc;
  }

  _start() {
    const proc = spawn(this.python(), ["-m", "venno.tagger"], {
      cwd: this.cwd,
      env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8", ...this.env() },
      windowsHide: true,
    });
    this.proc = proc;
    let buf = "";
    proc.stdout.on("data", (d) => {
      buf += d.toString();
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl);
        buf = buf.slice(nl + 1);
        let msg;
        try {
          msg = JSON.parse(line);
        } catch (e) {
          continue;
        }
        if (msg.event === "status") {
          this.status = msg.text;
          this.emit("status", msg.text);
        } else if (msg.id != null && this.pending.has(msg.id)) {
          const { resolve, reject } = this.pending.get(msg.id);
          this.pending.delete(msg.id);
          if (msg.ok) resolve(msg);
          else reject(new Error(msg.error || "tagger error"));
        }
      }
    });
    let errTail = "";
    proc.stderr.on("data", (d) => {
      errTail = (errTail + d.toString()).slice(-4000);
    });
    const fail = (why) => {
      if (this.proc !== proc) return;
      this.proc = null;
      this.loaded = null;
      this.status = "";
      const err = new Error(`Tagger process ${why}${errTail.trim() ? ":\n" + errTail.trim().split("\n").slice(-6).join("\n") : ""}`);
      for (const { reject } of this.pending.values()) reject(err);
      this.pending.clear();
      this.emit("exit");
    };
    proc.on("error", (e) => fail(`could not start (${e.message})`));
    proc.on("close", (code) => fail(`exited (code ${code})`));
  }

  request(cmd, body = {}) {
    if (!this.proc) this._start();
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.proc.stdin.write(JSON.stringify({ id, cmd, ...body }) + "\n");
    });
  }

  async tag(paths, { model, device, models_dir, hf_token, batch_size, floor = 0.1, options }) {
    const r = await this.request("tag", { model, device, paths, models_dir, hf_token, batch_size, floor, options });
    this.loaded = { model, device: r.device };
    return r;
  }

  async unload() {
    if (!this.proc) return;
    await this.request("unload");
    this.loaded = null;
    this.status = "";
  }

  stop() {
    if (this.proc) this.proc.kill();
  }
}

// Thumbnails are cached by path, size and mtime; requests arriving together
// are made in one batch by the tagger process.
class Thumbnails {
  constructor(service, dir) {
    this.service = service;
    this.dir = dir; // () => cache folder
    this.queue = [];
    this.timer = null;
  }

  pathFor(image, size) {
    const st = fs.statSync(image);
    const key = crypto.createHash("sha1").update(`${image}|${st.size}|${st.mtimeMs}|${size}`).digest("hex");
    return path.join(this.dir(), key.slice(0, 2), key + ".webp");
  }

  get(image, size) {
    const dst = this.pathFor(image, size);
    if (fs.existsSync(dst)) return Promise.resolve(dst);
    return new Promise((resolve, reject) => {
      this.queue.push({ src: image, dst, size, resolve, reject });
      if (!this.timer) this.timer = setTimeout(() => this._flush(), 25);
    });
  }

  async _flush() {
    this.timer = null;
    const batch = this.queue.splice(0, 64);
    if (this.queue.length) this.timer = setTimeout(() => this._flush(), 0);
    try {
      await this.service.request("thumbs", { items: batch.map(({ src, dst, size }) => ({ src, dst, size })) });
      for (const b of batch) (fs.existsSync(b.dst) ? b.resolve(b.dst) : b.reject(new Error("no thumbnail")));
    } catch (e) {
      for (const b of batch) b.reject(e);
    }
  }
}

// Undo history: each entry holds every touched caption file's previous text.
class UndoStore {
  constructor(file, max = 20) {
    this.file = file; // () => path
    this.max = max;
  }

  _read() {
    try {
      return JSON.parse(fs.readFileSync(this.file(), "utf8"));
    } catch (e) {
      return [];
    }
  }

  _write(entries) {
    fs.mkdirSync(path.dirname(this.file()), { recursive: true });
    fs.writeFileSync(this.file(), JSON.stringify(entries));
  }

  list() {
    return this._read().map((e) => ({ label: e.label, time: e.time, files: Object.keys(e.files).length }));
  }

  push(label, files) {
    if (!Object.keys(files).length) return;
    const entries = this._read();
    entries.push({ label, time: Date.now(), files });
    this._write(entries.slice(-this.max));
  }

  undo() {
    const entries = this._read();
    const last = entries.pop();
    if (!last) return null;
    for (const [file, text] of Object.entries(last.files)) writeCaptionFile(file, text);
    this._write(entries);
    return { label: last.label, files: Object.keys(last.files).length };
  }
}

// Bulk tag edit over the given images; returns how many captions changed.
function bulkEdit(items, captionExt, op, undo, label) {
  const before = {};
  let changed = 0;
  for (const it of items) {
    const file = captionPath(imageInFolder(it.dir, it.name), captionExt);
    const old = readCaption(file);
    const next = T.applyOp(old == null ? "" : old, op);
    if (next == null) continue;
    before[file] = old;
    writeCaptionFile(file, next);
    changed++;
  }
  undo.push(label, before);
  return changed;
}

// Auto-tag many images in chunks, writing captions as each chunk finishes.
class TagRun extends EventEmitter {
  constructor(service, undo, { items, captionExt, model, device, models_dir, hf_token, batch_size, options, mode }) {
    super();
    Object.assign(this, { service, undo, items, captionExt, model, device, models_dir, hf_token, batch_size, options, mode });
    this.state = { state: "running", done: 0, total: items.length, written: 0, skipped: 0, errors: [], model, device, mode, started: Date.now() };
    this.cancelled = false;
    this.before = {};
  }

  cancel() {
    this.cancelled = true;
  }

  async run() {
    const chunk = Math.max(4, (this.batch_size || 4) * 2);
    try {
      for (let k = 0; k < this.items.length && !this.cancelled; k += chunk) {
        const part = this.items.slice(k, k + chunk).map((it) => {
          const image = imageInFolder(it.dir, it.name);
          return { image, cap: captionPath(image, this.captionExt) };
        });
        // a caption written since the run started is left alone in skip mode
        const todo = this.mode === "skip" ? part.filter((p) => !fs.existsSync(p.cap)) : part;
        this.state.skipped += part.length - todo.length;
        if (todo.length) {
          const r = await this.service.tag(todo.map((p) => p.image), {
            model: this.model, device: this.device, models_dir: this.models_dir, hf_token: this.hf_token, batch_size: this.batch_size,
            options: this.options.model_options,
          });
          this.state.device = r.device;
          r.results.forEach((res, i) => {
            const { cap } = todo[i];
            if (res.error) return this.state.errors.push(`${path.basename(todo[i].image)}: ${res.error}`);
            const old = readCaption(cap);
            const next = T.mergeCaption(old, T.selectPredictions(res.preds, this.options), this.mode);
            if (next == null) return this.state.skipped++;
            if (!(cap in this.before)) this.before[cap] = old;
            writeCaptionFile(cap, next);
            this.state.written++;
          });
        }
        this.state.done = Math.min(this.items.length, k + chunk);
        this.emit("progress", { ...this.state });
      }
      this.state.state = this.cancelled ? "cancelled" : "done";
    } catch (e) {
      this.state.state = "error";
      this.state.error = e.message;
    } finally {
      this.undo.push(`Auto-tag ${this.state.written} image(s) (${this.mode})`, this.before);
      this.state.finished = Date.now();
      this.emit("progress", { ...this.state });
    }
    return this.state;
  }
}

module.exports = { TaggerService, Thumbnails, UndoStore, TagRun, bulkEdit, listFolder, captionPath, readCaption, writeCaptionFile };
