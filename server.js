// Venno Trainer: web UI for the bundled LoRA trainer (trainer/venno).
// Jobs live in ./jobs/<name>/job.json; on Train the job is written to
// jobs/<name>/run.json and `python -m venno.train run.json` is launched.
const express = require("express");
const fs = require("fs");
const path = require("path");
const http = require("http");
const { spawn, execFile } = require("child_process");
const WebSocket = require("ws");

const S = require("./public/js/schema");
const native = require("./lib/native");
const { scanFolder, listSubfolders, imageInFolder, writeCaption, IMAGE_EXTS } = require("./lib/datasets");
const { Runner, resolvePython, launchScript } = require("./lib/runner");

const args = process.argv.slice(2);
const argVal = (name, dflt) => {
  const a = args.find((x) => x.startsWith(`--${name}=`));
  return a ? a.split("=").slice(1).join("=") : dflt;
};
const PORT = parseInt(argVal("port", process.env.PORT || "3100"), 10);
// The UI launches processes, so it only listens locally unless asked.
const HOST = argVal("host", process.env.HOST || "127.0.0.1");

const ROOT = __dirname;
const JOBS_DIR = process.env.VENNO_JOBS_DIR || path.join(ROOT, "jobs");
const SETTINGS_PATH = process.env.VENNO_SETTINGS || path.join(ROOT, "settings.json");
fs.mkdirSync(JOBS_DIR, { recursive: true });

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });
const runner = new Runner();

app.use(express.json({ limit: "20mb" }));
app.use(express.static(path.join(ROOT, "public")));

// ---------------------------------------------------------------- settings

function defaultSettings() {
  const models = {};
  for (const [id, m] of Object.entries(S.MODELS)) {
    models[id] = Object.fromEntries(Object.keys(m.paths).map((k) => [k, ""]));
  }
  return {
    python: "", // venv folder or python executable; blank = ./venv
    cache_dir: "", // blank = ./cache
    models_dir: "", // downloads; blank = ./models
    models,
    hf_token: "",
    theme: "github-dark",
  };
}

function getSettings() {
  const s = defaultSettings();
  if (fs.existsSync(SETTINGS_PATH)) {
    try {
      S.deepMerge(s, JSON.parse(fs.readFileSync(SETTINGS_PATH, "utf8")));
    } catch (e) {
      console.error("settings.json is invalid:", e.message);
    }
  }
  return s;
}

app.get("/api/settings", (req, res) => res.json(getSettings()));

app.put("/api/settings", (req, res) => {
  const s = S.deepMerge(getSettings(), req.body || {});
  fs.writeFileSync(SETTINGS_PATH, JSON.stringify(s, null, 2));
  res.json(s);
});

// -------------------------------------------------------------------- jobs

function sanitizeName(name) {
  const safe = String(name || "").replace(/[<>:"/\\|?*\x00-\x1f]/g, "").replace(/^\.+/, "").trim();
  return safe.slice(0, 120);
}

function jobDir(name) {
  const safe = sanitizeName(name);
  if (!safe) throw Object.assign(new Error("Invalid job name"), { status: 400 });
  return path.join(JOBS_DIR, safe);
}

function readJob(name) {
  const p = path.join(jobDir(name), "job.json");
  if (!fs.existsSync(p)) throw Object.assign(new Error("Job not found"), { status: 404 });
  return S.normalizeJob(JSON.parse(fs.readFileSync(p, "utf8")));
}

function writeJob(name, job) {
  const dir = jobDir(name);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "job.json"), JSON.stringify(S.normalizeJob(job), null, 2));
}

// Express 4 does not catch async errors; wrap every handler.
const h = (fn) => async (req, res) => {
  try {
    await fn(req, res);
  } catch (e) {
    res.status(e.status || 500).json({ error: e.message });
  }
};

app.get("/api/jobs", h((req, res) => {
  const jobs = fs
    .readdirSync(JOBS_DIR, { withFileTypes: true })
    .filter((d) => d.isDirectory() && fs.existsSync(path.join(JOBS_DIR, d.name, "job.json")))
    .map((d) => {
      const p = path.join(JOBS_DIR, d.name, "job.json");
      let model = "anima";
      let variant = "";
      try {
        const j = JSON.parse(fs.readFileSync(p, "utf8"));
        model = j.model;
        variant = j.variant;
      } catch (e) {
        /* listed anyway */
      }
      return { name: d.name, model, variant, running: runner.isRunning(d.name), mtime: fs.statSync(p).mtimeMs };
    })
    .sort((a, b) => b.mtime - a.mtime);
  res.json(jobs);
}));

app.post("/api/jobs", h((req, res) => {
  const { name, model, variant } = req.body || {};
  const safe = sanitizeName(name);
  if (!safe) return res.status(400).json({ error: "Name required" });
  if (!S.MODELS[model]) return res.status(400).json({ error: "Unknown model" });
  if (fs.existsSync(jobDir(safe))) return res.status(409).json({ error: "Job already exists" });
  const job = S.defaultJob(model);
  if (variant && S.MODELS[model].variants[variant]) job.variant = variant;
  job.training.output_name = safe.replace(/\s+/g, "_");
  writeJob(safe, job);
  res.json({ name: safe });
}));

app.get("/api/jobs/:name", h((req, res) => {
  res.json({ name: sanitizeName(req.params.name), job: readJob(req.params.name), running: runner.isRunning(sanitizeName(req.params.name)) });
}));

app.put("/api/jobs/:name", h((req, res) => {
  readJob(req.params.name); // 404 if missing
  writeJob(req.params.name, req.body.job);
  res.json({ ok: true, issues: S.validateJob(readJob(req.params.name)) });
}));

app.delete("/api/jobs/:name", h((req, res) => {
  const name = sanitizeName(req.params.name);
  if (runner.isRunning(name)) return res.status(400).json({ error: "Stop the job first" });
  fs.rmSync(jobDir(name), { recursive: true, force: true });
  res.json({ ok: true });
}));

app.post("/api/jobs/:name/clone", h((req, res) => {
  const job = readJob(req.params.name);
  const newName = sanitizeName(req.body.name);
  if (!newName) return res.status(400).json({ error: "Name required" });
  if (fs.existsSync(jobDir(newName))) return res.status(409).json({ error: "Job already exists" });
  if (req.body.model && S.MODELS[req.body.model] && req.body.model !== job.model) {
    // switch model: keep dataset folders and prompts; resolutions, batch sizes
    // and everything else are tuned per model, so they take the new defaults
    const fresh = S.defaultJob(req.body.model);
    fresh.dataset.folders = job.dataset.folders;
    fresh.dataset.caption_ext = job.dataset.caption_ext;
    fresh.sampling.prompts = job.sampling.prompts;
    fresh.sampling.negative = job.sampling.negative;
    fresh.training.output_name = newName.replace(/\s+/g, "_");
    writeJob(newName, fresh);
  } else {
    job.training.output_name = newName.replace(/\s+/g, "_");
    writeJob(newName, job);
  }
  res.json({ name: newName });
}));

function buildForJob(name, job) {
  const settings = getSettings();
  return { settings, result: native.buildRun(job, settings, jobDir(name)) };
}

function pythonFor(settings) {
  return resolvePython(settings.python, native.REPO_ROOT);
}

app.post("/api/jobs/:name/preview", h((req, res) => {
  const job = S.normalizeJob(req.body.job || readJob(req.params.name));
  const { result } = buildForJob(req.params.name, job);
  res.json({
    files: [{ name: "run.json", path: result.runPath, content: JSON.stringify(result.run, null, 2) }],
    command: { cwd: result.cwd, args: result.args, env: result.env },
    issues: S.validateJob(job),
  });
}));

app.post("/api/jobs/:name/train/start", h((req, res) => {
  const name = sanitizeName(req.params.name);
  if (runner.isRunning(name)) return res.status(400).json({ error: "Job already running" });
  const job = readJob(name);
  const settings = getSettings();
  const errors = S.validateJob(job).filter((i) => i.level === "error").map((i) => i.msg);
  errors.push(...native.checkSettings(job, settings));
  if (errors.length) return res.status(400).json({ error: errors.join("\n") });

  const python = pythonFor(settings);
  const { result } = buildForJob(name, job);
  fs.mkdirSync(result.outputDir, { recursive: true });
  fs.writeFileSync(result.runPath, JSON.stringify(result.run, null, 2));
  const scriptName = process.platform === "win32" ? "launch_command.bat" : "launch_command.sh";
  fs.writeFileSync(path.join(jobDir(name), scriptName), launchScript(python, result.args, result.cwd, result.env), { mode: 0o755 });

  const header = `--- ${S.MODELS[job.model].label} (${S.variantOf(job).label || job.variant}) ---\n$ ${[python, ...result.args].join(" ")}\n\n`;
  const run = runner.start(name, {
    python,
    args: result.args,
    cwd: result.cwd,
    env: result.env,
    logsDir: result.logsDir,
    progressPath: path.join(jobDir(name), "progress.json"),
  });
  run.logBuffer.unshift(header);
  broadcast(name, { type: "log", data: header });
  res.json({ ok: true, pid: run.pid });
}));

app.post("/api/jobs/:name/train/stop", h((req, res) => {
  res.json({ ok: runner.stop(sanitizeName(req.params.name)) });
}));

app.get("/api/jobs/:name/progress", h((req, res) => {
  const name = sanitizeName(req.params.name);
  const run = runner.get(name);
  if (run) return res.json({ running: true, progress: run.progress, points: run.points });
  const p = path.join(jobDir(name), "progress.json");
  if (fs.existsSync(p)) return res.json({ running: false, ...JSON.parse(fs.readFileSync(p, "utf8")) });
  res.json({ running: false, progress: null, points: [] });
}));

function jobOutputs(name) {
  return buildForJob(name, readJob(name)).result;
}

function walkImages(dir, out = [], depth = 0) {
  if (!fs.existsSync(dir) || depth > 3) return out;
  for (const d of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, d.name);
    if (d.isDirectory()) walkImages(full, out, depth + 1);
    else if (IMAGE_EXTS.has(path.extname(d.name).toLowerCase())) {
      out.push({ path: full, name: d.name, mtime: fs.statSync(full).mtimeMs });
    }
  }
  return out;
}

app.get("/api/jobs/:name/samples", h((req, res) => {
  const { samplesDir } = jobOutputs(req.params.name);
  const base = jobDir(req.params.name);
  const images = walkImages(samplesDir)
    .sort((a, b) => b.mtime - a.mtime)
    .map((i) => ({ ...i, rel: path.relative(base, i.path) }));
  res.json({ dir: samplesDir, images });
}));

app.get("/api/jobs/:name/checkpoints", h((req, res) => {
  const { outputDir } = jobOutputs(req.params.name);
  const list = fs.existsSync(outputDir)
    ? fs
        .readdirSync(outputDir)
        .filter((f) => f.endsWith(".safetensors"))
        .map((f) => {
          const st = fs.statSync(path.join(outputDir, f));
          return { name: f, path: path.join(outputDir, f), size: st.size, mtime: st.mtimeMs };
        })
        .sort((a, b) => b.mtime - a.mtime)
    : [];
  res.json({ dir: outputDir, checkpoints: list });
}));

// Serve a file from inside the job folder only.
app.get("/api/jobs/:name/file", h((req, res) => {
  const base = path.resolve(jobDir(req.params.name));
  const target = path.resolve(base, String(req.query.rel || ""));
  if (target !== base && !target.startsWith(base + path.sep)) return res.status(403).end();
  if (!fs.existsSync(target)) return res.status(404).end();
  res.sendFile(target);
}));

app.delete("/api/jobs/:name/samples", h((req, res) => {
  const base = path.resolve(jobDir(req.params.name));
  let removed = 0;
  for (const rel of req.body.rels || []) {
    const target = path.resolve(base, rel);
    if (target.startsWith(base + path.sep) && fs.existsSync(target)) {
      fs.unlinkSync(target);
      removed++;
    }
  }
  res.json({ removed });
}));

// ---------------------------------------------------------------- datasets

app.post("/api/datasets/scan", h((req, res) => {
  const { paths = [], caption_ext = ".txt" } = req.body || {};
  res.json(paths.map((p) => ({ path: p, ...scanFolder(String(p || "").trim(), caption_ext) })));
}));

// Every image without a caption file, for the captioning window.
app.post("/api/datasets/uncaptioned", h((req, res) => {
  const { paths = [], caption_ext = ".txt" } = req.body || {};
  res.json(paths.map((p) => {
    const dir = String(p || "").trim();
    const s = scanFolder(dir, caption_ext, Infinity);
    return { dir, exists: s.exists, images: s.images, names: s.missing };
  }));
}));

app.get("/api/datasets/image", h((req, res) => {
  res.sendFile(imageInFolder(String(req.query.dir || ""), String(req.query.name || "")));
}));

app.post("/api/datasets/caption", h((req, res) => {
  const { dir, name, caption_ext = ".txt", text = "", overwrite = false } = req.body || {};
  if (!String(text).trim()) return res.status(400).json({ error: "Caption is empty" });
  res.json({ ok: true, path: writeCaption(String(dir || ""), String(name || ""), caption_ext, text, overwrite) });
}));

app.post("/api/datasets/subfolders", h((req, res) => {
  res.json(listSubfolders(String(req.body.path || "").trim(), req.body.caption_ext || ".txt"));
}));

// ------------------------------------------------------------------ system

function openNative(target) {
  if (process.platform === "win32") spawn("explorer", [target], { detached: true });
  else if (process.platform === "darwin") spawn("open", [target], { detached: true });
  else spawn("xdg-open", [target], { detached: true, stdio: "ignore" }).on("error", () => {});
}

app.post("/api/system/open-folder", h((req, res) => {
  let target = String(req.body.path || "");
  if (req.body.job) target = path.join(jobDir(req.body.job), target);
  if (!fs.existsSync(target)) return res.status(404).json({ error: `Not found: ${target}` });
  openNative(target);
  res.json({ ok: true });
}));

function dirSize(dir) {
  let total = 0;
  let files = 0;
  if (!fs.existsSync(dir)) return { bytes: 0, files: 0 };
  const stack = [dir];
  while (stack.length) {
    const d = stack.pop();
    for (const e of fs.readdirSync(d, { withFileTypes: true })) {
      const full = path.join(d, e.name);
      if (e.isDirectory()) stack.push(full);
      else {
        total += fs.statSync(full).size;
        files++;
      }
    }
  }
  return { bytes: total, files };
}

app.get("/api/system/cache", h((req, res) => {
  const dir = native.resolveDirs(getSettings()).cache;
  const models = {};
  for (const id of Object.keys(S.MODELS)) models[id] = dirSize(path.join(dir, id));
  res.json({ dir, models });
}));

app.delete("/api/system/cache/:model", h((req, res) => {
  if (!S.MODELS[req.params.model]) return res.status(400).json({ error: "Unknown model" });
  if ([...runner.runs.keys()].length) return res.status(400).json({ error: "Stop running jobs before clearing the cache" });
  fs.rmSync(path.join(native.resolveDirs(getSettings()).cache, req.params.model), { recursive: true, force: true });
  res.json({ ok: true });
}));

// Report the Python environment the trainer will use.
app.post("/api/system/check-python", h((req, res) => {
  let python;
  try {
    python = pythonFor(S.deepMerge(getSettings(), req.body || {}));
  } catch (e) {
    return res.json({ ok: false, error: e.message });
  }
  const code = [
    "import importlib, json, sys",
    "out = {'python_version': sys.version.split()[0]}",
    "for m in ['torch', 'transformers', 'safetensors', 'bitsandbytes', 'prodigyopt', 'fla']:",
    "    try:",
    "        mod = importlib.import_module(m); out[m] = getattr(mod, '__version__', 'ok')",
    "    except Exception as e:",
    "        out[m] = None",
    "try:",
    "    import torch; out['cuda'] = torch.cuda.is_available(); out['gpu'] = torch.cuda.get_device_name(0) if out['cuda'] else None",
    "except Exception: pass",
    "print(json.dumps(out))",
  ].join("\n");
  execFile(python, ["-c", code], { timeout: 60000 }, (err, stdout, stderr) => {
    if (err) return res.json({ ok: false, python, error: (stderr || err.message).slice(-2000) });
    try {
      res.json({ ok: true, python, ...JSON.parse(stdout.trim().split("\n").pop()) });
    } catch (e) {
      res.json({ ok: false, python, error: stdout.slice(-2000) });
    }
  });
}));

app.get("/api/system/gpus", h((req, res) => {
  execFile(
    "nvidia-smi",
    ["--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
    { timeout: 5000 },
    (err, stdout) => {
      if (err) return res.json([]);
      res.json(
        stdout
          .trim()
          .split("\n")
          .filter(Boolean)
          .map((l) => {
            const [index, name, total, used, util, temp] = l.split(",").map((s) => s.trim());
            return { index: +index, name, memTotal: +total, memUsed: +used, util: +util, temp: +temp };
          }),
      );
    },
  );
}));

// --------------------------------------------------------------- websocket

const subscribers = new Map(); // job -> Set<ws>

function broadcast(job, msg) {
  const set = subscribers.get(job);
  if (!set) return;
  const payload = JSON.stringify({ job, ...msg });
  for (const ws of set) if (ws.readyState === WebSocket.OPEN) ws.send(payload);
}

runner.on("log", (job, data) => broadcast(job, { type: "log", data }));
runner.on("progress", (job, data) => broadcast(job, { type: "progress", data }));
runner.on("status", (job, data) => {
  broadcast(job, { type: "status", data });
  // the sidebar shows running state for every job
  const payload = JSON.stringify({ job, type: "jobs-changed" });
  for (const ws of wss.clients) if (ws.readyState === WebSocket.OPEN) ws.send(payload);
});

wss.on("connection", (ws) => {
  ws.isAlive = true;
  ws.on("pong", () => (ws.isAlive = true));
  ws.on("message", (raw) => {
    let msg;
    try {
      msg = JSON.parse(raw);
    } catch (e) {
      return;
    }
    if (msg.type !== "subscribe") return;
    if (ws.job) subscribers.get(ws.job)?.delete(ws);
    ws.job = sanitizeName(msg.job);
    if (!subscribers.has(ws.job)) subscribers.set(ws.job, new Set());
    subscribers.get(ws.job).add(ws);
    const run = runner.get(ws.job);
    ws.send(JSON.stringify({ job: ws.job, type: "status", data: run ? "running" : "idle" }));
    if (run) ws.send(JSON.stringify({ job: ws.job, type: "log", data: run.logBuffer.join(""), replace: true }));
  });
  ws.on("close", () => ws.job && subscribers.get(ws.job)?.delete(ws));
});

setInterval(() => {
  for (const ws of wss.clients) {
    if (!ws.isAlive) ws.terminate();
    else {
      ws.isAlive = false;
      ws.ping();
    }
  }
}, 30000).unref();

for (const sig of ["SIGINT", "SIGTERM"]) {
  process.on(sig, () => {
    runner.stopAll();
    setTimeout(() => process.exit(0), 500);
  });
}

if (require.main === module) {
  server.listen(PORT, HOST, () => {
    console.log(`Venno Trainer running at http://${HOST === "0.0.0.0" ? "localhost" : HOST}:${PORT}`);
  });
}

module.exports = { app, server, runner };
