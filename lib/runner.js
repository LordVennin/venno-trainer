// Training process management: spawn, stream logs, parse progress, stop.
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const { EventEmitter } = require("events");

const isWindows = process.platform === "win32";
const MAX_LOG_CHUNKS = 4000;
const MAX_POINTS = 5000;

// tqdm lines from both trainers:
//   sd-scripts:  steps:  12%|█  | 120/1000 [01:23<10:00, 1.2it/s, avr_loss=0.0912]
//   ai-toolkit:  my_lora:  10%|█ | 300/3000 [05:00<45:00, 1.00s/it, lr: 1.0e-04 loss: 3.142e-01]
const PROGRESS_RE = /(\d+)\/(\d+)\s*\[([^\]<]*)<([^,\]]*)[^\]]*?(?:avr_loss=|loss[:=]\s*)([-+0-9.eE]+)/;

function parseProgress(line) {
  const m = PROGRESS_RE.exec(line);
  if (!m) return null;
  const loss = parseFloat(m[5]);
  if (!Number.isFinite(loss)) return null;
  const p = { step: +m[1], total: +m[2], elapsed: m[3].trim(), eta: m[4].trim(), loss };
  const v = /vram=([\d.]+)G/.exec(line);
  if (v) p.vram = parseFloat(v[1]);
  return p;
}

function resolvePython(pythonSetting, backendRoot) {
  const candidates = [];
  const base = pythonSetting && pythonSetting.trim() ? pythonSetting.trim() : path.join(backendRoot || "", "venv");
  if (fs.existsSync(base) && fs.statSync(base).isFile()) return base;
  candidates.push(
    isWindows ? path.join(base, "Scripts", "python.exe") : path.join(base, "bin", "python"),
    path.join(base, ".venv", isWindows ? "Scripts/python.exe" : "bin/python"),
  );
  const found = candidates.find((c) => fs.existsSync(c));
  if (!found) throw new Error(`Python not found in "${base}". Set the venv/python path in Global Settings.`);
  return found;
}

function shellQuote(a) {
  if (isWindows) return /[\s"]/.test(a) ? `"${a.replace(/"/g, '\\"')}"` : a;
  return /^[\w@%+=:,./-]+$/.test(a) ? a : `'${a.replace(/'/g, "'\\''")}'`;
}

// A script that reproduces the launch outside the UI.
function launchScript(python, args, cwd, env) {
  if (isWindows) {
    const sets = Object.entries(env).map(([k, v]) => `set "${k}=${v}"`).join("\r\n");
    return `@echo off\r\ncd /d ${shellQuote(cwd)}\r\n${sets}\r\n${[python, ...args].map(shellQuote).join(" ")}\r\n`;
  }
  const exports = Object.entries(env).map(([k, v]) => `export ${k}=${shellQuote(v)}`).join("\n");
  return `#!/usr/bin/env bash\ncd ${shellQuote(cwd)}\n${exports}\n${[python, ...args].map(shellQuote).join(" ")}\n`;
}

class Runner extends EventEmitter {
  constructor() {
    super();
    this.runs = new Map(); // job -> run
  }

  isRunning(job) {
    return this.runs.has(job);
  }

  get(job) {
    return this.runs.get(job);
  }

  start(job, { python, args, cwd, env, logsDir, progressPath }) {
    if (this.runs.has(job)) throw new Error("Job already running");
    fs.mkdirSync(logsDir, { recursive: true });
    const logFile = path.join(logsDir, `train_${new Date().toISOString().replace(/[:.]/g, "-")}.log`);
    const logStream = fs.createWriteStream(logFile, { flags: "a" });

    const proc = spawn(python, args, {
      cwd,
      env: {
        ...process.env,
        PYTHONIOENCODING: "utf-8",
        PYTHONUNBUFFERED: "1",
        TOKENIZERS_PARALLELISM: "false",
        ...env,
      },
      detached: !isWindows, // own process group so stop() reaches accelerate's children
      windowsHide: true,
    });

    const run = {
      proc,
      pid: proc.pid,
      startTime: Date.now(),
      logBuffer: [],
      points: [],
      progress: null,
      partial: "",
      progressPath,
    };
    this.runs.set(job, run);

    const onData = (data) => {
      const text = data.toString();
      run.logBuffer.push(text);
      if (run.logBuffer.length > MAX_LOG_CHUNKS) run.logBuffer.shift();
      logStream.write(text);
      this.emit("log", job, text);
      // tqdm redraws with \r, so split on both
      const lines = (run.partial + text).split(/[\r\n]/);
      run.partial = lines.pop();
      for (const line of lines) this._parse(job, run, line);
      if (run.partial.length > 4096) run.partial = "";
    };
    proc.stdout.on("data", onData);
    proc.stderr.on("data", onData);
    proc.stdout.on("error", () => {});
    proc.stderr.on("error", () => {});
    logStream.on("error", () => {});

    const finish = (msg) => {
      if (!this.runs.has(job)) return;
      logStream.end(msg);
      this.emit("log", job, msg);
      this._saveProgress(run);
      this.runs.delete(job);
      this.emit("status", job, "idle");
    };
    proc.on("close", (code, signal) =>
      finish(`\n--- ${code === 0 ? "Finished" : "Stopped"} (exit code: ${code ?? signal}) ---\n`),
    );
    proc.on("error", (err) => finish(`\nERROR: ${err.message}\n`));

    this.emit("status", job, "running");
    return run;
  }

  _parse(job, run, line) {
    const p = parseProgress(line);
    if (!p) return;
    const last = run.points[run.points.length - 1];
    if (last && last[0] === p.step) return; // tqdm repeats a step on redraw
    run.progress = p;
    run.points.push([p.step, p.loss]);
    if (run.points.length > MAX_POINTS) {
      // keep the curve's shape, halve the resolution
      run.points = run.points.filter((_, i) => i % 2 === 0);
    }
    this.emit("progress", job, p);
    if (run.points.length % 25 === 0) this._saveProgress(run);
  }

  _saveProgress(run) {
    if (!run.progressPath) return;
    try {
      fs.writeFileSync(run.progressPath, JSON.stringify({ progress: run.progress, points: run.points }));
    } catch (e) {
      /* the job folder may have been deleted */
    }
  }

  // SIGINT asks the trainer to finish its step and save; it is killed only if
  // it has not exited after graceMs.
  stop(job, graceMs = 180000) {
    const run = this.runs.get(job);
    if (!run) return false;
    const pid = run.pid;
    if (isWindows) {
      spawn("taskkill", ["/PID", String(pid), "/T", "/F"], { windowsHide: true });
      return true;
    }
    const signalGroup = (sig) => {
      try {
        process.kill(-pid, sig);
      } catch (e) {
        try {
          process.kill(pid, sig);
        } catch (e2) {
          /* already gone */
        }
      }
    };
    signalGroup("SIGINT");
    setTimeout(() => this.runs.has(job) && signalGroup("SIGKILL"), graceMs).unref();
    return true;
  }

  stopAll() {
    for (const job of this.runs.keys()) this.stop(job, 2000);
  }
}

module.exports = { Runner, parseProgress, resolvePython, launchScript };
