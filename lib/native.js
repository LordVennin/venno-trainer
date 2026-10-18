// Translates a job into the bundled Python trainer's run.json and command.
const path = require("path");
const S = require("../public/js/schema");

const REPO_ROOT = path.join(__dirname, "..");
const TRAINER_DIR = path.join(REPO_ROOT, "trainer");

function resolveDirs(settings) {
  return {
    cache: settings.cache_dir || path.join(REPO_ROOT, "cache"),
    models: settings.models_dir || path.join(REPO_ROOT, "models"),
  };
}

function buildRun(job, settings, jobDir) {
  const dirs = resolveDirs(settings);
  const outputDir = path.join(jobDir, "output");
  const run = {
    job: S.normalizeJob(job),
    model_paths: (settings.models && settings.models[job.model]) || {},
    output_dir: outputDir,
    cache_dir: dirs.cache,
    models_dir: dirs.models,
    hf_token: settings.hf_token || "",
  };
  const runPath = path.join(jobDir, "run.json");
  const gpu = String(job.training.gpu_ids || "").split(",").map((x) => x.trim()).filter(Boolean)[0];
  // expandable segments: fewer fragmentation OOMs as bucket shapes change
  const env = { PYTHONPATH: TRAINER_DIR, PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True" };
  // the trainer uses one GPU: cuda:0 of the visible set
  if (gpu !== undefined) env.CUDA_VISIBLE_DEVICES = gpu;
  if (settings.hf_token) env.HF_TOKEN = settings.hf_token;
  return {
    run,
    runPath,
    args: ["-m", "venno.train", runPath],
    cwd: TRAINER_DIR,
    env,
    outputDir,
    samplesDir: path.join(outputDir, "samples"),
    logsDir: path.join(jobDir, "logs"),
  };
}

// Model paths that must be set before a run can start.
function checkSettings(job, settings) {
  const errs = [];
  if (job.model === "anima") {
    const p = (settings.models && settings.models.anima) || {};
    for (const [k, def] of Object.entries(S.MODELS.anima.paths)) {
      if (!p[k]) errs.push(`Set "${def.label}" in Global Settings → Anima.`);
    }
  }
  return errs;
}

module.exports = { buildRun, checkSettings, resolveDirs, REPO_ROOT, TRAINER_DIR };
