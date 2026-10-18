// Dataset folder inspection: image/caption counts and subfolder discovery.
const fs = require("fs");
const path = require("path");
const S = require("../public/js/schema");

const IMAGE_EXTS = new Set([".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jxl", ".avif"]);

function scanFolder(dir, captionExt = ".txt") {
  const ext = captionExt.startsWith(".") ? captionExt : "." + captionExt;
  if (!dir || !fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) {
    return { exists: false, images: 0, captioned: 0, missing: [] };
  }
  const names = fs.readdirSync(dir);
  const present = new Set(names);
  let images = 0;
  let captioned = 0;
  const missing = [];
  for (const name of names) {
    const e = path.extname(name).toLowerCase();
    if (!IMAGE_EXTS.has(e)) continue;
    images++;
    if (present.has(name.slice(0, -e.length) + ext)) captioned++;
    else if (missing.length < 20) missing.push(name);
  }
  return { exists: true, images, captioned, missing };
}

function listSubfolders(dir, captionExt) {
  if (!dir || !fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) {
    throw new Error(`Not a folder: ${dir}`);
  }
  return fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((d) => d.isDirectory() && !d.name.startsWith("."))
    .map((d) => {
      const full = path.join(dir, d.name);
      return {
        name: d.name,
        path: full,
        repeats: S.repeatsFromFolderName(d.name),
        ...scanFolder(full, captionExt),
      };
    })
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
}

// Samples one epoch draws: images x repeats over enabled, non-reg folders,
// once per resolution (both trainers copy every folder per resolution).
function effectiveImages(job) {
  let total = 0;
  for (const f of job.dataset.folders) {
    if (!f.enabled || !f.path.trim() || f.is_reg) continue;
    total += scanFolder(f.path.trim(), job.dataset.caption_ext).images * Math.max(1, parseInt(f.repeats, 10) || 1);
  }
  return total * S.parseIntList(job.dataset.resolutions, [1024]).length;
}

module.exports = { IMAGE_EXTS, scanFolder, listSubfolders, effectiveImages };
