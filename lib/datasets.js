// Dataset folder inspection: image/caption counts and subfolder discovery.
const fs = require("fs");
const path = require("path");
const S = require("../public/js/schema");

const IMAGE_EXTS = new Set([".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jxl", ".avif"]);

function captionExtOf(captionExt) {
  return captionExt && captionExt.startsWith(".") ? captionExt : "." + (captionExt || "txt");
}

function scanFolder(dir, captionExt = ".txt", limit = 20) {
  const ext = captionExtOf(captionExt);
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
    else if (missing.length < limit) missing.push(name);
  }
  missing.sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  return { exists: true, images, captioned, missing };
}

// An image inside `dir`, by bare file name only (no path tricks).
function imageInFolder(dir, name) {
  if (!dir || !name || name !== path.basename(name) || !IMAGE_EXTS.has(path.extname(name).toLowerCase())) {
    throw Object.assign(new Error("Not an image in that folder"), { status: 400 });
  }
  const full = path.resolve(dir, name);
  if (!fs.existsSync(full)) throw Object.assign(new Error(`Not found: ${full}`), { status: 404 });
  return full;
}

// Write <image stem><ext>; refuses to replace an existing caption unless asked.
function writeCaption(dir, name, captionExt, text, overwrite = false) {
  const image = imageInFolder(dir, name);
  const cap = image.slice(0, -path.extname(image).length) + captionExtOf(captionExt);
  if (fs.existsSync(cap) && !overwrite) {
    throw Object.assign(new Error(`${path.basename(cap)} already exists`), { status: 409 });
  }
  fs.writeFileSync(cap, String(text).trim() + "\n", "utf8");
  return cap;
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

module.exports = { IMAGE_EXTS, scanFolder, listSubfolders, imageInFolder, writeCaption };
