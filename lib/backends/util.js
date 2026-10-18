const fs = require("fs");
const path = require("path");
const TOML = require("@iarna/toml");
const YAML = require("js-yaml");

function num(v, fallback) {
  const n = parseFloat(v);
  return Number.isFinite(n) ? n : fallback;
}

function int(v, fallback) {
  const n = parseInt(v, 10);
  return Number.isFinite(n) ? n : fallback;
}

// TOML cannot store undefined, and YAML would print it as null.
function dropUndefined(obj) {
  for (const k of Object.keys(obj)) if (obj[k] === undefined) delete obj[k];
  return obj;
}

// The Advanced tab takes raw TOML (Anima) or YAML (ai-toolkit) that is deep
// merged over the generated config.
function parseOverrides(text, format) {
  if (!text || !text.trim()) return {};
  try {
    const parsed = format === "toml" ? TOML.parse(text) : YAML.load(text);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed;
    throw new Error("must be a mapping");
  } catch (e) {
    throw new Error(`Advanced overrides are not valid ${format.toUpperCase()}: ${e.message}`);
  }
}

// Newest "*-state" folder sd-scripts wrote into the output dir.
function findLastStateDir(outputDir) {
  if (!fs.existsSync(outputDir)) return null;
  const states = fs
    .readdirSync(outputDir, { withFileTypes: true })
    .filter((d) => d.isDirectory() && /-state$/.test(d.name))
    .map((d) => {
      const full = path.join(outputDir, d.name);
      return { full, mtime: fs.statSync(full).mtimeMs };
    })
    .sort((a, b) => b.mtime - a.mtime);
  return states.length ? states[0].full : null;
}

module.exports = { num, int, dropUndefined, parseOverrides, findLastStateDir };
