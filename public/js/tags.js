/* Caption tag handling shared by the browser (Tagger view) and the server
   (bulk edits, folder auto-tagging): splitting and joining comma-separated
   tags, matching tags written differently ("long_hair" = "long hair" =
   "Long Hair", "\(" = "("), turning model predictions into tags, and bulk
   rename / remove / add. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.VennoTags = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // kept as written: replacing their underscores would break them
  const KAOMOJI = new Set(["0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"]);

  // which threshold a category uses; everything else uses "general"
  const NAMED = new Set(["character", "copyright", "artist"]);

  const CATEGORIES = {
    wd: ["general", "character", "rating"],
    jtp3: ["general", "species", "character", "copyright", "artist", "meta", "lore", "rating"],
    hydra35: ["general", "color", "species", "character", "copyright", "artist", "contributor", "meta", "lore", "rating"],
  };

  function splitTags(text) {
    return String(text || "")
      .split(/[,\n\r]+/)
      .map((t) => t.trim())
      .filter(Boolean);
  }

  function joinTags(tags) {
    return tags.join(", ");
  }

  function tagKey(tag) {
    return String(tag)
      .trim()
      .toLowerCase()
      .replace(/\\([()[\]{}])/g, "$1")
      .replace(/_/g, " ")
      .replace(/\s+/g, " ");
  }

  function formatTag(name, opts = {}) {
    let t = String(name);
    if (opts.underscores !== false && !KAOMOJI.has(t)) t = t.replace(/_/g, " ");
    if (opts.escape) t = t.replace(/([()[\]{}])/g, "\\$1");
    return t;
  }

  function dedupe(tags) {
    const seen = new Set();
    return tags.filter((t) => {
      const k = tagKey(t);
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
  }

  // Model predictions [[name, prob, category, picked?], ...] -> formatted tag
  // list. A model that picks its own tags (Hydra 3.5's calibration) sends
  // picked = true/false, which replaces the thresholds.
  // opts: {general, character} thresholds, include {category: bool},
  // exclude (comma list), prefix (comma list, always first), max_tags,
  // order ("confidence" | "alpha"), underscores, escape.
  function selectPredictions(preds, opts = {}) {
    const include = opts.include || {};
    const exclude = new Set(splitTags(opts.exclude).map(tagKey));
    const gen = Number(opts.general ?? 0.35);
    const named = Number(opts.character ?? 0.85);
    let rating = null;
    const picked = [];
    for (const [name, prob, cat, pickedByModel] of preds || []) {
      if (include[cat] === false) continue;
      if (cat === "rating") {
        // only the most likely rating, and only when asked for
        if (include.rating && (!rating || prob > rating[1])) rating = [name, prob];
        continue;
      }
      if (pickedByModel === undefined ? prob < (NAMED.has(cat) ? named : gen) : !pickedByModel) continue;
      if (exclude.has(tagKey(name))) continue;
      picked.push({ name, prob, named: NAMED.has(cat) || cat === "species" });
    }
    if (opts.order === "alpha") picked.sort((a, b) => a.name.localeCompare(b.name));
    else picked.sort((a, b) => b.prob - a.prob);
    // names (characters, species...) read best right after the trigger
    const ordered = [...picked.filter((p) => p.named), ...picked.filter((p) => !p.named)];
    let tags = ordered.map((p) => formatTag(p.name, opts));
    const max = parseInt(opts.max_tags, 10) || 0;
    if (max > 0) tags = tags.slice(0, max);
    if (rating) tags.push(formatTag(rating[0], opts));
    return dedupe([...splitTags(opts.prefix), ...tags]);
  }

  // Combine a caption with new tags. mode: "overwrite" | "append" | "skip"
  // ("skip" keeps an existing caption untouched).
  function mergeCaption(existing, newTags, mode) {
    const old = existing == null ? null : splitTags(existing);
    if (old && old.length && mode === "skip") return null;
    if (!old || !old.length || mode === "overwrite") return joinTags(dedupe(newTags));
    return joinTags(dedupe([...old, ...newTags]));
  }

  // Bulk edit one caption. op: {type: "remove"|"rename"|"add", tag, to, position}
  // Returns the new caption text, or null when nothing changes.
  function applyOp(text, op) {
    const tags = splitTags(text);
    const key = tagKey(op.tag || "");
    if (!key) return null;
    const has = tags.some((t) => tagKey(t) === key);
    let out;
    if (op.type === "remove") {
      if (!has) return null;
      out = tags.filter((t) => tagKey(t) !== key);
    } else if (op.type === "rename") {
      const to = String(op.to || "").trim();
      if (!has || !to) return null;
      out = dedupe(tags.map((t) => (tagKey(t) === key ? to : t)));
    } else if (op.type === "add") {
      if (has) return null;
      const tag = String(op.tag).trim();
      out = op.position === "end" ? [...tags, tag] : [tag, ...tags];
    } else {
      throw new Error(`Unknown tag operation: ${op.type}`);
    }
    return joinTags(out);
  }

  // Tag counts over captions (strings; null = no caption). The most common
  // spelling of each tag is the one shown.
  function countTags(captions) {
    const counts = new Map();
    for (const cap of captions) {
      if (cap == null) continue;
      const seen = new Set();
      for (const t of splitTags(cap)) {
        const k = tagKey(t);
        let e = counts.get(k);
        if (!e) counts.set(k, (e = { key: k, count: 0, forms: new Map() }));
        e.forms.set(t, (e.forms.get(t) || 0) + 1);
        if (seen.has(k)) continue;
        seen.add(k);
        e.count++;
      }
    }
    return [...counts.values()]
      .map((e) => ({ key: e.key, tag: [...e.forms.entries()].sort((a, b) => b[1] - a[1])[0][0], count: e.count, variants: e.forms.size }))
      .sort((a, b) => b.count - a.count || a.tag.localeCompare(b.tag));
  }

  // Filter expression: comma-separated terms, all must match. "tag" = has the
  // tag, "-tag" = lacks it, "name:foo" = file name contains foo, "*foo*" =
  // any tag containing foo.
  function matchesFilter(item, expr) {
    const terms = splitTags(expr);
    if (!terms.length) return true;
    const keys = item.caption == null ? [] : splitTags(item.caption).map(tagKey);
    return terms.every((raw) => {
      const neg = raw.startsWith("-");
      let term = neg ? raw.slice(1).trim() : raw;
      let ok;
      if (term.toLowerCase().startsWith("name:")) ok = item.name.toLowerCase().includes(term.slice(5).trim().toLowerCase());
      else if (term.includes("*")) {
        const re = new RegExp("^" + tagKey(term).split("*").map((s) => s.replace(/[.+?^${}()|[\]\\]/g, "\\$&")).join(".*") + "$");
        ok = keys.some((k) => re.test(k));
      } else ok = keys.includes(tagKey(term));
      return neg ? !ok : ok;
    });
  }

  return { KAOMOJI, CATEGORIES, splitTags, joinTags, tagKey, formatTag, dedupe, selectPredictions, mergeCaption, applyOp, countTags, matchesFilter };
});
