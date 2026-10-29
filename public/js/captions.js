/* Natural-language captions (JoyCaption) shared by the browser (Tagger
   view) and the server (caption runs): tag hints from a caption file or a
   tagger, the prompt template, cleaning the model's answer into a one-line
   caption, and the file layout (the caption alone, or tags + caption). Needs tags.js. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory(require("./tags"));
  else root.VennoCaptions = factory(root.VennoTags);
})(typeof self !== "undefined" ? self : this, function (T) {
  "use strict";

  const BACKUP_EXT = ".tags";

  // Watermarks are named (so "watermark" can go in a negative prompt) but not
  // copied: a URL or handle in the caption would teach the LoRA to write it.
  const WATERMARK = "If there is a watermark, signature or logo, say so and where it is, but do not write out its text, names or web addresses.";

  const DEFAULT_TEMPLATE =
    "Write a {length} descriptive caption for this image in a formal tone. These tags were written by a person and are accurate: {tags}. " +
    "Use them to correctly identify species, characters, and details, but only describe what is visible, and write natural sentences, not a list. " +
    WATERMARK;
  const DEFAULT_FALLBACK = "Write a {length} descriptive caption for this image in a formal tone. Only describe what is visible, and write natural sentences, not a list. " + WATERMARK;
  const DEFAULT_STRIP = ["This image shows", "This image depicts", "The image shows", "The image depicts", "This is an image of", "This is a picture of", "In this image,", "In the image,"];

  // {length} words for each setting; "mixed" picks one per image
  const LENGTHS = { short: "short", medium: "medium-length", long: "long" };

  // Tags from a caption file: comma/line separated, or e621-style space
  // separated when there is no comma. Only the tag line(s) of a "tags +
  // caption" file count; prose (an already-converted caption) gives no hints.
  const isProse = T.isProse;

  function parseTagText(text) {
    const s = T.splitCaption(text).tags;
    if (!s || isProse(s)) return [];
    return /[,\n]/.test(s) ? T.splitTags(s) : s.split(/\s+/);
  }

  // Clean hint tags: unescaped, underscores -> spaces (unless a kaomoji),
  // never-add list removed, duplicates removed, capped.
  function cleanHints(tags, opts = {}) {
    const never = new Set(T.splitTags(opts.exclude).map(T.tagKey));
    const out = T.dedupe(
      tags
        .map((t) => String(t).trim().replace(/\\([()[\]{}])/g, "$1"))
        .filter((t) => t && !never.has(T.tagKey(t)))
        .map((t) => T.formatTag(t, { underscores: opts.underscores !== false })),
    );
    const max = parseInt(opts.max_tags, 10);
    return max > 0 ? out.slice(0, max) : out;
  }

  // Hints from a tagger's predictions: its thresholds/calibration and the
  // categories chosen for hints (e.g. no artist or meta tags), no trigger prefix.
  function hintsFromPredictions(preds, taggerOptions, opts = {}) {
    const include = { ...(taggerOptions.include || {}), ...(opts.include || {}) };
    const tags = T.selectPredictions(preds, { ...taggerOptions, include, prefix: "", max_tags: 0, escape: false, underscores: false });
    return cleanHints(tags, opts);
  }

  function pickLength(length, rng = Math.random) {
    if (length === "mixed") {
      const keys = Object.keys(LENGTHS);
      return LENGTHS[keys[Math.floor(rng() * keys.length) % keys.length]];
    }
    return LENGTHS[length] || LENGTHS.long;
  }

  // The prompt for one image: the template with its tags, or the fallback
  // when it has none. Returns {prompt, length}.
  function buildPrompt(tags, opts = {}, rng = Math.random) {
    const length = pickLength(opts.length || "long", rng);
    const useTags = tags && tags.length;
    const tpl = String((useTags ? opts.template : opts.fallback_template) || (useTags ? DEFAULT_TEMPLATE : DEFAULT_FALLBACK));
    const prompt = tpl
      .replace(/\{length\}/g, length)
      .replace(/\{tags\}/g, useTags ? tags.join(", ") : "")
      .replace(/\s+/g, " ")
      .trim();
    return { prompt, length };
  }

  function stripList(opts) {
    if (Array.isArray(opts.strip)) return opts.strip;
    if (typeof opts.strip === "string") return opts.strip.split("\n").map((s) => s.trim()).filter(Boolean);
    return DEFAULT_STRIP;
  }

  // The model's answer as a caption: one line, markdown links and emphasis
  // reduced to their text, filler openers removed ("This image shows a fox"
  // -> "A fox"), whitespace trimmed.
  function cleanCaption(text, opts = {}) {
    let s = String(text || "")
      .replace(/\[([^\]]*)\]\([^)\s]*\)/g, "$1")
      .replace(/<(https?:\/\/[^>\s]+)>/g, "$1")
      .replace(/(\*\*|__)(?=\S)(.+?)(?<=\S)\1/g, "$2")
      .replace(/\s+/g, " ")
      .trim();
    const openers = stripList(opts).map((o) => o.trim()).filter(Boolean);
    for (let changed = true; changed; ) {
      changed = false;
      for (const o of openers) {
        if (s.toLowerCase().startsWith(o.toLowerCase())) {
          s = s.slice(o.length).replace(/^[\s,:;.-]+/, "");
          changed = true;
        }
      }
    }
    if (s) s = s[0].toUpperCase() + s.slice(1);
    return s;
  }

  // The caption file's text. layout "caption": the caption alone.
  // "tags_caption": the original tags as written on line 1, the caption on
  // line 2. `source` is where the tags come from (the .tags backup when a
  // run kept one, else the current file); a file without tags gets the hint
  // tags (e.g. a tagger's) instead, or the caption alone when there are none.
  function composeCaption(caption, layout, source, hints = []) {
    if (layout !== "tags_caption") return caption;
    const raw = T.splitCaption(source).tags;
    let tags = raw && !isProse(raw) ? (raw.includes("\n") ? T.joinTags(T.splitTags(raw)) : raw) : "";
    if (!tags && hints && hints.length) tags = T.joinTags(hints);
    return T.joinCaption(tags, caption);
  }

  function backupPath(image) {
    const dot = image.lastIndexOf(".");
    const slash = Math.max(image.lastIndexOf("/"), image.lastIndexOf("\\"));
    return (dot > slash ? image.slice(0, dot) : image) + BACKUP_EXT;
  }

  return { BACKUP_EXT, DEFAULT_TEMPLATE, DEFAULT_FALLBACK, DEFAULT_STRIP, LENGTHS, isProse, parseTagText, cleanHints, hintsFromPredictions, pickLength, buildPrompt, cleanCaption, composeCaption, backupPath };
});
