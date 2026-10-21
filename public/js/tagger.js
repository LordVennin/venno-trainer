/* Tagger view: browse dataset images, edit their captions, see tag counts,
   bulk rename/remove/add tags and auto-tag with WD v3 or JTP-3 Hydra.
   Uses app.js helpers (api, $, escapeHtml, showToast, showConfirm, ...). */
"use strict";

const T = window.VennoTags;
const TG_PAGE = 60;
const TG_TAG_PAGE = 200;

const tg = {
  dirs: [],
  ext: ".txt",
  items: [], // {dir, name, caption}
  filtered: [],
  page: 0,
  sel: -1, // index into filtered
  cfg: null, // settings.tagger
  models: [],
  categories: {},
  state: null, // server tagger state
  preds: null, // {key, preds, device, model}
  predicting: false,
  tagLimit: TG_TAG_PAGE,
  countsStale: true,
  lastRunState: null,
};

const tgKey = (it) => it.dir + "\n" + it.name;
const tgThumb = (it) => `/api/tagger/thumb?dir=${encodeURIComponent(it.dir)}&name=${encodeURIComponent(it.name)}&size=${tg.cfg ? tg.cfg.thumb_size : 256}`;
const tgDirName = (d) => d.split(/[\\/]/).filter(Boolean).pop() || d;

function taggerOpen() {
  return !$("tagger-view").classList.contains("hidden");
}

function showPrompt(title, message, value, onOk, okLabel = "OK") {
  $("prompt-title").textContent = title;
  $("prompt-message").textContent = message;
  $("prompt-input").value = value || "";
  $("btn-prompt-ok").textContent = okLabel;
  $("btn-prompt-ok").onclick = () => {
    const v = $("prompt-input").value.trim();
    closeModal("modal-prompt");
    if (v) onOk(v);
  };
  openModal("modal-prompt");
  $("prompt-input").focus();
  $("prompt-input").select();
}

// --------------------------------------------------------------- open/close

async function openTagger(dirs) {
  $("empty-state").classList.add("hidden");
  $("job-editor").classList.add("hidden");
  $("tagger-view").classList.remove("hidden");
  if (!tg.cfg) {
    try {
      settings = await api("/api/settings");
    } catch (e) {
      /* keep what we have */
    }
    tg.cfg = settings.tagger;
    bindTaggerSettings();
  }
  fillFolderPick(dirs);
  if (job) $("tg-ext").value = job.dataset.caption_ext || ".txt";
  loadTaggerModels();
  refreshTaggerState();
  refreshUndo();
  if (dirs && dirs.length) loadTaggerFolders(dirs);
  else if (!tg.items.length && job && job.dataset.folders.some((f) => f.path.trim())) {
    $("tg-folder-pick").value = "__job__";
    loadTaggerFolders(jobFolders());
  }
}

function closeTagger() {
  if (tgUnsaved()) return showConfirm("Unsaved caption", "Discard the unsaved caption edit?", () => { tg.sel = -1; closeTagger(); }, "Discard");
  $("tagger-view").classList.add("hidden");
  if (job) $("job-editor").classList.remove("hidden");
  else $("empty-state").classList.remove("hidden");
  if (job) {
    scanCache.clear(); // captions may have changed
    scanDatasets();
  }
}

function jobFolders() {
  return job ? [...new Set(job.dataset.folders.map((f) => f.path.trim()).filter(Boolean))] : [];
}

function fillFolderPick(dirs) {
  const folders = jobFolders();
  const opts = ['<option value="">Folder path…</option>'];
  if (folders.length) {
    opts.push(`<option value="__job__">All ${folders.length} folders of “${escapeHtml(currentJob)}”</option>`);
    for (const f of folders) opts.push(`<option value="${escapeHtml(f)}">${escapeHtml(tgDirName(f))}</option>`);
  }
  $("tg-folder-pick").innerHTML = opts.join("");
  $("tg-folder-pick").classList.toggle("hidden", !folders.length);
  if (dirs && dirs.length === 1) {
    $("tg-folder-pick").value = folders.includes(dirs[0]) ? dirs[0] : "";
    $("tg-path").value = dirs[0];
  }
}

function pickedDirs() {
  const pick = $("tg-folder-pick").value;
  if (pick === "__job__") return jobFolders();
  const p = $("tg-path").value.trim();
  return p ? [p] : [];
}

// -------------------------------------------------------------------- data

async function loadTaggerFolders(dirs = pickedDirs(), keepSelection = false) {
  if (!dirs.length) return showToast("Enter a folder path");
  const selKey = keepSelection && tg.filtered[tg.sel] ? tgKey(tg.filtered[tg.sel]) : null;
  tg.dirs = dirs;
  tg.ext = $("tg-ext").value.trim() || ".txt";
  try {
    const r = await api("/api/tagger/list", { method: "POST", body: { dirs, caption_ext: tg.ext } });
    tg.items = r.items;
    if (r.missing.length) showToast(`Not found: ${r.missing.join(", ")}`, 6000);
  } catch (e) {
    return showToast(e.message);
  }
  $("tg-source-label").textContent = dirs.length === 1 ? dirs[0] : `${dirs.length} folders`;
  tg.countsStale = true;
  if (!keepSelection) tg.page = 0;
  applyTaggerFilter(false);
  tg.sel = selKey ? tg.filtered.findIndex((it) => tgKey(it) === selKey) : -1;
  renderTaggerGrid();
  renderTaggerEditor();
  if (!$("tg-pane-tags").classList.contains("hidden")) renderTagCounts();
  renderRunScope();
}

function applyTaggerFilter(render = true) {
  const expr = $("tg-filter").value;
  const show = $("tg-show").value;
  tg.filtered = tg.items.filter(
    (it) => (show === "all" || (show === "uncaptioned") === (it.caption == null)) && T.matchesFilter(it, expr),
  );
  tg.countsStale = true;
  if (render) {
    tg.page = 0;
    tg.sel = -1;
    renderTaggerGrid();
    renderTaggerEditor();
    renderRunScope();
  }
}

function filterActive() {
  return $("tg-filter").value.trim() !== "" || $("tg-show").value !== "all";
}

// the images bulk actions apply to: the filtered set
function scopeItems() {
  return filterActive() ? tg.filtered : tg.items;
}

// -------------------------------------------------------------------- grid

function renderTaggerGrid() {
  const grid = $("tg-grid");
  const pages = Math.max(1, Math.ceil(tg.filtered.length / TG_PAGE));
  tg.page = Math.min(tg.page, pages - 1);
  const start = tg.page * TG_PAGE;
  $("tg-count").textContent = tg.items.length
    ? `${fmt(tg.filtered.length)} of ${fmt(tg.items.length)} images · ${fmt(tg.items.filter((i) => i.caption == null).length)} uncaptioned`
    : "";
  $("tg-page").textContent = tg.filtered.length ? `Page ${tg.page + 1} / ${pages}` : "";
  $("tg-prev-page").disabled = tg.page === 0;
  $("tg-next-page").disabled = tg.page >= pages - 1;
  $("tg-empty").classList.toggle("hidden", tg.items.length > 0);
  const multi = tg.dirs.length > 1;
  grid.innerHTML = tg.filtered
    .slice(start, start + TG_PAGE)
    .map((it, k) => {
      const i = start + k;
      const cap = it.caption == null ? '<span class="chip warn">no caption</span>' : escapeHtml(it.caption);
      return `<div class="tg-card ${i === tg.sel ? "selected" : ""} ${it.caption == null ? "uncap" : ""}" data-i="${i}" title="${escapeHtml((multi ? tgDirName(it.dir) + " / " : "") + it.name)}">
        <img loading="lazy" src="${tgThumb(it)}" alt="">
        <div class="tg-cap">${cap}</div></div>`;
    })
    .join("");
}

function selectTaggerImage(i) {
  if (i < 0 || i >= tg.filtered.length || i === tg.sel) return;
  const go = () => {
    tg.sel = i;
    tg.preds = null;
    const page = Math.floor(i / TG_PAGE);
    if (page !== tg.page) {
      tg.page = page;
      renderTaggerGrid();
    } else {
      document.querySelectorAll("#tg-grid .tg-card").forEach((c) => c.classList.toggle("selected", +c.dataset.i === i));
    }
    renderTaggerEditor();
    const card = document.querySelector(`#tg-grid .tg-card[data-i="${i}"]`);
    if (card) card.scrollIntoView({ block: "nearest" });
  };
  if (tgUnsaved()) {
    showConfirm("Unsaved caption", `Save the caption of ${tg.filtered[tg.sel].name} first?`, async () => {
      if (await saveTaggerCaption()) go();
    }, "Save");
  } else go();
}

// ------------------------------------------------------------------ editor

function tgUnsaved() {
  const it = tg.filtered[tg.sel];
  const ta = $("tg-caption");
  return !!(it && ta && ta.value.trim() !== (it.caption || ""));
}

function renderTaggerEditor() {
  const box = $("tg-editor");
  const it = tg.filtered[tg.sel];
  if (!it) {
    box.innerHTML = '<p class="hint">Select an image to edit its tags. Alt+← / Alt+→ move between images.</p>';
    return;
  }
  box.innerHTML = `
    <div class="tg-preview"><img src="${imageUrl(it)}" alt="" title="Open full size"></div>
    <div class="tg-name">
      <button class="btn-icon" id="tg-ed-prev" title="Previous (Alt+←)" ${tg.sel === 0 ? "disabled" : ""}>←</button>
      <span title="${escapeHtml(it.dir)}">${escapeHtml(tg.dirs.length > 1 ? tgDirName(it.dir) + " / " : "")}<b>${escapeHtml(it.name)}</b></span>
      <span class="hint">${tg.sel + 1} / ${tg.filtered.length}</span>
      <button class="btn-icon" id="tg-ed-next" title="Next (Alt+→)" ${tg.sel >= tg.filtered.length - 1 ? "disabled" : ""}>→</button>
    </div>
    <textarea id="tg-caption" rows="5" placeholder="Comma-separated tags">${escapeHtml(it.caption || "")}</textarea>
    <div class="inline" style="margin:6px 0;">
      <button class="btn btn-sm btn-primary" id="tg-save" title="Ctrl+S; Ctrl+Enter saves and opens the next image">Save</button>
      <button class="btn btn-sm" id="tg-revert">Revert</button>
      <span id="tg-dirty" class="chip warn hidden">unsaved</span>
      <span style="flex:1"></span>
      <button class="btn btn-sm" id="tg-predict" title="Run the tagger model on this image">✨ Auto-tag this image</button>
    </div>
    <div id="tg-chips" class="tg-chips"></div>
    <div id="tg-preds"></div>`;
  const ta = $("tg-caption");
  ta.addEventListener("input", () => {
    renderEditorChips();
    $("tg-dirty").classList.toggle("hidden", !tgUnsaved());
    if (tg.preds) renderPredictions();
  });
  ta.addEventListener("keydown", async (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      if (await saveTaggerCaption()) selectTaggerImage(tg.sel + 1);
    }
  });
  box.querySelector(".tg-preview img").onclick = () => window.open(imageUrl(it), "_blank");
  $("tg-ed-prev").onclick = () => selectTaggerImage(tg.sel - 1);
  $("tg-ed-next").onclick = () => selectTaggerImage(tg.sel + 1);
  $("tg-save").onclick = () => saveTaggerCaption();
  $("tg-revert").onclick = () => {
    ta.value = it.caption || "";
    ta.dispatchEvent(new Event("input"));
  };
  $("tg-predict").onclick = predictTaggerImage;
  renderEditorChips();
  if (tg.preds && tg.preds.key === tgKey(it)) renderPredictions();
}

function renderEditorChips() {
  const ta = $("tg-caption");
  const tags = T.splitTags(ta.value);
  $("tg-chips").innerHTML = tags
    .map((t, i) => `<span class="tg-chip">${escapeHtml(t)}<button data-i="${i}" title="Remove">×</button></span>`)
    .join("");
  $("tg-chips").querySelectorAll("button").forEach((b) => {
    b.onclick = () => {
      const list = T.splitTags(ta.value);
      list.splice(+b.dataset.i, 1);
      ta.value = T.joinTags(list);
      ta.dispatchEvent(new Event("input"));
    };
  });
}

async function saveTaggerCaption() {
  const it = tg.filtered[tg.sel];
  const ta = $("tg-caption");
  if (!it || !ta) return false;
  try {
    const r = await api("/api/tagger/caption", { method: "POST", body: { dir: it.dir, name: it.name, caption_ext: tg.ext, text: ta.value } });
    it.caption = r.caption;
    tg.countsStale = true;
    $("tg-dirty").classList.add("hidden");
    const card = document.querySelector(`#tg-grid .tg-card[data-i="${tg.sel}"]`);
    if (card) {
      card.classList.toggle("uncap", it.caption == null);
      card.querySelector(".tg-cap").innerHTML = it.caption == null ? '<span class="chip warn">no caption</span>' : escapeHtml(it.caption);
    }
    return true;
  } catch (e) {
    showToast(e.message);
    return false;
  }
}

// --------------------------------------------------------- single auto-tag

function modelKind(id = tg.cfg.model) {
  const m = tg.models.find((x) => x.id === id);
  return m ? m.kind : id.startsWith("jtp") ? "jtp3" : "wd";
}

const HYDRA_IMPL_HINTS = {
  inherit: "Tags implied by a picked tag are kept and take its probability (western dragon → dragon).",
  "enforce-inherit": "Also drops tags whose implied tags were not picked.",
  preserve: "Implied tags are kept with their own probability.",
  constrain: "Tags take the lowest probability of the tags they imply.",
  remove: "Drops tags implied by another picked tag (keeps western dragon, drops dragon).",
  off: "Raw model output; tags may contradict each other.",
};

function tagOptions() {
  const c = tg.cfg;
  const kind = modelKind();
  const th = c[kind] || {};
  const opts = {
    general: th.general,
    character: th.character,
    include: c.include,
    exclude: c.exclude,
    prefix: c.prefix,
    max_tags: c.max_tags,
    order: c.order,
    underscores: c.underscores,
    escape: c.escape,
  };
  if (kind === "hydra35") {
    // Hydra applies these before its implications, as its own tagger does
    opts.model_options = {
      confidence: th.confidence,
      implications: th.implications,
      aliases: th.aliases,
      exclude_categories: (tg.categories.hydra35 || []).filter((cat) => c.include[cat] === false || (cat === "rating" && !c.include.rating)),
      exclude_tags: T.splitTags(c.exclude).map((t) => t.replace(/ /g, "_")),
    };
  }
  return opts;
}

async function predictTaggerImage() {
  const it = tg.filtered[tg.sel];
  if (!it || tg.predicting) return;
  tg.predicting = true;
  const btn = $("tg-predict");
  btn.disabled = true;
  btn.textContent = "Tagging…";
  $("tg-preds").innerHTML = `<p class="hint">Running ${escapeHtml(modelLabel())}… the first run loads (and if needed downloads) the model.</p>`;
  try {
    const r = await api("/api/tagger/predict", {
      method: "POST",
      body: { dir: it.dir, name: it.name, model: tg.cfg.model, device: tg.cfg.device, model_options: tagOptions().model_options },
    });
    tg.preds = { key: tgKey(it), ...r };
    if (tg.filtered[tg.sel] === it) renderPredictions();
  } catch (e) {
    $("tg-preds").innerHTML = `<div class="issue error">${escapeHtml(e.message)}</div>`;
  } finally {
    tg.predicting = false;
    if ($("tg-predict")) {
      $("tg-predict").disabled = false;
      $("tg-predict").textContent = "✨ Auto-tag this image";
    }
  }
}

function modelLabel(id = tg.cfg.model) {
  const m = tg.models.find((x) => x.id === id);
  return m ? m.label : id;
}

function renderPredictions() {
  const box = $("tg-preds");
  if (!box || !tg.preds) return;
  const opts = tagOptions();
  const picked = new Set(T.selectPredictions(tg.preds.preds, { ...opts, prefix: "" }).map(T.tagKey));
  const have = new Set(T.splitTags($("tg-caption").value).map(T.tagKey));
  const floor = Math.min(0.2, opts.general ?? 0.35);
  const rows = tg.preds.preds
    .filter(([, p, cat, pickedByModel]) => p >= floor || cat === "rating" || pickedByModel)
    .map(([name, p, cat]) => {
      const tag = T.formatTag(name, opts);
      const k = T.tagKey(tag);
      const inCap = have.has(k);
      return `<label class="tg-pred ${inCap ? "have" : ""}" title="${escapeHtml(cat)}">
        <input type="checkbox" data-tag="${escapeHtml(tag)}" ${picked.has(k) && !inCap ? "checked" : ""} ${inCap ? "disabled" : ""}>
        <span class="tg-pred-name">${escapeHtml(tag)}</span>
        <span class="tg-pred-cat">${escapeHtml(cat)}</span>
        <span class="tg-pred-bar"><i style="width:${(p * 100).toFixed(0)}%"></i></span>
        <span class="tg-pred-p">${p.toFixed(2)}</span></label>`;
    });
  box.innerHTML = `
    <div class="tg-preds-head">
      <span class="hint">${escapeHtml(modelLabel(tg.preds.model))} on ${escapeHtml(tg.preds.device)} · ticked = ${modelKind(tg.preds.model) === "hydra35" ? "picked by Hydra's calibration" : "above your thresholds"} · greyed = already in the caption</span>
    </div>
    <div class="tg-pred-list">${rows.join("") || '<p class="hint">No tags above the lowest shown confidence.</p>'}</div>
    <div class="inline" style="margin-top:6px;">
      <button class="btn btn-sm btn-primary" id="tg-pred-add">Add ticked</button>
      <button class="btn btn-sm" id="tg-pred-replace" title="Replace the caption with your prefix and the ticked tags">Replace caption</button>
      <span class="hint">Then Save.</span>
    </div>`;
  const ticked = () => [...box.querySelectorAll("input[data-tag]:checked")].map((c) => c.dataset.tag);
  $("tg-pred-add").onclick = () => {
    const ta = $("tg-caption");
    ta.value = T.joinTags(T.dedupe([...T.splitTags(ta.value), ...ticked()]));
    ta.dispatchEvent(new Event("input"));
  };
  $("tg-pred-replace").onclick = () => {
    const ta = $("tg-caption");
    ta.value = T.joinTags(T.dedupe([...T.splitTags(tg.cfg.prefix), ...ticked()]));
    ta.dispatchEvent(new Event("input"));
  };
}

// -------------------------------------------------------------- tag counts

function renderTagCounts() {
  const items = scopeItems();
  const caps = items.map((i) => i.caption);
  const counts = T.countTags(caps);
  const captioned = caps.filter((c) => c != null).length;
  const totalTags = counts.reduce((s, c) => s + c.count, 0);
  $("tg-tag-summary").innerHTML = `
    <div class="stat"><div class="v">${fmt(items.length)}</div><div class="k">Images</div></div>
    <div class="stat"><div class="v">${fmt(captioned)}</div><div class="k">Captioned</div></div>
    <div class="stat"><div class="v">${fmt(counts.length)}</div><div class="k">Unique tags</div></div>
    <div class="stat"><div class="v">${captioned ? (totalTags / captioned).toFixed(1) : "–"}</div><div class="k">Tags / image</div></div>
    <div class="stat"><div class="v">${fmt(counts.filter((c) => c.count === 1).length)}</div><div class="k">Used once</div></div>`;
  $("tg-tag-scope").innerHTML = filterActive()
    ? `Counting the <b>${fmt(items.length)}</b> images shown by the Images filter; edits below apply to them only. Clear the filter to count every loaded image.`
    : `Counting all <b>${fmt(items.length)}</b> loaded images; edits below apply to all of them. Filter on the Images tab (e.g. a species tag) to count and edit a subset.`;
  const q = T.tagKey($("tg-tag-search").value);
  const sort = $("tg-tag-sort").value;
  let rows = q ? counts.filter((c) => c.key.includes(q)) : counts;
  if (sort === "rare") rows = [...rows].sort((a, b) => a.count - b.count || a.tag.localeCompare(b.tag));
  else if (sort === "name") rows = [...rows].sort((a, b) => a.key.localeCompare(b.key));
  const max = counts.length ? counts[0].count : 1;
  const shown = rows.slice(0, tg.tagLimit);
  $("tg-tag-table").innerHTML = shown.length
    ? `<div class="tg-tag-row tg-tag-headrow"><span>Tag</span><span>Images</span><span></span><span></span></div>` +
      shown
        .map(
          (c, i) => `<div class="tg-tag-row" data-i="${i}">
        <span class="tg-tag-name">${escapeHtml(c.tag)}${c.variants > 1 ? ` <span class="chip warn" title="Written ${c.variants} different ways">${c.variants} spellings</span>` : ""}</span>
        <span class="tg-tag-count">${fmt(c.count)} <span class="hint">${((c.count / Math.max(1, captioned)) * 100).toFixed(1)}%</span></span>
        <span class="tg-tag-bar"><i style="width:${(c.count / max) * 100}%"></i></span>
        <span class="tg-tag-actions">
          <button class="btn btn-xs" data-act="filter" title="Show these images">Show</button>
          ${c.variants > 1 ? '<button class="btn btn-xs" data-act="unify" title="Write every spelling the same way">Unify</button>' : ""}
          <button class="btn btn-xs" data-act="rename">Rename</button>
          <button class="btn btn-xs btn-danger" data-act="remove">Remove</button>
        </span></div>`,
        )
        .join("")
    : `<p class="hint">${items.length ? "No tags match." : "Load a folder first."}</p>`;
  $("tg-tags-more").classList.toggle("hidden", rows.length <= tg.tagLimit);
  $("tg-tags-more").textContent = `Show more (${fmt(rows.length - tg.tagLimit)} left)`;
  $("tg-tag-table").querySelectorAll("[data-act]").forEach((b) => {
    const c = shown[+b.closest(".tg-tag-row").dataset.i];
    b.onclick = () => tagAction(b.dataset.act, c);
  });
  tg.countsStale = false;
}

function tagAction(act, c) {
  const items = scopeItems();
  const n = `${fmt(c.count)} image(s)`;
  if (act === "filter") {
    $("tg-filter").value = c.tag;
    $("tg-show").value = "all";
    applyTaggerFilter();
    switchTaggerTab("images");
  } else if (act === "remove") {
    showConfirm("Remove tag", `Remove “${c.tag}” from ${n}? You can undo this.`, () => runBulk(items, { type: "remove", tag: c.tag }), "Remove");
  } else if (act === "rename") {
    showPrompt("Rename tag", `Rename “${c.tag}” in ${n}. An existing tag of the new name is merged.`, c.tag, (to) => runBulk(items, { type: "rename", tag: c.tag, to }), "Rename");
  } else if (act === "unify") {
    runBulk(items, { type: "rename", tag: c.tag, to: c.tag });
  }
}

async function runBulk(items, op) {
  try {
    const r = await api("/api/tagger/bulk", { method: "POST", body: { items: items.map(({ dir, name }) => ({ dir, name })), caption_ext: tg.ext, op } });
    showToast(`${fmt(r.changed)} caption(s) changed`);
  } catch (e) {
    showToast(e.message);
  }
  await loadTaggerFolders(tg.dirs, true);
  refreshUndo();
  if (!$("tg-pane-tags").classList.contains("hidden")) renderTagCounts();
}

async function refreshUndo() {
  try {
    const r = await api("/api/tagger/undo");
    const last = r.entries[r.entries.length - 1];
    $("tg-undo").disabled = !last;
    $("tg-undo").title = last ? `Undo: ${last.label} — ${last.files} file(s), ${new Date(last.time).toLocaleString()}` : "Nothing to undo";
  } catch (e) {
    $("tg-undo").disabled = true;
  }
}

// ---------------------------------------------------------------- auto-tag

async function loadTaggerModels() {
  try {
    const r = await api("/api/tagger/models");
    tg.models = r.models;
    tg.categories = r.categories;
  } catch (e) {
    $("tg-service").innerHTML = `<span style="color:var(--danger)">Tagger unavailable: ${escapeHtml(e.message)}</span>`;
    return;
  }
  $("tg-model").innerHTML = tg.models
    .map((m) => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.label)} · ${m.gb} GB${m.downloaded ? " ✓" : ""}</option>`)
    .join("");
  $("tg-model").value = tg.cfg.model;
  renderModelFields();
}

function renderModelFields() {
  const kind = modelKind();
  const th = tg.cfg[kind] || (tg.cfg[kind] = kind === "hydra35" ? { confidence: 0.5, implications: "inherit", aliases: false } : { general: 0.35, character: 0.85 });
  $("tg-hydra-fields").classList.toggle("hidden", kind !== "hydra35");
  $("tg-threshold-fields").classList.toggle("hidden", kind === "hydra35");
  if (kind === "hydra35") {
    $("tg-hy-conf").value = th.confidence;
    $("tg-hy-conf-val").textContent = Number(th.confidence).toFixed(2);
    $("tg-hy-impl").value = th.implications;
    $("tg-hy-impl-hint").textContent = HYDRA_IMPL_HINTS[th.implications] || "";
    $("tg-hy-aliases").checked = !!th.aliases;
  } else {
    $("tg-th-general").value = th.general;
    $("tg-th-character").value = th.character;
  }
  $("tg-th-general-hint").textContent = kind === "jtp3" ? "e621 tags. JTP-3's demo default is 0.65." : "Danbooru tags. 0.35 is the usual WD default.";
  const m = tg.models.find((x) => x.id === tg.cfg.model);
  $("tg-model-hint").textContent = m ? (m.downloaded ? "Downloaded." : `Downloads ~${m.gb} GB into the models folder on first use.`) : "";
  const cats = tg.categories[kind] || [];
  $("tg-cats").innerHTML = cats
    .map((c) => `<label class="hint inline"><input type="checkbox" data-cat="${c}" ${tg.cfg.include[c] !== false && (c !== "rating" || tg.cfg.include.rating) ? "checked" : ""}> ${c}</label>`)
    .join("");
  $("tg-cats").querySelectorAll("[data-cat]").forEach((el) => {
    el.onchange = () => {
      tg.cfg.include[el.dataset.cat] = el.checked;
      saveTaggerSettings();
      if (kind === "hydra35") repredictSoon();
    };
  });
}

function bindTaggerSettings() {
  document.querySelectorAll("#tg-pane-auto [data-t]").forEach((el) => {
    const v = tg.cfg[el.dataset.t];
    if (el.type === "checkbox") el.checked = !!v;
    else el.value = v ?? "";
    el.addEventListener(el.type === "checkbox" || el.tagName === "SELECT" ? "change" : "input", () => {
      const k = el.dataset.t;
      tg.cfg[k] = el.type === "checkbox" ? el.checked : el.type === "number" ? Number(el.value) : el.value;
      if (k === "model") renderModelFields();
      saveTaggerSettings();
    });
  });
  for (const k of ["general", "character"]) {
    $("tg-th-" + k).addEventListener("input", (e) => {
      tg.cfg[modelKind()][k] = Number(e.target.value);
      saveTaggerSettings();
    });
  }
  // Hydra picks tags in the model process, so a change re-runs the open image
  const hydraSet = (k, v) => {
    tg.cfg.hydra35[k] = v;
    renderModelFields();
    saveTaggerSettings();
    repredictSoon();
  };
  $("tg-hy-conf").addEventListener("input", (e) => {
    $("tg-hy-conf-val").textContent = Number(e.target.value).toFixed(2);
  });
  $("tg-hy-conf").addEventListener("change", (e) => hydraSet("confidence", Number(e.target.value)));
  $("tg-hy-impl").addEventListener("change", (e) => hydraSet("implications", e.target.value));
  $("tg-hy-aliases").addEventListener("change", (e) => hydraSet("aliases", e.target.checked));
}

const repredictSoon = debounce(() => {
  const it = tg.filtered[tg.sel];
  if (tg.preds && it && tg.preds.key === tgKey(it) && modelKind(tg.preds.model) === "hydra35") predictTaggerImage();
}, 500);

const saveTaggerSettings = debounce(async () => {
  try {
    settings = await api("/api/settings", { method: "PUT", body: { tagger: tg.cfg } });
  } catch (e) {
    showToast("Could not save tagger settings: " + e.message);
  }
  if (tg.preds) renderPredictions();
  renderRunScope();
}, 400);

function renderRunScope() {
  const sel = $("tg-run-scope");
  const prev = sel.value;
  const opts = [`<option value="all">All loaded images (${fmt(tg.items.length)})</option>`];
  if (filterActive()) opts.push(`<option value="filtered">Images shown by the filter (${fmt(tg.filtered.length)})</option>`);
  sel.innerHTML = opts.join("");
  if ([...sel.options].some((o) => o.value === prev)) sel.value = prev;
  const items = sel.value === "filtered" ? tg.filtered : tg.items;
  const mode = tg.cfg ? tg.cfg.mode : "skip";
  const n = mode === "skip" ? items.filter((i) => i.caption == null).length : items.length;
  $("tg-run").textContent = `▶ Auto-tag ${fmt(n)} image(s)`;
  $("tg-run").disabled = !n || (tg.state && tg.state.run && tg.state.run.state === "running");
}

function startTaggerRun() {
  const items = $("tg-run-scope").value === "filtered" ? tg.filtered : tg.items;
  const mode = tg.cfg.mode;
  const todo = mode === "skip" ? items.filter((i) => i.caption == null) : items;
  if (!todo.length) return showToast("No images to tag");
  const go = async () => {
    try {
      const r = await api("/api/tagger/run", {
        method: "POST",
        body: { items: todo.map(({ dir, name }) => ({ dir, name })), caption_ext: tg.ext, model: tg.cfg.model, device: tg.cfg.device, batch_size: tg.cfg.batch_size, options: tagOptions(), mode },
      });
      onTaggerMessage(r.state);
    } catch (e) {
      showToast(e.message, 6000);
    }
  };
  if (mode === "overwrite") showConfirm("Replace captions", `Replace the captions of ${fmt(todo.length)} image(s) with the tagger's tags? You can undo the run afterwards.`, go, "Replace");
  else go();
}

async function refreshTaggerState() {
  try {
    onTaggerMessage(await api("/api/tagger/state"));
  } catch (e) {
    /* server older than the tagger */
  }
}

function onTaggerMessage(data) {
  tg.state = data;
  if (!$("tg-service")) return;
  const svc = data.service;
  const device = tg.cfg && tg.cfg.device;
  let note = svc.loaded ? `Loaded: <b>${escapeHtml(modelLabel(svc.loaded.model))}</b> on ${escapeHtml(svc.loaded.device)}.` : "No model loaded.";
  if (svc.status && !/^Ready/.test(svc.status)) note += ` ${escapeHtml(svc.status)}`;
  if (data.training && device === "auto") note += ' <span style="color:var(--warning)">Training is running, so Auto uses the CPU (slower). Choose GPU to share the card.</span>';
  $("tg-service").innerHTML = note;
  const run = data.run;
  const box = $("tg-progress");
  box.classList.toggle("hidden", !run);
  const runningNow = run && run.state === "running";
  $("tg-cancel").classList.toggle("hidden", !runningNow);
  if (run) {
    const pct = run.total ? (run.done / run.total) * 100 : 0;
    $("tg-prog-fill").style.width = pct + "%";
    const verb = { running: "Tagging", done: "Done", cancelled: "Cancelled", error: "Failed" }[run.state];
    $("tg-prog-text").textContent = `${verb}: ${fmt(run.done)} / ${fmt(run.total)} · ${fmt(run.written)} written · ${fmt(run.skipped)} skipped${run.error_count ? ` · ${fmt(run.error_count)} errors` : ""} · ${run.device || ""}`;
    const secs = ((run.finished || Date.now()) - run.started) / 1000;
    $("tg-prog-eta").textContent = runningNow && run.done ? `ETA ${fmtDuration((secs / run.done) * (run.total - run.done))}` : fmtDuration(secs);
    $("tg-prog-errors").innerHTML = (run.error ? `<div class="issue error">${escapeHtml(run.error)}</div>` : "") + run.errors.map((e) => `<div>${escapeHtml(e)}</div>`).join("");
  }
  // a run finished: show the new captions
  const prev = tg.lastRunState;
  tg.lastRunState = run ? run.state : null;
  if (prev === "running" && run && run.state !== "running" && tg.dirs.length) {
    loadTaggerFolders(tg.dirs, true);
    refreshUndo();
    loadTaggerModels();
  }
  if (tg.cfg) renderRunScope();
}

function fmtDuration(s) {
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${m}m` : m ? `${m}m ${s % 60}s` : `${s}s`;
}

// -------------------------------------------------------------------- tabs

function switchTaggerTab(name) {
  document.querySelectorAll("#tg-tabs .tab").forEach((t) => t.classList.toggle("active", t.dataset.tgtab === name));
  for (const p of ["images", "tags", "auto"]) $("tg-pane-" + p).classList.toggle("hidden", p !== name);
  if (name === "tags" && tg.countsStale) renderTagCounts();
  if (name === "auto") renderRunScope();
}

// ------------------------------------------------------------------ events

(function bindTagger() {
  $("btn-tagger").onclick = () => openTagger();
  $("tg-close").onclick = closeTagger;
  $("tg-load").onclick = () => loadTaggerFolders();
  $("tg-path").onkeydown = (e) => {
    if (e.key === "Enter") loadTaggerFolders();
  };
  $("tg-path").oninput = () => ($("tg-folder-pick").value = "");
  $("tg-folder-pick").onchange = (e) => {
    const v = e.target.value;
    if (v && v !== "__job__") $("tg-path").value = v;
    if (v) loadTaggerFolders();
  };
  document.querySelectorAll("#tg-tabs .tab").forEach((t) => (t.onclick = () => switchTaggerTab(t.dataset.tgtab)));
  $("tg-filter").oninput = debounce(() => applyTaggerFilter(), 200);
  $("tg-show").onchange = () => applyTaggerFilter();
  $("tg-thumb").oninput = (e) => $("tg-grid").style.setProperty("--tg-thumb", e.target.value + "px");
  $("tg-prev-page").onclick = () => {
    tg.page--;
    renderTaggerGrid();
  };
  $("tg-next-page").onclick = () => {
    tg.page++;
    renderTaggerGrid();
  };
  $("tg-grid").addEventListener("click", (e) => {
    const card = e.target.closest(".tg-card");
    if (card) selectTaggerImage(+card.dataset.i);
  });
  $("tg-tag-search").oninput = debounce(() => {
    tg.tagLimit = TG_TAG_PAGE;
    renderTagCounts();
  }, 150);
  $("tg-tag-sort").onchange = renderTagCounts;
  $("tg-tags-more").onclick = () => {
    tg.tagLimit += TG_TAG_PAGE;
    renderTagCounts();
  };
  $("tg-add-btn").onclick = () => {
    const tag = $("tg-add-tag").value.trim();
    if (!tag) return showToast("Enter a tag to add");
    const items = scopeItems();
    showConfirm("Add tag", `Add “${tag}” at the ${$("tg-add-pos").value} of ${fmt(items.length)} image caption(s) that lack it? Images without a caption get one.`, () =>
      runBulk(items, { type: "add", tag, position: $("tg-add-pos").value }), "Add");
  };
  $("tg-undo").onclick = async () => {
    try {
      const r = await api("/api/tagger/undo", { method: "POST" });
      showToast(`Undone: ${r.label}`);
    } catch (e) {
      showToast(e.message);
    }
    refreshUndo();
    if (tg.dirs.length) loadTaggerFolders(tg.dirs, true);
  };
  $("tg-run").onclick = startTaggerRun;
  $("tg-run-scope").onchange = renderRunScope;
  $("tg-cancel").onclick = () => api("/api/tagger/cancel", { method: "POST" }).catch((e) => showToast(e.message));
  $("tg-unload").onclick = async () => {
    try {
      onTaggerMessage(await api("/api/tagger/unload", { method: "POST" }));
      showToast("Tagger model unloaded");
    } catch (e) {
      showToast(e.message);
    }
  };
  $("prompt-input").onkeydown = (e) => {
    if (e.key === "Enter") $("btn-prompt-ok").click();
  };
  document.addEventListener("keydown", (e) => {
    if (!taggerOpen() || document.querySelector(".modal:not(.hidden)")) return;
    if ((e.ctrlKey || e.metaKey) && e.key === "s") {
      e.preventDefault();
      saveTaggerCaption();
    } else if (e.altKey && (e.key === "ArrowRight" || e.key === "ArrowLeft")) {
      e.preventDefault();
      selectTaggerImage(tg.sel + (e.key === "ArrowRight" ? 1 : -1));
    }
  });
})();
