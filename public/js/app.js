/* Venno Trainer frontend. Form fields carry data-bind="section.key" and are
   synced with the in-memory job object; data-model / data-show control
   which fields a model shows. */
"use strict";

const S = window.VennoSchema;
const $ = (id) => document.getElementById(id);

let jobs = [];
let currentJob = null; // job name
let job = null; // job object being edited
let savedSnapshot = "";
let running = false;
let settings = null;
let ws = null;
let activeTab = "training";
const scanCache = new Map(); // folder path -> scan result
const selectedFolders = new Set(); // folder objects
let datasetFilter = "";
let lossPoints = [];

// ------------------------------------------------------------------ utils

async function api(url, opts = {}) {
  const init = { method: opts.method || "GET", headers: {} };
  if (opts.body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(opts.body);
  }
  const res = await fetch(url, init);
  let data = null;
  try {
    data = await res.json();
  } catch (e) {
    data = {};
  }
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function escapeHtml(str) {
  return String(str ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function showToast(msg, ms = 3500) {
  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), ms);
}

function openModal(id) {
  $(id).classList.remove("hidden");
}
function closeModal(id) {
  $(id).classList.add("hidden");
}

let confirmHandler = null;
function showConfirm(title, message, onOk, okLabel = "OK") {
  $("confirm-title").textContent = title;
  $("confirm-message").textContent = message;
  $("btn-confirm-ok").textContent = okLabel;
  confirmHandler = onOk;
  openModal("modal-confirm");
}

function debounce(fn, ms) {
  let t;
  return (...a) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...a), ms);
  };
}

function fmt(n) {
  return Number(n || 0).toLocaleString();
}

function getPath(obj, p) {
  return p.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);
}

function setPath(obj, p, v) {
  const keys = p.split(".");
  const last = keys.pop();
  keys.reduce((o, k) => (o[k] ??= {}), obj)[last] = v;
}


// collapsed state is UI-only, it should not mark the job dirty
function snapshot(j) {
  return JSON.stringify(j, (k, v) => (k === "collapsed" ? undefined : v));
}

// -------------------------------------------------------------- websocket

function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}`);
  ws.onopen = () => currentJob && ws.send(JSON.stringify({ type: "subscribe", job: currentJob }));
  ws.onclose = () => setTimeout(connectWS, 2000);
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === "jobs-changed") return loadJobs();
    if (msg.job !== currentJob) return;
    if (msg.type === "log") {
      if (msg.replace) resetConsole();
      appendConsole(msg.data);
    } else if (msg.type === "status") {
      setRunning(msg.data === "running");
    } else if (msg.type === "progress") {
      onProgress(msg.data);
    }
  };
}

// ------------------------------------------------------------------- jobs

async function loadJobs() {
  jobs = await api("/api/jobs");
  renderJobList();
}

function renderJobList() {
  const list = $("job-list");
  list.innerHTML = "";
  for (const j of jobs) {
    const el = document.createElement("div");
    el.className = "job-item" + (j.name === currentJob ? " active" : "");
    const model = S.MODELS[j.model];
    el.innerHTML = `
      ${j.running ? '<span class="running-dot" title="Training"></span>' : ""}
      <span class="job-name">${escapeHtml(j.name)}</span>
      <span class="badge badge-${escapeHtml(j.model)}">${escapeHtml(model ? model.label : j.model)}</span>`;
    el.onclick = () => trySelectJob(j.name);
    list.appendChild(el);
  }
}

function trySelectJob(name) {
  if (name === currentJob) return;
  if (isDirty()) {
    showConfirm("Unsaved changes", `Discard unsaved changes to "${currentJob}"?`, () => selectJob(name), "Discard");
  } else selectJob(name);
}

async function selectJob(name) {
  const data = await api(`/api/jobs/${encodeURIComponent(name)}`);
  currentJob = data.name;
  job = data.job;
  savedSnapshot = snapshot(job);
  selectedFolders.clear();
  $("empty-state").classList.add("hidden");
  $("job-editor").classList.remove("hidden");
  renderJobList();
  populate();
  setRunning(data.running);
  resetConsole();
  if (!data.running) appendConsole("Waiting for training to start...");
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "subscribe", job: currentJob }));
  loadProgress();
  scanDatasets();
  if (activeTab === "samples") loadSamples();
  try {
    localStorage.setItem("last_job", currentJob);
  } catch (e) {}
}

function isDirty() {
  return job && snapshot(job) !== savedSnapshot;
}

function checkDirty() {
  const dirty = isDirty();
  $("btn-save").classList.toggle("hidden", !dirty);
  $("btn-discard").classList.toggle("hidden", !dirty);
  renderIssues();
}

async function saveJob(quiet = false) {
  const res = await api(`/api/jobs/${encodeURIComponent(currentJob)}`, { method: "PUT", body: { job } });
  savedSnapshot = snapshot(job);
  checkDirty();
  if (!quiet) showToast("Saved");
  return res;
}

function setRunning(r) {
  running = r;
  $("btn-run").classList.toggle("hidden", r);
  $("btn-stop").classList.toggle("hidden", !r);
  $("btn-delete").disabled = r;
}

// --------------------------------------------------------------- populate

function fillOptions(select, key) {
  const list = S.OPTIONS[key] || [];
  select.innerHTML = list
    .map((o) => {
      const [v, l] = Array.isArray(o) ? o : [o, o];
      return `<option value="${escapeHtml(v)}">${escapeHtml(l)}</option>`;
    })
    .join("");
}

function populate() {
  const model = S.MODELS[job.model];
  $("job-title").textContent = currentJob;
  const badge = $("job-model-badge");
  badge.className = `badge badge-${job.model}`;
  badge.textContent = model.label;
  $("model-label").value = model.label;

  $("cfg-variant").innerHTML = Object.entries(model.variants)
    .map(([id, v]) => `<option value="${id}">${escapeHtml(v.label)}</option>`)
    .join("");
  document.querySelectorAll("#job-editor select[data-options]").forEach((s) => fillOptions(s, s.dataset.options));

  document.querySelectorAll("#job-editor [data-bind]").forEach((el) => {
    const v = getPath(job, el.dataset.bind);
    if (el.type === "checkbox") el.checked = !!v;
    else if (el.type === "radio") el.checked = el.value === v;
    else {
      el.value = v ?? "";
      // a stored value the option list does not know (e.g. hand-edited job)
      if (el.tagName === "SELECT" && v != null && el.value !== String(v)) {
        el.insertAdjacentHTML("beforeend", `<option value="${escapeHtml(v)}">${escapeHtml(v)} (custom)</option>`);
        el.value = v;
      }
    }
  });
  applyVisibility();
  renderDatasets();
  renderPrompts();
  checkDirty();
}

function readInput(el) {
  if (el.type === "checkbox") return el.checked;
  if (el.type === "number") return el.value === "" ? "" : Number(el.value);
  return el.value;
}

function onBoundInput(e) {
  const el = e.target.closest("[data-bind]");
  if (!el || !job) return;
  if (el.type === "radio" && !el.checked) return;
  setPath(job, el.dataset.bind, readInput(el));
  if (el.dataset.bind.startsWith("dataset.") || el.dataset.bind.startsWith("training.duration") || el.dataset.bind === "variant") {
    renderDatasets();
  }
  applyVisibility();
  checkDirty();
}

function evalCond(cond) {
  const ctx = job;
  let m;
  if ((m = /^(.+?)!=(.*)$/.exec(cond))) return String(getPath(ctx, m[1])) !== m[2];
  if ((m = /^(.+?)=(.*)$/.exec(cond))) return String(getPath(ctx, m[1])) === m[2];
  if (cond.startsWith("!")) return !getPath(ctx, cond.slice(1));
  return !!getPath(ctx, cond);
}

function applyVisibility() {
  document.querySelectorAll("#job-editor [data-model]").forEach((el) => {
    el.classList.toggle("hidden", el.dataset.model !== job.model);
  });
  document.querySelectorAll("#job-editor [data-show]").forEach((el) => {
    if (el.dataset.model && el.dataset.model !== job.model) return;
    // "a&b" = both, "a|b" = either
    const visible = el.dataset.show.split("&").every((term) => term.split("|").some(evalCond));
    el.classList.toggle("hidden", !visible);
  });
  updateEpochHint();
}

function renderIssues() {
  const box = $("issues");
  if (!job) return (box.innerHTML = "");
  box.innerHTML = S.validateJob(job)
    .map((i) => `<div class="issue ${i.level}">${i.level === "error" ? "⛔" : "⚠️"} ${escapeHtml(i.msg)}</div>`)
    .join("");
}

// --------------------------------------------------------------- datasets

const scanDatasets = debounce(async () => {
  if (!job) return;
  const paths = [...new Set(job.dataset.folders.map((f) => f.path.trim()).filter(Boolean))];
  if (!paths.length) return renderDatasetSummary();
  try {
    const results = await api("/api/datasets/scan", { method: "POST", body: { paths, caption_ext: job.dataset.caption_ext } });
    for (const r of results) scanCache.set(r.path, r);
  } catch (e) {
    showToast("Scan failed: " + e.message);
  }
  renderDatasets();
}, 300);

function folderStats(f) {
  return scanCache.get(f.path.trim());
}

// effective samples per epoch from one folder (all resolutions)
function folderEffective(f) {
  const s = folderStats(f);
  if (!f.enabled || !s || !s.exists || f.is_reg) return 0;
  return s.images * Math.max(1, parseInt(f.repeats, 10) || 1) * S.parseIntList(job.dataset.resolutions, [1024]).length;
}

function datasetTotals() {
  const folders = job.dataset.folders;
  const enabled = folders.filter((f) => f.enabled);
  let images = 0;
  let captioned = 0;
  let effective = 0;
  let scanned = true;
  for (const f of enabled) {
    const s = folderStats(f);
    if (!s) {
      scanned = false;
      continue;
    }
    images += s.images;
    captioned += s.captioned;
    effective += folderEffective(f);
  }
  const res = S.parseIntList(job.dataset.resolutions, [1024]);
  const batches = S.parseIntList(job.dataset.batch_sizes, [1]);
  const accum = Math.max(1, parseInt(job.training.grad_accum, 10) || 1);
  // each resolution steps with its own batch size (estimate: bucket
  // remainders add a few batches; the trainer logs the exact count)
  const perRes = effective / res.length;
  let stepsPerEpoch = res.reduce((sum, _, i) => sum + Math.ceil(perRes / (batches[i] ?? batches[batches.length - 1])), 0);
  stepsPerEpoch = Math.ceil(stepsPerEpoch / accum);
  return { total: folders.length, enabled: enabled.length, images, captioned, effective, stepsPerEpoch, scanned };
}

function renderDatasetSummary() {
  const t = datasetTotals();
  const unit = job.training.duration_unit;
  const totalSteps = unit === "epochs" ? t.stepsPerEpoch * (parseInt(job.training.max_epochs, 10) || 0) : parseInt(job.training.max_steps, 10) || 0;
  const epochs = unit === "steps" && t.stepsPerEpoch ? (totalSteps / t.stepsPerEpoch).toFixed(1) : job.training.max_epochs;
  const capPct = t.images ? Math.round((t.captioned / t.images) * 100) : 0;
  $("dataset-summary").innerHTML = `
    <div class="stat"><div class="v">${t.enabled} / ${t.total}</div><div class="k">Folders enabled</div></div>
    <div class="stat"><div class="v">${fmt(t.images)}</div><div class="k">Images</div></div>
    <div class="stat ${capPct < 100 && t.images ? "clickable" : ""}" id="stat-captioned" ${capPct < 100 && t.images ? 'title="Show every uncaptioned image"' : ""}><div class="v" style="color:${capPct < 100 && t.images ? "var(--warning)" : "inherit"}">${capPct}%</div><div class="k">Captioned${capPct < 100 && t.images ? " · review" : ""}</div></div>
    <div class="stat"><div class="v">${fmt(t.effective)}</div><div class="k">Samples / epoch</div></div>
    <div class="stat"><div class="v">${fmt(t.stepsPerEpoch)}</div><div class="k">Steps / epoch</div></div>
    <div class="stat"><div class="v">${fmt(totalSteps)}</div><div class="k">Total steps (${epochs} ep)</div></div>`;
  const stat = $("stat-captioned");
  if (stat && stat.classList.contains("clickable")) {
    stat.style.cursor = "pointer";
    stat.onclick = () => openCaptioner(job.dataset.folders.filter((f) => f.enabled && f.path.trim()).map((f) => f.path.trim()));
  }
  updateEpochHint();
}

// ---- caption uncaptioned images

let capItems = []; // {dir, name}
let capShown = 0;
const CAP_PAGE = 30;

async function openCaptioner(dirs) {
  const res = await api("/api/datasets/uncaptioned", { method: "POST", body: { paths: dirs, caption_ext: job.dataset.caption_ext } });
  capItems = res.flatMap((r) => r.names.map((name) => ({ dir: r.dir, name })));
  capShown = 0;
  $("cap-count").textContent = capItems.length;
  $("cap-where").textContent = dirs.length === 1 ? dirs[0] : `${dirs.length} folders`;
  $("cap-list").innerHTML = capItems.length ? "" : '<p class="hint">Every image has a caption.</p>';
  openModal("modal-captions");
  renderCaptionRows();
}

function imageUrl(it) {
  return `/api/datasets/image?dir=${encodeURIComponent(it.dir)}&name=${encodeURIComponent(it.name)}`;
}

function renderCaptionRows() {
  const list = $("cap-list");
  const multi = new Set(capItems.map((i) => i.dir)).size > 1;
  capItems.slice(capShown, capShown + CAP_PAGE).forEach((it) => {
    const row = document.createElement("div");
    row.className = "cap-row";
    row.innerHTML = `
      <img loading="lazy" src="${imageUrl(it)}" alt="" title="Open full size">
      <div>
        <div class="cap-name">${escapeHtml(multi ? it.dir.split(/[\\/]/).filter(Boolean).pop() + " / " : "")}${escapeHtml(it.name)}</div>
        <textarea placeholder="Caption for this image"></textarea>
      </div>
      <button class="btn btn-sm btn-primary">Save</button>`;
    const ta = row.querySelector("textarea");
    const save = async () => {
      if (!ta.value.trim()) return false;
      try {
        await api("/api/datasets/caption", { method: "POST", body: { dir: it.dir, name: it.name, caption_ext: job.dataset.caption_ext, text: ta.value } });
        row.classList.add("saved");
        row.querySelector("button").textContent = "Saved ✓";
        it.saved = true;
        scanCache.delete(it.dir); // refresh the folder's caption count
        scanDatasets();
        return true;
      } catch (e) {
        showToast(e.message);
        return false;
      }
    };
    row.querySelector("button").onclick = save;
    row.querySelector("img").onclick = () => window.open(imageUrl(it), "_blank");
    ta.addEventListener("keydown", async (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        if (await save()) {
          const next = [...list.querySelectorAll(".cap-row:not(.saved) textarea")][0];
          if (next) next.focus();
        }
      }
    });
    it.row = row;
    list.appendChild(row);
  });
  capShown = Math.min(capItems.length, capShown + CAP_PAGE);
  $("btn-cap-more").classList.toggle("hidden", capShown >= capItems.length);
  $("btn-cap-more").textContent = `Show more (${capItems.length - capShown} left)`;
}

async function saveAllCaptions() {
  let n = 0;
  for (const it of capItems) {
    if (!it.row || it.saved) continue;
    const ta = it.row.querySelector("textarea");
    if (ta.value.trim()) {
      it.row.querySelector("button").click();
      n++;
    }
  }
  showToast(n ? `Saving ${n} caption(s)` : "No filled boxes to save");
}

function updateEpochHint() {
  const el = $("epoch-steps-hint");
  if (!el || !job) return;
  const t = datasetTotals();
  if (!t.stepsPerEpoch) return (el.textContent = "");
  el.textContent =
    job.training.duration_unit === "steps"
      ? `About ${fmt(t.stepsPerEpoch)} steps per epoch → ${((parseInt(job.training.max_steps, 10) || 0) / t.stepsPerEpoch).toFixed(1)} epochs.`
      : `About ${fmt(t.stepsPerEpoch)} steps per epoch → ${fmt(t.stepsPerEpoch * (parseInt(job.training.max_epochs, 10) || 0))} steps.`;
}

function folderChips(f) {
  const s = folderStats(f);
  const chips = [];
  if (s && !s.exists && f.path.trim()) chips.push('<span class="chip bad">not found</span>');
  else if (s) {
    chips.push(`<span class="chip">${fmt(s.images)} img</span>`);
    if (s.captioned < s.images) chips.push(`<span class="chip warn clickable ds-uncaptioned" data-dir="${escapeHtml(f.path.trim())}" title="Show and caption these images">${fmt(s.images - s.captioned)} uncaptioned</span>`);
  }
  chips.push(`<span class="chip" title="Repeats">×${escapeHtml(f.repeats)}</span>`);
  if (Number(f.loss_weight) !== 1) chips.push(`<span class="chip" title="Loss weight">w ${escapeHtml(f.loss_weight)}</span>`);
  if (Number(f.caption_dropout) > 0) chips.push(`<span class="chip" title="Caption dropout">cap ${Math.round(f.caption_dropout * 100)}%</span>`);
  if (Number(f.tag_dropout) > 0) chips.push(`<span class="chip" title="Tag dropout">tag ${Math.round(f.tag_dropout * 100)}%</span>`);
  if (f.is_reg) chips.push('<span class="badge badge-reg">REG</span>');
  return chips.join("");
}

function folderMatchesFilter(f) {
  return !datasetFilter || f.path.toLowerCase().includes(datasetFilter) || (f.caption_prefix || "").toLowerCase().includes(datasetFilter);
}

function renderDatasets() {
  if (!job) return;
  renderDatasetSummary();
  const list = $("dataset-list");
  list.innerHTML = "";
  const folders = job.dataset.folders;
  const totalEff = folders.reduce((s, f) => s + folderEffective(f), 0);

  folders.forEach((f, idx) => {
    if (!folderMatchesFilter(f)) return;
    const card = document.createElement("div");
    card.className = "ds-card" + (selectedFolders.has(f) ? " selected" : "") + (f.enabled ? "" : " disabled-ds");
    const dirName = f.path.trim() ? f.path.trim().split(/[\\/]/).filter(Boolean).pop() : "No folder set";
    const share = totalEff ? (folderEffective(f) / totalEff) * 100 : 0;
    const s = folderStats(f);
    card.innerHTML = `
      <div class="ds-head">
        <input type="checkbox" class="ds-select" title="Select for bulk edit" ${selectedFolders.has(f) ? "checked" : ""}>
        <label class="switch" title="Include in training"><input type="checkbox" class="ds-enabled" ${f.enabled ? "checked" : ""}><span></span></label>
        <div class="ds-toggle">
          <span class="caret ${f.collapsed ? "closed" : ""}">▼</span>
          <span class="ds-title">Dataset ${idx + 1}</span>
          <span class="ds-dir" title="${escapeHtml(f.path)}">${escapeHtml(dirName)}</span>
        </div>
        <div class="ds-chips">${folderChips(f)}</div>
        <span class="chip" title="Share of each epoch">${share.toFixed(1)}%</span>
        <button class="btn-icon ds-up" title="Move up" ${idx === 0 ? "disabled" : ""}>↑</button>
        <button class="btn-icon ds-down" title="Move down" ${idx === folders.length - 1 ? "disabled" : ""}>↓</button>
        <button class="btn-icon ds-dup" title="Duplicate">⧉</button>
        <button class="btn-icon ds-del" title="Delete" ${folders.length <= 1 ? "disabled" : ""}>🗑</button>
      </div>
      <div class="share-bar"><div style="width:${share}%"></div></div>
      <div class="ds-body ${f.collapsed ? "hidden" : ""}">
        <div class="form-group">
          <label>Image Directory</label>
          <div class="inline">
            <input type="text" data-f="path" value="${escapeHtml(f.path)}" placeholder="/path/to/images">
            <button class="btn ds-open" title="Open folder">📂</button>
          </div>
          <small class="ds-scan">${s && s.exists ? `${fmt(s.images)} images, ${fmt(s.captioned)} with ${escapeHtml(job.dataset.caption_ext)} captions${s.missing.length ? ` — <a href="#" class="ds-uncaptioned" data-dir="${escapeHtml(f.path.trim())}">show the ${fmt(s.images - s.captioned)} uncaptioned</a>` : ""}` : s && f.path.trim() ? "Folder not found" : ""}</small>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label>Repeats</label>
            <input type="number" data-f="repeats" value="${escapeHtml(f.repeats)}" min="1">
          </div>
          <div class="form-group">
            <label>Loss Weight</label>
            <input type="number" data-f="loss_weight" value="${escapeHtml(f.loss_weight)}" step="0.1" min="0">
            <small>Multiplies this folder's loss.</small>
          </div>
          <div class="form-group">
            <label>Keep Tokens</label>
            <input type="number" data-f="keep_tokens" value="${escapeHtml(f.keep_tokens)}" min="0">
            <small>Leading tags never dropped/shuffled.</small>
          </div>
        </div>
        <div class="form-group">
          <label>Caption Prefix / Trigger</label>
          <input type="text" data-f="caption_prefix" value="${escapeHtml(f.caption_prefix)}" placeholder="e.g. sks character,">
          <small>Prepended to every caption in this folder (counts toward keep tokens).</small>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label>Caption Dropout</label>
            <input type="number" data-f="caption_dropout" value="${escapeHtml(f.caption_dropout)}" step="0.01" min="0" max="1">
            <small>Chance to train with an empty caption.</small>
          </div>
          <div class="form-group">
            <label>Tag Dropout</label>
            <input type="number" data-f="tag_dropout" value="${escapeHtml(f.tag_dropout)}" step="0.01" min="0" max="1">
            <small>Chance to drop each comma-separated tag.</small>
          </div>
          <div class="form-group">
            <label>Dropout Every N Epochs</label>
            <input type="number" data-f="dropout_every_n_epochs" value="${escapeHtml(f.dropout_every_n_epochs)}" min="0">
            <small>Whole epochs trained without captions. 0 = off.</small>
          </div>
        </div>
        <div class="checkbox-group" style="margin-bottom:10px;">
          <label class="hint inline"><input type="checkbox" data-f="shuffle" ${f.shuffle ? "checked" : ""}> Shuffle captions</label>
          <label class="hint inline"><input type="checkbox" data-f="flip" ${f.flip ? "checked" : ""}> Flip augmentation</label>
          <label class="hint inline"><input type="checkbox" data-f="is_reg" ${f.is_reg ? "checked" : ""}> Regularization dataset</label>
        </div>
      </div>`;

    const head = card.querySelector(".ds-head");
    card.querySelector(".ds-toggle").onclick = () => {
      f.collapsed = !f.collapsed;
      renderDatasets();
    };
    card.querySelector(".ds-select").onchange = (e) => {
      if (e.target.checked) selectedFolders.add(f);
      else selectedFolders.delete(f);
      card.classList.toggle("selected", e.target.checked);
    };
    card.querySelector(".ds-enabled").onchange = (e) => {
      f.enabled = e.target.checked;
      renderDatasets();
      checkDirty();
    };
    card.querySelector(".ds-up").onclick = () => moveFolder(idx, -1);
    card.querySelector(".ds-down").onclick = () => moveFolder(idx, 1);
    card.querySelector(".ds-dup").onclick = () => {
      const copy = JSON.parse(JSON.stringify(f));
      copy.path = "";
      folders.splice(idx + 1, 0, copy);
      renderDatasets();
      checkDirty();
    };
    card.querySelector(".ds-del").onclick = () => {
      if (folders.length <= 1) return;
      folders.splice(idx, 1);
      selectedFolders.delete(f);
      renderDatasets();
      checkDirty();
    };
    const openBtn = card.querySelector(".ds-open");
    if (openBtn) openBtn.onclick = () => openFolder(f.path.trim());

    card.querySelectorAll("[data-f]").forEach((input) => {
      const handler = () => {
        const key = input.dataset.f;
        f[key] = input.type === "checkbox" ? input.checked : input.type === "number" ? (input.value === "" ? 0 : Number(input.value)) : input.value;
        // refresh header chips without re-rendering the focused input
        head.querySelector(".ds-chips").innerHTML = folderChips(f);
        if (key === "path") {
          head.querySelector(".ds-dir").textContent = f.path.trim().split(/[\\/]/).filter(Boolean).pop() || "No folder set";
          scanDatasets();
        }
        if (key === "repeats" || key === "is_reg") updateShares();
        checkDirty();
      };
      input.addEventListener(input.type === "checkbox" ? "change" : "input", handler);
    });
    list.appendChild(card);
  });
  $("chk-select-all").checked = folders.length > 0 && folders.every((f) => selectedFolders.has(f));
}

// shares and summary change with repeats; avoid a full re-render while typing
function updateShares() {
  renderDatasetSummary();
  const folders = job.dataset.folders.filter(folderMatchesFilter);
  const totalEff = job.dataset.folders.reduce((s, f) => s + folderEffective(f), 0);
  document.querySelectorAll("#dataset-list .ds-card").forEach((card, i) => {
    const f = folders[i];
    if (!f) return;
    const share = totalEff ? (folderEffective(f) / totalEff) * 100 : 0;
    card.querySelector(".share-bar > div").style.width = share + "%";
    card.querySelector(".ds-head > .chip").textContent = share.toFixed(1) + "%";
  });
}

function moveFolder(idx, dir) {
  const folders = job.dataset.folders;
  const j = idx + dir;
  if (j < 0 || j >= folders.length) return;
  [folders[idx], folders[j]] = [folders[j], folders[idx]];
  renderDatasets();
  checkDirty();
}

function addFolder(props = {}) {
  const f = Object.assign(S.defaultFolder(), props);
  // a single untouched empty card is replaced instead of kept
  const folders = job.dataset.folders;
  if (folders.length === 1 && !folders[0].path.trim() && props.path) folders.length = 0;
  folders.push(f);
  return f;
}

async function openFolder(p, jobRel = false) {
  if (!p) return showToast("Enter a directory path first");
  try {
    await api("/api/system/open-folder", { method: "POST", body: jobRel ? { job: currentJob, path: p } : { path: p } });
  } catch (e) {
    showToast(e.message);
  }
}

// ---- add subfolders

let subfolderResults = [];

async function scanSubfolders() {
  const parent = $("subfolder-parent").value.trim();
  if (!parent) return;
  try {
    subfolderResults = await api("/api/datasets/subfolders", { method: "POST", body: { path: parent, caption_ext: job.dataset.caption_ext } });
  } catch (e) {
    $("subfolder-list").innerHTML = `<p class="hint" style="padding:10px;">${escapeHtml(e.message)}</p>`;
    return;
  }
  const existing = new Set(job.dataset.folders.map((f) => f.path.trim()));
  const defRepeats = parseInt($("subfolder-default-repeats").value, 10) || 1;
  $("subfolder-list").innerHTML =
    subfolderResults
      .map(
        (s, i) => `
      <div class="subfolder-row">
        <input type="checkbox" class="sf-check" data-i="${i}" ${existing.has(s.path) || !s.images ? "" : "checked"} ${existing.has(s.path) ? "disabled" : ""}>
        <span title="${escapeHtml(s.path)}">${escapeHtml(s.name)} ${existing.has(s.path) ? '<span class="chip">already added</span>' : ""}</span>
        <span class="hint">${fmt(s.images)} img${s.captioned < s.images ? ` <span class="chip warn">${fmt(s.images - s.captioned)} no cap</span>` : ""}</span>
        <input type="number" class="sf-repeats" data-i="${i}" min="1" value="${s.repeats ?? defRepeats}" title="Repeats">
      </div>`,
      )
      .join("") || '<p class="hint" style="padding:10px;">No subfolders found.</p>';
  const imgs = subfolderResults.reduce((a, s) => a + s.images, 0);
  $("subfolder-summary").textContent = `${subfolderResults.length} folders, ${fmt(imgs)} images`;
}

function addSelectedSubfolders() {
  const rows = [...document.querySelectorAll("#subfolder-list .sf-check")].filter((c) => c.checked && !c.disabled);
  const dropout = parseFloat($("subfolder-dropout").value) || 0;
  const tagDropout = parseFloat($("subfolder-tag-dropout").value) || 0;
  for (const c of rows) {
    const s = subfolderResults[+c.dataset.i];
    const repeats = parseInt(document.querySelector(`.sf-repeats[data-i="${c.dataset.i}"]`).value, 10) || 1;
    addFolder({ path: s.path, repeats, caption_dropout: dropout, tag_dropout: tagDropout, collapsed: true });
    scanCache.set(s.path, s);
  }
  closeModal("modal-subfolders");
  renderDatasets();
  checkDirty();
  showToast(`Added ${rows.length} dataset folder(s)`);
}

// ---- bulk edit

const BULK_FIELDS = [
  ["repeats", "Repeats", "number", "1"],
  ["loss_weight", "Loss weight", "number", "0.1"],
  ["caption_dropout", "Caption dropout", "number", "0.01"],
  ["tag_dropout", "Tag dropout", "number", "0.01"],
  ["dropout_every_n_epochs", "Dropout every N epochs", "number", "1"],
  ["keep_tokens", "Keep tokens", "number", "1"],
  ["caption_prefix", "Caption prefix / trigger", "text"],
  ["shuffle", "Shuffle captions", "checkbox"],
  ["flip", "Flip augmentation", "checkbox"],
  ["is_reg", "Regularization", "checkbox"],
  ["enabled", "Enabled", "checkbox"],
];

function bulkTargets() {
  return selectedFolders.size ? job.dataset.folders.filter((f) => selectedFolders.has(f)) : job.dataset.folders;
}

function openBulkEdit() {
  const targets = bulkTargets();
  $("bulk-target").textContent = selectedFolders.size
    ? `Applies to the ${targets.length} selected dataset(s).`
    : `No datasets selected — applies to all ${targets.length} datasets.`;
  const first = targets[0] || S.defaultFolder();
  $("bulk-fields").innerHTML = BULK_FIELDS
    .map(([k, label, type, step]) => {
      const input =
        type === "checkbox"
          ? `<input type="checkbox" class="bulk-val" data-k="${k}" ${first[k] ? "checked" : ""}>`
          : `<input type="${type}" class="bulk-val" data-k="${k}" value="${escapeHtml(first[k])}" ${step ? `step="${step}"` : ""}>`;
      return `<div class="bulk-row"><input type="checkbox" class="bulk-on" data-k="${k}"><span>${label}</span>${input}</div>`;
    })
    .join("");
  // editing a value ticks its row
  $("bulk-fields").querySelectorAll(".bulk-val").forEach((el) => {
    el.addEventListener(el.type === "checkbox" ? "change" : "input", () => {
      $("bulk-fields").querySelector(`.bulk-on[data-k="${el.dataset.k}"]`).checked = true;
    });
  });
  openModal("modal-bulk");
}

function applyBulkEdit() {
  const changes = {};
  $("bulk-fields").querySelectorAll(".bulk-on:checked").forEach((on) => {
    const el = $("bulk-fields").querySelector(`.bulk-val[data-k="${on.dataset.k}"]`);
    changes[on.dataset.k] = el.type === "checkbox" ? el.checked : el.type === "number" ? Number(el.value) : el.value;
  });
  const targets = bulkTargets();
  for (const f of targets) Object.assign(f, changes);
  closeModal("modal-bulk");
  renderDatasets();
  checkDirty();
  showToast(`Updated ${Object.keys(changes).length} field(s) on ${targets.length} dataset(s)`);
}

// ---------------------------------------------------------------- prompts

function renderPrompts() {
  const list = $("prompt-list");
  const prompts = job.sampling.prompts;
  $("prompts-empty").classList.toggle("hidden", prompts.length > 0);
  list.innerHTML = "";
  prompts.forEach((p, idx) => {
    const card = document.createElement("div");
    card.className = "prompt-card" + (p.enabled === false ? " skipped" : "");
    const s = job.sampling;
    const val = (k, d) => escapeHtml(p[k] ?? d);
    card.innerHTML = `
      <div class="prompt-card-header">
        <label class="hint inline"><input type="checkbox" data-p="enabled" ${p.enabled !== false ? "checked" : ""}> Prompt ${idx + 1}</label>
        <div>
          <button class="btn-icon p-dup" title="Duplicate">⧉</button>
          <button class="btn-icon p-del" title="Delete">🗑</button>
        </div>
      </div>
      <textarea data-p="text" rows="2" placeholder="Prompt text">${escapeHtml(p.text)}</textarea>
      <div class="prompt-params">
        <div class="compact-input"><label>W</label><input type="number" data-p="width" step="16" value="${val("width", s.width)}"></div>
        <div class="compact-input"><label>H</label><input type="number" data-p="height" step="16" value="${val("height", s.height)}"></div>
        <div class="compact-input"><label>Steps</label><input type="number" data-p="steps" value="${val("steps", s.steps)}"></div>
        <div class="compact-input"><label>CFG</label><input type="number" data-p="cfg" step="0.5" value="${val("cfg", s.cfg)}"></div>
        <div class="compact-input"><label>Seed</label><input type="number" data-p="seed" value="${val("seed", s.seed)}"></div>
      </div>`;
    card.querySelectorAll("[data-p]").forEach((el) => {
      el.addEventListener(el.type === "checkbox" ? "change" : "input", () => {
        const k = el.dataset.p;
        p[k] = el.type === "checkbox" ? el.checked : el.type === "number" ? Number(el.value) : el.value;
        if (k === "enabled") card.classList.toggle("skipped", !el.checked);
        checkDirty();
      });
    });
    card.querySelector(".p-dup").onclick = () => {
      prompts.splice(idx + 1, 0, JSON.parse(JSON.stringify(p)));
      renderPrompts();
      checkDirty();
    };
    card.querySelector(".p-del").onclick = () => {
      prompts.splice(idx, 1);
      renderPrompts();
      checkDirty();
    };
    list.appendChild(card);
  });
}

function newPrompt(text = "") {
  const s = job.sampling;
  return { text, enabled: true, width: s.width, height: s.height, steps: s.steps, cfg: s.cfg, seed: s.seed };
}

// ---------------------------------------------------------------- samples

let sampleImages = [];
const selectedSamples = new Set();
let lastClickedSample = -1;

async function loadSamples() {
  if (!currentJob) return;
  try {
    const [data, ck] = await Promise.all([
      api(`/api/jobs/${encodeURIComponent(currentJob)}/samples`),
      api(`/api/jobs/${encodeURIComponent(currentJob)}/checkpoints`),
    ]);
    sampleImages = data.images;
    renderSamples();
    $("ckpt-count").textContent = ck.checkpoints.length;
    $("ckpt-list").innerHTML =
      ck.checkpoints
        .map((c) => `<div><span>${escapeHtml(c.name)}</span><span class="hint">${(c.size / 1048576).toFixed(1)} MB · ${new Date(c.mtime).toLocaleString()}</span></div>`)
        .join("") || `<p class="hint">No checkpoints yet in <code>${escapeHtml(ck.dir)}</code>.</p>`;
  } catch (e) {
    showToast("Could not load samples: " + e.message);
  }
}

function renderSamples() {
  const limit = $("samples-limit").value;
  const shown = limit === "all" ? sampleImages : sampleImages.slice(0, parseInt(limit, 10));
  $("samples-count").textContent = sampleImages.length;
  $("samples-empty").classList.toggle("hidden", sampleImages.length > 0);
  const grid = $("samples-grid");
  grid.innerHTML = "";
  shown.forEach((img, i) => {
    const card = document.createElement("div");
    card.className = "sample-card" + (selectedSamples.has(img.rel) ? " selected" : "");
    const src = `/api/jobs/${encodeURIComponent(currentJob)}/file?rel=${encodeURIComponent(img.rel)}&t=${Math.round(img.mtime)}`;
    card.innerHTML = `<img loading="lazy" src="${src}" alt=""><div class="cap" title="${escapeHtml(img.name)}">${escapeHtml(img.name)}</div>`;
    card.onclick = (e) => {
      if (e.ctrlKey || e.metaKey) {
        selectedSamples.has(img.rel) ? selectedSamples.delete(img.rel) : selectedSamples.add(img.rel);
      } else if (e.shiftKey && lastClickedSample >= 0) {
        const [a, b] = [Math.min(lastClickedSample, i), Math.max(lastClickedSample, i)];
        for (let k = a; k <= b; k++) selectedSamples.add(shown[k].rel);
      } else {
        openLightbox(shown, i);
        return;
      }
      lastClickedSample = i;
      renderSamples();
    };
    grid.appendChild(card);
  });
  $("btn-delete-samples").classList.toggle("hidden", selectedSamples.size === 0);
}

function openLightbox(list, idx) {
  const lb = document.createElement("div");
  lb.className = "lightbox";
  const show = () => {
    const img = list[idx];
    lb.innerHTML = `<img src="/api/jobs/${encodeURIComponent(currentJob)}/file?rel=${encodeURIComponent(img.rel)}"><div class="lb-cap">${escapeHtml(img.name)} · ${idx + 1}/${list.length} · ← → to browse, Esc to close</div>`;
  };
  const onKey = (e) => {
    if (e.key === "ArrowRight") idx = Math.min(list.length - 1, idx + 1);
    else if (e.key === "ArrowLeft") idx = Math.max(0, idx - 1);
    else if (e.key === "Escape") return close();
    else return;
    show();
  };
  const close = () => {
    document.removeEventListener("keydown", onKey);
    lb.remove();
  };
  lb.onclick = close;
  document.addEventListener("keydown", onKey);
  show();
  document.body.appendChild(lb);
}

// ---------------------------------------------------------------- console

let consoleLines = [""];
let consoleRaf = 0;

function resetConsole() {
  consoleLines = [""];
  $("console-output").textContent = "";
}

function appendConsole(text) {
  const parts = text.split("\n");
  parts.forEach((part, i) => {
    if (i > 0) consoleLines.push("");
    // tqdm redraws the current line after \r
    const segs = part.split("\r");
    if (segs.length > 1) {
      const lastNonEmpty = segs.filter((s) => s.length).pop();
      if (lastNonEmpty !== undefined) consoleLines[consoleLines.length - 1] = lastNonEmpty;
    } else consoleLines[consoleLines.length - 1] += part;
  });
  if (consoleLines.length > 3000) consoleLines = consoleLines.slice(-3000);
  if (!consoleRaf) {
    consoleRaf = requestAnimationFrame(() => {
      consoleRaf = 0;
      const out = $("console-output");
      out.textContent = consoleLines.join("\n");
      if ($("chk-autoscroll").checked) out.scrollTop = out.scrollHeight;
    });
  }
}

async function loadProgress() {
  try {
    const p = await api(`/api/jobs/${encodeURIComponent(currentJob)}/progress`);
    lossPoints = p.points || [];
    if (p.progress) updateProgressLine(p.progress);
    else resetProgressLine();
    drawLossChart();
  } catch (e) {
    /* no progress yet */
  }
}

function resetProgressLine() {
  $("prog-step").textContent = "Step –";
  $("prog-loss").textContent = "Loss –";
  $("prog-vram").textContent = "VRAM –";
  $("prog-hint").textContent = "";
  $("prog-eta").textContent = "ETA –";
  $("prog-fill").style.width = "0";
}

// Turn the trainer's peak VRAM into a tuning hint for the GPU being used.
function vramHint(peakGb) {
  const idx = parseInt(String(job?.training.gpu_ids || "0").split(",")[0], 10) || 0;
  const gpu = gpus.find((g) => g.index === idx) || gpus[0];
  if (!gpu || !peakGb) return "";
  const total = gpu.memTotal / 1024;
  const frac = peakGb / total;
  if (frac < 0.7) {
    const opts = job.training.gradient_checkpointing ? "raise 'Blocks Without Checkpointing', " : "";
    return `${(total - peakGb).toFixed(0)} GB of ${gpu.name} unused: ${opts}a larger batch size or resolution would use it.`;
  }
  if (frac > 0.95) return "Close to the VRAM limit: lower batch/resolution or 'Blocks Without Checkpointing', or enable fp8 base weights.";
  return "";
}

function updateProgressLine(p) {
  $("prog-step").textContent = `Step ${fmt(p.step)} / ${fmt(p.total)}`;
  $("prog-loss").textContent = `Loss ${p.loss.toFixed(4)}`;
  $("prog-eta").textContent = `Elapsed ${p.elapsed} · ETA ${p.eta}`;
  if (p.vram) {
    $("prog-vram").textContent = `VRAM peak ${p.vram.toFixed(1)} GB`;
    $("prog-hint").textContent = vramHint(p.vram);
  }
  $("prog-fill").style.width = `${p.total ? (p.step / p.total) * 100 : 0}%`;
}

let lastSampleRefresh = 0;
function onProgress(p) {
  if (lossPoints.length && p.step < lossPoints[lossPoints.length - 1][0]) lossPoints = []; // new run
  lossPoints.push([p.step, p.loss]);
  updateProgressLine(p);
  if (activeTab === "console") drawLossChart();
  if (activeTab === "samples" && Date.now() - lastSampleRefresh > 15000) {
    lastSampleRefresh = Date.now();
    loadSamples();
  }
}

function drawLossChart() {
  const canvas = $("loss-chart");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (!w) return;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.documentElement);
  const muted = css.getPropertyValue("--text-muted").trim();
  const accent = css.getPropertyValue("--accent").trim();
  const border = css.getPropertyValue("--border").trim();
  ctx.font = "11px sans-serif";
  if (lossPoints.length < 2) {
    ctx.fillStyle = muted;
    ctx.fillText("Loss curve appears once training reports steps.", 10, h / 2);
    return;
  }
  // bias-corrected EMA over the raw points; the window grows with the run
  // so short runs are not flattened into a lagging line
  const win = Math.min(100, Math.max(3, lossPoints.length / 15));
  const beta = 1 - 1 / win;
  const smooth = [];
  let ema = 0;
  lossPoints.forEach(([, l], i) => {
    ema = ema * beta + l * (1 - beta);
    smooth.push(ema / (1 - Math.pow(beta, i + 1)));
  });
  const sorted = lossPoints.map((p) => p[1]).sort((a, b) => a - b);
  const lo = sorted[Math.floor(sorted.length * 0.01)];
  const hi = sorted[Math.floor(sorted.length * 0.99)];
  const pad = { l: 46, r: 10, t: 8, b: 18 };
  const x0 = lossPoints[0][0];
  const x1 = lossPoints[lossPoints.length - 1][0] || 1;
  const X = (s) => pad.l + ((s - x0) / Math.max(1, x1 - x0)) * (w - pad.l - pad.r);
  const Y = (l) => pad.t + (1 - (Math.min(hi, Math.max(lo, l)) - lo) / Math.max(1e-9, hi - lo)) * (h - pad.t - pad.b);
  ctx.strokeStyle = border;
  ctx.fillStyle = muted;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const v = lo + ((hi - lo) * i) / 3;
    const y = Y(v);
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
    ctx.fillText(v.toFixed(3), 4, y + 4);
  }
  ctx.fillText(`step ${fmt(x0)}`, pad.l, h - 4);
  const endLabel = `step ${fmt(x1)}`;
  ctx.fillText(endLabel, w - pad.r - ctx.measureText(endLabel).width, h - 4);
  ctx.globalAlpha = 0.25;
  ctx.strokeStyle = accent;
  ctx.beginPath();
  lossPoints.forEach(([s, l], i) => (i ? ctx.lineTo(X(s), Y(l)) : ctx.moveTo(X(s), Y(l))));
  ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.lineWidth = 2;
  ctx.beginPath();
  lossPoints.forEach(([s], i) => (i ? ctx.lineTo(X(s), Y(smooth[i])) : ctx.moveTo(X(s), Y(smooth[i]))));
  ctx.stroke();
}

// -------------------------------------------------------- global settings

function buildGlobalSettings() {
  const tabs = [
    {
      id: "trainer",
      label: "Trainer",
      html: `
        <div class="note">Training runs in the bundled Python trainer. Run <code>./setup.sh</code> once to create its environment in <code>./venv</code>.</div>
        <div class="form-group"><label>Python Venv (or python executable)</label><input type="text" data-g="python" placeholder="Blank = ./venv"></div>
        <div class="form-group"><div class="inline"><button class="btn btn-sm" id="btn-check-python">Check environment</button><span id="python-status" class="hint"></span></div></div>
        <div class="form-group"><label>Cache Folder</label><input type="text" data-g="cache_dir" placeholder="Blank = ./cache"><small>Latent and caption caches shared by all jobs.</small></div>
        <div class="form-group"><div class="inline"><span id="cache-status" class="hint">…</span></div></div>
        <div class="form-group"><label>Model Download Folder</label><input type="text" data-g="models_dir" placeholder="Blank = ./models"><small>Where MOSS-Image is downloaded when no model folder is set.</small></div>
        <div class="form-group"><label>HuggingFace Token</label><input type="password" data-g="hf_token" placeholder="hf_..."><small>Optional; faster, rate-limit-free downloads.</small></div>`,
    },
    ...Object.entries(S.MODELS).map(([id, m]) => ({
      id: "model-" + id,
      label: m.label,
      html: Object.entries(m.paths)
        .map(([k, p]) => `<div class="form-group"><label>${escapeHtml(p.label)}</label><input type="text" data-g="models.${id}.${k}" placeholder="${escapeHtml(p.placeholder)}"></div>`)
        .join(""),
    })),
    {
      id: "appearance",
      label: "Appearance",
      html: `<div class="form-group"><label>Theme</label><select data-g="theme">
        <option value="github-dark">GitHub Dark</option><option value="github-light">GitHub Light</option>
        <option value="midnight-blue">Midnight Blue</option><option value="cherry-pink">Cherry Pink</option></select></div>`,
    },
  ];
  $("global-tabs").innerHTML = tabs.map((t, i) => `<button class="tab ${i ? "" : "active"}" data-gtab="${t.id}">${escapeHtml(t.label)}</button>`).join("");
  $("global-panes").innerHTML = tabs.map((t, i) => `<div class="gtab-pane ${i ? "hidden" : ""}" id="gtab-${t.id}">${t.html}</div>`).join("");
  $("global-tabs").querySelectorAll("[data-gtab]").forEach((b) => {
    b.onclick = () => {
      $("global-tabs").querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
      document.querySelectorAll(".gtab-pane").forEach((p) => p.classList.toggle("hidden", p.id !== "gtab-" + b.dataset.gtab));
    };
  });
}

async function openGlobalSettings() {
  settings = await api("/api/settings");
  document.querySelectorAll("[data-g]").forEach((el) => (el.value = getPath(settings, el.dataset.g) ?? ""));
  $("python-status").textContent = "";
  openModal("modal-global-settings");
  loadCacheStatus();
}

function fmtBytes(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(1) + " GB";
  if (n >= 1e6) return (n / 1e6).toFixed(0) + " MB";
  return Math.ceil(n / 1e3) + " KB";
}

async function loadCacheStatus() {
  const box = $("cache-status");
  try {
    const c = await api("/api/system/cache");
    box.innerHTML = Object.entries(c.models)
      .map(([id, v]) => `${escapeHtml(S.MODELS[id].label)}: ${fmtBytes(v.bytes)} <button class="btn btn-xs" data-clear-cache="${id}" ${v.files ? "" : "disabled"}>Clear</button>`)
      .join(" &nbsp; ");
    box.querySelectorAll("[data-clear-cache]").forEach((b) => {
      b.onclick = () =>
        showConfirm("Clear cache", `Delete the cached latents and captions for ${S.MODELS[b.dataset.clearCache].label}? They are rebuilt on the next run.`, async () => {
          try {
            await api(`/api/system/cache/${b.dataset.clearCache}`, { method: "DELETE" });
          } catch (e) {
            showToast(e.message);
          }
          loadCacheStatus();
        }, "Clear");
    });
  } catch (e) {
    box.textContent = e.message;
  }
}

async function checkPython() {
  const box = $("python-status");
  box.textContent = "Checking…";
  const r = await api("/api/system/check-python", { method: "POST", body: { python: document.querySelector('[data-g="python"]').value.trim() } });
  if (!r.ok) {
    box.innerHTML = `<span style="color:var(--danger)">${escapeHtml(r.error || "failed")}</span>`;
    return;
  }
  const missing = ["torch", "transformers", "safetensors", "bitsandbytes"].filter((m) => !r[m]);
  box.innerHTML = `Python ${escapeHtml(r.python_version)} · torch ${escapeHtml(r.torch || "missing")} · ` +
    (r.cuda ? `GPU: ${escapeHtml(r.gpu)}` : '<span style="color:var(--warning)">no CUDA GPU</span>') +
    (missing.length ? ` · <span style="color:var(--danger)">missing: ${missing.join(", ")}</span>` : " · packages OK") +
    (r.fla ? "" : " · tip: pip install flash-linear-attention speeds up MOSS caption encoding");
}

async function saveGlobalSettings() {
  document.querySelectorAll("[data-g]").forEach((el) => setPath(settings, el.dataset.g, el.value.trim()));
  settings = await api("/api/settings", { method: "PUT", body: settings });
  applyTheme(settings.theme);
  closeModal("modal-global-settings");
  showToast("Settings saved");
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme || "github-dark");
  try {
    localStorage.setItem("ui_theme", theme || "github-dark");
  } catch (e) {}
}

// ------------------------------------------------------------ gpu monitor

let gpus = [];
async function pollGPUs() {
  try {
    gpus = await api("/api/system/gpus");
  } catch (e) {
    gpus = [];
  }
  const bar = $("hw-monitor");
  bar.innerHTML = gpus.length
    ? gpus
        .map(
          (g) => `<span class="hw-gpu">GPU${g.index} <b>${escapeHtml(g.name)}</b>
          VRAM<span class="hw-meter"><i style="width:${(g.memUsed / g.memTotal) * 100}%"></i></span>${(g.memUsed / 1024).toFixed(1)}/${(g.memTotal / 1024).toFixed(1)} GB
          · ${g.util}% · ${g.temp}°C</span>`,
        )
        .join("")
    : "<span>No NVIDIA GPU detected (nvidia-smi unavailable)</span>";
  renderGpuChips();
}

function renderGpuChips() {
  const box = $("gpu-chips");
  if (!box || !job) return;
  const selected = String(job.training.gpu_ids || "").split(",").map((s) => s.trim());
  box.innerHTML = gpus
    .map((g) => `<button class="btn btn-xs ${selected.includes(String(g.index)) ? "btn-primary" : ""}" data-gpu="${g.index}" title="${escapeHtml(g.name)}">GPU ${g.index}</button>`)
    .join("");
  box.querySelectorAll("[data-gpu]").forEach((b) => {
    b.onclick = () => {
      const set = new Set(String(job.training.gpu_ids || "").split(",").map((s) => s.trim()).filter(Boolean));
      set.has(b.dataset.gpu) ? set.delete(b.dataset.gpu) : set.add(b.dataset.gpu);
      job.training.gpu_ids = [...set].sort().join(",");
      document.querySelector('[data-bind="training.gpu_ids"]').value = job.training.gpu_ids;
      renderGpuChips();
      checkDirty();
    };
  });
}

// ------------------------------------------------------------------ tabs

function switchTab(name) {
  activeTab = name;
  document.querySelectorAll("#job-tabs .tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tab-content > .tab-pane").forEach((p) => p.classList.toggle("hidden", p.id !== "tab-" + name));
  if (name === "samples") loadSamples();
  if (name === "console") drawLossChart();
}

// ------------------------------------------------------------------- init

function fillModelSelect(sel) {
  sel.innerHTML = Object.entries(S.MODELS)
    .map(([id, m]) => `<option value="${id}">${escapeHtml(m.label)}</option>`)
    .join("");
}

function fillVariantSelect() {
  const m = S.MODELS[$("new-job-model").value];
  $("new-job-variant").innerHTML = Object.entries(m.variants).map(([id, v]) => `<option value="${id}">${escapeHtml(v.label)}</option>`).join("");
}

async function startTraining() {
  try {
    if (isDirty()) await saveJob(true);
    const errors = S.validateJob(job).filter((i) => i.level === "error");
    if (errors.length) return showToast(errors.map((e) => e.msg).join("\n"), 6000);
    resetConsole();
    lossPoints = [];
    resetProgressLine();
    await api(`/api/jobs/${encodeURIComponent(currentJob)}/train/start`, { method: "POST" });
    switchTab("console");
  } catch (e) {
    showToast(e.message, 7000);
  }
}

function bindEvents() {
  document.querySelector("#job-editor").addEventListener("input", onBoundInput);
  document.querySelector("#job-editor").addEventListener("change", onBoundInput);
  document.querySelectorAll("#job-tabs .tab").forEach((t) => (t.onclick = () => switchTab(t.dataset.tab)));
  document.querySelectorAll("[data-close]").forEach((b) => (b.onclick = () => closeModal(b.dataset.close)));
  $("btn-confirm-ok").onclick = () => {
    closeModal("modal-confirm");
    if (confirmHandler) confirmHandler();
  };

  // jobs
  $("btn-new-job").onclick = () => {
    $("new-job-name").value = "";
    fillVariantSelect();
    openModal("modal-new-job");
    $("new-job-name").focus();
  };
  $("new-job-model").onchange = fillVariantSelect;
  $("btn-create-job").onclick = async () => {
    const name = $("new-job-name").value.trim();
    if (!name) return showToast("Enter a job name");
    try {
      const r = await api("/api/jobs", { method: "POST", body: { name, model: $("new-job-model").value, variant: $("new-job-variant").value } });
      closeModal("modal-new-job");
      await loadJobs();
      selectJob(r.name);
    } catch (e) {
      showToast(e.message);
    }
  };
  // a handler property returning false cancels the keystroke, so never return the comparison
  $("new-job-name").onkeydown = (e) => {
    if (e.key === "Enter") $("btn-create-job").click();
  };
  $("btn-save").onclick = () => saveJob().catch((e) => showToast(e.message));
  $("btn-discard").onclick = () => selectJob(currentJob);
  const openClone = () => {
    $("clone-job-name").value = currentJob + "_copy";
    $("clone-job-model").value = job.model;
    openModal("modal-clone-job");
  };
  $("btn-clone").onclick = openClone;
  $("btn-clone-model").onclick = openClone;
  $("btn-confirm-clone").onclick = async () => {
    try {
      if (isDirty()) await saveJob(true);
      const r = await api(`/api/jobs/${encodeURIComponent(currentJob)}/clone`, {
        method: "POST",
        body: { name: $("clone-job-name").value.trim(), model: $("clone-job-model").value },
      });
      closeModal("modal-clone-job");
      await loadJobs();
      selectJob(r.name);
    } catch (e) {
      showToast(e.message);
    }
  };
  $("btn-delete").onclick = () =>
    showConfirm("Delete job", `Delete "${currentJob}" and everything in its folder (outputs, samples, logs)?`, async () => {
      try {
        await api(`/api/jobs/${encodeURIComponent(currentJob)}`, { method: "DELETE" });
        currentJob = null;
        job = null;
        $("job-editor").classList.add("hidden");
        $("empty-state").classList.remove("hidden");
        loadJobs();
      } catch (e) {
        showToast(e.message);
      }
    }, "Delete");
  $("btn-run").onclick = startTraining;
  $("btn-stop").onclick = () =>
    showConfirm("Stop training", "Stop the running training process?", () =>
      api(`/api/jobs/${encodeURIComponent(currentJob)}/train/stop`, { method: "POST" }).catch((e) => showToast(e.message)), "Stop");

  // datasets
  $("btn-add-dataset").onclick = () => {
    job.dataset.folders.push(S.defaultFolder());
    renderDatasets();
    checkDirty();
  };
  $("btn-add-subfolders").onclick = () => {
    $("subfolder-list").innerHTML = "";
    $("subfolder-summary").textContent = "";
    openModal("modal-subfolders");
    $("subfolder-parent").focus();
  };
  $("btn-scan-subfolders").onclick = scanSubfolders;
  $("subfolder-parent").onkeydown = (e) => {
    if (e.key === "Enter") scanSubfolders();
  };
  $("chk-subfolders-all").onchange = (e) =>
    document.querySelectorAll("#subfolder-list .sf-check:not(:disabled)").forEach((c) => (c.checked = e.target.checked));
  $("btn-add-selected-subfolders").onclick = addSelectedSubfolders;
  $("btn-bulk-edit").onclick = openBulkEdit;
  $("btn-cap-more").onclick = renderCaptionRows;
  $("btn-cap-save-all").onclick = saveAllCaptions;
  $("btn-cap-template").onclick = () => {
    const t = $("cap-template").value;
    $("cap-list").querySelectorAll(".cap-row:not(.saved) textarea").forEach((ta) => {
      if (!ta.value.trim()) ta.value = t;
    });
  };
  // chips are redrawn while editing, so listen on the list
  $("dataset-list").addEventListener("click", (e) => {
    const el = e.target.closest(".ds-uncaptioned");
    if (!el) return;
    e.preventDefault();
    openCaptioner([el.dataset.dir]);
  });
  $("btn-apply-bulk").onclick = applyBulkEdit;
  $("btn-rescan").onclick = () => {
    scanCache.clear();
    scanDatasets();
  };
  $("btn-expand-all").onclick = () => {
    job.dataset.folders.forEach((f) => (f.collapsed = false));
    renderDatasets();
  };
  $("btn-collapse-all").onclick = () => {
    job.dataset.folders.forEach((f) => (f.collapsed = true));
    renderDatasets();
  };
  $("chk-select-all").onchange = (e) => {
    job.dataset.folders.filter(folderMatchesFilter).forEach((f) => (e.target.checked ? selectedFolders.add(f) : selectedFolders.delete(f)));
    renderDatasets();
  };
  $("dataset-filter").oninput = debounce((e) => {
    datasetFilter = e.target.value.trim().toLowerCase();
    renderDatasets();
  }, 150);
  document.querySelector('[data-bind="dataset.caption_ext"]').addEventListener("change", () => {
    scanCache.clear();
    scanDatasets();
  });

  // prompts
  $("btn-add-prompt").onclick = () => {
    job.sampling.prompts.push(newPrompt());
    renderPrompts();
    checkDirty();
  };
  $("btn-paste-prompts").onclick = () => {
    $("paste-prompts-text").value = "";
    openModal("modal-paste-prompts");
  };
  $("btn-confirm-paste").onclick = () => {
    const lines = $("paste-prompts-text").value.split("\n").map((l) => l.trim()).filter(Boolean);
    lines.forEach((l) => job.sampling.prompts.push(newPrompt(l)));
    closeModal("modal-paste-prompts");
    renderPrompts();
    checkDirty();
  };
  $("btn-apply-global").onclick = () => {
    const s = job.sampling;
    job.sampling.prompts.forEach((p) => Object.assign(p, { width: s.width, height: s.height, steps: s.steps, cfg: s.cfg, seed: s.seed }));
    renderPrompts();
    checkDirty();
  };

  // samples
  $("btn-refresh-samples").onclick = loadSamples;
  $("samples-limit").onchange = renderSamples;
  $("thumb-size").oninput = (e) => $("samples-grid").style.setProperty("--thumb", e.target.value + "px");
  $("btn-open-samples").onclick = async () => {
    const data = await api(`/api/jobs/${encodeURIComponent(currentJob)}/samples`);
    openFolder(data.dir);
  };
  $("btn-delete-samples").onclick = () =>
    showConfirm("Delete samples", `Delete ${selectedSamples.size} selected image(s)?`, async () => {
      await api(`/api/jobs/${encodeURIComponent(currentJob)}/samples`, { method: "DELETE", body: { rels: [...selectedSamples] } });
      selectedSamples.clear();
      loadSamples();
    }, "Delete");

  // console
  $("btn-clear-console").onclick = resetConsole;
  window.addEventListener("resize", debounce(drawLossChart, 150));

  // advanced
  $("btn-preview").onclick = async () => {
    try {
      const r = await api(`/api/jobs/${encodeURIComponent(currentJob)}/preview`, { method: "POST", body: { job } });
      const cmd = ["python", ...r.command.args].join(" ");
      const env = Object.entries(r.command.env).map(([k, v]) => `${k}=${k === "HF_TOKEN" ? "***" : v}`).join(" ");
      $("preview-output").innerHTML =
        `<div class="preview-file"><strong>Command</strong> <small class="hint">(cwd: ${escapeHtml(r.command.cwd || "trainer folder not set")})</small><pre>${escapeHtml((env ? env + " " : "") + cmd)}</pre></div>` +
        r.files.map((f) => `<div class="preview-file"><strong>${escapeHtml(f.name)}</strong> <small class="hint">${escapeHtml(f.path)}</small><pre>${escapeHtml(f.content)}</pre></div>`).join("");
    } catch (e) {
      $("preview-output").innerHTML = `<div class="issue error">${escapeHtml(e.message)}</div>`;
    }
  };

  // job settings
  $("btn-open-job-folder").onclick = () => openFolder(".", true);
  $("btn-reset-config").onclick = () =>
    showConfirm("Reset settings", "Reset training, network and sampling settings to this model's defaults? Dataset folders and prompts are kept.", () => {
      const fresh = S.defaultJob(job.model);
      fresh.variant = job.variant;
      fresh.dataset.folders = job.dataset.folders;
      fresh.sampling.prompts = job.sampling.prompts;
      fresh.sampling.negative = job.sampling.negative;
      fresh.training.output_name = job.training.output_name;
      job = fresh;
      populate();
    }, "Reset");

  // global settings
  $("btn-global-settings").onclick = openGlobalSettings;
  $("btn-save-global").onclick = () => saveGlobalSettings().catch((e) => showToast(e.message));
  $("btn-check-python").onclick = () => checkPython().catch((e) => showToast(e.message));

  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "s" && job) {
      e.preventDefault();
      if (isDirty()) saveJob().catch((err) => showToast(err.message));
    }
    if (e.key === "Escape") document.querySelectorAll(".modal:not(.hidden)").forEach((m) => m.classList.add("hidden"));
  });
  window.addEventListener("beforeunload", (e) => {
    if (isDirty()) e.preventDefault();
  });
}

async function init() {
  fillModelSelect($("new-job-model"));
  fillModelSelect($("clone-job-model"));
  buildGlobalSettings();
  bindEvents();
  connectWS();
  try {
    settings = await api("/api/settings");
    applyTheme(settings.theme);
  } catch (e) {
    /* server default */
  }
  await loadJobs();
  pollGPUs();
  setInterval(pollGPUs, 4000);
  let last = null;
  try {
    last = localStorage.getItem("last_job");
  } catch (e) {}
  if (last && jobs.some((j) => j.name === last)) selectJob(last);
  if (settings && jobs.length === 0) {
    showToast("First run: open ⚙️ Global Settings → Trainer and press 'Check environment', then set your model paths.", 8000);
  }
}

init();
