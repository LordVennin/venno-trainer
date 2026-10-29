/* Auto-Caption (JoyCaption) in the Tagger view: caption the selected image
   (shown, not saved), preview a random sample, and caption many images with
   each image's tags as hints. Uses tagger.js state (tg) and app.js helpers. */
"use strict";

const CAP = window.VennoCaptions;
const CAP_DEFAULTS = { template: CAP.DEFAULT_TEMPLATE, fallback_template: CAP.DEFAULT_FALLBACK, strip: CAP.DEFAULT_STRIP.join("\n") };

function capCfg() {
  return tg.cfg.caption;
}

// Caption settings sent with a request; the never-add list and underscore
// handling are the Auto-Tag tab's.
function captionOptions() {
  return { ...capCfg(), exclude: tg.cfg.exclude, underscores: tg.cfg.underscores };
}

function captionScopeItems() {
  return $("tg-cap-scope").value === "filtered" ? tg.filtered : tg.items;
}

// ------------------------------------------------------------- settings

function bindCaptionSettings() {
  const c = capCfg();
  document.querySelectorAll("#tg-pane-caption [data-c]").forEach((el) => {
    const v = c[el.dataset.c];
    if (el.type === "checkbox") el.checked = !!v;
    else el.value = v ?? "";
    el.addEventListener(el.type === "checkbox" || el.tagName === "SELECT" ? "change" : "input", () => {
      c[el.dataset.c] = el.type === "checkbox" ? el.checked : el.type === "number" ? Number(el.value) : el.value;
      if (el.dataset.c === "hint_source") renderCaptionFields();
      saveTaggerSettings();
    });
  });
  document.querySelectorAll("#tg-pane-caption [data-reset]").forEach((b) => {
    b.onclick = () => {
      const k = b.dataset.reset;
      c[k] = CAP_DEFAULTS[k];
      document.querySelector(`#tg-pane-caption [data-c="${k}"]`).value = c[k];
      saveTaggerSettings();
    };
  });
}

function renderCaptionFields() {
  if (!tg.cfg) return;
  const c = capCfg();
  const fromTagger = c.hint_source === "tagger";
  $("tg-cap-cats-row").classList.toggle("hidden", !fromTagger);
  $("tg-cap-source-hint").textContent = fromTagger
    ? `${modelLabel()} (Auto-Tag tab settings) tags every image first; the hints are not saved.`
    : "Comma- or space-separated tags; a kept .tags backup is used instead once a caption was converted.";
  const kind = modelKind();
  const cats = (tg.categories[kind] || []).filter((x) => x !== "rating");
  const inc = c.hint_include || (c.hint_include = {});
  $("tg-cap-cats").innerHTML = cats
    .map((cat) => {
      const on = inc[cat] !== undefined ? inc[cat] : tg.cfg.include[cat] !== false;
      return `<label class="hint inline"><input type="checkbox" data-hcat="${cat}" ${on ? "checked" : ""}> ${cat}</label>`;
    })
    .join("");
  $("tg-cap-cats").querySelectorAll("[data-hcat]").forEach((el) => {
    el.onchange = () => {
      inc[el.dataset.hcat] = el.checked;
      saveTaggerSettings();
    };
  });
  const m = tg.captioners.find((x) => x.id === c.model);
  $("tg-cap-model-hint").textContent = m ? (m.downloaded ? `${m.label}: downloaded.` : `${m.label}: downloads ~${m.gb} GB into the models folder on first use.`) : "";
  renderCaptionScope();
}

function renderCaptionScope() {
  if (!tg.cfg || !$("tg-cap-scope")) return;
  const sel = $("tg-cap-scope");
  const prev = sel.value;
  const opts = [`<option value="all">All loaded images (${fmt(tg.items.length)})</option>`];
  if (filterActive()) opts.push(`<option value="filtered">Images shown by the filter (${fmt(tg.filtered.length)})</option>`);
  sel.innerHTML = opts.join("");
  if ([...sel.options].some((o) => o.value === prev)) sel.value = prev;
  const items = captionScopeItems();
  const n = capCfg().mode === "skip" ? items.filter((i) => i.caption == null).length : items.length;
  const run = tg.state && tg.state.caption_run;
  const busy = tg.state && ((run && run.state === "running") || (tg.state.run && tg.state.run.state === "running"));
  $("tg-cap-run").textContent = `▶ Caption ${capCfg().skip_converted ? "up to " : ""}${fmt(n)} image(s)`;
  $("tg-cap-run").disabled = !n || busy;
  $("tg-cap-preview").textContent = `Preview ${fmt(Math.min(items.length, capCfg().preview_n || 30))}`;
  $("tg-cap-preview").disabled = !items.length || busy;
}

// ----------------------------------------------------------- one image

async function captionThisImage() {
  const it = tg.filtered[tg.sel];
  const btn = $("tg-joycaption");
  if (!it || !btn || btn.disabled) return;
  btn.disabled = true;
  btn.textContent = "Captioning…";
  const c = captionOptions();
  const body = {
    dir: it.dir, name: it.name, caption_ext: tg.ext, options: c, tagger_options: tagOptions(), model: tg.cfg.model, device: tg.cfg.device,
    caption_text: $("tg-caption") ? $("tg-caption").value : it.caption,
  };
  // the tagger already ran on this image: use its tags instead of swapping models twice
  if (c.hint_source === "tagger" && tg.preds && tg.preds.key === tgKey(it) && tg.preds.model === tg.cfg.model) {
    body.tags = CAP.hintsFromPredictions(tg.preds.preds, tagOptions(), { ...c, include: c.hint_include });
  }
  $("tg-joy").innerHTML = '<p class="hint">Running JoyCaption… the first run loads (and if needed downloads) the model; any loaded tagger is unloaded first.</p>';
  try {
    const r = await api("/api/tagger/joycaption", { method: "POST", body });
    tg.joy = { key: tgKey(it), ...r };
    if (tg.filtered[tg.sel] === it) renderJoyResult();
  } catch (e) {
    $("tg-joy").innerHTML = `<div class="issue error">${escapeHtml(e.message)}</div>`;
  } finally {
    if ($("tg-joycaption")) {
      $("tg-joycaption").disabled = false;
      $("tg-joycaption").textContent = "📝 Caption this image (JoyCaption)";
    }
  }
}

function renderJoyResult() {
  const j = tg.joy;
  const box = $("tg-joy");
  if (!box || !j) return;
  box.innerHTML = `
    <div class="tg-joy">
      <div class="tg-preds-head"><span class="hint">JoyCaption on ${escapeHtml(j.device)} · ${escapeHtml(j.length)} · not saved</span></div>
      <div class="tg-joy-text">${escapeHtml(j.caption)}</div>
      <details><summary class="hint">Prompt used (${j.tags.length} tag hints)</summary><pre class="tg-joy-prompt">${escapeHtml(j.prompt)}</pre></details>
      <div class="inline" style="margin-top:6px;">
        <button class="btn btn-sm btn-primary" id="tg-joy-use">Use as caption</button>
        <span class="hint">${capCfg().layout === "tags_caption" ? "Keeps the tags on line 1 and puts this caption on line 2" : "Replaces the text above"}; then Save. Your trigger comes from the dataset card's caption prefix.</span>
      </div>
    </div>`;
  $("tg-joy-use").onclick = () => {
    const ta = $("tg-caption");
    ta.value = CAP.composeCaption(j.caption, capCfg().layout, ta.value, j.tags);
    ta.dispatchEvent(new Event("input"));
  };
}

// ------------------------------------------------------------ bulk runs

function startCaptionRun(preview) {
  const items = captionScopeItems();
  const c = captionOptions();
  if (!items.length) return showToast("No images loaded");
  const go = async () => {
    try {
      await api("/api/tagger/caption-run", {
        method: "POST",
        body: {
          items: items.map(({ dir, name }) => ({ dir, name })), caption_ext: tg.ext, options: c, tagger_options: tagOptions(),
          tagger: { model: tg.cfg.model, device: tg.cfg.device, batch_size: tg.cfg.batch_size },
          mode: c.mode, layout: c.layout, backup: c.backup, skip_converted: c.skip_converted, preview: preview ? c.preview_n || 30 : 0,
        },
      });
      // a quick run may finish (and broadcast "done") before this answer arrives
      refreshTaggerState();
    } catch (e) {
      showToast(e.message, 9000);
    }
  };
  if (!preview && c.mode === "replace" && !c.backup && c.layout !== "tags_caption") {
    showConfirm("Replace captions", `Replace the captions of up to ${fmt(items.length)} image(s) without keeping the original tags in .tags files? You can still undo the run.`, go, "Replace");
  } else go();
}

function renderCaptionState(data) {
  if (!$("tg-cap-service") || !tg.cfg) return;
  const svc = data.service;
  const isJoy = svc.loaded && tg.captioners.some((x) => x.id === svc.loaded.model);
  let note = svc.loaded
    ? `Loaded: <b>${escapeHtml(modelLabel(svc.loaded.model))}</b>${isJoy && svc.loaded.quant ? ` (${escapeHtml(svc.loaded.quant)})` : ""} on ${escapeHtml(svc.loaded.device)}.`
    : "No model loaded.";
  if (svc.status && !/^Ready/.test(svc.status)) note += ` ${escapeHtml(svc.status)}`;
  if (data.training && capCfg().device !== "cuda") note += ' <span style="color:var(--warning)">Training is running: JoyCaption will not start until it ends (or set Device to “GPU, even while training runs”).</span>';
  $("tg-cap-service").innerHTML = note;

  const run = data.caption_run;
  $("tg-cap-progress").classList.toggle("hidden", !run);
  const runningNow = run && run.state === "running";
  $("tg-cap-cancel").classList.toggle("hidden", !runningNow);
  if (run) {
    const hints = run.phase === "hints";
    const done = hints ? run.hint_done : run.done;
    const total = hints ? run.hint_total : run.total;
    $("tg-cap-prog-fill").style.width = (total ? (done / total) * 100 : 0) + "%";
    const verb = runningNow ? (hints ? "Tagging for hints" : run.phase === "captions" ? (run.preview ? "Previewing" : "Captioning") : "Preparing") : { done: "Done", cancelled: "Stopped", error: "Failed" }[run.state];
    $("tg-cap-prog-text").textContent =
      `${verb}: ${fmt(done)} / ${fmt(total)}` + (run.preview ? "" : ` · ${fmt(run.written)} written`) + ` · ${fmt(run.skipped)} skipped` +
      (run.error_count ? ` · ${fmt(run.error_count)} errors` : "") + (run.device ? ` · ${run.device}` : "");
    const secs = ((run.finished || Date.now()) - run.phase_started) / 1000;
    $("tg-cap-prog-eta").textContent = runningNow && done ? `ETA ${fmtDuration((secs / done) * (total - done))}` : fmtDuration(((run.finished || Date.now()) - run.started) / 1000);
    $("tg-cap-prog-errors").innerHTML = (run.error ? `<div class="issue error">${escapeHtml(run.error)}</div>` : "") + run.errors.map((e) => `<div>${escapeHtml(e)}</div>`).join("");
    renderPreviewResults(run);
  }
  const prev = tg.lastCaptionState;
  tg.lastCaptionState = run ? run.state : null;
  if (prev === "running" && run && run.state !== "running") {
    if (!run.preview && tg.dirs.length) {
      loadTaggerFolders(tg.dirs, true);
      refreshUndo();
    }
    loadTaggerModels();
  }
  renderCaptionScope();
}

function renderPreviewResults(run) {
  const box = $("tg-cap-results");
  if (!run.preview) return (box.innerHTML = "");
  if (box.dataset.count === String(run.results.length) && box.dataset.started === String(run.started)) return;
  box.dataset.count = run.results.length;
  box.dataset.started = run.started;
  box.innerHTML = run.results.length
    ? `<div class="tg-cap-table"><div class="tg-cap-row tg-tag-headrow"><span>Image</span><span>Tag hints</span><span>Prompt</span><span>${run.layout === "tags_caption" ? "Caption file (tags, then caption)" : "Caption"}</span></div>` +
      run.results
        .map(
          (r) => `<div class="tg-cap-row">
        <span><img loading="lazy" src="${tgThumb(r)}" alt="" title="${escapeHtml(r.name)}"><div class="hint">${escapeHtml(r.name)}</div></span>
        <span class="hint">${escapeHtml(r.tags.join(", ")) || "<i>none: fallback template</i>"}</span>
        <span class="hint tg-cap-prompt">${escapeHtml(r.prompt)}</span>
        <span class="tg-cap-text">${escapeHtml(r.text || r.caption)}</span></div>`,
        )
        .join("") +
      "</div>"
    : "";
}

(function bindCaptioner() {
  $("tg-cap-run").onclick = () => startCaptionRun(false);
  $("tg-cap-preview").onclick = () => startCaptionRun(true);
  $("tg-cap-scope").onchange = renderCaptionScope;
  $("tg-cap-cancel").onclick = () => api("/api/tagger/cancel", { method: "POST" }).catch((e) => showToast(e.message));
})();
