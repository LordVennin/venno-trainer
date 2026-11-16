/* Merge LoRAs dialog: rows of LoRA files and ratios, a rank report (how much of each LoRA a rank
   keeps, with the file size at that rank), and the merge itself. Both run in the background
   (python -m venno.merge on the CPU); the dialog polls /api/lora/task. Needs app.js. */

const mg = {
  rows: [{ path: "", ratio: 1 }, { path: "", ratio: 1 }],
  reports: {}, // path -> report of that file
  polling: false,
};

// MB with a decimal for small files
function mgMb(mb) {
  return mb < 10 ? mb.toFixed(1) : mb.toFixed(0);
}

function mgOthersRank() {
  return mg.rows.slice(1).reduce((n, r) => n + ((mg.reports[r.path] || {}).rank || 0), 0);
}

function mgOthersKnown() {
  return mg.rows.slice(1).every((r) => !r.path || mg.reports[r.path]);
}

// The first LoRA's rank in a concat merge: the result's rank minus the others' (null until known).
function mgFirstRank() {
  const rank = parseInt($("mg-rank").value, 10);
  if (!(rank > 0) || !mgOthersKnown()) return null;
  return rank - mgOthersRank();
}

function mgRenderRows() {
  $("mg-rows").innerHTML = mg.rows
    .map(
      (r, i) => `<div class="mg-row">
        <span class="mg-role">${i === 0 ? ($("mg-mode").value === "concat" ? "1st: compressed" : "1st") : i === 1 && $("mg-mode").value === "concat" ? "kept whole" : `${i + 1}${i === 1 ? "nd" : i === 2 ? "rd" : "th"}`}</span>
        <input type="text" class="mg-path" data-i="${i}" list="mg-files" value="${escapeHtml(r.path)}" placeholder="/path/to/lora.safetensors">
        <input type="number" class="mg-ratio" data-i="${i}" value="${escapeHtml(r.ratio)}" step="0.05" title="Strength of this LoRA in the result">
        <button class="btn-icon mg-del" data-i="${i}" title="Remove" ${mg.rows.length <= 2 ? "disabled" : ""}>✕</button>
      </div>`,
    )
    .join("");
  document.querySelectorAll(".mg-path").forEach((el) => {
    el.onchange = () => {
      mg.rows[+el.dataset.i].path = el.value.trim();
      if (+el.dataset.i === 0 && !$("mg-out").value.trim() && el.value.trim().endsWith(".safetensors")) {
        $("mg-out").value = el.value.trim().replace(/\.safetensors$/, "_merged.safetensors");
      }
      mgRenderHints();
      mgRenderReport();
    };
  });
  document.querySelectorAll(".mg-ratio").forEach((el) => (el.oninput = () => (mg.rows[+el.dataset.i].ratio = Number(el.value))));
  document.querySelectorAll(".mg-del").forEach((el) => {
    el.onclick = () => {
      mg.rows.splice(+el.dataset.i, 1);
      mgRenderRows();
      mgRenderHints();
    };
  });
}

function mgRenderHints() {
  const concat = $("mg-mode").value === "concat";
  $("mg-mode-hint").textContent = concat
    ? "Exactly the first LoRA (minus its weakest directions) plus the others, untouched: nothing of the others is lost."
    : "Exact when the rank holds all the inputs' ranks together; below that it keeps the strongest directions of the sum, which favours the strongest LoRA.";
  const first = mgFirstRank();
  let hint = "";
  if (concat) {
    if (first === null) hint = "Run the rank report to see how much of the first LoRA this rank keeps.";
    else if (first < 1) hint = `Too low: the others need ${mgOthersRank()}.`;
    else {
      const row = mgKeptAt(mg.rows[0].path, first);
      hint = `The first LoRA is kept at rank ${first}` + (row ? `: ${(row.kept * 100).toFixed(2)}% of its effect.` : ".");
    }
  } else {
    const all = mg.rows.reduce((n, r) => n + ((mg.reports[r.path] || {}).rank || 0), 0);
    if (mgOthersKnown() && mg.reports[mg.rows[0].path]) hint = `The inputs' ranks add up to ${all}: ${parseInt($("mg-rank").value, 10) >= all ? "the merge is exact" : "the sum is compressed"}.`;
  }
  $("mg-rank-hint").textContent = hint;
  mgRenderReport();
}

// Share of a file's effect kept at rank k, from its report (exact for reported ranks, else the
// next reported rank below, as a floor).
function mgKeptAt(path, k) {
  const r = mg.reports[path];
  if (!r) return null;
  const rows = r.rows.filter((row) => row.rank <= k);
  return rows.length ? rows[rows.length - 1] : null;
}

function mgRenderReport() {
  const box = $("mg-result");
  const paths = [...new Set(mg.rows.map((r) => r.path).filter((p) => mg.reports[p]))];
  if (!paths.length) {
    if (!box.querySelector(".mg-done")) box.innerHTML = "";
    return;
  }
  const first = $("mg-mode").value === "concat" ? mgFirstRank() : null;
  const done = box.querySelector(".mg-done");
  box.innerHTML =
    (done ? done.outerHTML : "") +
    `<div class="mg-report">` +
    paths
      .map((p) => {
        const r = mg.reports[p];
        const pick = p === mg.rows[0].path && first ? mgKeptAt(p, first) : null;
        return `<h5>${escapeHtml(p.split(/[\\/]/).pop())} · rank ${r.rank} · ${mgMb(r.size_mb)} MB</h5>
          <table class="mg-table"><tr><th>Rank</th><th>Effect kept</th><th>Weakest module</th><th>File size</th></tr>
          ${r.rows
            .map(
              (row) => `<tr class="${pick && row.rank === pick.rank ? "mg-pick" : ""}"><td>${row.rank}</td><td>${(row.kept * 100).toFixed(2)}%</td>
              <td title="${escapeHtml(row.worst_module)}">${(row.worst * 100).toFixed(2)}%</td><td>≈ ${mgMb((r.size_mb * row.rank) / r.rank)} MB</td></tr>`,
            )
            .join("")}</table>`;
      })
      .join("") +
    `<p class="hint">Effect kept: the share of the LoRA's change to the model that the rank keeps (squared singular values, over all its modules). Above about 99% the difference is rarely visible, but compare samples to be sure. File size scales with rank.</p></div>`;
}

async function mgPoll() {
  if (mg.polling) return;
  mg.polling = true;
  try {
    for (;;) {
      const t = await api("/api/lora/task");
      if (!t) break;
      const secs = Math.round(((t.finished || Date.now()) - t.started) / 1000);
      if (t.state === "running") {
        $("mg-status").textContent = `${t.kind === "report" ? "Measuring ranks" : "Merging"}… ${secs}s`;
        await new Promise((r) => setTimeout(r, 1500));
        continue;
      }
      $("mg-run").disabled = $("mg-report").disabled = false;
      if (t.state === "error") {
        $("mg-status").textContent = "";
        showToast(t.error, 9000);
      } else if (t.kind === "report") {
        for (const f of t.result.files) mg.reports[f.path] = f;
        const first = mg.reports[mg.rows[0].path];
        if (first && !$("mg-rank").value) $("mg-rank").value = first.rank;
        $("mg-status").textContent = `Rank report done in ${secs}s.`;
        mgRenderHints();
      } else {
        const r = t.result;
        $("mg-status").textContent = `Merged in ${secs}s.`;
        $("mg-result").innerHTML = `<div class="mg-done">Saved <code>${escapeHtml(r.out)}</code>: rank ${r.rank}, ${mgMb(r.size_mb)} MB.
          ${r.mode === "concat" ? `The first LoRA at rank ${r.first_rank} keeps ${(r.kept * 100).toFixed(2)}% of its effect (weakest module ${(r.worst * 100).toFixed(2)}%); the others are whole.` : `It keeps ${(r.kept * 100).toFixed(2)}% of the LoRAs' sum (weakest module ${(r.worst * 100).toFixed(2)}%).`}</div>`;
        mgRenderReport();
        loadLoraList();
      }
      break;
    }
  } catch (e) {
    showToast(e.message);
  } finally {
    mg.polling = false;
  }
}

async function mgStart(url, body) {
  try {
    await api(url, { method: "POST", body });
    $("mg-run").disabled = $("mg-report").disabled = true;
    mgPoll();
  } catch (e) {
    if (e.status === 409 && e.message === "exists") {
      showConfirm("Replace file", `${body.out} already exists. Replace it?`, () => mgStart(url, { ...body, overwrite: true }), "Replace");
    } else showToast(e.message, 8000);
  }
}

async function loadLoraList() {
  try {
    const files = await api("/api/loras");
    $("mg-files").innerHTML = files.map((f) => `<option value="${escapeHtml(f.path)}">${escapeHtml(f.job)} · ${escapeHtml(f.name)}</option>`).join("");
  } catch (e) {
    /* the lists are a convenience */
  }
}

(function bindMerge() {
  $("btn-merge").onclick = () => {
    openModal("modal-merge");
    mgRenderRows();
    mgRenderHints();
    loadLoraList();
    mgPoll(); // a merge started before the dialog was closed
  };
  $("mg-add").onclick = () => {
    mg.rows.push({ path: "", ratio: 1 });
    mgRenderRows();
  };
  $("mg-mode").onchange = () => {
    mgRenderRows();
    mgRenderHints();
  };
  $("mg-rank").oninput = mgRenderHints;
  $("mg-report").onclick = () => {
    const files = [...new Set(mg.rows.map((r) => r.path).filter(Boolean))].filter((p) => !mg.reports[p]);
    if (!files.length) return mg.rows.some((r) => r.path) ? showToast("Already measured") : showToast("Add a LoRA file first");
    mgStart("/api/lora/report", { files });
  };
  $("mg-run").onclick = () => {
    const loras = mg.rows.filter((r) => r.path);
    if (loras.length < 2) return showToast("Add at least two LoRA files");
    mgStart("/api/lora/merge", { loras, mode: $("mg-mode").value, rank: $("mg-rank").value, out: $("mg-out").value.trim(), dtype: $("mg-dtype").value });
  };
})();
