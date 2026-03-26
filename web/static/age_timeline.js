/* Life Monitoring — Age Timeline page */

let selectedFiles = [];   // { file: File, previewUrl: string }
let chartInstance  = null;

// ── Bootstrap ──────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  bindUpload();
  document.getElementById("runBtn").addEventListener("click", runBatch);
  document.getElementById("exportBtn").addEventListener("click", exportChart);
});

// ── File upload ────────────────────────────────────────────
function bindUpload() {
  const dropZone    = document.getElementById("dropZone");
  const fileInput   = document.getElementById("fileInput");
  const folderInput = document.getElementById("folderInput");

  document.getElementById("selectFilesBtn").addEventListener("click", e => {
    e.stopPropagation();
    fileInput.click();
  });
  document.getElementById("selectFolderBtn").addEventListener("click", e => {
    e.stopPropagation();
    folderInput.click();
  });

  fileInput.addEventListener("change",   () => addFiles(fileInput.files));
  folderInput.addEventListener("change", () => addFiles(folderInput.files));

  dropZone.addEventListener("dragover",  e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const items = e.dataTransfer.items;
    const files = [];
    if (items) {
      for (const item of items) {
        if (item.kind === "file") {
          const f = item.getAsFile();
          if (f && isImage(f)) files.push(f);
        }
      }
    } else {
      for (const f of e.dataTransfer.files) {
        if (isImage(f)) files.push(f);
      }
    }
    addFiles(files);
  });
}

function isImage(file) {
  return file.type.startsWith("image/") ||
         /\.(jpe?g|png|bmp|webp|tiff?)$/i.test(file.name);
}

function addFiles(fileList) {
  const existing = new Set(selectedFiles.map(f => f.file.name));
  const news = [];
  for (const f of fileList) {
    if (!isImage(f)) continue;
    if (existing.has(f.name)) continue;
    news.push(f);
    existing.add(f.name);
  }
  if (news.length === 0) return;

  // Generate preview URLs
  news.forEach(f => {
    selectedFiles.push({ file: f, previewUrl: URL.createObjectURL(f) });
  });

  // Natural sort
  selectedFiles.sort((a, b) => naturalCompare(a.file.name, b.file.name));

  renderFileList();
  document.getElementById("runBtn").disabled = selectedFiles.length === 0;
}

function renderFileList() {
  const list = document.getElementById("fileList");
  list.innerHTML = selectedFiles.map((item, i) => `
    <div class="file-item" id="fi-${i}">
      <img class="fi-thumb" src="${item.previewUrl}" alt="" />
      <span class="fi-name">${item.file.name}</span>
      <span class="fi-badge pending" id="badge-${i}">待处理</span>
    </div>
  `).join("");
}

// ── Run batch ──────────────────────────────────────────────
async function runBatch() {
  if (selectedFiles.length === 0) return;

  setRunning(true);

  const form = new FormData();
  selectedFiles.forEach(item => form.append("files", item.file, item.file.name));

  try {
    const data = await fetchForm("/api/faceage-batch", form);
    renderResults(data.results);
    showToast(`预测完成，共 ${data.results.length} 张`, "success");
  } catch (err) {
    showToast(`预测失败：${err.message}`, "error");
    setRunning(false);
  }
}

function setRunning(running) {
  const btn  = document.getElementById("runBtn");
  const icon = document.getElementById("runBtnIcon");
  const text = document.getElementById("runBtnText");
  btn.disabled      = running;
  icon.innerHTML    = running ? '<span class="spinner"></span>' : "▶";
  text.textContent  = running ? "预测中…" : "开始预测";

  if (running) {
    selectedFiles.forEach((_, i) => setBadge(i, "running", "预测中"));
  }
}

function setBadge(i, cls, label) {
  const el = document.getElementById(`badge-${i}`);
  if (el) { el.className = `fi-badge ${cls}`; el.textContent = label; }
}

// ── Render results ─────────────────────────────────────────
function renderResults(results) {
  setRunning(false);

  // Update badges
  results.forEach((r, i) => {
    const cls   = r.status === "success" ? "success" : "error";
    const label = r.status === "success" ? `${r.biological_age} 岁` : "失败";
    setBadge(i, cls, label);
  });

  renderChart(results);
  renderTable(results);

  document.getElementById("emptyState").style.display  = "none";
  document.getElementById("chartCard").style.display   = "";
  document.getElementById("tableCard").style.display   = "";
  document.getElementById("legendCard").style.display  = "";
}

// ── Parse real age from filename ───────────────────────────
// Supports "66-2008-11.png", "63-2005.png", "10-1952.png", etc.
// Returns the leading integer if present, otherwise null.
function parseRealAge(filename) {
  const stem = filename.replace(/\.[^.]+$/, "");
  const m = stem.match(/^(\d+)/);
  return m ? parseInt(m[1], 10) : null;
}

// ── Chart ──────────────────────────────────────────────────
function renderChart(results) {
  const labels    = results.map(r => shortName(r.filename));
  const bioAges   = results.map(r => r.status === "success" ? r.biological_age : null);
  const realAges  = results.map(r => parseRealAge(r.filename));
  const hasReal   = realAges.some(a => a !== null);

  const pointBg   = results.map(r => r.status === "success" ? "#2563EB" : "#EF4444");

  if (chartInstance) chartInstance.destroy();

  const datasets = [
    {
      label: "面部生物年龄",
      data: bioAges,
      borderColor: "#2563EB",
      backgroundColor: "rgba(37,99,235,.08)",
      borderWidth: 2.5,
      pointBackgroundColor: pointBg,
      pointBorderColor: "#fff",
      pointBorderWidth: 2,
      pointRadius: 6,
      pointHoverRadius: 9,
      fill: true,
      tension: 0.35,
      spanGaps: true,
      order: 1,
    },
  ];

  if (hasReal) {
    datasets.push({
      label: "真实年龄",
      data: realAges,
      borderColor: "#F59E0B",
      backgroundColor: "rgba(245,158,11,.06)",
      borderWidth: 2,
      borderDash: [6, 3],
      pointBackgroundColor: "#F59E0B",
      pointBorderColor: "#fff",
      pointBorderWidth: 2,
      pointRadius: 5,
      pointHoverRadius: 8,
      pointStyle: "rectRot",
      fill: false,
      tension: 0.35,
      spanGaps: true,
      order: 2,
    });
  }

  const ctx = document.getElementById("ageChart").getContext("2d");
  chartInstance = new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          display: hasReal,
          labels: { font: { size: 12 }, usePointStyle: true, padding: 16 },
        },
        tooltip: {
          callbacks: {
            title: ctx => results[ctx[0].dataIndex].filename,
            label: ctx => {
              const i = ctx.dataIndex;
              const r = results[i];
              if (ctx.datasetIndex === 0) {
                if (r.status !== "success") return "  生物年龄：❌ 未检测到人脸";
                const diff = hasReal && realAges[i] != null
                  ? ` (偏差 ${(r.biological_age - realAges[i]).toFixed(1)} 岁)`
                  : "";
                return [
                  `  生物年龄：${r.biological_age} 岁${diff}`,
                  `  置信度：${(r.detection_confidence * 100).toFixed(1)}%`,
                ];
              } else {
                return realAges[i] != null
                  ? `  真实年龄：${realAges[i]} 岁`
                  : "  真实年龄：未知";
              }
            },
          },
          backgroundColor: "#1E293B",
          titleFont: { size: 12 },
          bodyFont:  { size: 12 },
          padding: 10,
        },
      },
      scales: {
        x: {
          grid: { color: "rgba(0,0,0,.04)" },
          ticks: { font: { size: 12 }, maxRotation: 35 },
        },
        y: {
          title: { display: true, text: "年龄（岁）", font: { size: 12 } },
          grid:  { color: "rgba(0,0,0,.04)" },
          ticks: { font: { size: 12 } },
          suggestedMin: 0,
        },
      },
    },
  });
}

// ── Table ──────────────────────────────────────────────────
function renderTable(results) {
  const hasReal = results.some(r => parseRealAge(r.filename) !== null);

  // Update table header
  const thead = document.querySelector("#resultsTable thead tr");
  thead.innerHTML = [
    "<th>#</th>",
    "<th>文件名</th>",
    hasReal ? "<th>真实年龄</th>" : "",
    "<th>生物年龄（岁）</th>",
    hasReal ? "<th>偏差</th>" : "",
    "<th>置信度</th>",
    "<th>状态</th>",
  ].join("");

  const tbody = document.getElementById("tableBody");
  tbody.innerHTML = results.map((r, i) => {
    const realAge = parseRealAge(r.filename);
    const bioAge  = r.status === "success" ? r.biological_age : null;

    const realCell = hasReal
      ? `<td style="font-weight:700;color:#F59E0B">${realAge != null ? realAge : "—"}</td>`
      : "";

    const ageCell = bioAge != null
      ? `<span class="age-cell">${bioAge}</span>`
      : `<span class="age-cell error">—</span>`;

    let diffCell = "";
    if (hasReal) {
      if (bioAge != null && realAge != null) {
        const diff = bioAge - realAge;
        const sign = diff >= 0 ? "+" : "";
        const color = Math.abs(diff) <= 5 ? "#10B981"
                    : Math.abs(diff) <= 10 ? "#F59E0B"
                    : "#EF4444";
        diffCell = `<td style="font-weight:700;color:${color}">${sign}${diff.toFixed(1)}</td>`;
      } else {
        diffCell = `<td style="color:var(--muted)">—</td>`;
      }
    }

    const confCell = r.status === "success"
      ? `<div class="conf-bar">
           <div class="conf-track">
             <div class="conf-fill" style="width:${(r.detection_confidence * 100).toFixed(1)}%"></div>
           </div>
           <span style="font-size:11px;color:var(--text-2);width:36px;text-align:right">
             ${(r.detection_confidence * 100).toFixed(1)}%
           </span>
         </div>`
      : `<span style="color:var(--muted)">—</span>`;

    const statusPill = r.status === "success"
      ? `<span class="status-pill success">✓ 成功</span>`
      : `<span class="status-pill error" title="${r.error || ""}">✗ 失败</span>`;

    return `
      <tr>
        <td style="color:var(--muted)">${i + 1}</td>
        <td style="max-width:220px;word-break:break-all">${r.filename}</td>
        ${realCell}
        <td>${ageCell}</td>
        ${diffCell}
        <td>${confCell}</td>
        <td>${statusPill}</td>
      </tr>
    `;
  }).join("");
}

// ── Export chart ───────────────────────────────────────────
function exportChart() {
  if (!chartInstance) return;
  const link = document.createElement("a");
  link.download = "faceage_timeline.png";
  link.href = chartInstance.toBase64Image("image/png", 1);
  link.click();
}

// ── Utilities ──────────────────────────────────────────────
function shortName(filename) {
  // Strip extension for display
  return filename.replace(/\.[^.]+$/, "");
}

function naturalCompare(a, b) {
  const re = /(\d+)/g;
  const partsA = a.split(re);
  const partsB = b.split(re);
  for (let i = 0; i < Math.max(partsA.length, partsB.length); i++) {
    const pa = partsA[i] ?? "";
    const pb = partsB[i] ?? "";
    const na = parseInt(pa, 10), nb = parseInt(pb, 10);
    if (!isNaN(na) && !isNaN(nb) && na !== nb) return na - nb;
    if (pa.toLowerCase() !== pb.toLowerCase()) return pa.toLowerCase() < pb.toLowerCase() ? -1 : 1;
  }
  return 0;
}

async function fetchForm(url, form) {
  const res = await fetch(url, { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`HTTP ${res.status}: ${body.slice(0, 200)}`);
  }
  return res.json();
}

let toastTimer;
function showToast(msg, type = "") {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = `toast show ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 3500);
}
