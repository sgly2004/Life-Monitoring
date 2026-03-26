/* Life Monitoring — 综合分析页前端逻辑 */

const API = {
  modules:  "/api/modules",
  analyze:  "/api/analyze",
  template: "/api/config/template",
};

// ── State ──────────────────────────────────────────────────
let modulesMeta = [];
let enabledSet  = new Set();
let lastResults = {};

// ── Bootstrap ──────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  await loadModules();
  bindFileUploads();
  bindRunButton();
  bindPromptActions();
  bindTemplateModal();
});

// ── Module grid ────────────────────────────────────────────
async function loadModules() {
  try {
    modulesMeta = await fetchJson(API.modules);
  } catch {
    modulesMeta = fallbackModules();
  }

  const grid = document.getElementById("modulesGrid");
  grid.innerHTML = "";

  modulesMeta.forEach(mod => {
    if (mod.enabled_default !== false) enabledSet.add(mod.id);

    const card = document.createElement("div");
    card.className = `module-card ${enabledSet.has(mod.id) ? "enabled" : "disabled"}`;
    card.id = `module-card-${mod.id}`;

    const inputItems  = (mod.inputs  || []).map(i => `<div class="io-item">${i.name}</div>`).join("");
    const outputItems = (mod.outputs || []).map(o => `<div class="io-item">${o.name}${o.unit ? " ("+o.unit+")" : ""}</div>`).join("");

    card.innerHTML = `
      <div class="module-header">
        <div>
          <div class="module-name">${mod.name}</div>
          <div class="module-name-en">${mod.name_en || ""}</div>
        </div>
        <label class="toggle" title="开启/关闭此模块">
          <input type="checkbox" ${enabledSet.has(mod.id) ? "checked" : ""}
                 data-module-id="${mod.id}" />
          <span class="toggle-slider"></span>
        </label>
      </div>
      <div class="module-desc">${mod.description || ""}</div>
      <div class="module-io">
        <div class="io-block">
          <div class="io-label">输入</div>
          ${inputItems || '<div class="io-item" style="color:var(--muted)">无</div>'}
        </div>
        <div class="io-block">
          <div class="io-label">输出</div>
          ${outputItems || '<div class="io-item" style="color:var(--muted)">无</div>'}
        </div>
      </div>
      <div class="module-status status-idle" id="status-${mod.id}">
        <span class="status-dot"></span>
        <span>待机</span>
      </div>
    `;

    const checkbox = card.querySelector(`input[data-module-id="${mod.id}"]`);
    checkbox.addEventListener("change", () => toggleModule(mod.id, checkbox.checked));

    grid.appendChild(card);
  });
}

function toggleModule(id, enabled) {
  enabled ? enabledSet.add(id) : enabledSet.delete(id);
  const card = document.getElementById(`module-card-${id}`);
  if (card) {
    card.classList.toggle("enabled",  enabled);
    card.classList.toggle("disabled", !enabled);
  }
}

// ── File uploads ───────────────────────────────────────────
function bindFileUploads() {
  setupUploadZone("imageZone", "imageInput", "imageFileName", (file) => {
    const preview = document.getElementById("imagePreview");
    const reader  = new FileReader();
    reader.onload = e => { preview.src = e.target.result; preview.style.display = "block"; };
    reader.readAsDataURL(file);
  });
  setupUploadZone("audioZone", "audioInput", "audioFileName");
}

function setupUploadZone(zoneId, inputId, fileNameId, onFile) {
  const zone  = document.getElementById(zoneId);
  const input = document.getElementById(inputId);
  const label = document.getElementById(fileNameId);
  if (!zone || !input || !label) return;

  const handleFile = (file) => {
    if (!file) return;
    label.textContent = `✓ ${file.name}`;
    onFile && onFile(file);
  };

  input.addEventListener("change", () => handleFile(input.files[0]));
  zone.addEventListener("dragover",  e => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) {
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      handleFile(file);
    }
  });
}

// ── Run analysis ───────────────────────────────────────────
function bindRunButton() {
  document.getElementById("runBtn").addEventListener("click", runAnalysis);
}

async function runAnalysis() {
  const age   = parseInt(document.getElementById("ageInput").value, 10);
  const image = document.getElementById("imageInput").files[0];
  const audio = document.getElementById("audioInput").files[0];
  const mods  = [...enabledSet];

  if (mods.length === 0) { showToast("请至少启用一个模块", "error"); return; }
  if (isNaN(age) || age < 0 || age > 120) { showToast("请输入有效年龄（0-120）", "error"); return; }

  setRunning(true, mods);

  const form = new FormData();
  form.append("age", age);
  form.append("enabled_modules", JSON.stringify(mods));
  if (image) form.append("image", image);
  if (audio) form.append("audio", audio);

  try {
    const data = await fetchForm(API.analyze, form);
    lastResults = data;
    renderResults(data);
    showToast("分析完成", "success");
  } catch (err) {
    showToast(`分析失败：${err.message}`, "error");
  } finally {
    setRunning(false, mods);
  }
}

function setRunning(running, mods) {
  const btn   = document.getElementById("runBtn");
  const icon  = document.getElementById("runBtnIcon");
  const label = document.getElementById("runBtnText");
  btn.disabled      = running;
  icon.innerHTML    = running ? '<span class="spinner"></span>' : "▶";
  label.textContent = running ? "分析中…" : "开始分析";
  mods.forEach(mid => setModuleStatus(mid, running ? "running" : "idle"));
}

function setModuleStatus(id, status, text) {
  const el = document.getElementById(`status-${id}`);
  if (!el) return;
  const labels = { idle: "待机", running: "运行中", success: "成功", error: "出错" };
  el.className = `module-status status-${status}`;
  el.innerHTML = `
    <span class="status-dot ${status === "running" ? "pulse" : ""}"></span>
    <span>${text || labels[status] || status}</span>
  `;
}

// ── Render results ─────────────────────────────────────────
function renderResults(data) {
  const section = document.getElementById("resultsSection");
  const grid    = document.getElementById("resultGrid");
  section.classList.add("visible");
  grid.innerHTML = "";

  const results = data.results || {};
  for (const [mid, r] of Object.entries(results)) {
    setModuleStatus(mid, r.status === "success" ? "success" : "error");

    const card = document.createElement("div");
    card.className = `result-card ${r.status}`;

    const outputsHtml = r.status === "success"
      ? Object.entries(r.outputs || {})
          .filter(([, v]) => typeof v !== "object")
          .map(([k, v]) => `
            <div class="output-row">
              <span class="output-key">${formatKey(k)}</span>
              <span class="output-value">${formatValue(v)}</span>
            </div>
          `).join("")
      : "";

    const errorHtml = r.error
      ? `<div class="result-error">⚠ ${r.error}</div>`
      : "";

    card.innerHTML = `
      <div class="result-card-title">
        <span>${r.status === "success" ? "✅" : "❌"}</span>
        ${r.module_name}
      </div>
      <div class="result-summary">${r.summary}</div>
      <div class="result-outputs">${outputsHtml}</div>
      ${errorHtml}
    `;
    grid.appendChild(card);
  }

  // Fill prompt textarea (always visible, directly editable)
  const textarea = document.getElementById("promptTextarea");
  textarea.value = data.prompt || "";
  textarea.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function formatKey(key) {
  const map = {
    time_to_death_years:  "剩余寿命",
    life_expectancy_age:  "预期寿命（岁）",
    model_used:           "使用模型",
    biological_age:       "生物年龄（岁）",
    detection_confidence: "检测置信度",
    diagnosis:            "诊断结论",
    prob_parkinson:       "帕金森概率",
    prob_healthy:         "正常概率",
    risk_level:           "风险等级",
    final_verdict:        "最终判断",
    votes_positive:       "阳性模型数",
    total_models:         "模型总数",
    segments_analyzed:    "分析片段数",
    label:                "标签",
    is_positive:          "是否阳性",
  };
  return map[key] || key.replace(/_/g, " ");
}

function formatValue(v) {
  if (typeof v === "number") return Number.isInteger(v) ? v : parseFloat(v.toFixed(3));
  if (typeof v === "boolean") return v ? "是" : "否";
  return v;
}

// ── Prompt actions ─────────────────────────────────────────
function bindPromptActions() {
  document.getElementById("copyPromptBtn").addEventListener("click", async () => {
    const text = document.getElementById("promptTextarea").value;
    if (!text) { showToast("提示词为空", "error"); return; }
    await navigator.clipboard.writeText(text);
    const btn = document.getElementById("copyPromptBtn");
    btn.classList.add("copied");
    btn.textContent = "✅ 已复制";
    setTimeout(() => {
      btn.classList.remove("copied");
      btn.innerHTML = "📋 复制提示词";
    }, 2000);
  });
}

// ── Template modal ─────────────────────────────────────────
function bindTemplateModal() {
  const modal    = document.getElementById("templateModal");
  const area     = document.getElementById("templateEditorArea");
  const openBtn  = document.getElementById("editTemplateBtn");
  const closeBtn = document.getElementById("closeModalBtn");
  const cancelBtn= document.getElementById("cancelModalBtn");
  const saveBtn  = document.getElementById("saveTemplateBtn");

  openBtn.addEventListener("click", async () => {
    try {
      const { template } = await fetchJson(API.template);
      area.value = template;
    } catch { area.value = ""; }
    modal.classList.add("open");
  });

  const close = () => modal.classList.remove("open");
  closeBtn.addEventListener("click", close);
  cancelBtn.addEventListener("click", close);
  modal.addEventListener("click", e => { if (e.target === modal) close(); });

  saveBtn.addEventListener("click", async () => {
    try {
      await fetchJson(API.template, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ template: area.value }),
      });
      showToast("模板已保存", "success");
      close();
    } catch (err) {
      showToast(`保存失败：${err.message}`, "error");
    }
  });
}

// ── Utilities ──────────────────────────────────────────────
async function fetchJson(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`HTTP ${res.status}: ${body.slice(0, 200)}`);
  }
  return res.json();
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
  toastTimer = setTimeout(() => el.classList.remove("show"), 3000);
}

function fallbackModules() {
  return [
    { id: "faceage",     name: "面部生物年龄预测", name_en: "FaceAge",     description: "从面部照片估计生物年龄。",    enabled_default: true, inputs: [{name:"面部图片"}], outputs: [{name:"生物年龄", unit:"岁"}] },
    { id: "facettd",     name: "面部死亡时间预测", name_en: "FaceTTD",     description: "预测剩余寿命和预期寿命。",    enabled_default: true, inputs: [{name:"面部图片"},{name:"年龄"}], outputs: [{name:"剩余寿命",unit:"年"},{name:"预期寿命",unit:"岁"}] },
    { id: "parkinsons",  name: "帕金森声学检测",   name_en: "Parkinsons",  description: "声学特征检测帕金森风险。",    enabled_default: true, inputs: [{name:"语音录音"}], outputs: [{name:"诊断结果"},{name:"帕金森概率"}] },
    { id: "lung_cancer", name: "早期肺癌语音检测", name_en: "Lung Cancer", description: "集成模型检测早期肺癌信号。", enabled_default: true, inputs: [{name:"语音录音"}], outputs: [{name:"最终判断"},{name:"阳性模型数"}] },
  ];
}
