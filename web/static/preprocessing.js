/* Life Monitoring — 数据预处理页前端逻辑 */

const API = {
  faceSplit:  "/api/face-split",
  voiceSplit: "/api/voice-split",
};

// ── Bootstrap ──────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  bindFaceSplitInputs();
  bindVoiceSplitInputs();
});

// ── Upload zone helper ──────────────────────────────────────
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

// ── Face Split ─────────────────────────────────────────────
function bindFaceSplitInputs() {
  setupUploadZone("refFaceZone", "refFaceInput", "refFaceFileName", (file) => {
    const preview = document.getElementById("refFacePreview");
    const reader  = new FileReader();
    reader.onload = e => { preview.src = e.target.result; preview.style.display = "block"; };
    reader.readAsDataURL(file);
  });
  setupUploadZone("splitVideoZone", "splitVideoInput", "splitVideoFileName");

  const slider = document.getElementById("faceThreshold");
  const valEl  = document.getElementById("faceThresholdVal");
  slider.addEventListener("input", () => {
    valEl.textContent = parseFloat(slider.value).toFixed(2);
  });

  document.getElementById("runFaceSplitBtn").addEventListener("click", runFaceSplit);
}

async function runFaceSplit() {
  const refFace   = document.getElementById("refFaceInput").files[0];
  const video     = document.getElementById("splitVideoInput").files[0];
  const threshold = parseFloat(document.getElementById("faceThreshold").value);

  if (!refFace) { showToast("请上传目标人脸参考照片", "error"); return; }
  if (!video)   { showToast("请上传完整视频", "error"); return; }

  setSplitRunning("face", true);
  const form = new FormData();
  form.append("reference_image", refFace);
  form.append("video", video);
  form.append("threshold", threshold);

  try {
    const data = await fetchForm(API.faceSplit, form);
    renderFaceSplitResults(data);
    showToast("人脸切分完成", "success");
  } catch (err) {
    showToast(`人脸切分失败：${err.message}`, "error");
  } finally {
    setSplitRunning("face", false);
  }
}

function renderFaceSplitResults(data) {
  document.getElementById("faceSplitEmpty").style.display = "none";
  const container = document.getElementById("faceSplitResults");
  container.style.display = "block";
  container.innerHTML = buildSplitResultsHtml(data);
}

// ── Voice Split ─────────────────────────────────────────────
function bindVoiceSplitInputs() {
  setupUploadZone("refAudioZone", "refAudioInput", "refAudioFileName");
  setupUploadZone("splitAudioZone", "splitAudioInput", "splitAudioFileName");

  const slider = document.getElementById("voiceThreshold");
  const valEl  = document.getElementById("voiceThresholdVal");
  slider.addEventListener("input", () => {
    valEl.textContent = parseFloat(slider.value).toFixed(2);
  });

  document.getElementById("runVoiceSplitBtn").addEventListener("click", runVoiceSplit);
}

async function runVoiceSplit() {
  const refAudio  = document.getElementById("refAudioInput").files[0];
  const audio     = document.getElementById("splitAudioInput").files[0];
  const threshold = parseFloat(document.getElementById("voiceThreshold").value);

  if (!refAudio) { showToast("请上传目标说话人语音", "error"); return; }
  if (!audio)    { showToast("请上传完整音频", "error"); return; }

  setSplitRunning("voice", true);
  const form = new FormData();
  form.append("reference_audio", refAudio);
  form.append("audio", audio);
  form.append("threshold", threshold);

  try {
    const data = await fetchForm(API.voiceSplit, form);
    renderVoiceSplitResults(data);
    showToast("声纹切分完成", "success");
  } catch (err) {
    showToast(`声纹切分失败：${err.message}`, "error");
  } finally {
    setSplitRunning("voice", false);
  }
}

function renderVoiceSplitResults(data) {
  document.getElementById("voiceSplitEmpty").style.display = "none";
  const container = document.getElementById("voiceSplitResults");
  container.style.display = "block";
  container.innerHTML = buildSplitResultsHtml(data);
}

// ── Shared result renderer ──────────────────────────────────
function buildSplitResultsHtml(data) {
  const total    = data.total_segments    ?? 0;
  const matched  = data.matched_segments  ?? 0;
  const duration = data.total_duration    ?? "—";
  const segments = data.segments          ?? [];

  const statsHtml = `
    <div class="split-summary">
      <div class="split-stat">
        <span class="split-stat-num">${total}</span>
        <span class="split-stat-label">检测片段</span>
      </div>
      <div class="split-stat">
        <span class="split-stat-num">${matched}</span>
        <span class="split-stat-label">匹配片段</span>
      </div>
      <div class="split-stat">
        <span class="split-stat-num">${duration}</span>
        <span class="split-stat-label">匹配总时长</span>
      </div>
    </div>
  `;

  if (segments.length === 0) {
    return statsHtml + `<p class="tool-desc" style="margin-top:12px;">
      未检测到目标匹配片段，请尝试调低相似度阈值。
    </p>`;
  }

  const timelineHtml = segments.map((seg, i) => `
    <div class="split-seg ${seg.matched ? "matched" : "unmatched"}">
      <span class="seg-index">#${i + 1}</span>
      <span class="seg-time">${seg.start_time} → ${seg.end_time}</span>
      <span class="seg-score">${seg.score != null ? seg.score.toFixed(3) : "—"}</span>
      <span class="seg-badge ${seg.matched ? "badge-match" : "badge-no"}">
        ${seg.matched ? "✅ 匹配" : "⬜ 不匹配"}
      </span>
    </div>
  `).join("");

  return statsHtml + `<div class="split-timeline">${timelineHtml}</div>`;
}

// ── Running state ───────────────────────────────────────────
function setSplitRunning(type, running) {
  const isface = type === "face";
  const btn    = document.getElementById(isface ? "runFaceSplitBtn"  : "runVoiceSplitBtn");
  const icon   = document.getElementById(isface ? "faceSplitIcon"    : "voiceSplitIcon");
  const label  = document.getElementById(isface ? "faceSplitText"    : "voiceSplitText");
  if (!btn) return;
  btn.disabled      = running;
  icon.innerHTML    = running ? '<span class="spinner"></span>' : "▶";
  label.textContent = running ? "处理中…" : (isface ? "开始人脸切分" : "开始声纹切分");
}

// ── Utilities ──────────────────────────────────────────────
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
