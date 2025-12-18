// static/app.js

// ✅ Auto-pick the current host (works on LAN too)
// If your API is on the same host/port, just keep this.
const API_BASE = `${window.location.protocol}//${window.location.hostname}:8000`;

// Optional override (uncomment if needed)
// const API_BASE = "http://127.0.0.1:8000";

const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const submitBtn = document.getElementById("submitBtn");
const statusEl = document.getElementById("status");

const fileLabel = document.getElementById("fileLabel");
const fileLabelText = fileLabel.querySelector(".label-text");
const defaultLabelText = "Select a document...";

const resultBox = document.getElementById("result");
const resFilename = document.getElementById("resFilename");
const resClass = document.getElementById("resClass");
const resConf = document.getElementById("resConf");
const resId = document.getElementById("resId");
const resTop3 = document.getElementById("resTop3");

// ===== Helpers =====
function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  submitBtn.textContent = isLoading ? "Classifying..." : "Classify";
}

function setStatus(msg = "", { error = false, success = false } = {}) {
  statusEl.textContent = msg;
  statusEl.classList.remove("error", "success");
  if (error) statusEl.classList.add("error");
  if (success) statusEl.classList.add("success");
}

function hideResult() {
  resultBox.classList.add("hidden");
}

function showResult(data) {
  resFilename.textContent = data.filename || "-";
  resClass.textContent = data.predicted_class || "-";
  resId.textContent = data.id || "-";

  // Confidence
  if (data.confidence === null || data.confidence === undefined) {
    resConf.textContent = "N/A";
  } else {
    resConf.textContent = `${(data.confidence * 100).toFixed(2)}%`;
  }

  // Top-3
  if (Array.isArray(data.top_k) && data.top_k.length > 0) {
    resTop3.textContent = data.top_k
      .map(x => `${x.label} (${(x.confidence * 100).toFixed(1)}%)`)
      .join(" | ");
  } else {
    resTop3.textContent = "N/A";
  }

  resultBox.classList.remove("hidden");
}

async function safeJson(res) {
  const contentType = res.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) return null;
  try {
    return await res.json();
  } catch {
    return null;
  }
}

// ===== UI events =====
fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) {
    fileLabelText.textContent = file.name;
    fileLabel.classList.add("file-selected");
  } else {
    fileLabelText.textContent = defaultLabelText;
    fileLabel.classList.remove("file-selected");
  }
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const file = fileInput.files?.[0];
  if (!file) {
    setStatus("Pick a file first 😄", { error: true });
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  hideResult();
  setLoading(true);
  setStatus("Uploading + classifying…", { error: false });

  try {
    const res = await fetch(`${API_BASE}/classify`, {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const errBody = await safeJson(res);
      const msg = errBody?.detail || `Request failed (HTTP ${res.status})`;
      throw new Error(msg);
    }

    const data = await res.json();
    showResult(data);
    setStatus("Done ✅ Classification complete!", { success: true });

  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`, { error: true });

  } finally {
    setLoading(false);
  }
});
