// Popup UI controller.
// Drives the single Activate/Deactivate button, renders live detection results,
// and talks to the background service worker which owns capture + WebSocket state.

const DEFAULTS = {
  backendUrl: "ws://localhost:8765/ws",
  threshold: 0.7,
};

const $ = (id) => document.getElementById(id);

const els = {
  toggleBtn: $("toggle-btn"),
  btnLabel: document.querySelector("#toggle-btn .btn-label"),
  btnSub: document.querySelector("#toggle-btn .btn-sub"),
  statusDot: $("status-dot"),
  backendStatus: $("backend-status"),
  result: $("result"),
  verdict: $("verdict"),
  verdictIcon: $("verdict-icon"),
  verdictLabel: $("verdict-label"),
  verdictSub: $("verdict-sub"),
  verdictPct: $("verdict-pct"),
  confidence: $("confidence"),
  meter: $("meter"),
  thresholdLine: $("threshold-line"),
  score: $("score"),
  smoothed: $("smoothed"),
  alerts: $("alerts"),
  backendUrl: $("backend-url"),
  threshold: $("threshold"),
  thresholdValue: $("threshold-value"),
  saveSettings: $("save-settings"),
  stats: $("stats"),
};

let settings = { ...DEFAULTS };

async function loadSettings() {
  const stored = await chrome.storage.local.get(["backendUrl", "threshold"]);
  settings = { ...DEFAULTS, ...stored };
  els.backendUrl.value = settings.backendUrl;
  els.threshold.value = settings.threshold;
  els.thresholdValue.textContent = Number(settings.threshold).toFixed(2);
  els.thresholdLine.style.left = `${settings.threshold * 100}%`;
}

async function saveSettings() {
  const backendUrl = els.backendUrl.value.trim() || DEFAULTS.backendUrl;
  const threshold = parseFloat(els.threshold.value);
  settings = { backendUrl, threshold };
  await chrome.storage.local.set(settings);
  chrome.runtime.sendMessage({ type: "settings_updated", settings });
  els.saveSettings.textContent = "Saved";
  setTimeout(() => (els.saveSettings.textContent = "Save"), 1200);
}

els.threshold.addEventListener("input", () => {
  els.thresholdValue.textContent = Number(els.threshold.value).toFixed(2);
  els.thresholdLine.style.left = `${els.threshold.value * 100}%`;
});
els.saveSettings.addEventListener("click", saveSettings);

// ---------------------------------------------------------------
// Button <-> background
// ---------------------------------------------------------------

function setButton(state) {
  els.toggleBtn.classList.remove("idle", "running", "connecting");
  if (state === "running") {
    els.toggleBtn.classList.add("running");
    els.btnLabel.textContent = "Deactivate";
    els.btnSub.textContent = "Streaming tab audio";
    els.statusDot.classList.add("active");
    els.statusDot.classList.remove("error");
  } else if (state === "connecting") {
    els.toggleBtn.classList.add("connecting");
    els.btnLabel.textContent = "Connecting…";
    els.btnSub.textContent = "Waiting for backend";
    els.statusDot.classList.remove("active", "error");
  } else if (state === "error") {
    els.toggleBtn.classList.add("idle");
    els.btnLabel.textContent = "Activate";
    els.btnSub.textContent = "Retry";
    els.statusDot.classList.remove("active");
    els.statusDot.classList.add("error");
  } else {
    els.toggleBtn.classList.add("idle");
    els.btnLabel.textContent = "Activate";
    els.btnSub.textContent = "Start listening to this tab";
    els.statusDot.classList.remove("active", "error");
  }
}

function setBackendStatus(status) {
  els.backendStatus.classList.remove("online", "error");
  if (status === "online") {
    els.backendStatus.classList.add("online");
    els.backendStatus.textContent = "online";
  } else if (status === "error") {
    els.backendStatus.classList.add("error");
    els.backendStatus.textContent = "error";
  } else {
    els.backendStatus.textContent = "offline";
  }
}

els.toggleBtn.addEventListener("click", async () => {
  const { state } = await chrome.runtime.sendMessage({ type: "get_state" });
  if (state === "running" || state === "connecting") {
    chrome.runtime.sendMessage({ type: "stop" });
    return;
  }

  // Must call tabCapture.getMediaStreamId from a user gesture, from the
  // tab's own extension context. We do it here in the popup so Chrome
  // associates the capture with the active tab.
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;
  setButton("connecting");
  try {
    const streamId = await new Promise((resolve, reject) => {
      chrome.tabCapture.getMediaStreamId({ targetTabId: tab.id }, (sid) => {
        if (chrome.runtime.lastError || !sid) {
          reject(chrome.runtime.lastError || new Error("No stream id"));
          return;
        }
        resolve(sid);
      });
    });

    await chrome.runtime.sendMessage({
      type: "start",
      streamId,
      tabId: tab.id,
      tabUrl: tab.url,
      tabTitle: tab.title,
      settings,
    });
  } catch (e) {
    console.error(e);
    pushAlert({ level: "error", message: "Could not capture tab audio: " + (e?.message || e) });
    setButton("error");
  }
});

// ---------------------------------------------------------------
// Live updates from background / offscreen
// ---------------------------------------------------------------

chrome.runtime.onMessage.addListener((msg) => {
  switch (msg.type) {
    case "state":
      setButton(msg.state);
      break;
    case "backend_status":
      setBackendStatus(msg.status);
      break;
    case "detection":
      renderDetection(msg.detection);
      break;
    case "alert":
      pushAlert(msg.alert);
      break;
    case "stats":
      els.stats.textContent = `${msg.windows} windows · ${msg.aiDetections} AI detections`;
      break;
    case "error":
      pushAlert({ level: "error", message: msg.message });
      setButton("error");
      break;
  }
});

const VERDICT_MAP = {
  real:  { label: "Human",  sub: "No AI detected",            icon: "\u2714" },
  mixed: { label: "Mixed",  sub: "Partially AI-generated",    icon: "\u26A0" },
  ai:    { label: "AI",     sub: "AI-generated audio detected", icon: "\u2718" },
};

function renderDetection(d) {
  els.result.hidden = false;
  const score = d.smoothed_score ?? d.ai_score ?? 0;
  const pct = Math.round(score * 100);
  const label = d.label || "real";
  const v = VERDICT_MAP[label] || VERDICT_MAP.real;

  els.verdict.className = `verdict ${label}`;
  els.verdictIcon.textContent = v.icon;
  els.verdictLabel.textContent = v.label;
  els.verdictSub.textContent = v.sub;
  els.verdictPct.textContent = `${pct}%`;

  els.meter.style.width = `${Math.min(100, Math.max(0, pct)).toFixed(1)}%`;
  els.score.textContent = (d.ai_score ?? 0).toFixed(2);
  els.smoothed.textContent = (d.smoothed_score ?? 0).toFixed(2);
  els.confidence.textContent = `${Math.round((d.confidence ?? 0) * 100)}% conf`;
}

function pushAlert(a) {
  const div = document.createElement("div");
  div.className = "alert" + (a.level === "cleared" ? " cleared" : "");
  const ts = new Date().toLocaleTimeString();
  div.textContent = `[${ts}] ${a.message}`;
  els.alerts.prepend(div);
  while (els.alerts.children.length > 5) {
    els.alerts.removeChild(els.alerts.lastChild);
  }
}

// ---------------------------------------------------------------
// Init: hydrate UI from background's current state
// ---------------------------------------------------------------

(async () => {
  await loadSettings();
  const snapshot = await chrome.runtime.sendMessage({ type: "get_snapshot" });
  if (snapshot) {
    setButton(snapshot.state || "idle");
    setBackendStatus(snapshot.backendStatus || "offline");
    if (snapshot.lastDetection) renderDetection(snapshot.lastDetection);
    if (snapshot.stats) {
      els.stats.textContent = `${snapshot.stats.windows} windows · ${snapshot.stats.aiDetections} AI detections`;
    }
  }
})();
