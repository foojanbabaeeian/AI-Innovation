// Background service worker.
//
// Responsibilities:
//   - Own the logical "session" state (idle | connecting | running | error)
//   - Create/destroy the offscreen document that performs audio capture + WS
//   - Relay messages between popup and offscreen
//   - Emit native notifications on AI alerts
//
// Chrome MV3 service workers can be suspended; we use chrome.storage.session
// as a small persistence layer so the popup can rehydrate UI after reload.

const OFFSCREEN_URL = "offscreen.html";

// In-memory snapshot (also mirrored to session storage for cold-start popups).
const snapshot = {
  state: "idle", // idle | connecting | running | error
  backendStatus: "offline", // offline | online | error
  lastDetection: null,
  stats: { windows: 0, aiDetections: 0 },
  tabId: null,
};

async function persistSnapshot() {
  try {
    await chrome.storage.session.set({ snapshot });
  } catch (_) {
    /* session storage may not be available in all contexts */
  }
}

async function restoreSnapshot() {
  try {
    const { snapshot: s } = await chrome.storage.session.get("snapshot");
    if (s) Object.assign(snapshot, s);
  } catch (_) {}
}
restoreSnapshot();

// ---------------------------------------------------------------
// Offscreen document lifecycle
// ---------------------------------------------------------------

async function hasOffscreen() {
  if (!chrome.runtime.getContexts) return false;
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
    documentUrls: [chrome.runtime.getURL(OFFSCREEN_URL)],
  });
  return contexts.length > 0;
}

async function ensureOffscreen() {
  if (await hasOffscreen()) return;
  await chrome.offscreen.createDocument({
    url: OFFSCREEN_URL,
    reasons: ["USER_MEDIA"],
    justification: "Capture tab audio and stream it to the detection backend.",
  });
}

async function closeOffscreen() {
  if (await hasOffscreen()) {
    try {
      await chrome.offscreen.closeDocument();
    } catch (_) {}
  }
}

// ---------------------------------------------------------------
// Popup <-> background dispatch
// ---------------------------------------------------------------

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    switch (msg.type) {
      case "get_state":
        sendResponse({ state: snapshot.state });
        return;

      case "get_snapshot":
        sendResponse(snapshot);
        return;

      case "start":
        await handleStart(msg);
        sendResponse({ ok: true });
        return;

      case "stop":
        await handleStop();
        sendResponse({ ok: true });
        return;

      case "settings_updated":
        // Forward to offscreen so it can adjust threshold live.
        sendToOffscreen({ type: "settings_updated", settings: msg.settings });
        sendResponse({ ok: true });
        return;

      // Messages from offscreen -> relay to popup + update snapshot
      case "offscreen_state":
        snapshot.state = msg.state;
        await persistSnapshot();
        broadcast({ type: "state", state: msg.state });
        return;

      case "offscreen_backend_status":
        snapshot.backendStatus = msg.status;
        await persistSnapshot();
        broadcast({ type: "backend_status", status: msg.status });
        return;

      case "offscreen_detection":
        snapshot.lastDetection = msg.detection;
        snapshot.stats.windows += 1;
        if (msg.detection?.is_ai_detected) snapshot.stats.aiDetections += 1;
        await persistSnapshot();
        broadcast({ type: "detection", detection: msg.detection });
        broadcast({
          type: "stats",
          windows: snapshot.stats.windows,
          aiDetections: snapshot.stats.aiDetections,
        });
        return;

      case "offscreen_alert":
        broadcast({ type: "alert", alert: msg.alert });
        await maybeNotify(msg.alert);
        return;

      case "offscreen_error":
        snapshot.state = "error";
        await persistSnapshot();
        broadcast({ type: "error", message: msg.message });
        return;
    }
  })();
  return true; // keep sendResponse alive for async
});

function broadcast(msg) {
  chrome.runtime.sendMessage(msg).catch(() => {
    /* popup may be closed */
  });
}

function sendToOffscreen(msg) {
  chrome.runtime.sendMessage({ ...msg, target: "offscreen" }).catch(() => {});
}

// ---------------------------------------------------------------
// Start / Stop orchestration
// ---------------------------------------------------------------

async function handleStart({ streamId, tabId, tabUrl, tabTitle, settings }) {
  snapshot.state = "connecting";
  snapshot.tabId = tabId;
  snapshot.stats = { windows: 0, aiDetections: 0 };
  snapshot.lastDetection = null;
  await persistSnapshot();
  broadcast({ type: "state", state: "connecting" });

  await ensureOffscreen();

  // Give the offscreen document a tick to wire up its listener.
  await new Promise((r) => setTimeout(r, 50));

  sendToOffscreen({
    type: "start_capture",
    streamId,
    tabId,
    tabUrl,
    tabTitle,
    settings,
  });
}

async function handleStop() {
  sendToOffscreen({ type: "stop_capture" });
  snapshot.state = "idle";
  await persistSnapshot();
  broadcast({ type: "state", state: "idle" });
  // Give offscreen a moment to clean up before tearing it down.
  setTimeout(() => closeOffscreen(), 250);
}

// ---------------------------------------------------------------
// Notifications
// ---------------------------------------------------------------

async function maybeNotify(alert) {
  if (!alert || alert.level !== "ai_detected") return;
  try {
    await chrome.notifications.create({
      type: "basic",
      iconUrl: chrome.runtime.getURL("icons/icon128.png"),
      title: "AI audio detected",
      message: alert.message || "This tab appears to contain AI-generated audio.",
      priority: 2,
    });
  } catch (_) {
    /* notifications may be disabled */
  }
}
