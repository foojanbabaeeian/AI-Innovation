// Offscreen document: owns the audio capture graph and the WebSocket.
//
// The service worker cannot use Web Audio APIs, so we delegate the heavy
// lifting to this hidden document. We receive a streamId from the popup
// (created via chrome.tabCapture.getMediaStreamId), open a MediaStream,
// pipe it through an AudioWorklet that chunks it into ~100ms mono frames,
// and forward each frame as binary Float32 to the backend over a WebSocket.
//
// We also re-route the captured audio back to the speakers so the user can
// still hear the tab - tab capture mutes the original playback path.

const state = {
  ws: null,
  audioContext: null,
  mediaStream: null,
  workletNode: null,
  sourceNode: null,
  gainNode: null,
  settings: { backendUrl: "ws://localhost:8765/ws", threshold: 0.7 },
  tabUrl: null,
  tabTitle: null,
  pingInterval: null,
  running: false,
};

// ---------------------------------------------------------------
// Messaging
// ---------------------------------------------------------------

function toBackground(msg) {
  try {
    chrome.runtime.sendMessage(msg).catch(() => {});
  } catch (_) {}
}

function setState(s) {
  toBackground({ type: "offscreen_state", state: s });
}
function setBackendStatus(s) {
  toBackground({ type: "offscreen_backend_status", status: s });
}
function reportError(message) {
  console.error("[offscreen]", message);
  toBackground({ type: "offscreen_error", message });
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg?.target !== "offscreen") return;
  switch (msg.type) {
    case "start_capture":
      startCapture(msg).catch((e) => reportError(e?.message || String(e)));
      break;
    case "stop_capture":
      stopCapture();
      break;
    case "settings_updated":
      state.settings = { ...state.settings, ...msg.settings };
      sendJson({ type: "set_threshold", threshold: state.settings.threshold });
      break;
  }
});

// ---------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------

function openWebSocket() {
  return new Promise((resolve, reject) => {
    let ws;
    try {
      ws = new WebSocket(state.settings.backendUrl);
    } catch (e) {
      reject(e);
      return;
    }
    ws.binaryType = "arraybuffer";

    const timeout = setTimeout(() => {
      reject(new Error("Backend connection timed out"));
      try { ws.close(); } catch (_) {}
    }, 5000);

    ws.addEventListener("open", () => {
      clearTimeout(timeout);
      setBackendStatus("online");
      resolve(ws);
    });
    ws.addEventListener("error", () => {
      clearTimeout(timeout);
      setBackendStatus("error");
      reject(new Error("WebSocket error"));
    });
    ws.addEventListener("close", () => {
      setBackendStatus("offline");
      if (state.running) {
        reportError("Backend connection closed");
        stopCapture();
      }
    });
    ws.addEventListener("message", (evt) => {
      if (typeof evt.data !== "string") return;
      let msg;
      try { msg = JSON.parse(evt.data); } catch { return; }
      handleServerMessage(msg);
    });
  });
}

function sendJson(obj) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(obj));
  }
}

function handleServerMessage(msg) {
  switch (msg.type) {
    case "ready":
      // Server is initialized; we sent `start` right after open, but harmless.
      break;
    case "started":
      setState("running");
      break;
    case "stopped":
      break;
    case "detection":
      toBackground({ type: "offscreen_detection", detection: msg });
      break;
    case "alert":
      toBackground({ type: "offscreen_alert", alert: msg });
      break;
    case "pong":
      break;
    case "error":
      reportError(`[${msg.code}] ${msg.message}`);
      break;
  }
}

// ---------------------------------------------------------------
// Capture
// ---------------------------------------------------------------

async function startCapture({ streamId, tabUrl, tabTitle, settings }) {
  if (state.running) return;
  state.settings = { ...state.settings, ...(settings || {}) };
  state.tabUrl = tabUrl;
  state.tabTitle = tabTitle;

  setState("connecting");

  // 1. Open the media stream for the specified tab.
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        mandatory: {
          chromeMediaSource: "tab",
          chromeMediaSourceId: streamId,
        },
      },
      video: false,
    });
  } catch (e) {
    reportError("getUserMedia failed: " + (e?.message || e));
    setState("error");
    return;
  }
  state.mediaStream = stream;

  // 2. Build the audio graph.
  const audioContext = new AudioContext();
  state.audioContext = audioContext;

  const sourceNode = audioContext.createMediaStreamSource(stream);
  state.sourceNode = sourceNode;

  // 100ms frame @ the AudioContext's sample rate.
  const frameSize = Math.round(audioContext.sampleRate * 0.1);

  try {
    await audioContext.audioWorklet.addModule(chrome.runtime.getURL("audio-worklet.js"));
  } catch (e) {
    reportError("Failed to load audio worklet: " + (e?.message || e));
    cleanupAudio();
    setState("error");
    return;
  }

  const workletNode = new AudioWorkletNode(audioContext, "detector-processor", {
    processorOptions: { frameSize },
    numberOfInputs: 1,
    numberOfOutputs: 0,
  });
  state.workletNode = workletNode;

  // Passthrough so the user still hears the tab (tab capture mutes original).
  const gainNode = audioContext.createGain();
  gainNode.gain.value = 1.0;
  state.gainNode = gainNode;
  sourceNode.connect(gainNode).connect(audioContext.destination);

  // Tap for analysis.
  sourceNode.connect(workletNode);

  // 3. Open WebSocket, send start, then wire frame -> WS.
  try {
    state.ws = await openWebSocket();
  } catch (e) {
    reportError("Backend unreachable at " + state.settings.backendUrl);
    cleanupAudio();
    setState("error");
    return;
  }

  sendJson({
    type: "start",
    sample_rate: audioContext.sampleRate,
    tab_url: state.tabUrl || null,
    tab_title: state.tabTitle || null,
  });
  sendJson({ type: "set_threshold", threshold: state.settings.threshold });

  workletNode.port.onmessage = (evt) => {
    // evt.data is a Float32Array (transferred) from the processor.
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(evt.data.buffer);
    }
  };

  state.running = true;
  setState("running");

  // Keepalive.
  state.pingInterval = setInterval(() => sendJson({ type: "ping" }), 15000);

  // Handle the user stopping the capture from Chrome's UI.
  stream.getAudioTracks().forEach((t) => {
    t.addEventListener("ended", () => stopCapture());
  });
}

function cleanupAudio() {
  if (state.workletNode) {
    try { state.workletNode.port.onmessage = null; } catch (_) {}
    try { state.workletNode.disconnect(); } catch (_) {}
    state.workletNode = null;
  }
  if (state.sourceNode) {
    try { state.sourceNode.disconnect(); } catch (_) {}
    state.sourceNode = null;
  }
  if (state.gainNode) {
    try { state.gainNode.disconnect(); } catch (_) {}
    state.gainNode = null;
  }
  if (state.mediaStream) {
    try { state.mediaStream.getTracks().forEach((t) => t.stop()); } catch (_) {}
    state.mediaStream = null;
  }
  if (state.audioContext) {
    try { state.audioContext.close(); } catch (_) {}
    state.audioContext = null;
  }
}

function stopCapture() {
  if (!state.running && !state.ws && !state.audioContext) return;
  state.running = false;

  if (state.pingInterval) {
    clearInterval(state.pingInterval);
    state.pingInterval = null;
  }

  if (state.ws) {
    try { sendJson({ type: "stop" }); } catch (_) {}
    try { state.ws.close(); } catch (_) {}
    state.ws = null;
  }

  cleanupAudio();
  setBackendStatus("offline");
  setState("idle");
}
