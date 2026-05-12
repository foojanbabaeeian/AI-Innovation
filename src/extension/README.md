# AI Audio Detector — Browser Extension (MVP)

One-click Chrome extension that captures the audio of the active tab and
streams it to the detection backend (`src/backend`) over a WebSocket.
On every inference window the popup updates a live meter, and on
debounced state changes it surfaces a browser notification.

## Architecture

```
popup.html/js ──▶ background.js (service worker)
                      │
                      ├─ chrome.offscreen.createDocument
                      │
                      ▼
                 offscreen.html/js
                      │
                      ├─ getUserMedia(tab streamId) ─▶ AudioContext
                      │                                   │
                      │                                   ├─▶ destination (user still hears audio)
                      │                                   └─▶ AudioWorkletNode ─▶ Float32 frames
                      │
                      └─ WebSocket ──────▶ ws://localhost:8000/ws (FastAPI backend)
```

Key points:

- **Tab capture** uses `chrome.tabCapture.getMediaStreamId` from the popup
  (needed for the user-gesture requirement), then hands the `streamId` to
  the offscreen document which opens the actual `MediaStream`.
- **Web Audio is unavailable in service workers**, so an offscreen document
  owns the audio graph + WebSocket.
- **AudioWorklet** downmixes to mono and chunks into ~100 ms Float32 frames
  which are sent as binary `ArrayBuffer`s to the backend.
- The backend expects little-endian `float32` mono PCM; the browser's
  native sample rate is reported once via the `start` JSON message and the
  backend resamples internally.

## File layout

| File | Purpose |
| --- | --- |
| `manifest.json` | MV3 manifest (tabCapture, offscreen, storage, notifications). |
| `popup.html` / `popup.css` / `popup.js` | Single-button UI + live meter + settings. |
| `background.js` | Service worker; owns logical state, opens/closes the offscreen doc, fires notifications. |
| `offscreen.html` / `offscreen.js` | Captures audio, runs the worklet, maintains the WebSocket. |
| `audio-worklet.js` | `AudioWorkletProcessor` that downmixes + chunks to Float32 frames. |
| `icons/logo.png` | Extension icon. |

## Running it end-to-end

### 1. Start the backend

From the repo root:

```bash
# Mock detector (no model required), great for UI iteration
python -m src.backend.run --mock

# Or full mode (uses whatever detector detector.py is configured with)
python -m src.backend.run
```

By default the backend listens on `ws://localhost:8000/ws`.

### 2. Load the extension

1. Open `chrome://extensions`.
2. Enable **Developer mode** (top right).
3. Click **Load unpacked** and select `src/extension/`.
4. Pin the extension to the toolbar.

### 3. Use it

1. Navigate to a tab that is playing audio (YouTube, a podcast, etc.).
2. Click the extension icon.
3. Click **Activate**. The popup will go to *Connecting…* then *Streaming*.
4. The meter updates on every inference window; alerts appear on the
   real ↔ ai state flip and fire a Chrome notification.
5. Click **Deactivate** to stop.

## Settings

In the popup's `Settings` drawer:

- **Backend URL** — defaults to `ws://localhost:8000/ws`. Change to point
  at a remote deployment (include `wss://` for TLS). Host permissions in
  `manifest.json` currently whitelist only `localhost` — add your host
  there before deploying.
- **Threshold** — sent to the backend as a `set_threshold` message and
  also rendered as the vertical tick on the meter.

## Protocol summary

The extension talks the protocol in `src/backend/protocol.py`:

- **Client → server (JSON):** `start`, `stop`, `ping`, `set_threshold`
- **Client → server (binary):** little-endian `float32` mono PCM frames
- **Server → client (JSON):** `ready`, `started`, `stopped`, `detection`,
  `alert`, `pong`, `error`

The popup re-renders on every `detection` and pushes a native
notification on every `alert` with `level == "ai_detected"`.

## Troubleshooting

- **"Could not capture tab audio"** — make sure the tab is actually
  playing audio and that no other capture (DevTools recorder, Meet, etc.)
  is holding it. Tab capture only works on top-level http(s) pages, not
  on `chrome://` pages.
- **"Backend unreachable"** — check that `python -m src.backend.run` is
  running and listening on the port in the Backend URL setting. If you
  change the port, update the **Backend URL** in the popup.
- **I can't hear the tab anymore** — the offscreen document re-routes
  captured audio to `audioContext.destination`. If the context fails to
  start (rare autoplay edge cases) reload the tab and re-activate.
- **Service worker sleeping** — the popup rehydrates from
  `chrome.storage.session`, but while the popup is closed Chrome may
  suspend the worker. Capture continues inside the offscreen document
  regardless.

## Next steps (post-MVP)

- Per-tab activation (currently single global session).
- Waveform / spectrogram visualization driven by the same worklet.
- Options page for backend URL, auto-start domains, alert sound.
- Packaging for the Chrome Web Store (bump version, strip dev host perms).
