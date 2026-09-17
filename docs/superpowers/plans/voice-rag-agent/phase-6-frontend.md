# Phase 6: Web / mWeb Voice UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single static page, usable on desktop and mobile browsers, that captures mic audio, streams it to the backend over a WebSocket, and plays the spoken reply back as it arrives.

**Architecture:** Plain HTML/JS/CSS, no build step, no framework. The backend URL is typed in at runtime because the ngrok tunnel changes every Colab restart. Development and testing happen against a local mock server that speaks the same wire protocol, so this phase never needs Colab, a GPU or any model.

**Tech Stack:** Vanilla HTML/CSS/JS (getUserMedia, Web Audio API, WebSocket), plus a small FastAPI mock server for local development.

**Spec:** `docs/superpowers/specs/2026-09-17-voice-rag-agent-design.md`

**Index:** [README.md](README.md)

## Isolation

Depends on the **wire protocol only**, never on backend code. Task 6.1 builds a mock server that implements that protocol with canned responses, so the UI is fully developed and verified before the real backend is ever reachable. If the backend's internals change, nothing here moves; only a protocol change touches this document.

## Contract consumed

The WebSocket wire protocol defined in Phase 5. Endpoint: `/ws/{session_id}`.

**Client → server:**
- Binary frames: raw 16kHz mono PCM chunks.
- Text frame `"__end__"`: force end-of-turn, sent on push-to-talk release.

**Server → client, per turn, in order:**
1. Text `{"type": "transcript", "text": str, "language": str}`
2. Repeating per clause: text `{"type": "answer_text", "text": str}`, then a binary WAV frame for that clause.
3. Text `{"type": "timing", "stages": {...}}`
4. Text `{"type": "turn_end"}`

**Out of band:** text `{"type": "interrupted"}` — barge-in fired; stop playback and discard queued audio. No `turn_end` follows it.

## Two browser constraints that will bite if ignored

**1. The mic requires a secure context.** `getUserMedia` is unavailable on plain `http://` origins — `localhost` is exempt, but `http://192.168.x.x` on a phone is not. It fails without a useful error. **The frontend must be served over HTTPS for any mobile testing**, which means a second ngrok tunnel (or any static host: Netlify, GitHub Pages, Vercel).

**2. iOS ignores requested sample rates.** `new AudioContext({sampleRate: 16000})` is honoured on desktop Chrome but Safari/iOS runs its hardware rate (usually 44.1 or 48kHz) and either ignores the hint or throws. Audio then arrives at the server at the wrong rate, and both webrtcvad and Whisper produce silent nonsense rather than an error. **Always read `audioContext.sampleRate` and downsample explicitly.**

## Global Constraints

- The ngrok URL is not stable across Colab restarts — the backend URL must be user-supplied at runtime, never hardcoded. *(spec: Frontend)*
- Must work on mobile browsers as well as desktop, which requires HTTPS. *(spec: Frontend)*
- UI shows mic state, live transcript, connection status and the detected language. *(spec: Frontend)*
- On WebSocket close, show a disconnected state the user can retry from. *(spec: Error handling)*
- Audio sent must be 16kHz mono 16-bit PCM regardless of the device's native rate. *(spec: Data flow step 1)*
- Barge-in must stop playback immediately, including audio already queued. *(spec: Data flow step 8)*

---

### Task 6.1: Mock WebSocket server for frontend development

**Files:**
- Create: `frontend/mock_server.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a standalone server on port 8001 implementing the wire protocol above with canned responses.

- [ ] **Step 1: Write the mock server**

```python
# frontend/mock_server.py
import asyncio
import io
import json
import struct
import wave

import uvicorn
from fastapi import FastAPI, WebSocket

app = FastAPI()


def _fake_wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # ~0.25s of a simple tone so playback is audibly verifiable
        samples = [int(6000 * ((i // 20) % 2 * 2 - 1)) for i in range(4000)]
        wf.writeframes(struct.pack("<%dh" % len(samples), *samples))
    return buffer.getvalue()


@app.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    while True:
        message = await websocket.receive()

        if message.get("type") == "websocket.disconnect":
            break

        if message.get("text") != "__end__":
            continue  # ignore streamed audio chunks; this mock is turn-triggered

        await websocket.send_text(json.dumps({
            "type": "transcript", "text": "do you have headphones", "language": "en",
        }))
        await asyncio.sleep(0.3)

        for clause in ["Yes,", "we have Wireless Headphones in stock."]:
            await websocket.send_text(json.dumps({"type": "answer_text", "text": clause}))
            await websocket.send_bytes(_fake_wav_bytes())
            await asyncio.sleep(0.2)

        await websocket.send_text(json.dumps({"type": "timing", "stages": {
            "stt": 1800.0, "retrieval": 25.0, "llm_first_token": 900.0,
            "llm_total": 2100.0, "tts_total": 1200.0, "time_to_first_audio": 3100.0,
            "turn_total": 4300.0,
        }}))
        await websocket.send_text(json.dumps({"type": "turn_end"}))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

- [ ] **Step 2: Run it and verify it starts**

Run: `python -m frontend.mock_server`
Expected: `Uvicorn running on http://0.0.0.0:8001`, no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/mock_server.py
git commit -m "feat: add mock WebSocket server for frontend-only development"
```

---

### Task 6.2: Voice UI page

**Files:**
- Create: `frontend/index.html`
- Create: `frontend/app.js`
- Create: `frontend/style.css`

**Interfaces:**
- Consumes: the wire protocol above (served by Task 6.1's mock during development, by the real backend in Phase 7).
- Produces: a static page with a backend-URL field, a push-to-talk mic button, a live transcript log, and connection plus language status.

- [ ] **Step 1: Write the HTML shell**

```html
<!-- frontend/index.html -->
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Voice Inventory Assistant</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <main>
    <h1>Voice Inventory Assistant</h1>

    <label for="backend-url">Backend WebSocket URL</label>
    <input id="backend-url" type="text" placeholder="wss://xxxx.ngrok-free.app/ws/session1" />
    <button id="connect-btn">Connect</button>

    <p class="status-row">
      Status: <span id="status">disconnected</span>
      &middot; Language: <span id="language">—</span>
    </p>
    <p class="timing-row" id="timing"></p>

    <button id="mic-btn" disabled>Hold to talk</button>

    <section id="transcript-log" aria-live="polite"></section>
  </main>
  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write the client logic**

```javascript
// frontend/app.js
const TARGET_RATE = 16000;

let ws = null;
let captureContext = null;
let mediaStream = null;
let processor = null;
let playbackContext = null;

// Playback queue state. Clauses arrive faster than they play, so each buffer is
// scheduled to start when the previous one ends instead of immediately —
// otherwise overlapping clauses play on top of each other.
let nextPlayTime = 0;
let scheduledSources = [];
let decodeChain = Promise.resolve();

const statusEl = document.getElementById("status");
const languageEl = document.getElementById("language");
const timingEl = document.getElementById("timing");
const logEl = document.getElementById("transcript-log");
const micBtn = document.getElementById("mic-btn");
const connectBtn = document.getElementById("connect-btn");
const urlInput = document.getElementById("backend-url");

function log(role, text) {
  const p = document.createElement("p");
  p.className = `line line-${role}`;
  p.textContent = `${role}: ${text}`;
  logEl.appendChild(p);
  logEl.scrollTop = logEl.scrollHeight;
}

connectBtn.addEventListener("click", () => {
  const url = urlInput.value.trim();
  if (!url) return;

  statusEl.textContent = "connecting";
  ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    statusEl.textContent = "connected";
    micBtn.disabled = false;
  };

  ws.onclose = () => {
    statusEl.textContent = "disconnected — press Connect to retry";
    micBtn.disabled = true;
  };

  ws.onerror = () => {
    statusEl.textContent = "connection error";
  };

  ws.onmessage = (event) => {
    if (typeof event.data !== "string") {
      queueAudio(event.data);
      return;
    }
    const msg = JSON.parse(event.data);
    if (msg.type === "transcript") {
      log("you", msg.text);
      languageEl.textContent = msg.language;
    } else if (msg.type === "answer_text") {
      log("agent", msg.text);
    } else if (msg.type === "interrupted") {
      stopPlayback();
      log("system", "interrupted");
      statusEl.textContent = "listening";
    } else if (msg.type === "timing") {
      showTiming(msg.stages);
    } else if (msg.type === "turn_end") {
      statusEl.textContent = "connected";
    }
  };
});

function showTiming(stages) {
  const parts = Object.entries(stages).map(([k, v]) => `${k} ${v}ms`);
  timingEl.textContent = parts.join(" · ");
}

// ---------- playback ----------

function ensurePlaybackContext() {
  if (!playbackContext) {
    playbackContext = new (window.AudioContext || window.webkitAudioContext)();
  }
  // iOS suspends audio contexts until a user gesture resumes them.
  if (playbackContext.state === "suspended") playbackContext.resume();
  return playbackContext;
}

function queueAudio(arrayBuffer) {
  const ctx = ensurePlaybackContext();
  // Chain the decodes so clauses cannot be scheduled out of order when a later,
  // shorter clip finishes decoding first.
  decodeChain = decodeChain
    .then(() => ctx.decodeAudioData(arrayBuffer.slice(0)))
    .then((buffer) => {
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      const startAt = Math.max(ctx.currentTime, nextPlayTime);
      source.start(startAt);
      nextPlayTime = startAt + buffer.duration;
      scheduledSources.push(source);
      source.onended = () => {
        scheduledSources = scheduledSources.filter((s) => s !== source);
      };
    })
    .catch(() => {});
}

function stopPlayback() {
  scheduledSources.forEach((source) => {
    try {
      source.stop();
    } catch (e) {
      /* already stopped */
    }
  });
  scheduledSources = [];
  nextPlayTime = 0;
  decodeChain = Promise.resolve();
}

// ---------- capture ----------

// Box-filter downsample. The device rate is whatever the hardware gives us
// (typically 44100 or 48000); the backend requires exactly 16000.
function downsample(input, inputRate) {
  if (inputRate === TARGET_RATE) return input;
  const ratio = inputRate / TARGET_RATE;
  const outLength = Math.floor(input.length / ratio);
  const out = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(Math.floor((i + 1) * ratio), input.length);
    let sum = 0;
    for (let j = start; j < end; j++) sum += input[j];
    out[i] = end > start ? sum / (end - start) : 0;
  }
  return out;
}

function floatToPcm16(input) {
  const pcm16 = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const clamped = Math.max(-1, Math.min(1, input[i]));
    pcm16[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
  }
  return pcm16;
}

async function startMic(event) {
  event.preventDefault();
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (processor) return; // already capturing

  ensurePlaybackContext(); // unlock audio output on the same user gesture
  statusEl.textContent = "listening";

  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });

  // Do NOT request a sample rate here — iOS ignores it. Use whatever the
  // hardware provides and downsample in software.
  captureContext = new (window.AudioContext || window.webkitAudioContext)();
  const deviceRate = captureContext.sampleRate;

  const source = captureContext.createMediaStreamSource(mediaStream);
  processor = captureContext.createScriptProcessor(4096, 1, 1);

  processor.onaudioprocess = (e) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const input = e.inputBuffer.getChannelData(0);
    const resampled = downsample(input, deviceRate);
    ws.send(floatToPcm16(resampled).buffer);
  };

  source.connect(processor);
  processor.connect(captureContext.destination);
}

function stopMic(event) {
  event.preventDefault();
  if (processor) {
    processor.disconnect();
    processor = null;
  }
  if (captureContext) {
    captureContext.close();
    captureContext = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send("__end__");
    statusEl.textContent = "thinking";
  }
}

micBtn.addEventListener("mousedown", startMic);
micBtn.addEventListener("touchstart", startMic);
micBtn.addEventListener("mouseup", stopMic);
micBtn.addEventListener("touchend", stopMic);
```

> **Note on `createScriptProcessor`:** it is deprecated in favour of `AudioWorklet`, which runs on a dedicated audio thread and does not risk glitching when the main thread is busy. It still works in every current browser and is far less code, which is the right trade for a demo — but it is a legitimate thing to be asked about, so know the answer: *"ScriptProcessor runs on the main thread and is deprecated; AudioWorklet is the correct production choice."*

- [ ] **Step 3: Write the styling**

```css
/* frontend/style.css */
body {
  font-family: system-ui, -apple-system, sans-serif;
  max-width: 480px;
  margin: 0 auto;
  padding: 16px;
  color: #0f172a;
}

h1 {
  font-size: 20px;
}

label {
  display: block;
  font-size: 13px;
  color: #475569;
  margin-top: 12px;
}

input,
button {
  width: 100%;
  padding: 12px;
  margin: 6px 0;
  font-size: 16px; /* 16px avoids iOS zoom-on-focus */
  box-sizing: border-box;
}

.status-row {
  font-size: 13px;
  color: #475569;
}

.timing-row {
  font-size: 11px;
  color: #64748b;
  font-family: ui-monospace, monospace;
  min-height: 14px;
  word-break: break-word;
}

.line-system {
  color: #b45309;
  font-style: italic;
  font-size: 13px;
}

#mic-btn {
  background: #2563eb;
  color: #fff;
  border: none;
  border-radius: 8px;
  padding: 20px;
  font-size: 18px;
  touch-action: none; /* stop mobile long-press selection during push-to-talk */
  user-select: none;
}

#mic-btn:disabled {
  background: #94a3b8;
}

#transcript-log {
  margin-top: 16px;
  border-top: 1px solid #e2e8f0;
  padding-top: 8px;
  max-height: 50vh;
  overflow-y: auto;
}

.line {
  margin: 6px 0;
  line-height: 1.4;
}

.line-agent {
  color: #1d4ed8;
}
```

- [ ] **Step 4: Desktop test against the mock server**

In one terminal: `python -m frontend.mock_server`
In another: `python -m http.server 5500 --directory frontend`

Open `http://localhost:5500/index.html` — `localhost` is a secure context, so the mic works here without HTTPS. Enter `ws://localhost:8001/ws/session1`, press Connect, hold the mic button, speak, release.

Expected: status moves connecting → connected → listening → thinking → connected; the canned transcript and two answer clauses appear; language shows `en`; two tones play **one after the other, not on top of each other**; the timing row stays empty (the mock sends no timing message).

- [ ] **Step 5: Mobile test — requires HTTPS on both ends**

`http://<lan-ip>:5500` **will not work** — the mic is blocked on insecure origins with no clear error. Tunnel both the page and the mock server:

```bash
# terminal 1
python -m frontend.mock_server
# terminal 2
python -m http.server 5500 --directory frontend
# terminal 3 — two tunnels
ngrok http 5500    # serves the page over https
ngrok http 8001    # serves the mock server over wss
```

Open the `https://` page URL on a phone, paste the mock server's `wss://` URL into the backend field, and repeat the Step 4 interaction.

Expected: identical behaviour to desktop. If the mic button does nothing, check the browser console for a secure-context error before debugging anything else.

- [ ] **Step 6: Verify the audio is genuinely 16kHz**

Phones rarely run at 16kHz natively, so confirm the downsampling works rather than assuming it. Add a temporary log in `startMic`:

```javascript
console.log("device rate", deviceRate, "-> sending", TARGET_RATE);
```

Expected on a phone: something like `device rate 48000 -> sending 16000`. If the device rate reads 16000 on a phone, be suspicious — verify the downsample path is actually running.

- [ ] **Step 7: Commit**

```bash
git add frontend/index.html frontend/app.js frontend/style.css
git commit -m "feat: add mobile-responsive web voice UI with queued playback"
```

---

## Phase 6 done when

Steps 4–6 pass on both a desktop browser and a real mobile browser over HTTPS, against the mock server, with no real backend involved — including clauses playing sequentially rather than overlapping, and a confirmed downsample from the device's native rate to 16kHz.
