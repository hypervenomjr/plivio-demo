const TARGET_RATE = 16000;

let ws = null;
let captureContext = null;
let mediaStream = null;
let processor = null;
let playbackContext = null;

// Playback queue state. Clauses arrive faster than they play, so each buffer
// is scheduled to start when the previous one ends instead of immediately -
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
  // Chain the decodes so clauses cannot be scheduled out of order when a
  // later, shorter clip finishes decoding first.
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
// (typically 44100 or 48000); the backend requires exactly 16000. iOS
// ignores a requested AudioContext sample rate, so this cannot be skipped.
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

  // Do NOT request a sample rate here - iOS ignores it. Use whatever the
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
