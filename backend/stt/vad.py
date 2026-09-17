"""Voice-activity segmentation.

Turns a continuous PCM stream into discrete utterances. This is what replaces
a push-to-talk button: the caller simply stops talking and the segmenter
notices, which is the only turn-taking model that feels like a phone call.

`speech_active` exposes the in-progress state so the server can watch its
rising edge and cancel an in-flight answer when the caller talks over the
agent — the barge-in trigger.

`push` accepts PCM chunks of ANY length and re-frames internally. webrtcvad
only accepts exact 10/20/30ms frames, but a browser delivers whatever its
audio buffer size works out to (a 4096-sample buffer downsampled from 48kHz
to 16kHz arrives as ~1365 samples). Framing belongs here, once, rather than
in every caller.
"""
import webrtcvad

# 0-3. Higher is more aggressive about classifying audio as non-speech.
VAD_AGGRESSIVENESS = 2


class UtteranceSegmenter:
    def __init__(self, sample_rate: int = 16000, silence_ms: int = 600,
                 frame_ms: int = 30):
        if frame_ms not in (10, 20, 30):
            raise ValueError(f"webrtcvad supports 10, 20 or 30ms frames, got {frame_ms}")

        self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._silence_frames_needed = max(1, silence_ms // frame_ms)
        self._buffer = bytearray()      # the utterance being accumulated
        self._pending = bytearray()     # partial frame carried between pushes
        self._silence_run = 0
        self._speech_started = False

    @property
    def frame_bytes(self) -> int:
        """Exact byte length of one acceptable frame (16-bit mono)."""
        return int(self._sample_rate * self._frame_ms / 1000) * 2

    @property
    def speech_active(self) -> bool:
        """True while an utterance is in progress.

        The server watches the rising edge of this to detect the caller
        speaking over the agent, which triggers barge-in.
        """
        return self._speech_started

    def push(self, pcm_chunk: bytes) -> bytes | None:
        """Feed a PCM chunk of any length.

        Returns the completed utterance on end-of-speech, else None. Audio
        beyond a completed utterance stays buffered for the next call.
        """
        if len(pcm_chunk) % 2:
            raise ValueError(
                f"Expected 16-bit PCM (an even byte count), got {len(pcm_chunk)}"
            )

        self._pending.extend(pcm_chunk)

        while len(self._pending) >= self.frame_bytes:
            frame = bytes(self._pending[:self.frame_bytes])
            del self._pending[:self.frame_bytes]
            utterance = self._push_frame(frame)
            if utterance is not None:
                # Any remaining audio stays in _pending and is processed on
                # the next call, so nothing is dropped mid-chunk.
                return utterance
        return None

    def _push_frame(self, pcm_chunk: bytes) -> bytes | None:
        is_speech = self._vad.is_speech(pcm_chunk, self._sample_rate)

        if is_speech:
            self._speech_started = True
            self._silence_run = 0
            self._buffer.extend(pcm_chunk)
            return None

        if not self._speech_started:
            return None  # leading silence, nothing to accumulate yet

        # Trailing silence is kept in the buffer: Whisper transcribes better
        # with a little padding after the final word.
        self._buffer.extend(pcm_chunk)
        self._silence_run += 1

        if self._silence_run >= self._silence_frames_needed:
            utterance = bytes(self._buffer)
            self._reset()
            return utterance
        return None

    def _reset(self) -> None:
        self._buffer = bytearray()
        self._silence_run = 0
        self._speech_started = False
