"""Speech-to-text via faster-whisper.

Two behaviours here are defensive rather than incidental, and both were added
after observing them fail:

1. **Whisper hallucinates on non-speech.** Fed pure silence it returns "you";
   fed low-level noise it has returned a Korean news sign-off and a Spanish
   word. Left unguarded, a cough or a door slam that trips VAD becomes a
   spurious LLM turn. `vad_filter` drops non-speech before decoding, and
   segments whose no-speech probability is high are discarded.

2. **Detected language is clamped to the languages we can actually speak.**
   Whisper can return any of ~99 language codes. The TTS stage has voices for
   three. Returning "ko" downstream would raise mid-turn, so an unsupported
   detection falls back to English — the caller gets an answer rather than a
   dropped call.
"""
from faster_whisper import WhisperModel

# Must stay in step with the TTS voice map and the prompt builder's languages.
SUPPORTED_LANGUAGES = ("en", "hi", "mr")
FALLBACK_LANGUAGE = "en"

# Above this probability a segment is treated as non-speech and dropped.
NO_SPEECH_THRESHOLD = 0.6


class Transcriber:
    def __init__(self, model_size: str = "small", device: str = "cpu",
                 compute_type: str = None):
        # CPU is the default so the GPU stays free for the LLM, but this is a
        # parameter rather than a constant: Whisper is the slowest stage in the
        # pipeline and moving it to CUDA costs only ~1.5GB of VRAM against a
        # 15GB budget. backend/stt/benchmark.py measures both.
        self._device = device
        self._compute_type = self.resolve_compute_type(device, compute_type)
        self._model = WhisperModel(model_size, device=device,
                                   compute_type=self._compute_type)

    @staticmethod
    def resolve_compute_type(device: str, compute_type: str | None) -> str:
        if compute_type is not None:
            return compute_type
        return "float16" if device == "cuda" else "int8"

    @staticmethod
    def clamp_language(language: str) -> str:
        return language if language in SUPPORTED_LANGUAGES else FALLBACK_LANGUAGE

    def transcribe(self, audio_path: str) -> dict:
        try:
            segments, info = self._model.transcribe(
                audio_path,
                # Drop non-speech before decoding — the main hallucination guard.
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                # Stops the model carrying an invented phrase forward into
                # subsequent segments.
                condition_on_previous_text=False,
            )

            kept = [
                segment.text.strip()
                for segment in segments
                if getattr(segment, "no_speech_prob", 0.0) < NO_SPEECH_THRESHOLD
            ]
        except ValueError:
            # faster-whisper 1.0.3 raises "max() arg is an empty sequence" when
            # the VAD filter removes every segment, instead of returning an
            # empty result. That case means exactly one thing — the clip held
            # no speech — so report it as an empty transcript.
            return {"text": "", "language": FALLBACK_LANGUAGE}

        text = " ".join(part for part in kept if part).strip()

        # With no speech, the detected language is meaningless — don't let a
        # noise-derived code pick a voice.
        language = self.clamp_language(info.language) if text else FALLBACK_LANGUAGE
        return {"text": text, "language": language}
