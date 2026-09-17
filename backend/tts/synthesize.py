"""Text-to-speech via Meta MMS-TTS.

MMS-TTS is VITS-based — non-autoregressive, so a clause is one forward pass
rather than a token-by-token decode. That is what makes it viable on CPU,
which matters here because the GPU is reserved for the LLM.

Chosen over Piper (no Marathi voice) and AI4Bharat Indic-TTS (heavier, more
GPU-leaning): the hard constraints were CPU-native *and* Indic coverage, and
MMS ships per-language checkpoints for 1,100+ languages.

Checkpoints are loaded lazily and cached — each is ~150MB, so reloading one
per clause would make streaming impossible.
"""
import io

import scipy.io.wavfile
import torch
from transformers import AutoTokenizer, VitsModel

DEFAULT_LANGUAGE_MODEL_MAP = {
    "en": "facebook/mms-tts-eng",
    "hi": "facebook/mms-tts-hin",
    "mr": "facebook/mms-tts-mar",
}


class Synthesizer:
    def __init__(self, language_model_map: dict[str, str] = None):
        self._map = language_model_map or DEFAULT_LANGUAGE_MODEL_MAP
        self._models = {}
        self._tokenizers = {}

    def _load(self, language: str):
        if language not in self._map:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported: {sorted(self._map)}"
            )
        if language not in self._models:
            checkpoint = self._map[language]
            # CPU only — the GPU is reserved for the LLM.
            self._models[language] = VitsModel.from_pretrained(checkpoint)
            self._tokenizers[language] = AutoTokenizer.from_pretrained(checkpoint)
        return self._models[language], self._tokenizers[language]

    def preload(self, languages: list[str] = None) -> None:
        """Load checkpoints up front.

        Without this the first clause of the first call in each language pays
        a multi-second model load, which lands exactly where latency is most
        visible — the caller's first impression.
        """
        for language in (languages or list(self._map)):
            self._load(language)

    def synthesize(self, text: str, language: str = "en") -> bytes:
        model, tokenizer = self._load(language)
        sampling_rate = model.config.sampling_rate

        # A clause should never be empty, but raising mid-turn is a worse
        # failure than a zero-length clip.
        if not text.strip():
            return self._to_wav(torch.zeros(0).numpy(), sampling_rate)

        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            waveform = model(**inputs).waveform

        return self._to_wav(waveform.squeeze().numpy(), sampling_rate)

    @staticmethod
    def _to_wav(samples, sampling_rate: int) -> bytes:
        buffer = io.BytesIO()
        # MMS emits float32 in roughly [-1, 1]; browsers decode 16-bit PCM
        # far more reliably, so convert rather than shipping float WAV.
        if samples.size:
            peak = max(abs(float(samples.max())), abs(float(samples.min()))) or 1.0
            samples = (samples / peak * 0.95 * 32767).astype("int16")
        else:
            samples = samples.astype("int16")
        scipy.io.wavfile.write(buffer, rate=sampling_rate, data=samples)
        return buffer.getvalue()
