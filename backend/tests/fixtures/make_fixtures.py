"""Generate speech fixtures for the STT tests.

These are SYNTHETIC, produced by MMS-TTS rather than recorded from a person.

That is a deliberate tradeoff, and its limitation should be understood:
synthesized speech is clean, evenly paced and noise-free, so it is *easier*
for Whisper than real speech over a real microphone. These fixtures verify
that the Transcriber wrapper works — that it returns text and a language
code — not that transcription is accurate on realistic audio. Real-voice
accuracy is checked by hand during Phase 7 end-to-end verification.

Replacing any of these with a real recording at the same path is strictly
better for test signal:

    arecord -f S16_LE -r 16000 -c 1 -d 3 backend/tests/fixtures/hello_english.wav

Run: python -m backend.tests.fixtures.make_fixtures
"""
import os

import scipy.io.wavfile
import scipy.signal
import torch
from transformers import AutoTokenizer, VitsModel

TARGET_RATE = 16000
FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))

FIXTURES = [
    ("hello_english.wav", "facebook/mms-tts-eng",
     "hello, do you have headphones"),
    ("hello_hindi.wav", "facebook/mms-tts-hin",
     "क्या आपके पास हेडफ़ोन हैं"),
    ("hello_marathi.wav", "facebook/mms-tts-mar",
     "तुमच्याकडे हेडफोन आहेत का"),
]


def synthesize(checkpoint: str, text: str):
    model = VitsModel.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        waveform = model(**inputs).waveform
    return waveform.squeeze().numpy(), model.config.sampling_rate


def to_16k_pcm16(samples, source_rate: int):
    if source_rate != TARGET_RATE:
        n_target = int(len(samples) * TARGET_RATE / source_rate)
        samples = scipy.signal.resample(samples, n_target)
    peak = max(abs(samples.max()), abs(samples.min())) or 1.0
    normalized = samples / peak * 0.95
    return (normalized * 32767).astype("int16")


if __name__ == "__main__":
    for filename, checkpoint, text in FIXTURES:
        print(f"generating {filename} ...")
        samples, rate = synthesize(checkpoint, text)
        pcm16 = to_16k_pcm16(samples, rate)
        path = os.path.join(FIXTURE_DIR, filename)
        scipy.io.wavfile.write(path, TARGET_RATE, pcm16)
        seconds = len(pcm16) / TARGET_RATE
        print(f"  wrote {path}  ({seconds:.1f}s @ {TARGET_RATE}Hz)")
