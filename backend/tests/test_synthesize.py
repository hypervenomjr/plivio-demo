import io
import wave

import pytest

from backend.tts.synthesize import Synthesizer


@pytest.fixture(scope="module")
def synth():
    return Synthesizer()


def _wav_info(audio: bytes):
    with wave.open(io.BytesIO(audio), "rb") as wf:
        return {
            "channels": wf.getnchannels(),
            "sample_width": wf.getsampwidth(),
            "rate": wf.getframerate(),
            "seconds": wf.getnframes() / wf.getframerate(),
        }


def test_synthesize_returns_wav_bytes_english(synth):
    audio = synth.synthesize("hello, we have that in stock", language="en")
    assert audio[:4] == b"RIFF"
    assert len(audio) > 1000


def test_synthesize_returns_wav_bytes_hindi(synth):
    audio = synth.synthesize("नमस्ते, यह स्टॉक में है", language="hi")
    assert audio[:4] == b"RIFF"
    assert len(audio) > 1000


def test_synthesize_returns_wav_bytes_marathi(synth):
    audio = synth.synthesize("नमस्कार, हे स्टॉकमध्ये आहे", language="mr")
    assert audio[:4] == b"RIFF"
    assert len(audio) > 1000


def test_synthesize_unknown_language_raises(synth):
    with pytest.raises(ValueError):
        synth.synthesize("test", language="xx")


def test_output_is_mono_16bit(synth):
    """The browser decodes this directly; keep the format predictable."""
    info = _wav_info(synth.synthesize("a short test clause", language="en"))
    assert info["channels"] == 1
    assert info["sample_width"] == 2


def test_longer_text_produces_longer_audio(synth):
    short = _wav_info(synth.synthesize("yes.", language="en"))["seconds"]
    long = _wav_info(synth.synthesize(
        "yes, we have several of those in stock right now, in three colours.",
        language="en"))["seconds"]
    assert long > short


def test_empty_text_returns_empty_audio_without_raising(synth):
    """The clause chunker should never emit empty text, but a crash mid-turn
    is a worse failure than a silent clip if it ever does."""
    audio = synth.synthesize("", language="en")
    assert audio[:4] == b"RIFF"
    assert _wav_info(audio)["seconds"] == 0


def test_models_are_cached_between_calls(synth):
    """Each MMS checkpoint is ~150MB; reloading per clause would be fatal
    for a streaming pipeline."""
    synth.synthesize("first", language="en")
    loaded = dict(synth._models)
    synth.synthesize("second", language="en")
    assert synth._models["en"] is loaded["en"]
