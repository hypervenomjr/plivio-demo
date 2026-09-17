"""Transcriber tests.

The fixtures are synthetic (see fixtures/make_fixtures.py). They verify the
wrapper's behaviour — text out, language detected, device handling — not
transcription accuracy on realistic audio, which is checked by hand in
Phase 7.
"""
import os

import pytest

from backend.stt.transcribe import Transcriber

FIXTURE_DIR = "backend/tests/fixtures"
ENGLISH = f"{FIXTURE_DIR}/hello_english.wav"
HINDI = f"{FIXTURE_DIR}/hello_hindi.wav"
MARATHI = f"{FIXTURE_DIR}/hello_marathi.wav"

needs_fixtures = pytest.mark.skipif(
    not os.path.exists(ENGLISH),
    reason="Run python -m backend.tests.fixtures.make_fixtures first",
)


@pytest.fixture(scope="module")
def transcriber():
    return Transcriber(model_size="small", device="cpu")


def test_compute_type_defaults_to_int8_on_cpu():
    assert Transcriber.resolve_compute_type("cpu", None) == "int8"


def test_compute_type_defaults_to_float16_on_cuda():
    assert Transcriber.resolve_compute_type("cuda", None) == "float16"


def test_explicit_compute_type_is_respected():
    assert Transcriber.resolve_compute_type("cpu", "int8_float16") == "int8_float16"


@needs_fixtures
def test_transcribe_returns_expected_keys(transcriber):
    result = transcriber.transcribe(ENGLISH)
    assert set(result.keys()) == {"text", "language"}


@needs_fixtures
def test_transcribe_english(transcriber):
    result = transcriber.transcribe(ENGLISH)
    assert "headphone" in result["text"].lower()
    assert result["language"] == "en"


@needs_fixtures
def test_transcribe_detects_hindi(transcriber):
    result = transcriber.transcribe(HINDI)
    assert result["language"] == "hi"
    assert result["text"].strip() != ""


@needs_fixtures
def test_transcribe_detects_marathi(transcriber):
    result = transcriber.transcribe(MARATHI)
    # Marathi is the weakest language in Whisper's training data and is
    # sometimes labelled as Hindi, which shares the Devanagari script.
    assert result["language"] in ("mr", "hi")
    assert result["text"].strip() != ""


def test_clamp_language_keeps_supported_languages():
    assert Transcriber.clamp_language("en") == "en"
    assert Transcriber.clamp_language("hi") == "hi"
    assert Transcriber.clamp_language("mr") == "mr"


def test_clamp_language_falls_back_for_unsupported():
    """Whisper can return any of ~99 codes; TTS has three voices.

    Without this clamp a noise burst detected as Korean would reach the
    synthesizer and raise mid-turn.
    """
    assert Transcriber.clamp_language("ko") == "en"
    assert Transcriber.clamp_language("nn") == "en"


def _write_wav(path, samples):
    import scipy.io.wavfile
    scipy.io.wavfile.write(path, 16000, samples)
    return str(path)


@needs_fixtures
def test_transcribe_returns_empty_text_for_silence(transcriber, tmp_path):
    """Whisper hallucinates on silence — unguarded it returns "you".

    The server relies on an empty transcript to skip the turn without calling
    the LLM, so silence must come back empty rather than as invented words.
    """
    import numpy as np

    path = _write_wav(tmp_path / "silence.wav", np.zeros(32000, dtype="int16"))
    assert transcriber.transcribe(path)["text"] == ""


@needs_fixtures
def test_transcribe_does_not_hallucinate_on_noise(transcriber, tmp_path):
    """Unguarded, low-level noise produced a Korean news sign-off."""
    import numpy as np

    rng = np.random.default_rng(seed=0)
    noise = (rng.standard_normal(32000) * 50).astype("int16")
    path = _write_wav(tmp_path / "noise.wav", noise)

    result = transcriber.transcribe(path)
    assert result["text"] == ""
    assert result["language"] in ("en", "hi", "mr")


@needs_fixtures
def test_transcribe_never_reports_an_unsupported_language(transcriber, tmp_path):
    import numpy as np

    rng = np.random.default_rng(seed=1)
    noise = (rng.standard_normal(32000) * 300).astype("int16")
    path = _write_wav(tmp_path / "roomnoise.wav", noise)

    assert transcriber.transcribe(path)["language"] in ("en", "hi", "mr")
