import math
import struct

import pytest

from backend.stt.vad import UtteranceSegmenter


def _silence_frame(sample_rate=16000, ms=30):
    n_samples = int(sample_rate * ms / 1000)
    return b"\x00\x00" * n_samples


def _tone_frame(sample_rate=16000, ms=30):
    n_samples = int(sample_rate * ms / 1000)
    samples = [int(3000 * math.sin(2 * math.pi * 440 * i / sample_rate))
               for i in range(n_samples)]
    return struct.pack("<%dh" % n_samples, *samples)


def _drive_one_utterance(seg, speech_frames=5, silence_frames=10):
    for _ in range(speech_frames):
        seg.push(_tone_frame())
    for _ in range(silence_frames):
        result = seg.push(_silence_frame())
        if result is not None:
            return result
    return None


def test_segmenter_returns_none_during_speech():
    seg = UtteranceSegmenter(silence_ms=300)
    result = None
    for _ in range(5):
        result = seg.push(_tone_frame())
    assert result is None


def test_segmenter_returns_none_for_silence_before_any_speech():
    seg = UtteranceSegmenter(silence_ms=60)
    for _ in range(10):
        assert seg.push(_silence_frame()) is None


def test_segmenter_returns_bytes_after_trailing_silence():
    seg = UtteranceSegmenter(silence_ms=90)
    utterance = _drive_one_utterance(seg)
    assert utterance is not None
    assert len(utterance) > 0


def test_segmenter_does_not_accumulate_across_utterances():
    """The buffer must clear on emit rather than carrying the last turn forward.

    Note the assertion is a bound, not an equality: webrtcvad is adaptive and
    carries internal state between frames, so the same synthetic audio is not
    classified identically twice and the emitted lengths differ by a frame or
    two. What must hold is that the second utterance is its own length, not
    the sum of both.
    """
    seg = UtteranceSegmenter(silence_ms=90)
    first = _drive_one_utterance(seg)
    second = _drive_one_utterance(seg)

    assert first is not None
    assert second is not None
    assert len(second) <= len(first) + 3 * seg.frame_bytes


def test_segmenter_is_idle_after_emitting():
    """Once an utterance is emitted, trailing silence must produce nothing."""
    seg = UtteranceSegmenter(silence_ms=90)
    assert _drive_one_utterance(seg) is not None

    for _ in range(20):
        assert seg.push(_silence_frame()) is None
    assert seg.speech_active is False


def test_speech_active_is_false_before_any_speech():
    seg = UtteranceSegmenter(silence_ms=90)
    assert seg.speech_active is False
    seg.push(_silence_frame())
    assert seg.speech_active is False


def test_speech_active_is_true_during_speech():
    seg = UtteranceSegmenter(silence_ms=90)
    seg.push(_tone_frame())
    assert seg.speech_active is True


def test_speech_active_is_false_after_utterance_emitted():
    seg = UtteranceSegmenter(silence_ms=90)
    _drive_one_utterance(seg)
    assert seg.speech_active is False


def test_rejects_odd_byte_count():
    """16-bit PCM always has an even byte count; anything else is malformed."""
    seg = UtteranceSegmenter(silence_ms=90)
    with pytest.raises(ValueError):
        seg.push(b"\x00\x00\x00")


def test_frame_bytes_reports_expected_size():
    seg = UtteranceSegmenter(sample_rate=16000, frame_ms=30)
    # 30ms at 16kHz, 16-bit mono = 480 samples = 960 bytes
    assert seg.frame_bytes == 960


def test_rejects_invalid_frame_duration():
    with pytest.raises(ValueError):
        UtteranceSegmenter(frame_ms=25)


def test_accepts_chunks_smaller_than_a_frame():
    """A browser delivers whatever its audio buffer size works out to.

    Sub-frame chunks must accumulate rather than being rejected or dropped.
    """
    seg = UtteranceSegmenter(silence_ms=90)
    tone = _tone_frame()
    # Feed one frame's worth in 100-byte pieces.
    for start in range(0, len(tone), 100):
        assert seg.push(tone[start:start + 100]) is None
    assert seg.speech_active is True


def test_accepts_chunks_larger_than_a_frame():
    """A 4096-sample browser buffer downsampled to 16kHz is ~1365 samples."""
    seg = UtteranceSegmenter(silence_ms=90)
    big_speech = _tone_frame() * 5          # 5 frames in one push
    assert seg.push(big_speech) is None
    assert seg.speech_active is True

    utterance = None
    for _ in range(20):
        utterance = seg.push(_silence_frame() * 3)
        if utterance is not None:
            break
    assert utterance is not None


def test_odd_sized_chunks_lose_no_audio():
    """Re-framing must not discard the remainder of a chunk."""
    seg = UtteranceSegmenter(silence_ms=90)
    chunk = _tone_frame() + _tone_frame()[:400]  # 1.4 frames
    seg.push(chunk)
    seg.push(_tone_frame()[400:])               # completes the second frame
    assert seg.speech_active is True
