"""Measure transcription latency on CPU vs GPU.

The design reserves the GPU for the LLM, but Whisper is the slowest stage in
the pipeline. This script replaces an argument with a number: run it on the
target hardware and choose STT_DEVICE from the result.

Run: python -m backend.stt.benchmark
"""
import statistics
import time
import wave

from backend.stt.transcribe import Transcriber

FIXTURE = "backend/tests/fixtures/hello_english.wav"
RUNS = 5


def clip_seconds(path: str) -> float:
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def measure(device: str) -> list[float]:
    transcriber = Transcriber(model_size="small", device=device)
    transcriber.transcribe(FIXTURE)  # warm-up, excluded from results
    timings = []
    for _ in range(RUNS):
        start = time.perf_counter()
        transcriber.transcribe(FIXTURE)
        timings.append(time.perf_counter() - start)
    return timings


if __name__ == "__main__":
    duration = clip_seconds(FIXTURE)
    print(f"clip: {duration:.1f}s of audio, {RUNS} runs after a warm-up\n")

    for device in ["cpu", "cuda"]:
        try:
            timings = measure(device)
            mean = statistics.mean(timings)
            # Real-time factor: below 1.0 means faster than the audio's length.
            print(f"{device:5}  mean {mean:6.2f}s   min {min(timings):6.2f}s   "
                  f"max {max(timings):6.2f}s   RTF {mean / duration:.2f}")
        except Exception as exc:
            print(f"{device:5}  unavailable ({type(exc).__name__}: {exc})")

    print("\nRecord both lines in docs/interview-defense.md and set STT_DEVICE "
          "in backend/main.py accordingly.")
