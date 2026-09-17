"""Manual integration tests for the LLM wrapper.

These need the real GGUF weights and a CUDA-enabled llama-cpp-python build,
so they are skipped unless LLM_GGUF_PATH is set. Run them on Colab:

    LLM_GGUF_PATH=/content/qwen2.5-7b-instruct-q4_k_m.gguf \
        pytest backend/tests/test_generate_manual.py -v -s
"""
import os
import time

import pytest

MODEL_PATH = os.environ.get("LLM_GGUF_PATH", "")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH, reason="Set LLM_GGUF_PATH to run these manual tests"
)


@pytest.fixture(scope="module")
def engine():
    from backend.llm.generate import LLMEngine
    return LLMEngine(model_path=MODEL_PATH)


def test_generate_produces_nonempty_text(engine):
    chunks = list(engine.generate("assistant:", max_tokens=20))
    assert "".join(chunks).strip() != ""


def test_generate_streams_more_than_one_chunk(engine):
    chunks = list(engine.generate("Say three short sentences.\nassistant:",
                                  max_tokens=40))
    assert len(chunks) > 1


def test_generation_is_fast_enough_to_be_on_gpu(engine):
    """Guards against the silent CPU-only llama-cpp-python build.

    A CUDA build does 25-40 tok/s on a T4; a CPU build does 1-3. Anything
    under 10 tok/s means the GPU is not actually being used.
    """
    start = time.perf_counter()
    chunks = list(engine.generate("Count to twenty.\nassistant:", max_tokens=60))
    elapsed = time.perf_counter() - start
    rate = len(chunks) / elapsed
    print(f"\ngeneration rate: {rate:.1f} tok/s")
    assert rate > 10, (
        f"{rate:.1f} tok/s suggests a CPU-only llama-cpp-python build. "
        "Reinstall with CMAKE_ARGS=\"-DGGML_CUDA=on\"."
    )


def test_answers_from_retrieved_listings_in_english(engine):
    from backend.llm.prompt import build_prompt

    retrieved = [{
        "id": "1", "name": "Juniper Cordless Blender", "description": "...",
        "price": 49.99, "stock": 12, "category": "Home & Kitchen", "score": 0.1,
    }]
    prompt = build_prompt("do you have a cordless blender", retrieved, [])
    answer = "".join(engine.generate(prompt, max_tokens=80))
    print(f"\nEN answer: {answer}")
    assert "blender" in answer.lower()


def test_replies_in_hindi_when_asked_in_hindi(engine):
    """The multilingual feature lives or dies here.

    Listings are English; if the prompt's language instruction is ignored the
    model answers in English and the Hindi TTS voice renders it as noise.
    """
    from backend.llm.prompt import build_prompt

    retrieved = [{
        "id": "1", "name": "Juniper Cordless Blender", "description": "...",
        "price": 49.99, "stock": 12, "category": "Home & Kitchen", "score": 0.1,
    }]
    prompt = build_prompt("क्या आपके पास ब्लेंडर है", retrieved, [], language="hi")
    answer = "".join(engine.generate(prompt, max_tokens=80))
    print(f"\nHI answer: {answer}")
    # Devanagari block: the reply must actually be in Hindi script.
    assert any("ऀ" <= ch <= "ॿ" for ch in answer), (
        f"Expected a Devanagari reply, got: {answer!r}"
    )


def test_says_no_match_when_retrieval_is_empty(engine):
    from backend.llm.prompt import build_prompt

    prompt = build_prompt("do you sell live elephants", [], [])
    answer = "".join(engine.generate(prompt, max_tokens=80))
    print(f"\nempty-retrieval answer: {answer}")
    assert answer.strip() != ""
