"""Colab entrypoint: wires the real models into the server.

Not importable/runnable locally without a GPU and llama-cpp-python's CUDA
build - see backend/colab_notebook.md for the full startup sequence.
"""
import os

import uvicorn

from backend.llm.generate import LLMEngine
from backend.rag.store import InventoryStore
from backend.server.app import create_app
from backend.stt.transcribe import Transcriber
from backend.stt.vad import UtteranceSegmenter
from backend.tts.synthesize import Synthesizer

LLM_GGUF_PATH = os.environ.get(
    "LLM_GGUF_PATH", "/content/qwen2.5-7b-instruct-q4_k_m.gguf"
)

# Set STT_DEVICE=cuda to move Whisper onto the GPU. Default is cpu, which
# keeps the GPU entirely for the LLM; measure both with
# backend/stt/benchmark.py and pick based on the numbers, not design intent.
STT_DEVICE = os.environ.get("STT_DEVICE", "cpu")

synthesizer = Synthesizer()
# Loads all three MMS checkpoints up front (~2-3s each) so the cost lands at
# startup rather than on the caller's first clause in each language.
synthesizer.preload()

app = create_app(
    segmenter_factory=lambda: UtteranceSegmenter(),
    transcriber=Transcriber(model_size="small", device=STT_DEVICE),
    inventory_store=InventoryStore(persist_dir="backend/rag/chroma_db"),
    llm_engine=LLMEngine(model_path=LLM_GGUF_PATH, n_gpu_layers=-1),
    synthesizer=synthesizer,
    history_db_path="backend/history/history.db",
    scratch_audio_path="/tmp/scratch_utterance.wav",
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
