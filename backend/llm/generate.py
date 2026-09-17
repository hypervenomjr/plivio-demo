"""Streaming LLM generation via llama.cpp.

IMPORTANT: `pip install llama-cpp-python` installs a CPU-only build. It
accepts n_gpu_layers=-1 without complaint and then runs every layer on the
CPU at 1-3 tokens/sec, with no error pointing at the cause. Install with:

    CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
        pip install llama-cpp-python --no-cache-dir

Verify with verbose=True and look for "offloaded 29/29 layers to GPU".
"""

DEFAULT_STOP = ["user:", "\nuser"]


class LLMEngine:
    def __init__(self, model_path: str, n_gpu_layers: int = -1, n_ctx: int = 4096):
        from llama_cpp import Llama  # imported lazily so the module can be
                                     # imported (and tested) without the
                                     # CUDA build present

        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,  # -1 offloads every layer to the GPU
            n_ctx=n_ctx,
            verbose=False,
        )

    def generate(self, prompt: str, stop: list[str] = None, max_tokens: int = 200):
        """Yield text fragments as they are generated."""
        stream = self._llm(
            prompt,
            max_tokens=max_tokens,
            stop=stop or DEFAULT_STOP,
            stream=True,
            # Low temperature: this is retrieval-grounded factual answering,
            # not creative writing.
            temperature=0.3,
        )
        for chunk in stream:
            yield chunk["choices"][0]["text"]
