"""
LLM loader and LangChain wrapper for NUST Smart Banker.

Loads Qwen2.5-3B-Instruct with 4-bit quantisation (bitsandbytes) to fit
within 6 GB VRAM, then wraps it as a LangChain BaseLLM so it can plug
directly into the RAG chain.

Public API:
    get_llm()         → LangChain-compatible LLM singleton
    generate(prompt)  → raw string generation (bypass LangChain)
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, List, Optional

from configs.settings import (
    HF_TOKEN,
    LLM_LOAD_IN_4BIT,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL_NAME,
    LLM_REPETITION_PENALTY,
    LLM_TEMPERATURE,
)

logger = logging.getLogger(__name__)


# ─── Lazy imports (heavy; only load when first needed) ────────────────────────


def _load_model_and_tokenizer():
    """Load the quantised model and tokeniser.  Called once by the singleton."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TextIteratorStreamer,
    )


    # token=None is treated as "no token" by transformers, so passing an empty
    # string would cause a warning — normalise to None when unset.
    _hf_token: str | None = HF_TOKEN or None

    logger.info("Loading tokeniser for %s …", LLM_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_NAME,
        trust_remote_code=True,
        token=_hf_token,
    )

    if LLM_LOAD_IN_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        logger.info("Loading model in 4-bit (NF4) quantisation …")
    else:
        bnb_config = None
        logger.info("Loading model in full precision …")

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",  # automatically assigns layers to GPU/CPU
        trust_remote_code=True,
        torch_dtype=torch.float16,
        token=_hf_token,
    )
    # Double check if we can move entirely to CUDA if it wasn't assigned
    if torch.cuda.is_available() and str(model.device) == "cpu":
        logger.info("Manually moving model to CUDA...")
        model = model.to("cuda")

    model.eval()
    logger.info("Model loaded successfully.")
    return model, tokenizer


# ─── LangChain LLM Wrapper ────────────────────────────────────────────────────


class QwenLLM:
    """
    Thin wrapper around Qwen2.5-3B-Instruct that exposes the interface
    expected by the RAG chain:

        llm(prompt: str) -> str
        llm.stream(prompt: str) -> Iterator[str]

    We intentionally avoid subclassing langchain's BaseLLM to keep the
    dependency surface small and stay compatible across LangChain versions.
    """

    def __init__(self) -> None:
        from configs.settings import (
            LLM_MAX_NEW_TOKENS,
            LLM_REPETITION_PENALTY,
            LLM_TEMPERATURE,
        )

        self.max_new_tokens = LLM_MAX_NEW_TOKENS
        self.temperature = LLM_TEMPERATURE
        self.repetition_penalty = LLM_REPETITION_PENALTY
        self._model = None
        self._tokenizer = None

    # ── Lazy load ─────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model, self._tokenizer = _load_model_and_tokenizer()

    # ── Core generation ───────────────────────────────────────────────────────

    def __call__(self, prompt: str, **kwargs) -> str:
        """Generate a response for *prompt* and return it as a plain string."""
        return self.generate(prompt, **kwargs)

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """
        Run inference and return the assistant's reply as a string.

        *prompt* is expected to be a fully formatted chat string (with system
        and user turns already applied).  The RAG chain builds this via
        `build_prompt()` in rag_chain.py.
        """
        import torch

        self._ensure_loaded()

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3072,  # leave headroom for generation in 32K context
        ).to(self._model.device)

        n_tokens = max_new_tokens or self.max_new_tokens

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=n_tokens,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (exclude the prompt)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        response = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()

    def stream(self, prompt: str) -> Iterator[str]:
        """
        Streaming generation using HuggingFace TextIteratorStreamer.
        Yields token strings one-by-one as they are generated.
        Used by the Streamlit UI to display progressive responses.
        """
        import threading
        import torch
        from transformers import TextIteratorStreamer

        self._ensure_loaded()

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3072,
        ).to(self._model.device)

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "streamer": streamer,
        }

        # Run generation in a background thread so we can yield from the
        # streamer in the main thread
        thread = threading.Thread(
            target=self._model.generate,
            kwargs=generation_kwargs,
            daemon=True,
        )
        thread.start()

        for token_text in streamer:
            yield token_text

        thread.join()

    # ── LangChain compatibility shim ──────────────────────────────────────────

    def invoke(self, input: Any, **kwargs) -> str:
        """LangChain LCEL compatible invoke method."""
        prompt = input if isinstance(input, str) else str(input)
        return self.generate(prompt)

    @property
    def _llm_type(self) -> str:
        return "qwen2.5-3b-instruct"


# ─── Singleton ────────────────────────────────────────────────────────────────

_llm_instance: QwenLLM | None = None


def get_llm() -> QwenLLM:
    """Return (or lazily create) the global QwenLLM singleton."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = QwenLLM()
    return _llm_instance
