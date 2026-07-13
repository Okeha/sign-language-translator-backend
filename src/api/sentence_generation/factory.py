"""Factory for selecting the sentence-generation LLM backend.

Choose between a local on-device HuggingFace model and a remote cloud model
(OpenRouter) via the LLM_BACKEND config / env var. Imports are done lazily so
that only the dependencies for the selected backend are loaded (e.g. we don't
import `transformers` when running in cloud mode).
"""

import logging

from src.api.config import config

logger = logging.getLogger(__name__)

_LOCAL_ALIASES = {"local", "hf", "huggingface", "transformers"}
_CLOUD_ALIASES = {"cloud", "openrouter", "remote", "api"}


def create_sentence_service():
    """Instantiate the sentence service for the configured LLM_BACKEND.

    Returns the service instance, or raises ValueError for an unknown backend.
    """
    backend = (config.LLM_BACKEND or "cloud").strip().lower()

    if backend in _LOCAL_ALIASES:
        logger.info(
            f"🧠 LLM backend: LOCAL — loading '{config.LLM_MODEL_NAME}' onto the GPU/CPU"
        )
        from src.api.sentence_generation.sentence_service_local import LocalSentenceService
        return LocalSentenceService()

    if backend in _CLOUD_ALIASES:
        logger.info(
            f"☁️  LLM backend: CLOUD ({config.LLM_PROVIDER}) — remote API, no local weights"
        )
        from src.api.sentence_generation.sentence_service import CloudSentenceService
        return CloudSentenceService()

    raise ValueError(
        f"Unknown LLM_BACKEND '{backend}'. "
        f"Expected one of: {sorted(_LOCAL_ALIASES | _CLOUD_ALIASES)}."
    )
