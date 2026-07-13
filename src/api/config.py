"""Configuration module for Sign Language Detection API"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root (backend/) — resolve here so .env is found regardless of the
# working directory the server is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load .env before any os.getenv() calls below, otherwise every value would
# silently fall back to its hardcoded default.
load_dotenv(PROJECT_ROOT / ".env")


class Config:
    """Application configuration with environment variable overrides"""

    # Project paths
    PROJECT_ROOT = PROJECT_ROOT
    # Model source: a HuggingFace Hub repo ID (e.g. "owner/videomae-asl-finetuned")
    # or a local directory path. from_pretrained() auto-downloads + caches Hub repos
    # on first run, so no manual model placement is needed for a fresh setup.
    # TODO: replace the placeholder below with your actual public HF repo ID.
    MODEL_PATH = os.getenv(
        "MODEL_PATH",
        "okeha/videomae-sign-language-gloss-detector"  # public HF repo — override with MODEL_PATH env var
    )
    
    # Server settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # CORS settings - comma-separated list of allowed origins
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")
    
    # Inference settings
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.0))
    NUM_FRAMES_TO_SAMPLE = int(os.getenv("NUM_FRAMES_TO_SAMPLE", 16))
    
    # Connection limits
    MAX_CONNECTIONS_PER_IP = int(os.getenv("MAX_CONNECTIONS_PER_IP", 5))
    SESSION_TIMEOUT_HOURS = int(os.getenv("SESSION_TIMEOUT_HOURS", 2))
    
    # Gloss buffer settings
    MAX_GLOSSES_PER_SESSION = int(os.getenv("MAX_GLOSSES_PER_SESSION", 50))
    DEDUPLICATE_CONSECUTIVE = os.getenv("DEDUPLICATE_CONSECUTIVE", "true").lower() == "true"
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # LLM Settings for Sentence Generation.
    #
    # LLM_BACKEND selects HOW sentence/chat generation is served:
    #   "local" — load a local HuggingFace model (LLM_MODEL_NAME) onto the GPU/CPU.
    #   "cloud" — call a remote model over the OpenRouter API (OPENROUTER_MODEL).
    # ("openrouter"/"remote" are accepted as aliases for "cloud".)
    LLM_BACKEND = os.getenv("LLM_BACKEND", "cloud").strip().lower()

    # --- Local backend (LLM_BACKEND=local) ---
    # HuggingFace Hub repo ID or local path for the on-device model.
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3.5-4B")

    # --- Cloud backend (LLM_BACKEND=cloud) ---
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter")
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

    LLM_DEVICE = os.getenv("LLM_DEVICE", "auto")  # auto, cuda, cpu
    LLM_MAX_LENGTH = int(os.getenv("LLM_MAX_LENGTH", 100))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.5))
    LLM_USE_QUANTIZATION = os.getenv("LLM_USE_QUANTIZATION", "true").lower() == "true"
    
    # HuggingFace Token
    # Update here with hugging face token
    HF_TOKEN = os.getenv("HF_TOKEN","")

    
    # Prompts file path
    PROMPTS_FILE_PATH = os.getenv(
        "PROMPTS_FILE_PATH",
        str(PROJECT_ROOT / "src/api/sentence_generation/prompts.yml")
    )


# Singleton instance
config = Config()
