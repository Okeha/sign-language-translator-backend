"""Configuration module for Sign Language Detection API"""

import os
from pathlib import Path


class Config:
    """Application configuration with environment variable overrides"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    MODEL_PATH = os.getenv(
        "MODEL_PATH",
        str(PROJECT_ROOT / "src/model/finetune/videomae/video_mae_finetuned_final")
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
    
    # LLM Settings for Sentence Generation
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3.5-4B")
    # LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3-4B")
    # LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3-1.7B")
    # LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "microsoft/Phi-4-mini-reasoning")
    # LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "microsoft/Phi-4-mini-reasoning")


    LLM_DEVICE = os.getenv("LLM_DEVICE", "auto")  # auto, cuda, cpu
    LLM_MAX_LENGTH = int(os.getenv("LLM_MAX_LENGTH", 100))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.5))
    LLM_USE_QUANTIZATION = os.getenv("LLM_USE_QUANTIZATION", "true").lower() == "true"
    
    # HuggingFace Token
    # Update here with hugging face token
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    
    # Prompts file path
    PROMPTS_FILE_PATH = os.getenv(
        "PROMPTS_FILE_PATH",
        str(PROJECT_ROOT / "src/api/sentence_generation/prompts.yml")
    )


# Singleton instance
config = Config()
