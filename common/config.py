# config.py
import os
from dataclasses import dataclass

@dataclass
class Config:
    DEBUG: bool = os.getenv("DEBUG", "True") == "True"
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = 8000
    MODEL_PATH: str = os.getenv("MODEL_PATH", "/mnt/s/NLP/LocalModel/qwen2-0.5b/")
    TOKENIZER_PATH: str = os.getenv("TOKENIZER_PATH", "/mnt/s/NLP/LocalModel/qwen2-0.5b/")
    REDIS_HOST: str = "127.0.0.1"
    REDIS_PORT: int = 6379
    REDIS_MAX_CONNECTIONS: int = 20

