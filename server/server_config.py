# config.py
import os
from dataclasses import dataclass

@dataclass
class Config:
    DEBUG: bool = os.getenv("DEBUG", "True") == "True"
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", 8000))
    MODEL_PATH: str = os.getenv("MODEL_PATH", "/mnt/s/NLP/LocalModel/qwen2-0.5b/")
