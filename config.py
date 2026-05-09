# config.py

import os


from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

from pathlib import Path



BASE_DIR = Path(__file__).parent          # → /home/you/RAG-AI-AGENT/
DATA_DIR = BASE_DIR / "data"              # → /home/you/RAG-AI-AGENT/data/
PAPERS_DIR = DATA_DIR / "papers"          # → /home/you/RAG-AI-AGENT/data/papers/
VECTORSTORE_DIR = BASE_DIR / "vectorstore" # → /home/you/RAG-AI-AGENT/vectorstore/


# ─── Settings class ───────────────────────────────────────────────────────────

class Settings(BaseSettings):

    # ── LLM settings ─────────────────────────────────────────────────────────
    # These names MUST match the keys in your .env file exactly
    # pydantic reads GROQ_API_KEY from .env and puts it here
    groq_api_key: str = Field(..., validation_alias="GROQ_API_KEY")
    # tavily_api_key: str = Field(..., validation_alias="TAVILY_API_KEY")
    groq_model: str = Field("llama-3.3-70b-versatile", validation_alias="MODEL_GPT_GROQ")
    agent_groq_model: str = Field(
        "llama-3.1-8b-instant",
        validation_alias="MODEL_AGENT_GROQ",
    )

    # ── Ollama (local) settings ───────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    embedding_model: str = "all-MiniLM-L6-v2"

    retrieval_top_k: int = 3

    chunk_size: int = 1000
    chunk_overlap: int = 200

    max_agent_iterations: int = 5

    # Read from .env and ignore unrelated environment variables.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


PAPERS_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)