"""Configuration management for Architext.

This module centralizes environment-driven settings using Pydantic
BaseSettings so we can support .env files and type-safe defaults.
"""
from typing import Optional, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ArchitextSettings(BaseSettings):
    """Top-level configuration for LLM, embeddings, and retrieval."""

    # LLM configuration
    llm_provider: Literal["openai", "gemini", "local", "anthropic"] = Field(
        default="local", alias="LLM_PROVIDER"
    )
    llm_model: str = Field(default="local-model", alias="LLM_MODEL")
    openai_api_base: str = Field(default="http://127.0.0.1:5000/v1", alias="OPENAI_API_BASE")
    openai_api_key: str = Field(default="local", alias="OPENAI_API_KEY")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    llm_max_tokens: Optional[int] = Field(default=None, alias="LLM_MAX_TOKENS")
    system_prompt: Optional[str] = Field(default=None, alias="SYSTEM_PROMPT")

    # Embeddings
    embedding_provider: Literal["huggingface", "openai"] = Field(
        default="huggingface", alias="EMBEDDING_PROVIDER"
    )
    embedding_model_name: str = Field(
        default="sentence-transformers/all-mpnet-base-v2", alias="EMBEDDING_MODEL_NAME"
    )
    embedding_cache_dir: str = Field(default="models_cache", alias="EMBEDDING_CACHE_DIR")

    # Retrieval knobs
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    top_k: int = Field(default=5, alias="TOP_K")

    # Storage
    storage_path: str = Field(default="./storage", alias="STORAGE_PATH")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


def load_settings(env_file: Optional[str] = None) -> ArchitextSettings:
    """Load settings from env/.env with sensible defaults."""
    init_kwargs = {"_env_file": env_file} if env_file else {}
    return ArchitextSettings(**init_kwargs)
