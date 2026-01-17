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
    llm_provider: Literal["openai", "local"] = Field(default="local", alias="LLM_PROVIDER")
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
    chunking_strategy: Literal["logical", "file"] = Field(
        default="logical", alias="CHUNKING_STRATEGY"
    )

    # Retrieval enhancements
    enable_rerank: bool = Field(default=False, alias="ENABLE_RERANK")
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANK_MODEL"
    )
    rerank_top_n: int = Field(default=10, alias="RERANK_TOP_N")
    enable_hybrid: bool = Field(default=False, alias="ENABLE_HYBRID")
    hybrid_alpha: float = Field(default=0.7, alias="HYBRID_ALPHA")

    # Storage
    storage_path: str = Field(default="./storage", alias="STORAGE_PATH")
    vector_store_provider: Literal["chroma", "qdrant", "pinecone", "weaviate"] = Field(
        default="chroma", alias="VECTOR_STORE_PROVIDER"
    )
    vector_store_collection: str = Field(default="architext_db", alias="VECTOR_STORE_COLLECTION")
    qdrant_url: Optional[str] = Field(default=None, alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")
    pinecone_api_key: Optional[str] = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_index_name: Optional[str] = Field(default=None, alias="PINECONE_INDEX_NAME")
    weaviate_url: Optional[str] = Field(default=None, alias="WEAVIATE_URL")
    weaviate_api_key: Optional[str] = Field(default=None, alias="WEAVIATE_API_KEY")

    # Server security & limits
    rate_limit_per_minute: int = Field(default=120, alias="RATE_LIMIT_PER_MINUTE")
    allowed_source_roots: Optional[str] = Field(default=None, alias="ALLOWED_SOURCE_ROOTS")
    allowed_storage_roots: Optional[str] = Field(default=None, alias="ALLOWED_STORAGE_ROOTS")
    task_store_path: str = Field(default="~/.architext/task_store.json", alias="TASK_STORE_PATH")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


def load_settings(env_file: Optional[str] = None) -> ArchitextSettings:
    """Load settings from env/.env with sensible defaults."""
    init_kwargs: dict = {"_env_file": env_file} if env_file else {}
    return ArchitextSettings(**init_kwargs)
