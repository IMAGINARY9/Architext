"""Configuration management for Architext.

This module centralizes environment-driven settings using Pydantic
BaseSettings so we can support .env files and type-safe defaults.
"""
from typing import Optional, Literal

from pydantic import Field, ValidationError
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

    # Advanced / operational defaults
    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    ssh_key: Optional[str] = Field(default=None, alias="SSH_KEY_PATH")

    model_config = SettingsConfigDict(env_file=".env", extra="forbid", populate_by_name=True)


def load_settings(env_file: Optional[str] = None, config_file: Optional[str] = None) -> ArchitextSettings:
    """Load settings from env/.env and auto-detect a JSON config file in standard locations.

    Behavior:
    - Loads env/.env (when provided via --env-file or default .env).
    - If `config_file` is provided it will be loaded. Otherwise, tries these locations in order:
      1) `./architext.config.json` (current working directory)
      2) `~/.architext/config.json` (user config dir)

    This allows users to specify advanced/static defaults without adding CLI flags.
    """
    import json
    from pathlib import Path

    init_kwargs: dict = {"_env_file": env_file} if env_file else {}
    settings = ArchitextSettings(**init_kwargs)

    # Determine config file path
    candidate = None
    if config_file:
        candidate = Path(config_file)
    else:
        cwd_candidate = Path("./architext.config.json").resolve()
        home_candidate = Path.home() / ".architext" / "config.json"
        if cwd_candidate.exists():
            candidate = cwd_candidate
        elif home_candidate.exists():
            candidate = home_candidate

    if candidate:
        try:
            with open(candidate, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            # Create a temporary settings object to validate the config
            temp_settings = ArchitextSettings(**cfg)
            # If validation passes, merge with the main settings
            settings = settings.model_copy(update=cfg)
        except ValidationError as exc:
            # Extract unknown field names from the validation error
            unknown_fields = []
            for error in exc.errors():
                if error.get("type") == "extra_forbidden":
                    field_name = error.get("loc", ["unknown"])[0]
                    unknown_fields.append(field_name)
            
            if unknown_fields:
                known_fields = [field.alias or field_name for field_name, field in ArchitextSettings.model_fields.items()]
                raise RuntimeError(
                    f"Config file '{candidate}' contains unknown settings: {', '.join(unknown_fields)}. "
                    f"Valid settings are: {', '.join(sorted(known_fields))}"
                ) from exc
            else:
                raise RuntimeError(f"Invalid config file '{candidate}': {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to read config file {candidate}: {exc}") from exc

    return settings
