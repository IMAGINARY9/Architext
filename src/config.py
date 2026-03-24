"""Configuration management for Tekturo.

This module exposes a strict, section-first configuration model via
``AppSettings``. Flat legacy settings keys are intentionally rejected to keep
configuration contracts explicit and maintainable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppPathDefaults:
    """Canonical local path defaults used by runtime components."""

    APP_DIRNAME = ".tekturo"
    CONFIG_FILENAME = "tekturo.config.json"
    DEFAULT_COLLECTION = "tekturo_db"

    @classmethod
    def home_dir(cls) -> Path:
        return Path.home() / cls.APP_DIRNAME

    @classmethod
    def cache_dir(cls) -> Path:
        return cls.home_dir() / "cache"

    @classmethod
    def history_dir(cls) -> Path:
        return cls.home_dir() / "history"

    @classmethod
    def pipelines_dir(cls) -> Path:
        return cls.home_dir() / "pipelines"

    @classmethod
    def schedules_dir(cls) -> Path:
        return cls.home_dir() / "schedules"

    @classmethod
    def webhooks_dir(cls) -> Path:
        return cls.home_dir() / "webhooks"

    @classmethod
    def config_dir(cls) -> Path:
        return cls.home_dir() / "config"

    @classmethod
    def config_candidates(cls, explicit: Optional[str] = None) -> List[Path]:
        if explicit:
            return [Path(explicit)]
        return [Path(f"./{cls.CONFIG_FILENAME}").resolve(), cls.home_dir() / "config.json"]

    @classmethod
    def task_store_path(cls) -> str:
        return str(cls.home_dir() / "task_store.json")


_SECTION_SETTINGS_CONFIG = SettingsConfigDict(
    env_file=".env",
    extra="ignore",
    populate_by_name=True,
)


class LLMSettings(BaseSettings):
    llm_provider: Literal["openai", "local"] = Field(default="local", alias="LLM_PROVIDER")
    llm_model: str = Field(default="local-model", alias="LLM_MODEL")
    openai_api_base: str = Field(default="http://127.0.0.1:5000/v1", alias="OPENAI_API_BASE")
    openai_api_key: str = Field(default="local", alias="OPENAI_API_KEY")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    llm_max_tokens: Optional[int] = Field(default=None, alias="LLM_MAX_TOKENS")
    system_prompt: Optional[str] = Field(default=None, alias="SYSTEM_PROMPT")

    model_config = _SECTION_SETTINGS_CONFIG


class EmbeddingSettings(BaseSettings):
    embedding_provider: Literal["huggingface", "openai"] = Field(
        default="huggingface", alias="EMBEDDING_PROVIDER"
    )
    embedding_model_name: str = Field(
        default="sentence-transformers/all-mpnet-base-v2", alias="EMBEDDING_MODEL_NAME"
    )
    embedding_cache_dir: str = Field(default="models_cache", alias="EMBEDDING_CACHE_DIR")

    model_config = _SECTION_SETTINGS_CONFIG


class RetrievalSettings(BaseSettings):
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    top_k: int = Field(default=5, alias="TOP_K")
    chunking_strategy: Literal["logical", "file"] = Field(
        default="logical", alias="CHUNKING_STRATEGY"
    )
    index_max_files: int = Field(default=0, alias="INDEX_MAX_FILES")
    index_include_extensions: Optional[str] = Field(default=None, alias="INDEX_INCLUDE_EXTENSIONS")
    enable_rerank: bool = Field(default=False, alias="ENABLE_RERANK")
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANK_MODEL"
    )
    rerank_top_n: int = Field(default=10, alias="RERANK_TOP_N")
    enable_hybrid: bool = Field(default=True, alias="ENABLE_HYBRID")
    hybrid_alpha: float = Field(default=0.7, alias="HYBRID_ALPHA")

    model_config = _SECTION_SETTINGS_CONFIG


class StorageSettings(BaseSettings):
    storage_path: str = Field(default="./storage", alias="STORAGE_PATH")
    vector_store_provider: Literal["chroma", "qdrant", "pinecone", "weaviate"] = Field(
        default="chroma", alias="VECTOR_STORE_PROVIDER"
    )
    vector_store_collection: str = Field(
        default=AppPathDefaults.DEFAULT_COLLECTION,
        alias="VECTOR_STORE_COLLECTION",
    )
    qdrant_url: Optional[str] = Field(default=None, alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")
    pinecone_api_key: Optional[str] = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_index_name: Optional[str] = Field(default=None, alias="PINECONE_INDEX_NAME")
    weaviate_url: Optional[str] = Field(default=None, alias="WEAVIATE_URL")
    weaviate_api_key: Optional[str] = Field(default=None, alias="WEAVIATE_API_KEY")

    model_config = _SECTION_SETTINGS_CONFIG


class ServerSettings(BaseSettings):
    rate_limit_per_minute: int = Field(default=120, alias="RATE_LIMIT_PER_MINUTE")
    allowed_source_roots: Optional[str] = Field(default=None, alias="ALLOWED_SOURCE_ROOTS")
    allowed_storage_roots: Optional[str] = Field(default=None, alias="ALLOWED_STORAGE_ROOTS")
    task_store_path: str = Field(default_factory=AppPathDefaults.task_store_path, alias="TASK_STORE_PATH")

    model_config = _SECTION_SETTINGS_CONFIG


class RuntimeSettings(BaseSettings):
    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    ssh_key: Optional[str] = Field(default=None, alias="SSH_KEY_PATH")

    model_config = _SECTION_SETTINGS_CONFIG


SECTION_MODELS: Dict[str, Any] = {
    "llm": LLMSettings,
    "embedding": EmbeddingSettings,
    "retrieval": RetrievalSettings,
    "storage": StorageSettings,
    "server": ServerSettings,
    "runtime": RuntimeSettings,
}


def _split_flat_overrides(flat_values: Mapping[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    nested: Dict[str, Dict[str, Any]] = {name: {} for name in SECTION_MODELS}
    unknown: List[str] = []

    for key, value in flat_values.items():
        matched = False
        for section_name, model in SECTION_MODELS.items():
            for field_name, field in model.model_fields.items():
                accepted = {field_name, field.alias}
                if key in accepted:
                    nested[section_name][field_name] = value
                    matched = True
                    break
            if matched:
                break
        if not matched:
            unknown.append(str(key))

    nested = {k: v for k, v in nested.items() if v}
    return nested, unknown


def _merge_nested_overrides(overrides: Mapping[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    nested: Dict[str, Dict[str, Any]] = {name: {} for name in SECTION_MODELS}
    unknown: List[str] = []

    for key, value in overrides.items():
        if key in SECTION_MODELS:
            if not isinstance(value, dict):
                unknown.append(str(key))
                continue
            nested[key].update(value)
            continue

        unknown.append(str(key))

    nested = {k: v for k, v in nested.items() if v}
    return nested, unknown


def _all_known_field_names() -> List[str]:
    names: List[str] = []
    for model in SECTION_MODELS.values():
        for field_name, field in model.model_fields.items():
            names.append(field_name)
            if field.alias:
                names.append(field.alias)
    return sorted(set(names))


class AppSettings(BaseModel):
    """Strict composed application settings (section-first)."""

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)

    model_config = ConfigDict(extra="forbid")

    def __getattr__(self, item: str) -> Any:
        for section in (self.llm, self.embedding, self.retrieval, self.storage, self.server, self.runtime):
            if hasattr(section, item):
                return getattr(section, item)
        raise AttributeError(item)

    def with_overrides(self, overrides: Mapping[str, Any]) -> "AppSettings":
        """Return a copy with nested section overrides applied."""
        if not overrides:
            return self

        nested, unknown = _merge_nested_overrides(overrides)
        if unknown:
            flat_split, flat_unknown = _split_flat_overrides(overrides)
            if flat_split and not flat_unknown:
                raise ValueError(
                    "Flat override keys are not supported by AppSettings. "
                    "Use nested overrides, e.g. {'llm': {'llm_provider': 'openai'}}."
                )
            raise ValueError(
                "Unknown settings fields: "
                f"{', '.join(sorted(unknown))}. "
                "Use section names (llm/embedding/retrieval/storage/server/runtime)."
            )

        payload = self.model_dump(mode="python")
        for section_name, section_updates in nested.items():
            payload[section_name].update(section_updates)
        return AppSettings.model_validate(payload)


def load_settings(env_file: Optional[str] = None, config_file: Optional[str] = None) -> AppSettings:
    """Load strict section-first settings from env/.env and optional JSON config.

    Config file keys must be section names:
    - llm
    - embedding
    - retrieval
    - storage
    - server
    - runtime
    """

    init_kwargs: Dict[str, Any] = {"_env_file": env_file} if env_file else {}
    settings = AppSettings(
        llm=LLMSettings(**init_kwargs),
        embedding=EmbeddingSettings(**init_kwargs),
        retrieval=RetrievalSettings(**init_kwargs),
        storage=StorageSettings(**init_kwargs),
        server=ServerSettings(**init_kwargs),
        runtime=RuntimeSettings(**init_kwargs),
    )

    candidate: Optional[Path] = None
    for option in AppPathDefaults.config_candidates(explicit=config_file):
        if option.exists():
            candidate = option
            break

    if not candidate:
        return settings

    try:
        with open(candidate, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)

        if not isinstance(cfg, dict):
            raise RuntimeError(f"Invalid config file '{candidate}': expected a JSON object")

        nested, unknown = _merge_nested_overrides(cfg)
        if unknown:
            flat_split, flat_unknown = _split_flat_overrides(cfg)
            if flat_split and not flat_unknown:
                raise RuntimeError(
                    "Flat config keys are no longer supported. "
                    "Use nested sections only, e.g. {'llm': {'llm_provider': 'openai'}}."
                )
            raise RuntimeError(
                f"Config file '{candidate}' contains unknown settings: "
                f"{', '.join(sorted(set(unknown)))}. "
                "Valid top-level sections are: llm, embedding, retrieval, storage, server, runtime."
            )

        if nested:
            settings = settings.with_overrides(nested)

    except ValidationError as exc:
        raise RuntimeError(f"Invalid config file '{candidate}': {exc}") from exc
    except ValueError as exc:
        raise RuntimeError(f"Invalid config file '{candidate}': {exc}") from exc
    except RuntimeError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to read config file {candidate}: {exc}") from exc

    return settings
