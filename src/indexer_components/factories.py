"""Factory helpers for indexer components."""
from __future__ import annotations

import os
from typing import Optional

import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import ArchitextSettings


def build_llm(cfg: ArchitextSettings):
    """Create the configured LLM client.

    Currently supports local/OpenAI-compatible endpoints via OpenAILike.
    """

    provider = cfg.llm_provider.lower()
    if provider in {"local", "openai"}:
        llm_kwargs = {
            "model": cfg.llm_model,
            "api_base": cfg.openai_api_base,
            "api_key": cfg.openai_api_key,
            "temperature": cfg.llm_temperature,
            "is_chat_model": True,
        }
        if cfg.llm_max_tokens is not None:
            llm_kwargs["max_tokens"] = cfg.llm_max_tokens
        return OpenAILike(**llm_kwargs)  # type: ignore[arg-type]

    raise ValueError(f"Unsupported LLM provider: {cfg.llm_provider}")


def build_embedding(cfg: ArchitextSettings):
    """Create the configured embedding model."""

    provider = cfg.embedding_provider.lower()
    if provider == "huggingface":
        cache_folder = os.path.abspath(cfg.embedding_cache_dir)
        os.makedirs(cache_folder, exist_ok=True)

        return HuggingFaceEmbedding(
            model_name=cfg.embedding_model_name,
            cache_folder=cache_folder,
        )

    if provider == "openai":
        if not cfg.openai_api_key or cfg.openai_api_key == "local":
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")

        return OpenAIEmbedding(
            model=cfg.embedding_model_name,
            api_key=cfg.openai_api_key,
            api_base=cfg.openai_api_base,
        )

    raise ValueError(f"Unsupported embedding provider: {cfg.embedding_provider}")


def resolve_collection_name(cfg: ArchitextSettings) -> str:
    return cfg.vector_store_collection or "architext_db"


def build_vector_store(cfg: ArchitextSettings, storage_path: str):
    provider = cfg.vector_store_provider
    collection = resolve_collection_name(cfg)

    if provider == "chroma":
        db = chromadb.PersistentClient(path=storage_path)
        chroma_collection = db.get_or_create_collection(collection)
        return ChromaVectorStore(chroma_collection=chroma_collection)

    if provider == "qdrant":
        try:
            from qdrant_client import QdrantClient
            from llama_index.vector_stores.qdrant import QdrantVectorStore
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                "Qdrant adapter requires qdrant-client and llama-index-vector-stores-qdrant"
            ) from exc
        if not cfg.qdrant_url:
            raise ValueError("QDRANT_URL is required for qdrant vector store")
        client = QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)
        return QdrantVectorStore(client=client, collection_name=collection)

    if provider == "pinecone":
        try:
            from pinecone import Pinecone
            from llama_index.vector_stores.pinecone import PineconeVectorStore
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                "Pinecone adapter requires pinecone-client and llama-index-vector-stores-pinecone"
            ) from exc
        if not cfg.pinecone_api_key or not cfg.pinecone_index_name:
            raise ValueError("PINECONE_API_KEY and PINECONE_INDEX_NAME are required")
        client = Pinecone(api_key=cfg.pinecone_api_key)
        index = client.Index(cfg.pinecone_index_name)
        return PineconeVectorStore(pinecone_index=index)

    if provider == "weaviate":
        try:
            import weaviate
            from llama_index.vector_stores.weaviate import WeaviateVectorStore
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                "Weaviate adapter requires weaviate-client and llama-index-vector-stores-weaviate"
            ) from exc
        if not cfg.weaviate_url:
            raise ValueError("WEAVIATE_URL is required for weaviate vector store")
        client = weaviate.Client(
            url=cfg.weaviate_url,
            auth_client_secret=weaviate.AuthApiKey(cfg.weaviate_api_key)
            if cfg.weaviate_api_key
            else None,
        )
        return WeaviateVectorStore(weaviate_client=client, index_name=collection)

    raise ValueError(f"Unsupported vector store provider: {cfg.vector_store_provider}")
