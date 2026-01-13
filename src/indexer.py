import os
from typing import Optional

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

from src.config import ArchitextSettings, load_settings


def _build_llm(cfg: ArchitextSettings):
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
        return OpenAILike(**llm_kwargs)

    raise ValueError(f"Unsupported LLM provider: {cfg.llm_provider}")


def _build_embedding(cfg: ArchitextSettings):
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


def initialize_settings(settings: Optional[ArchitextSettings] = None) -> ArchitextSettings:
    """Configure global LlamaIndex settings for LLM and embeddings using config."""

    cfg = settings or load_settings()

    Settings.llm = _build_llm(cfg)
    Settings.embed_model = _build_embedding(cfg)

    return cfg

def load_documents(path: str):
    """Recursively read files from the directory, ignoring hidden files and common git folders."""
    if not os.path.exists(path):
        raise ValueError(f"Path does not exist: {path}")
        
    print(f"Loading documents from {path}...")
    # exclude_hidden=True helps avoid .git, .env, .vscode etc.
    reader = SimpleDirectoryReader(
        input_dir=path, 
        recursive=True,
        exclude_hidden=True,
        exclude=["**/.git/**", "**/__pycache__/**", "**/*.pyc"]
    )
    documents = reader.load_data()
    print(f"Loaded {len(documents)} documents.")
    return documents

def create_index(documents, storage_path="./storage"):
    """Create and persist a vector index from documents."""
    print(f"Initializing ChromaDB and Vector Store at {storage_path}...")
    
    # Create Chroma Client
    db = chromadb.PersistentClient(path=storage_path)
    chroma_collection = db.get_or_create_collection("architext_db")
    
    # Create Vector Store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print("Creating VectorStoreIndex (this may take a while)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    return index

def load_existing_index(storage_path="./storage"):
    """Load an existing index from ChromaDB."""
    print(f"Loading existing index from {storage_path}...")
    db = chromadb.PersistentClient(path=storage_path)
    chroma_collection = db.get_or_create_collection("architext_db")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
    )
    return index

def query_index(index, query_text: str):
    """Query the index and return the response."""
    print(f"Querying: {query_text}")
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    return response
