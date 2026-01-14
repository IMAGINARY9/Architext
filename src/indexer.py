import os
import re
from typing import Optional, List, Dict

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
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

def load_documents(path: str, progress_callback=None):
    """Recursively read files from the directory, ignoring hidden files and common git folders."""
    if not os.path.exists(path):
        raise ValueError(f"Path does not exist: {path}")
    
    if progress_callback:
        progress_callback({"stage": "loading", "message": "Scanning files..."})
        
    print(f"Loading documents from {path}...")
    
    from pathlib import Path
    
    # Manually collect files to index with explicit extension filtering
    source_extensions = [".py", ".md", ".rst", ".txt", ".java", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"]
    exclude_patterns = ["/.git/", "/__pycache__/", "/node_modules/", "/.venv/", "/.env/", "/build/", "/dist/", ".pyc", ".min.js", ".map"]
    
    source_path = Path(path)
    all_files = []
    
    for ext in source_extensions:
        for file_path in source_path.rglob(f"*{ext}"):
            str_path = str(file_path)
            # Skip hidden files (starting with .)
            if any(part.startswith('.') and part not in {'.py', '.md', '.txt', '.rs', '.c', '.h', '.cs', '.rb', '.php', '.js', '.ts', '.tsx', '.jsx', '.go', '.java', '.cpp', '.hpp', '.yaml', '.yml', '.json', '.toml', '.ini', '.cfg'} for part in file_path.parts):
                continue
            if not any(pattern in str_path.replace('\\', '/') for pattern in exclude_patterns):
                all_files.append(str(file_path))
    
    if not all_files:
        raise ValueError(f"No indexable files found in {path}")
    
    print(f"Found {len(all_files)} files to index")
    
    reader = SimpleDirectoryReader(input_files=all_files)
    documents = reader.load_data()
    print(f"Loaded {len(documents)} documents.")
    
    if progress_callback:
        progress_callback({"stage": "loading", "message": f"Loaded {len(documents)} documents", "count": len(documents)})
    
    return documents

def create_index(documents, storage_path="./storage", progress_callback=None):
    """Create and persist a vector index from documents."""
    print(f"Initializing ChromaDB and Vector Store at {storage_path}...")
    
    if progress_callback:
        progress_callback({"stage": "indexing", "message": "Initializing vector store..."})
    
    # Create Chroma Client
    db = chromadb.PersistentClient(path=storage_path)
    chroma_collection = db.get_or_create_collection("architext_db")
    
    # Create Vector Store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print("Creating VectorStoreIndex (this may take a while)...")
    if progress_callback:
        progress_callback({"stage": "indexing", "message": f"Embedding {len(documents)} documents..."})
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    if progress_callback:
        progress_callback({"stage": "indexing", "message": "Index created successfully", "completed": True})
    
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

def query_index(index, query_text: str, settings: Optional[ArchitextSettings] = None):
    """Query the index and return the response."""
    print(f"Querying: {query_text}")
    active_settings = settings or load_settings()
    query_engine = _build_query_engine(index, active_settings)
    response = query_engine.query(query_text)
    return response


def _build_query_engine(index: VectorStoreIndex, settings: ArchitextSettings):
    if settings.enable_hybrid or settings.enable_rerank:
        base_retriever = index.as_retriever(similarity_top_k=settings.top_k)
        hybrid_retriever = HybridRerankRetriever(base_retriever, settings)
        return RetrieverQueryEngine.from_args(retriever=hybrid_retriever)

    return index.as_query_engine(similarity_top_k=settings.top_k)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _keyword_score(query: str, document: str) -> float:
    query_terms = set(_tokenize(query))
    if not query_terms:
        return 0.0
    doc_terms = set(_tokenize(document))
    return len(query_terms & doc_terms) / len(query_terms)


class HybridRerankRetriever(BaseRetriever):
    """Retriever that optionally applies hybrid scoring and cross-encoder reranking."""

    def __init__(self, base_retriever: BaseRetriever, settings: ArchitextSettings):
        super().__init__()
        self._base_retriever = base_retriever
        self._settings = settings

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self._base_retriever.retrieve(query_bundle)
        if not nodes:
            return nodes

        query_text = query_bundle.query_str

        if self._settings.enable_hybrid:
            nodes = self._apply_hybrid(query_text, nodes)

        if self._settings.enable_rerank:
            nodes = _apply_cross_encoder_rerank(
                query_text,
                nodes,
                top_n=self._settings.rerank_top_n,
                model_name=self._settings.rerank_model,
            )

        return nodes

    def _apply_hybrid(self, query_text: str, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        alpha = max(0.0, min(1.0, self._settings.hybrid_alpha))
        scored: List[NodeWithScore] = []

        for node in nodes:
            vector_score = node.score or 0.0
            keyword = _keyword_score(query_text, node.node.get_content())
            node.score = alpha * vector_score + (1 - alpha) * keyword
            scored.append(node)

        scored.sort(key=lambda item: item.score or 0.0, reverse=True)
        return scored


def _apply_cross_encoder_rerank(
    query_text: str,
    nodes: List[NodeWithScore],
    top_n: int,
    model_name: str,
) -> List[NodeWithScore]:
    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARNING] Cross-encoder unavailable, skipping rerank: {exc}")
        return nodes

    rerank_limit = min(max(top_n, 1), len(nodes))
    candidates = nodes[:rerank_limit]
    pairs = [[query_text, node.node.get_content()] for node in candidates]

    try:
        model = _get_cross_encoder(model_name)
        scores = model.predict(pairs)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARNING] Rerank failed, returning base order: {exc}")
        return nodes

    for node, score in zip(candidates, scores):
        node.score = float(score)

    candidates.sort(key=lambda item: item.score or 0.0, reverse=True)
    return candidates + nodes[rerank_limit:]


_CROSS_ENCODER_CACHE: Dict[str, "CrossEncoder"] = {}
_CROSS_ENCODER_LOCK = None


def _get_cross_encoder(model_name: str):
    global _CROSS_ENCODER_LOCK

    if _CROSS_ENCODER_LOCK is None:
        import threading

        _CROSS_ENCODER_LOCK = threading.Lock()

    with _CROSS_ENCODER_LOCK:
        if model_name not in _CROSS_ENCODER_CACHE:
            from sentence_transformers import CrossEncoder

            _CROSS_ENCODER_CACHE[model_name] = CrossEncoder(model_name)
        return _CROSS_ENCODER_CACHE[model_name]
