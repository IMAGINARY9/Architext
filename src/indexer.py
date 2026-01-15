import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Iterable, Iterator, Tuple, TYPE_CHECKING

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings, Document
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

from src.config import ArchitextSettings, load_settings
from src.file_filters import should_skip_path

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


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


def _resolve_collection_name(cfg: ArchitextSettings) -> str:
    return cfg.vector_store_collection or "architext_db"


def _build_vector_store(cfg: ArchitextSettings, storage_path: str):
    provider = cfg.vector_store_provider
    collection = _resolve_collection_name(cfg)

    if provider == "chroma":
        db = chromadb.PersistentClient(path=storage_path)
        chroma_collection = db.get_or_create_collection(collection)
        return ChromaVectorStore(chroma_collection=chroma_collection)

    if provider == "qdrant":
        try:
            from qdrant_client import QdrantClient  # type: ignore[import-not-found]
            from llama_index.vector_stores.qdrant import QdrantVectorStore  # type: ignore[import-not-found]
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
            from pinecone import Pinecone  # type: ignore[import-not-found]
            from llama_index.vector_stores.pinecone import PineconeVectorStore  # type: ignore[import-not-found]
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
            import weaviate  # type: ignore[import-not-found]
            from llama_index.vector_stores.weaviate import WeaviateVectorStore  # type: ignore[import-not-found]
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

INDEXABLE_EXTENSIONS = {
    ".py",
    ".md",
    ".rst",
    ".txt",
    ".java",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
}

EXCLUDED_NAME_PATTERNS = {".min.js"}
EXCLUDED_SUFFIXES = {".pyc", ".map"}


def _is_indexable_file(file_path: Path) -> bool:
    if file_path.name.startswith("."):
        return False
    lowered = file_path.name.lower()
    if any(lowered.endswith(pattern) for pattern in EXCLUDED_NAME_PATTERNS):
        return False
    if file_path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return file_path.suffix.lower() in INDEXABLE_EXTENSIONS


def gather_index_files(path: str, progress_callback=None) -> List[str]:
    """Collect indexable file paths from a directory."""
    if not os.path.exists(path):
        raise ValueError(f"Path does not exist: {path}")

    if progress_callback:
        progress_callback({"stage": "loading", "message": "Scanning files..."})

    source_path = Path(path)
    all_files: List[str] = []

    for file_path in source_path.rglob("*"):
        if file_path.is_dir():
            continue
        if should_skip_path(file_path):
            continue
        if not _is_indexable_file(file_path):
            continue
        all_files.append(str(file_path))

    if not all_files:
        raise ValueError(f"No indexable files found in {path}")

    return all_files


def _get_ts_parser(language_name: str):
    try:
        from tree_sitter_languages import get_parser  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        return get_parser(language_name)
    except Exception:
        return None


def _node_chunks_for_language(content: str, language_name: str) -> List[Tuple[int, int, str]]:
    parser = _get_ts_parser(language_name)
    if parser is None:
        return []

    tree = parser.parse(content.encode("utf-8", errors="ignore"))
    root = tree.root_node
    if language_name == "python":
        target_types = {"function_definition", "class_definition"}
    elif language_name in {"javascript", "typescript", "tsx"}:
        target_types = {"function_declaration", "method_definition", "class_declaration"}
    elif language_name == "java":
        target_types = {"method_declaration", "class_declaration"}
    elif language_name == "go":
        target_types = {"function_declaration", "method_declaration", "type_declaration"}
    else:
        target_types = set()

    if not target_types:
        return []

    chunks: List[Tuple[int, int, str]] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in target_types:
            chunks.append((node.start_byte, node.end_byte, node.type))
        for child in reversed(node.children):
            stack.append(child)
    return chunks


def _documents_from_paths_logical(file_paths: List[str], cfg: ArchitextSettings) -> List[Document]:
    documents: List[Document] = []
    max_chars = max(2000, cfg.chunk_size * 6)

    language_by_ext = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "javascript",
        ".java": "java",
        ".go": "go",
    }

    for path in file_paths:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        suffix = Path(path).suffix.lower()
        language = language_by_ext.get(suffix)
        chunks = _node_chunks_for_language(text, language) if language else []

        if not chunks:
            if text:
                documents.append(Document(text=text, metadata={"file_path": str(path)}))
            continue

        for start, end, chunk_type in chunks:
            chunk_text = text[start:end]
            if not chunk_text.strip():
                continue
            start_line = text[:start].count("\n") + 1
            end_line = text[:end].count("\n") + 1

            if len(chunk_text) > max_chars:
                for offset in range(0, len(chunk_text), max_chars):
                    slice_text = chunk_text[offset : offset + max_chars]
                    slice_start_line = start_line + chunk_text[:offset].count("\n")
                    slice_end_line = slice_start_line + slice_text.count("\n")
                    documents.append(
                        Document(
                            text=slice_text,
                            metadata={
                                "file_path": str(path),
                                "start_line": slice_start_line,
                                "end_line": slice_end_line,
                                "chunk_type": chunk_type,
                                "slice_offset": offset,
                            },
                        )
                    )
            else:
                documents.append(
                    Document(
                        text=chunk_text,
                        metadata={
                            "file_path": str(path),
                            "start_line": start_line,
                            "end_line": end_line,
                            "chunk_type": chunk_type,
                        },
                    )
                )
    return documents


def iter_document_batches(
    file_paths: List[str],
    batch_size: int = 500,
    progress_callback=None,
    settings: Optional[ArchitextSettings] = None,
) -> Iterator[List]:
    """Yield documents in batches to avoid loading everything into memory."""
    total_files = len(file_paths)
    active_settings = settings or load_settings()
    for start in range(0, total_files, batch_size):
        batch_files = file_paths[start : start + batch_size]
        if active_settings.chunking_strategy == "logical":
            documents = _documents_from_paths_logical(batch_files, active_settings)
        else:
            reader = SimpleDirectoryReader(input_files=batch_files)
            documents = reader.load_data()
        if progress_callback:
            progress_callback(
                {
                    "stage": "loading",
                    "message": f"Loaded {min(start + len(batch_files), total_files)} / {total_files} files",
                    "count": min(start + len(batch_files), total_files),
                    "total": total_files,
                }
            )
        yield documents


def load_documents(path: str, progress_callback=None):
    """Load all documents into memory (legacy)."""
    all_files = gather_index_files(path, progress_callback=progress_callback)
    reader = SimpleDirectoryReader(input_files=all_files)
    documents = reader.load_data()
    if progress_callback:
        progress_callback(
            {"stage": "loading", "message": f"Loaded {len(documents)} documents", "count": len(documents)}
        )
    return documents

def create_index(documents, storage_path="./storage", progress_callback=None):
    """Create and persist a vector index from documents."""
    print(f"Initializing ChromaDB and Vector Store at {storage_path}...")
    
    if progress_callback:
        progress_callback({"stage": "indexing", "message": "Initializing vector store..."})
    
    settings = load_settings()
    vector_store = _build_vector_store(settings, storage_path)
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


def create_index_from_paths(
    file_paths: List[str],
    storage_path: str = "./storage",
    batch_size: int = 500,
    progress_callback=None,
    settings: Optional[ArchitextSettings] = None,
):
    """Create an index from file paths using batched document loading."""
    if not file_paths:
        raise ValueError("No files provided for indexing")

    if progress_callback:
        progress_callback({"stage": "indexing", "message": "Initializing vector store..."})

    active_settings = settings or load_settings()
    vector_store = _build_vector_store(active_settings, storage_path)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    batches = iter_document_batches(
        file_paths,
        batch_size=batch_size,
        progress_callback=progress_callback,
        settings=active_settings,
    )
    try:
        first_batch = next(batches)
    except StopIteration as exc:
        raise ValueError("No documents loaded for indexing") from exc

    index = VectorStoreIndex.from_documents(
        first_batch,
        storage_context=storage_context,
        show_progress=True,
    )

    indexed_count = len(first_batch)
    for batch in batches:
        if not batch:
            continue
        index.insert_documents(batch)
        indexed_count += len(batch)
        if progress_callback:
            progress_callback(
                {
                    "stage": "indexing",
                    "message": f"Indexed {indexed_count} documents",
                    "count": indexed_count,
                }
            )

    if progress_callback:
        progress_callback({"stage": "indexing", "message": "Index created successfully", "completed": True})

    return index

def load_existing_index(storage_path="./storage"):
    """Load an existing index from ChromaDB."""
    settings = load_settings()
    print(f"Loading existing index from {storage_path} (provider={settings.vector_store_provider})...")
    vector_store = _build_vector_store(settings, storage_path)
    
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
        raise RuntimeError(f"Cross-encoder unavailable for rerank: {exc}") from exc

    rerank_limit = min(max(top_n, 1), len(nodes))
    candidates = nodes[:rerank_limit]
    pairs = [[query_text, node.node.get_content()] for node in candidates]

    try:
        model = _get_cross_encoder(model_name)
        scores = model.predict(pairs)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"Rerank failed for model '{model_name}': {exc}") from exc

    for node, score in zip(candidates, scores):
        node.score = float(score)

    candidates.sort(key=lambda item: item.score or 0.0, reverse=True)
    return candidates + nodes[rerank_limit:]


import threading

_CROSS_ENCODER_CACHE: Dict[str, "CrossEncoder"] = {}
_CROSS_ENCODER_LOCK = threading.RLock()


def _get_cross_encoder(model_name: str):
    with _CROSS_ENCODER_LOCK:
        if model_name not in _CROSS_ENCODER_CACHE:
            from sentence_transformers import CrossEncoder

            _CROSS_ENCODER_CACHE[model_name] = CrossEncoder(model_name)
        return _CROSS_ENCODER_CACHE[model_name]
