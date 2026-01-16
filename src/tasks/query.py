"""Query diagnostics tasks."""
from __future__ import annotations

from typing import Any, Dict

import chromadb


def query_diagnostics(storage_path: str, query_text: str) -> Dict[str, Any]:
    from llama_index.core import VectorStoreIndex
    from llama_index.core.schema import QueryBundle
    from llama_index.vector_stores.chroma import ChromaVectorStore

    from src.config import load_settings
    from src.indexer import _keyword_score, _tokenize

    settings = load_settings()
    if settings.vector_store_provider != "chroma":
        raise ValueError("query_diagnostics requires chroma vector store")

    client = chromadb.PersistentClient(path=storage_path)
    collection = client.get_or_create_collection("architext_db")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    retriever = index.as_retriever(similarity_top_k=10)
    nodes = retriever.retrieve(QueryBundle(query_str=query_text))

    results = []
    for node in nodes:
        content = node.node.get_content()[:200]
        keyword_score = _keyword_score(query_text, node.node.get_content())
        vector_score = node.score or 0.0
        results.append(
            {
                "file": node.metadata.get("file_path", "unknown"),
                "content_preview": content,
                "vector_score": vector_score,
                "keyword_score": keyword_score,
                "hybrid_score": 0.7 * vector_score + 0.3 * keyword_score,
                "query_tokens": _tokenize(query_text),
                "matched_tokens": list(
                    set(_tokenize(query_text)) & set(_tokenize(node.node.get_content()))
                ),
            }
        )

    return {"query": query_text, "results": results}
