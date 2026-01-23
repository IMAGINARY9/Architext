"""CLI utilities for improved UX: verbosity, formatting, and output modes."""
import json
from typing import Any, Dict, Optional, List
from pathlib import Path


class VerboseLogger:
    """Simple logger that respects verbose mode."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def debug(self, msg: str):
        if self.verbose:
            print(f"[DEBUG] {msg}")

    def info(self, msg: str):
        print(f"[INFO] {msg}")

    def warning(self, msg: str):
        print(f"[WARNING] {msg}")

    def error(self, msg: str):
        print(f"[ERROR] {msg}")


def format_response(response: Any, format: str = "text") -> str:
    """Format a response for output.
    
    Args:
        response: The response object from LlamaIndex query engine.
        format: Output format ('text', 'json').
    
    Returns:
        Formatted string.
    """
    if format == "json":
        output: dict = {
            "response": str(response),
            "type": type(response).__name__,
        }
        sources = extract_sources(response)
        if sources:
            output["sources"] = sources
        return json.dumps(output, indent=2, default=str)
    return str(response)


def extract_sources(response: Any) -> List[Dict[str, Any]]:
    """Pull source metadata from a LlamaIndex response object."""

    sources: List[Dict[str, Any]] = []
    if hasattr(response, "source_nodes"):
        for node in response.source_nodes:
            metadata = getattr(node, "metadata", {}) or {}
            sources.append(
                {
                    "file": metadata.get("file_path", "unknown"),
                    "score": getattr(node, "score", None),
                    "metadata": metadata,
                    "start_line": metadata.get("start_line"),
                    "end_line": metadata.get("end_line"),
                }
            )
    return sources


def to_agent_response(response: Any) -> Dict[str, Any]:
    """Build a strict schema JSON payload for agent consumers."""

    sources = extract_sources(response)
    scores = [s["score"] for s in sources if s.get("score") is not None]
    confidence = max(scores) if scores else None

    return {
        "answer": str(response),
        "confidence": confidence,
        "sources": [
            {
                "file": s.get("file", "unknown"),
                "score": s.get("score"),
                "start_line": s.get("start_line"),
                "end_line": s.get("end_line"),
            }
            for s in sources
        ],
        "type": type(response).__name__,
    }


def to_agent_response_compact(response: Any) -> Dict[str, Any]:
    """Compact response schema for agent context windows."""
    sources = extract_sources(response)
    scores = [s["score"] for s in sources if s.get("score") is not None]
    confidence = max(scores) if scores else None

    compact_sources = []
    for source in sources:
        compact_sources.append(
            {
                "file": source.get("file", "unknown"),
                "line": source.get("start_line"),
                "score": source.get("score"),
            }
        )

    return {
        "answer": str(response),
        "confidence": confidence,
        "sources": compact_sources,
    }


def get_available_models_info() -> Dict[str, Any]:
    """Get information about available local models.
    
    Returns a dict with model discovery info for LLMs.
    For now, shows configuration points.
    """
    return {
        "local_llm": {
            "endpoint": "http://127.0.0.1:5000/v1",
            "note": "Requires Oobabooga/text-generation-webui running",
            "docs": "https://github.com/oobabooga/text-generation-webui",
        },
        "openai": {
            "endpoint": "https://api.openai.com/v1",
            "note": "Requires OPENAI_API_KEY in environment",
            "docs": "https://platform.openai.com",
        },
        "embedding_models": {
            "huggingface_local": "sentence-transformers/all-mpnet-base-v2 (recommended for local)",
            "openai": "text-embedding-3-small (recommended for cloud)",
        },
    }


class DryRunIndexer:
    """Simulates indexing without persistence for preview."""

    def __init__(self, logger: VerboseLogger):
        self.logger = logger
        self.document_count = 0
        self.file_types: dict[str, int] = {}

    def preview(self, source: str) -> Dict[str, Any]:
        """Preview what would be indexed.
        
        Args:
            source: Repository path or URL.
        
        Returns:
            Preview information.
        """
        from src.ingestor import resolve_source
        from src.indexer import gather_index_files

        try:
            repo_path = resolve_source(source, use_cache=True)
            self.logger.debug(f"Resolved source: {repo_path}")

            file_paths = gather_index_files(str(repo_path))
            self.document_count = len(file_paths)

            # Collect file type statistics
            for file_path in file_paths:
                ext = Path(file_path).suffix or "no-extension"
                self.file_types[ext] = self.file_types.get(ext, 0) + 1

            warnings = []
            if source.startswith("http") or source.startswith("git@"):
                warnings.append("Remote repository will be cloned/cached locally")
            # Add more warnings as needed, e.g., for billable providers

            return {
                "source": str(source),
                "resolved_path": str(repo_path),
                "documents": self.document_count,
                "file_types": self.file_types,
                "warnings": warnings,
                "would_index": True,
            }
        except Exception as e:
            self.logger.error(f"Preview failed: {e}")
            return {
                "source": str(source),
                "error": str(e),
                "warnings": ["Preview failed, check source path or network"],
                "would_index": False,
            }
