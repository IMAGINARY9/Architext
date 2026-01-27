"""API utilities shared between server and tasks."""

from typing import Any, Dict, List, Optional
import logging


def _calculate_confidence(response: Any) -> float:
    """Calculate confidence score from source node scores.
    
    Returns the average score of all source nodes with valid scores.
    Returns 0.0 if no valid scores are available.
    """
    if hasattr(response, "source_nodes") and response.source_nodes:
        scores = []
        for node in response.source_nodes:
            score = getattr(node, "score", None)
            if isinstance(score, (int, float)):
                scores.append(float(score))
        if scores:
            return sum(scores) / len(scores)
    return 0.0


def extract_sources(response: Any) -> List[Dict[str, Any]]:
    """Extract sources from a query response."""
    if hasattr(response, "source_nodes") and response.source_nodes:
        sources = []
        for node in response.source_nodes:
            # Handle both real nodes (node.node) and test mocks (node directly)
            # Prefer the nested 'node.node' structure when present and populated (real query
            # source nodes expose a 'node' attribute with metadata). Avoid treating
            # Mock objects that accidentally have a 'node' attribute as the nested case.
            if (
                hasattr(node, "node")
                and getattr(node, "node", None) is not None
                and hasattr(node.node, "metadata")
                and isinstance(getattr(node.node, "metadata", None), dict)
            ):
                metadata = node.node.metadata
                text = getattr(node.node, "text", "")
                score = getattr(node, "score", 0.0)
            else:
                # Test/mock structure or simplified node objects
                metadata = getattr(node, "metadata", {})
                text = getattr(node, "text", "") if hasattr(node, "text") else ""
                score = getattr(node, "score", 0.0)
            
            # Ensure text is a string
            if not isinstance(text, str):
                text = ""
            
            # Handle metadata which might be a dict or Mock
            file_path = ""
            start_line = None
            end_line = None
            if isinstance(metadata, dict):
                file_path = metadata.get("file_path", "")
                start_line = metadata.get("start_line")
                end_line = metadata.get("end_line")
            elif hasattr(metadata, "get"):
                # Handle Mock or dict-like
                file_path = metadata.get("file_path", "")
                start_line = metadata.get("start_line")
                end_line = metadata.get("end_line")
                # If it's a Mock, values might be Mocks, so check types
                if not isinstance(file_path, str):
                    file_path = ""
                if start_line is not None and not isinstance(start_line, int):
                    start_line = None
                if end_line is not None and not isinstance(end_line, int):
                    end_line = None
            else:
                file_path = ""
                start_line = None
                end_line = None
            
            # Normalize Windows paths to forward slashes for consistency
            if file_path:
                file_path = file_path.replace("\\", "/")
            
            sources.append({
                "file": file_path,
                "score": score,
                "start_line": start_line,
                "end_line": end_line,
            })
        return sources
    return []


def to_agent_response(response: Any, query: str) -> Dict[str, Any]:
    """Format a full agent response."""
    confidence = _calculate_confidence(response)
    
    reranked = False
    if hasattr(response, "reranked"):
        rerank_val = getattr(response, "reranked", False)
        if isinstance(rerank_val, bool):
            reranked = rerank_val
    
    hybrid_enabled = False
    if hasattr(response, "hybrid_enabled"):
        hybrid_val = getattr(response, "hybrid_enabled", False)
        if isinstance(hybrid_val, bool):
            hybrid_enabled = hybrid_val
    
    return {
        "answer": str(response),
        "confidence": confidence,
        "sources": extract_sources(response),
        "reranked": reranked,
        "hybrid_enabled": hybrid_enabled,
    }


def to_agent_response_compact(response: Any, query: str) -> Dict[str, Any]:
    """Format a compact agent response."""
    confidence = _calculate_confidence(response)

    return {
        "answer": str(response),
        "confidence": confidence,
        "sources": extract_sources(response),
    }


class DryRunIndexer:
    """Indexer that performs a dry run without actually creating an index."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def index(self, files: List[str], storage_path: str, settings: Any) -> Dict[str, Any]:
        """Perform a dry run indexing operation."""
        total_files = len(files)
        total_docs = total_files  # Simplified estimate
        
        if self.logger:
            self.logger.info(f"Dry run: Would index {total_docs} documents from {total_files} files")
        
        return {
            "files_processed": total_files,
            "documents_indexed": total_docs,
            "storage_path": storage_path,
            "dry_run": True,
        }