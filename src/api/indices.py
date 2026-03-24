"""Router for index listing and metadata endpoints.

Provides GET /indices, GET /indices/{name}, and GET /indices/{name}/files.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    IndexFileInfo,
    IndexFilesResponse,
    IndexInfo,
    IndexListResponse,
    IndexMetadataResponse,
)
from src.server_utils import (
    dir_disk_usage,
    dir_file_count,
    dir_last_modified,
    resolve_index_storage,
)


def build_indices_router(
    storage_roots: List[Path],
    base_settings,  # AppSettings (not typed to avoid circular import)
) -> APIRouter:
    """Create the indices router.

    Parameters
    ----------
    storage_roots:
        Resolved allowed storage root paths.
    base_settings:
        Application settings instance (``AppSettings``).
    """
    from src.indexer import load_existing_index
    from src.indexer_components.factories import resolve_collection_name

    router = APIRouter()

    # ------------------------------------------------------------------
    # GET /indices
    # ------------------------------------------------------------------
    @router.get("/indices", response_model=IndexListResponse, tags=["indices"])
    async def list_indices() -> IndexListResponse:
        """List available indices in configured storage paths."""
        indices: list[IndexInfo] = []
        for root in storage_roots:
            if not root.exists():
                continue
            try:
                _scan_root(root, indices, base_settings, resolve_collection_name)
                for item in root.iterdir():
                    if item.is_dir():
                        _scan_subdir(item, indices, base_settings, resolve_collection_name)
            except Exception:
                continue
        return IndexListResponse(indices=indices)

    # ------------------------------------------------------------------
    # GET /indices/{index_name}
    # ------------------------------------------------------------------
    @router.get(
        "/indices/{index_name}",
        response_model=IndexMetadataResponse,
        tags=["indices"],
    )
    async def get_index_metadata(index_name: str) -> IndexMetadataResponse:
        """Get metadata for a specific index."""
        for root in storage_roots:
            candidates = [root / index_name]
            if root.name == index_name:
                candidates.append(root)
            for index_path in candidates:
                if not (index_path.exists() and index_path.is_dir()):
                    continue
                try:
                    meta_file = index_path / "index_metadata.json"
                    if meta_file.exists():
                        meta = json.loads(meta_file.read_text(encoding="utf-8"))
                        return IndexMetadataResponse(
                            name=index_name,
                            path=str(index_path),
                            documents=meta.get("documents"),
                            provider=base_settings.vector_store_provider,
                            collection=resolve_collection_name(base_settings),
                            status="available",
                            metadata=meta,
                            error=None,
                            created_at=meta.get("created_at"),
                            updated_at=meta.get("updated_at"),
                            last_modified=dir_last_modified(index_path),
                            disk_usage_bytes=dir_disk_usage(index_path),
                            files_count=dir_file_count(index_path),
                        )

                    index = load_existing_index(str(index_path))
                    stats = index.vector_store.client.get_collection(
                        resolve_collection_name(base_settings)
                    )
                    doc_count = stats.count if hasattr(stats, "count") else 0

                    # build response explicitly so mypy can see required fields
                    metadata_val: Optional[Dict[str, Any]] = None
                    if hasattr(stats, "metadata") and stats.metadata:
                        metadata_val = stats.metadata

                    return IndexMetadataResponse(
                        name=index_name,
                        path=str(index_path),
                        documents=doc_count,
                        provider=base_settings.vector_store_provider,
                        collection=resolve_collection_name(base_settings),
                        status="available",
                        metadata=metadata_val,
                        error=None,
                        last_modified=dir_last_modified(index_path),
                        disk_usage_bytes=dir_disk_usage(index_path),
                        files_count=dir_file_count(index_path),
                        created_at=(metadata_val.get("created_at") if isinstance(metadata_val, dict) else None),
                        updated_at=(metadata_val.get("updated_at") if isinstance(metadata_val, dict) else None),
                    )
                except Exception as exc:
                    return IndexMetadataResponse(
                        name=index_name,
                        path=str(index_path),
                        documents=None,
                        provider=base_settings.vector_store_provider,
                        collection=resolve_collection_name(base_settings),
                        status="error",
                        metadata=None,
                        error=str(exc),
                        last_modified=None,
                        disk_usage_bytes=None,
                        files_count=None,
                        created_at=None,
                        updated_at=None,
                    )
        raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")

    # ------------------------------------------------------------------
    # GET /indices/{index_name}/files
    # ------------------------------------------------------------------
    @router.get(
        "/indices/{index_name}/files",
        response_model=IndexFilesResponse,
        tags=["indices"],
    )
    async def list_index_files(index_name: str) -> IndexFilesResponse:
        """List all files indexed in a specific index.

        Returns file paths with document chunk counts and metadata availability.
        """
        storage_path = resolve_index_storage(index_name, storage_roots)

        try:
            index = await asyncio.to_thread(load_existing_index, storage_path)
            vector_store = index.vector_store

            # Direct ChromaDB access for better performance
            metadatas: list = []
            if hasattr(vector_store, "_collection"):
                collection = vector_store._collection
                result = collection.get(include=["metadatas"])
                metadatas = result.get("metadatas", [])
            else:
                from llama_index.core.schema import QueryBundle

                retriever = index.as_retriever(similarity_top_k=10000)
                nodes = retriever.retrieve(QueryBundle(query_str="*"))
                metadatas = [
                    node.node.metadata
                    for node in nodes
                    if hasattr(node, "node")
                ]

            file_counts: dict[str, int] = {}
            file_has_lines: dict[str, bool] = {}

            for metadata in metadatas:
                if not isinstance(metadata, dict):
                    continue
                file_path = metadata.get("file_path", "")
                if not file_path:
                    continue
                file_path = file_path.replace("\\", "/")
                file_counts[file_path] = file_counts.get(file_path, 0) + 1
                if "start_line" in metadata or "end_line" in metadata:
                    file_has_lines[file_path] = True

            files = [
                IndexFileInfo(
                    file=path,
                    chunks=count,
                    has_line_info=file_has_lines.get(path, False),
                )
                for path, count in sorted(file_counts.items())
            ]

            return IndexFilesResponse(
                index_name=index_name,
                total_files=len(files),
                total_chunks=sum(file_counts.values()),
                files=files,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to list files: {str(exc)}"
            ) from exc

    return router


# ---------------------------------------------------------------------------
# Private helpers for list_indices
# ---------------------------------------------------------------------------

def _build_index_info_from_meta(
    path: Path, meta: dict, base_settings, resolve_collection_name
) -> IndexInfo:
    """Create an ``IndexInfo`` from persisted metadata."""
    return IndexInfo(
        name=path.name,
        path=str(path),
        documents=meta.get("documents"),
        provider=base_settings.vector_store_provider,
        collection=resolve_collection_name(base_settings),
        status="available",
        created_at=meta.get("created_at"),
        updated_at=meta.get("updated_at"),
        last_modified=dir_last_modified(path),
        disk_usage_bytes=dir_disk_usage(path),
        files_count=dir_file_count(path),
    )


def _build_index_info_from_db(
    path: Path, base_settings, resolve_collection_name
) -> IndexInfo:
    """Create an ``IndexInfo`` by inspecting the ChromaDB database."""
    from src.indexer import load_existing_index

    try:
        index = load_existing_index(str(path))
        doc_count = None
        try:
            stats = index.vector_store.client.get_collection(
                resolve_collection_name(base_settings)
            )
            doc_count = stats.count if hasattr(stats, "count") else None
        except Exception:
            pass
        return IndexInfo(
            name=path.name,
            path=str(path),
            documents=doc_count,
            provider=base_settings.vector_store_provider,
            collection=resolve_collection_name(base_settings),
            status="available",
            last_modified=dir_last_modified(path),
            disk_usage_bytes=dir_disk_usage(path),
            files_count=dir_file_count(path),
            created_at=None,
            updated_at=None,
        )
    except Exception:
        return IndexInfo(
            name=path.name,
            path=str(path),
            documents=None,
            provider=base_settings.vector_store_provider,
            collection=resolve_collection_name(base_settings),
            status="load_error",
            last_modified=None,
            disk_usage_bytes=None,
            files_count=None,
            created_at=None,
            updated_at=None,
        )


def _scan_index_dir(
    path: Path, indices: list, base_settings, resolve_collection_name
) -> bool:
    """Append an ``IndexInfo`` for *path* if it contains a ChromaDB index.

    Returns ``True`` if an entry was appended.
    """
    if not (path / "chroma.sqlite3").exists():
        return False

    try:
        meta_file = path / "index_metadata.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            indices.append(
                _build_index_info_from_meta(path, meta, base_settings, resolve_collection_name)
            )
            return True
    except Exception:
        pass

    indices.append(
        _build_index_info_from_db(path, base_settings, resolve_collection_name)
    )
    return True


def _scan_root(root: Path, indices: list, base_settings, resolve_collection_name) -> None:
    """Scan the storage root itself (not subdirectories)."""
    _scan_index_dir(root, indices, base_settings, resolve_collection_name)


def _scan_subdir(item: Path, indices: list, base_settings, resolve_collection_name) -> None:
    """Scan a single subdirectory of a storage root."""
    _scan_index_dir(item, indices, base_settings, resolve_collection_name)



