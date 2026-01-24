"""MCP endpoints for Architext."""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Type

from fastapi import APIRouter, Body, HTTPException

from src.indexer_components.factories import resolve_collection_name


QueryHandler = Callable[[Any], Awaitable[Dict[str, Any]]]
TaskRunner = Callable[[str, Any], Dict[str, Any]]
ToolSchema = Callable[[], List[Dict[str, Any]]]


def build_mcp_router(
    mcp_tools_schema: ToolSchema,
    query_handler: QueryHandler,
    ask_handler: QueryHandler,
    task_runner: TaskRunner,
    query_request_type: Type[Any],
    ask_request_type: Type[Any],
    task_request_type: Type[Any],
    mcp_run_request_type: Type[Any],
    storage_roots: List[Any] = None,
    base_settings: Any = None,
) -> APIRouter:
    router = APIRouter()

    @router.get("/mcp/tools")
    async def mcp_tools() -> Dict[str, Any]:
        return {"tools": mcp_tools_schema()}

    @router.post("/mcp/run")
    async def mcp_run(request: Dict[str, Any] = Body(..., examples={
        "query": {"summary": "Run architext.query", "value": {"tool": "architext.query", "arguments": {"text": "How does auth work?", "mode": "agent", "storage": "./my-index"}}},
        "ask": {"summary": "Run architext.ask", "value": {"tool": "architext.ask", "arguments": {"text": "How does auth work?", "compact": True}}},
        "task": {"summary": "Run architext.task inline", "value": {"tool": "architext.task", "arguments": {"task": "analyze-structure", "source": "./src", "output_format": "json", "background": False}}}
    })) -> Dict[str, Any]:
        payload = mcp_run_request_type.model_validate(request)
        tool = payload.tool
        args = payload.arguments or {}

        if tool == "architext.query":
            query_payload = query_request_type(**args)
            result = await query_handler(query_payload)
            # Convert Pydantic model results into plain dicts for MCP transport
            if not isinstance(result, dict):
                if hasattr(result, "model_dump"):
                    return result.model_dump()
                # Fallback: try to coerce to dict
                return dict(result)
            return result

        if tool == "architext.ask":
            ask_payload = ask_request_type(**args)
            result = await ask_handler(ask_payload)
            if not isinstance(result, dict):
                if hasattr(result, "model_dump"):
                    return result.model_dump()
                return dict(result)
            return result

        if tool == "architext.task":
            task_name = args.get("task")
            task_payload = task_request_type(
                storage=args.get("storage"),
                source=args.get("source"),
                output_format=args.get("output_format", "json"),
                depth=args.get("depth"),
                module=args.get("module"),
                output_dir=args.get("output_dir"),
                background=False,
            )
            try:
                result = task_runner(str(task_name), task_payload)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            return {"task": task_name, "result": result}

        if tool == "architext.list_indices":
            # List all available indices
            indices = []
            try:
                for root in storage_roots or []:
                    if not root.exists():
                        continue
                    for item in root.iterdir():
                        if item.is_dir():
                            chroma_path = item / "chroma.sqlite3"
                            if chroma_path.exists():
                                indices.append({
                                    "name": item.name,
                                    "path": str(item),
                                    "status": "available"
                                })
            except Exception:
                pass  # If we can't list indices, return empty list
            return {"indices": indices}

        if tool == "architext.get_index_metadata":
            index_name = args.get("index_name")
            if not index_name:
                raise HTTPException(status_code=400, detail="index_name is required")
            
            # Import necessary components
            from src.indexer import load_existing_index
            
            for root in storage_roots or []:
                index_path = root / index_name
                if index_path.exists() and index_path.is_dir():
                    try:
                        index = load_existing_index(str(index_path))
                        stats = index.vector_store.client.get_collection(
                            resolve_collection_name(base_settings)
                        )
                        doc_count = stats.count if hasattr(stats, 'count') else 0
                        
                        metadata = {
                            "name": index_name,
                            "path": str(index_path),
                            "documents": doc_count,
                            "provider": base_settings.vector_store_provider if base_settings else "unknown",
                            "collection": resolve_collection_name(base_settings) if base_settings else "unknown",
                            "status": "available",
                        }
                        
                        # Add any additional metadata if available
                        if hasattr(stats, 'metadata'):
                            metadata.update(stats.metadata)
                        
                        return metadata
                    except Exception as exc:
                        return {
                            "name": index_name,
                            "path": str(index_path),
                            "status": "error",
                            "error": str(exc),
                        }
            
            raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")

        raise HTTPException(status_code=400, detail="Unknown MCP tool")

    return router
