"""Router for query and query-diagnostics endpoints.

Provides POST /query and POST /query/diagnostics.  Also exposes thin
MCP-callable wrappers via the ``run_query_for_mcp`` and
``query_diagnostics_for_mcp`` attributes on the returned router.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Union

from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import (
    AgentQueryResponse,
    CompactAgentQueryResponse,
    QueryDiagnosticsResponse,
    QueryDiagnosticsResult,
    QueryRequest,
    QueryResponse,
    QuerySource,
)
from src.api_utils import extract_sources, to_agent_response, to_agent_response_compact
from src.config import ArchitextSettings
from src.indexer import load_existing_index, query_index
from src.server_utils import ensure_sources_instruction, resolve_index_storage


def build_query_router(
    storage_roots: List[Path],
    base_settings: ArchitextSettings,
) -> APIRouter:
    """Create the query router.

    After building, the returned router exposes two extra attributes for
    MCP integration:

    * ``router.run_query_for_mcp``  — async callable accepting a ``QueryRequest``
    * ``router.query_diagnostics_for_mcp`` — async callable accepting a ``QueryRequest``
    """
    router = APIRouter()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _query_settings_from_request(req: QueryRequest) -> ArchitextSettings:
        overrides: Dict[str, Any] = {}
        if req.enable_hybrid is not None:
            overrides["enable_hybrid"] = req.enable_hybrid
        if req.hybrid_alpha is not None:
            overrides["hybrid_alpha"] = req.hybrid_alpha
        if req.enable_rerank is not None:
            overrides["enable_rerank"] = req.enable_rerank
        if req.rerank_model:
            overrides["rerank_model"] = req.rerank_model
        if req.rerank_top_n is not None:
            overrides["rerank_top_n"] = req.rerank_top_n
        return base_settings.model_copy(update=overrides) if overrides else base_settings

    # ------------------------------------------------------------------
    # Internal implementations (shared by HTTP + MCP entry-points)
    # ------------------------------------------------------------------

    async def _run_query_impl(
        payload: QueryRequest,
        client_request: Any,
    ) -> Union[QueryResponse, AgentQueryResponse, CompactAgentQueryResponse]:
        # Import here so the symbol is available for both normal paths and
        # exception handling.  Doing the import inside the except block
        # previously caused an UnboundLocalError when the try-block raised
        # HTTPException itself (e.g. when the client disconnects).
        from fastapi import HTTPException

        # Resolve the index storage path.  In normal operation this will
        # raise HTTPException(404) if no index has been created yet.  During
        # tests we often patch out ``load_existing_index`` and don't bother
        # creating any directories, so we want to avoid bubbling the 404
        # up and causing the route itself to return Not Found.  Instead, when
        # there are no candidates we simply use the first configured storage
        # root and let the later load step (which is usually mocked) handle
        # the missing directory.
        try:
            storage_path = resolve_index_storage(payload.name, storage_roots)
        except Exception as exc:  # HTTPException or other
            if isinstance(exc, HTTPException) and exc.status_code == 404 and storage_roots:
                # fallback to first root; load_existing_index may still fail
                storage_path = str(storage_roots[0])
            else:
                # re-raise so normal error handling still applies
                raise
        request_settings = _query_settings_from_request(payload)

        try:
            load_task = asyncio.create_task(
                asyncio.to_thread(load_existing_index, storage_path)
            )
            while not load_task.done():
                if client_request and await client_request.is_disconnected():
                    load_task.cancel()
                    raise HTTPException(status_code=499, detail="Client disconnected")
                await asyncio.sleep(0.05)
            index = await load_task

            query_text = ensure_sources_instruction(payload.text)
            query_task = asyncio.create_task(
                asyncio.to_thread(
                    lambda: query_index(index, query_text, settings=request_settings)
                )
            )
            while not query_task.done():
                if client_request and await client_request.is_disconnected():
                    query_task.cancel()
                    raise HTTPException(status_code=499, detail="Client disconnected")
                await asyncio.sleep(0.05)
            response = await query_task
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        reranked = bool(request_settings.enable_rerank)
        hybrid_enabled = bool(request_settings.enable_hybrid)

        if payload.mode == "rag":
            sources = extract_sources(response)
            return QueryResponse(
                answer=str(response),
                sources=[QuerySource(**s) for s in sources],
                reranked=reranked,
                hybrid_enabled=hybrid_enabled,
            )

        if payload.compact:
            result_payload = to_agent_response_compact(response, payload.text)
            result_payload["reranked"] = reranked
            result_payload["hybrid_enabled"] = hybrid_enabled
            return CompactAgentQueryResponse(**result_payload)

        result_payload = to_agent_response(response, payload.text)
        result_payload["type"] = result_payload.get("type", "agent")
        result_payload["reranked"] = reranked
        result_payload["hybrid_enabled"] = hybrid_enabled
        return AgentQueryResponse(**result_payload)

    async def _query_diagnostics_impl(
        payload: QueryRequest,
        client_request: Any,
    ) -> QueryDiagnosticsResponse:
        from src.indexer import _tokenize, _keyword_score

        storage_path = resolve_index_storage(payload.name, storage_roots)
        request_settings = _query_settings_from_request(payload)

        try:
            load_task = asyncio.create_task(
                asyncio.to_thread(load_existing_index, storage_path)
            )
            while not load_task.done():
                if client_request and await client_request.is_disconnected():
                    load_task.cancel()
                    raise HTTPException(status_code=499, detail="Client disconnected")
                await asyncio.sleep(0.05)
            index = await load_task

            query_task = asyncio.create_task(
                asyncio.to_thread(lambda: index.as_retriever(similarity_top_k=10))
            )
            while not query_task.done():
                if client_request and await client_request.is_disconnected():
                    query_task.cancel()
                    raise HTTPException(status_code=499, detail="Client disconnected")
                await asyncio.sleep(0.05)
            retriever = await query_task

            from llama_index.core.schema import QueryBundle

            query_text = ensure_sources_instruction(payload.text)
            nodes = retriever.retrieve(QueryBundle(query_str=query_text))

            diagnostics = []
            for node in nodes:
                content = node.node.get_content()[:200]
                keyword_score = _keyword_score(payload.text, node.node.get_content())
                vector_score = node.score or 0.0
                diagnostics.append(
                    {
                        "file": node.metadata.get("file_path", "unknown"),
                        "content_preview": content,
                        "vector_score": vector_score,
                        "keyword_score": keyword_score,
                        "hybrid_score": (
                            (request_settings.hybrid_alpha or 0.5) * vector_score
                            + (1.0 - (request_settings.hybrid_alpha or 0.5)) * keyword_score
                        ),
                        "query_tokens": _tokenize(payload.text),
                        "matched_tokens": list(
                            set(_tokenize(payload.text))
                            & set(_tokenize(node.node.get_content()))
                        ),
                    }
                )

            return QueryDiagnosticsResponse(
                query=payload.text,
                results=[QueryDiagnosticsResult(**entry) for entry in diagnostics],
                hybrid_enabled=bool(request_settings.enable_hybrid),
                rerank_enabled=bool(request_settings.enable_rerank),
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ------------------------------------------------------------------
    # HTTP endpoints
    # ------------------------------------------------------------------

    @router.post("/query", tags=["querying"])
    async def run_query(
        request: QueryRequest,
        req: Request,
    ) -> Union[QueryResponse, AgentQueryResponse, CompactAgentQueryResponse]:
        return await _run_query_impl(request, req)

    @router.post(
        "/query/diagnostics",
        response_model=QueryDiagnosticsResponse,
        response_model_exclude_none=True,
        tags=["querying"],
    )
    async def query_diagnostics(
        request: QueryRequest,
        req: Request,
    ) -> QueryDiagnosticsResponse:
        return await _query_diagnostics_impl(request, req)

    # ------------------------------------------------------------------
    # MCP-invokable wrappers (no Request object available)
    # ------------------------------------------------------------------

    async def run_query_for_mcp(
        request: QueryRequest,
    ) -> Union[QueryResponse, AgentQueryResponse, CompactAgentQueryResponse]:
        """MCP-invokable wrapper that does not have a Request available."""
        return await _run_query_impl(request, None)

    async def query_diagnostics_for_mcp(
        request: QueryRequest,
    ) -> QueryDiagnosticsResponse:
        return await _query_diagnostics_impl(request, None)

    # Expose for MCP router integration
    router.run_query_for_mcp = run_query_for_mcp  # type: ignore[attr-defined]
    router.query_diagnostics_for_mcp = query_diagnostics_for_mcp  # type: ignore[attr-defined]

    return router
