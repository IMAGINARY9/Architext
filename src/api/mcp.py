"""MCP endpoints for Architext."""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Type

from fastapi import APIRouter, Body, HTTPException


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
) -> APIRouter:
    router = APIRouter()

    @router.get("/mcp/tools")
    async def mcp_tools() -> Dict[str, Any]:
        return {"tools": mcp_tools_schema()}

    @router.post("/mcp/run")
    async def mcp_run(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        payload = mcp_run_request_type.model_validate(request)
        tool = payload.tool
        args = payload.arguments or {}

        if tool == "architext.query":
            query_payload = query_request_type(**args)
            return await query_handler(query_payload)

        if tool == "architext.ask":
            ask_payload = ask_request_type(**args)
            return await ask_handler(ask_payload)

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
                result = task_runner(task_name, task_payload)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            return {"task": task_name, "result": result}

        raise HTTPException(status_code=400, detail="Unknown MCP tool")

    return router
