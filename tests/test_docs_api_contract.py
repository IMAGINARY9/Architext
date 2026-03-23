"""Prevent documentation endpoint drift against live OpenAPI schema."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.config import ArchitextSettings
from src.server import create_app


def test_core_documented_endpoints_exist_in_openapi(mocker) -> None:
    mocker.patch("src.server.initialize_settings")
    app = create_app(
        settings=ArchitextSettings(storage_path="./test-storage", task_store_path="./test-task-store.json")
    )
    client = TestClient(app)

    openapi = client.get("/openapi.json").json()
    paths = set(openapi.get("paths", {}).keys())

    expected = {
        "/index",
        "/index/preview",
        "/query",
        "/status/{task_id}",
        "/indices",
        "/mcp/tools",
    }
    assert expected.issubset(paths)


def test_docs_do_not_reference_stale_endpoints() -> None:
    readme = open("README.md", encoding="utf-8").read()
    development = open("docs/DEVELOPMENT.md", encoding="utf-8").read()
    combined = f"{readme}\n{development}"

    stale_markers = [
        "POST /ask",
        "GET /tasks/{id}",
        "/tasks/structure",
        "/tasks/anti-patterns",
        "/tasks/audit",
    ]
    for marker in stale_markers:
        assert marker not in combined


def test_docs_include_current_polling_and_index_selection_guidance() -> None:
    readme = open("README.md", encoding="utf-8").read()
    development = open("docs/DEVELOPMENT.md", encoding="utf-8").read()

    assert "GET /status/{task_id}" in readme
    assert "GET /indices" in readme
    assert "POST /query" in readme

    assert "GET /status/{task_id}" in development
    assert "GET /indices" in development
