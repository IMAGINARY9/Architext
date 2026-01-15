from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.config import ArchitextSettings
from src.server import create_app


@pytest.fixture
def base_settings(tmp_path, mocker):
    mocker.patch("src.server.initialize_settings")
    storage_root = tmp_path / "storage"
    storage_root.mkdir()

    return ArchitextSettings(
        storage_path=str(storage_root),
        task_store_path=str(tmp_path / "task_store.json"),
        allowed_storage_roots=str(tmp_path),
        allowed_source_roots=str(tmp_path),
        rate_limit_per_minute=120,
    )


def test_rate_limiting_blocks_after_burst(tmp_path, mocker):
    mocker.patch("src.server.initialize_settings")

    settings = ArchitextSettings(
        storage_path=str(tmp_path / "storage"),
        task_store_path=str(tmp_path / "task_store.json"),
        allowed_storage_roots=str(tmp_path),
        allowed_source_roots=str(tmp_path),
        rate_limit_per_minute=1,
    )

    app = create_app(settings=settings)
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.get("/health").status_code == 429


def test_source_allowlist_blocks_outside_roots(base_settings):
    app = create_app(settings=base_settings)
    client = TestClient(app)

    outside = Path.cwd()
    assert outside.exists() and outside.is_dir()

    response = client.post(
        "/tasks/analyze-structure",
        json={"source": str(outside), "background": False, "output_format": "json"},
    )
    assert response.status_code == 400
    assert "allowed roots" in response.json().get("detail", "").lower()


def test_task_store_persists_completed_tasks(tmp_path, mocker):
    mocker.patch("src.server.initialize_settings")

    storage_root = tmp_path / "storage"
    storage_root.mkdir()

    settings = ArchitextSettings(
        storage_path=str(storage_root),
        task_store_path=str(tmp_path / "task_store.json"),
        allowed_storage_roots=str(tmp_path),
        allowed_source_roots=str(tmp_path),
        rate_limit_per_minute=0,
    )

    mocker.patch("src.server.resolve_source", return_value=tmp_path)
    mocker.patch("src.server.gather_index_files", return_value=[str(tmp_path / "a.py")])
    mocker.patch("src.server.create_index_from_paths")

    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.post(
        "/index",
        json={"source": str(tmp_path), "background": False, "storage": str(storage_root)},
    )
    assert response.status_code == 202

    task_id = response.json()["task_id"]
    store_path = Path(settings.task_store_path).expanduser().resolve()
    assert store_path.exists()

    # New server instance should reload task status
    app2 = create_app(settings=settings)
    client2 = TestClient(app2)

    status = client2.get(f"/status/{task_id}")
    assert status.status_code == 200
    assert status.json().get("status") == "completed"


def test_rerank_fails_loudly(mocker):
    from src.indexer import _apply_cross_encoder_rerank

    mocker.patch("src.indexer._get_cross_encoder", side_effect=RuntimeError("boom"))

    node = mocker.Mock()
    node.get_content = mocker.Mock(return_value="content")

    wrapper = mocker.Mock()
    wrapper.node = node
    wrapper.score = 0.1

    with pytest.raises(RuntimeError):
        _apply_cross_encoder_rerank("query", [wrapper], top_n=1, model_name="model")
