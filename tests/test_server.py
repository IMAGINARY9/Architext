import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock

from src.config import ArchitextSettings
from src.server import create_app


@pytest.fixture
def patched_settings(mocker):
    mocker.patch("src.server.initialize_settings")
    settings = ArchitextSettings(storage_path="./test-storage")
    return settings


def test_index_endpoint_inline(mocker, tmp_path, patched_settings):
    mocker.patch("src.server.resolve_source", return_value=tmp_path)
    mocker.patch("src.server.load_documents", return_value=[Mock()])
    mocker.patch("src.server.create_index")

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post(
        "/index",
        json={"source": "repo", "background": False, "storage": str(tmp_path)},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "completed"
    assert "documents" in data


def test_query_endpoint_agent_mode(mocker, patched_settings):
    mock_response = Mock()
    mock_response.__str__ = Mock(return_value="answer")
    node = Mock()
    node.metadata = {"file_path": "app.py"}
    node.score = 0.9
    mock_response.source_nodes = [node]

    mocker.patch("src.server.load_existing_index", return_value=Mock())
    mocker.patch("src.server.query_index", return_value=mock_response)

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"text": "hello", "mode": "agent", "storage": "./test-storage"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "answer"
    assert data["sources"][0]["file"] == "app.py"
    assert data.get("confidence") == 0.9


def test_query_endpoint_override_flags(mocker, patched_settings):
    mock_response = Mock()
    mock_response.__str__ = Mock(return_value="answer")
    mock_response.source_nodes = []

    mocker.patch("src.server.load_existing_index", return_value=Mock())
    query_mock = mocker.patch("src.server.query_index", return_value=mock_response)

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post(
        "/query",
        json={
            "text": "hello",
            "storage": "./test-storage",
            "enable_hybrid": True,
            "hybrid_alpha": 0.5,
            "enable_rerank": True,
            "rerank_top_n": 3,
        },
    )

    assert response.status_code == 200
    _, kwargs = query_mock.call_args
    settings = kwargs["settings"]
    assert settings.enable_hybrid is True
    assert settings.hybrid_alpha == 0.5
    assert settings.enable_rerank is True
    assert settings.rerank_top_n == 3


def test_status_unknown_task_returns_404(patched_settings):
    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.get("/status/does-not-exist")
    assert response.status_code == 404


def test_list_tasks_endpoint(patched_settings):
    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.get("/tasks")
    assert response.status_code == 200
    data = response.json()
    assert "tasks" in data
