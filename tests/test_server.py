import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock

from src.config import ArchitextSettings
from src.server import create_app


@pytest.fixture
def patched_settings(mocker):
    mocker.patch("src.server.initialize_settings")
    settings = ArchitextSettings(storage_path="./test-storage", task_store_path="./test-task-store.json")
    return settings


def test_index_endpoint_inline(mocker, tmp_path, patched_settings):
    mocker.patch("src.server.resolve_source", return_value=tmp_path)
    mocker.patch("src.server.gather_index_files", return_value=[str(tmp_path / "a.py")])
    mocker.patch("src.server.create_index_from_paths")
    patched_settings.allowed_source_roots = str(tmp_path)
    patched_settings.allowed_storage_roots = str(tmp_path)

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


def test_index_endpoint_validation_failure(mocker, tmp_path, patched_settings):
    mocker.patch("src.server.resolve_source", return_value=tmp_path)
    mocker.patch("src.server.gather_index_files", return_value=[])  # Empty - no files to index
    patched_settings.allowed_source_roots = str(tmp_path)
    patched_settings.allowed_storage_roots = str(tmp_path)

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post(
        "/index",
        json={"source": "repo", "background": False, "storage": str(tmp_path)},
    )

    assert response.status_code == 400
    data = response.json()
    assert "Validation failed" in data["detail"]
    assert "No indexable files found" in data["detail"]


def test_providers_endpoint(patched_settings):
    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.get("/providers")

    assert response.status_code == 200
    data = response.json()
    assert "llm_providers" in data
    assert "embedding_providers" in data
    assert "vector_store_providers" in data
    assert "default_llm_provider" in data
    assert "default_embedding_provider" in data
    assert "default_storage_path" in data
    assert data["llm_providers"] == ["openai", "local"]
    assert data["embedding_providers"] == ["huggingface", "openai"]
    assert data["vector_store_providers"] == ["chroma", "qdrant", "pinecone", "weaviate"]


def test_query_endpoint_agent_mode(mocker, patched_settings):
    mock_response = Mock()
    mock_response.__str__ = Mock(return_value="answer")
    mock_response.confidence = 0.9
    # Create a mock node with proper structure
    class MockNode:
        def __init__(self):
            self.metadata = {"file_path": "app.py"}
            self.score = 0.9
    
    node = MockNode()
    mock_response.source_nodes = [node]

    mocker.patch("src.server.load_existing_index", return_value=Mock())
    mocker.patch("src.server.query_index", return_value=mock_response)

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"text": "hello", "mode": "agent"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "answer"
    assert data["sources"][0]["file"] == "app.py"
    assert data.get("confidence") == 0.9


def test_query_cancels_on_disconnect(mocker, tmp_path, patched_settings):
    # Simulate a client that disconnects immediately
    async def always_disconnected(self):
        return True

    mocker.patch("starlette.requests.Request.is_disconnected", always_disconnected)

    # Create a single index so the server will pick it automatically
    patched_settings.allowed_storage_roots = str(tmp_path)
    patched_settings.allowed_source_roots = str(tmp_path)
    idx = tmp_path / "idx"
    idx.mkdir()
    (idx / "chroma.sqlite3").write_text("")

    # Make load_existing_index slow so cancellation has time to trigger
    def slow_load(path):
        import time
        time.sleep(1)
        return Mock()

    mocker.patch("src.server.load_existing_index", side_effect=slow_load)

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post("/query", json={"text": "hello"})
    assert response.status_code == 499
    assert "Client disconnected" in response.json().get("detail", "")


def test_query_appends_sources_instruction_when_missing(mocker, tmp_path, patched_settings):
    # Ensure that when user doesn't ask for sources, server appends instruction
    patched_settings.allowed_storage_roots = str(tmp_path)
    patched_settings.allowed_source_roots = str(tmp_path)
    idx = tmp_path / "idx"
    idx.mkdir()
    (idx / "chroma.sqlite3").write_text("")

    captured = {}
    def fake_query(index, text, settings=None):
        captured['text'] = text
        mock_response = Mock()
        mock_response.__str__ = Mock(return_value="answer")
        mock_response.source_nodes = []
        return mock_response

    mocker.patch("src.server.load_existing_index", return_value=Mock())
    mocker.patch("src.server.query_index", side_effect=fake_query)

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post("/query", json={"text": "Find auth flow"})
    assert response.status_code == 200
    assert 'Please include sources' in captured['text']


def test_query_does_not_duplicate_instruction(mocker, tmp_path, patched_settings):
    patched_settings.allowed_storage_roots = str(tmp_path)
    patched_settings.allowed_source_roots = str(tmp_path)
    idx = tmp_path / "idx"
    idx.mkdir()
    (idx / "chroma.sqlite3").write_text("")

    captured = {}
    def fake_query(index, text, settings=None):
        captured['text'] = text
        mock_response = Mock()
        mock_response.__str__ = Mock(return_value="answer")
        mock_response.source_nodes = []
        return mock_response

    mocker.patch("src.server.load_existing_index", return_value=Mock())
    mocker.patch("src.server.query_index", side_effect=fake_query)

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    input_text = "Find auth flow. Please include sources (file path and line ranges)."
    response = client.post("/query", json={"text": input_text})
    assert response.status_code == 200
    assert captured['text'].strip().endswith("line ranges).")


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


def test_mcp_tools_lists_tools(patched_settings):
    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.get("/mcp/tools")
    assert response.status_code == 200
    data = response.json()
    names = {tool["name"] for tool in data["tools"]}
    assert "architext.query" in names
    assert "architext.task" in names


def test_mcp_run_query_dispatch(mocker, patched_settings):
    mock_response = Mock()
    mock_response.__str__ = Mock(return_value="answer")
    mock_response.source_nodes = []

    mocker.patch("src.server.load_existing_index", return_value=Mock())
    mocker.patch("src.server.query_index", return_value=mock_response)

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post(
        "/mcp/run",
        json={
            "tool": "architext.query",
            "arguments": {"text": "hello", "mode": "agent"},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "answer"


def test_query_requires_name_when_multiple_indices(tmp_path, patched_settings):
    # Configure storage roots to a temp dir and create two index directories
    patched_settings.allowed_storage_roots = str(tmp_path)
    patched_settings.allowed_source_roots = str(tmp_path)

    idx1 = tmp_path / "idx1"
    idx2 = tmp_path / "idx2"
    idx1.mkdir()
    idx2.mkdir()
    # Touch chroma.sqlite3 to simulate indices
    (idx1 / "chroma.sqlite3").write_text("")
    (idx2 / "chroma.sqlite3").write_text("")

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    # Without 'name' we expect an error because multiple indices exist
    response = client.post("/query", json={"text": "hello"})
    assert response.status_code == 400
    assert "Specify" in response.json().get("detail", "")


def test_query_with_name_works_when_multiple_indices(mocker, tmp_path, patched_settings):
    patched_settings.allowed_storage_roots = str(tmp_path)
    patched_settings.allowed_source_roots = str(tmp_path)

    idx1 = tmp_path / "idx1"
    idx2 = tmp_path / "idx2"
    idx1.mkdir()
    idx2.mkdir()
    (idx1 / "chroma.sqlite3").write_text("")
    (idx2 / "chroma.sqlite3").write_text("")

    mock_response = Mock()
    mock_response.__str__ = Mock(return_value="answer")
    node = Mock()
    node.metadata = {"file_path": "app.py", "start_line": 1, "end_line": 2}
    node.score = 0.9
    mock_response.source_nodes = [node]

    mocker.patch("src.server.load_existing_index", return_value=Mock())
    mocker.patch("src.server.query_index", return_value=mock_response)

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post("/query", json={"text": "hello", "name": "idx1", "mode": "agent"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "answer"


def test_mcp_run_task_dispatch(mocker, patched_settings):
    mocker.patch("src.server.analyze_structure", return_value={"format": "json", "summary": {}})

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post(
        "/mcp/run",
        json={
            "tool": "architext.task",
            "arguments": {
                "task": "analyze-structure",
                "storage": "./test-storage",
                "output_format": "json",
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "analyze-structure"
    assert data["result"]["format"] == "json"


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


def test_analyze_structure_task_inline(mocker, patched_settings):
    mocker.patch("src.server.analyze_structure", return_value={"format": "json", "summary": {}})

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post(
        "/tasks/analyze-structure",
        json={"storage": "./test-storage", "background": False, "output_format": "json"},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "completed"
    assert "result" in data


def test_tech_stack_task_inline(mocker, patched_settings):
    mocker.patch("src.server.tech_stack", return_value={"format": "json", "data": {}})

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post(
        "/tasks/tech-stack",
        json={"storage": "./test-storage", "background": False, "output_format": "json"},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "completed"
    assert "result" in data


def test_detect_anti_patterns_task_inline(mocker, patched_settings):
    mocker.patch("src.server.detect_anti_patterns", return_value={"issues": []})

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post(
        "/tasks/detect-anti-patterns",
        json={"storage": "./test-storage", "background": False},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "completed"
    assert "result" in data


def test_health_score_task_inline(mocker, patched_settings):
    mocker.patch("src.server.health_score", return_value={"score": 80})

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post(
        "/tasks/health-score",
        json={"storage": "./test-storage", "background": False},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "completed"
    assert "result" in data


def test_impact_analysis_task_inline(mocker, patched_settings):
    mocker.patch("src.server.impact_analysis", return_value={"affected": []})

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post(
        "/tasks/impact-analysis",
        json={"storage": "./test-storage", "background": False, "module": "module_x"},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "completed"
    assert "result" in data


def test_dependency_graph_task_inline(mocker, patched_settings):
    mocker.patch("src.server.dependency_graph_export", return_value={"format": "mermaid"})

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post(
        "/tasks/dependency-graph",
        json={"storage": "./test-storage", "background": False, "output_format": "mermaid"},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "completed"
    assert "result" in data


def test_test_mapping_task_inline(mocker, patched_settings):
    mocker.patch("src.server.test_mapping_analysis", return_value={"tested_ratio": 0.5})

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post(
        "/tasks/test-mapping",
        json={"storage": "./test-storage", "background": False},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "completed"
    assert "result" in data


def test_detect_patterns_task_inline(mocker, patched_settings):
    mocker.patch("src.server.architecture_pattern_detection", return_value={"patterns": []})

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post(
        "/tasks/detect-patterns",
        json={"storage": "./test-storage", "background": False},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "completed"
    assert "result" in data

