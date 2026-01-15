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


def test_ask_endpoint_compact_agent_schema(mocker, patched_settings):
    mock_response = Mock()
    mock_response.__str__ = Mock(return_value="answer")
    node = Mock()
    node.metadata = {"file_path": "app.py", "start_line": 10, "end_line": 20}
    node.score = 0.9
    mock_response.source_nodes = [node]

    mocker.patch("src.server.load_existing_index", return_value=Mock())
    mocker.patch("src.server.query_index", return_value=mock_response)

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post(
        "/ask",
        json={"text": "hello", "storage": "./test-storage", "compact": True},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "answer"
    assert data["sources"][0]["file"] == "app.py"
    assert "reranked" in data
    assert "hybrid_enabled" in data


def test_mcp_tools_lists_tools(patched_settings):
    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.get("/mcp/tools")
    assert response.status_code == 200
    data = response.json()
    names = {tool["name"] for tool in data["tools"]}
    assert "architext.query" in names
    assert "architext.ask" in names
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
            "arguments": {"text": "hello", "mode": "agent", "storage": "./test-storage"},
        },
    )

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


def test_refactoring_recommendations_task_inline(mocker, patched_settings):
    mocker.patch("src.server.refactoring_recommendations", return_value={"recommendations": []})

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post(
        "/tasks/refactoring-recommendations",
        json={"storage": "./test-storage", "background": False},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "completed"
    assert "result" in data


def test_generate_docs_task_inline(mocker, patched_settings):
    mocker.patch("src.server.generate_docs", return_value={"outputs": []})

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post(
        "/tasks/generate-docs",
        json={"storage": "./test-storage", "background": False, "output_dir": "./docs"},
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


def test_test_coverage_task_inline(mocker, patched_settings):
    mocker.patch("src.server.test_coverage_analysis", return_value={"coverage_ratio": 0.5})

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post(
        "/tasks/test-coverage",
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


def test_diff_architecture_task_inline(mocker, patched_settings):
    mocker.patch("src.server.diff_architecture_review", return_value={"added": [], "removed": []})

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post(
        "/tasks/diff-architecture",
        json={"storage": "./test-storage", "background": False},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "completed"
    assert "result" in data


def test_onboarding_guide_task_inline(mocker, patched_settings):
    mocker.patch("src.server.onboarding_guide", return_value={"entry_points": []})

    app = create_app(settings=patched_settings)
    client = TestClient(app)

    response = client.post(
        "/tasks/onboarding-guide",
        json={"storage": "./test-storage", "background": False},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "completed"
    assert "result" in data
