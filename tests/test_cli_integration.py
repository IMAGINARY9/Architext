import pytest
import sys
import os
from unittest.mock import Mock, MagicMock
from io import StringIO

from src.config import ArchitextSettings

def test_cli_index_command_success(temp_repo_path, mocker, capsys):
    """Test that 'index' command runs successfully with mocked dependencies."""
    # Mock all the heavy dependencies
    mock_initialize = mocker.patch("src.cli.initialize_settings")
    mock_gather_files = mocker.patch("src.cli.gather_index_files", return_value=[os.path.join(temp_repo_path, "main.py")])
    mock_create_index = mocker.patch("src.cli.create_index_from_paths")
    mock_load_settings = mocker.patch("src.cli.load_settings")
    mock_resolve_source = mocker.patch("src.cli.resolve_source", return_value=temp_repo_path)

    # Provide config so storage defaults are predictable
    cfg = ArchitextSettings(storage_path="./config_storage")
    mock_load_settings.return_value = cfg
    
    # Run the CLI command
    sys.argv = ["cli.py", "index", temp_repo_path]
    
    from src.cli import main
    
    # Should exit without error
    try:
        main()
    except SystemExit as e:
        assert e.code is None or e.code == 0, f"Expected successful exit, got {e.code}"
    
    # Verify the flow
    mock_load_settings.assert_called_once()
    mock_initialize.assert_called_once_with(cfg)
    mock_resolve_source.assert_called_once_with(temp_repo_path, use_cache=True, ssh_key=None)
    mock_gather_files.assert_called_once_with(temp_repo_path)
    file_paths = mock_gather_files.return_value
    mock_create_index.assert_called_once_with(file_paths, "./config_storage")
    
    # Check output
    captured = capsys.readouterr()
    assert "Indexing complete" in captured.out

def test_cli_index_command_invalid_path(mocker, capsys):
    """Test that 'index' command fails gracefully with invalid path."""
    mock_initialize = mocker.patch("src.cli.initialize_settings")
    mock_load_settings = mocker.patch("src.cli.load_settings")
    mock_resolve_source = mocker.patch("src.cli.resolve_source", side_effect=ValueError("Source not found"))

    from src.config import ArchitextSettings
    mock_load_settings.return_value = ArchitextSettings()
    
    # Run with non-existent path
    sys.argv = ["cli.py", "index", "/nonexistent/path"]
    
    from src.cli import main
    
    with pytest.raises(SystemExit) as exc_info:
        main()
    
    # Should exit with error code
    assert exc_info.value.code == 1
    
    captured = capsys.readouterr()
    assert "Error during indexing" in captured.out

def test_cli_query_command_success(mocker, capsys):
    """Test that 'query' command runs successfully with mocked dependencies."""
    # Mock dependencies
    mock_initialize = mocker.patch("src.cli.initialize_settings")
    mock_load_settings = mocker.patch("src.cli.load_settings")
    
    # Mock the index loading and query response
    mock_index = Mock()
    mock_response = Mock()
    mock_response.__str__ = Mock(return_value="The answer is 42")
    mock_query_engine = Mock()
    mock_query_engine.query = Mock(return_value=mock_response)
    mock_index.as_query_engine = Mock(return_value=mock_query_engine)
    
    mock_load_index = mocker.patch("src.cli.load_existing_index", return_value=mock_index)
    mock_load_settings.return_value = ArchitextSettings(storage_path="./config_storage")
    
    # Run query command
    sys.argv = ["cli.py", "query", "What is the answer?", "--storage", "./storage"]
    
    from src.cli import main
    
    try:
        main()
    except SystemExit as e:
        assert e.code is None or e.code == 0
    
    # Verify flow
    mock_load_settings.assert_called_once()
    mock_initialize.assert_called_once_with(mock_load_settings.return_value)
    # --storage flag should override config
    mock_load_index.assert_called_once_with("./storage")
    mock_query_engine.query.assert_called_once_with("What is the answer?")
    
    # Check output
    captured = capsys.readouterr()
    assert "RESPONSE" in captured.out
    assert "42" in captured.out

def test_cli_query_command_no_storage(mocker, capsys):
    """Test that 'query' command fails when storage doesn't exist."""
    mock_initialize = mocker.patch("src.cli.initialize_settings")
    mock_load_settings = mocker.patch("src.cli.load_settings")
    mock_load_index = mocker.patch("src.cli.load_existing_index", side_effect=Exception("No storage found"))

    mock_load_settings.return_value = ArchitextSettings()
    
    sys.argv = ["cli.py", "query", "test", "--storage", "/nonexistent/storage"]
    
    from src.cli import main
    
    with pytest.raises(SystemExit) as exc_info:
        main()
    
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Error during querying" in captured.out

def test_cli_no_command(capsys):
    """Test that CLI prints help when no command is given."""
    sys.argv = ["cli.py"]
    
    from src.cli import main
    
    with pytest.raises(SystemExit) as exc_info:
        main()
    
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    # Help should be printed
    assert "usage:" in captured.out or "Architext CLI" in captured.out

def test_cli_index_with_storage_param(temp_repo_path, mocker, capsys):
    """Test that 'index' command respects --storage parameter."""
    mock_initialize = mocker.patch("src.cli.initialize_settings")
    mocker.patch("src.cli.gather_index_files", return_value=[os.path.join(temp_repo_path, "main.py")])
    mock_create_index = mocker.patch("src.cli.create_index_from_paths")
    mock_resolve_source = mocker.patch("src.cli.resolve_source", return_value=temp_repo_path)
    mock_load_settings = mocker.patch("src.cli.load_settings", return_value=ArchitextSettings())
    
    custom_storage = "./custom_storage"
    sys.argv = ["cli.py", "index", temp_repo_path, "--storage", custom_storage]
    
    from src.cli import main
    
    try:
        main()
    except SystemExit as e:
        assert e.code is None or e.code == 0
    
    # Verify storage path was passed correctly
    mock_create_index.assert_called_once()
    call_args = mock_create_index.call_args
    assert call_args[0][1] == custom_storage


def test_cli_cache_cleanup(mocker, capsys):
    """Test that 'cache-cleanup' command works without LLM init."""
    mock_cleanup = mocker.patch("src.cli.cleanup_cache", return_value=3)
    
    sys.argv = ["cli.py", "cache-cleanup", "--max-age", "14"]
    
    from src.cli import main
    
    try:
        main()
    except SystemExit as e:
        assert e.code is None or e.code == 0
    
    mock_cleanup.assert_called_once_with(max_age_days=14)
    captured = capsys.readouterr()
    assert "Removed 3 cached repo(s)" in captured.out


def test_cli_analyze_structure(mocker, capsys):
    """Analyze-structure command runs without LLM init."""
    mock_load_settings = mocker.patch("src.cli.load_settings", return_value=ArchitextSettings())
    mock_task = mocker.patch("src.cli.analyze_structure", return_value={"format": "json", "summary": {}})

    sys.argv = ["cli.py", "analyze-structure", "--storage", "./storage", "--depth", "shallow"]

    from src.cli import main

    try:
        main()
    except SystemExit as e:
        assert e.code is None or e.code == 0

    mock_task.assert_called_once()
    captured = capsys.readouterr()
    assert "summary" in captured.out


def test_cli_tech_stack(mocker, capsys):
    """Tech-stack command runs without LLM init."""
    mock_load_settings = mocker.patch("src.cli.load_settings", return_value=ArchitextSettings())
    mock_task = mocker.patch("src.cli.tech_stack", return_value={"format": "json", "data": {}})

    sys.argv = ["cli.py", "tech-stack", "--storage", "./storage"]

    from src.cli import main

    try:
        main()
    except SystemExit as e:
        assert e.code is None or e.code == 0

    mock_task.assert_called_once()
    captured = capsys.readouterr()
    assert "data" in captured.out


def test_cli_detect_anti_patterns(mocker, capsys):
    """Detect-anti-patterns command runs without LLM init."""
    mocker.patch("src.cli.load_settings", return_value=ArchitextSettings())
    mock_task = mocker.patch("src.cli.detect_anti_patterns", return_value={"issues": []})

    sys.argv = ["cli.py", "detect-anti-patterns", "--storage", "./storage"]

    from src.cli import main

    try:
        main()
    except SystemExit as e:
        assert e.code is None or e.code == 0

    mock_task.assert_called_once()
    captured = capsys.readouterr()
    assert "issues" in captured.out


def test_cli_health_score(mocker, capsys):
    """Health-score command runs without LLM init."""
    mocker.patch("src.cli.load_settings", return_value=ArchitextSettings())
    mock_task = mocker.patch("src.cli.health_score", return_value={"score": 75})

    sys.argv = ["cli.py", "health-score", "--storage", "./storage"]

    from src.cli import main

    try:
        main()
    except SystemExit as e:
        assert e.code is None or e.code == 0

    mock_task.assert_called_once()
    captured = capsys.readouterr()
    assert "score" in captured.out


def test_cli_impact_analysis(mocker, capsys):
    """Impact-analysis command runs without LLM init."""
    mocker.patch("src.cli.load_settings", return_value=ArchitextSettings())
    mock_task = mocker.patch("src.cli.impact_analysis", return_value={"affected": []})

    sys.argv = ["cli.py", "impact-analysis", "module_x", "--storage", "./storage"]

    from src.cli import main

    try:
        main()
    except SystemExit as e:
        assert e.code is None or e.code == 0

    mock_task.assert_called_once()
    captured = capsys.readouterr()
    assert "affected" in captured.out


def test_cli_refactoring_recommendations(mocker, capsys):
    """Refactoring-recommendations command runs without LLM init."""
    mocker.patch("src.cli.load_settings", return_value=ArchitextSettings())
    mock_task = mocker.patch("src.cli.refactoring_recommendations", return_value={"recommendations": []})

    sys.argv = ["cli.py", "refactoring-recommendations", "--storage", "./storage"]

    from src.cli import main

    try:
        main()
    except SystemExit as e:
        assert e.code is None or e.code == 0

    mock_task.assert_called_once()
    captured = capsys.readouterr()
    assert "recommendations" in captured.out


def test_cli_generate_docs(mocker, capsys):
    """Generate-docs command runs without LLM init."""
    mocker.patch("src.cli.load_settings", return_value=ArchitextSettings())
    mock_task = mocker.patch("src.cli.generate_docs", return_value={"outputs": []})

    sys.argv = ["cli.py", "generate-docs", "--storage", "./storage", "--output", "./docs"]

    from src.cli import main

    try:
        main()
    except SystemExit as e:
        assert e.code is None or e.code == 0

    mock_task.assert_called_once()
    captured = capsys.readouterr()
    assert "outputs" in captured.out


def test_cli_dependency_graph(mocker, capsys):
    """Dependency-graph command runs without LLM init."""
    mocker.patch("src.cli.load_settings", return_value=ArchitextSettings())
    mock_task = mocker.patch(
        "src.cli.dependency_graph_export",
        return_value={"format": "mermaid", "content": "graph TD"},
    )

    sys.argv = ["cli.py", "dependency-graph", "--storage", "./storage", "--output-format", "mermaid"]

    from src.cli import main

    try:
        main()
    except SystemExit as e:
        assert e.code is None or e.code == 0

    mock_task.assert_called_once()
    captured = capsys.readouterr()
    assert "graph TD" in captured.out


def test_cli_test_coverage(mocker, capsys):
    """Test-coverage command runs without LLM init."""
    mocker.patch("src.cli.load_settings", return_value=ArchitextSettings())
    mock_task = mocker.patch("src.cli.test_coverage_analysis", return_value={"coverage_ratio": 0.5})

    sys.argv = ["cli.py", "test-coverage", "--storage", "./storage"]

    from src.cli import main

    try:
        main()
    except SystemExit as e:
        assert e.code is None or e.code == 0

    mock_task.assert_called_once()
    captured = capsys.readouterr()
    assert "coverage_ratio" in captured.out


def test_cli_detect_patterns(mocker, capsys):
    """Detect-patterns command runs without LLM init."""
    mocker.patch("src.cli.load_settings", return_value=ArchitextSettings())
    mock_task = mocker.patch("src.cli.architecture_pattern_detection", return_value={"patterns": []})

    sys.argv = ["cli.py", "detect-patterns", "--storage", "./storage"]

    from src.cli import main

    try:
        main()
    except SystemExit as e:
        assert e.code is None or e.code == 0

    mock_task.assert_called_once()
    captured = capsys.readouterr()
    assert "patterns" in captured.out


def test_cli_diff_architecture(mocker, capsys):
    """Diff-architecture command runs without LLM init."""
    mocker.patch("src.cli.load_settings", return_value=ArchitextSettings())
    mock_task = mocker.patch("src.cli.diff_architecture_review", return_value={"added": []})

    sys.argv = ["cli.py", "diff-architecture", "--storage", "./storage"]

    from src.cli import main

    try:
        main()
    except SystemExit as e:
        assert e.code is None or e.code == 0

    mock_task.assert_called_once()
    captured = capsys.readouterr()
    assert "added" in captured.out


def test_cli_onboarding_guide(mocker, capsys):
    """Onboarding-guide command runs without LLM init."""
    mocker.patch("src.cli.load_settings", return_value=ArchitextSettings())
    mock_task = mocker.patch("src.cli.onboarding_guide", return_value={"entry_points": []})

    sys.argv = ["cli.py", "onboarding-guide", "--storage", "./storage"]

    from src.cli import main

    try:
        main()
    except SystemExit as e:
        assert e.code is None or e.code == 0

    mock_task.assert_called_once()
    captured = capsys.readouterr()
    assert "entry_points" in captured.out
