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
    mock_load_docs = mocker.patch("src.cli.load_documents", return_value=[Mock()])
    mock_create_index = mocker.patch("src.cli.create_index")
    mock_load_settings = mocker.patch("src.cli.load_settings")

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
    mock_load_docs.assert_called_once_with(temp_repo_path)
    docs = mock_load_docs.return_value
    mock_create_index.assert_called_once_with(docs, "./config_storage")
    
    # Check output
    captured = capsys.readouterr()
    assert "Indexing complete" in captured.out

def test_cli_index_command_invalid_path(mocker, capsys):
    """Test that 'index' command fails gracefully with invalid path."""
    mock_initialize = mocker.patch("src.cli.initialize_settings")
    mock_load_settings = mocker.patch("src.cli.load_settings")

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
    mock_load_docs = mocker.patch("src.cli.load_documents", return_value=[Mock()])
    mock_create_index = mocker.patch("src.cli.create_index")
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
