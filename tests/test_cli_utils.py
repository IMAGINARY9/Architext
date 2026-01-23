"""Tests for CLI utilities and new Phase 1.6 features."""
import pytest
import sys
import json
from unittest.mock import Mock, patch

from src.config import ArchitextSettings
from src.cli_utils import (
    VerboseLogger,
    format_response,
    get_available_models_info,
    DryRunIndexer,
    to_agent_response,
)


class TestVerboseLogger:
    def test_verbose_mode_enabled(self, capsys):
        """Debug messages print when verbose=True."""
        logger = VerboseLogger(verbose=True)
        logger.debug("Test debug message")
        
        captured = capsys.readouterr()
        assert "[DEBUG]" in captured.out
        assert "Test debug message" in captured.out

    def test_verbose_mode_disabled(self, capsys):
        """Debug messages don't print when verbose=False."""
        logger = VerboseLogger(verbose=False)
        logger.debug("Test debug message")
        
        captured = capsys.readouterr()
        assert "[DEBUG]" not in captured.out

    def test_info_always_prints(self, capsys):
        """Info messages print regardless of verbose mode."""
        logger = VerboseLogger(verbose=False)
        logger.info("Test info")
        
        captured = capsys.readouterr()
        assert "[INFO]" in captured.out


class TestFormatResponse:
    def test_text_format(self):
        """Response formatted as text."""
        response = Mock()
        response.__str__ = Mock(return_value="The answer is 42")
        
        result = format_response(response, format="text")
        assert "42" in result

    def test_json_format(self):
        """Response formatted as JSON."""
        response = Mock()
        response.__str__ = Mock(return_value="The answer is 42")
        response.source_nodes = []
        
        result = format_response(response, format="json")
        
        # Should be valid JSON
        data = json.loads(result)
        assert "response" in data
        assert "42" in data["response"]

    def test_json_with_sources(self):
        """JSON format includes source nodes."""
        response = Mock()
        response.__str__ = Mock(return_value="Answer")
        
        # Mock source nodes
        node1 = Mock()
        node1.metadata = {"file_path": "auth.py"}
        node1.score = 0.95
        
        response.source_nodes = [node1]
        
        result = format_response(response, format="json")
        data = json.loads(result)
        
        assert "sources" in data
        assert len(data["sources"]) == 1
        assert data["sources"][0]["file"] == "auth.py"


class TestGetAvailableModelsInfo:
    def test_models_info_structure(self):
        """Available models info has expected structure."""
        info = get_available_models_info()
        
        assert "local_llm" in info
        assert "openai" in info
        assert "embedding_models" in info
        
        # Each should have descriptive info
        assert "endpoint" in info["local_llm"]
        assert "docs" in info["openai"]


class TestAgentResponse:
    def test_to_agent_response(self):
        response = Mock()
        response.__str__ = Mock(return_value="Agent answer")

        node = Mock()
        node.metadata = {"file_path": "service.py"}
        node.score = 0.87
        response.source_nodes = [node]

        payload = to_agent_response(response)

        assert payload["answer"] == "Agent answer"
        assert payload["confidence"] == 0.87
        assert payload["sources"][0]["file"] == "service.py"


class TestDryRunIndexer:
    def test_dry_run_preview(self, tmp_path, capsys):
        """Dry-run provides preview without indexing."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        (repo_dir / "test.py").write_text("def hello(): pass")

        logger = VerboseLogger(verbose=False)
        indexer = DryRunIndexer(logger)

        with patch("src.ingestor.resolve_source", return_value=repo_dir):
            with patch("src.indexer.gather_index_files") as mock_gather:
                mock_gather.return_value = [str(repo_dir / "test.py")]

                preview = indexer.preview(str(repo_dir))
        
        assert preview["documents"] == 1
        assert ".py" in preview["file_types"]

    def test_dry_run_with_invalid_source(self):
        """Dry-run handles invalid sources gracefully."""
        logger = VerboseLogger(verbose=False)
        indexer = DryRunIndexer(logger)
        
        with patch("src.ingestor.resolve_source", side_effect=ValueError("Not found")):
            preview = indexer.preview("/invalid/path")
        
        assert preview["would_index"] is False
        assert "error" in preview


class TestCLIWithNewFeatures:
    def test_cli_list_models_command(self, mocker, capsys):
        """list-models command works without LLM init."""
        sys.argv = ["cli.py", "list-models"]
        
        from src.cli import main
        
        try:
            main()
        except SystemExit as e:
            assert e.code is None or e.code == 0
        
        captured = capsys.readouterr()
        assert "AVAILABLE" in captured.out
        assert "LLM" in captured.out or "embedding" in captured.out.lower()

    def test_cli_verbose_flag(self, mocker, capsys):
        """Verbose flag enables debug logging."""
        mock_initialize = mocker.patch("src.cli.initialize_settings")
        mock_load_settings = mocker.patch("src.cli.load_settings", return_value=ArchitextSettings())
        mock_cleanup = mocker.patch("src.cli.cleanup_cache", return_value=0)
        
        sys.argv = ["cli.py", "--verbose", "cache-cleanup"]
        
        from src.cli import main
        
        try:
            main()
        except SystemExit:
            pass
        
        captured = capsys.readouterr()
        # Should show cleanup message
        assert "Cleanup complete" in captured.out

    def test_cli_dry_run_flag(self, mocker, capsys, tmp_path):
        """Dry-run flag previews without persisting."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        (repo_dir / "test.py").write_text("code")
        
        mock_initialize = mocker.patch("src.cli.initialize_settings")
        mock_load_settings = mocker.patch("src.cli.load_settings", return_value=ArchitextSettings())
        mock_resolve = mocker.patch("src.cli.resolve_source", return_value=repo_dir)
        mock_dry_run = mocker.patch("src.cli.DryRunIndexer")
        
        mock_indexer = Mock()
        mock_indexer.preview.return_value = {
            "source": str(repo_dir),
            "documents": 1,
            "file_types": {".py": 1},
        }
        mock_dry_run.return_value = mock_indexer
        
        sys.argv = ["cli.py", "index", str(repo_dir), "--dry-run"]
        
        from src.cli import main
        
        try:
            main()
        except SystemExit:
            pass
        
        captured = capsys.readouterr()
        assert "DRY-RUN" in captured.out
        assert "Documents" in captured.out

    def test_cli_query_json_format(self, mocker, capsys):
        """Query with --format json outputs JSON."""
        from src.config import ArchitextSettings
        
        mock_initialize = mocker.patch("src.cli.initialize_settings")
        mock_load_settings = mocker.patch("src.cli.load_settings", return_value=ArchitextSettings())
        
        # Mock the full query pipeline
        mock_response = Mock()
        mock_response.__str__ = Mock(return_value="Test answer")
        mock_response.source_nodes = []
        
        mock_index = Mock()
        mock_query_engine = Mock()
        mock_query_engine.query = Mock(return_value=mock_response)
        mock_index.as_query_engine = Mock(return_value=mock_query_engine)
        
        mocker.patch("src.cli.load_existing_index", return_value=mock_index)
        
        sys.argv = ["cli.py", "query", "test question", "--format", "json"]
        
        from src.cli import main
        
        try:
            main()
        except SystemExit:
            pass
        
        captured = capsys.readouterr()
        # Output should contain JSON markers
        assert "{" in captured.out and "}" in captured.out

    def test_cli_llm_provider_override(self, mocker):
        """CLI no longer supports provider overrides via flags; config/env should be used."""
        cfg = ArchitextSettings()
        mock_load_settings = mocker.patch("src.cli.load_settings", return_value=cfg)
        mock_initialize = mocker.patch("src.cli.initialize_settings")
        mock_resolve = mocker.patch("src.cli.resolve_source", side_effect=ValueError("test"))
        
        sys.argv = ["cli.py", "index", ".", "--llm-provider", "openai"]
        
        from src.cli import main
        
        # argparse should fail for unknown flag
        with pytest.raises(SystemExit):
            main()
        
        # Provider should remain unchanged (default in settings)
        assert cfg.llm_provider == "local"
