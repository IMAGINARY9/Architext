import pytest
import os
import shutil

@pytest.fixture
def temp_repo_path(tmp_path):
    """Creates a temporary repository structure for testing."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    # Regular file
    (repo_dir / "main.py").write_text("print('Hello World')")
    
    # Hidden file (should be ignored)
    (repo_dir / ".env").write_text("SECRET=123")
    
    # Git directory (should be ignored)
    git_dir = repo_dir / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main")
    
    return str(repo_dir)

@pytest.fixture
def mock_settings(mocker):
    """Mocks LlamaIndex Settings to prevent actual model loading."""
    mocker.patch("src.indexer.Settings.llm")
    mocker.patch("src.indexer.Settings.embed_model")
    return mocker.patch("src.indexer.initialize_settings")
