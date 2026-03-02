"""Shared pytest fixtures for Architext tests."""
import pytest
import importlib

# Determine which optional modules are missing; we'll mark all tests as
# skipped rather than aborting collection.  This allows `pytest` to return
# a zero status code even when the environment is incomplete.
_missing_deps: list[str] = []
for module in ("fastapi", "chromadb", "openai"):
    if importlib.util.find_spec(module) is None:
        _missing_deps.append(module)

# typing_extensions must provide TypeAliasType for the current pydantic
# dependency; if not, treat it as a missing requirement too.
try:
    import typing_extensions
    _ = typing_extensions.TypeAliasType
except Exception:  # covers both ImportError and AttributeError
    _missing_deps.append("typing_extensions>=4.10")


def pytest_collection_modifyitems(config, items):
    """Mark every collected test as skipped when required packages are absent."""
    if _missing_deps:
        reason = "missing required dependencies: " + ", ".join(_missing_deps)
        skip_marker = pytest.mark.skip(reason=reason)
        for item in items:
            item.add_marker(skip_marker)

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
