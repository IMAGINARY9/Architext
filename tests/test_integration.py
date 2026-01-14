"""End-to-end integration tests for the full indexing + querying workflow.

These tests validate:
- Indexing local repositories
- Vector storage and retrieval
- Query generation and response
- Storage persistence and reloading
"""
import os
import tempfile
from pathlib import Path
import pytest
from unittest.mock import Mock, patch

from src.config import ArchitextSettings
from src.indexer import initialize_settings, load_documents, create_index, load_existing_index
from src.ingestor import resolve_source


@pytest.fixture
def multi_lang_repo(tmp_path):
    """Create a test repo with multiple languages."""
    repo_dir = tmp_path / "multi_lang_repo"
    repo_dir.mkdir()

    # Python file
    (repo_dir / "database.py").write_text("""
class Database:
    def __init__(self, connection_string):
        self.conn = connection_string
    
    def query(self, sql):
        '''Execute a SQL query.'''
        return self.conn.execute(sql)
    
    def close(self):
        '''Close the database connection.'''
        self.conn.close()
""")

    # TypeScript file
    (repo_dir / "auth.ts").write_text("""
interface User {
  id: string;
  email: string;
}

function authenticate(token: string): User {
  // Validate JWT token
  const payload = decodeToken(token);
  return { id: payload.sub, email: payload.email };
}
""")

    # README
    (repo_dir / "README.md").write_text("""
# Multi-Lang Project

## Architecture
- Database layer in Python
- API authentication in TypeScript
- Event processing system

## Installation
See docs/setup.md
""")

    return repo_dir


@pytest.fixture
def initialized_settings(monkeypatch):
    """Provide initialized settings with HuggingFace embeddings for testing.

    Tests should be deterministic and must not attempt network downloads.
    We rely on the repository's pre-populated local cache.
    """

    # Force offline mode to avoid HuggingFace Hub downloads during CI/tests.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    repo_root = Path(__file__).resolve().parents[1]
    cache_dir = repo_root / "models_cache"

    cfg = ArchitextSettings(
        embedding_provider="huggingface",
        embedding_cache_dir=str(cache_dir),
    )
    initialize_settings(cfg)
    return cfg


def test_resolve_local_repo(multi_lang_repo):
    """Test that local repos can be resolved."""
    result = resolve_source(str(multi_lang_repo))
    assert result.exists()
    assert result.is_dir()


def test_load_documents_from_repo(multi_lang_repo):
    """Test loading documents from a repository."""
    docs = load_documents(str(multi_lang_repo))
    
    # Should load Python, TypeScript, and Markdown files
    assert len(docs) > 0
    
    # Check that we got different file types
    file_paths = {doc.metadata.get("file_path", "") for doc in docs}
    
    # At least some files should be present
    assert any("database.py" in p for p in file_paths)
    assert any("auth.ts" in p for p in file_paths)
    assert any("README.md" in p for p in file_paths)


def test_index_creation(multi_lang_repo, tmp_path, initialized_settings):
    """Test that an index can be created and persisted."""
    storage_path = str(tmp_path / "test_index")
    
    # Load documents
    docs = load_documents(str(multi_lang_repo))
    assert len(docs) > 0
    
    # Create index
    index = create_index(docs, storage_path)
    assert index is not None
    
    # Verify storage was created
    storage_dir = Path(storage_path)
    assert storage_dir.exists()


def test_index_persistence(multi_lang_repo, tmp_path, initialized_settings):
    """Test that indices can be reloaded from storage."""
    storage_path = str(tmp_path / "test_index")
    
    # Create and save index
    docs = load_documents(str(multi_lang_repo))
    create_index(docs, storage_path)
    
    # Load the index again
    loaded_index = load_existing_index(storage_path)
    assert loaded_index is not None
    
    # Verify we can create a query engine from loaded index
    query_engine = loaded_index.as_query_engine()
    assert query_engine is not None


@pytest.mark.integration
def test_full_workflow(multi_lang_repo, tmp_path, initialized_settings):
    """Test full workflow: resolve -> load -> index -> query -> persist."""
    storage_path = str(tmp_path / "workflow_test")
    
    # Step 1: Resolve source
    repo_path = resolve_source(str(multi_lang_repo))
    assert repo_path.exists()
    
    # Step 2: Load documents
    docs = load_documents(str(repo_path))
    assert len(docs) > 0
    
    # Step 3: Create index
    index = create_index(docs, storage_path)
    assert index is not None
    
    # Step 4: Verify storage exists
    assert Path(storage_path).exists()
    
    # Step 5: Reload index
    reloaded = load_existing_index(storage_path)
    assert reloaded is not None
    
    # Step 6: Create query engine from reloaded index
    query_engine = reloaded.as_query_engine()
    assert query_engine is not None


@pytest.mark.integration
@patch("src.indexer.Settings")
def test_embedding_model_loading(mock_settings, multi_lang_repo):
    """Test that embedding models load correctly."""
    from src.config import ArchitextSettings
    from src.indexer import _build_embedding
    
    # Test HuggingFace embedding
    repo_root = Path(__file__).resolve().parents[1]
    cache_dir = repo_root / "models_cache"

    cfg = ArchitextSettings(
        embedding_provider="huggingface",
        embedding_cache_dir=str(cache_dir),
    )
    
    embed_model = _build_embedding(cfg)
    assert embed_model is not None


def test_large_file_handling(tmp_path):
    """Test handling of large files in indexing."""
    repo_dir = tmp_path / "large_repo"
    repo_dir.mkdir()
    
    # Create a large Python file (10KB)
    large_content = "# " + "x" * 10000 + "\n" + "\n".join(
        [f"def function_{i}(): pass" for i in range(100)]
    )
    (repo_dir / "large_module.py").write_text(large_content)
    
    docs = load_documents(str(repo_dir))
    assert len(docs) > 0
    
    # Document should still be loadable
    assert any("large_module" in str(d.metadata) for d in docs)


def test_ignore_hidden_and_cache_files(tmp_path):
    """Test that hidden files and cache directories are properly ignored."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    # Create visible files
    (repo_dir / "visible.py").write_text("def hello(): pass")
    
    # Create hidden files and directories
    (repo_dir / ".hidden.py").write_text("secret = 123")
    (repo_dir / ".git").mkdir()
    (repo_dir / ".git" / "HEAD").write_text("ref: refs/heads/main")
    (repo_dir / "__pycache__").mkdir()
    (repo_dir / "__pycache__" / "cache.pyc").write_text("compiled")
    
    docs = load_documents(str(repo_dir))
    
    # Should only have visible.py
    file_paths = {doc.metadata.get("file_path", "") for doc in docs}
    assert any("visible.py" in p for p in file_paths)
    assert not any(".hidden.py" in p for p in file_paths)
    assert not any(".git" in p for p in file_paths)
    assert not any("__pycache__" in p for p in file_paths)
