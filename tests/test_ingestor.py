import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.ingestor import (
    resolve_source,
    _is_github_like_url,
    _compute_repo_hash,
    cleanup_cache,
)


class TestIsGithubLikeUrl:
    def test_github_url(self):
        assert _is_github_like_url("https://github.com/user/repo")
        assert _is_github_like_url("git@github.com:user/repo.git")

    def test_gitlab_url(self):
        assert _is_github_like_url("https://gitlab.com/user/repo")

    def test_local_path(self):
        assert not _is_github_like_url("/home/user/project")
        assert not _is_github_like_url("./local/path")


class TestComputeRepoHash:
    def test_stable_hash(self):
        url = "https://github.com/user/repo"
        h1 = _compute_repo_hash(url)
        h2 = _compute_repo_hash(url)
        assert h1 == h2
        assert len(h1) == 12

    def test_different_urls_different_hashes(self):
        h1 = _compute_repo_hash("https://github.com/user/repo1")
        h2 = _compute_repo_hash("https://github.com/user/repo2")
        assert h1 != h2


class TestResolveSource:
    def test_local_path_exists(self, tmp_path):
        """Resolving an existing local path returns its absolute path."""
        test_dir = tmp_path / "test_repo"
        test_dir.mkdir()
        
        result = resolve_source(str(test_dir))
        assert result == test_dir.resolve()

    def test_local_path_not_exists(self):
        """Resolving a non-existent local path raises ValueError."""
        with pytest.raises(ValueError, match="Source not found"):
            resolve_source("/nonexistent/path/that/does/not/exist")

    def test_remote_url_without_git(self):
        """Remote URL without GitPython raises RuntimeError."""
        with patch("src.ingestor.HAS_GIT", False):
            with pytest.raises(RuntimeError, match="GitPython is required"):
                resolve_source("https://github.com/user/repo")

    def test_remote_url_with_no_cache_fails(self):
        """Remote URL with use_cache=False raises ValueError."""
        with pytest.raises(ValueError, match="Remote source not allowed"):
            resolve_source("https://github.com/user/repo", use_cache=False)

    def test_ssh_format_url_recognized(self):
        """SSH-style git URLs are recognized as remote."""
        with pytest.raises(RuntimeError, match="GitPython is required"):
            with patch("src.ingestor.HAS_GIT", False):
                resolve_source("git@github.com:user/repo.git")

    @patch("src.ingestor.Repo")
    def test_clone_to_cache(self, mock_repo, tmp_path):
        """Remote URL is cloned to cache."""
        with patch("src.ingestor.CACHE_DIR", tmp_path):
            mock_repo.clone_from = MagicMock()
            
            # Mock successful clone
            result = resolve_source("https://github.com/user/repo")
            
            # Should create cache path
            assert "repo" in str(result).lower() or tmp_path.name in str(result)
            mock_repo.clone_from.assert_called_once()


class TestCleanupCache:
    @patch("src.ingestor.CACHE_DIR")
    def test_cleanup_nonexistent_cache(self, mock_cache_dir):
        """Cleanup on non-existent cache returns 0."""
        mock_cache_dir.exists.return_value = False
        
        result = cleanup_cache()
        assert result == 0

    @patch("src.ingestor.CACHE_DIR")
    def test_cleanup_empty_cache(self, mock_cache_dir, tmp_path):
        """Cleanup on empty cache returns 0."""
        mock_cache_dir.exists.return_value = True
        mock_cache_dir.iterdir.return_value = []
        
        result = cleanup_cache()
        assert result == 0
