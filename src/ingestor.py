"""Universal repository ingestion for local and remote sources.

This module handles:
- Local directory indexing
- Remote repo cloning (GitHub, GitLab, etc.)
- Authentication (SSH, PAT)
- Cache management
"""
import hashlib
import os
import shutil
from pathlib import Path
from typing import Optional, Union, Dict
from urllib.parse import urlparse

try:
    from git import Repo
    from git.exc import GitCommandError, InvalidGitRepositoryError
    HAS_GIT = True
except ImportError:
    HAS_GIT = False


CACHE_DIR = Path.home() / ".architext" / "cache"


def _is_github_like_url(source: str) -> bool:
    """Check if source is a GitHub/GitLab URL."""
    return any(
        domain in source.lower()
        for domain in ["github.com", "gitlab.com", "bitbucket.org", "gitea"]
    ) or source.startswith("git@")


def _compute_repo_hash(url: str) -> str:
    """Compute a stable hash for the repo URL."""
    return hashlib.sha256(url.encode()).hexdigest()[:12]


def _ensure_cache_dir() -> Path:
    """Ensure cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def resolve_source(source: str, use_cache: bool = True, ssh_key: Optional[str] = None) -> Path:
    """
    Resolve a source (local path or remote URL) to a local directory.
    
    Args:
        source: Local file path or remote git URL.
        use_cache: If True, clone remotes to ~/.architext/cache. If False, raise error.
    
    Returns:
        Path to the local directory to index.
    
    Raises:
        ValueError: If source is invalid or remote but use_cache=False.
        RuntimeError: If GitPython is not installed or cloning fails.
    """
    source = source.strip()
    
    # Check if it's a local path
    local_path = Path(source).expanduser()
    if local_path.exists() and local_path.is_dir():
        return local_path.resolve()
    
    # Check if it looks like a remote URL
    if _is_github_like_url(source):
        if not HAS_GIT:
            raise RuntimeError(
                "GitPython is required for remote repo support. "
                "Install: pip install GitPython"
            )
        
        if not use_cache:
            raise ValueError(
                f"Remote source not allowed without cache: {source}"
            )
        
        return _clone_to_cache(source, ssh_key=ssh_key)
    
    # Not found locally, not a recognized remote format
    if local_path.exists():
        raise ValueError(f"Source exists but is not a directory: {source}")
    
    raise ValueError(
        f"Source not found and not a recognized git URL: {source}\n"
        f"Expected: local path, GitHub URL, or git@host:repo format"
    )


def _clone_to_cache(url: str, ssh_key: Optional[str] = None) -> Path:
    """Clone a git repo to cache and return the path."""
    cache_dir = _ensure_cache_dir()
    repo_hash = _compute_repo_hash(url)
    repo_path = cache_dir / repo_hash

    env = _build_git_env(ssh_key)
    
    # If already cached, update it
    if repo_path.exists():
        print(f"Updating cached repo: {repo_path}")
        try:
            repo = Repo(repo_path)
            if env:
                repo.git.update_environment(**env)
            repo.remotes.origin.pull()
        except (InvalidGitRepositoryError, GitCommandError) as e:
            print(f"Warning: Could not update cache, using existing: {e}")
    else:
        # Clone fresh
        print(f"Cloning {url} to {repo_path}...")
        try:
            Repo.clone_from(url, repo_path, env=env)
        except GitCommandError as e:
            raise RuntimeError(f"Failed to clone {url}: {e}")
    
    return repo_path


def _build_git_env(ssh_key: Optional[str]) -> Dict[str, str]:
    if not ssh_key:
        return {}
    key_path = Path(ssh_key).expanduser().resolve()
    if not key_path.exists():
        raise ValueError(f"SSH key not found: {key_path}")
    ssh_command = f"ssh -i \"{key_path}\" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
    return {"GIT_SSH_COMMAND": ssh_command}


def cleanup_cache(max_age_days: int = 30) -> int:
    """
    Clean up old cached repos.
    
    Args:
        max_age_days: Remove repos not accessed in this many days.
    
    Returns:
        Number of repos removed.
    """
    import time
    
    if not CACHE_DIR.exists():
        return 0
    
    now = time.time()
    removed = 0
    
    for repo_dir in CACHE_DIR.iterdir():
        if not repo_dir.is_dir():
            continue
        
        # Check last access time
        atime = repo_dir.stat().st_atime
        age_days = (now - atime) / (24 * 3600)
        
        if age_days > max_age_days:
            print(f"Removing cached repo: {repo_dir.name} (unused for {age_days:.0f} days)")
            try:
                # On Windows, need to handle read-only files in .git
                def handle_remove_readonly(func, path, exc):
                    import stat
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                
                shutil.rmtree(repo_dir, onerror=handle_remove_readonly)
                removed += 1
            except Exception as e:
                print(f"Warning: Could not remove {repo_dir.name}: {e}")
    
    return removed
