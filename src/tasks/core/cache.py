"""Task result caching with disk persistence.

This module provides caching for task results to avoid redundant computation
when the same analysis is requested multiple times.

Features:
- Disk persistence for cache entries (JSON-based)
- TTL-based cache expiration
- Source file hash-based invalidation
- Thread-safe access
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""
    task_name: str
    result: Dict[str, Any]
    created_at: float
    source_hash: str  # Hash of source file modification times
    ttl_seconds: float = 3600.0  # Default 1 hour TTL
    
    def is_expired(self) -> bool:
        """Check if this entry has expired based on TTL."""
        return time.time() > (self.created_at + self.ttl_seconds)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "task_name": self.task_name,
            "result": self.result,
            "created_at": self.created_at,
            "source_hash": self.source_hash,
            "ttl_seconds": self.ttl_seconds,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Deserialize from dictionary."""
        return cls(
            task_name=data["task_name"],
            result=data["result"],
            created_at=data["created_at"],
            source_hash=data["source_hash"],
            ttl_seconds=data.get("ttl_seconds", 3600.0),
        )


class TaskResultCache:
    """
    Persistent cache for task results.
    
    Cache entries are invalidated when:
    - TTL expires (default: 1 hour)
    - Source files have changed (based on modification time hash)
    - Manual invalidation
    
    Usage:
        cache = TaskResultCache(cache_dir="~/.architext/cache")
        
        # Check for cached result
        result = cache.get("analyze-structure", source_path="src")
        if result is not None:
            return result
        
        # Compute and cache
        result = compute_expensive_analysis()
        cache.set("analyze-structure", source_path="src", result=result)
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        default_ttl: float = 3600.0,
        enabled: bool = True,
    ):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files. Defaults to ~/.architext/cache
            default_ttl: Default time-to-live in seconds for cache entries
            enabled: Whether caching is enabled
        """
        if cache_dir is None:
            cache_dir = str(Path.home() / ".architext" / "cache")
        
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        self.enabled = enabled
        self._lock = threading.Lock()
        self._memory_cache: Dict[str, CacheEntry] = {}
        
        # Create cache directory if needed
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_source_hash(self, source_path: Optional[str]) -> str:
        """
        Compute a hash of source file modification times.
        
        This is used to invalidate cache when source files change.
        """
        if not source_path:
            return "no-source"
        
        source = Path(source_path)
        if not source.exists():
            return "source-not-found"
        
        # Collect modification times of top-level files (for performance)
        mtimes = []
        try:
            for item in source.iterdir():
                if item.is_file():
                    mtimes.append(f"{item.name}:{item.stat().st_mtime}")
                elif item.is_dir() and not item.name.startswith("."):
                    # Include directory modification time
                    mtimes.append(f"{item.name}/:{item.stat().st_mtime}")
        except Exception:
            return f"source-error-{time.time()}"
        
        mtimes.sort()
        content = "\n".join(mtimes)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _cache_key(
        self,
        task_name: str,
        source_path: Optional[str] = None,
        storage_path: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a unique cache key for a task invocation."""
        parts = [task_name]
        if source_path:
            parts.append(f"src:{Path(source_path).resolve()}")
        if storage_path:
            parts.append(f"store:{Path(storage_path).resolve()}")
        # Include any additional kwargs that affect the result
        for k, v in sorted(kwargs.items()):
            if v is not None:
                parts.append(f"{k}:{v}")
        
        key = "|".join(parts)
        return hashlib.sha256(key.encode()).hexdigest()[:32]
    
    def _cache_file_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(
        self,
        task_name: str,
        source_path: Optional[str] = None,
        storage_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a cached result if available and valid.
        
        Returns None if:
        - Caching is disabled
        - No cache entry exists
        - Cache entry is expired
        - Source files have changed
        """
        if not self.enabled:
            return None
        
        cache_key = self._cache_key(task_name, source_path, storage_path, **kwargs)
        
        # Check memory cache first
        with self._lock:
            if cache_key in self._memory_cache:
                entry = self._memory_cache[cache_key]
                if not entry.is_expired():
                    current_hash = self._compute_source_hash(source_path)
                    if entry.source_hash == current_hash:
                        return entry.result
                # Invalid - remove from memory cache
                del self._memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self._cache_file_path(cache_key)
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            entry = CacheEntry.from_dict(data)
            
            # Validate entry
            if entry.is_expired():
                cache_file.unlink(missing_ok=True)
                return None
            
            current_hash = self._compute_source_hash(source_path)
            if entry.source_hash != current_hash:
                cache_file.unlink(missing_ok=True)
                return None
            
            # Valid - store in memory cache too
            with self._lock:
                self._memory_cache[cache_key] = entry
            
            return entry.result
            
        except Exception:
            # Corrupted cache file - remove it
            cache_file.unlink(missing_ok=True)
            return None
    
    def set(
        self,
        task_name: str,
        result: Dict[str, Any],
        source_path: Optional[str] = None,
        storage_path: Optional[str] = None,
        ttl: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Cache a task result.
        
        Args:
            task_name: Name of the task
            result: The task result to cache
            source_path: Source path used for the task
            storage_path: Storage path used for the task
            ttl: Time-to-live in seconds (uses default_ttl if None)
            **kwargs: Additional parameters that affect the cache key
        """
        if not self.enabled:
            return
        
        cache_key = self._cache_key(task_name, source_path, storage_path, **kwargs)
        source_hash = self._compute_source_hash(source_path)
        
        entry = CacheEntry(
            task_name=task_name,
            result=result,
            created_at=time.time(),
            source_hash=source_hash,
            ttl_seconds=ttl if ttl is not None else self.default_ttl,
        )
        
        # Store in memory
        with self._lock:
            self._memory_cache[cache_key] = entry
        
        # Store on disk
        cache_file = self._cache_file_path(cache_key)
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(entry.to_dict(), f, indent=2)
        except Exception:
            pass  # Disk write failure shouldn't break the application
    
    def invalidate(
        self,
        task_name: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries.
        
        Args:
            task_name: If provided, only invalidate entries for this task
            source_path: If provided, only invalidate entries for this source
            
        Returns:
            Number of entries invalidated
        """
        count = 0
        
        # Invalidate memory cache
        with self._lock:
            keys_to_remove = []
            for key, entry in self._memory_cache.items():
                if task_name and entry.task_name != task_name:
                    continue
                if source_path:
                    current_hash = self._compute_source_hash(source_path)
                    if entry.source_hash != current_hash:
                        keys_to_remove.append(key)
                        continue
                keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._memory_cache[key]
                count += 1
        
        # Invalidate disk cache
        if task_name is None and source_path is None:
            # Clear all
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                    count += 1
                except Exception:
                    pass
        else:
            # Selective invalidation - need to read each file
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if task_name and data.get("task_name") != task_name:
                        continue
                    cache_file.unlink()
                    count += 1
                except Exception:
                    pass
        
        return count
    
    def clear(self) -> int:
        """Clear all cache entries. Returns the number of entries cleared."""
        return self.invalidate()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        memory_count = len(self._memory_cache)
        disk_count = len(list(self.cache_dir.glob("*.json")))
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.json"))
        
        return {
            "enabled": self.enabled,
            "cache_dir": str(self.cache_dir),
            "memory_entries": memory_count,
            "disk_entries": disk_count,
            "disk_size_bytes": total_size,
            "default_ttl_seconds": self.default_ttl,
        }


# Global cache instance
_global_cache: Optional[TaskResultCache] = None
_cache_lock = threading.Lock()


def get_task_cache(
    cache_dir: Optional[str] = None,
    default_ttl: float = 3600.0,
    enabled: bool = True,
) -> TaskResultCache:
    """
    Get the global task cache instance.
    
    Creates the cache on first call, reuses it on subsequent calls.
    """
    global _global_cache
    
    with _cache_lock:
        if _global_cache is None:
            _global_cache = TaskResultCache(
                cache_dir=cache_dir,
                default_ttl=default_ttl,
                enabled=enabled,
            )
        return _global_cache


def cached_task(
    ttl: Optional[float] = None,
    cache_key_params: Optional[List[str]] = None,
) -> Callable:
    """
    Decorator to add caching to a task function.
    
    Usage:
        @cached_task(ttl=1800)  # 30 minute cache
        def analyze_structure(source_path=None, storage_path=None, progress_callback=None):
            ...
    
    Args:
        ttl: Time-to-live for cache entries in seconds
        cache_key_params: Additional parameter names to include in cache key
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(
            storage_path: Optional[str] = None,
            source_path: Optional[str] = None,
            progress_callback: Optional[Callable] = None,
            use_cache: bool = True,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            cache = get_task_cache()
            task_name = func.__name__
            
            # Build extra kwargs for cache key
            extra_kwargs = {}
            if cache_key_params:
                for param in cache_key_params:
                    if param in kwargs:
                        extra_kwargs[param] = kwargs[param]
            
            # Try to get from cache
            if use_cache:
                cached_result = cache.get(
                    task_name,
                    source_path=source_path,
                    storage_path=storage_path,
                    **extra_kwargs,
                )
                if cached_result is not None:
                    return cached_result
            
            # Execute the task
            result = func(
                storage_path=storage_path,
                source_path=source_path,
                progress_callback=progress_callback,
                **kwargs,
            )
            
            # Cache the result
            if use_cache:
                cache.set(
                    task_name,
                    result,
                    source_path=source_path,
                    storage_path=storage_path,
                    ttl=ttl,
                    **extra_kwargs,
                )
            
            return result
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator
