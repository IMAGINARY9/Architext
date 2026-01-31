"""Tests for task result caching."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.tasks.cache import (
    CacheEntry,
    TaskResultCache,
    cached_task,
    get_task_cache,
)


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_is_expired_false_for_new_entry(self):
        entry = CacheEntry(
            task_name="test-task",
            result={"data": "value"},
            created_at=time.time(),
            source_hash="abc123",
            ttl_seconds=3600,
        )
        assert not entry.is_expired()

    def test_is_expired_true_after_ttl(self):
        entry = CacheEntry(
            task_name="test-task",
            result={"data": "value"},
            created_at=time.time() - 7200,  # 2 hours ago
            source_hash="abc123",
            ttl_seconds=3600,  # 1 hour TTL
        )
        assert entry.is_expired()

    def test_serialization_roundtrip(self):
        original = CacheEntry(
            task_name="test-task",
            result={"data": "value", "count": 42},
            created_at=12345.67,
            source_hash="abc123",
            ttl_seconds=1800,
        )
        data = original.to_dict()
        restored = CacheEntry.from_dict(data)
        
        assert restored.task_name == original.task_name
        assert restored.result == original.result
        assert restored.created_at == original.created_at
        assert restored.source_hash == original.source_hash
        assert restored.ttl_seconds == original.ttl_seconds


class TestTaskResultCache:
    """Tests for TaskResultCache class."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create a cache with a temporary directory."""
        return TaskResultCache(cache_dir=str(tmp_path), default_ttl=3600)

    @pytest.fixture
    def source_dir(self, tmp_path):
        """Create a source directory with files."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("print('hello')")
        (src / "utils.py").write_text("def helper(): pass")
        return src

    def test_get_returns_none_when_empty(self, cache):
        result = cache.get("analyze-structure", source_path="src")
        assert result is None

    def test_set_and_get(self, cache, source_dir):
        expected_result = {"files": 10, "languages": ["python"]}
        
        cache.set(
            "analyze-structure",
            expected_result,
            source_path=str(source_dir),
        )
        
        result = cache.get("analyze-structure", source_path=str(source_dir))
        assert result == expected_result

    def test_get_returns_none_after_ttl(self, cache, source_dir):
        cache.set(
            "analyze-structure",
            {"files": 10},
            source_path=str(source_dir),
            ttl=0.1,  # 100ms TTL
        )
        
        # Should be available immediately
        assert cache.get("analyze-structure", source_path=str(source_dir)) is not None
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired
        assert cache.get("analyze-structure", source_path=str(source_dir)) is None

    def test_cache_invalidated_on_source_change(self, cache, source_dir):
        cache.set(
            "analyze-structure",
            {"files": 2},
            source_path=str(source_dir),
        )
        
        # Should be available
        assert cache.get("analyze-structure", source_path=str(source_dir)) is not None
        
        # Modify source
        (source_dir / "new_file.py").write_text("# new")
        
        # Should be invalidated
        assert cache.get("analyze-structure", source_path=str(source_dir)) is None

    def test_different_tasks_cached_separately(self, cache, source_dir):
        cache.set("task-a", {"a": 1}, source_path=str(source_dir))
        cache.set("task-b", {"b": 2}, source_path=str(source_dir))
        
        assert cache.get("task-a", source_path=str(source_dir)) == {"a": 1}
        assert cache.get("task-b", source_path=str(source_dir)) == {"b": 2}

    def test_disk_persistence(self, tmp_path, source_dir):
        cache_dir = str(tmp_path / "cache")
        
        # Create cache and store data
        cache1 = TaskResultCache(cache_dir=cache_dir)
        cache1.set("analyze-structure", {"files": 5}, source_path=str(source_dir))
        
        # Create new cache instance (simulating restart)
        cache2 = TaskResultCache(cache_dir=cache_dir)
        
        # Should retrieve from disk
        result = cache2.get("analyze-structure", source_path=str(source_dir))
        assert result == {"files": 5}

    def test_invalidate_by_task_name(self, cache, source_dir):
        cache.set("task-a", {"a": 1}, source_path=str(source_dir))
        cache.set("task-b", {"b": 2}, source_path=str(source_dir))
        
        # Invalidate only task-a
        count = cache.invalidate(task_name="task-a")
        assert count >= 1
        
        # task-a should be gone
        assert cache.get("task-a", source_path=str(source_dir)) is None
        # task-b should remain
        assert cache.get("task-b", source_path=str(source_dir)) == {"b": 2}

    def test_clear_removes_all(self, cache, source_dir):
        cache.set("task-a", {"a": 1}, source_path=str(source_dir))
        cache.set("task-b", {"b": 2}, source_path=str(source_dir))
        
        cache.clear()
        
        assert cache.get("task-a", source_path=str(source_dir)) is None
        assert cache.get("task-b", source_path=str(source_dir)) is None

    def test_get_stats(self, cache, source_dir):
        cache.set("task-a", {"a": 1}, source_path=str(source_dir))
        
        stats = cache.get_stats()
        
        assert stats["enabled"] is True
        assert stats["memory_entries"] >= 1
        assert stats["disk_entries"] >= 1
        assert stats["disk_size_bytes"] > 0

    def test_disabled_cache_returns_none(self, tmp_path, source_dir):
        cache = TaskResultCache(cache_dir=str(tmp_path), enabled=False)
        
        cache.set("task-a", {"a": 1}, source_path=str(source_dir))
        result = cache.get("task-a", source_path=str(source_dir))
        
        assert result is None


class TestCachedTaskDecorator:
    """Tests for the @cached_task decorator."""

    @pytest.fixture
    def source_dir(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("print('hello')")
        return src

    def test_decorator_caches_result(self, tmp_path, source_dir):
        call_count = 0
        
        @cached_task(ttl=3600)
        def my_task(source_path=None, storage_path=None, progress_callback=None):
            nonlocal call_count
            call_count += 1
            return {"computed": True}
        
        # Setup cache
        with patch("src.tasks.cache.get_task_cache") as mock_get_cache:
            cache = TaskResultCache(cache_dir=str(tmp_path / "cache"))
            mock_get_cache.return_value = cache
            
            # First call - computes
            result1 = my_task(source_path=str(source_dir))
            assert result1 == {"computed": True}
            assert call_count == 1
            
            # Second call - from cache
            result2 = my_task(source_path=str(source_dir))
            assert result2 == {"computed": True}
            assert call_count == 1  # Still 1, used cache

    def test_decorator_respects_use_cache_false(self, tmp_path, source_dir):
        call_count = 0
        
        @cached_task(ttl=3600)
        def my_task(source_path=None, storage_path=None, progress_callback=None):
            nonlocal call_count
            call_count += 1
            return {"computed": True}
        
        with patch("src.tasks.cache.get_task_cache") as mock_get_cache:
            cache = TaskResultCache(cache_dir=str(tmp_path / "cache"))
            mock_get_cache.return_value = cache
            
            # First call
            my_task(source_path=str(source_dir))
            assert call_count == 1
            
            # Second call with use_cache=False
            my_task(source_path=str(source_dir), use_cache=False)
            assert call_count == 2  # Recomputed


class TestGlobalCache:
    """Tests for global cache singleton."""

    def test_get_task_cache_returns_same_instance(self, tmp_path):
        # Reset global cache
        import src.tasks.cache as cache_module
        cache_module._global_cache = None
        
        cache1 = get_task_cache(cache_dir=str(tmp_path))
        cache2 = get_task_cache()
        
        assert cache1 is cache2
