"""Tests for BaseTask-based task implementations."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from typing import List

from src.tasks.analysis import (
    AntiPatternDetectionTask,
    SilentFailuresTask,
    TestMappingTask,
    HealthScoreTask,
    detect_anti_patterns_v2,
    identify_silent_failures_v2,
    test_mapping_analysis_v2,
    health_score_v2,
)
from src.tasks.core.base import FileInfo


# =============================================================================
# AntiPatternDetectionTask Tests
# =============================================================================

class TestAntiPatternDetectionTask:
    """Tests for anti-pattern detection task."""
    
    def test_detects_large_files(self):
        """Test that large files are detected."""
        task = AntiPatternDetectionTask()
        
        # Create mock file with many lines
        large_file = FileInfo(
            path="large.py",
            extension=".py",
            language="Python",
            content="\n" * 1000,  # 1000 lines
            line_count=1000,
        )
        
        result = task.analyze([large_file])
        
        issues = result["issues"]
        large_file_issues = [i for i in issues if i["type"] == "large_file"]
        assert len(large_file_issues) >= 1
        assert "1000 lines" in large_file_issues[0]["details"]
    
    def test_detects_god_objects(self):
        """Test that god objects (high function count) are detected."""
        task = AntiPatternDetectionTask()
        
        # Create file with many functions
        functions = "\n".join([f"def func_{i}(): pass" for i in range(50)])
        
        god_file = FileInfo(
            path="god.py",
            extension=".py",
            language="Python",
            content=functions,
            line_count=50,
        )
        
        result = task.analyze([god_file])
        
        issues = result["issues"]
        god_issues = [i for i in issues if i["type"] == "god_object"]
        assert len(god_issues) >= 1
        assert "function count" in god_issues[0]["details"].lower()
    
    def test_detects_missing_tests(self):
        """Test that missing tests are detected."""
        task = AntiPatternDetectionTask()
        
        # Files without any test files
        source_file = FileInfo(
            path="src/main.py",
            extension=".py",
            language="Python",
            content="print('hello')",
            line_count=1,
        )
        
        result = task.analyze([source_file])
        
        issues = result["issues"]
        test_issues = [i for i in issues if i["type"] == "missing_tests"]
        assert len(test_issues) >= 1
    
    def test_returns_summary(self):
        """Test that summary is included in result."""
        task = AntiPatternDetectionTask()
        
        file = FileInfo(
            path="app.py",
            extension=".py",
            language="Python",
            content="x = 1",
            line_count=1,
        )
        
        result = task.analyze([file])
        
        assert "summary" in result
        assert "total_issues" in result["summary"]
        assert "total_files" in result["summary"]


class TestSilentFailuresTask:
    """Tests for silent failures detection task."""
    
    def test_detects_python_silent_except(self):
        """Test detection of Python silent exception handlers."""
        task = SilentFailuresTask()
        
        code = """
try:
    something()
except:
    pass
"""
        
        file = FileInfo(
            path="silent.py",
            extension=".py",
            language="Python",
            content=code,
            line_count=5,
        )
        
        result = task.analyze([file])
        
        assert result["count"] >= 1
        assert any(f["type"] == "silent_exception" for f in result["findings"])
    
    def test_detects_python_silent_continue(self):
        """Test detection of except: continue patterns."""
        task = SilentFailuresTask()
        
        code = """
for item in items:
    try:
        process(item)
    except Exception:
        continue
"""
        
        file = FileInfo(
            path="loop.py",
            extension=".py",
            language="Python",
            content=code,
            line_count=6,
        )
        
        result = task.analyze([file])
        
        assert result["count"] >= 1
    
    def test_detects_js_empty_catch(self):
        """Test detection of empty JavaScript catch blocks."""
        task = SilentFailuresTask()
        
        code = """
try {
    doSomething();
} catch(e) {}
"""
        
        file = FileInfo(
            path="silent.js",
            extension=".js",
            language="JavaScript",
            content=code,
            line_count=4,
        )
        
        result = task.analyze([file])
        
        assert result["count"] >= 1
        assert any(f["snippet"] == "empty catch block" for f in result["findings"])
    
    def test_no_false_positives_for_proper_handling(self):
        """Test that proper exception handling is not flagged."""
        task = SilentFailuresTask()
        
        code = """
try:
    something()
except ValueError as e:
    logger.error(f"Error: {e}")
    raise
"""
        
        file = FileInfo(
            path="proper.py",
            extension=".py",
            language="Python",
            content=code,
            line_count=6,
        )
        
        result = task.analyze([file])
        
        # Should not flag proper error handling
        assert result["count"] == 0


class TestTestMappingTaskAnalysis:
    """Tests for test mapping task."""
    
    def test_maps_test_to_source(self):
        """Test that test files are mapped to source files."""
        task = TestMappingTask()
        
        files = [
            FileInfo(path="src/calculator.py", extension=".py", language="Python"),
            FileInfo(path="tests/test_calculator.py", extension=".py", language="Python"),
        ]
        
        result = task.analyze(files)
        
        assert result["total_sources"] == 1
        assert result["total_tests"] == 1
        assert "calculator.py" in str(result["mapping"])
    
    def test_identifies_untested_files(self):
        """Test that untested files are identified."""
        task = TestMappingTask()
        
        files = [
            FileInfo(path="src/calculator.py", extension=".py", language="Python"),
            FileInfo(path="src/utils.py", extension=".py", language="Python"),
            FileInfo(path="tests/test_calculator.py", extension=".py", language="Python"),
        ]
        
        result = task.analyze(files)
        
        assert len(result["untested"]) >= 1
        assert any("utils.py" in u for u in result["untested"])
    
    def test_skips_init_files(self):
        """Test that __init__.py files are not counted as untested."""
        task = TestMappingTask()
        
        files = [
            FileInfo(path="src/__init__.py", extension=".py", language="Python"),
            FileInfo(path="src/conftest.py", extension=".py", language="Python"),
        ]
        
        result = task.analyze(files)
        
        assert len(result["untested"]) == 0
    
    def test_recognizes_test_directories(self):
        """Test that files in test directories are recognized as tests."""
        task = TestMappingTask()
        
        assert task._is_test_file("tests/test_something.py")
        assert task._is_test_file("test/unit/test_foo.py")
        assert task._is_test_file("__tests__/component.test.js")
        assert not task._is_test_file("src/main.py")


class TestHealthScoreTask:
    """Tests for health score task."""
    
    def test_calculates_score(self):
        """Test that health score is calculated."""
        task = HealthScoreTask()
        
        files = [
            FileInfo(
                path="src/main.py",
                extension=".py",
                language="Python",
                content="x = 1",
                line_count=1,
            ),
            FileInfo(
                path="tests/test_main.py",
                extension=".py",
                language="Python",
                content="def test_x(): pass",
                line_count=1,
            ),
            FileInfo(
                path="README.md",
                extension=".md",
                language="Markdown",
                content="# Project",
                line_count=1,
            ),
        ]
        
        result = task.analyze(files)
        
        assert "score" in result
        assert "grade" in result
        assert 0 <= result["score"] <= 100
        assert result["grade"] in ["A", "B", "C", "D", "F"]
    
    def test_grade_calculation(self):
        """Test grade calculation from score."""
        assert HealthScoreTask._calculate_grade(95) == "A"
        assert HealthScoreTask._calculate_grade(85) == "B"
        assert HealthScoreTask._calculate_grade(75) == "C"
        assert HealthScoreTask._calculate_grade(65) == "D"
        assert HealthScoreTask._calculate_grade(50) == "F"
    
    def test_includes_metrics(self):
        """Test that individual metrics are included."""
        task = HealthScoreTask()
        
        files = [
            FileInfo(
                path="app.py",
                extension=".py",
                language="Python",
                content="print('hi')",
                line_count=1,
            ),
        ]
        
        result = task.analyze(files)
        
        assert "metrics" in result
        assert "test_presence" in result["metrics"]
        assert "doc_presence" in result["metrics"]
        assert "file_size" in result["metrics"]
    
    def test_includes_file_counts(self):
        """Test that file counts are included."""
        task = HealthScoreTask()
        
        files = [
            FileInfo(path="a.py", extension=".py", language="Python", content="x=1", line_count=1),
            FileInfo(path="b.py", extension=".py", language="Python", content="y=2", line_count=1),
        ]
        
        result = task.analyze(files)
        
        assert "file_counts" in result
        assert result["file_counts"]["total"] == 2


# =============================================================================
# Wrapper Function Tests
# =============================================================================

class TestWrapperFunctions:
    """Tests for wrapper functions."""
    
    @patch("src.tasks.analysis.anti_patterns.AntiPatternDetectionTask.run")
    def test_detect_anti_patterns_v2_wrapper(self, mock_run):
        """Test wrapper function calls task correctly."""
        mock_run.return_value = {"issues": []}
        
        result = detect_anti_patterns_v2(source_path="src")
        
        mock_run.assert_called_once()
        assert result == {"issues": []}
    
    @patch("src.tasks.analysis.quality.SilentFailuresTask.run")
    def test_identify_silent_failures_v2_wrapper(self, mock_run):
        """Test wrapper function calls task correctly."""
        mock_run.return_value = {"findings": [], "count": 0}
        
        result = identify_silent_failures_v2(source_path="src")
        
        mock_run.assert_called_once()
    
    @patch("src.tasks.analysis.quality.TestMappingTask.run")
    def test_test_mapping_analysis_v2_wrapper(self, mock_run):
        """Test wrapper function calls task correctly."""
        mock_run.return_value = {"tested_ratio": 0.5}
        
        result = test_mapping_analysis_v2(source_path="src")
        
        mock_run.assert_called_once()
    
    @patch("src.tasks.analysis.health.HealthScoreTask.run")
    def test_health_score_v2_wrapper(self, mock_run):
        """Test wrapper function calls task correctly."""
        mock_run.return_value = {"score": 75, "grade": "C"}
        
        result = health_score_v2(source_path="src")
        
        mock_run.assert_called_once()
        assert result["score"] == 75
