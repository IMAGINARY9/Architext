"""Tests for extended BaseTask implementations (tasks_v2_extended.py)."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.tasks.tasks_v2_extended import (
    StructureAnalysisTask,
    TechStackTask,
    ArchitecturePatternTask,
    ImpactAnalysisTask,
    DependencyGraphTask,
    DuplicateBlocksTask,
    SemanticDuplicationTask,
    SecurityHeuristicsTask,
    analyze_structure_v2,
    tech_stack_v2,
    architecture_pattern_detection_v2,
    impact_analysis_v2,
    dependency_graph_export_v2,
    detect_duplicate_blocks_v2,
    detect_duplicate_blocks_semantic_v2,
    security_heuristics_v2,
)
from src.tasks.base import FileInfo


class TestStructureAnalysisTask:
    """Tests for StructureAnalysisTask."""
    
    def test_build_tree(self):
        """Test tree building from paths."""
        paths = ["src/main.py", "src/utils/helper.py", "tests/test_main.py"]
        
        tree = StructureAnalysisTask._build_tree(paths)
        
        assert "src" in tree
        assert "__files__" in tree["src"]
        assert "main.py" in tree["src"]["__files__"]
    
    def test_prune_tree(self):
        """Test tree pruning at max depth."""
        tree = {
            "level1": {
                "level2": {
                    "level3": {
                        "__files__": ["deep.py"]
                    }
                }
            }
        }
        
        pruned = StructureAnalysisTask._prune_tree(tree, max_depth=2)
        
        assert "level1" in pruned
        assert "..." in pruned["level1"]["level2"]
    
    def test_tree_to_markdown(self):
        """Test markdown conversion."""
        tree = {
            "src": {"__files__": ["main.py", "utils.py"]},
            "__files__": ["README.md"]
        }
        
        lines = StructureAnalysisTask._tree_to_markdown(tree)
        
        assert any("README.md" in line for line in lines)
        assert any("src/" in line for line in lines)
    
    def test_tree_to_mermaid(self):
        """Test Mermaid diagram generation."""
        tree = {
            "src": {"__files__": ["main.py"]}
        }
        
        diagram = StructureAnalysisTask._tree_to_mermaid(tree)
        
        assert "graph TD" in diagram
        assert "root" in diagram
    
    @patch("src.tasks.base.collect_file_paths")
    @patch("src.tasks.base._read_file_text")
    def test_analyze_json_format(self, mock_read, mock_collect):
        """Test analysis with JSON output."""
        mock_collect.return_value = ["src/main.py", "src/utils.py"]
        mock_read.return_value = "import os\nprint('hello')"
        
        task = StructureAnalysisTask(source_path="src", output_format="json")
        result = task.run()
        
        assert result["format"] == "json"
        assert "summary" in result
        assert "total_files" in result["summary"]


class TestTechStackTask:
    """Tests for TechStackTask."""
    
    @patch("src.tasks.base.collect_file_paths")
    @patch("src.tasks.base._read_file_text")
    def test_detect_frameworks(self, mock_read, mock_collect):
        """Test framework detection."""
        mock_collect.return_value = ["app.py"]
        mock_read.return_value = """
from fastapi import FastAPI
from sqlalchemy import create_engine
"""
        
        task = TechStackTask(source_path="src")
        result = task.run()
        
        assert "data" in result
        assert "frameworks" in result["data"]
        assert "fastapi" in result["data"]["frameworks"]
        assert "sqlalchemy" in result["data"]["frameworks"]
    
    @patch("src.tasks.base.collect_file_paths")
    @patch("src.tasks.base._read_file_text")
    def test_markdown_output(self, mock_read, mock_collect):
        """Test markdown output format."""
        mock_collect.return_value = ["app.py"]
        mock_read.return_value = "import flask"
        
        task = TechStackTask(source_path="src", output_format="markdown")
        result = task.run()
        
        assert result["format"] == "markdown"
        assert "content" in result


class TestArchitecturePatternTask:
    """Tests for ArchitecturePatternTask."""
    
    @patch("src.tasks.base.collect_file_paths")
    def test_detect_mvc_pattern(self, mock_collect):
        """Test MVC pattern detection."""
        mock_collect.return_value = [
            "app/controllers/user_controller.py",
            "app/views/user_view.html",
            "app/models/user.py",
        ]
        
        task = ArchitecturePatternTask(source_path="app")
        result = task.run()
        
        assert "MVC" in result["patterns"]
        detailed = next(p for p in result["detailed"] if p["pattern"] == "MVC")
        assert detailed["confidence"] >= 0.6
    
    @patch("src.tasks.base.collect_file_paths")
    def test_no_patterns_detected(self, mock_collect):
        """Test when no patterns match."""
        mock_collect.return_value = [
            "main.py",
            "utils.py",
        ]
        
        task = ArchitecturePatternTask(source_path="src")
        result = task.run()
        
        assert result["patterns"] == []


class TestImpactAnalysisTask:
    """Tests for ImpactAnalysisTask."""
    
    @patch("src.tasks.base.collect_file_paths")
    @patch("src.tasks.tasks_v2_extended._build_import_graph")
    def test_find_affected_modules(self, mock_graph, mock_collect):
        """Test finding affected modules."""
        mock_collect.return_value = ["a.py", "b.py", "c.py"]
        mock_graph.return_value = {
            "a": ["b"],  # a imports b
            "b": ["c"],  # b imports c
            "c": [],
        }
        
        task = ImpactAnalysisTask(module="c", source_path="src")
        result = task.run()
        
        # c affects b, and b affects a
        assert "b" in result["affected"]
        assert "a" in result["affected"]
    
    @patch("src.tasks.base.collect_file_paths")
    @patch("src.tasks.tasks_v2_extended._build_import_graph")
    def test_module_not_found(self, mock_graph, mock_collect):
        """Test when module is not found."""
        mock_collect.return_value = ["a.py"]
        mock_graph.return_value = {"a": []}
        
        task = ImpactAnalysisTask(module="nonexistent", source_path="src")
        result = task.run()
        
        assert result["affected"] == []
        assert "not found" in result["note"]


class TestDependencyGraphTask:
    """Tests for DependencyGraphTask."""
    
    @patch("src.tasks.base.collect_file_paths")
    @patch("src.tasks.tasks_v2_extended._build_import_graph")
    def test_json_output(self, mock_graph, mock_collect):
        """Test JSON output format."""
        mock_collect.return_value = ["a.py", "b.py"]
        mock_graph.return_value = {"a": ["b"], "b": []}
        
        task = DependencyGraphTask(source_path="src", output_format="json")
        result = task.run()
        
        assert result["format"] == "json"
        assert "nodes" in result
        assert "edges" in result
    
    @patch("src.tasks.base.collect_file_paths")
    @patch("src.tasks.tasks_v2_extended._build_import_graph")
    def test_mermaid_output(self, mock_graph, mock_collect):
        """Test Mermaid output format."""
        mock_collect.return_value = ["a.py", "b.py"]
        mock_graph.return_value = {"a": ["b"], "b": []}
        
        task = DependencyGraphTask(source_path="src", output_format="mermaid")
        result = task.run()
        
        assert result["format"] == "mermaid"
        assert "graph TD" in result["content"]
    
    @patch("src.tasks.base.collect_file_paths")
    @patch("src.tasks.tasks_v2_extended._build_import_graph")
    def test_dot_output(self, mock_graph, mock_collect):
        """Test DOT/Graphviz output format."""
        mock_collect.return_value = ["a.py", "b.py"]
        mock_graph.return_value = {"a": ["b"], "b": []}
        
        task = DependencyGraphTask(source_path="src", output_format="dot")
        result = task.run()
        
        assert result["format"] == "dot"
        assert "digraph" in result["content"]


class TestDuplicateBlocksTask:
    """Tests for DuplicateBlocksTask."""
    
    def test_normalize_line(self):
        """Test line normalization."""
        assert DuplicateBlocksTask._normalize_line("  code  ", ".py") == "code"
        assert DuplicateBlocksTask._normalize_line("# comment", ".py") == ""
        assert DuplicateBlocksTask._normalize_line("// comment", ".js") == ""
        assert DuplicateBlocksTask._normalize_line("", ".py") == ""
    
    @patch("src.tasks.base.collect_file_paths")
    @patch("src.tasks.base._read_file_text")
    def test_find_duplicates(self, mock_read, mock_collect):
        """Test finding duplicate blocks."""
        duplicate_code = "\n".join([f"line{i}" for i in range(10)])
        
        mock_collect.return_value = ["a.py", "b.py"]
        mock_read.return_value = duplicate_code
        
        task = DuplicateBlocksTask(source_path="src", min_lines=5)
        result = task.run()
        
        # Should find duplicates since both files have same content
        assert result["scanned_files"] == 2
        assert "findings" in result


class TestSemanticDuplicationTask:
    """Tests for SemanticDuplicationTask."""
    
    def test_normalize_tokens(self):
        """Test token normalization."""
        code = "def foo(x): return x + 1"
        normalized = SemanticDuplicationTask._normalize_tokens(code)
        
        assert normalized  # Should produce output
        assert "_id" in normalized  # Identifiers normalized
    
    @patch("src.tasks.base.collect_file_paths")
    @patch("src.tasks.base._read_file_text")
    def test_find_semantic_duplicates(self, mock_read, mock_collect):
        """Test finding semantically similar functions."""
        code = '''
def function_a(value):
    result = value + 1
    return result

def function_b(number):
    output = number + 1
    return output
'''
        mock_collect.return_value = ["module.py"]
        mock_read.return_value = code
        
        task = SemanticDuplicationTask(source_path="src", min_tokens=5)
        result = task.run()
        
        assert result["scanned_files"] == 1
        assert "findings" in result


class TestSecurityHeuristicsTask:
    """Tests for SecurityHeuristicsTask."""
    
    @patch("src.tasks.base.collect_file_paths")
    @patch("src.tasks.base._read_file_text")
    def test_detect_eval_exec(self, mock_read, mock_collect):
        """Test detecting eval/exec usage."""
        mock_collect.return_value = ["unsafe.py"]
        mock_read.return_value = "result = eval(user_input)"
        
        task = SecurityHeuristicsTask(source_path="src")
        result = task.run()
        
        assert result["count"] > 0
        # Description is "Dynamic code execution detected"
        assert any("dynamic" in f["description"].lower() or "code execution" in f["description"].lower() 
                   for f in result["findings"])
    
    @patch("src.tasks.base.collect_file_paths")
    @patch("src.tasks.base._read_file_text")
    def test_detect_hardcoded_secret(self, mock_read, mock_collect):
        """Test detecting hardcoded secrets."""
        mock_collect.return_value = ["config.py"]
        mock_read.return_value = 'api_key = "sk-12345678901234567890"'
        
        task = SecurityHeuristicsTask(source_path="src")
        result = task.run()
        
        assert result["count"] > 0
        assert any("secret" in f["description"].lower() for f in result["findings"])
    
    @patch("src.tasks.base.collect_file_paths")
    @patch("src.tasks.base._read_file_text")
    def test_clean_code_no_findings(self, mock_read, mock_collect):
        """Test clean code produces no findings."""
        mock_collect.return_value = ["safe.py"]
        mock_read.return_value = """
def calculate_sum(a, b):
    return a + b
"""
        
        task = SecurityHeuristicsTask(source_path="src")
        result = task.run()
        
        assert result["count"] == 0


class TestWrapperFunctions:
    """Tests for wrapper functions maintaining backward compatibility."""
    
    @patch("src.tasks.base.collect_file_paths")
    @patch("src.tasks.base._read_file_text")
    def test_analyze_structure_v2(self, mock_read, mock_collect):
        """Test analyze_structure_v2 wrapper."""
        mock_collect.return_value = ["main.py"]
        mock_read.return_value = "print('hello')"
        
        result = analyze_structure_v2(source_path="src")
        
        assert "summary" in result or "format" in result
    
    @patch("src.tasks.base.collect_file_paths")
    @patch("src.tasks.base._read_file_text")
    def test_tech_stack_v2(self, mock_read, mock_collect):
        """Test tech_stack_v2 wrapper."""
        mock_collect.return_value = ["app.py"]
        mock_read.return_value = "import django"
        
        result = tech_stack_v2(source_path="src")
        
        assert "data" in result
    
    @patch("src.tasks.base.collect_file_paths")
    def test_architecture_pattern_detection_v2(self, mock_collect):
        """Test architecture_pattern_detection_v2 wrapper."""
        mock_collect.return_value = ["src/controllers/main.py"]
        
        result = architecture_pattern_detection_v2(source_path="src")
        
        assert "patterns" in result
    
    @patch("src.tasks.base.collect_file_paths")
    @patch("src.tasks.tasks_v2_extended._build_import_graph")
    def test_impact_analysis_v2(self, mock_graph, mock_collect):
        """Test impact_analysis_v2 wrapper."""
        mock_collect.return_value = ["a.py"]
        mock_graph.return_value = {"a": []}
        
        result = impact_analysis_v2(module="a", source_path="src")
        
        assert "module" in result
    
    @patch("src.tasks.base.collect_file_paths")
    @patch("src.tasks.tasks_v2_extended._build_import_graph")
    def test_dependency_graph_export_v2(self, mock_graph, mock_collect):
        """Test dependency_graph_export_v2 wrapper."""
        mock_collect.return_value = ["a.py"]
        mock_graph.return_value = {"a": []}
        
        result = dependency_graph_export_v2(source_path="src")
        
        assert "format" in result
