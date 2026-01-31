"""BaseTask-based implementations of analysis tasks.

This module provides class-based implementations of common analysis tasks
using the BaseTask pattern for consistency and reduced boilerplate.

Each class has a corresponding wrapper function that maintains backward
compatibility with the original function-based API.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from src.tasks.base import (
    BaseTask,
    FileInfo,
    PYTHON_EXTENSIONS,
    JS_TS_EXTENSIONS,
    CODE_EXTENSIONS,
    DOCUMENTATION_EXTENSIONS,
    calculate_ratio,
)
from src.tasks.shared import (
    DEFAULT_EXTENSIONS,
    _classify_import_clusters,
    _extract_imports,
    _line_number_from_index,
)


# =============================================================================
# Anti-Pattern Detection Task
# =============================================================================

class AntiPatternDetectionTask(BaseTask):
    """
    Detect code anti-patterns like god objects, large files, and mixed responsibilities.
    
    Analyzes Python and JavaScript/TypeScript files for common code smells.
    """
    
    # Thresholds for pattern detection
    LARGE_FILE_THRESHOLD = 800
    HIGH_FUNCTION_COUNT = 40
    MIXED_RESPONSIBILITY_MIN_LINES = 200
    MIXED_RESPONSIBILITY_MIN_FUNCTIONS = 20
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        source_path: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ):
        super().__init__(
            storage_path=storage_path,
            source_path=source_path,
            progress_callback=progress_callback,
            extensions=CODE_EXTENSIONS | DOCUMENTATION_EXTENSIONS,
            load_content=True,
        )
    
    def analyze(self, files: List[FileInfo]) -> Dict[str, Any]:
        """Analyze files for anti-patterns."""
        issues: List[Dict[str, Any]] = []
        large_files: List[tuple] = []
        
        total_files = len(files)
        directories = Counter(Path(f.path).parent for f in files)
        
        # Documentation and test file counts
        doc_files = [f for f in files if f.extension in DOCUMENTATION_EXTENSIONS]
        test_files = [f for f in files if "test" in Path(f.path).name.lower()]
        
        self._report_progress("analyze", f"Scanning {total_files} files for anti-patterns")
        
        for file in files:
            if not file.content:
                continue
            
            lines = file.content.splitlines()
            line_count = len(lines)
            
            # Check for large files
            if line_count > self.LARGE_FILE_THRESHOLD:
                large_files.append((file.path, line_count))
            
            # Count functions (Python and JS/TS)
            function_count = (
                len(re.findall(r"\bdef\s+\w+\b", file.content)) +
                len(re.findall(r"\bfunction\s+\w+\b", file.content))
            )
            
            # God object detection
            if function_count > self.HIGH_FUNCTION_COUNT:
                issues.append({
                    "type": "god_object",
                    "file": file.path,
                    "severity": "high",
                    "details": f"High function count: {function_count}",
                })
            
            # Mixed responsibilities detection
            imports = _extract_imports(file.path, file.content)
            clusters = _classify_import_clusters(imports)
            
            if (
                len(clusters) >= 2 and
                (line_count > self.MIXED_RESPONSIBILITY_MIN_LINES or 
                 function_count > self.MIXED_RESPONSIBILITY_MIN_FUNCTIONS)
            ):
                issues.append({
                    "type": "mixed_responsibilities",
                    "file": file.path,
                    "severity": "medium",
                    "details": f"Multiple import clusters detected: {', '.join(clusters)}",
                })
        
        # Add large file issues
        for path, count in large_files[:10]:
            issues.append({
                "type": "large_file",
                "file": path,
                "severity": "medium",
                "details": f"File has {count} lines",
            })
        
        # Check for missing tests and documentation
        doc_ratio = calculate_ratio(len(doc_files), total_files)
        test_ratio = calculate_ratio(len(test_files), total_files)
        
        if not test_files:
            issues.append({
                "type": "missing_tests",
                "severity": "medium",
                "details": "No test files detected (filenames containing 'test')",
            })
        elif test_ratio < 0.03:
            issues.append({
                "type": "low_test_presence",
                "severity": "low",
                "details": f"Test files ratio is low: {round(test_ratio, 3)}",
            })
        
        if not doc_files:
            issues.append({
                "type": "missing_documentation",
                "severity": "low",
                "details": "No .md or .rst documentation files found",
            })
        
        # Aggregate severity
        severity_counts = Counter(i["severity"] for i in issues)
        
        return {
            "issues": issues,
            "summary": {
                "total_issues": len(issues),
                "by_severity": dict(severity_counts),
                "total_files": total_files,
                "large_files": len(large_files),
                "doc_ratio": round(doc_ratio, 3),
                "test_ratio": round(test_ratio, 3),
            },
        }


def detect_anti_patterns_v2(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Detect code anti-patterns using BaseTask pattern.
    
    This is a wrapper function maintaining backward compatibility.
    """
    task = AntiPatternDetectionTask(
        storage_path=storage_path,
        source_path=source_path,
        progress_callback=progress_callback,
    )
    return task.run()


# =============================================================================
# Silent Failures Detection Task
# =============================================================================

class SilentFailuresTask(BaseTask):
    """
    Detect exception handlers that silently swallow errors.
    
    Finds patterns like:
    - except: pass
    - except Exception: continue
    - catch(e) {} (empty JS/TS catch blocks)
    """
    
    # Patterns that indicate silent exception handling
    SILENT_HANDLERS = {"pass", "continue", "return", "..."}
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        source_path: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ):
        super().__init__(
            storage_path=storage_path,
            source_path=source_path,
            progress_callback=progress_callback,
            extensions=PYTHON_EXTENSIONS | JS_TS_EXTENSIONS,
            load_content=True,
        )
    
    def analyze(self, files: List[FileInfo]) -> Dict[str, Any]:
        """Find silent exception handlers."""
        findings: List[Dict[str, Any]] = []
        
        self._report_progress("analyze", f"Scanning {len(files)} files for silent failures")
        
        for file in files:
            if not file.content:
                continue
            
            if file.extension == ".py":
                findings.extend(self._analyze_python(file))
            elif file.extension in JS_TS_EXTENSIONS:
                findings.extend(self._analyze_js_ts(file))
        
        return {
            "findings": findings,
            "count": len(findings),
            "summary": f"Found {len(findings)} silent exception handlers",
        }
    
    def _analyze_python(self, file: FileInfo) -> List[Dict[str, Any]]:
        """Analyze Python file for silent exception handlers."""
        findings = []
        lines = file.content.splitlines()
        
        for idx, line in enumerate(lines):
            # Look for except clauses
            if not re.match(r"\s*except\b", line):
                continue
            if not line.rstrip().endswith(":"):
                continue
            
            base_indent = len(line) - len(line.lstrip())
            cursor = idx + 1
            
            while cursor < len(lines):
                candidate = lines[cursor]
                
                # Skip empty lines and comments
                if not candidate.strip() or candidate.lstrip().startswith("#"):
                    cursor += 1
                    continue
                
                indent = len(candidate) - len(candidate.lstrip())
                if indent <= base_indent:
                    break
                
                stripped = candidate.strip()
                if stripped in self.SILENT_HANDLERS:
                    findings.append({
                        "file": file.path,
                        "line": cursor + 1,
                        "severity": "medium",
                        "type": "silent_exception",
                        "snippet": stripped,
                    })
                break
        
        return findings
    
    def _analyze_js_ts(self, file: FileInfo) -> List[Dict[str, Any]]:
        """Analyze JS/TS file for empty catch blocks."""
        findings = []
        
        # Match empty catch blocks (allowing comments inside)
        pattern = r"catch\s*\([^\)]*\)\s*\{\s*(?:\/\*.*?\*\/\s*|\/\/.*?\n\s*)*\}"
        
        for match in re.finditer(pattern, file.content, re.IGNORECASE | re.DOTALL):
            line = _line_number_from_index(file.content, match.start())
            findings.append({
                "file": file.path,
                "line": str(line),
                "severity": "medium",
                "type": "silent_exception",
                "snippet": "empty catch block",
            })
        
        return findings


def identify_silent_failures_v2(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Find silent exception handlers using BaseTask pattern.
    
    This is a wrapper function maintaining backward compatibility.
    """
    task = SilentFailuresTask(
        storage_path=storage_path,
        source_path=source_path,
        progress_callback=progress_callback,
    )
    return task.run()


# =============================================================================
# Test Mapping Task
# =============================================================================

# Common test file naming patterns
TEST_PATTERNS = [
    r"^test_(.+)$",      # test_module.py
    r"^(.+)_test$",      # module_test.py
    r"^tests?$",         # test.py or tests.py
    r"^(.+)\.test$",     # module.test.js
    r"^(.+)\.spec$",     # module.spec.ts
]

# Files to skip when checking for test coverage
SKIP_STEMS = {"__init__", "conftest", "__main__", "setup", "config", "settings"}


class TestMappingTask(BaseTask):
    """
    Map test files to source files based on naming conventions.
    
    This is NOT actual code coverage - it identifies which source files
    have corresponding test files based on naming patterns.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        source_path: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ):
        super().__init__(
            storage_path=storage_path,
            source_path=source_path,
            progress_callback=progress_callback,
            extensions=CODE_EXTENSIONS,
            load_content=False,  # Don't need content for mapping
        )
    
    def analyze(self, files: List[FileInfo]) -> Dict[str, Any]:
        """Map test files to source files."""
        source_files: List[str] = []
        test_files: List[str] = []
        
        # Separate source and test files
        for file in files:
            if self._is_test_file(file.path):
                test_files.append(file.path)
            else:
                source_files.append(file.path)
        
        self._report_progress("analyze", f"Mapping {len(test_files)} tests to {len(source_files)} sources")
        
        # Build mapping
        mapping: Dict[str, List[str]] = defaultdict(list)
        source_stems = {Path(src).stem.lower(): src for src in source_files}
        
        for test_path in test_files:
            subject = self._extract_test_subject(test_path)
            
            # Try direct match first
            if subject and subject in source_stems:
                mapping[source_stems[subject]].append(test_path)
                continue
            
            # Fallback: check if test filename contains source stem
            test_stem = Path(test_path).stem.lower()
            for src_stem, src_path in source_stems.items():
                if len(src_stem) < 4:  # Skip short generic names
                    continue
                if src_stem in test_stem and src_path not in mapping:
                    mapping[src_path].append(test_path)
        
        # Find untested files
        testable_sources = [
            src for src in source_files
            if Path(src).stem.lower() not in SKIP_STEMS
        ]
        
        untested = [src for src in testable_sources if src not in mapping]
        tested_ratio = 1 - calculate_ratio(len(untested), len(testable_sources))
        
        return {
            "total_sources": len(source_files),
            "testable_sources": len(testable_sources),
            "total_tests": len(test_files),
            "tested_ratio": round(tested_ratio, 2),
            "untested": untested[:50],
            "mapping": {k: v[:5] for k, v in list(mapping.items())[:30]},
            "note": "This analyzes test file presence, not actual code coverage.",
        }
    
    @staticmethod
    def _is_test_file(path: str) -> bool:
        """Check if a file is a test file."""
        path_lower = path.lower().replace("\\", "/")
        name = Path(path).name.lower()
        
        # Check directory patterns
        if "/tests/" in path_lower or "/test/" in path_lower or "/__tests__/" in path_lower:
            return True
        
        # Check filename patterns
        if name.startswith("test_") or name.endswith("_test.py"):
            return True
        if ".test." in name or ".spec." in name:
            return True
        if name in {"conftest.py", "fixtures.py"}:
            return True
        
        return False
    
    @staticmethod
    def _extract_test_subject(test_filename: str) -> Optional[str]:
        """Extract the subject module name from a test filename."""
        stem = Path(test_filename).stem
        
        for pattern in TEST_PATTERNS:
            match = re.match(pattern, stem, re.IGNORECASE)
            if match and match.groups():
                return match.group(1).lower()
        return None


def test_mapping_analysis_v2(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Analyze test file mapping using BaseTask pattern.
    
    This is a wrapper function maintaining backward compatibility.
    """
    task = TestMappingTask(
        storage_path=storage_path,
        source_path=source_path,
        progress_callback=progress_callback,
    )
    return task.run()


# =============================================================================
# Health Score Task
# =============================================================================

class HealthScoreTask(BaseTask):
    """
    Calculate a health score for the codebase.
    
    Scores are based on:
    - Test presence and ratio
    - Documentation presence
    - File size distribution
    - Code complexity indicators
    """
    
    # Weights for health score calculation
    WEIGHTS = {
        "test_presence": 25,
        "doc_presence": 15,
        "file_size": 20,
        "complexity": 20,
        "structure": 20,
    }
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        source_path: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ):
        super().__init__(
            storage_path=storage_path,
            source_path=source_path,
            progress_callback=progress_callback,
            extensions=None,  # All extensions
            load_content=True,
        )
    
    def analyze(self, files: List[FileInfo]) -> Dict[str, Any]:
        """Calculate health score."""
        self._report_progress("analyze", "Calculating health metrics")
        
        code_files = [f for f in files if f.extension in CODE_EXTENSIONS]
        test_files = [f for f in code_files if "test" in Path(f.path).name.lower()]
        doc_files = [f for f in files if f.extension in DOCUMENTATION_EXTENSIONS]
        
        total = len(code_files)
        
        metrics = {
            "test_presence": self._score_test_presence(test_files, code_files),
            "doc_presence": self._score_doc_presence(doc_files, total),
            "file_size": self._score_file_sizes(code_files),
            "complexity": self._score_complexity(code_files),
            "structure": self._score_structure(files),
        }
        
        # Calculate weighted score
        total_score = sum(
            metrics[key] * (self.WEIGHTS[key] / 100)
            for key in self.WEIGHTS
        )
        
        grade = self._calculate_grade(total_score)
        
        return {
            "score": round(total_score, 1),
            "grade": grade,
            "metrics": {k: round(v, 1) for k, v in metrics.items()},
            "weights": self.WEIGHTS,
            "file_counts": {
                "total": len(files),
                "code": len(code_files),
                "tests": len(test_files),
                "docs": len(doc_files),
            },
        }
    
    def _score_test_presence(self, test_files: List[FileInfo], code_files: List[FileInfo]) -> float:
        """Score based on test file ratio."""
        if not code_files:
            return 0.0
        ratio = len(test_files) / len(code_files)
        # Ideal ratio around 0.15-0.25
        if ratio >= 0.20:
            return 100.0
        elif ratio >= 0.10:
            return 75.0 + (ratio - 0.10) * 250  # Scale 75-100
        elif ratio >= 0.05:
            return 50.0 + (ratio - 0.05) * 500  # Scale 50-75
        else:
            return ratio * 1000  # Scale 0-50
    
    def _score_doc_presence(self, doc_files: List[FileInfo], total: int) -> float:
        """Score based on documentation presence."""
        if not doc_files:
            return 0.0
        # Having any docs is good, more is better up to a point
        doc_count = len(doc_files)
        if doc_count >= 5:
            return 100.0
        elif doc_count >= 3:
            return 80.0
        elif doc_count >= 1:
            return 60.0
        return 0.0
    
    def _score_file_sizes(self, files: List[FileInfo]) -> float:
        """Score based on file size distribution."""
        if not files:
            return 100.0
        
        large_files = [f for f in files if f.line_count > 500]
        huge_files = [f for f in files if f.line_count > 1000]
        
        large_ratio = len(large_files) / len(files)
        huge_ratio = len(huge_files) / len(files)
        
        score = 100.0
        score -= large_ratio * 30  # Penalty for large files
        score -= huge_ratio * 50  # Extra penalty for huge files
        
        return max(0.0, score)
    
    def _score_complexity(self, files: List[FileInfo]) -> float:
        """Score based on code complexity indicators."""
        if not files:
            return 100.0
        
        high_complexity = 0
        for f in files:
            if not f.content:
                continue
            
            # Count functions/methods
            func_count = (
                len(re.findall(r"\bdef\s+\w+", f.content)) +
                len(re.findall(r"\bfunction\s+\w+", f.content))
            )
            
            # High complexity if many functions in one file
            if func_count > 30:
                high_complexity += 1
        
        ratio = high_complexity / len(files)
        return max(0.0, 100.0 - ratio * 100)
    
    def _score_structure(self, files: List[FileInfo]) -> float:
        """Score based on project structure."""
        paths = [f.path for f in files]
        
        # Check for common good structure indicators
        has_tests_dir = any("/tests/" in p.replace("\\", "/").lower() for p in paths)
        has_src_dir = any("/src/" in p.replace("\\", "/").lower() for p in paths)
        has_docs_dir = any("/docs/" in p.replace("\\", "/").lower() for p in paths)
        
        score = 40.0  # Base score
        if has_tests_dir:
            score += 20
        if has_src_dir:
            score += 20
        if has_docs_dir:
            score += 20
        
        return min(100.0, score)
    
    @staticmethod
    def _calculate_grade(score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"


def health_score_v2(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Calculate codebase health score using BaseTask pattern.
    
    This is a wrapper function maintaining backward compatibility.
    """
    task = HealthScoreTask(
        storage_path=storage_path,
        source_path=source_path,
        progress_callback=progress_callback,
    )
    return task.run()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Task classes
    "AntiPatternDetectionTask",
    "SilentFailuresTask",
    "TestMappingTask",
    "HealthScoreTask",
    # Wrapper functions
    "detect_anti_patterns_v2",
    "identify_silent_failures_v2",
    "test_mapping_analysis_v2",
    "health_score_v2",
]
