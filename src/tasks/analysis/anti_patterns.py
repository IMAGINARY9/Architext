"""Anti-pattern detection task.

Detects code smells like god objects, large files, and mixed responsibilities.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.tasks.core.base import (
    BaseTask,
    FileInfo,
    CODE_EXTENSIONS,
    DOCUMENTATION_EXTENSIONS,
    calculate_ratio,
)
from src.tasks.shared import (
    _classify_import_clusters,
    _extract_imports,
)


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


__all__ = ["AntiPatternDetectionTask", "detect_anti_patterns_v2"]
