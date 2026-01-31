"""Codebase health scoring task.

Calculates a health score based on tests, documentation, file sizes, and structure.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.tasks.core.base import (
    BaseTask,
    FileInfo,
    CODE_EXTENSIONS,
    DOCUMENTATION_EXTENSIONS,
)


# Default weights for health score calculation
DEFAULT_WEIGHTS = {
    "test_presence": 25,
    "doc_presence": 15,
    "file_size": 20,
    "complexity": 20,
    "structure": 20,
}


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
    WEIGHTS = DEFAULT_WEIGHTS
    
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
    """Calculate codebase health score using BaseTask pattern."""
    task = HealthScoreTask(
        storage_path=storage_path,
        source_path=source_path,
        progress_callback=progress_callback,
    )
    return task.run()


__all__ = [
    "HealthScoreTask",
    "health_score_v2",
    "DEFAULT_WEIGHTS",
]
