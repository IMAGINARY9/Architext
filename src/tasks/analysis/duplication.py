"""Duplication detection tasks.

Includes exact duplicate block detection and semantic similarity detection.
"""
from __future__ import annotations

import ast
import hashlib
import io
import keyword
import tokenize
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from src.tasks.core.base import BaseTask, FileInfo, CODE_EXTENSIONS, PYTHON_EXTENSIONS


# =============================================================================
# Exact Duplicate Blocks Detection
# =============================================================================

class DuplicateBlocksTask(BaseTask):
    """
    Detect exact duplicate code blocks.
    
    Uses sliding window hash comparison to find duplicated code.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        source_path: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        min_lines: int = 8,
        max_findings: int = 50,
        max_windows_per_file: int = 6000,
    ):
        super().__init__(
            storage_path=storage_path,
            source_path=source_path,
            progress_callback=progress_callback,
            extensions=CODE_EXTENSIONS,
            load_content=True,
        )
        self.min_lines = min_lines
        self.max_findings = max_findings
        self.max_windows_per_file = max_windows_per_file
    
    def analyze(self, files: List[FileInfo]) -> Dict[str, Any]:
        """Find duplicate code blocks."""
        self._report_progress("analyze", "Scanning for duplicate blocks")
        
        duplicates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        scanned = 0
        
        for f in files:
            if f.extension not in {".py", ".js", ".ts", ".jsx", ".tsx"}:
                continue
            if not f.content:
                continue
            
            raw_lines = f.content.splitlines()
            if len(raw_lines) < self.min_lines:
                continue
            
            normalized = [self._normalize_line(line, f.extension) for line in raw_lines]
            window_limit = min(self.max_windows_per_file, max(len(raw_lines) - self.min_lines + 1, 0))
            
            for idx in range(min(window_limit, len(raw_lines) - self.min_lines + 1)):
                window = normalized[idx:idx + self.min_lines]
                if not all(window):
                    continue
                block = "\n".join(window)
                digest = hashlib.sha1(block.encode("utf-8")).hexdigest()
                duplicates[digest].append({
                    "file": f.path,
                    "start_line": idx + 1,
                    "end_line": idx + self.min_lines,
                })
            
            scanned += 1
        
        # Filter to actual duplicates (2+ occurrences)
        findings = [
            {
                "fingerprint": digest,
                "occurrence_count": len(occ),
                "occurrences": occ[:10],
            }
            for digest, occ in duplicates.items()
            if len(occ) >= 2
        ]
        findings.sort(key=lambda x: x["occurrence_count"], reverse=True)
        
        return {
            "count": len(findings),
            "scanned_files": scanned,
            "min_lines": self.min_lines,
            "findings": findings[:self.max_findings],
        }
    
    @staticmethod
    def _normalize_line(line: str, suffix: str) -> str:
        """Normalize line for comparison."""
        stripped = line.strip()
        if not stripped:
            return ""
        if suffix == ".py" and stripped.startswith("#"):
            return ""
        if suffix in {".js", ".ts", ".jsx", ".tsx"} and stripped.startswith("//"):
            return ""
        return stripped


def detect_duplicate_blocks_v2(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    min_lines: int = 8,
    max_findings: int = 50,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Detect duplicate blocks using BaseTask pattern."""
    task = DuplicateBlocksTask(
        storage_path=storage_path,
        source_path=source_path,
        progress_callback=progress_callback,
        min_lines=min_lines,
        max_findings=max_findings,
    )
    return task.run()


# Alias
duplicate_blocks_detection_v2 = detect_duplicate_blocks_v2


# =============================================================================
# Semantic Duplication Detection
# =============================================================================

class SemanticDuplicationTask(BaseTask):
    """
    Detect semantically similar functions using token normalization.
    
    Normalizes identifiers and literals to find structurally similar code.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        source_path: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        min_tokens: int = 40,
        max_findings: int = 50,
    ):
        super().__init__(
            storage_path=storage_path,
            source_path=source_path,
            progress_callback=progress_callback,
            extensions=PYTHON_EXTENSIONS,
            load_content=True,
        )
        self.min_tokens = min_tokens
        self.max_findings = max_findings
    
    def analyze(self, files: List[FileInfo]) -> Dict[str, Any]:
        """Find semantically similar functions."""
        self._report_progress("analyze", "Scanning for semantic duplication")
        
        fingerprints: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        scanned = 0
        
        for f in files:
            if not f.ast_tree or not f.content:
                continue
            
            for node in ast.walk(f.ast_tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                
                segment = ast.get_source_segment(f.content, node) or ""
                normalized = self._normalize_tokens(segment)
                if not normalized:
                    continue
                
                token_count = len(normalized.split())
                if token_count < self.min_tokens:
                    continue
                
                digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
                fingerprints[digest].append({
                    "file": f.path,
                    "name": getattr(node, "name", "<anonymous>"),
                    "line": getattr(node, "lineno", 0),
                    "tokens": token_count,
                })
            
            scanned += 1
        
        # Filter to actual duplicates
        findings = [
            {
                "fingerprint": digest,
                "occurrence_count": len(occ),
                "occurrences": occ[:10],
            }
            for digest, occ in fingerprints.items()
            if len(occ) >= 2
        ]
        findings.sort(key=lambda x: x["occurrence_count"], reverse=True)
        
        return {
            "count": len(findings),
            "scanned_files": scanned,
            "min_tokens": self.min_tokens,
            "findings": findings[:self.max_findings],
        }
    
    @staticmethod
    def _normalize_tokens(segment: str) -> str:
        """Normalize Python tokens for semantic comparison."""
        tokens: List[str] = []
        try:
            for tok in tokenize.generate_tokens(io.StringIO(segment).readline):
                if tok.type in {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT}:
                    continue
                if tok.type == tokenize.COMMENT:
                    continue
                if tok.type == tokenize.NAME:
                    if keyword.iskeyword(tok.string):
                        tokens.append(tok.string)
                    else:
                        tokens.append("_id")
                elif tok.type == tokenize.STRING:
                    tokens.append("S")
                elif tok.type == tokenize.NUMBER:
                    tokens.append("0")
                else:
                    tokens.append(tok.string)
        except Exception:
            return ""
        return " ".join(tokens)


def detect_duplicate_blocks_semantic_v2(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    min_tokens: int = 40,
    max_findings: int = 50,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Detect semantic duplication using BaseTask pattern."""
    task = SemanticDuplicationTask(
        storage_path=storage_path,
        source_path=source_path,
        progress_callback=progress_callback,
        min_tokens=min_tokens,
        max_findings=max_findings,
    )
    return task.run()


# Alias
semantic_duplication_detection_v2 = detect_duplicate_blocks_semantic_v2


__all__ = [
    "DuplicateBlocksTask",
    "SemanticDuplicationTask",
    "detect_duplicate_blocks_v2",
    "duplicate_blocks_detection_v2",
    "detect_duplicate_blocks_semantic_v2",
    "semantic_duplication_detection_v2",
]
