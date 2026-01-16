"""Duplication detection tasks."""
from __future__ import annotations

import ast
import hashlib
import io
import keyword
import tokenize
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tasks.shared import _progress, _read_file_text, collect_file_paths


def _normalize_duplication_line(line: str, suffix: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if suffix == ".py" and stripped.startswith("#"):
        return ""
    if suffix in {".js", ".ts", ".jsx", ".tsx"} and stripped.startswith("//"):
        return ""
    return stripped


def _normalize_python_tokens(segment: str) -> str:
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


def detect_duplicate_blocks_semantic(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    min_tokens: int = 40,
    max_findings: int = 50,
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)

    fingerprints: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    scanned_files = 0

    _progress(progress_callback, {"stage": "analyze", "message": "Scanning for semantic duplication"})
    for path in files:
        if Path(path).suffix.lower() != ".py":
            continue
        content = _read_file_text(path)
        if not content:
            continue
        try:
            tree = ast.parse(content)
        except Exception:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            segment = ast.get_source_segment(content, node) or ""
            normalized = _normalize_python_tokens(segment)
            if not normalized:
                continue
            token_count = len(normalized.split())
            if token_count < min_tokens:
                continue
            digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
            fingerprints[digest].append(
                {
                    "file": path,
                    "name": getattr(node, "name", "<anonymous>"),
                    "line": getattr(node, "lineno", 0),
                    "tokens": token_count,
                }
            )
        scanned_files += 1

    findings: List[Dict[str, Any]] = []
    for digest, occ in fingerprints.items():
        if len(occ) < 2:
            continue
        findings.append(
            {
                "fingerprint": digest,
                "occurrence_count": len(occ),
                "occurrences": occ[:10],
            }
        )

    findings.sort(key=lambda item: item.get("occurrence_count", 0), reverse=True)

    return {
        "count": len(findings),
        "scanned_files": scanned_files,
        "min_tokens": min_tokens,
        "findings": findings[:max_findings],
    }


def detect_duplicate_blocks(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    min_lines: int = 8,
    max_findings: int = 50,
    max_windows_per_file: int = 6000,
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)

    duplicates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    scanned_files = 0

    _progress(progress_callback, {"stage": "analyze", "message": "Scanning for duplicate blocks"})
    for path in files:
        suffix = Path(path).suffix.lower()
        if suffix not in {".py", ".js", ".ts", ".jsx", ".tsx"}:
            continue
        content = _read_file_text(path)
        if not content:
            continue
        raw_lines = content.splitlines()
        if len(raw_lines) < min_lines:
            continue

        normalized = [_normalize_duplication_line(line, suffix) for line in raw_lines]
        window_limit = min(max_windows_per_file, max(len(raw_lines) - min_lines + 1, 0))
        windows_seen = 0
        for idx in range(0, len(raw_lines) - min_lines + 1):
            if windows_seen >= window_limit:
                break
            window = normalized[idx : idx + min_lines]
            if not any(window):
                continue
            key = "\n".join(window)
            if len(key) < 20:
                continue
            duplicates[key].append(
                {
                    "file": path,
                    "start_line": idx + 1,
                    "end_line": idx + min_lines,
                }
            )
            windows_seen += 1
        scanned_files += 1

    findings: List[Dict[str, Any]] = []
    for key, occ in duplicates.items():
        if len(occ) < 2:
            continue
        findings.append(
            {
                "line_count": min_lines,
                "occurrences": occ[:10],
                "occurrence_count": len(occ),
                "snippet": "\n".join(key.splitlines()[:min_lines])[:500],
            }
        )

    findings.sort(key=lambda item: item.get("occurrence_count", 0), reverse=True)

    return {
        "count": len(findings),
        "scanned_files": scanned_files,
        "min_lines": min_lines,
        "findings": findings[:max_findings],
    }
