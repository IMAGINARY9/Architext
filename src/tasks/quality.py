"""Quality analysis tasks: test mapping and silent failure detection."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tasks.shared import (
    DEFAULT_EXTENSIONS,
    _line_number_from_index,
    _progress,
    _read_file_text,
    collect_file_paths,
)


# Common test file naming patterns
TEST_PATTERNS = [
    r"^test_(.+)$",      # test_module.py
    r"^(.+)_test$",      # module_test.py
    r"^tests?$",         # test.py or tests.py (generic)
    r"^(.+)\.test$",     # module.test.js
    r"^(.+)\.spec$",     # module.spec.ts
]


def _extract_test_subject(test_filename: str) -> Optional[str]:
    """Extract the subject module name from a test filename."""
    stem = Path(test_filename).stem
    
    for pattern in TEST_PATTERNS:
        match = re.match(pattern, stem, re.IGNORECASE)
        if match and match.groups():
            return match.group(1).lower()
    return None


def _is_test_file(path: str) -> bool:
    """Check if a file is a test file based on path and name."""
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


def test_mapping_analysis(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Analyze test file presence and map tests to source files.
    
    This is NOT actual code coverage - it maps test files to source files
    based on naming conventions to identify potentially untested modules.
    """
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)
    
    # Separate source and test files
    source_files: List[str] = []
    test_files: List[str] = []
    
    for path in files:
        suffix = Path(path).suffix.lower()
        if suffix not in DEFAULT_EXTENSIONS:
            continue
        if _is_test_file(path):
            test_files.append(path)
        else:
            source_files.append(path)

    # Build mapping from source files to their tests
    mapping: Dict[str, List[str]] = defaultdict(list)
    source_stems = {Path(src).stem.lower(): src for src in source_files}
    
    _progress(progress_callback, {"stage": "analyze", "message": "Mapping tests to sources"})
    
    for test_path in test_files:
        subject = _extract_test_subject(test_path)
        if subject and subject in source_stems:
            mapping[source_stems[subject]].append(test_path)
            continue
        
        # Fallback: check if test filename contains source stem
        test_stem = Path(test_path).stem.lower()
        for src_stem, src_path in source_stems.items():
            # Avoid matching generic names like "test", "utils", "helpers"
            if len(src_stem) < 4:
                continue
            if src_stem in test_stem and src_path not in mapping:
                mapping[src_path].append(test_path)

    # Find untested files (excluding __init__.py, conftest.py, etc.)
    skip_names = {"__init__", "conftest", "__main__", "setup"}
    testable_sources = [
        src for src in source_files 
        if Path(src).stem.lower() not in skip_names
    ]
    
    untested = [src for src in testable_sources if src not in mapping]
    tested_ratio = 1 - (len(untested) / max(len(testable_sources), 1))

    return {
        "total_sources": len(source_files),
        "testable_sources": len(testable_sources),
        "total_tests": len(test_files),
        "tested_ratio": round(tested_ratio, 2),
        "untested": untested[:50],
        "mapping": {k: v[:5] for k, v in list(mapping.items())[:30]},
        "note": "This analyzes test file presence, not actual code coverage.",
    }


def identify_silent_failures(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Find exception handlers that silently swallow errors.
    
    Detects patterns like:
    - except: pass
    - except Exception: continue
    - catch(e) {} (empty JS/TS catch blocks)
    """
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)
    findings: List[Dict[str, Any]] = []

    for path in files:
        suffix = Path(path).suffix.lower()
        content = _read_file_text(path)
        if not content:
            continue

        if suffix == ".py":
            lines = content.splitlines()
            for idx, line in enumerate(lines):
                if not re.match(r"\s*except\b", line):
                    continue
                if not line.rstrip().endswith(":"):
                    continue
                base_indent = len(line) - len(line.lstrip())
                cursor = idx + 1
                while cursor < len(lines):
                    candidate = lines[cursor]
                    if not candidate.strip() or candidate.lstrip().startswith("#"):
                        cursor += 1
                        continue
                    indent = len(candidate) - len(candidate.lstrip())
                    if indent <= base_indent:
                        break
                    stripped = candidate.strip()
                    if stripped in {"pass", "continue", "return", "..."}:
                        findings.append(
                            {
                                "file": path,
                                "line": cursor + 1,
                                "severity": "medium",
                                "type": "silent_exception",
                                "snippet": stripped,
                            }
                        )
                    break
                    
        elif suffix in {".js", ".ts", ".jsx", ".tsx"}:
            for match in re.finditer(
                r"catch\s*\([^\)]*\)\s*\{\s*(?:\/\*.*?\*\/\s*|\/\/.*?\n\s*)*\}",
                content,
                re.IGNORECASE | re.DOTALL,
            ):
                line = str(_line_number_from_index(content, match.start()))
                findings.append(
                    {
                        "file": path,
                        "line": line,
                        "severity": "medium",
                        "type": "silent_exception",
                        "snippet": "empty catch block",
                    }
                )

    return {
        "findings": findings,
        "count": len(findings),
    }
