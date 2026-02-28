"""Anti-pattern detection tasks."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tasks.graph import _build_import_graph, _find_cycles
from src.tasks.shared import (
    ProgressCallback,
    _classify_import_clusters,
    _extract_imports,
    _progress,
    _read_file_text,
    collect_file_paths,
)

# ---------------------------------------------------------------------------
# Thresholds (module-level so tests can patch if needed)
# ---------------------------------------------------------------------------
LARGE_FILE_THRESHOLD = 800
HIGH_FUNCTION_COUNT = 40
MIXED_RESP_LINE_THRESHOLD = 200
MIXED_RESP_FUNC_THRESHOLD = 20
FLAT_STRUCTURE_THRESHOLD = 30
CONCENTRATION_RATIO = 0.6
CONCENTRATION_MIN_FILES = 20
DUPLICATE_STEM_MIN = 3
LOW_TEST_RATIO = 0.03
LOW_DOC_RATIO = 0.02
HIGH_COUPLING_THRESHOLD = 12
MIN_FILES_FOR_HYGIENE = 10

# CI config path tokens recognised by the detector
_CI_TOKENS = [
    ".github/workflows/",
    ".gitlab-ci",
    "circleci/config.yml",
    "azure-pipelines.yml",
    "azure-pipelines.yaml",
    "appveyor.yml",
]

# Formatting tool config filenames
_FORMATTING_FILES = {
    ".editorconfig",
    ".prettierrc",
    ".prettierrc.json",
    ".prettierrc.yml",
    ".prettierrc.yaml",
    "prettier.config.js",
    "pyproject.toml",
    "setup.cfg",
    "tox.ini",
    "ruff.toml",
}


# ---------------------------------------------------------------------------
# Helper detectors — each returns a list of issue dicts
# ---------------------------------------------------------------------------

def _check_code_smells(files: List[str]) -> tuple[List[Dict[str, Any]], List[tuple]]:
    """Detect god objects, mixed responsibilities, and large files."""
    issues: List[Dict[str, Any]] = []
    large_files: List[tuple] = []

    for path in files:
        text = _read_file_text(path)
        if not text:
            continue
        lines = text.splitlines()
        if len(lines) > LARGE_FILE_THRESHOLD:
            large_files.append((path, len(lines)))

        function_count = len(re.findall(r"\bdef\s+\w+\b", text)) + len(
            re.findall(r"\bfunction\s+\w+\b", text)
        )
        if function_count > HIGH_FUNCTION_COUNT:
            issues.append({
                "type": "god_object",
                "file": path,
                "severity": "high",
                "details": f"High function count: {function_count}",
            })

        imports = _extract_imports(path, text)
        clusters = _classify_import_clusters(imports)
        if len(clusters) >= 2 and (
            len(lines) > MIXED_RESP_LINE_THRESHOLD or function_count > MIXED_RESP_FUNC_THRESHOLD
        ):
            issues.append({
                "type": "mixed_responsibilities",
                "file": path,
                "severity": "medium",
                "details": f"Multiple import clusters detected: {', '.join(clusters)}",
            })

    for path, count in large_files[:10]:
        issues.append({
            "type": "large_file",
            "file": path,
            "severity": "medium",
            "details": f"File has {count} lines",
        })

    return issues, large_files


def _check_test_hygiene(files: List[str], total_files: int) -> List[Dict[str, Any]]:
    """Check for missing or low test presence."""
    issues: List[Dict[str, Any]] = []
    test_files = [p for p in files if "test" in Path(p).name.lower()]
    test_ratio = len(test_files) / max(total_files, 1)

    if not test_files:
        issues.append({
            "type": "missing_tests",
            "severity": "medium",
            "details": "No test files detected (filenames containing 'test')",
        })
    elif test_ratio < LOW_TEST_RATIO:
        issues.append({
            "type": "low_test_presence",
            "severity": "low",
            "details": f"Test files ratio is low: {round(test_ratio, 3)}",
        })
    return issues


def _check_doc_hygiene(files: List[str], total_files: int) -> List[Dict[str, Any]]:
    """Check for missing or low documentation presence."""
    issues: List[Dict[str, Any]] = []
    doc_files = [p for p in files if Path(p).suffix in {".md", ".rst"}]
    doc_ratio = len(doc_files) / max(total_files, 1)

    if not doc_files:
        issues.append({
            "type": "missing_docs",
            "severity": "low",
            "details": "No documentation files detected (.md/.rst)",
        })
    elif doc_ratio < LOW_DOC_RATIO:
        issues.append({
            "type": "low_doc_presence",
            "severity": "low",
            "details": f"Documentation files ratio is low: {round(doc_ratio, 3)}",
        })
    return issues


def _check_structure(
    files: List[str],
    directories: Counter,
    avg_files_per_dir: float,
    total_files: int,
) -> List[Dict[str, Any]]:
    """Check for flat structure, high concentration, and duplicate stems."""
    issues: List[Dict[str, Any]] = []

    if avg_files_per_dir > FLAT_STRUCTURE_THRESHOLD:
        issues.append({
            "type": "flat_structure",
            "severity": "medium",
            "details": f"Average files per directory is high: {round(avg_files_per_dir, 2)}",
        })

    if directories:
        max_dir, max_count = max(directories.items(), key=lambda item: item[1])
        concentration = max_count / max(total_files, 1)
        if concentration > CONCENTRATION_RATIO and total_files >= CONCENTRATION_MIN_FILES:
            issues.append({
                "type": "single_directory_concentration",
                "severity": "medium",
                "details": f"{max_count} files ({round(concentration, 2)}) in {max_dir}",
            })

    stems = [Path(p).stem.lower() for p in files]
    stem_counts = Counter(stems)
    duplicate_stems = [s for s, c in stem_counts.items() if c >= DUPLICATE_STEM_MIN]
    if duplicate_stems:
        issues.append({
            "type": "duplicate_file_stems",
            "severity": "low",
            "details": f"Duplicate stems found (>={DUPLICATE_STEM_MIN} occurrences): "
                       f"{', '.join(duplicate_stems[:5])}",
        })

    return issues


def _check_project_hygiene(files: List[str], total_files: int) -> List[Dict[str, Any]]:
    """Check for missing CI, license, and formatting configuration."""
    issues: List[Dict[str, Any]] = []
    if total_files < MIN_FILES_FOR_HYGIENE:
        return issues

    # CI configuration
    ci_files = [
        p for p in files
        if any(t in str(p).replace("\\", "/").lower() for t in _CI_TOKENS)
    ]
    if not ci_files:
        issues.append({
            "type": "missing_ci_config",
            "severity": "low",
            "details": "No CI configuration detected",
        })

    # License
    if not any(Path(p).name.lower() in {"license", "license.md", "license.txt"} for p in files):
        issues.append({
            "type": "missing_license",
            "severity": "low",
            "details": "No LICENSE file detected",
        })

    # Formatting config
    if not any(Path(p).name.lower() in _FORMATTING_FILES for p in files):
        issues.append({
            "type": "missing_formatting_config",
            "severity": "low",
            "details": "No formatting configuration detected "
                       "(.editorconfig/prettier/pyproject/etc)",
        })

    return issues


def _check_coupling_and_cycles(
    graph: Dict[str, List[str]],
) -> tuple[List[Dict[str, Any]], float]:
    """Detect high coupling and circular dependencies."""
    issues: List[Dict[str, Any]] = []
    coupling = sum(len(deps) for deps in graph.values()) / max(len(graph), 1)

    if coupling > HIGH_COUPLING_THRESHOLD:
        issues.append({
            "type": "high_coupling",
            "severity": "medium",
            "details": f"Average dependencies per module is high: {round(coupling, 2)}",
        })

    for cycle in _find_cycles(graph):
        issues.append({
            "type": "circular_dependency",
            "severity": "high",
            "details": " -> ".join(cycle),
        })

    return issues, coupling


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_anti_patterns(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    """Detect code anti-patterns, structural issues, and project hygiene gaps.

    Scans for god objects, large files, mixed responsibilities, circular
    dependencies, missing tests/docs/CI, duplicated stems, flat structure, and
    other common code smells.
    """
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)

    total_files = len(files)
    directories = Counter(Path(path).parent for path in files)
    avg_files_per_dir = total_files / max(len(directories), 1)

    # --- Per-file code smells ---
    issues: List[Dict[str, Any]] = []
    code_issues, large_files = _check_code_smells(files)
    issues.extend(code_issues)

    # --- Hygiene checks ---
    doc_files = [p for p in files if Path(p).suffix in {".md", ".rst"}]
    test_files = [p for p in files if "test" in Path(p).name.lower()]
    doc_ratio = len(doc_files) / max(total_files, 1)
    test_ratio = len(test_files) / max(total_files, 1)

    issues.extend(_check_test_hygiene(files, total_files))
    issues.extend(_check_doc_hygiene(files, total_files))
    issues.extend(_check_structure(files, directories, avg_files_per_dir, total_files))
    issues.extend(_check_project_hygiene(files, total_files))

    # --- Coupling & cycles ---
    _progress(progress_callback, {"stage": "analyze", "message": "Building dependency graph"})
    graph = _build_import_graph(files)
    coupling_issues, coupling = _check_coupling_and_cycles(graph)
    issues.extend(coupling_issues)

    severity_counts = Counter(issue.get("severity", "unknown") for issue in issues)

    return {
        "issues": issues,
        "counts": Counter(issue["type"] for issue in issues),
        "severity_counts": dict(severity_counts),
        "metrics": {
            "total_files": total_files,
            "doc_files": len(doc_files),
            "test_files": len(test_files),
            "doc_ratio": round(doc_ratio, 3),
            "test_ratio": round(test_ratio, 3),
            "avg_files_per_dir": round(avg_files_per_dir, 2),
            "avg_dependencies": round(coupling, 2),
        },
    }
