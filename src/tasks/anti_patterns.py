"""Anti-pattern detection tasks."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tasks.graph import _build_import_graph, _find_cycles
from src.tasks.shared import (
    _classify_import_clusters,
    _extract_imports,
    _progress,
    _read_file_text,
    collect_file_paths,
)


def detect_anti_patterns(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)

    total_files = len(files)
    directories = Counter(Path(path).parent for path in files)
    avg_files_per_dir = total_files / max(len(directories), 1)

    issues = []
    large_files = []
    for path in files:
        text = _read_file_text(path)
        if not text:
            continue
        lines = text.splitlines()
        if len(lines) > 800:
            large_files.append((path, len(lines)))

        function_count = len(re.findall(r"\bdef\s+\w+\b", text)) + len(
            re.findall(r"\bfunction\s+\w+\b", text)
        )
        if function_count > 40:
            issues.append(
                {
                    "type": "god_object",
                    "file": path,
                    "severity": "high",
                    "details": f"High function count: {function_count}",
                }
            )

        imports = _extract_imports(path, text)
        clusters = _classify_import_clusters(imports)
        if len(clusters) >= 2 and (len(lines) > 200 or function_count > 20):
            issues.append(
                {
                    "type": "mixed_responsibilities",
                    "file": path,
                    "severity": "medium",
                    "details": f"Multiple import clusters detected: {', '.join(clusters)}",
                }
            )

    if large_files:
        for path, count in large_files[:10]:
            issues.append(
                {
                    "type": "large_file",
                    "file": path,
                    "severity": "medium",
                    "details": f"File has {count} lines",
                }
            )

    # Documentation and test hygiene signals
    doc_files = [path for path in files if Path(path).suffix in {".md", ".rst"}]
    test_files = [path for path in files if "test" in Path(path).name.lower()]
    doc_ratio = len(doc_files) / max(total_files, 1)
    test_ratio = len(test_files) / max(total_files, 1)

    if not test_files:
        issues.append(
            {
                "type": "missing_tests",
                "severity": "medium",
                "details": "No test files detected (filenames containing 'test')",
            }
        )
    elif test_ratio < 0.03:
        issues.append(
            {
                "type": "low_test_presence",
                "severity": "low",
                "details": f"Test files ratio is low: {round(test_ratio, 3)}",
            }
        )

    if not doc_files:
        issues.append(
            {
                "type": "missing_docs",
                "severity": "low",
                "details": "No documentation files detected (.md/.rst)",
            }
        )
    elif doc_ratio < 0.02:
        issues.append(
            {
                "type": "low_doc_presence",
                "severity": "low",
                "details": f"Documentation files ratio is low: {round(doc_ratio, 3)}",
            }
        )

    # Flat structure signal (lots of files per directory)
    if avg_files_per_dir > 30:
        issues.append(
            {
                "type": "flat_structure",
                "severity": "medium",
                "details": f"Average files per directory is high: {round(avg_files_per_dir, 2)}",
            }
        )

    # Excessive single-directory concentration
    if directories:
        max_dir, max_count = max(directories.items(), key=lambda item: item[1])
        concentration = max_count / max(total_files, 1)
        if concentration > 0.6 and total_files >= 20:
            issues.append(
                {
                    "type": "single_directory_concentration",
                    "severity": "medium",
                    "details": f"{max_count} files ({round(concentration, 2)}) in {max_dir}",
                }
            )

    # Duplicate file stems (excluding extensions)
    stems = [Path(path).stem.lower() for path in files]
    stem_counts = Counter(stems)
    duplicate_stems = [stem for stem, count in stem_counts.items() if count >= 3]
    if duplicate_stems:
        sample = duplicate_stems[:5]
        issues.append(
            {
                "type": "duplicate_file_stems",
                "severity": "low",
                "details": f"Duplicate stems found (>=3 occurrences): {', '.join(sample)}",
            }
        )

    # Missing CI configuration (GitHub Actions / GitLab / CircleCI / Azure Pipelines)
    ci_files = [
        path
        for path in files
        if any(
            token in str(path).replace("\\", "/").lower()
            for token in [
                ".github/workflows/",
                ".gitlab-ci",
                "circleci/config.yml",
                "azure-pipelines.yml",
                "azure-pipelines.yaml",
                "appveyor.yml",
            ]
        )
    ]
    if not ci_files and total_files >= 10:
        issues.append(
            {
                "type": "missing_ci_config",
                "severity": "low",
                "details": "No CI configuration detected",
            }
        )

    # Missing license file
    has_license = any(
        Path(path).name.lower() in {"license", "license.md", "license.txt"} for path in files
    )
    if not has_license and total_files >= 10:
        issues.append(
            {
                "type": "missing_license",
                "severity": "low",
                "details": "No LICENSE file detected",
            }
        )

    # Missing formatting config (Python/JS/TS/Prettier/EditorConfig)
    formatting_files = {
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
    has_formatting = any(Path(path).name.lower() in formatting_files for path in files)
    if not has_formatting and total_files >= 10:
        issues.append(
            {
                "type": "missing_formatting_config",
                "severity": "low",
                "details": "No formatting configuration detected (.editorconfig/prettier/pyproject/etc)",
            }
        )

    _progress(progress_callback, {"stage": "analyze", "message": "Building dependency graph"})
    graph = _build_import_graph(files)

    coupling = sum(len(deps) for deps in graph.values()) / max(len(graph), 1)
    if coupling > 12:
        issues.append(
            {
                "type": "high_coupling",
                "severity": "medium",
                "details": f"Average dependencies per module is high: {round(coupling, 2)}",
            }
        )

    cycles = _find_cycles(graph)
    for cycle in cycles:
        issues.append(
            {
                "type": "circular_dependency",
                "severity": "high",
                "details": " -> ".join(cycle),
            }
        )

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
