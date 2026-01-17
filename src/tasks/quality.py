"""Quality, coverage, and logic gap analysis tasks."""
from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.tasks.anti_patterns import detect_anti_patterns
from src.tasks.health import health_score
from src.tasks.shared import (
    DEFAULT_EXTENSIONS,
    FRAMEWORK_SETTINGS_IGNORES,
    _line_number_from_index,
    _progress,
    _read_file_text,
    collect_file_paths,
)


def refactoring_recommendations(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    anti_patterns = detect_anti_patterns(storage_path, source_path)
    score = health_score(storage_path, source_path)

    recommendations = []
    for issue in anti_patterns.get("issues", []):
        if issue["type"] == "god_object":
            recommendations.append(
                {
                    "title": "Split large modules",
                    "file": issue.get("file"),
                    "effort": "medium",
                    "benefit": "high",
                    "rationale": issue.get("details"),
                }
            )
        if issue["type"] == "circular_dependency":
            recommendations.append(
                {
                    "title": "Break circular dependencies",
                    "effort": "high",
                    "benefit": "high",
                    "rationale": issue.get("details"),
                }
            )

    if score.get("details", {}).get("documentation", 100) < 40:
        recommendations.append(
            {
                "title": "Improve documentation coverage",
                "effort": "low",
                "benefit": "medium",
                "rationale": "Documentation coverage below 40%",
            }
        )

    if score.get("details", {}).get("testing", 100) < 30:
        recommendations.append(
            {
                "title": "Increase test coverage",
                "effort": "medium",
                "benefit": "high",
                "rationale": "Test coverage below 30%",
            }
        )

    return {"recommendations": recommendations, "health": score}


def test_coverage_analysis(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)
    source_files = [path for path in files if Path(path).suffix in DEFAULT_EXTENSIONS]
    test_files = [path for path in files if "test" in Path(path).name.lower()]

    mapping: Dict[str, List[str]] = defaultdict(list)
    for src in source_files:
        stem = Path(src).stem
        for test in test_files:
            if stem in Path(test).stem:
                mapping[src].append(test)

    uncovered = [src for src in source_files if src not in mapping]
    coverage_ratio = 1 - (len(uncovered) / max(len(source_files), 1))

    return {
        "total_sources": len(source_files),
        "total_tests": len(test_files),
        "coverage_ratio": round(coverage_ratio, 2),
        "uncovered": uncovered[:50],
        "mapping": {k: v[:5] for k, v in mapping.items()},
    }


def logic_gap_analysis(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)

    config_file = None
    settings_fields: List[Tuple[str, int]] = []
    is_pydantic_settings = False
    for path in files:
        if Path(path).name.lower() != "config.py":
            continue
        content = _read_file_text(path)
        if "ArchitextSettings" not in content:
            continue
        try:
            tree = ast.parse(content)
        except Exception:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "ArchitextSettings":
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == "BaseSettings":
                        is_pydantic_settings = True
                    elif isinstance(base, ast.Attribute) and base.attr == "BaseSettings":
                        is_pydantic_settings = True
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        settings_fields.append((item.target.id, item.lineno))
                    elif isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                settings_fields.append((target.id, item.lineno))
        if settings_fields:
            config_file = path
            break

    if not settings_fields:
        return {
            "note": "No ArchitextSettings fields found",
            "config_file": config_file,
            "unused_settings": [],
        }

    used = set()
    candidate_files = [
        path for path in files if Path(path).suffix == ".py" and path != config_file
    ]
    for path in candidate_files:
        content = _read_file_text(path)
        if not content:
            continue
        for field, _lineno in settings_fields:
            if field in used:
                continue
            if re.search(rf"\b{re.escape(field)}\b", content):
                used.add(field)

    ignored_settings = set()
    if is_pydantic_settings:
        ignored_settings.update(FRAMEWORK_SETTINGS_IGNORES.get("pydantic", set()))

    unused = [
        {"name": field, "defined_at": lineno}
        for field, lineno in settings_fields
        if field not in used and field not in ignored_settings
    ]

    return {
        "config_file": config_file,
        "settings_defined": len(settings_fields),
        "settings_used": len(used),
        "unused_settings": unused,
        "ignored_settings": sorted(ignored_settings),
    }


def identify_silent_failures(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
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
