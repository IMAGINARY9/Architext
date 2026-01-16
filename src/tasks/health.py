"""Health score and documentation coverage tasks."""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tasks.graph import _build_import_graph
from src.tasks.shared import _progress, _read_file_text, collect_file_paths


def _count_python_docstrings(files: List[str]) -> int:
    count = 0
    for path in files:
        if Path(path).suffix != ".py":
            continue
        content = _read_file_text(path)
        if not content:
            continue
        try:
            tree = ast.parse(content)
        except Exception:
            continue

        if ast.get_docstring(tree):
            count += 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if ast.get_docstring(node):
                    count += 1
    return count


def health_score(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)
    if not files:
        return {"score": 0, "details": {}}

    total_files = len(files)
    directories = Counter(Path(path).parent for path in files)
    avg_files_per_dir = total_files / max(len(directories), 1)

    graph = _build_import_graph(files)
    coupling = sum(len(deps) for deps in graph.values()) / max(len(graph), 1)

    doc_files = [path for path in files if Path(path).suffix in {".md", ".rst"}]
    doc_coverage = min(len(doc_files) / max(total_files, 1), 1.0)
    docstring_count = _count_python_docstrings(files)
    py_files = [path for path in files if Path(path).suffix == ".py"]
    docstring_coverage = min(docstring_count / max(len(py_files), 1), 1.0)

    test_files = [path for path in files if "test" in Path(path).name.lower()]
    test_coverage = min(len(test_files) / max(total_files, 1), 1.0)

    modularity_score = max(0, 100 - avg_files_per_dir * 2)
    coupling_score = max(0, 100 - coupling * 5)
    documentation_score = (doc_coverage * 0.6 + docstring_coverage * 0.4) * 100
    testing_score = test_coverage * 100

    overall = (modularity_score + coupling_score + documentation_score + testing_score) / 4

    return {
        "score": round(overall, 2),
        "details": {
            "modularity": round(modularity_score, 2),
            "coupling": round(coupling_score, 2),
            "documentation": round(documentation_score, 2),
            "testing": round(testing_score, 2),
            "avg_files_per_dir": round(avg_files_per_dir, 2),
            "avg_dependencies": round(coupling, 2),
            "doc_files": len(doc_files),
            "docstrings": docstring_count,
        },
    }
