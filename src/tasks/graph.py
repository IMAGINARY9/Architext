"""Dependency graph utilities."""
from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from src.tasks.shared import (
    _extract_imports,
    _module_key,
    _module_name,
    _read_file_text,
    _resolve_python_relative_import,
    _resolve_relative_import,
)


def _build_import_graph(files: List[str]) -> Dict[str, List[str]]:
    graph: Dict[str, List[str]] = defaultdict(list)
    if not files:
        return graph

    root = Path(os.path.commonpath(files)).resolve()
    file_lookup = {_module_key(path): path for path in files}
    module_lookup: Dict[str, str] = {}
    for path in files:
        try:
            module_lookup[_module_name(path, root)] = _module_key(path)
        except Exception:
            continue

    for path in files:
        content = _read_file_text(path)
        if not content:
            continue
        module = _module_key(path)
        suffix = Path(path).suffix
        current_module = None
        if suffix == ".py":
            try:
                current_module = _module_name(path, root)
            except Exception:
                current_module = None

        for imported in _extract_imports(path, content):
            if suffix == ".py" and imported.startswith("."):
                resolved = _resolve_python_relative_import(current_module, imported)
                if resolved and resolved in module_lookup:
                    graph[module].append(module_lookup[resolved])
                continue

            if imported.startswith("."):
                resolved = _resolve_relative_import(path, imported)
                if resolved in file_lookup:
                    graph[module].append(resolved)
                continue

            if imported in module_lookup:
                graph[module].append(module_lookup[imported])
                continue

            for key in file_lookup:
                if key.endswith(imported.replace(".", "/")):
                    graph[module].append(key)
    return graph


def _find_cycles(
    graph: Dict[str, List[str]],
    limit: int = 10,
    time_limit: float = 1.0,
    max_depth: int = 12,
) -> List[List[str]]:
    cycles: List[List[str]] = []
    path_stack: List[str] = []
    visited: Dict[str, str] = {}
    start_time = time.time()

    def dfs(node: str):
        if time.time() - start_time > time_limit:
            return
        if len(path_stack) > max_depth:
            return
        if len(cycles) >= limit:
            return
        visited[node] = "visiting"
        path_stack.append(node)
        for neighbor in graph.get(node, []):
            state = visited.get(neighbor)
            if state == "visiting":
                idx = path_stack.index(neighbor)
                cycles.append(path_stack[idx:] + [neighbor])
            elif state != "visited":
                dfs(neighbor)
        path_stack.pop()
        visited[node] = "visited"

    for node in graph:
        if visited.get(node) is None:
            dfs(node)
        if len(cycles) >= limit or time.time() - start_time > time_limit:
            break

    return cycles
