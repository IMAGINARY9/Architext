"""Architecture and dependency analysis tasks."""
from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tasks.graph import _build_import_graph
from src.tasks.shared import _progress, _read_file_text, collect_file_paths


def impact_analysis(
    module: str,
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)
    graph = _build_import_graph(files)

    targets = [key for key in graph if module in key]
    if not targets:
        return {"module": module, "affected": [], "note": "Module not found"}

    reverse_graph: Dict[str, List[str]] = defaultdict(list)
    for src, deps in graph.items():
        for dep in deps:
            reverse_graph[dep].append(src)

    affected = set()
    stack = list(targets)
    while stack:
        current = stack.pop()
        for dep in reverse_graph.get(current, []):
            if dep not in affected:
                affected.add(dep)
                stack.append(dep)

    return {
        "module": module,
        "targets": targets,
        "affected": sorted(affected),
        "affected_count": len(affected),
    }


def dependency_graph_export(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    output_format: str = "mermaid",
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)
    graph = _build_import_graph(files)

    edges = []
    for src, deps in graph.items():
        for dep in deps:
            edges.append((src, dep))

    if output_format == "json":
        return {"format": "json", "nodes": list(graph.keys()), "edges": edges}

    if output_format == "graphml":
        lines = ["<graphml>", "<graph edgedefault=\"directed\">"]
        for src, dep in edges:
            lines.append(f"  <edge source=\"{src}\" target=\"{dep}\"/>")
        lines.append("</graph>")
        lines.append("</graphml>")
        return {"format": "graphml", "content": "\n".join(lines)}

    if output_format == "mermaid":
        lines = ["graph TD"]
        for src, dep in edges:
            src_id = src.replace("-", "_")
            dep_id = dep.replace("-", "_")
            lines.append(f"  {src_id}[{Path(src).name}] --> {dep_id}[{Path(dep).name}]")
        return {"format": "mermaid", "content": "\n".join(lines), "edge_count": len(edges)}

    return {"format": "json", "nodes": list(graph.keys()), "edges": edges}


def architecture_pattern_detection(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)

    patterns = []
    files_lower = [str(path).lower() for path in files]

    if any("controllers" in path for path in files_lower) and any("views" in path for path in files_lower):
        patterns.append("MVC")
    if any("services" in path for path in files_lower) and any("repositories" in path for path in files_lower):
        patterns.append("Service-Repository")
    if any("microservice" in path for path in files_lower) or any("docker" in path for path in files_lower):
        patterns.append("Microservices")
    if any("plugins" in path for path in files_lower) or any("extensions" in path for path in files_lower):
        patterns.append("Plugin Architecture")
    if any("event" in path for path in files_lower) or any("kafka" in path for path in files_lower):
        patterns.append("Event-Driven")

    return {"patterns": patterns, "evidence": files_lower[:20]}


def diff_architecture_review(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    baseline_files: Optional[List[str]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    current_files = set(collect_file_paths(storage_path, source_path))
    baseline_set = set(baseline_files or [])

    added = sorted(current_files - baseline_set)
    removed = sorted(baseline_set - current_files)

    return {
        "added": added[:100],
        "removed": removed[:100],
        "added_count": len(added),
        "removed_count": len(removed),
    }


def onboarding_guide(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)
    root_files = [Path(path).name for path in files if len(Path(path).parts) <= 3]

    entry_points = [
        f
        for f in root_files
        if f.lower() in {"readme.md", "setup.py", "pyproject.toml", "package.json", "main.py"}
    ]
    suggestions = ["Start with README and configuration files", "Review entry points and tests"]

    return {
        "entry_points": entry_points,
        "suggestions": suggestions,
        "root_files": root_files[:50],
    }


def code_knowledge_graph(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    max_edges: int = 2000,
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    unsupported: List[str] = []

    for path in files:
        suffix = Path(path).suffix.lower()
        if suffix != ".py":
            if suffix in {".js", ".ts", ".tsx", ".jsx"}:
                unsupported.append(path)
            continue
        content = _read_file_text(path)
        if not content:
            continue
        try:
            tree = ast.parse(content)
        except Exception:
            continue

        function_stack: List[str] = []

        class Visitor(ast.NodeVisitor):
            def visit_ClassDef(self, node: ast.ClassDef):
                name = f"{path}:{node.name}"
                nodes[name] = {
                    "id": name,
                    "type": "class",
                    "file": path,
                    "line": node.lineno,
                }
                function_stack.append(name)
                self.generic_visit(node)
                function_stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef):
                parent = function_stack[-1] if function_stack else None
                name = f"{path}:{node.name}"
                nodes[name] = {
                    "id": name,
                    "type": "function",
                    "file": path,
                    "line": node.lineno,
                }
                if parent:
                    edges.append({"source": parent, "target": name, "type": "defines"})
                function_stack.append(name)
                self.generic_visit(node)
                function_stack.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                self.visit_FunctionDef(node)  # type: ignore[arg-type]

            def visit_Call(self, node: ast.Call):
                if len(edges) >= max_edges:
                    return
                caller = function_stack[-1] if function_stack else f"{path}:<module>"
                if caller not in nodes:
                    nodes[caller] = {
                        "id": caller,
                        "type": "module",
                        "file": path,
                        "line": 1,
                    }
                callee = None
                if isinstance(node.func, ast.Name):
                    callee = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    callee = node.func.attr
                if callee:
                    edges.append({"source": caller, "target": callee, "type": "calls"})
                self.generic_visit(node)

        Visitor().visit(tree)

    return {
        "nodes": list(nodes.values()),
        "edges": edges[:max_edges],
        "unsupported_files": unsupported[:50],
    }
