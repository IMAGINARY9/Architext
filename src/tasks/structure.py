"""Structure analysis task."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.tasks.shared import (
    DEFAULT_EXTENSIONS,
    _classify_import_clusters,
    _extract_imports,
    _progress,
    _read_file_text,
    collect_file_paths,
)


def _build_tree(paths: Iterable[str]) -> Dict[str, Any]:
    tree: Dict[str, Any] = {}
    for path in paths:
        parts = Path(path).parts
        cursor = tree
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor.setdefault("__files__", []).append(parts[-1])
    return tree


def _prune_tree(tree: Dict[str, Any], max_depth: int, depth: int = 0) -> Dict[str, Any]:
    if depth >= max_depth:
        return {"...": "truncated"}

    pruned: Dict[str, Any] = {}
    for key, value in tree.items():
        if key == "__files__":
            pruned[key] = value
        elif isinstance(value, dict):
            pruned[key] = _prune_tree(value, max_depth, depth + 1)
    return pruned


def _tree_to_markdown(tree: Dict[str, Any], indent: int = 0) -> List[str]:
    lines: List[str] = []
    for key, value in sorted(tree.items()):
        if key == "__files__":
            for file in sorted(value):
                lines.append("  " * indent + f"- {file}")
        elif isinstance(value, dict):
            lines.append("  " * indent + f"- {key}/")
            lines.extend(_tree_to_markdown(value, indent + 1))
    return lines


def _tree_to_mermaid(tree: Dict[str, Any], root_label: str = "root") -> str:
    lines = ["graph TD", f"  {root_label}[{root_label}]"]
    node_id = 0

    def walk(node: Dict[str, Any], parent: str):
        nonlocal node_id
        for key, value in node.items():
            if key == "__files__":
                for file in value:
                    node_id += 1
                    file_id = f"node{node_id}"
                    lines.append(f"  {file_id}[{file}]")
                    lines.append(f"  {parent} --> {file_id}")
            elif isinstance(value, dict):
                node_id += 1
                dir_id = f"node{node_id}"
                lines.append(f"  {dir_id}[{key}/]")
                lines.append(f"  {parent} --> {dir_id}")
                walk(value, dir_id)

    walk(tree, root_label)
    return "\n".join(lines)


def analyze_structure(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    depth: str = "shallow",
    output_format: str = "json",
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)

    depth_map = {"shallow": 2, "detailed": 4, "exhaustive": 8}
    max_depth = depth_map.get(depth, 2)

    _progress(progress_callback, {"stage": "analyze", "message": "Building structure tree"})
    tree = _build_tree(files)
    pruned = _prune_tree(tree, max_depth)

    extensions = Counter(Path(path).suffix for path in files)
    languages: Counter[str] = Counter()
    for ext, count in extensions.items():
        languages[DEFAULT_EXTENSIONS.get(ext, "Other")] += count

    import_cluster_counts: Counter[str] = Counter()
    for path in files:
        text = _read_file_text(path)
        if not text:
            continue
        imports = _extract_imports(path, text)
        clusters = _classify_import_clusters(imports)
        import_cluster_counts.update(clusters)

    summary = {
        "total_files": len(files),
        "total_extensions": len(extensions),
        "languages": dict(languages),
        "top_extensions": dict(extensions.most_common(10)),
        "import_clusters": dict(import_cluster_counts),
    }

    if output_format == "markdown":
        lines = ["# Repository Structure", "", "## Summary", json.dumps(summary, indent=2), "", "## Tree"]
        lines.extend(_tree_to_markdown(pruned))
        return {"format": "markdown", "content": "\n".join(lines)}

    if output_format == "mermaid":
        diagram = _tree_to_mermaid(pruned)
        return {"format": "mermaid", "content": diagram, "summary": summary}

    return {"format": "json", "summary": summary, "tree": pruned}
