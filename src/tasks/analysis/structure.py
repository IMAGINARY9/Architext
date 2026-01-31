"""Structure analysis task.

Analyzes repository file structure and language statistics.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.tasks.core.base import BaseTask, FileInfo
from src.tasks.shared import _classify_import_clusters, _extract_imports


class StructureAnalysisTask(BaseTask):
    """
    Analyze repository file structure and language statistics.
    
    Generates file trees, language breakdowns, and import cluster analysis.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        source_path: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        depth: str = "shallow",
        output_format: str = "json",
    ):
        super().__init__(
            storage_path=storage_path,
            source_path=source_path,
            progress_callback=progress_callback,
            extensions=None,  # All files
            load_content=True,  # Need content for import analysis
        )
        self.depth = depth
        self.output_format = output_format
        self._depth_map = {"shallow": 2, "detailed": 4, "exhaustive": 8}
    
    def analyze(self, files: List[FileInfo]) -> Dict[str, Any]:
        """Analyze repository structure."""
        self._report_progress("analyze", "Building structure tree")
        
        paths = [f.path for f in files]
        tree = self._build_tree(paths)
        max_depth = self._depth_map.get(self.depth, 2)
        pruned = self._prune_tree(tree, max_depth)
        
        # Count extensions and languages
        extensions = Counter(f.extension for f in files)
        languages: Counter[str] = Counter()
        for f in files:
            languages[f.language] += 1
        
        # Import cluster analysis
        import_cluster_counts: Counter[str] = Counter()
        for f in files:
            if f.content:
                imports = _extract_imports(f.path, f.content)
                clusters = _classify_import_clusters(imports)
                import_cluster_counts.update(clusters)
        
        summary = {
            "total_files": len(files),
            "total_extensions": len(extensions),
            "languages": dict(languages),
            "top_extensions": dict(extensions.most_common(10)),
            "import_clusters": dict(import_cluster_counts),
        }
        
        if self.output_format == "markdown":
            lines = [
                "# Repository Structure", "",
                "## Summary", json.dumps(summary, indent=2), "",
                "## Tree"
            ]
            lines.extend(self._tree_to_markdown(pruned))
            return {"format": "markdown", "content": "\n".join(lines)}
        
        if self.output_format == "mermaid":
            diagram = self._tree_to_mermaid(pruned)
            return {"format": "mermaid", "content": diagram, "summary": summary}
        
        return {"format": "json", "summary": summary, "tree": pruned}
    
    @staticmethod
    def _build_tree(paths: List[str]) -> Dict[str, Any]:
        """Build nested dictionary tree from paths."""
        tree: Dict[str, Any] = {}
        for path in paths:
            parts = Path(path).parts
            cursor = tree
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor.setdefault("__files__", []).append(parts[-1])
        return tree
    
    @staticmethod
    def _prune_tree(tree: Dict[str, Any], max_depth: int, depth: int = 0) -> Dict[str, Any]:
        """Prune tree to max depth."""
        if depth >= max_depth:
            return {"...": "truncated"}
        
        pruned: Dict[str, Any] = {}
        for key, value in tree.items():
            if key == "__files__":
                pruned[key] = value
            elif isinstance(value, dict):
                pruned[key] = StructureAnalysisTask._prune_tree(value, max_depth, depth + 1)
        return pruned
    
    @staticmethod
    def _tree_to_markdown(tree: Dict[str, Any], indent: int = 0) -> List[str]:
        """Convert tree to markdown list."""
        lines: List[str] = []
        for key, value in sorted(tree.items()):
            if key == "__files__":
                for file in sorted(value):
                    lines.append("  " * indent + f"- {file}")
            elif isinstance(value, dict):
                lines.append("  " * indent + f"- {key}/")
                lines.extend(StructureAnalysisTask._tree_to_markdown(value, indent + 1))
        return lines
    
    @staticmethod
    def _tree_to_mermaid(tree: Dict[str, Any], root_label: str = "root") -> str:
        """Convert tree to Mermaid diagram."""
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


def analyze_structure_v2(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    depth: str = "shallow",
    output_format: str = "json",
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Analyze repository structure using BaseTask pattern."""
    task = StructureAnalysisTask(
        storage_path=storage_path,
        source_path=source_path,
        progress_callback=progress_callback,
        depth=depth,
        output_format=output_format,
    )
    return task.run()


# Alias
structure_analysis_v2 = analyze_structure_v2


__all__ = [
    "StructureAnalysisTask",
    "analyze_structure_v2",
    "structure_analysis_v2",
]
