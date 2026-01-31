"""Architecture analysis tasks.

Includes pattern detection, impact analysis, and dependency graph export.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.tasks.core.base import BaseTask, FileInfo, CODE_EXTENSIONS
from src.tasks.graph import _build_import_graph


# =============================================================================
# Architecture Pattern Detection
# =============================================================================

PATTERN_RULES = {
    "MVC": {
        "required": [["controllers", "controller"], ["views", "view", "templates"]],
        "optional": [["models", "model"]],
        "min_confidence": 0.6,
    },
    "Service-Repository": {
        "required": [["services", "service"], ["repositories", "repository", "repos"]],
        "optional": [["entities", "entity", "models"]],
        "min_confidence": 0.6,
    },
    "Layered Architecture": {
        "required": [["domain", "core"], ["infrastructure", "adapters"]],
        "optional": [["application", "usecases", "use_cases"]],
        "min_confidence": 0.5,
    },
    "Hexagonal/Ports-Adapters": {
        "required": [["ports"], ["adapters"]],
        "optional": [["domain", "core"]],
        "min_confidence": 0.7,
    },
    "CQRS": {
        "required": [["commands", "command"], ["queries", "query"]],
        "optional": [["handlers", "handler"]],
        "min_confidence": 0.7,
    },
    "Plugin Architecture": {
        "required": [["plugins", "plugin", "extensions", "extension"]],
        "optional": [["hooks", "hook"]],
        "min_confidence": 0.5,
    },
    "Event-Driven": {
        "required": [["events", "event"], ["handlers", "handler", "listeners", "subscriber"]],
        "optional": [["kafka", "rabbitmq", "pubsub"]],
        "min_confidence": 0.6,
    },
    "Microservices": {
        "required": [["docker-compose", "kubernetes", "k8s"]],
        "optional": [["gateway", "api-gateway"], ["service-"]],
        "min_confidence": 0.5,
    },
}


class ArchitecturePatternTask(BaseTask):
    """
    Detect architectural patterns with confidence scoring.
    
    Uses multi-signal analysis to identify patterns like MVC, microservices,
    event-driven, etc.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        source_path: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ):
        super().__init__(
            storage_path=storage_path,
            source_path=source_path,
            progress_callback=progress_callback,
            extensions=None,
            load_content=False,  # Only need file paths
        )
    
    def analyze(self, files: List[FileInfo]) -> Dict[str, Any]:
        """Detect architecture patterns."""
        self._report_progress("analyze", "Detecting architecture patterns")
        
        detected_patterns: List[Dict[str, Any]] = []
        files_lower = [f.path.replace("\\", "/").lower() for f in files]
        
        for pattern_name, rules in PATTERN_RULES.items():
            required_matches = []
            optional_matches = []
            evidence = []
            
            # Check required patterns
            for keywords in rules["required"]:
                matching = [f for f in files_lower if any(kw in f for kw in keywords)]
                if matching:
                    required_matches.append(keywords[0])
                    evidence.extend(matching[:3])
            
            # Check optional patterns
            for keywords in rules.get("optional", []):
                matching = [f for f in files_lower if any(kw in f for kw in keywords)]
                if matching:
                    optional_matches.append(keywords[0])
                    evidence.extend(matching[:2])
            
            # Calculate confidence
            if len(required_matches) == len(rules["required"]):
                base_confidence = 0.6
                optional_bonus = 0.1 * len(optional_matches)
                confidence = min(base_confidence + optional_bonus, 1.0)
                
                if confidence >= rules["min_confidence"]:
                    detected_patterns.append({
                        "pattern": pattern_name,
                        "confidence": round(confidence, 2),
                        "required_signals": required_matches,
                        "optional_signals": optional_matches,
                        "evidence": list(set(evidence))[:5],
                    })
        
        # Sort by confidence
        detected_patterns.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "patterns": [p["pattern"] for p in detected_patterns],
            "detailed": detected_patterns,
            "files_analyzed": len(files),
        }


def architecture_pattern_detection_v2(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Detect architecture patterns using BaseTask pattern."""
    task = ArchitecturePatternTask(
        storage_path=storage_path,
        source_path=source_path,
        progress_callback=progress_callback,
    )
    return task.run()


# =============================================================================
# Impact Analysis Task
# =============================================================================

class ImpactAnalysisTask(BaseTask):
    """
    Analyze module dependencies and find affected modules.
    
    Given a module name, finds all modules that depend on it (directly or transitively).
    """
    
    def __init__(
        self,
        module: str,
        storage_path: Optional[str] = None,
        source_path: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ):
        super().__init__(
            storage_path=storage_path,
            source_path=source_path,
            progress_callback=progress_callback,
            extensions=CODE_EXTENSIONS,
            load_content=False,
        )
        self.module = module
    
    def analyze(self, files: List[FileInfo]) -> Dict[str, Any]:
        """Find affected modules."""
        self._report_progress("analyze", "Building dependency graph")
        
        paths = [f.path for f in files]
        graph = _build_import_graph(paths)
        
        targets = [key for key in graph if self.module in key]
        if not targets:
            return {"module": self.module, "affected": [], "note": "Module not found"}
        
        # Build reverse graph
        reverse_graph: Dict[str, List[str]] = defaultdict(list)
        for src, deps in graph.items():
            for dep in deps:
                reverse_graph[dep].append(src)
        
        # Find all affected modules (transitive)
        affected = set()
        stack = list(targets)
        while stack:
            current = stack.pop()
            for dep in reverse_graph.get(current, []):
                if dep not in affected:
                    affected.add(dep)
                    stack.append(dep)
        
        return {
            "module": self.module,
            "targets": targets,
            "affected": sorted(affected),
            "affected_count": len(affected),
        }


def impact_analysis_v2(
    module: str,
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Perform impact analysis using BaseTask pattern."""
    task = ImpactAnalysisTask(
        module=module,
        storage_path=storage_path,
        source_path=source_path,
        progress_callback=progress_callback,
    )
    return task.run()


# =============================================================================
# Dependency Graph Task
# =============================================================================

class DependencyGraphTask(BaseTask):
    """
    Export module dependency graph in various formats.
    
    Supports: JSON, Mermaid, GraphML, DOT (Graphviz)
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        source_path: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        output_format: str = "mermaid",
    ):
        super().__init__(
            storage_path=storage_path,
            source_path=source_path,
            progress_callback=progress_callback,
            extensions=CODE_EXTENSIONS,
            load_content=False,
        )
        self.output_format = output_format
    
    def analyze(self, files: List[FileInfo]) -> Dict[str, Any]:
        """Export dependency graph."""
        self._report_progress("analyze", "Building dependency graph")
        
        paths = [f.path for f in files]
        graph = _build_import_graph(paths)
        
        edges = [(src, dep) for src, deps in graph.items() for dep in deps]
        
        if self.output_format == "json":
            return {"format": "json", "nodes": list(graph.keys()), "edges": edges}
        
        if self.output_format == "graphml":
            lines = ["<graphml>", "<graph edgedefault=\"directed\">"]
            for src, dep in edges:
                lines.append(f'  <edge source="{src}" target="{dep}"/>')
            lines.extend(["</graph>", "</graphml>"])
            return {"format": "graphml", "content": "\n".join(lines)}
        
        if self.output_format == "mermaid":
            lines = ["graph TD"]
            for src, dep in edges:
                src_id = src.replace("-", "_").replace(".", "_").replace("/", "_")
                dep_id = dep.replace("-", "_").replace(".", "_").replace("/", "_")
                lines.append(f"  {src_id}[{Path(src).name}] --> {dep_id}[{Path(dep).name}]")
            return {"format": "mermaid", "content": "\n".join(lines), "edge_count": len(edges)}
        
        if self.output_format == "dot":
            lines = ["digraph dependencies {", "  rankdir=LR;", "  node [shape=box];"]
            for src, dep in edges:
                src_label = Path(src).name.replace('"', '\\"')
                dep_label = Path(dep).name.replace('"', '\\"')
                lines.append(f'  "{src_label}" -> "{dep_label}";')
            lines.append("}")
            return {"format": "dot", "content": "\n".join(lines), "edge_count": len(edges)}
        
        return {"format": "json", "nodes": list(graph.keys()), "edges": edges}


def dependency_graph_export_v2(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    output_format: str = "mermaid",
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Export dependency graph using BaseTask pattern."""
    task = DependencyGraphTask(
        storage_path=storage_path,
        source_path=source_path,
        progress_callback=progress_callback,
        output_format=output_format,
    )
    return task.run()


__all__ = [
    "ArchitecturePatternTask",
    "ImpactAnalysisTask",
    "DependencyGraphTask",
    "architecture_pattern_detection_v2",
    "impact_analysis_v2",
    "dependency_graph_export_v2",
    "PATTERN_RULES",
]
