"""Extended BaseTask-based implementations for remaining tasks.

This module completes the migration of all tasks to the BaseTask pattern.
Includes: structure analysis, tech stack, architecture patterns, security,
duplication detection, and more.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import keyword
import re
import tokenize
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from src.tasks.base import (
    BaseTask,
    FileInfo,
    PYTHON_EXTENSIONS,
    JS_TS_EXTENSIONS,
    CODE_EXTENSIONS,
    DOCUMENTATION_EXTENSIONS,
    CONFIG_EXTENSIONS,
)
from src.tasks.graph import _build_import_graph
from src.tasks.shared import (
    DEFAULT_EXTENSIONS,
    _get_ts_parser,
    _classify_import_clusters,
    _extract_imports,
)


# =============================================================================
# Structure Analysis Task
# =============================================================================

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


# =============================================================================
# Tech Stack Task
# =============================================================================

FRAMEWORK_PATTERNS = {
    "django": ["django"],
    "flask": ["flask"],
    "fastapi": ["fastapi"],
    "requests": ["requests"],
    "sqlalchemy": ["sqlalchemy"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "pytest": ["pytest"],
    "react": ["react", "react-dom"],
    "vue": ["vue"],
    "angular": ["@angular"],
    "express": ["express"],
    "nestjs": ["@nestjs"],
    "spring": ["springframework", "spring-boot"],
}


class TechStackTask(BaseTask):
    """
    Detect technologies, frameworks, and languages in use.
    
    Scans for framework patterns in imports and code.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        source_path: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        output_format: str = "json",
    ):
        super().__init__(
            storage_path=storage_path,
            source_path=source_path,
            progress_callback=progress_callback,
            extensions=None,
            load_content=True,
        )
        self.output_format = output_format
    
    def analyze(self, files: List[FileInfo]) -> Dict[str, Any]:
        """Detect tech stack from files."""
        self._report_progress("analyze", "Scanning for framework usage")
        
        extensions = Counter(f.extension for f in files)
        languages: Counter[str] = Counter()
        for f in files:
            languages[f.language] += 1
        
        framework_hits: Dict[str, int] = defaultdict(int)
        framework_files: Dict[str, List[str]] = defaultdict(list)
        
        for f in files:
            if not f.content:
                continue
            lowered = f.content.lower()
            for framework, tokens in FRAMEWORK_PATTERNS.items():
                if any(token in lowered for token in tokens):
                    framework_hits[framework] += 1
                    if len(framework_files[framework]) < 10:
                        framework_files[framework].append(f.path)
        
        result = {
            "languages": dict(languages),
            "extensions": dict(extensions.most_common(15)),
            "frameworks": dict(framework_hits),
            "examples": framework_files,
        }
        
        if self.output_format == "markdown":
            lines = [
                "# Technology Stack", "",
                "## Languages", json.dumps(dict(languages), indent=2), "",
                "## Frameworks", json.dumps(dict(framework_hits), indent=2)
            ]
            return {"format": "markdown", "content": "\n".join(lines)}
        
        return {"format": "json", "data": result}


def tech_stack_v2(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    output_format: str = "json",
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Detect tech stack using BaseTask pattern."""
    task = TechStackTask(
        storage_path=storage_path,
        source_path=source_path,
        progress_callback=progress_callback,
        output_format=output_format,
    )
    return task.run()


# =============================================================================
# Architecture Pattern Detection Task
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


# =============================================================================
# Duplication Detection Tasks
# =============================================================================

class DuplicateBlocksTask(BaseTask):
    """
    Detect exact duplicate code blocks.
    
    Uses sliding window hash comparison to find duplicated code.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        source_path: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        min_lines: int = 8,
        max_findings: int = 50,
        max_windows_per_file: int = 6000,
    ):
        super().__init__(
            storage_path=storage_path,
            source_path=source_path,
            progress_callback=progress_callback,
            extensions=CODE_EXTENSIONS,
            load_content=True,
        )
        self.min_lines = min_lines
        self.max_findings = max_findings
        self.max_windows_per_file = max_windows_per_file
    
    def analyze(self, files: List[FileInfo]) -> Dict[str, Any]:
        """Find duplicate code blocks."""
        self._report_progress("analyze", "Scanning for duplicate blocks")
        
        duplicates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        scanned = 0
        
        for f in files:
            if f.extension not in {".py", ".js", ".ts", ".jsx", ".tsx"}:
                continue
            if not f.content:
                continue
            
            raw_lines = f.content.splitlines()
            if len(raw_lines) < self.min_lines:
                continue
            
            normalized = [self._normalize_line(line, f.extension) for line in raw_lines]
            window_limit = min(self.max_windows_per_file, max(len(raw_lines) - self.min_lines + 1, 0))
            
            for idx in range(min(window_limit, len(raw_lines) - self.min_lines + 1)):
                window = normalized[idx:idx + self.min_lines]
                if not all(window):
                    continue
                block = "\n".join(window)
                digest = hashlib.sha1(block.encode("utf-8")).hexdigest()
                duplicates[digest].append({
                    "file": f.path,
                    "start_line": idx + 1,
                    "end_line": idx + self.min_lines,
                })
            
            scanned += 1
        
        # Filter to actual duplicates (2+ occurrences)
        findings = [
            {
                "fingerprint": digest,
                "occurrence_count": len(occ),
                "occurrences": occ[:10],
            }
            for digest, occ in duplicates.items()
            if len(occ) >= 2
        ]
        findings.sort(key=lambda x: x["occurrence_count"], reverse=True)
        
        return {
            "count": len(findings),
            "scanned_files": scanned,
            "min_lines": self.min_lines,
            "findings": findings[:self.max_findings],
        }
    
    @staticmethod
    def _normalize_line(line: str, suffix: str) -> str:
        """Normalize line for comparison."""
        stripped = line.strip()
        if not stripped:
            return ""
        if suffix == ".py" and stripped.startswith("#"):
            return ""
        if suffix in {".js", ".ts", ".jsx", ".tsx"} and stripped.startswith("//"):
            return ""
        return stripped


def detect_duplicate_blocks_v2(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    min_lines: int = 8,
    max_findings: int = 50,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Detect duplicate blocks using BaseTask pattern."""
    task = DuplicateBlocksTask(
        storage_path=storage_path,
        source_path=source_path,
        progress_callback=progress_callback,
        min_lines=min_lines,
        max_findings=max_findings,
    )
    return task.run()


class SemanticDuplicationTask(BaseTask):
    """
    Detect semantically similar functions using token normalization.
    
    Normalizes identifiers and literals to find structurally similar code.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        source_path: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        min_tokens: int = 40,
        max_findings: int = 50,
    ):
        super().__init__(
            storage_path=storage_path,
            source_path=source_path,
            progress_callback=progress_callback,
            extensions=PYTHON_EXTENSIONS,
            load_content=True,
        )
        self.min_tokens = min_tokens
        self.max_findings = max_findings
    
    def analyze(self, files: List[FileInfo]) -> Dict[str, Any]:
        """Find semantically similar functions."""
        self._report_progress("analyze", "Scanning for semantic duplication")
        
        fingerprints: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        scanned = 0
        
        for f in files:
            if not f.ast_tree or not f.content:
                continue
            
            for node in ast.walk(f.ast_tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                
                segment = ast.get_source_segment(f.content, node) or ""
                normalized = self._normalize_tokens(segment)
                if not normalized:
                    continue
                
                token_count = len(normalized.split())
                if token_count < self.min_tokens:
                    continue
                
                digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
                fingerprints[digest].append({
                    "file": f.path,
                    "name": getattr(node, "name", "<anonymous>"),
                    "line": getattr(node, "lineno", 0),
                    "tokens": token_count,
                })
            
            scanned += 1
        
        # Filter to actual duplicates
        findings = [
            {
                "fingerprint": digest,
                "occurrence_count": len(occ),
                "occurrences": occ[:10],
            }
            for digest, occ in fingerprints.items()
            if len(occ) >= 2
        ]
        findings.sort(key=lambda x: x["occurrence_count"], reverse=True)
        
        return {
            "count": len(findings),
            "scanned_files": scanned,
            "min_tokens": self.min_tokens,
            "findings": findings[:self.max_findings],
        }
    
    @staticmethod
    def _normalize_tokens(segment: str) -> str:
        """Normalize Python tokens for semantic comparison."""
        tokens: List[str] = []
        try:
            for tok in tokenize.generate_tokens(io.StringIO(segment).readline):
                if tok.type in {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT}:
                    continue
                if tok.type == tokenize.COMMENT:
                    continue
                if tok.type == tokenize.NAME:
                    if keyword.iskeyword(tok.string):
                        tokens.append(tok.string)
                    else:
                        tokens.append("_id")
                elif tok.type == tokenize.STRING:
                    tokens.append("S")
                elif tok.type == tokenize.NUMBER:
                    tokens.append("0")
                else:
                    tokens.append(tok.string)
        except Exception:
            return ""
        return " ".join(tokens)


def detect_duplicate_blocks_semantic_v2(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    min_tokens: int = 40,
    max_findings: int = 50,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Detect semantic duplication using BaseTask pattern."""
    task = SemanticDuplicationTask(
        storage_path=storage_path,
        source_path=source_path,
        progress_callback=progress_callback,
        min_tokens=min_tokens,
        max_findings=max_findings,
    )
    return task.run()


# =============================================================================
# Security Tasks
# =============================================================================

SECURITY_RULES = [
    {
        "id": "py-open-user-input",
        "severity": "high",
        "extensions": {".py"},
        "pattern": re.compile(
            r"\bopen\(\s*[^)]*(request|input|user|filename|filepath|path)[^)]*\)",
            re.IGNORECASE,
        ),
        "description": "Potential file operation with user-controlled input",
    },
    {
        "id": "py-eval-exec",
        "severity": "high",
        "extensions": {".py"},
        "pattern": re.compile(r"\b(eval|exec)\(.*\)", re.IGNORECASE),
        "description": "Dynamic code execution detected",
    },
    {
        "id": "hardcoded-secret",
        "severity": "medium",
        "extensions": None,
        "pattern": re.compile(
            r"\b(api_key|secret|password|token|access_key)\b\s*[:=]\s*['\"][^'\"]{6,}['\"]",
            re.IGNORECASE,
        ),
        "description": "Potential hardcoded secret",
    },
]


class SecurityHeuristicsTask(BaseTask):
    """
    Security scanning using regex and AST patterns.
    
    Detects common security issues like hardcoded secrets, command injection,
    and dangerous function usage.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        source_path: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        max_findings: int = 500,
    ):
        super().__init__(
            storage_path=storage_path,
            source_path=source_path,
            progress_callback=progress_callback,
            extensions=CODE_EXTENSIONS,
            load_content=True,
        )
        self.max_findings = max_findings
    
    def analyze(self, files: List[FileInfo]) -> Dict[str, Any]:
        """Run security heuristics scan."""
        self._report_progress("analyze", "Running security heuristics")
        
        findings: List[Dict[str, Any]] = []
        
        for f in files:
            if not f.content:
                continue
            
            lines = f.content.splitlines()
            for idx, line in enumerate(lines, start=1):
                for rule in SECURITY_RULES:
                    extensions = rule.get("extensions")
                    if extensions and f.extension not in extensions:
                        continue
                    
                    pattern = rule.get("pattern")
                    if pattern and pattern.search(line):
                        findings.append({
                            "rule_id": rule["id"],
                            "severity": rule["severity"],
                            "description": rule["description"],
                            "file": f.path,
                            "line": idx,
                            "snippet": line.strip()[:300],
                        })
                        
                        if len(findings) >= self.max_findings:
                            return self._format_result(findings)
            
            # AST analysis for Python
            if f.extension == ".py" and f.ast_tree:
                findings.extend(self._analyze_python_ast(f))
        
        return self._format_result(findings)
    
    def _analyze_python_ast(self, f: FileInfo) -> List[Dict[str, Any]]:
        """Analyze Python AST for security issues."""
        findings: List[Dict[str, Any]] = []
        
        for node in ast.walk(f.ast_tree):
            if not isinstance(node, ast.Call):
                continue
            
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    name = f"{node.func.value.id}.{node.func.attr}"
            
            if name in {"eval", "exec"}:
                findings.append({
                    "rule_id": "py-ast-eval-exec",
                    "severity": "high",
                    "description": "Dynamic code execution detected (AST)",
                    "file": f.path,
                    "line": getattr(node, "lineno", 0),
                    "snippet": name,
                })
        
        return findings
    
    def _format_result(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format security scan results."""
        by_severity = Counter(f["severity"] for f in findings)
        return {
            "findings": findings[:self.max_findings],
            "count": len(findings),
            "by_severity": dict(by_severity),
        }


def security_heuristics_v2(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    max_findings: int = 500,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Run security heuristics using BaseTask pattern."""
    task = SecurityHeuristicsTask(
        storage_path=storage_path,
        source_path=source_path,
        progress_callback=progress_callback,
        max_findings=max_findings,
    )
    return task.run()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Task classes
    "StructureAnalysisTask",
    "TechStackTask",
    "ArchitecturePatternTask",
    "ImpactAnalysisTask",
    "DependencyGraphTask",
    "DuplicateBlocksTask",
    "SemanticDuplicationTask",
    "SecurityHeuristicsTask",
    # Wrapper functions
    "analyze_structure_v2",
    "tech_stack_v2",
    "architecture_pattern_detection_v2",
    "impact_analysis_v2",
    "dependency_graph_export_v2",
    "detect_duplicate_blocks_v2",
    "detect_duplicate_blocks_semantic_v2",
    "security_heuristics_v2",
]
