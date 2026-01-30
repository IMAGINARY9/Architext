"""Architecture and dependency analysis tasks."""
from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tasks.graph import _build_import_graph
from src.tasks.shared import _get_ts_parser, _progress, _read_file_text, collect_file_paths


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
            src_id = src.replace("-", "_").replace(".", "_").replace("/", "_")
            dep_id = dep.replace("-", "_").replace(".", "_").replace("/", "_")
            lines.append(f"  {src_id}[{Path(src).name}] --> {dep_id}[{Path(dep).name}]")
        return {"format": "mermaid", "content": "\n".join(lines), "edge_count": len(edges)}

    if output_format == "dot":
        lines = ["digraph dependencies {", "  rankdir=LR;", "  node [shape=box];"]
        for src, dep in edges:
            src_label = Path(src).name.replace('"', '\\"')
            dep_label = Path(dep).name.replace('"', '\\"')
            lines.append(f'  "{src_label}" -> "{dep_label}";')
        lines.append("}")
        return {"format": "dot", "content": "\n".join(lines), "edge_count": len(edges)}

    return {"format": "json", "nodes": list(graph.keys()), "edges": edges}


# Architecture pattern detection rules with confidence scoring
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


def architecture_pattern_detection(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """Detect architectural patterns with confidence scoring and evidence."""
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)

    detected_patterns: List[Dict[str, Any]] = []
    files_lower = [str(path).replace("\\", "/").lower() for path in files]

    for pattern_name, rules in PATTERN_RULES.items():
        required_matches = []
        optional_matches = []
        evidence = []

        # Check required patterns (all must have at least one match)
        for keywords in rules["required"]:
            matching_files = [
                f for f in files_lower
                if any(kw in f for kw in keywords)
            ]
            if matching_files:
                required_matches.append(keywords[0])
                evidence.extend(matching_files[:3])

        # Check optional patterns
        for keywords in rules.get("optional", []):
            matching_files = [
                f for f in files_lower
                if any(kw in f for kw in keywords)
            ]
            if matching_files:
                optional_matches.append(keywords[0])
                evidence.extend(matching_files[:2])

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

    # Sort by confidence descending
    detected_patterns.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "patterns": [p["pattern"] for p in detected_patterns],
        "detailed": detected_patterns,
        "files_analyzed": len(files),
    }


def code_knowledge_graph(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    max_edges: int = 2000,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Build a call graph showing classes, functions, and their relationships.
    
    Supports:
    - Python (.py) - Full AST parsing
    - JavaScript/TypeScript (.js, .ts, .jsx, .tsx) - Tree-sitter parsing
    """
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    languages_parsed: Dict[str, int] = defaultdict(int)

    for path in files:
        suffix = Path(path).suffix.lower()
        content = _read_file_text(path)
        if not content:
            continue
        
        if suffix == ".py":
            _parse_python_knowledge_graph(path, content, nodes, edges, max_edges)
            languages_parsed["python"] += 1
        elif suffix in {".js", ".jsx"}:
            if _parse_js_ts_knowledge_graph(path, content, nodes, edges, max_edges, "javascript"):
                languages_parsed["javascript"] += 1
        elif suffix in {".ts", ".tsx"}:
            lang = "tsx" if suffix == ".tsx" else "typescript"
            if _parse_js_ts_knowledge_graph(path, content, nodes, edges, max_edges, lang):
                languages_parsed["typescript"] += 1

    return {
        "nodes": list(nodes.values()),
        "edges": edges[:max_edges],
        "languages_parsed": dict(languages_parsed),
        "total_nodes": len(nodes),
        "total_edges": min(len(edges), max_edges),
    }


def _parse_python_knowledge_graph(
    path: str,
    content: str,
    nodes: Dict[str, Dict[str, Any]],
    edges: List[Dict[str, Any]],
    max_edges: int,
) -> None:
    """Parse Python file and extract knowledge graph nodes/edges."""
    try:
        tree = ast.parse(content)
    except Exception:
        return

    function_stack: List[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef):
            name = f"{path}:{node.name}"
            nodes[name] = {
                "id": name,
                "type": "class",
                "file": path,
                "line": node.lineno,
                "language": "python",
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
                "language": "python",
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
                    "language": "python",
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


def _parse_js_ts_knowledge_graph(
    path: str,
    content: str,
    nodes: Dict[str, Dict[str, Any]],
    edges: List[Dict[str, Any]],
    max_edges: int,
    language: str,
) -> bool:
    """Parse JavaScript/TypeScript file and extract knowledge graph nodes/edges."""
    parser = _get_ts_parser(language)
    if not parser:
        return False
    
    try:
        tree = parser.parse(content.encode("utf-8", errors="ignore"))
    except Exception:
        return False
    
    root = tree.root_node
    function_stack: List[str] = []
    lang_name = "typescript" if language in ("typescript", "tsx") else "javascript"
    
    def get_node_text(node) -> str:
        return content[node.start_byte:node.end_byte]
    
    def walk(node):
        if len(edges) >= max_edges:
            return
            
        # Class declarations
        if node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                class_name = get_node_text(name_node)
                node_id = f"{path}:{class_name}"
                nodes[node_id] = {
                    "id": node_id,
                    "type": "class",
                    "file": path,
                    "line": node.start_point[0] + 1,
                    "language": lang_name,
                }
                function_stack.append(node_id)
                for child in node.children:
                    walk(child)
                function_stack.pop()
                return
        
        # Function declarations and arrow functions
        if node.type in ("function_declaration", "method_definition", "arrow_function"):
            name = None
            if node.type == "function_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = get_node_text(name_node)
            elif node.type == "method_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = get_node_text(name_node)
            elif node.type == "arrow_function":
                # Try to get name from variable declaration parent
                parent = node.parent
                if parent and parent.type == "variable_declarator":
                    name_node = parent.child_by_field_name("name")
                    if name_node:
                        name = get_node_text(name_node)
            
            if name:
                parent_ctx = function_stack[-1] if function_stack else None
                node_id = f"{path}:{name}"
                nodes[node_id] = {
                    "id": node_id,
                    "type": "function",
                    "file": path,
                    "line": node.start_point[0] + 1,
                    "language": lang_name,
                }
                if parent_ctx:
                    edges.append({"source": parent_ctx, "target": node_id, "type": "defines"})
                function_stack.append(node_id)
                for child in node.children:
                    walk(child)
                function_stack.pop()
                return
        
        # Function calls
        if node.type == "call_expression":
            caller = function_stack[-1] if function_stack else f"{path}:<module>"
            if caller not in nodes:
                nodes[caller] = {
                    "id": caller,
                    "type": "module",
                    "file": path,
                    "line": 1,
                    "language": lang_name,
                }
            
            func_node = node.child_by_field_name("function")
            if func_node:
                callee = None
                if func_node.type == "identifier":
                    callee = get_node_text(func_node)
                elif func_node.type == "member_expression":
                    prop = func_node.child_by_field_name("property")
                    if prop:
                        callee = get_node_text(prop)
                if callee:
                    edges.append({"source": caller, "target": callee, "type": "calls"})
        
        # Recurse
        for child in node.children:
            walk(child)
    
    walk(root)
    return True
