"""Security analysis tasks."""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tasks.shared import (
    SECURITY_RULES,
    SEMANTIC_VULNERABILITY_QUERIES,
    ProgressCallback,
    _progress,
    _read_file_text,
    collect_file_paths,
)


def _extract_call_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base_name = _extract_call_name(node.value)
        if base_name:
            return f"{base_name}.{node.attr}"
        return node.attr
    return None


def _is_truthy_constant(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return bool(node.value)
    if isinstance(node, ast.NameConstant):
        return bool(node.value)
    return False


def _scan_security_rules(files: List[str], max_findings: int = 500) -> List[Dict[str, Any]]:
    """Apply regex-based security rules against every source line."""
    findings: List[Dict[str, Any]] = []
    for path in files:
        suffix = Path(path).suffix.lower()
        content = _read_file_text(path)
        if not content:
            continue
        lines = content.splitlines()
        for idx, line in enumerate(lines, start=1):
            for rule in SECURITY_RULES:
                extensions = rule.get("extensions")
                if extensions and suffix not in extensions:
                    continue
                pattern = rule.get("pattern")
                if pattern and pattern.search(line):
                    findings.append(
                        {
                            "rule_id": rule.get("id"),
                            "severity": rule.get("severity"),
                            "description": rule.get("description"),
                            "file": path,
                            "line": idx,
                            "snippet": line.strip()[:300],
                        }
                    )
                    if len(findings) >= max_findings:
                        return findings
    return findings


def _scan_python_ast_security(path: str, content: str) -> List[Dict[str, Any]]:
    """Walk the Python AST to find dangerous calls (eval, exec, subprocess, etc.)."""
    findings: List[Dict[str, Any]] = []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return findings

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            name = _extract_call_name(node.func)

            if name in {"eval", "exec"}:
                findings.append(
                    {
                        "rule_id": "py-ast-eval-exec",
                        "severity": "high",
                        "description": "Dynamic code execution detected (AST)",
                        "file": path,
                        "line": getattr(node, "lineno", 0),
                        "snippet": name,
                    }
                )
            if name in {"subprocess.run", "subprocess.Popen", "os.system"}:
                findings.append(
                    {
                        "rule_id": "py-ast-command-exec",
                        "severity": "high",
                        "description": "Command execution detected (AST)",
                        "file": path,
                        "line": getattr(node, "lineno", 0),
                        "snippet": name,
                    }
                )

            if name in {"subprocess.run", "subprocess.Popen"}:
                for kw in node.keywords:
                    if kw.arg == "shell" and _is_truthy_constant(kw.value):
                        findings.append(
                            {
                                "rule_id": "py-ast-subprocess-shell-true",
                                "severity": "high",
                                "description": "subprocess call with shell=True detected (AST)",
                                "file": path,
                                "line": getattr(node, "lineno", 0),
                                "snippet": name,
                            }
                        )
                        break

            if name == "yaml.load":
                has_safe_loader = any(
                    kw.arg == "Loader"
                    and _extract_call_name(kw.value) in {"yaml.SafeLoader", "SafeLoader"}
                    for kw in node.keywords
                )
                if not has_safe_loader:
                    findings.append(
                        {
                            "rule_id": "py-ast-yaml-unsafe-load",
                            "severity": "high",
                            "description": "yaml.load call without SafeLoader detected (AST)",
                            "file": path,
                            "line": getattr(node, "lineno", 0),
                            "snippet": name,
                        }
                    )

            self.generic_visit(node)

    Visitor().visit(tree)
    return findings


def _scan_python_taint_security(path: str, content: str) -> List[Dict[str, Any]]:
    """Simple taint analysis: detect user-input flowing into sensitive sinks."""
    findings: List[Dict[str, Any]] = []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return findings

    source_keywords = {
        "request",
        "req",
        "user",
        "input",
        "query",
        "params",
        "path",
        "filename",
        "filepath",
        "body",
    }
    sinks = {
        "open",
        "eval",
        "exec",
        "os.system",
        "subprocess.run",
        "subprocess.Popen",
        "Path",
        "read_text",
        "read_bytes",
    }

    def is_tainted_name(name: str) -> bool:
        lowered = name.lower()
        return any(token in lowered for token in source_keywords)

    def extract_name(node: ast.AST) -> Optional[str]:
        return _extract_call_name(node)

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.tainted_stack: List[set[str]] = [set()]

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            tainted = set()
            for arg in node.args.args:
                if is_tainted_name(arg.arg):
                    tainted.add(arg.arg)
            self.tainted_stack.append(tainted)
            self.generic_visit(node)
            self.tainted_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            tainted = set()
            for arg in node.args.args:
                if is_tainted_name(arg.arg):
                    tainted.add(arg.arg)
            self.tainted_stack.append(tainted)
            self.generic_visit(node)
            self.tainted_stack.pop()

        def visit_Assign(self, node: ast.Assign) -> None:
            current = self.tainted_stack[-1]
            value_name = extract_name(node.value)
            if value_name and is_tainted_name(value_name):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        current.add(target.id)
            if isinstance(node.value, ast.Name) and node.value.id in current:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        current.add(target.id)
            if isinstance(node.value, ast.Call):
                call_name = extract_name(node.value.func) or ""
                if call_name in {"input"}:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            current.add(target.id)
            if isinstance(node.value, ast.JoinedStr):
                has_tainted_part = False
                for value in node.value.values:
                    if isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name):
                        if value.value.id in current:
                            has_tainted_part = True
                            break
                if has_tainted_part:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            current.add(target.id)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            call_name = extract_name(node.func) or ""
            current = self.tainted_stack[-1]

            tainted_args = []
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id in current:
                    tainted_args.append(arg.id)
                elif isinstance(arg, ast.Attribute):
                    base = extract_name(arg)
                    if base and any(name in base for name in current):
                        tainted_args.append(base)
                elif isinstance(arg, ast.JoinedStr):
                    for value in arg.values:
                        if isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name):
                            if value.value.id in current:
                                tainted_args.append(value.value.id)

            for kw in node.keywords:
                if isinstance(kw.value, ast.Name) and kw.value.id in current:
                    tainted_args.append(kw.value.id)
                elif isinstance(kw.value, ast.JoinedStr):
                    for value in kw.value.values:
                        if isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name):
                            if value.value.id in current:
                                tainted_args.append(value.value.id)

            if call_name in sinks and tainted_args:
                findings.append(
                    {
                        "rule_id": "py-ast-taint-flow",
                        "severity": "high",
                        "description": "Potential user input flows into sensitive sink",
                        "file": path,
                        "line": getattr(node, "lineno", 0),
                        "snippet": call_name,
                        "tainted_args": tainted_args,
                    }
                )

            self.generic_visit(node)

    Visitor().visit(tree)
    return findings


def security_heuristics(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    max_findings: int = 500,
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    """Run a combined regex + AST + taint security scan on all source files."""
    if max_findings <= 0:
        return {
            "findings": [],
            "counts": {
                "total": 0,
                "by_severity": {},
                "by_rule": {},
            },
        }

    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)
    _progress(progress_callback, {"stage": "analyze", "message": "Scanning for heuristic matches"})

    findings = _scan_security_rules(files, max_findings=max_findings)
    if len(findings) < max_findings:
        for path in files:
            if Path(path).suffix.lower() != ".py":
                continue
            content = _read_file_text(path)
            if not content:
                continue

            for item in _scan_python_ast_security(path, content):
                if len(findings) >= max_findings:
                    break
                findings.append(item)

            if len(findings) >= max_findings:
                break

            for item in _scan_python_taint_security(path, content):
                if len(findings) >= max_findings:
                    break
                findings.append(item)

            if len(findings) >= max_findings:
                break

    findings.sort(
        key=lambda item: (
            str(item.get("file", "")),
            int(item.get("line", 0) or 0),
            str(item.get("rule_id", "")),
        )
    )
    severity_counts = Counter(item.get("severity", "unknown") for item in findings)
    rule_counts = Counter(item.get("rule_id", "unknown") for item in findings)

    return {
        "findings": findings,
        "counts": {
            "total": len(findings),
            "by_severity": dict(severity_counts),
            "by_rule": dict(rule_counts),
        },
    }


def detect_vulnerabilities(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    """Run heuristic scan plus optional semantic (RAG) vulnerability queries."""
    heuristics = security_heuristics(
        storage_path=storage_path,
        source_path=source_path,
        progress_callback=progress_callback,
    )

    semantic_results = []
    semantic_enabled = False
    semantic_error = None

    if storage_path:
        try:
            from src.indexer import load_existing_index, query_index, initialize_settings
            from src.config import load_settings
            from src.api_utils import extract_sources

            initialize_settings(load_settings())
            index = load_existing_index(storage_path)
            semantic_enabled = True
            for query in SEMANTIC_VULNERABILITY_QUERIES:
                response = query_index(index, query["prompt"])
                semantic_results.append(
                    {
                        "id": query["id"],
                        "prompt": query["prompt"],
                        "answer": str(response),
                        "sources": extract_sources(response),
                    }
                )
        except Exception as exc:  # pylint: disable=broad-except
            semantic_error = str(exc)

    return {
        "heuristics": heuristics,
        "semantic": semantic_results,
        "semantic_enabled": semantic_enabled,
        "semantic_error": semantic_error,
    }
