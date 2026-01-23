"""Security analysis tasks."""
from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tasks.shared import _progress, _read_file_text, collect_file_paths

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
        "id": "py-path-read-user-input",
        "severity": "high",
        "extensions": {".py"},
        "pattern": re.compile(
            r"\b(read_text|read_bytes)\(\s*[^)]*(request|input|user|filename|filepath|path)[^)]*\)",
            re.IGNORECASE,
        ),
        "description": "Potential file read with user-controlled input",
    },
    {
        "id": "py-subprocess-user-input",
        "severity": "high",
        "extensions": {".py"},
        "pattern": re.compile(
            r"\b(subprocess\.run|subprocess\.popen|os\.system)\([^)]*(request|input|user|params|query)[^)]*\)",
            re.IGNORECASE,
        ),
        "description": "Potential command execution using user input",
    },
    {
        "id": "py-eval-exec",
        "severity": "high",
        "extensions": {".py"},
        "pattern": re.compile(r"\b(eval|exec)\(.*\)", re.IGNORECASE),
        "description": "Dynamic code execution detected",
    },
    {
        "id": "js-fs-user-input",
        "severity": "high",
        "extensions": {".js", ".ts", ".jsx", ".tsx"},
        "pattern": re.compile(
            r"\bfs\.(readFile|readFileSync|writeFile|writeFileSync|createReadStream|createWriteStream)\(.*(req\.|request|params|query|body)\b",
            re.IGNORECASE,
        ),
        "description": "Potential fs operation with request data",
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

SEMANTIC_VULNERABILITY_QUERIES = [
    {
        "id": "unvalidated-file-io",
        "prompt": "Where is user input passed into file operations without validation?",
    },
    {
        "id": "path-traversal",
        "prompt": "Find any code that constructs file paths from user input without sanitizing traversal like ..",
    },
    {
        "id": "silent-exceptions",
        "prompt": "Locate try/except blocks that swallow errors without logging or re-throwing.",
    },
    {
        "id": "dynamic-code-exec",
        "prompt": "Find dynamic code execution (eval/exec) or unsafe command execution paths.",
    },
]


def _scan_security_rules(files: List[str], max_findings: int = 500) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    for path in files:
        suffix = Path(path).suffix.lower()
        content = _read_file_text(path)
        if not content:
            continue
        lines = content.splitlines()
        for idx, line in enumerate(lines, start=1):
            for rule in SECURITY_RULES:
                extensions = rule.get("extensions")  # type: ignore[attr-defined]
                if extensions and suffix not in extensions:
                    continue
                pattern = rule.get("pattern")  # type: ignore[attr-defined]
                if pattern and pattern.search(line):
                    findings.append(
                        {
                            "rule_id": rule.get("id"),  # type: ignore[attr-defined]
                            "severity": rule.get("severity"),  # type: ignore[attr-defined]
                            "description": rule.get("description"),  # type: ignore[attr-defined]
                            "file": path,
                            "line": idx,
                            "snippet": line.strip()[:300],
                        }
                    )
                    if len(findings) >= max_findings:
                        return findings
    return findings


def _scan_python_ast_security(path: str, content: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    try:
        tree = ast.parse(content)
    except Exception:
        return findings

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                base = node.func.value
                if isinstance(base, ast.Name):
                    name = f"{base.id}.{node.func.attr}"
                else:
                    name = node.func.attr

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

            self.generic_visit(node)

    Visitor().visit(tree)
    return findings


def _scan_python_taint_security(path: str, content: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    try:
        tree = ast.parse(content)
    except Exception:
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
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                return f"{node.value.id}.{node.attr}"
            return node.attr
        return None

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.tainted_stack: List[set[str]] = [set()]

        def visit_FunctionDef(self, node: ast.FunctionDef):
            tainted = set()
            for arg in node.args.args:
                if is_tainted_name(arg.arg):
                    tainted.add(arg.arg)
            self.tainted_stack.append(tainted)
            self.generic_visit(node)
            self.tainted_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            self.visit_FunctionDef(node)  # type: ignore[arg-type]

        def visit_Assign(self, node: ast.Assign):
            current = self.tainted_stack[-1]
            value_name = extract_name(node.value)
            if value_name and is_tainted_name(value_name):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        current.add(target.id)
            if isinstance(node.value, ast.Call):
                call_name = extract_name(node.value.func) or ""
                if call_name in {"input"}:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            current.add(target.id)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call):
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
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)
    _progress(progress_callback, {"stage": "analyze", "message": "Scanning for heuristic matches"})

    findings = _scan_security_rules(files, max_findings=max_findings)
    for path in files:
        if Path(path).suffix.lower() != ".py":
            continue
        content = _read_file_text(path)
        if not content:
            continue
        findings.extend(_scan_python_ast_security(path, content))
        findings.extend(_scan_python_taint_security(path, content))
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
    progress_callback=None,
) -> Dict[str, Any]:
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
