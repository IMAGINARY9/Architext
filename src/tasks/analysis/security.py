"""Security analysis tasks.

Security scanning using regex and AST patterns.
"""
from __future__ import annotations

import ast
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

from src.tasks.core.base import BaseTask, FileInfo, CODE_EXTENSIONS


# Security detection rules
SECURITY_PATTERNS = [
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
                for rule in SECURITY_PATTERNS:
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


__all__ = [
    "SecurityHeuristicsTask",
    "security_heuristics_v2",
    "SECURITY_PATTERNS",
]
