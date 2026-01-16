"""Phase 2.5 analysis tasks for Architext."""
from __future__ import annotations

import ast
import json
import os
import re
import time
import hashlib
import io
import keyword
import tokenize
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import chromadb

from src.file_filters import should_skip_path


DEFAULT_EXTENSIONS = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".jsx": "JavaScript",
    ".java": "Java",
    ".go": "Go",
    ".rs": "Rust",
    ".c": "C",
    ".cpp": "C++",
    ".h": "C/C++",
    ".hpp": "C/C++",
    ".cs": "C#",
    ".rb": "Ruby",
    ".php": "PHP",
    ".md": "Markdown",
    ".rst": "reStructuredText",
}


ALLOWED_ANALYSIS_EXTENSIONS = set(DEFAULT_EXTENSIONS.keys()) | {
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
    ".txt",
}


ALLOWED_ANALYSIS_BASENAMES = {
    "dockerfile",
    "makefile",
}

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


IMPORT_CLUSTER_RULES = {
    "http_api": {"fastapi", "flask", "django", "starlette"},
    "mcp": {"mcp"},
    "vector_store": {"chromadb", "qdrant", "pinecone", "weaviate"},
    "llm": {"llama_index", "openai", "langchain", "transformers"},
    "cli": {"typer", "click", "argparse"},
    "database": {"sqlalchemy", "psycopg", "psycopg2", "sqlite3", "pymongo", "redis"},
}


FRAMEWORK_SETTINGS_IGNORES = {
    "pydantic": {"model_config"},
}


def _progress(progress_callback, payload: Dict[str, Any]):
    if progress_callback:
        progress_callback(payload)


@lru_cache(maxsize=32)
def _load_files_from_storage_cached(storage_path: str) -> tuple[str, ...]:
    return tuple(_load_files_from_storage(storage_path))


@lru_cache(maxsize=32)
def _gather_files_cached(source_path: str) -> tuple[str, ...]:
    return tuple(_gather_files(source_path))


def _load_files_from_storage(storage_path: str) -> List[str]:
    try:
        from src.config import load_settings
    except Exception:
        load_settings = None

    if load_settings:
        settings = load_settings()
        if settings.vector_store_provider != "chroma":
            raise ValueError("storage-based tasks require chroma vector store; use --source instead")
    client = chromadb.PersistentClient(path=storage_path)
    collection = client.get_or_create_collection("architext_db")
    data = collection.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []
    file_paths = []
    for meta in metadatas:
        if not meta:
            continue
        path = meta.get("file_path")
        if path:
            file_paths.append(path)
    return sorted(set(file_paths))


def _gather_files(source_path: str) -> List[str]:
    root = Path(source_path)
    files: List[str] = []
    for file_path in root.rglob("*"):
        if file_path.is_dir():
            continue
        if should_skip_path(file_path):
            continue
        suffix = file_path.suffix.lower()
        if suffix:
            if suffix not in ALLOWED_ANALYSIS_EXTENSIONS:
                continue
        else:
            if file_path.name.lower() not in ALLOWED_ANALYSIS_BASENAMES:
                continue
        files.append(str(file_path))
    return files


def collect_file_paths(storage_path: Optional[str], source_path: Optional[str]) -> List[str]:
    if source_path:
        return list(_gather_files_cached(str(source_path)))
    if storage_path:
        return list(_load_files_from_storage_cached(str(storage_path)))
    raise ValueError("Either storage_path or source_path must be provided")


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


@lru_cache(maxsize=512)
def _read_file_text(path: str, max_bytes: int = 200_000) -> str:
    try:
        data = Path(path).read_bytes()
    except Exception:
        return ""

    if len(data) > max_bytes:
        data = data[:max_bytes]
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _get_ts_parser(language_name: str):
    try:
        from tree_sitter_languages import get_parser  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        return get_parser(language_name)
    except Exception:
        return None


def _line_number_from_index(text: str, idx: int) -> int:
    return text[:idx].count("\n") + 1


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

    source_keywords = {"request", "req", "user", "input", "query", "params", "path", "filename", "filepath", "body"}
    sinks = {"open", "eval", "exec", "os.system", "subprocess.run", "subprocess.Popen", "Path", "read_text", "read_bytes"}

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
            self.visit_FunctionDef(node)

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


def tech_stack(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    output_format: str = "json",
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)

    extensions = Counter(Path(path).suffix for path in files)
    languages: Counter[str] = Counter()
    for ext, count in extensions.items():
        languages[DEFAULT_EXTENSIONS.get(ext, "Other")] += count

    framework_hits: Dict[str, int] = defaultdict(int)
    framework_files: Dict[str, List[str]] = defaultdict(list)

    _progress(progress_callback, {"stage": "analyze", "message": "Scanning for framework usage"})
    for path in files:
        text = _read_file_text(path)
        if not text:
            continue
        lowered = text.lower()
        for framework, tokens in FRAMEWORK_PATTERNS.items():
            if any(token in lowered for token in tokens):
                framework_hits[framework] += 1
                if len(framework_files[framework]) < 10:
                    framework_files[framework].append(path)

    result = {
        "languages": dict(languages),
        "extensions": dict(extensions.most_common(15)),
        "frameworks": dict(framework_hits),
        "examples": framework_files,
    }

    if output_format == "markdown":
        lines = ["# Technology Stack", "", "## Languages", json.dumps(dict(languages), indent=2)]
        lines.extend(["", "## Frameworks", json.dumps(dict(framework_hits), indent=2)])
        return {"format": "markdown", "content": "\n".join(lines)}

    return {"format": "json", "data": result}


def query_diagnostics(storage_path: str, query_text: str) -> Dict[str, Any]:
    from llama_index.core import VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core.schema import QueryBundle
    from src.indexer import _tokenize, _keyword_score
    from src.config import load_settings

    settings = load_settings()
    if settings.vector_store_provider != "chroma":
        raise ValueError("query_diagnostics requires chroma vector store")

    client = chromadb.PersistentClient(path=storage_path)
    collection = client.get_or_create_collection("architext_db")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    retriever = index.as_retriever(similarity_top_k=10)
    nodes = retriever.retrieve(QueryBundle(query_str=query_text))

    results = []
    for node in nodes:
        content = node.node.get_content()[:200]
        keyword_score = _keyword_score(query_text, node.node.get_content())
        vector_score = node.score or 0.0
        results.append(
            {
                "file": node.metadata.get("file_path", "unknown"),
                "content_preview": content,
                "vector_score": vector_score,
                "keyword_score": keyword_score,
                "hybrid_score": 0.7 * vector_score + 0.3 * keyword_score,
                "query_tokens": _tokenize(query_text),
                "matched_tokens": list(
                    set(_tokenize(query_text)) & set(_tokenize(node.node.get_content()))
                ),
            }
        )

    return {"query": query_text, "results": results}


IMPORT_PATTERNS = {
    ".js": [re.compile(r"import\s+.*?from\s+['\"](.*?)['\"]")],
    ".ts": [re.compile(r"import\s+.*?from\s+['\"](.*?)['\"]")],
    ".jsx": [re.compile(r"import\s+.*?from\s+['\"](.*?)['\"]")],
    ".tsx": [re.compile(r"import\s+.*?from\s+['\"](.*?)['\"]")],
    ".java": [re.compile(r"^\s*import\s+([\w\.]+)", re.MULTILINE)],
}


def _module_key(path: str) -> str:
    return str(Path(path).with_suffix(""))


def _module_name(path: str, root: Path) -> str:
    relative = Path(path).resolve().relative_to(root)
    if relative.name == "__init__.py":
        return ".".join(relative.parent.parts)
    return ".".join(relative.with_suffix("").parts)


def _extract_imports(path: str, content: str) -> List[str]:
    suffix = Path(path).suffix
    if suffix == ".py":
        try:
            tree = ast.parse(content)
        except Exception:
            return []

        imports: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                level = node.level or 0
                if level > 0:
                    imports.append(f".{level}:{module or ''}")
                elif module:
                    imports.append(module)
        return imports

    language_map = {
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".java": "java",
    }
    language = language_map.get(suffix)
    parser = _get_ts_parser(language) if language else None
    if parser:
        imports: List[str] = []
        tree = parser.parse(content.encode("utf-8", errors="ignore"))
        root = tree.root_node
        stack = [root]
        while stack:
            node = stack.pop()
            if node.type in {"import_statement", "import_declaration"}:
                statement = content[node.start_byte : node.end_byte]
                match = re.search(r"['\"](.*?)['\"]", statement)
                if match:
                    imports.append(match.group(1))
            for child in reversed(node.children):
                stack.append(child)
        if imports:
            return imports

    patterns = IMPORT_PATTERNS.get(suffix, [])
    imports: List[str] = []
    for pattern in patterns:
        for match in pattern.findall(content):
            if isinstance(match, tuple):
                match = match[0]
            imports.append(match)
    return imports


def _classify_import_clusters(imports: List[str]) -> List[str]:
    clusters: set[str] = set()
    for imported in imports:
        if not imported:
            continue
        if imported.startswith(".") or ":" in imported:
            continue
        normalized = imported.replace("/", ".")
        root = normalized.split(".")[0]
        for cluster, tokens in IMPORT_CLUSTER_RULES.items():
            if root in tokens or normalized in tokens:
                clusters.add(cluster)
    return sorted(clusters)


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


def _resolve_relative_import(path: str, import_path: str) -> str:
    parts = import_path.split("/")
    current = Path(path).parent
    while parts and parts[0] == ".":
        parts.pop(0)
    return str(current.joinpath(*parts))


def _resolve_python_relative_import(current_module: Optional[str], encoded: str) -> Optional[str]:
    if not current_module:
        return None
    try:
        level_text, module = encoded.split(":", 1)
        level = int(level_text.lstrip(".")) if level_text else 0
    except ValueError:
        return None

    base_parts = current_module.split(".")
    if len(base_parts) > 0:
        base_parts = base_parts[:-1]

    if level > 1:
        base_parts = base_parts[: max(len(base_parts) - (level - 1), 0)]

    if module:
        base_parts.extend(module.split("."))

    return ".".join(base_parts) if base_parts else None


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


def detect_anti_patterns(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)

    total_files = len(files)
    directories = Counter(Path(path).parent for path in files)
    avg_files_per_dir = total_files / max(len(directories), 1)

    issues = []
    large_files = []
    for path in files:
        text = _read_file_text(path)
        if not text:
            continue
        lines = text.splitlines()
        if len(lines) > 800:
            large_files.append((path, len(lines)))

        function_count = len(re.findall(r"\bdef\s+\w+\b", text)) + len(
            re.findall(r"\bfunction\s+\w+\b", text)
        )
        if function_count > 40:
            issues.append(
                {
                    "type": "god_object",
                    "file": path,
                    "severity": "high",
                    "details": f"High function count: {function_count}",
                }
            )

        imports = _extract_imports(path, text)
        clusters = _classify_import_clusters(imports)
        if len(clusters) >= 2 and (len(lines) > 200 or function_count > 20):
            issues.append(
                {
                    "type": "mixed_responsibilities",
                    "file": path,
                    "severity": "medium",
                    "details": f"Multiple import clusters detected: {', '.join(clusters)}",
                }
            )

    if large_files:
        for path, count in large_files[:10]:
            issues.append(
                {
                    "type": "large_file",
                    "file": path,
                    "severity": "medium",
                    "details": f"File has {count} lines",
                }
            )

    # Documentation and test hygiene signals
    doc_files = [path for path in files if Path(path).suffix in {".md", ".rst"}]
    test_files = [path for path in files if "test" in Path(path).name.lower()]
    doc_ratio = len(doc_files) / max(total_files, 1)
    test_ratio = len(test_files) / max(total_files, 1)

    if not test_files:
        issues.append(
            {
                "type": "missing_tests",
                "severity": "medium",
                "details": "No test files detected (filenames containing 'test')",
            }
        )
    elif test_ratio < 0.03:
        issues.append(
            {
                "type": "low_test_presence",
                "severity": "low",
                "details": f"Test files ratio is low: {round(test_ratio, 3)}",
            }
        )

    if not doc_files:
        issues.append(
            {
                "type": "missing_docs",
                "severity": "low",
                "details": "No documentation files detected (.md/.rst)",
            }
        )
    elif doc_ratio < 0.02:
        issues.append(
            {
                "type": "low_doc_presence",
                "severity": "low",
                "details": f"Documentation files ratio is low: {round(doc_ratio, 3)}",
            }
        )

    # Flat structure signal (lots of files per directory)
    if avg_files_per_dir > 30:
        issues.append(
            {
                "type": "flat_structure",
                "severity": "medium",
                "details": f"Average files per directory is high: {round(avg_files_per_dir, 2)}",
            }
        )

    # Excessive single-directory concentration
    if directories:
        max_dir, max_count = max(directories.items(), key=lambda item: item[1])
        concentration = max_count / max(total_files, 1)
        if concentration > 0.6 and total_files >= 20:
            issues.append(
                {
                    "type": "single_directory_concentration",
                    "severity": "medium",
                    "details": f"{max_count} files ({round(concentration, 2)}) in {max_dir}",
                }
            )

    # Duplicate file stems (excluding extensions)
    stems = [Path(path).stem.lower() for path in files]
    stem_counts = Counter(stems)
    duplicate_stems = [stem for stem, count in stem_counts.items() if count >= 3]
    if duplicate_stems:
        sample = duplicate_stems[:5]
        issues.append(
            {
                "type": "duplicate_file_stems",
                "severity": "low",
                "details": f"Duplicate stems found (>=3 occurrences): {', '.join(sample)}",
            }
        )

    # Missing CI configuration (GitHub Actions / GitLab / CircleCI / Azure Pipelines)
    ci_files = [
        path
        for path in files
        if any(
            token in str(path).replace("\\", "/").lower()
            for token in [
                ".github/workflows/",
                ".gitlab-ci",
                "circleci/config.yml",
                "azure-pipelines.yml",
                "azure-pipelines.yaml",
                "appveyor.yml",
            ]
        )
    ]
    if not ci_files and total_files >= 10:
        issues.append(
            {
                "type": "missing_ci_config",
                "severity": "low",
                "details": "No CI configuration detected",
            }
        )

    # Missing license file
    has_license = any(Path(path).name.lower() in {"license", "license.md", "license.txt"} for path in files)
    if not has_license and total_files >= 10:
        issues.append(
            {
                "type": "missing_license",
                "severity": "low",
                "details": "No LICENSE file detected",
            }
        )

    # Missing formatting config (Python/JS/TS/Prettier/EditorConfig)
    formatting_files = {
        ".editorconfig",
        ".prettierrc",
        ".prettierrc.json",
        ".prettierrc.yml",
        ".prettierrc.yaml",
        "prettier.config.js",
        "pyproject.toml",
        "setup.cfg",
        "tox.ini",
        "ruff.toml",
    }
    has_formatting = any(Path(path).name.lower() in formatting_files for path in files)
    if not has_formatting and total_files >= 10:
        issues.append(
            {
                "type": "missing_formatting_config",
                "severity": "low",
                "details": "No formatting configuration detected (.editorconfig/prettier/pyproject/etc)",
            }
        )

    _progress(progress_callback, {"stage": "analyze", "message": "Building dependency graph"})
    graph = _build_import_graph(files)

    coupling = sum(len(deps) for deps in graph.values()) / max(len(graph), 1)
    if coupling > 12:
        issues.append(
            {
                "type": "high_coupling",
                "severity": "medium",
                "details": f"Average dependencies per module is high: {round(coupling, 2)}",
            }
        )

    cycles = _find_cycles(graph)
    for cycle in cycles:
        issues.append(
            {
                "type": "circular_dependency",
                "severity": "high",
                "details": " -> ".join(cycle),
            }
        )

    severity_counts = Counter(issue.get("severity", "unknown") for issue in issues)

    return {
        "issues": issues,
        "counts": Counter(issue["type"] for issue in issues),
        "severity_counts": dict(severity_counts),
        "metrics": {
            "total_files": total_files,
            "doc_files": len(doc_files),
            "test_files": len(test_files),
            "doc_ratio": round(doc_ratio, 3),
            "test_ratio": round(test_ratio, 3),
            "avg_files_per_dir": round(avg_files_per_dir, 2),
            "avg_dependencies": round(coupling, 2),
        },
    }


def _count_python_docstrings(files: List[str]) -> int:
    count = 0
    for path in files:
        if Path(path).suffix != ".py":
            continue
        content = _read_file_text(path)
        if not content:
            continue
        try:
            tree = ast.parse(content)
        except Exception:
            continue

        if ast.get_docstring(tree):
            count += 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if ast.get_docstring(node):
                    count += 1
    return count


def health_score(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)
    if not files:
        return {"score": 0, "details": {}}

    total_files = len(files)
    directories = Counter(Path(path).parent for path in files)
    avg_files_per_dir = total_files / max(len(directories), 1)

    graph = _build_import_graph(files)
    coupling = sum(len(deps) for deps in graph.values()) / max(len(graph), 1)

    doc_files = [path for path in files if Path(path).suffix in {".md", ".rst"}]
    doc_coverage = min(len(doc_files) / max(total_files, 1), 1.0)
    docstring_count = _count_python_docstrings(files)
    py_files = [path for path in files if Path(path).suffix == ".py"]
    docstring_coverage = min(docstring_count / max(len(py_files), 1), 1.0)

    test_files = [path for path in files if "test" in Path(path).name.lower()]
    test_coverage = min(len(test_files) / max(total_files, 1), 1.0)

    modularity_score = max(0, 100 - avg_files_per_dir * 2)
    coupling_score = max(0, 100 - coupling * 5)
    documentation_score = (doc_coverage * 0.6 + docstring_coverage * 0.4) * 100
    testing_score = test_coverage * 100

    overall = (modularity_score + coupling_score + documentation_score + testing_score) / 4

    return {
        "score": round(overall, 2),
        "details": {
            "modularity": round(modularity_score, 2),
            "coupling": round(coupling_score, 2),
            "documentation": round(documentation_score, 2),
            "testing": round(testing_score, 2),
            "avg_files_per_dir": round(avg_files_per_dir, 2),
            "avg_dependencies": round(coupling, 2),
            "doc_files": len(doc_files),
            "docstrings": docstring_count,
        },
    }


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


def refactoring_recommendations(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    anti_patterns = detect_anti_patterns(storage_path, source_path)
    score = health_score(storage_path, source_path)

    recommendations = []
    for issue in anti_patterns.get("issues", []):
        if issue["type"] == "god_object":
            recommendations.append(
                {
                    "title": "Split large modules",
                    "file": issue.get("file"),
                    "effort": "medium",
                    "benefit": "high",
                    "rationale": issue.get("details"),
                }
            )
        if issue["type"] == "circular_dependency":
            recommendations.append(
                {
                    "title": "Break circular dependencies",
                    "effort": "high",
                    "benefit": "high",
                    "rationale": issue.get("details"),
                }
            )

    if score.get("details", {}).get("documentation", 100) < 40:
        recommendations.append(
            {
                "title": "Improve documentation coverage",
                "effort": "low",
                "benefit": "medium",
                "rationale": "Documentation coverage below 40%",
            }
        )

    if score.get("details", {}).get("testing", 100) < 30:
        recommendations.append(
            {
                "title": "Increase test coverage",
                "effort": "medium",
                "benefit": "high",
                "rationale": "Test coverage below 30%",
            }
        )

    return {"recommendations": recommendations, "health": score}


def generate_docs(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "analyze", "message": "Compiling documentation"})
    structure = analyze_structure(storage_path, source_path, output_format="markdown")
    tech = tech_stack(storage_path, source_path, output_format="markdown")
    health = health_score(storage_path, source_path)
    anti = detect_anti_patterns(storage_path, source_path)

    docs = {
        "structure.md": structure.get("content", ""),
        "tech-stack.md": tech.get("content", ""),
        "health.json": json.dumps(health, indent=2),
        "anti-patterns.json": json.dumps(anti, indent=2),
    }

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for name, content in docs.items():
            (out_path / name).write_text(content, encoding="utf-8")

    return {"outputs": list(docs.keys()), "output_dir": output_dir, "inline": docs}


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


def test_coverage_analysis(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)
    source_files = [path for path in files if Path(path).suffix in DEFAULT_EXTENSIONS]
    test_files = [path for path in files if "test" in Path(path).name.lower()]

    mapping: Dict[str, List[str]] = defaultdict(list)
    for src in source_files:
        stem = Path(src).stem
        for test in test_files:
            if stem in Path(test).stem:
                mapping[src].append(test)

    uncovered = [src for src in source_files if src not in mapping]
    coverage_ratio = 1 - (len(uncovered) / max(len(source_files), 1))

    return {
        "total_sources": len(source_files),
        "total_tests": len(test_files),
        "coverage_ratio": round(coverage_ratio, 2),
        "uncovered": uncovered[:50],
        "mapping": {k: v[:5] for k, v in mapping.items()},
    }


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
    baseline_files = set(baseline_files or [])

    added = sorted(current_files - baseline_files)
    removed = sorted(baseline_files - current_files)

    return {"added": added[:100], "removed": removed[:100], "added_count": len(added), "removed_count": len(removed)}


def onboarding_guide(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)
    root_files = [Path(path).name for path in files if len(Path(path).parts) <= 3]

    entry_points = [f for f in root_files if f.lower() in {"readme.md", "setup.py", "pyproject.toml", "package.json", "main.py"}]
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
                self.visit_FunctionDef(node)

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


def synthesis_roadmap(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "analyze", "message": "Gathering structural signals"})
    anti_patterns = detect_anti_patterns(storage_path, source_path)
    health = health_score(storage_path, source_path)
    silent = identify_silent_failures(storage_path, source_path)
    logic_gaps = logic_gap_analysis(storage_path, source_path)
    heuristics = security_heuristics(storage_path, source_path)
    duplication = detect_duplicate_blocks(storage_path, source_path)
    semantic_duplication = detect_duplicate_blocks_semantic(storage_path, source_path)

    opportunities: List[Dict[str, Any]] = []

    if anti_patterns.get("issues"):
        opportunities.append(
            {
                "title": "Address architectural anti-patterns",
                "priority": "high",
                "score": 0.9,
                "evidence": anti_patterns.get("issues", [])[:5],
            }
        )

    if health.get("score", 100) < 60:
        opportunities.append(
            {
                "title": "Improve architecture health score",
                "priority": "high",
                "score": 0.85,
                "evidence": health.get("details", {}),
            }
        )

    if heuristics.get("counts", {}).get("total", 0) > 0:
        opportunities.append(
            {
                "title": "Resolve security heuristic findings",
                "priority": "high",
                "score": 0.95,
                "evidence": heuristics.get("findings", [])[:5],
            }
        )

    if silent.get("count", 0) > 0:
        opportunities.append(
            {
                "title": "Eliminate silent exception handling",
                "priority": "medium",
                "score": 0.7,
                "evidence": silent.get("findings", [])[:5],
            }
        )

    if logic_gaps.get("unused_settings"):
        opportunities.append(
            {
                "title": "Resolve unused configuration settings",
                "priority": "medium",
                "score": 0.6,
                "evidence": logic_gaps.get("unused_settings", [])[:5],
            }
        )

    if duplication.get("count", 0) > 0:
        opportunities.append(
            {
                "title": "Reduce duplicated code blocks",
                "priority": "medium",
                "score": 0.65,
                "evidence": duplication.get("findings", [])[:5],
            }
        )

    if semantic_duplication.get("count", 0) > 0:
        opportunities.append(
            {
                "title": "Consolidate semantically duplicated functions",
                "priority": "medium",
                "score": 0.68,
                "evidence": semantic_duplication.get("findings", [])[:5],
            }
        )

    opportunities.sort(key=lambda item: item.get("score", 0), reverse=True)

    return {
        "summary": {
            "health_score": health.get("score"),
            "anti_pattern_count": len(anti_patterns.get("issues", [])),
            "security_findings": heuristics.get("counts", {}).get("total", 0),
            "silent_failures": silent.get("count", 0),
            "unused_settings": len(logic_gaps.get("unused_settings", [])),
            "duplication_findings": duplication.get("count", 0),
            "semantic_duplication_findings": semantic_duplication.get("count", 0),
        },
        "roadmap": opportunities,
    }


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
            from src.cli_utils import extract_sources

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


def logic_gap_analysis(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)

    config_file = None
    settings_fields: List[Tuple[str, int]] = []
    is_pydantic_settings = False
    for path in files:
        if Path(path).name.lower() != "config.py":
            continue
        content = _read_file_text(path)
        if "ArchitextSettings" not in content:
            continue
        try:
            tree = ast.parse(content)
        except Exception:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "ArchitextSettings":
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == "BaseSettings":
                        is_pydantic_settings = True
                    elif isinstance(base, ast.Attribute) and base.attr == "BaseSettings":
                        is_pydantic_settings = True
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        settings_fields.append((item.target.id, item.lineno))
                    elif isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                settings_fields.append((target.id, item.lineno))
        if settings_fields:
            config_file = path
            break

    if not settings_fields:
        return {
            "note": "No ArchitextSettings fields found",
            "config_file": config_file,
            "unused_settings": [],
        }

    used = set()
    candidate_files = [path for path in files if Path(path).suffix == ".py" and path != config_file]
    for path in candidate_files:
        content = _read_file_text(path)
        if not content:
            continue
        for field, _lineno in settings_fields:
            if field in used:
                continue
            if re.search(rf"\b{re.escape(field)}\b", content):
                used.add(field)

    ignored_settings = set()
    if is_pydantic_settings:
        ignored_settings.update(FRAMEWORK_SETTINGS_IGNORES.get("pydantic", set()))

    unused = [
        {"name": field, "defined_at": lineno}
        for field, lineno in settings_fields
        if field not in used and field not in ignored_settings
    ]

    return {
        "config_file": config_file,
        "settings_defined": len(settings_fields),
        "settings_used": len(used),
        "unused_settings": unused,
        "ignored_settings": sorted(ignored_settings),
    }


def identify_silent_failures(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)
    findings: List[Dict[str, Any]] = []

    for path in files:
        suffix = Path(path).suffix.lower()
        content = _read_file_text(path)
        if not content:
            continue

        if suffix == ".py":
            lines = content.splitlines()
            for idx, line in enumerate(lines):
                if not re.match(r"\s*except\b", line):
                    continue
                if not line.rstrip().endswith(":"):
                    continue
                base_indent = len(line) - len(line.lstrip())
                cursor = idx + 1
                while cursor < len(lines):
                    candidate = lines[cursor]
                    if not candidate.strip() or candidate.lstrip().startswith("#"):
                        cursor += 1
                        continue
                    indent = len(candidate) - len(candidate.lstrip())
                    if indent <= base_indent:
                        break
                    stripped = candidate.strip()
                    if stripped in {"pass", "continue", "return", "..."}:
                        findings.append(
                            {
                                "file": path,
                                "line": cursor + 1,
                                "severity": "medium",
                                "type": "silent_exception",
                                "snippet": stripped,
                            }
                        )
                    break
        elif suffix in {".js", ".ts", ".jsx", ".tsx"}:
            for match in re.finditer(
                r"catch\s*\([^\)]*\)\s*\{\s*(?:\/\*.*?\*\/\s*|\/\/.*?\n\s*)*\}",
                content,
                re.IGNORECASE | re.DOTALL,
            ):
                line = _line_number_from_index(content, match.start())
                findings.append(
                    {
                        "file": path,
                        "line": line,
                        "severity": "medium",
                        "type": "silent_exception",
                        "snippet": "empty catch block",
                    }
                )

    return {
        "findings": findings,
        "count": len(findings),
    }


def _normalize_duplication_line(line: str, suffix: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if suffix == ".py" and stripped.startswith("#"):
        return ""
    if suffix in {".js", ".ts", ".jsx", ".tsx"} and stripped.startswith("//"):
        return ""
    return stripped


def _normalize_python_tokens(segment: str) -> str:
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


def detect_duplicate_blocks_semantic(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    min_tokens: int = 40,
    max_findings: int = 50,
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)

    fingerprints: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    scanned_files = 0

    _progress(progress_callback, {"stage": "analyze", "message": "Scanning for semantic duplication"})
    for path in files:
        if Path(path).suffix.lower() != ".py":
            continue
        content = _read_file_text(path)
        if not content:
            continue
        try:
            tree = ast.parse(content)
        except Exception:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            segment = ast.get_source_segment(content, node) or ""
            normalized = _normalize_python_tokens(segment)
            if not normalized:
                continue
            token_count = len(normalized.split())
            if token_count < min_tokens:
                continue
            digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
            fingerprints[digest].append(
                {
                    "file": path,
                    "name": getattr(node, "name", "<anonymous>"),
                    "line": getattr(node, "lineno", 0),
                    "tokens": token_count,
                }
            )
        scanned_files += 1

    findings: List[Dict[str, Any]] = []
    for digest, occ in fingerprints.items():
        if len(occ) < 2:
            continue
        findings.append(
            {
                "fingerprint": digest,
                "occurrence_count": len(occ),
                "occurrences": occ[:10],
            }
        )

    findings.sort(key=lambda item: item.get("occurrence_count", 0), reverse=True)

    return {
        "count": len(findings),
        "scanned_files": scanned_files,
        "min_tokens": min_tokens,
        "findings": findings[:max_findings],
    }


def detect_duplicate_blocks(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    min_lines: int = 8,
    max_findings: int = 50,
    max_windows_per_file: int = 6000,
    progress_callback=None,
) -> Dict[str, Any]:
    _progress(progress_callback, {"stage": "scan", "message": "Collecting files"})
    files = collect_file_paths(storage_path, source_path)

    duplicates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    scanned_files = 0

    _progress(progress_callback, {"stage": "analyze", "message": "Scanning for duplicate blocks"})
    for path in files:
        suffix = Path(path).suffix.lower()
        if suffix not in {".py", ".js", ".ts", ".jsx", ".tsx"}:
            continue
        content = _read_file_text(path)
        if not content:
            continue
        raw_lines = content.splitlines()
        if len(raw_lines) < min_lines:
            continue

        normalized = [_normalize_duplication_line(line, suffix) for line in raw_lines]
        window_limit = min(max_windows_per_file, max(len(raw_lines) - min_lines + 1, 0))
        windows_seen = 0
        for idx in range(0, len(raw_lines) - min_lines + 1):
            if windows_seen >= window_limit:
                break
            window = normalized[idx : idx + min_lines]
            if not any(window):
                continue
            key = "\n".join(window)
            if len(key) < 20:
                continue
            duplicates[key].append(
                {
                    "file": path,
                    "start_line": idx + 1,
                    "end_line": idx + min_lines,
                }
            )
            windows_seen += 1
        scanned_files += 1

    findings: List[Dict[str, Any]] = []
    for key, occ in duplicates.items():
        if len(occ) < 2:
            continue
        findings.append(
            {
                "line_count": min_lines,
                "occurrences": occ[:10],
                "occurrence_count": len(occ),
                "snippet": "\n".join(key.splitlines()[:min_lines])[:500],
            }
        )

    findings.sort(key=lambda item: item.get("occurrence_count", 0), reverse=True)

    return {
        "count": len(findings),
        "scanned_files": scanned_files,
        "min_lines": min_lines,
        "findings": findings[:max_findings],
    }
