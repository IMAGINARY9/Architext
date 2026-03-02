"""Shared helpers for analysis tasks."""
from __future__ import annotations

import ast
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# `chromadb` is an optional runtime dependency used by many tasks.  In
# test environments it may not be installed; wrap the import so failing
# imports do not break module importation.  Functions that actually need
# chromadb will raise an error at use time, and tests can mock them.
try:
    import chromadb
except ImportError:  # pragma: no cover - fallbacks for lightweight testing
    class _DummyChroma:
        class PersistentClient:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("chromadb is required to use the persistent client")
    chromadb: Any = _DummyChroma()

from src.file_filters import should_skip_path

# Type alias for progress callback used across all task functions.
ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


# === Task Execution Context ===

@dataclass
class TaskContext:
    """
    Shared context for task execution to avoid redundant file scanning and parsing.
    
    This context caches:
    - File paths collected from storage or source
    - File contents already read
    - Parsed ASTs for Python files
    - Import graphs
    
    Usage:
        ctx = TaskContext(source_path="src")
        files = ctx.get_files()  # Cached after first call
        content = ctx.get_file_content(path)  # Cached per file
    """
    storage_path: Optional[str] = None
    source_path: Optional[str] = None

    # Cached data
    _files: Optional[List[str]] = field(default=None, repr=False)
    _file_contents: Dict[str, str] = field(default_factory=dict, repr=False)
    _parsed_asts: Dict[str, ast.AST] = field(default_factory=dict, repr=False)
    _import_graph: Optional[Dict[str, List[str]]] = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get_files(self) -> List[str]:
        """Get all file paths, with caching."""
        if self._files is None:
            with self._lock:
                if self._files is None:
                    self._files = collect_file_paths(self.storage_path, self.source_path)
        return self._files

    def get_file_content(self, path: str) -> str:
        """Get file content, with caching."""
        if path not in self._file_contents:
            self._file_contents[path] = _read_file_text(path)
        return self._file_contents[path]

    def get_parsed_ast(self, path: str) -> Optional[ast.AST]:
        """Get parsed Python AST, with caching."""
        if path not in self._parsed_asts:
            content = self.get_file_content(path)
            if content and path.endswith(".py"):
                try:
                    self._parsed_asts[path] = ast.parse(content)
                except Exception:
                    self._parsed_asts[path] = None  # type: ignore
        return self._parsed_asts.get(path)

    def get_import_graph(self) -> Dict[str, List[str]]:
        """Get import graph, with caching."""
        if self._import_graph is None:
            with self._lock:
                if self._import_graph is None:
                    from src.tasks.graph import _build_import_graph
                    self._import_graph = _build_import_graph(self.get_files())
        return self._import_graph

    def clear_cache(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._files = None
            self._file_contents.clear()
            self._parsed_asts.clear()
            self._import_graph = None


# Thread-local storage for the current context
_current_context: threading.local = threading.local()


def get_current_context() -> Optional[TaskContext]:
    """Get the current task context if one is active."""
    return getattr(_current_context, 'context', None)


def set_current_context(ctx: Optional[TaskContext]) -> None:
    """Set the current task context."""
    _current_context.context = ctx


class task_context:
    """
    Context manager for setting a shared TaskContext during task execution.
    
    Usage:
        with task_context(source_path="src") as ctx:
            result1 = task_a(source_path="src")  # Uses shared context
            result2 = task_b(source_path="src")  # Reuses cached files
    """
    def __init__(self, storage_path: Optional[str] = None, source_path: Optional[str] = None):
        self.ctx = TaskContext(storage_path=storage_path, source_path=source_path)
        self._previous: Optional[TaskContext] = None

    def __enter__(self) -> TaskContext:
        self._previous = get_current_context()
        set_current_context(self.ctx)
        return self.ctx

    def __exit__(self, *args) -> None:
        set_current_context(self._previous)


# === Constants ===

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


IMPORT_PATTERNS = {
    ".js": [re.compile(r"import\s+.*?from\s+['\"](.*?)['\"]")],
    ".ts": [re.compile(r"import\s+.*?from\s+['\"](.*?)['\"]")],
    ".jsx": [re.compile(r"import\s+.*?from\s+['\"](.*?)['\"]")],
    ".tsx": [re.compile(r"import\s+.*?from\s+['\"](.*?)['\"]")],
    ".java": [re.compile(r"^\s*import\s+([\w\.]+)", re.MULTILINE)],
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


# --- Constants shared between v1 (top-level) and v2 (analysis/) task files ---

FRAMEWORK_PATTERNS: Dict[str, List[str]] = {
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

PATTERN_RULES: Dict[str, Dict[str, Any]] = {
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

TEST_PATTERNS: List[str] = [
    r"^test_(.+)$",      # test_module.py
    r"^(.+)_test$",      # module_test.py
    r"^tests?$",         # test.py or tests.py (generic)
    r"^(.+)\.test$",     # module.test.js
    r"^(.+)\.spec$",     # module.spec.ts
]

SECURITY_RULES: List[Dict[str, Any]] = [
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

SEMANTIC_VULNERABILITY_QUERIES: List[Dict[str, str]] = [
    {
        "id": "unvalidated-file-io",
        "prompt": "Where is user input passed into file operations without validation?",
    },
    {
        "id": "path-traversal",
        "prompt": "Find any code that constructs file paths from user input without sanitizing "
                  "traversal like ..",
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


def _progress(progress_callback: ProgressCallback, payload: Dict[str, Any]) -> None:
    """Invoke the progress callback if provided."""
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
        from src.config import load_settings as _load_settings_func
        settings = _load_settings_func()
        if settings.vector_store_provider != "chroma":
            raise ValueError("storage-based tasks require chroma vector store; use --source instead")
    except Exception:
        pass
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
    return sorted(set(file_paths))  # type: ignore[arg-type]


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
    """
    Collect file paths from storage or source directory.
    
    If a TaskContext is active and matches the parameters, uses the cached file list.
    """
    # Check if we have an active context with matching parameters
    ctx = get_current_context()
    if ctx is not None:
        if (ctx.source_path == source_path and ctx.storage_path == storage_path):
            return ctx.get_files()

    # Fall back to cached functions
    if source_path:
        return list(_gather_files_cached(str(source_path)))
    if storage_path:
        return list(_load_files_from_storage_cached(str(storage_path)))
    raise ValueError("Either storage_path or source_path must be provided")


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
        from tree_sitter_languages import get_parser
    except Exception:
        return None
    try:
        return get_parser(language_name)
    except Exception:
        return None


def _line_number_from_index(text: str, idx: int) -> int:
    return text[:idx].count("\n") + 1


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
        ts_imports: List[str] = []
        tree = parser.parse(content.encode("utf-8", errors="ignore"))
        root = tree.root_node
        stack = [root]
        while stack:
            node = stack.pop()
            if node.type in {"import_statement", "import_declaration"}:
                statement = content[node.start_byte : node.end_byte]
                match = re.search(r"['\"](.*?)['\"]", statement)
                if match:
                    ts_imports.append(match.group(1))
            for child in reversed(node.children):
                stack.append(child)
        if ts_imports:
            return ts_imports

    patterns = IMPORT_PATTERNS.get(suffix, [])
    fallback_imports: List[str] = []
    for pattern in patterns:
        for match in pattern.findall(content):
            if isinstance(match, tuple):
                match = match[0]
            fallback_imports.append(match)
    return fallback_imports


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
