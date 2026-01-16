"""Shared helpers for analysis tasks."""
from __future__ import annotations

import ast
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
