"""Phase 2.5 analysis tasks for Architext."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import chromadb


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


def _should_skip(path: Path) -> bool:
    skip_fragments = {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        ".env",
        "dist",
        "build",
        ".pytest_cache",
        "models_cache",
        "storage",
    }
    for part in path.parts:
        lower = part.lower()
        if lower == "storage" or lower.startswith("storage-") or lower.startswith("storage_"):
            return True
    return any(fragment in path.parts for fragment in skip_fragments) or path.name.startswith(".")


def _gather_files(source_path: str) -> List[str]:
    root = Path(source_path)
    files: List[str] = []
    for file_path in root.rglob("*"):
        if file_path.is_dir():
            continue
        if _should_skip(file_path):
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

    summary = {
        "total_files": len(files),
        "total_extensions": len(extensions),
        "languages": dict(languages),
        "top_extensions": dict(extensions.most_common(10)),
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
    ".py": [
        re.compile(r"^\s*import\s+([\w\.]+)", re.MULTILINE),
        re.compile(r"^\s*from\s+([\w\.]+)\s+import", re.MULTILINE),
    ],
    ".js": [re.compile(r"import\s+.*?from\s+['\"](.*?)['\"]")],
    ".ts": [re.compile(r"import\s+.*?from\s+['\"](.*?)['\"]")],
    ".jsx": [re.compile(r"import\s+.*?from\s+['\"](.*?)['\"]")],
    ".tsx": [re.compile(r"import\s+.*?from\s+['\"](.*?)['\"]")],
    ".java": [re.compile(r"^\s*import\s+([\w\.]+)", re.MULTILINE)],
}


def _module_key(path: str) -> str:
    return str(Path(path).with_suffix(""))


def _extract_imports(path: str, content: str) -> List[str]:
    suffix = Path(path).suffix
    patterns = IMPORT_PATTERNS.get(suffix, [])
    imports: List[str] = []
    for pattern in patterns:
        for match in pattern.findall(content):
            if isinstance(match, tuple):
                match = match[0]
            imports.append(match)
    return imports


def _build_import_graph(files: List[str]) -> Dict[str, List[str]]:
    graph: Dict[str, List[str]] = defaultdict(list)
    file_lookup = {_module_key(path): path for path in files}
    for path in files:
        content = _read_file_text(path)
        if not content:
            continue
        module = _module_key(path)
        for imported in _extract_imports(path, content):
            if imported.startswith("."):
                resolved = _resolve_relative_import(path, imported)
                if resolved in file_lookup:
                    graph[module].append(resolved)
            else:
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


def _find_cycles(graph: Dict[str, List[str]], limit: int = 10) -> List[List[str]]:
    cycles: List[List[str]] = []
    path_stack: List[str] = []
    visited: Dict[str, str] = {}

    def dfs(node: str):
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

    test_files = [path for path in files if "test" in Path(path).name.lower()]
    test_coverage = min(len(test_files) / max(total_files, 1), 1.0)

    modularity_score = max(0, 100 - avg_files_per_dir * 2)
    coupling_score = max(0, 100 - coupling * 5)
    documentation_score = doc_coverage * 100
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
