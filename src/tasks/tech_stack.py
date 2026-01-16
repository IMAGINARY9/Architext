"""Technology stack analysis task."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tasks.shared import DEFAULT_EXTENSIONS, _progress, _read_file_text, collect_file_paths


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
