"""Documentation generation tasks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.tasks.anti_patterns import detect_anti_patterns
from src.tasks.health import health_score
from src.tasks.structure import analyze_structure
from src.tasks.tech_stack import tech_stack
from src.tasks.shared import _progress


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
