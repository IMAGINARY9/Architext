"""Tech stack detection task.

Detects technologies, frameworks, and languages in use.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional

from src.tasks.core.base import BaseTask, FileInfo


# Framework detection patterns
FRAMEWORK_SIGNATURES = {
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

# Language indicators by file extension
LANGUAGE_INDICATORS = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".jsx": "React/JSX",
    ".tsx": "React/TSX",
    ".java": "Java",
    ".rb": "Ruby",
    ".go": "Go",
    ".rs": "Rust",
    ".cs": "C#",
    ".cpp": "C++",
    ".c": "C",
    ".php": "PHP",
    ".swift": "Swift",
    ".kt": "Kotlin",
}


class TechStackTask(BaseTask):
    """
    Detect technologies, frameworks, and languages in use.
    
    Scans for framework patterns in imports and code.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        source_path: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        output_format: str = "json",
    ):
        super().__init__(
            storage_path=storage_path,
            source_path=source_path,
            progress_callback=progress_callback,
            extensions=None,
            load_content=True,
        )
        self.output_format = output_format
    
    def analyze(self, files: List[FileInfo]) -> Dict[str, Any]:
        """Detect tech stack from files."""
        self._report_progress("analyze", "Scanning for framework usage")
        
        extensions = Counter(f.extension for f in files)
        languages: Counter[str] = Counter()
        for f in files:
            languages[f.language] += 1
        
        framework_hits: Dict[str, int] = defaultdict(int)
        framework_files: Dict[str, List[str]] = defaultdict(list)
        
        for f in files:
            if not f.content:
                continue
            lowered = f.content.lower()
            for framework, tokens in FRAMEWORK_SIGNATURES.items():
                if any(token in lowered for token in tokens):
                    framework_hits[framework] += 1
                    if len(framework_files[framework]) < 10:
                        framework_files[framework].append(f.path)
        
        result = {
            "languages": dict(languages),
            "extensions": dict(extensions.most_common(15)),
            "frameworks": dict(framework_hits),
            "examples": framework_files,
        }
        
        if self.output_format == "markdown":
            lines = [
                "# Technology Stack", "",
                "## Languages", json.dumps(dict(languages), indent=2), "",
                "## Frameworks", json.dumps(dict(framework_hits), indent=2)
            ]
            return {"format": "markdown", "content": "\n".join(lines)}
        
        return {"format": "json", "data": result}


def tech_stack_v2(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    output_format: str = "json",
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Detect tech stack using BaseTask pattern."""
    task = TechStackTask(
        storage_path=storage_path,
        source_path=source_path,
        progress_callback=progress_callback,
        output_format=output_format,
    )
    return task.run()


# Alias
tech_stack_detection_v2 = tech_stack_v2


__all__ = [
    "TechStackTask",
    "tech_stack_v2",
    "tech_stack_detection_v2",
    "FRAMEWORK_SIGNATURES",
    "LANGUAGE_INDICATORS",
]
