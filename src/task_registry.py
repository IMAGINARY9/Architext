"""Central registry for analysis tasks."""
from __future__ import annotations

import inspect
from typing import Any, Callable, Dict

from src.tasks import (
    analyze_structure,
    tech_stack,
    detect_anti_patterns,
    health_score,
    impact_analysis,
    dependency_graph_export,
    test_mapping_analysis,
    architecture_pattern_detection,
    detect_vulnerabilities,
    identify_silent_failures,
    security_heuristics,
    code_knowledge_graph,
    synthesis_roadmap,
    detect_duplicate_blocks,
    detect_duplicate_blocks_semantic,
)


TASK_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "analyze-structure": analyze_structure,
    "tech-stack": tech_stack,
    "detect-anti-patterns": detect_anti_patterns,
    "health-score": health_score,
    "impact-analysis": impact_analysis,
    "dependency-graph": dependency_graph_export,
    "test-mapping": test_mapping_analysis,
    "detect-patterns": architecture_pattern_detection,
    "detect-vulnerabilities": detect_vulnerabilities,
    "identify-silent-failures": identify_silent_failures,
    "security-heuristics": security_heuristics,
    "code-knowledge-graph": code_knowledge_graph,
    "synthesis-roadmap": synthesis_roadmap,
    "detect-duplication": detect_duplicate_blocks,
    "detect-duplication-semantic": detect_duplicate_blocks_semantic,
}


def list_task_names() -> list[str]:
    return sorted(TASK_REGISTRY.keys())


def get_task_handler(task_name: str) -> Callable[..., Dict[str, Any]]:
    try:
        return TASK_REGISTRY[task_name]
    except KeyError as exc:
        raise ValueError(f"Unknown task: {task_name}") from exc


def run_task(task_name: str, **kwargs: Any) -> Dict[str, Any]:
    handler = get_task_handler(task_name)
    signature = inspect.signature(handler)
    filtered = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters and value is not None
    }
    return handler(**filtered)
