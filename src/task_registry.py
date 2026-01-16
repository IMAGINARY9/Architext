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
    refactoring_recommendations,
    generate_docs,
    dependency_graph_export,
    test_coverage_analysis,
    architecture_pattern_detection,
    diff_architecture_review,
    onboarding_guide,
    detect_vulnerabilities,
    logic_gap_analysis,
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
    "refactoring-recommendations": refactoring_recommendations,
    "generate-docs": generate_docs,
    "dependency-graph": dependency_graph_export,
    "test-coverage": test_coverage_analysis,
    "detect-patterns": architecture_pattern_detection,
    "diff-architecture": diff_architecture_review,
    "onboarding-guide": onboarding_guide,
    "detect-vulnerabilities": detect_vulnerabilities,
    "logic-gap-analysis": logic_gap_analysis,
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
