"""Synthesis roadmap tasks."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.tasks.anti_patterns import detect_anti_patterns
from src.tasks.health import health_score
from src.tasks.quality import identify_silent_failures
from src.tasks.security import security_heuristics
from src.tasks.duplication import detect_duplicate_blocks, detect_duplicate_blocks_semantic
from src.tasks.shared import _progress, task_context


def synthesis_roadmap(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Generate a prioritized improvement roadmap by aggregating findings from multiple analysis tasks.
    
    Combines: anti-patterns, health score, security heuristics, silent failures, and duplication.
    
    Uses a shared TaskContext to cache file collection across all sub-tasks,
    significantly improving performance for large codebases.
    """
    # Use shared context to cache file collection across all sub-tasks
    with task_context(storage_path=storage_path, source_path=source_path):
        _progress(progress_callback, {"stage": "analyze", "message": "Gathering structural signals"})
        
        # All these tasks will share the cached file list from the context
        anti_patterns = detect_anti_patterns(storage_path, source_path)
        health = health_score(storage_path, source_path)
        silent = identify_silent_failures(storage_path, source_path)
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
            "duplication_findings": duplication.get("count", 0),
            "semantic_duplication_findings": semantic_duplication.get("count", 0),
        },
        "roadmap": opportunities,
    }
