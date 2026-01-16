"""Synthesis roadmap tasks."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.tasks.anti_patterns import detect_anti_patterns
from src.tasks.health import health_score
from src.tasks.quality import identify_silent_failures, logic_gap_analysis
from src.tasks.security import security_heuristics
from src.tasks.duplication import detect_duplicate_blocks, detect_duplicate_blocks_semantic
from src.tasks.shared import _progress


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
