"""Analysis tasks package."""
from __future__ import annotations

from src.tasks.anti_patterns import detect_anti_patterns
from src.tasks.architecture import (
	architecture_pattern_detection,
	code_knowledge_graph,
	dependency_graph_export,
	impact_analysis,
)
from src.tasks.duplication import detect_duplicate_blocks, detect_duplicate_blocks_semantic
from src.tasks.health import health_score
from src.tasks.quality import (
	identify_silent_failures,
	test_mapping_analysis,
)
from src.tasks.query import query_diagnostics
from src.tasks.roadmap import synthesis_roadmap
from src.tasks.security import detect_vulnerabilities, security_heuristics
from src.tasks.structure import analyze_structure
from src.tasks.tech_stack import tech_stack

__all__ = [
	"analyze_structure",
	"tech_stack",
	"detect_anti_patterns",
	"health_score",
	"impact_analysis",
	"dependency_graph_export",
	"test_mapping_analysis",
	"architecture_pattern_detection",
	"detect_vulnerabilities",
	"identify_silent_failures",
	"security_heuristics",
	"code_knowledge_graph",
	"synthesis_roadmap",
	"detect_duplicate_blocks",
	"detect_duplicate_blocks_semantic",
	"query_diagnostics",
]
