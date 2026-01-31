"""Analysis task implementations.

This subpackage contains all analysis task classes using the BaseTask pattern.
Tasks are organized by domain:
- anti_patterns: Code smell and anti-pattern detection
- architecture: Pattern detection, dependency graphs, impact analysis
- duplication: Exact and semantic duplication detection  
- health: Codebase health scoring
- quality: Silent failures, test mapping
- security: Security heuristics and vulnerability scanning
- structure: Repository structure analysis
- tech_stack: Technology stack detection
"""
from __future__ import annotations

# Anti-patterns
from src.tasks.analysis.anti_patterns import (
    AntiPatternDetectionTask,
    detect_anti_patterns_v2,
)

# Architecture
from src.tasks.analysis.architecture import (
    ArchitecturePatternTask,
    ImpactAnalysisTask,
    DependencyGraphTask,
    architecture_pattern_detection_v2,
    impact_analysis_v2,
    dependency_graph_export_v2,
    PATTERN_RULES,
)

# Duplication
from src.tasks.analysis.duplication import (
    DuplicateBlocksTask,
    SemanticDuplicationTask,
    detect_duplicate_blocks_v2,
    duplicate_blocks_detection_v2,
    detect_duplicate_blocks_semantic_v2,
    semantic_duplication_detection_v2,
)

# Health
from src.tasks.analysis.health import (
    HealthScoreTask,
    health_score_v2,
    DEFAULT_WEIGHTS,
)

# Quality
from src.tasks.analysis.quality import (
    SilentFailuresTask,
    TestMappingTask,
    identify_silent_failures_v2,
    silent_failures_detection_v2,
    test_mapping_analysis_v2,
    test_mapping_v2,
    TEST_PATTERNS,
    SKIP_STEMS,
)

# Security
from src.tasks.analysis.security import (
    SecurityHeuristicsTask,
    security_heuristics_v2,
    SECURITY_PATTERNS,
)

# Structure
from src.tasks.analysis.structure import (
    StructureAnalysisTask,
    analyze_structure_v2,
    structure_analysis_v2,
)

# Tech Stack
from src.tasks.analysis.tech_stack import (
    TechStackTask,
    tech_stack_v2,
    tech_stack_detection_v2,
    FRAMEWORK_SIGNATURES,
    LANGUAGE_INDICATORS,
)

__all__ = [
    # Anti-patterns
    "AntiPatternDetectionTask",
    "detect_anti_patterns_v2",
    # Architecture
    "ArchitecturePatternTask",
    "ImpactAnalysisTask", 
    "DependencyGraphTask",
    "architecture_pattern_detection_v2",
    "impact_analysis_v2",
    "dependency_graph_export_v2",
    "PATTERN_RULES",
    # Duplication
    "DuplicateBlocksTask",
    "SemanticDuplicationTask",
    "detect_duplicate_blocks_v2",
    "duplicate_blocks_detection_v2",
    "detect_duplicate_blocks_semantic_v2",
    "semantic_duplication_detection_v2",
    # Health
    "HealthScoreTask",
    "health_score_v2",
    "DEFAULT_WEIGHTS",
    # Quality
    "SilentFailuresTask",
    "TestMappingTask",
    "identify_silent_failures_v2",
    "silent_failures_detection_v2",
    "test_mapping_analysis_v2",
    "test_mapping_v2",
    "TEST_PATTERNS",
    "SKIP_STEMS",
    # Security
    "SecurityHeuristicsTask",
    "security_heuristics_v2",
    "SECURITY_PATTERNS",
    # Structure
    "StructureAnalysisTask",
    "analyze_structure_v2",
    "structure_analysis_v2",
    # Tech Stack
    "TechStackTask",
    "tech_stack_v2",
    "tech_stack_detection_v2",
    "FRAMEWORK_SIGNATURES",
    "LANGUAGE_INDICATORS",
]
