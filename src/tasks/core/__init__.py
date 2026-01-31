"""Core infrastructure for task execution.

This subpackage contains:
- BaseTask: Abstract base class for task implementations
- TaskContext: Shared context with caching for multi-task execution
- TaskResultCache: Persistent disk caching with TTL
- Type definitions for task results
"""
from __future__ import annotations

from src.tasks.core.base import (
    BaseTask,
    FileInfo,
    TaskResult,
    PYTHON_EXTENSIONS,
    JAVASCRIPT_EXTENSIONS,
    TYPESCRIPT_EXTENSIONS,
    JS_TS_EXTENSIONS,
    CODE_EXTENSIONS,
    DOCUMENTATION_EXTENSIONS,
    CONFIG_EXTENSIONS,
    filter_files_by_extension,
    count_by_extension,
    count_by_language,
    get_test_files,
    get_documentation_files,
    calculate_ratio,
)

from src.tasks.core.context import (
    TaskContext,
    task_context,
    get_current_context,
    set_current_context,
)

from src.tasks.core.cache import (
    TaskResultCache,
    cached_task,
    get_task_cache,
)

from src.tasks.core.types import (
    AntiPatternResult,
    DependencyGraphResult,
    DuplicationResult,
    HealthResult,
    ImpactAnalysisResult,
    KnowledgeGraphResult,
    PatternDetectionResult,
    SecurityHeuristicsResult,
    SemanticDuplicationResult,
    SilentFailureResult,
    StructureResult,
    SynthesisRoadmapResult,
    TechStackResult,
    TestMappingResult,
    VulnerabilityResult,
)

__all__ = [
    # Base task
    "BaseTask",
    "FileInfo",
    "TaskResult",
    # Extension constants
    "PYTHON_EXTENSIONS",
    "JAVASCRIPT_EXTENSIONS",
    "TYPESCRIPT_EXTENSIONS",
    "JS_TS_EXTENSIONS",
    "CODE_EXTENSIONS",
    "DOCUMENTATION_EXTENSIONS",
    "CONFIG_EXTENSIONS",
    # Utility functions
    "filter_files_by_extension",
    "count_by_extension",
    "count_by_language",
    "get_test_files",
    "get_documentation_files",
    "calculate_ratio",
    # Context
    "TaskContext",
    "task_context",
    "get_current_context",
    "set_current_context",
    # Caching
    "TaskResultCache",
    "cached_task",
    "get_task_cache",
    # Type definitions
    "AntiPatternResult",
    "DependencyGraphResult",
    "DuplicationResult",
    "HealthResult",
    "ImpactAnalysisResult",
    "KnowledgeGraphResult",
    "PatternDetectionResult",
    "SecurityHeuristicsResult",
    "SemanticDuplicationResult",
    "SilentFailureResult",
    "StructureResult",
    "SynthesisRoadmapResult",
    "TechStackResult",
    "TestMappingResult",
    "VulnerabilityResult",
]
