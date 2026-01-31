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
from src.tasks.shared import TaskContext, task_context, set_current_context, get_current_context
from src.tasks.structure import analyze_structure
from src.tasks.tech_stack import tech_stack
from src.tasks.cache import TaskResultCache, cached_task, get_task_cache
from src.tasks.base import (
    BaseTask,
    FileInfo,
    TaskResult,
    PYTHON_EXTENSIONS,
    JS_TS_EXTENSIONS,
    CODE_EXTENSIONS,
    filter_files_by_extension,
    count_by_extension,
    count_by_language,
    get_test_files,
    get_documentation_files,
    calculate_ratio,
)
from src.tasks.history import (
    TaskExecution,
    TaskAnalytics,
    TaskExecutionHistory,
    ExecutionTracker,
    get_task_history,
)
from src.tasks.pipeline import (
    PipelineStep,
    ParallelGroup,
    TaskPipeline,
    PipelineExecutor,
    PipelineStore,
    BUILTIN_PIPELINES,
    list_builtin_pipelines,
    get_builtin_pipeline,
    get_pipeline_store,
)
from src.tasks.recommendations import (
    TaskRecommendation,
    RecommendationConfig,
    TaskRecommendationEngine,
    get_recommendation_engine,
    get_task_recommendations,
)
from src.tasks.metrics import (
    ExecutionTrend,
    TaskMetrics,
    DashboardMetrics,
    MetricsDashboard,
    get_metrics_dashboard,
    get_dashboard_metrics,
)
from src.tasks.tasks_v2 import (
    AntiPatternDetectionTask,
    SilentFailuresTask,
    TestMappingTask,
    HealthScoreTask,
    detect_anti_patterns_v2,
    identify_silent_failures_v2,
    test_mapping_analysis_v2,
    health_score_v2,
)

# Import types for type hints (re-export for backwards compatibility)
from src.tasks.types import (
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
	# Task functions
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
	# Task context utilities
	"TaskContext",
	"task_context",
	"set_current_context",
	"get_current_context",
	# Caching
	"TaskResultCache",
	"cached_task",
	"get_task_cache",
	# Base task utilities
	"BaseTask",
	"FileInfo",
	"TaskResult",
	"PYTHON_EXTENSIONS",
	"JS_TS_EXTENSIONS",
	"CODE_EXTENSIONS",
	"filter_files_by_extension",
	"count_by_extension",
	"count_by_language",
	"get_test_files",
	"get_documentation_files",
	"calculate_ratio",
	# Execution history and analytics
	"TaskExecution",
	"TaskAnalytics",
	"TaskExecutionHistory",
	"ExecutionTracker",
	"get_task_history",
	# Pipeline composition
	"PipelineStep",
	"ParallelGroup",
	"TaskPipeline",
	"PipelineExecutor",
	"PipelineStore",
	"BUILTIN_PIPELINES",
	"list_builtin_pipelines",
	"get_builtin_pipeline",
	"get_pipeline_store",
	# Task recommendations
	"TaskRecommendation",
	"RecommendationConfig",
	"TaskRecommendationEngine",
	"get_recommendation_engine",
	"get_task_recommendations",
	# Metrics dashboard
	"ExecutionTrend",
	"TaskMetrics",
	"DashboardMetrics",
	"MetricsDashboard",
	"get_metrics_dashboard",
	"get_dashboard_metrics",
	# BaseTask implementations (v2)
	"AntiPatternDetectionTask",
	"SilentFailuresTask",
	"TestMappingTask",
	"HealthScoreTask",
	"detect_anti_patterns_v2",
	"identify_silent_failures_v2",
	"test_mapping_analysis_v2",
	"health_score_v2",
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
