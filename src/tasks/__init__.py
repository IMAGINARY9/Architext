"""Analysis tasks package.

This package provides a comprehensive set of code analysis tasks organized into:
- core/: Infrastructure (BaseTask, cache, context, types)
- analysis/: All analysis task implementations
- orchestration/: Execution infrastructure (history, metrics, pipelines, etc.)

Primary API: Import task classes directly from this package.
"""
from __future__ import annotations

# =============================================================================
# Core Infrastructure (from core/)
# =============================================================================
from src.tasks.core import (
    # Base task utilities
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
    # Context management
    TaskContext,
    task_context,
    set_current_context,
    get_current_context,
    # Caching
    TaskResultCache,
    cached_task,
    get_task_cache,
    # Type definitions
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

# =============================================================================
# Legacy Task Functions (must load BEFORE orchestration to avoid circular imports)
# =============================================================================
from src.tasks.anti_patterns import detect_anti_patterns
from src.tasks.architecture import (
    architecture_pattern_detection,
    code_knowledge_graph,
    dependency_graph_export,
    impact_analysis,
)
from src.tasks.duplication import detect_duplicate_blocks, detect_duplicate_blocks_semantic
from src.tasks.health import health_score
from src.tasks.quality import identify_silent_failures, test_mapping_analysis
from src.tasks.query import query_diagnostics
from src.tasks.roadmap import synthesis_roadmap
from src.tasks.security import detect_vulnerabilities, security_heuristics
from src.tasks.structure import analyze_structure
from src.tasks.tech_stack import tech_stack

# =============================================================================
# Analysis Task Classes (from analysis/)
# =============================================================================
from src.tasks.analysis import (
    # Anti-patterns
    AntiPatternDetectionTask,
    detect_anti_patterns_v2,
    # Architecture
    ArchitecturePatternTask,
    ImpactAnalysisTask,
    DependencyGraphTask,
    architecture_pattern_detection_v2,
    impact_analysis_v2,
    dependency_graph_export_v2,
    # Duplication
    DuplicateBlocksTask,
    SemanticDuplicationTask,
    detect_duplicate_blocks_v2,
    duplicate_blocks_detection_v2,
    detect_duplicate_blocks_semantic_v2,
    semantic_duplication_detection_v2,
    # Health
    HealthScoreTask,
    health_score_v2,
    DEFAULT_WEIGHTS,
    # Quality
    SilentFailuresTask,
    TestMappingTask,
    identify_silent_failures_v2,
    silent_failures_detection_v2,
    test_mapping_analysis_v2,
    test_mapping_v2,
    # Security
    SecurityHeuristicsTask,
    security_heuristics_v2,
    # Structure
    StructureAnalysisTask,
    analyze_structure_v2,
    structure_analysis_v2,
    # Tech Stack
    TechStackTask,
    tech_stack_v2,
    tech_stack_detection_v2,
)

# =============================================================================
# Orchestration (from orchestration/) - loads after legacy functions
# =============================================================================
from src.tasks.orchestration import (
    # History
    TaskExecution,
    TaskAnalytics,
    TaskExecutionHistory,
    ExecutionTracker,
    get_task_history,
    # Metrics
    ExecutionTrend,
    TaskMetrics,
    DashboardMetrics,
    MetricsDashboard,
    get_metrics_dashboard,
    get_dashboard_metrics,
    # Pipeline
    PipelineStep,
    ParallelGroup,
    TaskPipeline,
    PipelineExecutor,
    PipelineStore,
    PipelineResult,
    get_pipeline_store,
    BUILTIN_PIPELINES,
    list_builtin_pipelines,
    get_builtin_pipeline,
    # Recommendations
    TaskRecommendation,
    ScoringWeights,
    ScoringWeightsStore,
    RecommendationConfig,
    TaskRecommendationEngine,
    get_recommendation_engine,
    get_task_recommendations,
    get_scoring_weights,
    update_scoring_weights,
    reset_scoring_weights,
    get_weight_presets,
    apply_weight_preset,
    # Scheduler
    ScheduleType,
    ScheduleConfig,
    ScheduleExecution,
    TaskScheduler,
    get_task_scheduler,
    create_interval_schedule,
    create_cron_schedule,
    create_one_time_schedule,
    # Webhooks
    WebhookEvent,
    WebhookConfig,
    WebhookPayload,
    WebhookDelivery,
    WebhookManager,
    get_webhook_manager,
    emit_task_started,
    emit_task_completed,
    emit_task_failed,
    emit_task_cached,
    emit_pipeline_started,
    emit_pipeline_completed,
)

__all__ = [
    # ==========================================================================
    # Core Infrastructure
    # ==========================================================================
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
    "TaskContext",
    "task_context",
    "set_current_context",
    "get_current_context",
    "TaskResultCache",
    "cached_task",
    "get_task_cache",
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
    
    # ==========================================================================
    # Analysis Task Classes
    # ==========================================================================
    "AntiPatternDetectionTask",
    "ArchitecturePatternTask",
    "DependencyGraphTask",
    "DuplicateBlocksTask",
    "HealthScoreTask",
    "ImpactAnalysisTask",
    "SecurityHeuristicsTask",
    "SemanticDuplicationTask",
    "SilentFailuresTask",
    "StructureAnalysisTask",
    "TechStackTask",
    "TestMappingTask",
    # v2 wrapper functions
    "detect_anti_patterns_v2",
    "architecture_pattern_detection_v2",
    "impact_analysis_v2",
    "dependency_graph_export_v2",
    "detect_duplicate_blocks_v2",
    "duplicate_blocks_detection_v2",
    "detect_duplicate_blocks_semantic_v2",
    "semantic_duplication_detection_v2",
    "health_score_v2",
    "identify_silent_failures_v2",
    "silent_failures_detection_v2",
    "test_mapping_analysis_v2",
    "test_mapping_v2",
    "security_heuristics_v2",
    "analyze_structure_v2",
    "structure_analysis_v2",
    "tech_stack_v2",
    "tech_stack_detection_v2",
    "DEFAULT_WEIGHTS",
    
    # ==========================================================================
    # Orchestration
    # ==========================================================================
    "TaskExecution",
    "TaskAnalytics",
    "TaskExecutionHistory",
    "ExecutionTracker",
    "get_task_history",
    "ExecutionTrend",
    "TaskMetrics",
    "DashboardMetrics",
    "MetricsDashboard",
    "get_metrics_dashboard",
    "get_dashboard_metrics",
    "PipelineStep",
    "ParallelGroup",
    "TaskPipeline",
    "PipelineExecutor",
    "PipelineStore",
    "PipelineResult",
    "get_pipeline_store",
    "BUILTIN_PIPELINES",
    "list_builtin_pipelines",
    "get_builtin_pipeline",
    "TaskRecommendation",
    "ScoringWeights",
    "ScoringWeightsStore",
    "RecommendationConfig",
    "TaskRecommendationEngine",
    "get_recommendation_engine",
    "get_task_recommendations",
    "get_scoring_weights",
    "update_scoring_weights",
    "reset_scoring_weights",
    "get_weight_presets",
    "apply_weight_preset",
    "ScheduleType",
    "ScheduleConfig",
    "ScheduleExecution",
    "TaskScheduler",
    "get_task_scheduler",
    "create_interval_schedule",
    "create_cron_schedule",
    "create_one_time_schedule",
    "WebhookEvent",
    "WebhookConfig",
    "WebhookPayload",
    "WebhookDelivery",
    "WebhookManager",
    "get_webhook_manager",
    "emit_task_started",
    "emit_task_completed",
    "emit_task_failed",
    "emit_task_cached",
    "emit_pipeline_started",
    "emit_pipeline_completed",
    
    # ==========================================================================
    # Legacy Task Functions
    # ==========================================================================
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
