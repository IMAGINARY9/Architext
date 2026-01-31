"""Orchestration subpackage for task scheduling, pipelines, and coordination.

This subpackage re-exports from existing orchestration modules:
- history: Task execution history and analytics
- metrics: Execution metrics and dashboard
- pipeline: Custom task pipelines and composition
- recommendations: Intelligent task recommendations
- scheduler: Task scheduling and automation
- webhooks: Webhook notifications for task events
"""
from __future__ import annotations

# Re-export from history module
from src.tasks.history import (
    TaskExecution,
    TaskAnalytics,
    TaskExecutionHistory,
    ExecutionTracker,
    get_task_history,
)

# Re-export from metrics module
from src.tasks.metrics import (
    ExecutionTrend,
    TaskMetrics,
    DashboardMetrics,
    MetricsDashboard,
    get_metrics_dashboard,
    get_dashboard_metrics,
)

# Re-export from pipeline module
from src.tasks.pipeline import (
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
)

# Re-export from recommendations module
from src.tasks.recommendations import (
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
)

# Re-export from scheduler module
from src.tasks.scheduler import (
    ScheduleType,
    ScheduleConfig,
    ScheduleExecution,
    TaskScheduler,
    get_task_scheduler,
    create_interval_schedule,
    create_cron_schedule,
    create_one_time_schedule,
)

# Re-export from webhooks module
from src.tasks.webhooks import (
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
    # History
    "TaskExecution",
    "TaskAnalytics",
    "TaskExecutionHistory",
    "ExecutionTracker",
    "get_task_history",
    # Metrics
    "ExecutionTrend",
    "TaskMetrics",
    "DashboardMetrics",
    "MetricsDashboard",
    "get_metrics_dashboard",
    "get_dashboard_metrics",
    # Pipeline
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
    # Recommendations
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
    # Scheduler
    "ScheduleType",
    "ScheduleConfig",
    "ScheduleExecution",
    "TaskScheduler",
    "get_task_scheduler",
    "create_interval_schedule",
    "create_cron_schedule",
    "create_one_time_schedule",
    # Webhooks
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
]
