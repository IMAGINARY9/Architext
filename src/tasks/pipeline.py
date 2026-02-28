"""Compatibility shim — re-exports from src.tasks.orchestration.pipeline.

All implementations live in src/tasks/orchestration/pipeline.py.
Import from ``src.tasks.orchestration`` or ``src.tasks`` for the public API.
"""
from src.tasks.orchestration.pipeline import *  # noqa: F401,F403
from src.tasks.orchestration.pipeline import (  # explicit re-exports for type checkers
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

__all__ = [
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
]
