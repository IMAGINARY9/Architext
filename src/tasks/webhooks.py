"""Compatibility shim — re-exports from src.tasks.orchestration.webhooks.

All implementations live in src/tasks/orchestration/webhooks.py.
Import from ``src.tasks.orchestration`` or ``src.tasks`` for the public API.
"""
from src.tasks.orchestration.webhooks import *  # noqa: F401,F403
from src.tasks.orchestration.webhooks import (  # explicit re-exports for type checkers
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
