"""Webhook management endpoints.

Extracted from src/api/tasks.py — handles CRUD for webhooks,
delivery history, and test event emission.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException


def build_webhooks_router() -> APIRouter:
    """Create an APIRouter with webhook endpoints."""
    router = APIRouter()

    @router.get("/tasks/webhooks")
    async def list_webhooks() -> Dict[str, Any]:
        """List all registered webhooks."""
        from src.tasks.orchestration.webhooks import get_webhook_manager

        manager = get_webhook_manager()
        webhooks = manager.list_webhooks()

        return {
            "webhooks": [w.to_dict() for w in webhooks],
            "count": len(webhooks),
        }

    @router.post("/tasks/webhooks")
    async def register_webhook(
        request: Dict[str, Any] = Body(
            ...,
            examples=[
                {
                    "summary": "Subscribe to all events",
                    "value": {
                        "url": "https://example.com/webhook",
                        "events": ["task.started", "task.completed", "task.failed"],
                    },
                },
                {
                    "summary": "With HMAC signature",
                    "value": {
                        "url": "https://example.com/webhook",
                        "events": ["task.completed"],
                        "secret": "my-secret-key",
                    },
                },
            ],
        ),
    ) -> Dict[str, Any]:
        """Register a new webhook.

        Args:
            url: The URL to send webhook notifications to
            events: List of events to subscribe to
            secret: Optional secret for HMAC signature verification
            max_retries: Maximum retry attempts (default: 3)
            enabled: Whether webhook is active (default: True)
        """
        from src.tasks.orchestration.webhooks import WebhookConfig, WebhookEvent, get_webhook_manager

        url = request.get("url")
        if not url:
            raise HTTPException(status_code=400, detail="url is required")

        event_names = request.get("events", [])
        if not event_names:
            raise HTTPException(status_code=400, detail="events list is required")

        # Convert event names to WebhookEvent enum
        try:
            events = [WebhookEvent(name) for name in event_names]
        except ValueError:
            valid_events = [e.value for e in WebhookEvent]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid event name. Valid events: {valid_events}"
            )

        config = WebhookConfig(
            url=url,
            events=events,
            secret=request.get("secret"),
            enabled=request.get("enabled", True),
        )
        if "max_retries" in request:
            # set attribute separately to avoid constructor mismatch
            config.max_retries = request.get("max_retries", 3)

        manager = get_webhook_manager()
        manager.register_webhook(config)

        return {
            "status": "registered",
            "webhook_id": config.id,
            "url": config.url,
            "events": [e.value for e in config.events],
        }

    @router.get("/tasks/webhooks/{webhook_id}")
    async def get_webhook(webhook_id: str) -> Dict[str, Any]:
        """Get details of a specific webhook."""
        from src.tasks.orchestration.webhooks import get_webhook_manager

        manager = get_webhook_manager()
        webhook = manager.get_webhook(webhook_id)

        if webhook is None:
            raise HTTPException(
                status_code=404,
                detail=f"Webhook not found: {webhook_id}"
            )

        return webhook.to_dict()

    @router.put("/tasks/webhooks/{webhook_id}")
    async def update_webhook(
        webhook_id: str,
        request: Dict[str, Any] = Body(...),
    ) -> Dict[str, Any]:
        """Update a webhook configuration.

        Args:
            url: New URL (optional)
            events: New events list (optional)
            secret: New secret (optional)
            enabled: Enable/disable (optional)
        """
        from src.tasks.orchestration.webhooks import WebhookEvent, get_webhook_manager

        manager = get_webhook_manager()

        updates = {}
        if "url" in request:
            updates["url"] = request["url"]
        if "events" in request:
            try:
                updates["events"] = [WebhookEvent(name) for name in request["events"]]
            except ValueError:
                valid_events = [e.value for e in WebhookEvent]
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid event. Valid: {valid_events}"
                )
        if "secret" in request:
            updates["secret"] = request["secret"]
        if "enabled" in request:
            updates["enabled"] = request["enabled"]
        if "max_retries" in request:
            updates["max_retries"] = request["max_retries"]

        webhook = manager.update_webhook(webhook_id, updates)

        if webhook is None:
            raise HTTPException(
                status_code=404,
                detail=f"Webhook not found: {webhook_id}"
            )

        return {
            "status": "updated",
            "webhook": webhook.to_dict(),
        }

    @router.delete("/tasks/webhooks/{webhook_id}")
    async def delete_webhook(webhook_id: str) -> Dict[str, Any]:
        """Delete a webhook."""
        from src.tasks.orchestration.webhooks import get_webhook_manager

        manager = get_webhook_manager()
        deleted = manager.unregister_webhook(webhook_id)

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Webhook not found: {webhook_id}"
            )

        return {"status": "deleted", "webhook_id": webhook_id}

    @router.get("/tasks/webhooks/{webhook_id}/deliveries")
    async def get_webhook_deliveries(
        webhook_id: str,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Get delivery history for a webhook."""
        from src.tasks.orchestration.webhooks import get_webhook_manager

        manager = get_webhook_manager()
        webhook = manager.get_webhook(webhook_id)

        if webhook is None:
            raise HTTPException(
                status_code=404,
                detail=f"Webhook not found: {webhook_id}"
            )

        deliveries = manager.get_deliveries(webhook_id, limit)

        return {
            "webhook_id": webhook_id,
            "deliveries": [d.to_dict() for d in deliveries],
            "count": len(deliveries),
        }

    @router.post("/tasks/webhooks/{webhook_id}/test")
    async def test_webhook(webhook_id: str) -> Dict[str, Any]:
        """Send a test event to a webhook."""
        from src.tasks.orchestration.webhooks import WebhookEvent, get_webhook_manager

        manager = get_webhook_manager()
        webhook = manager.get_webhook(webhook_id)

        if webhook is None:
            raise HTTPException(
                status_code=404,
                detail=f"Webhook not found: {webhook_id}"
            )

        # Emit a test event
        delivery = manager.emit(
            event=WebhookEvent.TASK_COMPLETED,
            data={
                "task_name": "test-webhook",
                "task_id": "test-123",
                "status": "success",
                "duration_seconds": 0.1,
                "is_test": True,
            },
            webhook_ids=[webhook_id],
            async_delivery=False,  # Wait for result
        )

        if delivery:
            return {
                "status": "sent",
                "delivery": delivery[0].to_dict() if delivery else None,
            }
        return {"status": "no_delivery", "reason": "Webhook may not subscribe to task.completed"}

    return router
