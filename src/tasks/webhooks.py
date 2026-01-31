"""Task execution webhooks and notifications.

This module provides webhook support for task execution events,
allowing external systems to be notified of task completions,
failures, and other events.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

class WebhookEvent(str, Enum):
    """Types of webhook events."""
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_CACHED = "task.cached"
    PIPELINE_STARTED = "pipeline.started"
    PIPELINE_COMPLETED = "pipeline.completed"
    PIPELINE_FAILED = "pipeline.failed"


@dataclass
class WebhookConfig:
    """Configuration for a webhook endpoint."""
    url: str
    events: List[WebhookEvent]
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    secret: Optional[str] = None
    enabled: bool = True
    timeout_seconds: int = 10
    retry_count: int = 3
    retry_delay_seconds: float = 1.0
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Alias for retry_count
    @property
    def max_retries(self) -> int:
        return self.retry_count
    
    @max_retries.setter
    def max_retries(self, value: int) -> None:
        self.retry_count = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "url": self.url,
            "events": [e.value for e in self.events],
            "secret": "***" if self.secret else None,  # Don't expose secret
            "enabled": self.enabled,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "headers": self.headers,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebhookConfig":
        """Create from dictionary."""
        events = [WebhookEvent(e) for e in data.get("events", [])]
        return cls(
            id=data["id"],
            url=data["url"],
            events=events,
            secret=data.get("secret"),
            enabled=data.get("enabled", True),
            timeout_seconds=data.get("timeout_seconds", 10),
            retry_count=data.get("retry_count", 3),
            retry_delay_seconds=data.get("retry_delay_seconds", 1.0),
            headers=data.get("headers", {}),
        )


@dataclass
class WebhookPayload:
    """Payload sent to webhook endpoints."""
    event: WebhookEvent
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    webhook_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event": self.event.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt."""
    webhook_id: str
    event: WebhookEvent
    url: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    error: Optional[str] = None
    delivered_at: datetime = field(default_factory=datetime.now)
    attempts: int = 1
    success: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "webhook_id": self.webhook_id,
            "event": self.event.value,
            "url": self.url,
            "status_code": self.status_code,
            "error": self.error,
            "delivered_at": self.delivered_at.isoformat(),
            "attempts": self.attempts,
            "success": self.success,
        }


class WebhookManager:
    """
    Manages webhook registrations and deliveries.
    
    Webhooks can be registered to receive notifications for specific
    task execution events.
    """
    
    DEFAULT_STORAGE_PATH = Path.home() / ".architext" / "webhooks"
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        async_delivery: bool = True,
    ):
        """
        Initialize the webhook manager.
        
        Args:
            storage_path: Path to store webhook configurations
            async_delivery: Whether to deliver webhooks asynchronously
        """
        self.storage_path = storage_path or self.DEFAULT_STORAGE_PATH
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.async_delivery = async_delivery
        
        self._webhooks: Dict[str, WebhookConfig] = {}
        self._deliveries: List[WebhookDelivery] = []
        self._callbacks: List[Callable[[WebhookPayload], None]] = []
        self._lock = threading.Lock()
        
        self._load_webhooks()
    
    def _load_webhooks(self) -> None:
        """Load webhooks from storage."""
        config_file = self.storage_path / "webhooks.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    data = json.load(f)
                for item in data.get("webhooks", []):
                    webhook = WebhookConfig.from_dict(item)
                    self._webhooks[webhook.id] = webhook
            except Exception:
                pass  # Ignore load errors
    
    def _save_webhooks(self) -> None:
        """Save webhooks to storage."""
        config_file = self.storage_path / "webhooks.json"
        data = {
            "webhooks": [
                {**w.to_dict(), "secret": w.secret}  # Include actual secret
                for w in self._webhooks.values()
            ]
        }
        with open(config_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def register(self, config: WebhookConfig) -> WebhookConfig:
        """
        Register a new webhook.
        
        Args:
            config: Webhook configuration
            
        Returns:
            The registered webhook config
        """
        with self._lock:
            self._webhooks[config.id] = config
            self._save_webhooks()
        return config
    
    def unregister(self, webhook_id: str) -> bool:
        """
        Unregister a webhook.
        
        Args:
            webhook_id: ID of webhook to remove
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if webhook_id in self._webhooks:
                del self._webhooks[webhook_id]
                self._save_webhooks()
                return True
        return False
    
    def get(self, webhook_id: str) -> Optional[WebhookConfig]:
        """Get a webhook by ID."""
        return self._webhooks.get(webhook_id)
    
    def list_webhooks(self) -> List[WebhookConfig]:
        """List all registered webhooks."""
        return list(self._webhooks.values())
    
    def update(self, webhook_id: str, updates: Dict[str, Any]) -> Optional[WebhookConfig]:
        """
        Update a webhook configuration.
        
        Args:
            webhook_id: ID of webhook to update
            updates: Dictionary of fields to update
            
        Returns:
            Updated webhook config or None if not found
        """
        with self._lock:
            if webhook_id not in self._webhooks:
                return None
            
            webhook = self._webhooks[webhook_id]
            
            if "url" in updates:
                webhook.url = updates["url"]
            if "events" in updates:
                webhook.events = updates["events"] if isinstance(updates["events"][0], WebhookEvent) else [WebhookEvent(e) for e in updates["events"]]
            if "enabled" in updates:
                webhook.enabled = updates["enabled"]
            if "secret" in updates:
                webhook.secret = updates["secret"]
            if "timeout_seconds" in updates:
                webhook.timeout_seconds = updates["timeout_seconds"]
            if "headers" in updates:
                webhook.headers = updates["headers"]
            if "max_retries" in updates:
                webhook.retry_count = updates["max_retries"]
            
            self._save_webhooks()
            return webhook
    
    # Alias methods for cleaner API
    def register_webhook(self, config: WebhookConfig) -> WebhookConfig:
        """Alias for register()."""
        return self.register(config)
    
    def unregister_webhook(self, webhook_id: str) -> bool:
        """Alias for unregister()."""
        return self.unregister(webhook_id)
    
    def get_webhook(self, webhook_id: str) -> Optional[WebhookConfig]:
        """Alias for get()."""
        return self.get(webhook_id)
    
    def update_webhook(self, webhook_id: str, updates: Dict[str, Any]) -> Optional[WebhookConfig]:
        """Alias for update()."""
        return self.update(webhook_id, updates)
    
    def get_deliveries(self, webhook_id: Optional[str] = None, limit: int = 50) -> List[WebhookDelivery]:
        """Alias for get_recent_deliveries()."""
        return self.get_recent_deliveries(limit=limit, webhook_id=webhook_id)
    
    def add_callback(self, callback: Callable[[WebhookPayload], None]) -> None:
        """
        Add a local callback for webhook events.
        
        Useful for in-process notifications without HTTP.
        """
        self._callbacks.append(callback)
    
    def emit(
        self,
        event: Optional[WebhookEvent] = None,
        data: Optional[Dict[str, Any]] = None,
        payload: Optional[WebhookPayload] = None,
        webhook_ids: Optional[List[str]] = None,
        async_delivery: Optional[bool] = None,
    ) -> List[WebhookDelivery]:
        """
        Emit a webhook event.
        
        Can be called with either:
        - A WebhookPayload object: emit(payload=my_payload)
        - Event and data: emit(event=WebhookEvent.TASK_COMPLETED, data={...})
        
        Args:
            event: The event type
            data: Event data dictionary
            payload: Pre-built WebhookPayload (alternative to event+data)
            webhook_ids: Specific webhook IDs to deliver to (optional)
            async_delivery: Override instance async_delivery setting
            
        Returns:
            List of delivery records (empty list if async)
        """
        # Build payload if not provided
        if payload is None:
            if event is None:
                raise ValueError("Either payload or event must be provided")
            payload = WebhookPayload(
                event=event,
                data=data or {},
            )
        
        # Notify local callbacks
        for callback in self._callbacks:
            try:
                callback(payload)
            except Exception:
                pass  # Don't fail on callback errors
        
        # Find matching webhooks
        if webhook_ids:
            matching = [
                self._webhooks[wid] for wid in webhook_ids
                if wid in self._webhooks and self._webhooks[wid].enabled
            ]
        else:
            matching = [
                w for w in self._webhooks.values()
                if w.enabled and payload.event in w.events
            ]
        
        if not matching:
            return []
        
        # Determine delivery mode
        use_async = async_delivery if async_delivery is not None else self.async_delivery
        
        # Deliver webhooks
        if use_async:
            thread = threading.Thread(
                target=self._deliver_all,
                args=(matching, payload),
                daemon=True,
            )
            thread.start()
            return []
        else:
            return self._deliver_all_sync(matching, payload)
    
    def _deliver_all(
        self,
        webhooks: List[WebhookConfig],
        payload: WebhookPayload,
    ) -> None:
        """Deliver payload to multiple webhooks (async version, no return)."""
        for webhook in webhooks:
            self._deliver(webhook, payload)
    
    def _deliver_all_sync(
        self,
        webhooks: List[WebhookConfig],
        payload: WebhookPayload,
    ) -> List[WebhookDelivery]:
        """Deliver payload to multiple webhooks (sync version with returns)."""
        deliveries = []
        for webhook in webhooks:
            delivery = self._deliver(webhook, payload)
            deliveries.append(delivery)
        return deliveries
    
    def _deliver_to_webhook(
        self,
        webhook: WebhookConfig,
        payload: WebhookPayload,
    ) -> WebhookDelivery:
        """Alias for _deliver for tests."""
        return self._deliver(webhook, payload)
    
    def _deliver(
        self,
        webhook: WebhookConfig,
        payload: WebhookPayload,
    ) -> WebhookDelivery:
        """Deliver payload to a single webhook with retries."""
        payload_dict = payload.to_dict()
        body = json.dumps(payload_dict).encode("utf-8")
        
        delivery = WebhookDelivery(
            webhook_id=webhook.id,
            event=payload.event,
            url=webhook.url,
            payload=payload_dict,
        )
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Architext-Webhook/1.0",
            "X-Webhook-Event": payload.event.value,
            **webhook.headers,
        }
        
        if webhook.secret:
            import hashlib
            import hmac
            signature = hmac.new(
                webhook.secret.encode(),
                body,
                hashlib.sha256,
            ).hexdigest()
            headers["X-Webhook-Signature"] = f"sha256={signature}"
        
        for attempt in range(webhook.retry_count):
            delivery.attempts = attempt + 1
            
            try:
                request = Request(
                    webhook.url,
                    data=body,
                    headers=headers,
                    method="POST",
                )
                
                with urlopen(request, timeout=webhook.timeout_seconds) as response:
                    delivery.status_code = response.status
                    delivery.response_body = response.read().decode("utf-8")[:1000]
                    delivery.success = 200 <= response.status < 300
                    
                if delivery.success:
                    break
                    
            except URLError as e:
                delivery.error = str(e)
            except Exception as e:
                delivery.error = str(e)
            
            # Wait before retry
            if attempt < webhook.retry_count - 1:
                time.sleep(webhook.retry_delay_seconds * (attempt + 1))
        
        with self._lock:
            self._deliveries.append(delivery)
            # Keep only recent deliveries
            if len(self._deliveries) > 1000:
                self._deliveries = self._deliveries[-500:]
        
        return delivery
    
    def get_recent_deliveries(
        self,
        limit: int = 50,
        webhook_id: Optional[str] = None,
    ) -> List[WebhookDelivery]:
        """
        Get recent webhook deliveries.
        
        Args:
            limit: Maximum number of deliveries to return
            webhook_id: Filter by webhook ID
            
        Returns:
            List of recent deliveries
        """
        with self._lock:
            deliveries = self._deliveries
            if webhook_id:
                deliveries = [d for d in deliveries if d.webhook_id == webhook_id]
            return deliveries[-limit:]
    
    def _generate_signature(self, body: bytes, secret: str) -> str:
        """Generate HMAC signature for webhook payload."""
        import hashlib
        import hmac
        signature = hmac.new(
            secret.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()
        return f"sha256={signature}"

# Convenience functions for emitting events

def emit_task_started(
    task_name: str,
    task_id: str,
    source_path: Optional[str] = None,
    **metadata: Any,
) -> None:
    """Emit task started event."""
    manager = get_webhook_manager()
    manager.emit(payload=WebhookPayload(
        event=WebhookEvent.TASK_STARTED,
        data={
            "task_name": task_name,
            "task_id": task_id,
            "source_path": source_path,
            "status": "started",
            **metadata,
        },
    ))


def emit_task_completed(
    task_name: str,
    task_id: str,
    duration_seconds: float,
    result_summary: Optional[Dict[str, Any]] = None,
    cached: bool = False,
    **metadata: Any,
) -> None:
    """Emit task completed event."""
    manager = get_webhook_manager()
    event = WebhookEvent.TASK_CACHED if cached else WebhookEvent.TASK_COMPLETED
    manager.emit(payload=WebhookPayload(
        event=event,
        data={
            "task_name": task_name,
            "task_id": task_id,
            "status": "completed",
            "duration_seconds": duration_seconds,
            "result_summary": result_summary,
            "cached": cached,
            **metadata,
        },
    ))


def emit_task_cached(
    task_name: str,
    task_id: str = "",
    duration_seconds: float = 0.0,
    result_summary: Optional[Dict[str, Any]] = None,
    **metadata: Any,
) -> None:
    """Emit task cached event (result returned from cache)."""
    manager = get_webhook_manager()
    manager.emit(payload=WebhookPayload(
        event=WebhookEvent.TASK_CACHED,
        data={
            "task_name": task_name,
            "task_id": task_id,
            "status": "cached",
            "duration_seconds": duration_seconds,
            "result_summary": result_summary,
            **metadata,
        },
    ))


def emit_task_failed(
    task_name: str,
    task_id: str,
    error: str,
    duration_seconds: Optional[float] = None,
    **metadata: Any,
) -> None:
    """Emit task failed event."""
    manager = get_webhook_manager()
    manager.emit(payload=WebhookPayload(
        event=WebhookEvent.TASK_FAILED,
        data={
            "task_name": task_name,
            "task_id": task_id,
            "status": "failed",
            "duration_seconds": duration_seconds,
            "error": error,
            **metadata,
        },
    ))


def emit_pipeline_started(
    pipeline_name: str,
    pipeline_id: str = "",
    **metadata: Any,
) -> None:
    """Emit pipeline started event."""
    manager = get_webhook_manager()
    manager.emit(payload=WebhookPayload(
        event=WebhookEvent.PIPELINE_STARTED,
        data={
            "pipeline_name": pipeline_name,
            "pipeline_id": pipeline_id,
            "status": "started",
            **metadata,
        },
    ))


def emit_pipeline_completed(
    pipeline_name: str,
    pipeline_id: str = "",
    duration_seconds: float = 0.0,
    tasks_executed: int = 0,
    tasks_failed: int = 0,
    **metadata: Any,
) -> None:
    """Emit pipeline completed event."""
    manager = get_webhook_manager()
    event = WebhookEvent.PIPELINE_FAILED if tasks_failed > 0 else WebhookEvent.PIPELINE_COMPLETED
    manager.emit(payload=WebhookPayload(
        event=event,
        data={
            "pipeline_name": pipeline_name,
            "pipeline_id": pipeline_id,
            "status": "completed" if tasks_failed == 0 else "failed",
            "duration_seconds": duration_seconds,
            "tasks_executed": tasks_executed,
            "tasks_failed": tasks_failed,
            **metadata,
        },
    ))


# Singleton instance
_webhook_manager: Optional[WebhookManager] = None


def get_webhook_manager() -> WebhookManager:
    """Get the singleton webhook manager instance."""
    global _webhook_manager
    if _webhook_manager is None:
        _webhook_manager = WebhookManager()
    return _webhook_manager


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
