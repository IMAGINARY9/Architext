"""Tests for webhook notification system."""
from __future__ import annotations

import hashlib
import hmac
import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.tasks.webhooks import (
    WebhookConfig,
    WebhookDelivery,
    WebhookEvent,
    WebhookManager,
    WebhookPayload,
    emit_task_completed,
    emit_task_failed,
    emit_task_started,
    emit_task_cached,
    get_webhook_manager,
)


class TestWebhookEvent:
    """Tests for WebhookEvent enum."""
    
    def test_event_values(self):
        """Test event enum values."""
        assert WebhookEvent.TASK_STARTED.value == "task.started"
        assert WebhookEvent.TASK_COMPLETED.value == "task.completed"
        assert WebhookEvent.TASK_FAILED.value == "task.failed"
        assert WebhookEvent.TASK_CACHED.value == "task.cached"
        assert WebhookEvent.PIPELINE_STARTED.value == "pipeline.started"
        assert WebhookEvent.PIPELINE_COMPLETED.value == "pipeline.completed"
    
    def test_event_from_string(self):
        """Test creating event from string."""
        event = WebhookEvent("task.started")
        assert event == WebhookEvent.TASK_STARTED


class TestWebhookConfig:
    """Tests for WebhookConfig dataclass."""
    
    def test_create_config(self):
        """Test creating webhook config."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
        )
        
        assert config.url == "https://example.com/webhook"
        assert config.events == [WebhookEvent.TASK_COMPLETED]
        assert config.enabled is True
        assert config.id is not None
    
    def test_config_with_secret(self):
        """Test config with secret for signing."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
            secret="my-secret-key",
        )
        
        assert config.secret == "my-secret-key"
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED, WebhookEvent.TASK_FAILED],
            retry_count=5,
        )
        
        data = config.to_dict()
        
        assert data["url"] == "https://example.com/webhook"
        assert data["events"] == ["task.completed", "task.failed"]
        assert data["retry_count"] == 5
        assert "secret" not in data or data["secret"] is None
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "id": "webhook-123",
            "url": "https://example.com/webhook",
            "events": ["task.started", "task.completed"],
            "enabled": True,
        }
        
        config = WebhookConfig.from_dict(data)
        
        assert config.id == "webhook-123"
        assert config.url == "https://example.com/webhook"
        assert WebhookEvent.TASK_STARTED in config.events
        assert WebhookEvent.TASK_COMPLETED in config.events


class TestWebhookPayload:
    """Tests for WebhookPayload dataclass."""
    
    def test_create_payload(self):
        """Test creating webhook payload."""
        payload = WebhookPayload(
            event=WebhookEvent.TASK_COMPLETED,
            data={"task_name": "health-score", "score": 85},
        )
        
        assert payload.event == WebhookEvent.TASK_COMPLETED
        assert payload.data["task_name"] == "health-score"
        assert payload.timestamp is not None
    
    def test_to_dict(self):
        """Test converting payload to dictionary."""
        payload = WebhookPayload(
            event=WebhookEvent.TASK_COMPLETED,
            data={"task_name": "health-score"},
            webhook_id="webhook-123",
        )
        
        data = payload.to_dict()
        
        assert data["event"] == "task.completed"
        assert data["data"]["task_name"] == "health-score"
        assert "timestamp" in data


class TestWebhookDelivery:
    """Tests for WebhookDelivery dataclass."""
    
    def test_create_delivery(self):
        """Test creating delivery record."""
        delivery = WebhookDelivery(
            webhook_id="webhook-123",
            event=WebhookEvent.TASK_COMPLETED,
            url="https://example.com/webhook",
        )
        
        assert delivery.webhook_id == "webhook-123"
        assert delivery.success is False  # Default
        assert delivery.attempts == 1  # Default is 1
    
    def test_delivery_success(self):
        """Test successful delivery record."""
        delivery = WebhookDelivery(
            webhook_id="webhook-123",
            event=WebhookEvent.TASK_COMPLETED,
            url="https://example.com/webhook",
            success=True,
            status_code=200,
            attempts=1,
        )
        
        assert delivery.success is True
        assert delivery.status_code == 200
    
    def test_to_dict(self):
        """Test converting delivery to dictionary."""
        delivery = WebhookDelivery(
            webhook_id="webhook-123",
            event=WebhookEvent.TASK_COMPLETED,
            url="https://example.com/webhook",
            success=True,
            status_code=200,
        )
        
        data = delivery.to_dict()
        
        assert data["webhook_id"] == "webhook-123"
        assert data["event"] == "task.completed"
        assert data["success"] is True


class TestWebhookManager:
    """Tests for WebhookManager class."""
    
    @pytest.fixture
    def manager(self, tmp_path):
        """Create a manager with temporary storage."""
        return WebhookManager(storage_path=tmp_path / "webhooks")
    
    def test_register_webhook(self, manager):
        """Test registering a webhook."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
        )
        
        manager.register_webhook(config)
        
        assert len(manager.list_webhooks()) == 1
        assert manager.get_webhook(config.id) is not None
    
    def test_unregister_webhook(self, manager):
        """Test unregistering a webhook."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
        )
        manager.register_webhook(config)
        
        result = manager.unregister_webhook(config.id)
        
        assert result is True
        assert manager.get_webhook(config.id) is None
    
    def test_unregister_nonexistent(self, manager):
        """Test unregistering non-existent webhook."""
        result = manager.unregister_webhook("nonexistent")
        assert result is False
    
    def test_list_webhooks(self, manager):
        """Test listing webhooks."""
        config1 = WebhookConfig(
            url="https://example1.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
        )
        config2 = WebhookConfig(
            url="https://example2.com/webhook",
            events=[WebhookEvent.TASK_FAILED],
        )
        
        manager.register_webhook(config1)
        manager.register_webhook(config2)
        
        webhooks = manager.list_webhooks()
        
        assert len(webhooks) == 2
    
    def test_update_webhook(self, manager):
        """Test updating webhook configuration."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
        )
        manager.register_webhook(config)
        
        updated = manager.update_webhook(config.id, {
            "url": "https://new-url.com/webhook",
            "enabled": False,
        })
        
        assert updated is not None
        assert updated.url == "https://new-url.com/webhook"
        assert updated.enabled is False
    
    def test_update_nonexistent(self, manager):
        """Test updating non-existent webhook."""
        result = manager.update_webhook("nonexistent", {"enabled": False})
        assert result is None
    
    @patch("src.tasks.webhooks.urlopen")
    def test_emit_event(self, mock_urlopen, manager):
        """Test emitting an event."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"ok": true}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response
        
        config = WebhookConfig(
            url="https://example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
        )
        manager.register_webhook(config)
        
        deliveries = manager.emit(
            event=WebhookEvent.TASK_COMPLETED,
            data={"task_name": "health-score"},
            async_delivery=False,
        )
        
        assert len(deliveries) == 1
        assert deliveries[0].success is True
    
    @patch("src.tasks.webhooks.urlopen")
    def test_emit_event_no_matching_webhook(self, mock_urlopen, manager):
        """Test emitting event with no matching webhooks."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
        )
        manager.register_webhook(config)
        
        deliveries = manager.emit(
            event=WebhookEvent.TASK_FAILED,  # Different event
            data={"task_name": "health-score"},
            async_delivery=False,
        )
        
        assert len(deliveries) == 0
        mock_urlopen.assert_not_called()
    
    @patch("src.tasks.webhooks.urlopen")
    def test_emit_disabled_webhook(self, mock_urlopen, manager):
        """Test disabled webhooks don't receive events."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
            enabled=False,
        )
        manager.register_webhook(config)
        
        deliveries = manager.emit(
            event=WebhookEvent.TASK_COMPLETED,
            data={"task_name": "health-score"},
            async_delivery=False,
        )
        
        assert len(deliveries) == 0
        mock_urlopen.assert_not_called()
    
    @patch("src.tasks.webhooks.urlopen")
    def test_emit_to_specific_webhooks(self, mock_urlopen, manager):
        """Test emitting to specific webhook IDs only."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"ok": true}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response
        
        config1 = WebhookConfig(
            url="https://example1.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
        )
        config2 = WebhookConfig(
            url="https://example2.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
        )
        manager.register_webhook(config1)
        manager.register_webhook(config2)
        
        deliveries = manager.emit(
            event=WebhookEvent.TASK_COMPLETED,
            data={"task_name": "health-score"},
            webhook_ids=[config1.id],
            async_delivery=False,
        )
        
        assert len(deliveries) == 1
        assert deliveries[0].webhook_id == config1.id
    
    def test_get_deliveries(self, manager):
        """Test getting delivery history."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
        )
        manager.register_webhook(config)
        
        # Add some fake deliveries
        manager._deliveries.append(WebhookDelivery(
            webhook_id=config.id,
            event=WebhookEvent.TASK_COMPLETED,
            url=config.url,
            success=True,
            status_code=200,
        ))
        
        deliveries = manager.get_deliveries(config.id)
        
        assert len(deliveries) == 1
    
    def test_persistence(self, tmp_path):
        """Test webhook configs are persisted."""
        storage = tmp_path / "webhooks"
        
        # Create and save webhook
        manager1 = WebhookManager(storage_path=storage)
        config = WebhookConfig(
            url="https://example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
        )
        manager1.register_webhook(config)
        webhook_id = config.id
        
        # Create new manager and check persistence
        manager2 = WebhookManager(storage_path=storage)
        
        assert manager2.get_webhook(webhook_id) is not None
        assert manager2.get_webhook(webhook_id).url == "https://example.com/webhook"


class TestSignatureGeneration:
    """Tests for HMAC signature generation."""
    
    @pytest.fixture
    def manager(self, tmp_path):
        """Create a manager with temporary storage."""
        return WebhookManager(storage_path=tmp_path / "webhooks")
    
    def test_generate_signature(self, manager):
        """Test HMAC signature generation."""
        secret = "my-secret-key"
        body = json.dumps({"event": "task.completed"}).encode()
        
        signature = manager._generate_signature(body, secret)
        
        # Verify signature format
        assert signature.startswith("sha256=")
        
        # Verify signature is correct
        expected = hmac.new(
            secret.encode(),
            body,
            hashlib.sha256
        ).hexdigest()
        assert signature == f"sha256={expected}"
    
    @patch("src.tasks.webhooks.urlopen")
    def test_signature_in_headers(self, mock_urlopen, manager):
        """Test signature is included in request headers when secret is set."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"ok": true}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response
        
        config = WebhookConfig(
            url="https://example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
            secret="my-secret-key",
        )
        manager.register_webhook(config)
        
        deliveries = manager.emit(
            event=WebhookEvent.TASK_COMPLETED,
            data={"task_name": "test"},
            async_delivery=False,
        )
        
        # Verify urlopen was called
        mock_urlopen.assert_called()
        # Get the Request object that was passed to urlopen
        call_args = mock_urlopen.call_args
        request = call_args[0][0]  # First positional argument
        # Verify it has the signature header (urllib lowercases header keys)
        assert "X-webhook-signature" in request.headers


class TestConvenienceFunctions:
    """Tests for convenience emit functions."""
    
    @patch("src.tasks.webhooks.get_webhook_manager")
    def test_emit_task_started(self, mock_get_manager):
        """Test emit_task_started function."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        
        emit_task_started("health-score", "task-123", "./src")
        
        mock_manager.emit.assert_called_once()
        call_args = mock_manager.emit.call_args
        # The payload is passed as keyword argument
        payload = call_args[1]["payload"]
        assert payload.event == WebhookEvent.TASK_STARTED
        assert payload.data["task_name"] == "health-score"
    
    @patch("src.tasks.webhooks.get_webhook_manager")
    def test_emit_task_completed(self, mock_get_manager):
        """Test emit_task_completed function."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        
        emit_task_completed(
            "health-score", 
            "task-123", 
            5.5, 
            {"score": 85}, 
            False
        )
        
        mock_manager.emit.assert_called_once()
        call_args = mock_manager.emit.call_args
        payload = call_args[1]["payload"]
        assert payload.event == WebhookEvent.TASK_COMPLETED
        assert payload.data["duration_seconds"] == 5.5
    
    @patch("src.tasks.webhooks.get_webhook_manager")
    def test_emit_task_failed(self, mock_get_manager):
        """Test emit_task_failed function."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        
        emit_task_failed("health-score", "task-123", "Connection error", 2.5)
        
        mock_manager.emit.assert_called_once()
        call_args = mock_manager.emit.call_args
        payload = call_args[1]["payload"]
        assert payload.event == WebhookEvent.TASK_FAILED
        assert payload.data["error"] == "Connection error"


class TestRetryLogic:
    """Tests for webhook delivery retry logic."""
    
    @pytest.fixture
    def manager(self, tmp_path):
        """Create a manager with temporary storage."""
        return WebhookManager(storage_path=tmp_path / "webhooks")
    
    @patch("src.tasks.webhooks.urlopen")
    def test_retry_on_failure(self, mock_urlopen, manager):
        """Test retries on delivery failure."""
        # Create successful response mock
        mock_success_response = MagicMock()
        mock_success_response.status = 200
        mock_success_response.read.return_value = b'{"ok": true}'
        mock_success_response.__enter__ = MagicMock(return_value=mock_success_response)
        mock_success_response.__exit__ = MagicMock(return_value=False)
        
        # Fail first two attempts, succeed on third
        mock_urlopen.side_effect = [
            Exception("Connection refused"),
            Exception("Timeout"),
            mock_success_response,
        ]
        
        config = WebhookConfig(
            url="https://example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
            retry_count=3,
            retry_delay_seconds=0.01,  # Fast retries for tests
        )
        manager.register_webhook(config)
        
        # Use internal deliver method to test retries
        delivery = manager._deliver_to_webhook(
            config,
            WebhookPayload(
                event=WebhookEvent.TASK_COMPLETED,
                data={"task_name": "test"},
            ),
        )
        
        # Should eventually succeed after retries
        assert delivery.attempts == 3
    
    @patch("src.tasks.webhooks.urlopen")
    def test_max_retries_exceeded(self, mock_urlopen, manager):
        """Test failure after max retries exceeded."""
        mock_urlopen.side_effect = Exception("Connection refused")
        
        config = WebhookConfig(
            url="https://example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
            retry_count=2,
            retry_delay_seconds=0.01,  # Fast retries for tests
        )
        manager.register_webhook(config)
        
        delivery = manager._deliver_to_webhook(
            config,
            WebhookPayload(
                event=WebhookEvent.TASK_COMPLETED,
                data={"task_name": "test"},
            ),
        )
        
        assert delivery.success is False
        assert delivery.attempts == 2
        assert delivery.error is not None
