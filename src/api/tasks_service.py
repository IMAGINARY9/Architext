"""Task service helpers for background analysis tasks."""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from src.task_registry import run_task
from src.ingestor import resolve_source


def _is_within_any(candidate: Path, roots: List[Path]) -> bool:
    for root in roots:
        try:
            candidate.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def resolve_task_store_path(raw_path: Optional[str]) -> Path:
    candidate = Path(raw_path or "~/.architext/task_store.json").expanduser().resolve()
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _sanitize_task_store(store: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    sanitized: Dict[str, Dict[str, Any]] = {}
    for task_id, payload in store.items():
        data = {}
        for key, value in payload.items():
            if key == "future":
                continue
            data[key] = value
        sanitized[task_id] = data
    return sanitized


def load_task_store(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    for payload in data.values():
        if payload.get("status") in {"queued", "running"}:
            payload["status"] = "stale"
            payload["note"] = "Server restarted before completion"
    return data


def persist_task_store(path: Path, store: Dict[str, Dict[str, Any]]) -> None:
    try:
        sanitized = _sanitize_task_store(store)
        path.write_text(json.dumps(sanitized, indent=2, default=str), encoding="utf-8")
    except Exception:
        return


class AnalysisTaskService:
    def __init__(
        self,
        task_store: Dict[str, Dict[str, Any]],
        task_store_path: Path,
        lock,
        executor,
        storage_roots: List[Path],
        source_roots: List[Path],
        base_settings,
    ) -> None:
        self.task_store = task_store
        self.task_store_path = task_store_path
        self.lock = lock
        self.executor = executor
        self.storage_roots = storage_roots
        self.source_roots = source_roots
        self.base_settings = base_settings

    def update_task(self, task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            existing = self.task_store.get(task_id, {})
            existing.update(payload)
            self.task_store[task_id] = existing
            persist_task_store(self.task_store_path, self.task_store)
            return dict(existing)

    def resolve_storage_path(self, raw_path: Optional[str]) -> str:
        candidate = Path(raw_path or self.base_settings.storage_path).expanduser().resolve()
        if not _is_within_any(candidate, self.storage_roots):
            raise HTTPException(status_code=400, detail="storage must be within allowed roots")
        return str(candidate)

    def resolve_output_dir(self, raw_path: Optional[str]) -> Optional[str]:
        if not raw_path:
            return None
        candidate = Path(raw_path).expanduser().resolve()
        if not _is_within_any(candidate, self.source_roots):
            raise HTTPException(status_code=400, detail="output_dir must be within allowed roots")
        return str(candidate)

    def resolve_task_source(self, raw_path: Optional[str]) -> Optional[str]:
        if not raw_path:
            return None
        try:
            candidate = resolve_source(raw_path, use_cache=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not candidate.exists():
            raise HTTPException(status_code=400, detail="source path does not exist")
        if not candidate.is_dir():
            raise HTTPException(status_code=400, detail="source path must be a directory")
        if not _is_within_any(candidate, self.source_roots):
            raise HTTPException(status_code=400, detail="source path must be within allowed roots")
        return str(candidate)

    def run_analysis_task(self, task_name: str, payload: Any, progress_update=None) -> Dict[str, Any]:
        if task_name == "impact-analysis" and not getattr(payload, "module", None):
            raise ValueError("module is required for impact analysis")

        storage_path = self.resolve_storage_path(getattr(payload, "storage", None))
        source_path = self.resolve_task_source(getattr(payload, "source", None))
        return run_task(
            task_name,
            storage_path=storage_path if not getattr(payload, "source", None) else None,
            source_path=source_path,
            output_format=getattr(payload, "output_format", "json"),
            depth=getattr(payload, "depth", "shallow") or "shallow",
            module=getattr(payload, "module", None),
            output_dir=self.resolve_output_dir(getattr(payload, "output_dir", None)),
            progress_callback=progress_update,
        )

    def submit_analysis_task(self, task_name: str, payload: Any, task_id: str) -> str:
        self.update_task(task_id, {"status": "queued", "task": task_name})

        def _run():
            def progress_update(info: Dict[str, Any]):
                self.update_task(task_id, {"progress": info})

            self.update_task(task_id, {"status": "running"})
            try:
                result = self.run_analysis_task(task_name, payload, progress_update=progress_update)
                self.update_task(task_id, {"status": "completed", "result": result})
            except Exception as exc:  # pylint: disable=broad-except
                self.update_task(
                    task_id,
                    {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()},
                )

        future = self.executor.submit(_run)
        future.add_done_callback(lambda fut: fut.result() if not fut.cancelled() else None)
        self.update_task(task_id, {"future": future})
        return task_id
