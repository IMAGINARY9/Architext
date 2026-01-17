# Migration Guide

**Last Updated:** 2026-01-17

This guide covers the refactor from the monolithic task module and inline server task helpers to the modular task package and task service.

## Overview of Changes

### 1) Tasks are now a package
- **Before:** `src/tasks.py`
- **After:** `src/tasks/` package

**Action:** Update imports to use `src.tasks.<module>` or `from src.tasks import <function>`.

### 2) Task dispatch is centralized
- **Before:** Server/CLI wired tasks individually.
- **After:** `src/task_registry.py` is the single source of truth.

**Action:** Register new tasks in `TASK_REGISTRY` and reference by name.

### 3) Task lifecycle is centralized
- **Before:** Task store helpers lived in `src/server.py`.
- **After:** `src/api/tasks_service.py` contains the task store, validation, and background execution.

**Action:** Use `AnalysisTaskService` for task execution in new endpoints.

### 4) Indexer components split
- **Before:** `src/indexer.py` contained factories and query helpers.
- **After:** `src/indexer_components/factories.py` and `src/indexer_components/querying.py`.

**Compatibility:** `src/indexer.py` retains helper functions (`_build_llm`, `_build_embedding`, `_tokenize`, `_keyword_score`) for test and external compatibility.

## Common Update Examples

### Task imports
**Old**
```python
from src.tasks import analyze_structure
```

**New**
```python
from src.tasks.structure import analyze_structure
# or
from src.tasks import analyze_structure
```

### Adding a task
1. Implement the task in an appropriate `src/tasks/<module>.py`.
2. Export it in `src/tasks/__init__.py` if it should be public.
3. Register it in `src/task_registry.py`.
4. Add CLI wiring if needed in `src/cli.py`.

## Deprecations
- `src/tasks.py` has been removed. Use `src/tasks/` instead.
- Legacy task shims were removed after refactor validation.

## Planned Removals
- Compatibility helpers in `src/indexer.py` will be removed in a future release after downstream callers migrate to `src/indexer_components/`.
