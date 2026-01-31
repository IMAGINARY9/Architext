# Architext Tasks Analysis & Refactoring Plan

**Date:** January 31, 2026  
**Initial Tasks Analyzed:** 20  
**Current Tasks:** 15 ✅

---

## Executive Summary

This document provides a comprehensive analysis of all default project tasks, identifies issues, and tracks the refactoring progress.

### 🎉 Refactoring Status: PHASE 9 COMPLETE ✅

| Metric | Before | After |
|--------|--------|-------|
| Total Tasks | 20 | 15 |
| Tasks Deleted | - | 5 |
| Tasks Improved | 0 | 6 |
| Tests | 58 | 306 |

### Changes Made:
- **Deleted 5 tasks** (low value / project-specific / redundant)
- **Improved `detect-patterns`** - added confidence scoring, more patterns
- **Renamed `test-coverage` → `test-mapping`** - improved implementation
- **Enhanced `code-knowledge-graph`** - added JS/TS support via tree-sitter
- **Enhanced `dependency-graph`** - added DOT/Graphviz export format
- **Added TypedDict types** - proper type definitions for all task results
- **Added TaskContext** - shared context with caching for multi-task execution
- **Added parallel execution** - run multiple tasks concurrently with shared caching
- **Added task categories** - group related tasks for batch execution
- **Added disk caching** - persistent task result cache with TTL and source-change detection
- **Added BaseTask class** - reusable base class for task implementation
- **Added execution history** - track all task runs with analytics
- **Added task pipelines** - compose and run task sequences
- **Added metrics dashboard** - comprehensive execution analytics
- **Added task recommendations** - intelligent task suggestions
- **Added webhooks/notifications** - event-driven task notifications
- **Added task scheduling** - automated task execution with cron support
- **Added customizable scoring weights** - configurable recommendation scoring
- **Added extended BaseTask migrations** - 8 additional task classes

---

## 📋 Current Task List (15 Tasks)

| Task Name | Module | Purpose | Status |
|-----------|--------|---------|--------|
| `analyze-structure` | structure.py | Repository file tree & language stats | ✅ Active |
| `tech-stack` | tech_stack.py | Detect frameworks & languages | ✅ Active |
| `detect-anti-patterns` | anti_patterns.py | Find code smells & structural issues | ✅ Active |
| `health-score` | health.py | Calculate overall codebase health | ✅ Active |
| `impact-analysis` | architecture.py | Find affected modules when changing code | ✅ Active |
| `dependency-graph` | architecture.py | Export module dependency graph | ✅ Active |
| `test-mapping` | quality.py | Map test files to source modules | ✅ **Renamed & Improved** |
| `detect-patterns` | architecture.py | Detect architecture patterns with confidence | ✅ **Improved** |
| `detect-vulnerabilities` | security.py | Security scanning with semantic queries | ✅ Active |
| `identify-silent-failures` | quality.py | Find swallowed exceptions | ✅ Active |
| `security-heuristics` | security.py | Regex + AST security scanning | ✅ Active |
| `code-knowledge-graph` | architecture.py | Build call graph from AST | ✅ Active |
| `synthesis-roadmap` | roadmap.py | Aggregate all findings into roadmap | ✅ Active |
| `detect-duplication` | duplication.py | Find exact duplicate code blocks | ✅ Active |
| `detect-duplication-semantic` | duplication.py | Find semantically similar functions | ✅ Active |

---

## ❌ Deleted Tasks (5 Tasks)

| Task | Reason for Removal | Alternative |
|------|-------------------|-------------|
| `diff-architecture` | Trivial set difference - git diff does this better | Use `git diff --name-status` |
| `onboarding-guide` | Returned hardcoded generic suggestions, no analysis | Manual README review |
| `refactoring-recommendations` | Redundant with `synthesis-roadmap` which does it better | Use `synthesis-roadmap` |
| `generate-docs` | Just dumped other task outputs to files, no real generation | Users can save outputs directly |
| `logic-gap-analysis` | Hardcoded to `ArchitextSettings` - project-specific, not reusable | External tool / future project |

### Deleted Files
- `src/tasks/docs.py` - Entire module removed

---

## ✅ Improvements Made

### 1. `detect-patterns` - Enhanced with Confidence Scoring

**Before:** Simple string matching in file paths with high false positive rate

```python
# Old implementation - prone to false positives
if any("controllers" in path for path in files_lower):
    patterns.append("MVC")
if any("docker" in path for path in files_lower):
    patterns.append("Microservices")  # docker != microservices!
```

**After:** Multi-signal pattern detection with confidence scores

**New Features:**
- ✅ Confidence scoring (0.0-1.0) based on evidence strength
- ✅ Evidence collection for each detected pattern  
- ✅ Added more patterns: Layered Architecture, Hexagonal/Ports-Adapters, CQRS
- ✅ Required + optional signal separation (prevents false positives)
- ✅ Fixed docker != microservices false positive

**Output Format:**
```json
{
  "patterns": [
    {
      "name": "MVC",
      "confidence": 0.85,
      "evidence": ["controllers/", "models/", "views/"]
    }
  ]
}
```

---

### 2. `test-mapping` - Renamed and Improved (was `test-coverage`)

**Before:** Misleading name, naive stem matching

```python
# Old implementation - too loose matching
for src in source_files:
    stem = Path(src).stem
    for test in test_files:
        if stem in Path(test).stem:  # "config" matches "test_config_manager"
            mapping[src].append(test)
```

**After:** Clear naming, improved test file detection

**New Features:**
- ✅ Better test file detection using multiple patterns (directory + filename)
- ✅ Common test naming pattern extraction (`test_X.py`, `X_test.py`, `tests/test_X.py`)
- ✅ Skip untestable files (`__init__.py`, `conftest.py`)
- ✅ Clear disclaimer that this is NOT actual code coverage
- ✅ Coverage percentage calculation (files with tests / total testable files)

**Name Change Rationale:**
The old name `test-coverage` implied actual code coverage metrics (line/branch coverage), which it never provided. The new name `test-mapping` accurately describes what it does - mapping test files to source files.

---

## 📁 Files Modified

```
src/
├── task_registry.py          # Removed 5 task registrations ✅
│                             # Added TASK_DEPENDENCIES, TASK_CATEGORIES ✅
│                             # Added run_tasks_parallel(), run_category() ✅
├── server.py                 # Updated backward-compatible imports ✅
├── api/
│   └── tasks.py              # Removed 5 routes, renamed test-mapping ✅
│                             # Added /tasks/categories endpoint ✅
│                             # Added /tasks/run-parallel endpoint ✅
│                             # Added /tasks/run-category/{category} endpoint ✅
├── tasks/
│   ├── __init__.py           # Updated exports, added type exports ✅
│   │                         # Added TaskContext, task_context exports ✅
│   ├── types.py              # NEW: TypedDict definitions for all tasks ✅
│   ├── shared.py             # Added TaskContext class and context manager ✅
│   ├── architecture.py       # Removed diff_architecture_review, onboarding_guide ✅
│   │                         # Improved architecture_pattern_detection ✅
│   │                         # Added JS/TS support to code_knowledge_graph ✅
│   │                         # Added DOT format to dependency_graph_export ✅
│   ├── quality.py            # Removed refactoring_recommendations, logic_gap_analysis ✅
│   │                         # Renamed test_coverage → test_mapping_analysis ✅
│   ├── roadmap.py            # Removed logic_gap_analysis import ✅
│   │                         # Added TaskContext usage for caching ✅
│   └── docs.py               # DELETED ✅

tests/
├── test_server.py            # Removed tests for deleted tasks ✅
│                             # Updated test-mapping test ✅
```

---

## ✅ Phase 3 Improvements

### 1. JS/TS Support in `code-knowledge-graph`

The knowledge graph now supports JavaScript and TypeScript files via tree-sitter parsing:

**Supported Languages:**
- Python (.py) - Full AST parsing
- JavaScript (.js, .jsx) - Tree-sitter parsing
- TypeScript (.ts, .tsx) - Tree-sitter parsing

**New Output Fields:**
```json
{
  "nodes": [...],
  "edges": [...],
  "languages_parsed": {"python": 30, "javascript": 5, "typescript": 3},
  "total_nodes": 308,
  "total_edges": 500
}
```

Each node now includes a `language` field indicating its source language.

### 2. DOT Format for `dependency-graph`

Added Graphviz DOT format export:

```bash
# Use via API
POST /tasks/dependency-graph
{"source_path": "src", "output_format": "dot"}
```

**Output:**
```dot
digraph dependencies {
  rankdir=LR;
  node [shape=box];
  "server" -> "architecture";
  ...
}
```

### 3. TypedDict Definitions

Created [types.py](../src/tasks/types.py) with comprehensive type definitions:

- `StructureResult`, `HealthResult`, `TechStackResult`
- `PatternDetectionResult`, `ImpactAnalysisResult`, `DependencyGraphResult`
- `KnowledgeGraphResult`, `TestMappingResult`, `AntiPatternResult`
- `SecurityHeuristicsResult`, `VulnerabilityResult`
- `DuplicationResult`, `SemanticDuplicationResult`
- `SilentFailureResult`, `SynthesisRoadmapResult`
- `TaskContext` - For future shared context implementation

---

## ✅ Phase 4 Improvements

### TaskContext - Shared Caching for Multi-Task Execution

Added a `TaskContext` class that caches file collections and parsed data across multiple task invocations:

**Usage:**
```python
from src.tasks import TaskContext, task_context, synthesis_roadmap

# Option 1: Use context manager (recommended)
with task_context(source_path="src") as ctx:
    # All tasks within this block share cached files
    result = synthesis_roadmap(source_path="src")

# Option 2: Manual context management
ctx = TaskContext(source_path="src")
files = ctx.get_files()  # Cached after first call
content = ctx.get_file_content(path)  # Cached per file
ast_tree = ctx.get_parsed_ast(path)  # Cached Python AST
```

**Benefits:**
- File collection happens once, shared across tasks
- `synthesis-roadmap` now uses context internally (6 sub-tasks share cache)
- Thread-safe with locking for concurrent access
- Reduces I/O overhead for large codebases

**TaskContext Methods:**
| Method | Description |
|--------|-------------|
| `get_files()` | Get all file paths (cached) |
| `get_file_content(path)` | Get file content (cached) |
| `get_parsed_ast(path)` | Get Python AST (cached) |
| `get_import_graph()` | Get import graph (cached) |
| `clear_cache()` | Clear all cached data |

### Parameter Consistency Verification

All 15 tasks now have consistent signatures:
- `storage_path: Optional[str]` - Path to ChromaDB storage
- `source_path: Optional[str]` - Path to source directory  
- `progress_callback` - Optional callback for progress updates

---

## ✅ Phase 5 Improvements

### Task Categories

Tasks are now organized into logical categories:

| Category | Tasks |
|----------|-------|
| `structure` | analyze-structure, tech-stack, detect-patterns |
| `quality` | detect-anti-patterns, health-score, test-mapping, identify-silent-failures |
| `security` | detect-vulnerabilities, security-heuristics |
| `duplication` | detect-duplication, detect-duplication-semantic |
| `architecture` | impact-analysis, dependency-graph, code-knowledge-graph |
| `synthesis` | synthesis-roadmap |

### Task Dependency Graph

A dependency graph tracks which tasks depend on others:

```python
TASK_DEPENDENCIES = {
    "synthesis-roadmap": {
        "detect-anti-patterns", "health-score", 
        "identify-silent-failures", "security-heuristics",
        "detect-duplication", "detect-duplication-semantic",
    },
    # All other tasks are independent
}
```

### Parallel Task Execution

New functions for concurrent task execution:

```python
from src.task_registry import run_tasks_parallel, run_category

# Run multiple tasks in parallel
results = run_tasks_parallel(
    ["analyze-structure", "tech-stack", "detect-patterns"],
    source_path="src",
    max_workers=4,
)

# Run all tasks in a category
results = run_category("quality", source_path="src")
```

**Benefits:**
- Tasks run concurrently using ThreadPoolExecutor
- Shared TaskContext caches files across all tasks
- Pre-warms file cache before starting parallel execution
- Progress callbacks supported for monitoring

### New API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tasks/categories` | GET | List all task categories |
| `/tasks/run-parallel` | POST | Run multiple tasks in parallel |
| `/tasks/run-category/{category}` | POST | Run all tasks in a category |

**Example: Run parallel tasks via API**
```bash
curl -X POST http://localhost:8000/tasks/run-parallel \
  -H "Content-Type: application/json" \
  -d '{"tasks": ["analyze-structure", "tech-stack"], "source": "./src"}'
```

**Example: Run category via API**
```bash
curl -X POST http://localhost:8000/tasks/run-category/quality \
  -H "Content-Type: application/json" \
  -d '{"source": "./src"}'
```

---

## ✅ Phase 6 Improvements

### Task Result Caching

Added persistent disk caching for task results with automatic invalidation:

**Features:**
- ✅ Disk persistence (JSON files in `~/.architext/cache/`)
- ✅ TTL-based expiration (default: 1 hour)
- ✅ Source file change detection (invalidates cache when files change)
- ✅ Memory + disk hybrid caching
- ✅ Thread-safe access
- ✅ Configurable via decorator or direct API

**Usage:**
```python
from src.tasks.cache import TaskResultCache, cached_task, get_task_cache

# Direct cache usage
cache = get_task_cache()
result = cache.get("analyze-structure", source_path="src")
if result is None:
    result = run_analysis()
    cache.set("analyze-structure", result, source_path="src")

# Or use the decorator
@cached_task(ttl=1800)  # 30 minute cache
def my_analysis(source_path=None, storage_path=None, progress_callback=None):
    # Expensive computation
    return {"result": "data"}

# Run task with caching
result = run_task("analyze-structure", source_path="src", use_cache=True)
```

### Cache API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tasks/cache/stats` | GET | Get cache statistics |
| `/tasks/cache/clear` | POST | Clear cache (optionally by task name) |

**Example: Get cache stats**
```bash
curl http://localhost:8000/tasks/cache/stats
# Returns: {"enabled": true, "memory_entries": 5, "disk_entries": 5, "disk_size_bytes": 12345}
```

**Example: Clear cache**
```bash
curl -X POST http://localhost:8000/tasks/cache/clear \
  -H "Content-Type: application/json" \
  -d '{"task_name": "analyze-structure"}'
```

### BaseTask Abstract Class

Added a reusable base class for implementing new tasks:

```python
from src.tasks.base import BaseTask, FileInfo, PYTHON_EXTENSIONS

class MyCustomTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(extensions=PYTHON_EXTENSIONS, **kwargs)
    
    def analyze(self, files: List[FileInfo]) -> Dict[str, Any]:
        results = []
        for file in files:
            if file.ast_tree:
                # Analyze Python AST
                pass
        return {"results": results}

# Usage
task = MyCustomTask(source_path="src")
result = task.run()
```

**Benefits:**
- Automatic file collection and filtering
- Progress reporting
- Context-aware caching integration
- Standardized error handling

### Parallel Execution Fix

Fixed infinite loading issue in parallel task execution:

**Problem:** Thread-local storage (`threading.local()`) wasn't propagated to ThreadPoolExecutor worker threads.

**Solution:** 
- Create `TaskContext` explicitly before spawning threads
- Pass context to each worker via `set_current_context(ctx)` 
- Pre-warm file cache synchronously before parallel execution
- Add timeout handling to prevent hangs

---

## ✅ Phase 7 Improvements

### Task Execution History

Added automatic tracking of all task executions with analytics:

**Features:**
- ✅ Automatic tracking of every task execution
- ✅ Tracks success/failure status, duration, cache hits
- ✅ Disk persistence (`~/.architext/history/executions.json`)
- ✅ Configurable max history entries (default: 1000)
- ✅ Analytics: success rate, average duration, cache hit rate
- ✅ Filter history by task name, status, date range

**Usage:**
```python
from src.tasks.history import get_task_history, TaskExecution

# Get singleton history instance
history = get_task_history()

# Get all history
executions = history.get_history()

# Filter by task name
pattern_history = history.get_history(task_name="detect-patterns")

# Get analytics for a task
analytics = history.get_analytics("detect-patterns")
print(f"Success rate: {analytics.success_rate}%")
print(f"Avg duration: {analytics.average_duration_seconds}s")
print(f"Cache hit rate: {analytics.cache_hit_rate}%")

# Tracking is automatic when using run_task()
result = run_task("analyze-structure", source_path="src")  # Tracked!

# Or manually with context manager
with history.track("my-task") as tracker:
    # do work
    tracker.set_cached(False)
```

### Task Pipelines

Added custom task pipelines for composing multiple tasks:

**Features:**
- ✅ Sequential and parallel step execution
- ✅ Error handling modes: stop, skip, or continue
- ✅ 5 built-in pipelines: quick-scan, full-analysis, security-audit, code-quality, architecture-review
- ✅ Custom pipeline persistence (`~/.architext/pipelines/`)
- ✅ Pipeline execution results with per-task status

**Built-in Pipelines:**
| Pipeline | Tasks | Description |
|----------|-------|-------------|
| `quick-scan` | structure, tech-stack, health | Fast overview of codebase |
| `full-analysis` | 11 tasks | Comprehensive analysis |
| `security-audit` | vulnerabilities, security-heuristics, silent-failures | Security-focused scan |
| `code-quality` | anti-patterns, duplication, quality, health | Code quality metrics |
| `architecture-review` | structure, dependencies, graph, impact-analysis | Architecture analysis |

**Usage:**
```python
from src.tasks.pipeline import (
    PipelineExecutor, PipelineStore, TaskPipeline, PipelineStep, ParallelGroup,
    get_builtin_pipeline, BUILTIN_PIPELINES
)

# Run a built-in pipeline
executor = PipelineExecutor(source_path="src", storage_path="storage")
result = executor.execute(get_builtin_pipeline("quick-scan"))
print(f"Success: {result.success}")
print(f"Duration: {result.total_duration_seconds}s")

# Create a custom pipeline
custom = TaskPipeline(
    name="my-pipeline",
    description="Custom analysis",
    steps=[
        PipelineStep(task_name="analyze-structure"),
        ParallelGroup(steps=[
            PipelineStep(task_name="tech-stack"),
            PipelineStep(task_name="detect-patterns"),
        ]),
        PipelineStep(task_name="health-score", on_error="skip"),
    ]
)

# Save custom pipeline
store = PipelineStore()
store.save(custom)

# List all pipelines
all_pipelines = store.list_pipelines()
```

### History & Pipeline API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tasks/history` | GET | Get task execution history |
| `/tasks/history` | DELETE | Clear history (optionally by task) |
| `/tasks/history/analytics` | GET | Get analytics for a task |
| `/tasks/pipelines` | GET | List all pipelines (built-in + custom) |
| `/tasks/pipelines` | POST | Create a custom pipeline |
| `/tasks/pipelines/{id}` | GET | Get pipeline by ID |
| `/tasks/pipelines/{id}` | DELETE | Delete a custom pipeline |
| `/tasks/pipelines/{id}/run` | POST | Execute a pipeline |

**Example: Get execution history**
```bash
curl "http://localhost:8000/tasks/history?task_name=analyze-structure&limit=10"
```

**Example: Get task analytics**
```bash
curl "http://localhost:8000/tasks/history/analytics?task_name=detect-patterns"
# Returns: {"task_name": "detect-patterns", "total_executions": 15, "success_rate": 93.3, ...}
```

**Example: Run a pipeline**
```bash
curl -X POST http://localhost:8000/tasks/pipelines/quick-scan/run \
  -H "Content-Type: application/json" \
  -d '{"source": "./src"}'
# Returns: {"success": true, "total_duration_seconds": 2.5, "step_results": [...]}
```

**Example: Create custom pipeline**
```bash
curl -X POST http://localhost:8000/tasks/pipelines \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-custom-pipeline",
    "description": "Quick security check",
    "steps": [
      {"task_name": "detect-vulnerabilities"},
      {"task_name": "security-heuristics", "on_error": "skip"}
    ]
  }'
```

---

## ✅ Phase 8 Improvements

### BaseTask-Based Task Implementations

Migrated several tasks to use the `BaseTask` class pattern for cleaner, more maintainable code:

**New Task Classes (`src/tasks/tasks_v2.py`):**
- `AntiPatternDetectionTask` - Detects code smells and anti-patterns
- `SilentFailuresTask` - Finds silent exception handlers
- `TestMappingTask` - Maps test files to source files
- `HealthScoreTask` - Calculates codebase health score

**Benefits:**
- Automatic file collection and filtering
- Built-in progress reporting
- Context-aware caching integration
- Standardized error handling
- Cleaner separation of concerns

**Usage:**
```python
from src.tasks.tasks_v2 import AntiPatternDetectionTask, HealthScoreTask

# Class-based usage
task = AntiPatternDetectionTask(source_path="src")
result = task.run()

# Or use wrapper functions for backward compatibility
from src.tasks import detect_anti_patterns_v2, health_score_v2
result = detect_anti_patterns_v2(source_path="src")
```

### Task Recommendation Engine

Added intelligent task recommendations based on execution history:

**Features:**
- ✅ Score-based ranking (0-100)
- ✅ Never-run task boosting
- ✅ Stale task detection (24h/7d thresholds)
- ✅ Success rate analysis
- ✅ Category preference support
- ✅ Related task suggestions
- ✅ Quick-scan recommendations
- ✅ Configurable scoring weights

**Usage:**
```python
from src.tasks.recommendations import (
    get_recommendation_engine, get_task_recommendations
)

# Get top 5 recommendations
recommendations = get_task_recommendations(limit=5)

# Get category-specific recommendations
engine = get_recommendation_engine()
security_recs = engine.get_category_recommendations("security", limit=3)

# Get related task suggestions
related = engine.get_related_recommendations("health-score")

# Quick scan recommendations (essential tasks)
quick = engine.get_quick_scan_recommendation()
```

### Metrics Dashboard

Added comprehensive task execution metrics and analytics:

**Features:**
- ✅ Overall success rate and cache hit rate
- ✅ Per-task metrics (executions, duration, success rate)
- ✅ Per-category metrics with task coverage
- ✅ Daily execution trends
- ✅ Top performers (most run, fastest, slowest)
- ✅ Health indicators (never-run tasks, failing tasks)

**Usage:**
```python
from src.tasks.metrics import get_dashboard_metrics, get_metrics_dashboard

# Get full dashboard
metrics = get_dashboard_metrics(days=30)
print(f"Total executions: {metrics['summary']['total_executions']}")
print(f"Success rate: {metrics['summary']['overall_success_rate']}%")

# Get details for specific task
dashboard = get_metrics_dashboard()
task_details = dashboard.get_task_details("analyze-structure", days=7)
```

### Recommendations & Metrics API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tasks/recommendations` | GET | Get task recommendations |
| `/tasks/recommendations/quick-scan` | GET | Get essential quick-scan tasks |
| `/tasks/recommendations/category/{category}` | GET | Get category recommendations |
| `/tasks/recommendations/related/{task_name}` | GET | Get related task suggestions |
| `/tasks/metrics/dashboard` | GET | Get full metrics dashboard |
| `/tasks/metrics/task/{task_name}` | GET | Get metrics for specific task |
| `/tasks/metrics/summary` | GET | Get metrics summary only |
| `/tasks/metrics/trends` | GET | Get daily execution trends |
| `/tasks/metrics/categories` | GET | Get per-category metrics |

**Example: Get recommendations**
```bash
curl "http://localhost:8000/tasks/recommendations?limit=5&category=quality"
# Returns: {"recommendations": [...], "count": 5}
```

**Example: Get dashboard**
```bash
curl "http://localhost:8000/tasks/metrics/dashboard?days=7"
# Returns: {"summary": {...}, "task_metrics": [...], "daily_trends": [...]}
```

**Example: Get task metrics**
```bash
curl "http://localhost:8000/tasks/metrics/task/analyze-structure"
# Returns: {"task_name": "...", "metrics": {...}, "duration": {...}, "recent_executions": [...]}
```

---

## 🔧 Remaining Code Quality Issues

These are recommendations for future improvement:

### 1. **Inconsistent Parameter Handling**
```python
# Some tasks use storage_path, some source_path, some both
def task_a(storage_path=None, source_path=None): ...
def task_b(storage_path=None): ...  # Missing source_path
```
**Recommendation:** Standardize all tasks to accept both parameters uniformly.

### 2. **Duplicated File Collection Logic**
Every task calls `collect_file_paths()` separately.

**Recommendation:** Create a task execution context that pre-collects files once.

### 3. **No Caching Between Related Tasks**
`synthesis_roadmap` calls 7 other tasks, each re-scanning files.

**Recommendation:** Implement result caching or shared context between tasks.

### 4. **Missing Type Hints on Return Values**
```python
def analyze_structure(...) -> Dict[str, Any]:  # Too generic
```
**Recommendation:** Create TypedDict or dataclasses for task results.

---

## 📊 Future Enhancement Ideas

These tasks are good but could be improved further:

| Task | Potential Enhancement |
|------|----------------------|
| `analyze-structure` | Add file size analysis, gitignore awareness |
| `detect-anti-patterns` | Add severity aggregation, trend tracking |
| `health-score` | Add historical comparison, configurable weights |
| `impact-analysis` | Add depth limiting, impact severity levels |
| `dependency-graph` | Add DOT format, clustering by package |
| `detect-vulnerabilities` | Add CVSS scoring, remediation suggestions |
| `security-heuristics` | Add rule categories, false positive suppression |
| `identify-silent-failures` | Add exception type analysis |
| `code-knowledge-graph` | Add JavaScript/TypeScript support via tree-sitter |
| `synthesis-roadmap` | Add time estimates, dependency ordering |

---

## 📝 Implementation Checklist

### Phase 1: Cleanup ✅ DONE
- [x] Remove `diff-architecture` task
- [x] Remove `onboarding-guide` task
- [x] Remove `refactoring-recommendations` task
- [x] Remove `generate-docs` task (entire docs.py module)
- [x] Remove `logic-gap-analysis` task
- [x] Update task registry imports
- [x] Update API routes
- [x] Update tests
- [x] All 58 tests passing

### Phase 2: Core Improvements ✅ DONE
- [x] Rewrite `detect-patterns` with confidence scores
- [x] Rename `test-coverage` to `test-mapping`
- [x] Improve test file detection logic

### Phase 3: Enhanced Capabilities ✅ DONE
- [x] Add JS/TS support to `code-knowledge-graph` (using tree-sitter)
- [x] Add DOT/Graphviz format to `dependency-graph`
- [x] Create TypedDict definitions for all task return values
- [x] Add `language` field to knowledge graph nodes

### Phase 4: Performance & Consistency ✅ DONE
- [x] Add shared TaskContext with file caching
- [x] Update `synthesis-roadmap` to use TaskContext for caching
- [x] Verify all tasks have consistent parameters (storage_path, source_path, progress_callback)
- [x] Export TaskContext and task_context from package

### Phase 5: Parallel Execution & Categories ✅ DONE
- [x] Add task dependency graph (TASK_DEPENDENCIES)
- [x] Add task categories (TASK_CATEGORIES)
- [x] Add `run_tasks_parallel()` for concurrent task execution
- [x] Add `run_category()` for running all tasks in a category
- [x] Add API endpoints: `/tasks/categories`, `/tasks/run-parallel`, `/tasks/run-category/{category}`

### Phase 6: Caching & Infrastructure ✅ DONE
- [x] Add task result caching across sessions (persist to disk)
- [x] Add `TaskResultCache` class with TTL and source-change invalidation
- [x] Add `@cached_task` decorator for easy caching
- [x] Add cache API endpoints: `/tasks/cache/stats`, `/tasks/cache/clear`
- [x] Fix parallel execution thread-local context propagation
- [x] Add `BaseTask` abstract class for task implementation
- [x] Add file utility functions (filter_files_by_extension, get_test_files, etc.)
- [x] Add 33 new tests (12 parallel + 16 cache + 5 API = 91 total)

### Phase 7: History & Pipelines ✅ DONE
- [x] Add task execution history and analytics
- [x] Add `TaskExecutionHistory` class with disk persistence
- [x] Add `TaskExecution` and `TaskAnalytics` dataclasses
- [x] Add `ExecutionTracker` context manager for automatic tracking
- [x] Integrate history tracking into `run_task()` function
- [x] Add custom task composition (user-defined task pipelines)
- [x] Add `TaskPipeline`, `PipelineStep`, `ParallelGroup` dataclasses
- [x] Add `PipelineExecutor` for running sequential/parallel pipelines
- [x] Add `PipelineStore` for persisting custom pipelines
- [x] Add 5 built-in pipelines: quick-scan, full-analysis, security-audit, code-quality, architecture-review
- [x] Add API endpoints for history: `/tasks/history`, `/tasks/history/analytics`
- [x] Add API endpoints for pipelines: `/tasks/pipelines`, `/tasks/pipelines/{id}`, `/tasks/pipelines/{id}/run`
- [x] Add 52 new tests (21 history + 31 pipeline = 144 total)

### Phase 8: Metrics & Recommendations ✅ DONE
- [x] Migrate existing tasks to use BaseTask class
- [x] Create `tasks_v2.py` with class-based task implementations
- [x] Add `AntiPatternDetectionTask`, `SilentFailuresTask`, `TestMappingTask`, `HealthScoreTask` classes
- [x] Add backward-compatible wrapper functions (`detect_anti_patterns_v2`, etc.)
- [x] Add task execution metrics dashboard (`metrics.py`)
- [x] Add `MetricsDashboard`, `TaskMetrics`, `ExecutionTrend`, `DashboardMetrics` classes
- [x] Add per-task and per-category metrics aggregation
- [x] Add daily execution trend analysis
- [x] Add task recommendation engine (`recommendations.py`)
- [x] Add `TaskRecommendationEngine` with configurable scoring
- [x] Add quick-scan, category-based, and related task recommendations
- [x] Add API endpoints for recommendations and metrics dashboard
- [x] Add 52 new tests (21 tasks_v2 + 15 recommendations + 16 metrics = 195 total)

### Phase 9: Webhooks, Scheduling & Extended Tasks ✅ DONE
- [x] Add webhooks/notifications system (`webhooks.py`)
- [x] Add `WebhookEvent`, `WebhookConfig`, `WebhookPayload`, `WebhookDelivery` classes
- [x] Add `WebhookManager` with registration, emit, signature generation, persistence
- [x] Add task scheduling/automation (`scheduler.py`)
- [x] Add `ScheduleType`, `TaskSchedule`, `ScheduledRun` classes
- [x] Add `TaskScheduler` with cron parsing, APScheduler integration
- [x] Add customizable scoring weights for recommendations
- [x] Add `ScoringWeights`, `ScoringWeightsStore` with 5 presets
- [x] Add API endpoints for webhooks: `/webhooks`, `/webhooks/{id}`, `/webhooks/test`
- [x] Add API endpoints for scheduling: `/schedules`, `/schedules/{id}`, `/schedules/{id}/run`
- [x] Add API endpoints for scoring weights: `/recommendations/weights`, `/weights/presets`
- [x] Complete extended BaseTask migrations (`tasks_v2_extended.py`)
- [x] Add 8 new task classes: `StructureAnalysisTask`, `TechStackTask`, `ArchitecturePatternTask`, `ImpactAnalysisTask`, `DependencyGraphTask`, `DuplicateBlocksTask`, `SemanticDuplicationTask`, `SecurityHeuristicsTask`
- [x] Add 111 new tests (30 webhooks + 32 scheduler + 23 scoring + 26 extended = 306 total)

### Phase 10: Final Cleanup & Organization (Planned)
- [ ] Consolidate v2/extended files into canonical task modules
- [ ] Remove deprecated wrapper functions after migration
- [ ] Reorganize tasks folder structure by domain
- [ ] Update all imports to use new canonical paths
- [ ] Remove redundant code and unused utilities
- [ ] Finalize public API exports in `__init__.py`
- [ ] Update documentation for final architecture
- [ ] Final test suite validation

---

## Conclusion

**Final Statistics:**
- Started with: 20 tasks
- Deleted: 5 tasks (25%)
- Improved: 6 tasks (30%)
- Final count: 15 tasks

---

## 📋 Phase Summary (Phases 1-9)

| Phase | Focus | Key Deliverables | Tests Added |
|-------|-------|------------------|-------------|
| **1** | Cleanup | Removed 5 low-value tasks, deleted docs.py | 58 baseline |
| **2** | Core Improvements | `detect-patterns` confidence scoring, renamed `test-mapping` | — |
| **3** | Enhanced Capabilities | JS/TS support, DOT format, TypedDict definitions | — |
| **4** | Performance | `TaskContext` with file caching, consistent parameters | — |
| **5** | Parallel Execution | Task categories, `run_tasks_parallel()`, category APIs | — |
| **6** | Caching Infrastructure | Disk cache with TTL, `BaseTask` class, `@cached_task` | +33 → 91 |
| **7** | History & Pipelines | Execution tracking, 5 built-in pipelines, custom pipelines | +52 → 144 |
| **8** | Metrics & Recommendations | Dashboard analytics, smart recommendations, 4 BaseTask classes | +51 → 195 |
| **9** | Webhooks, Scheduling & Extended Tasks | Event notifications, cron scheduling, scoring weights, 8 more BaseTask classes | +111 → 306 |

---

## 🧹 Phase 10: Final Cleanup & Organization (PLANNED)

Phase 10 consolidates the refactoring work into a clean, production-ready codebase by removing deprecated files, reorganizing the folder structure, and establishing canonical imports.

### Goals
1. **Eliminate v2/extended proliferation** - Merge into canonical modules
2. **Clean folder structure** - Organize by domain/responsibility
3. **Single source of truth** - One implementation per task
4. **Clean public API** - Minimal, well-documented exports

### 10.1 File Consolidation

**Current State (messy):**
```
src/tasks/
├── anti_patterns.py       # Original implementation
├── tasks_v2.py           # AntiPatternDetectionTask (duplicate)
├── tasks_v2_extended.py  # 8 more BaseTask classes
├── architecture.py       # Original functions
├── quality.py            # Original functions
├── structure.py          # Original functions (mostly unused?)
├── ... (24 files total)
```

**Target State (clean):**
```
src/tasks/
├── core/                    # Core infrastructure
│   ├── __init__.py
│   ├── base.py              # BaseTask, FileInfo, utilities
│   ├── cache.py             # TaskResultCache, @cached_task
│   ├── context.py           # TaskContext (from shared.py)
│   └── types.py             # TypedDict definitions
│
├── analysis/                # Analysis task implementations
│   ├── __init__.py
│   ├── anti_patterns.py     # AntiPatternDetectionTask (merged)
│   ├── architecture.py      # ArchitecturePatternTask, ImpactAnalysisTask, DependencyGraphTask
│   ├── duplication.py       # DuplicateBlocksTask, SemanticDuplicationTask
│   ├── health.py            # HealthScoreTask
│   ├── quality.py           # SilentFailuresTask, TestMappingTask
│   ├── security.py          # SecurityHeuristicsTask, VulnerabilityTask
│   ├── structure.py         # StructureAnalysisTask
│   └── tech_stack.py        # TechStackTask
│
├── orchestration/           # Execution infrastructure
│   ├── __init__.py
│   ├── history.py           # TaskExecutionHistory
│   ├── metrics.py           # MetricsDashboard
│   ├── pipeline.py          # PipelineExecutor, PipelineStore
│   ├── recommendations.py   # TaskRecommendationEngine, ScoringWeights
│   ├── scheduler.py         # TaskScheduler
│   └── webhooks.py          # WebhookManager
│
├── __init__.py              # Clean public API exports
└── registry.py              # Task registry (moved from task_registry.py)
```

### 10.2 Files to Delete

| File | Reason | Replacement |
|------|--------|-------------|
| `tasks_v2.py` | Temporary migration file | Merge into `analysis/*.py` |
| `tasks_v2_extended.py` | Temporary extended migration | Merge into `analysis/*.py` |
| `shared.py` | TaskContext moved to core | `core/context.py` |
| `graph.py` | Unused/redundant | Remove or merge |
| `query.py` | Unused query utilities | Remove or merge |
| `roadmap.py` | Can be simplified | Merge into orchestration |

### 10.3 Deprecated Functions to Remove

After consolidation, remove legacy wrapper functions:
- `detect_anti_patterns()` → Use `AntiPatternDetectionTask`
- `identify_silent_failures()` → Use `SilentFailuresTask`
- `analyze_structure()` → Use `StructureAnalysisTask`
- `tech_stack()` → Use `TechStackTask`
- All `*_v2()` functions → Use class-based tasks directly

### 10.4 Import Updates

**Before:**
```python
from src.tasks import detect_anti_patterns_v2
from src.tasks.tasks_v2 import AntiPatternDetectionTask
from src.tasks.tasks_v2_extended import StructureAnalysisTask
```

**After:**
```python
from src.tasks import AntiPatternDetectionTask, StructureAnalysisTask
# or
from src.tasks.analysis import AntiPatternDetectionTask
```

### 10.5 Public API Design

**`src/tasks/__init__.py` exports:**
```python
# Core Infrastructure
from .core import BaseTask, FileInfo, TaskContext, TaskResultCache, cached_task

# Task Classes (primary API)
from .analysis import (
    AntiPatternDetectionTask,
    ArchitecturePatternTask,
    DependencyGraphTask,
    DuplicateBlocksTask,
    HealthScoreTask,
    ImpactAnalysisTask,
    SecurityHeuristicsTask,
    SemanticDuplicationTask,
    SilentFailuresTask,
    StructureAnalysisTask,
    TechStackTask,
    TestMappingTask,
)

# Orchestration
from .orchestration import (
    TaskExecutionHistory,
    TaskRecommendationEngine,
    MetricsDashboard,
    PipelineExecutor,
    TaskScheduler,
    WebhookManager,
    ScoringWeights,
)

# Registry & Utilities
from .registry import run_task, run_tasks_parallel, TASK_CATEGORIES
```

### 10.6 Implementation Checklist

- [ ] Create `src/tasks/core/` directory with base modules
- [ ] Create `src/tasks/analysis/` directory with task implementations
- [ ] Create `src/tasks/orchestration/` directory with infrastructure
- [ ] Merge `tasks_v2.py` classes into appropriate analysis modules
- [ ] Merge `tasks_v2_extended.py` classes into appropriate analysis modules
- [ ] Move `TaskContext` from `shared.py` to `core/context.py`
- [ ] Delete deprecated files: `tasks_v2.py`, `tasks_v2_extended.py`, `shared.py`
- [ ] Delete unused files: `graph.py`, `query.py` (if confirmed unused)
- [ ] Update `src/tasks/__init__.py` with clean exports
- [ ] Update all internal imports to new paths
- [ ] Update `task_registry.py` → `registry.py` location
- [ ] Update API layer imports (`src/api/tasks.py`)
- [ ] Update test imports
- [ ] Remove deprecated `*_v2()` wrapper functions
- [ ] Run full test suite to verify no regressions
- [ ] Update documentation with final architecture

### 10.7 Test Updates Required

| Test File | Updates Needed |
|-----------|----------------|
| `test_tasks_v2.py` | Rename to `test_analysis_tasks.py`, update imports |
| `test_tasks_v2_extended.py` | Merge into domain-specific test files |
| `test_scoring_weights.py` | Update imports to new paths |
| `test_webhooks.py` | Update imports |
| `test_scheduler.py` | Update imports |
| `test_recommendations.py` | Update imports |

### 10.8 Success Criteria

- [ ] No `v2` or `extended` in any filename
- [ ] All tasks accessible from `src.tasks` directly
- [ ] Folder structure organized by domain (core/analysis/orchestration)
- [ ] Zero deprecated wrapper functions
- [ ] All 306 tests passing
- [ ] Clean imports with no circular dependencies
- [ ] Documentation updated with final architecture diagram

---

## All Phases Summary

| Phase | Status | Tests | Description |
|-------|--------|-------|-------------|
| 1 | ✅ Done | 58 | Cleanup: Removed 5 low-value tasks |
| 2 | ✅ Done | — | Core: detect-patterns, test-mapping improvements |
| 3 | ✅ Done | — | Enhanced: JS/TS support, DOT format, TypedDict |
| 4 | ✅ Done | — | Performance: TaskContext caching |
| 5 | ✅ Done | — | Parallel: Categories, concurrent execution |
| 6 | ✅ Done | 91 | Infrastructure: Disk cache, BaseTask class |
| 7 | ✅ Done | 144 | Pipelines: History tracking, custom pipelines |
| 8 | ✅ Done | 195 | Analytics: Metrics dashboard, recommendations |
| 9 | ✅ Done | 306 | Automation: Webhooks, scheduling, scoring weights |
| 10 | 🔲 Planned | 306+ | Cleanup: File consolidation, folder reorganization |

---

### Test Verification
All 306 tests pass after Phase 9:
```
pytest tests/ -v --tb=short
306 passed in 26.xx s
```
