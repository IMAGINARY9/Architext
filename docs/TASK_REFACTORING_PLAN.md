# Architext Tasks Analysis & Refactoring Plan

**Date:** January 31, 2026  
**Initial Tasks Analyzed:** 20  
**Current Tasks:** 15 ✅

---

## Executive Summary

This document provides a comprehensive analysis of all default project tasks, identifies issues, and tracks the refactoring progress.

### 🎉 Refactoring Status: COMPLETED ✅

| Metric | Before | After |
|--------|--------|-------|
| Total Tasks | 20 | 15 |
| Tasks Deleted | - | 5 |
| Tasks Improved | 0 | 6 |
| Tests | 58 | 91 |

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
  -d '{"source": "./src"}'
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

### Phase 7: Future Enhancements (Backlog)
- [ ] Add task execution history and analytics
- [ ] Add custom task composition (user-defined task pipelines)
- [ ] Migrate existing tasks to use BaseTask class

---

## Conclusion

**Final Statistics:**
- Started with: 20 tasks
- Deleted: 5 tasks (25%)
- Improved: 6 tasks (30%)
- Final count: 15 tasks

**All Phases Completed:**
- ✅ Phase 1: Cleanup (deleted 5 low-value tasks)
- ✅ Phase 2: Core Improvements (detect-patterns, test-mapping)
- ✅ Phase 3: Enhanced Capabilities (JS/TS support, DOT format, TypedDict)
- ✅ Phase 4: Performance & Consistency (TaskContext caching)
- ✅ Phase 5: Parallel Execution & Categories (concurrent tasks, API endpoints)
- ✅ Phase 6: Caching & Infrastructure (disk caching, BaseTask class)

The refactoring:
1. Removed tasks that provided no real value
2. Improved remaining tasks for accuracy and usability
3. Added multi-language support (JS/TS via tree-sitter)
4. Implemented shared caching for performance
5. Added parallel execution for efficiency
6. Organized tasks into logical categories
7. Added persistent disk caching with automatic invalidation
8. Created base classes for future task development

The codebase is now cleaner, faster, better typed, and more extensible.

### Test Verification
All 91 tests pass after refactoring:
```
pytest tests/ -v --tb=short
91 passed in 23.18s
```
