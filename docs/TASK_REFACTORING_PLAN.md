# Architext Tasks Analysis & Refactoring Plan

**Date:** January 30, 2026  
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
| Tasks Improved | 0 | 4 |

### Changes Made:
- **Deleted 5 tasks** (low value / project-specific / redundant)
- **Improved `detect-patterns`** - added confidence scoring, more patterns
- **Renamed `test-coverage` → `test-mapping`** - improved implementation
- **Enhanced `code-knowledge-graph`** - added JS/TS support via tree-sitter
- **Enhanced `dependency-graph`** - added DOT/Graphviz export format
- **Added TypedDict types** - proper type definitions for all task results

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
├── server.py                 # Updated backward-compatible imports ✅
├── api/
│   └── tasks.py              # Removed 5 routes, renamed test-mapping ✅
├── tasks/
│   ├── __init__.py           # Updated exports, added type exports ✅
│   ├── types.py              # NEW: TypedDict definitions for all tasks ✅
│   ├── architecture.py       # Removed diff_architecture_review, onboarding_guide ✅
│   │                         # Improved architecture_pattern_detection ✅
│   │                         # Added JS/TS support to code_knowledge_graph ✅
│   │                         # Added DOT format to dependency_graph_export ✅
│   ├── quality.py            # Removed refactoring_recommendations, logic_gap_analysis ✅
│   │                         # Renamed test_coverage → test_mapping_analysis ✅
│   ├── roadmap.py            # Removed logic_gap_analysis import ✅
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

### Phase 4: Future Enhancements (Backlog)
- [ ] Add shared task execution context with file caching
- [ ] Add task result caching for `synthesis-roadmap`
- [ ] Standardize parameter handling across all tasks

---

## Conclusion

**Final Statistics:**
- Started with: 20 tasks
- Deleted: 5 tasks (25%)
- Improved: 4 tasks (20%)
- Final count: 15 tasks

**Phase 3 Enhancements:**
- JS/TS support via tree-sitter
- DOT format export
- TypedDict definitions for IDE support

The refactoring removed tasks that provided no real value, improved remaining tasks for better accuracy and usability, and added multi-language support. The codebase is now cleaner, more extensible, and better typed.

### Test Verification
All 58 tests pass after refactoring:
```
pytest tests/ -v --tb=short
58 passed in 60.88s
```
