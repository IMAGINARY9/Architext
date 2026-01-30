# Architext Tasks Analysis & Refactoring Plan

**Date:** January 30, 2026  
**Total Tasks Analyzed:** 20

---

## Executive Summary

This document provides a comprehensive analysis of all default project tasks, identifies issues, and proposes a refactoring plan to improve clarity, usability, and efficiency.

---

## 📋 Task Overview

| Task Name | Module | Purpose | Value Rating | Status |
|-----------|--------|---------|--------------|--------|
| `analyze-structure` | structure.py | Repository file tree & language stats | ⭐⭐⭐⭐⭐ | **Keep** |
| `tech-stack` | tech_stack.py | Detect frameworks & languages | ⭐⭐⭐⭐ | **Keep** |
| `detect-anti-patterns` | anti_patterns.py | Find code smells & structural issues | ⭐⭐⭐⭐⭐ | **Keep** |
| `health-score` | health.py | Calculate overall codebase health | ⭐⭐⭐⭐⭐ | **Keep** |
| `impact-analysis` | architecture.py | Find affected modules when changing code | ⭐⭐⭐⭐ | **Keep** |
| `refactoring-recommendations` | quality.py | Generate refactoring suggestions | ⭐⭐⭐ | **Improve** |
| `generate-docs` | docs.py | Auto-generate documentation files | ⭐⭐⭐ | **Improve** |
| `dependency-graph` | architecture.py | Export module dependency graph | ⭐⭐⭐⭐ | **Keep** |
| `test-coverage` | quality.py | Analyze test file coverage | ⭐⭐ | **Improve** |
| `detect-patterns` | architecture.py | Detect architecture patterns (MVC, etc.) | ⭐⭐ | **Improve** |
| `diff-architecture` | architecture.py | Compare file lists for changes | ⭐ | **Remove** |
| `onboarding-guide` | architecture.py | Generate onboarding suggestions | ⭐ | **Remove** |
| `detect-vulnerabilities` | security.py | Security scanning with semantic queries | ⭐⭐⭐⭐ | **Keep** |
| `logic-gap-analysis` | quality.py | Find unused config settings | ⭐⭐ | **Improve** |
| `identify-silent-failures` | quality.py | Find swallowed exceptions | ⭐⭐⭐⭐ | **Keep** |
| `security-heuristics` | security.py | Regex + AST security scanning | ⭐⭐⭐⭐ | **Keep** |
| `code-knowledge-graph` | architecture.py | Build call graph from AST | ⭐⭐⭐ | **Improve** |
| `synthesis-roadmap` | roadmap.py | Aggregate all findings into roadmap | ⭐⭐⭐⭐⭐ | **Keep** |
| `detect-duplication` | duplication.py | Find exact duplicate code blocks | ⭐⭐⭐⭐ | **Keep** |
| `detect-duplication-semantic` | duplication.py | Find semantically similar functions | ⭐⭐⭐⭐ | **Keep** |

---

## 🔴 Tasks Recommended for REMOVAL

### 1. `diff-architecture` - **REMOVE**

**Location:** [architecture.py](../src/tasks/architecture.py#L112-L128)

**Current Implementation:**
```python
def diff_architecture_review(baseline_files: Optional[List[str]] = None, ...):
    current_files = set(collect_file_paths(...))
    baseline_set = set(baseline_files or [])
    added = sorted(current_files - baseline_set)
    removed = sorted(baseline_set - current_files)
    return {"added": added[:100], "removed": removed[:100], ...}
```

**Problems:**
- Trivial set difference operation with minimal value
- Requires user to provide `baseline_files` list manually (impractical)
- No meaningful architecture analysis - just file list comparison
- Git diff provides better functionality natively
- Zero adoption potential - too basic to be useful

**Verdict:** ❌ **DELETE** - Functionality is trivial and better served by git

---

### 2. `onboarding-guide` - **REMOVE**

**Location:** [architecture.py](../src/tasks/architecture.py#L131-L148)

**Current Implementation:**
```python
def onboarding_guide(...):
    files = collect_file_paths(...)
    root_files = [Path(path).name for path in files if len(Path(path).parts) <= 3]
    entry_points = [f for f in root_files if f.lower() in {"readme.md", "setup.py", ...}]
    suggestions = ["Start with README...", "Review entry points..."]
    return {"entry_points": entry_points, "suggestions": suggestions, "root_files": root_files[:50]}
```

**Problems:**
- Returns hardcoded generic suggestions with no project-specific analysis
- Simply lists common files (README, setup.py) - obvious to any developer
- No actual "guide" generation - just file filtering
- Provides zero actionable intelligence
- Name is misleading - it doesn't generate a guide

**Verdict:** ❌ **DELETE** - Too simplistic, no real value

---

## 🟡 Tasks Recommended for IMPROVEMENT

### 3. `detect-patterns` - **NEEDS MAJOR IMPROVEMENT**

**Location:** [architecture.py](../src/tasks/architecture.py#L83-L110)

**Current Implementation Issues:**
```python
# Naive string matching in file paths
if any("controllers" in path for path in files_lower):
    patterns.append("MVC")
if any("docker" in path for path in files_lower):
    patterns.append("Microservices")  # Having docker != microservices!
```

**Problems:**
- False positives: "docker" doesn't mean microservices
- False positives: Any file with "event" triggers "Event-Driven"
- No confidence scores or evidence quality
- Missing many patterns: DDD, CQRS, Hexagonal, Clean Architecture
- No analysis of actual code structure

**Refactoring Plan:**
1. Add confidence scores based on evidence strength
2. Require multiple signals before pattern detection
3. Add more patterns with proper detection logic
4. Analyze actual code structure, not just paths
5. Return evidence for each detected pattern

---

### 4. `logic-gap-analysis` - **NEEDS IMPROVEMENT**

**Location:** [quality.py](../src/tasks/quality.py#L99-L166)

**Current Implementation Issues:**
- Hardcoded to only look for `ArchitextSettings` class
- Only works for Pydantic settings pattern
- Project-specific logic that won't work for other codebases

**Refactoring Plan:**
1. Make configurable for any settings class name
2. Support multiple settings patterns (env vars, config files)
3. Add generic unused code detection
4. Rename to `config-gap-analysis` for clarity

---

### 5. `test-coverage` - **NEEDS IMPROVEMENT**

**Location:** [quality.py](../src/tasks/quality.py#L70-L97)

**Current Implementation Issues:**
```python
# Naive stem matching
for src in source_files:
    stem = Path(src).stem
    for test in test_files:
        if stem in Path(test).stem:  # Too loose!
            mapping[src].append(test)
```

**Problems:**
- Matches are too loose (e.g., "config" matches "test_config_manager")
- No actual coverage metrics - just file presence
- Doesn't understand test frameworks
- Name is misleading - it's not real coverage analysis

**Refactoring Plan:**
1. Rename to `test-mapping` or `test-presence`
2. Use smarter matching logic
3. Add support for pytest markers and test discovery
4. Consider integration with actual coverage tools

---

### 6. `refactoring-recommendations` - **NEEDS IMPROVEMENT**

**Location:** [quality.py](../src/tasks/quality.py#L22-L68)

**Current Implementation Issues:**
- Simply aggregates anti-patterns and health score
- Limited recommendation types
- No prioritization logic
- Generic suggestions without specifics

**Refactoring Plan:**
1. Add more recommendation categories
2. Include effort/impact matrix
3. Add specific file-level recommendations
4. Include code examples for fixes

---

### 7. `generate-docs` - **NEEDS IMPROVEMENT**

**Location:** [docs.py](../src/tasks/docs.py)

**Current Implementation Issues:**
- Just aggregates other task outputs into files
- No actual documentation generation
- Outputs JSON/Markdown dumps, not useful docs

**Refactoring Plan:**
1. Generate actual README sections
2. Add API documentation extraction
3. Create architecture diagrams
4. Generate developer guides from code analysis

---

### 8. `code-knowledge-graph` - **NEEDS IMPROVEMENT**

**Location:** [architecture.py](../src/tasks/architecture.py#L151-L232)

**Current Implementation Issues:**
- Python-only (JS/TS marked as "unsupported")
- Call targets are just function names, not fully qualified
- Large graphs can be overwhelming
- No visualization format options

**Refactoring Plan:**
1. Add JavaScript/TypeScript support using tree-sitter
2. Improve call resolution to use full paths
3. Add graph filtering options
4. Support DOT/Graphviz export format

---

## ✅ Tasks to KEEP (with minor improvements)

### Core Analysis Tasks (High Value)
| Task | Minor Improvements Suggested |
|------|------------------------------|
| `analyze-structure` | Add file size analysis, gitignore awareness |
| `detect-anti-patterns` | Add severity aggregation, trend tracking |
| `health-score` | Add historical comparison, configurable weights |
| `impact-analysis` | Add depth limiting, impact severity levels |
| `dependency-graph` | Add DOT format, clustering by package |

### Security Tasks (High Value)
| Task | Minor Improvements Suggested |
|------|------------------------------|
| `detect-vulnerabilities` | Add CVSS scoring, remediation suggestions |
| `security-heuristics` | Add rule categories, false positive suppression |

### Quality Tasks (High Value)
| Task | Minor Improvements Suggested |
|------|------------------------------|
| `identify-silent-failures` | Add exception type analysis |
| `detect-duplication` | Add configurable thresholds via params |
| `detect-duplication-semantic` | Add cross-file deduplication suggestions |

### Aggregation Tasks (High Value)
| Task | Minor Improvements Suggested |
|------|------------------------------|
| `synthesis-roadmap` | Add time estimates, dependency ordering |
| `tech-stack` | Add version detection, deprecation warnings |

---

## 🔧 Code Quality Issues Across All Tasks

### 1. **Inconsistent Parameter Handling**
```python
# Some tasks use storage_path, some source_path, some both
def task_a(storage_path=None, source_path=None): ...
def task_b(storage_path=None): ...  # Missing source_path
```
**Fix:** Standardize all tasks to accept both parameters uniformly.

### 2. **Duplicated File Collection Logic**
Every task calls `collect_file_paths()` separately - should be passed in.

**Fix:** Create a task execution context that pre-collects files once.

### 3. **No Caching Between Related Tasks**
`synthesis_roadmap` calls 7 other tasks, each re-scanning files.

**Fix:** Implement result caching or shared context between tasks.

### 4. **Missing Type Hints on Return Values**
```python
def analyze_structure(...) -> Dict[str, Any]:  # Too generic
```
**Fix:** Create TypedDict or dataclasses for task results.

### 5. **No Task Dependencies Declaration**
Tasks that depend on others (e.g., `refactoring_recommendations`) don't declare it.

**Fix:** Add task metadata with dependencies, enabling parallel execution.

---

## 📊 Refactoring Priority Matrix

| Priority | Task | Effort | Impact | Action |
|----------|------|--------|--------|--------|
| 🔴 P0 | `diff-architecture` | Low | High | Delete |
| 🔴 P0 | `onboarding-guide` | Low | High | Delete |
| 🟠 P1 | `detect-patterns` | Medium | High | Rewrite |
| 🟠 P1 | `logic-gap-analysis` | Medium | Medium | Generalize |
| 🟡 P2 | `test-coverage` | Medium | Medium | Rename + Improve |
| 🟡 P2 | `code-knowledge-graph` | High | Medium | Add JS/TS support |
| 🟢 P3 | `refactoring-recommendations` | Medium | Medium | Enhance |
| 🟢 P3 | `generate-docs` | High | Medium | Expand scope |

---

## 📝 Implementation Checklist

### Phase 1: Cleanup (Week 1)
- [x] Remove `diff-architecture` task
- [x] Remove `onboarding-guide` task
- [x] Update task registry imports
- [ ] Update documentation
- [ ] Update tests

### Phase 2: Core Improvements (Week 2-3)
- [ ] Rewrite `detect-patterns` with confidence scores
- [ ] Generalize `logic-gap-analysis` 
- [ ] Rename `test-coverage` to `test-mapping`
- [ ] Add shared task execution context

### Phase 3: Enhancements (Week 4+)
- [ ] Add JS/TS support to `code-knowledge-graph`
- [ ] Enhance `refactoring-recommendations`
- [ ] Improve `generate-docs` output quality
- [ ] Add task result caching

---

## 📁 Files to Modify

```
src/
├── task_registry.py          # Remove 2 task registrations
├── tasks/
│   ├── __init__.py           # Remove 2 exports
│   ├── architecture.py       # Remove diff_architecture_review, onboarding_guide
│   │                         # Improve architecture_pattern_detection
│   ├── quality.py            # Generalize logic_gap_analysis
│   │                         # Rename/improve test_coverage_analysis
│   └── docs.py               # Enhance generate_docs
```

---

## Conclusion

Out of 20 tasks:
- **2 tasks (10%)** should be **deleted** - provide no real value
- **5 tasks (25%)** need **significant improvement**
- **13 tasks (65%)** are **good to keep** with minor enhancements

The codebase would benefit most from removing dead weight and focusing on making the core tasks more robust and configurable.
