"""Type definitions for task results using TypedDict.

These types provide better IDE support and documentation for task return values.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


# === Structure Analysis Types ===

class StructureResult(TypedDict):
    """Result from analyze-structure task."""
    total_files: int
    file_tree: Dict[str, Any]
    file_tree_markdown: str
    language_stats: Dict[str, int]
    import_clusters: List[str]


# === Health Score Types ===

class HealthDetails(TypedDict):
    """Detailed health metrics."""
    modularity: float
    coupling: float
    documentation: float
    testing: float
    avg_files_per_dir: float
    avg_dependencies: float
    doc_files: int
    docstrings: int


class HealthResult(TypedDict):
    """Result from health-score task."""
    score: float
    details: HealthDetails


# === Tech Stack Types ===

class TechStackResult(TypedDict):
    """Result from tech-stack task."""
    languages: Dict[str, int]
    frameworks: List[str]
    build_tools: List[str]
    testing_frameworks: List[str]
    config_files: List[str]


# === Architecture Types ===

class PatternDetail(TypedDict):
    """Details of a detected architecture pattern."""
    pattern: str
    confidence: float
    required_signals: List[str]
    optional_signals: List[str]
    evidence: List[str]


class PatternDetectionResult(TypedDict):
    """Result from detect-patterns task."""
    patterns: List[str]
    detailed: List[PatternDetail]
    files_analyzed: int


class ImpactAnalysisResult(TypedDict):
    """Result from impact-analysis task."""
    module: str
    targets: List[str]
    affected: List[str]
    affected_count: int


class DependencyEdge(TypedDict):
    """An edge in the dependency graph."""
    source: str
    target: str


class DependencyGraphResult(TypedDict):
    """Result from dependency-graph task."""
    format: str
    content: Optional[str]
    nodes: Optional[List[str]]
    edges: Optional[List[tuple]]
    edge_count: Optional[int]


# === Knowledge Graph Types ===

class KnowledgeNode(TypedDict):
    """A node in the knowledge graph."""
    id: str
    type: str  # "class", "function", "module"
    file: str
    line: int
    language: Optional[str]


class KnowledgeEdge(TypedDict):
    """An edge in the knowledge graph."""
    source: str
    target: str
    type: str  # "calls", "defines", "imports"


class KnowledgeGraphResult(TypedDict):
    """Result from code-knowledge-graph task."""
    nodes: List[KnowledgeNode]
    edges: List[KnowledgeEdge]
    unsupported_files: List[str]
    languages_parsed: Dict[str, int]


# === Test Mapping Types ===

class TestMappingResult(TypedDict):
    """Result from test-mapping task."""
    disclaimer: str
    source_files: int
    test_files: int
    files_with_tests: int
    files_without_tests: int
    coverage_percentage: float
    mapping: Dict[str, List[str]]
    untested_files: List[str]


# === Anti-Pattern Types ===

class AntiPatternFinding(TypedDict):
    """A detected anti-pattern."""
    type: str
    file: str
    line: int
    message: str
    severity: str


class AntiPatternResult(TypedDict):
    """Result from detect-anti-patterns task."""
    total_findings: int
    findings: List[AntiPatternFinding]
    by_type: Dict[str, int]
    by_severity: Dict[str, int]


# === Security Types ===

class SecurityFinding(TypedDict):
    """A security finding."""
    id: str
    severity: str
    file: str
    line: int
    description: str
    snippet: Optional[str]


class SecurityHeuristicsResult(TypedDict):
    """Result from security-heuristics task."""
    total_findings: int
    findings: List[SecurityFinding]
    by_severity: Dict[str, int]
    by_rule: Dict[str, int]


class VulnerabilityResult(TypedDict):
    """Result from detect-vulnerabilities task."""
    queries_run: int
    findings: List[Dict[str, Any]]
    summary: Dict[str, int]


# === Duplication Types ===

class DuplicateBlock(TypedDict):
    """A duplicate code block occurrence."""
    file: str
    start_line: int
    end_line: int
    lines: int


class DuplicationFinding(TypedDict):
    """A group of duplicate code blocks."""
    hash: str
    occurrences: List[DuplicateBlock]
    sample: str


class DuplicationResult(TypedDict):
    """Result from detect-duplication task."""
    total_duplicate_groups: int
    findings: List[DuplicationFinding]
    files_with_duplication: int


class SemanticDuplicateOccurrence(TypedDict):
    """A semantically similar code occurrence."""
    file: str
    name: str
    line: int
    tokens: int


class SemanticDuplicationFinding(TypedDict):
    """A group of semantically similar code."""
    occurrences: List[SemanticDuplicateOccurrence]


class SemanticDuplicationResult(TypedDict):
    """Result from detect-duplication-semantic task."""
    total_groups: int
    findings: List[SemanticDuplicationFinding]
    files_scanned: int


# === Silent Failure Types ===

class SilentFailureFinding(TypedDict):
    """A silent failure (swallowed exception)."""
    file: str
    line: int
    exception_types: List[str]
    body_summary: str


class SilentFailureResult(TypedDict):
    """Result from identify-silent-failures task."""
    total_findings: int
    findings: List[SilentFailureFinding]


# === Roadmap Types ===

class RoadmapItem(TypedDict):
    """A roadmap item/recommendation."""
    priority: str
    category: str
    title: str
    description: str
    affected_files: List[str]


class SynthesisRoadmapResult(TypedDict):
    """Result from synthesis-roadmap task."""
    health_score: float
    total_issues: int
    items: List[RoadmapItem]
    summary: Dict[str, int]


# Note: The runtime TaskContext class is defined in src.tasks.shared
# and provides caching functionality for task execution.
# Import it from there: from src.tasks import TaskContext, task_context
