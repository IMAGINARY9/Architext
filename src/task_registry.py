"""Central registry for analysis tasks."""
from __future__ import annotations

import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Set

from src.tasks import (
    analyze_structure,
    tech_stack,
    detect_anti_patterns,
    health_score,
    impact_analysis,
    dependency_graph_export,
    test_mapping_analysis,
    architecture_pattern_detection,
    detect_vulnerabilities,
    identify_silent_failures,
    security_heuristics,
    code_knowledge_graph,
    synthesis_roadmap,
    detect_duplicate_blocks,
    detect_duplicate_blocks_semantic,
    TaskContext,
    task_context,
)


TASK_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "analyze-structure": analyze_structure,
    "tech-stack": tech_stack,
    "detect-anti-patterns": detect_anti_patterns,
    "health-score": health_score,
    "impact-analysis": impact_analysis,
    "dependency-graph": dependency_graph_export,
    "test-mapping": test_mapping_analysis,
    "detect-patterns": architecture_pattern_detection,
    "detect-vulnerabilities": detect_vulnerabilities,
    "identify-silent-failures": identify_silent_failures,
    "security-heuristics": security_heuristics,
    "code-knowledge-graph": code_knowledge_graph,
    "synthesis-roadmap": synthesis_roadmap,
    "detect-duplication": detect_duplicate_blocks,
    "detect-duplication-semantic": detect_duplicate_blocks_semantic,
}


# Task dependency graph - tasks that must complete before others
# Key: task name, Value: set of task names it depends on
TASK_DEPENDENCIES: Dict[str, Set[str]] = {
    # synthesis-roadmap depends on these tasks (runs them internally)
    "synthesis-roadmap": {
        "detect-anti-patterns",
        "health-score", 
        "identify-silent-failures",
        "security-heuristics",
        "detect-duplication",
        "detect-duplication-semantic",
    },
    # Most tasks are independent and can run in parallel
    "analyze-structure": set(),
    "tech-stack": set(),
    "detect-anti-patterns": set(),
    "health-score": set(),
    "impact-analysis": set(),
    "dependency-graph": set(),
    "test-mapping": set(),
    "detect-patterns": set(),
    "detect-vulnerabilities": set(),
    "identify-silent-failures": set(),
    "security-heuristics": set(),
    "code-knowledge-graph": set(),
    "detect-duplication": set(),
    "detect-duplication-semantic": set(),
}


# Task categories for grouping related tasks
TASK_CATEGORIES: Dict[str, List[str]] = {
    "structure": ["analyze-structure", "tech-stack", "detect-patterns"],
    "quality": ["detect-anti-patterns", "health-score", "test-mapping", "identify-silent-failures"],
    "security": ["detect-vulnerabilities", "security-heuristics"],
    "duplication": ["detect-duplication", "detect-duplication-semantic"],
    "architecture": ["impact-analysis", "dependency-graph", "code-knowledge-graph"],
    "synthesis": ["synthesis-roadmap"],
}


def list_task_names() -> list[str]:
    return sorted(TASK_REGISTRY.keys())


def list_task_categories() -> Dict[str, List[str]]:
    """Return task categories with their task names."""
    return TASK_CATEGORIES.copy()


def get_task_dependencies(task_name: str) -> Set[str]:
    """Get the set of tasks that must complete before this task."""
    return TASK_DEPENDENCIES.get(task_name, set()).copy()


def get_task_handler(task_name: str) -> Callable[..., Dict[str, Any]]:
    try:
        return TASK_REGISTRY[task_name]
    except KeyError as exc:
        raise ValueError(f"Unknown task: {task_name}") from exc


def run_task(task_name: str, **kwargs: Any) -> Dict[str, Any]:
    handler = get_task_handler(task_name)
    signature = inspect.signature(handler)
    filtered = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters and value is not None
    }
    return handler(**filtered)


def run_tasks_parallel(
    task_names: List[str],
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    max_workers: int = 4,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Run multiple tasks in parallel with shared context caching.
    
    Tasks are executed concurrently using a thread pool, with file collection
    cached and shared across all tasks via TaskContext.
    
    Args:
        task_names: List of task names to run
        storage_path: Path to ChromaDB storage
        source_path: Path to source directory
        max_workers: Maximum number of parallel workers (default: 4)
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dict mapping task names to their results
        
    Example:
        results = run_tasks_parallel(
            ["analyze-structure", "tech-stack", "detect-anti-patterns"],
            source_path="src"
        )
    """
    # Validate task names
    for name in task_names:
        if name not in TASK_REGISTRY:
            raise ValueError(f"Unknown task: {name}")
    
    results: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}
    
    # Use shared context for caching
    with task_context(storage_path=storage_path, source_path=source_path) as ctx:
        # Pre-warm the file cache
        _ = ctx.get_files()
        
        def run_single_task(task_name: str) -> tuple[str, Dict[str, Any]]:
            if progress_callback:
                progress_callback({"task": task_name, "status": "started"})
            try:
                result = run_task(
                    task_name,
                    storage_path=storage_path,
                    source_path=source_path,
                    progress_callback=progress_callback,
                )
                if progress_callback:
                    progress_callback({"task": task_name, "status": "completed"})
                return (task_name, result)
            except Exception as e:
                if progress_callback:
                    progress_callback({"task": task_name, "status": "failed", "error": str(e)})
                return (task_name, {"error": str(e), "task": task_name})
        
        # Run tasks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_single_task, name): name for name in task_names}
            
            for future in as_completed(futures):
                task_name, result = future.result()
                results[task_name] = result
    
    return results


def run_category(
    category: str,
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    max_workers: int = 4,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Run all tasks in a category in parallel.
    
    Args:
        category: Category name (structure, quality, security, duplication, architecture, synthesis)
        storage_path: Path to ChromaDB storage
        source_path: Path to source directory
        max_workers: Maximum number of parallel workers
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dict mapping task names to their results
        
    Example:
        results = run_category("quality", source_path="src")
    """
    if category not in TASK_CATEGORIES:
        raise ValueError(f"Unknown category: {category}. Valid: {list(TASK_CATEGORIES.keys())}")
    
    task_names = TASK_CATEGORIES[category]
    return run_tasks_parallel(
        task_names,
        storage_path=storage_path,
        source_path=source_path,
        max_workers=max_workers,
        progress_callback=progress_callback,
    )
