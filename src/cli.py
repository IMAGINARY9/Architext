import argparse
import sys
import os
import json
from datetime import date
from pathlib import Path

# Ensure src can be imported if resolving paths is tricky, though running as module (python -m src.cli) usually handles this.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_settings
from src.indexer import (
    initialize_settings,
    gather_index_files,
    create_index_from_paths,
    load_existing_index,
    query_index,
)
from src.tasks import (
    analyze_structure,
    tech_stack,
    detect_anti_patterns,
    health_score,
    refactoring_recommendations,
    dependency_graph_export,
    test_coverage_analysis,
    logic_gap_analysis,
    identify_silent_failures,
    security_heuristics,
    code_knowledge_graph,
    onboarding_guide,
    synthesis_roadmap,
    detect_duplicate_blocks,
    detect_duplicate_blocks_semantic,
    query_diagnostics,
)
from src.task_registry import run_task
from src.ingestor import resolve_source, cleanup_cache
from src.cli_utils import (
    VerboseLogger,
    format_response,
    get_available_models_info,
    DryRunIndexer,
    to_agent_response,
    to_agent_response_compact,
)

def _compute_auto_storage(source: str, default_storage: str) -> str:
    """Compute auto storage path if not provided."""
    import hashlib
    from urllib.parse import urlparse

    # Extract repo name
    if source.startswith("http") or source.startswith("git@"):
        parsed = urlparse(source)
        repo_name = Path(parsed.path).stem  # e.g., 'requests' from '/psf/requests.git'
    else:
        repo_name = Path(source).resolve().name

def _compute_auto_storage(source: str, default_storage: str) -> str:
    """Compute auto storage path if not provided."""
    import hashlib
    from urllib.parse import urlparse

    # Extract repo name
    if source.startswith("http") or source.startswith("git@"):
        parsed = urlparse(source)
        repo_name = Path(parsed.path).stem  # e.g., 'requests' from '/psf/requests.git'
    else:
        repo_name = Path(source).resolve().name

    # Hash the full source for uniqueness
    source_hash = hashlib.sha256(source.encode()).hexdigest()[:8]
    return os.path.join(default_storage, f"{repo_name}-{source_hash}")


def _build_parser():
    parser = argparse.ArgumentParser(description="Architext CLI: Local Codebase RAG")
    parser.add_argument(
        "--env-file",
        dest="env_file",
        help="Optional path to .env file for configuration overrides",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging for debugging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a repository")
    index_parser.add_argument(
        "source",
        help="Local path or remote git URL (GitHub/GitLab/Gitea)",
    )
    
    # Basic options
    basic_group = index_parser.add_argument_group('basic options')
    basic_group.add_argument("--storage", help="Path to save the vector DB (auto-generated if omitted)")
    basic_group.add_argument(
        "--preview",
        action="store_true",
        help="Preview indexing plan in JSON format",
    )
    basic_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview indexing without persisting (human-readable)",
    )
    


    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a question about the indexed code")
    
    # Basic options
    query_basic = query_parser.add_argument_group('basic options')
    query_basic.add_argument("text", help="The question/query string")
    query_basic.add_argument("--storage", help="Path to load the vector DB from (overrides config)")
    query_basic.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for the response",
    )
    


    ask_parser = subparsers.add_parser("ask", help="Agent-optimized query")
    
    # Basic options
    ask_basic = ask_parser.add_argument_group('basic options')
    ask_basic.add_argument("text", help="The question/query string")
    ask_basic.add_argument("--storage", help="Path to load the vector DB from (overrides config)")
    ask_basic.add_argument(
        "--compact",
        action="store_true",
        help="Return compact agent schema output",
    )
    


    # List models command
    list_parser = subparsers.add_parser("list-models", help="Show available LLM and embedding models")

    # Cache cleanup command
    cache_parser = subparsers.add_parser("cache-cleanup", help="Clean up old cached repos")
    cache_parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Remove repos unused for more than N days (default: 30)",
    )

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the Architext API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable autoreload (dev only)")

    # Phase 2.5 tasks
    structure_parser = subparsers.add_parser("analyze-structure", help="Analyze repository structure")
    structure_parser.add_argument("--storage", help="Path to load the vector DB from")
    structure_parser.add_argument("--source", help="Source repo path (if not using storage)")
    structure_parser.add_argument(
        "--depth",
        choices=["shallow", "detailed", "exhaustive"],
        default="shallow",
        help="Depth of structure analysis",
    )
    structure_parser.add_argument(
        "--output-format",
        choices=["json", "markdown", "mermaid"],
        default="json",
        help="Output format",
    )

    tech_parser = subparsers.add_parser("tech-stack", help="Analyze technology stack")
    tech_parser.add_argument("--storage", help="Path to load the vector DB from")
    tech_parser.add_argument("--source", help="Source repo path (if not using storage)")
    tech_parser.add_argument(
        "--output-format",
        choices=["json", "markdown"],
        default="json",
        help="Output format",
    )

    diag_parser = subparsers.add_parser("query-diagnostics", help="Export hybrid/keyword diagnostics for a query")
    diag_parser.add_argument("text", help="Query text to diagnose")
    diag_parser.add_argument("--storage", help="Path to load the vector DB from")
    diag_parser.add_argument("--output", help="Optional path to save JSON diagnostics")

    anti_parser = subparsers.add_parser("detect-anti-patterns", help="Detect architectural anti-patterns")
    anti_parser.add_argument("--storage", help="Path to load the vector DB from")
    anti_parser.add_argument("--source", help="Source repo path (if not using storage)")

    health_parser = subparsers.add_parser("health-score", help="Compute architectural health score")
    health_parser.add_argument("--storage", help="Path to load the vector DB from")
    health_parser.add_argument("--source", help="Source repo path (if not using storage)")

    impact_parser = subparsers.add_parser("impact-analysis", help="Analyze impact of a module change")
    impact_parser.add_argument("module", help="Module name or path fragment")
    impact_parser.add_argument("--storage", help="Path to load the vector DB from")
    impact_parser.add_argument("--source", help="Source repo path (if not using storage)")

    refactor_parser = subparsers.add_parser("refactoring-recommendations", help="Suggest refactoring improvements")
    refactor_parser.add_argument("--storage", help="Path to load the vector DB from")
    refactor_parser.add_argument("--source", help="Source repo path (if not using storage)")

    docs_parser = subparsers.add_parser("generate-docs", help="Generate documentation bundle")
    docs_parser.add_argument("--storage", help="Path to load the vector DB from")
    docs_parser.add_argument("--source", help="Source repo path (if not using storage)")
    docs_parser.add_argument("--output", help="Output directory for docs")

    graph_parser = subparsers.add_parser("dependency-graph", help="Export dependency graph")
    graph_parser.add_argument("--storage", help="Path to load the vector DB from")
    graph_parser.add_argument("--source", help="Source repo path (if not using storage)")
    graph_parser.add_argument(
        "--output-format",
        choices=["mermaid", "json", "graphml"],
        default="mermaid",
        help="Output format",
    )

    coverage_parser = subparsers.add_parser("test-coverage", help="Analyze test coverage")
    coverage_parser.add_argument("--storage", help="Path to load the vector DB from")
    coverage_parser.add_argument("--source", help="Source repo path (if not using storage)")

    pattern_parser = subparsers.add_parser("detect-patterns", help="Detect architecture patterns")
    pattern_parser.add_argument("--storage", help="Path to load the vector DB from")
    pattern_parser.add_argument("--source", help="Source repo path (if not using storage)")

    diff_parser = subparsers.add_parser("diff-architecture", help="Review architecture diff")
    diff_parser.add_argument("--storage", help="Path to load the vector DB from")
    diff_parser.add_argument("--source", help="Source repo path (if not using storage)")
    diff_parser.add_argument(
        "--baseline",
        help="Path to a JSON file containing baseline file list",
    )

    onboard_parser = subparsers.add_parser("onboarding-guide", help="Generate onboarding guide")
    onboard_parser.add_argument("--storage", help="Path to load the vector DB from")
    onboard_parser.add_argument("--source", help="Source repo path (if not using storage)")

    vuln_parser = subparsers.add_parser("detect-vulnerabilities", help="Semantic + heuristic vulnerability sweep")
    vuln_parser.add_argument("--storage", help="Path to load the vector DB from")
    vuln_parser.add_argument("--source", help="Source repo path (if not using storage)")

    logic_parser = subparsers.add_parser("logic-gap-analysis", help="Find defined but unused config settings")
    logic_parser.add_argument("--storage", help="Path to load the vector DB from")
    logic_parser.add_argument("--source", help="Source repo path (if not using storage)")

    silent_parser = subparsers.add_parser("identify-silent-failures", help="Find silent exception handling")
    silent_parser.add_argument("--storage", help="Path to load the vector DB from")
    silent_parser.add_argument("--source", help="Source repo path (if not using storage)")

    sec_parser = subparsers.add_parser("security-heuristics", help="Run regex-based security heuristics")
    sec_parser.add_argument("--storage", help="Path to load the vector DB from")
    sec_parser.add_argument("--source", help="Source repo path (if not using storage)")

    knowledge_parser = subparsers.add_parser("code-knowledge-graph", help="Generate code knowledge graph")
    knowledge_parser.add_argument("--storage", help="Path to load the vector DB from")
    knowledge_parser.add_argument("--source", help="Source repo path (if not using storage)")

    roadmap_parser = subparsers.add_parser("synthesis-roadmap", help="Generate synthesis refactor roadmap")
    roadmap_parser.add_argument("--storage", help="Path to load the vector DB from")
    roadmap_parser.add_argument("--source", help="Source repo path (if not using storage)")

    dup_parser = subparsers.add_parser("detect-duplication", help="Detect duplicated code blocks")
    dup_parser.add_argument("--storage", help="Path to load the vector DB from")
    dup_parser.add_argument("--source", help="Source repo path (if not using storage)")
    dup_parser.add_argument("--min-lines", type=int, default=8, help="Minimum lines per block")
    dup_parser.add_argument("--max-findings", type=int, default=50, help="Maximum findings to return")

    dup_sem_parser = subparsers.add_parser(
        "detect-duplication-semantic",
        help="Detect semantically duplicated Python functions/classes",
    )
    dup_sem_parser.add_argument("--storage", help="Path to load the vector DB from")
    dup_sem_parser.add_argument("--source", help="Source repo path (if not using storage)")
    dup_sem_parser.add_argument("--min-tokens", type=int, default=40, help="Minimum tokens per block")
    dup_sem_parser.add_argument("--max-findings", type=int, default=50, help="Maximum findings to return")

    audit_parser = subparsers.add_parser(
        "audit",
        help="Run a headless analysis suite and export JSON/mermaid outputs",
    )
    audit_parser.add_argument(
        "--source",
        default=".",
        help="Source repo path (default: .)",
    )
    audit_parser.add_argument(
        "--output",
        help="Output directory (default: storage-refactor-data-YYYY-MM-DD under source)",
    )
    audit_parser.add_argument(
        "--min-dup-lines",
        type=int,
        default=8,
        help="Minimum lines per duplicated block",
    )
    audit_parser.add_argument(
        "--max-dup-findings",
        type=int,
        default=50,
        help="Maximum duplication findings to save",
    )
    audit_parser.add_argument("--ci", action="store_true", help="Enable CI mode (exit non-zero on thresholds)")
    audit_parser.add_argument("--min-health-score", type=float, help="Fail if health score below this")
    audit_parser.add_argument("--max-anti-patterns", type=int, help="Fail if anti-pattern count above this")
    audit_parser.add_argument("--max-silent-failures", type=int, help="Fail if silent failures above this")
    audit_parser.add_argument("--max-duplication", type=int, help="Fail if duplication findings above this")
    audit_parser.add_argument(
        "--max-semantic-duplication",
        type=int,
        help="Fail if semantic duplication findings above this",
    )
    audit_parser.add_argument("--max-security-findings", type=int, help="Fail if security findings above this")

    return parser


def _resolve_storage(args_storage: str, default_storage: str) -> str:
    return args_storage if args_storage else default_storage


def main():
    parser = _build_parser()
    args = parser.parse_args()

    logger = VerboseLogger(verbose=args.verbose)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # List models doesn't need LLM/embedding init
    if args.command == "list-models":
        models_info = get_available_models_info()
        print("\n" + "="*60)
        print("AVAILABLE LLM & EMBEDDING MODELS")
        print("="*60)
        for category, info in models_info.items():
            if isinstance(info, dict):
                print(f"\n{category.upper()}:")
                for key, value in info.items():
                    if isinstance(value, dict):
                        for k, v in value.items():
                            print(f"  {k}: {v}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"  {category}: {info}")
        print("\n" + "="*60 + "\n")
        return

    # Cache cleanup doesn't need LLM/embedding init
    if args.command == "cache-cleanup":
        removed = cleanup_cache(max_age_days=args.max_age)
        print(f"Cleanup complete. Removed {removed} cached repo(s).")
        return

    # Serve command (defer to uvicorn)
    if args.command == "serve":
        from uvicorn import run
        from src.server import create_app

        app = create_app()
        run(app=app, host=args.host, port=args.port, reload=args.reload)
        return

    # Task commands do not need LLM init
    if args.command in {
        "analyze-structure",
        "tech-stack",
        "query-diagnostics",
        "detect-anti-patterns",
        "health-score",
        "impact-analysis",
        "refactoring-recommendations",
        "generate-docs",
        "dependency-graph",
        "test-coverage",
        "detect-patterns",
        "diff-architecture",
        "onboarding-guide",
        "detect-vulnerabilities",
        "logic-gap-analysis",
        "identify-silent-failures",
        "security-heuristics",
        "code-knowledge-graph",
        "synthesis-roadmap",
        "detect-duplication",
        "detect-duplication-semantic",
        "audit",
    }:
        if args.command == "audit":
            source_path = args.source or "."
            source_root = Path(source_path).resolve()
            out_dir = Path(args.output) if args.output else source_root / f"storage-refactor-data-{date.today().isoformat()}"
            out_dir.mkdir(parents=True, exist_ok=True)

            results = {}
            results["analyze_structure_json"] = analyze_structure(
                source_path=source_path,
                depth="exhaustive",
                output_format="json",
            )
            results["analyze_structure_mermaid"] = analyze_structure(
                source_path=source_path,
                depth="detailed",
                output_format="mermaid",
            )
            results["tech_stack"] = tech_stack(source_path=source_path, output_format="json")
            results["anti_patterns"] = detect_anti_patterns(source_path=source_path)
            results["health_score"] = health_score(source_path=source_path)
            results["refactoring_recommendations"] = refactoring_recommendations(source_path=source_path)
            results["dependency_graph_mermaid"] = dependency_graph_export(
                source_path=source_path,
                output_format="mermaid",
            )
            results["test_coverage"] = test_coverage_analysis(source_path=source_path)
            results["logic_gap_analysis"] = logic_gap_analysis(source_path=source_path)
            results["silent_failures"] = identify_silent_failures(source_path=source_path)
            results["security_heuristics"] = security_heuristics(source_path=source_path)
            results["code_knowledge_graph"] = code_knowledge_graph(source_path=source_path)
            results["onboarding_guide"] = onboarding_guide(source_path=source_path)
            results["synthesis_roadmap"] = synthesis_roadmap(source_path=source_path)
            results["duplication"] = detect_duplicate_blocks(
                source_path=source_path,
                min_lines=args.min_dup_lines,
                max_findings=args.max_dup_findings,
            )
            results["duplication_semantic"] = detect_duplicate_blocks_semantic(
                source_path=source_path,
            )

            _write_json(out_dir / "analyze-structure.json", results["analyze_structure_json"])
            (out_dir / "structure.mmd").write_text(
                results["analyze_structure_mermaid"]["content"],
                encoding="utf-8",
            )
            _write_json(out_dir / "tech-stack.json", results["tech_stack"])
            _write_json(out_dir / "anti-patterns.json", results["anti_patterns"])
            _write_json(out_dir / "health-score.json", results["health_score"])
            _write_json(out_dir / "refactoring-recommendations.json", results["refactoring_recommendations"])
            (out_dir / "dependency-graph.mmd").write_text(
                results["dependency_graph_mermaid"].get("content", ""),
                encoding="utf-8",
            )
            _write_json(out_dir / "test-coverage.json", results["test_coverage"])
            _write_json(out_dir / "logic-gap-analysis.json", results["logic_gap_analysis"])
            _write_json(out_dir / "silent-failures.json", results["silent_failures"])
            _write_json(out_dir / "security-heuristics.json", results["security_heuristics"])
            _write_json(out_dir / "code-knowledge-graph.json", results["code_knowledge_graph"])
            _write_json(out_dir / "onboarding-guide.json", results["onboarding_guide"])
            _write_json(out_dir / "synthesis-roadmap.json", results["synthesis_roadmap"])
            _write_json(out_dir / "duplication.json", results["duplication"])
            _write_json(out_dir / "duplication-semantic.json", results["duplication_semantic"])

            summary = {
                "health_score": results["health_score"].get("score"),
                "anti_pattern_count": len(results["anti_patterns"].get("issues", [])),
                "silent_failures": results["silent_failures"].get("count", 0),
                "duplication_findings": results["duplication"].get("count", 0),
                "semantic_duplication_findings": results["duplication_semantic"].get("count", 0),
                "security_findings": results["security_heuristics"].get("counts", {}).get("total", 0),
            }
            _write_json(out_dir / "summary.json", summary)
            _write_json(
                out_dir / "_meta.json",
                {
                    "date": date.today().isoformat(),
                    "repo_root": str(source_root),
                    "notes": [
                        "Generated via architext audit command",
                        "Uses Architext tasks in src/tasks.py (static + heuristic; no vector store required).",
                    ],
                },
            )

            thresholds = {
                "min_health_score": args.min_health_score,
                "max_anti_patterns": args.max_anti_patterns,
                "max_silent_failures": args.max_silent_failures,
                "max_duplication": args.max_duplication,
                "max_semantic_duplication": args.max_semantic_duplication,
                "max_security_findings": args.max_security_findings,
            }
            failures = []
            if thresholds["min_health_score"] is not None and summary["health_score"] is not None:
                if summary["health_score"] < thresholds["min_health_score"]:
                    failures.append("health_score")
            if thresholds["max_anti_patterns"] is not None:
                if summary["anti_pattern_count"] > thresholds["max_anti_patterns"]:
                    failures.append("anti_pattern_count")
            if thresholds["max_silent_failures"] is not None:
                if summary["silent_failures"] > thresholds["max_silent_failures"]:
                    failures.append("silent_failures")
            if thresholds["max_duplication"] is not None:
                if summary["duplication_findings"] > thresholds["max_duplication"]:
                    failures.append("duplication_findings")
            if thresholds["max_semantic_duplication"] is not None:
                if summary["semantic_duplication_findings"] > thresholds["max_semantic_duplication"]:
                    failures.append("semantic_duplication_findings")
            if thresholds["max_security_findings"] is not None:
                if summary["security_findings"] > thresholds["max_security_findings"]:
                    failures.append("security_findings")

            print(f"Wrote refactor data to: {out_dir}")
            if failures and (args.ci or any(value is not None for value in thresholds.values())):
                print(f"Audit failed thresholds: {', '.join(failures)}")
                sys.exit(2)
            return

        settings = load_settings(env_file=args.env_file)
        storage_path = _resolve_storage(getattr(args, "storage", None), settings.storage_path)

        def _run_task_cli(task_name: str, **extra_kwargs):
            return run_task(
                task_name,
                storage_path=storage_path if not args.source else None,
                source_path=args.source,
                output_format=getattr(args, "output_format", None),
                depth=getattr(args, "depth", None),
                module=getattr(args, "module", None),
                output_dir=getattr(args, "output", None),
                **extra_kwargs,
            )

        if args.command == "analyze-structure":
            result = _run_task_cli("analyze-structure")
            _print_task_result(result)
            return

        if args.command == "tech-stack":
            result = _run_task_cli("tech-stack")
            _print_task_result(result)
            return

        if args.command == "query-diagnostics":
            if not storage_path:
                print("Storage path is required for diagnostics")
                sys.exit(1)
            result = query_diagnostics(storage_path=storage_path, query_text=args.text)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as handle:
                    json.dump(result, handle, indent=2)
                print(f"Diagnostics saved to {args.output}")
            else:
                print(json.dumps(result, indent=2))
            return

        if args.command == "detect-anti-patterns":
            result = _run_task_cli("detect-anti-patterns")
            _print_task_result(result)
            return

        if args.command == "health-score":
            result = _run_task_cli("health-score")
            _print_task_result(result)
            return

        if args.command == "impact-analysis":
            result = _run_task_cli("impact-analysis")
            _print_task_result(result)
            return

        if args.command == "refactoring-recommendations":
            result = _run_task_cli("refactoring-recommendations")
            _print_task_result(result)
            return

        if args.command == "generate-docs":
            result = _run_task_cli("generate-docs")
            _print_task_result(result)
            return

        if args.command == "dependency-graph":
            result = _run_task_cli("dependency-graph")
            _print_task_result(result)
            return

        if args.command == "test-coverage":
            result = _run_task_cli("test-coverage")
            _print_task_result(result)
            return

        if args.command == "detect-patterns":
            result = _run_task_cli("detect-patterns")
            _print_task_result(result)
            return

        if args.command == "diff-architecture":
            baseline_files = None
            if args.baseline:
                if not os.path.exists(args.baseline):
                    print(f"Baseline file not found: {args.baseline}")
                    print("Provide a JSON file containing a list of file paths.")
                    sys.exit(1)
                try:
                    with open(args.baseline, "r", encoding="utf-8") as handle:
                        baseline_files = json.load(handle)
                except Exception as exc:
                    print(f"Failed to read baseline JSON: {exc}")
                    sys.exit(1)
            result = _run_task_cli("diff-architecture", baseline_files=baseline_files)
            _print_task_result(result)
            return

        if args.command == "onboarding-guide":
            result = _run_task_cli("onboarding-guide")
            _print_task_result(result)
            return

        if args.command == "detect-vulnerabilities":
            result = _run_task_cli("detect-vulnerabilities")
            _print_task_result(result)
            return

        if args.command == "logic-gap-analysis":
            result = _run_task_cli("logic-gap-analysis")
            _print_task_result(result)
            return

        if args.command == "identify-silent-failures":
            result = _run_task_cli("identify-silent-failures")
            _print_task_result(result)
            return

        if args.command == "security-heuristics":
            result = _run_task_cli("security-heuristics")
            _print_task_result(result)
            return

        if args.command == "code-knowledge-graph":
            result = _run_task_cli("code-knowledge-graph")
            _print_task_result(result)
            return

        if args.command == "synthesis-roadmap":
            result = _run_task_cli("synthesis-roadmap")
            _print_task_result(result)
            return

        if args.command == "detect-duplication":
            result = _run_task_cli(
                "detect-duplication",
                min_lines=args.min_lines,
                max_findings=args.max_findings,
            )
            _print_task_result(result)
            return

        if args.command == "detect-duplication-semantic":
            result = _run_task_cli(
                "detect-duplication-semantic",
                min_tokens=args.min_tokens,
                max_findings=args.max_findings,
            )
            _print_task_result(result)
            return

    # All other commands need settings and LLM init
    try:
        settings = load_settings(env_file=args.env_file)

        # Provider choices should be configured via .env or config file (no CLI overrides)
        logger.info("Initializing AI models...")
        initialize_settings(settings)
    except Exception as e:
        logger.error(f"Failed to initialize AI models: {e}")
        print("Ensure LLM/embedding backends are reachable and configuration is valid.")
        sys.exit(1)

    if args.command == "index":
        # Compute storage path (auto if not provided)
        if args.storage:
            storage_path = _resolve_storage(args.storage, settings.storage_path)
        else:
            storage_path = _compute_auto_storage(args.source, settings.storage_path)
            logger.info(f"Using auto storage path: {storage_path}")

        # Handle preview/dry-run mode
        if args.preview or args.dry_run:
            logger.info("Running in PREVIEW mode (no indexing)")
            indexer = DryRunIndexer(logger)
            preview = indexer.preview(args.source)
            if args.preview:
                # Output structured JSON plan
                plan = {
                    "plan_id": f"plan-{hash(args.source) % 1000000}",  # Simple ID
                    "source": args.source,
                    "resolved_path": str(preview.get("resolved_path", "")),
                    "file_count": preview.get("documents", 0),
                    "doc_estimate": preview.get("documents", 0),  # Placeholder
                    "suggested_storage": storage_path,
                    "warnings": preview.get("warnings", []),
                }
                print(json.dumps(plan, indent=2))
            else:
                # Legacy dry-run output
                print("\n" + "="*60)
                print("DRY-RUN PREVIEW")
                print("="*60)
                print(f"Source: {preview.get('source')}")
                print(f"Resolved: {preview.get('resolved_path')}")
                print(f"Documents: {preview.get('documents', 'N/A')}")
                if "file_types" in preview:
                    print("\nFile types:")
                    for ext, count in preview["file_types"].items():
                        print(f"  {ext}: {count}")
                if "error" in preview:
                    print(f"Error: {preview['error']}")
                print("="*60 + "\n")
            return

        try:
            logger.info(f"Resolving source: {args.source}")
            # Use operational defaults from settings (no CLI flags for these)
            source_path = resolve_source(
                args.source,
                use_cache=settings.cache_enabled,
                ssh_key=settings.ssh_key,
            )
            logger.info(f"Starting indexing for: {source_path}")
            file_paths = gather_index_files(str(source_path))
            logger.info(f"Found {len(file_paths)} files to index")
            create_index_from_paths(file_paths, storage_path, settings=settings)
            logger.info(f"Indexing complete! Data saved to {storage_path}")
        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            sys.exit(1)

    elif args.command == "query":
        storage_path = _resolve_storage(args.storage, settings.storage_path)
        try:
            logger.info(f"Loading index from: {storage_path}")
            index = load_existing_index(storage_path)
            logger.info("Generating answer...")
            response = query_index(index, args.text, settings=settings)
            
            # Format output
            formatted = format_response(response, format=args.format)
            print("\n" + "="*40)
            print(f"RESPONSE:\n{formatted}")
            print("="*40 + "\n")
        except Exception as e:
            logger.error(f"Error during querying: {e}")
            sys.exit(1)

    elif args.command == "ask":
        storage_path = _resolve_storage(args.storage, settings.storage_path)
        try:
            logger.info(f"Loading index from: {storage_path}")
            index = load_existing_index(storage_path)
            logger.info("Generating agent response...")
            response = query_index(index, args.text, settings=settings)

            payload = (
                to_agent_response_compact(response)
                if args.compact
                else to_agent_response(response)
            )
            print(json.dumps(payload, indent=2))
        except Exception as e:
            logger.error(f"Error during ask: {e}")
            sys.exit(1)


def _print_task_result(result):
    if result.get("format") == "markdown" or result.get("format") == "mermaid":
        print(result.get("content", ""))
    else:
        print(json.dumps(result, indent=2))


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

if __name__ == "__main__":
    main()
