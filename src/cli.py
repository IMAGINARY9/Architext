import argparse
import sys
import os
import json

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
    query_diagnostics,
    detect_anti_patterns,
    health_score,
    impact_analysis,
    refactoring_recommendations,
    generate_docs,
    dependency_graph_export,
    test_coverage_analysis,
    architecture_pattern_detection,
    diff_architecture_review,
    onboarding_guide,
    detect_vulnerabilities,
    logic_gap_analysis,
    identify_silent_failures,
    security_heuristics,
)
from src.ingestor import resolve_source, cleanup_cache
from src.cli_utils import VerboseLogger, format_response, get_available_models_info, DryRunIndexer

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
    index_parser.add_argument("--storage", help="Path to save the vector DB (overrides config)")
    index_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable remote repo caching (local paths only)",
    )
    index_parser.add_argument(
        "--ssh-key",
        help="Path to SSH private key for cloning private repos",
    )
    index_parser.add_argument(
        "--llm-provider",
        choices=["local", "openai"],
        help="Override LLM provider from config",
    )
    index_parser.add_argument(
        "--embedding-provider",
        choices=["huggingface", "openai"],
        help="Override embedding provider from config",
    )
    index_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview indexing without persisting (shows file counts, etc)",
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a question about the indexed code")
    query_parser.add_argument("text", help="The question/query string")
    query_parser.add_argument("--storage", help="Path to load the vector DB from (overrides config)")
    query_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for the response",
    )
    query_parser.add_argument(
        "--enable-hybrid",
        action="store_true",
        help="Enable hybrid keyword+vector scoring for this query",
    )
    query_parser.add_argument(
        "--hybrid-alpha",
        type=float,
        help="Hybrid weight for vector score (0-1). Higher favors vectors",
    )
    query_parser.add_argument(
        "--enable-rerank",
        action="store_true",
        help="Enable cross-encoder reranking for this query",
    )
    query_parser.add_argument(
        "--rerank-model",
        help="Cross-encoder model name for reranking",
    )
    query_parser.add_argument(
        "--rerank-top-n",
        type=int,
        help="Number of top results to rerank",
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
    }:
        settings = load_settings(env_file=args.env_file)
        storage_path = _resolve_storage(getattr(args, "storage", None), settings.storage_path)

        if args.command == "analyze-structure":
            result = analyze_structure(
                storage_path=storage_path if not args.source else None,
                source_path=args.source,
                depth=args.depth,
                output_format=args.output_format,
            )
            _print_task_result(result)
            return

        if args.command == "tech-stack":
            result = tech_stack(
                storage_path=storage_path if not args.source else None,
                source_path=args.source,
                output_format=args.output_format,
            )
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
            result = detect_anti_patterns(
                storage_path=storage_path if not args.source else None,
                source_path=args.source,
            )
            _print_task_result(result)
            return

        if args.command == "health-score":
            result = health_score(
                storage_path=storage_path if not args.source else None,
                source_path=args.source,
            )
            _print_task_result(result)
            return

        if args.command == "impact-analysis":
            result = impact_analysis(
                module=args.module,
                storage_path=storage_path if not args.source else None,
                source_path=args.source,
            )
            _print_task_result(result)
            return

        if args.command == "refactoring-recommendations":
            result = refactoring_recommendations(
                storage_path=storage_path if not args.source else None,
                source_path=args.source,
            )
            _print_task_result(result)
            return

        if args.command == "generate-docs":
            result = generate_docs(
                storage_path=storage_path if not args.source else None,
                source_path=args.source,
                output_dir=args.output,
            )
            _print_task_result(result)
            return

        if args.command == "dependency-graph":
            result = dependency_graph_export(
                storage_path=storage_path if not args.source else None,
                source_path=args.source,
                output_format=args.output_format,
            )
            _print_task_result(result)
            return

        if args.command == "test-coverage":
            result = test_coverage_analysis(
                storage_path=storage_path if not args.source else None,
                source_path=args.source,
            )
            _print_task_result(result)
            return

        if args.command == "detect-patterns":
            result = architecture_pattern_detection(
                storage_path=storage_path if not args.source else None,
                source_path=args.source,
            )
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
            result = diff_architecture_review(
                storage_path=storage_path if not args.source else None,
                source_path=args.source,
                baseline_files=baseline_files,
            )
            _print_task_result(result)
            return

        if args.command == "onboarding-guide":
            result = onboarding_guide(
                storage_path=storage_path if not args.source else None,
                source_path=args.source,
            )
            _print_task_result(result)
            return

        if args.command == "detect-vulnerabilities":
            result = detect_vulnerabilities(
                storage_path=storage_path if not args.source else None,
                source_path=args.source,
            )
            _print_task_result(result)
            return

        if args.command == "logic-gap-analysis":
            result = logic_gap_analysis(
                storage_path=storage_path if not args.source else None,
                source_path=args.source,
            )
            _print_task_result(result)
            return

        if args.command == "identify-silent-failures":
            result = identify_silent_failures(
                storage_path=storage_path if not args.source else None,
                source_path=args.source,
            )
            _print_task_result(result)
            return

        if args.command == "security-heuristics":
            result = security_heuristics(
                storage_path=storage_path if not args.source else None,
                source_path=args.source,
            )
            _print_task_result(result)
            return

    # All other commands need settings and LLM init
    try:
        settings = load_settings(env_file=args.env_file)

        # Override providers if specified
        if hasattr(args, "llm_provider") and args.llm_provider:
            settings.llm_provider = args.llm_provider
            logger.debug(f"Overriding LLM provider: {args.llm_provider}")
        if hasattr(args, "embedding_provider") and args.embedding_provider:
            settings.embedding_provider = args.embedding_provider
            logger.debug(f"Overriding embedding provider: {args.embedding_provider}")

        logger.info("Initializing AI models...")
        initialize_settings(settings)
    except Exception as e:
        logger.error(f"Failed to initialize AI models: {e}")
        print("Ensure LLM/embedding backends are reachable and configuration is valid.")
        sys.exit(1)

    if args.command == "index":
        storage_path = _resolve_storage(args.storage, settings.storage_path)

        # Handle dry-run mode
        if args.dry_run:
            logger.info("Running in DRY-RUN mode (no indexing)")
            indexer = DryRunIndexer(logger)
            preview = indexer.preview(args.source)
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
            source_path = resolve_source(
                args.source,
                use_cache=not args.no_cache,
                ssh_key=args.ssh_key,
            )
            logger.info(f"Starting indexing for: {source_path}")
            file_paths = gather_index_files(str(source_path))
            logger.info(f"Found {len(file_paths)} files to index")
            create_index_from_paths(file_paths, storage_path)
            logger.info(f"Indexing complete! Data saved to {storage_path}")
        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            sys.exit(1)

    elif args.command == "query":
        storage_path = _resolve_storage(args.storage, settings.storage_path)
        if args.enable_hybrid:
            settings.enable_hybrid = True
        if args.hybrid_alpha is not None:
            settings.hybrid_alpha = args.hybrid_alpha
        if args.enable_rerank:
            settings.enable_rerank = True
        if args.rerank_model:
            settings.rerank_model = args.rerank_model
        if args.rerank_top_n is not None:
            settings.rerank_top_n = args.rerank_top_n
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


def _print_task_result(result):
    if result.get("format") == "markdown" or result.get("format") == "mermaid":
        print(result.get("content", ""))
    else:
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
