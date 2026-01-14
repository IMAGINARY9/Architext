import argparse
import sys
import os

# Ensure src can be imported if resolving paths is tricky, though running as module (python -m src.cli) usually handles this.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_settings
from src.indexer import initialize_settings, load_documents, create_index, load_existing_index, query_index
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
        "--llm-provider",
        choices=["local", "openai", "gemini", "anthropic"],
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
            source_path = resolve_source(args.source, use_cache=not args.no_cache)
            logger.info(f"Starting indexing for: {source_path}")
            documents = load_documents(str(source_path))
            logger.info(f"Loaded {len(documents)} documents")
            create_index(documents, storage_path)
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

if __name__ == "__main__":
    main()
