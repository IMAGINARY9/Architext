import argparse
import sys
import os

# Ensure src can be imported if resolving paths is tricky, though running as module (python -m src.cli) usually handles this.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_settings
from src.indexer import initialize_settings, load_documents, create_index, load_existing_index, query_index
from src.ingestor import resolve_source, cleanup_cache

def _build_parser():
    parser = argparse.ArgumentParser(description="Architext CLI: Local Codebase RAG")
    parser.add_argument(
        "--env-file",
        dest="env_file",
        help="Optional path to .env file for configuration overrides",
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

    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a question about the indexed code")
    query_parser.add_argument("text", help="The question/query string")
    query_parser.add_argument("--storage", help="Path to load the vector DB from (overrides config)")

    # Cache cleanup command
    cache_parser = subparsers.add_parser("cache-cleanup", help="Clean up old cached repos")
    cache_parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Remove repos unused for more than N days (default: 30)",
    )

    return parser


def _resolve_storage(args_storage: str, default_storage: str) -> str:
    return args_storage if args_storage else default_storage


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command not in ["index", "query", "cache-cleanup"]:
        parser.print_help()
        sys.exit(1)

    # Cache cleanup doesn't need LLM/embedding init
    if args.command == "cache-cleanup":
        removed = cleanup_cache(max_age_days=args.max_age)
        print(f"Cleanup complete. Removed {removed} cached repo(s).")
        return

    try:
        settings = load_settings(env_file=args.env_file)
        print("Initializing AI models...")
        initialize_settings(settings)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize AI models: {e}")
        print("Ensure LLM/embedding backends are reachable and configuration is valid.")
        sys.exit(1)

    if args.command == "index":
        storage_path = _resolve_storage(args.storage, settings.storage_path)
        try:
            print(f"Resolving source: {args.source}")
            source_path = resolve_source(args.source, use_cache=not args.no_cache)
            print(f"Starting indexing for: {source_path}")
            documents = load_documents(str(source_path))
            create_index(documents, storage_path)
            print("Indexing complete! Data saved to " + storage_path)
        except Exception as e:
            print(f"Error during indexing: {e}")
            sys.exit(1)

    elif args.command == "query":
        storage_path = _resolve_storage(args.storage, settings.storage_path)
        try:
            print(f"Loading index from: {storage_path}")
            index = load_existing_index(storage_path)
            print("Generating answer...")
            response = query_index(index, args.text)
            print("\n" + "="*40)
            print(f"RESPONSE:\n{response}")
            print("="*40 + "\n")
        except Exception as e:
            print(f"Error during querying: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
