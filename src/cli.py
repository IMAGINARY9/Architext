import argparse
import sys
import os

# Ensure src can be imported if resolving paths is tricky, though running as module (python -m src.cli) usually handles this.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_settings
from src.indexer import initialize_settings, load_documents, create_index, load_existing_index, query_index

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
    index_parser.add_argument("path", help="Path to the repository or folder to index")
    index_parser.add_argument("--storage", help="Path to save the vector DB (overrides config)")

    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a question about the indexed code")
    query_parser.add_argument("text", help="The question/query string")
    query_parser.add_argument("--storage", help="Path to load the vector DB from (overrides config)")

    return parser


def _resolve_storage(args_storage: str, default_storage: str) -> str:
    return args_storage if args_storage else default_storage


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command not in ["index", "query"]:
        parser.print_help()
        sys.exit(1)

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
            print(f"Starting indexing for: {args.path}")
            documents = load_documents(args.path)
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
