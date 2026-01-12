import argparse
import sys
import os

# Ensure src can be imported if resolving paths is tricky, though running as module (python -m src.cli) usually handles this.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.indexer import initialize_settings, load_documents, create_index, load_existing_index, query_index

def main():
    parser = argparse.ArgumentParser(description="Architext CLI: Local Codebase RAG")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a repository")
    index_parser.add_argument("path", help="Path to the repository or folder to index")
    index_parser.add_argument("--storage", default="./storage", help="Path to save the vector DB")

    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a question about the indexed code")
    query_parser.add_argument("text", help="The question/query string")
    query_parser.add_argument("--storage", default="./storage", help="Path to load the vector DB from")

    args = parser.parse_args()

    if args.command not in ["index", "query"]:
        parser.print_help()
        sys.exit(1)

    print("Initializing AI models...")
    try:
        initialize_settings()
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize AI models: {e}")
        print("Ensure Oobabooga is running and internet access is available for downloading embedding models.")
        sys.exit(1)

    if args.command == "index":
        try:
            print(f"Starting indexing for: {args.path}")
            documents = load_documents(args.path)
            create_index(documents, args.storage)
            print("Indexing complete! Data saved to " + args.storage)
        except Exception as e:
            print(f"Error during indexing: {e}")
            sys.exit(1)

    elif args.command == "query":
        try:
            print(f"Loading index from: {args.storage}")
            index = load_existing_index(args.storage)
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
