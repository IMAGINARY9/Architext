"""Quick reindexing script to recreate index with logical chunking and line numbers."""

import asyncio
import shutil
from pathlib import Path


async def reindex_with_logical_chunking():
    """Recreate the src-5640d5cd index with proper metadata."""
    
    # Import after we know we're in the right environment
    from src.indexer import create_index_from_paths, gather_index_files
    from src.config import load_settings
    
    print("🔄 Reindexing with logical chunking enabled...")
    print()
    
    # Configuration
    source_path = "./src"
    old_storage = "./storage/src-5640d5cd"
    new_storage = "./storage/src-5640d5cd-v2"
    
    # Load settings (chunking_strategy is already "logical" by default)
    settings = load_settings()
    print(f"✓ Chunking strategy: {settings.chunking_strategy}")
    print(f"✓ Hybrid search: {settings.enable_hybrid}")
    print(f"✓ Source path: {source_path}")
    print(f"✓ New storage: {new_storage}")
    print()
    
    # Check if new storage exists
    new_storage_path = Path(new_storage)
    if new_storage_path.exists():
        print(f"⚠️  Storage path {new_storage} already exists")
        response = input("Delete and recreate? (y/N): ")
        if response.lower() != 'y':
            print("❌ Aborted")
            return
        print(f"🗑️  Removing {new_storage}...")
        shutil.rmtree(new_storage_path)
    
    # Gather files
    print("📂 Gathering files...")
    files = gather_index_files(source_path)
    print(f"✓ Found {len(files)} files to index")
    print()
    
    # Create index
    print("🔨 Creating index with logical chunking...")
    print("   (This will take a few moments)")
    
    def progress_callback(info):
        if info.get("stage") == "loading":
            count = info.get("count", 0)
            total = info.get("total", 0)
            if total > 0:
                pct = (count / total) * 100
                print(f"   Loading: {count}/{total} files ({pct:.0f}%)")
        elif info.get("stage") == "embedding":
            print(f"   Embedding: {info.get('message', '')}")
    
    result = await asyncio.to_thread(
        create_index_from_paths,
        file_paths=files,
        storage_path=new_storage,
        progress_callback=progress_callback,
        settings=settings
    )
    
    print()
    print("✅ Indexing complete!")
    print(f"   Documents indexed: {result.get('documents_indexed', 0)}")
    print(f"   Storage path: {result.get('storage_path', '')}")
    print()
    
    # Verify metadata
    print("🔍 Verifying metadata structure...")
    import chromadb
    import json
    
    client = chromadb.PersistentClient(path=new_storage)
    collection = client.get_collection("architext_db")
    sample = collection.get(limit=1, include=["metadatas"])
    
    if sample["metadatas"]:
        metadata = sample["metadatas"][0]
        print(f"   Sample metadata keys: {list(metadata.keys())}")
        
        # Check for line numbers
        has_lines = "start_line" in metadata or "end_line" in metadata
        if has_lines:
            print("   ✅ Line number metadata present!")
            print(f"      file_path: {metadata.get('file_path', 'N/A')}")
            print(f"      start_line: {metadata.get('start_line', 'N/A')}")
            print(f"      end_line: {metadata.get('end_line', 'N/A')}")
        else:
            print("   ⚠️  No line number metadata found")
            print("      This may indicate logical chunking didn't work correctly")
    else:
        print("   ⚠️  No documents found in index")
    
    print()
    print("📝 Next steps:")
    print("   1. Start the server: ./start_windows.bat --api")
    print(f"   2. Update queries to use: name='src-5640d5cd-v2'")
    print("   3. Test queries and verify start_line/end_line are populated")
    print()
    print("   Or rename the new index to replace the old one:")
    print(f"      mv {new_storage} {old_storage}")


if __name__ == "__main__":
    asyncio.run(reindex_with_logical_chunking())
