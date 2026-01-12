import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

def initialize_settings():
    """Configure global LlamaIndex settings for local LLM and Embeddings."""
    # LLM Setup (Oobabooga)
    api_base = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:5000/v1")
    api_key = os.getenv("OPENAI_API_KEY", "local")
    
    print(f"Connecting to LLM at {api_base}...")
    llm = OpenAILike(
        model="local-model",
        api_base=api_base,
        api_key=api_key,
        temperature=0.1,
        is_chat_model=True
    )
    
    # Embedding Setup (Local HuggingFace)
    print("Loading embedding model (sentence-transformers/all-mpnet-base-v2)...")
    # Use a local cache directory to avoid path length issues and permission errors
    cache_folder = os.path.join(os.getcwd(), "models_cache")
    os.makedirs(cache_folder, exist_ok=True)
    
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-mpnet-base-v2",
        cache_folder=cache_folder
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

def load_documents(path: str):
    """Recursively read files from the directory."""
    if not os.path.exists(path):
        raise ValueError(f"Path does not exist: {path}")
        
    print(f"Loading documents from {path}...")
    # recursive=True allows reading subdirectories
    reader = SimpleDirectoryReader(input_dir=path, recursive=True)
    documents = reader.load_data()
    print(f"Loaded {len(documents)} documents.")
    return documents

def create_index(documents, storage_path="./storage"):
    """Create and persist a vector index from documents."""
    print(f"Initializing ChromaDB and Vector Store at {storage_path}...")
    
    # Create Chroma Client
    db = chromadb.PersistentClient(path=storage_path)
    chroma_collection = db.get_or_create_collection("architext_db")
    
    # Create Vector Store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print("Creating VectorStoreIndex (this may take a while)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    return index

def load_existing_index(storage_path="./storage"):
    """Load an existing index from ChromaDB."""
    print(f"Loading existing index from {storage_path}...")
    db = chromadb.PersistentClient(path=storage_path)
    chroma_collection = db.get_or_create_collection("architext_db")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
    )
    return index

def query_index(index, query_text: str):
    """Query the index and return the response."""
    print(f"Querying: {query_text}")
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    return response
