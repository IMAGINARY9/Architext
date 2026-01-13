import pytest
from src.indexer import load_documents, initialize_settings
import os

def test_load_documents_security(temp_repo_path):
    """Ensure load_documents ignores hidden files and .git directories."""
    documents = load_documents(temp_repo_path)
    
    loaded_filenames = [os.path.basename(doc.metadata["file_path"]) for doc in documents]
    
    assert "main.py" in loaded_filenames
    assert ".env" not in loaded_filenames
    assert "HEAD" not in loaded_filenames
    assert len(documents) == 1

def test_initialize_settings_graceful_fallback(mocker):
    """Test that settings initialization handles model checks securely."""
    # Mock OS environ
    mocker.patch.dict(os.environ, {"OPENAI_API_BASE": "http://fake:5000/v1"})
    
    # Mock requests.get to raise connection error (simulating Oobabooga down)
    mock_get = mocker.patch("requests.get", side_effect=Exception("Connection refused"))
    
    # Mock OpenAILike and HuggingFaceEmbedding
    # We must ensure the return value of OpenAILike() is an instance of LLM for LlamaIndex validation
    from llama_index.core.llms import LLM
    mock_llm_cls = mocker.patch("src.indexer.OpenAILike")
    mock_llm_instance = mocker.Mock(spec=LLM)
    mock_llm_cls.return_value = mock_llm_instance
    
    # Also mock embeddings
    mock_embed = mocker.patch("src.indexer.HuggingFaceEmbedding")

    # Mock Settings to prevent real assignment logic triggering validation if we can't easily satisfy it, 
    # OR rely on the spec=LLM above. 
    # Let's try relying on spec=LLM first, but since Settings.llm does introspection, it might need more.
    # Actually, failure happened at verify_llm: assert isinstance(llm, LLM)
    # mocker.Mock(spec=LLM) should pass isinstance check if using spec properly? 
    # Wait, 'spec' in Mock usually works for 'isinstance' if setup right, but sometimes strict type checks fail.
    # Alternative: patch Settings directly in the test to verify assignment happened without validation logic.
    
    mock_settings = mocker.patch("src.indexer.Settings")
    
    # Run init
    initialize_settings()
    
    # Should still initialize LLM (with generic settings or fallback) and Embeddings
    assert mock_llm_cls.called
    assert mock_embed.called
    # Verify assignment
    assert mock_settings.llm == mock_llm_instance
