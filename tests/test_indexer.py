import os
import pytest

from src.config import ArchitextSettings
from src.indexer import initialize_settings, load_documents

def test_load_documents_security(temp_repo_path):
    """Ensure load_documents ignores hidden files and .git directories."""
    documents = load_documents(temp_repo_path)
    
    loaded_filenames = [os.path.basename(doc.metadata["file_path"]) for doc in documents]
    
    assert "main.py" in loaded_filenames
    assert ".env" not in loaded_filenames
    assert "HEAD" not in loaded_filenames
    assert len(documents) == 1

def test_initialize_settings_graceful_fallback(mocker):
    """Ensure we wire LLM and embedding via config without hitting real services."""

    cfg = ArchitextSettings()
    mock_llm = mocker.Mock()
    mock_embed = mocker.Mock()

    mock_build_llm = mocker.patch("src.indexer._build_llm", return_value=mock_llm)
    mock_build_embed = mocker.patch("src.indexer._build_embedding", return_value=mock_embed)
    mock_settings = mocker.patch("src.indexer.Settings")

    initialize_settings(cfg)

    mock_build_llm.assert_called_once_with(cfg)
    mock_build_embed.assert_called_once_with(cfg)
    assert mock_settings.llm == mock_llm
    assert mock_settings.embed_model == mock_embed


def test_openai_embedding_requires_key(mocker):
    cfg = ArchitextSettings(embedding_provider="openai", openai_api_key="")
    with pytest.raises(ValueError):
        initialize_settings(cfg)


def test_openai_embedding_builds(mocker):
    cfg = ArchitextSettings(
        embedding_provider="openai",
        embedding_model_name="text-embedding-3-small",
        openai_api_key="key",
        openai_api_base="https://api.openai.com/v1",
    )

    mock_openai_embed = mocker.patch("src.indexer.OpenAIEmbedding")
    mock_llm = mocker.patch("src.indexer._build_llm", return_value=mocker.Mock())
    mock_settings = mocker.patch("src.indexer.Settings")

    initialize_settings(cfg)

    mock_openai_embed.assert_called_once_with(
        model="text-embedding-3-small",
        api_key="key",
        api_base="https://api.openai.com/v1",
    )
    mock_llm.assert_called_once_with(cfg)
    assert mock_settings.embed_model == mock_openai_embed.return_value
