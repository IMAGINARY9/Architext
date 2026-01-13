# Developer Quick Reference

**For status and completion details, see [PHASE_1_COMPLETE.md](PHASE_1_COMPLETE.md)**

## Project Structure

```
src/
  config.py      # Pydantic settings + provider config
  ingestor.py    # Repo resolution (local/remote + caching)
  indexer.py     # Indexing pipeline (LlamaIndex + ChromaDB)
  cli.py         # CLI framework (7 commands)
  cli_utils.py   # Logging, formatting, model discovery

tests/
  test_ingestor.py          # 13 repo resolution tests
  test_indexer.py           # 4 indexing tests
  test_cli_integration.py   # 7 CLI integration tests
  test_cli_utils.py         # 14 utility tests
  test_integration.py       # 8 end-to-end tests

docs/
  IMPLEMENTATION_PLAN.md    # Roadmap (Phases 1-3)

storage/                    # Default ChromaDB vector store
.env                        # Configuration
```

## Core API Reference

### config.py - Pydantic Settings
```python
from src.config import ArchitextSettings
settings = ArchitextSettings()  # Loads from .env
```
Providers: `llm_provider` (local|openai|gemini|anthropic), `embedding_provider` (huggingface|openai)

### ingestor.py - Source Resolution
```python
from src.ingestor import resolve_source, cleanup_cache
path = resolve_source("./repo" or "https://github.com/user/repo")
cleanup_cache(max_age_days=30)
```

### indexer.py - Indexing Pipeline
```python
from src.indexer import initialize_settings, load_documents, create_index
settings = initialize_settings(ArchitextSettings())
docs = load_documents("/path/to/repo")
index = create_index(docs, storage_path="./storage")
```

### cli.py - CLI Commands
7 subcommands: `index`, `query`, `list-models`, `cache-cleanup`, `serve` (Phase 2)

Flags: `--storage`, `--llm-provider`, `--embedding-provider`, `--verbose`, `--dry-run`, `--format`, `--env-file`

## Testing

### Run Tests
```bash
pytest tests/ -v                    # All 38 tests
pytest tests/test_cli_utils.py -v   # 14 CLI utility tests
pytest tests/test_ingestor.py -v    # 13 ingestor tests
pytest tests/test_indexer.py -v     # 4 indexer tests
pytest tests/test_cli_integration.py -v  # 7 CLI tests
```

## Configuration

Edit `.env` to switch providers. See [PHASE_1_COMPLETE.md](PHASE_1_COMPLETE.md#configuration) for templates.

## Common Tasks

### Add a New CLI Command
1. Add subparser in `cli.py` `_build_parser()`
2. Implement handler function
3. Add tests in `tests/test_cli_integration.py`

### Add a New LLM Provider
1. Extend `ArchitextSettings` in `config.py`
2. Add factory logic in `indexer.py` `_build_llm()`
3. Update `cli_utils.py` `get_available_models_info()`
4. Add tests in `tests/test_indexer.py`
