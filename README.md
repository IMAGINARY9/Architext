# Architext

An intelligent "Codebase Architect" tool that indexes repositories and answers high-level architectural questions using Retrieval-Augmented Generation (RAG).

## 🎉 Phase 1 Complete (Production-Ready)

✅ **38/38 Tests Passing** | ✅ **3,500+ Lines** | ✅ **7 CLI Commands** | ✅ **Full Documentation**

**Key Features:**
- Universal ingestion (local paths, GitHub/GitLab/Gitea/SSH with caching)
- Flexible LLM backend (OpenAI, local Oobabooga, extensible to Gemini/Anthropic)
- Configuration-driven (Pydantic + .env with provider selection)
- Rich CLI (7 commands, verbose logging, dry-run mode, JSON output)
- Privacy-first (local models supported, no forced cloud calls)

**[→ See PHASE_1_COMPLETE.md for full completion details](PHASE_1_COMPLETE.md)**

## Getting Started

### Installation
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Quick Start
```bash
# Index a local repo
python -m src.cli index ./my-project --storage ./index --verbose

# Index a remote repo
python -m src.cli index https://github.com/user/repo --storage ./index

# Query the index
python -m src.cli query "How is authentication handled?" --storage ./index

# List available models
python -m src.cli list-models

# Clean cached repos
python -m src.cli cache-cleanup --max-age 30
```

### Run Tests
```bash
pytest tests/ -v
# Expected: 38/38 passing in ~7 seconds
```
