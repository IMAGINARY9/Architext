# Documentation

## Quick Links

| Document | Purpose | Lines |
|----------|---------|-------|
| **[README.md](README.md)** | Getting started, installation, quick examples | 36 |
| **[PHASE_1_COMPLETE.md](PHASE_1_COMPLETE.md)** | Phase 1 summary, test results, architecture | 186 |
| **[DEVELOPER_REFERENCE.md](DEVELOPER_REFERENCE.md)** | Module API, common tasks, testing | 65 |
| **[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)** | Full roadmap (Phases 1-3) | 250 |

## Entry Points

**I want to...**
- Get started → [README.md](README.md)
- Understand Phase 1 completion → [PHASE_1_COMPLETE.md](PHASE_1_COMPLETE.md)
- Develop / extend code → [DEVELOPER_REFERENCE.md](DEVELOPER_REFERENCE.md)
- See the roadmap → [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)

## Quick Commands

```bash
# Setup
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Test (all 38 passing)
pytest tests/ -v

# Index a repo
python -m src.cli index ./src --storage ./my-index

# Query
python -m src.cli query "How does this work?" --storage ./my-index
```
