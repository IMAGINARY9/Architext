"""Query helpers — backward-compatibility shim.

Canonical implementations now live in ``src.indexer``.
"""
from __future__ import annotations

from src.indexer import _keyword_score, _tokenize  # noqa: F401

__all__ = ["_tokenize", "_keyword_score"]
