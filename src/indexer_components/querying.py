"""Query helpers and retriever utilities."""
from __future__ import annotations

import re
from typing import List


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _keyword_score(query: str, document: str) -> float:
    query_terms = set(_tokenize(query))
    if not query_terms:
        return 0.0
    doc_terms = set(_tokenize(document))
    return len(query_terms & doc_terms) / len(query_terms)


