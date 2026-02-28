"""Structure analysis task — thin shim delegating to analysis/structure.py.

Backward-compatibility wrapper. All logic lives in
``src.tasks.analysis.structure.StructureAnalysisTask``.
"""
from __future__ import annotations

from src.tasks.analysis.structure import (  # noqa: F401
    StructureAnalysisTask,
    analyze_structure_v2 as analyze_structure,
)
