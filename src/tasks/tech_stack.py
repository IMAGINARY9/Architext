"""Technology stack analysis task — thin shim delegating to analysis/tech_stack.py.

Backward-compatibility wrapper. All logic lives in
``src.tasks.analysis.tech_stack.TechStackTask``.
"""
from __future__ import annotations

from src.tasks.analysis.tech_stack import (  # noqa: F401
    TechStackTask,
    tech_stack_v2 as tech_stack,
)
