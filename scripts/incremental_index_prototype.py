"""Prototype benchmark for incremental vs full indexing file discovery.

This script compares:
1. Full manifest scan time.
2. Incremental target selection time after synthetic file changes.

It writes a markdown report to docs/benchmarks/INCREMENTAL_INDEX_PROTOTYPE.md.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.indexer_components.incremental import (
    build_manifest,
    select_incremental_index_targets,
)


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE = WORKSPACE / "src"
REPORT = WORKSPACE / "docs" / "benchmarks" / "INCREMENTAL_INDEX_PROTOTYPE.md"
JSON_REPORT = WORKSPACE / "docs" / "benchmarks" / "incremental_index_prototype_metrics.json"


def _ns_to_ms(delta_ns: int) -> float:
    return round(delta_ns / 1_000_000, 3)


def main() -> None:
    start = time.perf_counter_ns()
    baseline_manifest = build_manifest(str(SOURCE))
    full_scan_ns = time.perf_counter_ns() - start

    # Synthetic change simulation: mark every 20th file as changed.
    current_manifest = dict(baseline_manifest)
    for idx, path in enumerate(sorted(current_manifest.keys())):
        if idx % 20 == 0:
            current_manifest[path] = f"{current_manifest[path]}:changed"

    incremental_start = time.perf_counter_ns()
    targets, fallback = select_incremental_index_targets(
        baseline_manifest,
        current_manifest,
        change_ratio_threshold=0.35,
    )
    incremental_ns = time.perf_counter_ns() - incremental_start

    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE),
        "total_files": len(baseline_manifest),
        "changed_files": len(targets),
        "fallback_to_full": fallback,
        "full_scan_ms": _ns_to_ms(full_scan_ns),
        "incremental_selection_ms": _ns_to_ms(incremental_ns),
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "\n".join(
            [
                "# Incremental Indexing Prototype",
                "",
                f"Generated: {metrics['timestamp']}",
                "",
                "## Results",
                f"- Total files: {metrics['total_files']}",
                f"- Incremental target files: {metrics['changed_files']}",
                f"- Full scan time (ms): {metrics['full_scan_ms']}",
                f"- Incremental selection time (ms): {metrics['incremental_selection_ms']}",
                f"- Fallback to full indexing: {metrics['fallback_to_full']}",
                "",
                "## Trade-offs",
                "- Incremental mode reduces candidate set when file-change ratio is low.",
                "- Full re-index remains safer fallback when change ratio exceeds threshold.",
                "- Prototype currently tracks file metadata (size + mtime), not semantic deltas.",
            ]
        ),
        encoding="utf-8",
    )
    JSON_REPORT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
