"""Benchmark harness for indexing/query latency and process profile.

This script is deterministic and local-only. It avoids LLM synthesis latency by
using query diagnostics retrieval for query timing.
"""
from __future__ import annotations

import json
import shutil
import statistics
import time
import tracemalloc
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from src.config import load_settings
from src.indexer import create_index_from_paths, gather_index_files, initialize_settings
from src.tasks.query import query_diagnostics


@dataclass
class IterationResult:
    profile: str
    iteration: int
    file_count: int
    index_seconds: float
    query_seconds: float
    peak_python_memory_mb: float
    cpu_process_ratio: float


@dataclass
class ProfileSummary:
    profile: str
    path: str
    iterations: int
    files_mean: float
    index_p50: float
    index_p95: float
    query_p50: float
    query_p95: float
    peak_memory_mb_p95: float
    cpu_ratio_p95: float


def _percentile(values: Iterable[float], pct: float) -> float:
    seq = sorted(values)
    if not seq:
        return 0.0
    if len(seq) == 1:
        return float(seq[0])
    rank = (len(seq) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(seq) - 1)
    if low == high:
        return float(seq[low])
    fraction = rank - low
    return float(seq[low] + (seq[high] - seq[low]) * fraction)


def _run_profile(profile: str, source_path: Path, root_storage: Path, iterations: int) -> tuple[list[IterationResult], ProfileSummary]:
    results: list[IterationResult] = []
    file_paths = gather_index_files(str(source_path))

    for idx in range(1, iterations + 1):
        storage_path = root_storage / f"{profile}_{idx}"
        if storage_path.exists():
            shutil.rmtree(storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)

        tracemalloc.start()
        start_wall = time.perf_counter()
        start_cpu = time.process_time()

        create_index_from_paths(
            file_paths=file_paths,
            storage_path=str(storage_path),
            batch_size=300,
        )

        index_seconds = time.perf_counter() - start_wall

        query_start = time.perf_counter()
        query_diagnostics(
            storage_path=str(storage_path),
            query_text="How is task execution history implemented?",
        )
        query_seconds = time.perf_counter() - query_start

        end_cpu = time.process_time()
        wall_total = time.perf_counter() - start_wall

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        cpu_ratio = (end_cpu - start_cpu) / wall_total if wall_total > 0 else 0.0

        results.append(
            IterationResult(
                profile=profile,
                iteration=idx,
                file_count=len(file_paths),
                index_seconds=round(index_seconds, 4),
                query_seconds=round(query_seconds, 4),
                peak_python_memory_mb=round(peak / (1024 * 1024), 2),
                cpu_process_ratio=round(cpu_ratio, 4),
            )
        )

    index_vals = [row.index_seconds for row in results]
    query_vals = [row.query_seconds for row in results]
    mem_vals = [row.peak_python_memory_mb for row in results]
    cpu_vals = [row.cpu_process_ratio for row in results]
    files = [row.file_count for row in results]

    summary = ProfileSummary(
        profile=profile,
        path=str(source_path).replace("\\", "/"),
        iterations=iterations,
        files_mean=round(statistics.mean(files), 2),
        index_p50=round(_percentile(index_vals, 0.5), 4),
        index_p95=round(_percentile(index_vals, 0.95), 4),
        query_p50=round(_percentile(query_vals, 0.5), 4),
        query_p95=round(_percentile(query_vals, 0.95), 4),
        peak_memory_mb_p95=round(_percentile(mem_vals, 0.95), 2),
        cpu_ratio_p95=round(_percentile(cpu_vals, 0.95), 4),
    )
    return results, summary


def main() -> None:
    initialize_settings(load_settings())

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "docs" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    storage_root = repo_root / "test-storage" / "bench"
    storage_root.mkdir(parents=True, exist_ok=True)

    profiles = {
        "small": repo_root / "src" / "tasks" / "analysis",
        "medium": repo_root / "src",
    }
    iterations = 5

    all_iterations: list[IterationResult] = []
    summaries: list[ProfileSummary] = []

    for profile_name, profile_path in profiles.items():
        results, summary = _run_profile(profile_name, profile_path, storage_root, iterations)
        all_iterations.extend(results)
        summaries.append(summary)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "iterations": [asdict(row) for row in all_iterations],
        "summaries": [asdict(row) for row in summaries],
        "notes": [
            "Query timing uses query_diagnostics retrieval path for deterministic local measurement.",
            "Memory is peak Python allocation (tracemalloc), not system-wide RSS.",
            "CPU ratio approximates process CPU time / wall time over each iteration.",
        ],
    }

    json_path = out_dir / f"metrics_{now}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Benchmark Matrix",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "| Profile | Source Path | Files (mean) | Index p50 (s) | Index p95 (s) | Query p50 (s) | Query p95 (s) | Peak Python Mem p95 (MB) | CPU Ratio p95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.profile} | {summary.path} | {summary.files_mean} | {summary.index_p50} | {summary.index_p95} | {summary.query_p50} | {summary.query_p95} | {summary.peak_memory_mb_p95} | {summary.cpu_ratio_p95} |"
        )

    lines.extend([
        "",
        "## Notes",
        "- Query measurements use retrieval diagnostics (no external LLM synthesis latency).",
        "- Memory metric is Python allocation peak via tracemalloc.",
        "- CPU ratio is process CPU time divided by wall-clock time for each run.",
        "",
        f"JSON artifact: docs/benchmarks/{json_path.name}",
    ])

    md_path = out_dir / "BENCHMARK_MATRIX.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(str(md_path))
    print(str(json_path))


if __name__ == "__main__":
    main()
