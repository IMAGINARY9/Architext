"""Run the UX release gate and write generated outputs to ignored local paths."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute UX release gate and save generated summaries under .local/ux/."
    )
    parser.add_argument(
        "--report",
        default="docs/research/simulation-runs-2026-03-23.md",
        help="Path to simulation report markdown.",
    )
    parser.add_argument(
        "--out-dir",
        default=".local/ux",
        help="Output directory for generated summaries (ignored by git).",
    )
    parser.add_argument("--min-completion-rate", type=float, default=85.0)
    parser.add_argument("--max-time-to-query", type=float, default=15.0)
    parser.add_argument("--max-wrong-endpoint", type=float, default=1.0)
    parser.add_argument("--min-integration-correctness", type=float, default=3.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    python_executable = sys.executable
    base_cmd = [python_executable, "scripts/ux_simulation_kpi_summary.py", "--report", args.report]

    text_out = out_dir / "kpi-summary.txt"
    json_out = out_dir / "kpi-summary.json"
    csv_out = out_dir / "kpi-summary.csv"

    commands = [
        base_cmd + ["--format", "text", "--output", str(text_out)],
        base_cmd + ["--format", "json", "--output", str(json_out)],
        base_cmd + ["--format", "csv", "--output", str(csv_out)],
    ]

    for command in commands:
        result = _run(command)
        if result.returncode != 0:
            sys.stderr.write(result.stderr)
            return result.returncode

    gate_cmd = base_cmd + [
        "--check-thresholds",
        "--min-completion-rate",
        str(args.min_completion_rate),
        "--max-time-to-query",
        str(args.max_time_to_query),
        "--max-wrong-endpoint",
        str(args.max_wrong_endpoint),
        "--min-integration-correctness",
        str(args.min_integration_correctness),
    ]

    gate_result = _run(gate_cmd)
    if gate_result.stdout:
        print(gate_result.stdout, end="")
    if gate_result.returncode != 0:
        if gate_result.stderr:
            sys.stderr.write(gate_result.stderr)
        return gate_result.returncode

    print("UX release gate passed.")
    print(f"Generated outputs: {text_out}, {json_out}, {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
