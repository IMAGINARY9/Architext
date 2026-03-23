"""Run the UX release gate and write generated outputs to ignored local paths."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True, capture_output=True)


def _extract_latest_cycle_number(report_text: str) -> int:
    matches = re.findall(r"^##\s+Cycle\s+(\d+)\s+Aggregated Metrics\s*$", report_text, re.MULTILINE)
    if not matches:
        return 1
    return max(int(match) for match in matches)


def _extract_latest_cycle_actions(report_text: str, cycle_number: int) -> list[str]:
    heading = f"## Cycle {cycle_number} Findings"
    start = report_text.find(heading)
    if start == -1:
        return []

    section_start = start + len(heading)
    next_heading = report_text.find("\n## ", section_start)
    section = report_text[section_start:] if next_heading == -1 else report_text[section_start:next_heading]

    actions: list[str] = []
    for line in section.splitlines():
        line = line.strip()
        if line.startswith("- Action:"):
            actions.append(line.replace("- Action:", "", 1).strip())
    return actions


def _append_release_gate_log(
    log_file: Path,
    release_tag: str,
    gate_command: str,
    decision: str,
    rationale: str,
    latest_metrics: dict[str, Any],
    actions: list[str],
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines: list[str] = []
    lines.append(f"\n## {timestamp} - {release_tag}\n")
    lines.append(f"- Release tag/version: {release_tag}")
    lines.append(f"- Gate command: `{gate_command}`")
    lines.append(f"- Decision: {decision}")
    lines.append(f"- Rationale: {rationale}")
    lines.append("- KPI snapshot:")
    lines.append(f"  - completion rate: {latest_metrics['completion_rate']:.2f}%")
    lines.append(
        f"  - time to first successful query: {latest_metrics['time_to_query_min']:.2f} min"
    )
    lines.append(
        f"  - wrong-endpoint attempts (median): {latest_metrics['wrong_endpoint_attempts']:.2f}"
    )
    lines.append(
        f"  - integration correctness: {latest_metrics['integration_correctness']:.2f}/4"
    )

    if actions:
        lines.append("- Candidate improvements (from latest cycle findings):")
        for action in actions:
            lines.append(f"  - {action}")
    else:
        lines.append("- Candidate improvements: none extracted from latest cycle findings.")

    lines.append("- Follow-up: continue lightweight cycle per release; escalate on threshold failure.")

    with log_file.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


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
    parser.add_argument(
        "--release-tag",
        default="release-candidate",
        help="Release tag/name used in release gate log entry.",
    )
    parser.add_argument(
        "--log-file",
        default="docs/research/release-gate-log.md",
        help="Path to markdown release gate log.",
    )
    parser.add_argument(
        "--skip-log",
        action="store_true",
        help="Skip appending a release gate log entry.",
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

    report_text = Path(args.report).read_text(encoding="utf-8")
    latest_cycle_number = _extract_latest_cycle_number(report_text)
    latest_actions = _extract_latest_cycle_actions(report_text, latest_cycle_number)
    summary_json = json.loads(json_out.read_text(encoding="utf-8"))
    latest_metrics = summary_json["cycles"][-1]
    gate_command_text = (
        f"python scripts/run_ux_release_gate.py --release-tag {args.release_tag}"
    )

    gate_result = _run(gate_cmd)
    if gate_result.stdout:
        print(gate_result.stdout, end="")
    if gate_result.returncode != 0:
        if not args.skip_log:
            _append_release_gate_log(
                log_file=Path(args.log_file),
                release_tag=args.release_tag,
                gate_command=gate_command_text,
                decision="NO-GO",
                rationale="One or more KPI threshold checks failed.",
                latest_metrics=latest_metrics,
                actions=latest_actions,
            )
        if gate_result.stderr:
            sys.stderr.write(gate_result.stderr)
        return gate_result.returncode

    if not args.skip_log:
        _append_release_gate_log(
            log_file=Path(args.log_file),
            release_tag=args.release_tag,
            gate_command=gate_command_text,
            decision="GO",
            rationale="All KPI threshold checks passed.",
            latest_metrics=latest_metrics,
            actions=latest_actions,
        )

    print("UX release gate passed.")
    print(f"Generated outputs: {text_out}, {json_out}, {csv_out}")
    if args.skip_log:
        print("Release gate log update skipped (--skip-log).")
    else:
        print(f"Release gate log updated: {args.log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
