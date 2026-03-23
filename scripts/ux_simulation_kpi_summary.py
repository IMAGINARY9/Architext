"""Summarize UX simulation KPIs across cycles from the simulation report."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CycleMetrics:
    name: str
    completion_rate: float
    time_to_query_min: float
    wrong_endpoint_attempts: float
    integration_correctness: float


def _extract_float(pattern: str, text: str) -> float:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Pattern not found: {pattern}")
    return float(match.group(1))


def _extract_section(report: str, heading: str, next_headings: list[str]) -> str:
    start_idx = report.find(heading)
    if start_idx == -1:
        raise ValueError(f"Heading not found: {heading}")

    end_idx = len(report)
    for next_heading in next_headings:
        idx = report.find(next_heading, start_idx + len(heading))
        if idx != -1:
            end_idx = min(end_idx, idx)

    return report[start_idx:end_idx]


def _parse_cycle(report: str, heading: str, next_headings: list[str], name: str) -> CycleMetrics:
    section = _extract_section(report, heading, next_headings)

    completion_rate = _extract_float(r"completion rate:\s*([0-9]+(?:\.[0-9]+)?)%", section)
    time_to_query_min = _extract_float(
        r"time to first successful query:\s*([0-9]+(?:\.[0-9]+)?)\s*minutes",
        section,
    )
    wrong_endpoint_attempts = _extract_float(
        r"wrong-endpoint attempts:\s*([0-9]+(?:\.[0-9]+)?)",
        section,
    )
    integration_correctness = _extract_float(
        r"integration correctness:\s*([0-9]+(?:\.[0-9]+)?)\/4",
        section,
    )

    return CycleMetrics(
        name=name,
        completion_rate=completion_rate,
        time_to_query_min=time_to_query_min,
        wrong_endpoint_attempts=wrong_endpoint_attempts,
        integration_correctness=integration_correctness,
    )


def _fmt_signed(value: float, suffix: str = "") -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}{suffix}"


def summarize(report_path: Path) -> str:
    report = report_path.read_text(encoding="utf-8")

    cycle1 = _parse_cycle(
        report,
        "## Aggregated Metrics",
        ["## Cycle 2 Scope"],
        "cycle-1",
    )
    cycle2 = _parse_cycle(
        report,
        "## Cycle 2 Aggregated Metrics",
        ["## Cycle 2 Delta vs Cycle 1", "## Cycle 3 Scope"],
        "cycle-2",
    )
    cycle3 = _parse_cycle(
        report,
        "## Cycle 3 Aggregated Metrics",
        ["## Cycle 3 Delta vs Cycle 2", "## Stability Section (Cycles 1-3)"],
        "cycle-3",
    )

    cycles = [cycle1, cycle2, cycle3]

    lines: list[str] = []
    lines.append("UX Simulation KPI Summary")
    lines.append(f"Source: {report_path}")
    lines.append("")
    lines.append("Cycle metrics:")

    for cycle in cycles:
        lines.append(
            "- "
            f"{cycle.name}: completion={cycle.completion_rate:.1f}%, "
            f"time={cycle.time_to_query_min:.1f}m, "
            f"wrong-endpoint={cycle.wrong_endpoint_attempts:.1f}, "
            f"integration={cycle.integration_correctness:.2f}/4"
        )

    def delta(a: CycleMetrics, b: CycleMetrics) -> list[str]:
        return [
            f"completion {_fmt_signed(b.completion_rate - a.completion_rate, ' pp')}",
            f"time {_fmt_signed(b.time_to_query_min - a.time_to_query_min, ' min')}",
            f"wrong-endpoint {_fmt_signed(b.wrong_endpoint_attempts - a.wrong_endpoint_attempts)}",
            f"integration {_fmt_signed(b.integration_correctness - a.integration_correctness)}",
        ]

    lines.append("")
    lines.append("Deltas:")
    lines.append("- cycle-2 vs cycle-1: " + ", ".join(delta(cycle1, cycle2)))
    lines.append("- cycle-3 vs cycle-2: " + ", ".join(delta(cycle2, cycle3)))
    lines.append("- cycle-3 vs cycle-1: " + ", ".join(delta(cycle1, cycle3)))

    completion_values = [c.completion_rate for c in cycles]
    time_values = [c.time_to_query_min for c in cycles]
    wrong_values = [c.wrong_endpoint_attempts for c in cycles]
    integration_values = [c.integration_correctness for c in cycles]

    lines.append("")
    lines.append("Stability ranges (cycles 1-3):")
    lines.append(
        f"- completion spread: {max(completion_values) - min(completion_values):.2f} pp"
    )
    lines.append(
        f"- time spread: {max(time_values) - min(time_values):.2f} min"
    )
    lines.append(
        f"- wrong-endpoint spread: {max(wrong_values) - min(wrong_values):.2f}"
    )
    lines.append(
        f"- integration spread: {max(integration_values) - min(integration_values):.2f}"
    )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize UX simulation metrics from simulation-runs markdown report."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/research/simulation-runs-2026-03-23.md"),
        help="Path to simulation report markdown file.",
    )
    args = parser.parse_args()

    summary = summarize(args.report)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
