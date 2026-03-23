"""Summarize UX simulation KPIs across cycles from the simulation report."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CycleMetrics:
    name: str
    cycle_number: int
    completion_rate: float
    time_to_query_min: float
    wrong_endpoint_attempts: float
    integration_correctness: float


def _extract_float(pattern: str, text: str) -> float:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Pattern not found: {pattern}")
    return float(match.group(1))


def _parse_cycle(section: str, cycle_number: int) -> CycleMetrics:

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
        name=f"cycle-{cycle_number}",
        cycle_number=cycle_number,
        completion_rate=completion_rate,
        time_to_query_min=time_to_query_min,
        wrong_endpoint_attempts=wrong_endpoint_attempts,
        integration_correctness=integration_correctness,
    )


def _extract_cycle_sections(report: str) -> list[tuple[int, str]]:
    heading_regex = re.compile(r"^##\s+(Cycle\s+(\d+)\s+)?Aggregated Metrics\s*$", re.MULTILINE)
    matches = list(heading_regex.finditer(report))
    if not matches:
        raise ValueError("No aggregated metrics sections found in report.")

    cycle_sections: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        section_start = match.start()
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(report)
        section = report[section_start:section_end]

        cycle_number = int(match.group(2)) if match.group(2) else 1
        cycle_sections.append((cycle_number, section))

    cycle_sections.sort(key=lambda item: item[0])
    return cycle_sections


def _fmt_signed(value: float, suffix: str = "") -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}{suffix}"


def summarize(report_path: Path) -> str:
    report = report_path.read_text(encoding="utf-8")
    cycle_sections = _extract_cycle_sections(report)
    cycles = [_parse_cycle(section, cycle_number) for cycle_number, section in cycle_sections]
    if len(cycles) < 2:
        raise ValueError("At least two cycles are required for delta analysis.")

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
    for prev_cycle, current_cycle in zip(cycles, cycles[1:]):
        lines.append(
            f"- {current_cycle.name} vs {prev_cycle.name}: "
            + ", ".join(delta(prev_cycle, current_cycle))
        )

    baseline_cycle = cycles[0]
    latest_cycle = cycles[-1]
    lines.append(
        f"- {latest_cycle.name} vs {baseline_cycle.name}: "
        + ", ".join(delta(baseline_cycle, latest_cycle))
    )

    completion_values = [c.completion_rate for c in cycles]
    time_values = [c.time_to_query_min for c in cycles]
    wrong_values = [c.wrong_endpoint_attempts for c in cycles]
    integration_values = [c.integration_correctness for c in cycles]

    lines.append("")
    lines.append(f"Stability ranges (cycles 1-{len(cycles)}):")
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
