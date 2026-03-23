"""Summarize UX simulation KPIs across cycles from the simulation report."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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


def _compute_summary_data(report: str) -> dict[str, Any]:
    cycle_sections = _extract_cycle_sections(report)
    cycles = [_parse_cycle(section, cycle_number) for cycle_number, section in cycle_sections]
    if len(cycles) < 2:
        raise ValueError("At least two cycles are required for delta analysis.")

    deltas: list[dict[str, Any]] = []

    def add_delta(a: CycleMetrics, b: CycleMetrics) -> None:
        deltas.append(
            {
                "from": a.name,
                "to": b.name,
                "completion_pp": round(b.completion_rate - a.completion_rate, 2),
                "time_min": round(b.time_to_query_min - a.time_to_query_min, 2),
                "wrong_endpoint": round(
                    b.wrong_endpoint_attempts - a.wrong_endpoint_attempts,
                    2,
                ),
                "integration": round(
                    b.integration_correctness - a.integration_correctness,
                    2,
                ),
            }
        )

    for prev_cycle, current_cycle in zip(cycles, cycles[1:]):
        add_delta(prev_cycle, current_cycle)

    baseline_cycle = cycles[0]
    latest_cycle = cycles[-1]
    add_delta(baseline_cycle, latest_cycle)

    completion_values = [c.completion_rate for c in cycles]
    time_values = [c.time_to_query_min for c in cycles]
    wrong_values = [c.wrong_endpoint_attempts for c in cycles]
    integration_values = [c.integration_correctness for c in cycles]

    stability = {
        "cycle_count": len(cycles),
        "completion_spread_pp": round(max(completion_values) - min(completion_values), 2),
        "time_spread_min": round(max(time_values) - min(time_values), 2),
        "wrong_endpoint_spread": round(max(wrong_values) - min(wrong_values), 2),
        "integration_spread": round(max(integration_values) - min(integration_values), 2),
    }

    return {
        "cycles": [asdict(c) for c in cycles],
        "deltas": deltas,
        "stability": stability,
    }


def _render_text(report_path: Path, summary_data: dict[str, Any]) -> str:
    cycles = [CycleMetrics(**entry) for entry in summary_data["cycles"]]
    deltas = summary_data["deltas"]
    stability = summary_data["stability"]

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

    lines.append("")
    lines.append("Deltas:")
    for delta in deltas[:-1]:
        lines.append(
            f"- {delta['to']} vs {delta['from']}: "
            f"completion {_fmt_signed(delta['completion_pp'], ' pp')}, "
            f"time {_fmt_signed(delta['time_min'], ' min')}, "
            f"wrong-endpoint {_fmt_signed(delta['wrong_endpoint'])}, "
            f"integration {_fmt_signed(delta['integration'])}"
        )

    baseline_delta = deltas[-1]
    lines.append(
        f"- {baseline_delta['to']} vs {baseline_delta['from']}: "
        f"completion {_fmt_signed(baseline_delta['completion_pp'], ' pp')}, "
        f"time {_fmt_signed(baseline_delta['time_min'], ' min')}, "
        f"wrong-endpoint {_fmt_signed(baseline_delta['wrong_endpoint'])}, "
        f"integration {_fmt_signed(baseline_delta['integration'])}"
    )

    lines.append("")
    lines.append(f"Stability ranges (cycles 1-{stability['cycle_count']}):")
    lines.append(f"- completion spread: {stability['completion_spread_pp']:.2f} pp")
    lines.append(f"- time spread: {stability['time_spread_min']:.2f} min")
    lines.append(f"- wrong-endpoint spread: {stability['wrong_endpoint_spread']:.2f}")
    lines.append(f"- integration spread: {stability['integration_spread']:.2f}")

    return "\n".join(lines) + "\n"


def _render_csv(summary_data: dict[str, Any]) -> str:
    rows: list[list[str]] = [[
        "cycle",
        "completion_rate",
        "time_to_first_query_min",
        "wrong_endpoint_attempts",
        "integration_correctness",
    ]]
    for cycle in summary_data["cycles"]:
        rows.append(
            [
                cycle["name"],
                str(cycle["completion_rate"]),
                str(cycle["time_to_query_min"]),
                str(cycle["wrong_endpoint_attempts"]),
                str(cycle["integration_correctness"]),
            ]
        )

    output_lines: list[str] = []
    for row in rows:
        output_lines.append(",".join(row))
    return "\n".join(output_lines) + "\n"


def _evaluate_thresholds(
    summary_data: dict[str, Any],
    min_completion_rate: float,
    max_time_to_query: float,
    max_wrong_endpoint: float,
    min_integration_correctness: float,
) -> tuple[bool, list[str]]:
    latest = summary_data["cycles"][-1]
    failures: list[str] = []

    if latest["completion_rate"] < min_completion_rate:
        failures.append(
            f"completion_rate {latest['completion_rate']:.2f}% < {min_completion_rate:.2f}%"
        )
    if latest["time_to_query_min"] > max_time_to_query:
        failures.append(
            f"time_to_query_min {latest['time_to_query_min']:.2f} > {max_time_to_query:.2f}"
        )
    if latest["wrong_endpoint_attempts"] > max_wrong_endpoint:
        failures.append(
            "wrong_endpoint_attempts "
            f"{latest['wrong_endpoint_attempts']:.2f} > {max_wrong_endpoint:.2f}"
        )
    if latest["integration_correctness"] < min_integration_correctness:
        failures.append(
            "integration_correctness "
            f"{latest['integration_correctness']:.2f} < {min_integration_correctness:.2f}"
        )

    return (len(failures) == 0, failures)


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
    parser.add_argument(
        "--format",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file path to write output.",
    )
    parser.add_argument(
        "--check-thresholds",
        action="store_true",
        help="Fail with exit code 2 when latest-cycle thresholds are not met.",
    )
    parser.add_argument("--min-completion-rate", type=float, default=85.0)
    parser.add_argument("--max-time-to-query", type=float, default=15.0)
    parser.add_argument("--max-wrong-endpoint", type=float, default=1.0)
    parser.add_argument("--min-integration-correctness", type=float, default=3.0)
    args = parser.parse_args()

    report = args.report.read_text(encoding="utf-8")
    summary_data = _compute_summary_data(report)

    if args.format == "json":
        rendered = json.dumps(summary_data, indent=2) + "\n"
    elif args.format == "csv":
        rendered = _render_csv(summary_data)
    else:
        rendered = _render_text(args.report, summary_data)

    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")

    if args.check_thresholds:
        passed, failures = _evaluate_thresholds(
            summary_data,
            min_completion_rate=args.min_completion_rate,
            max_time_to_query=args.max_time_to_query,
            max_wrong_endpoint=args.max_wrong_endpoint,
            min_integration_correctness=args.min_integration_correctness,
        )
        if not passed:
            failure_text = "Threshold check FAILED:\n- " + "\n- ".join(failures)
            print(failure_text)
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
