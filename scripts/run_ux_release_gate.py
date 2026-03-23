"""Run the UX release gate and write generated outputs to ignored local paths.

This gate combines KPI threshold checks with optional structured findings input.
Decision rules:
- NO-GO when KPI thresholds fail.
- NO-GO when unresolved Critical findings exist.
- NO-GO when unresolved High findings exist and strict mode is enabled.
- CONDITIONAL GO when thresholds pass but unresolved Medium/Low remain.
- GO only when thresholds pass and no unresolved Critical/High remain.
"""

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


def _severity_counts(raw: dict[str, Any] | None) -> dict[str, int]:
    raw = raw or {}
    result: dict[str, int] = {}
    for key in ("critical", "high", "medium", "low"):
        value = raw.get(key, 0)
        try:
            result[key] = max(int(value), 0)
        except (TypeError, ValueError):
            result[key] = 0
    return result


def _load_findings(findings_file: str | None) -> dict[str, Any]:
    default_payload = {
        "source": "none",
        "found": {"critical": 0, "high": 0, "medium": 0, "low": 0},
        "fixed": {"critical": 0, "high": 0, "medium": 0, "low": 0},
        "unresolved": {"critical": 0, "high": 0, "medium": 0, "low": 0},
        "highlights": [],
        "recommendations": [],
    }
    if not findings_file:
        return default_payload

    findings_path = Path(findings_file)
    if not findings_path.exists():
        raise FileNotFoundError(f"Findings file not found: {findings_file}")

    payload = json.loads(findings_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    return {
        "source": str(findings_path).replace("\\", "/"),
        "found": _severity_counts(summary.get("found") if isinstance(summary, dict) else None),
        "fixed": _severity_counts(summary.get("fixed") if isinstance(summary, dict) else None),
        "unresolved": _severity_counts(
            summary.get("unresolved") if isinstance(summary, dict) else None
        ),
        "highlights": [str(item) for item in payload.get("highlights", [])][:10],
        "recommendations": [str(item) for item in payload.get("recommendations", [])][:10],
    }


def _decide_release(
    *,
    threshold_passed: bool,
    findings: dict[str, Any],
    fail_on_unresolved_high: bool,
) -> tuple[str, str]:
    unresolved = findings["unresolved"]
    unresolved_critical = unresolved["critical"]
    unresolved_high = unresolved["high"]
    unresolved_medium = unresolved["medium"]
    unresolved_low = unresolved["low"]

    if not threshold_passed:
        return "NO-GO", "KPI threshold checks failed."

    if unresolved_critical > 0:
        return "NO-GO", "Unresolved Critical findings remain."

    if unresolved_high > 0 and fail_on_unresolved_high:
        return "NO-GO", "Unresolved High findings remain (strict mode)."

    if unresolved_high > 0 or unresolved_medium > 0 or unresolved_low > 0:
        return (
            "CONDITIONAL GO",
            "Thresholds passed with unresolved non-blocking findings; remediation and owner tracking required.",
        )

    return "GO", "Thresholds passed and no unresolved Critical/High findings remain."


def _requires_findings_file(release_tag: str, findings_file: str) -> bool:
    """Return whether the gate must fail due to missing findings-file policy.

    Policy: any explicit non-default release tag requires a findings file.
    """
    return release_tag != "release-candidate" and not findings_file.strip()


def _append_release_gate_log(
    log_file: Path,
    release_tag: str,
    gate_command: str,
    decision: str,
    rationale: str,
    gate_result: str,
    findings: dict[str, Any],
    latest_metrics: dict[str, Any],
    actions: list[str],
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines: list[str] = []
    lines.append(f"\n## {timestamp} - {release_tag}\n")
    lines.append(f"- Release tag/version: {release_tag}")
    lines.append(f"- Gate command: `{gate_command}`")
    lines.append(f"- Gate result: {gate_result}")
    lines.append("- Findings result:")
    lines.append(
        "  - found: "
        f"C={findings['found']['critical']}, H={findings['found']['high']}, "
        f"M={findings['found']['medium']}, L={findings['found']['low']}"
    )
    lines.append(
        "  - fixed: "
        f"C={findings['fixed']['critical']}, H={findings['fixed']['high']}, "
        f"M={findings['fixed']['medium']}, L={findings['fixed']['low']}"
    )
    lines.append(
        "  - unresolved: "
        f"C={findings['unresolved']['critical']}, H={findings['unresolved']['high']}, "
        f"M={findings['unresolved']['medium']}, L={findings['unresolved']['low']}"
    )
    lines.append(f"  - source: {findings['source']}")
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

    recommendations = findings.get("recommendations", [])
    if recommendations:
        lines.append("- Recommended improvements (from findings file):")
        for recommendation in recommendations:
            lines.append(f"  - {recommendation}")
    elif actions:
        lines.append("- Candidate improvements (from latest cycle findings):")
        for action in actions:
            lines.append(f"  - {action}")
    else:
        lines.append("- Candidate improvements: none extracted from findings file or latest cycle.")

    highlights = findings.get("highlights", [])
    if highlights:
        lines.append("- Findings highlights:")
        for highlight in highlights:
            lines.append(f"  - {highlight}")

    if decision == "GO":
        lines.append("- Follow-up: continue lightweight cycle per release; escalate on threshold failure.")
    else:
        lines.append("- Follow-up: run remediation, update findings report, and re-run release gate.")

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
    parser.add_argument(
        "--findings-file",
        default="",
        help=(
            "Optional path to structured findings JSON. "
            "Used to determine unresolved findings and GO/CONDITIONAL GO/NO-GO decisions."
        ),
    )
    parser.add_argument(
        "--allow-unresolved-high",
        action="store_true",
        help="Allow unresolved High findings (will yield CONDITIONAL GO if thresholds pass).",
    )
    parser.add_argument("--min-completion-rate", type=float, default=85.0)
    parser.add_argument("--max-time-to-query", type=float, default=15.0)
    parser.add_argument("--max-wrong-endpoint", type=float, default=1.0)
    parser.add_argument("--min-integration-correctness", type=float, default=3.0)
    args = parser.parse_args()

    if _requires_findings_file(args.release_tag, args.findings_file):
        parser.error(
            "--findings-file is required when --release-tag is explicitly provided "
            "(non-default release tags)."
        )

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
    findings = _load_findings(args.findings_file)
    gate_command_text = (
        f"python scripts/run_ux_release_gate.py --release-tag {args.release_tag}"
    )
    if args.findings_file:
        gate_command_text += f" --findings-file {args.findings_file}"
    if args.allow_unresolved_high:
        gate_command_text += " --allow-unresolved-high"

    gate_result = _run(gate_cmd)
    if gate_result.stdout:
        print(gate_result.stdout, end="")

    threshold_passed = gate_result.returncode == 0
    fail_on_unresolved_high = not args.allow_unresolved_high
    decision, rationale = _decide_release(
        threshold_passed=threshold_passed,
        findings=findings,
        fail_on_unresolved_high=fail_on_unresolved_high,
    )
    gate_status = "PASS" if threshold_passed else "FAIL"

    if gate_result.returncode != 0 and gate_result.stderr:
        sys.stderr.write(gate_result.stderr)

    if not args.skip_log:
        _append_release_gate_log(
            log_file=Path(args.log_file),
            release_tag=args.release_tag,
            gate_command=gate_command_text,
            decision=decision,
            rationale=rationale,
            gate_result=gate_status,
            findings=findings,
            latest_metrics=latest_metrics,
            actions=latest_actions,
        )

    if decision == "NO-GO":
        print("UX release gate decision: NO-GO")
        print(f"Rationale: {rationale}")
        print(f"Generated outputs: {text_out}, {json_out}, {csv_out}")
        if args.skip_log:
            print("Release gate log update skipped (--skip-log).")
        else:
            print(f"Release gate log updated: {args.log_file}")
        return 1

    print(f"UX release gate decision: {decision}")
    print(f"Rationale: {rationale}")
    print(f"Generated outputs: {text_out}, {json_out}, {csv_out}")
    if args.skip_log:
        print("Release gate log update skipped (--skip-log).")
    else:
        print(f"Release gate log updated: {args.log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
