"""Build a compact unresolved-findings dashboard from release gate log entries."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ENTRY_SPLIT_PATTERN = re.compile(r"\n(?=##\s+\d{4}-\d{2}-\d{2}\s+-\s+)")
HEADER_PATTERN = re.compile(r"^##\s+(?P<date>\d{4}-\d{2}-\d{2})\s+-\s+(?P<tag>.+)$")
UNRESOLVED_PATTERN = re.compile(
    r"^\s*-\s*unresolved:\s*C=(?P<c>\d+),\s*H=(?P<h>\d+),\s*M=(?P<m>\d+),\s*L=(?P<l>\d+)\s*$"
)


def _split_entries(text: str) -> list[str]:
    parts = ENTRY_SPLIT_PATTERN.split(text.strip())
    return [part for part in parts if part.strip().startswith("## ")]


def _extract_field(lines: list[str], prefix: str) -> str:
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""


def parse_release_gate_log(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []

    for entry in _split_entries(text):
        lines = [line.rstrip() for line in entry.splitlines() if line.strip()]
        header_match = HEADER_PATTERN.match(lines[0])
        if not header_match:
            continue

        unresolved = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for line in lines:
            unresolved_match = UNRESOLVED_PATTERN.match(line)
            if unresolved_match:
                unresolved = {
                    "critical": int(unresolved_match.group("c")),
                    "high": int(unresolved_match.group("h")),
                    "medium": int(unresolved_match.group("m")),
                    "low": int(unresolved_match.group("l")),
                }
                break

        rows.append(
            {
                "date": header_match.group("date"),
                "release_tag": header_match.group("tag"),
                "gate_result": _extract_field(lines, "- Gate result:"),
                "decision": _extract_field(lines, "- Decision:"),
                "unresolved_critical": unresolved["critical"],
                "unresolved_high": unresolved["high"],
                "unresolved_medium": unresolved["medium"],
                "unresolved_low": unresolved["low"],
            }
        )

    return rows


def _build_text(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("Release Gate Findings Dashboard")
    lines.append("")
    lines.append("| Date | Tag | Gate | Decision | Unresolved C/H/M/L |")
    lines.append("|---|---|---|---|---|")
    for row in rows:
        lines.append(
            "| "
            f"{row['date']} | {row['release_tag']} | {row['gate_result'] or '-'} | "
            f"{row['decision'] or '-'} | "
            f"{row['unresolved_critical']}/{row['unresolved_high']}/"
            f"{row['unresolved_medium']}/{row['unresolved_low']} |"
        )

    if rows:
        latest = rows[-1]
        lines.append("")
        lines.append("Latest entry summary:")
        lines.append(
            "- "
            f"{latest['release_tag']}: unresolved C/H/M/L = "
            f"{latest['unresolved_critical']}/{latest['unresolved_high']}/"
            f"{latest['unresolved_medium']}/{latest['unresolved_low']}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build unresolved findings dashboard from release gate log.")
    parser.add_argument("--log-file", default="docs/research/release-gate-log.md")
    parser.add_argument("--format", choices=["text", "json", "csv"], default="text")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    log_file = Path(args.log_file)
    rows = parse_release_gate_log(log_file)

    if args.format == "json":
        content = json.dumps({"entries": rows}, indent=2)
    elif args.format == "csv":
        headers = [
            "date",
            "release_tag",
            "gate_result",
            "decision",
            "unresolved_critical",
            "unresolved_high",
            "unresolved_medium",
            "unresolved_low",
        ]
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)
            print(str(out_path))
            return 0
        output_lines = [",".join(headers)]
        for row in rows:
            output_lines.append(
                ",".join(
                    str(row[key])
                    for key in headers
                )
            )
        content = "\n".join(output_lines) + "\n"
    else:
        content = _build_text(rows)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        print(str(out_path))
        return 0

    print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
