"""Validate release gate log policy for findings-aware entries."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def _split_entries(text: str) -> list[str]:
    parts = re.split(r"\n(?=##\s+\d{4}-\d{2}-\d{2}\s+-\s+)", text.strip())
    return [part for part in parts if part.strip().startswith("## ")]


def validate_release_gate_log(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    entries = _split_entries(text)

    for entry in entries:
        header = entry.splitlines()[0].strip()
        gate_command = ""
        for line in entry.splitlines():
            if line.startswith("- Gate command:"):
                gate_command = line
                break

        if "--release-tag" in gate_command and "--findings-file" not in gate_command:
            errors.append(f"{header}: release-tag entry missing --findings-file in gate command")

        if "- Gate result:" not in entry:
            errors.append(f"{header}: missing '- Gate result:'")
        if "- Findings result:" not in entry:
            errors.append(f"{header}: missing '- Findings result:'")
        if "- unresolved:" not in entry:
            errors.append(f"{header}: missing unresolved severity summary")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate findings-aware release gate log entries.")
    parser.add_argument(
        "--log-file",
        default="docs/research/release-gate-log.md",
        help="Path to release gate markdown log.",
    )
    args = parser.parse_args()

    log_file = Path(args.log_file)
    if not log_file.exists():
        print(f"ERROR: release gate log not found: {log_file}")
        return 1

    errors = validate_release_gate_log(log_file)
    if errors:
        print("Release gate log validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Release gate log validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
