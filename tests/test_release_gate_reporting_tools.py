"""Tests for release-gate reporting helper scripts."""

from __future__ import annotations

from pathlib import Path

from scripts.release_gate_findings_dashboard import parse_release_gate_log
from scripts.validate_release_gate_log import validate_release_gate_log


def test_validate_release_gate_log_accepts_findings_aware_entry(tmp_path: Path) -> None:
    log_file = tmp_path / "release-gate-log.md"
    log_file.write_text(
        """
# Release UX Gate Log

## 2026-03-24 - rc1

- Release tag/version: rc1
- Gate command: `python scripts/run_ux_release_gate.py --release-tag rc1 --findings-file docs/research/f.json`
- Gate result: PASS
- Findings result:
  - found: C=1, H=2, M=3, L=4
  - fixed: C=1, H=2, M=3, L=4
  - unresolved: C=0, H=0, M=0, L=0
  - source: docs/research/f.json
- Decision: GO
- Rationale: Thresholds passed and no unresolved Critical/High findings remain.
""".strip()
        + "\n",
        encoding="utf-8",
    )

    errors = validate_release_gate_log(log_file)
    assert errors == []


def test_validate_release_gate_log_flags_missing_findings_file(tmp_path: Path) -> None:
    log_file = tmp_path / "release-gate-log.md"
    log_file.write_text(
        """
## 2026-03-24 - rc1

- Release tag/version: rc1
- Gate command: `python scripts/run_ux_release_gate.py --release-tag rc1`
- Decision: GO
""".strip()
        + "\n",
        encoding="utf-8",
    )

    errors = validate_release_gate_log(log_file)
    assert any("missing --findings-file" in message for message in errors)


def test_parse_release_gate_log_extracts_unresolved_counts(tmp_path: Path) -> None:
    log_file = tmp_path / "release-gate-log.md"
    log_file.write_text(
        """
## 2026-03-24 - rc1

- Release tag/version: rc1
- Gate command: `python scripts/run_ux_release_gate.py --release-tag rc1 --findings-file docs/research/f.json`
- Gate result: PASS
- Findings result:
  - found: C=1, H=0, M=0, L=0
  - fixed: C=1, H=0, M=0, L=0
  - unresolved: C=0, H=0, M=2, L=1
  - source: docs/research/f.json
- Decision: CONDITIONAL GO
""".strip()
        + "\n",
        encoding="utf-8",
    )

    rows = parse_release_gate_log(log_file)
    assert len(rows) == 1
    assert rows[0]["release_tag"] == "rc1"
    assert rows[0]["unresolved_medium"] == 2
    assert rows[0]["unresolved_low"] == 1
