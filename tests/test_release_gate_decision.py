"""Tests for UX release gate decision and findings parsing logic."""

from __future__ import annotations

import json

from scripts.run_ux_release_gate import (
    _decide_release,
    _load_findings,
    _requires_findings_file,
    _severity_counts,
)


def test_severity_counts_normalizes_values() -> None:
    counts = _severity_counts({"critical": "2", "high": -3, "medium": "x", "low": 1})
    assert counts == {"critical": 2, "high": 0, "medium": 0, "low": 1}


def test_load_findings_defaults_without_file() -> None:
    payload = _load_findings("")
    assert payload["source"] == "none"
    assert payload["unresolved"] == {"critical": 0, "high": 0, "medium": 0, "low": 0}


def test_load_findings_reads_summary(tmp_path) -> None:
    findings_path = tmp_path / "findings.json"
    findings_path.write_text(
        json.dumps(
            {
                "summary": {
                    "found": {"critical": 1, "high": 2, "medium": 3, "low": 4},
                    "fixed": {"critical": 1, "high": 1, "medium": 1, "low": 0},
                    "unresolved": {"critical": 0, "high": 1, "medium": 2, "low": 0},
                },
                "highlights": ["H1"],
                "recommendations": ["R1", "R2"],
            }
        ),
        encoding="utf-8",
    )

    payload = _load_findings(str(findings_path))
    assert payload["unresolved"]["high"] == 1
    assert payload["highlights"] == ["H1"]
    assert payload["recommendations"] == ["R1", "R2"]


def test_decide_release_no_go_on_threshold_failure() -> None:
    decision, rationale = _decide_release(
        threshold_passed=False,
        findings={"unresolved": {"critical": 0, "high": 0, "medium": 0, "low": 0}},
        fail_on_unresolved_high=True,
    )
    assert decision == "NO-GO"
    assert "threshold" in rationale.lower()


def test_decide_release_no_go_on_unresolved_critical() -> None:
    decision, _ = _decide_release(
        threshold_passed=True,
        findings={"unresolved": {"critical": 1, "high": 0, "medium": 0, "low": 0}},
        fail_on_unresolved_high=True,
    )
    assert decision == "NO-GO"


def test_decide_release_conditional_go_for_medium_low() -> None:
    decision, _ = _decide_release(
        threshold_passed=True,
        findings={"unresolved": {"critical": 0, "high": 0, "medium": 1, "low": 2}},
        fail_on_unresolved_high=True,
    )
    assert decision == "CONDITIONAL GO"


def test_decide_release_go_without_unresolved_critical_high() -> None:
    decision, rationale = _decide_release(
        threshold_passed=True,
        findings={"unresolved": {"critical": 0, "high": 0, "medium": 0, "low": 0}},
        fail_on_unresolved_high=True,
    )
    assert decision == "GO"
    assert "no unresolved critical/high" in rationale.lower()


def test_requires_findings_file_for_non_default_release_tag() -> None:
    assert _requires_findings_file("v1.2.3-rc1", "") is True
    assert _requires_findings_file("v1.2.3-rc1", "docs/research/f.json") is False
    assert _requires_findings_file("release-candidate", "") is False
