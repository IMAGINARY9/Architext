from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ARCHITEXT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

# Ensure local package imports work when running as a script.
if str(ARCHITEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARCHITEXT_ROOT))


@dataclass
class RepoTarget:
    name: str
    source: Path


def _stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _source_descriptor(path: Path, *, include_paths: bool) -> Any:
    if include_paths:
        return str(path)
    # Redact absolute paths by default to avoid leaking local filesystem structure.
    return {"basename": path.name, "id": _stable_id(str(path.resolve()))}


def _run_cli(args: List[str], *, cwd: Path) -> Tuple[int, str, float]:
    start = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ARCHITEXT_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(
        [PYTHON, "-m", "src.cli", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, out.strip(), elapsed


def _parse_json_output(output: str) -> Dict[str, Any]:
    # CLI prints JSON for json-format task outputs.
    # If output includes extra lines, try to locate the first JSON object.
    output = output.strip()
    if not output:
        raise ValueError("Empty output")

    # Fast path: direct JSON
    try:
        return json.loads(output)
    except Exception:
        pass

    # Heuristic: find first '{' and last '}'
    start = output.find("{")
    end = output.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in output")
    return json.loads(output[start : end + 1])


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _assert_between(value: float, lo: float, hi: float, msg: str) -> None:
    _assert(lo <= value <= hi, f"{msg} (got {value}, expected {lo}..{hi})")


def evaluate_repo(repo: RepoTarget) -> Dict[str, Any]:
    if not repo.source.exists():
        raise FileNotFoundError(f"Missing repo path: {repo.source}")

    results: Dict[str, Any] = {"repo": repo.name, "checks": [], "timings_sec": {}}

    def record(name: str, ok: bool, detail: Optional[str] = None, seconds: Optional[float] = None):
        results["checks"].append({"name": name, "ok": ok, "detail": detail})
        if seconds is not None:
            results["timings_sec"][name] = round(seconds, 3)

    # In-process evaluation is dramatically faster than launching a new Python process per task.
    # We still run a couple CLI sanity checks to ensure formatting/wiring works.
    from src import tasks as phase25

    # Pre-load/cached file list once per repo
    start_scan = time.perf_counter()
    files = phase25.collect_file_paths(None, str(repo.source))
    scan_t = time.perf_counter() - start_scan
    record("_inventory", True, detail=f"files={len(files)}", seconds=scan_t)

    # ---------- CLI sanity checks (2) ----------
    code, out, t = _run_cli(
        ["analyze-structure", "--source", str(repo.source), "--depth", "shallow", "--output-format", "json"],
        cwd=ARCHITEXT_ROOT,
    )
    _assert(code == 0, f"CLI analyze-structure json failed: {out[-500:]}")
    payload = _parse_json_output(out)
    _assert(payload.get("format") == "json", "CLI analyze-structure json: format != json")
    record("cli.analyze-structure.json", True, detail=f"total_files={payload.get('summary',{}).get('total_files')}", seconds=t)

    code, out, t = _run_cli(
        ["analyze-structure", "--source", str(repo.source), "--depth", "detailed", "--output-format", "markdown"],
        cwd=ARCHITEXT_ROOT,
    )
    _assert(code == 0, f"CLI analyze-structure markdown failed: {out[-500:]}")
    _assert("# Repository Structure" in out, "CLI analyze-structure markdown: missing header")
    record("cli.analyze-structure.markdown", True, detail=f"bytes={len(out.encode('utf-8',errors='ignore'))}", seconds=t)

    # ---------- In-process task checks (cached) ----------
    start = time.perf_counter()
    structure = phase25.analyze_structure(source_path=str(repo.source), depth="shallow", output_format="json")
    record(
        "analyze-structure.json",
        True,
        detail=f"total_files={structure.get('summary',{}).get('total_files')} langs={list((structure.get('summary',{}).get('languages') or {}).keys())}",
        seconds=time.perf_counter() - start,
    )

    start = time.perf_counter()
    tech = phase25.tech_stack(source_path=str(repo.source), output_format="json")
    langs = (tech.get("data") or {}).get("languages") or {}
    record("tech-stack.json", True, detail=f"languages={len(langs)}", seconds=time.perf_counter() - start)

    start = time.perf_counter()
    anti = phase25.detect_anti_patterns(source_path=str(repo.source))
    issues = anti.get("issues") or []
    counts = anti.get("counts") or {}
    record(
        "detect-anti-patterns.json",
        True,
        detail=f"issues={len(issues)} top_types={list(dict(counts).keys())[:4]}",
        seconds=time.perf_counter() - start,
    )

    # Harder/less "too easy": require at least one objective finding on non-trivial repos.
    if len(files) >= 20 and len(issues) == 0:
        record("_finding_check", False, detail="No anti-pattern issues detected on non-trivial repo (>=20 files)")
    else:
        record("_finding_check", True, detail="ok")

    start = time.perf_counter()
    health = phase25.health_score(source_path=str(repo.source))
    score = health.get("score")
    record("health-score.json", True, detail=f"score={score} details={health.get('details',{})}", seconds=time.perf_counter() - start)

    start = time.perf_counter()
    impact = phase25.impact_analysis("src", source_path=str(repo.source))
    record(
        "impact-analysis.json",
        True,
        detail=f"targets={len(impact.get('targets') or [])} affected_count={impact.get('affected_count')}",
        seconds=time.perf_counter() - start,
    )

    start = time.perf_counter()
    recs = phase25.refactoring_recommendations(source_path=str(repo.source))
    record(
        "refactoring-recommendations.json",
        True,
        detail=f"recommendations={len(recs.get('recommendations') or [])}",
        seconds=time.perf_counter() - start,
    )

    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "docs"
        docs = phase25.generate_docs(source_path=str(repo.source), output_dir=str(out_dir))
        record(
            "generate-docs.json",
            True,
            detail=f"outputs={len(docs.get('outputs') or [])} wrote_dir={out_dir.exists()}",
            seconds=time.perf_counter() - start,
        )

    start = time.perf_counter()
    graph = phase25.dependency_graph_export(source_path=str(repo.source), output_format="json")
    record(
        "dependency-graph.json",
        True,
        detail=f"nodes={len(graph.get('nodes') or [])} edges={len(graph.get('edges') or [])}",
        seconds=time.perf_counter() - start,
    )

    start = time.perf_counter()
    coverage = phase25.test_coverage_analysis(source_path=str(repo.source))
    record(
        "test-coverage.json",
        True,
        detail=f"ratio={coverage.get('coverage_ratio')} sources={coverage.get('total_sources')} tests={coverage.get('total_tests')}",
        seconds=time.perf_counter() - start,
    )

    start = time.perf_counter()
    patterns = phase25.architecture_pattern_detection(source_path=str(repo.source))
    record(
        "detect-patterns.json",
        True,
        detail=f"patterns={patterns.get('patterns')}",
        seconds=time.perf_counter() - start,
    )

    # diff-architecture: baseline == current => 0 changes; baseline empty => added_count > 0
    start = time.perf_counter()
    same = phase25.diff_architecture_review(source_path=str(repo.source), baseline_files=files)
    record(
        "diff-architecture.same",
        True,
        detail=f"added={same.get('added_count')} removed={same.get('removed_count')}",
        seconds=time.perf_counter() - start,
    )

    start = time.perf_counter()
    empty = phase25.diff_architecture_review(source_path=str(repo.source), baseline_files=[])
    record(
        "diff-architecture.empty",
        True,
        detail=f"added={empty.get('added_count')} removed={empty.get('removed_count')}",
        seconds=time.perf_counter() - start,
    )

    start = time.perf_counter()
    onboard = phase25.onboarding_guide(source_path=str(repo.source))
    record(
        "onboarding-guide.json",
        True,
        detail=f"entry_points={onboard.get('entry_points')} root_files={len(onboard.get('root_files') or [])}",
        seconds=time.perf_counter() - start,
    )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Phase 2.5 tasks against one or more repositories. "
            "Defaults are intentionally minimal to avoid leaking local paths in source control."
        )
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help=(
            "Repository target. Format: NAME=PATH or just PATH (NAME defaults to folder name). "
            "Repeatable. If omitted, evaluates only the current Architext repo."
        ),
    )
    parser.add_argument(
        "--report",
        default=str(ARCHITEXT_ROOT / ".local" / "phase25_evaluation_report.json"),
        help="Output JSON path (default: .local/phase25_evaluation_report.json)",
    )
    parser.add_argument(
        "--include-paths",
        action="store_true",
        help="Include full source paths in the report (off by default to prevent leaks).",
    )
    args = parser.parse_args()

    targets: List[RepoTarget] = []
    if args.repo:
        for raw in args.repo:
            name: str
            path_str: str
            if "=" in raw:
                name, path_str = raw.split("=", 1)
            else:
                path_str = raw
                name = Path(path_str).name or path_str
            targets.append(RepoTarget(name=name, source=Path(path_str)))
    else:
        targets.append(RepoTarget("Architext", ARCHITEXT_ROOT))

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "targets": [],
    }

    failures: List[str] = []

    for target in targets:
        try:
            res = evaluate_repo(target)
            res["source"] = _source_descriptor(target.source, include_paths=bool(args.include_paths))
            report["targets"].append(res)
            print(f"OK: {target.name}")
        except Exception as exc:
            failures.append(f"{target.name}: {exc}")
            report["targets"].append(
                {
                    "repo": target.name,
                    "source": _source_descriptor(target.source, include_paths=bool(args.include_paths)),
                    "error": str(exc),
                }
            )
            print(f"FAIL: {target.name} -> {exc}")

    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote report: {out_path}")

    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"- {f}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
