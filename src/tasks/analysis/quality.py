"""Code quality analysis tasks.

Includes silent failures detection and test mapping.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.tasks.core.base import (
    BaseTask,
    FileInfo,
    PYTHON_EXTENSIONS,
    JS_TS_EXTENSIONS,
    CODE_EXTENSIONS,
    calculate_ratio,
)
from src.tasks.shared import _line_number_from_index


# =============================================================================
# Silent Failures Detection Task
# =============================================================================

class SilentFailuresTask(BaseTask):
    """
    Detect exception handlers that silently swallow errors.
    
    Finds patterns like:
    - except: pass
    - except Exception: continue
    - catch(e) {} (empty JS/TS catch blocks)
    """
    
    # Patterns that indicate silent exception handling
    SILENT_HANDLERS = {"pass", "continue", "return", "..."}
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        source_path: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ):
        super().__init__(
            storage_path=storage_path,
            source_path=source_path,
            progress_callback=progress_callback,
            extensions=PYTHON_EXTENSIONS | JS_TS_EXTENSIONS,
            load_content=True,
        )
    
    def analyze(self, files: List[FileInfo]) -> Dict[str, Any]:
        """Find silent exception handlers."""
        findings: List[Dict[str, Any]] = []
        
        self._report_progress("analyze", f"Scanning {len(files)} files for silent failures")
        
        for file in files:
            if not file.content:
                continue
            
            if file.extension == ".py":
                findings.extend(self._analyze_python(file))
            elif file.extension in JS_TS_EXTENSIONS:
                findings.extend(self._analyze_js_ts(file))
        
        return {
            "findings": findings,
            "count": len(findings),
            "summary": f"Found {len(findings)} silent exception handlers",
        }
    
    def _analyze_python(self, file: FileInfo) -> List[Dict[str, Any]]:
        """Analyze Python file for silent exception handlers."""
        findings = []
        lines = file.content.splitlines()
        
        for idx, line in enumerate(lines):
            # Look for except clauses
            if not re.match(r"\s*except\b", line):
                continue
            if not line.rstrip().endswith(":"):
                continue
            
            base_indent = len(line) - len(line.lstrip())
            cursor = idx + 1
            
            while cursor < len(lines):
                candidate = lines[cursor]
                
                # Skip empty lines and comments
                if not candidate.strip() or candidate.lstrip().startswith("#"):
                    cursor += 1
                    continue
                
                indent = len(candidate) - len(candidate.lstrip())
                if indent <= base_indent:
                    break
                
                stripped = candidate.strip()
                if stripped in self.SILENT_HANDLERS:
                    findings.append({
                        "file": file.path,
                        "line": cursor + 1,
                        "severity": "medium",
                        "type": "silent_exception",
                        "snippet": stripped,
                    })
                break
        
        return findings
    
    def _analyze_js_ts(self, file: FileInfo) -> List[Dict[str, Any]]:
        """Analyze JS/TS file for empty catch blocks."""
        findings = []
        
        # Match empty catch blocks (allowing comments inside)
        pattern = r"catch\s*\([^\)]*\)\s*\{\s*(?:\/\*.*?\*\/\s*|\/\/.*?\n\s*)*\}"
        
        for match in re.finditer(pattern, file.content, re.IGNORECASE | re.DOTALL):
            line = _line_number_from_index(file.content, match.start())
            findings.append({
                "file": file.path,
                "line": str(line),
                "severity": "medium",
                "type": "silent_exception",
                "snippet": "empty catch block",
            })
        
        return findings


def identify_silent_failures_v2(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Find silent exception handlers using BaseTask pattern."""
    task = SilentFailuresTask(
        storage_path=storage_path,
        source_path=source_path,
        progress_callback=progress_callback,
    )
    return task.run()


# Alias for backward compatibility
silent_failures_detection_v2 = identify_silent_failures_v2


# =============================================================================
# Test Mapping Task
# =============================================================================

# Common test file naming patterns
TEST_PATTERNS = [
    r"^test_(.+)$",      # test_module.py
    r"^(.+)_test$",      # module_test.py
    r"^tests?$",         # test.py or tests.py
    r"^(.+)\.test$",     # module.test.js
    r"^(.+)\.spec$",     # module.spec.ts
]

# Files to skip when checking for test coverage
SKIP_STEMS = {"__init__", "conftest", "__main__", "setup", "config", "settings"}


class TestMappingTask(BaseTask):
    """
    Map test files to source files based on naming conventions.
    
    This is NOT actual code coverage - it identifies which source files
    have corresponding test files based on naming patterns.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        source_path: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ):
        super().__init__(
            storage_path=storage_path,
            source_path=source_path,
            progress_callback=progress_callback,
            extensions=CODE_EXTENSIONS,
            load_content=False,  # Don't need content for mapping
        )
    
    def analyze(self, files: List[FileInfo]) -> Dict[str, Any]:
        """Map test files to source files."""
        source_files: List[str] = []
        test_files: List[str] = []
        
        # Separate source and test files
        for file in files:
            if self._is_test_file(file.path):
                test_files.append(file.path)
            else:
                source_files.append(file.path)
        
        self._report_progress("analyze", f"Mapping {len(test_files)} tests to {len(source_files)} sources")
        
        # Build mapping
        mapping: Dict[str, List[str]] = defaultdict(list)
        source_stems = {Path(src).stem.lower(): src for src in source_files}
        
        for test_path in test_files:
            subject = self._extract_test_subject(test_path)
            
            # Try direct match first
            if subject and subject in source_stems:
                mapping[source_stems[subject]].append(test_path)
                continue
            
            # Fallback: check if test filename contains source stem
            test_stem = Path(test_path).stem.lower()
            for src_stem, src_path in source_stems.items():
                if len(src_stem) < 4:  # Skip short generic names
                    continue
                if src_stem in test_stem and src_path not in mapping:
                    mapping[src_path].append(test_path)
        
        # Find untested files
        testable_sources = [
            src for src in source_files
            if Path(src).stem.lower() not in SKIP_STEMS
        ]
        
        untested = [src for src in testable_sources if src not in mapping]
        tested_ratio = 1 - calculate_ratio(len(untested), len(testable_sources))
        
        return {
            "total_sources": len(source_files),
            "testable_sources": len(testable_sources),
            "total_tests": len(test_files),
            "tested_ratio": round(tested_ratio, 2),
            "untested": untested[:50],
            "mapping": {k: v[:5] for k, v in list(mapping.items())[:30]},
            "note": "This analyzes test file presence, not actual code coverage.",
        }
    
    @staticmethod
    def _is_test_file(path: str) -> bool:
        """Check if a file is a test file."""
        path_lower = path.lower().replace("\\", "/")
        name = Path(path).name.lower()
        
        # Check directory patterns
        if "/tests/" in path_lower or "/test/" in path_lower or "/__tests__/" in path_lower:
            return True
        
        # Check filename patterns
        if name.startswith("test_") or name.endswith("_test.py"):
            return True
        if ".test." in name or ".spec." in name:
            return True
        if name in {"conftest.py", "fixtures.py"}:
            return True
        
        return False
    
    @staticmethod
    def _extract_test_subject(test_filename: str) -> Optional[str]:
        """Extract the subject module name from a test filename."""
        stem = Path(test_filename).stem
        
        for pattern in TEST_PATTERNS:
            match = re.match(pattern, stem, re.IGNORECASE)
            if match and match.groups():
                return match.group(1).lower()
        return None


def test_mapping_analysis_v2(
    storage_path: Optional[str] = None,
    source_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Analyze test file mapping using BaseTask pattern."""
    task = TestMappingTask(
        storage_path=storage_path,
        source_path=source_path,
        progress_callback=progress_callback,
    )
    return task.run()


# Alias for backward compatibility
test_mapping_v2 = test_mapping_analysis_v2


__all__ = [
    "SilentFailuresTask",
    "TestMappingTask",
    "identify_silent_failures_v2",
    "silent_failures_detection_v2",
    "test_mapping_analysis_v2",
    "test_mapping_v2",
    "TEST_PATTERNS",
    "SKIP_STEMS",
]
