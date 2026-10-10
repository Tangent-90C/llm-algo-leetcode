#!/usr/bin/env python3
"""Audit Part 0 / Part 1 notebooks with profile-based checks.

Profiles:
    - codecell_run: execute code cells sequentially and report per-notebook status
    - structure_only: inspect markdown structure, link counts, and cell-id hygiene
    - all: run both profiles
"""

from __future__ import annotations

import argparse
import contextlib
import json
import io
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRS = [
    ROOT / "00_Prerequisites",
    ROOT / "01_Hardware_Math_and_Systems",
]
PART1_DIRNAME = "01_Hardware_Math_and_Systems"
SECTION_TITLES = ("本节导读", "前置阅读", "相关阅读")
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


@dataclass
class CodeRunResult:
    path: Path
    ok: bool
    executed_cells: int
    failed_cell: int | None = None
    error: str | None = None
    output: str = ""


@dataclass
class StructureResult:
    path: Path
    ok: bool
    hard_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_ids: int = 0
    duplicate_ids: int = 0
    prereq_links: int = 0
    related_links: int = 0


def iter_notebooks(base_dirs: list[Path], pattern: str) -> list[Path]:
    notebooks: list[Path] = []
    for base in base_dirs:
        if base.is_file() and base.suffix == ".ipynb":
            notebooks.append(base.resolve())
            continue
        if base.exists():
            notebooks.extend(sorted(p.resolve() for p in base.glob(pattern)))

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in notebooks:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def read_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_code_cells(path: Path, *, show_output: bool) -> CodeRunResult:
    nb = read_notebook(path)
    ns: dict[str, object] = {"__name__": "__main__"}
    executed = 0
    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        for idx, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if not source.strip():
                continue
            try:
                exec(compile(source, str(path), "exec"), ns, ns)
                executed += 1
            except Exception as exc:
                traceback_str = "".join(traceback.format_exception(exc))
                return CodeRunResult(
                    path=path,
                    ok=False,
                    executed_cells=executed,
                    failed_cell=idx,
                    error=traceback_str.strip(),
                    output=buffer.getvalue().strip(),
                )

    output = buffer.getvalue().strip() if show_output else ""
    return CodeRunResult(path=path, ok=True, executed_cells=executed, output=output)


def collect_markdown_lines(nb: dict) -> list[str]:
    lines: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        lines.extend("".join(cell.get("source", [])).splitlines())
        lines.append("")
    return lines


def extract_section(lines: list[str], title: str) -> list[str]:
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == f"## {title}":
            start = idx + 1
            break
    if start is None:
        return []

    end = len(lines)
    for idx in range(start, len(lines)):
        stripped = lines[idx].strip()
        if stripped.startswith("## ") and stripped != f"## {title}":
            end = idx
            break
    return lines[start:end]


def count_links(lines: list[str]) -> int:
    count = 0
    for line in lines:
        count += len(LINK_RE.findall(line))
    return count


def section_cell_indexes(nb: dict, title: str) -> list[int]:
    """Return markdown-cell indexes containing one exact level-2 section title."""
    expected = f"## {title}"
    indexes: list[int] = []
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "markdown":
            continue
        lines = "".join(cell.get("source", [])).splitlines()
        if any(line.strip() == expected for line in lines):
            indexes.append(idx)
    return indexes


def last_nonempty_cell_index(nb: dict) -> int | None:
    indexes = [
        idx
        for idx, cell in enumerate(nb.get("cells", []))
        if "".join(cell.get("source", [])).strip()
    ]
    return indexes[-1] if indexes else None


def audit_structure(path: Path, *, treat_missing_id_as_fail: bool) -> StructureResult:
    nb = read_notebook(path)
    lines = collect_markdown_lines(nb)
    result = StructureResult(path=path, ok=True)

    ids: list[str] = []
    for cell in nb.get("cells", []):
        cell_id = cell.get("id")
        if cell_id is None:
            result.missing_ids += 1
        else:
            ids.append(cell_id)

    seen: set[str] = set()
    duplicates: set[str] = set()
    for cell_id in ids:
        if cell_id in seen:
            duplicates.add(cell_id)
        seen.add(cell_id)
    result.duplicate_ids = len(duplicates)

    prereq_lines = extract_section(lines, "前置阅读")
    related_lines = extract_section(lines, "相关阅读")
    result.prereq_links = count_links(prereq_lines)
    result.related_links = count_links(related_lines)

    is_part1 = path.parent.name == PART1_DIRNAME
    for title in ("本节导读", "相关阅读"):
        indexes = section_cell_indexes(nb, title)
        if not indexes:
            result.hard_failures.append(f"missing section: {title}")
        elif len(indexes) > 1:
            result.hard_failures.append(f"duplicate section: {title}")

    related_cells = section_cell_indexes(nb, "相关阅读")
    last_cell = last_nonempty_cell_index(nb)
    if len(related_cells) == 1 and related_cells[0] != last_cell:
        result.hard_failures.append(
            f"相关阅读 must be the final non-empty cell: cell={related_cells[0]}, last={last_cell}"
        )

    if is_part1:
        for title in ("前置阅读",):
            if not any(line.strip() == f"## {title}" for line in lines):
                result.hard_failures.append(f"missing section: {title}")
        if result.prereq_links > 3:
            result.hard_failures.append(f"前置阅读 has {result.prereq_links} links (> 3)")
        if result.related_links > 3:
            result.hard_failures.append(f"相关阅读 has {result.related_links} links (> 3)")

    if result.duplicate_ids:
        result.hard_failures.append(f"duplicate cell ids: {result.duplicate_ids}")

    if result.missing_ids:
        message = f"missing cell ids: {result.missing_ids}"
        if treat_missing_id_as_fail:
            result.hard_failures.append(message)
        else:
            result.warnings.append(message)

    result.ok = not result.hard_failures
    return result


def print_codecell_summary(results: list[CodeRunResult]) -> bool:
    print("\n" + "=" * 72)
    print("Profile: codecell_run")
    print("=" * 72)
    ok = True
    for result in results:
        rel = result.path.relative_to(ROOT)
        if result.ok:
            print(f"PASS {rel} | executed_cells={result.executed_cells}")
            if result.output:
                print(result.output)
            continue
        ok = False
        print(f"FAIL {rel} | executed_cells={result.executed_cells} | failed_cell={result.failed_cell}")
        if result.output:
            print(result.output)
        if result.error:
            print(result.error)
    return ok


def print_structure_summary(results: list[StructureResult]) -> bool:
    print("\n" + "=" * 72)
    print("Profile: structure_only")
    print("=" * 72)
    ok = True
    for result in results:
        rel = result.path.relative_to(ROOT)
        status = "PASS" if result.ok else "FAIL"
        detail = (
            f"prereq_links={result.prereq_links}, related_links={result.related_links}, "
            f"missing_ids={result.missing_ids}, duplicate_ids={result.duplicate_ids}"
        )
        print(f"{status} {rel} | {detail}")
        for failure in result.hard_failures:
            print(f"  hard-failure: {failure}")
        for warning in result.warnings:
            print(f"  warning: {warning}")
        if result.hard_failures:
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Part 0 / Part 1 notebooks.")
    parser.add_argument(
        "--dir",
        action="append",
        default=None,
        help="Directory or notebook path to scan. Can be passed multiple times.",
    )
    parser.add_argument(
        "--pattern",
        default="*.ipynb",
        help="Glob pattern to match notebooks. Defaults to all .ipynb files.",
    )
    parser.add_argument(
        "--profile",
        choices=("codecell_run", "structure_only", "all"),
        default="all",
        help="Audit profile to run.",
    )
    parser.add_argument(
        "--treat-missing-id-as-fail",
        action="store_true",
        help="Escalate missing cell ids from warnings to failures.",
    )
    parser.add_argument(
        "--show-cell-output",
        action="store_true",
        help="Print captured notebook stdout/stderr for passing codecell runs.",
    )
    args = parser.parse_args()

    base_dirs = [Path(d).resolve() for d in args.dir] if args.dir else DEFAULT_DIRS
    notebooks = iter_notebooks(base_dirs, args.pattern)
    if not notebooks:
        print("No notebooks found.")
        return 1

    overall_ok = True

    if args.profile in {"codecell_run", "all"}:
        code_results = [run_code_cells(path, show_output=args.show_cell_output) for path in notebooks]
        overall_ok = print_codecell_summary(code_results) and overall_ok

    if args.profile in {"structure_only", "all"}:
        structure_results = [
            audit_structure(path, treat_missing_id_as_fail=args.treat_missing_id_as_fail)
            for path in notebooks
        ]
        overall_ok = print_structure_summary(structure_results) and overall_ok

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
