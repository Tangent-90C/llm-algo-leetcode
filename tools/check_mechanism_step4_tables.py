#!/usr/bin/env python3
"""Check the exact Step 4 table contract for Part 02 mechanism notebooks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


EXPECTED_HEADER = "| TODO / 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |"
REPO_ROOT = Path(__file__).resolve().parents[1]
PART02_ROOT = REPO_ROOT / "02_PyTorch_Algorithms"
PROJECT_REGISTRY = REPO_ROOT / "benchmarks/project_registry.json"
TODO_RE = re.compile(r"\bTODO\s*(\d+)\b")


def registered_project_paths(registry_path: Path = PROJECT_REGISTRY) -> set[Path]:
    """Return project notebook paths declared by the project registry."""
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    return {
        Path(item["notebook"]).resolve()
        for item in payload.get("projects", [])
        if item.get("notebook")
    }


def discover_part02_mechanisms() -> list[Path]:
    """Discover mechanism notebooks while excluding projects and reserved slots."""
    projects = registered_project_paths()
    return sorted(
        path
        for path in PART02_ROOT.glob("*.ipynb")
        if path.resolve() not in projects and "Reserved" not in path.name
    )


def expand_paths(paths: list[Path]) -> list[Path]:
    """Expand explicit notebook and directory arguments in stable order."""
    notebooks: set[Path] = set()
    for path in paths:
        if path.is_dir():
            notebooks.update(path.glob("*.ipynb"))
        elif path.suffix == ".ipynb":
            notebooks.add(path)
    return sorted(notebooks)


def step4_markdown(notebook: dict) -> str | None:
    """Return markdown starting at the first mechanism Step 4 heading."""
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        text = "".join(cell.get("source", []))
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if line.startswith("### Step 4"):
                return "\n".join(lines[index:])
    return None


def table_lines(markdown: str) -> list[str]:
    """Return lines from the first Markdown table in a Step 4 cell."""
    lines = markdown.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        result: list[str] = []
        for candidate in lines[index:]:
            if not candidate.startswith("|"):
                break
            result.append(candidate)
        return result
    return []


def audit(path: Path) -> list[str]:
    """Return teaching-template violations for one notebook."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    markdown = step4_markdown(notebook)
    if markdown is None:
        return ["缺少以 `### Step 4` 开头的 Markdown 单元"]

    table = table_lines(markdown)
    if not table:
        return ["Step 4 缺少机制职责表"]

    issues: list[str] = []
    if table[0] != EXPECTED_HEADER:
        issues.append(f"表头漂移：{table[0]!r}")

    for row_number, row in enumerate(table[2:], start=1):
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        if len(cells) != 4:
            issues.append(f"第 {row_number} 行不是四列：{row!r}")
            continue
        first_cell = cells[0]
        if not re.match(r"^TODO\s+\d+", first_cell) or "`" not in first_cell:
            issues.append(
                f"第 {row_number} 行第一列必须同时包含 TODO 编号和反引号函数或方法名：{first_cell!r}"
            )
        if any(not cell for cell in cells):
            issues.append(f"第 {row_number} 行存在空字段：{row!r}")

    question_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if "tutorial-question" in cell.get("metadata", {}).get("tags", [])
    )
    todo_numbers = set(TODO_RE.findall(question_source))
    body_rows = table[2:]
    if todo_numbers and len(body_rows) != len(todo_numbers):
        issues.append(
            "职责行数与题目区 TODO 数量不一致："
            f"表格 {len(body_rows)} 行，TODO {len(todo_numbers)} 个"
        )
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check exact Step 4 tables in Part 02 mechanism notebooks."
    )
    parser.add_argument("notebooks", nargs="*", type=Path)
    parser.add_argument(
        "--all-part02",
        action="store_true",
        help="自动扫描 Part 02 中未登记为项目且不是保留位的机制 Notebook。",
    )
    args = parser.parse_args()

    notebooks = discover_part02_mechanisms() if args.all_part02 else expand_paths(args.notebooks)
    if not notebooks:
        parser.error("请传入机制 Notebook，或使用 --all-part02")

    project_paths = registered_project_paths()
    wrong_type = [path for path in notebooks if path.resolve() in project_paths]
    if wrong_type:
        names = ", ".join(path.as_posix() for path in wrong_type)
        parser.error(f"以下文件已登记为项目页，不能使用机制页 Step 4 检查器：{names}")

    failed = False
    for path in notebooks:
        issues = audit(path)
        if not issues:
            print(f"PASS {path}")
            continue
        failed = True
        print(f"FAIL {path}")
        for issue in issues:
            print(f"  - {issue}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
