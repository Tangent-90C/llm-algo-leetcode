#!/usr/bin/env python3
"""Validate structural arrow rules that are easy to miss in SVG review."""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


MOVETO_RE = re.compile(r"(?<![A-Za-z])[Mm](?=[\s+\-.0-9])")
CLASS_MARKER_RE = re.compile(r"\.([A-Za-z_][\w-]*)\s*\{[^}]*marker-end\s*:", re.DOTALL)
URL_MARKER_RE = re.compile(r"url\(#([^)]+)\)")


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def audit(path: Path) -> list[str]:
    issues: list[str] = []
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        return [f"XML 解析失败：{exc}"]

    root = tree.getroot()
    styles = "\n".join(
        node.text or "" for node in root.iter() if local_name(node.tag) == "style"
    )
    marker_classes = set(CLASS_MARKER_RE.findall(styles))
    marker_ids = {
        node.attrib.get("id") for node in root.iter()
        if local_name(node.tag) == "marker" and node.attrib.get("id")
    }

    for node in root.iter():
        if local_name(node.tag) != "path":
            continue
        classes = set(node.attrib.get("class", "").split())
        marker_value = node.attrib.get("marker-end", "")
        has_arrow = bool(classes & marker_classes) or bool(marker_value)
        if not has_arrow:
            continue
        data = node.attrib.get("d", "")
        if len(MOVETO_RE.findall(data)) > 1:
            issues.append(
                "一条带箭头的 path 含多个 M/m 子路径；请拆成‘一条语义边一个 path’"
            )
        for marker_id in URL_MARKER_RE.findall(marker_value):
            if marker_id not in marker_ids:
                issues.append(f"marker-end 引用了不存在的 marker：{marker_id}")

    for marker_class in marker_classes:
        match = re.search(rf"\.{re.escape(marker_class)}\s*\{{([^}}]*)\}}", styles, re.DOTALL)
        if match:
            for marker_id in URL_MARKER_RE.findall(match.group(1)):
                if marker_id not in marker_ids:
                    issues.append(f"样式 .{marker_class} 引用了不存在的 marker：{marker_id}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    failed = False
    for path in args.files:
        issues = audit(path)
        if issues:
            failed = True
            for issue in issues:
                print(f"{path}: {issue}")
        else:
            print(f"OK {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
