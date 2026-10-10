"""Validate common contracts and mechanism-specific evidence in quantization JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


COMMON_FIELDS = {
    "schema_version",
    "chapter",
    "environment_profile",
    "run_mode",
    "framework",
    "hardware",
    "workload",
    "failure",
    "evidence_level",
    "decision",
}

MECHANISM_FIELDS = {
    "25": {"measurement_matrix", "summary"},
    "40": {"candidates", "mechanism_metrics"},
    "41": {"candidates", "mechanism_metrics"},
    "53": {"candidates", "mechanism_metrics"},
    "54": {"measurement_matrix", "mechanism_metrics"},
    "QUANT-MIXED-BIT": {"sensitivity", "candidates", "selected_candidate"},
    "QUANT-QAT": {"ptq", "qat_candidates", "selected_candidate", "artifact_manifest"},
}


def validate_result(payload: Mapping[str, Any], path: Path) -> list[str]:
    """Return actionable contract errors for one mechanism result."""

    errors: list[str] = []
    missing = COMMON_FIELDS - set(payload)
    if missing:
        errors.append(f"缺少公共字段：{sorted(missing)}")
    schema = payload.get("schema_version")
    if not isinstance(schema, str) or not schema.startswith("quantization-mechanism-gpu/"):
        errors.append(f"schema_version 非量化机制契约：{schema!r}")
    if payload.get("run_mode") == "real_gpu":
        hardware = payload.get("hardware") or {}
        if not hardware.get("cuda_available"):
            errors.append("real_gpu 结果没有确认 CUDA 可用")
        if not hardware.get("device"):
            errors.append("real_gpu 结果缺少 GPU 名称")
    if payload.get("failure") is None and payload.get("evidence_level") in {
        None,
        "dry_run",
        "environment_preflight",
        "not_executed",
    }:
        errors.append("成功结果仍停留在预检或未执行证据等级")

    chapter = str(payload.get("chapter"))
    required = MECHANISM_FIELDS.get(chapter)
    if required is None:
        errors.append(f"未登记的量化机制 chapter：{chapter!r}")
    else:
        mechanism_missing = required - set(payload)
        if mechanism_missing:
            errors.append(f"缺少 {chapter} 专属字段：{sorted(mechanism_missing)}")

    workload = payload.get("workload") or {}
    if chapter != "25" and not workload.get("resolved_model_commit"):
        errors.append("workload 缺少 resolved_model_commit")
    if chapter == "25" and not (
        workload.get("resolved_model_commit") or workload.get("loader_revision")
    ):
        errors.append("workload 缺少解析后的模型版本")
    return [f"{path}: {message}" for message in errors]


def main() -> int:
    parser = argparse.ArgumentParser(description="验证量化机制 GPU JSON 契约")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    for path in args.paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        errors.extend(validate_result(payload, path))
    if errors:
        print("\n".join(errors))
        return 1
    print(f"quantization result contracts: ok ({len(args.paths)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
