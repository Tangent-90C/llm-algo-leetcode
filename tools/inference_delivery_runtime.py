"""Contracts and decision logic for the Part 05 inference delivery project."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CONFIG_SCHEMA = "inference-service-delivery-config/v1"
HANDOFF_SCHEMA = "inference-deployment-handoff/v1"
RESULT_SCHEMA = "inference-service-delivery/v1"
SEMANTIC_ID = "INF-INTEGRATED-DELIVERY"


def _metric(record: Mapping[str, Any], key: str, statistic: str | None = None) -> float | None:
    metrics = record.get("metrics", {})
    value = metrics.get(key) if isinstance(metrics, Mapping) else None
    if isinstance(value, Mapping):
        value = value.get(statistic or "p50")
    return float(value) if isinstance(value, (int, float)) else None


def _has_failure(record: Mapping[str, Any]) -> bool:
    failure = record.get("failure")
    if failure in (None, {}, "", "not_recorded"):
        return False
    if isinstance(failure, Mapping):
        return any(value not in (None, "", {}, []) for value in failure.values())
    return True


def load_json_record(path: str | Path) -> dict[str, Any]:
    """Load one JSON object without guessing around a missing path."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"项目记录不存在：{source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"项目记录必须是 JSON object：{source}")
    return dict(payload)


def normalized_backend_result(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the shared inference record stored by the backend benchmark."""

    normalized = record.get("normalized_result")
    if isinstance(normalized, Mapping):
        return dict(normalized)
    if record.get("schema_version") == "inference-benchmark/v1" and "metrics" in record:
        return dict(record)
    raise ValueError("后端结果缺少 normalized_result 或 inference-benchmark/v1 记录")


def load_quality_score(path: str | Path | None) -> float | None:
    """Read a direction-normalized quality score from an external report."""

    if not path:
        return None
    record = load_json_record(path)
    value = record.get("quality_score")
    if value is None and isinstance(record.get("quality"), Mapping):
        value = record["quality"].get("score")
    if not isinstance(value, (int, float)):
        raise ValueError(f"质量报告缺少数值 quality_score：{path}")
    return float(value)


def validate_delivery_config(config: Mapping[str, Any]) -> list[str]:
    """Validate identities, workload and gates before contacting a backend."""

    errors: list[str] = []
    if config.get("schema_version") != CONFIG_SCHEMA:
        errors.append(f"schema_version 必须是 {CONFIG_SCHEMA}")
    if config.get("semantic_id") != SEMANTIC_ID:
        errors.append(f"semantic_id 必须是 {SEMANTIC_ID}")
    if not str(config.get("run_label", "")).strip() or config.get("run_label") == "replace_with_unique_label":
        errors.append("run_label 必须替换为唯一值")
    handoff = config.get("inputs", {}).get("deployment_handoff")
    if not str(handoff or "").strip():
        errors.append("inputs.deployment_handoff 不能为空")
    workload = config.get("workload", {})
    for key in ("path", "num_prompts", "max_tokens", "concurrency", "warmup", "repeats"):
        if key not in workload:
            errors.append(f"workload 缺少 {key}")
    for key in ("num_prompts", "max_tokens", "concurrency", "repeats"):
        if key in workload and int(workload[key]) <= 0:
            errors.append(f"workload.{key} 必须大于 0")
    candidates = config.get("candidates")
    if not isinstance(candidates, Mapping) or not candidates:
        errors.append("candidates 至少包含一个服务候选")
    else:
        roles = [item.get("role") for item in candidates.values() if isinstance(item, Mapping)]
        if roles.count("baseline") != 1:
            errors.append("candidates 必须且只能包含一个 baseline")
        for name, item in candidates.items():
            if not isinstance(item, Mapping):
                errors.append(f"candidate {name} 必须是 mapping")
                continue
            for key in ("role", "model", "backend", "dtype", "base_url", "result_path"):
                if not str(item.get(key, "")).strip():
                    errors.append(f"candidate {name} 缺少 {key}")
    gates = config.get("gates", {})
    for key in (
        "minimum_success_rate",
        "maximum_quality_regression",
        "maximum_ttft_p95_ms",
        "maximum_tpot_p95_ms",
        "minimum_output_throughput_tps",
        "maximum_peak_memory_gib",
    ):
        if key not in gates:
            errors.append(f"gates 缺少 {key}")
    return errors


def validate_deployment_handoff(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the single direct Part 02 handoff consumed by Part 05."""

    required = {
        "schema_version",
        "semantic_id",
        "run_label",
        "workload",
        "candidate_records",
        "decision",
        "rollback",
    }
    missing = sorted(required - set(record))
    if missing:
        raise KeyError(f"单卡部署交接清单缺少字段：{missing}")
    if record.get("schema_version") != HANDOFF_SCHEMA:
        raise ValueError(f"交接清单版本必须是 {HANDOFF_SCHEMA}")
    if record.get("semantic_id") != "INF-SINGLE-GPU-DEPLOY":
        raise ValueError("交接清单 semantic_id 必须是 INF-SINGLE-GPU-DEPLOY")
    ready = [
        dict(item)
        for item in record.get("candidate_records", [])
        if isinstance(item, Mapping) and item.get("handoff_status") == "ready_for_integrated_evaluation"
    ]
    return {
        "ready_candidates": ready,
        "handoff_decision": record.get("decision"),
        "rollback_ready": bool(record.get("rollback", {}).get("ready")),
    }


def validate_comparable_runs(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Check that a measured candidate uses the frozen Part 05 workload."""

    fixed_config = ("model", "prompt_tokens", "generated_tokens", "batch", "concurrency")
    baseline_config = baseline.get("config", {})
    candidate_config = candidate.get("config", {})
    mismatches = [
        key for key in fixed_config if baseline_config.get(key) != candidate_config.get(key)
    ]
    return {"comparable": not mismatches, "mismatches": mismatches}


def evaluate_service_candidate(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    gates: Mapping[str, Any],
    *,
    quality_score: float | None,
    baseline_quality_score: float | None,
    artifact_verified: bool,
    rollback_ready: bool,
) -> dict[str, Any]:
    """Apply availability gates before SLO, quality and capacity gates."""

    comparison = validate_comparable_runs(baseline, candidate)
    hard_failures: list[str] = []
    missing_evidence: list[str] = []
    failed_requests = _metric(candidate, "failed_requests")
    successful_requests = _metric(candidate, "successful_requests")
    total_requests = None
    if failed_requests is not None and successful_requests is not None:
        total_requests = failed_requests + successful_requests
    success_rate = successful_requests / total_requests if total_requests else None

    if not comparison["comparable"]:
        hard_failures.append("workload_mismatch")
    if _has_failure(candidate):
        hard_failures.append("backend_failure")
    if not artifact_verified:
        hard_failures.append("artifact_unverified")
    if not rollback_ready:
        hard_failures.append("rollback_unavailable")

    ttft_p95 = _metric(candidate, "ttft_ms", "p95")
    tpot_p95 = _metric(candidate, "tpot_ms", "p95")
    throughput = _metric(candidate, "throughput_tokens_per_s")
    peak_memory_mb = _metric(candidate, "peak_memory_mb")
    peak_memory_gib = peak_memory_mb / 1024 if peak_memory_mb is not None else None
    evidence = {
        "success_rate": success_rate,
        "quality_score": quality_score,
        "ttft_p95_ms": ttft_p95,
        "tpot_p95_ms": tpot_p95,
        "output_throughput_tps": throughput,
        "peak_memory_gib": peak_memory_gib,
    }
    for key, value in evidence.items():
        if value is None:
            missing_evidence.append(key)

    failed_gates: list[str] = []
    if success_rate is not None and success_rate < gates["minimum_success_rate"]:
        failed_gates.append("success_rate")
    if quality_score is not None and quality_score < gates.get("minimum_quality_score", 0.0):
        failed_gates.append("quality")
    if quality_score is not None and baseline_quality_score is not None:
        if baseline_quality_score - quality_score > gates["maximum_quality_regression"]:
            failed_gates.append("quality_regression")
    if ttft_p95 is not None and ttft_p95 > gates["maximum_ttft_p95_ms"]:
        failed_gates.append("ttft_p95")
    if tpot_p95 is not None and tpot_p95 > gates["maximum_tpot_p95_ms"]:
        failed_gates.append("tpot_p95")
    if throughput is not None and throughput < gates["minimum_output_throughput_tps"]:
        failed_gates.append("throughput")
    if peak_memory_gib is not None and peak_memory_gib > gates["maximum_peak_memory_gib"]:
        failed_gates.append("peak_memory")

    if hard_failures:
        decision = "reject"
    elif missing_evidence or failed_gates:
        decision = "tune"
    else:
        decision = "accept"
    return {
        "decision": decision,
        "hard_failures": hard_failures,
        "missing_evidence": missing_evidence,
        "failed_gates": failed_gates,
        "comparison": comparison,
        "evidence": evidence,
    }


def select_service_candidate(
    decisions: Mapping[str, Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Select one accepted candidate by TPOT, then TTFT, then throughput."""

    accepted = [name for name, item in decisions.items() if item.get("decision") == "accept"]
    if not accepted:
        fallback = "tune" if any(item.get("decision") == "tune" for item in decisions.values()) else "reject"
        return {"decision": fallback, "selected_candidate": None, "accepted_candidates": []}

    def rank(name: str) -> tuple[float, float, float]:
        record = results[name]
        tpot = _metric(record, "tpot_ms", "p95")
        ttft = _metric(record, "ttft_ms", "p95")
        throughput = _metric(record, "throughput_tokens_per_s")
        return (
            tpot if tpot is not None else float("inf"),
            ttft if ttft is not None else float("inf"),
            -(throughput if throughput is not None else 0.0),
        )

    selected = min(accepted, key=rank)
    return {"decision": "accept", "selected_candidate": selected, "accepted_candidates": sorted(accepted)}


def build_delivery_record(
    *,
    run_label: str,
    handoff_path: str | Path,
    workload: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable final record without hiding candidate-level evidence."""

    source = Path(handoff_path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
    return {
        "schema_version": RESULT_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "run_label": run_label,
        "source_handoff": {"path": str(source), "sha256": digest},
        "workload": dict(workload),
        "candidate_results": {name: dict(value) for name, value in results.items()},
        "candidate_decisions": {name: dict(value) for name, value in decisions.items()},
        "decision": selection.get("decision"),
        "selected_candidate": selection.get("selected_candidate"),
        "rollback_candidate": "baseline",
        "evidence_level": "integrated_system_evidence",
    }


def save_delivery_record(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Save a delivery decision without silently replacing an earlier run."""

    output = Path(path)
    if output.exists():
        raise FileExistsError(f"结果已存在，请更换 run_label：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(record), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
