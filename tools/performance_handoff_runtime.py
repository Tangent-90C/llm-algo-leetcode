"""Qualification logic for the Part 02 performance handoff project."""

from __future__ import annotations

from typing import Any, Mapping


HANDOFF_SCHEMA = "performance-optimization-handoff/v1"
SEMANTIC_ID = "PERF-PROFILING-CLOSURE"


def build_workload_identity(
    baseline_report: Mapping[str, Any], strategy_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Build and validate the fixed workload shared by projects 73 and 76."""

    baseline = baseline_report.get("config", {})
    strategy = strategy_report.get("config", {})
    fields = ("model_id", "batch_size", "seq_len", "dtype", "optimizer", "workload", "seed")
    mismatches = [name for name in fields if baseline.get(name) != strategy.get(name)]
    if mismatches:
        raise ValueError(f"73/76 工作负载身份不一致：{mismatches}")
    return {
        "family": "training",
        "workload_id": f"{strategy.get('workload')}-b{strategy.get('batch_size')}-s{strategy.get('seq_len')}",
        "model": strategy.get("model_id"),
        "input": {"batch_size": strategy.get("batch_size"), "seq_len": strategy.get("seq_len")},
        "hardware": strategy.get("device"),
        "runtime": {"torch": strategy.get("torch"), "cuda": strategy.get("torch_cuda"), "dtype": strategy.get("dtype")},
        "seed": strategy.get("seed"),
    }


def qualify_memory_candidates(
    strategy_report: Mapping[str, Any],
    budget_report: Mapping[str, Any],
    profile_report: Mapping[str, Any],
    assessments: Mapping[str, Mapping[str, Any]],
    gates: Mapping[str, float],
) -> dict[str, Any]:
    """Qualify candidates using budget, paired metrics and interpreted traces."""

    strategy_by_name = {item.get("name"): item for item in strategy_report.get("candidates", [])}
    profile_by_name = {item.get("name"): item for item in profile_report.get("candidates", [])}
    budget_status = {
        item.get("name"): item
        for item in budget_report.get("summary", {}).get("evaluations", [])
    }
    baseline = strategy_by_name.get("baseline")
    profile_baseline = profile_by_name.get("baseline")
    if not baseline or not profile_baseline:
        raise ValueError("76 或 74 profiler 报告缺少 baseline")

    records: list[dict[str, Any]] = []
    for name, candidate in strategy_by_name.items():
        if name == "baseline":
            continue
        hard_failures: list[str] = []
        missing_evidence: list[str] = []
        failed_gates: list[str] = []
        profile_candidate = profile_by_name.get(name)
        assessment = assessments.get(name, {})
        if candidate.get("status") != "ok":
            hard_failures.append("candidate_execution_failed")
        if budget_status.get(name, {}).get("status") != "feasible":
            hard_failures.append("budget_rejected")
        # Candidates rejected by a hard budget/execution gate are intentionally
        # not profiled.  Do not report their skipped evidence as an accidental
        # data-quality gap.
        if not hard_failures:
            if not profile_candidate or profile_candidate.get("status") != "ok":
                missing_evidence.append("paired_profile_result")
            if assessment.get("status") != "supported":
                missing_evidence.append("interpreted_bottleneck_evidence")
            for key in ("bottleneck_before", "bottleneck_after", "evidence_path"):
                if not assessment.get(key):
                    missing_evidence.append(key)

        memory_saving_mb = baseline["peak_memory_mb"] - candidate["peak_memory_mb"]
        throughput_ratio = candidate["samples_per_s"] / baseline["samples_per_s"]
        loss_delta = abs(candidate.get("eval_loss", candidate.get("loss")) - baseline.get("eval_loss", baseline.get("loss")))
        if memory_saving_mb < gates["minimum_memory_saving_mb"]:
            failed_gates.append("memory_saving")
        if throughput_ratio < gates["minimum_throughput_ratio"]:
            failed_gates.append("throughput")
        if loss_delta > gates["maximum_loss_delta"]:
            failed_gates.append("quality")

        if hard_failures:
            decision = "reject"
        elif missing_evidence or failed_gates:
            decision = "tune"
        else:
            decision = "accept"
        records.append({
            "name": name,
            "decision": decision,
            "hard_failures": hard_failures,
            "missing_evidence": sorted(set(missing_evidence)),
            "failed_gates": failed_gates,
            "metrics": {
                "memory_saving_mb": round(memory_saving_mb, 3),
                "throughput_ratio": round(throughput_ratio, 6),
                "loss_delta": round(loss_delta, 8),
                "profile_step_time_ratio": (
                    round(profile_candidate["step_time_ms"] / profile_baseline["step_time_ms"], 6)
                    if profile_candidate else None
                ),
            },
            "bottleneck_assessment": dict(assessment),
        })
    return {
        "candidate_records": records,
        "qualified_candidates": [item for item in records if item["decision"] == "accept"],
    }


def build_performance_handoff(
    *,
    run_label: str,
    workload: Mapping[str, Any],
    baseline_path: str,
    profiling_report_path: str,
    qualification: Mapping[str, Any],
    source_artifacts: Mapping[str, Any] | None = None,
    decision_scopes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a handoff whose decision reflects 74's own qualification."""

    qualified = list(qualification.get("qualified_candidates", []))
    candidate_records = list(qualification.get("candidate_records", []))
    if qualified:
        decision = "accept"
    elif any(item.get("decision") == "tune" for item in candidate_records):
        decision = "tune"
    else:
        decision = "reject"
    result = {
        "schema_version": HANDOFF_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "run_label": run_label,
        "workload": dict(workload),
        "baseline": {"result_path": baseline_path},
        "candidate_records": candidate_records,
        "qualified_candidates": qualified,
        "profiling_report": profiling_report_path,
        "rollback": {"ready": bool(baseline_path), "target": "training_baseline"},
        "decision": decision,
    }
    if source_artifacts is not None:
        result["source_artifacts"] = dict(source_artifacts)
    if decision_scopes is not None:
        result["decision_scopes"] = {
            name: dict(scope) for name, scope in decision_scopes.items()
        }
    return result
