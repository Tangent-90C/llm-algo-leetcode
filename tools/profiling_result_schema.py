"""Result schema for project 74, profiling-driven end-to-end optimization.

Project 74 shares the common experiment envelope with projects 73, 76 and 75,
but adds evidence about the diagnosed bottleneck.  The profiling fields are
optional so a CPU-first notebook can still produce a valid report before a
real GPU trace is available.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "profiling-optimization-benchmark/v1"
PHASE_MEMORY_SCHEMA_VERSION = "training-phase-memory/v1"
SEMANTIC_ID = "PERF-PROFILING-CLOSURE"
PROJECT_ID = "74"


def make_result(
    *,
    config: Mapping[str, Any],
    baseline: Mapping[str, Any],
    tuned: Mapping[str, Any],
    decision: Mapping[str, Any],
    profile: Mapping[str, Any] | None = None,
    bottleneck: Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
    source: str | None = None,
    run_label: str | None = None,
    semantic_id: str = SEMANTIC_ID,
    workload_identity: Mapping[str, Any] | None = None,
    sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a portable 74 report while preserving raw candidate metrics."""

    normalized_config = {
        "model": config.get("model", config.get("model_id")),
        "device": config.get("device"),
        "torch": config.get("torch"),
        "cuda": config.get("cuda", config.get("torch_cuda")),
        "dtype": config.get("dtype", config.get("amp_dtype")),
        "batch_size": config.get("batch_size"),
        "seq_len": config.get("seq_len"),
        "warmup": config.get("warmup"),
        "iters": config.get("iters"),
        "workload": config.get("workload"),
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "semantic_id": semantic_id,
        "project": PROJECT_ID,
        "stage": "profiling_driven_optimization",
        "config": normalized_config,
        "candidates": [
            {"name": "baseline", "metrics": dict(baseline)},
            {"name": "tuned", "metrics": dict(tuned)},
        ],
        "profile": dict(profile or {}),
        "bottleneck": dict(bottleneck or {}),
        "validation": dict(validation or {}),
        "decision": dict(decision),
    }
    if run_label:
        result["run_label"] = run_label
    if workload_identity is not None:
        result["workload_identity"] = dict(workload_identity)
    if sources is not None:
        result["sources"] = dict(sources)
    if source:
        result["source"] = source
    return result


def validate_result(result: Mapping[str, Any]) -> list[str]:
    """Validate the project-74 contract without interpreting benchmark values."""

    errors: list[str] = []
    if result.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if result.get("semantic_id") != SEMANTIC_ID:
        errors.append(f"semantic_id must be {SEMANTIC_ID}")
    for key in ("config", "candidates", "profile", "bottleneck", "validation", "decision"):
        if key not in result:
            errors.append(f"missing required field: {key}")
    profile = result.get("profile", {})
    if isinstance(profile, Mapping) and profile.get("timing_for_adoption") is not False:
        errors.append("profile.timing_for_adoption must be false")
    validation = result.get("validation", {})
    if isinstance(validation, Mapping):
        roles = validation.get("evidence_roles", {})
        if roles.get("adoption_metrics") != "unprofiled_repeated_measurements":
            errors.append("validation.evidence_roles.adoption_metrics is invalid")
        if roles.get("profiler_metrics") != "bottleneck_attribution_only":
            errors.append("validation.evidence_roles.profiler_metrics is invalid")
    decision_scopes = result.get("decision_scopes")
    if decision_scopes is not None:
        if not isinstance(decision_scopes, Mapping):
            errors.append("decision_scopes must be an object")
        else:
            full_training = decision_scopes.get("full_training_adoption", {})
            activation = decision_scopes.get("activation_memory_mechanism", {})
            if full_training.get("scope") != "full_training_adoption":
                errors.append("decision_scopes.full_training_adoption.scope is invalid")
            if activation.get("scope") != "activation_memory_isolation":
                errors.append("decision_scopes.activation_memory_mechanism.scope is invalid")
            if activation.get("decision") == "accept":
                errors.append(
                    "activation-memory acceptance must use accept_for_activation_memory_scope"
                )
            if activation.get("decision") == "accept_for_activation_memory_scope":
                if activation.get("optimizer") != "SGD":
                    errors.append("activation-memory acceptance must identify optimizer=SGD")
                if activation.get("promotable_to_full_training") is not False:
                    errors.append(
                        "activation-memory acceptance must not be promotable to full training"
                    )
            top_decision = result.get("decision", {}).get("decision")
            if full_training.get("decision") != top_decision:
                errors.append("top-level decision must match full_training_adoption")
    return errors


def validate_phase_memory_result(result: Mapping[str, Any]) -> list[str]:
    """Validate repeated forward/backward/optimizer memory attribution."""

    errors: list[str] = []
    if result.get("schema_version") != PHASE_MEMORY_SCHEMA_VERSION:
        errors.append(f"schema_version must be {PHASE_MEMORY_SCHEMA_VERSION}")
    if result.get("semantic_id") != SEMANTIC_ID:
        errors.append(f"semantic_id must be {SEMANTIC_ID}")
    measurement = result.get("measurement", {})
    if measurement.get("role") != "phase_memory_attribution_only":
        errors.append("measurement.role must be phase_memory_attribution_only")
    if measurement.get("timing_for_adoption") is not False:
        errors.append("measurement.timing_for_adoption must be false")
    candidates = result.get("candidates", [])
    names = {item.get("name") for item in candidates if isinstance(item, Mapping)}
    if names != {"baseline", "checkpoint"}:
        errors.append("candidates must contain baseline and checkpoint")
    for item in candidates:
        if not isinstance(item, Mapping) or item.get("status") != "ok":
            continue
        phase_metrics = item.get("phase_metrics", {})
        if set(phase_metrics) != {"forward", "backward", "optimizer"}:
            errors.append(f"{item.get('name')}: incomplete phase_metrics")
        if item.get("successful_runs") != result.get("config", {}).get("repeats"):
            errors.append(f"{item.get('name')}: successful_runs does not match repeats")
    return errors


def save_result(path: str | Path, result: Mapping[str, Any]) -> Path:
    """Save a profiling result for notebook and local-GPU workflows."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"profiling result already exists: {output}")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
