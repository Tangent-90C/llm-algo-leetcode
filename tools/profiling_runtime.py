"""Shared profiler collection for the real GPU optimization project."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Mapping


def infer_bottleneck_assessments(
    profile_report: Mapping[str, Any],
    *,
    evidence_path: str,
    phase_report: Mapping[str, Any] | None = None,
    phase_evidence_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Build conservative bottleneck assessments from a paired trace report.

    The helper only marks checkpoint evidence as supported when both paired
    candidates succeeded and the checkpoint operator table contains an
    explicit checkpoint/recomputation event.  Adoption still depends on the
    unprofiled measurements and qualification gates.
    """

    candidates = {
        item.get("name"): item
        for item in profile_report.get("candidates", [])
        if isinstance(item, Mapping) and item.get("name")
    }
    baseline = candidates.get("baseline", {})
    checkpoint = candidates.get("checkpoint", {})
    baseline_operators = str(baseline.get("top_operators", ""))
    checkpoint_operators = str(checkpoint.get("top_operators", ""))
    required_metrics = ("step_time_ms", "peak_memory_mb")
    metrics_available = all(
        item.get(key) is not None
        for item in (baseline, checkpoint)
        for key in required_metrics
    ) and float(baseline.get("step_time_ms", 0)) > 0
    paired_ok = (
        baseline.get("status") == "ok"
        and checkpoint.get("status") == "ok"
        and metrics_available
    )
    matmul_visible = "aten::mm" in baseline_operators
    recomputation_visible = "CheckpointFunct" in checkpoint_operators

    checkpoint_assessment: dict[str, Any]
    if paired_ok and matmul_visible and recomputation_visible:
        baseline_step_ms = float(baseline["step_time_ms"])
        checkpoint_step_ms = float(checkpoint["step_time_ms"])
        checkpoint_assessment = {
            "status": "supported",
            "reason": (
                "checkpoint_recomputation_increases_work_with_"
                "negligible_allocated_memory_saving"
            ),
            "bottleneck_before": "matmul_dominated_training_step",
            "bottleneck_after": "checkpoint_recomputation_overhead",
            "evidence_path": evidence_path,
            "observations": {
                "profile_step_time_ratio": round(
                    checkpoint_step_ms / baseline_step_ms, 6
                ),
                "profile_allocated_memory_saving_mb": round(
                    float(baseline["peak_memory_mb"])
                    - float(checkpoint["peak_memory_mb"]),
                    3,
                ),
                "checkpoint_event_visible": True,
            },
        }
        phase_candidates = {
            item.get("name"): item
            for item in (phase_report or {}).get("candidates", [])
            if isinstance(item, Mapping) and item.get("name")
        }
        phase_baseline = phase_candidates.get("baseline", {})
        phase_checkpoint = phase_candidates.get("checkpoint", {})
        if (
            phase_baseline.get("status") == "ok"
            and phase_checkpoint.get("status") == "ok"
        ):
            phase_savings = {}
            for name in ("forward", "backward", "optimizer"):
                baseline_phase = phase_baseline.get("phase_metrics", {}).get(name, {})
                checkpoint_phase = phase_checkpoint.get("phase_metrics", {}).get(name, {})
                if (
                    baseline_phase.get("peak_allocated_mb") is not None
                    and checkpoint_phase.get("peak_allocated_mb") is not None
                ):
                    phase_savings[name] = round(
                        float(baseline_phase["peak_allocated_mb"])
                        - float(checkpoint_phase["peak_allocated_mb"]),
                        3,
                    )
            checkpoint_assessment["phase_memory"] = {
                "status": "collected",
                "evidence_path": phase_evidence_path,
                "baseline_global_peak_phase": phase_baseline.get("global_peak_phase"),
                "checkpoint_global_peak_phase": phase_checkpoint.get("global_peak_phase"),
                "allocated_memory_saving_mb": phase_savings,
            }
            checkpoint_assessment["observations"]["phase_memory_collected"] = True
        elif phase_report is not None:
            checkpoint_assessment["phase_memory"] = {
                "status": "invalid",
                "evidence_path": phase_evidence_path,
            }
    else:
        missing = []
        if not paired_ok:
            missing.append("paired_profile_result")
        if not matmul_visible:
            missing.append("baseline_matmul_event")
        if not recomputation_visible:
            missing.append("checkpoint_event")
        checkpoint_assessment = {
            "status": "pending",
            "reason": "missing_" + "_and_".join(missing),
        }

    return {
        "checkpoint": checkpoint_assessment,
        "offload": {
            "status": "not_profiled",
            "reason": "rejected_by_budget_throughput_gate",
        },
        "hybrid": {
            "status": "not_profiled",
            "reason": "rejected_by_budget_throughput_gate",
        },
    }


def summarize_phase_memory_runs(
    name: str,
    runs: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate repeated phase-memory runs without hiding failed repeats."""

    successful = [run for run in runs if run.get("status") == "ok"]
    if not successful:
        return {
            "name": name,
            "status": "failed",
            "runs": [dict(run) for run in runs],
            "successful_runs": 0,
        }

    phase_metrics: dict[str, dict[str, float]] = {}
    for phase in ("forward", "backward", "optimizer"):
        rows = [run["phase_metrics"][phase] for run in successful]
        phase_metrics[phase] = {
            key: round(sum(float(row[key]) for row in rows) / len(rows), 3)
            for key in (
                "time_ms_mean",
                "peak_allocated_mb",
                "peak_reserved_mb",
                "allocated_growth_mb",
            )
        }

    peak_phase_counts: dict[str, int] = {}
    for run in successful:
        phase = str(run["global_peak_phase"])
        peak_phase_counts[phase] = peak_phase_counts.get(phase, 0) + 1
    global_peak_phase = max(peak_phase_counts, key=peak_phase_counts.get)
    return {
        "name": name,
        "status": "ok" if len(successful) == len(runs) else "partial",
        "step_time_ms": round(
            sum(float(run["step_time_ms"]) for run in successful) / len(successful), 3
        ),
        "samples_per_s": round(
            sum(float(run["samples_per_s"]) for run in successful) / len(successful), 3
        ),
        "loss": round(
            sum(float(run["loss"]) for run in successful) / len(successful), 6
        ),
        "peak_memory_mb": round(
            sum(float(run["peak_memory_mb"]) for run in successful) / len(successful), 3
        ),
        "peak_reserved_mb": round(
            sum(float(run["peak_reserved_mb"]) for run in successful) / len(successful), 3
        ),
        "phase_metrics": phase_metrics,
        "global_peak_phase": global_peak_phase,
        "global_peak_phase_counts": peak_phase_counts,
        "runs": [dict(run) for run in runs],
        "successful_runs": len(successful),
        "measurement_role": "phase_memory_attribution_only",
        "timing_for_adoption": False,
    }


def collect_training_trace(
    train_step: Callable[[], Any],
    *,
    torch_module: Any,
    output_dir: str | Path,
    warmup: int,
    iters: int,
    batch_size: int,
) -> dict[str, Any]:
    """Profile fixed optimizer steps and return comparable trace metadata."""

    if warmup < 0 or iters <= 0 or batch_size <= 0:
        raise ValueError("warmup must be >= 0, iters and batch_size must be > 0")
    if not torch_module.cuda.is_available():
        raise RuntimeError("collect_training_trace requires CUDA")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for _ in range(warmup):
        train_step()
    torch_module.cuda.synchronize()
    torch_module.cuda.reset_peak_memory_stats()

    activities = [
        torch_module.profiler.ProfilerActivity.CPU,
        torch_module.profiler.ProfilerActivity.CUDA,
    ]
    # iters=1 时不能先 warmup 再 active，否则循环结束前不会产出 trace。
    schedule = torch_module.profiler.schedule(
        wait=0, warmup=1 if iters > 1 else 0,
        active=max(1, iters - 1) if iters > 1 else 1,
        repeat=1,
    )
    handler = torch_module.profiler.tensorboard_trace_handler(str(output_path))
    losses: list[float] = []
    with torch_module.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        start = time.perf_counter()
        for _ in range(iters):
            value = train_step()
            losses.append(float(value.detach().item() if hasattr(value, "detach") else value))
            profiler.step()
        torch_module.cuda.synchronize()
        elapsed = time.perf_counter() - start
        top_operators = profiler.key_averages().table(
            sort_by="cuda_time_total", row_limit=10
        )

    return {
        "step_time_ms": round(elapsed * 1000 / iters, 3),
        "samples_per_s": round(batch_size * iters / elapsed, 3),
        "loss": round(losses[-1], 6),
        "peak_memory_mb": round(torch_module.cuda.max_memory_allocated() / (1024**2), 2),
        "peak_reserved_mb": round(torch_module.cuda.max_memory_reserved() / (1024**2), 2),
        "allocator_slack_mb": round(
            (
                torch_module.cuda.max_memory_reserved()
                - torch_module.cuda.max_memory_allocated()
            )
            / (1024**2),
            2,
        ),
        "top_operators": top_operators,
        "profile": {
            "tool": "torch.profiler",
            "activities": [activity.name for activity in activities],
            "trace_dir": str(output_path),
            "warmup": warmup,
            "iters": iters,
            "active_steps": max(1, iters - 1) if iters > 1 else 1,
            "repeat": 1,
            "measurement_role": "bottleneck_attribution_only",
            "timing_for_adoption": False,
            "status": "collected",
        },
    }
