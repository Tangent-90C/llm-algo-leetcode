"""Shared measurement helpers for the real training-memory notebooks.

Model construction and strategy setup stay in the notebooks.  This module
only keeps timing and CUDA memory accounting consistent across projects 73,
76 and 74.
"""

from __future__ import annotations

import time
from typing import Any, Callable


def synchronize_if_cuda(torch_module: Any) -> None:
    """Synchronize CUDA before and after a timed region when available."""

    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def measure_training_run(
    train_step: Callable[[], float],
    *,
    torch_module: Any,
    batch_size: int,
    warmup: int,
    iters: int,
) -> dict[str, float | str]:
    """Measure complete optimizer steps and return common result fields."""

    if batch_size <= 0 or warmup < 0 or iters <= 0:
        raise ValueError("batch_size must be > 0, warmup >= 0 and iters > 0")

    for _ in range(warmup):
        train_step()
    synchronize_if_cuda(torch_module)
    if torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    losses = [float(train_step()) for _ in range(iters)]
    synchronize_if_cuda(torch_module)
    elapsed = time.perf_counter() - start

    result: dict[str, float | str] = {
        "step_time_ms": round(elapsed * 1000 / iters, 3),
        "samples_per_s": round(batch_size * iters / elapsed, 3),
        "loss": round(losses[-1], 6),
    }
    if torch_module.cuda.is_available():
        result.update(
            {
                "peak_mem_mb": round(torch_module.cuda.max_memory_allocated() / (1024**2), 2),
                "peak_reserved_mb": round(torch_module.cuda.max_memory_reserved() / (1024**2), 2),
            }
        )
    return result


def measure_phased_training_run(
    train_step: Callable[[Callable[[str, Callable[[], Any]], Any]], Any],
    *,
    torch_module: Any,
    batch_size: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    """Measure absolute CUDA peaks independently for each training phase.

    ``train_step`` receives a ``record_phase(name, operation)`` callback and
    must use the phase names ``forward``, ``backward`` and ``optimizer``.
    Resetting peak counters does not free allocations, so each phase records
    its absolute resident peak while preserving the real training lifecycle.
    Timing from this helper is diagnostic only because phase boundaries add
    synchronization overhead.
    """

    if batch_size <= 0 or warmup < 0 or iters <= 0:
        raise ValueError("batch_size must be > 0, warmup >= 0 and iters > 0")
    if not torch_module.cuda.is_available():
        raise RuntimeError("measure_phased_training_run requires CUDA")

    expected_phases = ("forward", "backward", "optimizer")

    def run_without_measurement(_name: str, operation: Callable[[], Any]) -> Any:
        return operation()

    for _ in range(warmup):
        train_step(run_without_measurement)
    synchronize_if_cuda(torch_module)

    observations: dict[str, list[dict[str, float]]] = {
        name: [] for name in expected_phases
    }
    losses: list[float] = []
    measured_start = time.perf_counter()
    for _ in range(iters):
        seen_phases: list[str] = []

        def record_phase(name: str, operation: Callable[[], Any]) -> Any:
            if name not in observations:
                raise ValueError(f"unsupported training phase: {name}")
            if name in seen_phases:
                raise ValueError(f"training phase recorded more than once: {name}")
            seen_phases.append(name)
            synchronize_if_cuda(torch_module)
            start_allocated = torch_module.cuda.memory_allocated()
            start_reserved = torch_module.cuda.memory_reserved()
            torch_module.cuda.reset_peak_memory_stats()
            phase_start = time.perf_counter()
            value = operation()
            synchronize_if_cuda(torch_module)
            elapsed = time.perf_counter() - phase_start
            peak_allocated = torch_module.cuda.max_memory_allocated()
            peak_reserved = torch_module.cuda.max_memory_reserved()
            observations[name].append(
                {
                    "time_ms": elapsed * 1000,
                    "start_allocated_mb": start_allocated / (1024**2),
                    "start_reserved_mb": start_reserved / (1024**2),
                    "peak_allocated_mb": peak_allocated / (1024**2),
                    "peak_reserved_mb": peak_reserved / (1024**2),
                    "allocated_growth_mb": max(
                        0.0, (peak_allocated - start_allocated) / (1024**2)
                    ),
                }
            )
            return value

        loss = train_step(record_phase)
        missing = [name for name in expected_phases if name not in seen_phases]
        if missing:
            raise ValueError(f"train_step did not record required phases: {missing}")
        losses.append(float(loss.detach().item() if hasattr(loss, "detach") else loss))

    synchronize_if_cuda(torch_module)
    elapsed = time.perf_counter() - measured_start
    phase_metrics: dict[str, dict[str, float]] = {}
    for name, phase_observations in observations.items():
        phase_metrics[name] = {
            "time_ms_mean": round(
                sum(item["time_ms"] for item in phase_observations) / iters, 3
            ),
            "peak_allocated_mb": round(
                max(item["peak_allocated_mb"] for item in phase_observations), 2
            ),
            "peak_reserved_mb": round(
                max(item["peak_reserved_mb"] for item in phase_observations), 2
            ),
            "allocated_growth_mb": round(
                max(item["allocated_growth_mb"] for item in phase_observations), 2
            ),
        }

    global_peak_phase = max(
        expected_phases,
        key=lambda name: phase_metrics[name]["peak_allocated_mb"],
    )
    return {
        "step_time_ms": round(elapsed * 1000 / iters, 3),
        "samples_per_s": round(batch_size * iters / elapsed, 3),
        "loss": round(losses[-1], 6),
        "phase_metrics": phase_metrics,
        "global_peak_phase": global_peak_phase,
        "peak_memory_mb": phase_metrics[global_peak_phase]["peak_allocated_mb"],
        "peak_reserved_mb": max(
            item["peak_reserved_mb"] for item in phase_metrics.values()
        ),
        "measurement_role": "phase_memory_attribution_only",
        "timing_for_adoption": False,
    }
