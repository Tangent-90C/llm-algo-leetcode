from __future__ import annotations

import tempfile
from pathlib import Path

from tools.profiling_result_schema import (
    make_result,
    save_result,
    validate_phase_memory_result,
    validate_result,
)


def main() -> None:
    result = make_result(
        config={"model_id": "tiny", "batch_size": 1, "seq_len": 128},
        baseline={"step_time_ms": 10.0},
        tuned={"step_time_ms": 11.0},
        profile={"timing_for_adoption": False},
        bottleneck={"before": "activation", "after": "recompute"},
        validation={
            "evidence_roles": {
                "adoption_metrics": "unprofiled_repeated_measurements",
                "profiler_metrics": "bottleneck_attribution_only",
            }
        },
        decision={"decision": "tune"},
        run_label="smoke-v1",
    )
    assert validate_result(result) == []
    scoped = dict(result)
    scoped["decision_scopes"] = {
        "full_training_adoption": {
            "scope": "full_training_adoption",
            "decision": "tune",
        },
        "activation_memory_mechanism": {
            "scope": "activation_memory_isolation",
            "decision": "accept_for_activation_memory_scope",
            "optimizer": "SGD",
            "promotable_to_full_training": False,
        },
    }
    assert validate_result(scoped) == []
    invalid_scope = dict(scoped)
    invalid_scope["decision_scopes"] = {
        **scoped["decision_scopes"],
        "activation_memory_mechanism": {
            **scoped["decision_scopes"]["activation_memory_mechanism"],
            "decision": "accept",
        },
    }
    assert "activation-memory acceptance must use accept_for_activation_memory_scope" in validate_result(invalid_scope)
    phase_result = {
        "schema_version": "training-phase-memory/v1",
        "semantic_id": "PERF-PROFILING-CLOSURE",
        "config": {"repeats": 3},
        "measurement": {
            "role": "phase_memory_attribution_only",
            "timing_for_adoption": False,
        },
        "candidates": [
            {
                "name": name,
                "status": "ok",
                "successful_runs": 3,
                "phase_metrics": {
                    phase: {"peak_allocated_mb": 1.0}
                    for phase in ("forward", "backward", "optimizer")
                },
            }
            for name in ("baseline", "checkpoint")
        ],
    }
    assert validate_phase_memory_result(phase_result) == []
    invalid = dict(result)
    invalid["profile"] = {"timing_for_adoption": True}
    assert "profile.timing_for_adoption must be false" in validate_result(invalid)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "result.json"
        save_result(path, result)
        try:
            save_result(path, result)
        except FileExistsError:
            pass
        else:
            raise AssertionError("save_result must not overwrite an existing result")
    print("profiling result schema: ok")


if __name__ == "__main__":
    main()
