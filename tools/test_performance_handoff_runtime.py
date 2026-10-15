from __future__ import annotations

from tools.performance_handoff_runtime import (
    build_performance_handoff,
    build_workload_identity,
    qualify_memory_candidates,
)
from tools.profiling_runtime import (
    infer_bottleneck_assessments,
    summarize_phase_memory_runs,
)


def main() -> None:
    config = {"model_id": "tiny", "batch_size": 1, "seq_len": 128, "dtype": "float32", "optimizer": "AdamW", "workload": "pressure", "seed": 42, "device": "gpu", "torch": "2", "torch_cuda": "12.8"}
    workload = build_workload_identity({"config": dict(config)}, {"config": dict(config)})
    strategy = {"candidates": [
        {"name": "baseline", "status": "ok", "peak_memory_mb": 1000.0, "samples_per_s": 10.0, "eval_loss": 1.0},
        {"name": "checkpoint", "status": "ok", "peak_memory_mb": 700.0, "samples_per_s": 8.5, "eval_loss": 1.0},
    ]}
    budget = {"summary": {"evaluations": [{"name": "checkpoint", "status": "feasible"}]}}
    profile = {"candidates": [
        {"name": "baseline", "status": "ok", "step_time_ms": 100.0, "peak_memory_mb": 1000.0, "top_operators": "aten::mm"},
        {"name": "checkpoint", "status": "ok", "step_time_ms": 115.0, "peak_memory_mb": 990.0, "top_operators": "CheckpointFunction"},
    ]}
    gates = {"minimum_memory_saving_mb": 256.0, "minimum_throughput_ratio": 0.8, "maximum_loss_delta": 0.01}
    missing = qualify_memory_candidates(strategy, budget, profile, {}, gates)
    assert missing["candidate_records"][0]["decision"] == "tune"
    phase_run = {
        "status": "ok", "step_time_ms": 120.0, "samples_per_s": 8.0,
        "loss": 1.0, "peak_memory_mb": 900.0, "peak_reserved_mb": 950.0,
        "global_peak_phase": "optimizer",
        "phase_metrics": {
            name: {
                "time_ms_mean": 10.0, "peak_allocated_mb": peak,
                "peak_reserved_mb": peak + 50.0, "allocated_growth_mb": 100.0,
            }
            for name, peak in (("forward", 700.0), ("backward", 800.0), ("optimizer", 900.0))
        },
    }
    phase_baseline = summarize_phase_memory_runs("baseline", [phase_run, phase_run, phase_run])
    checkpoint_run = dict(phase_run)
    checkpoint_run["phase_metrics"] = {
        name: dict(values) for name, values in phase_run["phase_metrics"].items()
    }
    checkpoint_run["phase_metrics"]["backward"]["peak_allocated_mb"] = 650.0
    phase_checkpoint = summarize_phase_memory_runs(
        "checkpoint", [checkpoint_run, checkpoint_run, checkpoint_run]
    )
    phase_report = {"candidates": [phase_baseline, phase_checkpoint]}
    assessments = infer_bottleneck_assessments(
        profile, evidence_path="trace.json",
        phase_report=phase_report, phase_evidence_path="phase.json",
    )
    assert assessments["checkpoint"]["status"] == "supported"
    assert assessments["checkpoint"]["observations"]["profile_step_time_ratio"] == 1.15
    assert assessments["checkpoint"]["phase_memory"]["allocated_memory_saving_mb"]["backward"] == 150.0
    qualified = qualify_memory_candidates(strategy, budget, profile, assessments, gates)
    assert qualified["qualified_candidates"][0]["name"] == "checkpoint"
    handoff = build_performance_handoff(
        run_label="smoke-v1", workload=workload, baseline_path="baseline.json",
        profiling_report_path="profile.json", qualification=qualified,
        source_artifacts={"73": {"path": "baseline.json", "sha256": "abc123"}},
        decision_scopes={
            "full_training_adoption": {
                "scope": "full_training_adoption", "decision": "accept",
            },
            "activation_memory_mechanism": {
                "scope": "activation_memory_isolation",
                "decision": "accept_for_activation_memory_scope",
                "optimizer": "SGD",
                "promotable_to_full_training": False,
            },
        },
    )
    assert handoff["decision"] == "accept"
    assert handoff["qualified_candidates"][0]["metrics"]["memory_saving_mb"] == 300.0
    assert handoff["source_artifacts"]["73"]["sha256"] == "abc123"
    assert handoff["decision_scopes"]["activation_memory_mechanism"]["promotable_to_full_training"] is False
    print("performance handoff runtime: ok")


if __name__ == "__main__":
    main()
