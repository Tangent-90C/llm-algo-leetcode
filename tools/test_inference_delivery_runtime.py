from __future__ import annotations

from tools.inference_delivery_runtime import (
    CONFIG_SCHEMA,
    SEMANTIC_ID,
    evaluate_service_candidate,
    select_service_candidate,
    validate_delivery_config,
    validate_deployment_handoff,
)


def _config() -> dict:
    return {
        "schema_version": CONFIG_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "run_label": "smoke-v1",
        "inputs": {"deployment_handoff": "handoff.json"},
        "workload": {
            "path": "workload.jsonl",
            "num_prompts": 5,
            "max_tokens": 32,
            "concurrency": 1,
            "warmup": 1,
            "repeats": 3,
        },
        "candidates": {
            "baseline": {
                "role": "baseline",
                "model": "model",
                "backend": "vllm",
                "dtype": "bf16",
                "base_url": "http://127.0.0.1:8000",
                "result_path": "baseline.json",
            },
            "candidate": {
                "role": "candidate",
                "model": "model",
                "backend": "vllm",
                "dtype": "bf16",
                "base_url": "http://127.0.0.1:8000",
                "result_path": "candidate.json",
            },
        },
        "gates": {
            "minimum_success_rate": 1.0,
            "minimum_quality_score": 0.8,
            "maximum_quality_regression": 0.01,
            "maximum_ttft_p95_ms": 100.0,
            "maximum_tpot_p95_ms": 20.0,
            "minimum_output_throughput_tps": 10.0,
            "maximum_peak_memory_gib": 11.0,
        },
    }


def _result(*, tpot: float, ttft: float, throughput: float, peak_mb: float) -> dict:
    return {
        "schema_version": "inference-benchmark/v1",
        "config": {
            "model": "model",
            "prompt_tokens": 64,
            "generated_tokens": 32,
            "batch": 1,
            "concurrency": 1,
        },
        "metrics": {
            "successful_requests": 5,
            "failed_requests": 0,
            "ttft_ms": {"p95": ttft},
            "tpot_ms": {"p95": tpot},
            "throughput_tokens_per_s": throughput,
            "peak_memory_mb": peak_mb,
        },
        "failure": {"failure_reason": None, "retest_path": None},
    }


def main() -> None:
    config = _config()
    assert validate_delivery_config(config) == []
    handoff = {
        "schema_version": "inference-deployment-handoff/v1",
        "semantic_id": "INF-SINGLE-GPU-DEPLOY",
        "run_label": "handoff-v1",
        "workload": {},
        "candidate_records": [
            {"group": "G2", "handoff_status": "ready_for_integrated_evaluation"}
        ],
        "decision": "accept",
        "rollback": {"ready": True},
    }
    checked = validate_deployment_handoff(handoff)
    assert checked["rollback_ready"] is True
    assert checked["ready_candidates"][0]["group"] == "G2"

    baseline = _result(tpot=18.0, ttft=80.0, throughput=12.0, peak_mb=9 * 1024)
    candidate = _result(tpot=12.0, ttft=70.0, throughput=18.0, peak_mb=10 * 1024)
    accepted = evaluate_service_candidate(
        candidate,
        baseline,
        config["gates"],
        quality_score=0.90,
        baseline_quality_score=0.90,
        artifact_verified=True,
        rollback_ready=True,
    )
    assert accepted["decision"] == "accept"
    missing_memory = _result(tpot=12.0, ttft=70.0, throughput=18.0, peak_mb=10 * 1024)
    missing_memory["metrics"]["peak_memory_mb"] = None
    assert evaluate_service_candidate(
        missing_memory,
        baseline,
        config["gates"],
        quality_score=0.90,
        baseline_quality_score=0.90,
        artifact_verified=True,
        rollback_ready=True,
    )["decision"] == "tune"
    assert evaluate_service_candidate(
        candidate,
        baseline,
        config["gates"],
        quality_score=0.90,
        baseline_quality_score=0.90,
        artifact_verified=False,
        rollback_ready=True,
    )["decision"] == "reject"
    selection = select_service_candidate(
        {"baseline": accepted, "candidate": accepted},
        {"baseline": baseline, "candidate": candidate},
    )
    assert selection["selected_candidate"] == "candidate"
    print("inference delivery runtime: ok")


if __name__ == "__main__":
    main()
