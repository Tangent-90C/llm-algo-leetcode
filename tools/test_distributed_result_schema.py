from __future__ import annotations

from tools.distributed_result_schema import make_record, validate_record
from tools.project_asset_relations import validate_asset_payload


def main() -> None:
    cases = [
        ("79", "training_state_parallel", "distributed.training_state"),
        ("87", "model_internal_parallel", "distributed.model_internal"),
        ("80", "moe_expert_parallel", "distributed.moe_dispatch"),
    ]
    for project, role, asset_id in cases:
        record = make_record(
            project=project,
            role=role,
            workload={"id": "fixed"},
            hardware={"world_size": 2},
            metrics={"throughput": 1.0},
            evidence_level="real_multi_gpu_benchmark",
            decision={"decision": "tune"},
        )
        assert validate_record(record) == []
        assert validate_asset_payload(asset_id, record) == []
        wrong_identity = {**record, "semantic_id": "PAR-DISTRIBUTED-INFERENCE"}
        assert any("semantic_id" in error for error in validate_record(wrong_identity))
        wrong_role = {**record, "role": "distributed_serving"}
        assert any("role must be" in error for error in validate_record(wrong_role))
    print("distributed result schema: ok")


if __name__ == "__main__":
    main()
