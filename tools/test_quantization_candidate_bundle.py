from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tools.quantization_runtime import (
    build_candidate_bundle,
    save_candidate_bundle,
    validate_candidate_bundle,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        artifact = root / "artifact"
        artifact.mkdir()
        (artifact / "config.json").write_text("{}", encoding="utf-8")
        (artifact / "model.safetensors").write_bytes(b"weights")
        (artifact / "tokenizer.json").write_text("{}", encoding="utf-8")
        result_path = root / "result.json"
        result_path.write_text("{}", encoding="utf-8")
        result = {
            "workload": {
                "model_id": "toy/model", "resolved_model_commit": "abc123",
                "scheme": "W4A16", "max_nll_ratio_vs_bf16": 1.1,
            },
            "dataset": {
                "dataset_id": "toy/data", "config": "raw", "revision": "d1",
                "calibration_split": "train", "evaluation_split": "validation",
                "calibration_samples": 8, "evaluation_samples": 8,
                "calibration_ids_hash": "cal-hash", "evaluation_ids_hash": "eval-hash",
            },
            "baseline": {"token_weighted_nll": 2.0},
            "candidates": {"gptq": {
                "token_weighted_nll": 2.1, "nll_ratio_vs_bf16": 1.05,
                "artifact_manifest": {"quantization_config": {
                    "format": "pack-quantized", "quant_method": "compressed-tensors", "version": "1",
                    "config_groups": {"group_0": {"weights": {"num_bits": 4, "group_size": 128}}},
                }},
            }},
            "evidence_level": "artifact_generation", "failure": None,
        }
        bundle = build_candidate_bundle(
            result, candidate_name="gptq", artifact_dir=artifact,
            project_root=root, source_result_path=result_path,
        )
        bundle_path = save_candidate_bundle(artifact / "candidate_bundle.json", bundle)
        loaded = json.loads(bundle_path.read_text(encoding="utf-8"))
        assert validate_candidate_bundle(loaded, project_root=root)["ready"]
        assert loaded["evaluation"]["direction"] == "lower_is_better"
        assert loaded["evaluation"]["passed"] is True
    print("quantization candidate bundle: ok")


if __name__ == "__main__":
    main()
