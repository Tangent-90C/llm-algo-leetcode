"""Dependency-light regression tests for shared quantization notebook helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tools.notebook_runtime import failure_record, hash_ids, package_version, save_json
from tools.quantization_runtime import (
    artifact_manifest,
    mean_prefix_agreement,
    token_prefix_agreement,
)


def main() -> None:
    assert package_version("a-distribution-that-does-not-exist") is None
    assert hash_ids([3, 1, 4]) == hash_ids([3, 1, 4])
    assert hash_ids([3, 1, 4]) != hash_ids([1, 3, 4])
    assert token_prefix_agreement([1, 2, 3], [1, 2, 4]) == 2 / 3
    assert mean_prefix_agreement([[1, 2], [3]], [[1, 9], [3]]) == 0.75

    try:
        raise ValueError("expected")
    except ValueError as exc:
        failure = failure_record("unit_test", exc)
    assert failure["stage"] == "unit_test"
    assert failure["type"] == "ValueError"
    assert failure["traceback_tail"]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "config.json").write_text(
            json.dumps({"quantization_config": {"bits": 4}}),
            encoding="utf-8",
        )
        (root / "model.safetensors").write_bytes(b"weights")
        manifest = artifact_manifest(root)
        assert manifest["has_config"] is True
        assert manifest["has_weight_file"] is True
        assert manifest["quantization_config"]["bits"] == 4

        output = save_json(root / "result.json", {"status": "ok"})
        assert json.loads(output.read_text(encoding="utf-8"))["status"] == "ok"

    print("quantization runtime smoke: ok")


if __name__ == "__main__":
    main()
