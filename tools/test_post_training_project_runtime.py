"""Focused tests for shared post-training project fact collection."""

from __future__ import annotations

import tempfile
from pathlib import Path

from tools.post_training_project_runtime import (
    audit_preference_records,
    audit_sft_records,
    build_causal_sft_examples,
    build_preference_pair_sequences,
    extract_final_number,
    hash_directory,
    save_json_record,
)


class _Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": list(range(len(text.split())))}


def main() -> None:
    records = [
        {"instruction": "solve one", "input": "", "output": "answer one"},
        {"instruction": "solve two", "input": "", "output": "answer two"},
        {"instruction": "solve one", "input": "", "output": "answer one"},
        {"instruction": "solve two", "input": "", "output": "new answer"},
    ]
    audit = audit_sft_records(records, _Tokenizer(), max_length=32, train_count=2)
    assert audit["duplicate_count"] == 1
    assert audit["split_leakage_count"] == 2
    assert audit["zero_supervision_count"] == 0
    assert audit["train_prompt_ids_hash"] == audit["evaluation_prompt_ids_hash"]
    examples = build_causal_sft_examples(records[:2], _Tokenizer(), max_length=32)
    assert len(examples) == 2
    assert all(len(example["input_ids"]) == 32 for example in examples)
    assert all(-100 in example["labels"] for example in examples)

    preference_audit = audit_preference_records(
        [
            {
                "chosen": [{"content": "question", "role": "user"}, {"content": "good", "role": "assistant"}],
                "rejected": [{"content": "question", "role": "user"}, {"content": "bad", "role": "assistant"}],
            },
            {
                "chosen": [{"content": "question", "role": "user"}, {"content": "same", "role": "assistant"}],
                "rejected": [{"content": "question", "role": "user"}, {"content": "same", "role": "assistant"}],
            },
        ]
    )
    assert preference_audit["valid_ratio"] == 1.0
    assert preference_audit["identical_response_count"] == 1
    long_pair = build_preference_pair_sequences(
        _Tokenizer(),
        " ".join(f"prompt-{index}" for index in range(40)),
        "chosen answer",
        "rejected answer with more tokens",
        max_length=12,
    )
    assert long_pair["prompt_truncated"]
    assert long_pair["chosen"]["response_start"] == long_pair["rejected"]["response_start"]
    assert 0 < long_pair["chosen"]["response_start"] < len(long_pair["chosen"]["input_ids"])
    assert 0 < long_pair["rejected"]["response_start"] < len(long_pair["rejected"]["input_ids"])
    assert len(long_pair["chosen"]["input_ids"]) <= 12
    assert len(long_pair["rejected"]["input_ids"]) <= 12
    assert extract_final_number("reasoning... #### 1,024") == "1024"

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "result.json"
        save_json_record(output, {"status": "ok"})
        assert output.is_file()
        artifact = Path(directory) / "artifact"
        artifact.mkdir()
        (artifact / "weights.bin").write_bytes(b"weights")
        first_hash = hash_directory(artifact)
        assert first_hash == hash_directory(artifact)
        try:
            save_json_record(output, {"status": "unexpected overwrite"})
        except FileExistsError:
            pass
        else:
            raise AssertionError("save_json_record must reject an existing path")

    print("post-training project runtime: ok")


if __name__ == "__main__":
    main()
