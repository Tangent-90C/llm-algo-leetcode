"""Shared fact collection for post-training project notebooks.

The helpers in this module collect dataset and artifact facts.  They do not
choose thresholds or return accept/tune/reject decisions; those project rules
remain visible in the notebooks.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def normalized_text(value: object) -> str:
    """Normalize one text field for deterministic identity checks."""

    return " ".join(str(value or "").strip().split())


def stable_text_hash(parts: Iterable[object]) -> str:
    """Return a stable SHA-256 identity for normalized text fields."""

    payload = "\u241f".join(normalized_text(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit_sft_records(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_length: int,
    train_count: int,
) -> dict[str, Any]:
    """Collect SFT admission facts without applying project thresholds.

    Records use ``instruction``, ``input`` and ``output`` fields.  The stable
    input order defines the train/evaluation split so leakage checks are fully
    reproducible from the returned hashes and configuration.
    """

    if max_length < 8:
        raise ValueError("max_length 必须至少为 8")
    if train_count <= 0 or train_count >= len(records):
        raise ValueError("train_count 必须位于 0 与样本总数之间")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer 必须提供 eos_token_id")

    pair_hashes: list[str] = []
    prompt_hashes: list[str] = []
    supervised_tokens: list[int] = []
    empty_instruction_count = 0
    empty_response_count = 0
    truncated_count = 0
    zero_supervision_count = 0

    for record in records:
        instruction = normalized_text(record.get("instruction"))
        context = normalized_text(record.get("input"))
        response = normalized_text(record.get("output"))
        empty_instruction_count += int(not instruction)
        empty_response_count += int(not response)

        prompt = f"### Instruction:\n{instruction}\n"
        if context:
            prompt += f"### Input:\n{context}\n"
        prompt += "### Response:\n"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        response_ids = response_ids + [tokenizer.eos_token_id]
        available = max(max_length - len(prompt_ids), 0)
        kept_response_tokens = min(len(response_ids), available)
        total_tokens = len(prompt_ids) + len(response_ids)

        pair_hashes.append(stable_text_hash((instruction, context, response)))
        prompt_hashes.append(stable_text_hash((instruction, context)))
        supervised_tokens.append(kept_response_tokens)
        truncated_count += int(total_tokens > max_length)
        zero_supervision_count += int(kept_response_tokens == 0)

    train_prompt_hashes = set(prompt_hashes[:train_count])
    evaluation_prompt_hashes = set(prompt_hashes[train_count:])
    duplicate_count = len(pair_hashes) - len(set(pair_hashes))
    split_leakage_count = len(train_prompt_hashes & evaluation_prompt_hashes)
    sample_count = len(records)

    return {
        "sample_count": sample_count,
        "train_count": train_count,
        "evaluation_count": sample_count - train_count,
        "empty_instruction_count": empty_instruction_count,
        "empty_response_count": empty_response_count,
        "duplicate_count": duplicate_count,
        "duplicate_ratio": duplicate_count / sample_count if sample_count else 0.0,
        "truncated_count": truncated_count,
        "truncation_ratio": truncated_count / sample_count if sample_count else 0.0,
        "zero_supervision_count": zero_supervision_count,
        "split_leakage_count": split_leakage_count,
        "minimum_supervised_tokens": min(supervised_tokens, default=0),
        "mean_supervised_tokens": (
            sum(supervised_tokens) / len(supervised_tokens) if supervised_tokens else 0.0
        ),
        "sample_ids_hash": stable_text_hash(pair_hashes),
        "train_prompt_ids_hash": stable_text_hash(prompt_hashes[:train_count]),
        "evaluation_prompt_ids_hash": stable_text_hash(prompt_hashes[train_count:]),
    }


def save_json_record(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Save one JSON object without silently replacing an existing run."""

    output = Path(path)
    if output.exists():
        raise FileExistsError(f"结果已存在，请更换 RUN_LABEL：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def build_causal_sft_examples(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_length: int,
) -> list[dict[str, list[int]]]:
    """Build fixed-length causal-LM examples with prompt labels masked."""

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer 必须提供 pad_token_id 或 eos_token_id")

    examples: list[dict[str, list[int]]] = []
    for record in records:
        instruction = normalized_text(record.get("instruction"))
        context = normalized_text(record.get("input"))
        response = normalized_text(record.get("output"))
        prompt = f"### Instruction:\n{instruction}\n"
        if context:
            prompt += f"### Input:\n{context}\n"
        prompt += "### Response:\n"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        response_ids = response_ids + [tokenizer.eos_token_id]
        input_ids = (prompt_ids + response_ids)[:max_length]
        supervised = max(len(input_ids) - min(len(prompt_ids), max_length), 0)
        if supervised == 0:
            raise ValueError("样本在当前 max_length 下没有有效监督 Token")
        labels = [-100] * (len(input_ids) - supervised) + input_ids[-supervised:]
        attention_mask = [1] * len(input_ids)
        padding = max_length - len(input_ids)
        examples.append(
            {
                "input_ids": input_ids + [pad_token_id] * padding,
                "attention_mask": attention_mask + [0] * padding,
                "labels": labels + [-100] * padding,
            }
        )
    return examples


def run_lora_candidate(
    *,
    model_id: str,
    revision: str,
    tokenizer: Any,
    train_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    max_length: int,
    update_steps: int,
    learning_rate: float,
    seed: int,
    dtype_name: str = "bfloat16",
    save_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Run one real LoRA training candidate and return project-level evidence.

    The function reloads the base model for every candidate so peak memory,
    initial loss and optimizer state are comparable.  Candidate selection stays
    in the notebook and is never hidden in this runtime helper.
    """

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("真实 LoRA 候选实验需要 CUDA")
    if update_steps <= 0:
        raise ValueError("update_steps 必须大于 0")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dtype = getattr(torch, dtype_name)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU 不支持 BF16，请改用 float16")

    train_examples = build_causal_sft_examples(
        train_records, tokenizer, max_length=max_length
    )
    evaluation_examples = build_causal_sft_examples(
        evaluation_records, tokenizer, max_length=max_length
    )
    device = torch.device("cuda")

    def tensor_batch(example: Mapping[str, Sequence[int]]) -> dict[str, Any]:
        return {
            key: torch.tensor(value, dtype=torch.long, device=device).unsqueeze(0)
            for key, value in example.items()
        }

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
    ).to(device)
    model.config.use_cache = False
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(candidate["rank"]),
        lora_alpha=int(candidate["alpha"]),
        lora_dropout=float(candidate["dropout"]),
        target_modules=list(candidate["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )

    def evaluate() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for example in evaluation_examples:
                losses.append(float(model(**tensor_batch(example)).loss.detach().float().item()))
        return statistics.fmean(losses)

    initial_eval_loss = evaluate()
    torch.cuda.reset_peak_memory_stats()
    step_times_ms: list[float] = []
    train_losses: list[float] = []
    supervised_tokens = 0
    model.train()
    for step in range(update_steps):
        example = train_examples[step % len(train_examples)]
        batch = tensor_batch(example)
        supervised_tokens += sum(token != -100 for token in example["labels"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        output.loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        step_times_ms.append((time.perf_counter() - started) * 1000.0)
        train_losses.append(float(output.loss.detach().float().item()))

    final_eval_loss = evaluate()
    peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
    peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
    elapsed_seconds = sum(step_times_ms) / 1000.0
    if save_directory is not None:
        output_dir = Path(save_directory)
        output_dir.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(output_dir)

    return {
        "name": str(candidate["name"]),
        "config": dict(candidate),
        "seed": seed,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / total_params,
        "initial_eval_loss": initial_eval_loss,
        "final_eval_loss": final_eval_loss,
        "eval_loss_improvement": initial_eval_loss - final_eval_loss,
        "initial_train_loss": train_losses[0],
        "final_train_loss": train_losses[-1],
        "median_step_time_ms": statistics.median(step_times_ms),
        "p95_step_time_ms": sorted(step_times_ms)[max(0, int(0.95 * len(step_times_ms)) - 1)],
        "supervised_tokens_per_second": supervised_tokens / elapsed_seconds,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "status": "ok",
    }


def hash_directory(directory: str | Path) -> str:
    """Hash relative names and bytes for every file in one artifact directory."""

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"artifact directory not found: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"artifact directory is empty: {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_lora_delivery(
    *,
    model_id: str,
    revision: str,
    tokenizer: Any,
    train_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
    lora_config: Mapping[str, Any],
    max_length: int,
    total_steps: int,
    checkpoint_step: int,
    learning_rate: float,
    seed: int,
    dtype_name: str,
    checkpoint_directory: str | Path,
    artifact_directory: str | Path,
) -> dict[str, Any]:
    """Train, checkpoint, resume, save and reload one selected LoRA adapter."""

    import gc

    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    if not 0 < checkpoint_step < total_steps:
        raise ValueError("checkpoint_step 必须位于 0 与 total_steps 之间")
    checkpoint_root = Path(checkpoint_directory)
    artifact_root = Path(artifact_directory)
    if checkpoint_root.exists() or artifact_root.exists():
        raise FileExistsError("checkpoint 或 artifact 目录已存在，请修改 RUN_LABEL")
    checkpoint_root.mkdir(parents=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dtype = getattr(torch, dtype_name)
    train_examples = build_causal_sft_examples(train_records, tokenizer, max_length=max_length)
    evaluation_examples = build_causal_sft_examples(
        evaluation_records, tokenizer, max_length=max_length
    )
    device = torch.device("cuda")

    def tensor_batch(example: Mapping[str, Sequence[int]]) -> dict[str, Any]:
        return {
            key: torch.tensor(value, dtype=torch.long, device=device).unsqueeze(0)
            for key, value in example.items()
        }

    def load_base():
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=dtype
        ).to(device)
        model.config.use_cache = False
        return model

    def evaluate(model: Any) -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for example in evaluation_examples:
                losses.append(float(model(**tensor_batch(example)).loss.detach().float().item()))
        return statistics.fmean(losses)

    base = load_base()
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_config["rank"]),
        lora_alpha=int(lora_config["alpha"]),
        lora_dropout=float(lora_config["dropout"]),
        target_modules=list(lora_config["target_modules"]),
        bias="none",
    )
    model = get_peft_model(base, peft_config)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    probe_parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
    initial_eval_loss = evaluate(model)
    torch.cuda.reset_peak_memory_stats()
    losses: list[float] = []
    times_ms: list[float] = []
    gradient_norms: list[float] = []
    parameter_delta_norms: list[float] = []
    learning_rates: list[float] = []
    finite_flags: list[bool] = []

    def train_range(start: int, stop: int) -> None:
        model.train()
        for step in range(start, stop):
            batch = tensor_batch(train_examples[step % len(train_examples)])
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            output.loss.backward()
            gradient_norm = 0.0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    gradient_norm += float(parameter.grad.detach().float().pow(2).sum().item())
            gradient_norm = gradient_norm**0.5
            probe_before = probe_parameter.detach().float().reshape(-1)[:4096].cpu().clone()
            learning_rates.append(float(optimizer.param_groups[0]["lr"]))
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - started) * 1000.0)
            losses.append(float(output.loss.detach().float().item()))
            gradient_norms.append(gradient_norm)
            parameter_delta_norms.append(
                float(
                    (probe_parameter.detach().float().reshape(-1)[:4096].cpu() - probe_before)
                    .pow(2)
                    .sum()
                    .sqrt()
                    .item()
                )
            )
            finite_flags.append(
                bool(torch.isfinite(output.loss).item()) and torch.isfinite(torch.tensor(gradient_norm)).item()
            )

    train_range(0, checkpoint_step)
    checkpoint_adapter = checkpoint_root / "adapter"
    model.save_pretrained(checkpoint_adapter)
    torch.save(optimizer.state_dict(), checkpoint_root / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_root / "scheduler.pt")
    torch.save(
        {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)},
        checkpoint_root / "rng.pt",
    )
    (checkpoint_root / "progress.json").write_text(
        json.dumps({"global_step": checkpoint_step, "total_steps": total_steps}, indent=2) + "\n",
        encoding="utf-8",
    )
    checkpoint_hash = hash_directory(checkpoint_root)

    del scheduler, optimizer, model, base
    gc.collect()
    torch.cuda.empty_cache()

    base = load_base()
    model = PeftModel.from_pretrained(base, checkpoint_adapter, is_trainable=True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    optimizer.load_state_dict(
        torch.load(checkpoint_root / "optimizer.pt", map_location=device, weights_only=True)
    )
    scheduler.load_state_dict(torch.load(checkpoint_root / "scheduler.pt", weights_only=True))
    rng_state = torch.load(checkpoint_root / "rng.pt", map_location="cpu", weights_only=True)
    torch.set_rng_state(rng_state["cpu"])
    torch.cuda.set_rng_state(rng_state["cuda"], device)
    probe_parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
    resumed_from_step = json.loads(
        (checkpoint_root / "progress.json").read_text(encoding="utf-8")
    )["global_step"]
    train_range(resumed_from_step, total_steps)
    final_eval_loss = evaluate(model)
    model.save_pretrained(artifact_root)
    tokenizer.save_pretrained(artifact_root)
    artifact_hash = hash_directory(artifact_root)
    peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
    peak_reserved_bytes = int(torch.cuda.max_memory_reserved())

    del scheduler, optimizer, model, base
    gc.collect()
    torch.cuda.empty_cache()
    reloaded_base = load_base()
    reloaded = PeftModel.from_pretrained(reloaded_base, artifact_root)
    reloaded_eval_loss = evaluate(reloaded)

    return {
        "initial_eval_loss": initial_eval_loss,
        "final_eval_loss": final_eval_loss,
        "reloaded_eval_loss": reloaded_eval_loss,
        "reload_loss_abs_diff": abs(reloaded_eval_loss - final_eval_loss),
        "initial_train_loss": losses[0],
        "final_train_loss": losses[-1],
        "median_step_time_ms": statistics.median(times_ms),
        "update_dynamics": {
            "finite_all": all(finite_flags),
            "gradient_norm_median": statistics.median(gradient_norms),
            "parameter_delta_norm_median": statistics.median(parameter_delta_norms),
            "learning_rate_start": learning_rates[0],
            "learning_rate_end": learning_rates[-1],
        },
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "checkpoint": {
            "path": str(checkpoint_root),
            "sha256": checkpoint_hash,
            "saved_step": checkpoint_step,
            "resumed_from_step": resumed_from_step,
            "completed_step": total_steps,
            "training_update_count": len(losses),
            "optimizer_state_present": True,
            "scheduler_state_present": True,
            "rng_state_present": True,
        },
        "artifact": {"path": str(artifact_root), "sha256": artifact_hash},
        "status": "ok",
    }


def preference_record_to_text(record: Mapping[str, Any]) -> tuple[str, str, str]:
    """Extract one shared prompt and two responses from a chat preference pair."""

    chosen = record.get("chosen")
    rejected = record.get("rejected")
    if not isinstance(chosen, Sequence) or not isinstance(rejected, Sequence):
        raise ValueError("chosen/rejected 必须是消息列表")
    if len(chosen) < 2 or len(rejected) < 2:
        raise ValueError("偏好对必须包含 prompt 与 response")
    chosen_prompt = normalized_text(chosen[0].get("content"))
    rejected_prompt = normalized_text(rejected[0].get("content"))
    if chosen_prompt != rejected_prompt:
        raise ValueError("chosen/rejected 的 prompt 不一致")
    chosen_response = normalized_text(chosen[-1].get("content"))
    rejected_response = normalized_text(rejected[-1].get("content"))
    if not chosen_prompt or not chosen_response or not rejected_response:
        raise ValueError("偏好对存在空字段")
    return chosen_prompt, chosen_response, rejected_response


def audit_preference_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collect pair validity, identity and duplicate facts without thresholds."""

    valid_pairs: list[tuple[str, str, str]] = []
    invalid_count = 0
    identical_response_count = 0
    for record in records:
        try:
            pair = preference_record_to_text(record)
        except (TypeError, ValueError, AttributeError):
            invalid_count += 1
            continue
        identical_response_count += int(pair[1] == pair[2])
        valid_pairs.append(pair)
    pair_hashes = [stable_text_hash(pair) for pair in valid_pairs]
    prompt_hashes = [stable_text_hash((pair[0],)) for pair in valid_pairs]
    return {
        "sample_count": len(records),
        "valid_count": len(valid_pairs),
        "valid_ratio": len(valid_pairs) / len(records) if records else 0.0,
        "invalid_count": invalid_count,
        "identical_response_count": identical_response_count,
        "duplicate_count": len(pair_hashes) - len(set(pair_hashes)),
        "pair_ids_hash": stable_text_hash(pair_hashes),
        "prompt_ids_hash": stable_text_hash(prompt_hashes),
    }


def build_preference_pair_sequences(
    tokenizer: Any,
    prompt: str,
    chosen: str,
    rejected: str,
    *,
    max_length: int,
) -> dict[str, Any]:
    """Encode one preference pair while preserving a shared prompt boundary.

    Long prompts are truncated from the left.  Chosen and rejected use the
    same retained prompt, and each side keeps at least one response token.
    """

    if max_length < 2:
        raise ValueError("max_length 必须至少为 2")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer 必须提供 eos_token_id")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    chosen_ids = tokenizer(chosen, add_special_tokens=False)["input_ids"] + [
        tokenizer.eos_token_id
    ]
    rejected_ids = tokenizer(rejected, add_special_tokens=False)["input_ids"] + [
        tokenizer.eos_token_id
    ]
    if not prompt_ids or not chosen_ids or not rejected_ids:
        raise ValueError("偏好样本的 prompt、chosen 和 rejected 均必须产生 Token")

    maximum_response_tokens = max_length - 1
    chosen_kept = chosen_ids[:maximum_response_tokens]
    rejected_kept = rejected_ids[:maximum_response_tokens]
    shared_prompt_budget = max_length - max(len(chosen_kept), len(rejected_kept))
    shared_prompt = prompt_ids[-shared_prompt_budget:]
    if not shared_prompt:
        raise ValueError("偏好样本没有保留 prompt Token")

    def sequence(response_ids: Sequence[int], kept: Sequence[int]) -> dict[str, Any]:
        input_ids = list(shared_prompt) + list(kept)
        return {
            "input_ids": input_ids,
            "response_start": len(shared_prompt),
            "response_tokens_total": len(response_ids),
            "response_tokens_retained": len(kept),
            "response_truncated": len(kept) < len(response_ids),
        }

    return {
        "chosen": sequence(chosen_ids, chosen_kept),
        "rejected": sequence(rejected_ids, rejected_kept),
        "prompt_tokens_total": len(prompt_ids),
        "prompt_tokens_retained": len(shared_prompt),
        "prompt_truncated": len(shared_prompt) < len(prompt_ids),
    }


def run_dpo_delivery(
    *,
    model_id: str,
    revision: str,
    sft_adapter_path: str | Path,
    tokenizer: Any,
    train_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
    max_length: int,
    update_steps: int,
    beta: float,
    learning_rate: float,
    seed: int,
    dtype_name: str,
    artifact_directory: str | Path,
) -> dict[str, Any]:
    """Run a fixed DPO candidate against a frozen reference and save its adapter."""

    import gc

    import torch
    import torch.nn.functional as functional
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("真实 DPO 项目需要 CUDA")
    if beta <= 0 or update_steps <= 0:
        raise ValueError("beta 和 update_steps 必须大于 0")
    artifact_root = Path(artifact_directory)
    if artifact_root.exists():
        raise FileExistsError(f"artifact 已存在：{artifact_root}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dtype = getattr(torch, dtype_name)
    device = torch.device("cuda")

    def load_policy(trainable: bool):
        base = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=dtype
        ).to(device)
        base.config.use_cache = False
        return PeftModel.from_pretrained(base, sft_adapter_path, is_trainable=trainable)

    policy = load_policy(True)
    reference = load_policy(False)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )

    def response_logp(model: Any, sequence: Mapping[str, Any], *, grad: bool) -> Any:
        input_ids = sequence["input_ids"]
        response_start = int(sequence["response_start"])
        if not 0 < response_start < len(input_ids):
            raise ValueError("偏好序列必须同时保留 prompt 与 response Token")
        tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        mask = torch.ones_like(tensor)
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            logits = model(input_ids=tensor, attention_mask=mask).logits[:, :-1]
            targets = tensor[:, 1:]
            token_logps = functional.log_softmax(logits.float(), dim=-1).gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)
            start = max(response_start - 1, 0)
            return token_logps[:, start:].mean()

    def encode_record(record: Mapping[str, Any]) -> dict[str, Any]:
        prompt, chosen, rejected = preference_record_to_text(record)
        return build_preference_pair_sequences(
            tokenizer, prompt, chosen, rejected, max_length=max_length
        )

    train_pairs = [encode_record(record) for record in train_records]
    evaluation_pairs = [encode_record(record) for record in evaluation_records]

    def evaluate(model: Any) -> dict[str, float]:
        margins = []
        for pair in evaluation_pairs:
            chosen_logp = response_logp(model, pair["chosen"], grad=False)
            rejected_logp = response_logp(model, pair["rejected"], grad=False)
            margins.append(float((chosen_logp - rejected_logp).item()))
        return {
            "preference_accuracy": sum(margin > 0 for margin in margins) / len(margins),
            "margin_mean": statistics.fmean(margins),
            "margin_min": min(margins),
        }

    baseline = evaluate(policy)
    losses: list[float] = []
    times_ms: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    policy.train()
    for step in range(update_steps):
        pair = train_pairs[step % len(train_pairs)]
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        policy_chosen = response_logp(policy, pair["chosen"], grad=True)
        policy_rejected = response_logp(policy, pair["rejected"], grad=True)
        reference_chosen = response_logp(reference, pair["chosen"], grad=False)
        reference_rejected = response_logp(reference, pair["rejected"], grad=False)
        logits = beta * (
            (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
        )
        loss = -functional.logsigmoid(logits)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        losses.append(float(loss.detach().item()))
        times_ms.append((time.perf_counter() - started) * 1000.0)

    candidate = evaluate(policy)
    policy.save_pretrained(artifact_root)
    tokenizer.save_pretrained(artifact_root)
    artifact_hash = hash_directory(artifact_root)
    result = {
        "baseline": baseline,
        "candidate": candidate,
        "comparison": {
            "preference_accuracy_gain": candidate["preference_accuracy"] - baseline["preference_accuracy"],
            "margin_mean_gain": candidate["margin_mean"] - baseline["margin_mean"],
            "margin_floor_gain": candidate["margin_min"] - baseline["margin_min"],
        },
        "training": {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "median_step_time_ms": statistics.median(times_ms),
            "update_steps": update_steps,
        },
        "sequence_contract": {
            "max_length": max_length,
            "train_pair_count": len(train_pairs),
            "evaluation_pair_count": len(evaluation_pairs),
            "train_prompt_truncated_count": sum(pair["prompt_truncated"] for pair in train_pairs),
            "evaluation_prompt_truncated_count": sum(pair["prompt_truncated"] for pair in evaluation_pairs),
            "train_response_truncated_count": sum(
                pair[side]["response_truncated"]
                for pair in train_pairs for side in ("chosen", "rejected")
            ),
            "evaluation_response_truncated_count": sum(
                pair[side]["response_truncated"]
                for pair in evaluation_pairs for side in ("chosen", "rejected")
            ),
            "prompt_truncation_ratio": (
                sum(pair["prompt_truncated"] for pair in train_pairs + evaluation_pairs)
                / (len(train_pairs) + len(evaluation_pairs))
            ),
            "response_truncation_ratio": (
                sum(
                    pair[side]["response_truncated"]
                    for pair in train_pairs + evaluation_pairs
                    for side in ("chosen", "rejected")
                )
                / (2 * (len(train_pairs) + len(evaluation_pairs)))
            ),
        },
        "resources": {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "artifact": {"path": str(artifact_root), "sha256": artifact_hash},
        "status": "ok",
    }
    del optimizer, policy, reference
    gc.collect()
    torch.cuda.empty_cache()
    return result


def extract_final_number(text: str) -> str | None:
    """Extract the final normalized numeric answer from generated text."""

    matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", str(text))
    return matches[-1].replace(",", "") if matches else None


def run_grpo_delivery(
    *,
    model_id: str,
    revision: str,
    sft_adapter_path: str | Path,
    tokenizer: Any,
    train_tasks: Sequence[Mapping[str, Any]],
    evaluation_tasks: Sequence[Mapping[str, Any]],
    group_size: int,
    max_prompt_length: int,
    max_new_tokens: int,
    clip_epsilon: float,
    kl_coefficient: float,
    learning_rate: float,
    seed: int,
    dtype_name: str,
    artifact_directory: str | Path,
    source_policy_version: str,
    verifier_version: str,
) -> dict[str, Any]:
    """Run real grouped rollouts, verifier rewards and finite GRPO updates."""

    import gc

    import torch
    import torch.nn.functional as functional
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if group_size < 2:
        raise ValueError("group_size 必须至少为 2")
    if not torch.cuda.is_available():
        raise RuntimeError("真实 GRPO 项目需要 CUDA")
    artifact_root = Path(artifact_directory)
    if artifact_root.exists():
        raise FileExistsError(f"artifact 已存在：{artifact_root}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dtype = getattr(torch, dtype_name)
    device = torch.device("cuda")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def load_policy(trainable: bool):
        base = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=dtype
        ).to(device)
        base.config.use_cache = True
        return PeftModel.from_pretrained(base, sft_adapter_path, is_trainable=trainable)

    policy = load_policy(True)
    reference = load_policy(False)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )

    def gold_answer(task: Mapping[str, Any]) -> str | None:
        return extract_final_number(str(task["answer"]).split("####")[-1])

    def generate_group(model: Any, prompt: str, generation_seed: int) -> tuple[list[str], list[int]]:
        torch.manual_seed(generation_seed)
        torch.cuda.manual_seed_all(generation_seed)
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
        ).to(device)
        model.eval()
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                num_return_sequences=group_size,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_tokens = encoded["input_ids"].shape[1]
        generated_tokens = generated[:, prompt_tokens:]
        responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        response_lengths = []
        for row in generated_tokens:
            token_ids = row.tolist()
            if tokenizer.eos_token_id in token_ids:
                response_lengths.append(token_ids.index(tokenizer.eos_token_id) + 1)
            else:
                response_lengths.append(len(token_ids))
        return responses, response_lengths

    def response_logp(model: Any, prompt: str, response: str, *, grad: bool) -> Any:
        prompt_ids = tokenizer(
            prompt, add_special_tokens=False, truncation=True, max_length=max_prompt_length
        )["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"] + [
            tokenizer.eos_token_id
        ]
        input_ids = torch.tensor(
            prompt_ids + response_ids, dtype=torch.long, device=device
        ).unsqueeze(0)
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            logits = model(input_ids=input_ids).logits[:, :-1]
            targets = input_ids[:, 1:]
            token_logps = functional.log_softmax(logits.float(), dim=-1).gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)
            return token_logps[:, max(len(prompt_ids) - 1, 0) :].mean()

    def rollout_summary(model: Any, tasks: Sequence[Mapping[str, Any]], seed_offset: int):
        rewards: list[float] = []
        pass_at_group: list[float] = []
        response_lengths: list[int] = []
        for index, task in enumerate(tasks):
            responses, lengths = generate_group(
                model, str(task["question"]), seed + seed_offset + index
            )
            gold = gold_answer(task)
            group_rewards = [float(extract_final_number(response) == gold) for response in responses]
            rewards.extend(group_rewards)
            response_lengths.extend(lengths)
            pass_at_group.append(float(any(group_rewards)))
        return {
            "reward_mean": statistics.fmean(rewards) if rewards else 0.0,
            "pass_at_group": statistics.fmean(pass_at_group) if pass_at_group else 0.0,
            "response_count": len(rewards),
            "response_length_mean": (
                statistics.fmean(response_lengths) if response_lengths else 0.0
            ),
            "response_length_p95": (
                sorted(response_lengths)[
                    min(len(response_lengths) - 1, int(0.95 * len(response_lengths)))
                ]
                if response_lengths
                else 0
            ),
            "max_length_hit_ratio": (
                sum(length >= max_new_tokens for length in response_lengths)
                / len(response_lengths)
                if response_lengths
                else 0.0
            ),
        }

    evaluation_seed_offset = 10_000
    baseline = rollout_summary(policy, evaluation_tasks, evaluation_seed_offset)
    valid_groups = 0
    total_groups = 0
    update_losses: list[float] = []
    rollout_seconds = 0.0
    update_seconds = 0.0
    trajectory_records: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()

    for task_index, task in enumerate(train_tasks):
        total_groups += 1
        prompt = str(task["question"])
        started = time.perf_counter()
        responses, response_lengths = generate_group(policy, prompt, seed + task_index)
        rollout_seconds += time.perf_counter() - started
        gold = gold_answer(task)
        rewards = [float(extract_final_number(response) == gold) for response in responses]
        reward_mean = statistics.fmean(rewards)
        reward_std = statistics.pstdev(rewards)
        prompt_id = stable_text_hash((prompt,))
        trace_ids = [
            stable_text_hash(
                (
                    source_policy_version,
                    verifier_version,
                    prompt_id,
                    task_index,
                    sample_index,
                    seed,
                )
            )
            for sample_index in range(group_size)
        ]
        trajectory_records.append(
            {
                "prompt_id": prompt_id,
                "trace_ids": trace_ids,
                "source_policy_version": source_policy_version,
                "verifier_version": verifier_version,
                "rewards": rewards,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "response_length_mean": statistics.fmean(response_lengths),
                "max_length_hit_ratio": sum(
                    length >= max_new_tokens for length in response_lengths
                )
                / len(response_lengths),
            }
        )
        if reward_std == 0.0:
            continue
        valid_groups += 1
        advantages = [(reward - reward_mean) / reward_std for reward in rewards]
        old_logps = [
            response_logp(policy, prompt, response, grad=False).detach()
            for response in responses
        ]
        reference_logps = [
            response_logp(reference, prompt, response, grad=False).detach()
            for response in responses
        ]
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        terms = []
        started = time.perf_counter()
        for response, advantage, old_logp, reference_logp in zip(
            responses, advantages, old_logps, reference_logps
        ):
            current_logp = response_logp(policy, prompt, response, grad=True)
            ratio = torch.exp(current_logp - old_logp)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
            policy_term = -torch.minimum(ratio * advantage, clipped_ratio * advantage)
            kl_proxy = (current_logp - reference_logp).pow(2)
            terms.append(policy_term + kl_coefficient * kl_proxy)
        loss = torch.stack(terms).mean()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        update_seconds += time.perf_counter() - started
        update_losses.append(float(loss.detach().item()))

    candidate = rollout_summary(policy, evaluation_tasks, evaluation_seed_offset)
    policy.save_pretrained(artifact_root)
    tokenizer.save_pretrained(artifact_root)
    artifact_hash = hash_directory(artifact_root)
    valid_trajectory_count = valid_groups * group_size
    result = {
        "baseline": baseline,
        "candidate": candidate,
        "comparison": {
            "reward_mean_gain": candidate["reward_mean"] - baseline["reward_mean"],
            "pass_at_group_gain": candidate["pass_at_group"] - baseline["pass_at_group"],
            "paired_evaluation_seed_offset": evaluation_seed_offset,
        },
        "rollout": {
            "group_size": group_size,
            "group_count": total_groups,
            "valid_group_count": valid_groups,
            "valid_group_ratio": valid_groups / total_groups if total_groups else 0.0,
            "trajectory_records": trajectory_records,
            "rollout_seconds": rollout_seconds,
        },
        "update": {
            "update_count": len(update_losses),
            "initial_loss": update_losses[0] if update_losses else None,
            "final_loss": update_losses[-1] if update_losses else None,
            "update_seconds": update_seconds,
            "effective_trajectories_per_second": (
                valid_trajectory_count / (rollout_seconds + update_seconds)
                if rollout_seconds + update_seconds > 0
                else 0.0
            ),
        },
        "resources": {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "artifact": {"path": str(artifact_root), "sha256": artifact_hash},
        "status": "ok" if valid_groups else "no_effective_update",
    }
    del optimizer, policy, reference
    gc.collect()
    torch.cuda.empty_cache()
    return result
