"""Shared runtime helpers for real-model training mechanism notebooks."""

from __future__ import annotations

import statistics
from typing import Iterable

import torch

from tools.notebook_runtime import resolve_model_commit


def prepare_causal_lm_batch(
    model_id: str,
    revision: str,
    prompts: Iterable[str],
    batch_size: int,
    seq_len: int,
    *,
    local_files_only: bool = False,
):
    """Resolve one immutable snapshot and build a deterministic CPU token batch."""
    from transformers import AutoTokenizer

    commit, resolution_source = resolve_model_commit(
        model_id,
        revision,
        local_files_only=local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=commit,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_list = tuple(prompts)
    if not prompt_list:
        raise ValueError("prompts 不能为空")
    texts = [prompt_list[index % len(prompt_list)] for index in range(batch_size)]
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
        "resolved_model_commit": commit,
        "revision_resolution_source": resolution_source,
        "tokenizer_class": type(tokenizer).__name__,
    }


def load_causal_lm(
    model_id: str,
    resolved_revision: str,
    device: torch.device,
    *,
    parameter_dtype: torch.dtype = torch.float32,
    local_files_only: bool = False,
):
    """Load a causal LM from the same immutable snapshot for each candidate."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=resolved_revision,
        torch_dtype=parameter_dtype,
        local_files_only=local_files_only,
    )
    model.config.use_cache = False
    return model.to(device)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device):
    """Move only tensor fields to the target device."""
    return {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}


def gradient_l2_norm(parameters) -> float:
    """Return the global FP32 L2 norm of all finite parameter gradients."""
    squared = None
    for parameter in parameters:
        if parameter.grad is not None:
            term = parameter.grad.detach().float().pow(2).sum()
            squared = term if squared is None else squared + term
    return 0.0 if squared is None else float(squared.sqrt())


def median_and_p95(values: list[float]) -> tuple[float, float]:
    """Summarize a non-empty timing sample."""
    if not values:
        raise ValueError("timing samples 不能为空")
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, (95 * len(ordered) + 99) // 100 - 1))
    return float(statistics.median(ordered)), float(ordered[p95_index])


def logit_agreement(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    """Summarize output agreement without letting one BF16 outlier dominate."""
    if reference.shape != candidate.shape:
        raise ValueError(f"logit shape mismatch: {reference.shape} != {candidate.shape}")
    reference_fp32 = reference.detach().float()
    candidate_fp32 = candidate.detach().float()
    absolute_difference = (reference_fp32 - candidate_fp32).abs()
    cosine_similarity = torch.nn.functional.cosine_similarity(
        reference_fp32.reshape(1, -1),
        candidate_fp32.reshape(1, -1),
    ).item()
    top1_agreement = (
        reference_fp32.argmax(dim=-1) == candidate_fp32.argmax(dim=-1)
    ).float().mean().item()
    return {
        "max_abs_diff": float(absolute_difference.max().item()),
        "mean_abs_diff": float(absolute_difference.mean().item()),
        "cosine_similarity": float(cosine_similarity),
        "top1_agreement": float(top1_agreement),
    }


def cuda_hardware_snapshot() -> dict[str, object]:
    """Record the minimum hardware identity required by result JSON."""
    return {
        "gpu": torch.cuda.get_device_name(0),
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }
