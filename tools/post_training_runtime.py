"""Shared runtime helpers for post-training mechanism notebooks.

The helpers in this module own tokenization and sequence-score bookkeeping.
They deliberately do not choose preference pairs, rewards, beta values, update
rules, thresholds, or decisions; those remain visible in each notebook.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def build_response_batch(
    tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    *,
    max_length: int,
) -> dict[str, torch.Tensor]:
    """Tokenize prompt/response pairs and mark tokens scored as responses."""

    if len(prompts) != len(responses) or not prompts:
        raise ValueError("prompts 与 responses 必须非空且长度一致")
    if max_length < 8:
        raise ValueError("max_length 过小，无法容纳稳定的回复边界")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows: list[list[int]] = []
    masks: list[list[bool]] = []
    for prompt, response in zip(prompts, responses):
        prompt_ids = tokenizer(str(prompt), add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(str(response), add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            response_ids = response_ids + [tokenizer.eos_token_id]
        available = max_length - len(prompt_ids)
        if available <= 0:
            raise ValueError("prompt 已占满 max_length，没有可评分的回复 token")
        response_ids = response_ids[:available]
        if not response_ids:
            raise ValueError("每条样本至少需要一个回复 token")
        token_ids = prompt_ids + response_ids
        response_mask = [False] * len(prompt_ids) + [True] * len(response_ids)
        padding = max_length - len(token_ids)
        rows.append(token_ids + [tokenizer.pad_token_id] * padding)
        masks.append(response_mask + [False] * padding)

    input_ids = torch.tensor(rows, dtype=torch.long)
    response_mask = torch.tensor(masks, dtype=torch.bool)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
    }


def move_response_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move a response-scoring batch to one device."""

    return {key: value.to(device) for key, value in batch.items()}


def response_sequence_logps(
    model,
    batch: dict[str, torch.Tensor],
    *,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return response-only sequence log-probabilities and scored-token counts."""

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    shifted_targets = batch["input_ids"][:, 1:]
    shifted_mask = batch["response_mask"][:, 1:]
    token_logps = F.log_softmax(outputs.logits[:, :-1].float(), dim=-1)
    selected = token_logps.gather(-1, shifted_targets.unsqueeze(-1)).squeeze(-1)
    counts = shifted_mask.sum(dim=-1)
    if torch.any(counts == 0):
        raise ValueError("每条样本至少需要一个可评分的回复 token")
    sequence_logps = (selected * shifted_mask).sum(dim=-1)
    if normalize:
        sequence_logps = sequence_logps / counts
    return sequence_logps, counts


def build_token_rewards(
    task_reward: float,
    actor_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    beta_kl: float,
    terminal_index: int = -1,
) -> torch.Tensor:
    """Combine per-token KL shaping with one terminal task reward."""

    if actor_log_probs.ndim != 1 or actor_log_probs.shape != reference_log_probs.shape:
        raise ValueError("actor/reference log probabilities must be same-shape 1D tensors")
    if beta_kl < 0:
        raise ValueError("beta_kl must be non-negative")
    if not -actor_log_probs.numel() <= terminal_index < actor_log_probs.numel():
        raise IndexError("terminal_index 超出 Token 序列范围")
    token_rewards = -beta_kl * (actor_log_probs - reference_log_probs)
    token_rewards = token_rewards.clone()
    token_rewards[terminal_index] += task_reward
    return token_rewards


def discounted_returns(token_rewards: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Accumulate discounted rewards from the end of one token trajectory."""

    if token_rewards.ndim != 1:
        raise ValueError("token_rewards must be a 1D tensor")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    returns = torch.empty_like(token_rewards)
    running_return = torch.zeros((), dtype=token_rewards.dtype, device=token_rewards.device)
    for index in range(token_rewards.numel() - 1, -1, -1):
        running_return = token_rewards[index] + gamma * running_return
        returns[index] = running_return
    return returns


def compute_advantages(
    returns: torch.Tensor,
    values: torch.Tensor,
    *,
    normalize: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Subtract a value baseline and optionally normalize one advantage vector."""

    if returns.shape != values.shape:
        raise ValueError("returns and values must have the same shape")
    advantages = returns - values
    if normalize:
        std = advantages.std(unbiased=False)
        if std > eps:
            advantages = (advantages - advantages.mean()) / (std + eps)
        else:
            advantages = advantages - advantages.mean()
    return advantages
