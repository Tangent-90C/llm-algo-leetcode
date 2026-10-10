"""Shared GPU runtime for speculative-decoding mechanism notebooks."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("P95 requires at least one value")
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _token_hash(token_ids: Sequence[int]) -> str:
    payload = ",".join(str(int(token_id)) for token_id in token_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _first_mismatch(reference: Sequence[int], candidate: Sequence[int]) -> int | None:
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if int(left) != int(right):
            return index
    return None if len(reference) == len(candidate) else min(len(reference), len(candidate))


def _output_diagnostics(
    reference_outputs: Sequence[Sequence[int]],
    candidate_outputs: Sequence[Sequence[int]],
    prompt_lengths: Sequence[int],
) -> list[dict[str, Any]]:
    diagnostics = []
    for prompt_index, (reference, candidate, prompt_length) in enumerate(
        zip(reference_outputs, candidate_outputs, prompt_lengths)
    ):
        absolute_mismatch = _first_mismatch(reference, candidate)
        generated_mismatch = (
            None if absolute_mismatch is None else max(0, absolute_mismatch - int(prompt_length))
        )
        diagnostics.append({
            "prompt_index": prompt_index,
            "match": list(reference) == list(candidate),
            "prompt_tokens": int(prompt_length),
            "reference_generated_tokens": max(0, len(reference) - int(prompt_length)),
            "candidate_generated_tokens": max(0, len(candidate) - int(prompt_length)),
            "first_mismatch_absolute_index": absolute_mismatch,
            "first_mismatch_generated_index": generated_mismatch,
            "reference_hash": _token_hash(reference),
            "candidate_hash": _token_hash(candidate),
            "reference_token_at_mismatch": (
                None if absolute_mismatch is None or absolute_mismatch >= len(reference)
                else int(reference[absolute_mismatch])
            ),
            "candidate_token_at_mismatch": (
                None if absolute_mismatch is None or absolute_mismatch >= len(candidate)
                else int(candidate[absolute_mismatch])
            ),
        })
    return diagnostics


def _target_margin_diagnostics(
    model: Any,
    reference_outputs: Sequence[Sequence[int]],
    output_diagnostics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Measure the target model's top-1/top-2 margin at each first mismatch."""
    import torch

    mantissa_bits = {
        torch.bfloat16: 7,
        torch.float16: 10,
        torch.float32: 23,
        torch.float64: 52,
    }

    def dtype_ulp(value: float, dtype: Any) -> float | None:
        bits = mantissa_bits.get(dtype)
        if bits is None or not math.isfinite(value) or value == 0.0:
            return None
        exponent = math.floor(math.log2(abs(value)))
        return math.ldexp(1.0, exponent - bits)

    rows = []
    for reference, diagnostic in zip(reference_outputs, output_diagnostics):
        mismatch = diagnostic.get("first_mismatch_absolute_index")
        if mismatch is None or int(mismatch) <= 0:
            rows.append({"prompt_index": diagnostic["prompt_index"], "available": False})
            continue
        prefix = torch.tensor([reference[: int(mismatch)]], dtype=torch.long, device=model.device)
        with torch.inference_mode():
            raw_logits = model(input_ids=prefix, use_cache=False).logits[0, -1]
        logit_dtype = raw_logits.dtype
        values, indices = torch.topk(raw_logits.float(), k=2)
        margin = float(values[0] - values[1])
        top1_ulp = dtype_ulp(float(values[0]), logit_dtype)
        margin_in_ulps = None if top1_ulp in (None, 0.0) else margin / top1_ulp
        rows.append({
            "prompt_index": diagnostic["prompt_index"],
            "available": True,
            "target_top1_token_id": int(indices[0]),
            "target_top2_token_id": int(indices[1]),
            "target_top1_logit": float(values[0]),
            "target_top2_logit": float(values[1]),
            "target_top1_top2_margin": margin,
            "target_logit_dtype": str(logit_dtype).removeprefix("torch."),
            "target_top1_ulp": top1_ulp,
            "margin_in_target_ulps": margin_in_ulps,
            "near_tie_within_two_ulps": (
                margin_in_ulps is not None and margin_in_ulps <= 2.0
            ),
            "reference_token_at_mismatch": diagnostic.get("reference_token_at_mismatch"),
            "candidate_token_at_mismatch": diagnostic.get("candidate_token_at_mismatch"),
        })
    return rows


def _write_json(result: Mapping[str, Any], output_path: str | Path) -> dict[str, Any]:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(result)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_asset_gate_result(
    *,
    semantic_id: str,
    profile_id: str,
    run_mode: str,
    run_label: str,
    workload_id: str,
    workload_version: str,
    mechanism: str,
    required_assets: Sequence[Mapping[str, Any]],
    required_packages: Sequence[str],
    output_path: str | Path,
    not_ready_decision: str = "not_ready",
    not_ready_message: str = "真实模型或后端资产尚未配置；未执行替代性 GPU 探针。",
) -> dict[str, Any]:
    """Record whether a mechanism has the real assets needed for a GPU benchmark."""
    package_status = {name: importlib.util.find_spec(name) is not None for name in required_packages}
    assets = []
    for asset in required_assets:
        item = dict(asset)
        value = item.get("value")
        item["configured"] = bool(value and str(value).strip())
        assets.append(item)
    missing_packages = [name for name, available in package_status.items() if not available]
    missing_assets = [item.get("name") for item in assets if not item["configured"]]
    ready = not missing_packages and not missing_assets
    result = {
        "schema_version": "inference-mechanism-gpu/v2",
        "semantic_id": semantic_id,
        "environment_profile": profile_id,
        "run_mode": run_mode,
        "run_label": run_label,
        "framework": {"python": platform.python_version(), "packages": package_status},
        "workload": {
            "id": workload_id,
            "version": workload_version,
            "mechanism": mechanism,
            "required_assets": assets,
        },
        "asset_gate": {
            "ready": ready,
            "missing_packages": missing_packages,
            "missing_assets": missing_assets,
        },
        "baseline": None,
        "candidates": [],
        "failure": None if ready else {
            "type": "assets_not_ready",
            "message": not_ready_message,
        },
        "evidence_level": "asset_preflight" if ready else "not_ready",
        "decision": "ready_to_measure" if ready else not_ready_decision,
    }
    return _write_json(result, output_path)


def run_transformers_assisted_benchmark(
    *,
    semantic_id: str,
    profile_id: str,
    run_mode: str,
    run_label: str,
    workload_id: str,
    workload_version: str,
    target_model_id: str,
    target_revision: str,
    assistant_model_id: str,
    assistant_revision: str,
    dtype_name: str,
    prompts: Sequence[str],
    max_new_tokens: int,
    assistant_tokens: int,
    warmup_steps: int,
    repeat_steps: int,
    seed: int,
    output_path: str | Path,
) -> dict[str, Any]:
    """Compare target-only generation with a fixed-budget real assistant model."""
    import torch

    from tools.notebook_runtime import resolve_model_commit

    result: dict[str, Any] = {
        "schema_version": "inference-mechanism-gpu/v2",
        "semantic_id": semantic_id,
        "environment_profile": profile_id,
        "run_mode": run_mode,
        "run_label": run_label,
        "framework": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
        },
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "workload": {
            "id": workload_id,
            "version": workload_version,
            "target_model_id": target_model_id,
            "target_revision": target_revision,
            "target_commit": None,
            "assistant_model_id": assistant_model_id,
            "assistant_revision": assistant_revision,
            "assistant_commit": None,
            "dtype": dtype_name,
            "prompt_count": len(prompts),
            "max_new_tokens": max_new_tokens,
            "assistant_tokens": assistant_tokens,
            "warmup_steps": warmup_steps,
            "repeat_steps": repeat_steps,
            "seed": seed,
        },
        "baseline": None,
        "candidates": [],
        "failure": None,
        "evidence_level": "dry_run",
        "decision": "not_measured",
    }
    if run_mode != "real_gpu":
        return _write_json(result, output_path)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("RUN_MODE=real_gpu，但 CUDA 不可用")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cache_dir = os.getenv("HF_HUB_CACHE") or None
        local_only = os.getenv("HF_HUB_OFFLINE") == "1"
        target_commit, _ = resolve_model_commit(target_model_id, target_revision, cache_dir, local_only)
        assistant_commit, _ = resolve_model_commit(assistant_model_id, assistant_revision, cache_dir, local_only)
        result["workload"]["target_commit"] = target_commit
        result["workload"]["assistant_commit"] = assistant_commit
        dtype = getattr(torch, dtype_name)
        torch.manual_seed(seed)
        tokenizer = AutoTokenizer.from_pretrained(
            target_model_id, revision=target_commit, cache_dir=cache_dir, local_files_only=local_only
        )
        prompt_lengths = [len(tokenizer(prompt).input_ids) for prompt in prompts]
        target = AutoModelForCausalLM.from_pretrained(
            target_model_id, revision=target_commit, cache_dir=cache_dir,
            local_files_only=local_only, dtype=dtype,
        ).to("cuda").eval()

        def greedy_config(model):
            config = copy.deepcopy(model.generation_config)
            config.do_sample = False
            config.temperature = None
            config.top_p = None
            config.top_k = None
            config.repetition_penalty = 1.0
            config.max_new_tokens = max_new_tokens
            config.min_new_tokens = 0
            config.use_cache = True
            config.pad_token_id = tokenizer.eos_token_id
            return config

        target_config = greedy_config(target)
        target.generation_config = copy.deepcopy(target_config)

        def measure(assistant=None):
            target_calls = 0
            assistant_calls = 0

            def count_target(_module, _inputs):
                nonlocal target_calls
                target_calls += 1

            def count_assistant(_module, _inputs):
                nonlocal assistant_calls
                assistant_calls += 1

            target_hook = target.register_forward_pre_hook(count_target)
            assistant_hook = assistant.register_forward_pre_hook(count_assistant) if assistant is not None else None
            outputs = []
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            torch.cuda.reset_peak_memory_stats()
            try:
                started.record()
                with torch.inference_mode():
                    for prompt in prompts:
                        inputs = tokenizer(prompt, return_tensors="pt").to(target.device)
                        generated = target.generate(
                            **inputs, assistant_model=assistant, generation_config=target_config
                        )
                        outputs.append(generated[0].detach().cpu().tolist())
                finished.record()
                torch.cuda.synchronize()
            finally:
                target_hook.remove()
                if assistant_hook is not None:
                    assistant_hook.remove()
            generated_tokens = sum(
                max(0, len(ids) - len(tokenizer(prompt).input_ids))
                for ids, prompt in zip(outputs, prompts)
            )
            return {
                "outputs": outputs,
                "latency_ms": started.elapsed_time(finished),
                "target_forward_calls": target_calls,
                "assistant_forward_calls": assistant_calls,
                "generated_tokens": generated_tokens,
                "tokens_per_target_call": generated_tokens / max(1, target_calls),
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            }

        for _ in range(warmup_steps):
            measure()
        baseline_runs = [measure() for _ in range(repeat_steps)]
        reference_outputs = baseline_runs[0]["outputs"]
        baseline_stable = all(item["outputs"] == reference_outputs for item in baseline_runs)
        baseline_median = statistics.median(item["latency_ms"] for item in baseline_runs)
        result["baseline"] = {
            "name": "target_only",
            "outputs_stable_across_repeats": baseline_stable,
            "output_hashes": [_token_hash(output) for output in reference_outputs],
            "generated_tokens_per_prompt": [
                max(0, len(output) - prompt_length)
                for output, prompt_length in zip(reference_outputs, prompt_lengths)
            ],
            "median_latency_ms": baseline_median,
            "p95_latency_ms": _p95([item["latency_ms"] for item in baseline_runs]),
            "median_target_forward_calls": statistics.median(
                item["target_forward_calls"] for item in baseline_runs
            ),
            "peak_memory_bytes": max(item["peak_memory_bytes"] for item in baseline_runs),
        }

        assistant = AutoModelForCausalLM.from_pretrained(
            assistant_model_id, revision=assistant_commit, cache_dir=cache_dir,
            local_files_only=local_only, dtype=dtype,
        ).to("cuda").eval()
        target_vocab_size = int(getattr(target.config, "vocab_size", -1))
        assistant_vocab_size = int(getattr(assistant.config, "vocab_size", -1))
        target_eos = getattr(target.generation_config, "eos_token_id", None)
        assistant_eos = getattr(assistant.generation_config, "eos_token_id", None)
        compatibility = {
            "target_vocab_size": target_vocab_size,
            "assistant_vocab_size": assistant_vocab_size,
            "vocab_size_match": target_vocab_size == assistant_vocab_size,
            "tokenizer_vocab_size": len(tokenizer),
            "target_eos_token_id": target_eos,
            "assistant_eos_token_id": assistant_eos,
            "eos_match": target_eos == assistant_eos,
            "pad_token_id": tokenizer.pad_token_id,
        }
        result["workload"]["model_compatibility"] = compatibility
        if not compatibility["vocab_size_match"]:
            raise RuntimeError("目标模型与草稿模型词表大小不一致，不能进行 assisted generation")
        assistant.generation_config = greedy_config(assistant)
        assistant.generation_config.num_assistant_tokens_schedule = "constant"
        assistant.generation_config.num_assistant_tokens = assistant_tokens
        assistant.generation_config.assistant_confidence_threshold = 0.0
        for _ in range(warmup_steps):
            assistant.generation_config.num_assistant_tokens = assistant_tokens
            measure(assistant)
        assisted_runs = []
        for _ in range(repeat_steps):
            assistant.generation_config.num_assistant_tokens = assistant_tokens
            assisted_runs.append(measure(assistant))
        median_ms = statistics.median(item["latency_ms"] for item in assisted_runs)
        outputs_match = baseline_stable and all(
            item["outputs"] == reference_outputs for item in assisted_runs
        )
        per_repeat_diagnostics = [
            _output_diagnostics(reference_outputs, item["outputs"], prompt_lengths)
            for item in assisted_runs
        ]
        assisted_reference = assisted_runs[0]["outputs"]
        assisted_stable = all(item["outputs"] == assisted_reference for item in assisted_runs)
        assisted_hashes_by_repeat = [
            [_token_hash(output) for output in item["outputs"]]
            for item in assisted_runs
        ]
        margin_diagnostics = _target_margin_diagnostics(
            target, reference_outputs, per_repeat_diagnostics[0]
        )
        mismatch_prompt_indices = {
            int(item["prompt_index"])
            for item in per_repeat_diagnostics[0]
            if not item["match"]
        }
        mismatch_margins = [
            item for item in margin_diagnostics
            if int(item["prompt_index"]) in mismatch_prompt_indices
        ]
        all_mismatches_near_tie = bool(mismatch_margins) and all(
            item.get("near_tie_within_two_ulps", False)
            for item in mismatch_margins
        )
        result["candidates"].append({
            "candidate": "fixed_budget_assistant",
            "outputs_match_baseline": outputs_match,
            "baseline_stable": baseline_stable,
            "candidate_stable_across_repeats": assisted_stable,
            "output_hashes_by_repeat": assisted_hashes_by_repeat,
            "output_diagnostics": per_repeat_diagnostics[0],
            "target_margin_diagnostics": margin_diagnostics,
            "all_mismatches_near_tie": all_mismatches_near_tie,
            "all_repeat_matches": [
                all(entry["match"] for entry in diagnostics)
                for diagnostics in per_repeat_diagnostics
            ],
            "median_latency_ms": median_ms,
            "p95_latency_ms": _p95([item["latency_ms"] for item in assisted_runs]),
            "latency_ratio_vs_baseline": median_ms / baseline_median,
            "median_target_forward_calls": statistics.median(
                item["target_forward_calls"] for item in assisted_runs
            ),
            "median_assistant_forward_calls": statistics.median(
                item["assistant_forward_calls"] for item in assisted_runs
            ),
            "median_tokens_per_target_call": statistics.median(
                item["tokens_per_target_call"] for item in assisted_runs
            ),
            "peak_memory_bytes": max(item["peak_memory_bytes"] for item in assisted_runs),
            "path_confirmed": True,
            "acceptance_rate": None,
            "acceptance_rate_note": "Transformers generate 未暴露逐候选接受日志；本页不从前向次数反推接受率。",
        })
        candidate = result["candidates"][0]
        result["evidence_level"] = "local_mechanism_benchmark"
        if not outputs_match:
            result["decision"] = (
                "tune_numeric_equivalence" if all_mismatches_near_tie else "reject"
            )
        elif candidate["latency_ratio_vs_baseline"] < 0.95:
            result["decision"] = "accept_for_fixed_budget_path"
        else:
            result["decision"] = "tune"
    except Exception as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
        result["evidence_level"] = "failed_real_gpu_attempt"
        result["decision"] = "reject_until_fixed"
    return _write_json(result, output_path)


def read_mechanism_result(output_path: str | Path) -> dict[str, Any]:
    path = Path(output_path)
    if not path.exists():
        return {
            "failure": {"type": "not_run", "message": str(path)},
            "candidates": [],
            "decision": "not_run",
        }
    return json.loads(path.read_text(encoding="utf-8"))
