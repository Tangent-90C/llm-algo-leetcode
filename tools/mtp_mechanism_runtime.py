"""Isolated vLLM benchmark for a model checkpoint with native MTP layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("P95 requires at least one value")
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def _token_hash(token_ids: Sequence[int]) -> str:
    payload = ",".join(str(int(token_id)) for token_id in token_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(payload: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    result = dict(payload)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _metric_snapshot(llm: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for metric in llm.get_metrics():
        name = str(getattr(metric, "name", ""))
        if not name:
            continue
        if hasattr(metric, "values"):
            snapshot[name] = [float(value) for value in metric.values]
        elif hasattr(metric, "value"):
            snapshot[name] = float(metric.value)
    return snapshot


def _metric_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for name, after_value in after.items():
        before_value = before.get(name)
        if isinstance(after_value, list):
            previous = before_value if isinstance(before_value, list) else [0.0] * len(after_value)
            delta[name] = [
                float(value) - float(previous[index] if index < len(previous) else 0.0)
                for index, value in enumerate(after_value)
            ]
        elif isinstance(after_value, (int, float)):
            delta[name] = float(after_value) - float(before_value or 0.0)
    return delta


def _metric_value(metrics: Mapping[str, Any], suffix: str) -> float:
    for name, value in metrics.items():
        if name.endswith(suffix) and isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _worker(config: Mapping[str, Any]) -> dict[str, Any]:
    # vLLM 0.31.0 defaults to ``fork`` for local workers.  This benchmark is
    # itself launched as an isolated subprocess, so use ``spawn`` before
    # importing torch/vLLM and before either library can initialize CUDA.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    import torch
    from vllm import LLM, SamplingParams

    speculative_tokens = int(config["speculative_tokens"])
    llm_kwargs: dict[str, Any] = {
        "model": config["model_id"],
        "revision": config["model_commit"],
        "dtype": config["dtype"],
        "tensor_parallel_size": 1,
        "max_model_len": int(config["max_model_len"]),
        "max_num_seqs": len(config["prompts"]),
        "gpu_memory_utilization": float(config["gpu_memory_utilization"]),
        "enforce_eager": bool(config["enforce_eager"]),
        "disable_log_stats": False,
        "seed": int(config["seed"]),
        "trust_remote_code": False,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
    }
    if speculative_tokens:
        llm_kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": speculative_tokens,
        }

    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=int(config["max_new_tokens"]),
        min_tokens=int(config["max_new_tokens"]),
        ignore_eos=True,
        seed=int(config["seed"]),
    )

    def generate_once() -> tuple[list[list[int]], float, int]:
        started = time.perf_counter()
        outputs = llm.generate(config["prompts"], sampling, use_tqdm=False)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        token_ids = [list(item.outputs[0].token_ids) for item in outputs]
        return token_ids, elapsed_ms, sum(len(ids) for ids in token_ids)

    for _ in range(int(config["warmup_steps"])):
        generate_once()

    metrics_before = _metric_snapshot(llm)
    latencies_ms: list[float] = []
    throughputs: list[float] = []
    outputs_by_repeat: list[list[list[int]]] = []
    used_memory_samples: list[int] = []
    for _ in range(int(config["repeat_steps"])):
        token_ids, elapsed_ms, output_tokens = generate_once()
        outputs_by_repeat.append(token_ids)
        latencies_ms.append(elapsed_ms)
        throughputs.append(output_tokens / max(elapsed_ms / 1000.0, 1e-9))
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        used_memory_samples.append(int(total_bytes - free_bytes))
    metrics_after = _metric_snapshot(llm)
    metric_delta = _metric_delta(metrics_before, metrics_after)

    reference_outputs = outputs_by_repeat[0]
    num_drafts = _metric_value(metric_delta, "spec_decode_num_drafts")
    num_draft_tokens = _metric_value(metric_delta, "spec_decode_num_draft_tokens")
    num_accepted_tokens = _metric_value(metric_delta, "spec_decode_num_accepted_tokens")
    acceptance = None
    if speculative_tokens:
        acceptance = {
            "num_drafts": int(num_drafts),
            "num_draft_tokens": int(num_draft_tokens),
            "num_accepted_draft_tokens": int(num_accepted_tokens),
            "draft_acceptance_rate": (
                num_accepted_tokens / num_draft_tokens if num_draft_tokens else 0.0
            ),
            "mean_acceptance_length": (
                1.0 + num_accepted_tokens / num_drafts if num_drafts else 1.0
            ),
        }

    return {
        "name": "mtp_1" if speculative_tokens else "target_only",
        "speculative_tokens": speculative_tokens,
        "framework": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": _package_version("vllm"),
            "transformers": _package_version("transformers"),
            "vllm_worker_multiproc_method": os.environ[
                "VLLM_WORKER_MULTIPROC_METHOD"
            ],
        },
        "hardware": {
            "device": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        },
        "latency_ms": {
            "median": statistics.median(latencies_ms),
            "p95": _p95(latencies_ms),
            "samples": latencies_ms,
        },
        "throughput_tokens_per_second": {
            "median": statistics.median(throughputs),
            "p95": _p95(throughputs),
            "samples": throughputs,
        },
        "amortized_ms_per_output_token": statistics.median(
            latency / max(1, sum(len(ids) for ids in outputs))
            for latency, outputs in zip(latencies_ms, outputs_by_repeat)
        ),
        "peak_observed_device_memory_bytes": max(used_memory_samples),
        "output_token_ids": reference_outputs,
        "output_hashes": [_token_hash(ids) for ids in reference_outputs],
        "outputs_stable_across_repeats": all(
            outputs == reference_outputs for outputs in outputs_by_repeat
        ),
        "acceptance": acceptance,
        "metric_delta": metric_delta,
    }


def _run_worker(config: Mapping[str, Any], temporary_directory: Path, label: str) -> dict[str, Any]:
    config_path = temporary_directory / f"{label}_config.json"
    output_path = temporary_directory / f"{label}_result.json"
    config_path.write_text(json.dumps(dict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-config",
            str(config_path),
            "--worker-output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError(
            f"{label} worker failed (exit={completed.returncode}): "
            + (completed.stderr or completed.stdout)[-4000:]
        )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if payload.get("failure"):
        raise RuntimeError(f"{label} worker failed: {payload['failure']}")
    return payload["result"]


def run_native_mtp_benchmark(
    *,
    semantic_id: str,
    profile_id: str,
    run_mode: str,
    run_label: str,
    workload_id: str,
    workload_version: str,
    model_id: str,
    model_revision: str,
    dtype: str,
    prompts: Sequence[str],
    max_new_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    warmup_steps: int,
    repeat_steps: int,
    speculative_tokens: int,
    seed: int,
    output_path: str | Path,
) -> dict[str, Any]:
    """Compare target-only and native MTP in separate vLLM worker processes."""

    result: dict[str, Any] = {
        "schema_version": "inference-mtp-mechanism/v1",
        "semantic_id": semantic_id,
        "environment_profile": profile_id,
        "run_mode": run_mode,
        "run_label": run_label,
        "workload": {
            "id": workload_id,
            "version": workload_version,
            "model_id": model_id,
            "requested_revision": model_revision,
            "resolved_commit": None,
            "dtype": dtype,
            "prompt_count": len(prompts),
            "max_new_tokens": max_new_tokens,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "warmup_steps": warmup_steps,
            "repeat_steps": repeat_steps,
            "speculative_tokens": speculative_tokens,
            "seed": seed,
        },
        "model_contract": None,
        "baseline": None,
        "candidate": None,
        "comparison": None,
        "failure": None,
        "evidence_level": "dry_run",
        "decision": "not_measured",
    }
    if run_mode != "real_backend":
        return _write_json(result, output_path)

    try:
        vllm_version = _package_version("vllm")
        if vllm_version != "0.31.0+cu129":
            raise RuntimeError("本实验要求 vllm 0.31.0 CUDA 12.9 构建；请使用 INF-VLLM-MTP Profile")
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("INF-VLLM-MTP 环境缺少 PyTorch") from error
        if torch.version.cuda != "12.9":
            raise RuntimeError(
                f"本实验要求 PyTorch CUDA 12.9 构建，当前为 {torch.version.cuda!r}；"
                "请创建新的 INF-VLLM-MTP 环境，不要复用默认 CUDA 13 环境"
            )
        from transformers import AutoConfig

        from tools.notebook_runtime import resolve_model_commit

        cache_dir = os.getenv("HF_HUB_CACHE") or None
        local_only = os.getenv("HF_HUB_OFFLINE") == "1"
        model_commit, resolution_source = resolve_model_commit(
            model_id, model_revision, cache_dir, local_only
        )
        config = AutoConfig.from_pretrained(
            model_id,
            revision=model_commit,
            cache_dir=cache_dir,
            local_files_only=local_only,
            trust_remote_code=False,
        )
        text_config = getattr(config, "text_config", config)
        mtp_layers = int(getattr(text_config, "mtp_num_hidden_layers", 0) or 0)
        if mtp_layers < speculative_tokens:
            raise RuntimeError(
                f"checkpoint 只有 {mtp_layers} 个原生 MTP 层，不能运行 MTP-{speculative_tokens}"
            )
        result["workload"]["resolved_commit"] = model_commit
        result["model_contract"] = {
            "model_type": getattr(config, "model_type", None),
            "text_model_type": getattr(text_config, "model_type", None),
            "mtp_num_hidden_layers": mtp_layers,
            "resolution_source": resolution_source,
        }
        worker_config = {
            "model_id": model_id,
            "model_commit": model_commit,
            "dtype": dtype,
            "prompts": list(prompts),
            "max_new_tokens": max_new_tokens,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "warmup_steps": warmup_steps,
            "repeat_steps": repeat_steps,
            "seed": seed,
        }
        with tempfile.TemporaryDirectory(prefix="mtp-mechanism-") as directory:
            temporary_directory = Path(directory)
            result["baseline"] = _run_worker(
                {**worker_config, "speculative_tokens": 0}, temporary_directory, "baseline"
            )
            result["candidate"] = _run_worker(
                {**worker_config, "speculative_tokens": speculative_tokens},
                temporary_directory,
                "candidate",
            )

        baseline = result["baseline"]
        candidate = result["candidate"]
        outputs_match = baseline["output_token_ids"] == candidate["output_token_ids"]
        latency_ratio = (
            candidate["latency_ms"]["median"] / baseline["latency_ms"]["median"]
        )
        throughput_ratio = (
            candidate["throughput_tokens_per_second"]["median"]
            / baseline["throughput_tokens_per_second"]["median"]
        )
        acceptance = candidate.get("acceptance") or {}
        result["comparison"] = {
            "outputs_match": outputs_match,
            "latency_ratio": latency_ratio,
            "speedup": 1.0 / latency_ratio,
            "throughput_ratio": throughput_ratio,
            "memory_delta_bytes": (
                candidate["peak_observed_device_memory_bytes"]
                - baseline["peak_observed_device_memory_bytes"]
            ),
            "mean_acceptance_length": acceptance.get("mean_acceptance_length"),
            "draft_acceptance_rate": acceptance.get("draft_acceptance_rate"),
        }
        result["evidence_level"] = "local_backend_mechanism_benchmark"
        if not outputs_match:
            result["decision"] = "tune_numeric_equivalence"
        elif float(acceptance.get("mean_acceptance_length", 1.0)) <= 1.0:
            result["decision"] = "reject_no_effective_progress"
        elif latency_ratio < 1.0:
            result["decision"] = "accept"
        else:
            result["decision"] = "tune"
    except Exception as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
        result["evidence_level"] = "failed_real_backend_attempt"
        result["decision"] = "reject_until_fixed"
    return _write_json(result, output_path)


def _worker_main(config_path: str, output_path: str) -> int:
    try:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        payload = {"result": _worker(config), "failure": None}
        return_code = 0
    except Exception as error:
        payload = {
            "result": None,
            "failure": {"type": type(error).__name__, "message": str(error)},
        }
        return_code = 1
    _write_json(payload, output_path)
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one isolated native-MTP vLLM worker")
    parser.add_argument("--worker-config")
    parser.add_argument("--worker-output")
    args = parser.parse_args()
    if not args.worker_config or not args.worker_output:
        parser.error("--worker-config and --worker-output are required")
    return _worker_main(args.worker_config, args.worker_output)


if __name__ == "__main__":
    raise SystemExit(main())
