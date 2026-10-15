"""Notebook helpers shared by inference projects 66--70.

The helper keeps the optional GPU path explicit: a notebook can stay runnable
without vLLM, while a GPU learner gets model resolution, dtype selection, free
port selection, service cleanup and result persistence from one place.
"""

from __future__ import annotations

import json
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


def admit_speculative_candidate(candidate: Mapping[str, Any]) -> dict[str, str]:
    """Return the first failing speculative-decoding admission gate."""

    if not candidate.get("enabled", False):
        return {"run_status": "skipped", "reason": "candidate_disabled"}
    if not candidate.get("backend_supported", False):
        return {"run_status": "unsupported", "reason": "backend_method_unavailable"}
    if not candidate.get("asset_ready", False):
        return {"run_status": "asset_missing", "reason": "required_model_asset_unavailable"}
    if not candidate.get("compatible", False):
        return {"run_status": "incompatible", "reason": "target_proposer_contract_failed"}
    estimated = candidate.get("estimated_memory_gb")
    limit = candidate.get("memory_limit_gb")
    if estimated is not None and limit is not None and estimated > limit:
        return {"run_status": "capacity_blocked", "reason": "estimated_memory_exceeds_limit"}
    return {"run_status": "runnable", "reason": "static_admission_passed"}


def build_speculative_pair_evidence(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a completed baseline/candidate pair and derive ratios."""

    completed = baseline.get("run_status") == candidate.get("run_status") == "completed"
    if not completed:
        return {
            "evidence_status": "not_evaluated",
            "missing_evidence": ["completed_pair"],
            "candidate_run_status": str(candidate.get("run_status", "unknown")),
        }
    fixed_fields = ("target_snapshot", "workload_key", "sampling_key")
    mismatches = [name for name in fixed_fields if baseline.get(name) != candidate.get(name)]
    required = (
        "throughput_tokens_per_s",
        "ttft_p95_ms",
        "e2e_p95_ms",
        "peak_memory_mb",
        "quality_ok",
    )
    missing = [
        name for name in required
        if candidate.get(name) is None or baseline.get(name) is None
    ]
    path_ok = (
        baseline.get("execution_path") == "target_only"
        and candidate.get("execution_path") == "speculative"
    )
    if not path_ok:
        missing.append("speculative_execution_path")
    if candidate.get("mechanism_active") is not True:
        missing.append("observed_speculative_proposals")
    if mismatches:
        return {"evidence_status": "incomparable", "mismatches": mismatches, "missing_evidence": missing}
    if missing:
        return {"evidence_status": "incomplete", "mismatches": [], "missing_evidence": missing}
    return {
        "evidence_status": "comparable",
        "mismatches": [],
        "missing_evidence": [],
        "throughput_speedup": candidate["throughput_tokens_per_s"] / baseline["throughput_tokens_per_s"],
        "ttft_ratio": candidate["ttft_p95_ms"] / baseline["ttft_p95_ms"],
        "e2e_p95_ratio": candidate["e2e_p95_ms"] / baseline["e2e_p95_ms"],
        "candidate_peak_memory_mb": candidate["peak_memory_mb"],
        "quality_ok": bool(candidate["quality_ok"]),
        "mechanism_active": True,
    }


def decide_speculative_candidate(
    evidence: Mapping[str, Any], thresholds: Mapping[str, float]
) -> dict[str, str]:
    """Apply quality, performance, latency, and memory adoption gates."""

    status = evidence.get("evidence_status")
    if status not in ("comparable", "incomplete"):
        return {"decision": "not_evaluated", "reason": f"evidence_{status}"}
    if status == "incomplete":
        return {"decision": "tune", "reason": "collect_missing_evidence"}
    if not evidence.get("quality_ok", False):
        return {"decision": "reject", "reason": "quality_gate_failed"}
    gates = (
        evidence["throughput_speedup"] >= thresholds["min_throughput_speedup"],
        evidence["ttft_ratio"] <= thresholds["max_ttft_regression_ratio"],
        evidence["e2e_p95_ratio"] <= thresholds["max_e2e_p95_regression_ratio"],
        evidence["candidate_peak_memory_mb"] <= thresholds["max_peak_memory_mb"],
    )
    if all(gates):
        return {"decision": "accept", "reason": "quality_and_net_benefit_passed"}
    return {"decision": "tune", "reason": "one_or_more_performance_gates_failed"}


def inspect_quantization_upstream_contract(
    *,
    baseline_path: str | Path,
    adapter_manifest_path: str | Path | None = None,
    expected_baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Inspect the optional 65/66 inputs used by project 67 without a GPU.

    A 65 manifest records the origin of an adapter or merged model.  It is
    deliberately *not* accepted as a GPTQ/AWQ/GGUF artifact.  A 66 result is
    reusable only when it has the standard baseline contract and agrees with
    the selected model/workload/backend fields.  Project 67 may still run its
    own paired G0/G1 measurement when either upstream record is unavailable.
    """

    baseline_file = Path(baseline_path)
    expected = dict(expected_baseline or {})
    required = {
        "schema_version",
        "project",
        "experiment_group",
        "role",
        "quantization_format",
        "model_revision",
        "backend",
        "workload_path",
        "dtype",
        "evidence_level",
    }
    baseline: dict[str, Any] = {
        "status": "pending_baseline",
        "path": str(baseline_file),
        "missing": ["result_json"],
        "mismatches": {},
        "contract": None,
    }
    if baseline_file.exists():
        try:
            payload = json.loads(baseline_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            baseline.update(status="invalid_json", missing=["valid_json"], error=str(error))
        else:
            contract = payload.get("experiment_contract") or payload.get("baseline_contract") or {}
            missing = sorted(required - set(contract))
            mismatches = {
                key: {"expected": value, "actual": contract.get(key)}
                for key, value in expected.items()
                if value is not None and contract.get(key) != value
            }
            failure = payload.get("failure") or contract.get("failure") or {}
            if failure.get("status") == "recorded":
                status = "failure_recorded"
            elif missing:
                status = "legacy_or_incomplete"
            elif mismatches:
                status = "mismatch"
            else:
                status = "compatible"
            baseline.update(status=status, missing=missing, mismatches=mismatches, contract=contract)

    manifest: dict[str, Any] = {
        "status": "not_provided",
        "path": str(adapter_manifest_path) if adapter_manifest_path else None,
        "artifact_role": "provenance_only",
        "can_replace_quantization_artifact": False,
    }
    if adapter_manifest_path:
        manifest_file = Path(adapter_manifest_path)
        if not manifest_file.exists():
            manifest["status"] = "pending_manifest"
        else:
            try:
                payload = json.loads(manifest_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                manifest.update(status="invalid_json", error=str(error))
            else:
                ready = payload.get("status") == "ready" and bool(payload.get("artifact_path"))
                manifest.update(
                    status="ready_provenance" if ready else "not_ready",
                    model_revision=payload.get("model_revision") or payload.get("model_id"),
                    artifact_format=payload.get("artifact_format"),
                    result_json=payload.get("result_json"),
                )

    return {
        "status": "compatible" if baseline["status"] == "compatible" else "standalone_pair_required",
        "baseline": baseline,
        "adapter_manifest": manifest,
        "matched_measurement": baseline["status"] == "compatible",
        "standalone_pair_allowed": True,
    }


def locate_repo_root(start: str | Path | None = None) -> Path:
    """Find the repository root from a notebook working directory."""

    current = Path(start or Path.cwd()).resolve()
    candidates = (current, *current.parents)
    for candidate in candidates:
        if (candidate / "tools").is_dir() and (candidate / "02_PyTorch_Algorithms").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError("未找到项目根目录，请从仓库启动 Notebook 或先 clone 仓库。")


def shared_project_config(
    *,
    model: str,
    backend: str = "local",
    dtype: str = "unknown",
    prompt_tokens: int | None = None,
    generated_tokens: int | None = None,
    batch: int = 1,
    concurrency: int = 1,
    cache_policy: str = "default",
    **extra: Any,
) -> dict[str, Any]:
    """Build the common 66--70 configuration block."""

    config = {
        "model": model,
        "backend": backend,
        "dtype": dtype,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "batch": batch,
        "concurrency": concurrency,
        "cache_policy": cache_policy,
    }
    errors = validate_backend_config(config)
    if errors:
        raise ValueError("invalid backend config: " + "; ".join(errors))
    config.update({key: value for key, value in extra.items() if value is not None})
    return config


def validate_backend_config(config: Mapping[str, Any]) -> list[str]:
    """Validate common 66--70 fields without starting a server."""

    errors: list[str] = []
    for key in ("model", "backend", "dtype", "cache_policy"):
        if not str(config.get(key, "")).strip():
            errors.append(f"missing config: {key}")
    for key in ("batch", "concurrency"):
        if key in config and int(config[key]) <= 0:
            errors.append(f"invalid config: {key}")
    for key in ("prompt_tokens", "generated_tokens"):
        if config.get(key) is not None and int(config[key]) <= 0:
            errors.append(f"invalid config: {key}")
    return errors


def runtime_snapshot(torch_module: Any | None = None) -> dict[str, Any]:
    """Capture the environment used by a backend run."""

    snapshot: dict[str, Any] = {"python": sys.version, "platform": platform.platform()}
    if torch_module is None:
        return snapshot
    snapshot["torch"] = getattr(torch_module, "__version__", None)
    snapshot["torch_cuda"] = getattr(getattr(torch_module, "version", None), "cuda", None)
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        snapshot["device"] = cuda.get_device_name(0)
    else:
        snapshot["device"] = "cpu"
    return snapshot


def save_project_result(
    path: str | Path,
    *,
    project: str,
    strategy: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    quality: Mapping[str, Any] | None = None,
    decision: Mapping[str, Any] | None = None,
    strategy_metrics: Mapping[str, Any] | None = None,
) -> Path:
    """Save one project result using the shared schema."""

    from tools.inference_result_schema import make_result, save_result

    result = make_result(
        project=project,
        strategy=strategy,
        config=config,
        metrics=metrics,
        quality=quality,
        decision=decision,
        strategy_metrics=strategy_metrics,
    )
    output = save_result(path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return output


def start_optional_vllm(
    *,
    model_id: str,
    model_source: str = "auto",
    cache_dir: str | Path = "model_cache",
    dtype: str = "auto",
    vllm_command: str | None = None,
    vllm_environment: str | None = None,
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.8,
               enforce_eager: bool = True,
    served_model_name: str | None = None,
    enable_prefix_caching: bool = False,
    scheduling_policy: str = "fcfs",
    max_num_seqs: int | None = None,
    quantization_args: list[str] | None = None,
    generation_config: str | None = None,
    log_path: str | Path | None = None,
) -> tuple[Any, Any, int, str, str]:
    """Resolve a model and launch vLLM for an optional Practice-P2 run.

    Returns ``(server, log_handle, port, selected_dtype, model_path)``. The
    caller owns the server and must call ``stop_optional_vllm`` in ``finally``.
    """

    root = locate_repo_root()
    from tools.backend_runtime import resolve_model, start_vllm

    model_path = resolve_model(model_id, model_source, cache_dir=cache_dir)
    server, log_handle, port, selected_dtype = start_vllm(
        model_path,
        dtype,
        vllm_command=vllm_command,
        vllm_environment=vllm_environment,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        served_model_name=served_model_name or model_id,
        enable_prefix_caching=enable_prefix_caching,
        scheduling_policy=scheduling_policy,
        max_num_seqs=max_num_seqs,
        quantization_args=quantization_args,
        generation_config=generation_config,
        **({"log_path": str(log_path)} if log_path is not None else {}),
    )
    return server, log_handle, port, selected_dtype, str(model_path)


def start_speculative_vllm(
    *,
    target_model_id: str,
    draft_model_id: str,
    model_source: str = "auto",
    cache_dir: str | Path = "model_cache",
    dtype: str = "auto",
    proposal_length: int = 5,
    vllm_command: str | None = None,
    vllm_environment: str | None = None,
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.8,
    enforce_eager: bool = True,
    served_model_name: str | None = None,
    generation_config: str | None = None,
    log_path: str | Path | None = None,
):
    """Launch vLLM speculative decoding using detected CLI syntax.

    The function fails before model loading when the installed CLI does not
    expose a supported speculative interface. It does not verify model
    compatibility or acceptance metrics; callers must check those separately.
    """

    if target_model_id == draft_model_id:
        raise ValueError("target_model_id 与 draft_model_id 不能相同")
    from tools.backend_runtime import (
        build_vllm_speculative_args,
        probe_vllm_speculative_support,
        resolve_model,
        start_vllm,
    )

    capability = probe_vllm_speculative_support(vllm_command, vllm_environment)
    if capability["status"] != "supported":
        raise RuntimeError(
            "当前 vLLM CLI 不支持可识别的 speculative 参数："
            + json.dumps(capability, ensure_ascii=False)
        )
    target_path = resolve_model(target_model_id, model_source, cache_dir=cache_dir)
    draft_path = resolve_model(draft_model_id, model_source, cache_dir=cache_dir)
    capability = {
        **capability,
        "resolved_target_snapshot": Path(target_path).name,
        "resolved_draft_snapshot": Path(draft_path).name,
    }
    speculative_args = build_vllm_speculative_args(
        capability, draft_model=str(draft_path), proposal_length=proposal_length
    )
    server, log_handle, port, selected_dtype = start_vllm(
        target_path,
        dtype,
        vllm_command=vllm_command,
        vllm_environment=vllm_environment,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        served_model_name=served_model_name or target_model_id,
        speculative_args=speculative_args,
        generation_config=generation_config,
        **({"log_path": str(log_path)} if log_path is not None else {}),
    )
    return server, log_handle, port, selected_dtype, str(target_path), capability


def start_speculative_strategy_vllm(
    *,
    target_model_id: str,
    method: str | None,
    proposal_length: int,
    proposer_model_id: str | None = None,
    method_options: Mapping[str, Any] | None = None,
    model_source: str = "auto",
    cache_dir: str | Path = "model_cache",
    dtype: str = "auto",
    vllm_command: str | None = None,
    vllm_environment: str | None = None,
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.8,
    enforce_eager: bool = True,
    served_model_name: str | None = None,
    generation_config: str | None = None,
    log_path: str | Path | None = None,
):
    """Launch one admitted speculative family through vLLM's generic config.

    This adapter supports asset-free proposers such as n-gram, target-native
    MTP, and proposer checkpoints such as EAGLE.  Project 68 performs the
    capability and asset gates before calling it.
    """

    from tools.backend_runtime import (
        build_vllm_speculative_strategy_args,
        probe_vllm_speculative_support,
        resolve_model,
        start_vllm,
    )

    capability = probe_vllm_speculative_support(vllm_command, vllm_environment)
    if capability["status"] != "supported":
        raise RuntimeError(
            "当前 vLLM CLI 不支持可识别的 speculative 接口："
            + json.dumps(capability, ensure_ascii=False)
        )
    target_path = resolve_model(target_model_id, model_source, cache_dir=cache_dir)
    proposer_path = None
    if proposer_model_id:
        proposer_path = resolve_model(proposer_model_id, model_source, cache_dir=cache_dir)
    capability = {
        **capability,
        "resolved_target_snapshot": Path(target_path).name,
        "resolved_proposer_snapshot": Path(proposer_path).name if proposer_path else None,
    }
    speculative_args = build_vllm_speculative_strategy_args(
        capability,
        method=method,
        proposal_length=proposal_length,
        model=str(proposer_path) if proposer_path else None,
        method_options=method_options,
    )
    server, log_handle, port, selected_dtype = start_vllm(
        target_path,
        dtype,
        vllm_command=vllm_command,
        vllm_environment=vllm_environment,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        served_model_name=served_model_name or target_model_id,
        speculative_args=speculative_args,
        generation_config=generation_config,
        **({"log_path": str(log_path)} if log_path is not None else {}),
    )
    return server, log_handle, port, selected_dtype, str(target_path), capability


def stop_optional_vllm(server: Any, log_handle: Any) -> None:
    """Stop a server started by :func:`start_optional_vllm` and close its log."""

    from tools.backend_runtime import stop_backend

    # ``start_vllm`` returns the already-open TextIOWrapper used as the
    # subprocess stdout target.  ``stop_backend`` owns closing that handle;
    # turning it into ``Path(...)`` raises TypeError and can hide completed
    # benchmark evidence during cleanup.
    stop_backend(server, log_handle)


def start_external_openai_backend(
    command_template: str | list[str],
    *,
    model_path: str,
    port: int,
    log_path: str | Path,
    ready_timeout_s: float = 120.0,
) -> tuple[Any, Path]:
    """Start a user-specified OpenAI-compatible backend.

    The template must contain ``{model_path}`` and ``{port}``.  This adapter
    deliberately does not assume a GGUF engine or its CLI flags; the learner
    must provide the engine-specific command and record it in the report.
    """

    if not model_path:
        raise ValueError("model_path 不能为空")
    if port <= 0:
        raise ValueError("port 必须大于 0")
    tokens = shlex.split(command_template) if isinstance(command_template, str) else list(command_template)
    if not tokens or "{model_path}" not in tokens or "{port}" not in tokens:
        raise ValueError("command_template 必须包含 {model_path} 和 {port} 占位符")
    command = [token.format(model_path=model_path, port=port) for token in tokens]
    log_file_path = Path(log_path)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_file_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
    try:
        from tools.backend_runtime import wait_until_ready
        wait_until_ready(port, timeout_s=ready_timeout_s)
    except Exception:
        from tools.backend_runtime import stop_backend
        stop_backend(process, log_file)
        raise
    return process, log_file_path


def run_backend_benchmark(
    *,
    project: str,
    base_url: str,
    model: str,
    label: str,
    output: str | Path,
    workload: str = "benchmarks/workloads/fixed.jsonl",
    num_prompts: int = 5,
    max_tokens: int = 64,
    concurrency: int = 1,
    warmup: int = 1,
    repeats: int = 1,
    temperature: float = 0.0,
    timeout_s: float = 180.0,
    backend: str = "vllm",
    dtype: str = "unknown",
    batch: int = 1,
    cache_policy: str = "default",
    role: str | None = None,
    peak_memory_mb: float | None = None,
    measure_device_peak_memory: bool = False,
    include_requests: bool = False,
    send_workload_priority: bool = False,
    synchronize_start: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run the common OpenAI-compatible benchmark and load its JSON output."""

    root = locate_repo_root()
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = root / output_path
    command = [
        sys.executable,
        str(root / "tools/benchmark_inference_backend.py"),
        "--base-url", base_url,
        "--model", model,
        "--label", label,
        "--project", project,
        "--backend", backend,
        "--dtype", dtype,
        "--batch", str(batch),
        "--cache-policy", cache_policy,
        "--role", role or "candidate",
        "--workload", str(root / workload),
        "--num-prompts", str(num_prompts),
        "--max-tokens", str(max_tokens),
        "--concurrency", str(concurrency),
        "--warmup", str(warmup),
        "--repeats", str(repeats),
        "--temperature", str(temperature),
        "--timeout", str(timeout_s),
        "--output", str(output_path),
    ]
    if peak_memory_mb is not None:
        command.extend(["--peak-memory-mb", str(peak_memory_mb)])
    if include_requests:
        command.append("--include-requests")
    if send_workload_priority:
        command.append("--send-workload-priority")
    if synchronize_start:
        command.append("--synchronize-start")
    if not measure_device_peak_memory or peak_memory_mb is not None:
        subprocess.run(
            command,
            cwd=root,
            check=True,
            **({"stdout": subprocess.DEVNULL} if quiet else {}),
        )
        return json.loads(output_path.read_text(encoding="utf-8"))

    def read_device_memory_mb() -> float | None:
        """Read total used memory on visible GPU 0; return None when unavailable."""
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi", "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits", "--id=0",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return float(completed.stdout.strip().splitlines()[0])
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None

    process = subprocess.Popen(
        command,
        cwd=root,
        **({"stdout": subprocess.DEVNULL} if quiet else {}),
    )
    samples: list[float] = []
    while process.poll() is None:
        sample = read_device_memory_mb()
        if sample is not None:
            samples.append(sample)
        time.sleep(0.2)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)

    report = json.loads(output_path.read_text(encoding="utf-8"))
    measured_peak = max(samples) if samples else None
    report.setdefault("metrics", {})["peak_memory_mb"] = measured_peak
    normalized = report.get("normalized_result") or {}
    normalized.setdefault("metrics", {})["peak_memory_mb"] = measured_peak
    report["normalized_result"] = normalized
    report["resource_measurement"] = {
        "metric": "device_memory_used_peak_mb",
        "scope": "visible_gpu_0_total_device_usage",
        "sample_count": len(samples),
        "includes_non_service_processes": True,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
