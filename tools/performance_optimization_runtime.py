"""Shared GPU workload for performance candidate and composition mechanism pages."""

from __future__ import annotations

import statistics
import time
from typing import Iterable

import torch


class DecoderMLP(torch.nn.Module):
    """Small SwiGLU MLP with decoder-like tensor shapes."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(inputs)) * self.up_proj(inputs))


def _p95(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    return ordered[int(0.95 * (len(ordered) - 1))]


def _summary(values: list[float]) -> dict[str, object]:
    return {
        "median_ms": float(statistics.median(values)),
        "p95_ms": _p95(values),
        "samples_ms": values,
    }


def _serial_pipeline(
    host_batches: list[torch.Tensor], model: DecoderMLP, device_buffers: list[torch.Tensor]
) -> tuple[float, float]:
    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
    checksum = torch.zeros((), device="cuda")
    begin.record()
    for index, host_batch in enumerate(host_batches):
        slot = index % len(device_buffers)
        device_buffers[slot].copy_(host_batch, non_blocking=True)
        checksum = checksum + model(device_buffers[slot]).float().sum()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end)), float(checksum.item())


def _overlap_pipeline(
    host_batches: list[torch.Tensor], model: DecoderMLP, device_buffers: list[torch.Tensor]
) -> tuple[float, float]:
    compute_stream = torch.cuda.current_stream()
    copy_stream = torch.cuda.Stream()
    slots = len(device_buffers)
    copy_done = [torch.cuda.Event() for _ in range(slots)]
    compute_done = [torch.cuda.Event() for _ in range(slots)]
    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
    checksum = torch.zeros((), device="cuda")
    begin.record(compute_stream)
    copy_stream.wait_event(begin)
    for index, host_batch in enumerate(host_batches):
        slot = index % slots
        with torch.cuda.stream(copy_stream):
            if index >= slots:
                copy_stream.wait_event(compute_done[slot])
            device_buffers[slot].copy_(host_batch, non_blocking=True)
            copy_done[slot].record(copy_stream)
        compute_stream.wait_event(copy_done[slot])
        with torch.cuda.stream(compute_stream):
            checksum = checksum + model(device_buffers[slot]).float().sum()
            compute_done[slot].record(compute_stream)
    end.record(compute_stream)
    end.synchronize()
    return float(begin.elapsed_time(end)), float(checksum.item())


def _phase_service_times(
    host_batch: torch.Tensor, model: DecoderMLP, device_buffer: torch.Tensor, repeats: int
) -> dict[str, float]:
    copy_samples, compute_samples = [], []
    device_buffer.copy_(host_batch, non_blocking=True)
    torch.cuda.synchronize()
    for _ in range(repeats):
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        device_buffer.copy_(host_batch, non_blocking=True)
        end.record()
        end.synchronize()
        copy_samples.append(float(begin.elapsed_time(end)))

        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        with torch.no_grad():
            model(device_buffer)
        end.record()
        end.synchronize()
        compute_samples.append(float(begin.elapsed_time(end)))
    return {
        "h2d_median_ms": float(statistics.median(copy_samples)),
        "compute_median_ms": float(statistics.median(compute_samples)),
    }


def _correctness_output(
    host_batches: list[torch.Tensor], model: DecoderMLP, dtype: torch.dtype
) -> torch.Tensor:
    outputs = []
    with torch.no_grad():
        for host_batch in host_batches:
            outputs.append(model(host_batch.to("cuda", dtype=dtype, non_blocking=True)).float().cpu())
    return torch.cat(outputs, dim=0)


def run_candidate_composition_matrix(
    *,
    hidden_size: int,
    intermediate_size: int,
    token_counts: list[int],
    pipeline_batches: int,
    warmup_steps: int,
    repeat_steps: int,
    seed: int,
    variants: list[str],
) -> dict[str, object]:
    """Benchmark variants with rotating order inside every measured round."""
    if not torch.cuda.is_available():
        raise RuntimeError("真实候选矩阵需要 CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU 不支持 BF16；不能执行固定候选矩阵")
    allowed = {"baseline_fp32_serial", "bf16_serial", "fp32_overlap", "bf16_overlap"}
    if not variants or set(variants) - allowed:
        raise ValueError(f"variants 必须来自 {sorted(allowed)}")
    if any(value <= 0 for value in (hidden_size, intermediate_size, pipeline_batches, warmup_steps, repeat_steps)):
        raise ValueError("工作负载尺寸、预热和重复次数必须为正")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    reference = DecoderMLP(hidden_size, intermediate_size).eval()
    state = {name: tensor.detach().clone() for name, tensor in reference.state_dict().items()}
    host_inputs = {
        token_count: [
            torch.randn(token_count, hidden_size, dtype=torch.float32).pin_memory()
            for _ in range(pipeline_batches)
        ]
        for token_count in token_counts
    }
    if variants[0] != "baseline_fp32_serial":
        raise ValueError("variants 的第一项必须是 baseline_fp32_serial")

    models: dict[str, DecoderMLP] = {}
    variant_specs: dict[str, tuple[torch.dtype, bool]] = {}
    for variant in variants:
        dtype = torch.bfloat16 if variant.startswith("bf16") else torch.float32
        overlap = variant.endswith("overlap")
        model = DecoderMLP(hidden_size, intermediate_size).eval()
        model.load_state_dict(state)
        models[variant] = model.to(device="cuda", dtype=dtype)
        variant_specs[variant] = (dtype, overlap)

    records = {
        variant: {
            "dtype": str(dtype).replace("torch.", ""),
            "overlap": overlap,
            "loads": [],
        }
        for variant, (dtype, overlap) in variant_specs.items()
    }
    for token_count in token_counts:
        host_by_variant, buffers_by_variant = {}, {}
        for variant, (dtype, _) in variant_specs.items():
            host_by_variant[variant] = [
                batch if batch.dtype == dtype else batch.to(dtype=dtype).pin_memory()
                for batch in host_inputs[token_count]
            ]
            buffers_by_variant[variant] = [
                torch.empty(token_count, hidden_size, device="cuda", dtype=dtype) for _ in range(2)
            ]

        first_execution, checksums = {}, {}
        for variant in variants:
            _, overlap = variant_specs[variant]
            runner = _overlap_pipeline if overlap else _serial_pipeline
            torch.cuda.synchronize()
            started_at = time.perf_counter()
            first_cuda_ms, checksum = runner(
                host_by_variant[variant], models[variant], buffers_by_variant[variant]
            )
            torch.cuda.synchronize()
            first_execution[variant] = {
                "first_cuda_ms": first_cuda_ms,
                "first_wall_ms": (time.perf_counter() - started_at) * 1000.0,
            }
            checksums[variant] = checksum

        for warmup_round in range(warmup_steps):
            offset = warmup_round % len(variants)
            order = variants[offset:] + variants[:offset]
            for variant in order:
                _, overlap = variant_specs[variant]
                runner = _overlap_pipeline if overlap else _serial_pipeline
                runner(host_by_variant[variant], models[variant], buffers_by_variant[variant])

        samples = {variant: [] for variant in variants}
        torch.cuda.reset_peak_memory_stats()
        for trial in range(repeat_steps):
            offset = trial % len(variants)
            order = variants[offset:] + variants[:offset]
            for variant in order:
                _, overlap = variant_specs[variant]
                runner = _overlap_pipeline if overlap else _serial_pipeline
                elapsed_ms, checksum = runner(
                    host_by_variant[variant], models[variant], buffers_by_variant[variant]
                )
                samples[variant].append(elapsed_ms)
                checksums[variant] = checksum
        matrix_peak = int(torch.cuda.max_memory_allocated())

        outputs = {
            variant: _correctness_output(
                host_by_variant[variant], models[variant], variant_specs[variant][0]
            )
            for variant in variants
        }
        baseline_output = outputs["baseline_fp32_serial"]
        for variant in variants:
            dtype, _ = variant_specs[variant]
            raw_cosine = float(torch.nn.functional.cosine_similarity(
                baseline_output.flatten(), outputs[variant].flatten(), dim=0
            ).item())
            cosine = min(1.0, max(-1.0, raw_cosine))
            model_bytes = sum(
                parameter.numel() * parameter.element_size() for parameter in models[variant].parameters()
            )
            buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in buffers_by_variant[variant])
            services = _phase_service_times(
                host_by_variant[variant][0], models[variant], buffers_by_variant[variant][0], repeat_steps
            )
            records[variant]["loads"].append({
                "token_count": token_count,
                **first_execution[variant],
                "latency": _summary(samples[variant]),
                "service_times": services,
                "model_bytes": model_bytes,
                "buffer_bytes": buffer_bytes,
                "matrix_peak_allocated_bytes": matrix_peak,
                "cosine_vs_fp32": cosine,
                "checksum": checksums[variant],
            })
    return {
        "protocol": {
            "measurement_order": "rotating_order_within_each_round",
            "paired_rounds": repeat_steps,
        },
        "variants": records,
    }
