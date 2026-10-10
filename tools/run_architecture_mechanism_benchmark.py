"""Run controlled single-GPU mechanism benchmarks for architecture lessons.

The runner measures representation and state-update costs.  It deliberately
does not claim model-quality or production-backend evidence; those belong to
the architecture project notebooks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("列表必须包含正整数")
    return values


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("kv_representation", "linear_recurrence", "memory_growth"),
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--semantic-id", required=True)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--run-label", default="default")
    parser.add_argument("--profile-id", default="ARCH-CUDA-MECHANISM")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--sequence-lengths", type=_parse_int_list, default=[512, 2048, 8192])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--query-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--attention-every", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    return parser.parse_args()


def _workload_record(args: argparse.Namespace) -> dict:
    """Keep experiment inputs separate from CLI identity and output fields."""

    workload = {
        "dtype": args.dtype,
        "sequence_lengths": args.sequence_lengths,
        "batch_size": args.batch_size,
        "query_heads": args.query_heads,
        "head_dim": args.head_dim,
        "warmup": args.warmup,
        "repeats": args.repeats,
    }
    if args.mode == "kv_representation":
        workload.update({
            "kv_heads": args.kv_heads,
            "latent_dim": args.latent_dim,
            "rope_dim": args.rope_dim,
        })
    elif args.mode == "memory_growth":
        workload.update({
            "num_layers": args.num_layers,
            "attention_every": args.attention_every,
        })
    return workload


def _measure(torch, fn, warmup: int, repeats: int, device) -> tuple[float, float, bool]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup 必须非负且 repeats 必须为正整数")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    output = None
    start.record()
    for _ in range(repeats):
        output = fn()
    end.record()
    torch.cuda.synchronize(device)
    output_finite = bool(torch.isfinite(output).all().item())
    if not output_finite:
        raise RuntimeError("GPU 机制输出包含非有限值")
    return (
        start.elapsed_time(end) / repeats,
        torch.cuda.max_memory_allocated(device) / 1024**2,
        output_finite,
    )


def _sdpa(torch, query, key, value, enable_gqa: bool = False):
    if enable_gqa:
        try:
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, enable_gqa=True
            )
        except TypeError as exc:
            raise RuntimeError("当前 PyTorch 不支持 SDPA enable_gqa；请升级后复测") from exc
    return torch.nn.functional.scaled_dot_product_attention(query, key, value)


def _run_kv_representation(torch, args, device, dtype) -> dict:
    if args.query_heads % args.kv_heads:
        raise ValueError("query-heads 必须能被 kv-heads 整除")
    if args.latent_dim <= 0 or args.rope_dim <= 0:
        raise ValueError("latent-dim 与 rope-dim 必须为正整数")
    rows = []
    element_size = torch.tensor([], dtype=dtype).element_size()
    for sequence_length in args.sequence_lengths:
        torch.manual_seed(701 + sequence_length)
        query = torch.randn(
            args.batch_size, args.query_heads, 1, args.head_dim, device=device, dtype=dtype
        )

        mha_key = torch.randn(
            args.batch_size, args.query_heads, sequence_length, args.head_dim,
            device=device, dtype=dtype,
        )
        mha_value = torch.randn_like(mha_key)
        elapsed_ms, peak_mb, output_finite = _measure(
            torch, lambda: _sdpa(torch, query, mha_key, mha_value),
            args.warmup, args.repeats, device,
        )
        rows.append({
            "sequence_length": sequence_length,
            "representation": "mha_full_kv",
            "cache_bytes": 2 * mha_key.numel() * element_size,
            "materialized_kv_bytes": 2 * mha_key.numel() * element_size,
            "mean_step_ms": round(elapsed_ms, 5),
            "peak_memory_mb": round(peak_mb, 3),
            "reconstruction": False,
            "output_finite": output_finite,
        })
        del mha_key, mha_value

        gqa_key = torch.randn(
            args.batch_size, args.kv_heads, sequence_length, args.head_dim,
            device=device, dtype=dtype,
        )
        gqa_value = torch.randn_like(gqa_key)
        elapsed_ms, peak_mb, output_finite = _measure(
            torch, lambda: _sdpa(torch, query, gqa_key, gqa_value, enable_gqa=True),
            args.warmup, args.repeats, device,
        )
        rows.append({
            "sequence_length": sequence_length,
            "representation": "gqa_shared_kv",
            "cache_bytes": 2 * gqa_key.numel() * element_size,
            "materialized_kv_bytes": 2 * gqa_key.numel() * element_size,
            "mean_step_ms": round(elapsed_ms, 5),
            "peak_memory_mb": round(peak_mb, 3),
            "reconstruction": False,
            "output_finite": output_finite,
        })
        del gqa_key, gqa_value

        latent = torch.randn(
            args.batch_size, sequence_length, args.latent_dim, device=device, dtype=dtype
        )
        rope_state = torch.randn(
            args.batch_size, sequence_length, args.rope_dim, device=device, dtype=dtype
        )
        key_projection = torch.randn(
            args.latent_dim, args.kv_heads * args.head_dim, device=device, dtype=dtype
        ) / args.latent_dim**0.5
        value_projection = torch.randn_like(key_projection)

        def latent_reconstruction_attention():
            key = (latent @ key_projection).view(
                args.batch_size, sequence_length, args.kv_heads, args.head_dim
            ).transpose(1, 2)
            value = (latent @ value_projection).view(
                args.batch_size, sequence_length, args.kv_heads, args.head_dim
            ).transpose(1, 2)
            return _sdpa(torch, query, key, value, enable_gqa=True)

        elapsed_ms, peak_mb, output_finite = _measure(
            torch, latent_reconstruction_attention, args.warmup, args.repeats, device
        )
        rows.append({
            "sequence_length": sequence_length,
            "representation": "latent_kv_reconstruction_proxy",
            "cache_bytes": (latent.numel() + rope_state.numel()) * element_size,
            "materialized_kv_bytes": (
                2 * args.batch_size * args.kv_heads
                * sequence_length * args.head_dim * element_size
            ),
            "mean_step_ms": round(elapsed_ms, 5),
            "peak_memory_mb": round(peak_mb, 3),
            "reconstruction": True,
            "output_finite": output_finite,
        })

        # Absorb the up-projection into query scoring and context projection.
        # This is algebraically equivalent to the content-only latent path but
        # deliberately does not materialize a full historical K/V tensor.
        query_heads_per_kv_head = args.query_heads // args.kv_heads
        absorbed_key_projection = (
            key_projection.view(args.latent_dim, args.kv_heads, args.head_dim)
            .permute(1, 2, 0)
            .repeat_interleave(query_heads_per_kv_head, dim=0)
        )
        absorbed_value_projection = (
            value_projection.view(args.latent_dim, args.kv_heads, args.head_dim)
            .permute(1, 0, 2)
            .repeat_interleave(query_heads_per_kv_head, dim=0)
        )

        def latent_absorbed_attention():
            latent_query = torch.einsum(
                "bhqd,hdl->bhql", query, absorbed_key_projection
            )
            scores = torch.matmul(
                latent_query, latent.transpose(1, 2).unsqueeze(1)
            ) * (args.head_dim ** -0.5)
            weights = torch.softmax(scores, dim=-1)
            latent_context = torch.matmul(weights, latent.unsqueeze(1))
            return torch.einsum(
                "bhql,hld->bhqd", latent_context, absorbed_value_projection
            )

        elapsed_ms, peak_mb, output_finite = _measure(
            torch, latent_absorbed_attention, args.warmup, args.repeats, device
        )
        reconstruction_output = latent_reconstruction_attention()
        absorbed_output = latent_absorbed_attention()
        cosine_vs_reconstruction = torch.nn.functional.cosine_similarity(
            reconstruction_output.float().reshape(1, -1),
            absorbed_output.float().reshape(1, -1),
        ).item()
        max_abs_diff_vs_reconstruction = (
            reconstruction_output.float() - absorbed_output.float()
        ).abs().max().item()
        rows.append({
            "sequence_length": sequence_length,
            "representation": "latent_kv_absorbed_proxy",
            "cache_bytes": (latent.numel() + rope_state.numel()) * element_size,
            "materialized_kv_bytes": 0,
            "mean_step_ms": round(elapsed_ms, 5),
            "peak_memory_mb": round(peak_mb, 3),
            "reconstruction": False,
            "cosine_vs_reconstruction": round(cosine_vs_reconstruction, 8),
            "max_abs_diff_vs_reconstruction": round(
                max_abs_diff_vs_reconstruction, 8
            ),
            "output_finite": output_finite,
        })
        del latent, rope_state, key_projection, value_projection, query
        del absorbed_key_projection, absorbed_value_projection
        torch.cuda.empty_cache()
    return {"comparison_type": "controlled_content_path_proxy", "rows": rows}


def _run_memory_growth(torch, args, device, dtype) -> dict:
    if args.attention_every <= 0 or args.num_layers <= 0:
        raise ValueError("num-layers 与 attention-every 必须为正整数")
    hybrid_attention_layers = max(1, args.num_layers // args.attention_every)
    hybrid_recurrent_layers = args.num_layers - hybrid_attention_layers
    rows = []
    element_size = torch.tensor([], dtype=dtype).element_size()
    for sequence_length in args.sequence_lengths:
        # Full-Attention candidate: allocate one independent K/V history per layer.
        torch.manual_seed(801 + sequence_length)
        query = torch.randn(
            args.batch_size, args.query_heads, 1, args.head_dim, device=device, dtype=dtype
        )
        key_history = torch.randn(
            args.num_layers, args.batch_size, args.query_heads,
            sequence_length, args.head_dim,
            device=device, dtype=dtype,
        )
        value_history = torch.randn_like(key_history)

        def attention_step():
            output = None
            for layer in range(args.num_layers):
                output = _sdpa(torch, query, key_history[layer], value_history[layer])
            return output

        elapsed_ms, peak_mb, output_finite = _measure(
            torch, attention_step, args.warmup, args.repeats, device
        )
        rows.append({
            "sequence_length": sequence_length,
            "representation": "explicit_attention",
            "state_bytes": (key_history.numel() + value_history.numel()) * element_size,
            "mean_step_ms": round(elapsed_ms, 5),
            "peak_memory_mb": round(peak_mb, 3),
            "attention_layers": args.num_layers,
            "output_finite": output_finite,
        })
        del query, key_history, value_history
        torch.cuda.empty_cache()

        # Fully recurrent candidate: one fixed matrix state per layer.
        torch.manual_seed(801 + sequence_length)
        query = torch.randn(
            args.batch_size, args.query_heads, 1, args.head_dim, device=device, dtype=dtype
        )
        recurrent_state = torch.randn(
            args.num_layers, args.batch_size, args.query_heads,
            args.head_dim, args.head_dim, device=device, dtype=dtype,
        )
        key_now = torch.randn(
            args.num_layers, args.batch_size, args.query_heads, args.head_dim,
            device=device, dtype=dtype,
        )
        value_now = torch.randn_like(key_now)

        def recurrent_step():
            output = None
            for layer in range(args.num_layers):
                update = key_now[layer].unsqueeze(-1) * value_now[layer].unsqueeze(-2)
                updated_state = 0.99 * recurrent_state[layer] + update
                output = torch.matmul(query, updated_state)
            return output

        elapsed_ms, peak_mb, output_finite = _measure(
            torch, recurrent_step, args.warmup, args.repeats, device
        )
        rows.append({
            "sequence_length": sequence_length,
            "representation": "fixed_recurrent_state",
            "state_bytes": recurrent_state.numel() * element_size,
            "mean_step_ms": round(elapsed_ms, 5),
            "peak_memory_mb": round(peak_mb, 3),
            "attention_layers": 0,
            "output_finite": output_finite,
        })
        del query, recurrent_state, key_now, value_now
        torch.cuda.empty_cache()

        # Hybrid candidate: allocate only the states used by its layer plan.
        torch.manual_seed(801 + sequence_length)
        query = torch.randn(
            args.batch_size, args.query_heads, 1, args.head_dim, device=device, dtype=dtype
        )
        hybrid_key_history = torch.randn(
            hybrid_attention_layers, args.batch_size, args.query_heads,
            sequence_length, args.head_dim, device=device, dtype=dtype,
        )
        hybrid_value_history = torch.randn_like(hybrid_key_history)
        hybrid_recurrent_state = torch.randn(
            hybrid_recurrent_layers, args.batch_size, args.query_heads,
            args.head_dim, args.head_dim, device=device, dtype=dtype,
        )
        hybrid_key_now = torch.randn(
            hybrid_recurrent_layers, args.batch_size, args.query_heads, args.head_dim,
            device=device, dtype=dtype,
        )
        hybrid_value_now = torch.randn_like(hybrid_key_now)

        def hybrid_step():
            output = None
            for layer in range(hybrid_attention_layers):
                output = _sdpa(
                    torch, query,
                    hybrid_key_history[layer], hybrid_value_history[layer],
                )
            for layer in range(hybrid_recurrent_layers):
                update = (
                    hybrid_key_now[layer].unsqueeze(-1)
                    * hybrid_value_now[layer].unsqueeze(-2)
                )
                updated_state = 0.99 * hybrid_recurrent_state[layer] + update
                output = torch.matmul(query, updated_state)
            return output

        elapsed_ms, peak_mb, output_finite = _measure(
            torch, hybrid_step, args.warmup, args.repeats, device
        )
        rows.append({
            "sequence_length": sequence_length,
            "representation": "hybrid_attention_recurrent",
            "state_bytes": (
                hybrid_key_history.numel()
                + hybrid_value_history.numel()
                + hybrid_recurrent_state.numel()
            ) * element_size,
            "mean_step_ms": round(elapsed_ms, 5),
            "peak_memory_mb": round(peak_mb, 3),
            "attention_layers": hybrid_attention_layers,
            "output_finite": output_finite,
        })
        del (
            query, hybrid_key_history, hybrid_value_history,
            hybrid_recurrent_state, hybrid_key_now, hybrid_value_now,
        )
        torch.cuda.empty_cache()
    return {"comparison_type": "controlled_state_interface_proxy", "rows": rows}


def _run_linear_recurrence(torch, args, device, dtype) -> dict:
    """Compare one-layer explicit history with a fixed recurrent state."""
    rows = []
    element_size = torch.tensor([], dtype=dtype).element_size()
    for sequence_length in args.sequence_lengths:
        # Measure each representation with only its own live state.  Keeping
        # both candidates allocated would make peak memory incomparable.
        torch.manual_seed(751 + sequence_length)
        query = torch.randn(
            args.batch_size, args.query_heads, 1, args.head_dim, device=device, dtype=dtype
        )
        key_history = torch.randn(
            args.batch_size, args.query_heads, sequence_length, args.head_dim,
            device=device, dtype=dtype,
        )
        value_history = torch.randn_like(key_history)
        def explicit_history_step():
            return _sdpa(torch, query, key_history, value_history)

        elapsed_ms, peak_mb, output_finite = _measure(
            torch, explicit_history_step, args.warmup, args.repeats, device
        )
        rows.append({
            "sequence_length": sequence_length,
            "representation": "explicit_attention_history",
            "state_bytes": 2 * key_history.numel() * element_size,
            "mean_step_ms": round(elapsed_ms, 5),
            "peak_memory_mb": round(peak_mb, 3),
            "output_finite": output_finite,
        })
        del query, key_history, value_history
        torch.cuda.empty_cache()

        torch.manual_seed(751 + sequence_length)
        query = torch.randn(
            args.batch_size, args.query_heads, 1, args.head_dim, device=device, dtype=dtype
        )
        recurrent_state = torch.randn(
            args.batch_size, args.query_heads, args.head_dim, args.head_dim,
            device=device, dtype=dtype,
        )
        key_now = torch.randn(
            args.batch_size, args.query_heads, args.head_dim, device=device, dtype=dtype
        )
        value_now = torch.randn_like(key_now)

        def recurrent_state_step():
            update = key_now.unsqueeze(-1) * value_now.unsqueeze(-2)
            updated_state = 0.99 * recurrent_state + update
            return torch.matmul(query, updated_state)

        elapsed_ms, peak_mb, output_finite = _measure(
            torch, recurrent_state_step, args.warmup, args.repeats, device
        )
        rows.append({
            "sequence_length": sequence_length,
            "representation": "fixed_recurrent_state",
            "state_bytes": recurrent_state.numel() * element_size,
            "mean_step_ms": round(elapsed_ms, 5),
            "peak_memory_mb": round(peak_mb, 3),
            "output_finite": output_finite,
        })
        del query, recurrent_state, key_now, value_now
        torch.cuda.empty_cache()
    return {"comparison_type": "controlled_single_layer_state_proxy", "rows": rows}


def main() -> int:
    args = _args()
    import torch

    result = {
        "schema_version": "architecture-mechanism-gpu/v1",
        "identity": {
            "semantic_id": args.semantic_id,
            "workload_id": args.workload_id,
            "run_label": args.run_label,
            "profile_id": args.profile_id,
        },
        "mechanism": args.mode,
        "workload": _workload_record(args),
        "framework": {"torch": torch.__version__},
        "hardware": {},
        "baseline": {},
        "candidates": [],
        "metrics": {},
        "failure": None,
        "evidence_level": "environment_preflight",
        "decision": "not_started",
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("架构机制 GPU 实验需要 CUDA GPU")
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 GPU 不支持 BF16；请显式改用 float32，并作为新的运行记录保存")
        dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
        result["hardware"] = {
            "gpu": torch.cuda.get_device_name(device),
            "cuda": torch.version.cuda,
            "dtype": args.dtype,
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        }
        if args.mode == "kv_representation":
            result["baseline"] = {"name": "mha_full_kv"}
            result["candidates"] = [
                "gqa_shared_kv",
                "latent_kv_reconstruction_proxy",
                "latent_kv_absorbed_proxy",
            ]
            result["metrics"] = _run_kv_representation(torch, args, device, dtype)
        elif args.mode == "linear_recurrence":
            result["baseline"] = {"name": "explicit_attention_history"}
            result["candidates"] = ["fixed_recurrent_state"]
            result["metrics"] = _run_linear_recurrence(torch, args, device, dtype)
        else:
            result["baseline"] = {"name": "explicit_attention"}
            result["candidates"] = [
                "fixed_recurrent_state", "hybrid_attention_recurrent"
            ]
            result["metrics"] = _run_memory_growth(torch, args, device, dtype)
        result["evidence_level"] = "single_gpu_mechanism_benchmark"
        if args.mode == "kv_representation":
            rows = result["metrics"]["rows"]
            grouped = {
                sequence_length: {
                    row["representation"]: row
                    for row in rows
                    if row["sequence_length"] == sequence_length
                }
                for sequence_length in args.sequence_lengths
            }
            compact = all(
                group["latent_kv_reconstruction_proxy"]["cache_bytes"]
                == group["latent_kv_absorbed_proxy"]["cache_bytes"]
                < group["gqa_shared_kv"]["cache_bytes"]
                < group["mha_full_kv"]["cache_bytes"]
                for group in grouped.values()
            )
            absorbed_without_materialized_kv = all(
                group["latent_kv_reconstruction_proxy"]["materialized_kv_bytes"] > 0
                and group["latent_kv_absorbed_proxy"]["materialized_kv_bytes"] == 0
                for group in grouped.values()
            )
            absorbed_matches_reconstruction = all(
                group["latent_kv_absorbed_proxy"]["cosine_vs_reconstruction"] >= 0.99
                for group in grouped.values()
            )
            finite = all(row["output_finite"] for row in rows)
            result["decision"] = (
                "accept_latent_representation_and_absorbed_path_mechanism"
                if (
                    compact
                    and absorbed_without_materialized_kv
                    and absorbed_matches_reconstruction
                    and finite
                )
                else "reject_latent_representation_or_absorbed_path_mechanism"
            )
        elif args.mode == "linear_recurrence":
            rows = result["metrics"]["rows"]
            explicit = [row for row in rows if row["representation"] == "explicit_attention_history"]
            recurrent = [row for row in rows if row["representation"] == "fixed_recurrent_state"]
            fixed_state = len({row["state_bytes"] for row in recurrent}) == 1
            growing_history = all(
                later["state_bytes"] > earlier["state_bytes"]
                for earlier, later in zip(explicit, explicit[1:])
            )
            finite = all(row["output_finite"] for row in rows)
            result["decision"] = (
                "accept_fixed_state_growth_mechanism"
                if fixed_state and growing_history and finite
                else "reject_state_growth_mechanism"
            )
        else:
            rows = result["metrics"]["rows"]
            grouped = {
                sequence_length: {
                    row["representation"]: row
                    for row in rows
                    if row["sequence_length"] == sequence_length
                }
                for sequence_length in args.sequence_lengths
            }
            recurrent_state_is_fixed = len({
                group["fixed_recurrent_state"]["state_bytes"]
                for group in grouped.values()
            }) == 1
            hybrid_is_between = all(
                group["fixed_recurrent_state"]["state_bytes"]
                < group["hybrid_attention_recurrent"]["state_bytes"]
                < group["explicit_attention"]["state_bytes"]
                for group in grouped.values()
            )
            full_attention_growth = (
                grouped[args.sequence_lengths[-1]]["explicit_attention"]["state_bytes"]
                - grouped[args.sequence_lengths[0]]["explicit_attention"]["state_bytes"]
            )
            hybrid_growth = (
                grouped[args.sequence_lengths[-1]]["hybrid_attention_recurrent"]["state_bytes"]
                - grouped[args.sequence_lengths[0]]["hybrid_attention_recurrent"]["state_bytes"]
            )
            finite = all(row["output_finite"] for row in rows)
            result["decision"] = (
                "accept_hybrid_state_growth_mechanism"
                if (
                    recurrent_state_is_fixed
                    and hybrid_is_between
                    and hybrid_growth < full_attention_growth
                    and finite
                )
                else "reject_hybrid_state_growth_mechanism"
            )
    except Exception as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
        result["decision"] = "mechanism_measurement_failed"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["failure"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
