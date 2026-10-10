"""Load one quantized artifact with vLLM and write a project-82 loader record."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "failed",
        "framework": {
            "python": platform.python_version(),
            "torch": None,
            "cuda": None,
        },
        "hardware": {
            "cuda_available": False,
            "device": None,
            "total_memory_bytes": None,
        },
        "backend_version": None,
        "load_status": "failed",
        "roundtrip_quality": {"non_empty_generations": 0, "prompt_count": 2},
        "peak_memory_mb": None,
        "kernel": "not_verified_by_loader_gate",
        "fallback": "not_verified_by_loader_gate",
        "failure": None,
    }
    try:
        import torch
        from vllm import LLM, SamplingParams

        result["framework"].update({
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        })
        result["hardware"].update({
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "total_memory_bytes": (
                int(torch.cuda.get_device_properties(0).total_memory)
                if torch.cuda.is_available()
                else None
            ),
        })
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is unavailable")
        result["backend_version"] = importlib.metadata.version("vllm")
        free_before, _ = torch.cuda.mem_get_info()
        engine = LLM(
            model=str(Path(args.artifact).resolve()),
            dtype="auto",
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=False,
        )
        prompts = ["量化部署验证：请简要解释校准。", "Quantization deployment check:"]
        outputs = engine.generate(
            prompts,
            SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
            use_tqdm=False,
        )
        generated = [item.outputs[0].text if item.outputs else "" for item in outputs]
        free_after, _ = torch.cuda.mem_get_info()
        non_empty = sum(bool(text.strip()) for text in generated)
        result.update({
            "status": "ok" if non_empty == len(prompts) else "failed",
            "load_status": "loaded" if non_empty == len(prompts) else "roundtrip_failed",
            "roundtrip_quality": {
                "non_empty_generations": non_empty,
                "prompt_count": len(prompts),
            },
            "peak_memory_mb": round(max(0, free_before - free_after) / (1024 ** 2), 3),
            "failure": None if non_empty == len(prompts) else {"reason": "empty generation"},
        })
    except Exception as exc:
        result["failure"] = {"type": type(exc).__name__, "reason": str(exc)}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
