"""Shared OpenAI-compatible runtime for serving mechanism notebooks."""

from __future__ import annotations

import concurrent.futures
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from typing import Any, Iterable


def percentile(values: list[float], q: float) -> float | None:
    """Return a linearly interpolated percentile."""
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q / 100.0
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def probe_openai_backend(base_url: str, expected_model: str) -> dict[str, Any]:
    """Check the model endpoint and return the advertised model IDs."""
    url = base_url.rstrip("/") + "/v1/models"
    with urllib.request.urlopen(url, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    model_ids = [item.get("id") for item in payload.get("data", [])]
    if expected_model not in model_ids:
        raise RuntimeError(f"服务未暴露预期模型 {expected_model!r}；当前模型：{model_ids}")
    return {"url": url, "model_ids": model_ids}


def _parse_sse_line(line: bytes) -> dict[str, Any] | None:
    text = line.decode("utf-8", errors="replace").strip()
    if not text.startswith("data:") or text[5:].strip() == "[DONE]":
        return None
    try:
        return json.loads(text[5:].strip())
    except json.JSONDecodeError:
        return None


def _run_request(
    base_url: str,
    model: str,
    request_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    experiment_started: float,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        body["ignore_eos"] = True
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    usage: dict[str, Any] | None = None
    chunks = 0
    chunk_times: list[float] = []
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                payload = _parse_sse_line(raw_line)
                if payload is None:
                    continue
                if payload.get("usage"):
                    usage = payload["usage"]
                choices = payload.get("choices") or []
                content = ((choices[0].get("delta") or {}).get("content") or "") if choices else ""
                if content:
                    chunk_at = time.perf_counter()
                    first_token_at = first_token_at or chunk_at
                    chunk_times.append(chunk_at)
                    chunks += 1
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"request_id": request_id, "ok": False, "error": str(exc)}

    finished = time.perf_counter()
    first_token_at = first_token_at or finished
    output_tokens = (usage or {}).get("completion_tokens") or max(1, chunks)
    ttft_ms = (first_token_at - started) * 1000.0
    e2e_ms = (finished - started) * 1000.0
    stream_gaps_ms = [
        (current - previous) * 1000.0
        for previous, current in zip(chunk_times, chunk_times[1:])
    ]
    return {
        "request_id": request_id,
        "ok": True,
        "submitted_ms": round((started - experiment_started) * 1000.0, 3),
        "completed_ms": round((finished - experiment_started) * 1000.0, 3),
        "ttft_ms": round(ttft_ms, 3),
        "tpot_ms": round((e2e_ms - ttft_ms) / max(1, output_tokens - 1), 3),
        "e2e_ms": round(e2e_ms, 3),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "output_tokens": output_tokens,
        "stream_chunks": chunks,
        "stream_gap_ms": {
            "median": round(statistics.median(stream_gaps_ms), 3) if stream_gaps_ms else None,
            "p95": round(percentile(stream_gaps_ms, 95), 3) if stream_gaps_ms else None,
            "max": round(max(stream_gaps_ms), 3) if stream_gaps_ms else None,
        },
    }


def _run_group(
    requests: list[dict[str, Any]],
    *,
    concurrency: int,
    request_fn: Any,
) -> list[dict[str, Any]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(request_fn, item) for item in requests]
        return [future.result() for future in futures]


def summarize_schedule(results: list[dict[str, Any]], duration_s: float) -> dict[str, Any]:
    """Summarize request-level streaming evidence without discarding raw rows."""
    successful = [item for item in results if item.get("ok")]
    failed = [item for item in results if not item.get("ok")]
    output_tokens = sum(int(item["output_tokens"]) for item in successful)

    def distribution(name: str) -> dict[str, float | None]:
        values = [float(item[name]) for item in successful]
        return {
            "median": round(statistics.median(values), 3) if values else None,
            "p95": round(percentile(values, 95), 3) if values else None,
            "p99": round(percentile(values, 99), 3) if values else None,
        }

    return {
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "duration_s": round(duration_s, 4),
        "request_throughput_per_s": round(len(successful) / duration_s, 4) if duration_s else 0.0,
        "output_token_throughput_per_s": round(output_tokens / duration_s, 4) if duration_s else 0.0,
        "ttft_ms": distribution("ttft_ms"),
        "tpot_ms": distribution("tpot_ms"),
        "e2e_ms": distribution("e2e_ms"),
        "completion_order": [item["request_id"] for item in sorted(successful, key=lambda row: row["completed_ms"])],
    }


def run_arrival_schedule(
    *,
    base_url: str,
    model: str,
    prompts: Iterable[str | dict[str, Any]],
    schedule: str,
    concurrency: int,
    max_tokens: int,
    warmup: int = 1,
    temperature: float = 0.0,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    """Compare barrier waves with rolling replacement on one fixed backend."""
    prompt_list = list(prompts)
    if schedule not in {"barrier_waves", "rolling_admission"}:
        raise ValueError("schedule 必须是 barrier_waves 或 rolling_admission")
    if concurrency <= 0 or max_tokens <= 0 or not prompt_list:
        raise ValueError("concurrency、max_tokens 和 prompts 必须有效")
    probe = probe_openai_backend(base_url, model)
    for index in range(warmup):
        warmup_started = time.perf_counter()
        first_prompt = prompt_list[0]["prompt"] if isinstance(prompt_list[0], dict) else prompt_list[0]
        warmup_result = _run_request(
            base_url, model, f"warmup-{index}", str(first_prompt), min(max_tokens, 8),
            temperature, timeout_s, warmup_started,
        )
        if not warmup_result.get("ok"):
            raise RuntimeError(f"预热请求失败：{warmup_result.get('error')}")

    indexed = []
    for index, item in enumerate(prompt_list):
        if isinstance(item, dict):
            prompt = str(item["prompt"])
            request_max_tokens = int(item.get("max_tokens", max_tokens))
            arrival_delay_s = float(item.get("arrival_delay_s", 0.0))
            request_role = str(item.get("request_role", "default"))
            ignore_eos = bool(item.get("ignore_eos", False))
        else:
            prompt = str(item)
            request_max_tokens = max_tokens
            arrival_delay_s = 0.0
            request_role = "default"
            ignore_eos = False
        if request_max_tokens <= 0:
            raise ValueError("每个请求的 max_tokens 必须为正数")
        if arrival_delay_s < 0:
            raise ValueError("arrival_delay_s 不能为负数")
        indexed.append({
            "request_id": f"request-{index:03d}",
            "prompt": prompt,
            "max_tokens": request_max_tokens,
            "arrival_delay_s": arrival_delay_s,
            "request_role": request_role,
            "ignore_eos": ignore_eos,
        })
    experiment_started = time.perf_counter()

    def request_fn(item: dict[str, Any]) -> dict[str, Any]:
        remaining_delay = item["arrival_delay_s"] - (time.perf_counter() - experiment_started)
        if remaining_delay > 0:
            time.sleep(remaining_delay)
        result = _run_request(
            base_url, model, item["request_id"], item["prompt"], int(item["max_tokens"]),
            temperature, timeout_s, experiment_started, ignore_eos=bool(item["ignore_eos"]),
        )
        result["request_role"] = item["request_role"]
        result["scheduled_arrival_ms"] = round(item["arrival_delay_s"] * 1000.0, 3)
        return result

    if schedule == "barrier_waves":
        results: list[dict[str, Any]] = []
        for start in range(0, len(indexed), concurrency):
            results.extend(_run_group(indexed[start:start + concurrency], concurrency=concurrency, request_fn=request_fn))
    else:
        results = _run_group(indexed, concurrency=concurrency, request_fn=request_fn)

    duration_s = time.perf_counter() - experiment_started
    return {
        "schedule": schedule,
        "backend_probe": probe,
        "request_count": len(indexed),
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "metrics": summarize_schedule(results, duration_s),
        "requests": results,
    }
