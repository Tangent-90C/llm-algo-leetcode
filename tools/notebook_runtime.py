"""Shared infrastructure helpers for runnable tutorial notebooks.

The helpers in this module own environment, repository, download identity and
result-I/O boilerplate.  They deliberately do not choose experiment
candidates, thresholds or decisions; those remain visible in each notebook.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import random
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping

from tools.project_runtime import ensure_output_path, resolve_project_root


def package_version(distribution: str) -> str | None:
    """Return an installed distribution version without importing it."""

    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def repository_output_path(relative_path: str | Path) -> Path:
    """Resolve a repository-relative output path and create its parent."""

    return ensure_output_path(resolve_project_root(), relative_path)


def proxy_transport_status() -> dict[str, Any]:
    """Report SOCKS transport availability without exposing proxy secrets."""

    proxy = os.getenv("ALL_PROXY") or os.getenv("all_proxy")
    scheme = proxy.partition(":")[0].lower() if proxy else None
    return {
        "all_proxy_configured": bool(proxy),
        "proxy_scheme": scheme,
        "socks_support_available": importlib.util.find_spec("socksio") is not None,
    }


def resolve_model_commit(
    model_id: str,
    requested_revision: str,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> tuple[str, str]:
    """Resolve a model branch or tag to the immutable commit used by a run."""

    if local_files_only:
        from huggingface_hub.constants import HF_HUB_CACHE

        hub_cache = Path(cache_dir) if cache_dir else Path(HF_HUB_CACHE)
        repo_cache = hub_cache / f"models--{model_id.replace('/', '--')}"
        ref_path = repo_cache / "refs" / requested_revision
        snapshots_dir = repo_cache / "snapshots"
        resolved = ref_path.read_text(encoding="utf-8").strip() if ref_path.is_file() else ""
        source = "local_cache_ref"

        if not resolved and (snapshots_dir / requested_revision).is_dir():
            resolved = requested_revision
            source = "local_cache_commit"

        if not resolved and snapshots_dir.is_dir():
            cached_snapshots = sorted(path.name for path in snapshots_dir.iterdir() if path.is_dir())
            if len(cached_snapshots) == 1:
                resolved = cached_snapshots[0]
                source = "local_cache_single_snapshot"

        if not resolved or not (snapshots_dir / resolved).is_dir():
            raise FileNotFoundError(
                f"本地缓存无法解析 {model_id}@{requested_revision}；"
                "请联网运行一次、填写已缓存的 commit hash，"
                "或清理多个旧 snapshot 后重试"
            )
    else:
        from huggingface_hub import HfApi

        resolved = HfApi().model_info(
            repo_id=model_id,
            revision=requested_revision,
        ).sha
        source = "huggingface_hub_api"

    if not resolved:
        raise RuntimeError(f"无法将 {model_id}@{requested_revision} 解析为 commit hash")
    return str(resolved), source


def stable_sample(dataset: Any, count: int, seed: int, *, text_column: str = "text"):
    """Select reproducible non-empty rows and return their original indices."""

    eligible = [
        index
        for index, text in enumerate(dataset[text_column])
        if text and str(text).strip()
    ]
    if len(eligible) < count:
        raise ValueError(f"可用样本 {len(eligible)} 少于请求数量 {count}")
    selected = random.Random(seed).sample(eligible, count)
    return dataset.select(selected), selected


def hash_ids(ids: Iterable[Any]) -> str:
    """Return the stable SHA-256 identity of an ordered sample-ID sequence."""

    payload = ",".join(map(str, ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def failure_record(stage: str, error: BaseException, *, traceback_lines: int = 12) -> dict[str, Any]:
    """Create a compact JSON-safe failure record for an experiment stage."""

    lines = traceback.format_exception(type(error), error, error.__traceback__)
    rendered = "".join(lines).strip().splitlines()
    return {
        "stage": stage,
        "type": type(error).__name__,
        "message": str(error),
        "traceback_tail": rendered[-traceback_lines:],
    }


def save_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write one UTF-8 JSON result with a trailing newline."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output
