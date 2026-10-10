"""Small execution adapters for project notebooks.

Adapters translate a project-neutral ``ExperimentSpec`` into one backend call.
They do not select candidates, change workloads, install dependencies, or turn
a smoke result into benchmark evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Mapping, Protocol

from .project_contracts import make_run_result


class ExecutionAdapter(Protocol):
    """Minimal interface implemented by PyTorch, Transformers, or serving runners."""

    name: str

    def preflight(self, spec: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def run(self, spec: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class CallableExecutionAdapter:
    """Wrap notebook callables without introducing a framework dependency."""

    name: str
    runner: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    preflight_runner: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None

    def preflight(self, spec: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.preflight_runner is None:
            return {"status": "ok", "adapter": self.name}
        return dict(self.preflight_runner(spec))

    def run(self, spec: Mapping[str, Any]) -> Mapping[str, Any]:
        return dict(self.runner(spec))


@dataclass(frozen=True)
class CommandExecutionAdapter:
    """Run an explicit command that writes one project result JSON.

    The command is a sequence rather than a shell string.  This keeps quoting
    predictable and prevents a notebook configuration field from becoming an
    implicit shell program.
    """

    name: str
    command: tuple[str, ...]
    output_path: str | Path

    def preflight(self, spec: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self.command:
            return {"status": "blocked", "ready": False, "reason": "adapter command is empty"}
        executable = self.command[0]
        if not (Path(executable).is_file() or shutil.which(executable)):
            return {
                "status": "blocked",
                "ready": False,
                "reason": f"adapter executable not found: {executable}",
            }
        output = Path(self.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        return {"status": "ok", "ready": True, "adapter": self.name, "output_path": str(output)}

    def run(self, spec: Mapping[str, Any]) -> Mapping[str, Any]:
        completed = subprocess.run(list(self.command), capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            return {
                "status": "failed",
                "failure": {
                    "stage": "execution",
                    "returncode": completed.returncode,
                    "stderr": completed.stderr[-2000:],
                },
                "metrics": {},
                "decision": "pending",
            }
        output = Path(self.output_path)
        if not output.is_file():
            return {
                "status": "failed",
                "failure": {"stage": "result_read", "message": f"adapter result missing: {output}"},
                "metrics": {},
                "decision": "pending",
            }
        payload = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("adapter result must be a JSON object")
        return dict(payload)


def execute_with_adapter(
    adapter: ExecutionAdapter,
    spec: Mapping[str, Any],
    *,
    default_evidence_level: str = "gpu_smoke",
) -> dict[str, Any]:
    """Execute one adapter and normalize success, blocked, and failure records."""

    preflight = dict(adapter.preflight(spec))
    if preflight.get("status") == "blocked" or preflight.get("ready") is False:
        return make_run_result(
            spec=spec,
            status="blocked",
            evidence_level=default_evidence_level,
            metrics={},
            failure={
                "stage": "preflight",
                "adapter": adapter.name,
                "details": preflight,
            },
            decision="pending",
            next_action="按预检提示修复环境后复测",
        )

    try:
        payload = dict(adapter.run(spec))
    except Exception as exc:
        return make_run_result(
            spec=spec,
            status="failed",
            evidence_level=default_evidence_level,
            metrics={},
            failure={
                "stage": "execution",
                "adapter": adapter.name,
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
            decision="pending",
            next_action="保留失败记录，修复后使用同一 experiment_id 复测",
        )

    status = str(payload.get("status", "ok"))
    failure = dict(payload.get("failure", {}))
    if status == "failed" and not failure:
        failure = {
            "stage": "execution",
            "adapter": adapter.name,
            "message": "adapter 返回 failed，但未提供专项失败详情",
        }
    return make_run_result(
        spec=spec,
        status=status,
        evidence_level=str(payload.get("evidence_level", default_evidence_level)),
        metrics=dict(payload.get("metrics", {})),
        strategy_metrics=dict(payload.get("strategy_metrics", {})),
        quality=dict(payload.get("quality", {})),
        artifacts=list(payload.get("artifacts", [])),
        failure=failure,
        decision=str(payload.get("decision", "pending")),
        decision_reason=payload.get("decision_reason"),
        next_action=payload.get("next_action"),
    )
