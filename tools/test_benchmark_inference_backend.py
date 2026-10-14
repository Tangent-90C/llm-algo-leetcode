from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools import benchmark_inference_backend as benchmark_module
from tools import backend_runtime
from tools import inference_project_runtime


class PriorityMetricTests(unittest.TestCase):
    def test_first_token_score_uses_first_token_rank(self) -> None:
        results = [
            {"request_id": "high-1", "priority": 0, "first_token_rank": 1},
            {"request_id": "low-1", "priority": 5, "first_token_rank": 2},
            {"request_id": "high-2", "priority": 0, "first_token_rank": 3},
            {"request_id": "low-2", "priority": 5, "first_token_rank": 4},
        ]
        self.assertEqual(
            benchmark_module._priority_pair_score(results, "first_token_rank"),
            0.75,
        )

    def test_repeated_report_marks_request_result_scope(self) -> None:
        reports = [
            {
                "workload": {"repeats": 1},
                "metrics": {"priority_first_token_order_score": score},
                "results": [{"request_id": f"run-{index}"}],
            }
            for index, score in enumerate((0.5, 1.0), start=1)
        ]
        aggregate = benchmark_module.aggregate_repeated_reports(reports)
        self.assertEqual(aggregate["metrics"]["priority_first_token_order_score"], 0.75)
        self.assertEqual(aggregate["metrics_aggregation"], "mean_of_run_metrics")
        self.assertIn("first_run", aggregate["results_scope"])
        self.assertEqual(len(aggregate["runs"]), 2)

    def test_synchronized_start_requires_full_concurrency(self) -> None:
        args = SimpleNamespace(
            workload=None,
            prompt="test",
            num_prompts=2,
            synchronize_start=True,
            concurrency=1,
        )
        with mock.patch.object(
            benchmark_module,
            "_read_request_specs",
            return_value=[{"prompt": "a"}, {"prompt": "b"}],
        ):
            with self.assertRaisesRegex(ValueError, "concurrency"):
                benchmark_module.benchmark(args)


class BackendQueuePressureTests(unittest.TestCase):
    def test_optional_runtime_forwards_queue_pressure(self) -> None:
        with (
            mock.patch.object(backend_runtime, "resolve_model", return_value="/models/test"),
            mock.patch.object(
                backend_runtime,
                "start_vllm",
                return_value=(mock.Mock(), mock.Mock(), 8123, "bfloat16"),
            ) as start,
        ):
            result = inference_project_runtime.start_optional_vllm(
                model_id="test/model",
                max_num_seqs=2,
                scheduling_policy="priority",
            )
        self.assertEqual(result[2:], (8123, "bfloat16", "/models/test"))
        self.assertEqual(start.call_args.kwargs["max_num_seqs"], 2)
        self.assertEqual(start.call_args.kwargs["scheduling_policy"], "priority")

    def test_start_vllm_passes_max_num_seqs(self) -> None:
        process = mock.Mock()
        with tempfile.TemporaryDirectory() as temporary_dir:
            with (
                mock.patch.object(backend_runtime, "find_free_port", return_value=8123),
                mock.patch.object(backend_runtime, "choose_dtype", return_value="bfloat16"),
                mock.patch.object(backend_runtime, "resolve_vllm_command", return_value="vllm"),
                mock.patch.object(backend_runtime, "wait_until_ready"),
                mock.patch.object(backend_runtime.subprocess, "Popen", return_value=process) as popen,
            ):
                _, log_handle, _, _ = backend_runtime.start_vllm(
                    "/models/test",
                    max_num_seqs=2,
                    scheduling_policy="priority",
                    log_path=Path(temporary_dir) / "server.log",
                )
                log_handle.close()
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "2")
        self.assertEqual(command[command.index("--scheduling-policy") + 1], "priority")


if __name__ == "__main__":
    unittest.main()
