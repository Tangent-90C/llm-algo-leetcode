"""Exercise the DPO chapter and its teaching checks without downloads or GPUs.

Run with a Python environment containing PyTorch:
    python tools/test_dpo_tutorial.py

This deliberately executes the notebook's own reference implementations instead
of maintaining a second copy of the teaching code. The student stubs stay empty.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
from copy import deepcopy
from pathlib import Path
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "02_PyTorch_Algorithms/17_DPO_Loss_Tutorial.ipynb"


class DPOTutorialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, cls.old_threads)
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.ns = {"__name__": "dpo_notebook_test"}
        with contextlib.redirect_stdout(io.StringIO()):
            for index, cell in enumerate(cls.notebook["cells"]):
                if cell["cell_type"] == "code":
                    source = "".join(cell["source"])
                    exec(compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec"), cls.ns)

    def test_reference_functions_pass_the_visible_checks(self):
        for name in ("response_logps", "dpo_loss", "dpo_step"):
            with self.subTest(name=name), contextlib.redirect_stdout(io.StringIO()):
                self.ns[f"check_{name}"](self.ns[f"{name}_solution"])

    def test_student_answers_are_not_silently_filled_in(self):
        cases = (
            ("response_logps", (None, None, None)),
            ("dpo_loss", (None, None, None, None, 0.1)),
            ("dpo_step", (None, None, None, None)),
        )
        for name, args in cases:
            with self.subTest(name=name), self.assertRaises(NotImplementedError):
                self.ns[name](*args)

    def test_response_scores_match_a_token_by_token_oracle(self):
        batch = self.ns["make_batch"]([
            ("1 + 1 =", "2", "3 4"),
            ("2 + 3 =", "5 4 3", "2"),
        ])
        ids, mask = batch["input_ids"], batch["response_mask"]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            logits = torch.randn(*ids.shape, len(self.ns["VOCAB"]), dtype=torch.float64)
        expected = []
        for b in range(len(ids)):
            total = logits.new_zeros(())
            for t in range(1, ids.shape[1]):
                if mask[b, t]:
                    total = total + logits[b, t - 1].log_softmax(-1)[ids[b, t]]
            expected.append(total)
        torch.testing.assert_close(
            self.ns["response_logps_solution"](logits, ids, mask), torch.stack(expected)
        )

    def test_half_precision_scoring_uses_finite_fp32_outputs_and_gradients(self):
        ids = torch.tensor([[0, 1, 2]])
        mask = torch.tensor([[False, True, True]])
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                logits = torch.tensor([[[1000., -1000., 0.], [0., 1000., -1000.], [0., 0., 0.]]], dtype=dtype, requires_grad=True)
                scores = self.ns["response_logps_solution"](logits, ids, mask)
                self.assertEqual(scores.dtype, torch.float32)
                self.assertTrue(torch.isfinite(scores).all())
                scores.sum().backward()
                self.assertTrue(torch.isfinite(logits.grad).all())

    def test_visible_checks_reject_core_scoring_mistakes(self):
        def no_shift(logits, ids, mask):
            return logits.log_softmax(-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1).masked_fill(~mask, 0).sum(-1)

        def no_mask(logits, ids, mask):
            return logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1).sum(-1)

        def mean_instead_of_sum(logits, ids, mask):
            return self.ns["response_logps_solution"](logits, ids, mask) / mask[:, 1:].sum(-1)

        def detached(logits, ids, mask):
            return self.ns["response_logps_solution"](logits, ids, mask).detach()

        for mutation in (no_shift, no_mask, mean_instead_of_sum, detached):
            with self.subTest(mutation=mutation.__name__):
                with self.assertRaises((AssertionError, RuntimeError)):
                    self.ns["check_response_logps"](mutation)

    def test_visible_checks_reject_core_loss_mistakes(self):
        def no_reference(pw, pl, rw, rl, beta):
            delta = pw - pl
            return -F.logsigmoid(beta * delta), delta

        def reversed(pw, pl, rw, rl, beta):
            delta = (pl - rl) - (pw - rw)
            return -F.logsigmoid(beta * delta), delta

        def unstable(pw, pl, rw, rl, beta):
            delta = (pw - rw) - (pl - rl)
            return -torch.sigmoid(beta * delta).log(), delta

        def detached_policy(pw, pl, rw, rl, beta):
            return self.ns["dpo_loss_solution"](pw.detach(), pl.detach(), rw, rl, beta)

        for mutation in (no_reference, reversed, unstable, detached_policy):
            with self.subTest(mutation=mutation.__name__):
                with self.assertRaises((AssertionError, RuntimeError)):
                    self.ns["check_dpo_loss"](mutation)

    def test_visible_train_check_rejects_stale_gradients_and_missing_updates(self):
        def no_zero_grad(policy, reference, batch, optimizer, beta):
            with mock.patch.object(optimizer, "zero_grad", return_value=None):
                return self.ns["dpo_step_solution"](policy, reference, batch, optimizer, beta)

        def no_update(policy, reference, batch, optimizer, beta):
            with mock.patch.object(optimizer, "step", return_value=None):
                return self.ns["dpo_step_solution"](policy, reference, batch, optimizer, beta)

        for mutation in (no_zero_grad, no_update):
            with self.subTest(mutation=mutation.__name__), self.assertRaises(AssertionError):
                self.ns["check_dpo_step"](mutation)

    def test_step_matches_three_independently_computed_sgd_updates(self):
        policy, reference = self.ns["make_models"]()
        expected = deepcopy(policy)
        actual_optimizer = torch.optim.SGD(policy.parameters(), lr=0.05)
        expected_optimizer = torch.optim.SGD(expected.parameters(), lr=0.05)
        batch = self.ns["make_batch"](self.ns["PAIRS"])
        ids, mask, n = batch["input_ids"], batch["response_mask"], batch["n_pairs"]

        def slow_scores(model):
            logits = model(ids)
            return torch.stack([
                sum(logits[b, t - 1].log_softmax(-1)[ids[b, t]]
                    for t in range(1, ids.shape[1]) if mask[b, t])
                for b in range(len(ids))
            ])

        with torch.no_grad():
            reference_scores = slow_scores(reference)
        for _ in range(3):
            actual = self.ns["dpo_step_solution"](policy, reference, batch, actual_optimizer, beta=0.2)
            expected_optimizer.zero_grad(set_to_none=True)
            scores = slow_scores(expected)
            delta = scores[:n] - scores[n:] - reference_scores[:n] + reference_scores[n:]
            expected_loss = F.softplus(-0.2 * delta).mean()
            self.assertAlmostEqual(actual, expected_loss.item(), places=6)
            expected_loss.backward()
            expected_optimizer.step()
            for actual_parameter, expected_parameter in zip(policy.parameters(), expected.parameters()):
                torch.testing.assert_close(actual_parameter, expected_parameter)

    def test_counterexample_uses_valid_distributions(self):
        reference = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)
        updated = torch.tensor([0.4, 0.1, 0.5], dtype=torch.float64)
        self.assertAlmostEqual(updated.sum().item(), 1.0)
        losses, margin = self.ns["dpo_loss_solution"](
            updated[0].log(), updated[1].log(), reference[0].log(), reference[1].log(), beta=0.1,
        )
        self.assertLess(updated[0], reference[0])
        self.assertGreater(margin.item(), 0)
        self.assertLess(losses.item(), math.log(2))

    def test_full_experiment_learns_pairs_without_updating_reference(self):
        history = self.ns["history"]
        self.assertEqual([row["step"] for row in history], [0, 1, 5, 15, 30, 60])
        self.assertAlmostEqual(history[0]["loss"], math.log(2), places=6)
        self.assertEqual(history[0]["margin"], 0)
        self.assertLess(history[-1]["loss"], 0.1)
        self.assertGreater(history[-1]["margin"], 10)
        _, initial_reference = self.ns["make_models"]()
        for initial, final in zip(initial_reference.parameters(), self.ns["reference"].parameters()):
            torch.testing.assert_close(initial, final, atol=0, rtol=0)
            self.assertIsNone(final.grad)

    def test_notebook_has_three_explicit_exercises_and_no_errors(self):
        cells = self.notebook["cells"]
        exercises = [c for c in cells if "dpo-exercise" in c.get("metadata", {}).get("tags", [])]
        self.assertEqual(len(exercises), 3)
        ids = [c["id"] for c in cells]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in cells:
            self.assertFalse(any(o.get("output_type") == "error" for o in cell.get("outputs", [])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
