# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 inkling-4090-engine contributors
"""Offline regression tests for logit_compare.py (no GPU, no model): python3 -m unittest."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logit_compare as lc  # noqa: E402

RULE = {"policy": "num-2", "argmax_margin": 0.5,
        "limits": {"max_near": {"mean": 0.2, "p90": 0.3, "max": 0.4}, "kl": {"mean": 1e-3, "p90": 2e-3, "max": 4e-3}}}


def row(name, n, zr, zc):
    r = {"name": name, "len": n}
    r.update(lc.metrics(zr, zc))
    return r


class Metrics(unittest.TestCase):
    def test_shift_invariant(self):
        z = np.array([3.0, 1.0, -2.0, -30.0])
        m = lc.metrics(z, z + 3.0)
        self.assertAlmostEqual(m["max_near"], 0.0, places=12)
        self.assertAlmostEqual(m["kl"], 0.0, places=12)

    def test_window_is_reference_side(self):
        # The token that drops from -15 to -17 stays in the window because only the reference decides.
        zr = np.array([0.0, -15.0, -40.0])
        zc = np.array([0.0, -17.0, -40.0])
        m = lc.metrics(zr, zc)
        self.assertEqual(m["window"], 2)
        self.assertAlmostEqual(m["max_near"], 2.0, places=6)

    def test_flip_gaps(self):
        m = lc.metrics(np.array([1.0, 0.8, -5.0]), np.array([0.7, 0.9, -5.0]))
        self.assertEqual((m["argmax_ref"], m["argmax_cand"]), (0, 1))
        self.assertAlmostEqual(m["gap_ref"], 0.2, places=6)
        self.assertAlmostEqual(m["gap_cand"], 0.2, places=6)

    def test_nonfinite(self):
        self.assertEqual(lc.metrics(np.array([0.0, np.nan]), np.array([0.0, 1.0]))["nonfinite"], 1)


class Judge(unittest.TestCase):
    exp = [("a", 4), ("b", 4)]

    def rows(self, delta=0.0, flip=False):
        zr = np.array([2.0, 1.0, -3.0])
        zc = zr.copy()
        zc[1] += delta
        if flip:
            zc[1] += 2.0
        return [row("a", 4, zr, zr), row("b", 4, zr, zc)]

    def test_pass(self):
        v = lc.judge(self.exp, self.rows(0.05), RULE)
        self.assertTrue(v["complete"] and v["passed"], v)

    def test_threshold_fail(self):
        v = lc.judge(self.exp, self.rows(0.9), RULE)
        self.assertTrue(v["complete"])
        self.assertFalse(v["passed"])

    def test_argmax_flip_at_large_margin_fails(self):
        v = lc.judge(self.exp, self.rows(flip=True), dict(RULE, limits={}))
        self.assertFalse(v["checks"]["argmax"]["ok"])

    def test_empty_is_incomplete(self):
        v = lc.judge(self.exp, [], RULE)
        self.assertFalse(v["complete"] or v["passed"])

    def test_truncated_is_incomplete(self):
        v = lc.judge(self.exp, self.rows()[:1], RULE)
        self.assertFalse(v["complete"])

    def test_duplicate_is_incomplete(self):
        r = self.rows()
        v = lc.judge(self.exp, r + r[:1], RULE)
        self.assertFalse(v["complete"])

    def test_no_expected_is_incomplete(self):
        self.assertFalse(lc.judge([], self.rows(), RULE)["complete"])


class EndToEnd(unittest.TestCase):
    def test_compare_calibrate_judge_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            man = os.path.join(d, "m.tsv")
            with open(man, "w") as f:
                f.write("p0\t2,3\t1 2 3\n")
            rng = np.random.default_rng(0)
            for n in (2, 3):
                z = rng.normal(size=50).astype("<f4")
                z.tofile(os.path.join(d, "p0_%d.single.f32" % n))
                (z + rng.normal(scale=1e-3, size=50).astype("<f4")).tofile(os.path.join(d, "p0_%d.batch.f32" % n))
            pos = os.path.join(d, "pos.jsonl")
            rule = os.path.join(d, "rule.json")
            self.assertEqual(lc.main(["compare", man, d + ":single", d + ":batch", pos]), 0)
            self.assertEqual(lc.main(["calibrate", pos, rule]), 0)
            self.assertEqual(lc.main(["judge", man, pos, rule]), 0)
            os.remove(os.path.join(d, "p0_3.batch.f32"))
            self.assertEqual(lc.main(["compare", man, d + ":single", d + ":batch", pos]), 2)
            self.assertEqual(lc.main(["judge", man, pos, rule]), 2)
            with open(rule) as f:
                self.assertEqual(json.load(f)["policy"], "num-2")


if __name__ == "__main__":
    unittest.main()
