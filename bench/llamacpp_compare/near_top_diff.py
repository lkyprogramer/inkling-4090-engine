# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 inkling-4090-engine contributors
"""Per-position max |log-prob difference| among tokens with log p > -16 on both sides.

Measures llama.cpp's own batched-vs-single noise (or any two logits files) in the same
units on both sides. Note: astral cmp-prefill reports raw-logit differences and a
step-side window instead, so its numbers are not directly comparable to this output.

    python3 near_top_diff.py <reference.bin> <candidate.bin> [<candidate2.bin> ...]
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kld_compare as kld

def run(pf, qf):
    P, Q = kld.load(pf), kld.load(qf); nv = P["n_vocab"]; near = []
    for s in range(0, P["rec"].shape[0], 64):
        lp, lq = kld.logp(P["rec"][s:s + 64], nv), kld.logp(Q["rec"][s:s + 64], nv)
        m = (lp > -16) & (lq > -16)
        near.append(np.where(m, np.abs(lp - lq), 0).max(1))
    near = np.concatenate(near)
    return {"positions": int(near.size), "median": float(np.median(near)), "p90": float(np.percentile(near, 90)),
            "p99": float(np.percentile(near, 99)), "max": float(near.max()), "frac_gt_1": float((near > 1).mean())}

if __name__ == "__main__":
    print(json.dumps({q: run(sys.argv[1], q) for q in sys.argv[2:]}, indent=1))
