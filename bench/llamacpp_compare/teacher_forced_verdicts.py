# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 inkling-4090-engine contributors
"""V5 / V6 / V7 verdicts from astral score and cmp-prefill outputs against greedy_reference.py.

    astral score <model> $(cat NN.ids) > score_NN.txt
    astral cmp-prefill <model> $(cat NN.ids) > cmp_NN.txt
    python3 teacher_forced_verdicts.py <dir> [--v7-near-max X --v7-kl-max Y]

V5: the first disagreement in the first 128 generated tokens has reference margin < 0.5.
V6: every disagreement has margin <= 1.0 (both Astral's and the reference's margin).
V7: argmax equal; with --v7-near-max / --v7-kl-max also the near-top raw-logit diff and
    KL(step||prefill) printed by cmp-prefill. Without them V7 checks argmax only and reports
    the diagnostics. The thresholds must be calibrated in the same units (raw logits,
    step-side window, float64 full-vocabulary KL), not taken from near_top_diff.py.
Any missing or incomplete input fails. Exit status 0 only if every verdict passes.
"""
import argparse, json, re, sys

POLICY = "tfv-1"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--v7-near-max", type=float)
    ap.add_argument("--v7-kl-max", type=float)
    a = ap.parse_args()
    ref = json.load(open(f"{a.dir}/ref.json"))
    rows, ok, errors = [], {"V5": True, "V6": True, "V7": True}, []
    if not ref:
        errors.append("ref.json is empty")
    for r in ref:
        i, plen, ngen = r["i"], r["plen"], len(r["gen"])
        try:
            txt = open(f"{a.dir}/score_{i:02d}.txt").read()
            c = open(f"{a.dir}/cmp_{i:02d}.txt").read()
        except OSError as e:
            errors.append(str(e)); continue
        pos = {}
        for m in re.finditer(r"pos\s+(\d+)\s+next\s+(\d+)\s+want\s+\d+\s+(?:XX|  )\s+margin ([\d.]+)", txt):
            p = int(m[1])
            if p in pos:
                errors.append(f"prompt {i}: duplicate position {p}")
            pos[p] = (int(m[2]), float(m[3]))
        want = range(plen - 1, plen - 1 + ngen)
        missing = [p for p in want if p not in pos]
        if missing:
            errors.append(f"prompt {i}: {len(missing)} generated positions missing from score output")
            continue
        bad = []
        for p in want:
            nxt, am = pos[p]
            g = p - (plen - 1)
            if nxt != r["ref_top1"][g]:
                bad.append((g, am, r["ref_margin"][g]))
        first = bad[0] if bad else None
        v5 = first is None or first[0] >= 128 or (first[2] is not None and first[2] < 0.5)
        v6 = all(max(am, rm if rm is not None else 0.0) <= 1.0 for _, am, rm in bad)
        m = re.search(r"max \|logit diff\| ([\d.]+); argmax step \((\d+), [^)]*\) prefill \((\d+),", c)
        nd = re.search(r"near-top max \|diff\| ([\d.]+) at .*KL\(step\|\|prefill\) ([-\d.e+]+)", c)
        if not m:
            errors.append(f"prompt {i}: cmp-prefill output not found"); continue
        v7 = m[2] == m[3]
        if a.v7_near_max is not None or a.v7_kl_max is not None:
            if not nd:
                errors.append(f"prompt {i}: cmp-prefill diagnostics missing"); continue
            v7 = v7 and (a.v7_near_max is None or float(nd[1]) <= a.v7_near_max) and (a.v7_kl_max is None or float(nd[2]) <= a.v7_kl_max)
        for k, v in (("V5", v5), ("V6", v6), ("V7", v7)):
            ok[k] &= v
        rows.append({"i": i, "plen": plen, "n_gen_checked": ngen, "ignore_eos_forced": sum(x != y for x, y in zip(r["gen"], r["ref_top1"])),
                     "disagree": [{"gen_pos": g, "astral_margin": am, "ref_margin": None if rm is None else round(rm, 3)} for g, am, rm in bad],
                     "v5_first_div": None if first is None else first[0], "v5": v5, "v6": v6,
                     "v7_full_vocab_max_diff": float(m[1]), "v7_near_top_max_diff": float(nd[1]) if nd else None,
                     "v7_kl_step_prefill": float(nd[2]) if nd else None, "v7": v7})
    passed = not errors and all(ok.values())
    print(json.dumps({"policy": POLICY, "v7_near_max": a.v7_near_max, "v7_kl_max": a.v7_kl_max, **ok,
                      "errors": errors, "pass": passed, "rows": rows}, indent=1))
    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()
