#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 inkling-4090-engine contributors
"""One calculator for comparing two sets of raw logits (policies "num-2" and "num-3").

Both sides are raw f32 logit files for the same token prefixes, as written by
`astral dump-final-logits` (kinds step / prefill / decode), `astral dump-verify-logits` (verify<k>)
and `final_logits.cpp` (single / batch):
`<dir>/<name>_<L>.<kind>.f32`. Everything is computed in float64 the same way for every pair:

  lp = log_softmax(z_ref), lq = log_softmax(z_cand)
  S  = {i : lp_i >= max(lp) - 16}             (window chosen on the reference side only)
  max_near = max_{i in S} |lp_i - lq_i|
  near_w   = sum_{i in S} p_i |lp_i - lq_i|      (probability-weighted, num-3)
  kl       = sum_i p_i (lp_i - lq_i)            (whole vocabulary)
  mass_out = probability mass of the reference outside S
  argmax, top-2 margins on both sides, gap_ref = z_ref[a_ref] - z_ref[a_cand],
  gap_cand = z_cand[a_cand] - z_cand[a_ref], number of non-finite values.

Subcommands:
  compare   <manifest> <ref dir>:<kind> <cand dir>:<kind> <positions.jsonl>
  calibrate <positions.jsonl> <rule.json> [--factor 2] [--margin 0.5]
  judge     <manifest> <positions.jsonl> <rule.json>                     (num-2: absolute limits)
  judge-paired <manifest> <cand.jsonl> <ref.jsonl> <rule.json>         (num-3: same positions)

num-3 compares a candidate with a reference control measured on the same positions: for each
statistic in the rule, candidate / reference must not exceed the given ratio; argmax flips are
allowed only where the reference-side top-2 margin is below argmax_margin; max_near is recorded,
not judged.

compare exits 2 when an expected file is missing or has the wrong size. judge checks that
every manifest position is present exactly once and finite, then applies the rule; it prints
a JSON verdict and exits 0 (pass), 1 (fail) or 2 (incomplete input).
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

POLICY = "num-2"
WINDOW = 16.0
STATS = ("mean", "p90", "max")


def read_manifest(path):
    jobs = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            name, lens, _ids = line.rstrip("\n").split("\t", 2)
            jobs.append((name, sorted({int(x) for x in lens.split(",")})))
    if not jobs:
        raise ValueError("%s: no sequences" % path)
    return jobs


def expected_positions(manifest):
    return [(name, n) for name, lens in read_manifest(manifest) for n in lens]


def log_softmax(z):
    m = np.max(z)
    return z - (m + math.log(np.sum(np.exp(z - m))))


def top2(z):
    i = int(np.argmax(z))
    rest = np.delete(z, i)
    return i, float(z[i] - np.max(rest))


def metrics(zr, zc):
    """Comparison of one reference and one candidate logit vector (float64 inside)."""
    zr = np.asarray(zr, dtype=np.float64)
    zc = np.asarray(zc, dtype=np.float64)
    if zr.shape != zc.shape:
        raise ValueError("shape mismatch %s vs %s" % (zr.shape, zc.shape))
    nonfinite = int(np.count_nonzero(~np.isfinite(zr)) + np.count_nonzero(~np.isfinite(zc)))
    if nonfinite:
        return {"nonfinite": nonfinite}
    lp, lq = log_softmax(zr), log_softmax(zc)
    s = lp >= np.max(lp) - WINDOW
    p = np.exp(lp)
    ar, mr = top2(zr)
    ac, mc = top2(zc)
    return {
        "nonfinite": 0,
        "max_near": float(np.max(np.abs(lp[s] - lq[s]))),
        "near_w": float(np.sum(p[s] * np.abs(lp[s] - lq[s]))),
        "kl": float(np.sum(p * (lp - lq))),
        "mass_out": float(np.sum(p[~s])),
        "window": int(np.count_nonzero(s)),
        "argmax_ref": ar,
        "argmax_cand": ac,
        "margin_ref": mr,
        "margin_cand": mc,
        "gap_ref": float(zr[ar] - zr[ac]),
        "gap_cand": float(zc[ac] - zc[ar]),
    }


def load(path, n_vocab=None):
    v = np.fromfile(path, dtype="<f4")
    if v.size == 0 or (n_vocab is not None and v.size != n_vocab):
        raise ValueError("%s: %d values, expected %s" % (path, v.size, n_vocab))
    return v


def cmd_compare(a):
    rdir, rkind = a.ref.rsplit(":", 1)
    cdir, ckind = a.cand.rsplit(":", 1)
    rows, missing, n_vocab = [], [], None
    for name, n in expected_positions(a.manifest):
        rp = os.path.join(rdir, "%s_%d.%s.f32" % (name, n, rkind))
        cp = os.path.join(cdir, "%s_%d.%s.f32" % (name, n, ckind))
        if not (os.path.isfile(rp) and os.path.isfile(cp)):
            missing.append("%s_%d" % (name, n))
            continue
        zr = load(rp, n_vocab)
        n_vocab = zr.size
        row = {"name": name, "len": n}
        row.update(metrics(zr, load(cp, n_vocab)))
        rows.append(row)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"ref": a.ref, "cand": a.cand, "positions": len(rows), "missing": missing,
                      "n_vocab": n_vocab, "summary": summarize(rows)}, indent=1))
    return 2 if missing else 0


def read_rows(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def stat(values, which):
    v = np.asarray(values, dtype=np.float64)
    if which == "mean":
        return float(np.mean(v))
    if which == "p90":
        return float(np.quantile(v, 0.9, method="higher"))
    return float(np.max(v))


def summarize(rows):
    ok = [r for r in rows if r.get("nonfinite") == 0]
    if not ok:
        return {}
    out = {m: {s: stat([r[m] for r in ok], s) for s in STATS} for m in ("max_near", "near_w", "kl", "mass_out") if all(m in r for r in ok)}
    out["argmax_mismatch"] = sum(r["argmax_ref"] != r["argmax_cand"] for r in ok)
    return out


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def cmd_calibrate(a):
    rows = read_rows(a.positions)
    bad = [r for r in rows if r.get("nonfinite") != 0]
    if not rows or bad:
        print("calibrate: %d rows, %d non-finite" % (len(rows), len(bad)), file=sys.stderr)
        return 2
    s = summarize(rows)
    rule = {
        "policy": POLICY,
        "source": os.path.basename(a.positions),
        "source_sha256": sha256(a.positions),
        "source_positions": len(rows),
        "factor": a.factor,
        "argmax_margin": a.margin,
        "reference": {m: s[m] for m in ("max_near", "kl")},
        "limits": {m: {k: a.factor * v for k, v in s[m].items()} for m in ("max_near", "kl")},
    }
    with open(a.rule, "w") as f:
        json.dump(rule, f, indent=1)
    print(json.dumps(rule, indent=1))
    return 0


def completeness(expected, rows):
    problems = []
    seen = {}
    for r in rows:
        key = (r.get("name"), r.get("len"))
        seen[key] = seen.get(key, 0) + 1
    exp = set(expected)
    missing = sorted(k for k in exp if k not in seen)
    dup = sorted(k for k, c in seen.items() if c > 1)
    extra = sorted(k for k in seen if k not in exp)
    nonfinite = sum(1 for r in rows if r.get("nonfinite") != 0)
    if not expected:
        problems.append("no expected positions")
    if missing:
        problems.append("missing %d positions" % len(missing))
    if dup:
        problems.append("duplicated %d positions" % len(dup))
    if extra:
        problems.append("unexpected %d positions" % len(extra))
    if nonfinite:
        problems.append("%d positions with non-finite logits" % nonfinite)
    return problems


def judge_paired(expected, cand, ref, rule):
    """num-3 verdict: candidate vs a reference control on the same positions."""
    problems = ["candidate: " + p for p in completeness(expected, cand)] + ["reference: " + p for p in completeness(expected, ref)]
    out = {"policy": rule.get("policy"), "positions": len(cand), "expected": len(expected), "problems": problems}
    if problems:
        out.update(complete=False, passed=False)
        return out
    checks = {}
    flips = [r for r in cand if r["argmax_ref"] != r["argmax_cand"]]
    hard = [(r["name"], r["len"], r["margin_ref"]) for r in flips if r["margin_ref"] >= rule["argmax_margin"]]
    checks["argmax"] = {"flips": len(flips), "flips_at_margin_ge_limit": hard, "ok": not hard}
    sc, sr = summarize(cand), summarize(ref)
    for m, lim in rule["ratio_max"].items():
        for k, v in lim.items():
            c, f = sc[m][k], sr[m][k]
            ratio = c / f if f > 0 else (0.0 if c == 0 else math.inf)
            checks["%s.%s" % (m, k)] = {"cand": c, "ref": f, "ratio": ratio, "limit": v, "ok": ratio <= v}
    out.update(complete=True, checks=checks, recorded={m: {"cand": sc[m], "ref": sr[m]} for m in rule.get("record_only", [])},
               passed=all(c["ok"] for c in checks.values()))
    return out


def cmd_judge_paired(a):
    with open(a.rule) as f:
        rule = json.load(f)
    v = judge_paired(expected_positions(a.manifest), read_rows(a.cand), read_rows(a.ref), rule)
    print(json.dumps(v, indent=1))
    return 2 if not v["complete"] else (0 if v["passed"] else 1)


def judge(expected, rows, rule):
    """num-2 verdict dict for rows against rule; complete=False when the input is not usable."""
    problems = completeness(expected, rows)
    out = {"policy": rule.get("policy"), "positions": len(rows), "expected": len(expected), "problems": problems}
    if problems:
        out.update(complete=False, passed=False)
        return out
    checks = {}
    flips = [r for r in rows if r["argmax_ref"] != r["argmax_cand"]]
    hard = [(r["name"], r["len"], r["margin_ref"]) for r in flips if r["margin_ref"] >= rule["argmax_margin"]]
    checks["argmax"] = {"flips": len(flips), "flips_at_margin_ge_limit": hard, "ok": not hard}
    s = summarize(rows)
    for m, lim in rule["limits"].items():
        for k, v in lim.items():
            checks["%s.%s" % (m, k)] = {"value": s[m][k], "limit": v, "ok": s[m][k] <= v}
    per_prompt = {}
    for r in rows:
        per_prompt[r["name"]] = max(per_prompt.get(r["name"], 0.0), r["max_near"])
    out.update(complete=True, checks=checks, worst_prompt_max_near=sorted(per_prompt.items(), key=lambda x: -x[1])[:5],
               passed=all(c["ok"] for c in checks.values()))
    return out


def cmd_judge(a):
    with open(a.rule) as f:
        rule = json.load(f)
    v = judge(expected_positions(a.manifest), read_rows(a.positions), rule)
    print(json.dumps(v, indent=1))
    return 2 if not v["complete"] else (0 if v["passed"] else 1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("manifest")
    c.add_argument("ref")
    c.add_argument("cand")
    c.add_argument("out")
    c.set_defaults(fn=cmd_compare)
    k = sub.add_parser("calibrate")
    k.add_argument("positions")
    k.add_argument("rule")
    k.add_argument("--factor", type=float, default=2.0)
    k.add_argument("--margin", type=float, default=0.5)
    k.set_defaults(fn=cmd_calibrate)
    j = sub.add_parser("judge")
    j.add_argument("manifest")
    j.add_argument("positions")
    j.add_argument("rule")
    j.set_defaults(fn=cmd_judge)
    jp = sub.add_parser("judge-paired")
    jp.add_argument("manifest")
    jp.add_argument("cand")
    jp.add_argument("ref")
    jp.add_argument("rule")
    jp.set_defaults(fn=cmd_judge_paired)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
