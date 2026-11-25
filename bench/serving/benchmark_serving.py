# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Synthetic Minds LTD
"""Single-stream decode throughput of an OpenAI-compatible server.

For every prompt of a prompt set, the chat-template token ids are sent to streaming
/v1/completions (512 tokens, greedy, ignore_eos), and decode speed is measured from the
first streamed token to the last. One warm-up request runs first.

    python bench/serving/benchmark_serving.py --base-url http://localhost:8000 \
        --prompts bench/serving/prompts_set_a.json --label astral

The prompt token ids are obtained once from the server's /tokenize endpoint (Astral and vLLM
both implement it); pass --ids to reuse a saved file with another server.
"""

import argparse
import json
import time
import urllib.request


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=900)


def tokenize(base, prompts):
    return [json.load(post(base, "/tokenize", {"messages": [{"role": "user", "content": p}]}))["tokens"] for p in prompts]


def run(base, model, ids, max_tokens):
    t0 = time.perf_counter()
    ttft, usage, chunks = None, None, 0
    body = {"model": model, "prompt": ids, "max_tokens": max_tokens, "ignore_eos": True, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    for raw in post(base, "/v1/completions", body):
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        ch = json.loads(line[5:])
        if ch.get("usage"):
            usage = ch["usage"]
        if ch.get("choices") and ch["choices"][0].get("text"):
            chunks += 1
            if ttft is None:
                ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    n = usage["completion_tokens"] if usage else chunks
    return (n - 1) / (total - ttft), ttft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="model")
    ap.add_argument("--prompts", help="JSON list of user messages")
    ap.add_argument("--ids", help="JSON list of prompt token-id lists (skips /tokenize)")
    ap.add_argument("--save-ids", help="write the prompt token ids here")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--label", default="server")
    a = ap.parse_args()
    if a.ids:
        ids = json.load(open(a.ids))
    else:
        ids = tokenize(a.base_url, json.load(open(a.prompts)))
        if a.save_ids:
            json.dump(ids, open(a.save_ids, "w"))
    run(a.base_url, a.model, ids[0], a.max_tokens)  # warm-up
    rates = []
    for i, x in enumerate(ids):
        r, t = run(a.base_url, a.model, x, a.max_tokens)
        rates.append(r)
        print(f"{a.label} prompt {i:2d}: {r:6.1f} tok/s (ttft {t * 1e3:.0f} ms)", flush=True)
    print(f"{a.label} MEAN {sum(rates) / len(rates):.1f} tok/s over {len(rates)} prompts")


if __name__ == "__main__":
    main()
