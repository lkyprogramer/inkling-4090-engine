# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 inkling-4090-engine contributors
"""Chat-template token ids for prompt sets, from a server's /tokenize endpoint (Astral).

    python3 chat_prompt_ids.py http://127.0.0.1:8000 ids.json prompts_set_a.json prompts_set_b.json
"""
import json, sys, urllib.request
base, out = sys.argv[1], sys.argv[2]; res = []
for s, path in (("a", sys.argv[3]), ("b", sys.argv[4])):
    for i, p in enumerate(json.load(open(path))):
        req = urllib.request.Request(base + "/tokenize", data=json.dumps({"messages": [{"role": "user", "content": p}]}).encode(), headers={"Content-Type": "application/json"})
        res.append({"set": s, "case": i, "ids": json.load(urllib.request.urlopen(req))["tokens"]})
json.dump(res, open(out, "w")); print(len(res))
