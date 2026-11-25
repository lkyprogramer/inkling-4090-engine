#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Synthetic Minds LTD
# Chat completion against a local Astral server; set ASTRAL_URL to use another address.
set -euo pipefail
URL="${ASTRAL_URL:-http://localhost:8000}"
curl -s "$URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "inkling-2-mini",
        "messages": [{"role": "user", "content": "What is 17 * 23? Answer briefly."}],
        "max_tokens": 400,
        "reasoning_effort": "low"
      }'
echo
