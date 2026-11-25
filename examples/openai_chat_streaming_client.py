# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Synthetic Minds LTD
"""Stream a chat answer from an Astral server: reasoning first, then the answer."""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
model = client.models.list().data[0].id

stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Write a haiku about the sea."}],
    max_tokens=600,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta if chunk.choices else None
    if delta is None:
        continue
    text = getattr(delta, "reasoning_content", None) or delta.content
    if text:
        print(text, end="", flush=True)
print()
