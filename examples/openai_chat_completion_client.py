# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Synthetic Minds LTD
"""Chat with an Astral server through the OpenAI Python client (pip install openai).

    astral serve Inkling-2-Mini-UD-Q4_K_XL.gguf --checkpoint Inkling-2-Mini/
    python examples/openai_chat_completion_client.py
"""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
model = client.models.list().data[0].id

messages = [{"role": "user", "content": "Explain what a hash map is in three sentences."}]
resp = client.chat.completions.create(model=model, messages=messages, max_tokens=600)
msg = resp.choices[0].message
print("Reasoning:", getattr(msg, "reasoning_content", None))
print("Answer:", msg.content)
print("Usage:", resp.usage)
