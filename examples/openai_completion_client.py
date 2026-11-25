# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Synthetic Minds LTD
"""Plain text completion with a stop string."""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
model = client.models.list().data[0].id

resp = client.completions.create(model=model, prompt="1, 2, 3, 4, 5,", max_tokens=40, stop=["9"])
print(repr(resp.choices[0].text), resp.choices[0].finish_reason)
