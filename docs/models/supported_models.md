# Supported Models

Each architecture in Astral is implemented natively, kernel by kernel, and verified against a reference implementation.

| Model | Architecture | Weights | Server | Speculative decoding | Notes |
|---|---|---|---|---|---|
| Inkling 2 Mini | `inkling2_loop` (looped MoE, attention residuals, short convolutions) | Native safetensors checkpoint, or GGUF | ✅ | ✅ DSpark draft (`mtp.safetensors`) | Verified layer by layer against llama.cpp |
| Qwen3.8-27B | Hybrid gated delta-net + attention | Native safetensors checkpoint | CLI only | — | Greedy output matches vLLM |

Support for more models is coming soon.

## Inputs

- Text only. Image input to Inkling 2 Mini is not supported yet.
- The checkpoint's own tokenizer and chat template are used. Astral's chat template port produces the same token ids as the reference implementation.
