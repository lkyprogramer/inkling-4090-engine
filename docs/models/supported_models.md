# Supported Models

Each architecture in Astral is implemented natively, kernel by kernel, and verified against a reference implementation.

| Model | Architecture | Weights | Server | Speculative decoding | Notes |
|---|---|---|---|---|---|
| Inkling 2 Mini | `inkling2_loop` (looped MoE, attention residuals, short convolutions) | GGUF (`UD-Q4_K_XL`: Q8_0, NVFP4, BF16, F32) | ✅ | ✅ DSpark draft (`mtp.safetensors`) | Verified layer by layer against llama.cpp |
| Qwen3.8-27B | Hybrid gated delta-net + attention | MXFP4 safetensors | CLI only | — | Greedy output matches vLLM |

## Weight formats

| Format | Decode | Prefill |
|---|---|---|
| Q8_0 | dp4a GEMV; int8 tensor-core GEMV for 2–8 tokens | Tensor-core GEMM |
| NVFP4 | Shared-memory GEMV; grouped GEMV for MoE | wgmma GEMM (Hopper), mma.sync GEMM |
| MXFP4 | Native-scale GEMV | Tensor-core GEMM |
| BF16 / F32 | Row-block GEMV | — |

## Inputs

- Text only. Image input to Inkling 2 Mini is not supported yet.
- The checkpoint's own tokenizer and chat template are used. Astral's chat template port produces the same token ids as the reference implementation.
