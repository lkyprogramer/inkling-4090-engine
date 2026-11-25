# 06b — Model spec: Qwen3.8-27B (second test model)

Role: **comparison model**, not the flagship. Chosen 2026-11-23 because vLLM and SGLang cannot run Inkling 2 Mini on the A100 (STATUS.md), but should be able to run a 4-bit Qwen3.8-27B there. This lets "faster than vLLM/SGLang" be measured on the dev server. Sources (read 2026-11-24): `[Q]` Qwen/Qwen3.8-27B model card + `config.json`, the vLLM recipe page and the SGLang cookbook page; checkpoint metadata from the Hugging Face API. Licence Apache-2.0.

## 1. Architecture `[Q]`
- `Qwen3_5ForConditionalGeneration`, `model_type qwen3_5`. **Dense** (not MoE), 27B language-model parameters; vision tower (27 layers, hidden 1152) also present.
- 64 layers, hidden 5120, FFN 17,408 (SiLU), RMSNorm eps 1e-6, vocab 248,320 (padded), untied embeddings.
- **Hybrid layout:** 16 repeats of [3 × (Gated DeltaNet → FFN), 1 × (Gated Attention → FFN)]; full attention at layers 3, 7, 11, …, 63 (16 layers), linear attention in the other 48.
  - Gated DeltaNet (linear attention): 16 QK heads, 48 V heads, head_dim 128, short conv kernel 4, SSM state in float32.
  - Gated full attention: 24 Q heads, 4 KV heads, head_dim 256, output gate, partial rotary 25% (64 of 256 dims), rope_theta 1e7, multimodal rope sections [11, 11, 10].
- Context 262,144 native (1M with static YaRN ×4). One MTP layer.

## 2. Memory profile (why it is a good second test)
- KV only in 16 full-attention layers: 4 KV heads × 256 × 2 × 2 B = 4 KiB per layer-token → **64 KiB/token BF16** (SGLang cookbook: 65.5 KB bf16, 32.8 KB fp8).
- Fixed per-sequence recurrent state for 48 GDN layers: **78.4 MB (bf16) / 153.9 MB (fp32) per request slot** (SGLang cookbook).
- It stresses Astral's typed pools differently from Inkling: Inkling has 8 global + 40 sliding-window KV slots plus small conv state; Qwen has 16 global KV layers plus large recurrent state and no sliding window. Both are poorly served by single-page-size KV managers (spec 04 §1).

## 3. Checkpoints used on the A100
| Engine | Checkpoint | Size | Format |
|---|---|---|---|
| vLLM, SGLang | `RedHatAI/Qwen3.8-27B-INT4` | 19.47 GB | compressed-tensors pack-quantized, 4-bit weights, group 128, symmetric, BF16 activations (W4A16); 304 modules ignored (left BF16) — includes, per similar repos, GDN projections, `lm_head`, vision and MTP (to verify) |
| llama.cpp (and Astral later) | `unsloth/Qwen3.8-27B-GGUF` → `Qwen3.8-27B-UD-Q4_K_XL.gguf` | 17.66 GB | Unsloth Dynamic mixed 4-bit GGUF, arch `qwen35` |

BF16 (55.6 GB) and FP8 (30.9 GB) official checkpoints exist; BF16 does not fit, FP8 would leave ~9 GB and the A100 has no FP8 compute.

## 4. Engine requirements `[Q]`
- vLLM ≥ 0.17.0 and transformers ≥ 5.8.0 (dev server image: vLLM 0.34.0, transformers 5.22.2). `--language-model-only` drops the vision tower. MTP via `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` (not used in the baseline).
- SGLang cookbook validated on v0.5.19; MTP via EAGLE flags; GDN state strategy flags (`--mamba-*`).
- **Neither recipe lists the A100 or any sm_80 GPU.** Whether their GDN (linear-attention) kernels run on sm_80 is unverified; the baseline run on the dev server is the test.

## 5. Scope in Astral
P2 or later, after Inkling P1: GGUF loader support for `qwen35`, GDN kernels (chunked prefill + recurrent decode), gated attention with partial RoPE, recurrent-state pool. Not on the P1 critical path. The immediate use is the incumbent baseline (bench/baseline_servers.sh with `MODEL=qwen38`).
