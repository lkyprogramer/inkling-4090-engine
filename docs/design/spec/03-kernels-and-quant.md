# 03 — Kernels and quantisation

Requirements: R-KQ-1..5. Crates: `kernels`, `kernels-src`, `quant`.

## 1. Four kernel families [R]
| Family | Arch | Core instructions | Notes |
|---|---|---|---|
| F80 | sm_80 (and base of sm_12x) | `mma.sync`, `cp.async` | **Only family testable locally** |
| F90 | sm_90a | WGMMA + TMA | FP8; FA3-style attention |
| F100 | sm_100a/103a | tcgen05 + TMEM, 2-CTA MMA | NVFP4/MXFP4/MXFP8; INT8 reduced on sm_103 |
| F120 | sm_120a/121a | block-scaled `mma.sync`, ~100 KB smem | tile shapes/pipeline depth differ (best Colfax sm_120 tile: 2 stages vs 4) [R] |

Tile shapes tuned for Hopper/B200 do not fit sm_12x [R]. Each family has its own tuning tables; shared code is limited to epilogues, layouts and reference implementations.

## 2. Rust vs CUDA split
CUDA C++/CUTLASS own: dequant-matmul, GEMM, attention (prefill/decode, SWA/global), MoE grouped GEMM, fused RMSNorm/RoPE/quant, sampling kernels, custom all-reduce (Enterprise edition). Triton/CuTe-DSL AOT: fused glue and experiments only; produces cubins consumed by Rust (no Rust production precedent found; Triton `compile.py` is legacy and may need dummy launch params [R]) — hence **not on the P1 critical path**. cuTile Rust / cuda-oxide: feature-flagged, non-critical kernels only.

## 3. A100 / GGUF kernel set (P1 — the only locally validated path)
Target files: Unsloth `UD-*` GGUFs [G]: `UD-Q4_K_XL` (17.6 GB, default Q4), `UD-Q4_K_M` (16.6 GB), `UD-IQ4_XS` (15.7 GB), `UD-Q8_K_XL` (22.9 GB, quality reference).

**Measured from the real header (2026-11-23, first 16 MiB of `Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file`; GGUF v3, 744 tensors, 63 KV):** despite the name, this file contains **no K-quants**. Tensor types: **NVFP4 (ggml type id 40) ×40** = all expert `gate_up`/`down` tensors of layers 0–19 (20.38 G elements, each with an F32 per-expert `.scale` [128]); **Q8_0 ×154** = all attention projections, `attn_rel`, `attn_gate`, `token_embd`, `output` and the *expert* tensors of layers 20–23 (the native MXFP8 layers); **F32 ×518** (norms, conv kernels, router, selection bias, attn-res queries, relative bias/clip, all loop controllers); **BF16 ×32** (`attn_sink_proj`, `attn_{k,v}_conv_proj`). Element counts reproduce the 17.6 GB file size (NVFP4 36 B per 64 elements = 4.5 bit/element). So **`UD-Q4_K_XL` is essentially the native NVFP4 checkpoint wrapped in GGUF, with MXFP8 layers and all other weights at Q8_0**. Other `UD-*` files (e.g. `Q4_K_M`, `IQ4_XS`, `Q8_K_XL`) were **not** read and may use K/IQ types; R-KQ-1 scope for P1 is therefore {NVFP4, Q8_0, F32, BF16}, and K-quant/IQ kernels are deferred until a file that needs them is chosen (reading each file's header costs ~16 MiB).

Block formats [L, ggml-common.h]: **Q8_0** = 32 int8 + F16 scale (34 B/32 elems). **NVFP4** = block of **64 elements = four 16-element sub-blocks**: 4 bytes of E4M3-class scales (one per sub-block) + 32 bytes of packed E2M1 → 36 B per 64 elements; E2M1 value table is stored *doubled* `{0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12}` (shared with MXFP4), so integer products with the doubled table are exact and the halving folds into the scale. (UE4M3 vs E4M3 is labelled inconsistently across two headers; read the decode routine before writing the kernel.) A separate F32 **per-expert `.scale`** multiplies the matmul output, applied by `build_lora_mm_id` via `get_w_scale` [L]. llama.cpp sets **BF16 accumulation** for NVFP4 matmuls [L] — Astral's A100 reference must match, or document the difference.

Kernels per type (all derived from the GGML block-format definitions; llama.cpp, MIT, is the reference; attribution headers required):
- **MMVQ-style** (matrix × few vectors; decode, batch 1–8): reads packed blocks, dequantises in registers, dot-product with quantised activations (int8 `dp4a`) — memory-bandwidth bound. For **NVFP4 on A100** (no FP4 hardware): table-lookup E2M1→int8 (doubled values), `dp4a` against Q8 activations, then apply the per-sub-block scale and per-expert scale in float. This *is* the Marlin-style W4A8-ish path on sm_80 and is the P1 critical decode kernel.
- **MMQ-style** (prefill/larger batches): int8 tensor-core (`mma.sync` m16n8k32) over dequantised tiles; activations quantised to Q8 blocks; NVFP4 tiles expanded to int8 in shared memory with per-16 scales applied in the epilogue accumulation.
- **Dequant-to-FP16 + cuBLASLt GEMM** fallback for very large prefill chunks `[I]` (choose by benchmark).
- **MoE grouped variants** (R-KQ-2): 128 experts, top-8; token-sorted grouped dispatch; fused `gate_up` and `down` expert weights stay quantised; **F32 router; sigmoid gating; `exp_probs_b` bias added for top-k *selection only* (weights come from the unbiased sigmoid); selected weights normalised by their sum (clamped) then ×8.0; SITU activation `beta·tanh(gate/beta)·sigmoid(gate) · lb·tanh(up/lb)` with beta 4.0, lb 25.0** on the fused `gate_up` split [L] (spec 06 §3).
- **Attention (R-KQ-3) — FlashAttention/FlashInfer are not drop-in for this model.** Inkling 2 Mini uses **no RoPE**; positions enter through a **learned relative-position bias** (per-head vector `R` of size 16 dotted with a distance-indexed table, sinks folded in), a **sigmoid output gate**, Q/K/R **L2 clipping**, and on global layers a **log-length scale τ**. llama.cpp materialises the bias with `get_rows` and passes it as an additive `kq_b` mask [L]. Astral needs its own attention kernels: paged KV, GQA 24/8, head_dim 128, SWA (window 1024, relative extent 1024) and global variants, additive bias computed on the fly from `(R, table, distance)` rather than materialised `[heads × tokens × ctx]`, FP16/BF16 KV, split-KV for global slots at long context. Reference expectation: FA2-class on sm_80 [R]; no FP8 on A100. This is real kernel-authoring work and is the largest P1 kernel risk.
- **Fused ops (all new relative to a Llama-class engine):** RMSNorm (weighted and weight-less), value RMS-norm, QK/R L2-clip (`clip` operator, spec 06 §3), **attention-residual `mix`** (depth-softmax over up to ~9 stacked states with RMS-normalised states), **loop controllers** (F32 `down`/`gain`/`forget` with tanh, per core layer ×4, passes ≥1), **short causal conv at 4 sites** with per-slot state and optional input-dependent per-head taps on global layers, sigmoid attention gate, optional logit softcap, sampling (temp/top-p/top-k/min-p, grammar mask application). The F32 loop/attn-res tensors are small but sit on the critical path of every one of 48 slots — fuse aggressively once correct.

**Correctness rule:** each quant kernel must produce results bit-exact (integer paths) or within a stated ULP bound (float paths) vs `quant::dequant_ref()` on random blocks and on real tensors. Accuracy at model level is measured by KLD and same-top-token vs UD-Q8_K_XL (Unsloth's reported numbers: Q4_K_XL KLD 0.0107 / 98.0%, Q4_K_M 0.0108 / 97.9%, IQ4_XS 0.0109 / 97.7% [G]); Astral must match llama.cpp's behaviour on the same file, not "improve" it.

**Roofline (A100), computed from the real tensor list `[I]`:** at batch 1, bytes read per token = non-expert weights ≈ **1.79 GB** (attention/gate/rel at Q8_0 across 48 slots, plus F32 loop controllers) + top-8 experts ≈ **1.85 GB** (44 NVFP4 slots at ~35.8 MB + 4 Q8_0 slots) + `output` head ≈ **0.49 GB** = **≈4.1 GB/token**. **The dev server's GPU is an A100-PCIE-40GB (confirmed by `nvidia-smi`)**. `astral hw --bench` (2026-11-23) measured **1,483 GB/s best / 1,471 GB/s median DRAM read bandwidth** (95% of the 1,555 GB/s derived from clocks); at the measured figure the batch-1 ceiling is ≈**360 tok/s**; an 80 GB SXM part (2,039 GB/s) would give ≈494 tok/s but is not what we have. Observations: (a) the Q8_0 attention weights cost as much as the 4-bit experts — an A100 decode win may come from quantising or fusing these, but changing their precision departs from the file; (b) the `output` head alone is 12% of traffic; (c) distinct experts touched grows with batch, so per-token cost rises at batch >1 until experts saturate (all experts of one pass ≈ 15.8 GB). Published utilisation context: stock stacks reach ~50% of bandwidth at batch 1 on H100 for a 1B model vs 78% for a megakernel [R]; so R-KQ-4 (≥70% on weight reads) is an aggressive but measurable target.

## 4. Native low-precision formats (P3, unvalidated)
Native checkpoint is `quant_method inkling2`: **only MoE expert weights are quantised — NVFP4 for layers 0–19, MXFP8 for layers 20–23; all attention/dense/router/vision tensors are bf16** [T] (spec 06 §6). So native Blackwell kernels needed first are *grouped block-scaled expert GEMMs*, not dense FP4 GEMMs. NVFP4 = E2M1 values with 16-element E4M3 block scales; MXFP4 = 32-element E8M0 scales [R]. `quant` parameterises block size and scale type so both are one code path.
- sm_100/103: tcgen05 block-scaled paths.
- sm_120/121: block-scaled `mma.sync`. Colfax measured a hand-tuned FP4 GEMM at 1,666 TFLOP/s on RTX PRO 6000 (~83% of clock-adjusted peak) [R], so the silicon is not the limit; the gap is MoE/decode kernels and maturity. GB10 community data: NVFP4 only ~1.2–1.3× over FP8 at ~36% of bandwidth ceiling; compile failures, NaN logits and sm_100a-only templates reported [R].
- Fallback on any arch where native FP4 fails validation: Marlin-style W4A16 dequant path (the incumbents' usual workaround [R]) — R-KQ-5 requires this fallback to exist before native FP4 is enabled by default.
- NVFP4 accuracy: vendor reports ~1 point vs FP8 on DeepSeek-R1; community GB10 report of larger drops is disputed [R] — re-measure per model.
- A100 has no native path for these; GGUF is the A100 route (decision, ADR-0001 note). Since `UD-Q4_K_XL` stores the same NVFP4 experts as the native checkpoint, the A100 NVFP4 dequant kernel (§3) and the Blackwell native block-scaled kernel (here) must agree on layout semantics — share `quant::nvfp4` block definitions and the CPU reference between them.

## 5. Per-architecture guidance (design intent, only F80 testable)
- A100: GGUF K-quants (MMVQ/MMQ), FA2-style attention, BF16/FP16 KV.
- H100: FP8 weights/activations, WGMMA/TMA, FA3-style attention; Machete-class W4A16 `[R]`.
- B200/GB300: NVFP4/MXFP4, tcgen05, FA4-class attention, fused MoE; avoid INT8 on sm_103.
- sm_12x: block-scaled `mma.sync` with small tiles, FlashInfer-derived attention, Astral-tuned MoE; FA4-class public kernels for sm_120/121 were not found [R].

## 6. Differentiation tracks (not P1)
(1) Autotuned per-arch kernel families with cached results. (2) Low-batch decode megakernel: MPK reports 1.0–1.7× vs SGLang/vLLM at batch 1–16; Hazy Llama-1B 78% bandwidth; **never tested on sm_120/121, no large-batch benefit** [R] — research track only. (3) Fused layer sequences (norm+quant, RoPE+KV-write). (4) Batch-invariant kernel mode (off by default; ~20% matmul cost [R]).

## 7. Build pipeline
`kernels-src/<family>/<op>.cu` → `nvcc -gencode` per arch → cubins → packaged into fatbin + manifest (op, arch, quant, shape-class, launch-config fn). CI: disassembly assertions (spec 01 §7); reference tests on A100 for F80 only; mock-backend tests for logic. Kernel naming and manifest are stable API for the `kernels` crate.

## 8. Open items
Type inventory of the other `UD-*` files (only `UD-Q4_K_XL` is read; K-quant/IQ kernels deferred); NVFP4 scale decode (E4M3 vs UE4M3); cuBLASLt crossover for prefill; whether to re-quantise the Q8_0 attention weights for A100 speed (departs from the file); A100 SKU bandwidth.
