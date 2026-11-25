# 05 — Scheduler and batching

Requirements: R-SC-1..4. Crates: `sched`, `runtime`.

## 1. Which technique improves which metric [R]
| Technique | TTFT | ITL / per-user tok/s | Throughput | Concurrency / long ctx | Cost per M tokens |
|---|---|---|---|---|---|
| Async/overlap scheduling | small gain | gains on small models/high TP (~10–33%) | same | neutral | modest |
| Chunked prefill + stall-free batching | slight cost | tightens p99 ITL | 2.6–5.6× serving capacity under SLOs (2024 baseline) | raises SLO-goodput | raises goodput |
| Prefix caching | large cut on shared prefixes | neutral | up to ~2× claimed | strong for multi-turn/agents | strong |
| Speculative decoding | neutral | 1.7–2.5× at batch 1–8 | often <1× at batch 32+ | falls with concurrency | large at high interactivity |
| Disaggregation / wide EP | isolates TTFT | isolates ITL | largest at scale | large MoE, many GPUs | large at scale (Enterprise edition) |
Numbers are from published sources, mostly 2024-era or vendor-reported; they are *priors for design*, not Astral results.

## 2. Core loop (R-SC-1, R-SC-2)
Each step the scheduler produces a `BatchPlan`:
- **Token budget per step** (default derived from roofline: enough prefill tokens to fill compute without breaking the decode ITL target) `[I]`.
- **Decode first:** all running sequences get a decode (or speculative-verify) slot.
- **Chunked prefill fills the remaining budget** (Sarathi-Serve-style stall-free batching [R]); chunk size adapts to current decode batch.
- **Admission** checks the KV arbiter (spec 04) for worst-case blocks; otherwise queue.
- **Preemption:** reclaim prefix-cache/SWA blocks first; then recompute-preempt the lowest-priority/youngest sequence (swap-to-host only when offload tier enabled).
- **Overlap:** `plan(n+1)` runs while step n executes; finished sequences cost one extra decode step of latency (same trade as TRT-LLM's overlap scheduler [R]) — accepted and documented.

## 3. Mixture-of-experts and loop considerations `[I]`
- 128 experts/top-8: batch composition changes distinct-experts-touched; the planner estimates bytes read per step from batch size to keep roofline estimates honest.
- Looped model: one scheduler "token step" = 48 slot applications (24 stored layers; layers 6–17 run 3×); the runtime owns the loop (spec 06 §2), so the scheduler sees only token-level work. The planner's cost model uses 48 slots for FLOPs and KV reads, but only 24 layers' worth of weight *bytes* in cache terms — weights are re-read each pass unless they stay L2-resident (12 core layers' active experts will not), so decode weight traffic is ≈ prelude + 3 × core + postlude `[I]`. This is why "effective active" is ~5.4B, not 3.4B [M].
- Per-slot state (KV + short-conv) means preemption/swap cost is ≈ 160 MiB SWA + 32 KiB/token global + ~2 MB conv per sequence (spec 04); recompute-preemption is usually cheaper for short sequences.

## 4. Speculative decoding (R-SC-3)
- **GGUF/A100 path: DSpark drafter** (`dspark-Inkling-2-Mini-*.gguf`, 504 MB Q8_0 [G]). Mainline llama.cpp support is an open PR (#32218), the drafter file format changed, and `--spec-draft-n-max 5` was fastest while 2 can be slower than no drafter [G]. Spec: implement against the *new* drafter format, treat the format as unstable, version-gate on file metadata.
- **Safetensors path (P3):** native MTP head (`mtp.safetensors`: 3 layers, block_size 8, `target_layer_ids [5,23,41,48]`) [M]. Not usable from the GGUF repo (no MTP file) [G].
- **Load-adaptive depth (the differentiator):** throttle draft length by batch size and measured acceptance; disable at high concurrency where speculation is ≤1× [R] (EAGLE: 1.73× at batch 1 → 1.21× at batch 128 on Llama-3.1-8B; MTP per-position acceptance falling 0.92→0.38 on GLM-4.5-Air). MoE caveat: verification fetches extra experts, shrinking benefit [R].
- **Rollback:** Inkling carries per-slot short-conv state (4 sites, last 3 inputs per channel, 48 slots) [L]. Because the state is just the last `n_conv−1 = 3` inputs, rollback after a rejected draft token is cheap by construction: keep the verifier's per-position inputs for the last k+3 positions and restore by re-slicing, no recompute `[I]`. KV rollback is block-pointer truncation. Cost remains unmeasured; validate in P2. (Attention-residual `blocks` and loop messages are per-token intra-step activations, not carried state, per spec 04 §4 — confirm.) Loop passes also mean a draft token's verification costs a full 48-slot forward per draft position.
- Suffix/lookup decoding (up to 4.5× on agentic workloads at low concurrency [R]) is a P2+ option behind the same verifier.
- Author-reported claims such as DFlash >6× are unreplicated and not design inputs [R].

## 5. Priorities and multi-tenancy (R-SC-4, Enterprise edition)
Priority classes with preemption; per-tenant token-rate quotas. VTC is a research baseline with published unfairness critique [R]; implement quotas first, fairness policy later. SLO-aware admission uses TTFT/ITL targets per class.

## 6. Structured output
Grammar masks computed on the output thread (llguidance, ~50 µs/token CPU for 128k vocab [R]; Inkling vocab is 201,024 [M], so re-measure) and applied by the sampling kernel; overlaps the forward pass.

## 7. Determinism
Default: not batch-invariant. Optional batch-invariant mode (fixed reduction order, ~20% matmul cost; 80 unique outputs → 1 over 1,000 runs [R]). Scheduler must expose a "deterministic" flag that pins batch composition-independent kernels.

## 8. Instrumentation required from day one
Per request: queue time, prefill time, TTFT, per-token ITL, tokens/s, speculative accept rate, preemptions, prefix hit tokens. Per step: batch composition, tokens, planned vs actual duration, CPU time in plan/launch. Without these the 20% claim cannot be defended (spec 08).

## 9. Out of scope for v1
Disaggregated prefill/decode, wide expert parallelism, cache-aware routing, NIXL-style KV transfer — all part of the Enterprise edition, and they only pay off for large MoE on many GPUs with fast interconnect [R].
