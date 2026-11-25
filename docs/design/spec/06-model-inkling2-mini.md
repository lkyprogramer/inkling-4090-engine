# 06 — Model spec: Inkling 2 Mini

Requirements: R-MD-1..3. Crates: `model`, `loader`.

Sources (read 2026-11-23): `[M]` HF model card + `config.json`; `[G]` Unsloth GGUF repo pages; **`[L]` llama.cpp `master` source for `inkling2_loop` (`src/models/inkling2.cpp`, `inkling2-loop.cpp`, `llama-arch.*`, `conversion/inkling2.py`, PRs #32694/#32704)**; **`[T]` repo files `chat_template.jinja`, `tokenizer_config.json`, `generation_config.json`, `hf_quant_config.json`, and the Thinking Machines blog**. Licence Apache-2.0; the card links a Model Acceptable Use Policy.

**Status of this document:** the architecture below is taken from llama.cpp's implementation of the model, i.e. a reimplementation that reports perplexity 7.653 vs 7.651 against Unsloth's own build [G]. It is the best available ground truth but is **not** Thinking Machines' reference code (the HF repo ships none [T]). The reference-parity risk is handled in §9. Items still unknown are in §8.

## 1. Architecture summary
- `Inkling2LoopForConditionalGeneration`; multimodal in, text out. 24 stored layers, hidden size 2304, sparse MoE FFN, 26.4B total / 3.4B active / ~5.4B effective active parameters [M].
- Four blocks of six layers; the middle two blocks (layers 6–17, 0-indexed) run **three fixed passes** with shared weights = **48 layer applications ("slots") per token** [T blog, L]. Passes are fixed; no token-dependent early exit in the code (`is_varloop` only changes a minimum-pass check) [L].
- Vocabulary 201,024 (unpadded 200,058), EOS 200006, context 1,048,576 [M, T]. `bfloat16` base dtype.

## 2. The loop (what llama.cpp does) `[L]`
- `n_slot = n_phys + (n_passes−1)·(core_end−core_start)` = 24 + 2·12 = **48**. `layer_phys(il) = il < n_phys ? il : core_start + (il − n_phys) % (core_end − core_start)`.
- Graph order: embed → **prelude** layers 0..5 once → **pass 0** core (layers 6..17) → **passes 1 and 2**, each *restarting from the prelude output* and running the core on its own slots, with the input of each core layer modified by loop controllers from "messages" built from the previous pass → **postlude** layers 18..23 once → output.
- **KV cache: per pass, not shared.** Weights are shared (`layers[il] = layers[layer_phys(il)]`), but "each cache slot keeps its own KV and conv state" [L, inferred from indexing: attention and recurrent state are addressed by slot index `il`; I did not read the KV allocation code]. Third-party arithmetic agrees: 48 slots = **8 global + 40 sliding-window** [L commit message]. Layers 6–17 hold 10 local + 2 global layers (11, 17); ×3 passes plus the 2 global (5, 23) and 10 local outside the core gives 8 global + 40 SWA. **This resolves the earlier open question: spec 04 pools must size 48 slots.** Open llama.cpp issue #32718 concerns co-locating per-pass KV with shared weights under multi-GPU layer split — relevant to P4 `dist`.
- **Loop controllers** (core layers only; four per layer: Q, K, V, FFN): `gate = concat(rms(x), rms(m))` (2·n_embd) → `down` (`2·n_embd × rank`) → `gain = tanh(W_gain·gate)`, `forget = tanh(W_forget·gate)` → `x' = x − forget·x + gain·m`. `rank = loop.gate_rank` (64 in config). No nonlinearity between `down` and the two projections. Controller tensors are F32 [L]. `loop_msg_query` (`{n_embd, 4}`) builds messages.
- **Messages:** `mix(bank, bank_out, msg_query)` where `bank` = previous pass's residual blocks after the first core block and `bank_out` = previous pass's final core output [L]. HF names these `middle_cycle_cross_pass_queries/kv_queries` (F32) [T].

## 3. Layer structure (per slot) `[L]`
Order in `build_layer`:
1. **Attention-residual mix (inputs):** if `attn_res_query` exists (every layer except 0), `inputs = mix(blocks, cur, attn_res_query)` → one input for SWA layers, **three (Q,K,V) for global layers** (`attn_res_query` shape `{n_embd,3}` vs `{n_embd,1}`). `mix`: stack states `[blocks newest-first…, cur]` along depth; **RMS-normalise each (no weight, eps 1e-6)**; scores = `softmax_depth(states·q / sqrt(n_embd))`; output = Σ score·normalised-state. (There is no `attn_res_normalize_values` switch: it always normalises [L].)
2. **Loop control** on Q/K/V inputs (core layers, passes ≥1).
3. **Norm:** SWA layers use `attn_norm`; global layers use separate `attn_{q,k,v}_inp_norm`.
4. **Projections:** `Q = Wq·in0`; `K = clip(conv_K(Wk·in1))`; `V = rms(conv_V(Wv·in2))` (value RMS-norm with no weight, after conv); relative vector `R = clip(attn_rel·in0)`.
5. **Global layers only:** `Q *= τ`, `R *= τ` with `τ = 1 + α·log(max(1, (pos+1)/floor))` (`log_scale_alpha` 0.1, `log_scale_floor` 8192).
6. **No RoPE** (`rope_pattern` all false). Positions enter via a **learned relative-position bias**: `attn_rel_bias` `{n_rel, n_extent}`; bias(h, t, s) = Σ_k `bias[k, d]·R[k,h,t]` indexed by distance d = t−s; `n_rel = d_rel = 16`; extent = `relative_extent` (1024; window size for SWA layers, full relative extent for global). **Attention sinks** are folded into the table (`table = attn_rel_bias^T·R − sink`) [L]. Score scale `1/sqrt(head_dim)`.
7. **Attention** (GQA 24/8, head_dim 128), then **output gating:** `attn_out = Wo(attn · sigmoid(attn_gate·in0))`, then post-conv (`attn_post_conv`).
8. **Residual:** if `ip % attn_res.block_size == 0` (block_size 6) push the layer input to `blocks` and *do not add*; else `cur = in + attn_out`. Core start/end must be multiples of the block size.
9. **FFN input:** `mix(blocks, cur, ffn_res_query)` (+ FFN loop control) → `ffn_norm` (RMS) → **MoE** → post-conv (`ffn_post_conv`) → residual.
10. **Output:** `mix(blocks, cur, output_res_query)` → `output_norm` → `output` (F32 accumulate) → optional logit softcap → trim padded vocab.

### Clip operator `[L]`
`clip(x, scale, thr)`: optional per-channel `x *= scale`; `n = ‖x‖₂ + 1e-6`; `x *= clamp(thr/n, 0, 1)` (row L2 norm capped at `thr`). Q: per head, scale `attn_q_clip`, `thr = qk_clip` (22.627…); K: per head, no scale, `thr = qk_clip`; R: per head over `n_rel`, scale `attn_rel_clip`, `thr = relative_clip` (8.0).

### Short convolution (`sconv`) `[L]`
Kernel `n_conv = shortconv.l_cache` (config 4; code requires 2–64). Depthwise causal conv applied at **four sites**: K, V, attention output, FFN output. Recurrent state = last `n_conv−1` inputs per channel, **per slot**; channel layout offsets: K 0, V `n_embd_k_gqa`, attn-out `2·n_embd_k_gqa`, FFN-out `2·n_embd_k_gqa + n_embd` (`n_embd_k_gqa` = 8×128 = 1024). Each conv: `x = concat(past(n_conv−1), current)` → depthwise conv → `cur + out`, scaled by `1/sqrt((dyn?2:1)·n_conv + 1)`. **Global layers add input-dependent per-head taps** (`attn_{k,v}_conv_proj`); SWA layers do not.

### MoE `[L]`
Router `ffn_gate_inp` F32 `[2304,128]`; **sigmoid gating** `probs = sigmoid(logits)`; selection scores = `probs + exp_probs_b` (bias used **only to pick the top-8**; the weights are gathered from the *unbiased* `probs`); selected weights divided by their sum (clamped ≥ 6.1e-5, or +eps) then ×`expert_weights_scale` (**8.0** in the GGUF = config `route_scale`); fused experts `gate_up {2304, 2304, 128}` (gate = first 1152 rows, up = next 1152), `down {1152, 2304, 128}`; no shared experts. **Activation SITU [L, llama-graph.cpp]:** `act = β·tanh(gate/β)·sigmoid(gate)`; `up' = lb·tanh(up/lb)` (if lb > 0); `out = act · up'`, with **β = `situ_beta` = 4.0, lb = `situ_linear_beta` = 25.0** — these are exactly the config's `mlp_gate_softcap 4.0` / `mlp_up_softcap 25.0` (resolved). SITU is MoE-path only (dense FFN path aborts "not yet supported"). Each NVFP4 expert tensor has an F32 per-expert `.scale` `[128]` multiplied into the matmul output (spec 03 §3). **`final_logit_softcapping = 30.0` is written into the GGUF** (resolved).

## 4. Local vs global layers
`sliding_window_pattern` (per physical layer, copied to slots): SWA layers 0–4, 6–10, 12–16, 18–22 (window 1024); **global 5, 11, 17, 23** [M, L]. Heads 24/8/128 for both.

## 5. GGUF keys and tensors (Mainline layout) `[L]`
Arch string: **`inkling2_loop`** (base arch `inkling2` also exists for the non-looped model) [L].
- Loop keys: `%s.loop.start`, `.end`, `.count`, `.gate_rank`, `.varying`.
- Other keys: `%s.attention.layer_norm_rms_epsilon`, `.qk_clip`, `.relative_length`, `.relative_extent`, `.relative_clip`, `.log_scale_floor`, `.log_scale_alpha`, `.sliding_window`, `.sliding_window_pattern` (array); `%s.attn_res.block_size`; `%s.shortconv.l_cache`; `%s.vocab_size`, `.vocab_size_unpadded`; `%s.expert_feed_forward_length`; `%s.activation.situ_beta`, `.situ_linear_beta`; `%s.expert_weights_scale`; `%s.final_logit_softcapping`; standard block/head/expert counts from the base loader (not read).
- Name map from config: `n_layers_per_attn_res_block`→`attn_res.block_size`; `d_rel`/`rel_extent`→`relative_length`/`relative_extent`; `sconv_kernel_size`→`shortconv.l_cache`; `log_scaling_*`→`log_scale_*`; `route_scale`→`expert_weights_scale`.
- Tensors (physical layers): global `token_embd`, `token_embd_norm`, `output`, `output_norm`, `output_res_query`; per layer `attn_q/k/v/output/gate`, `attn_q_clip`, `attn_sink_proj`, `attn_rel`, `attn_rel_bias`, `attn_rel_clip`, `attn_k_conv`, `attn_v_conv`, `attn_post_conv`, `ffn_post_conv`, `attn_res_query`, `ffn_res_query`, `ffn_norm`, `ffn_gate_inp`, `ffn_exp_probs_b`, `ffn_gate_up_exps`, `ffn_down_exps`; SWA: `attn_norm`; global: `attn_{q,k,v}_inp_norm`, `attn_{k,v}_conv_proj`; core layers: `loop_msg_query`, `loop_{q,k,v,ffn}_ctrl_{down,gain,forget}` (converter forces these to F32). Names and shapes below were confirmed against the real `UD-Q4_K_XL` header (§7).
- **Loader rule (from PR #32704):** loop-slot tensor aliasing must happen *after* all optional scale tensors (e.g. NVFP4 `.scale`) are attached, or passes 2–3 run experts without scales (UD-IQ1_S perplexity 168.5 → 8.53 after the fix) [L]. Astral's loader aliases by reference to physical tensors, which avoids this class of bug; add a regression test.

## 6. Native checkpoint (P3) `[T]`
`quant_method: "inkling2"`; **only expert weights are quantised**: layers 0–19 `mlp.experts.w13_weight`/`w2_weight` **NVFP4** (block 16, E4M3 scales, per-token E2M1 activations); layers 20–23 experts **MXFP8** (block 32, UE8M0 scales). Everything else (attention, norms, sconv, controllers, router, embeddings, vision tower) bf16; `attn_res.*`, gate biases and cross-pass queries F32. MTP weights bf16 in `mtp.safetensors`. No explicit exclude list.

## 7. A100 GGUF path (P1) — measured from the real header
Files [G]: `UD-Q4_K_XL` (17.6 GB, speed), `UD-Q8_K_XL` (22.9 GB, quality reference); alternates `UD-Q4_K_M`, `UD-IQ4_XS`. Mainline layout = `Mainline_Support/*.gguf_file`, build ≥ b12611, PRs #32694/#32704. Unsloth KLD vs Q8: Q4_K_XL 0.0107 (98.0% same top token).

**Header of `Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file` (first 16 MiB fetched by HTTP range request, 2026-11-23; the 17.6 GB body was not downloaded):**
- GGUF v3, 744 tensors, 63 KV pairs; header + tensor-info table fit in 16 MiB (tokenizer arrays dominate). `general.architecture = inkling2_loop`, `general.file_type = 7` (MOSTLY_Q8_0 — a generic label, not meaningful for this mixed file).
- Keys confirmed: `block_count 24`, `context_length 1048576`, `embedding_length 2304`, per-layer `head_count`=24 and `head_count_kv`=8 arrays, `key_length`/`value_length` 128, `expert_count 128`, `expert_used_count 8`, `expert_feed_forward_length 1152`, `loop.{start 6, end 18, count 3, gate_rank 64, varying false}`, `rope.dimension_count 0`, `attention.sliding_window 1024`, `sliding_window_pattern` = `True` (SWA) for layers 0–4, 6–10, 12–16, 18–22 and `False` (global) for 5, 11, 17, 23, `shortconv.l_cache 4`, `expert_weights_scale 8.0`, `activation.situ_beta 4.0`, `activation.situ_linear_beta 25.0`, `attn_res.block_size 6`, `final_logit_softcapping 30.0`, `attention.relative_length 16`, `relative_extent 1024`, `qk_clip 22.6274`, `relative_clip 8.0`, `log_scale_floor 8192`, `log_scale_alpha 0.1`.
- Tokenizer: `gpt2`-style BPE, `pre = inkling2` (a llama.cpp-specific pre-tokenizer id), 200,058 tokens, 424,413 merges, EOS/EOT 200006, pad 199999, `add_bos/eos false`. **The Jinja chat template is embedded** (7,041 chars) and the file embeds default sampling (`temp 1.0, top_k 0, top_p 1.0, min_p 0.0`).
- imatrix provenance: `unsloth_calibration_Inkling-2-Mini.conv.jsonl`, 892 chunks, 392 entries.
- Tensor shapes (physical layers): `attn_q/attn_gate [2304,3072]`, `attn_k/v [2304,1024]`, `attn_output [3072,2304]`, `attn_rel [2304,384]` (=24 heads × 16), `attn_rel_bias [16,1024]`, `attn_rel_clip [16,24]`, `attn_q_clip [128,24]`, `attn_sink_proj [2304,24]` (BF16), `attn_{k,v}_conv [4,1024]`, `attn_post_conv`/`ffn_post_conv [4,2304]`, global-only `attn_{k,v}_conv_proj [2304,32]` (BF16, = 8 KV heads × 4 taps) and `attn_{q,k,v}_inp_norm [2304]`, `attn_res_query [2304]` on layers 1–22 (SWA) / `[2304,3]` on global layers 5, 11, 17, 23 (layer 0 has none), `exp_probs_b [128]`, `ffn_res_query [2304]`; core layers 6–17: `loop_msg_query [2304,4]` and `loop_{q,k,v,ffn}_ctrl_down [4608,64]`, `_gain`/`_forget [64,2304]` (all F32).
- Types: see spec 03 §3 — NVFP4 experts for layers 0–19, Q8_0 for everything quantised otherwise (including layers 20–23 experts, `token_embd`, `output`), no K-quants.
- **Identity check:** the HF web viewer showed the same SHA-256 and size for the root and Mainline copies of this quant, contradicting the Mainline README's claim that tensor names/metadata differ. Not resolved by this fetch (hashing needs the full 17.6 GB); the Mainline file parsed here uses the `inkling2_loop` names above regardless.

## 8. Remaining open questions
**Resolved 2026-11-23 (this pass):** per-tensor types of `UD-Q4_K_XL` (spec 03 §3); SITU formula and its β/lb = config softcaps; `final_logit_softcapping 30.0` is written; `exp_probs_b` is selection-only; sigmoid/normalise/×8.0 routing; NVFP4 block layout and per-expert `.scale`; GGUF metadata and tensor names (§7); chat template embedded in the GGUF; llama.cpp gives every layer index its own attention KV and recurrent state (hybrid-iswa memory with all-true attention/recurrent filters; whether loop slots are distinct layer indices is consistent with this but the slot-table code was not read).
**Still open:**
1. `UD-Q4_K_M`, `UD-IQ4_XS`, `UD-Q8_K_XL` per-tensor types (each header is ~16 MiB; read before choosing the Q8 reference file's kernels — expected Q8_0 experts, unverified).
2. `attn_sink_proj` semantics (BF16 `[2304,24]`; sink folded into the relative-bias table per [L]) and `attn_gate` being per-channel `[2304,3072]` not per-head — confirm with the llama.cpp kernel when writing the attention reference.
3. Exact E4M3 vs UE4M3 decoding of NVFP4 sub-block scales (two llama.cpp headers disagree) — read the dequant routine in ggml before writing the kernel.
4. **Generation prompt:** the embedded template ends with the thinking-effort system message and emits **no** assistant header (confirmed by reading its tail). llama.cpp's parser appends `<|message_model|>` only when *continuing* an assistant message. Expectation (`[I]`): the model itself generates `<|message_model|>` as its first token. Verify on llama.cpp in P1 by logging the first generated token.
5. Rollback semantics for per-slot sconv state under speculation (design in spec 05).
6. Known upstream bugs affecting parity testing: CUDA perplexity NaN (#32710, open when seen), long-prompt garbage with experts on CPU (#32755, closed), chunked mask build (#32752, open), chat-parser grammar issue (#32745).
7. The `pre = inkling2` BPE pre-tokenizer must be matched: Astral should tokenise with the repo's `tokenizer.json` (HF `tokenizers`) and **test token-for-token equality against llama.cpp's tokeniser** on a corpus.

## 9. Reference-parity strategy
Because llama.cpp is a reimplementation, **Astral parity targets llama.cpp on the same GGUF** (as specified in spec 08) and is additionally checked against Hugging Face Transformers (`AutoModelForMultimodalLM`) in BF16 on a smaller-context sample where memory allows on the A100 (`[I]`: BF16 weights ≈ 53 GB). Disagreements between llama.cpp and Transformers are documented, not silently resolved.

## 10. Chat, tools, reasoning `[T]`
- Roles map to special tokens: `<|message_user|>`(200000), `<|message_model|>`(200001), `<|message_system|>`(200002), `<|message_tool|>`(200003); content kinds `<|content_text|>`(200004), `<|content_image|>`(200005), `<|content_thinking|>`(200008), `<|content_xml|>`(200024), `<|content_invoke_tool_json|>`(200049); `<|end_message|>`(200010); EOS `<|content_model_end_sampling|>`(200006); pad 199999. `developer` role raises an error.
- **Reasoning effort:** parameter `reasoning_effort`; map `none 0.0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh 0.99, max 0.99`; numeric in [0, 0.99]; default 0.9; rendered as a system message "Thinking effort level: <n>" before the first non-system message. Card: minimum effort does not fully disable reasoning.
- **Tools:** declared in a system block `tool_declare<|content_xml|>` + compact sorted JSON; assistant calls rendered as `NAME<|content_invoke_tool_json|>{"name":NAME,"args":{…}}<|end_message|>`; tool results as `<|message_tool|>NAME<|content_text|>…<|end_message|>`. Arguments must be a parsed mapping.
- **Output parsing:** `tokenizer_config.json` carries a `response_template` parse schema: `reasoning_content` between `<|content_thinking|>`…`<|end_message|>`; `content` between `<|content_text|>`/`<|content_tool_error|>`/`<|content_invoke_tool_text|>`…`<|end_message|>`; `tool_calls` between `<|content_invoke_tool_json|>`…`<|end_message|>`. Astral's parser is driven by this schema (spec 07).
- Images: `<|content_image|><|unused_200054|><|end_message|>`; audio rejected.
- Sampling: `generation_config.json` has only `do_sample true, temperature 1.0`; Unsloth recommends temperature 1.0, top-p 1.0, top-k 0, min-p 0 [G].
- No vLLM/SGLang parser names or flags are given on the card or blog [T] (recipe pages not read).

## 11. Vision (R-MD-3, P2) `[M, G]`
23-layer ViT (hidden 1024, 16 heads, patch 20, temporal patch 2, merge 2, projection 4096); side length 40–4096 px; GGUF `mmproj-{F16,BF16,F32}.gguf`. Image placeholder token `200054`. Encoder is a separate stage; embeddings cached in `VisionCache`.

## 12. Speculation heads
Native `mtp.safetensors` (3 layers, `block_size 8`, `target_layer_ids [5,23,41,48]` — probably slot indices over the 48 applications, inferred) [M]; GGUF path uses the DSpark drafter (spec 05).

## 13. Execution graph for `model` crate
`LayerSpec { kind: Local|Global, moe, attn_res_block_start: bool }`, `LoopSpec { core: 6..18, passes: 3, gate_rank: 64, controllers: [Q,K,V,FFN] }`; `slot_to_phys()` mirrors `layer_phys`. The loop driver lives in `model`/`runtime`; kernels see only per-slot ops. One CUDA graph per batch bucket covers all 48 slots.

## 14. Acceptance (R-MD-2)
On identical GGUF and prompts: perplexity within the llama.cpp-vs-llama.cpp noise floor of ≈7.65 on UD-Q8_K_XL; KLD/top-token agreement vs llama.cpp logits above thresholds set at P1 start; greedy first-128-token identity on deterministic prompts. See spec 08.
