# Read-only probes of the looped forward pass

`astral trace <model> <manifest> <out dir>` records what happens inside the 48 layer visits of
Inkling 2 Mini at chosen token positions, without changing the computation. Set up for
interpretability work on the loop (which blocks a visit reads, what the controllers do with the
previous pass's message, how routing and attention change across passes).

## How it works

- Only decode steps are probed. The manifest (`name<TAB>L1,L2,...<TAB>ids`, as for
  `dump-final-logits`) is fed token by token; the step that consumes token `L - 1` is probed
  for each listed `L`, and the next token, if present, is the readout target.
- A probed step runs eagerly with the kernels of the graph replay (the fused paths stay on).
  Hooks queue device-to-device copies of small buffers into one staging buffer on the same
  stream; nothing synchronises until the step ends, when the staging is read back once.
- Derived quantities are recomputed on the host in f64 from the kernels' formulas and compared
  with the device output. Each record carries the relative error of the reconstruction
  (`recompute_rel_err`), and the head record of each position carries the largest error per
  probe (`checks`). Errors around 1e-6 mean the reading of the kernel is right.
- With probes off, the code paths are unchanged. On an RTX 4090 (80 positions): probes off,
  probes on and `ASTRAL_NO_GRAPH=1` gave bit-identical logits, and decode speed was unchanged.

Not covered: prefill and verify passes (a probed position always takes the decode path), the
`ASTRAL_NO_BATCH_CTRL=1` controller path (its `cd` buffer is overwritten between controllers),
and coda-lens readouts (only the raw lens below).

Environment: `ASTRAL_TRACE_VALIDATE=all` recomputes the attention output (reads the V cache) at
every probed position (default: the first one of each sequence); `ASTRAL_TRACE_LENS=0` skips the
raw lens; `ASTRAL_TRACE_PREFILL=1` prefills each sequence up to its first probed step
(so the probed attention reads KV written by the prefill path); `ASTRAL_TRACE_FLOATS` sizes the staging buffer (default 24 Mi floats, 96 MB).

## Records

`trace.jsonl` holds one JSON object per layer visit and one per head, in execution order.
`<name>_<L>.trace.f32` is the probed step's final logits.

Common fields:

| Field | Meaning |
|---|---|
| `seq`, `len` | manifest name and prefix length L (the token at zero-based position `pos` = L - 1 was consumed) |
| `pos`, `token`, `target` | position, the token consumed, the next token of the sequence (or null) |
| `visit` | logical visit 0..47 in execution order (null on the head record) |
| `slot` | cache slot (KV and conv state) of the visit |
| `layer` | physical layer 0..23 (parameters shared across passes) |
| `pass` | 0, 1, 2 for core layers 6..17; null in prelude and coda |
| `stage` | `prelude`, `core`, `coda` or `head` |

Per visit (present when the layer has them):

| Field | Content |
|---|---|
| `attn_mix`, `ffn_mix` | attention-residual depth mixes before attention and before the MoE: `sources` (`vN` = the bank row written at visit N, `cur` = the current stream), softmax weights `p` per query, output norms |
| `msg_mix` | passes 1–2: the cross-pass message mix over the previous pass's blocks and its output (4 queries: Q, K, V and FFN messages) |
| `ctrl` | passes 1–2, controllers `q`, `k`, `v`, `ffn`: `out = x - tanh(f) x + tanh(g) msg`. Distributions (mean, mean of abs, p01–p99) of `gain` = tanh(g), `forget` = tanh(f), `keep` = 1 - forget; norms of x, msg, the forget and gain terms and the output; cos(x, msg), cos(out, x). tanh values are not probabilities: keep can exceed 1 and gain can be negative |
| `router` | the picked expert ids and weights, the top k + 1 selection scores (sigmoid + bias) with probabilities, the k-th vs (k+1)-th margin, and whether the host recompute picks the same ids |
| `attn` | from `q . k / sqrt(hd) + relative bias - sink` over the window (sink logit 0): mean over heads of the probability mass on the sink, the token itself, distance 1–63, 64–1023 and ≥ 1024; per head sink and far mass, entropy and argmax distance |
| `stream` | the residual stream after the visit: RMS, max abs, median and p99 of abs, the 8 largest coordinates `[index, value, energy share]`, the energy share of the top 4 |
| `lens_raw` | raw lens (output norm and LM head applied to the stream): entropy, top 5 `[id, prob]`, the target's log-prob and rank |

Head record: `head_mix` (the output depth mix), `native` (readout of the final logits) and
`checks`.
