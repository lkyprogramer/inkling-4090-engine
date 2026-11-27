# Speculative Decoding

Astral accelerates decoding with the draft model that ships with the checkpoint: the DSpark draft of Inkling 2 Mini (`mtp.safetensors`, `mtp_config` in `config.json`). Greedy output is unchanged: every token is verified by the target model.

## How it works

- **The draft** is a 3-layer dense transformer.
  - Each target token leaves four hidden states: the residual stream entering layer visits 5, 23 and 41, plus the final-norm output.
  - Every draft layer mixes them and projects them into its own K/V cache.
  - A draft step runs one 8-token block (the last accepted token, then 7 mask tokens) bidirectionally over that cache.
  - A recurrent Markov head turns the mask outputs into drafted tokens, left to right.
- **A round** verifies the anchor and *k* drafts in one batched target pass. It accepts the matching prefix plus the target's own next token, and rolls the model's short-convolution states back to the accepted prefix. Then it extends the draft's cache and drafts the next block.
- **Each round is one CUDA graph,** covering verification, acceptance, rollback and drafting. The host only reads back the new tokens.

## Adaptive draft length

The draft can propose up to 7 tokens per round, but verifying more drafts costs more per round, and acceptance varies with the text. Astral picks *k* every round:

- after a round that accepted every draft, it verifies one more draft next time;
- after an early rejection, it verifies one fewer;
- it stays between `ASTRAL_SPEC_KMIN` (default 2) and the server's `--k` (default 6).

A round verifying *k* drafts only drafts the *k* + 1 tokens the next round can use.

On a 16-prompt benchmark, adaptive length reached 241.2 tok/s against 234.6 tok/s for a fixed *k* = 3.

## Tuning

| Variable | Default | Effect |
|---|---|---|
| `ASTRAL_SPEC_FIXED=1` | off | Verify exactly `--k` drafts every round |
| `ASTRAL_SPEC_KMIN` | `2` | Fewest drafts the adaptive policy verifies |
| `ASTRAL_SPEC_POLICY=model` | burst rule | Pick *k* from per-position acceptance rates and measured round times instead |
| `ASTRAL_SPEC_GROW`, `ASTRAL_SPEC_SHRINK` | `1`, `dec` | Burst-rule step sizes (`dec`, `jump`, `half`) |
| `ASTRAL_DRAFT_HEAD_FP4=1` | off | NVFP4 copy of the LM head for the draft's base logits |
| `ASTRAL_DRAFT_KV_F32=1` | off (f16) | Draft K/V stored as f32 (twice the draft cache). The f16 default changes the drafts in rounding only: identical generated tokens, rounds and acceptance at 16K and 100K |
| `ASTRAL_DRAFT_KV_FULL=1` | off | Keep every position for the draft's sliding-window layers too (by default they keep a ring of `max(window + block, tmax)` rows; the drafts are bit-identical either way) |
| `ASTRAL_SPEC_LOG=<file>` | off | Append one JSON line per round: position, drafts verified, accepted, wall ms |

## Correctness

Verification uses batched kernels, which differ from one-token decoding by up to about one logit. `astral spec-generate` reruns plain decoding along the speculative output and reports every position where the speculative token is not plain decoding's top choice.

On 32 benchmark prompts with the default settings, mismatches occur only at near-ties, where the logit gap is at most 0.95.
