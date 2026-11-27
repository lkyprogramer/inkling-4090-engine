# Environment Variables

The defaults are the fastest measured configuration. The variables below exist for building, tuning and diagnosing.

## Build

| Variable | Default | Description |
|---|---|---|
| `ASTRAL_CUDA_ARCHS` | `80` | Comma-separated kernel targets, e.g. `80,90a` |

## Runtime

| Variable | Default | Description |
|---|---|---|
| `ASTRAL_PDL` | on (cc ≥ 9.0) | `0` disables programmatic dependent launch |
| `ASTRAL_NO_GRAPH` | off | Run decode steps eagerly instead of as CUDA graphs |
| `ASTRAL_ATT_SPLITS` | `32` | Most attention splits in graph-captured multi-token passes |
| `ASTRAL_DECODE_DP4A` | off | dp4a instead of tensor-core GEMVs for decode projections (about 10% faster on an RTX 4090, but decode moves further from prefill; see `decode_q8_mma`) |
| `ASTRAL_SMALL_DP4A` | off (on for sm_89) | dp4a instead of tensor-core GEMVs for 2–8-token passes (DSpark verify and draft); `ASTRAL_SMALL_TC=1` forces tensor cores |
| `ASTRAL_LM_HEAD_Q8` | off | Qwen3.8-27B: quantise the BF16 LM head to Q8_0 at load |

## Speculative decoding

See [speculative decoding](../features/speculative_decoding.md#tuning): `ASTRAL_SPEC_FIXED`, `ASTRAL_SPEC_KMIN`, `ASTRAL_SPEC_POLICY`, `ASTRAL_SPEC_GROW`, `ASTRAL_SPEC_SHRINK`, `ASTRAL_SPEC_LOG`, `ASTRAL_DRAFT_HEAD_FP4`, `ASTRAL_DRAFT_KV_F16`, `ASTRAL_DRAFT_KV_FULL`.

## Fusion switches

Each fused kernel can be split back into its parts, for comparison: `ASTRAL_SIDE_BF16`, `ASTRAL_NO_FUSE_MIX`, `ASTRAL_NO_FUSE_ROUTER`, `ASTRAL_NO_BATCH_CTRL`. Their effect on speed is recorded in the [engineering log](../design/engineering-log.md).
