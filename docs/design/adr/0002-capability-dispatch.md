# ADR-0002: Dispatch on capabilities, never on `major >= N`

**Status:** Proposed · **Date:** 2026-11-23

## Context
vLLM's backend auto-selection on GB10 routed any capability ≥100 (including sm_121) to FlashInfer CUTLASS paths that need tcgen05/sm_100a templates, producing `ptxas` failures and NaN logits [R]. sm_100/103 and sm_120/121 differ in TMEM, tcgen05, shared memory and cluster support.

## Decision
`hw` derives `ArchProfile` (six variants) and a `Capabilities` struct. Kernel manifest selection keys on `(op, quant, ArchProfile, shape class)` with explicit fallback chain (exact → shared base family → reference). Unknown CC fails loudly. Memory accounting keys on the **integrated** flag, not on sm_121.

## Consequences
+ New architecture = one enum variant, one capability row, one kernel family.
+ A table-driven test asserts no `major >= N` branches.
− More upfront tables; unverified spec cells must be datasheet-confirmed.
