# ADR-0003: Typed multi-pool KV arena under one byte budget

**Status:** Proposed · **Date:** 2026-11-23

## Context
vLLM's hybrid KV manager pads all layer types to one page size, can force attention block sizes above 400 tokens for Mamba-style state, and limits prefix caching to two groups [R]. Inkling 2 Mini mixes 20 sliding-window and 4 global physical layers with a 3× looped core, giving 48 cache slots (8 global + 40 SWA) with **separate KV and short-conv state per pass** [L, resolved 2026-11-23], plus a drafter.

## Decision
Separate slab classes (`GlobalKv`, `SwaKv`, `ConvState`, `DraftKv`, `VisionCache`) carved from a VMM arena, governed by one `BudgetArbiter`. Pools are sized from a per-slot table produced by `model::slot_to_phys()` so variable-loop or future variants are configuration. Prefix cache spans all pool kinds (conv state via block-boundary checkpoints) with cost-aware eviction.

## Consequences
+ No padding waste; SWA blocks freed on window slide; fixed addresses for CUDA graphs.
− More allocator complexity; prefix semantics through SWA/loop need careful tests.
− Benefit is an inference from documented incumbent limits; not yet measured.
