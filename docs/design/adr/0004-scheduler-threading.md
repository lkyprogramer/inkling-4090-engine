# ADR-0004: Pinned engine thread; Tokio only at the edges

**Status:** Proposed · **Date:** 2026-11-23

## Context
Tokio is cooperative and starves tasks that never yield; Tokio+Rayon oversubscription caused p99 spikes under CPU throttling in practitioner reports [R]. The vLLM 2024 profile (33% HTTP, 29% scheduling/input prep, 38% GPU) is outdated; V1 recovered much of it. SGLang's zero-overhead scheduler gave ~1.1× on a 3B model [R].

## Decision
A dedicated pinned OS thread runs plan → build inputs → launch with no `.await`. Tokio handles HTTP/SSE/gRPC; a blocking pool handles tokenisation/templates/grammar compile; a separate output thread detokenises and streams, overlapping step n+1. Bounded lock-free channels between them. No Rayon in the hot path.

## Consequences
+ Predictable tail latency; clear ownership of GPU launch.
− The Rust-vs-Python scheduler win versus current vLLM is **unmeasured**; P1/P2 must measure it before it is claimed.
