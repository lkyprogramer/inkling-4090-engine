# 02 — Workspace, crates and runtime

Requirements: R-RT-1..4. The layout is a design proposal informed by Atlas's trait structure and Dynamo's Rust/Python split; it has **not been prototyped** [R]. Trait granularity vs hot-loop dispatch cost is unmeasured: use enum dispatch/generics in the step loop, trait objects only at setup.

## 1. Crates (17)
| Crate | Responsibility | Key public interface |
|---|---|---|
| `core` | IDs, errors, config, dtype/shape types, tracing glue. No GPU dependency | `RequestId`, `DType`, `AstralError`, `Config` |
| `hw` | Probing, `ArchProfile`, `Capabilities`, microbench, topology | `probe() -> Vec<DeviceInfo>` (spec 01) |
| `gpu` | **Only `unsafe` boundary.** Context, streams, events, graphs, device buffers, module loader | `trait GpuBackend`; `Stream`, `Event`, `Graph`, `DeviceBuffer<T>` |
| `mem` | Pinned host pool, VMM arena, slab classes, byte-budget arbiter | `Arena`, `SlabPool<T>`, `BudgetArbiter` |
| `kernels` | Manifest, fatbin loader, typed launch wrappers | `KernelRegistry::select(op, ArchProfile, shape)` |
| `kernels-src` | `.cu`/CUTLASS (and optional Triton/CuTe AOT) sources, built to cubin by `build.rs`/xtask | build artefacts only |
| `quant` | Quant type descriptors (GGML K/IQ/Q types; NVFP4/MXFP4/MXFP8), packing, CPU reference dequant | `QuantType`, `dequant_ref()` |
| `loader` | GGUF (primary) and safetensors readers, mmap, concurrent upload | `trait WeightSource`, `load_model()` |
| `model` | Layer/cache specs, model implementations, loop driver | `trait Model`, `LayerSpec`, `CacheSpec` |
| `kvcache` | Typed pools, prefix tree, offload tiers (spec 04) | `KvManager`, `trait StorageBackend` |
| `sched` | Policy, continuous batching, chunked prefill, preemption, speculation control | `Scheduler::plan(&State) -> BatchPlan` |
| `runtime` | Engine thread, graph buckets, step pipeline, sampler, grammar masks | `Engine::run()`, `StepPipeline` |
| `api` | OpenAI-compatible server, SSE, parsers | axum routers, `ChatTemplate` |
| `obs` | Prometheus, OpenTelemetry, NVTX hooks | `Metrics`, `trace_span!` |
| `testing` | Mock backend, golden harness, bench harness | `MockBackend`, `Golden`, `Bench` |

Dependency direction (no cycles), as implemented in P0: `core` (incl. `ArchProfile`/`Capabilities` types) ← `gpu` (driver + NVML FFI) ← `kernels-src` / `kernels` ← `hw` (probing + microbench, needs kernels) ← `testing` ← `cli`; later `mem` ← {`quant`,`loader`,`kvcache`,`model`} ← `sched` ← `runtime` ← `api`. Capability *types* live in `core` so kernel selection never depends on `hw`. Multi-node components plug in at `runtime` and are part of the Enterprise edition, which builds on these crates.

## 2. The `gpu` boundary (R-RT-1)
All raw driver calls go through `cudarc::driver::sys` or a thin in-house layer. cudarc covers driver, NVRTC, cuBLAS/cuBLASLt, NCCL, cuDNN across CUDA 11.4–13.4 under Apache-2.0/MIT [R], but **its safe layer was not confirmed to wrap CUDA graphs, VMM (`cuMemCreate`/`cuMemMap`), multicast or TMA descriptors** [R] — plan to maintain a thin layer over `sys` or contribute upstream.
- `DeviceBuffer<T>` carries a lifetime tied to its `Context`; kernel launches take typed arg structs generated per kernel, not `*mut c_void` arrays.
- Stream ordering protects only same-stream work; cross-stream use requires an explicit `Event` (API makes it unrepresentable otherwise).
- Graph/exec/memory drop order is solved by arena ownership: graphs borrow from the arena, arena outlives them.
- Dynamic loading: binary starts without CUDA present and reports a clear error.

## 3. Threading (R-RT-2)
```
Tokio multi-thread  : HTTP/SSE/gRPC, auth, request parsing
Blocking pool       : tokenisation, template rendering, grammar compile
Engine thread (pinned OS thread, no .await):
      loop { drain request channel → sched.plan() → build inputs (pinned buf)
             → launch step (graph replay or eager) → record event
             → while GPU runs: prepare step n+1 }
Output thread       : sample results → detokenise → push to per-request SSE channel (overlaps step n+1)
```
- Bounded lock-free channels (e.g. `crossbeam`/`flume` class `[I]`); backpressure surfaces as HTTP 429/queueing, never unbounded growth.
- Tokio is cooperative and starves tasks that never yield; Tokio+Rayon oversubscription causes p99 spikes under CPU throttling [R] — **do not** use Rayon in the hot path; fixed thread counts, documented.
- Evidence caveat [R]: the 2024 vLLM profile (33% HTTP, 29% scheduling/input prep, 38% GPU) is outdated; the Rust advantage vs current vLLM must be measured in P1/P2.

## 4. Graph management (R-RT-4)
Capture per batch-size bucket (default set `{1,2,4,8,16,24,32,48,64}` `[I]`; trimmed to reduce capture time, a cold-start lever [R]). Static device addresses come from the VMM arena. Prefill runs eagerly (variable shapes); decode and speculative-verify use graphs. Graph pools are accounted in the memory budget (spec 04).

## 5. Mock backend (R-RT-3)
`testing::MockBackend` implements `GpuBackend` on CPU (slow reference kernels, simulated memory). Used for: scheduler/KV/API unit and property tests, CI without GPUs, and the golden harness. Also the executable form of "unvalidated arch" behaviour: selecting an `ArchProfile` whose kernels cannot run locally falls back to mock for logic tests only.

## 6. Configuration
Single typed `Config` (TOML + CLI + env). Autodetected defaults from `hw`; every autodetected value is logged with its source and can be overridden. `astral doctor` prints detection, driver/toolkit compatibility, memory budget and the kernel selected for each op.

## 7. Build and packaging
Cargo workspace; `kernels-src` built via `build.rs`/`xtask` with pinned CUDA 13.x toolchain; optional Triton/CuTe AOT step requires Python **at build time only** (pinned, artefacts cached in CI) [R]. Release artefact: one binary + fatbin (aarch64 and x86_64); users need only the NVIDIA driver.
