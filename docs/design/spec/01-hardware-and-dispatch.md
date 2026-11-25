# 01 — Hardware detection, abstraction and dispatch

Crate: `hw`. Requirements: R-HW-1..5.

## 1. Target chips
| Chip | CC | Memory / bandwidth | Key ISA/precision notes | Local test? |
|---|---|---|---|---|
| A100 | 8.0 | **Dev server: A100-PCIE-40GB, 40,960 MiB, driver 580.105.08, max SM clock 1410 MHz, memory clock 1215 MHz (measured via `nvidia-smi`, 2026-11-23)**; bandwidth ~1,555 GB/s `[UNVERIFIED]` spec value, measure with the startup microbench | `mma.sync`, `cp.async`, BF16/FP16/INT8 tensor cores; **no FP8/FP4** | **Yes (only)** |
| H100 | 9.0 | 80 GB, ~3.35 TB/s `[UNVERIFIED]` | WGMMA, TMA, FP8, FA3 | No |
| B200 | 10.0 | 192 GB HBM3e, 8 TB/s [R] | tcgen05, TMEM, 2-CTA MMA, NVFP4/MXFP4/MXFP8 | No |
| GB300/B300 | 10.3 | 288 GB HBM3e, 8 TB/s; NVLink 1,800 GB/s/GPU [R] | as B200, tuned for NVFP4, reduced INT8 [R] | No |
| RTX PRO 6000 | 12.0 | 96 GB GDDR7, 1,792 GB/s, no NVLink [R]; SM count `[UNVERIFIED]` | block-scaled FP4 via warp-level `mma.sync`; ~100 KB smem/SM; no TMEM/tcgen05; no cluster path [R] | No |
| GB10 | 12.1 | 128 GB unified LPDDR5x, 273 GB/s spec, ~200–213 GB/s measured (community) [R] | as sm_120; unified memory | No |

All `[UNVERIFIED]` cells must be replaced from NVIDIA datasheets before they enter `Capabilities` tables in code.

## 2. Types
```rust
pub enum ArchProfile { Sm80, Sm90a, Sm100a, Sm103a, Sm120a, Sm121a }   // kernel-family selection
pub struct Capabilities {
    pub cc: (u32,u32), pub sm_count: u32,
    pub max_smem_optin: u32, pub l2_bytes: u64,
    pub has_wgmma: bool, pub has_tma: bool, pub has_tcgen05: bool, pub has_tmem: bool,
    pub has_clusters: bool, pub has_cp_async: bool,
    pub fp8: Fp8Support, pub fp4: Fp4Support,        // None | Mma | Tcgen05 ; formats: nvfp4/mxfp4/mxfp8
    pub int8_tc: bool,
    pub memory: MemoryModel,                          // Discrete{hbm_bytes} | Unified{total, reclaimable_probe}
    pub bw_spec_gbps: f32, pub bw_measured_gbps: Option<f32>,
    pub link: Topology,                               // see §5
}
```
`ArchProfile` is derived from `cc` plus an explicit table; **`Capabilities` is what kernels and the planner consult**. Unknown CC → `ArchProfile::Unsupported` with a clear error, not a silent fallback to the nearest arch.

## 3. Detection (R-HW-1)
Authoritative source: **CUDA driver API** (`cuDeviceGetAttribute`: compute capability, SM count, `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`, memory clock, bus width, `INTEGRATED`, `PAGEABLE_MEMORY_ACCESS`, `CONCURRENT_MANAGED_ACCESS`). NVML (via raw `-sys` bindings; `nvml-wrapper` targets an older NVML [R]) for topology, NVLink, fabric info (`GpuFabricInfo` cluster UUID/clique ID identifies NVL72 domains [R]), clocks, ECC. Every NVML call tolerates `NOT_SUPPORTED`. cudarc loads libraries dynamically so the binary needs no toolkit at build/run beyond the driver [R].

`astral hw` CLI prints the full `Capabilities` JSON and is the first deliverable of P0.

## 4. Memory accounting (R-HW-3)
- `Discrete`: budget = `cuMemGetInfo` free × `gpu_memory_utilization` (default 0.90) minus analytic reserve (spec 04).
- `Unified` (GB10; keyed on the **integrated flag, not sm_121** [I]): NVML memory queries return `NOT_SUPPORTED`, `cudaMemGetInfo` counts reclaimable page cache as free, cgroup limits reportedly do not bound CUDA allocations, and OOM can wedge the host [R]. Therefore: budget from `/proc/meminfo` `MemAvailable` with an explicit headroom reserve (default 8 GiB `[I]`), load weights with `O_DIRECT` or `posix_fadvise(DONTNEED)` after load, treat "CPU offload" as meaningless (one pool), and add a watchdog that refuses admission when available memory falls below the reserve.

## 5. Interconnect and multi-GPU (detection only; multi-node serving is part of the Enterprise edition)
Enumerate NVLink via NVML, PCIe topology via sysfs, NUMA via `/sys/devices/system/node`, InfiniBand/RoCE via `/sys/class/infiniband` `[I]` (general knowledge, unsourced). NVLink-C2C bandwidth (900 GB/s vendor claim), IMEX and NVL72 detection were **not sourced** [R]; mark `Topology` fields `Option` and do not branch on them until verified on hardware.

## 6. Startup microbenchmark (R-HW-4)
100–300 ms: device-memory triad (bandwidth) and a small GEMM (clock behaviour). Purpose: detect GPUs stuck in low power states and record measured vs spec bandwidth (GB10 measured ~75% of spec [R]). Cache keyed by (device UUID, driver version, clock state). Result feeds the roofline estimates in the planner, never a hard gate.

## 7. Compilation and dispatch (R-HW-5)
- **Per-arch cubins in a fatbin, built with CUDA 13.x** (`-gencode`, **not** `-arch=sm_121a`, which may silently drop the `a` feature set [R]). `a` targets have no forward compatibility; `compute_90a` PTX does not run on Blackwell [R]. PTX only for the portable subset.
- Family targets (`sm_100f` covers 10.0/10.3, CUDA 12.9+ [R]). Whether `12.0f` covers 12.1 is inferred, not stated by NVIDIA [R]: ship explicit `sm_121a` cubins.
- **CI disassembly gate:** `cuobjdump -sass` per cubin asserts presence of the arch's characteristic instructions (e.g. block-scaled FP4 MMA on sm_120a, `tcgen05` on sm_100a, `HGMMA` on sm_90a). This is the only automated check for non-A100 arches that does not need hardware.
- NVRTC for shape-specialised kernels; cache on disk keyed by (arch, driver, NVRTC version, source hash).
- Build both `x86_64` and `aarch64` (GB10/Grace are Arm) [R].
- Driver minimums: 525 (CUDA 12.x) and 580 (CUDA 13.x) [R]. Minimum for sm_121 / sm_103 specifically, and which toolkit first supports `sm_103a`/`sm_121a`, **conflict between sources `[UNVERIFIED]`** — resolve with a real toolchain before P3.

## 8. Kernel selection
Manifest (`kernels` crate) maps `(op, dtype/quant, ArchProfile, shape class)` → kernel artifact + launch-config function. Selection order: exact arch family → shared base family (sm_12x falls back to the sm_80 `mma.sync` family) → reference implementation (slow, correct, marked in logs). **Never** select by `cc.major >= 10` — that rule is the documented source of vLLM's GB10 misrouting [R]. Adding a new arch = new `ArchProfile` variant + capability row + kernel family; the planner, scheduler, KV and API crates are untouched.

## 9. Open items
Datasheet confirmation for all `[UNVERIFIED]` cells; A100 SKU; toolkit/driver minimums for sm_103/121.
