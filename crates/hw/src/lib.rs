// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Synthetic Minds LTD
//! `astral-hw`: builds a [`SystemReport`] from the CUDA driver (authoritative) and NVML
//! (optional), and optionally measures memory bandwidth (spec 01 §3, §6).

#![forbid(unsafe_code)]

pub mod bench;

use astral_core::arch::{derived_dram_bandwidth_gbps, ArchProfile, Capabilities, MemoryModel, Validation};
use astral_core::Result;
use astral_gpu::{attr, nvml, Context, Device};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct NvmlReport {
    pub pcie_gen_current: Option<u32>,
    pub pcie_gen_max: Option<u32>,
    pub pcie_width_current: Option<u32>,
    pub pcie_width_max: Option<u32>,
    pub power_limit_w: Option<f64>,
    pub sm_clock_mhz: Option<u32>,
    pub mem_clock_mhz: Option<u32>,
    pub max_sm_clock_mhz: Option<u32>,
    pub max_mem_clock_mhz: Option<u32>,
    pub nvlink_active_links: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DeviceReport {
    pub ordinal: usize,
    pub name: String,
    pub uuid: String,
    pub capabilities: Capabilities,
    pub free_mem_bytes: u64,
    pub nvml: Option<NvmlReport>,
    pub memory_budget: MemoryBudget,
    pub bench: Option<bench::BenchResult>,
    pub notes: Vec<String>,
}

/// Whether and how to run the bandwidth microbenchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchMode {
    Skip,
    /// Reuse a cached result for this GPU/driver/clock state if one exists.
    Cached,
    /// Always measure (and refresh the cache).
    Fresh,
}

/// Memory Astral may plan with on this device (R-HW-3, spec 01 §4).
#[derive(Debug, Clone, Serialize)]
pub struct MemoryBudget {
    pub usable_bytes: u64,
    /// Where the figure came from: `cuMemGetInfo` (discrete) or `MemAvailable` (unified).
    pub source: &'static str,
    pub reserve_bytes: u64,
    pub utilization: f64,
}

/// Default share of free device memory Astral plans with on discrete GPUs.
pub const DEFAULT_UTILIZATION: f64 = 0.90;
/// Headroom kept free on unified-memory systems, where OOM can wedge the host.
pub const DEFAULT_UMA_RESERVE: u64 = 8 << 30;

/// Parses `MemAvailable` (kB) from `/proc/meminfo` text, in bytes.
pub fn parse_mem_available(meminfo: &str) -> Option<u64> {
    meminfo.lines().find_map(|l| {
        let rest = l.strip_prefix("MemAvailable:")?;
        let kb: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
        Some(kb * 1024)
    })
}

/// Computes the plannable memory budget.
///
/// - Discrete: `cuda_free x utilization`.
/// - Unified (keyed on the INTEGRATED flag, not on sm_121): `cuMemGetInfo` counts
///   reclaimable page cache as free, so plan with `min(cuda_free, MemAvailable) - reserve`.
///   If `/proc/meminfo` is unreadable, fall back to `cuda_free / 2 - reserve` (conservative).
pub fn memory_budget(model: MemoryModel, cuda_free: u64, meminfo: Option<&str>, utilization: f64, uma_reserve: u64) -> MemoryBudget {
    match model {
        MemoryModel::Discrete => MemoryBudget { usable_bytes: (cuda_free as f64 * utilization) as u64, source: "cuMemGetInfo", reserve_bytes: 0, utilization },
        MemoryModel::Unified => {
            let (avail, source) = match meminfo.and_then(parse_mem_available) {
                Some(a) => (a.min(cuda_free), "MemAvailable"),
                None => (cuda_free / 2, "cuMemGetInfo/2 (meminfo unreadable)"),
            };
            MemoryBudget { usable_bytes: avail.saturating_sub(uma_reserve), source, reserve_bytes: uma_reserve, utilization: 1.0 }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SystemReport {
    /// e.g. "13.0" from `cuDriverGetVersion`.
    pub cuda_driver_api: String,
    pub nvml_driver_version: Option<String>,
    pub kernels_built_for: Vec<&'static str>,
    pub nvcc: &'static str,
    pub devices: Vec<DeviceReport>,
}

fn flag(dev: &Device, a: i32) -> Option<bool> {
    dev.attribute(a).ok().map(|v| v != 0)
}

fn khz_to_mhz(v: Option<i32>) -> Option<u32> {
    v.filter(|&k| k > 0).map(|k| (k / 1000) as u32)
}

/// Reads the device's capabilities. Unknown compute capabilities are reported, not guessed.
pub fn capabilities(dev: &Device) -> Result<Capabilities> {
    let (major, minor) = dev.compute_capability()?;
    let profile = ArchProfile::from_compute_capability(major, minor);
    let u = |a: i32| -> Result<u32> { Ok(dev.attribute(a)?.max(0) as u32) };
    let mem_clock_mhz = khz_to_mhz(dev.attribute(attr::MEMORY_CLOCK_RATE_KHZ).ok());
    let bus = u(attr::GLOBAL_MEMORY_BUS_WIDTH)?;
    Ok(Capabilities {
        compute_capability: (major, minor),
        profile,
        kernel_family: profile.map(|p| p.kernel_family()),
        validation: profile.map(|p| p.validation()),
        isa: profile.map(|p| p.isa_features()),
        sm_count: u(attr::MULTIPROCESSOR_COUNT)?,
        warp_size: u(attr::WARP_SIZE)?,
        max_threads_per_block: u(attr::MAX_THREADS_PER_BLOCK)?,
        max_threads_per_sm: u(attr::MAX_THREADS_PER_MULTIPROCESSOR)?,
        regs_per_sm: u(attr::MAX_REGISTERS_PER_MULTIPROCESSOR)?,
        smem_per_sm_bytes: u(attr::MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)?,
        smem_per_block_optin_bytes: u(attr::MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)?,
        l2_bytes: u(attr::L2_CACHE_SIZE)? as u64,
        sm_clock_mhz: khz_to_mhz(dev.attribute(attr::CLOCK_RATE_KHZ).ok()),
        mem_clock_mhz,
        mem_bus_width_bits: bus,
        total_mem_bytes: dev.total_mem()? as u64,
        memory_model: if flag(dev, attr::INTEGRATED) == Some(true) { MemoryModel::Unified } else { MemoryModel::Discrete },
        ecc_enabled: flag(dev, attr::ECC_ENABLED),
        vmm_supported: flag(dev, attr::VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED),
        memory_pools_supported: flag(dev, attr::MEMORY_POOLS_SUPPORTED),
        managed_memory: flag(dev, attr::MANAGED_MEMORY),
        pageable_memory_access: flag(dev, attr::PAGEABLE_MEMORY_ACCESS),
        concurrent_managed_access: flag(dev, attr::CONCURRENT_MANAGED_ACCESS),
        cluster_launch: flag(dev, attr::CLUSTER_LAUNCH),
        dram_bandwidth_gbps_derived: mem_clock_mhz.filter(|_| bus > 0).map(|m| derived_dram_bandwidth_gbps(m, bus)),
    })
}

fn notes_for(caps: &Capabilities) -> Vec<String> {
    let mut n = Vec::new();
    let (maj, min) = caps.compute_capability;
    match caps.profile {
        None => n.push(format!("compute capability {maj}.{min} is not an Astral target (supported: 8.0, 9.0, 10.0, 10.3, 12.0, 12.1)")),
        Some(p) if p.validation() == Validation::Specified => n.push(format!("{p:?}: kernels for this architecture are specified but UNVALIDATED on real silicon (spec 08)")),
        Some(_) => {}
    }
    if caps.memory_model == MemoryModel::Unified {
        n.push("unified memory: free memory from cuMemGetInfo over-reports; budget from MemAvailable instead (spec 01 §4)".into());
    }
    if caps.dram_bandwidth_gbps_derived.is_some() {
        n.push("dram_bandwidth_gbps_derived assumes double data rate; trust the measured bench figure".into());
    }
    n
}

/// Probes every CUDA device. With `run_bench`, also measures memory bandwidth.
pub fn probe(bench_mode: BenchMode) -> Result<SystemReport> {
    let v = astral_gpu::driver_version()?;
    let nv = nvml::nvml();
    let mut devices = Vec::new();
    for i in 0..Device::count()? {
        let dev = Device::get(i)?;
        let caps = capabilities(&dev)?;
        let uuid = dev.uuid()?;
        let ctx = Context::new(dev)?;
        let (free, _) = ctx.mem_info()?;
        let nvml_info = nv.and_then(|m| m.device_info(&uuid)).map(|d| NvmlReport {
            pcie_gen_current: d.pcie_gen_current,
            pcie_gen_max: d.pcie_gen_max,
            pcie_width_current: d.pcie_width_current,
            pcie_width_max: d.pcie_width_max,
            power_limit_w: d.power_limit_w,
            sm_clock_mhz: d.sm_clock_mhz,
            mem_clock_mhz: d.mem_clock_mhz,
            max_sm_clock_mhz: d.max_sm_clock_mhz,
            max_mem_clock_mhz: d.max_mem_clock_mhz,
            nvlink_active_links: d.nvlink_active_links,
        });
        let mut notes = notes_for(&caps);
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok();
        let budget = memory_budget(caps.memory_model, free as u64, meminfo.as_deref(), DEFAULT_UTILIZATION, DEFAULT_UMA_RESERVE);
        let key = bench::cache_key(&uuid, &format!("{v}"), nvml_info.as_ref().and_then(|n| n.max_sm_clock_mhz), nvml_info.as_ref().and_then(|n| n.max_mem_clock_mhz));
        let bench = if bench_mode != BenchMode::Skip {
            match bench::run_cached(&ctx, &caps, &key, bench_mode == BenchMode::Fresh) {
                Ok(b) => Some(b),
                Err(e) => {
                    notes.push(format!("bandwidth bench skipped: {e}"));
                    None
                }
            }
        } else {
            None
        };
        devices.push(DeviceReport { ordinal: i, name: dev.name()?, uuid, capabilities: caps, free_mem_bytes: free as u64, memory_budget: budget, nvml: nvml_info, bench, notes });
    }
    Ok(SystemReport {
        cuda_driver_api: format!("{}.{}", v / 1000, (v % 1000) / 10),
        nvml_driver_version: nv.and_then(|m| m.driver_version()),
        kernels_built_for: astral_kernels::BUILT_ARCHS.to_vec(),
        nvcc: astral_kernels::NVCC_VERSION,
        devices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    const MEMINFO: &str = "MemTotal:       131072000 kB\nMemFree:         2000000 kB\nMemAvailable:   100000000 kB\nBuffers:          123 kB\n";

    #[test]
    fn parses_mem_available() {
        assert_eq!(parse_mem_available(MEMINFO), Some(100_000_000 * 1024));
        assert_eq!(parse_mem_available("MemTotal: 1 kB\n"), None);
    }

    #[test]
    fn discrete_uses_cuda_free_times_utilization() {
        let b = memory_budget(MemoryModel::Discrete, 40 * GIB, Some(MEMINFO), 0.9, DEFAULT_UMA_RESERVE);
        assert_eq!(b.usable_bytes, (40.0 * GIB as f64 * 0.9) as u64);
        assert_eq!(b.source, "cuMemGetInfo");
    }

    #[test]
    fn unified_ignores_page_cache_and_keeps_reserve() {
        // cuMemGetInfo claims 120 GiB free (page cache counted); MemAvailable says ~95 GiB.
        let b = memory_budget(MemoryModel::Unified, 120 * GIB, Some(MEMINFO), 0.9, 8 * GIB);
        assert_eq!(b.usable_bytes, 100_000_000 * 1024 - 8 * GIB);
        assert_eq!(b.source, "MemAvailable");
    }

    #[test]
    fn unified_without_meminfo_is_conservative() {
        let b = memory_budget(MemoryModel::Unified, 120 * GIB, None, 0.9, 8 * GIB);
        assert_eq!(b.usable_bytes, 60 * GIB - 8 * GIB);
    }

    #[test]
    fn cache_key_changes_with_clock_limits() {
        let a = bench::cache_key("GPU-x", "13000", Some(1410), Some(1215));
        let b = bench::cache_key("GPU-x", "13000", Some(1200), Some(1215));
        assert_ne!(a, b);
    }
}
