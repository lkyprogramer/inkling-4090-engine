// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Synthetic Minds LTD
//! Startup memory-bandwidth microbenchmark (R-HW-4, spec 01 §6).
//!
//! Decode is bound by DRAM *reads*, so the read probe is the headline number used by
//! the planner's roofline. Copy and DMA figures are reported for context.

use astral_core::arch::Capabilities;
use astral_core::Result;
use astral_gpu::{Context, DeviceBuffer, Event, Stream};
use astral_kernels::ops::{CopyBandwidth, ReadBandwidth};
use astral_kernels::KernelRegistry;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    pub buffer_bytes: u64,
    pub iterations: u32,
    pub read_gbps_best: f64,
    pub read_gbps_median: f64,
    /// Bytes read + bytes written, per second.
    pub copy_gbps_best: f64,
    pub dma_dtod_gbps_best: f64,
    pub read_fraction_of_derived: Option<f64>,
    pub warnings: Vec<String>,
    /// True when this result was loaded from the on-disk cache rather than measured now.
    #[serde(default)]
    pub from_cache: bool,
}

/// Idle GPUs park their SM clock very low (210 MHz observed on the A100 dev server) and
/// take time to ramp; measuring immediately under-reports bandwidth by ~30%. Warm up for
/// a fixed wall-clock time, not a fixed iteration count.
const WARMUP: std::time::Duration = std::time::Duration::from_millis(300);
const ITERS: u32 = 20;

fn time_ms(ctx: &Arc<Context>, stream: &Stream, mut f: impl FnMut() -> Result<()>) -> Result<Vec<f64>> {
    let (start, end) = (Event::new(ctx)?, Event::new(ctx)?);
    let t = std::time::Instant::now();
    while t.elapsed() < WARMUP {
        f()?;
        stream.synchronize()?;
    }
    let mut out = Vec::with_capacity(ITERS as usize);
    for _ in 0..ITERS {
        start.record(stream)?;
        f()?;
        end.record(stream)?;
        end.synchronize()?;
        out.push(start.elapsed_ms(&end)? as f64);
    }
    Ok(out)
}

fn gbps(bytes: f64, ms: f64) -> f64 {
    bytes / (ms * 1e-3) / 1e9
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

pub fn run(ctx: &Arc<Context>, caps: &Capabilities) -> Result<BenchResult> {
    let reg = KernelRegistry::load(ctx)?;
    let read = ReadBandwidth::new(&reg)?;
    let copy = CopyBandwidth::new(&reg)?;
    let stream = Stream::new(ctx)?;

    // 1 GiB, or a quarter of free memory if smaller; a multiple of 4 floats.
    let (free, _) = ctx.mem_info()?;
    let bytes = (1usize << 30).min(free / 4) & !0xFFFF;
    let n = bytes / 4;
    let grid = caps.sm_count.max(1) * 8;

    let src = DeviceBuffer::<f32>::zeroed(ctx, n)?;
    let mut sink = DeviceBuffer::<f32>::zeroed(ctx, 1)?;
    let read_ms = time_ms(ctx, &stream, || read.launch(&stream, &src, &mut sink, grid))?;

    let half = (n / 2) & !3;
    let a = DeviceBuffer::<f32>::zeroed(ctx, half)?;
    let mut b = DeviceBuffer::<f32>::zeroed(ctx, half)?;
    let copy_ms = time_ms(ctx, &stream, || copy.launch(&stream, &a, &mut b, grid))?;
    let dma_ms = time_ms(ctx, &stream, || b.copy_from_device_async(&a, &stream))?;

    let read_best = gbps(bytes as f64, read_ms.iter().cloned().fold(f64::MAX, f64::min));
    let read_median = gbps(bytes as f64, median(read_ms));
    let moved = 2.0 * (half * 4) as f64;
    let copy_best = gbps(moved, copy_ms.iter().cloned().fold(f64::MAX, f64::min));
    let dma_best = gbps(moved, dma_ms.iter().cloned().fold(f64::MAX, f64::min));

    let fraction = caps.dram_bandwidth_gbps_derived.map(|d| read_best / d);
    let mut warnings = Vec::new();
    if let Some(f) = fraction {
        if f < 0.6 {
            warnings.push(format!("measured read bandwidth is {:.0}% of the clock-derived figure: the GPU may be throttled or in a low power state", f * 100.0));
        }
    }
    Ok(BenchResult {
        buffer_bytes: bytes as u64,
        iterations: ITERS,
        read_gbps_best: read_best,
        read_gbps_median: read_median,
        copy_gbps_best: copy_best,
        dma_dtod_gbps_best: dma_best,
        read_fraction_of_derived: fraction,
        warnings,
        from_cache: false,
    })
}

/// Cache key: results are only reused for the same GPU, driver and clock limits
/// (spec 01 §6). A changed clock limit (e.g. a power cap) invalidates the entry.
pub fn cache_key(uuid: &str, driver: &str, max_sm_mhz: Option<u32>, max_mem_mhz: Option<u32>) -> String {
    format!("v1|{uuid}|{driver}|sm{}|mem{}", max_sm_mhz.unwrap_or(0), max_mem_mhz.unwrap_or(0))
}

fn cache_path() -> Option<std::path::PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME").map(std::path::PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))?;
    Some(base.join("astral").join("hw-bench-v1.json"))
}

type Cache = std::collections::BTreeMap<String, BenchResult>;

fn load_cache() -> Cache {
    cache_path().and_then(|p| std::fs::read(p).ok()).and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

fn store_cache(cache: &Cache) {
    let Some(p) = cache_path() else { return };
    if let Some(dir) = p.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(json) = serde_json::to_vec_pretty(cache) {
        // Write-then-rename so a crash never leaves a truncated cache.
        let tmp = p.with_extension("json.tmp");
        if std::fs::write(&tmp, json).is_ok() {
            let _ = std::fs::rename(tmp, p);
        }
    }
}

/// Returns a cached result for `key`, or measures, stores and returns a fresh one.
pub fn run_cached(ctx: &Arc<Context>, caps: &Capabilities, key: &str, fresh: bool) -> Result<BenchResult> {
    let mut cache = load_cache();
    if !fresh {
        if let Some(hit) = cache.get(key) {
            let mut r = hit.clone();
            r.from_cache = true;
            return Ok(r);
        }
    }
    let r = run(ctx, caps)?;
    cache.insert(key.to_string(), r.clone());
    store_cache(&cache);
    Ok(r)
}
