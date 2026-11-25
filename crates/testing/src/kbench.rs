// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! Single-kernel micro-benchmarks, small enough to run under Nsight Compute.

use crate::Rng;
use astral_core::Result;
use astral_gpu::{Context, DeviceBuffer, Event, Stream};
use astral_kernels::ops::{GemmMxfp4MmaF32, GemvMxfp4F32};
use astral_kernels::KernelRegistry;
use std::sync::Arc;
use std::time::Instant;

pub struct KbenchResult {
    pub us_per_call: f64,
    pub weight_gbps: f64,
}

/// Times the MXFP4 GEMV on random weights (300 ms warm-up, then `iters` timed calls).
pub fn gemv_mxfp4(ctx: &Arc<Context>, n: usize, k: usize, b: usize, iters: u32) -> Result<KbenchResult> {
    let reg = KernelRegistry::load(ctx)?;
    let gemv = GemvMxfp4F32::new(&reg)?;
    let stream = Stream::new(ctx)?;
    let mut rng = Rng::new(7);
    let blocks: Vec<u8> = (0..n * k / 32 * 17).map(|i| if i % 17 == 0 { 120 + (rng.next_u64() % 15) as u8 } else { rng.next_u64() as u8 }).collect();
    let (qs, es) = astral_quant::repack_mxfp4(&blocks)?;
    let bytes = (qs.len() + es.len()) as f64;
    let dqs = DeviceBuffer::from_slice(ctx, &qs)?;
    let des = DeviceBuffer::from_slice(ctx, &es)?;
    let dx = DeviceBuffer::from_slice(ctx, &rng.vec_f32(b * k, 1.0))?;
    let mut dy = DeviceBuffer::<f32>::zeroed(ctx, b * n)?;
    let run = |dy: &mut DeviceBuffer<f32>| gemv.launch(&stream, &dqs, &des, &dx, dy, n as u32, k as u32, b as u32);
    let t = Instant::now();
    while t.elapsed().as_millis() < 300 {
        run(&mut dy)?;
        stream.synchronize()?;
    }
    let (e0, e1) = (Event::new(ctx)?, Event::new(ctx)?);
    e0.record(&stream)?;
    for _ in 0..iters.max(1) {
        run(&mut dy)?;
    }
    e1.record(&stream)?;
    e1.synchronize()?;
    let us = e0.elapsed_ms(&e1)? as f64 * 1e3 / iters.max(1) as f64;
    Ok(KbenchResult { us_per_call: us, weight_gbps: bytes / (us * 1e-6) / 1e9 })
}

/// Times the tensor-core MXFP4 kernel (same protocol as [`gemv_mxfp4`]).
pub fn gemm_mxfp4_mma(ctx: &Arc<Context>, n: usize, k: usize, b: usize, iters: u32, f16_x: bool) -> Result<KbenchResult> {
    let reg = KernelRegistry::load(ctx)?;
    let mma = GemmMxfp4MmaF32::new(&reg)?;
    let stream = Stream::new(ctx)?;
    let mut rng = Rng::new(7);
    let blocks: Vec<u8> = (0..n * k / 32 * 17).map(|i| if i % 17 == 0 { 120 + (rng.next_u64() % 15) as u8 } else { rng.next_u64() as u8 }).collect();
    let (q, sc) = astral_quant::repack_mxfp4_mma(&blocks, n, k)?;
    let bytes = (q.len() + sc.len()) as f64;
    let (dq, ds) = (DeviceBuffer::from_slice(ctx, &q)?, DeviceBuffer::from_slice(ctx, &sc)?);
    let xf = rng.vec_f32(b * k, 1.0);
    let dx = DeviceBuffer::from_slice(ctx, &xf)?;
    let dxh = DeviceBuffer::from_slice(ctx, &xf.iter().map(|&v| astral_quant::f32_to_f16(v)).collect::<Vec<u16>>())?;
    let mut dy = DeviceBuffer::<f32>::zeroed(ctx, b * n)?;
    let run = |dy: &mut DeviceBuffer<f32>| {
        if f16_x {
            mma.launch_f16(&stream, &dq, &ds, &dxh, dy, n as u32, k as u32, b as u32)
        } else {
            mma.launch(&stream, &dq, &ds, &dx, dy, n as u32, k as u32, b as u32)
        }
    };
    let t = Instant::now();
    while t.elapsed().as_millis() < 300 {
        run(&mut dy)?;
        stream.synchronize()?;
    }
    let (e0, e1) = (Event::new(ctx)?, Event::new(ctx)?);
    e0.record(&stream)?;
    for _ in 0..iters.max(1) {
        run(&mut dy)?;
    }
    e1.record(&stream)?;
    e1.synchronize()?;
    let us = e0.elapsed_ms(&e1)? as f64 * 1e3 / iters.max(1) as f64;
    Ok(KbenchResult { us_per_call: us, weight_gbps: bytes / (us * 1e-6) / 1e9 })
}
