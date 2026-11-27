// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! GPU self-test (spec 08 L2/L3): each kernel against its CPU reference, the CPU and
//! CUDA backends on identical code, the VMM arena, and CUDA-graph replay vs eager.

use crate::{compare, ErrorStats, Rng};
use astral_core::Result;
use astral_gpu::{Context, DeviceBuffer, Event, Stream};
use astral_kernels::backend::{rmsnorm_chain, CpuBackend, CudaBackend};
use astral_kernels::ops::{GemmMxfp4MmaF32, Gemv, GemvMatrix, GemvMxfp4F32, GemvType, RmsNormF32};
use astral_kernels::{reference, KernelRegistry};
use astral_mem::{GpuArena, SlabPool};
use serde::Serialize;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Serialize)]
pub struct CaseResult {
    pub kernel: &'static str,
    pub case: String,
    pub error: ErrorStats,
    pub tolerance: f64,
    pub pass: bool,
    /// Free-form measurement attached to the case (e.g. timings).
    pub info: Option<String>,
}

/// RMSNorm f32 tolerance: f32 accumulation + rsqrtf vs f64 reference.
const RMSNORM_TOL: f64 = 2e-5;
/// 48 chained norms compound rounding; still far below any model-level effect.
const CHAIN_TOL: f64 = 2e-4;

fn exact(kernel: &'static str, case: String, pass: bool, info: Option<String>) -> CaseResult {
    let error = ErrorStats { max_abs: if pass { 0.0 } else { f64::INFINITY }, max_scaled: if pass { 0.0 } else { f64::INFINITY }, worst_index: 0 };
    CaseResult { kernel, case, error, tolerance: 0.0, pass, info }
}

pub fn run(ctx: &Arc<Context>) -> Result<Vec<CaseResult>> {
    let mut out = Vec::new();
    rmsnorm_cases(ctx, &mut out)?;
    backend_cases(ctx, &mut out)?;
    arena_cases(ctx, &mut out)?;
    graph_cases(ctx, &mut out)?;
    mxfp4_cases(ctx, &mut out)?;
    mxfp4_mma_cases(ctx, &mut out)?;
    gemv_cases(ctx, &mut out)?;
    split_k_cases(ctx, &mut out)?;
    Ok(out)
}

fn rmsnorm_cases(ctx: &Arc<Context>, out: &mut Vec<CaseResult>) -> Result<()> {
    let reg = KernelRegistry::load(ctx)?;
    let stream = Stream::new(ctx)?;
    let rms = RmsNormF32::new(&reg)?;
    let mut rng = Rng::new(0xA57A1);
    // Shapes include Inkling's hidden size (2304), non-multiples of the block, and 1x1.
    let shapes: &[(usize, usize)] = &[(1, 2304), (7, 2304), (64, 2304), (3, 1000), (1, 1), (5, 4097), (2, 128)];
    for &(rows, cols) in shapes {
        for weighted in [false, true] {
            let x = rng.vec_f32(rows * cols, 4.0);
            let w = rng.vec_f32(cols, 1.5);
            let want = reference::rmsnorm_f32(&x, weighted.then_some(w.as_slice()), rows, cols, 1e-6);
            let dx = DeviceBuffer::from_slice(ctx, &x)?;
            let dw = DeviceBuffer::from_slice(ctx, &w)?;
            let mut dy = DeviceBuffer::<f32>::zeroed(ctx, rows * cols)?;
            rms.launch(&stream, &dx, weighted.then_some(&dw as &dyn astral_gpu::Dev<f32>), &mut dy, rows as u32, cols as u32, 1e-6)?;
            stream.synchronize()?;
            let error = compare(&dy.to_vec()?, &want);
            out.push(CaseResult {
                kernel: "astral_rmsnorm_f32",
                case: format!("{rows}x{cols}{}", if weighted { " weighted" } else { " weightless" }),
                error,
                tolerance: RMSNORM_TOL,
                pass: error.max_scaled <= RMSNORM_TOL,
                info: None,
            });
        }
    }
    Ok(())
}

/// Same backend-generic code on CPU (mock) and CUDA: 48 chained norms, like 48 slots.
fn backend_cases(ctx: &Arc<Context>, out: &mut Vec<CaseResult>) -> Result<()> {
    let cuda = CudaBackend::new(ctx)?;
    let mut rng = Rng::new(0xBACC);
    let (rows, cols) = (16, 2304);
    let x = rng.vec_f32(rows * cols, 3.0);
    let w = rng.vec_f32(cols, 1.2);
    let cpu_y = rmsnorm_chain(&CpuBackend, &x, &w, rows, cols, 48)?;
    let gpu_y = rmsnorm_chain(&cuda, &x, &w, rows, cols, 48)?;
    let error = compare(&gpu_y, &cpu_y);
    out.push(CaseResult { kernel: "backend: cpu vs cuda", case: format!("{rows}x{cols} x48 chain"), error, tolerance: CHAIN_TOL, pass: error.max_scaled <= CHAIN_TOL, info: None });
    Ok(())
}

/// VMM arena: two pools with different block sizes, unique byte patterns, reuse after free.
fn arena_cases(ctx: &Arc<Context>, out: &mut Vec<CaseResult>) -> Result<()> {
    let mut arena = GpuArena::new(ctx, 256 << 20)?;
    let g = arena.granule();
    let pa = arena.arbiter.register("global_kv", 0)?;
    let pb = arena.arbiter.register("conv_state", 8 << 20)?;
    let mut kv = SlabPool::new(pa, 64 << 10, g);
    let mut st = SlabPool::new(pb, 3 << 20, g); // 3 MiB: not a granule multiple
    let mut blocks = Vec::new();
    for _ in 0..40 {
        blocks.push((0usize, kv.alloc(&mut arena.arbiter).expect("kv block")));
    }
    for _ in 0..4 {
        blocks.push((1usize, st.alloc(&mut arena.arbiter).expect("state block")));
    }
    arena.sync_mapping()?;
    let pattern = |i: usize, n: usize| -> Vec<u8> { (0..n).map(|j| ((i * 131 + j * 7) % 251) as u8).collect() };
    let pools = [&kv, &st];
    for (i, &(p, id)) in blocks.iter().enumerate() {
        let off = pools[p].offset(id).unwrap();
        arena.range().write(off, &pattern(i, pools[p].block_bytes()))?;
    }
    let mut ok = true;
    for (i, &(p, id)) in blocks.iter().enumerate() {
        let off = pools[p].offset(id).unwrap();
        let mut back = vec![0u8; pools[p].block_bytes()];
        arena.range().read(off, &mut back)?;
        ok &= back == pattern(i, back.len());
    }
    // Free every other kv block and reallocate: must reuse addresses, not grow.
    let hw = arena.arbiter.high_water();
    let freed: Vec<_> = blocks.iter().filter(|(p, _)| *p == 0).step_by(2).map(|&(_, id)| id).collect();
    let mut freed_offsets: Vec<usize> = freed.iter().map(|&id| kv.offset(id).unwrap()).collect();
    for id in &freed {
        kv.free(*id)?;
    }
    let mut again: Vec<usize> = (0..freed.len()).map(|_| kv.alloc(&mut arena.arbiter).and_then(|id| kv.offset(id)).unwrap()).collect();
    freed_offsets.sort();
    again.sort();
    ok &= freed_offsets == again && arena.arbiter.high_water() == hw;
    let info = format!("granule {} KiB, mapped {} MiB, kv {} blocks/chunk, state {} B chunk", g >> 10, arena.mapped_bytes() >> 20, kv.blocks_per_chunk(), st.chunk_bytes());
    out.push(exact("mem: VMM arena + slab pools", "write/read/reuse".into(), ok, Some(info)));
    Ok(())
}

/// CUDA graph: capture 48 RMSNorm launches, replay, compare bit-exact with eager, and time both.
fn graph_cases(ctx: &Arc<Context>, out: &mut Vec<CaseResult>) -> Result<()> {
    let reg = KernelRegistry::load(ctx)?;
    let rms = RmsNormF32::new(&reg)?;
    let stream = Stream::new(ctx)?;
    let (rows, cols, steps) = (1u32, 2304u32, 48usize);
    let mut rng = Rng::new(0x6EA9);
    let x = rng.vec_f32((rows * cols) as usize, 2.0);
    let w = rng.vec_f32(cols as usize, 1.0);
    let dx = DeviceBuffer::from_slice(ctx, &x)?;
    let dw = DeviceBuffer::from_slice(ctx, &w)?;
    let mut a = DeviceBuffer::<f32>::zeroed(ctx, x.len())?;
    let mut b = DeviceBuffer::<f32>::zeroed(ctx, x.len())?;

    // One "forward": x -> a, then 47 more alternating a <-> b.
    let enqueue = |s: &Stream, a: &mut DeviceBuffer<f32>, b: &mut DeviceBuffer<f32>| -> Result<()> {
        rms.launch(s, &dx, Some(&dw as &dyn astral_gpu::Dev<f32>), a, rows, cols, 1e-6)?;
        for i in 1..steps {
            if i % 2 == 1 {
                rms.launch(s, a, None, b, rows, cols, 1e-6)?;
            } else {
                rms.launch(s, b, Some(&dw as &dyn astral_gpu::Dev<f32>), a, rows, cols, 1e-6)?;
            }
        }
        Ok(())
    };

    enqueue(&stream, &mut a, &mut b)?;
    stream.synchronize()?;
    let eager = b.to_vec()?;

    let graph = stream.capture(|s| enqueue(s, &mut a, &mut b))?;
    let zeros = vec![0f32; x.len()];
    a.copy_from_slice(&zeros)?;
    b.copy_from_slice(&zeros)?;
    graph.launch(&stream)?;
    stream.synchronize()?;
    let replay = b.to_vec()?;
    let same = eager.iter().zip(&replay).all(|(p, q)| p.to_bits() == q.to_bits());

    // Timing: host time to enqueue + GPU time per forward, eager vs graph.
    let iters = 200;
    let (e0, e1) = (Event::new(ctx)?, Event::new(ctx)?);
    let t = Instant::now();
    e0.record(&stream)?;
    for _ in 0..iters {
        enqueue(&stream, &mut a, &mut b)?;
    }
    e1.record(&stream)?;
    let eager_host_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    e1.synchronize()?;
    let eager_gpu_us = e0.elapsed_ms(&e1)? as f64 * 1e3 / iters as f64;
    let t = Instant::now();
    e0.record(&stream)?;
    for _ in 0..iters {
        graph.launch(&stream)?;
    }
    e1.record(&stream)?;
    let graph_host_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    e1.synchronize()?;
    let graph_gpu_us = e0.elapsed_ms(&e1)? as f64 * 1e3 / iters as f64;
    let info = format!("{steps} launches/forward: eager {eager_gpu_us:.1} us GPU ({eager_host_us:.1} us host enqueue), graph {graph_gpu_us:.1} us GPU ({graph_host_us:.1} us host)");
    out.push(exact("gpu: CUDA graph replay", format!("{steps}-launch forward == eager"), same, Some(info)));
    Ok(())
}

/// Random MXFP4 blocks with scales near 1 (E8M0 120..=134) so outputs stay well scaled.
fn random_mxfp4(rng: &mut Rng, n_blocks: usize) -> Vec<u8> {
    let mut v = vec![0u8; n_blocks * 17];
    for b in v.chunks_mut(17) {
        b[0] = 120 + (rng.next_u64() % 15) as u8;
        for q in &mut b[1..] {
            *q = rng.next_u64() as u8;
        }
    }
    v
}

/// MXFP4 GEMV vs the f64 reference, then effective weight bandwidth on a matrix larger than L2.
fn mxfp4_cases(ctx: &Arc<Context>, out: &mut Vec<CaseResult>) -> Result<()> {
    let reg = KernelRegistry::load(ctx)?;
    let gemv = GemvMxfp4F32::new(&reg)?;
    let stream = Stream::new(ctx)?;
    let mut rng = Rng::new(0x3F4);
    // Inkling: hidden 2304, expert 1152; Qwen3.8-27B: hidden 5120, FFN 17408.
    let shapes: &[(usize, usize)] = &[(2304, 2304), (2304, 1152), (1152, 2304), (5120, 17408), (8, 64)];
    for &(n, k) in shapes {
        let blocks = random_mxfp4(&mut rng, n * k / 32);
        let (qs, es) = astral_quant::repack_mxfp4(&blocks)?;
        let dqs = DeviceBuffer::from_slice(ctx, &qs)?;
        let des = DeviceBuffer::from_slice(ctx, &es)?;
        for b in [1usize, 2, 4] {
            let x = rng.vec_f32(b * k, 1.0);
            let (want, l1) = reference::gemv_mxfp4(&blocks, &x, n, k, b);
            let dx = DeviceBuffer::from_slice(ctx, &x)?;
            let mut dy = DeviceBuffer::<f32>::zeroed(ctx, b * n)?;
            gemv.launch(&stream, &dqs, &des, &dx, &mut dy, n as u32, k as u32, b as u32)?;
            stream.synchronize()?;
            // Error relative to sum|w*x|: any f32 summation order stays within ~(depth * eps)
            // of the exact result in this metric. 4e-6 is ~32 ulp, far below model-level effects.
            let tol = 4e-6;
            let got = dy.to_vec()?;
            let mut error = ErrorStats { max_abs: 0.0, max_scaled: 0.0, worst_index: 0 };
            for (i, ((&g, &w), &a)) in got.iter().zip(&want).zip(&l1).enumerate() {
                let d = (g as f64 - w as f64).abs();
                let sc = if g.is_finite() { d / a.max(1e-30) } else { f64::INFINITY };
                if sc > error.max_scaled {
                    error = ErrorStats { max_abs: d, max_scaled: sc, worst_index: i };
                }
            }
            out.push(CaseResult { kernel: "astral_gemv_mxfp4_f32", case: format!("n{n} k{k} b{b}"), error, tolerance: tol, pass: error.max_scaled <= tol, info: None });
        }
    }

    // Bandwidth: 16384 x 8192 MXFP4 = 71.3 MB of weights + scales (> 50 MB L2).
    let (n, k) = (16384usize, 8192usize);
    let blocks = random_mxfp4(&mut rng, n * k / 32);
    let (qs, es) = astral_quant::repack_mxfp4(&blocks)?;
    let dqs = DeviceBuffer::from_slice(ctx, &qs)?;
    let des = DeviceBuffer::from_slice(ctx, &es)?;
    let weight_bytes = (qs.len() + es.len()) as f64;
    let mut info = Vec::new();
    for b in [1usize, 4] {
        let dx = DeviceBuffer::from_slice(ctx, &rng.vec_f32(b * k, 1.0))?;
        let mut dy = DeviceBuffer::<f32>::zeroed(ctx, b * n)?;
        let (e0, e1) = (Event::new(ctx)?, Event::new(ctx)?);
        let warm = Instant::now();
        while warm.elapsed().as_millis() < 300 {
            gemv.launch(&stream, &dqs, &des, &dx, &mut dy, n as u32, k as u32, b as u32)?;
            stream.synchronize()?;
        }
        let iters = 50;
        e0.record(&stream)?;
        for _ in 0..iters {
            gemv.launch(&stream, &dqs, &des, &dx, &mut dy, n as u32, k as u32, b as u32)?;
        }
        e1.record(&stream)?;
        e1.synchronize()?;
        let us = e0.elapsed_ms(&e1)? as f64 * 1e3 / iters as f64;
        info.push(format!("b{b}: {us:.1} us, {:.0} GB/s of weights", weight_bytes / (us * 1e-6) / 1e9));
    }
    out.push(exact("perf: mxfp4 gemv 16384x8192", "timing".into(), true, Some(info.join("; "))));
    Ok(())
}

fn err_vs_l1(got: &[f32], want: &[f32], l1: &[f64]) -> ErrorStats {
    let mut e = ErrorStats { max_abs: 0.0, max_scaled: 0.0, worst_index: 0 };
    for (i, ((&g, &w), &a)) in got.iter().zip(want).zip(l1).enumerate() {
        let d = (g as f64 - w as f64).abs();
        let sc = if g.is_finite() { d / a.max(1e-30) } else { f64::INFINITY };
        if sc > e.max_scaled {
            e = ErrorStats { max_abs: d, max_scaled: sc, worst_index: i };
        }
    }
    e
}

/// Tensor-core MXFP4 kernel vs the f64 reference on f16-rounded activations (its contract),
/// then timing on the same 16384 x 8192 matrix as the CUDA-core kernel, batches 1..8.
fn mxfp4_mma_cases(ctx: &Arc<Context>, out: &mut Vec<CaseResult>) -> Result<()> {
    let reg = KernelRegistry::load(ctx)?;
    let mma = GemmMxfp4MmaF32::new(&reg)?;
    let stream = Stream::new(ctx)?;
    let mut rng = Rng::new(0x3F5);
    let shapes: &[(usize, usize)] = &[(2304, 2304), (2304, 1152), (1152, 2304), (5120, 17408), (16, 64), (48, 4160)];
    for &(n, k) in shapes {
        let blocks = random_mxfp4(&mut rng, n * k / 32);
        let (q, sc) = astral_quant::repack_mxfp4_mma(&blocks, n, k)?;
        let (dq, ds) = (DeviceBuffer::from_slice(ctx, &q)?, DeviceBuffer::from_slice(ctx, &sc)?);
        for b in [1usize, 3, 8] {
            let x = rng.vec_f32(b * k, 1.0);
            let xh: Vec<f32> = x.iter().map(|&v| astral_quant::round_through_f16(v)).collect();
            let (want, l1) = reference::gemv_mxfp4(&blocks, &xh, n, k, b);
            let dx = DeviceBuffer::from_slice(ctx, &x)?;
            let mut dy = DeviceBuffer::<f32>::zeroed(ctx, b * n)?;
            mma.launch(&stream, &dq, &ds, &dx, &mut dy, n as u32, k as u32, b as u32)?;
            stream.synchronize()?;
            let error = err_vs_l1(&dy.to_vec()?, &want, &l1);
            out.push(CaseResult { kernel: "astral_gemm_mxfp4_mma_f32", case: format!("n{n} k{k} b{b}"), error, tolerance: 4e-6, pass: error.max_scaled <= 4e-6, info: None });
            // Same inputs given as f16 bit patterns must give the same result.
            let xbits: Vec<u16> = x.iter().map(|&v| astral_quant::f32_to_f16(v)).collect();
            let dxh = DeviceBuffer::from_slice(ctx, &xbits)?;
            let mut dyh = DeviceBuffer::<f32>::zeroed(ctx, b * n)?;
            mma.launch_f16(&stream, &dq, &ds, &dxh, &mut dyh, n as u32, k as u32, b as u32)?;
            stream.synchronize()?;
            let error = err_vs_l1(&dyh.to_vec()?, &want, &l1);
            out.push(CaseResult { kernel: "astral_gemm_mxfp4_mma_f16x", case: format!("n{n} k{k} b{b}"), error, tolerance: 4e-6, pass: error.max_scaled <= 4e-6, info: None });
        }
    }
    for &(n, k) in &[(16384usize, 8192usize), (5120, 17408), (2304, 2304)] {
        let blocks = random_mxfp4(&mut rng, n * k / 32);
        let (q, sc) = astral_quant::repack_mxfp4_mma(&blocks, n, k)?;
        let bytes = (q.len() + sc.len()) as f64;
        let (dq, ds) = (DeviceBuffer::from_slice(ctx, &q)?, DeviceBuffer::from_slice(ctx, &sc)?);
        let mut info = Vec::new();
        for b in [1usize, 2, 4, 8] {
            let dx = DeviceBuffer::from_slice(ctx, &rng.vec_f32(b * k, 1.0))?;
            let mut dy = DeviceBuffer::<f32>::zeroed(ctx, b * n)?;
            let warm = Instant::now();
            while warm.elapsed().as_millis() < 300 {
                mma.launch(&stream, &dq, &ds, &dx, &mut dy, n as u32, k as u32, b as u32)?;
                stream.synchronize()?;
            }
            let (e0, e1) = (Event::new(ctx)?, Event::new(ctx)?);
            e0.record(&stream)?;
            for _ in 0..50 {
                mma.launch(&stream, &dq, &ds, &dx, &mut dy, n as u32, k as u32, b as u32)?;
            }
            e1.record(&stream)?;
            e1.synchronize()?;
            let us = e0.elapsed_ms(&e1)? as f64 * 1e3 / 50.0;
            info.push(format!("b{b}: {us:.1} us {:.0} GB/s", bytes / (us * 1e-6) / 1e9));
        }
        out.push(exact("perf: mxfp4 mma", format!("{n}x{k}"), true, Some(info.join("; "))));
    }
    Ok(())
}

/// Random GGUF-format bytes for `n_mats` matrices of `n x k` elements of type `ty`, with scales
/// chosen so values stay well scaled.
fn random_ggml(rng: &mut Rng, ty: GemvType, n_mats: usize, n: usize, k: usize) -> Vec<u8> {
    let elems = n_mats * n * k;
    match ty {
        GemvType::F32 => rng.vec_f32(elems, 1.0).iter().flat_map(|v| v.to_le_bytes()).collect(),
        GemvType::Bf16 => rng.vec_f32(elems, 1.0).iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect(),
        GemvType::Q8_0 => (0..elems / 32)
            .flat_map(|_| {
                let d = astral_quant::f32_to_f16(0.01 + (rng.next_u64() % 100) as f32 * 1e-4);
                let mut b = d.to_le_bytes().to_vec();
                b.extend((0..32).map(|_| rng.next_u64() as u8));
                b
            })
            .collect(),
        GemvType::Nvfp4 => (0..elems / 64)
            .flat_map(|_| {
                let mut b: Vec<u8> = (0..4).map(|_| 0x30 + (rng.next_u64() % 24) as u8).collect(); // scales ~0.25..4
                b.extend((0..32).map(|_| rng.next_u64() as u8));
                b
            })
            .collect(),
    }
}

/// Deterministic split-K (prefill GEMMs). The grouped tensor-core GEMM reads the f16 activation
/// plane, so each shape first checks that the inputs reach it: the output is nonzero, matches a
/// CPU reference on the operands as the kernel rounds them, and a change to one activation moves
/// only its token's outputs; an unwritten (all-zero) f16 plane must fail those checks. Then the
/// split workspace runs twice (bit-identical) and against the atomic-add path. Last, the prefill
/// loop-controller projection (`ctrl_t`) runs twice (bit-identical).
fn split_k_cases(ctx: &Arc<Context>, out: &mut Vec<CaseResult>) -> Result<()> {
    use astral_kernels::inkling::{InklingOps, CTRL_T_SPLITS};
    use astral_kernels::ops::{MmaWork, QAct};
    let reg = KernelRegistry::load(ctx)?;
    let atomic = Gemv::new(&reg)?;
    let mut det = Gemv::new(&reg)?;
    det.reserve_split_workspace(ctx, 1 << 20)?;
    let stream = Stream::new(ctx)?;
    let mut rng = Rng::new(0x5B1);
    // (type, n, k, matrices, tokens): attention output and expert down at a short prompt.
    let cases: &[(GemvType, usize, usize, usize, usize)] = &[(GemvType::Q8_0, 2304, 3072, 1, 5), (GemvType::Nvfp4, 2304, 1152, 3, 6)];
    for &(ty, n, k, mats, nt) in cases {
        let raw = random_ggml(&mut rng, ty, mats, n, k);
        let (qs, aux, gt) = match ty {
            GemvType::Q8_0 => {
                let (q, a) = astral_quant::repack_q8_0(&raw)?;
                (q, a, astral_quant::GgmlType::Q8_0)
            }
            _ => {
                let (q, a) = astral_quant::repack_nvfp4(&raw)?;
                (q, a, astral_quant::GgmlType::Nvfp4)
            }
        };
        let m = GemvMatrix::upload(ctx, ty, n as u32, k as u32, mats as u32, &qs, &aux, None)?;
        // Signed activations whose magnitude changes by 100x across 32-value blocks.
        let mut x = rng.vec_f32(nt * k, 1.0);
        for (i, v) in x.iter_mut().enumerate() {
            *v *= [0.03f32, 0.3, 1.0, 3.0][(i / 32) % 4];
        }
        let mut work = MmaWork::default();
        for e in 0..mats as u32 {
            let rows: Vec<(u32, u32)> = (0..nt as u32).filter(|t| t % mats as u32 == e).map(|t| (t, t)).collect();
            work.push(e, &rows);
        }
        let (items, xidx, yidx) = (DeviceBuffer::from_slice(ctx, &work.items)?, DeviceBuffer::from_slice(ctx, &work.xidx)?, DeviceBuffer::from_slice(ctx, &work.yidx)?);
        let (n_items, idx_len) = ((work.items.len() / 3) as u32, work.xidx.len() as u32);
        let quantized = |x: &[f32]| -> Result<QAct> {
            let dx = DeviceBuffer::from_slice(ctx, x)?;
            let mut qa = QAct::with_f16(ctx, k as u32, nt as u32)?;
            atomic.quantize(&stream, &dx, k as u32, nt as u32, &mut qa)?;
            stream.synchronize()?;
            Ok(qa)
        };
        let run = |g: &Gemv, qa: &QAct| -> Result<Vec<f32>> {
            let mut y = DeviceBuffer::<f32>::zeroed(ctx, nt * n)?;
            g.launch_mma(&stream, &m, qa, nt as u32, &mut y, nt as u32, &items, n_items, &xidx, &yidx, idx_len)?;
            stream.synchronize()?;
            y.to_vec()
        };
        // CPU reference on the operands the kernel multiplies: f16 activations and weights
        // dequantised to f16 (Q8_0: int8 x f16 scale rounded once; NVFP4: E2M1 x E4M3 is exact
        // in f16), products summed in f64. Token t uses matrix t % mats.
        let w = astral_quant::dequantize(gt, &raw, mats * n * k)?;
        let (mut want, mut l1) = (vec![0f32; nt * n], vec![0f64; nt * n]);
        for t in 0..nt {
            let e = t % mats;
            let xr: Vec<f64> = x[t * k..(t + 1) * k].iter().map(|&v| astral_quant::round_through_f16(v) as f64).collect();
            for r in 0..n {
                let (mut acc, mut a) = (0f64, 0f64);
                for (wv, xv) in w[(e * n + r) * k..(e * n + r + 1) * k].iter().zip(&xr) {
                    let p = astral_quant::round_through_f16(*wv) as f64 * xv;
                    acc += p;
                    a += p.abs();
                }
                want[t * n + r] = acc as f32;
                l1[t * n + r] = a;
            }
        }
        let tol = 1e-5;
        let check = |y: &[f32]| (y.iter().filter(|v| **v != 0.0).count(), err_vs_l1(y, &want, &l1));
        let case = |what: &str| format!("{ty:?} n{n} k{k} t{nt}: {what}");
        let qa = quantized(&x)?;
        let (d1, d2, a1) = (run(&det, &qa)?, run(&det, &qa)?, run(&atomic, &qa)?);
        let (nz, error) = check(&d1);
        out.push(exact("gemm_mma3 f16 activations", case("output nonzero"), nz == d1.len(), Some(format!("{nz} of {} nonzero", d1.len()))));
        out.push(CaseResult { kernel: "gemm_mma3 f16 activations", case: case("vs f16-operand reference"), error, tolerance: tol, pass: error.max_scaled <= tol, info: None });
        // One activation of the last token changed: that token's outputs move, no other's.
        let mut x2 = x.clone();
        x2[(nt - 1) * k + 7] += 0.5;
        let y2 = run(&det, &quantized(&x2)?)?;
        let moved = |t: usize| (0..n).filter(|&r| d1[t * n + r].to_bits() != y2[t * n + r].to_bits()).count();
        let (last, others) = (moved(nt - 1), (0..nt - 1).map(moved).sum::<usize>());
        out.push(exact(
            "gemm_mma3 f16 activations",
            case("one input moves only its token"),
            last >= n / 2 && others == 0,
            Some(format!("{last} of {n} outputs of that token, {others} of the others")),
        ));
        // Negative control: the f16 plane as allocated (zero), which is what an earlier version
        // of this test multiplied. Both checks above must reject it.
        let (bnz, berr) = check(&run(&det, &QAct::with_f16(ctx, k as u32, nt as u32)?)?);
        out.push(exact(
            "gemm_mma3 f16 activations",
            case("negative control: unwritten f16 plane rejected"),
            bnz != d1.len() && berr.max_scaled > tol,
            Some(format!("{bnz} nonzero, scaled error {:.2e}", berr.max_scaled)),
        ));
        let same = d1.iter().zip(&d2).all(|(a, b)| a.to_bits() == b.to_bits());
        out.push(CaseResult { kernel: "gemm_mma3 split-K workspace", case: case("repeat bit-identical"), error: err_vs_l1(&d2, &d1, &l1), tolerance: 0.0, pass: same, info: None });
        let error = err_vs_l1(&d1, &a1, &l1);
        out.push(CaseResult { kernel: "gemm_mma3 split-K workspace", case: case("vs atomic adds"), error, tolerance: tol, pass: error.max_scaled <= tol, info: None });
    }
    // Loop-controller projection for a short prefill (dim 2304, rank 64).
    let ops = InklingOps::new(&reg)?;
    let (dim, rank, nt) = (2304usize, 64usize, 7usize);
    let up = |r: &mut Rng, n: usize, k: usize| -> Result<GemvMatrix> {
        let w = r.vec_f32(n * k, 0.05);
        let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        GemvMatrix::upload(ctx, GemvType::F32, n as u32, k as u32, 1, &bytes, &[], None)
    };
    let (wd, wgf) = (up(&mut rng, rank, 2 * dim)?, up(&mut rng, 2 * dim, rank)?);
    let x = DeviceBuffer::from_slice(ctx, &rng.vec_f32(nt * dim, 1.0))?;
    let msg = DeviceBuffer::from_slice(ctx, &rng.vec_f32(nt * dim, 1.0))?;
    let ctrl = || -> Result<Vec<f32>> {
        let mut rms = DeviceBuffer::<f32>::zeroed(ctx, 2 * nt)?;
        let mut cd = DeviceBuffer::<f32>::zeroed(ctx, (1 + CTRL_T_SPLITS as usize) * nt * rank)?;
        let mut gf = DeviceBuffer::<f32>::zeroed(ctx, nt * 2 * dim)?;
        let mut o = DeviceBuffer::<f32>::zeroed(ctx, nt * dim)?;
        ops.ctrl_t(&stream, &wd, &wgf, &x, dim, &msg, dim, &mut rms, &mut cd, &mut gf, &mut o, dim as u32, nt as u32)?;
        stream.synchronize()?;
        o.to_vec()
    };
    let (c1, c2) = (ctrl()?, ctrl()?);
    let same = c1.iter().zip(&c2).all(|(a, b)| a.to_bits() == b.to_bits());
    let l1 = vec![1.0f64; c1.len()];
    out.push(CaseResult { kernel: "ctrl_t split-K", case: format!("dim{dim} rank{rank} t{nt}: repeat bit-identical"), error: err_vs_l1(&c2, &c1, &l1), tolerance: 0.0, pass: same, info: None });
    Ok(())
}

/// General GEMV (all four weight types) vs the CPU decoders on the original GGUF bytes,
/// including the expert-id path with per-expert scales.
fn gemv_cases(ctx: &Arc<Context>, out: &mut Vec<CaseResult>) -> Result<()> {
    let reg = KernelRegistry::load(ctx)?;
    let gemv = Gemv::new(&reg)?;
    let stream = Stream::new(ctx)?;
    let mut rng = Rng::new(0x6E3);
    let cases: &[(GemvType, usize, usize)] = &[
        (GemvType::F32, 128, 2304),    // router
        (GemvType::F32, 64, 4608),     // loop controller down
        (GemvType::Bf16, 24, 2304),    // sink projection
        (GemvType::Bf16, 32, 2304),    // conv tap projection
        (GemvType::Q8_0, 3072, 2304),  // attention q / gate
        (GemvType::Q8_0, 2304, 3072),  // attention output
        (GemvType::Nvfp4, 2304, 2304), // expert gate_up
        (GemvType::Nvfp4, 2304, 1152), // expert down
    ];
    let ggml = |t: GemvType| match t {
        GemvType::F32 => astral_quant::GgmlType::F32,
        GemvType::Bf16 => astral_quant::GgmlType::Bf16,
        GemvType::Q8_0 => astral_quant::GgmlType::Q8_0,
        GemvType::Nvfp4 => astral_quant::GgmlType::Nvfp4,
    };
    for &(ty, n, k) in cases {
        let mats = 3usize;
        let raw = random_ggml(&mut rng, ty, mats, n, k);
        let (qs, aux) = match ty {
            GemvType::Q8_0 => astral_quant::repack_q8_0(&raw)?,
            GemvType::Nvfp4 => astral_quant::repack_nvfp4(&raw)?,
            _ => (raw.clone(), Vec::new()),
        };
        let scales = [0.5f32, 2.0, 1.25];
        let m = GemvMatrix::upload(ctx, ty, n as u32, k as u32, mats as u32, &qs, &aux, Some(&scales))?;
        let w = astral_quant::dequantize(ggml(ty), &raw, mats * n * k)?;
        // Expert path: slots use matrices [2, 0, 1], each with its own x.
        let ids = [2i32, 0, 1];
        let x = rng.vec_f32(mats * k, 1.0);
        let dx = DeviceBuffer::from_slice(ctx, &x)?;
        let mut dy = DeviceBuffer::<f32>::zeroed(ctx, mats * n)?;
        gemv.launch_ids_checked(&stream, &m, &dx, k as u32, &mut dy, &ids)?;
        let got = dy.to_vec()?;
        let (mut want, mut l1) = (vec![0f32; mats * n], vec![0f64; mats * n]);
        for (sl, &e) in ids.iter().enumerate() {
            for r in 0..n {
                let row = &w[(e as usize * n + r) * k..(e as usize * n + r + 1) * k];
                let (mut acc, mut a) = (0f64, 0f64);
                for (wv, xv) in row.iter().zip(&x[sl * k..(sl + 1) * k]) {
                    acc += *wv as f64 * *xv as f64;
                    a += (*wv as f64 * *xv as f64).abs();
                }
                want[sl * n + r] = (acc * scales[e as usize] as f64) as f32;
                l1[sl * n + r] = a * scales[e as usize] as f64;
            }
        }
        let error = err_vs_l1(&got, &want, &l1);
        out.push(CaseResult { kernel: "astral_gemv (ids)", case: format!("{ty:?} n{n} k{k}"), error, tolerance: 4e-6, pass: error.max_scaled <= 4e-6, info: None });
    }
    Ok(())
}
