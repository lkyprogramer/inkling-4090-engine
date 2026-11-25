// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! The `Backend` trait the runtime is written against (R-RT-3).
//!
//! [`CpuBackend`] runs the CPU reference kernels and is the mock used in CPU-only CI;
//! [`CudaBackend`] runs the real kernels. The same model code runs on both, and a
//! CPU/GPU comparison of that code is the model-level golden test.

#![forbid(unsafe_code)]

use crate::ops::RmsNormF32;
use crate::{reference, KernelRegistry};
use astral_core::{AstralError, Result};
use astral_gpu::{Context, DeviceBuffer, Stream};
use std::sync::Arc;

pub trait Backend {
    type Buf;

    fn name(&self) -> &'static str;
    fn upload(&self, data: &[f32]) -> Result<Self::Buf>;
    fn zeros(&self, n: usize) -> Result<Self::Buf>;
    fn download(&self, buf: &Self::Buf) -> Result<Vec<f32>>;
    fn synchronize(&self) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn rmsnorm(&self, x: &Self::Buf, weight: Option<&Self::Buf>, y: &mut Self::Buf, rows: usize, cols: usize, eps: f32) -> Result<()>;
}

/// Reference backend: plain Rust on the host. Slow, deterministic, always available.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    type Buf = Vec<f32>;

    fn name(&self) -> &'static str {
        "cpu-reference"
    }

    fn upload(&self, data: &[f32]) -> Result<Vec<f32>> {
        Ok(data.to_vec())
    }

    fn zeros(&self, n: usize) -> Result<Vec<f32>> {
        Ok(vec![0.0; n])
    }

    fn download(&self, buf: &Vec<f32>) -> Result<Vec<f32>> {
        Ok(buf.clone())
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn rmsnorm(&self, x: &Vec<f32>, weight: Option<&Vec<f32>>, y: &mut Vec<f32>, rows: usize, cols: usize, eps: f32) -> Result<()> {
        if y.len() < rows * cols {
            return Err(AstralError::Invalid("rmsnorm: output too small".into()));
        }
        let out = reference::rmsnorm_f32(x, weight.map(|w| w.as_slice()), rows, cols, eps);
        y[..rows * cols].copy_from_slice(&out);
        Ok(())
    }
}

/// CUDA backend: one context, one stream, kernels loaded once.
pub struct CudaBackend {
    ctx: Arc<Context>,
    stream: Stream,
    rmsnorm: RmsNormF32,
}

impl CudaBackend {
    pub fn new(ctx: &Arc<Context>) -> Result<Self> {
        let reg = KernelRegistry::load(ctx)?;
        Ok(Self { ctx: ctx.clone(), stream: Stream::new(ctx)?, rmsnorm: RmsNormF32::new(&reg)? })
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }
}

impl Backend for CudaBackend {
    type Buf = DeviceBuffer<f32>;

    fn name(&self) -> &'static str {
        "cuda"
    }

    fn upload(&self, data: &[f32]) -> Result<DeviceBuffer<f32>> {
        DeviceBuffer::from_slice(&self.ctx, data)
    }

    fn zeros(&self, n: usize) -> Result<DeviceBuffer<f32>> {
        DeviceBuffer::zeroed(&self.ctx, n)
    }

    fn download(&self, buf: &DeviceBuffer<f32>) -> Result<Vec<f32>> {
        self.stream.synchronize()?;
        buf.to_vec()
    }

    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }

    fn rmsnorm(&self, x: &DeviceBuffer<f32>, weight: Option<&DeviceBuffer<f32>>, y: &mut DeviceBuffer<f32>, rows: usize, cols: usize, eps: f32) -> Result<()> {
        let (r, c) = (u32::try_from(rows), u32::try_from(cols));
        let (Ok(r), Ok(c)) = (r, c) else { return Err(AstralError::Invalid("rmsnorm: dims exceed u32".into())) };
        self.rmsnorm.launch(&self.stream, x, weight.map(|w| w as &dyn astral_gpu::Dev<f32>), y, r, c, eps)
    }
}

/// A backend-generic computation used by tests: `steps` chained RMSNorms, alternating
/// buffers, mimicking a stack of layers. Identical code runs on CPU and GPU.
pub fn rmsnorm_chain<B: Backend>(b: &B, x: &[f32], w: &[f32], rows: usize, cols: usize, steps: usize) -> Result<Vec<f32>> {
    let mut a = b.upload(x)?;
    let mut t = b.zeros(rows * cols)?;
    let wb = b.upload(w)?;
    for i in 0..steps {
        let weight = (i % 2 == 0).then_some(&wb);
        b.rmsnorm(&a, weight, &mut t, rows, cols, 1e-6)?;
        std::mem::swap(&mut a, &mut t);
    }
    b.synchronize()?;
    b.download(&a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_runs_without_a_gpu() {
        let x: Vec<f32> = (0..2304).map(|i| ((i * 37 % 101) as f32 - 50.0) / 10.0).collect();
        let w: Vec<f32> = (0..2304).map(|i| 1.0 + (i % 7) as f32 * 0.1).collect();
        let y = rmsnorm_chain(&CpuBackend, &x, &w, 1, 2304, 48).unwrap();
        assert_eq!(y.len(), 2304);
        assert!(y.iter().all(|v| v.is_finite()));
        // After a weightless norm (odd step last) the row has unit RMS.
        let ms: f64 = y.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / 2304.0;
        assert!((ms - 1.0).abs() < 1e-4, "{ms}");
    }
}
