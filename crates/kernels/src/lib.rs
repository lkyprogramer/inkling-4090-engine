// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Synthetic Minds LTD
//! `astral-kernels`: loads the embedded fatbins and exposes typed, bounds-checked
//! launchers. Each launcher in [`ops`] is the single place that mirrors a kernel's
//! C signature; that is the only `unsafe` allowed outside `astral-gpu` (spec 08, R-RT-1).
//!
//! [`reference`] holds slow, obviously-correct CPU versions every GPU kernel is
//! tested against.

/// `DeviceBuffer<f32>`, for launchers that need the whole buffer (e.g. to zero it).
pub type DeviceBufferF32 = astral_gpu::DeviceBuffer<f32>;

pub mod backend;
pub mod dspark;
pub mod inkling;
pub mod ops;
pub mod qwen;
pub mod reference;

use astral_core::{AstralError, Result};
use astral_gpu::{Context, Function, Module};
use std::collections::HashMap;
use std::sync::Arc;

pub use astral_kernels_src::{BUILT_ARCHS, NVCC_VERSION};

/// All kernel modules loaded into one context.
pub struct KernelRegistry {
    modules: HashMap<&'static str, Arc<Module>>,
}

impl KernelRegistry {
    pub fn load(ctx: &Arc<Context>) -> Result<Self> {
        if astral_kernels_src::FATBINS.is_empty() {
            return Err(AstralError::Unsupported("this build contains no GPU kernels (nvcc was not found when astral-kernels-src was built)".into()));
        }
        let mut modules = HashMap::new();
        for (name, image) in astral_kernels_src::FATBINS {
            let m = Module::load(ctx, image).map_err(|e| match e {
                AstralError::Cuda { code: 209, .. } => {
                    AstralError::Unsupported(format!("kernel module `{name}` has no binary for this GPU (built for sm_{BUILT_ARCHS:?}); rebuild with ASTRAL_CUDA_ARCHS"))
                }
                other => other,
            })?;
            modules.insert(*name, m);
        }
        Ok(Self { modules })
    }

    pub fn function(&self, module: &str, symbol: &str) -> Result<Function> {
        let f = self.modules.get(module).ok_or_else(|| AstralError::Invalid(format!("no kernel module `{module}`")))?.function(symbol)?;
        // Experiment / tuning knob: a fixed carveout preference for every kernel.
        if let Some(p) = std::env::var("ASTRAL_CARVEOUT").ok().and_then(|v| v.parse::<i32>().ok()) {
            f.set_carveout(p)?;
        }
        Ok(f)
    }

    pub fn module_names(&self) -> Vec<&'static str> {
        let mut v: Vec<_> = self.modules.keys().copied().collect();
        v.sort();
        v
    }
}
