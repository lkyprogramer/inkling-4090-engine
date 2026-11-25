// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Synthetic Minds LTD
use crate::BudgetArbiter;
use astral_core::Result;
use astral_gpu::{Context, DevicePtr, VirtualRange};
use std::sync::Arc;

/// A fixed-address GPU arena: an arbiter whose offsets map into one reserved VMM range.
///
/// Usage: allocate through pools with `&mut arena.arbiter`, then call
/// [`GpuArena::sync_mapping`] before touching new blocks on the GPU.
pub struct GpuArena {
    range: VirtualRange,
    pub arbiter: BudgetArbiter,
}

impl GpuArena {
    /// Reserves `budget_bytes` of address space; physical memory is mapped lazily.
    pub fn new(ctx: &Arc<Context>, budget_bytes: usize) -> Result<Self> {
        let range = VirtualRange::reserve(ctx, budget_bytes)?;
        let arbiter = BudgetArbiter::new(range.reserved(), range.granule());
        Ok(Self { range, arbiter })
    }

    /// Maps physical memory up to the arbiter's high-water mark.
    pub fn sync_mapping(&mut self) -> Result<()> {
        self.range.map_up_to(self.arbiter.high_water())
    }

    pub fn granule(&self) -> usize {
        self.range.granule()
    }

    pub fn mapped_bytes(&self) -> usize {
        self.range.mapped()
    }

    /// Device address of an arena offset. Stable for the life of the arena.
    pub fn device_ptr(&self, offset: usize) -> DevicePtr {
        self.range.base() + offset as u64
    }

    pub fn range(&self) -> &VirtualRange {
        &self.range
    }
}
