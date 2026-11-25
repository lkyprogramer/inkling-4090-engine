// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! Virtual memory management (spec 04 §5): reserve one large device address range,
//! then map physical memory into it on demand. Addresses never move, which is what
//! CUDA graphs need, and the range can grow without copying.

use crate::{check, drv, sys, Context};
use astral_core::{AstralError, Result};
use std::ffi::c_void;
use std::sync::Arc;

fn device_prop(ctx: &Context) -> sys::CUmemAllocationProp {
    sys::CUmemAllocationProp {
        type_: sys::CU_MEM_ALLOCATION_TYPE_PINNED,
        requested_handle_types: 0,
        location: sys::CUmemLocation { type_: sys::CU_MEM_LOCATION_TYPE_DEVICE, id: ctx.device().ordinal() as i32 },
        win32_handle_meta_data: std::ptr::null_mut(),
        alloc_flags: sys::CUmemAllocFlags::default(),
    }
}

fn round_up(v: usize, to: usize) -> usize {
    v.div_ceil(to) * to
}

/// A reserved device virtual address range, partially backed by physical memory.
pub struct VirtualRange {
    ctx: Arc<Context>,
    base: sys::CUdeviceptr,
    reserved: usize,
    granularity: usize,
    mapped: usize,
    /// (offset, size, handle) of each physical allocation, in mapping order.
    maps: Vec<(usize, usize, sys::CUmemGenericAllocationHandle)>,
}

// SAFETY: plain handles; all operations re-bind the context.
unsafe impl Send for VirtualRange {}

impl VirtualRange {
    /// Recommended physical allocation granularity for this device (2 MiB on current GPUs).
    pub fn granularity(ctx: &Arc<Context>) -> Result<usize> {
        ctx.bind()?;
        let d = drv()?;
        let prop = device_prop(ctx);
        let mut g = 0usize;
        // SAFETY: out-pointer to a local; `prop` outlives the call.
        check("cuMemGetAllocationGranularity", unsafe { (d.fns.cuMemGetAllocationGranularity)(&mut g, &prop, sys::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) })?;
        Ok(g.max(1))
    }

    /// Reserves `bytes` (rounded up to the granularity) of address space. Maps nothing yet.
    pub fn reserve(ctx: &Arc<Context>, bytes: usize) -> Result<Self> {
        let granularity = Self::granularity(ctx)?;
        let reserved = round_up(bytes.max(1), granularity);
        let d = drv()?;
        let mut base = 0;
        // SAFETY: out-pointer to a local; freed in Drop.
        check("cuMemAddressReserve", unsafe { (d.fns.cuMemAddressReserve)(&mut base, reserved, granularity, 0, 0) })?;
        Ok(Self { ctx: ctx.clone(), base, reserved, granularity, mapped: 0, maps: Vec::new() })
    }

    pub fn base(&self) -> sys::CUdeviceptr {
        self.base
    }

    pub fn reserved(&self) -> usize {
        self.reserved
    }

    pub fn mapped(&self) -> usize {
        self.mapped
    }

    pub fn granule(&self) -> usize {
        self.granularity
    }

    /// Ensures `[0, bytes)` is backed by physical memory (rounded up to the granularity).
    pub fn map_up_to(&mut self, bytes: usize) -> Result<()> {
        let target = round_up(bytes, self.granularity);
        if target <= self.mapped {
            return Ok(());
        }
        if target > self.reserved {
            return Err(AstralError::Invalid(format!("map_up_to {target} exceeds reservation {}", self.reserved)));
        }
        self.ctx.bind()?;
        let d = drv()?;
        let size = target - self.mapped;
        let prop = device_prop(&self.ctx);
        let mut handle = 0;
        // SAFETY: out-pointer to a local; released in Drop after unmapping.
        check("cuMemCreate", unsafe { (d.fns.cuMemCreate)(&mut handle, size, &prop, 0) })?;
        let at = self.base + self.mapped as u64;
        // SAFETY: [at, at+size) lies inside our reservation and is currently unmapped.
        if let Err(e) = check("cuMemMap", unsafe { (d.fns.cuMemMap)(at, size, 0, handle, 0) }) {
            // SAFETY: handle was just created and is not mapped.
            unsafe { (d.fns.cuMemRelease)(handle) };
            return Err(e);
        }
        self.maps.push((self.mapped, size, handle));
        self.mapped = target;
        let access = sys::CUmemAccessDesc { location: prop.location, flags: sys::CU_MEM_ACCESS_FLAGS_PROT_READWRITE };
        // SAFETY: range just mapped; descriptor outlives the call.
        check("cuMemSetAccess", unsafe { (d.fns.cuMemSetAccess)(at, size, &access, 1) })
    }

    fn bounds(&self, offset: usize, len: usize) -> Result<()> {
        match offset.checked_add(len) {
            Some(end) if end <= self.mapped => Ok(()),
            _ => Err(AstralError::Invalid(format!("range [{offset}, +{len}) outside mapped {}", self.mapped))),
        }
    }

    /// Synchronous host-to-device write into the mapped region.
    pub fn write(&self, offset: usize, data: &[u8]) -> Result<()> {
        self.bounds(offset, data.len())?;
        if data.is_empty() {
            return Ok(());
        }
        self.ctx.bind()?;
        let d = drv()?;
        // SAFETY: destination is inside the mapped range (checked); source slice is valid.
        check("cuMemcpyHtoD_v2", unsafe { (d.fns.cuMemcpyHtoD_v2)(self.base + offset as u64, data.as_ptr() as *const c_void, data.len()) })
    }

    /// Synchronous device-to-host read from the mapped region.
    pub fn read(&self, offset: usize, out: &mut [u8]) -> Result<()> {
        self.bounds(offset, out.len())?;
        if out.is_empty() {
            return Ok(());
        }
        self.ctx.bind()?;
        let d = drv()?;
        // SAFETY: source is inside the mapped range (checked); destination slice is valid for writes.
        check("cuMemcpyDtoH_v2", unsafe { (d.fns.cuMemcpyDtoH_v2)(out.as_mut_ptr() as *mut c_void, self.base + offset as u64, out.len()) })
    }
}

impl Drop for VirtualRange {
    fn drop(&mut self) {
        let Ok(d) = sys::driver() else { return };
        let _ = self.ctx.bind();
        // SAFETY: each mapping is unmapped and released exactly once, then the reservation freed.
        unsafe {
            for &(off, size, handle) in self.maps.iter().rev() {
                (d.fns.cuMemUnmap)(self.base + off as u64, size);
                (d.fns.cuMemRelease)(handle);
            }
            (d.fns.cuMemAddressFree)(self.base, self.reserved);
        }
    }
}
