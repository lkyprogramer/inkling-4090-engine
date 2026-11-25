// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Synthetic Minds LTD
//! Optional NVML access (topology, PCIe, clocks, power). Every query tolerates
//! `NOT_SUPPORTED` by returning `None` (spec 01 §3).

use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::OnceLock;

pub type NvmlReturn = i32;
pub type NvmlDevice = *mut c_void;

api_table! {
    pub struct NvmlFns -> NvmlReturn {
        nvmlInit_v2();
        nvmlSystemGetDriverVersion(*mut c_char, u32);
        nvmlDeviceGetHandleByUUID(*const c_char, *mut NvmlDevice);
        nvmlDeviceGetCurrPcieLinkGeneration(NvmlDevice, *mut u32);
        nvmlDeviceGetMaxPcieLinkGeneration(NvmlDevice, *mut u32);
        nvmlDeviceGetCurrPcieLinkWidth(NvmlDevice, *mut u32);
        nvmlDeviceGetMaxPcieLinkWidth(NvmlDevice, *mut u32);
        nvmlDeviceGetPowerManagementLimit(NvmlDevice, *mut u32);
        nvmlDeviceGetClockInfo(NvmlDevice, u32, *mut u32);
        nvmlDeviceGetMaxClockInfo(NvmlDevice, u32, *mut u32);
        nvmlDeviceGetNvLinkState(NvmlDevice, u32, *mut u32);
    }
}

pub struct Nvml {
    fns: NvmlFns,
    _lib: libloading::Library,
}

static NVML: OnceLock<Option<Nvml>> = OnceLock::new();

/// Returns NVML if it can be loaded and initialised; `None` otherwise.
pub fn nvml() -> Option<&'static Nvml> {
    NVML.get_or_init(|| {
        for name in ["libnvidia-ml.so.1", "libnvidia-ml.so"] {
            // SAFETY: loading the system NVML library.
            let Ok(lib) = (unsafe { libloading::Library::new(name) }) else { continue };
            // SAFETY: symbol signatures follow nvml.h.
            let Ok(fns) = (unsafe { NvmlFns::load(&lib) }) else { continue };
            // SAFETY: documented first call.
            if unsafe { (fns.nvmlInit_v2)() } == 0 {
                return Some(Nvml { fns, _lib: lib });
            }
        }
        None
    })
    .as_ref()
}

/// NVML clock domains.
const CLOCK_SM: u32 = 1;
const CLOCK_MEM: u32 = 2;
const MAX_NVLINKS: u32 = 18;

#[derive(Debug, Clone, Default)]
pub struct NvmlDeviceInfo {
    pub pcie_gen_current: Option<u32>,
    pub pcie_gen_max: Option<u32>,
    pub pcie_width_current: Option<u32>,
    pub pcie_width_max: Option<u32>,
    pub power_limit_w: Option<f64>,
    pub sm_clock_mhz: Option<u32>,
    pub mem_clock_mhz: Option<u32>,
    pub max_sm_clock_mhz: Option<u32>,
    pub max_mem_clock_mhz: Option<u32>,
    /// `None` when NVLink queries are not supported (e.g. PCIe-only cards).
    pub nvlink_active_links: Option<u32>,
}

impl Nvml {
    pub fn driver_version(&self) -> Option<String> {
        let mut buf = [0 as c_char; 96];
        // SAFETY: length passed; NVML NUL-terminates.
        if unsafe { (self.fns.nvmlSystemGetDriverVersion)(buf.as_mut_ptr(), buf.len() as u32) } != 0 {
            return None;
        }
        // SAFETY: NUL-terminated by NVML.
        Some(unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned())
    }

    /// Looks the device up by UUID (CUDA and NVML ordinals can differ).
    pub fn device_info(&self, uuid: &str) -> Option<NvmlDeviceInfo> {
        let c = CString::new(uuid).ok()?;
        let mut h: NvmlDevice = std::ptr::null_mut();
        // SAFETY: out-pointer to a local; `c` outlives the call.
        if unsafe { (self.fns.nvmlDeviceGetHandleByUUID)(c.as_ptr(), &mut h) } != 0 {
            return None;
        }
        let q = |f: unsafe extern "C" fn(NvmlDevice, *mut u32) -> NvmlReturn| {
            let mut v = 0u32;
            // SAFETY: valid handle, out-pointer to a local.
            (unsafe { f(h, &mut v) } == 0).then_some(v)
        };
        let clk = |f: unsafe extern "C" fn(NvmlDevice, u32, *mut u32) -> NvmlReturn, domain: u32| {
            let mut v = 0u32;
            // SAFETY: valid handle, out-pointer to a local.
            (unsafe { f(h, domain, &mut v) } == 0).then_some(v)
        };
        let mut nvlink: Option<u32> = None;
        for link in 0..MAX_NVLINKS {
            let mut state = 0u32;
            // SAFETY: valid handle, out-pointer to a local.
            if unsafe { (self.fns.nvmlDeviceGetNvLinkState)(h, link, &mut state) } == 0 {
                *nvlink.get_or_insert(0) += u32::from(state == 1);
            } else if link == 0 {
                break;
            }
        }
        Some(NvmlDeviceInfo {
            pcie_gen_current: q(self.fns.nvmlDeviceGetCurrPcieLinkGeneration),
            pcie_gen_max: q(self.fns.nvmlDeviceGetMaxPcieLinkGeneration),
            pcie_width_current: q(self.fns.nvmlDeviceGetCurrPcieLinkWidth),
            pcie_width_max: q(self.fns.nvmlDeviceGetMaxPcieLinkWidth),
            power_limit_w: q(self.fns.nvmlDeviceGetPowerManagementLimit).map(|mw| mw as f64 / 1000.0),
            sm_clock_mhz: clk(self.fns.nvmlDeviceGetClockInfo, CLOCK_SM),
            mem_clock_mhz: clk(self.fns.nvmlDeviceGetClockInfo, CLOCK_MEM),
            max_sm_clock_mhz: clk(self.fns.nvmlDeviceGetMaxClockInfo, CLOCK_SM),
            max_mem_clock_mhz: clk(self.fns.nvmlDeviceGetMaxClockInfo, CLOCK_MEM),
            nvlink_active_links: nvlink,
        })
    }
}
