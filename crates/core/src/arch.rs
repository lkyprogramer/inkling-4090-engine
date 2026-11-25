// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! Architecture profiles and capability descriptions (spec 01, ADR-0002).
//!
//! Rule: kernel selection keys on [`ArchProfile`] / [`KernelFamily`] and the
//! capability flags here. Never on `compute_capability.major >= N`.

use serde::Serialize;

/// The six Astral target architectures. Anything else is unsupported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchProfile {
    /// A100 (sm_80).
    Sm80,
    /// H100 (sm_90a).
    Sm90a,
    /// B200 (sm_100a).
    Sm100a,
    /// GB300 / B300 (sm_103a).
    Sm103a,
    /// RTX PRO 6000 Blackwell (sm_120a).
    Sm120a,
    /// GB10 / DGX Spark (sm_121a).
    Sm121a,
}

/// Kernel families (spec 03 §1). Each family has its own tuned kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelFamily {
    /// `mma.sync` + `cp.async` (sm_80; also the base of sm_12x).
    F80,
    /// WGMMA + TMA (sm_90a).
    F90,
    /// tcgen05 + TMEM (sm_100a / sm_103a).
    F100,
    /// Block-scaled `mma.sync`, ~100 KB smem (sm_120a / sm_121a).
    F120,
}

/// How far a family has been validated on real silicon (spec 08 §1, R-VA-2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Validation {
    Validated,
    Static,
    Logic,
    Specified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Fp4Path {
    None,
    BlockScaledMmaSync,
    Tcgen05,
}

/// ISA-level features implied by the architecture (static table, spec 01 §1).
#[derive(Debug, Clone, Serialize)]
pub struct IsaFeatures {
    pub cp_async: bool,
    pub wgmma: bool,
    /// Conservative: reported false where sources were not confirmed.
    pub tma: bool,
    pub tcgen05: bool,
    pub tmem: bool,
    pub clusters: bool,
    pub int8_tensor_core: bool,
    pub fp8_tensor_core: bool,
    pub fp4: Fp4Path,
    pub low_precision_formats: Vec<&'static str>,
}

impl ArchProfile {
    /// Exact table lookup. Unknown capabilities return `None` and must fail loudly.
    pub fn from_compute_capability(major: i32, minor: i32) -> Option<Self> {
        match (major, minor) {
            (8, 0) => Some(Self::Sm80),
            (9, 0) => Some(Self::Sm90a),
            (10, 0) => Some(Self::Sm100a),
            (10, 3) => Some(Self::Sm103a),
            (12, 0) => Some(Self::Sm120a),
            (12, 1) => Some(Self::Sm121a),
            _ => None,
        }
    }

    pub fn kernel_family(self) -> KernelFamily {
        match self {
            Self::Sm80 => KernelFamily::F80,
            Self::Sm90a => KernelFamily::F90,
            Self::Sm100a | Self::Sm103a => KernelFamily::F100,
            Self::Sm120a | Self::Sm121a => KernelFamily::F120,
        }
    }

    /// Family to fall back to when the exact family lacks a kernel.
    /// sm_12x shares the sm_80 `mma.sync` base; datacentre families have no fallback.
    pub fn fallback_family(self) -> Option<KernelFamily> {
        match self.kernel_family() {
            KernelFamily::F120 => Some(KernelFamily::F80),
            _ => None,
        }
    }

    /// The nvcc target string used for this profile's cubins.
    pub fn sm_target(self) -> &'static str {
        match self {
            Self::Sm80 => "80",
            Self::Sm90a => "90a",
            Self::Sm100a => "100a",
            Self::Sm103a => "103a",
            Self::Sm120a => "120a",
            Self::Sm121a => "121a",
        }
    }

    /// How far each family has been exercised on real silicon (spec 08 §1).
    /// sm_80 (A100-PCIE-40GB, 2026-11-23) and sm_90a (H100 PCIe, 2026-11-24) have run the
    /// P0 kernels and self-test; no model kernels exist yet. Everything else is design-only.
    pub fn validation(self) -> Validation {
        match self {
            Self::Sm80 | Self::Sm90a => Validation::Logic,
            _ => Validation::Specified,
        }
    }

    pub fn isa_features(self) -> IsaFeatures {
        match self {
            Self::Sm80 => IsaFeatures {
                cp_async: true,
                wgmma: false,
                tma: false,
                tcgen05: false,
                tmem: false,
                clusters: false,
                int8_tensor_core: true,
                fp8_tensor_core: false,
                fp4: Fp4Path::None,
                low_precision_formats: vec![],
            },
            Self::Sm90a => IsaFeatures {
                cp_async: true,
                wgmma: true,
                tma: true,
                tcgen05: false,
                tmem: false,
                clusters: true,
                int8_tensor_core: true,
                fp8_tensor_core: true,
                fp4: Fp4Path::None,
                low_precision_formats: vec!["fp8_e4m3", "fp8_e5m2"],
            },
            Self::Sm100a | Self::Sm103a => IsaFeatures {
                cp_async: true,
                wgmma: false,
                tma: true,
                tcgen05: true,
                tmem: true,
                clusters: true,
                // sm_103 trades INT8 throughput for NVFP4 (spec 01 §1).
                int8_tensor_core: self == Self::Sm100a,
                fp8_tensor_core: true,
                fp4: Fp4Path::Tcgen05,
                low_precision_formats: vec!["fp8_e4m3", "fp8_e5m2", "mxfp8", "mxfp4", "nvfp4"],
            },
            Self::Sm120a | Self::Sm121a => IsaFeatures {
                cp_async: true,
                wgmma: false,
                tma: false,
                tcgen05: false,
                tmem: false,
                clusters: false,
                int8_tensor_core: true,
                fp8_tensor_core: true,
                fp4: Fp4Path::BlockScaledMmaSync,
                low_precision_formats: vec!["fp8_e4m3", "fp8_e5m2", "mxfp8", "mxfp4", "nvfp4"],
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryModel {
    /// Separate device memory (A100, H100, B200, GB300, RTX PRO 6000).
    Discrete,
    /// Host and device share one pool (GB10). Keyed on the driver's INTEGRATED flag.
    Unified,
}

/// Measured + derived description of one device. Filled by `astral-hw`.
#[derive(Debug, Clone, Serialize)]
pub struct Capabilities {
    pub compute_capability: (i32, i32),
    pub profile: Option<ArchProfile>,
    pub kernel_family: Option<KernelFamily>,
    pub validation: Option<Validation>,
    pub isa: Option<IsaFeatures>,
    pub sm_count: u32,
    pub warp_size: u32,
    pub max_threads_per_block: u32,
    pub max_threads_per_sm: u32,
    pub regs_per_sm: u32,
    pub smem_per_sm_bytes: u32,
    pub smem_per_block_optin_bytes: u32,
    pub l2_bytes: u64,
    pub sm_clock_mhz: Option<u32>,
    pub mem_clock_mhz: Option<u32>,
    pub mem_bus_width_bits: u32,
    pub total_mem_bytes: u64,
    pub memory_model: MemoryModel,
    pub ecc_enabled: Option<bool>,
    pub vmm_supported: Option<bool>,
    pub memory_pools_supported: Option<bool>,
    pub managed_memory: Option<bool>,
    pub pageable_memory_access: Option<bool>,
    pub concurrent_managed_access: Option<bool>,
    pub cluster_launch: Option<bool>,
    /// 2 x memory clock x bus width. A DDR assumption: correct for HBM,
    /// possibly wrong for GDDR7 / LPDDR5x. Always prefer a measured figure.
    pub dram_bandwidth_gbps_derived: Option<f64>,
}

/// 2 x clock x bus width / 8 (double data rate), in GB/s.
pub fn derived_dram_bandwidth_gbps(mem_clock_mhz: u32, bus_width_bits: u32) -> f64 {
    2.0 * mem_clock_mhz as f64 * 1e6 * bus_width_bits as f64 / 8.0 / 1e9
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn six_targets_map_exactly() {
        assert_eq!(ArchProfile::from_compute_capability(8, 0), Some(ArchProfile::Sm80));
        assert_eq!(ArchProfile::from_compute_capability(9, 0), Some(ArchProfile::Sm90a));
        assert_eq!(ArchProfile::from_compute_capability(10, 0), Some(ArchProfile::Sm100a));
        assert_eq!(ArchProfile::from_compute_capability(10, 3), Some(ArchProfile::Sm103a));
        assert_eq!(ArchProfile::from_compute_capability(12, 0), Some(ArchProfile::Sm120a));
        assert_eq!(ArchProfile::from_compute_capability(12, 1), Some(ArchProfile::Sm121a));
    }

    #[test]
    fn unknown_capabilities_are_rejected_not_rounded() {
        // A ">= N" rule would wrongly accept all of these.
        for (maj, min) in [(8, 6), (8, 9), (9, 1), (10, 1), (10, 2), (11, 0), (12, 2), (13, 0)] {
            assert_eq!(ArchProfile::from_compute_capability(maj, min), None, "{maj}.{min}");
        }
    }

    #[test]
    fn gb10_is_not_routed_to_datacentre_blackwell() {
        // The documented vLLM failure: sm_121 sent to tcgen05 paths.
        let p = ArchProfile::from_compute_capability(12, 1).unwrap();
        assert_eq!(p.kernel_family(), KernelFamily::F120);
        assert!(!p.isa_features().tcgen05);
        assert!(!p.isa_features().tmem);
        assert_eq!(p.fallback_family(), Some(KernelFamily::F80));
    }

    #[test]
    fn a100_has_no_fp8_or_fp4() {
        let f = ArchProfile::Sm80.isa_features();
        assert!(!f.fp8_tensor_core);
        assert_eq!(f.fp4, Fp4Path::None);
    }

    #[test]
    fn only_a100_and_h100_have_run_on_silicon() {
        assert_eq!(ArchProfile::Sm80.validation(), Validation::Logic);
        assert_eq!(ArchProfile::Sm90a.validation(), Validation::Logic);
        for p in [ArchProfile::Sm100a, ArchProfile::Sm103a, ArchProfile::Sm120a, ArchProfile::Sm121a] {
            assert_eq!(p.validation(), Validation::Specified);
        }
    }

    #[test]
    fn a100_pcie_40gb_bandwidth_from_clocks() {
        // 1215 MHz HBM2e x 5120-bit bus ~= 1555 GB/s.
        let bw = derived_dram_bandwidth_gbps(1215, 5120);
        assert!((bw - 1555.2).abs() < 0.5, "{bw}");
    }
}
