// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! Compiles every `cu/*.cu` file into one fatbin per file with explicit per-architecture
//! cubins (`-gencode arch=compute_X,code=sm_X`; never `-arch=sm_Xa`, spec 01 §7) and
//! generates `FATBINS` for embedding.
//!
//! Environment:
//! - `ASTRAL_CUDA_ARCHS`: comma-separated targets, default `80` (e.g. `80,90a,100a,120a`).
//! - `NVCC`: path to nvcc (default: `nvcc` on PATH, then `/usr/local/cuda/bin/nvcc`).
//!
//! If nvcc is absent the crate still builds with an empty kernel list, so host-only
//! development (and the mock backend) works on machines without CUDA.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_nvcc() -> Option<PathBuf> {
    if let Ok(p) = env::var("NVCC") {
        return Some(PathBuf::from(p));
    }
    for cand in ["nvcc", "/usr/local/cuda/bin/nvcc"] {
        if Command::new(cand).arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
            return Some(PathBuf::from(cand));
        }
    }
    None
}

fn nvcc_version(nvcc: &Path) -> String {
    Command::new(nvcc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.lines().rev().find(|l| l.contains("release")).map(|l| l.trim().to_string()))
        .unwrap_or_else(|| "unknown".into())
}

fn main() {
    println!("cargo:rerun-if-changed=cu");
    println!("cargo:rerun-if-env-changed=ASTRAL_CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=ASTRAL_NVCC_DEFINES");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let archs: Vec<String> = env::var("ASTRAL_CUDA_ARCHS").unwrap_or_else(|_| "80".into()).split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();

    let mut sources: Vec<PathBuf> =
        fs::read_dir(manifest.join("cu")).expect("kernels-src/cu directory").filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "cu")).collect();
    sources.sort();

    let nvcc = find_nvcc();
    let mut table = String::new();
    let (built_archs, version) = match &nvcc {
        None => {
            println!("cargo:warning=nvcc not found: building astral-kernels-src WITHOUT GPU kernels (host-only build)");
            (Vec::new(), "none".to_string())
        }
        Some(nvcc) => {
            let headers = manifest.join("cu");
            for src in &sources {
                let stem = src.file_stem().unwrap().to_string_lossy().to_string();
                let dst = out.join(format!("{stem}.fatbin"));
                let mut cmd = Command::new(nvcc);
                cmd.arg("-fatbin").arg("-O3").arg("-std=c++17").arg("-lineinfo").arg("--expt-relaxed-constexpr").arg("-I").arg(&headers);
                // Extra -D defines for kernel tuning experiments, e.g. ASTRAL_NVCC_DEFINES="ASTRAL_MMA_LOAD=1".
                for d in env::var("ASTRAL_NVCC_DEFINES").unwrap_or_default().split_whitespace() {
                    cmd.arg(format!("-D{d}"));
                }
                for a in &archs {
                    // "90a" -> compute_90a / sm_90a
                    cmd.arg("-gencode").arg(format!("arch=compute_{a},code=sm_{a}"));
                }
                cmd.arg("-o").arg(&dst).arg(src);
                let status = cmd.status().unwrap_or_else(|e| panic!("failed to run nvcc: {e}"));
                if !status.success() {
                    panic!("nvcc failed for {} (archs {:?})", src.display(), archs);
                }
                writeln!(table, "    ({stem:?}, include_bytes!({:?}) as &[u8]),", dst.display().to_string()).unwrap();
            }
            (archs.clone(), nvcc_version(nvcc))
        }
    };

    let mut generated = String::new();
    writeln!(generated, "/// (module name, fatbin image) for every compiled `cu/*.cu` file.").unwrap();
    writeln!(generated, "pub static FATBINS: &[(&str, &[u8])] = &[\n{table}];").unwrap();
    writeln!(generated, "/// SM targets embedded in every fatbin (empty when nvcc was unavailable).").unwrap();
    writeln!(generated, "pub const BUILT_ARCHS: &[&str] = &{built_archs:?};").unwrap();
    writeln!(generated, "/// `nvcc --version` release line used for this build.").unwrap();
    writeln!(generated, "pub const NVCC_VERSION: &str = {version:?};").unwrap();
    fs::write(out.join("fatbins.rs"), generated).unwrap();
}
