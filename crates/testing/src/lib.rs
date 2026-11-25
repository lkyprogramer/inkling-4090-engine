// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Synthetic Minds LTD
//! `astral-testing`: deterministic RNG, numeric comparison, and the GPU self-test that
//! checks every kernel against its CPU reference (spec 08, layer L2).

#![forbid(unsafe_code)]

pub mod golden;
pub mod kbench;
pub mod selftest;

/// Small deterministic xorshift64* generator (no external dependency, reproducible).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [-1, 1).
    pub fn unit_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    pub fn vec_f32(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.unit_f32() * scale).collect()
    }
}

/// Worst-case error of `got` against `want`, using |a-b| / max(1, |b|).
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ErrorStats {
    pub max_abs: f64,
    pub max_scaled: f64,
    pub worst_index: usize,
}

pub fn compare(got: &[f32], want: &[f32]) -> ErrorStats {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let mut s = ErrorStats { max_abs: 0.0, max_scaled: 0.0, worst_index: 0 };
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let d = (g as f64 - w as f64).abs();
        let scaled = if g.is_nan() != w.is_nan() { f64::INFINITY } else { d / (w as f64).abs().max(1.0) };
        if scaled > s.max_scaled {
            s = ErrorStats { max_abs: d, max_scaled: scaled, worst_index: i };
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic_and_bounded() {
        let (mut a, mut b) = (Rng::new(7), Rng::new(7));
        for _ in 0..1000 {
            let x = a.unit_f32();
            assert_eq!(x, b.unit_f32());
            assert!((-1.0..1.0).contains(&x));
        }
    }

    #[test]
    fn compare_flags_nan() {
        assert!(compare(&[f32::NAN], &[1.0]).max_scaled.is_infinite());
        assert_eq!(compare(&[1.0, 2.0], &[1.0, 2.0]).max_scaled, 0.0);
    }
}
