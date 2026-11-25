// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! CPU reference implementations. Slow on purpose: accumulate in f64, no tricks.

#![forbid(unsafe_code)]

/// Row-wise RMSNorm with optional weight.
pub fn rmsnorm_f32(x: &[f32], weight: Option<&[f32]>, rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    assert!(x.len() >= rows * cols, "x too short");
    if let Some(w) = weight {
        assert!(w.len() >= cols, "weight too short");
    }
    let mut y = vec![0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let ms: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / cols as f64;
        let inv = 1.0 / (ms + eps as f64).sqrt();
        for c in 0..cols {
            let w = weight.map_or(1.0, |w| w[c] as f64);
            y[r * cols + c] = (row[c] as f64 * inv * w) as f32;
        }
    }
    y
}

/// MXFP4 GEMV reference: `y[b, n] = sum_k W[n, k] * x[b, k]`, W given as GGUF MXFP4 blocks
/// (row-major, `k / 32` blocks per row), dequantised with the bit-exact ggml decoder and
/// accumulated in f64. Also returns `l1[b, n] = sum_k |W[n, k] * x[b, k]|`, the scale that
/// bounds rounding error in any f32 summation order.
pub fn gemv_mxfp4(blocks: &[u8], x: &[f32], n: usize, k: usize, b: usize) -> (Vec<f32>, Vec<f64>) {
    let w = astral_quant::dequantize(astral_quant::GgmlType::Mxfp4, blocks, n * k).expect("valid mxfp4");
    let mut y = vec![0f32; b * n];
    let mut l1 = vec![0f64; b * n];
    for bi in 0..b {
        for r in 0..n {
            let row = &w[r * k..(r + 1) * k];
            let xb = &x[bi * k..(bi + 1) * k];
            let (mut s, mut a) = (0f64, 0f64);
            for (&wv, &xv) in row.iter().zip(xb) {
                let t = wv as f64 * xv as f64;
                s += t;
                a += t.abs();
            }
            y[bi * n + r] = s as f32;
            l1[bi * n + r] = a;
        }
    }
    (y, l1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_unit_vector() {
        // mean(x^2) = 1 -> output equals input (eps 0).
        let x = [1.0f32, -1.0, 1.0, -1.0];
        assert_eq!(rmsnorm_f32(&x, None, 1, 4, 0.0), x.to_vec());
    }

    #[test]
    fn rmsnorm_weighted_known_values() {
        let x = [3.0f32, 4.0];
        // mean(x^2) = 12.5, inv = 1/sqrt(12.5)
        let y = rmsnorm_f32(&x, Some(&[2.0, 0.5]), 1, 2, 0.0);
        let inv = 1.0 / 12.5f64.sqrt();
        assert!((y[0] as f64 - 3.0 * inv * 2.0).abs() < 1e-6);
        assert!((y[1] as f64 - 4.0 * inv * 0.5).abs() < 1e-6);
    }
}
