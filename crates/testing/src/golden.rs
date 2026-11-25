// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Synthetic Minds LTD
//! Golden checks against llama.cpp `llama-eval-callback` dumps (spec 08 L3).
//!
//! The dump prints, per op, the first and last three values of each row and the tensor
//! sum. For the token-embedding lookup, finding rows of `token_embd.weight` that match
//! those printed values identifies the prompt's token ids and checks the loader
//! (offsets, row size) and the dequantiser against llama.cpp on the same file.

use astral_core::{AstralError, Result};
use astral_loader::Gguf;
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct PrintedTensor {
    pub name: String,
    /// (first three, last three) values of each printed row.
    pub rows: Vec<([f64; 3], [f64; 3])>,
    pub sum: f64,
}

fn parse_row(line: &str) -> Option<([f64; 3], [f64; 3])> {
    let inner = line.trim().trim_end_matches(',').trim_start_matches('[').trim_end_matches(']');
    let vals: Vec<f64> = inner.split(',').map(str::trim).filter(|s| !s.is_empty() && *s != "...").filter_map(|s| s.parse().ok()).collect();
    (vals.len() == 6).then(|| ([vals[0], vals[1], vals[2]], [vals[3], vals[4], vals[5]]))
}

/// Parses the first printed tensor whose name is `name` (e.g. `embd`).
pub fn parse_first(log: &str, name: &str) -> Option<PrintedTensor> {
    let marker = format!(" {name} = ");
    let mut lines = log.lines().skip_while(|l| !(l.contains("common_debug_cb_eval:") && l.contains(&marker)));
    lines.next()?;
    let mut rows = Vec::new();
    for l in lines {
        let t = l.trim();
        if let Some(rest) = t.strip_prefix("sum =") {
            return Some(PrintedTensor { name: name.to_string(), rows, sum: rest.trim().parse().ok()? });
        }
        if t.contains("...") {
            rows.push(parse_row(t)?);
        }
        if t.starts_with("common_debug_cb_eval:") {
            break;
        }
    }
    None
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbdMatch {
    pub tensor: String,
    pub token_ids: Vec<Option<u64>>,
    pub our_sum: f64,
    pub printed_sum: f64,
    pub rows_scanned: u64,
}

impl EmbdMatch {
    /// Every printed row identified, and sums agree within `rel_tol`.
    pub fn pass(&self, rel_tol: f64) -> bool {
        self.token_ids.iter().all(|t| t.is_some()) && (self.our_sum - self.printed_sum).abs() <= rel_tol * self.printed_sum.abs().max(1.0)
    }
}

/// Finds the rows of `tensor` that match each printed row (values printed to 4 decimals).
pub fn match_embedding_rows(g: &Gguf, tensor: &str, printed: &PrintedTensor) -> Result<EmbdMatch> {
    let t = g.tensor(tensor).ok_or_else(|| AstralError::Invalid(format!("no tensor {tensor}")))?;
    let cols = t.dims[0] as usize;
    let rows = t.n_rows();
    let (blk, _) = t.ty.layout().unwrap();
    let bytes = g.read_tensor(t)?;
    let rb = t.row_bytes() as usize;
    let tol = 6e-5;
    let close = |a: &[f32], b: &[f64; 3]| a.iter().zip(b).all(|(&x, &y)| (x as f64 - y).abs() <= tol);
    let mut ids = vec![None; printed.rows.len()];
    for r in 0..rows as usize {
        let row = &bytes[r * rb..(r + 1) * rb];
        // Decode only the first and last block of the row for the search.
        let bb = rb / (cols / blk);
        let head = astral_quant::dequantize(t.ty, &row[..bb], blk)?;
        let tail = astral_quant::dequantize(t.ty, &row[rb - bb..], blk)?;
        for (i, (h, tl)) in printed.rows.iter().enumerate() {
            if ids[i].is_none() && close(&head[..3], h) && close(&tail[blk - 3..], tl) {
                ids[i] = Some(r as u64);
            }
        }
        if ids.iter().all(|x| x.is_some()) {
            break;
        }
    }
    let mut sum = 0f64;
    for id in ids.iter().flatten() {
        let r = *id as usize;
        sum += astral_quant::dequantize(t.ty, &bytes[r * rb..(r + 1) * rb], cols)?.iter().map(|&v| v as f64).sum::<f64>();
    }
    Ok(EmbdMatch { tensor: tensor.to_string(), token_ids: ids, our_sum: sum, printed_sum: printed.sum, rows_scanned: rows })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_llama_cpp_dump_block() {
        let log = "noise\ncommon_debug_cb_eval:                     embd = (f32)   GET_ROWS(token_embd.weight{2304, 201024, 1, 1}, inp_tokens{2, 1, 1, 1}}) = {2304, 2, 1, 1}\n    [\n        [\n            [      2.4843,       0.0391,       0.2739,    ...,      -0.8621,       1.2931,       0.1257  ],\n            [      0.7514,      -0.4032,       0.5314,    ...,      -1.4172,       1.0629,      -0.3149  ],\n        ],\n    ]\n    sum = -144.010513\ncommon_debug_cb_eval: next = ...\n";
        let p = parse_first(log, "embd").unwrap();
        assert_eq!(p.rows.len(), 2);
        assert_eq!(p.rows[0].0, [2.4843, 0.0391, 0.2739]);
        assert_eq!(p.rows[1].1, [-1.4172, 1.0629, -0.3149]);
        assert_eq!(p.sum, -144.010513);
    }
}
