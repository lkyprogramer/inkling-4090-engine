// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! Inkling 2 Mini from its native safetensors checkpoint.
//!
//! The checkpoint is presented as a [`Gguf`] with the tensor names, types and layouts of the
//! GGUF conversion, so the model loads through the same code path and gets the same kernels:
//!
//! - dense projections and the embeddings (BF16) are quantised to Q8_0 exactly as ggml does;
//! - NVFP4 experts are repacked into GGUF NVFP4 blocks without requantisation, with the
//!   per-expert global scale `amax / outer_scale_denominator`;
//! - MXFP8 experts are dequantised and quantised to Q8_0;
//! - the fused gate/up expert rows, stored interleaved (gate, up, gate, ...), are split into a
//!   gate half followed by an up half;
//! - norms, short-convolution taps, routing and loop-controller weights become F32, with the
//!   transposes and the time reversal (dynamic short-conv static taps) the graph expects.
//!
//! Each tensor is converted when it is read; nothing is written to disk.

use crate::safetensors::{Safetensors, StTensor};
use crate::{Gguf, TensorInfo, Value};
use astral_core::{AstralError, Result};
use astral_quant::GgmlType;
use serde_json::Value as Json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn bad(m: String) -> AstralError {
    AstralError::Invalid(format!("inkling checkpoint: {m}"))
}

/// Layout change applied to an F32 tensor.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Xf {
    /// Same element order; GGML dims are the safetensors shape reversed.
    Id,
    /// `[a, b]` -> GGML `[a, b]`: element `(i, j)` lands at `j * a + i`.
    Transpose,
    /// Transpose with the first axis reversed (static short-conv taps).
    FlipTranspose,
}

#[derive(Clone, Debug)]
enum Recipe {
    F32(String, Xf),
    Bf16(String),
    Q8(String),
    /// Loop message queries of core layer `i`.
    MsgQuery(usize),
    /// Per-expert NVFP4 global scale: `<base>_amax / <base>_outer_scale_denominator`.
    ExpertScale(String),
    /// Expert weights `<base>` (+ `<base>_scale`); `true` splits interleaved gate/up rows.
    Nvfp4(String, bool),
    Mxfp8ToQ8(String, bool),
}

struct Source {
    shards: Vec<Safetensors>,
    index: HashMap<String, usize>,
}

impl Source {
    fn open(dir: &Path) -> Result<Self> {
        let mut files: Vec<PathBuf> = Vec::new();
        let idx = dir.join("model.safetensors.index.json");
        if idx.exists() {
            let j: Json = serde_json::from_slice(&std::fs::read(&idx)?).map_err(|e| bad(format!("{}: {e}", idx.display())))?;
            let map = j["weight_map"].as_object().ok_or_else(|| bad("index has no weight_map".into()))?;
            for f in map.values().filter_map(|v| v.as_str()) {
                let p = dir.join(f);
                if !files.contains(&p) {
                    files.push(p);
                }
            }
        } else {
            for e in std::fs::read_dir(dir)? {
                let p = e?.path();
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or_default();
                if name.ends_with(".safetensors") && name != "mtp.safetensors" {
                    files.push(p);
                }
            }
            files.sort();
        }
        if files.is_empty() {
            return Err(bad(format!("no model safetensors in {}", dir.display())));
        }
        let mut shards = Vec::with_capacity(files.len());
        let mut index = HashMap::new();
        for (i, f) in files.iter().enumerate() {
            let st = Safetensors::open(f)?;
            for n in st.names() {
                index.insert(n.to_string(), i);
            }
            shards.push(st);
        }
        Ok(Self { shards, index })
    }

    fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    fn get(&self, name: &str) -> Result<(&Safetensors, &StTensor)> {
        let &i = self.index.get(name).ok_or_else(|| bad(format!("missing tensor {name}")))?;
        let st = &self.shards[i];
        Ok((st, st.tensor(name).expect("indexed")))
    }

    fn shape(&self, name: &str) -> Result<Vec<u64>> {
        Ok(self.get(name)?.1.shape.clone())
    }

    fn dtype(&self, name: &str) -> Result<String> {
        Ok(self.get(name)?.1.dtype.clone())
    }

    fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let (st, t) = self.get(name)?;
        st.read_f32(t)
    }

    fn raw(&self, name: &str) -> Result<Vec<u8>> {
        let (st, t) = self.get(name)?;
        st.read(t)
    }
}

/// Shape without size-1 axes (at least one axis kept).
fn squeeze(s: &[u64]) -> Vec<u64> {
    let v: Vec<u64> = s.iter().copied().filter(|&d| d != 1).collect();
    if v.is_empty() {
        vec![1]
    } else {
        v
    }
}

fn reversed(s: &[u64]) -> Vec<u64> {
    let mut v = squeeze(s);
    v.reverse();
    v
}

/// The text model's configuration as GGUF metadata.
fn metadata(cfg: &Json, name: &str) -> Result<Vec<(String, Value)>> {
    let t = &cfg["text_config"];
    let u = |k: &str| t[k].as_u64().map(|v| v as u32).ok_or_else(|| bad(format!("config: missing {k}")));
    let f = |k: &str| t[k].as_f64().map(|v| v as f32).ok_or_else(|| bad(format!("config: missing {k}")));
    let lc = &t["loop_config"];
    let lu = |k: &str| lc[k].as_u64().map(|v| v as u32).ok_or_else(|| bad(format!("config: missing loop_config.{k}")));
    let n_layer = u("num_hidden_layers")?;
    let local: Vec<u64> = t["local_layer_ids"].as_array().ok_or_else(|| bad("config: missing local_layer_ids".into()))?.iter().filter_map(|v| v.as_u64()).collect();
    if t["sliding_window_size"].as_u64() != t["rel_extent"].as_u64() {
        return Err(bad("sliding window differs from the relative extent".into()));
    }
    let a = "inkling2_loop";
    let mut m: Vec<(String, Value)> = vec![("general.architecture".into(), Value::Str(a.into())), ("general.name".into(), Value::Str(name.into())), ("general.alignment".into(), Value::U32(32))];
    let mut put = |k: &str, v: Value| m.push((format!("{a}.{k}"), v));
    put("block_count", Value::U32(n_layer));
    put("context_length", Value::U32(u("model_max_length")?));
    put("embedding_length", Value::U32(u("hidden_size")?));
    put("feed_forward_length", Value::U32(u("intermediate_size")?));
    put("attention.head_count", Value::U32(u("num_attention_heads")?));
    put("attention.head_count_kv", Value::U32(u("num_key_value_heads")?));
    put("attention.layer_norm_rms_epsilon", Value::F32(f("rms_norm_eps")?));
    put("attention.key_length", Value::U32(u("head_dim")?));
    put("attention.value_length", Value::U32(u("head_dim")?));
    put("expert_count", Value::U32(u("n_routed_experts")?));
    put("expert_used_count", Value::U32(u("num_experts_per_tok")?));
    put("expert_feed_forward_length", Value::U32(u("intermediate_size")?));
    put("expert_weights_scale", Value::F32(f("route_scale")?));
    put("loop.count", Value::U32(lu("num_passes")?));
    put("loop.start", Value::U32(lu("core_start")?));
    put("loop.end", Value::U32(lu("core_end")?));
    put("loop.gate_rank", Value::U32(lu("gate_rank")?));
    put("vocab_size", Value::U32(u("vocab_size")?));
    put("vocab_size_unpadded", Value::U32(u("unpadded_vocab_size")?));
    put("attention.sliding_window", Value::U32(u("sliding_window_size")?));
    put("attention.sliding_window_pattern", Value::Array((0..n_layer as u64).map(|l| Value::Bool(local.contains(&l))).collect()));
    put("attention.relative_length", Value::U32(u("d_rel")?));
    put("attention.relative_extent", Value::U32(u("rel_extent")?));
    put("attention.qk_clip", Value::F32(f("qk_l2_clip_threshold")?));
    put("attention.relative_clip", Value::F32(f("rel_l2_clip_threshold")?));
    put("attention.log_scale_floor", Value::F32(f("log_scaling_n_floor")?));
    put("attention.log_scale_alpha", Value::F32(f("log_scaling_alpha")?));
    put("activation.situ_beta", Value::F32(f("mlp_gate_softcap")?));
    put("activation.situ_linear_beta", Value::F32(f("mlp_up_softcap")?));
    put("attn_res.block_size", Value::U32(u("n_layers_per_attn_res_block")?));
    put("final_logit_softcapping", Value::F32(f("final_logit_softcapping")?));
    put("shortconv.l_cache", Value::U32(u("sconv_kernel_size")?));
    Ok(m)
}

type Plan = Vec<(String, Vec<u64>, GgmlType, Recipe)>;

/// The GGUF tensor table and how to build each tensor.
fn plan(src: &Source, cfg: &Json) -> Result<Plan> {
    let t = &cfg["text_config"];
    let n_layer = t["num_hidden_layers"].as_u64().unwrap_or(0) as usize;
    let lc = &t["loop_config"];
    let (core_start, core_end) = (lc["core_start"].as_u64().unwrap_or(0) as usize, lc["core_end"].as_u64().unwrap_or(0) as usize);
    let mut p: Plan = Vec::new();
    let f32_id = |p: &mut Plan, gg: String, st: String| -> Result<()> {
        let dims = reversed(&src.shape(&st)?);
        p.push((gg, dims, GgmlType::F32, Recipe::F32(st, Xf::Id)));
        Ok(())
    };
    let q8 = |p: &mut Plan, gg: String, st: String| -> Result<()> {
        let dims = reversed(&src.shape(&st)?);
        p.push((gg, dims, GgmlType::Q8_0, Recipe::Q8(st)));
        Ok(())
    };
    let m = "model.llm.";
    q8(&mut p, "token_embd.weight".into(), format!("{m}embed.weight"))?;
    f32_id(&mut p, "token_embd_norm.weight".into(), format!("{m}embed_norm.weight"))?;
    q8(&mut p, "output.weight".into(), format!("{m}unembed.weight"))?;
    f32_id(&mut p, "output_norm.weight".into(), format!("{m}norm.weight"))?;
    f32_id(&mut p, "output_res_query.weight".into(), format!("{m}attn_res.output_query"))?;

    for l in 0..n_layer {
        let s = |x: &str| format!("{m}layers.{l}.{x}");
        let g = |x: &str| format!("blk.{l}.{x}");
        for (gg, st) in [
            ("attn_norm.weight", "attn_norm.weight"),
            ("attn_q_inp_norm.weight", "q_input_norm.weight"),
            ("attn_k_inp_norm.weight", "k_input_norm.weight"),
            ("attn_v_inp_norm.weight", "v_input_norm.weight"),
            ("ffn_norm.weight", "mlp_norm.weight"),
        ] {
            if src.has(&s(st)) {
                f32_id(&mut p, g(gg), s(st))?;
            }
        }
        for (gg, st) in [
            ("attn_q.weight", "attn.wq_du.weight"),
            ("attn_k.weight", "attn.wk_dv.weight"),
            ("attn_v.weight", "attn.wv_dv.weight"),
            ("attn_output.weight", "attn.wo_ud.weight"),
            ("attn_gate.weight", "attn.gate_attn.weight"),
            ("attn_rel.weight", "attn.wr_du.weight"),
        ] {
            q8(&mut p, g(gg), s(st))?;
        }
        f32_id(&mut p, g("attn_q_clip.weight"), s("attn.q_l2_clip.scale"))?;
        f32_id(&mut p, g("attn_rel_clip.weight"), s("attn.rel_l2_clip.scale"))?;
        let rel = s("attn.rel_logits_proj.proj");
        let rs = squeeze(&src.shape(&rel)?);
        p.push((g("attn_rel_bias.weight"), rs.clone(), GgmlType::F32, Recipe::F32(rel, Xf::Transpose)));
        let sink = s("attn.attn_sink_proj.weight");
        p.push((g("attn_sink_proj.weight"), reversed(&src.shape(&sink)?), GgmlType::Bf16, Recipe::Bf16(sink)));
        for kv in ["k", "v"] {
            let plain = s(&format!("attn.{kv}_sconv.weight"));
            if src.has(&plain) {
                f32_id(&mut p, g(&format!("attn_{kv}_conv.weight")), plain)?;
            } else {
                let st = s(&format!("attn.{kv}_sconv.static_w"));
                let dims = squeeze(&src.shape(&st)?);
                p.push((g(&format!("attn_{kv}_conv.weight")), dims, GgmlType::F32, Recipe::F32(st, Xf::FlipTranspose)));
                let proj = s(&format!("attn.{kv}_sconv.weight_proj"));
                p.push((g(&format!("attn_{kv}_conv_proj.weight")), reversed(&src.shape(&proj)?), GgmlType::Bf16, Recipe::Bf16(proj)));
            }
        }
        f32_id(&mut p, g("attn_post_conv.weight"), s("attn_sconv.weight"))?;
        f32_id(&mut p, g("ffn_post_conv.weight"), s("mlp_sconv.weight"))?;
        f32_id(&mut p, g("ffn_gate_inp.weight"), s("mlp.gate.weight"))?;
        f32_id(&mut p, g("exp_probs_b.bias"), s("mlp.gate.bias"))?;
        if l >= 1 {
            f32_id(&mut p, g("attn_res_query.weight"), format!("{m}attn_res.attn_queries.{}", l - 1))?;
        }
        f32_id(&mut p, g("ffn_res_query.weight"), format!("{m}attn_res.mlp_queries.{l}"))?;

        for (gg, st, split) in [("ffn_gate_up_exps", "mlp.experts.w13_weight", true), ("ffn_down_exps", "mlp.experts.w2_weight", false)] {
            let base = s(st);
            let shape = src.shape(&base)?;
            let [e, n, kb] = shape[..] else { return Err(bad(format!("{base}: shape {shape:?}"))) };
            match src.dtype(&base)?.as_str() {
                "U8" => {
                    p.push((g(&format!("{gg}.weight")), vec![kb * 2, n, e], GgmlType::Nvfp4, Recipe::Nvfp4(base.clone(), split)));
                    p.push((g(&format!("{gg}.scale")), vec![e], GgmlType::F32, Recipe::ExpertScale(base)));
                }
                "F8_E4M3" => p.push((g(&format!("{gg}.weight")), vec![kb, n, e], GgmlType::Q8_0, Recipe::Mxfp8ToQ8(base, split))),
                d => return Err(bad(format!("{base}: expert dtype {d} not supported"))),
            }
        }

        if (core_start..core_end).contains(&l) {
            for (gg, st) in [("q", "attn"), ("k", "k"), ("v", "v"), ("ffn", "mlp")] {
                for w in ["down", "forget", "gain"] {
                    f32_id(&mut p, g(&format!("loop_{gg}_ctrl_{w}.weight")), s(&format!("controllers.{st}.{w}_weight")))?;
                }
            }
            let dim = t["hidden_size"].as_u64().unwrap_or(0);
            p.push((g("loop_msg_query.weight"), vec![dim, 4], GgmlType::F32, Recipe::MsgQuery(l - core_start)));
        }
    }
    Ok(p)
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// FP8 E4M3 (fn): bias 7, no infinities, `0x7f` / `0xff` are NaN.
fn e4m3(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let (e, m) = ((b >> 3) & 0xf, (b & 7) as f32);
    if b & 0x7f == 0x7f {
        return f32::NAN;
    }
    sign * if e == 0 { m / 8.0 * 2f32.powi(-6) } else { (1.0 + m / 8.0) * 2f32.powi(e as i32 - 7) }
}

/// UE8M0 block scale `2^(s - 127)`.
fn e8m0(s: u8) -> f32 {
    match s {
        0 => f32::from_bits(1 << 22),
        255 => f32::NAN,
        s => f32::from_bits((s as u32) << 23),
    }
}

/// Source row for GGUF row `r` of `n`: interleaved gate/up rows become gate half, up half.
fn src_row(r: usize, n: usize, split: bool) -> usize {
    if !split {
        r
    } else if r < n / 2 {
        2 * r
    } else {
        2 * (r - n / 2) + 1
    }
}

fn build(src: &Source, recipe: &Recipe, info: &TensorInfo, dim: usize) -> Result<Vec<u8>> {
    Ok(match recipe {
        Recipe::F32(name, xf) => {
            let v = src.f32(name)?;
            match xf {
                Xf::Id => f32_bytes(&v),
                Xf::Transpose | Xf::FlipTranspose => {
                    let s = squeeze(&src.shape(name)?);
                    let [a, b] = s[..] else { return Err(bad(format!("{name}: cannot transpose shape {s:?}"))) };
                    let (a, b) = (a as usize, b as usize);
                    let mut out = vec![0f32; a * b];
                    for i in 0..a {
                        let si = if *xf == Xf::FlipTranspose { a - 1 - i } else { i };
                        for j in 0..b {
                            out[j * a + i] = v[si * b + j];
                        }
                    }
                    f32_bytes(&out)
                }
            }
        }
        Recipe::Bf16(name) => {
            if src.dtype(name)? != "BF16" {
                return Err(bad(format!("{name}: expected BF16")));
            }
            src.raw(name)?
        }
        Recipe::Q8(name) => astral_quant::quantize_q8_0(&src.f32(name)?)?,
        Recipe::MsgQuery(i) => {
            let q = src.f32("model.llm.middle_cycle_cross_pass_queries")?;
            let kv = src.f32("model.llm.middle_cycle_cross_pass_kv_queries")?;
            let row = |v: &[f32], r: usize| v.get(r * dim..(r + 1) * dim).map(<[f32]>::to_vec).ok_or_else(|| bad(format!("cross-pass queries: no row {r}")));
            let mut out = Vec::with_capacity(4 * dim);
            for part in [row(&q, 2 * i)?, row(&kv, 2 * i)?, row(&kv, 2 * i + 1)?, row(&q, 2 * i + 1)?] {
                out.extend(part);
            }
            f32_bytes(&out)
        }
        Recipe::ExpertScale(base) => {
            let amax = src.f32(&format!("{base}_amax"))?;
            let den = src.f32(&format!("{base}_outer_scale_denominator"))?;
            let d = *den.first().ok_or_else(|| bad(format!("{base}: empty outer scale denominator")))?;
            f32_bytes(&amax.iter().map(|a| a / d).collect::<Vec<_>>())
        }
        Recipe::Nvfp4(base, split) => {
            let (k, n, e) = (info.dims[0] as usize, info.dims[1] as usize, info.dims[2] as usize);
            let w = src.raw(base)?;
            let sc = src.raw(&format!("{base}_scale"))?;
            if w.len() != e * n * k / 2 || sc.len() != e * n * k / 16 {
                return Err(bad(format!("{base}: unexpected NVFP4 sizes")));
            }
            let nb = k / 64;
            let mut out = vec![0u8; e * n * nb * 36];
            let mut vals = vec![0u8; k];
            for ex in 0..e {
                for r in 0..n {
                    let sr = ex * n + src_row(r, n, *split);
                    let qs = &w[sr * k / 2..(sr + 1) * k / 2];
                    let ds = &sc[sr * k / 16..(sr + 1) * k / 16];
                    for (j, &b) in qs.iter().enumerate() {
                        vals[2 * j] = b & 0xf;
                        vals[2 * j + 1] = b >> 4;
                    }
                    let dst = &mut out[(ex * n + r) * nb * 36..(ex * n + r + 1) * nb * 36];
                    for b in 0..nb {
                        let blk = &mut dst[b * 36..(b + 1) * 36];
                        blk[..4].copy_from_slice(&ds[b * 4..b * 4 + 4]);
                        for s in 0..4 {
                            let v = &vals[b * 64 + s * 16..b * 64 + s * 16 + 16];
                            for j in 0..8 {
                                blk[4 + s * 8 + j] = v[j] | (v[j + 8] << 4);
                            }
                        }
                    }
                }
            }
            out
        }
        Recipe::Mxfp8ToQ8(base, split) => {
            let (k, n, e) = (info.dims[0] as usize, info.dims[1] as usize, info.dims[2] as usize);
            let w = src.raw(base)?;
            let sc = src.raw(&format!("{base}_scale"))?;
            if w.len() != e * n * k || sc.len() != e * n * k / 32 {
                return Err(bad(format!("{base}: unexpected MXFP8 sizes")));
            }
            let lut: Vec<f32> = (0..=255u8).map(e4m3).collect();
            let mut row = vec![0f32; k];
            let mut out = Vec::with_capacity(e * n * k / 32 * 34);
            for ex in 0..e {
                for r in 0..n {
                    let sr = ex * n + src_row(r, n, *split);
                    for (j, x) in row.iter_mut().enumerate() {
                        *x = lut[w[sr * k + j] as usize] * e8m0(sc[sr * k / 32 + j / 32]);
                    }
                    out.extend(astral_quant::quantize_q8_0(&row)?);
                }
            }
            out
        }
    })
}

/// Whether `dir` holds an Inkling 2 Mini checkpoint (`config.json` with `inkling2_loop`).
pub fn is_inkling_checkpoint(dir: &Path) -> bool {
    std::fs::read(dir.join("config.json")).ok().and_then(|b| serde_json::from_slice::<Json>(&b).ok()).is_some_and(|c| c["model_type"].as_str() == Some("inkling2_loop"))
}

/// Opens an Inkling 2 Mini safetensors checkpoint directory as a [`Gguf`].
pub fn open(dir: impl AsRef<Path>) -> Result<Gguf> {
    let dir = dir.as_ref();
    let cfg: Json = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?).map_err(|e| bad(format!("config.json: {e}")))?;
    if cfg["model_type"].as_str() != Some("inkling2_loop") {
        return Err(bad(format!("{}: model_type {:?}, expected inkling2_loop", dir.display(), cfg["model_type"])));
    }
    let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("inkling-2-mini").to_string();
    let src = Arc::new(Source::open(dir)?);
    let plan = plan(&src, &cfg)?;
    let dim = cfg["text_config"]["hidden_size"].as_u64().unwrap_or(0) as usize;
    let recipes: HashMap<String, Recipe> = plan.iter().map(|(n, _, _, r)| (n.clone(), r.clone())).collect();
    let tensors = plan.into_iter().map(|(n, d, t, _)| (n, d, t)).collect();
    let read = Box::new(move |info: &TensorInfo| -> Result<Vec<u8>> {
        let r = recipes.get(&info.name).ok_or_else(|| bad(format!("no recipe for {}", info.name)))?;
        build(&src, r, info, dim)
    });
    Gguf::converted(dir.to_path_buf(), metadata(&cfg, &name)?, tensors, read)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_decoders() {
        assert_eq!(e4m3(0x38), 1.0);
        assert_eq!(e4m3(0xb8), -1.0);
        assert_eq!(e4m3(0x7e), 448.0);
        assert_eq!(e4m3(0x01), 2f32.powi(-9));
        assert!(e4m3(0x7f).is_nan());
        assert_eq!(e8m0(127), 1.0);
        assert_eq!(e8m0(130), 8.0);
        assert_eq!(e8m0(0), 2f32.powi(-127));
    }

    #[test]
    fn gate_up_rows_are_split() {
        let n = 8;
        let rows: Vec<usize> = (0..n).map(|r| src_row(r, n, true)).collect();
        assert_eq!(rows, vec![0, 2, 4, 6, 1, 3, 5, 7]);
        assert_eq!(src_row(5, n, false), 5);
    }

    /// Compares every tensor of a checkpoint with a GGUF conversion of it. Run with
    /// `ASTRAL_TEST_CKPT=<dir> ASTRAL_TEST_GGUF=<file> cargo test -p astral-loader -- --ignored`.
    #[test]
    #[ignore]
    fn matches_gguf_conversion() {
        let (Ok(ckpt), Ok(gguf)) = (std::env::var("ASTRAL_TEST_CKPT"), std::env::var("ASTRAL_TEST_GGUF")) else {
            eprintln!("ASTRAL_TEST_CKPT / ASTRAL_TEST_GGUF not set");
            return;
        };
        let a = open(&ckpt).unwrap();
        let b = Gguf::open(&gguf).unwrap();
        let mut names_a: Vec<&str> = a.tensors.iter().map(|t| t.name.as_str()).collect();
        let mut names_b: Vec<&str> = b.tensors.iter().map(|t| t.name.as_str()).collect();
        names_a.sort_unstable();
        names_b.sort_unstable();
        assert_eq!(names_a, names_b, "tensor sets differ");
        let (mut same, mut close, mut differ) = (0, 0, Vec::new());
        for tb in &b.tensors {
            let ta = a.tensor(&tb.name).unwrap();
            assert_eq!((&ta.dims, ta.ty), (&tb.dims, tb.ty), "{}: dims/type", tb.name);
            let (ra, rb) = (a.read_tensor(ta).unwrap(), b.read_tensor(tb).unwrap());
            if ra == rb {
                same += 1;
                continue;
            }
            let n = tb.n_elements as usize;
            let (va, vb) = (astral_quant::dequantize(ta.ty, &ra, n).unwrap(), astral_quant::dequantize(tb.ty, &rb, n).unwrap());
            let scale = vb.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-30);
            let err = va.iter().zip(&vb).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale;
            if err < 1e-6 {
                close += 1;
            } else {
                differ.push(format!("{} rel err {err:.2e}", tb.name));
            }
        }
        eprintln!("{same} tensors byte-identical, {close} equal within 1e-6, {} differ", differ.len());
        assert!(differ.is_empty(), "{differ:#?}");
    }
}
