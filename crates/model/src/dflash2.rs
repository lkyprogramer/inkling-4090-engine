// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! Inkling 2 DFlash2 speculative decoding with `incoai/Inkling-2-Mini-DFlash2` (a directory with
//! `config.json` holding `dflash_config`, and `model.safetensors`).
//!
//! The draft reads the same four target features as DSpark (the residual stream entering layer
//! visits 5, 23 and 41, and the final-norm output), so the verify pass, acceptance, short-conv
//! rollback and round graphs of [`crate::dspark::Spec`] are unchanged; only the context ingest
//! and the draft step differ:
//! - ingest: `c = rms_norm(fc · [f5 | f23 | f41 | f48]) · hidden_norm`, then every layer's own
//!   `k_proj` (with `k_norm` and neox RoPE) and `v_proj` of `c` into a 2048-key sliding window;
//! - draft step: a block of 8 rows (the anchor's normalised target embedding, then 7 rows of the
//!   mask token 200056) through 5 Qwen3-like layers whose attention and MLP are each wrapped in a
//!   two-tap grouped dynamic convolution (input side before, output side after), non-causal
//!   inside the block, each row with its own sliding window; the target LM head on rows 1..=n;
//!   top-16 per row; and the candidate selector walks one path greedily from the anchor
//!   (`score = 30 tanh((logit + <pred[prev] * gate, succ[c]>) / 30)`).
//!
//! Mirrors llama.cpp's `dflash2` graph (PR #33073, b12823), greedy only. Weights are Q8_0 here
//! (BF16 in the checkpoint); the selector codebooks stay bf16.

use crate::dspark::{item, kv_half, mm_small, q8, f32m, KvBuf};
use crate::inkling::{DevWork, Weights as TargetWeights};
use astral_core::{AstralError, Result};
use astral_gpu::{Context, Dev, DeviceBuffer, Stream};
use astral_kernels::dspark::{DevIndex, DsparkOps, HeadDst, DF2_PARTS, DF2_TOPK, HEAD_DIM};
use astral_kernels::inkling::InklingOps;
use astral_kernels::ops::{Gemv, GemvMatrix, QAct};
#[allow(unused_imports)]
use astral_gpu::DevMut;
use astral_loader::safetensors::Safetensors;
use std::path::Path;
use std::sync::Arc;

fn bad(m: String) -> AstralError {
    AstralError::Invalid(m)
}

/// `dflash_config` and the Qwen3 fields of the draft's `config.json`.
#[derive(Debug, Clone)]
pub struct Df2Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub inter: usize,
    pub eps: f32,
    pub theta: f32,
    pub window: u32,
    pub block: usize,
    pub mask_token: u32,
    pub rank: usize,
    pub cap: f32,
    pub taps: Vec<usize>,
}

impl Df2Config {
    /// `None` when `dir/config.json` has no `dflash_config` (a DSpark checkpoint).
    pub fn from_dir(dir: &Path) -> Result<Option<Self>> {
        let text = std::fs::read_to_string(dir.join("config.json"))?;
        let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| bad(format!("config.json: {e}")))?;
        let Some(m) = v.get("dflash_config") else { return Ok(None) };
        let u = |o: &serde_json::Value, k: &str| o.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).ok_or_else(|| bad(format!("DFlash2 config: {k} missing")));
        let f = |o: &serde_json::Value, k: &str| o.get(k).and_then(|x| x.as_f64()).map(|x| x as f32).ok_or_else(|| bad(format!("DFlash2 config: {k} missing")));
        let theta = v.get("rope_parameters").and_then(|r| r.get("rope_theta")).or_else(|| v.get("rope_theta")).and_then(|x| x.as_f64()).unwrap_or(10000.0) as f32;
        let taps: Vec<usize> = m.get("target_layer_ids").and_then(|x| x.as_array()).map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect()).unwrap_or_default();
        let c = Self {
            hidden: u(&v, "hidden_size")?,
            layers: u(&v, "num_hidden_layers")?,
            heads: u(&v, "num_attention_heads")?,
            kv_heads: u(&v, "num_key_value_heads")?,
            inter: u(&v, "intermediate_size")?,
            eps: f(&v, "rms_norm_eps")?,
            theta,
            window: u(&v, "sliding_window")? as u32,
            block: u(m, "block_size")?,
            mask_token: u(m, "mask_token_id")? as u32,
            rank: u(m, "selector_rank")?,
            cap: f(m, "final_logit_softcapping").unwrap_or(0.0),
            taps,
        };
        let ok = u(&v, "head_dim")? == HEAD_DIM as usize
            && c.taps.len() == 4
            && u(m, "conv_kernel_size")? == 2
            && u(m, "conv_group_size")? == 16
            && u(m, "selector_top_k")? == DF2_TOPK as usize
            && m.get("selector_softcap_mode").and_then(|x| x.as_str()) == Some("after_correction")
            && c.hidden.is_multiple_of(64)
            && c.block >= 2;
        if !ok {
            return Err(bad(format!("unsupported DFlash2 config {m}")));
        }
        Ok(Some(c))
    }
}

struct Layer {
    wqkv: GemvMatrix,
    wo: GemvMatrix,
    wgu: GemvMatrix,
    wdown: GemvMatrix,
    /// `[k_proj; v_proj]` again, for the context rows.
    wkv: GemvMatrix,
    aproj: GemvMatrix,
    mproj: GemvMatrix,
    abase: DeviceBuffer<f32>,
    mbase: DeviceBuffer<f32>,
    in_ln: DeviceBuffer<f32>,
    post_ln: DeviceBuffer<f32>,
    q_norm: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
}

/// The DFlash2 draft: weights, its sliding-window K/V and the block buffers.
pub struct Df2 {
    pub cfg: Df2Config,
    layers: Vec<Layer>,
    fc: GemvMatrix,
    hidden_norm: DeviceBuffer<f32>,
    norm: DeviceBuffer<f32>,
    hproj: GemvMatrix,
    pred: DeviceBuffer<u16>,
    succ: DeviceBuffer<u16>,
    vocab: u32,
    pub bytes: usize,
    /// The 7 mask rows' input (normalised target embedding of the mask token).
    pub mask_row: Vec<f32>,
    kc: Vec<KvBuf>,
    vc: Vec<KvBuf>,
    cat: DeviceBuffer<f32>,
    qa_cat: QAct,
    ctx: DeviceBuffer<f32>,
    qa_ctx: QAct,
    kv: DeviceBuffer<f32>,
    xin: DeviceBuffer<f32>,
    emb: DeviceBuffer<f32>,
    res: DeviceBuffer<f32>,
    hn: DeviceBuffer<f32>,
    cbuf: DeviceBuffer<f32>,
    tmp: DeviceBuffer<f32>,
    mlp: DeviceBuffer<f32>,
    hid: DeviceBuffer<f32>,
    qkv: DeviceBuffer<f32>,
    att: DeviceBuffer<f32>,
    gu: DeviceBuffer<f32>,
    dyn_a: DeviceBuffer<f32>,
    dyn_m: DeviceBuffer<f32>,
    qa_d: QAct,
    qa_o: QAct,
    qa_f: QAct,
    qa_h: QAct,
    logits: DeviceBuffer<f32>,
    part_v: DeviceBuffer<f32>,
    part_i: DeviceBuffer<u32>,
    cand_v: DeviceBuffer<f32>,
    cand_i: DeviceBuffer<i32>,
    gate: DeviceBuffer<f32>,
    /// The selector's softmax probability of each drafted token (for a round planner).
    pub chance: DeviceBuffer<f32>,
    zero_ids: DeviceBuffer<i32>,
    items_blk: DeviceBuffer<u32>,
    items_head: DeviceBuffer<u32>,
    items_ctx: DeviceBuffer<u32>,
    capacity: u32,
}

fn bf16_bits(x: &[f32], rows: usize, cols: usize) -> Vec<u16> {
    // The checkpoint is bf16, so the upper half of each f32 is the exact value.
    x[..rows * cols].iter().map(|v| (v.to_bits() >> 16) as u16).collect()
}

impl Df2 {
    /// Loads the draft from `dir` for a target with `tmax`-token passes verifying `k` drafts per
    /// round; `capacity` is the draft cache capacity of [`crate::dspark::Spec`].
    #[allow(clippy::too_many_arguments)]
    pub fn load(ctx: &Arc<Context>, dir: &Path, cfg: Df2Config, tw: &TargetWeights, iops: &InklingOps, ops: &DsparkOps, k: usize, tmax: usize, capacity: u32) -> Result<Self> {
        let st = Safetensors::open(dir.join("model.safetensors"))?;
        let get = |name: &str| -> Result<(Vec<f32>, Vec<u64>)> {
            let t = st.tensor(name).ok_or_else(|| bad(format!("DFlash2 model.safetensors: missing {name}")))?;
            Ok((st.read_f32(t)?, t.shape.clone()))
        };
        let vecd = |name: &str| -> Result<DeviceBuffer<f32>> { DeviceBuffer::from_slice(ctx, &get(name)?.0) };
        let (d, nb, r) = (cfg.hidden, cfg.block, cfg.rank);
        let (qw, kvw) = (cfg.heads * HEAD_DIM as usize, cfg.kv_heads * HEAD_DIM as usize);
        let groups = d / 16;
        let mut bytes = 0usize;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = |s: &str| format!("layers.{i}.{s}");
            let kp = get(&p("self_attn.k_proj.weight"))?.0;
            let vp = get(&p("self_attn.v_proj.weight"))?.0;
            let mut qkv = get(&p("self_attn.q_proj.weight"))?.0;
            qkv.extend_from_slice(&kp);
            qkv.extend_from_slice(&vp);
            let mut kv = kp;
            kv.extend(vp);
            let mut gu = get(&p("mlp.gate_proj.weight"))?.0;
            gu.extend(get(&p("mlp.up_proj.weight"))?.0);
            let (ab, abs) = get(&p("attention_conv.base_kernel"))?;
            if abs != [2, 2, d as u64] {
                return Err(bad(format!("DFlash2 conv base shape {abs:?}")));
            }
            let l = Layer {
                wqkv: q8(ctx, &qkv, qw + 2 * kvw, d)?,
                wo: q8(ctx, &get(&p("self_attn.o_proj.weight"))?.0, d, qw)?,
                wgu: q8(ctx, &gu, 2 * cfg.inter, d)?,
                wdown: q8(ctx, &get(&p("mlp.down_proj.weight"))?.0, d, cfg.inter)?,
                wkv: q8(ctx, &kv, 2 * kvw, d)?,
                aproj: q8(ctx, &get(&p("attention_conv.kernel_projection.weight"))?.0, 4 * groups, d)?,
                mproj: q8(ctx, &get(&p("mlp_conv.kernel_projection.weight"))?.0, 4 * groups, d)?,
                abase: DeviceBuffer::from_slice(ctx, &ab)?,
                mbase: vecd(&p("mlp_conv.base_kernel"))?,
                in_ln: vecd(&p("input_layernorm.weight"))?,
                post_ln: vecd(&p("post_attention_layernorm.weight"))?,
                q_norm: vecd(&p("self_attn.q_norm.weight"))?,
                k_norm: vecd(&p("self_attn.k_norm.weight"))?,
            };
            bytes += [&l.wqkv, &l.wo, &l.wgu, &l.wdown, &l.wkv, &l.aproj, &l.mproj].iter().map(|m| m.bytes()).sum::<usize>();
            layers.push(l);
        }
        let (fcw, fcs) = get("fc.weight")?;
        if fcs != [d as u64, 4 * d as u64] {
            return Err(bad(format!("DFlash2 fc shape {fcs:?}")));
        }
        let fc = q8(ctx, &fcw, d, 4 * d)?;
        let hproj = f32m(ctx, &get("candidate_selector.hidden_projection.weight")?.0, r, d)?;
        let vocab = (tw.cfg.vocab as u32).min(tw.output.n);
        let (pc, ps) = get("candidate_selector.predecessor_codebook")?;
        let (sc, ss) = get("candidate_selector.successor_codebook")?;
        if ps.len() != 2 || ps[1] != r as u64 || ss != ps || ps[0] < vocab as u64 {
            return Err(bad(format!("DFlash2 codebook shape {ps:?} / {ss:?}")));
        }
        let pred = DeviceBuffer::from_slice(ctx, &bf16_bits(&pc, vocab as usize, r))?;
        let succ = DeviceBuffer::from_slice(ctx, &bf16_bits(&sc, vocab as usize, r))?;
        bytes += fc.bytes() + hproj.bytes() + 2 * pred.bytes();

        // The mask rows: the target's embedding of the mask token, normalised by its embed norm.
        let mask_row = {
            let s = Stream::new(ctx)?;
            let tok = DeviceBuffer::from_slice(ctx, &[cfg.mask_token as i32])?;
            let mut e = DeviceBuffer::<f32>::zeroed(ctx, d)?;
            let mut y = DeviceBuffer::<f32>::zeroed(ctx, d)?;
            iops.embed(&s, &tw.tok_embd, &tok, &mut e, 1)?;
            ops.add_rms_q8(&s, e.dptr(), d, None, None, &tw.tok_norm, Some(&mut y), None, 1, d as u32, cfg.eps)?;
            s.synchronize()?;
            y.to_vec()?
        };
        let mut xin = vec![0f32; nb * d];
        for row in 1..nb {
            xin[row * d..(row + 1) * d].copy_from_slice(&mask_row);
        }
        // Sliding-window K/V: a ring that holds the widest row window (W + block keys) and one
        // whole context write (tmax rows), unless capacity is smaller.
        let ring = (cfg.window as usize + nb).max(tmax);
        let (rows, is_ring) = if ring < capacity as usize { (ring, true) } else { (capacity as usize, false) };
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(ctx, n);
        let vp = tw.output.n as usize;
        let vmax = k + 1;
        let nd = nb - 1;
        Ok(Self {
            kc: (0..cfg.layers).map(|_| KvBuf::new(ctx, (rows, is_ring), kvw, kv_half())).collect::<Result<_>>()?,
            vc: (0..cfg.layers).map(|_| KvBuf::new(ctx, (rows, is_ring), kvw, kv_half())).collect::<Result<_>>()?,
            cat: z(tmax * 4 * d)?,
            qa_cat: QAct::with_f16(ctx, 4 * d as u32, tmax as u32)?,
            ctx: z(tmax * d)?,
            qa_ctx: QAct::with_f16(ctx, d as u32, tmax as u32)?,
            kv: z(tmax * 2 * kvw)?,
            xin: DeviceBuffer::from_slice(ctx, &xin)?,
            emb: z(d)?,
            res: z(nb * d)?,
            hn: z(nb * d)?,
            cbuf: z(nb * d)?,
            tmp: z(nb * d)?,
            mlp: z(nb * d)?,
            hid: z(nb * d)?,
            qkv: z(nb * (qw + 2 * kvw))?,
            att: z(nb * qw)?,
            gu: z(nb * 2 * cfg.inter)?,
            dyn_a: z(nb * 4 * groups)?,
            dyn_m: z(nb * 4 * groups)?,
            qa_d: QAct::with_f16(ctx, d as u32, nb as u32)?,
            qa_o: QAct::with_f16(ctx, qw as u32, nb as u32)?,
            qa_f: QAct::with_f16(ctx, cfg.inter as u32, nb as u32)?,
            qa_h: QAct::new(ctx, d as u32, nb as u32)?,
            logits: z(nd * vp)?,
            part_v: z(nd * (DF2_PARTS * DF2_TOPK) as usize)?,
            part_i: DeviceBuffer::zeroed(ctx, nd * (DF2_PARTS * DF2_TOPK) as usize)?,
            cand_v: z(nd * DF2_TOPK as usize)?,
            cand_i: DeviceBuffer::zeroed(ctx, nd * DF2_TOPK as usize)?,
            gate: z(nd * r)?,
            chance: z(nd)?,
            zero_ids: DeviceBuffer::zeroed(ctx, nb)?,
            items_blk: item(ctx, nb, 0)?,
            items_head: item(ctx, k, 0)?,
            items_ctx: item(ctx, vmax, 0)?,
            capacity,
            layers,
            fc,
            hidden_norm: vecd("hidden_norm.weight")?,
            norm: vecd("norm.weight")?,
            hproj,
            pred,
            succ,
            vocab,
            bytes,
            mask_row,
            cfg,
        })
    }

    /// Context K/V for `rows` target tokens (feature rows `0..rows` of `feats`, `[4][tmax][d]`) at
    /// positions `pos..`; `verify` selects the fixed small-row path, else `work_pf` (uploaded by the
    /// caller for `rows`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn store_ctx(&mut self, st: &Stream, gemv: &Gemv, ops: &DsparkOps, feats: &DeviceBuffer<f32>, tmax: usize, rows: usize, pos: DevIndex, verify: bool, work_pf: &DevWork) -> Result<()> {
        let c = &self.cfg;
        let (d, kvh) = (c.hidden as u32, c.kv_heads as u32);
        let kvw = kvh * HEAD_DIM;
        let ru = rows as u32;
        ops.df2_gather4(st, feats, tmax * c.hidden, &mut self.cat, ru, d)?;
        ops.quant(st, &self.cat, 4 * d, ru, &mut self.qa_cat)?;
        let wk = work_pf;
        if verify {
            mm_small(gemv, st, &self.fc, &self.qa_cat, ru, &mut self.ctx, &self.items_ctx)?;
        } else {
            gemv.launch_mma(st, &self.fc, &self.qa_cat, ru, &mut self.ctx, ru, &wk.items, wk.n_items, &wk.xidx, &wk.yidx, wk.idx_len)?;
        }
        ops.add_rms_q8(st, self.ctx.dptr(), self.ctx.len(), None, None, &self.hidden_norm, None, Some(&mut self.qa_ctx), ru, d, c.eps)?;
        for l in 0..c.layers {
            let lw = &self.layers[l];
            if verify {
                mm_small(gemv, st, &lw.wkv, &self.qa_ctx, ru, &mut self.kv, &self.items_ctx)?;
            } else {
                gemv.launch_mma(st, &lw.wkv, &self.qa_ctx, ru, &mut self.kv, ru, &wk.items, wk.n_items, &wk.xidx, &wk.yidx, wk.idx_len)?;
            }
            let kp = self.kv.dptr();
            ops.head_prep(st, kp, 2 * kvw, HeadDst::Cache(self.kc[l].r()), Some(&lw.k_norm), true, pos, ru, kvh, c.theta, c.eps)?;
            ops.head_prep(st, kp + 4 * kvw as u64, 2 * kvw, HeadDst::Cache(self.vc[l].r()), None, false, pos, ru, kvh, c.theta, c.eps)?;
        }
        Ok(())
    }

    /// One draft step: the block for anchor `token[0]` at position `info[2]`; writes `n` drafts to
    /// `token[1..=n]` (and their selector chances).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draft(&mut self, st: &Stream, gemv: &Gemv, ops: &DsparkOps, iops: &InklingOps, tw: &TargetWeights, token: &DeviceBuffer<i32>, info: &DeviceBuffer<i32>, n: usize) -> Result<()> {
        let c = &self.cfg;
        let (d, nb) = (c.hidden, c.block);
        let n = n.clamp(1, nb - 1);
        let (du, nbu) = (d as u32, nb as u32);
        let (heads, kvh) = (c.heads as u32, c.kv_heads as u32);
        let (qw, kvw) = (heads * HEAD_DIM, kvh * HEAD_DIM);
        let qkvw = qw + 2 * kvw;
        let start = DevIndex::of(info, 2)?;
        let blk = &self.items_blk;
        iops.embed(st, &tw.tok_embd, token, &mut self.emb, 1)?;
        ops.add_rms_q8(st, self.emb.dptr(), d, None, None, &tw.tok_norm, Some(&mut self.xin.view_mut(0, d)?), None, 1, du, c.eps)?;
        for l in 0..c.layers {
            let lw = &self.layers[l];
            let (x, add) = if l == 0 { (self.xin.dptr(), None) } else { (self.res.dptr(), Some(&self.cbuf as &dyn Dev<f32>)) };
            ops.add_rms_q8(st, x, nb * d, add, Some(self.res.dptr()), &lw.in_ln, Some(&mut self.hn), Some(&mut self.qa_d), nbu, du, c.eps)?;
            mm_small(gemv, st, &lw.aproj, &self.qa_d, nbu, &mut self.dyn_a, blk)?;
            ops.df2_conv(st, &self.hn, &self.dyn_a, &lw.abase, 0, None, Some(&mut self.qa_d), nbu, du)?;
            mm_small(gemv, st, &lw.wqkv, &self.qa_d, nbu, &mut self.qkv, blk)?;
            let qp = self.qkv.dptr();
            ops.head_prep(st, qp, qkvw, HeadDst::Rows { buf: &mut self.qkv, stride: qkvw }, Some(&lw.q_norm), true, start, nbu, heads, c.theta, c.eps)?;
            ops.head_prep(st, qp + 4 * qw as u64, qkvw, HeadDst::Cache(self.kc[l].r()), Some(&lw.k_norm), true, start, nbu, kvh, c.theta, c.eps)?;
            ops.head_prep(st, qp + 4 * (qw + kvw) as u64, qkvw, HeadDst::Cache(self.vc[l].r()), None, false, start, nbu, kvh, c.theta, c.eps)?;
            ops.attn(st, qp, qkvw, self.kc[l].r(), self.vc[l].r(), &mut self.att, start, nbu, heads, kvh, c.window + nbu - 1, self.capacity, true)?;
            ops.quant(st, &self.att, qw, nbu, &mut self.qa_o)?;
            mm_small(gemv, st, &lw.wo, &self.qa_o, nbu, &mut self.tmp, blk)?;
            ops.df2_conv(st, &self.tmp, &self.dyn_a, &lw.abase, 1, Some(&mut self.cbuf), None, nbu, du)?;
            ops.add_rms_q8(st, self.res.dptr(), nb * d, Some(&self.cbuf), Some(self.res.dptr()), &lw.post_ln, Some(&mut self.hn), Some(&mut self.qa_d), nbu, du, c.eps)?;
            mm_small(gemv, st, &lw.mproj, &self.qa_d, nbu, &mut self.dyn_m, blk)?;
            ops.df2_conv(st, &self.hn, &self.dyn_m, &lw.mbase, 0, None, Some(&mut self.qa_d), nbu, du)?;
            mm_small(gemv, st, &lw.wgu, &self.qa_d, nbu, &mut self.gu, blk)?;
            ops.swiglu_q8(st, &self.gu, nbu, &mut self.qa_f)?;
            mm_small(gemv, st, &lw.wdown, &self.qa_f, nbu, &mut self.mlp, blk)?;
            ops.df2_conv(st, &self.mlp, &self.dyn_m, &lw.mbase, 1, Some(&mut self.cbuf), None, nbu, du)?;
        }
        ops.add_rms_q8(st, self.res.dptr(), nb * d, Some(&self.cbuf), Some(self.res.dptr()), &self.norm, Some(&mut self.hid), None, nbu, du, c.eps)?;
        // Head, top-16 and selector on the mask rows 1..=n.
        let kr = n as u32;
        let vp = tw.output.n;
        ops.quant(st, &self.hid.view(d, n * d)?, du, kr, &mut self.qa_h)?;
        mm_small(gemv, st, &tw.output, &self.qa_h, kr, &mut self.logits, &self.items_head)?;
        ops.df2_topk(st, &self.logits, vp, self.vocab, kr, &mut self.part_v, &mut self.part_i, &mut self.cand_v, &mut self.cand_i)?;
        gemv.launch_ids(st, &self.hproj, &self.hid.view(d, n * d)?, du, &mut self.gate, &self.zero_ids.view(0, n)?)?;
        ops.df2_select(st, &self.cand_i, &self.cand_v, &self.gate, &self.pred, &self.succ, c.rank as u32, self.vocab, token, kr, c.cap, &mut self.chance)
    }
}
