// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! Inkling 2 Mini on the GPU (spec 06), P1 correctness-first decode: one token per step.
//!
//! The forward pass follows llama.cpp's `inkling2` / `inkling2_loop` graphs (MIT) step for step:
//! 24 stored layers run as 48 cache slots (layers 6..18 three times, each pass restarting from
//! the prelude output and steered by loop "messages"), with per-slot KV and short-conv state.

use crate::dspark::{hist_offsets, DraftWeights, Spec, SpecStats};
use astral_core::{AstralError, Result};
use astral_gpu::{Context, Dev, DevMut, DeviceBuffer, GraphExec, Stream};
use astral_kernels::dspark::{DevIndex, DsparkOps};
use astral_kernels::inkling::{AttnPrep, AttnShape, InklingOps};
use astral_kernels::ops::{Gemv, GemvMatrix, GemvType, MmaWork, MultiItem, QAct, QItem, RmsNormF32, QITEM_ROWS};
use astral_kernels::KernelRegistry;
use astral_loader::{Gguf, TensorInfo};
use astral_quant::GgmlType;
use std::sync::Arc;

mod probe;

fn bad(m: String) -> AstralError {
    AstralError::Invalid(format!("inkling: {m}"))
}

/// Model hyper-parameters, read from the GGUF.
#[derive(Debug, Clone)]
pub struct Config {
    pub n_layer: usize,
    pub dim: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub vocab_padded: usize,
    pub vocab: usize,
    pub core_start: usize,
    pub core_end: usize,
    pub passes: usize,
    pub swa: Vec<bool>,
    pub n_swa: u32,
    pub n_rel: u32,
    pub rel_extent: u32,
    pub rms_eps: f32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_floor: f32,
    pub log_alpha: f32,
    pub situ_beta: f32,
    pub situ_lb: f32,
    pub expert_scale: f32,
    pub softcap: f32,
    pub attn_res_block: usize,
    /// `loop.varying` (varloop checkpoints): trained on several pass counts.
    pub varying: bool,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        if g.architecture() != Some("inkling2_loop") {
            return Err(bad(format!("architecture {:?}, expected inkling2_loop", g.architecture())));
        }
        let u = |k: &str| g.arch_key(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| bad(format!("missing key {k}")));
        let f = |k: &str| g.arch_key(k).and_then(|v| v.as_f64()).map(|v| v as f32).ok_or_else(|| bad(format!("missing key {k}")));
        let first = |k: &str| -> Result<usize> {
            let v = g.arch_key(k).ok_or_else(|| bad(format!("missing key {k}")))?;
            v.as_u64().or_else(|| v.as_array().and_then(|a| a.first()).and_then(|x| x.as_u64())).map(|x| x as usize).ok_or_else(|| bad(format!("bad key {k}")))
        };
        let swa: Vec<bool> = g
            .arch_key("attention.sliding_window_pattern")
            .and_then(|v| v.as_array())
            .ok_or_else(|| bad("missing sliding_window_pattern".into()))?
            .iter()
            .map(|v| v.as_bool().unwrap_or(false))
            .collect();
        let c = Self {
            n_layer: u("block_count")?,
            dim: u("embedding_length")?,
            heads: first("attention.head_count")?,
            kv_heads: first("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            n_ff_exp: u("expert_feed_forward_length")?,
            vocab_padded: u("vocab_size")?,
            vocab: u("vocab_size_unpadded")?,
            core_start: u("loop.start")?,
            core_end: u("loop.end")?,
            passes: u("loop.count")?,
            swa,
            n_swa: u("attention.sliding_window")? as u32,
            n_rel: u("attention.relative_length")? as u32,
            rel_extent: u("attention.relative_extent")? as u32,
            rms_eps: f("attention.layer_norm_rms_epsilon")?,
            qk_clip: f("attention.qk_clip")?,
            rel_clip: f("attention.relative_clip")?,
            log_floor: f("attention.log_scale_floor")?,
            log_alpha: f("attention.log_scale_alpha")?,
            situ_beta: f("activation.situ_beta")?,
            situ_lb: f("activation.situ_linear_beta")?,
            expert_scale: g.arch_key("expert_weights_scale").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
            softcap: g.arch_key("final_logit_softcapping").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            attn_res_block: u("attn_res.block_size")?,
            varying: g.arch_key("loop.varying").and_then(|v| v.as_bool()).unwrap_or(false),
        };
        let mut c = c;
        // Pass-count override (research): ASTRAL_PASSES=R rebuilds the slots, KV, conv state and
        // the attention-residual bank for R passes. Only varloop checkpoints (loop.varying) were
        // trained on more than one count (1-5).
        if let Some(r) = std::env::var("ASTRAL_PASSES").ok().filter(|v| !v.is_empty()) {
            let r: usize = r.parse().map_err(|_| bad(format!("ASTRAL_PASSES={r:?} is not a pass count")))?;
            if !(1..=MAX_PASSES).contains(&r) {
                return Err(bad(format!("ASTRAL_PASSES={r} outside 1..={MAX_PASSES}")));
            }
            if r != c.passes {
                eprintln!("inkling: {r} passes instead of {} ({} loop)", c.passes, if c.varying { "variable" } else { "fixed" });
            }
            c.passes = r;
        }
        if c.swa.len() != c.n_layer || c.core_end > c.n_layer || c.core_start >= c.core_end || c.passes < 1 {
            return Err(bad("inconsistent layer / loop configuration".into()));
        }
        if c.n_swa != c.rel_extent {
            return Err(bad("P1 assumes the SWA window equals the relative extent".into()));
        }
        Ok(c)
    }

    /// Cache slots: n_layer + (passes - 1) * core length.
    pub fn n_slots(&self) -> usize {
        self.n_layer + (self.passes - 1) * (self.core_end - self.core_start)
    }

    fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// Physical layer of cache slot `slot` (slots past `n_layer` are the later core passes).
    pub fn slot_layer(&self, slot: usize) -> usize {
        if slot < self.n_layer {
            slot
        } else {
            self.core_start + (slot - self.n_layer) % (self.core_end - self.core_start)
        }
    }

    /// KV-cache rows of `slot` for `capacity` positions and passes of up to `tmax` tokens.
    /// Sliding-window layers keep a ring of window + max(tmax, QITEM_ROWS) rows (rounded up
    /// to 64): a pass writes all its keys before attending, so the ring must hold the earliest
    /// query's window plus the whole pass. Rejected speculative rows only overwrite positions
    /// no later query reads. `ASTRAL_NO_SWA_RING=1` keeps every position.
    pub fn kv_rows(&self, slot: usize, capacity: u32, tmax: usize) -> u32 {
        if !self.swa[self.slot_layer(slot)] || std::env::var_os("ASTRAL_NO_SWA_RING").is_some() {
            return capacity;
        }
        let ring = (self.n_swa as usize + tmax.max(QITEM_ROWS)).next_multiple_of(64);
        capacity.min(ring as u32)
    }

    /// Attention-residual bank rows: one per block start over all visits, plus headroom.
    pub(crate) fn bank_rows(&self) -> usize {
        let b = self.attn_res_block.max(1);
        self.n_slots().div_ceil(b) + 2
    }

    /// Short-conv channels: K, V, attention output, FFN output.
    fn conv_channels(&self) -> usize {
        2 * self.kv_width() + 2 * self.dim
    }
}

struct Layer {
    swa: bool,
    wq: GemvMatrix,
    wk: GemvMatrix,
    wv: GemvMatrix,
    wo: GemvMatrix,
    wgate: GemvMatrix,
    wrel: GemvMatrix,
    sink: GemvMatrix,
    k_conv_proj: Option<GemvMatrix>,
    v_conv_proj: Option<GemvMatrix>,
    /// Q8_0 copies of the three above, fused into the decode / small-batch projection launch.
    sink_q8: GemvMatrix,
    kp_q8: Option<GemvMatrix>,
    vp_q8: Option<GemvMatrix>,
    q_clip: DeviceBuffer<f32>,
    rel_bias: DeviceBuffer<f32>,
    rel_clip: DeviceBuffer<f32>,
    k_conv: DeviceBuffer<f32>,
    v_conv: DeviceBuffer<f32>,
    attn_post_conv: DeviceBuffer<f32>,
    ffn_post_conv: DeviceBuffer<f32>,
    attn_norm: Option<DeviceBuffer<f32>>,
    qkv_inp_norm: Option<[DeviceBuffer<f32>; 3]>,
    attn_res_query: Option<DeviceBuffer<f32>>,
    ffn_res_query: DeviceBuffer<f32>,
    ffn_norm: DeviceBuffer<f32>,
    router: GemvMatrix,
    probs_b: Option<DeviceBuffer<f32>>,
    gate_up: GemvMatrix,
    down: GemvMatrix,
}

struct LoopLayer {
    msg_query: DeviceBuffer<f32>,
    down: [GemvMatrix; 4],
    /// Gain rows then forget rows (`[2 * dim, rank]`): one GEMV gives both.
    gain_forget: [GemvMatrix; 4],
}

/// All weights of the model on the GPU.
pub struct Weights {
    pub cfg: Config,
    pub(crate) tok_embd: GemvMatrix,
    pub(crate) tok_norm: DeviceBuffer<f32>,
    pub(crate) output: GemvMatrix,
    output_norm: DeviceBuffer<f32>,
    output_res_query: DeviceBuffer<f32>,
    layers: Vec<Layer>,
    loops: Vec<LoopLayer>,
    pub bytes: usize,
}

struct Loader<'a> {
    g: &'a Gguf,
    ctx: &'a Arc<Context>,
    bytes: usize,
}

impl Loader<'_> {
    fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.g.tensor(name).ok_or_else(|| bad(format!("missing tensor {name}")))
    }

    /// A weight matrix (`[k, n]` or `[k, n, mats]` in GGML order) in GEMV layout.
    fn mat(&mut self, name: &str) -> Result<GemvMatrix> {
        let t = self.info(name)?.clone();
        let (k, n, mats) = match t.dims.as_slice() {
            [k, n] => (*k, *n, 1),
            [k, n, m] => (*k, *n, *m),
            d => return Err(bad(format!("{name}: unexpected dims {d:?}"))),
        };
        let raw = self.g.read_tensor(&t)?;
        let (ty, qs, aux) = match t.ty {
            GgmlType::F32 => (GemvType::F32, raw, Vec::new()),
            GgmlType::Bf16 => (GemvType::Bf16, raw, Vec::new()),
            GgmlType::Q8_0 => {
                let (q, a) = astral_quant::repack_q8_0(&raw)?;
                (GemvType::Q8_0, q, a)
            }
            GgmlType::Nvfp4 => {
                let (q, a) = astral_quant::repack_nvfp4(&raw)?;
                (GemvType::Nvfp4, q, a)
            }
            other => return Err(bad(format!("{name}: type {} not supported by the P1 GEMV", other.name()))),
        };
        let scale_name = name.strip_suffix(".weight").map(|b| format!("{b}.scale"));
        let scale = match scale_name.as_deref().and_then(|s| self.g.tensor(s)) {
            Some(st) => Some(astral_quant::dequantize(st.ty, &self.g.read_tensor(st)?, st.n_elements as usize)?),
            None => None,
        };
        let m = GemvMatrix::upload(self.ctx, ty, n as u32, k as u32, mats as u32, &qs, &aux, scale.as_deref())?;
        self.bytes += m.bytes();
        Ok(m)
    }

    /// A `[k, n]` matrix of any dequantisable type, requantised to Q8_0 (fused decode GEMV).
    fn mat_q8(&mut self, name: &str) -> Result<GemvMatrix> {
        let t = self.info(name)?.clone();
        let [k, n] = t.dims.as_slice() else { return Err(bad(format!("{name}: expected a matrix"))) };
        let v = astral_quant::dequantize(t.ty, &self.g.read_tensor(&t)?, t.n_elements as usize)?;
        let (qs, aux) = astral_quant::quantize_q8_0_rows(&v, *n as usize, *k as usize)?;
        let m = GemvMatrix::upload(self.ctx, GemvType::Q8_0, *n as u32, *k as u32, 1, &qs, &aux, None)?;
        self.bytes += m.bytes();
        Ok(m)
    }

    /// Matrices with the same `k`, stacked by rows into one F32 matrix.
    fn stacked_f32(&mut self, names: &[&str]) -> Result<GemvMatrix> {
        let (mut rows, mut k0, mut data) = (0usize, None, Vec::new());
        for name in names {
            let t = self.info(name)?.clone();
            let [k, n] = t.dims.as_slice() else { return Err(bad(format!("{name}: expected a matrix"))) };
            if k0.is_some_and(|k0| k0 != *k) {
                return Err(bad(format!("{name}: k {k} differs from the stack")));
            }
            k0 = Some(*k);
            rows += *n as usize;
            let v = astral_quant::dequantize(t.ty, &self.g.read_tensor(&t)?, t.n_elements as usize)?;
            data.extend(v.iter().flat_map(|x| x.to_le_bytes()));
        }
        let k = k0.ok_or_else(|| bad("empty stack".into()))?;
        let m = GemvMatrix::upload(self.ctx, GemvType::F32, rows as u32, k as u32, 1, &data, &[], None)?;
        self.bytes += m.bytes();
        Ok(m)
    }

    /// A small tensor as f32 on the GPU.
    fn vec(&mut self, name: &str) -> Result<DeviceBuffer<f32>> {
        let t = self.info(name)?.clone();
        let v = astral_quant::dequantize(t.ty, &self.g.read_tensor(&t)?, t.n_elements as usize)?;
        self.bytes += v.len() * 4;
        DeviceBuffer::from_slice(self.ctx, &v)
    }

    fn opt_vec(&mut self, name: &str) -> Result<Option<DeviceBuffer<f32>>> {
        if self.g.tensor(name).is_some() {
            self.vec(name).map(Some)
        } else {
            Ok(None)
        }
    }
}

impl Weights {
    pub fn load(ctx: &Arc<Context>, g: &Gguf) -> Result<Self> {
        let cfg = Config::from_gguf(g)?;
        let mut l = Loader { g, ctx, bytes: 0 };
        let tok_embd = l.mat("token_embd.weight")?;
        let tok_norm = l.vec("token_embd_norm.weight")?;
        let output = l.mat("output.weight")?;
        let output_norm = l.vec("output_norm.weight")?;
        let output_res_query = l.vec("output_res_query.weight")?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let b = |s: &str| format!("blk.{i}.{s}");
            let swa = cfg.swa[i];
            layers.push(Layer {
                swa,
                wq: l.mat(&b("attn_q.weight"))?,
                wk: l.mat(&b("attn_k.weight"))?,
                wv: l.mat(&b("attn_v.weight"))?,
                wo: l.mat(&b("attn_output.weight"))?,
                wgate: l.mat(&b("attn_gate.weight"))?,
                wrel: l.mat(&b("attn_rel.weight"))?,
                sink: l.mat(&b("attn_sink_proj.weight"))?,
                k_conv_proj: if swa { None } else { Some(l.mat(&b("attn_k_conv_proj.weight"))?) },
                v_conv_proj: if swa { None } else { Some(l.mat(&b("attn_v_conv_proj.weight"))?) },
                sink_q8: l.mat_q8(&b("attn_sink_proj.weight"))?,
                kp_q8: if swa { None } else { Some(l.mat_q8(&b("attn_k_conv_proj.weight"))?) },
                vp_q8: if swa { None } else { Some(l.mat_q8(&b("attn_v_conv_proj.weight"))?) },
                q_clip: l.vec(&b("attn_q_clip.weight"))?,
                rel_bias: l.vec(&b("attn_rel_bias.weight"))?,
                rel_clip: l.vec(&b("attn_rel_clip.weight"))?,
                k_conv: l.vec(&b("attn_k_conv.weight"))?,
                v_conv: l.vec(&b("attn_v_conv.weight"))?,
                attn_post_conv: l.vec(&b("attn_post_conv.weight"))?,
                ffn_post_conv: l.vec(&b("ffn_post_conv.weight"))?,
                attn_norm: if swa { Some(l.vec(&b("attn_norm.weight"))?) } else { None },
                qkv_inp_norm: if swa { None } else { Some([l.vec(&b("attn_q_inp_norm.weight"))?, l.vec(&b("attn_k_inp_norm.weight"))?, l.vec(&b("attn_v_inp_norm.weight"))?]) },
                attn_res_query: l.opt_vec(&b("attn_res_query.weight"))?,
                ffn_res_query: l.vec(&b("ffn_res_query.weight"))?,
                ffn_norm: l.vec(&b("ffn_norm.weight"))?,
                router: l.mat(&b("ffn_gate_inp.weight"))?,
                probs_b: l.opt_vec(&b("exp_probs_b.bias"))?,
                gate_up: l.mat(&b("ffn_gate_up_exps.weight"))?,
                down: l.mat(&b("ffn_down_exps.weight"))?,
            });
        }
        let mut loops = Vec::new();
        for i in cfg.core_start..cfg.core_end {
            let b = |s: &str| format!("blk.{i}.{s}");
            let mut m = |kind: &str| -> Result<[GemvMatrix; 4]> {
                Ok([
                    l.mat(&b(&format!("loop_q_ctrl_{kind}.weight")))?,
                    l.mat(&b(&format!("loop_k_ctrl_{kind}.weight")))?,
                    l.mat(&b(&format!("loop_v_ctrl_{kind}.weight")))?,
                    l.mat(&b(&format!("loop_ffn_ctrl_{kind}.weight")))?,
                ])
            };
            let down = m("down")?;
            let mut gf = Vec::with_capacity(4);
            for c in ["q", "k", "v", "ffn"] {
                gf.push(l.stacked_f32(&[&b(&format!("loop_{c}_ctrl_gain.weight")), &b(&format!("loop_{c}_ctrl_forget.weight"))])?);
            }
            let gain_forget: [GemvMatrix; 4] = gf.try_into().map_err(|_| bad("loop controllers".into()))?;
            loops.push(LoopLayer { msg_query: l.vec(&b("loop_msg_query.weight"))?, down, gain_forget });
        }
        Ok(Self { cfg, tok_embd, tok_norm, output, output_norm, output_res_query, layers, loops, bytes: l.bytes })
    }
}

/// Most prompt tokens one forward pass takes (prefill chunk).
pub const PREFILL_TMAX: usize = 2048;

/// Scratch buffers for up to `tmax` tokens per forward pass, all token-major (decode uses row
/// 0). The attention-residual bank holds `Config::bank_rows` rows of `[tmax][dim]`.
struct Scratch {
    tmax: usize,
    cur: DeviceBuffer<f32>,
    inp_sa: DeviceBuffer<f32>,
    mixed: DeviceBuffer<f32>,
    inputs: [DeviceBuffer<f32>; 3],
    normed: [DeviceBuffer<f32>; 3],
    msgs: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    kdyn: DeviceBuffer<f32>,
    vdyn: DeviceBuffer<f32>,
    r: DeviceBuffer<f32>,
    sinks: DeviceBuffer<f32>,
    attn: DeviceBuffer<f32>,
    attn_work: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    o: DeviceBuffer<f32>,
    m: DeviceBuffer<f32>,
    ctmp: DeviceBuffer<f32>,
    mn: DeviceBuffer<f32>,
    rlogits: DeviceBuffer<f32>,
    ids: DeviceBuffer<i32>,
    zero_ids: DeviceBuffer<i32>,
    /// Per-token completion counters of the fused router (always zero between launches).
    router_cnt: DeviceBuffer<u32>,
    w: DeviceBuffer<f32>,
    gu: DeviceBuffer<f32>,
    y: DeviceBuffer<f32>,
    cd: DeviceBuffer<f32>,
    crms: DeviceBuffer<f32>,
    bank: DeviceBuffer<f32>,
    prelude: DeviceBuffer<f32>,
    bank_out: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    token: DeviceBuffer<i32>,
    pos: DeviceBuffer<i32>,
    next: DeviceBuffer<i32>,
    qa: [QAct; 3],
    qa_attn: QAct,
    qa_ff: QAct,
    qa_last: QAct,
    /// Prefill work items: dense rows, MoE gate_up and MoE down.
    items_dense: DeviceBuffer<u32>,
    items_gu: DeviceBuffer<u32>,
    items_dn: DeviceBuffer<u32>,
    /// Tensor-core prefill work lists: dense rows, MoE gate_up and MoE down.
    work: [DevWork; 3],
    /// Layer intermediates (name, values), collected on decode steps when set (verification).
    dbg: Option<Vec<(&'static str, Vec<f32>)>>,
    /// Read-only probes of the next decode step ([`Session::arm_probe`]).
    probe: Option<Box<probe::Probe>>,
    /// Decode projections on tensor cores rather than dp4a ([`decode_q8_mma`]).
    decode_mma: bool,
}

/// A [`MmaWork`] on the device (fixed-size buffers, rewritten per use).
pub(crate) struct DevWork {
    pub(crate) items: DeviceBuffer<u32>,
    pub(crate) xidx: DeviceBuffer<u32>,
    pub(crate) yidx: DeviceBuffer<u32>,
    pub(crate) n_items: u32,
    pub(crate) idx_len: u32,
}

impl DevWork {
    pub(crate) fn new(ctx: &Arc<Context>, max_items: usize, max_idx: usize) -> Result<Self> {
        Ok(Self { items: DeviceBuffer::zeroed(ctx, 3 * max_items)?, xidx: DeviceBuffer::zeroed(ctx, max_idx)?, yidx: DeviceBuffer::zeroed(ctx, max_idx)?, n_items: 0, idx_len: 0 })
    }

    /// Synchronous upload; the caller guarantees no kernel is reading the buffers.
    pub(crate) fn upload(&mut self, w: &MmaWork) -> Result<()> {
        let put = |buf: &mut DeviceBuffer<u32>, v: &[u32]| -> Result<()> {
            if v.len() > buf.len() {
                return Err(bad("prefill: work list too long".into()));
            }
            let mut p = v.to_vec();
            p.resize(buf.len(), 0);
            buf.copy_from_slice(&p)
        };
        put(&mut self.items, &w.items)?;
        put(&mut self.xidx, &w.xidx)?;
        put(&mut self.yidx, &w.yidx)?;
        self.n_items = w.n_items();
        self.idx_len = w.xidx.len() as u32;
        Ok(())
    }
}

/// Largest context a session accepts: 128 Ki positions (the longest validated on an RTX 4090)
/// plus room for the reply. The KV of the 8 full-attention slots grows by 32 KiB per position;
/// the sliding-window slots are rings.
pub const MAX_CONTEXT: u32 = 128 * 1024 + 1024;

/// Largest pass count `ASTRAL_PASSES` accepts.
pub const MAX_PASSES: usize = 10;

/// Kernels and the stream, borrowed together with the scratch buffers.
struct Exec<'a> {
    st: &'a Stream,
    gemv: &'a Gemv,
    ops: &'a InklingOps,
    rms: &'a RmsNormF32,
}

impl Exec<'_> {
    /// `y = W x` for `ntok` quantised rows in `qa`: the int8 GEMV for one token, the grouped
    /// GEMM over the dense items for more.
    fn mm(&self, m: &GemvMatrix, qa: &QAct, y: &mut DeviceBuffer<f32>, ntok: usize, dense: &DeviceBuffer<u32>, work: &DevWork) -> Result<()> {
        if ntok == 1 {
            self.gemv.launch_q(self.st, m, qa, y, None)
        } else if small_batch(ntok) {
            // Every weight row read once for all rows (verify passes): the fused T-token GEMV,
            // else one grouped item.
            let mut it = [MultiItem { m, x: qa, y }];
            if Gemv::multi_t_ok(&it, ntok as u32) {
                return self.gemv.launch_q_multi_t(self.st, &mut it, ntok as u32);
            }
            let [MultiItem { y, .. }] = it;
            self.gemv.launch_items(self.st, m, qa, ntok as u32, y, ntok as u32, dense, 1)
        } else if use_mma() && m.n.is_multiple_of(16) {
            let n = ntok as u32;
            self.gemv.launch_mma(self.st, m, qa, n, y, n, &work.items, work.n_items, &work.xidx, &work.yidx, work.idx_len)
        } else {
            let items = ntok.div_ceil(QITEM_ROWS) as u32;
            self.gemv.launch_items(self.st, m, qa, ntok as u32, y, ntok as u32, dense, items)
        }
    }

    /// `y[t] = W x[t]` with f32 activations (F32 / BF16 weights), `x` token-major of width k.
    fn mm_f32(&self, m: &GemvMatrix, x: &DeviceBuffer<f32>, y: &mut DeviceBuffer<f32>, ntok: usize, zero_ids: &DeviceBuffer<i32>) -> Result<()> {
        if ntok == 1 {
            self.gemv.launch(self.st, m, x, 0, y)
        } else {
            self.gemv.launch_ids(self.st, m, x, m.k, y, &zero_ids.view(0, ntok)?)
        }
    }

    fn norm(&self, x: &DeviceBuffer<f32>, w: &DeviceBuffer<f32>, out: &mut DeviceBuffer<f32>, rows: usize, d: usize, eps: f32) -> Result<()> {
        self.rms.launch(self.st, x, Some(w as &dyn Dev<f32>), out, rows as u32, d as u32, eps)
    }
}

/// One sequence on the GPU: batched prefill, then one token per decode step.
pub struct Session {
    pub w: Weights,
    stream: Stream,
    /// Stream for eager work (prefill, verification); the decode graph replays on `stream`.
    eager_stream: Stream,
    gemv: Gemv,
    ops: InklingOps,
    rms: RmsNormF32,
    kc: Vec<DeviceBuffer<u16>>,
    vc: Vec<DeviceBuffer<u16>>,
    conv: Vec<DeviceBuffer<f32>>,
    s: Scratch,
    capacity: u32,
    pub pos: u32,
    /// When set, `l_out` of every slot is copied to the host each forward (verification).
    pub capture: bool,
    pub captured: Vec<(usize, Vec<f32>)>,
    graph: Option<GraphExec>,
    /// `reset()` as a graph (the conv-state memsets), for the same reason as graph prefill.
    reset_graph: Option<GraphExec>,
    /// DSpark speculative decoding, when enabled ([`Session::enable_spec`]).
    spec: Option<Box<Spec>>,
    /// Host copies of small weights for [`Session::probe_report`], read on first use.
    probe_w: Option<probe::HostW>,
}

/// Floats of attention workspace: decode at full capacity, or `tmax` tokens over 512 positions.
fn attn_work_len(c: &Config, capacity: u32, tmax: usize) -> usize {
    use astral_kernels::inkling::{attn_workspace, ATT_CHUNK};
    let (h, hd) = (c.heads as u32, c.head_dim as u32);
    // Decode at full capacity (one chunk per split), and passes of up to QITEM_ROWS tokens
    // (speculative verification) with a capped split count.
    let decode = attn_workspace(1, h, hd, capacity.div_ceil(ATT_CHUNK)).max(attn_workspace(QITEM_ROWS.min(tmax) as u32, h, hd, capacity.div_ceil(ATT_CHUNK).min(graph_splits())));
    if split_prefill() {
        decode.max(attn_workspace(tmax.min(512) as u32, h, hd, 512u32.div_ceil(ATT_CHUNK)))
    } else {
        decode
    }
}

impl Session {
    pub fn new(ctx: &Arc<Context>, w: Weights, capacity: u32) -> Result<Self> {
        if capacity == 0 || capacity > MAX_CONTEXT {
            return Err(bad(format!("capacity {capacity} outside 1..={MAX_CONTEXT}")));
        }
        let reg = KernelRegistry::load(ctx)?;
        let c = &w.cfg;
        let tmax = std::env::var("ASTRAL_TMAX").ok().and_then(|v| v.parse().ok()).unwrap_or(PREFILL_TMAX).clamp(1, PREFILL_TMAX).min(capacity as usize);
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(ctx, n);
        let (d, kvw, slots, t, ke) = (c.dim, c.kv_width(), c.n_slots(), tmax, c.n_expert_used);
        let mut kc = Vec::with_capacity(slots);
        let mut vc = Vec::with_capacity(slots);
        let mut conv = Vec::with_capacity(slots);
        for slot in 0..slots {
            let rows = c.kv_rows(slot, capacity, tmax) as usize;
            kc.push(DeviceBuffer::<u16>::zeroed(ctx, rows * kvw)?);
            vc.push(DeviceBuffer::<u16>::zeroed(ctx, rows * kvw)?);
            conv.push(z(3 * c.conv_channels())?);
        }
        let rank = w.loops.first().map_or(1, |l| l.down[0].n as usize);
        let dynw = w.layers.iter().filter_map(|l| l.k_conv_proj.as_ref()).map(|m| m.n as usize).max().unwrap_or(4);
        let max_items = t + c.n_expert + 1;
        let s = Scratch {
            tmax,
            cur: z(t * d)?,
            inp_sa: z(t * d)?,
            mixed: z(3 * t * d)?,
            inputs: [z(t * d)?, z(t * d)?, z(t * d)?],
            normed: [z(t * d)?, z(t * d)?, z(t * d)?],
            msgs: z(4 * t * d)?,
            q: z(t * c.heads * c.head_dim)?,
            k: z(t * kvw)?,
            v: z(t * kvw)?,
            kdyn: z(t * dynw)?,
            vdyn: z(t * dynw)?,
            r: z(t * c.heads * c.n_rel as usize)?,
            sinks: z(t * c.heads)?,
            attn: z(t * c.heads * c.head_dim)?,
            attn_work: z(attn_work_len(c, capacity, t))?,
            gate: z(t * c.heads * c.head_dim)?,
            o: z(t * d)?,
            m: z(t * d)?,
            ctmp: z(t * d)?,
            mn: z(t * d)?,
            rlogits: z(t * c.n_expert)?,
            ids: DeviceBuffer::<i32>::zeroed(ctx, t * ke)?,
            zero_ids: DeviceBuffer::<i32>::zeroed(ctx, t)?,
            router_cnt: DeviceBuffer::<u32>::zeroed(ctx, t)?,
            w: z(t * ke)?,
            gu: z(t * ke * 2 * c.n_ff_exp)?,
            y: z(t * ke * d)?,
            cd: z((3 * t * rank).max((1 + astral_kernels::inkling::CTRL_T_SPLITS as usize) * t * rank))?,
            crms: z(2 * t)?,
            bank: z(c.bank_rows() * t * d)?,
            prelude: z(t * d)?,
            bank_out: z(t * d)?,
            logits: z(c.vocab_padded)?,
            token: DeviceBuffer::<i32>::zeroed(ctx, t)?,
            pos: DeviceBuffer::<i32>::zeroed(ctx, 1)?,
            next: DeviceBuffer::<i32>::zeroed(ctx, 1)?,
            qa: [QAct::with_f16(ctx, d as u32, t as u32)?, QAct::with_f16(ctx, d as u32, t as u32)?, QAct::with_f16(ctx, d as u32, t as u32)?],
            qa_attn: QAct::with_f16(ctx, (c.heads * c.head_dim) as u32, t as u32)?,
            qa_ff: QAct::with_f16(ctx, c.n_ff_exp as u32, (t * ke) as u32)?,
            qa_last: QAct::new(ctx, d as u32, 1)?,
            items_dense: DeviceBuffer::<u32>::zeroed(ctx, max_items * QItem::WORDS)?,
            items_gu: DeviceBuffer::<u32>::zeroed(ctx, max_items * QItem::WORDS)?,
            items_dn: DeviceBuffer::<u32>::zeroed(ctx, max_items * QItem::WORDS)?,
            work: {
                let (mi, mx) = (t + c.n_expert + 1, t * ke);
                [DevWork::new(ctx, mi, mx)?, DevWork::new(ctx, mi, mx)?, DevWork::new(ctx, mi, mx)?]
            },
            dbg: None,
            probe: None,
            decode_mma: decode_q8_mma(),
        };
        let mut me = Self {
            stream: Stream::new(ctx)?,
            eager_stream: Stream::new(ctx)?,
            gemv: {
                // Deterministic split-K for the prefill GEMMs (bit-reproducible prefill).
                let mut g = Gemv::new(&reg)?;
                g.reserve_split_workspace(ctx, SPLIT_WORKSPACE_FLOATS)?;
                g.set_small_mma(small_q8_mma(ctx)?);
                g
            },
            ops: InklingOps::new(&reg)?,
            rms: RmsNormF32::new(&reg)?,
            kc,
            vc,
            conv,
            s,
            capacity,
            pos: 0,
            capture: false,
            captured: Vec::new(),
            graph: None,
            reset_graph: None,
            spec: None,
            probe_w: None,
            w,
        };
        me.capture_graph()?;
        {
            let Session { stream, conv, reset_graph, .. } = &mut me;
            *reset_graph = Some(stream.capture(|st| {
                for c in conv.iter_mut() {
                    c.zero_async(st)?;
                }
                Ok(())
            })?);
        }
        Ok(me)
    }

    /// Records the one-token forward pass as a CUDA graph (nothing runs), so the first decode
    /// step does not pay for capture and instantiation. No-op with `ASTRAL_NO_GRAPH=1`.
    fn capture_graph(&mut self) -> Result<()> {
        if self.graph.is_some() || std::env::var_os("ASTRAL_NO_GRAPH").is_some() {
            return Ok(());
        }
        let Session { w, stream, gemv, ops, rms, kc, vc, conv, s, capacity, graph, .. } = self;
        let g = stream.capture(|st| forward(&Exec { st, gemv, ops, rms }, w, kc, vc, conv, s, *capacity, 0, 1, None, &mut SpecIo::none()))?;
        *graph = Some(g);
        Ok(())
    }

    /// Runs one token at position `self.pos`; returns the greedy next token and advances.
    ///
    /// The first plain step captures the whole forward pass into a CUDA graph; later steps
    /// replay it (token and position live in device buffers, so the graph never changes).
    /// Verification (`capture`, debug taps) and `ASTRAL_NO_GRAPH=1` run eagerly instead.
    pub fn step(&mut self, token: u32) -> Result<u32> {
        if self.pos >= self.capacity {
            return Err(bad(format!("context full ({} positions)", self.capacity)));
        }
        let Session { w, stream, gemv, ops, rms, kc, vc, conv, s, capacity, pos, capture, captured, graph, .. } = self;
        // The previous call ended with a stream sync, so these synchronous copies cannot race.
        let t0 = std::time::Instant::now();
        let mut tok = vec![0i32; s.tmax];
        tok[0] = token as i32;
        s.token.copy_from_slice(&tok)?;
        s.pos.copy_from_slice(&[*pos as i32])?;
        let t1 = std::time::Instant::now();
        let eager = *capture || s.dbg.is_some() || s.probe.as_ref().is_some_and(|p| p.armed) || std::env::var_os("ASTRAL_NO_GRAPH").is_some();
        if eager {
            captured.clear();
            let ex = Exec { st: stream, gemv, ops, rms };
            forward(&ex, w, kc, vc, conv, s, *capacity, *pos, 1, capture.then_some(captured), &mut SpecIo::none())?;
        } else {
            if graph.is_none() {
                let g = stream.capture(|st| forward(&Exec { st, gemv, ops, rms }, w, kc, vc, conv, s, *capacity, 0, 1, None, &mut SpecIo::none()))?;
                *graph = Some(g);
            }
            graph.as_ref().expect("graph captured above").launch(stream)?;
        }
        let t2 = std::time::Instant::now();
        stream.synchronize()?;
        let t3 = std::time::Instant::now();
        *pos += 1;
        let r = s.next.to_vec()?[0] as u32;
        if std::env::var_os("ASTRAL_STEP_TIMING").is_some() {
            eprintln!(
                "step copy {:.0} us launch {:.0} us sync {:.0} us read {:.0} us",
                (t1 - t0).as_secs_f64() * 1e6,
                (t2 - t1).as_secs_f64() * 1e6,
                (t3 - t2).as_secs_f64() * 1e6,
                t3.elapsed().as_secs_f64() * 1e6
            );
        }
        Ok(r)
    }

    /// Processes `tokens` (the prompt) in batched chunks; returns the greedy next token after
    /// the last one. Eager (no graph): every chunk shape differs and MoE grouping reads the
    /// router's picks on the host once per layer.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<u32> {
        if tokens.is_empty() {
            return Err(bad("prefill: no tokens".into()));
        }
        if self.pos as usize + tokens.len() > self.capacity as usize {
            return Err(bad(format!("prefill: {} tokens at position {} exceed the context of {}", tokens.len(), self.pos, self.capacity)));
        }
        use astral_kernels::inkling::{attn_workspace, ATT_CHUNK};
        let mut next = 0;
        let mut done = 0;
        self.captured.clear();
        while done < tokens.len() {
            // Largest chunk whose attention workspace fits.
            let (h, hd) = (self.w.cfg.heads as u32, self.w.cfg.head_dim as u32);
            let mut t = (tokens.len() - done).min(self.s.tmax);
            // Only the split-K fallback needs workspace; flash-attention prefill does not.
            while split_prefill() && t > 1 && attn_workspace(t as u32, h, hd, (self.pos + t as u32).div_ceil(ATT_CHUNK)) > self.s.attn_work.len() {
                t /= 2;
            }
            let chunk = &tokens[done..done + t];
            let Session { w, eager_stream: stream, gemv, ops, rms, kc, vc, conv, s, capacity, pos, capture, captured, spec, .. } = self;
            let mut tok = vec![0i32; s.tmax];
            for (i, &x) in chunk.iter().enumerate() {
                tok[i] = x as i32;
            }
            s.token.copy_from_slice(&tok)?;
            s.pos.copy_from_slice(&[*pos as i32])?;
            if t > 1 {
                let mut words = QItem::words(&QItem::dense(t as u32));
                words.resize(s.items_dense.len(), 0);
                s.items_dense.copy_from_slice(&words)?;
                let mut dense = MmaWork::default();
                let rows: Vec<(u32, u32)> = (0..t as u32).map(|r| (r, r)).collect();
                dense.push(0, &rows);
                s.work[0].upload(&dense)?;
            }
            let ex = Exec { st: stream, gemv, ops, rms };
            if let Some(k) = std::env::var("ASTRAL_EAGER_DIAG").ok().and_then(|v| v.parse().ok()) {
                DIAG_LIMIT.store(k, std::sync::atomic::Ordering::Relaxed);
            }
            // Prefill runs as a one-off CUDA graph: bursts of eager launches (or memsets) leave the
            // process's later graph replays ~6% slower on this driver (measured: decode 135 vs 144
            // tok/s), so a serving process makes none. ASTRAL_PREFILL_EAGER=1 launches eagerly.
            let r = if std::env::var_os("ASTRAL_PREFILL_EAGER").is_none() && !*capture {
                let p0 = *pos;
                let cap = *capacity;
                // PDL helps decode's small kernels but slows prefill's large ones: off while capturing.
                let prev = stream.context().set_pdl(std::env::var_os("ASTRAL_PREFILL_PDL").is_some());
                let captured_graph = stream.capture(|st| forward(&Exec { st, gemv: ex.gemv, ops: ex.ops, rms: ex.rms }, w, kc, vc, conv, s, cap, p0, t, None, &mut spec_io(spec)));
                stream.context().set_pdl(prev);
                match captured_graph {
                    Ok(g) => g.launch(stream).and_then(|_| stream.synchronize()),
                    Err(e) => Err(e),
                }
            } else {
                forward(&ex, w, kc, vc, conv, s, *capacity, *pos, t, capture.then_some(captured), &mut spec_io(spec))
            };
            DIAG_LIMIT.store(usize::MAX, std::sync::atomic::Ordering::Relaxed);
            r?;
            stream.synchronize()?;
            // Speculative decoding: the draft's context K/V for this chunk (one-off graph).
            if let Some(sp) = spec.as_deref_mut() {
                let mut wk = MmaWork::default();
                wk.push(0, &(0..t as u32).map(|r| (r, r)).collect::<Vec<_>>());
                sp.work_pf.upload(&wk)?;
                let at = DevIndex::of(&s.pos, 0)?;
                let prev = stream.context().set_pdl(false);
                let g = stream.capture(|st| sp.store_ctx(st, gemv, t, at, false));
                stream.context().set_pdl(prev);
                g?.launch(stream)?;
                stream.synchronize()?;
                sp.last_rows = t;
            }
            *pos += t as u32;
            next = s.next.to_vec()?[0] as u32;
            done += t;
        }
        if std::env::var_os("ASTRAL_RECAPTURE").is_some() {
            self.graph = None;
            self.capture_graph()?;
        }
        Ok(next)
    }

    /// Enables DSpark speculative decoding with the draft of the native checkpoint in `dir`
    /// (`config.json` with `mtp_config`, `mtp.safetensors`), verifying `k` drafts per round.
    pub fn enable_spec(&mut self, dir: &std::path::Path, k: usize) -> Result<()> {
        let ctx = self.stream.context().clone();
        let reg = KernelRegistry::load(&ctx)?;
        let dops = DsparkOps::new(&reg)?;
        let dw = DraftWeights::load(&ctx, dir, &self.w, &self.gemv, &dops)?;
        let c = &self.w.cfg;
        if dw.cfg.taps[3] != c.n_slots() || dw.cfg.taps[..3].iter().any(|&v| v >= c.n_slots()) {
            return Err(bad(format!("draft taps {:?} do not fit {} layer visits", dw.cfg.taps, c.n_slots())));
        }
        let mut sp = Spec::new(&ctx, dw, dops, k, self.s.tmax, self.capacity, c.n_slots(), c.kv_width(), c.dim, self.w.output.n as usize)?;
        // Conv rollback table: (state, hist, channels) per slot and site.
        let (kvw, d) = (c.kv_width(), c.dim);
        let ho = hist_offsets(k + 1, kvw, d);
        let so = [0, 3 * kvw, 6 * kvw, (2 * kvw + d) * 3];
        let ch = [kvw, kvw, d, d];
        let mut t = Vec::new();
        for (cv, h) in self.conv.iter().zip(&sp.hist) {
            for i in 0..4 {
                t.extend([cv.device_ptr() + 4 * so[i] as u64, h.device_ptr() + 4 * ho[i] as u64, ch[i] as u64]);
            }
        }
        sp.table.copy_from_slice(&t)?;
        sp.entries = (t.len() / 3) as u32;
        sp.max_ch = kvw.max(d) as u32;
        if let Some(p) = std::env::var_os("ASTRAL_SPEC_LOG") {
            let f = std::fs::OpenOptions::new().create(true).append(true).open(&p).map_err(|e| bad(format!("ASTRAL_SPEC_LOG {}: {e}", p.to_string_lossy())))?;
            sp.log = Some(std::io::BufWriter::new(f));
        }
        eprintln!("dspark draft: {:.2} GB, k = {k}", sp.w.bytes as f64 / 1e9);
        self.spec = Some(Box::new(sp));
        Ok(())
    }

    /// Diagnostics: make speculative rounds accept against `seq` (prompt + continuation, token
    /// by position) instead of the target's greedy picks. Call before the first `spec_generate`.
    pub fn spec_force(&mut self, seq: &[u32]) -> Result<()> {
        let sp = self.spec.as_deref_mut().ok_or_else(|| bad("spec_force: speculative decoding is not enabled".into()))?;
        let mut v: Vec<i32> = seq.iter().map(|&t| t as i32).collect();
        v.resize(self.capacity as usize + 16, 0);
        sp.forced = Some(DeviceBuffer::from_slice(self.stream.context(), &v)?);
        sp.forced_host = seq.to_vec();
        sp.rounds.iter_mut().for_each(|r| *r = None);
        Ok(())
    }

    pub fn spec_stats(&self) -> Option<SpecStats> {
        self.spec.as_ref().map(|s| s.stats)
    }

    /// Greedy generation of `n` tokens after `prompt` with speculative decoding (needs
    /// [`Session::enable_spec`]). Each round is one CUDA graph: verify the anchor and `k` drafts,
    /// accept the matching prefix plus the target's next pick, roll the short-conv states back,
    /// extend the draft's context and draft the next block.
    pub fn spec_generate(&mut self, prompt: &[u32], n: usize) -> Result<Vec<u32>> {
        let mut out = vec![self.spec_begin(prompt)?];
        while out.len() < n {
            let r = self.spec_next()?;
            if r.is_empty() {
                break;
            }
            out.extend(r);
        }
        // Diagnostics: time the draft-only graph (ASTRAL_SPEC_PROFILE=1).
        if std::env::var_os("ASTRAL_SPEC_PROFILE").is_some() {
            let Session { stream, spec, .. } = self;
            let sp = spec.as_deref_mut().expect("spec_begin checked");
            let reps = 50;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                sp.first.as_ref().expect("captured in spec_begin").launch(stream)?;
            }
            stream.synchronize()?;
            eprintln!("profile: draft step {:.3} ms (graph replay)", t.elapsed().as_secs_f64() / reps as f64 * 1e3);
        }
        out.truncate(n);
        Ok(out)
    }

    /// Starts a speculative generation (needs [`Session::enable_spec`]): prefills `prompt`
    /// from the current position (call [`Session::reset`] first for a new sequence) and drafts
    /// the first block. Returns the first generated token; [`Session::spec_next`] yields the rest.
    pub fn spec_begin(&mut self, prompt: &[u32]) -> Result<u32> {
        if self.spec.is_none() {
            return Err(bad("spec_begin: speculative decoding is not enabled".into()));
        }
        let mut next = self.prefill(prompt)?;
        if let Some(&t) = self.spec.as_ref().and_then(|sp| sp.forced_host.get(prompt.len())) {
            next = t;
        }
        let Session { w, stream, gemv, ops, rms, kc, vc, conv, s, capacity, pos, spec, .. } = self;
        let sp = spec.as_deref_mut().expect("checked above");
        let vmax = sp.k + 1;
        // The verify pass's dense work list (prefill left its own in work[0]).
        let mut dense = MmaWork::default();
        dense.push(0, &(0..vmax as u32).map(|r| (r, r)).collect::<Vec<_>>());
        s.work[0].upload(&dense)?;
        let mut words = QItem::words(&QItem::dense(vmax as u32));
        words.resize(s.items_dense.len(), 0);
        s.items_dense.copy_from_slice(&words)?;
        let mut tok = vec![0i32; s.tmax];
        tok[0] = next as i32;
        s.token.copy_from_slice(&tok)?;
        s.pos.copy_from_slice(&[*pos as i32])?;
        sp.info.copy_from_slice(&[*pos as i32, sp.last_rows as i32 - 1, *pos as i32, 0])?;
        if sp.first.is_none() {
            let kmax = sp.k;
            let g = stream.capture(|st| sp.draft(st, gemv, ops, w, &s.token, kmax))?;
            sp.first = Some(g);
        }
        // One round graph per verified length (captured once; capture runs nothing).
        let (lo, hi) = if sp.policy.adaptive { (sp.policy.kmin, sp.k) } else { (sp.k, sp.k) };
        for kv in lo..=hi {
            if sp.rounds[kv].is_some() {
                continue;
            }
            let taps = [sp.w.cfg.taps[0], sp.w.cfg.taps[1], sp.w.cfg.taps[2]];
            let nv = kv + 1;
            let g = stream.capture(|st| {
                let ex = Exec { st, gemv, ops, rms };
                {
                    let mut io = SpecIo {
                        feats: Some(&mut sp.feats),
                        taps,
                        hist: Some(&mut sp.hist),
                        hist_rows: vmax,
                        all: Some((&mut sp.logits_all, &mut sp.next_all)),
                        argmax: Some((&sp.ops, &mut sp.part_v, &mut sp.part_i)),
                    };
                    forward(&ex, w, kc, vc, conv, s, *capacity, 0, nv, None, &mut io)?;
                }
                sp.ops.accept(st, &sp.next_all, &mut s.token, kv as u32, &mut sp.out, &mut sp.info, &mut s.pos, sp.forced.as_ref().map(|f| f as &dyn Dev<i32>))?;
                sp.ops.conv_commit(st, &sp.table, sp.entries, sp.max_ch, &sp.info)?;
                let at = DevIndex::of(&sp.info, 0)?;
                sp.store_ctx(st, gemv, nv, at, true)?;
                let nd = sp.policy.drafts_after(kv);
                sp.draft(st, gemv, ops, w, &s.token, nd)
            })?;
            sp.rounds[kv] = Some(g);
        }
        sp.first.as_ref().expect("captured above").launch(stream)?;
        stream.synchronize()?;
        Ok(next)
    }

    /// One speculative round after [`Session::spec_begin`]: verifies the drafts and returns the
    /// accepted tokens plus the target's next one (1..=k+1 tokens), or none when the context
    /// is full.
    pub fn spec_next(&mut self) -> Result<Vec<u32>> {
        let Session { stream, capacity, pos, spec, .. } = self;
        let sp = spec.as_deref_mut().ok_or_else(|| bad("spec_next: speculative decoding is not enabled".into()))?;
        let vmax = sp.k + 1;
        if *pos as usize + vmax + 1 > *capacity as usize {
            return Ok(Vec::new());
        }
        let kv = sp.policy.cur;
        let round = sp.rounds[kv].as_ref().ok_or_else(|| bad("spec_next: call spec_begin first".into()))?;
        let t0 = std::time::Instant::now();
        round.launch(stream)?;
        stream.synchronize()?;
        let o = sp.out.to_vec()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let c = (o[0].max(1) as usize).min(kv + 1);
        if let Some(log) = sp.log.as_mut() {
            use std::io::Write;
            writeln!(log, "{{\"pos\":{},\"k\":{kv},\"acc\":{},\"ms\":{ms:.4}}}", *pos, c - 1).and_then(|()| log.flush()).map_err(|e| bad(format!("ASTRAL_SPEC_LOG: {e}")))?;
        }
        *pos += c as u32;
        sp.stats.rounds += 1;
        sp.stats.accepted += (c - 1) as u64;
        sp.stats.drafted += kv as u64;
        sp.policy.update(kv, c - 1, ms);
        Ok(o[1..=c].iter().map(|&t| t as u32).collect())
    }

    /// Read-only probes (docs/trace.md): the next [`Session::step`] runs eagerly with the same
    /// kernels as the graph replay and stages small intermediates; read them with
    /// [`Session::probe_report`]. Staging size: `ASTRAL_TRACE_FLOATS` (default 24 Mi floats);
    /// `ASTRAL_TRACE_LENS=0` skips the per-visit raw-lens head.
    pub fn arm_probe(&mut self) -> Result<()> {
        if self.s.probe.is_none() {
            let floats = std::env::var("ASTRAL_TRACE_FLOATS").ok().and_then(|v| v.parse().ok()).unwrap_or(24usize << 20);
            self.s.probe = Some(Box::new(probe::Probe::new(self.stream.context(), self.w.cfg.dim, self.w.cfg.vocab_padded, floats, self.w.cfg.bank_rows())?));
        }
        let p = self.s.probe.as_deref_mut().expect("allocated above");
        p.lens = std::env::var("ASTRAL_TRACE_LENS").map_or(true, |v| v != "0");
        p.arm();
        Ok(())
    }

    /// JSON records of the probed step (one per layer visit, then the head), for `token` at
    /// position `pos - 1`; `target` adds its log-prob to the readouts; `validate` also
    /// recomputes the attention output (reads the V cache).
    pub fn probe_report(&mut self, token: u32, target: Option<u32>, validate: bool) -> Result<Vec<serde_json::Value>> {
        if !self.s.probe.as_ref().is_some_and(|p| p.armed) {
            return Err(bad("probe_report: no armed probe (call arm_probe, then step)".into()));
        }
        if self.pos == 0 {
            return Err(bad("probe_report: no step was run".into()));
        }
        self.stream.synchronize()?;
        let staged = self.s.probe.as_deref_mut().expect("checked above").take()?;
        if self.probe_w.is_none() {
            self.probe_w = Some(probe::HostW::load(&self.w)?);
        }
        let native = self.logits()?;
        let kvw = self.w.cfg.kv_width();
        let (kc, vc) = (&self.kc, &self.vc);
        // Positions lo..lo+n of a slot, through its ring (sliding-window slots) if it has one.
        let kv = |slot: usize, lo: usize, n: usize, is_v: bool| -> Result<Vec<u16>> {
            let b = if is_v { &vc[slot] } else { &kc[slot] };
            let rows = b.len() / kvw;
            let (r0, first) = (lo % rows, n.min(rows - lo % rows));
            let mut v = b.to_vec_range(r0 * kvw, first * kvw)?;
            if first < n {
                v.extend(b.to_vec_range(0, (n - first) * kvw)?);
            }
            Ok(v)
        };
        let a = probe::ReportIn { w: &self.w, hw: self.probe_w.as_ref().expect("loaded above"), pos: self.pos - 1, token, target, kv: &kv, validate, native: &native };
        probe::report(&staged, &a)
    }

    /// Diagnostics: the logits of every row of the last speculative verify pass (rows `0..=k`,
    /// unpadded vocabulary). Rows past the accepted prefix saw rejected drafts.
    pub fn spec_round_logits(&self) -> Result<Vec<Vec<f32>>> {
        let sp = self.spec.as_deref().ok_or_else(|| bad("spec_round_logits: speculative decoding is not enabled".into()))?;
        let vp = self.w.output.n as usize;
        let all = sp.logits_all.to_vec()?;
        Ok((0..=sp.k).map(|r| all[r * vp..r * vp + self.w.cfg.vocab].to_vec()).collect())
    }

    /// Whether [`Session::enable_spec`] was called.
    pub fn spec_enabled(&self) -> bool {
        self.spec.is_some()
    }

    /// Context capacity in positions.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Collect layer intermediates on every following decode step (see [`Session::take_debug`]).
    pub fn set_debug(&mut self, on: bool) {
        self.s.dbg = on.then(Vec::new);
    }

    pub fn take_debug(&mut self) -> Vec<(&'static str, Vec<f32>)> {
        self.s.dbg.as_mut().map(std::mem::take).unwrap_or_default()
    }

    /// Logits of the last token of the last call (unpadded vocabulary), copied to the host.
    pub fn logits(&self) -> Result<Vec<f32>> {
        let mut v = self.s.logits.to_vec()?;
        v.truncate(self.w.cfg.vocab);
        Ok(v)
    }

    /// Diagnostic: one tiny eager kernel launch (argmax over the logits) on `stream` ("same") or
    /// the eager stream ("eager"), to test whether any non-graph launch slows later graph replays.
    pub fn touch(&mut self, which: &str) -> Result<()> {
        if let Some(rest) = which.strip_prefix("warmgraph_") {
            self.step(0)?;
            self.reset()?;
            return self.touch(rest);
        }
        let st = &self.eager_stream;
        match which {
            "memcpy" => {
                let d = self.w.cfg.dim;
                self.s.prelude.copy_range_from_async(0, &self.s.cur, 0, d, st)?;
            }
            "gemv" => {
                let l = &self.w.layers[0];
                self.gemv.launch_q(st, &l.wo, &self.s.qa_attn, &mut self.s.o, None)?;
            }
            "embed" => self.ops.embed(st, &self.w.tok_embd, &self.s.token, &mut self.s.inp_sa, 1)?,
            "rmsnorm" => {
                let d = self.w.cfg.dim;
                self.rms.launch(st, &self.s.inp_sa, Some(&self.w.tok_norm as &dyn Dev<f32>), &mut self.s.cur, 1, d as u32, 1e-6)?;
            }
            "mix" => {
                let d = self.w.cfg.dim as u32;
                let rs = self.s.tmax * self.w.cfg.dim;
                self.ops.mix(st, &self.s.bank, &[], rs, &self.s.cur, &self.w.output_res_query, 1, &mut self.s.m, d, 1)?;
            }
            "rms_q8" => {
                let d = self.w.cfg.dim;
                self.ops.rms_q8(st, &self.s.m, d, Some(&self.w.output_norm), Some(&mut self.s.mn), &mut self.s.qa_last, 1, 1e-6)?;
            }
            "softcap" => self.ops.softcap(st, &mut self.s.logits, 16, 30.0)?,
            "many" => {
                for _ in 0..100 {
                    self.ops.argmax(st, &self.s.logits, 16, &mut self.s.next)?;
                }
            }
            "many_same" => {
                for _ in 0..100 {
                    self.ops.argmax(&self.stream, &self.s.logits, 16, &mut self.s.next)?;
                }
                self.stream.synchronize()?;
            }
            _ => self.ops.argmax(st, &self.s.logits, 16, &mut self.s.next)?,
        }
        st.synchronize()
    }

    /// Forget the sequence: position 0, zeroed short-conv state (the KV cache is overwritten).
    pub fn reset(&mut self) -> Result<()> {
        match &self.reset_graph {
            Some(g) => g.launch(&self.stream)?,
            None => {
                for c in &mut self.conv {
                    c.zero_async(&self.stream)?;
                }
            }
        }
        self.stream.synchronize()?;
        self.pos = 0;
        Ok(())
    }
}

/// Speculative-decoding extras of one forward pass ([`SpecIo::none`] for plain passes).
pub(crate) struct SpecIo<'a> {
    /// Target features `[4][tmax][dim]`: the stream entering visits `taps`, then the final-norm
    /// output, for every token of the pass.
    pub(crate) feats: Option<&'a mut DeviceBuffer<f32>>,
    pub(crate) taps: [usize; 3],
    /// Verify: per slot, the short-conv inputs are saved here (`rows` rows per site) instead of
    /// advancing the conv state; `astral_dsp_conv_commit` applies the accepted prefix later.
    pub(crate) hist: Option<&'a mut [DeviceBuffer<f32>]>,
    pub(crate) hist_rows: usize,
    /// Verify: logits (`[ntok][vocab_padded]`) and greedy pick of every token.
    pub(crate) all: Option<(&'a mut DeviceBuffer<f32>, &'a mut DeviceBuffer<i32>)>,
    /// Verify: the multi-block argmax (ops and partials) for those picks.
    pub(crate) argmax: Option<(&'a DsparkOps, &'a mut DeviceBuffer<f32>, &'a mut DeviceBuffer<u32>)>,
}

impl SpecIo<'_> {
    pub(crate) fn none() -> Self {
        Self { feats: None, taps: [usize::MAX; 3], hist: None, hist_rows: 0, all: None, argmax: None }
    }
}

/// A slot's conv-input history for a verify pass.
struct Hist<'a> {
    buf: &'a mut DeviceBuffer<f32>,
    rows: usize,
}

/// The whole forward pass for `ntok` tokens (`s.token[..ntok]`, positions from `s.pos`), ending
/// with the greedy pick after the last token in `s.next`. With one token it enqueues work only
/// (graph-capturable); with more it syncs once per layer to group the MoE experts. `pos0` is the
/// host copy of the first position (prefill only; sizes the attention splits).
#[allow(clippy::too_many_arguments)]
fn forward(
    ex: &Exec,
    w: &Weights,
    kc: &mut [DeviceBuffer<u16>],
    vc: &mut [DeviceBuffer<u16>],
    conv: &mut [DeviceBuffer<f32>],
    s: &mut Scratch,
    capacity: u32,
    pos0: u32,
    ntok: usize,
    mut captured: Option<&mut Vec<(usize, Vec<f32>)>>,
    io: &mut SpecIo,
) -> Result<()> {
    let cfg = &w.cfg;
    let d = cfg.dim;
    let td = ntok * d;
    let rs = s.tmax * d;
    // Embedding and its norm.
    ex.ops.embed(ex.st, &w.tok_embd, &s.token, &mut s.inp_sa, ntok as u32)?;
    ex.norm(&s.inp_sa, &w.tok_norm, &mut s.cur, ntok, d, cfg.rms_eps)?;

    let mut blocks: Vec<u32> = Vec::new();
    let mut next_row = 0usize;
    let mut visit = 0usize;
    // Probes (decode steps only; see docs/trace.md).
    let armed = ntok == 1 && s.probe.as_ref().is_some_and(|p| p.armed);
    let (core_n, n_layer, core0, core1) = (cfg.core_end - cfg.core_start, cfg.n_layer, cfg.core_start, cfg.core_end);
    let taps = io.taps;
    let hist_rows = io.hist_rows;
    let mut run = |slot: usize, ip: usize, loop_i: Option<usize>, blocks: &mut Vec<u32>, next_row: &mut usize, s: &mut Scratch| -> Result<()> {
        // Diagnostic (ASTRAL_EAGER_DIAG=k): an eager pass stops after k layer slots.
        if pos0 != u32::MAX && ntok >= 1 && slot >= DIAG_LIMIT.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(());
        }
        if let Some(f) = io.feats.as_deref_mut()
            && let Some(t) = taps.iter().position(|&v| v == visit)
        {
            f.copy_range_from_async(t * s.tmax * d, &s.cur, 0, td, ex.st)?;
        }
        if armed && let Some(p) = s.probe.as_deref_mut() {
            let pass = if loop_i.is_some() {
                Some((slot - n_layer) / core_n + 1)
            } else if (core0..core1).contains(&ip) {
                Some(0)
            } else {
                None
            };
            p.ctx = probe::Ctx { visit, slot, ip, pass };
        }
        visit += 1;
        let mut hist = io.hist.as_deref_mut().map(|h| Hist { buf: &mut h[slot], rows: hist_rows });
        layer(ex, w, s, &mut kc[slot], &mut vc[slot], &mut conv[slot], ip, loop_i, blocks, next_row, capacity, pos0, ntok, hist.as_mut())?;
        if armed && let Some(p) = s.probe.as_deref_mut() {
            p.f(ex.st, "out", &s.cur, 0, d, Vec::new())?;
            p.lens(ex, w, &s.cur)?;
        }
        if let Some(c) = captured.as_deref_mut() {
            ex.st.synchronize()?;
            let mut v = s.cur.to_vec()?;
            v.truncate(td);
            c.push((slot, v));
        }
        Ok(())
    };

    let (start, end) = (cfg.core_start, cfg.core_end);
    let core = end - start;
    for ip in 0..start {
        run(ip, ip, None, &mut blocks, &mut next_row, s)?;
    }
    // Every pass starts from the prelude output.
    s.prelude.copy_range_from_async(0, &s.cur, 0, td, ex.st)?;
    let prelude_blocks = blocks.clone();
    for pass in 0..cfg.passes {
        let mut bank: Vec<u32> = Vec::new();
        if pass > 0 {
            // The previous pass's blocks after its first core block, and its output.
            bank = blocks.get(prelude_blocks.len() + 1..).map(|b| b.to_vec()).unwrap_or_default();
            s.bank_out.copy_range_from_async(0, &s.cur, 0, td, ex.st)?;
            s.cur.copy_range_from_async(0, &s.prelude, 0, td, ex.st)?;
            blocks = prelude_blocks.clone();
        }
        for ip in start..end {
            if pass == 0 {
                run(ip, ip, None, &mut blocks, &mut next_row, s)?;
                continue;
            }
            let li = ip - start;
            let mq = &w.loops[li].msg_query;
            ex.ops.mix(ex.st, &s.bank, &bank, rs, &s.bank_out, mq, (mq.len() / d) as u32, &mut s.msgs, d as u32, ntok as u32)?;
            if armed && let Some(p) = s.probe.as_deref_mut() {
                p.ctx = probe::Ctx { visit: start + pass * core + li, slot: cfg.n_layer + (pass - 1) * core + li, ip, pass: Some(pass) };
                p.states(ex.st, "msg_mix.states", &s.bank, &bank, rs, &s.bank_out, d)?;
                p.f(ex.st, "msg_mix.out", &s.msgs, 0, mq.len(), Vec::new())?;
            }
            run(cfg.n_layer + (pass - 1) * core + li, ip, Some(li), &mut blocks, &mut next_row, s)?;
        }
    }
    for ip in end..cfg.n_layer {
        run(ip, ip, None, &mut blocks, &mut next_row, s)?;
    }

    // Speculative decoding: the final-norm output of every token (a feature), and in verify
    // passes the logits and greedy pick of every token.
    if io.feats.is_some() || io.all.is_some() {
        ex.ops.mix(ex.st, &s.bank, &blocks, rs, &s.cur, &w.output_res_query, 1, &mut s.m, d as u32, ntok as u32)?;
        ex.ops.rms_q8(ex.st, &s.m, d, Some(&w.output_norm), Some(&mut s.mn), &mut s.qa[0], ntok as u32, cfg.rms_eps)?;
        if let Some(f) = io.feats.as_deref_mut() {
            f.copy_range_from_async(3 * s.tmax * d, &s.mn, 0, td, ex.st)?;
        }
        if let Some((lg, na)) = io.all.as_mut() {
            let vp = w.output.n as usize;
            ex.mm(&w.output, &s.qa[0], lg, ntok, &s.items_dense, &s.work[0])?;
            if cfg.softcap != 0.0 {
                ex.ops.softcap(ex.st, &mut **lg, (vp * ntok) as u32, cfg.softcap)?;
            }
            match io.argmax.as_mut() {
                Some((dops, pv, pi)) => dops.argmax_rows(ex.st, &**lg, vp as u32, cfg.vocab as u32, ntok as u32, &mut **pv, &mut **pi, &mut **na)?,
                None => {
                    for t in 0..ntok {
                        ex.ops.argmax(ex.st, &lg.view(t * vp, vp)?, cfg.vocab as u32, &mut na.view_mut(t, 1)?)?;
                    }
                }
            }
            return Ok(());
        }
    }

    // Output for the last token: depth mix, norm, LM head, softcap, greedy pick.
    let last = (ntok - 1) * d;
    {
        let bank = s.bank.view(last, s.bank.len() - last)?;
        let cur = s.cur.view(last, d)?;
        ex.ops.mix(ex.st, &bank, &blocks, rs, &cur, &w.output_res_query, 1, &mut s.m, d as u32, 1)?;
    }
    if armed && let Some(p) = s.probe.as_deref_mut() {
        p.ctx = probe::Ctx { visit: usize::MAX, slot: usize::MAX, ip: cfg.n_layer, pass: None };
        p.states(ex.st, "head_mix.states", &s.bank, &blocks, rs, &s.cur, d)?;
        p.f(ex.st, "head_mix.out", &s.m, 0, d, Vec::new())?;
    }
    ex.ops.rms_q8(ex.st, &s.m, d, Some(&w.output_norm), Some(&mut s.mn), &mut s.qa_last, 1, cfg.rms_eps)?;
    match w.output.ty {
        GemvType::Q8_0 | GemvType::Nvfp4 => ex.gemv.launch_q(ex.st, &w.output, &s.qa_last, &mut s.logits, None)?,
        _ => ex.gemv.launch(ex.st, &w.output, &s.mn, 0, &mut s.logits)?,
    }
    if cfg.softcap != 0.0 {
        ex.ops.softcap(ex.st, &mut s.logits, w.output.n, cfg.softcap)?;
    }
    ex.ops.argmax(ex.st, &s.logits, cfg.vocab as u32, &mut s.next)
}

/// The prefill extras when speculative decoding is on: every token's features.
fn spec_io(spec: &mut Option<Box<Spec>>) -> SpecIo<'_> {
    match spec.as_deref_mut() {
        Some(sp) => {
            let t = &sp.w.cfg.taps;
            let taps = [t[0], t[1], t[2]];
            SpecIo { feats: Some(&mut sp.feats), taps, hist: None, hist_rows: 0, all: None, argmax: None }
        }
        None => SpecIo::none(),
    }
}

/// Most attention splits of a graph-captured pass (decode, verify); `ASTRAL_ATT_SPLITS` to tune.
fn graph_splits() -> u32 {
    std::env::var("ASTRAL_ATT_SPLITS").ok().and_then(|v| v.parse().ok()).filter(|&n: &u32| n > 0).unwrap_or(32)
}

/// Prefill attention through the split-K decode kernel instead of flash attention.
fn split_prefill() -> bool {
    std::env::var_os("ASTRAL_ATTN_SPLIT_PREFILL").is_some()
}

/// Whether decode runs the fused attention-input projections (q, gate, rel, k, v and the side
/// projections) on tensor cores (`launch_q_multi_mma`, default) or with dp4a (`launch_q_multi`,
/// `ASTRAL_DECODE_DP4A=1`, or `ASTRAL_SMALL_DP4A=1` for every small product). On an RTX 4090 dp4a
/// decodes about 10% faster (154.2 vs 140.1 tok/s at 1K tokens of context), but it moves decode
/// further from prefill: the step-vs-decode check of the hold-out set (num-3, 320 positions) then
/// fails on its largest KL (0.082 against 0.023 for llama.cpp's own -ub 1 vs batch), so it is not
/// the default there.
fn decode_q8_mma() -> bool {
    std::env::var_os("ASTRAL_DECODE_DP4A").is_none() && std::env::var_os("ASTRAL_SMALL_DP4A").is_none()
}

/// Whether the Q8_0 products of 2..=8-token passes (DSpark verify and draft) run on tensor cores
/// (`launch_q_multi_mma`) or with dp4a (`launch_q_multi_t`). Tensor cores on most GPUs, dp4a on
/// an RTX 4090 (sm_89), where the tensor-core kernel reads the 201024-row LM head
/// at about 390 GB/s; DSpark 171.0 vs 206.3 tok/s there. `ASTRAL_SMALL_DP4A=1` or
/// `ASTRAL_SMALL_TC=1` forces one of them.
fn small_q8_mma(ctx: &Context) -> Result<bool> {
    if std::env::var_os("ASTRAL_SMALL_DP4A").is_some() {
        return Ok(false);
    }
    if std::env::var_os("ASTRAL_SMALL_TC").is_some() {
        return Ok(true);
    }
    let (major, minor) = ctx.device().compute_capability()?;
    Ok(astral_core::arch::ArchProfile::from_compute_capability(major, minor) != Some(astral_core::arch::ArchProfile::Sm89))
}

/// Passes of 2..=8 tokens (speculative verification) use the grouped dp4a GEMV instead of the
/// tensor-core prefill GEMMs (`ASTRAL_SMALL_MMA=1` to keep the latter).
fn small_batch(ntok: usize) -> bool {
    ntok > 1 && ntok <= QITEM_ROWS && std::env::var_os("ASTRAL_SMALL_MMA").is_none()
}

/// Depth mix fused with the following RMSNorm + quantisation (`ASTRAL_NO_FUSE_MIX=1` to split).
fn fuse_mix() -> bool {
    std::env::var_os("ASTRAL_NO_FUSE_MIX").is_none()
}

/// Sink and conv-tap projections as Q8_0 inside the fused projection launch (decode and small
/// batches); `ASTRAL_SIDE_BF16=1` keeps the separate BF16 GEMVs.
fn side_q8() -> bool {
    std::env::var_os("ASTRAL_SIDE_BF16").is_none()
}

/// Group MoE picks on the host (one sync per layer) instead of on the GPU.
fn host_group() -> bool {
    std::env::var_os("ASTRAL_HOST_MOE_GROUP").is_some()
}

/// Prefill GEMMs on tensor cores (default) or the dp4a grouped kernel (`ASTRAL_PREFILL_DP4A=1`).
fn use_mma() -> bool {
    std::env::var_os("ASTRAL_PREFILL_DP4A").is_none()
}

/// Device floats reserved for deterministic split-K in the prefill GEMMs (16 MiB). Launches that
/// would need more use fewer K splits.
const SPLIT_WORKSPACE_FLOATS: usize = 4 << 20;

static DIAG_LIMIT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);

/// Physical layer whose intermediates the debug taps collect (`ASTRAL_TAP_LAYER`, default 0).
pub fn tap_layer() -> usize {
    std::env::var("ASTRAL_TAP_LAYER").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

/// Groups the router's (token, expert) picks by expert into gate_up and down work items.
fn moe_items(ids: &[i32], k: usize, n_expert: usize) -> (Vec<QItem>, Vec<QItem>) {
    let mut by_e: Vec<Vec<u32>> = vec![Vec::new(); n_expert];
    for (p, &e) in ids.iter().enumerate() {
        by_e[(e.max(0) as usize).min(n_expert - 1)].push(p as u32);
    }
    let (mut gu, mut dn) = (Vec::new(), Vec::new());
    for (e, pairs) in by_e.iter().enumerate() {
        for chunk in pairs.chunks(QITEM_ROWS) {
            let mut a = QItem { e: e as u32, n: chunk.len() as u32, ..Default::default() };
            let mut b = a;
            for (i, &p) in chunk.iter().enumerate() {
                a.xi[i] = p / k as u32; // gate_up reads the token's normalised input
                a.yi[i] = p;
                b.xi[i] = p; // down reads the pair's activation
                b.yi[i] = p;
            }
            gu.push(a);
            dn.push(b);
        }
    }
    (gu, dn)
}

/// One `build_layer` of llama.cpp's inkling2 graph for `ntok` tokens; `s.cur` is updated in
/// place. `ip` is the physical layer; `loop_i` selects the loop controller (later passes only).
#[allow(clippy::too_many_arguments)]
fn layer(
    ex: &Exec,
    w: &Weights,
    s: &mut Scratch,
    kc: &mut DeviceBuffer<u16>,
    vc: &mut DeviceBuffer<u16>,
    conv: &mut DeviceBuffer<f32>,
    ip: usize,
    loop_i: Option<usize>,
    blocks: &mut Vec<u32>,
    next_row: &mut usize,
    capacity: u32,
    pos0: u32,
    ntok: usize,
    mut hist: Option<&mut Hist>,
) -> Result<()> {
    let cfg = &w.cfg;
    let l = &w.layers[ip];
    let hoff = hist.as_ref().map(|h| crate::dspark::hist_offsets(h.rows, cfg.kv_width(), cfg.dim));
    let (st, d, du, nt) = (ex.st, cfg.dim, cfg.dim as u32, ntok as u32);
    let (heads, kvh, hd, n_rel) = (cfg.heads as u32, cfg.kv_heads as u32, cfg.head_dim as u32, cfg.n_rel);
    let rows = (kc.len() / cfg.kv_width()) as u32;
    let kvw = cfg.kv_width();
    let rs = s.tmax * d;
    let tap = ntok == 1 && s.dbg.is_some() && loop_i.is_none() && ip == tap_layer();
    macro_rules! tap {
        ($name:literal, $buf:ident) => {
            if tap {
                st.synchronize()?;
                let mut v = s.$buf.to_vec()?;
                v.truncate(v.len() / s.tmax);
                s.dbg.as_mut().unwrap().push(($name, v));
            }
        };
    }

    // Attention inputs: SWA layers mix one, full-attention layers one each for Q, K and V; the
    // loop controllers then make all three differ. `n_in` distinct inputs are normalised.
    let nq = l.attn_res_query.as_ref().map_or(0, |q| q.len() / d);
    let row = |j: usize| if l.swa || nq <= 1 { 0 } else { j };
    let x_stride = if nq > 0 { nq * d } else { d };
    let n_in = if loop_i.is_some() || !l.swa { 3 } else { 1 };
    // Outside the loop the mix feeds the input norms directly: one fused kernel.
    let fuse_in = fuse_mix() && loop_i.is_none() && nq == n_in && s.dbg.is_none();
    if let Some(qry) = &l.attn_res_query {
        if fuse_in {
            let norms: Vec<&DeviceBuffer<f32>> = (0..n_in)
                .map(|j| match (&l.attn_norm, &l.qkv_inp_norm) {
                    (Some(an), _) => Ok(an),
                    (None, Some(qkv)) => Ok(&qkv[j]),
                    _ => Err(bad(format!("layer {ip}: no attention norm"))),
                })
                .collect::<Result<_>>()?;
            let Scratch { bank, cur, mixed, normed, qa, .. } = &mut *s;
            let mut outs: Vec<(&dyn Dev<f32>, Option<&mut dyn DevMut<f32>>, &mut QAct)> = Vec::with_capacity(n_in);
            for ((w, y), a) in norms.iter().zip(normed.iter_mut()).zip(qa.iter_mut()).take(n_in) {
                outs.push((*w as &dyn Dev<f32>, Some(y as &mut dyn DevMut<f32>), a));
            }
            ex.ops.mix_rms_q8(st, &*bank, blocks, rs, &*cur, qry, Some(mixed as &mut dyn DevMut<f32>), du, nt, &mut outs, cfg.rms_eps)?;
        } else {
            ex.ops.mix(st, &s.bank, blocks, rs, &s.cur, qry, nq as u32, &mut s.mixed, du, nt)?;
        }
    }
    let armed = ntok == 1 && s.probe.as_ref().is_some_and(|p| p.armed);
    if armed
        && nq > 0
        && let Some(p) = s.probe.as_deref_mut()
    {
        p.states(st, "attn_mix.states", &s.bank, blocks, rs, &s.cur, d)?;
        p.f(st, "attn_mix.out", &s.mixed, 0, nq * d, vec![nq as u32])?;
    }
    let batched_ctrl = (ntok == 1 || small_batch(ntok)) && std::env::var_os("ASTRAL_NO_BATCH_CTRL").is_none();
    if let (Some(li), true) = (loop_i, batched_ctrl) {
        // Q, K and V controllers in two launches, then their three input norms in one.
        let lp = &w.loops[li];
        let rank = lp.down[0].n as usize;
        let Scratch { mixed, cur, msgs, cd, inputs, normed, qa, .. } = &mut *s;
        let xs: Vec<_> = (0..3).map(|j| if nq > 0 { mixed.view(row(j) * d, mixed.len() - row(j) * d) } else { cur.view(0, cur.len()) }).collect::<Result<_>>()?;
        let ms: Vec<_> = (0..3).map(|j| msgs.view(j * d, msgs.len() - j * d)).collect::<Result<_>>()?;
        let (cd0, rest) = cd.split_at_mut_views(ntok * rank)?;
        let (cd1, cd2) = rest.split_at_mut_views(ntok * rank)?;
        let [i0, i1, i2] = inputs;
        {
            let mut cds = [cd0, cd1, cd2];
            let [c0, c1, c2] = &mut cds;
            let mut jobs: [(&GemvMatrix, &GemvMatrix, &dyn Dev<f32>, &dyn Dev<f32>, &mut dyn DevMut<f32>, &mut dyn DevMut<f32>); 3] = [
                (&lp.down[0], &lp.gain_forget[0], &xs[0], &ms[0], c0, &mut *i0),
                (&lp.down[1], &lp.gain_forget[1], &xs[1], &ms[1], c1, &mut *i1),
                (&lp.down[2], &lp.gain_forget[2], &xs[2], &ms[2], c2, &mut *i2),
            ];
            ex.ops.ctrl_multi(st, &mut jobs, x_stride, 4 * d, du, nt)?;
        }
        let norm = |j: usize| -> Result<&DeviceBuffer<f32>> {
            match (&l.attn_norm, &l.qkv_inp_norm) {
                (Some(an), _) => Ok(an),
                (None, Some(qkv)) => Ok(&qkv[j]),
                _ => Err(bad(format!("layer {ip}: no attention norm"))),
            }
        };
        let [n0, n1, n2] = normed;
        let [q0, q1, q2] = qa;
        let mut rj: [(&dyn Dev<f32>, &dyn Dev<f32>, Option<&mut dyn DevMut<f32>>, &mut QAct); 3] = [(&*i0, norm(0)?, Some(n0), q0), (&*i1, norm(1)?, Some(n1), q1), (&*i2, norm(2)?, Some(n2), q2)];
        ex.ops.rms_q8_multi(st, &mut rj, d, nt, cfg.rms_eps)?;
    } else if let Some(li) = loop_i {
        let lp = &w.loops[li];
        for j in 0..3 {
            let x = if nq > 0 { s.mixed.view(row(j) * d, s.mixed.len() - row(j) * d)? } else { s.cur.view(0, s.cur.len())? };
            let msg = s.msgs.view(j * d, s.msgs.len() - j * d)?;
            if ntok == 1 || small_batch(ntok) {
                ex.ops.ctrl_down(st, &lp.down[j], &x, x_stride, &msg, 4 * d, &mut s.cd, du, nt)?;
                ex.ops.ctrl_combine(st, &lp.gain_forget[j], &s.cd, &x, x_stride, &msg, 4 * d, &mut s.inputs[j], du, nt)?;
            } else {
                ex.ops.ctrl_t(st, &lp.down[j], &lp.gain_forget[j], &x, x_stride, &msg, 4 * d, &mut s.crms, &mut s.cd, &mut s.gu, &mut s.inputs[j], du, nt)?;
            }
        }
    }
    for j in 0..n_in {
        if (fuse_in && nq > 0) || (loop_i.is_some() && batched_ctrl) {
            break;
        }
        let nw: &DeviceBuffer<f32> = match (&l.attn_norm, &l.qkv_inp_norm) {
            (Some(an), _) => an,
            (None, Some(qkv)) => &qkv[j],
            _ => return Err(bad(format!("layer {ip}: no attention norm"))),
        };
        let (x, xs) = if loop_i.is_some() {
            (s.inputs[j].view(0, s.inputs[j].len())?, d)
        } else if nq > 0 {
            (s.mixed.view(row(j) * d, s.mixed.len() - row(j) * d)?, x_stride)
        } else {
            (s.cur.view(0, s.cur.len())?, d)
        };
        ex.ops.rms_q8(st, &x, xs, Some(nw), Some(&mut s.normed[j]), &mut s.qa[j], nt, cfg.rms_eps)?;
    }
    if armed
        && batched_ctrl
        && let (Some(li), Some(p)) = (loop_i, s.probe.as_deref_mut())
    {
        let rank = w.loops[li].down[0].n as usize;
        p.f(st, "ctrl.cd", &s.cd, 0, 3 * rank, Vec::new())?;
        for (j, x) in s.inputs.iter().enumerate() {
            p.f(st, "ctrl.out", x, 0, d, vec![j as u32])?;
        }
    }
    let (ik, iv) = if n_in == 3 { (1, 2) } else { (0, 0) };
    if tap {
        st.synchronize()?;
        let mut v = s.normed[0].to_vec()?;
        v.truncate(d);
        s.dbg.as_mut().unwrap().push(("attn_in_normed", v));
    }

    // Projections: Q, gate, R, K and V (one launch when decoding); the BF16 sink and tap
    // projections.
    // Decode and small batches: one launch for Q, gate, R, K, V and (as Q8_0 copies) the sink
    // and conv-tap projections.
    let mut side_fused = false;
    let multi_path = ntok == 1 || (small_batch(ntok) && [&l.wq, &l.wgate, &l.wrel, &l.wk, &l.wv].iter().all(|m| m.ty == GemvType::Q8_0 && m.n_mats == 1 && m.mat_scale.is_none()));
    if multi_path {
        let side = side_q8() && s.dbg.is_none();
        let mut items: Vec<MultiItem> = vec![
            MultiItem { m: &l.wq, x: &s.qa[0], y: &mut s.q },
            MultiItem { m: &l.wgate, x: &s.qa[0], y: &mut s.gate },
            MultiItem { m: &l.wrel, x: &s.qa[0], y: &mut s.r },
            MultiItem { m: &l.wk, x: &s.qa[ik], y: &mut s.k },
            MultiItem { m: &l.wv, x: &s.qa[iv], y: &mut s.v },
        ];
        if side {
            items.push(MultiItem { m: &l.sink_q8, x: &s.qa[0], y: &mut s.sinks });
            if let (Some(kq), Some(vq)) = (l.kp_q8.as_ref(), l.vp_q8.as_ref()) {
                items.push(MultiItem { m: kq, x: &s.qa[ik], y: &mut s.kdyn });
                items.push(MultiItem { m: vq, x: &s.qa[iv], y: &mut s.vdyn });
            }
        }
        // Decode: one fused GEMV, on tensor cores unless dp4a is asked for (decode_q8_mma).
        if ntok == 1 && s.decode_mma && l.wq.k.is_multiple_of(128) {
            ex.gemv.launch_q_multi_mma(st, &mut items, 1)?;
        } else if ntok == 1 {
            ex.gemv.launch_q_multi(st, &mut items)?;
        } else {
            ex.gemv.launch_q_multi_t(st, &mut items, nt)?;
        }
        side_fused = side;
    } else {
        ex.mm(&l.wq, &s.qa[0], &mut s.q, ntok, &s.items_dense, &s.work[0])?;
        ex.mm(&l.wgate, &s.qa[0], &mut s.gate, ntok, &s.items_dense, &s.work[0])?;
        ex.mm(&l.wrel, &s.qa[0], &mut s.r, ntok, &s.items_dense, &s.work[0])?;
        ex.mm(&l.wk, &s.qa[ik], &mut s.k, ntok, &s.items_dense, &s.work[0])?;
        ex.mm(&l.wv, &s.qa[iv], &mut s.v, ntok, &s.items_dense, &s.work[0])?;
    }
    let (kp, vp) = (l.k_conv_proj.as_ref(), l.v_conv_proj.as_ref());
    if !side_fused {
        ex.mm_f32(&l.sink, &s.normed[0], &mut s.sinks, ntok, &s.zero_ids)?;
        if let (Some(kp), Some(vp)) = (kp, vp) {
            ex.mm_f32(kp, &s.normed[ik], &mut s.kdyn, ntok, &s.zero_ids)?;
            ex.mm_f32(vp, &s.normed[iv], &mut s.vdyn, ntok, &s.zero_ids)?;
        }
    }

    // Short convs (per token in attn_prep when decoding; sequentially over the chunk when
    // prefilling), clips, value norm, log-length scale and the KV append.
    let dyn_heads = kp.map_or(0, |m| m.n / 4);
    // Prefill: convolved K and V go to s.attn and s.o (free until the attention and the output
    // projection), which attn_prep then reads.
    if ntok > 1 {
        let taps_k = kp.map(|m| (&s.kdyn as &dyn Dev<f32>, m.n as usize, dyn_heads));
        let taps_v = vp.map(|m| (&s.vdyn as &dyn Dev<f32>, m.n as usize, dyn_heads));
        // Verify passes: the conv kernels also save their inputs for the rollback (no state update).
        {
            let mut hk = match (hist.as_mut(), hoff) {
                (Some(h), Some(o)) => Some(h.buf.view_mut(o[0], ntok * kvw)?),
                _ => None,
            };
            ex.ops.conv_par(st, &s.k, kvw, &mut s.attn, kvw, nt, &conv.view(0, 3 * kvw)?, &l.k_conv, taps_k, kvw as u32, hk.as_mut().map(|v| v as &mut dyn DevMut<f32>))?;
            if hk.is_none() {
                ex.ops.conv_state(st, &s.k, kvw, nt, &mut conv.view_mut(0, 3 * kvw)?, kvw as u32)?;
            }
        }
        {
            let mut hv = match (hist.as_mut(), hoff) {
                (Some(h), Some(o)) => Some(h.buf.view_mut(o[1], ntok * kvw)?),
                _ => None,
            };
            ex.ops.conv_par(st, &s.v, kvw, &mut s.o, kvw, nt, &conv.view(3 * kvw, 3 * kvw)?, &l.v_conv, taps_v, kvw as u32, hv.as_mut().map(|v| v as &mut dyn DevMut<f32>))?;
            if hv.is_none() {
                ex.ops.conv_state(st, &s.v, kvw, nt, &mut conv.view_mut(3 * kvw, 3 * kvw)?, kvw as u32)?;
            }
        }
    }
    let (kbuf, vbuf): (&mut DeviceBuffer<f32>, &mut DeviceBuffer<f32>) = if ntok > 1 { (&mut s.attn, &mut s.o) } else { (&mut s.k, &mut s.v) };
    ex.ops.attn_prep(
        st,
        AttnPrep {
            q: &mut s.q,
            r: &mut s.r,
            k: kbuf,
            v: vbuf,
            q_scale: &l.q_clip,
            r_scale: &l.rel_clip,
            conv_kv: &mut conv.view_mut(0, 6 * kvw)?,
            k_conv: &l.k_conv,
            v_conv: &l.v_conv,
            kdyn: if ntok == 1 { kp.map(|_| &s.kdyn as &dyn Dev<f32>) } else { None },
            vdyn: if ntok == 1 { vp.map(|_| &s.vdyn as &dyn Dev<f32>) } else { None },
            dyn_heads,
            ntok: nt,
            conv: ntok == 1,
            kc: &mut *kc,
            vc: &mut *vc,
            pos: &s.pos,
            heads,
            kv_heads: kvh,
            head_dim: hd,
            n_rel,
            thr_qk: cfg.qk_clip,
            thr_rel: cfg.rel_clip,
            eps: cfg.rms_eps,
            global: !l.swa,
            log_floor: cfg.log_floor,
            log_alpha: cfg.log_alpha,
            capacity,
            rows,
        },
    )?;
    tap!("Qcur", q);
    tap!("Kcur", k);
    tap!("Vcur", v);
    tap!("Rcur", r);
    tap!("sinks", sinks);
    if armed && let Some(p) = s.probe.as_deref_mut() {
        let hh = cfg.heads * cfg.head_dim;
        p.f(st, "attn.q", &s.q, 0, hh, Vec::new())?;
        p.f(st, "attn.r", &s.r, 0, cfg.heads * n_rel as usize, Vec::new())?;
        p.f(st, "attn.sinks", &s.sinks, 0, cfg.heads, Vec::new())?;
        p.f(st, "attn.gate", &s.gate, 0, hh, Vec::new())?;
    }

    // Attention, output gate, output projection, post conv into the residual stream. The
    // first layer of each block starts a new stream; the old one joins the blocks.
    let shape = AttnShape { heads, kv_heads: kvh, head_dim: hd, n_rel, extent: cfg.rel_extent, window: if l.swa { cfg.n_swa } else { 0 }, capacity, rows };
    use astral_kernels::inkling::ATT_CHUNK;
    let mut gated = false;
    if ntok == 1 && (hd != 128 || n_rel > 32 || std::env::var_os("ASTRAL_ATTN_SIMPLE").is_some()) {
        ex.ops.attn_decode(st, &s.q, &s.r, &s.sinks, &*kc, &*vc, &l.rel_bias, &mut s.attn, &s.pos, shape)?;
    } else if ntok > 1 && !small_batch(ntok) && hd == 128 && n_rel <= 16 && n_rel.is_multiple_of(4) && !split_prefill() {
        ex.ops.attn_prefill(st, &s.q, &s.r, &s.sinks, &*kc, &*vc, &l.rel_bias, &mut s.attn, &s.pos, shape, nt)?;
    } else {
        // Decode: a fixed split count for the whole context (graph-capturable); prefill: enough
        // splits for the chunk's last position.
        // Small batches (verify passes, graph-captured at any position) use the decode split.
        // (capped: each split loops over chunks, so a large context does not launch idle blocks.)
        // Decode: one chunk per split over the whole capacity (measured fastest). Verify passes:
        // a capped split count with each split looping over chunks (no idle blocks at large
        // contexts). Prefill: exact splits for the chunk's last position.
        let chunk_loop = small_batch(ntok);
        // A sliding-window query spans at most the window: its splits past that exit at once.
        let span = if l.swa { capacity.min(cfg.n_swa) } else { capacity };
        let splits = if ntok == 1 {
            span.div_ceil(ATT_CHUNK)
        } else if chunk_loop {
            capacity.div_ceil(ATT_CHUNK).min(graph_splits())
        } else {
            (pos0 + nt).div_ceil(ATT_CHUNK)
        };
        // The merge also applies the output gate and quantises for the O projection (except
        // when the debug taps want the ungated attention output).
        let fuse_gate = s.dbg.is_none();
        let gate = if fuse_gate { Some((&s.gate as &dyn Dev<f32>, &mut s.qa_attn)) } else { None };
        ex.ops.attn_decode_split(st, &s.q, &s.r, &s.sinks, &*kc, &*vc, &l.rel_bias, &mut s.attn_work, &mut s.attn, &s.pos, shape, nt, splits, chunk_loop, gate)?;
        gated = fuse_gate;
    }
    tap!("kqv_out", attn);
    if !gated {
        ex.ops.gate_q8(st, &mut s.attn, &s.gate, &mut s.qa_attn, nt)?;
    }
    if armed && let Some(p) = s.probe.as_deref_mut() {
        p.f(st, "attn.out", &s.attn, 0, cfg.heads * cfg.head_dim, Vec::new())?;
    }
    ex.mm(&l.wo, &s.qa_attn, &mut s.o, ntok, &s.items_dense, &s.work[0])?;
    let starts_block = ip % cfg.attn_res_block == 0;
    if starts_block && *next_row >= cfg.bank_rows() {
        return Err(bad("attention-residual bank full".into()));
    }
    {
        let mut bank_row = if starts_block { Some(s.bank.view_mut(*next_row * rs, rs)?) } else { None };
        let br = bank_row.as_mut().map(|b| b as &mut dyn DevMut<f32>);
        if ntok == 1 {
            ex.ops.conv_residual(st, &s.o, None, nt, &mut conv.view_mut(2 * kvw * 3, 3 * d)?, &l.attn_post_conv, &mut s.cur, br, du)?;
        } else {
            let mut ha = match (hist.as_mut(), hoff) {
                (Some(h), Some(o)) => Some(h.buf.view_mut(o[2], ntok * d)?),
                _ => None,
            };
            ex.ops.conv_res_par(st, &s.o, nt, &conv.view(2 * kvw * 3, 3 * d)?, &l.attn_post_conv, &mut s.cur, br, du, ha.as_mut().map(|v| v as &mut dyn DevMut<f32>))?;
            if ha.is_none() {
                ex.ops.conv_state(st, &s.o, d, nt, &mut conv.view_mut(2 * kvw * 3, 3 * d)?, du)?;
            }
        }
    }
    if starts_block {
        if armed && let Some(p) = s.probe.as_deref_mut() {
            p.row_visit[*next_row] = Some(p.ctx.visit);
        }
        blocks.push(*next_row as u32);
        *next_row += 1;
    }
    tap!("ffn_inp", cur);

    // MoE.
    let fuse_ffn = fuse_mix() && loop_i.is_none() && s.dbg.is_none();
    if fuse_ffn {
        let Scratch { bank, cur, m, mn, qa, .. } = &mut *s;
        let mut outs: [(&dyn Dev<f32>, Option<&mut dyn DevMut<f32>>, &mut QAct); 1] = [(&l.ffn_norm, Some(mn as &mut dyn DevMut<f32>), &mut qa[0])];
        ex.ops.mix_rms_q8(st, &*bank, blocks, rs, &*cur, &l.ffn_res_query, Some(m as &mut dyn DevMut<f32>), du, nt, &mut outs, cfg.rms_eps)?;
    } else {
        ex.ops.mix(st, &s.bank, blocks, rs, &s.cur, &l.ffn_res_query, 1, &mut s.m, du, nt)?;
    }
    if armed && let Some(p) = s.probe.as_deref_mut() {
        p.states(st, "ffn_mix.states", &s.bank, blocks, rs, &s.cur, d)?;
        p.f(st, "ffn_mix.out", &s.m, 0, d, Vec::new())?;
    }
    if let Some(li) = loop_i {
        let lp = &w.loops[li];
        let msg = s.msgs.view(3 * d, s.msgs.len() - 3 * d)?;
        if ntok == 1 || small_batch(ntok) {
            ex.ops.ctrl_down(st, &lp.down[3], &s.m, d, &msg, 4 * d, &mut s.cd, du, nt)?;
            ex.ops.ctrl_combine(st, &lp.gain_forget[3], &s.cd, &s.m, d, &msg, 4 * d, &mut s.ctmp, du, nt)?;
        } else {
            ex.ops.ctrl_t(st, &lp.down[3], &lp.gain_forget[3], &s.m, d, &msg, 4 * d, &mut s.crms, &mut s.cd, &mut s.gu, &mut s.ctmp, du, nt)?;
        }
        if armed && let Some(p) = s.probe.as_deref_mut() {
            p.f(st, "ffn_ctrl.cd", &s.cd, 0, lp.down[3].n as usize, Vec::new())?;
            p.f(st, "ffn_ctrl.out", &s.ctmp, 0, d, Vec::new())?;
        }
    }
    if !fuse_ffn {
        let ffn_x: &DeviceBuffer<f32> = if loop_i.is_some() { &s.ctmp } else { &s.m };
        ex.ops.rms_q8(st, ffn_x, d, Some(&l.ffn_norm), Some(&mut s.mn), &mut s.qa[0], nt, cfg.rms_eps)?;
    }
    tap!("ffn_normed", mn);
    let (ne, k) = (cfg.n_expert as u32, cfg.n_expert_used as u32);
    if l.router.ty == GemvType::F32 && ntok <= QITEM_ROWS && s.dbg.is_none() && std::env::var_os("ASTRAL_NO_FUSE_ROUTER").is_none() {
        // Router GEMV + top-k in one kernel (the last block per token selects).
        let bias = l.probs_b.as_ref().map(|b| b as &dyn Dev<f32>);
        ex.ops.router_fused(st, &l.router, &s.mn, &mut s.rlogits, bias, &mut s.ids, &mut s.w, k, cfg.expert_scale, nt, &mut s.router_cnt)?;
    } else {
        ex.mm_f32(&l.router, &s.mn, &mut s.rlogits, ntok, &s.zero_ids)?;
        tap!("ffn_moe_logits", rlogits);
        ex.ops.router_topk(st, &s.rlogits, l.probs_b.as_ref(), &mut s.ids, &mut s.w, ne, k, cfg.expert_scale, nt)?;
    }
    if armed && let Some(p) = s.probe.as_deref_mut() {
        p.f(st, "router.logits", &s.rlogits, 0, ne as usize, Vec::new())?;
        p.i(st, "router.ids", &s.ids, 0, k as usize)?;
        p.f(st, "router.w", &s.w, 0, k as usize, Vec::new())?;
    }
    let pairs = ntok * k as usize;
    if ntok == 1 {
        let ids = s.ids.view(0, k as usize)?;
        ex.gemv.launch_q(st, &l.gate_up, &s.qa[0], &mut s.gu, Some(&ids as &dyn Dev<i32>))?;
        ex.ops.situ_q8(st, &s.gu, &mut s.qa_ff, k, cfg.situ_beta, cfg.situ_lb)?;
        ex.gemv.launch_q(st, &l.down, &s.qa_ff, &mut s.y, Some(&ids as &dyn Dev<i32>))?;
    } else {
        let p32 = pairs as u32;
        if small_batch(ntok) {
            // Few tokens (verify passes): dp4a grouped GEMV, each used expert read once, grouped
            // on the GPU (graph-capturable).
            let mi = astral_kernels::inkling::moe_items_bound(p32, cfg.n_expert as u32);
            ex.ops.moe_group_q(st, &s.ids, p32, k, cfg.n_expert as u32, &mut s.items_gu, &mut s.items_dn, mi)?;
            ex.gemv.launch_items(st, &l.gate_up, &s.qa[0], nt, &mut s.gu, p32, &s.items_gu, mi)?;
            ex.ops.situ_q8(st, &s.gu, &mut s.qa_ff, p32, cfg.situ_beta, cfg.situ_lb)?;
            ex.gemv.launch_items(st, &l.down, &s.qa_ff, p32, &mut s.y, p32, &s.items_dn, mi)?;
        } else if use_mma() && !host_group() {
            // Group the (token, expert) pairs by expert on the GPU: no host sync.
            // Few tokens (verify passes): launch only as many items as can be used.
            let max_items = ((s.work[1].items.len() / 3) as u32).min((p32.div_ceil(64) + cfg.n_expert as u32).min(p32));
            let [_, w1, w2] = &mut s.work;
            ex.ops.moe_group(st, &s.ids, p32, k, cfg.n_expert as u32, (&mut w1.items, &mut w1.xidx, &mut w1.yidx), (&mut w2.items, &mut w2.xidx, &mut w2.yidx), max_items)?;
            let (w1, w2) = (&s.work[1], &s.work[2]);
            ex.gemv.launch_mma(st, &l.gate_up, &s.qa[0], nt, &mut s.gu, p32, &w1.items, max_items, &w1.xidx, &w1.yidx, p32)?;
            ex.ops.situ_q8(st, &s.gu, &mut s.qa_ff, p32, cfg.situ_beta, cfg.situ_lb)?;
            ex.gemv.launch_mma(st, &l.down, &s.qa_ff, p32, &mut s.y, p32, &w2.items, max_items, &w2.xidx, &w2.yidx, p32)?;
        } else {
            // Group the (token, expert) pairs by expert on the host, so each expert's weights are
            // read once per 8 (dp4a) or 64 (tensor cores) of its tokens.
            st.synchronize()?;
            let mut ids = s.ids.to_vec()?;
            ids.truncate(pairs);
            if use_mma() {
                let (mut gu, mut dn) = (MmaWork::default(), MmaWork::default());
                let mut by_e: Vec<Vec<u32>> = vec![Vec::new(); cfg.n_expert];
                for (p, &e) in ids.iter().enumerate() {
                    by_e[(e.max(0) as usize).min(cfg.n_expert - 1)].push(p as u32);
                }
                for (e, ps) in by_e.iter().enumerate() {
                    if ps.is_empty() {
                        continue;
                    }
                    let g: Vec<(u32, u32)> = ps.iter().map(|&p| (p / k, p)).collect();
                    let dd: Vec<(u32, u32)> = ps.iter().map(|&p| (p, p)).collect();
                    gu.push(e as u32, &g);
                    dn.push(e as u32, &dd);
                }
                s.work[1].upload(&gu)?;
                s.work[2].upload(&dn)?;
                let (w1, w2) = (&s.work[1], &s.work[2]);
                ex.gemv.launch_mma(st, &l.gate_up, &s.qa[0], nt, &mut s.gu, p32, &w1.items, w1.n_items, &w1.xidx, &w1.yidx, w1.idx_len)?;
                ex.ops.situ_q8(st, &s.gu, &mut s.qa_ff, p32, cfg.situ_beta, cfg.situ_lb)?;
                ex.gemv.launch_mma(st, &l.down, &s.qa_ff, p32, &mut s.y, p32, &w2.items, w2.n_items, &w2.xidx, &w2.yidx, w2.idx_len)?;
            } else {
                let (gu_items, dn_items) = moe_items(&ids, k as usize, cfg.n_expert);
                for (items, buf) in [(&gu_items, &mut s.items_gu), (&dn_items, &mut s.items_dn)] {
                    let mut words = QItem::words(items);
                    if words.len() > buf.len() {
                        return Err(bad("moe: too many work items".into()));
                    }
                    words.resize(buf.len(), 0);
                    buf.copy_from_slice(&words)?;
                }
                ex.gemv.launch_items(st, &l.gate_up, &s.qa[0], nt, &mut s.gu, p32, &s.items_gu, gu_items.len() as u32)?;
                ex.ops.situ_q8(st, &s.gu, &mut s.qa_ff, p32, cfg.situ_beta, cfg.situ_lb)?;
                ex.gemv.launch_items(st, &l.down, &s.qa_ff, p32, &mut s.y, p32, &s.items_dn, dn_items.len() as u32)?;
            }
        }
    }
    // Combine, post conv and the residual add.
    let wv = s.w.view(0, pairs)?;
    let st_off = (2 * kvw + d) * 3;
    if ntok == 1 {
        ex.ops.conv_residual(st, &s.y, Some((&wv as &dyn Dev<f32>, k)), nt, &mut conv.view_mut(st_off, 3 * d)?, &l.ffn_post_conv, &mut s.cur, None, du)
    } else {
        ex.ops.moe_combine_t(st, &s.y, &wv, &mut s.o, du, k, nt)?;
        let mut hf = match (hist.as_mut(), hoff) {
            (Some(h), Some(o)) => Some(h.buf.view_mut(o[3], ntok * d)?),
            _ => None,
        };
        ex.ops.conv_res_par(st, &s.o, nt, &conv.view(st_off, 3 * d)?, &l.ffn_post_conv, &mut s.cur, None, du, hf.as_mut().map(|v| v as &mut dyn DevMut<f32>))?;
        if hf.is_none() {
            ex.ops.conv_state(st, &s.o, d, nt, &mut conv.view_mut(st_off, 3 * d)?, du)?;
        }
        Ok(())
    }
}
