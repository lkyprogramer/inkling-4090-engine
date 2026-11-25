// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Synthetic Minds LTD
//! Typed launchers for `cu/qwen.cu` (Qwen3.8-27B / `qwen3_5`). Each checks buffer sizes before
//! launching; device-side indices (token, position) are clamped by the kernels.

use crate::ops::QAct;
use crate::KernelRegistry;
use astral_core::{AstralError, Result};
use astral_gpu::{param, Dev, DevMut, Function, LaunchConfig, Stream};

fn invalid(m: String) -> AstralError {
    AstralError::Invalid(m)
}

fn need(what: &str, have: usize, want: usize) -> Result<()> {
    if have < want {
        return Err(invalid(format!("{what}: buffer has {have} elements, need {want}")));
    }
    Ok(())
}

/// Head size of the Gated DeltaNet kernels (`GDN_D`) and of the attention kernels (`QA_HD`).
pub const GDN_D: u32 = 128;
/// Dynamic shared memory of the chunked delta-rule kernel (`astral_qw_gdn_chunk`).
const GDN_CHUNK_SMEM: u32 = 4 * (128 * 64 + 3 * 64 * 132 + 2 * 64 * 68 + 2 * 64 * 65 + 4 * 64);
/// Dynamic shared memory of the two-phase delta rule (prep: K, Q [64][132], V [64][68], M
/// [64][65], 3 x 64; scan: S [128][32], K, Q, W [64][132], U [64][36], P [64][65], 64).
const GDN_PREP2_SMEM: u32 = 4 * (2 * 64 * 132 + 64 * 68 + 64 * 65 + 3 * 64);
const GDN_SCAN2_SMEM: u32 = 4 * (128 * 32 + 3 * 64 * 132 + 64 * 36 + 64 * 65 + 64);

/// Floats of two-phase delta-rule scratch for `ntok` tokens and `vh` value heads (W, U: 64 x 128,
/// P: 64 x 64, G: 64 per head and 64-token chunk).
pub fn gdn_scratch(ntok: u32, vh: u32) -> usize {
    (vh * ntok.div_ceil(64)) as usize * (64 * 128 * 2 + 64 * 64 + 64)
}

/// Prefill chunks use the chunked kernel from this many tokens (the recurrent one below).
const GDN_CHUNK_MIN: u32 = 32;
pub const ATTN_HD: u32 = 256;
pub const ATTN_CHUNK: u32 = 64;
/// Dynamic shared memory of `astral_qw_attn_prefill`: K [64][264] + V half [64][136] (f16).
const ATTN_PREFILL_SMEM: u32 = 2 * 64 * (264 + 136);

/// Floats of split-attention workspace for `ntok` queries, `heads` and `max_splits`.
pub fn attn_workspace(ntok: u32, heads: u32, max_splits: u32) -> usize {
    ntok as usize * (heads * max_splits * (ATTN_HD + 2)) as usize
}

pub struct QwenOps {
    embed: Function,
    conv_silu: Function,
    conv_state: Function,
    gdn_prep: Function,
    gdn_recur: Function,
    gdn_chunk: Function,
    gdn_prep2: Function,
    gdn_scan2: Function,
    gated_norm: Function,
    attn_prep: Function,
    attn_split: Function,
    attn_merge: Function,
    attn_prefill: Function,
    swiglu: Function,
    add: Function,
    add_rms: Function,
}

/// Linear-attention (Gated DeltaNet) geometry.
#[derive(Debug, Clone, Copy)]
pub struct GdnShape {
    pub k_heads: u32,
    pub v_heads: u32,
    pub head_dim: u32,
}

impl GdnShape {
    pub fn qkv_width(&self) -> u32 {
        (2 * self.k_heads + self.v_heads) * self.head_dim
    }
    fn check(&self) -> Result<()> {
        if self.head_dim != GDN_D || self.k_heads == 0 || !self.v_heads.is_multiple_of(self.k_heads) {
            return Err(invalid(format!("gdn: unsupported shape {self:?} (head_dim {GDN_D})")));
        }
        Ok(())
    }
}

impl QwenOps {
    pub fn new(reg: &KernelRegistry) -> Result<Self> {
        let f = |n: &str| reg.function("qwen", n);
        Ok(Self {
            embed: f("astral_qw_embed_bf16")?,
            conv_silu: f("astral_qw_conv_silu")?,
            conv_state: f("astral_qw_conv_state")?,
            gdn_prep: f("astral_qw_gdn_prep")?,
            gdn_recur: f("astral_qw_gdn_recur")?,
            gdn_prep2: {
                let fp = f("astral_qw_gdn_prep2")?;
                fp.set_max_dynamic_shared(GDN_PREP2_SMEM)?;
                fp
            },
            gdn_scan2: {
                let fs = f("astral_qw_gdn_scan2")?;
                fs.set_max_dynamic_shared(GDN_SCAN2_SMEM)?;
                fs
            },
            gdn_chunk: {
                let fc = f("astral_qw_gdn_chunk")?;
                fc.set_max_dynamic_shared(GDN_CHUNK_SMEM)?;
                fc
            },
            gated_norm: f("astral_qw_gated_norm_q8")?,
            attn_prep: f("astral_qw_attn_prep")?,
            attn_split: f("astral_qw_attn_split")?,
            attn_merge: f("astral_qw_attn_merge")?,
            attn_prefill: {
                let fa = f("astral_qw_attn_prefill")?;
                fa.set_max_dynamic_shared(ATTN_PREFILL_SMEM)?;
                fa
            },
            swiglu: f("astral_qw_swiglu_q8")?,
            add: f("astral_qw_add")?,
            add_rms: f("astral_qw_add_rms_q8")?,
        })
    }

    fn go(f: &Function, s: &Stream, cfg: LaunchConfig, p: &mut [*mut std::ffi::c_void]) -> Result<()> {
        // SAFETY: every caller builds `p` to match the kernel's C signature in qwen.cu and checks
        // buffer sizes first; device-side indices are clamped by the kernels.
        unsafe { f.launch(cfg, s, p) }
    }

    /// `out[t] = table[token[t]]` from a BF16 table of `rows x dim`.
    pub fn embed(&self, s: &Stream, table: &impl Dev<u16>, rows: u32, dim: u32, token: &impl Dev<i32>, out: &mut impl DevMut<f32>, ntok: u32) -> Result<()> {
        need("embed table", table.dlen(), (rows * dim) as usize)?;
        need("embed token", token.dlen(), ntok as usize)?;
        need("embed out", out.dlen(), (ntok * dim) as usize)?;
        let (mut pt, mut pk, mut po, mut d, mut r) = (table.dptr(), token.dptr(), out.dptr(), dim, rows);
        let mut p = [param(&mut pt), param(&mut pk), param(&mut po), param(&mut d), param(&mut r)];
        Self::go(&self.embed, s, LaunchConfig { grid: (ntok, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// Causal 4-tap conv + SiLU of `src` (rows `src_stride` apart) into `dst` (`[t][channels]`),
    /// then the state update.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_silu(&self, s: &Stream, src: &impl Dev<f32>, src_stride: usize, dst: &mut impl DevMut<f32>, ntok: u32, state: &mut impl DevMut<f32>, w: &impl Dev<f32>, channels: u32) -> Result<()> {
        let (c, nt) = (channels as usize, ntok as usize);
        if nt == 0 || src_stride < c {
            return Err(invalid("conv: need tokens > 0 and src_stride >= channels".into()));
        }
        need("conv src", src.dlen(), (nt - 1) * src_stride + c)?;
        need("conv dst", dst.dlen(), nt * c)?;
        need("conv state", state.dlen(), 3 * c)?;
        need("conv weight", w.dlen(), 4 * c)?;
        let (mut ps, mut ss, mut pd, mut n, mut pst, mut pw, mut ch) = (src.dptr(), src_stride as u32, dst.dptr(), ntok, state.dptr(), w.dptr(), channels);
        let mut p = [param(&mut ps), param(&mut ss), param(&mut pd), param(&mut n), param(&mut pst), param(&mut pw), param(&mut ch)];
        Self::go(&self.conv_silu, s, LaunchConfig { grid: (channels.div_ceil(128), ntok, 1), block: (128, 1, 1), shared_mem_bytes: 0 }, &mut p)?;
        let (mut ps, mut ss, mut n, mut pst, mut ch) = (src.dptr(), src_stride as u32, ntok, state.dptr(), channels);
        let mut p = [param(&mut ps), param(&mut ss), param(&mut n), param(&mut pst), param(&mut ch)];
        Self::go(&self.conv_state, s, LaunchConfig { grid: (channels.div_ceil(128), 1, 1), block: (128, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// Gated DeltaNet prep (in place on the convolved qkv) and the recurrence over `ntok`
    /// tokens; `out` is `[t][v_heads][head_dim]`, `state` `[v_heads][head_dim][head_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn(
        &self,
        s: &Stream,
        sh: GdnShape,
        qkv: &mut impl DevMut<f32>,
        b: &impl Dev<f32>,
        a: &impl Dev<f32>,
        ab_stride: usize,
        neg_a: &impl Dev<f32>,
        dt_bias: &impl Dev<f32>,
        beta: &mut impl DevMut<f32>,
        g: &mut impl DevMut<f32>,
        state: &mut impl DevMut<f32>,
        out: &mut impl DevMut<f32>,
        ntok: u32,
        scratch: Option<&mut dyn DevMut<f32>>,
    ) -> Result<()> {
        sh.check()?;
        let (nt, vh, d) = (ntok as usize, sh.v_heads as usize, sh.head_dim as usize);
        need("gdn qkv", qkv.dlen(), nt * sh.qkv_width() as usize)?;
        if nt == 0 || ab_stride < vh {
            return Err(invalid("gdn: need tokens > 0 and ab_stride >= v_heads".into()));
        }
        need("gdn b", b.dlen(), (nt - 1) * ab_stride + vh)?;
        need("gdn a", a.dlen(), (nt - 1) * ab_stride + vh)?;
        need("gdn beta", beta.dlen(), nt * vh)?;
        need("gdn g", g.dlen(), nt * vh)?;
        need("gdn neg_a", neg_a.dlen(), vh)?;
        need("gdn dt_bias", dt_bias.dlen(), vh)?;
        need("gdn state", state.dlen(), vh * d * d)?;
        need("gdn out", out.dlen(), nt * vh * d)?;
        let (mut pq, mut qs, mut pb, mut pa, mut abs, mut pn, mut pdt, mut pbe, mut pg, mut kh, mut dk, mut vhh) =
            (qkv.dptr(), sh.qkv_width(), b.dptr(), a.dptr(), ab_stride as u32, neg_a.dptr(), dt_bias.dptr(), beta.dptr(), g.dptr(), sh.k_heads, sh.head_dim, sh.v_heads);
        let mut p = [
            param(&mut pq),
            param(&mut qs),
            param(&mut pb),
            param(&mut pa),
            param(&mut abs),
            param(&mut pn),
            param(&mut pdt),
            param(&mut pbe),
            param(&mut pg),
            param(&mut kh),
            param(&mut dk),
            param(&mut vhh),
        ];
        Self::go(&self.gdn_prep, s, LaunchConfig { grid: (sh.k_heads, ntok, 1), block: (sh.head_dim, 1, 1), shared_mem_bytes: 0 }, &mut p)?;
        let (mut pq, mut qs, mut pbe, mut pg, mut pst, mut po, mut nn, mut kh, mut vhh) = (qkv.dptr(), sh.qkv_width(), beta.dptr(), g.dptr(), state.dptr(), out.dptr(), ntok, sh.k_heads, sh.v_heads);
        let mut p = [param(&mut pq), param(&mut qs), param(&mut pbe), param(&mut pg), param(&mut pst), param(&mut po), param(&mut nn), param(&mut kh), param(&mut vhh)];
        // Two-phase (parallel prep + sequential scan) measured slower than one pass on H100
        // (scan 632 us vs one pass 507 us per 512-token layer): opt-in only.
        if ntok >= GDN_CHUNK_MIN && std::env::var_os("ASTRAL_GDN_RECUR").is_none() && std::env::var_os("ASTRAL_GDN_TWOPASS").is_some() && scratch.is_some() {
            let sc = scratch.expect("checked");
            need("gdn scratch", sc.dlen(), gdn_scratch(ntok, sh.v_heads))?;
            let nch = ntok.div_ceil(64);
            let per = (sh.v_heads * nch) as u64 * 64;
            let base = sc.dptr();
            let (wsc, usc) = (base, base + per * 128 * 4);
            let (psc, gsc) = (usc + per * 128 * 4, usc + per * 128 * 4 + per * 64 * 4);
            let (mut pq, mut qs, mut pbe, mut pg, mut pw, mut pu, mut pp, mut pgs, mut n, mut kh, mut vhh) =
                (qkv.dptr(), sh.qkv_width(), beta.dptr(), g.dptr(), wsc, usc, psc, gsc, ntok, sh.k_heads, sh.v_heads);
            let mut p =
                [param(&mut pq), param(&mut qs), param(&mut pbe), param(&mut pg), param(&mut pw), param(&mut pu), param(&mut pp), param(&mut pgs), param(&mut n), param(&mut kh), param(&mut vhh)];
            Self::go(&self.gdn_prep2, s, LaunchConfig { grid: (sh.v_heads, nch, 2), block: (256, 1, 1), shared_mem_bytes: GDN_PREP2_SMEM }, &mut p)?;
            let (mut pq, mut qs, mut pw, mut pu, mut pp, mut pgs, mut pst, mut po, mut n, mut kh, mut vhh, mut nc) =
                (qkv.dptr(), sh.qkv_width(), wsc, usc, psc, gsc, state.dptr(), out.dptr(), ntok, sh.k_heads, sh.v_heads, nch);
            let mut p = [
                param(&mut pq),
                param(&mut qs),
                param(&mut pw),
                param(&mut pu),
                param(&mut pp),
                param(&mut pgs),
                param(&mut pst),
                param(&mut po),
                param(&mut n),
                param(&mut kh),
                param(&mut vhh),
                param(&mut nc),
            ];
            Self::go(&self.gdn_scan2, s, LaunchConfig { grid: (sh.v_heads, GDN_D / 32, 1), block: (256, 1, 1), shared_mem_bytes: GDN_SCAN2_SMEM }, &mut p)
        } else if ntok >= GDN_CHUNK_MIN && std::env::var_os("ASTRAL_GDN_RECUR").is_none() {
            Self::go(&self.gdn_chunk, s, LaunchConfig { grid: (sh.v_heads, GDN_D / 64, 1), block: (256, 1, 1), shared_mem_bytes: GDN_CHUNK_SMEM }, &mut p)
        } else {
            Self::go(&self.gdn_recur, s, LaunchConfig { grid: (sh.v_heads, GDN_D / 32, 1), block: (128, 1, 1), shared_mem_bytes: 0 }, &mut p)
        }
    }

    /// `y = rms_norm(x_h) * w * silu(z_h)` per value head, quantised into `qa` (`qa.k = vh * dv`).
    #[allow(clippy::too_many_arguments)]
    pub fn gated_norm_q8(&self, s: &Stream, x: &impl Dev<f32>, z: &impl Dev<f32>, z_stride: usize, w: &impl Dev<f32>, qa: &mut QAct, vh: u32, dv: u32, eps: f32, ntok: u32) -> Result<()> {
        if dv != GDN_D || qa.k != vh * dv {
            return Err(invalid(format!("gated_norm: dv {dv}, activations k {} != {}", qa.k, vh * dv)));
        }
        let n = (ntok * vh * dv) as usize;
        need("gated_norm x", x.dlen(), n)?;
        if ntok == 0 || z_stride < (vh * dv) as usize {
            return Err(invalid("gated_norm: need tokens > 0 and z_stride >= vh * dv".into()));
        }
        need("gated_norm z", z.dlen(), (ntok as usize - 1) * z_stride + (vh * dv) as usize)?;
        need("gated_norm w", w.dlen(), dv as usize)?;
        let (mut pq, mut pd, mut ph) = qa.planes_for(ntok)?;
        let (mut px, mut pz, mut zs, mut pw, mut v, mut d, mut e) = (x.dptr(), z.dptr(), z_stride as u32, w.dptr(), vh, dv, eps);
        let mut p = [param(&mut px), param(&mut pz), param(&mut zs), param(&mut pw), param(&mut pq), param(&mut pd), param(&mut ph), param(&mut v), param(&mut d), param(&mut e)];
        Self::go(&self.gated_norm, s, LaunchConfig { grid: (vh, ntok, 1), block: (dv, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// Full-attention prep (q/k RMS norms, partial NeoX RoPE, KV append) for `ntok` tokens. `qkv`
    /// rows (`row_stride` apart) hold [heads x (query, gate) | kvh x k | kvh x v].
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prep(
        &self,
        s: &Stream,
        qkv: &impl Dev<f32>,
        row_stride: usize,
        qn: &impl Dev<f32>,
        kn: &impl Dev<f32>,
        q_out: &mut impl DevMut<f32>,
        gate_out: &mut impl DevMut<f32>,
        kc: &mut impl DevMut<u16>,
        vc: &mut impl DevMut<u16>,
        pos: &impl Dev<i32>,
        heads: u32,
        kvh: u32,
        rot: u32,
        theta: f32,
        eps: f32,
        capacity: u32,
        ntok: u32,
    ) -> Result<()> {
        let hd = ATTN_HD;
        let width = ((2 * heads + 2 * kvh) * hd) as usize;
        if kvh == 0 || !heads.is_multiple_of(kvh) || rot > hd || !rot.is_multiple_of(2) || capacity == 0 || ntok == 0 || row_stride < width {
            return Err(invalid(format!("attn_prep: heads {heads} kv {kvh} rot {rot} stride {row_stride}")));
        }
        let nt = ntok as usize;
        need("attn qkv", qkv.dlen(), (nt - 1) * row_stride + width)?;
        need("attn q norm", qn.dlen(), hd as usize)?;
        need("attn k norm", kn.dlen(), hd as usize)?;
        need("attn q out", q_out.dlen(), nt * (heads * hd) as usize)?;
        need("attn gate out", gate_out.dlen(), nt * (heads * hd) as usize)?;
        need("attn k cache", kc.dlen(), (capacity * kvh * hd) as usize)?;
        need("attn v cache", vc.dlen(), (capacity * kvh * hd) as usize)?;
        need("attn pos", pos.dlen(), 1)?;
        let kp = qkv.dptr() + (2 * heads * hd) as u64 * 4;
        let vp = kp + (kvh * hd) as u64 * 4;
        let (mut a, mut b, mut c, mut rs, mut d, mut e, mut f, mut g, mut h, mut i, mut j) =
            (qkv.dptr(), kp, vp, row_stride as u32, qn.dptr(), kn.dptr(), q_out.dptr(), gate_out.dptr(), kc.dptr(), vc.dptr(), pos.dptr());
        let (mut hh, mut kk, mut dd, mut rr, mut th, mut ep, mut cap) = (heads, kvh, hd, rot, theta, eps, capacity);
        let mut p = [
            param(&mut a),
            param(&mut b),
            param(&mut c),
            param(&mut rs),
            param(&mut d),
            param(&mut e),
            param(&mut f),
            param(&mut g),
            param(&mut h),
            param(&mut i),
            param(&mut j),
            param(&mut hh),
            param(&mut kk),
            param(&mut dd),
            param(&mut rr),
            param(&mut th),
            param(&mut ep),
            param(&mut cap),
        ];
        Self::go(&self.attn_prep, s, LaunchConfig { grid: (heads + 2 * kvh, ntok, 1), block: (hd, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// Causal split-K attention (head_dim 256, no sink) for `ntok` queries at positions `pos[0]..`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        s: &Stream,
        q: &impl Dev<f32>,
        kc: &impl Dev<u16>,
        vc: &impl Dev<u16>,
        work: &mut impl DevMut<f32>,
        out: &mut impl DevMut<f32>,
        pos: &impl Dev<i32>,
        heads: u32,
        kvh: u32,
        capacity: u32,
        ntok: u32,
        max_splits: u32,
    ) -> Result<()> {
        if kvh == 0 || !heads.is_multiple_of(kvh) || capacity == 0 || max_splits == 0 || ntok == 0 {
            return Err(invalid(format!("attention: heads {heads} kv {kvh} splits {max_splits}")));
        }
        let nt = ntok as usize;
        need("attn q", q.dlen(), nt * (heads * ATTN_HD) as usize)?;
        need("attn k cache", kc.dlen(), (capacity * kvh * ATTN_HD) as usize)?;
        need("attn v cache", vc.dlen(), (capacity * kvh * ATTN_HD) as usize)?;
        need("attn work", work.dlen(), attn_workspace(ntok, heads, max_splits))?;
        need("attn out", out.dlen(), nt * (heads * ATTN_HD) as usize)?;
        need("attn pos", pos.dlen(), 1)?;
        let (mut pq, mut pk, mut pv, mut pw, mut pp, mut hh, mut kk, mut cap, mut ms) = (q.dptr(), kc.dptr(), vc.dptr(), work.dptr(), pos.dptr(), heads, kvh, capacity, max_splits);
        let mut sc = 1.0f32 / (ATTN_HD as f32).sqrt();
        let mut p = [param(&mut pq), param(&mut pk), param(&mut pv), param(&mut pw), param(&mut pp), param(&mut hh), param(&mut kk), param(&mut sc), param(&mut cap), param(&mut ms)];
        Self::go(&self.attn_split, s, LaunchConfig { grid: (heads, max_splits, ntok), block: (128, 1, 1), shared_mem_bytes: 0 }, &mut p)?;
        let (mut pw, mut po, mut pp, mut cap, mut ms) = (work.dptr(), out.dptr(), pos.dptr(), capacity, max_splits);
        let mut p = [param(&mut pw), param(&mut po), param(&mut pp), param(&mut cap), param(&mut ms)];
        Self::go(&self.attn_merge, s, LaunchConfig { grid: (heads, ntok, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// `silu(g) * u` over `ntok` rows of `qa.k`, quantised into `qa`; `gu` rows (`stride` apart)
    /// hold [g (qa.k) | u (qa.k)].
    pub fn swiglu_q8(&self, s: &Stream, gu: &impl Dev<f32>, stride: usize, qa: &mut QAct, ntok: u32) -> Result<()> {
        let n = qa.k as usize;
        if ntok == 0 || stride < 2 * n {
            return Err(invalid("swiglu: need tokens > 0 and stride >= 2 n".into()));
        }
        need("swiglu gu", gu.dlen(), (ntok as usize - 1) * stride + 2 * n)?;
        let (mut pq, mut pd, mut ph) = qa.planes_for(ntok)?;
        let total = n as u64 * ntok as u64;
        let (mut pg, mut pu, mut nn, mut ss, mut tt) = (gu.dptr(), gu.dptr() + n as u64 * 4, n as u32, stride as u32, total);
        let mut p = [param(&mut pg), param(&mut pu), param(&mut nn), param(&mut ss), param(&mut pq), param(&mut pd), param(&mut ph), param(&mut tt)];
        if !n.is_multiple_of(32) {
            return Err(invalid(format!("swiglu: n {n} must be a multiple of 32")));
        }
        Self::go(&self.swiglu, s, LaunchConfig { grid: ((n as u32).div_ceil(256), ntok, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// `x += y` over `n` floats.
    pub fn add(&self, s: &Stream, x: &mut impl DevMut<f32>, y: &impl Dev<f32>, n: usize) -> Result<()> {
        need("add x", x.dlen(), n)?;
        need("add y", y.dlen(), n)?;
        let (mut px, mut py, mut nn) = (x.dptr(), y.dptr(), n as u64);
        let mut p = [param(&mut px), param(&mut py), param(&mut nn)];
        Self::go(&self.add, s, LaunchConfig { grid: ((n as u32).div_ceil(256).clamp(1, 2048), 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }

    /// Causal flash-attention prefill (head_dim 256, no sink) on tensor cores for `ntok` queries
    /// at positions `pos[0]..`; keys already in the cache. No workspace.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill(
        &self,
        s: &Stream,
        q: &impl Dev<f32>,
        kc: &impl Dev<u16>,
        vc: &impl Dev<u16>,
        out: &mut impl DevMut<f32>,
        pos: &impl Dev<i32>,
        heads: u32,
        kvh: u32,
        capacity: u32,
        ntok: u32,
    ) -> Result<()> {
        if kvh == 0 || !heads.is_multiple_of(kvh) || capacity == 0 || ntok == 0 {
            return Err(invalid(format!("attention prefill: heads {heads} kv {kvh}")));
        }
        let nt = ntok as usize;
        need("attn q", q.dlen(), nt * (heads * ATTN_HD) as usize)?;
        need("attn k cache", kc.dlen(), (capacity * kvh * ATTN_HD) as usize)?;
        need("attn v cache", vc.dlen(), (capacity * kvh * ATTN_HD) as usize)?;
        need("attn out", out.dlen(), nt * (heads * ATTN_HD) as usize)?;
        need("attn pos", pos.dlen(), 1)?;
        let (mut pq, mut pk, mut pv, mut po, mut pp, mut n, mut hh, mut kk, mut cap) = (q.dptr(), kc.dptr(), vc.dptr(), out.dptr(), pos.dptr(), ntok, heads, kvh, capacity);
        let mut sc = 1.0f32 / (ATTN_HD as f32).sqrt();
        let mut p = [param(&mut pq), param(&mut pk), param(&mut pv), param(&mut po), param(&mut pp), param(&mut n), param(&mut hh), param(&mut kk), param(&mut sc), param(&mut cap)];
        Self::go(&self.attn_prefill, s, LaunchConfig { grid: (ntok.div_ceil(64), heads, 2), block: (128, 1, 1), shared_mem_bytes: ATTN_PREFILL_SMEM }, &mut p)
    }

    /// `x += add`, then `y = rms_norm(x) * w` (y optional) quantised into `qa`, for `ntok` rows of
    /// `qa.k`: the residual add fused into the following norm.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rms_q8(&self, s: &Stream, x: &mut impl DevMut<f32>, add: &impl Dev<f32>, w: &impl Dev<f32>, y: Option<&mut dyn DevMut<f32>>, qa: &mut QAct, ntok: u32, eps: f32) -> Result<()> {
        let dim = qa.k;
        let n = (dim * ntok) as usize;
        need("add_rms x", x.dlen(), n)?;
        need("add_rms add", add.dlen(), n)?;
        need("add_rms w", w.dlen(), dim as usize)?;
        let py = match y {
            Some(y) => {
                need("add_rms y", y.dlen(), n)?;
                y.dptr()
            }
            None => 0,
        };
        let (mut pq, mut pd, mut ph) = qa.planes_for(ntok)?;
        let (mut px, mut pa, mut pw, mut py, mut d, mut e) = (x.dptr(), add.dptr(), w.dptr(), py, dim, eps);
        let mut p = [param(&mut px), param(&mut pa), param(&mut pw), param(&mut py), param(&mut pq), param(&mut pd), param(&mut ph), param(&mut d), param(&mut e)];
        let block = (dim / 4).next_multiple_of(32).clamp(32, 1024);
        Self::go(&self.add_rms, s, LaunchConfig { grid: (ntok, 1, 1), block: (block, 1, 1), shared_mem_bytes: 0 }, &mut p)
    }
}
