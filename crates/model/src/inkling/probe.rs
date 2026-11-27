// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 inkling-4090-engine contributors
//! Read-only probes of the looped forward pass (see docs/trace.md).
//!
//! A probed decode step runs eagerly with exactly the kernels of the graph replay. Hooks copy
//! small device buffers (block-mix states and outputs, cross-pass messages, controller inputs
//! and outputs, router logits and picks, attention inputs and output, the stream after every
//! visit and an optional raw-lens head) into one staging buffer on the stream, without
//! synchronising; [crate::inkling::Session::probe_report] reads the staging back once and
//! recomputes the derived quantities on the host (f64) from the formulas of the kernels, with
//! a reconstruction error against the device output for each of them.

use super::{Exec, Weights};
use astral_core::{AstralError, Result};
use astral_gpu::{Context, DeviceBuffer, Stream};
use astral_kernels::ops::{GemvType, QAct};
use serde_json::{json, Value};
use std::sync::Arc;

fn bad(m: String) -> AstralError {
    AstralError::Invalid(m)
}

/// Coordinates of the visit being run: logical visit, cache slot, physical layer, pass.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Ctx {
    pub visit: usize,
    pub slot: usize,
    pub ip: usize,
    pub pass: Option<usize>,
}

struct Entry {
    ctx: Ctx,
    key: &'static str,
    int: bool,
    off: usize,
    len: usize,
    meta: Vec<u32>,
}

/// Device staging for one probed step.
pub(crate) struct Probe {
    pub(crate) armed: bool,
    pub(crate) ctx: Ctx,
    pub(crate) lens: bool,
    f: DeviceBuffer<f32>,
    i: DeviceBuffer<i32>,
    f_off: usize,
    i_off: usize,
    lens_qa: QAct,
    lens_mn: DeviceBuffer<f32>,
    lens_logits: DeviceBuffer<f32>,
    entries: Vec<Entry>,
    /// Visit that wrote each attention-residual bank row in this step.
    pub(crate) row_visit: Vec<Option<usize>>,
}

impl Probe {
    pub(crate) fn new(ctx: &Arc<Context>, dim: usize, vocab_padded: usize, floats: usize, bank_rows: usize) -> Result<Self> {
        Ok(Self {
            armed: false,
            ctx: Ctx::default(),
            lens: true,
            f: DeviceBuffer::zeroed(ctx, floats)?,
            i: DeviceBuffer::zeroed(ctx, 1 << 14)?,
            f_off: 0,
            i_off: 0,
            lens_qa: QAct::new(ctx, dim as u32, 1)?,
            lens_mn: DeviceBuffer::zeroed(ctx, dim)?,
            lens_logits: DeviceBuffer::zeroed(ctx, vocab_padded)?,
            entries: Vec::new(),
            row_visit: vec![None; bank_rows],
        })
    }

    pub(crate) fn arm(&mut self) {
        self.armed = true;
        self.f_off = 0;
        self.i_off = 0;
        self.entries.clear();
        self.row_visit.iter_mut().for_each(|r| *r = None);
        self.ctx = Ctx::default();
    }

    /// Queues a copy of src[off..off + len] into the staging buffer.
    pub(crate) fn f(&mut self, st: &Stream, key: &'static str, src: &DeviceBuffer<f32>, off: usize, len: usize, meta: Vec<u32>) -> Result<()> {
        if self.f_off + len > self.f.len() {
            return Err(bad(format!("probe staging full at {key} (visit {}): raise ASTRAL_TRACE_FLOATS", self.ctx.visit)));
        }
        self.f.copy_range_from_async(self.f_off, src, off, len, st)?;
        self.entries.push(Entry { ctx: self.ctx, key, int: false, off: self.f_off, len, meta });
        self.f_off += len;
        Ok(())
    }

    pub(crate) fn i(&mut self, st: &Stream, key: &'static str, src: &DeviceBuffer<i32>, off: usize, len: usize) -> Result<()> {
        if self.i_off + len > self.i.len() {
            return Err(bad(format!("probe int staging full at {key}")));
        }
        self.i.copy_range_from_async(self.i_off, src, off, len, st)?;
        self.entries.push(Entry { ctx: self.ctx, key, int: true, off: self.i_off, len, meta: Vec::new() });
        self.i_off += len;
        Ok(())
    }

    /// Bank rows `rows` (token 0 of each) followed by `cur`: the states a depth mix reads.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn states(&mut self, st: &Stream, key: &'static str, bank: &DeviceBuffer<f32>, rows: &[u32], row_stride: usize, cur: &DeviceBuffer<f32>, d: usize) -> Result<()> {
        for &r in rows {
            self.f(st, key, bank, r as usize * row_stride, d, vec![r])?;
        }
        self.f(st, key, cur, 0, d, vec![u32::MAX])
    }

    /// Raw lens of the stream: output norm, LM head and softcap (the head's kernels, into the
    /// probe's own buffers), staged as one row of logits.
    pub(crate) fn lens(&mut self, ex: &Exec, w: &Weights, cur: &DeviceBuffer<f32>) -> Result<()> {
        if !self.lens {
            return Ok(());
        }
        let (cfg, d) = (&w.cfg, w.cfg.dim);
        let x = cur.view(0, d)?;
        ex.ops.rms_q8(ex.st, &x, d, Some(&w.output_norm), Some(&mut self.lens_mn), &mut self.lens_qa, 1, cfg.rms_eps)?;
        match w.output.ty {
            GemvType::Q8_0 | GemvType::Nvfp4 => ex.gemv.launch_q(ex.st, &w.output, &self.lens_qa, &mut self.lens_logits, None)?,
            _ => ex.gemv.launch(ex.st, &w.output, &self.lens_mn, 0, &mut self.lens_logits)?,
        }
        if cfg.softcap != 0.0 {
            ex.ops.softcap(ex.st, &mut self.lens_logits, w.output.n, cfg.softcap)?;
        }
        let len = cfg.vocab;
        if self.f_off + len > self.f.len() {
            return Err(bad(format!("probe staging full at lens (visit {}): raise ASTRAL_TRACE_FLOATS", self.ctx.visit)));
        }
        self.f.copy_range_from_async(self.f_off, &self.lens_logits, 0, len, ex.st)?;
        self.entries.push(Entry { ctx: self.ctx, key: "lens", int: false, off: self.f_off, len, meta: Vec::new() });
        self.f_off += len;
        Ok(())
    }
}

/// What a probed step left on the host.
pub(crate) struct Staged {
    f: Vec<f32>,
    i: Vec<i32>,
    entries: Vec<Entry>,
    row_visit: Vec<Option<usize>>,
}

impl Probe {
    pub(crate) fn take(&mut self) -> Result<Staged> {
        self.armed = false;
        let f = self.f.to_vec_range(0, self.f_off)?;
        let i = self.i.to_vec_range(0, self.i_off)?;
        Ok(Staged { f, i, entries: std::mem::take(&mut self.entries), row_visit: self.row_visit.clone() })
    }
}

/// Host copies of the small weights the report needs (read once).
pub(crate) struct HostW {
    attn_q: Vec<Option<Vec<f32>>>,
    ffn_q: Vec<Vec<f32>>,
    msg_q: Vec<Vec<f32>>,
    out_q: Vec<f32>,
    rel_bias: Vec<Vec<f32>>,
    probs_b: Vec<Option<Vec<f32>>>,
    /// Per loop layer, per controller (q, k, v, ffn): gain rows then forget rows, [2 dim][rank].
    gf: Vec<[Vec<f32>; 4]>,
    rank: usize,
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

impl HostW {
    pub(crate) fn load(w: &Weights) -> Result<Self> {
        let mut gf = Vec::new();
        let mut rank = 0;
        for lp in &w.loops {
            let mut v: Vec<Vec<f32>> = Vec::new();
            for m in &lp.gain_forget {
                if m.ty != GemvType::F32 {
                    return Err(bad(format!("probe: controller gain/forget is {:?}, expected F32", m.ty)));
                }
                rank = m.k as usize;
                v.push(f32s(&m.qs.to_vec()?));
            }
            gf.push(v.try_into().map_err(|_| bad("probe: four controllers expected".into()))?);
        }
        Ok(Self {
            attn_q: w.layers.iter().map(|l| l.attn_res_query.as_ref().map(|q| q.to_vec()).transpose()).collect::<Result<_>>()?,
            ffn_q: w.layers.iter().map(|l| l.ffn_res_query.to_vec()).collect::<Result<_>>()?,
            msg_q: w.loops.iter().map(|l| l.msg_query.to_vec()).collect::<Result<_>>()?,
            out_q: w.output_res_query.to_vec()?,
            rel_bias: w.layers.iter().map(|l| l.rel_bias.to_vec()).collect::<Result<_>>()?,
            probs_b: w.layers.iter().map(|l| l.probs_b.as_ref().map(|b| b.to_vec()).transpose()).collect::<Result<_>>()?,
            gf,
            rank,
        })
    }
}

pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            // Subnormal: normalise.
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            (s << 31) | ((e2 as u32) << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 {
        (s << 31) | 0x7f80_0000 | (m << 13)
    } else {
        (s << 31) | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

fn norm(x: &[f32]) -> f64 {
    x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt()
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    d / (norm(a) * norm(b)).max(1e-30)
}

fn max_abs_diff(a: &[f64], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| (x - y as f64).abs()).fold(0.0, f64::max)
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let i = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[i]
}

fn dist_summary(v: &[f64]) -> Value {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let mean = s.iter().sum::<f64>() / s.len().max(1) as f64;
    let mabs = s.iter().map(|x| x.abs()).sum::<f64>() / s.len().max(1) as f64;
    json!({"mean": mean, "mean_abs": mabs, "p01": quantile(&s, 0.01), "p10": quantile(&s, 0.1), "p50": quantile(&s, 0.5), "p90": quantile(&s, 0.9), "p99": quantile(&s, 0.99)})
}

/// RMS, largest magnitudes, quantiles of |x| and the energy share of the top coordinates.
fn stream_stats(x: &[f32]) -> Value {
    let n = x.len() as f64;
    let e: f64 = x.iter().map(|&v| v as f64 * v as f64).sum();
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[b].abs().total_cmp(&x[a].abs()));
    let mut a: Vec<f64> = x.iter().map(|v| v.abs() as f64).collect();
    a.sort_by(|p, q| p.total_cmp(q));
    let top: Vec<Value> = idx.iter().take(8).map(|&i| json!([i, x[i], (x[i] as f64).powi(2) / e.max(1e-30)])).collect();
    let share4: f64 = idx.iter().take(4).map(|&i| (x[i] as f64).powi(2)).sum::<f64>() / e.max(1e-30);
    json!({"rms": (e / n).sqrt(), "max_abs": a.last().copied().unwrap_or(0.0), "abs_p50": quantile(&a, 0.5), "abs_p99": quantile(&a, 0.99), "top": top, "top4_energy": share4})
}

/// The depth mix of astral_ink_mix: states RMS-normalised (eps 1e-6), p = softmax(states . q /
/// sqrt(dim)) per query, out = sum p * normalised state. Returns (p per query, out per query).
fn depth_mix(states: &[&[f32]], query: &[f32], d: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let nq = query.len() / d;
    let inv: Vec<f64> = states.iter().map(|s| 1.0 / (s.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / d as f64 + 1e-6).sqrt()).collect();
    let (mut ps, mut outs) = (Vec::new(), Vec::new());
    for j in 0..nq {
        let q = &query[j * d..(j + 1) * d];
        let sc: Vec<f64> = states.iter().zip(&inv).map(|(s, iv)| s.iter().zip(q).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() * iv / (d as f64).sqrt()).collect();
        let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = sc.iter().map(|v| (v - m).exp()).collect();
        let z: f64 = e.iter().sum();
        let p: Vec<f64> = e.iter().map(|v| v / z).collect();
        let mut o = vec![0.0f64; d];
        for ((s, pv), iv) in states.iter().zip(&p).zip(&inv) {
            for (oi, &si) in o.iter_mut().zip(s.iter()) {
                *oi += pv * iv * si as f64;
            }
        }
        ps.push(p);
        outs.push(o);
    }
    (ps, outs)
}

fn log_softmax(z: &[f32]) -> Vec<f64> {
    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let s: f64 = z.iter().map(|&v| (v as f64 - m).exp()).sum();
    let l = s.ln() + m;
    z.iter().map(|&v| v as f64 - l).collect()
}

/// Readout of a logits row: entropy, top-5 and, with a target, its log-prob and rank.
pub(crate) fn readout(z: &[f32], target: Option<u32>) -> Value {
    let lp = log_softmax(z);
    let mut idx: Vec<usize> = (0..lp.len()).collect();
    idx.select_nth_unstable_by(5, |&a, &b| lp[b].total_cmp(&lp[a]));
    let mut top: Vec<usize> = idx[..5].to_vec();
    top.sort_by(|&a, &b| lp[b].total_cmp(&lp[a]));
    let h: f64 = -lp.iter().map(|&l| l.exp() * l).sum::<f64>();
    let mut out = json!({"entropy": h, "top5": top.iter().map(|&i| json!([i, lp[i].exp()])).collect::<Vec<_>>()});
    if let Some(t) = target {
        let lt = lp[t as usize];
        out["target_logprob"] = json!(lt);
        out["target_rank"] = json!(lp.iter().filter(|&&l| l > lt).count());
    }
    out
}

pub(crate) struct ReportIn<'a> {
    pub w: &'a Weights,
    pub hw: &'a HostW,
    pub pos: u32,
    pub token: u32,
    pub target: Option<u32>,
    /// K / V rows of a slot, [lo, pos] ([n][kv width] f16 bits); None skips attention.
    pub kv: &'a dyn Fn(usize, usize, usize, bool) -> Result<Vec<u16>>,
    pub validate: bool,
    pub native: &'a [f32],
}

fn slice<'a>(st: &'a Staged, e: &Entry) -> &'a [f32] {
    &st.f[e.off..e.off + e.len]
}

fn entries<'a>(es: &[&'a Entry], k: &str) -> Vec<&'a Entry> {
    es.iter().filter(|e| e.key == k && !e.int).copied().collect()
}

/// Builds one JSON record per visit plus a head record from a staged step.
pub(crate) fn report(st: &Staged, a: &ReportIn) -> Result<Vec<Value>> {
    let w = a.w;
    let cfg = &w.cfg;
    let d = cfg.dim;
    let (start, end) = (cfg.core_start, cfg.core_end);
    let label = |r: u32| -> Value {
        if r == u32::MAX {
            json!("cur")
        } else {
            match st.row_visit.get(r as usize).copied().flatten() {
                Some(v) => json!(format!("v{v}")),
                None => json!(format!("row{r}")),
            }
        }
    };
    // Group entries by visit, keeping order.
    let mut visits: Vec<(Ctx, Vec<&Entry>)> = Vec::new();
    for e in &st.entries {
        match visits.last_mut() {
            Some((c, v)) if c.visit == e.ctx.visit && c.slot == e.ctx.slot => v.push(e),
            _ => visits.push((e.ctx, vec![e])),
        }
    }
    let fv = |e: &Entry| -> &[f32] { slice(st, e) };
    let mut out = Vec::new();
    let mut worst = json!({});
    let mut upd = |k: &str, v: f64| {
        let cur = worst.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0);
        if v > cur || !v.is_finite() || worst.get(k).is_none() {
            worst[k] = json!(v);
        }
    };
    for (c, es) in &visits {
        let get = |k: &str| -> Vec<&Entry> { entries(es, k) };
        let one = |k: &str| -> Option<&[f32]> { entries(es, k).first().map(|e| slice(st, e)) };
        let stage = if c.visit == usize::MAX {
            "head"
        } else if c.ip < start {
            "prelude"
        } else if c.ip < end {
            "core"
        } else {
            "coda"
        };
        let mut rec = json!({"pos": a.pos, "token": a.token, "target": a.target, "visit": if c.visit == usize::MAX { Value::Null } else { json!(c.visit) },
                             "slot": c.slot, "layer": c.ip, "pass": c.pass, "stage": stage});
        let mix = |skey: &str, okey: &str, query: &[f32], upd: &mut dyn FnMut(&str, f64)| -> Option<Value> {
            let ss = get(skey);
            let o = one(okey)?;
            if ss.is_empty() || query.is_empty() {
                return None;
            }
            let states: Vec<&[f32]> = ss.iter().map(|e| fv(e)).collect();
            let (ps, outs) = depth_mix(&states, query, d);
            let err = outs.iter().enumerate().map(|(j, oj)| max_abs_diff(oj, &o[j * d..(j + 1) * d])).fold(0.0, f64::max);
            let scale = o.iter().map(|v| v.abs() as f64).fold(0.0, f64::max).max(1e-30);
            upd(okey, err / scale);
            let labels: Vec<Value> = ss.iter().map(|e| label(e.meta[0])).collect();
            Some(json!({"sources": labels, "p": ps, "out_norm": (0..ps.len()).map(|j| norm(&o[j * d..(j + 1) * d])).collect::<Vec<_>>(), "recompute_rel_err": err / scale}))
        };
        if c.visit == usize::MAX {
            rec["slot"] = Value::Null;
            rec["layer"] = Value::Null;
            if let Some(m) = mix("head_mix.states", "head_mix.out", &a.hw.out_q, &mut upd) {
                rec["head_mix"] = m;
            }
            rec["native"] = readout(a.native, a.target);
            out.push(rec);
            continue;
        }
        let l = &w.layers[c.ip];
        if let Some(q) = &a.hw.attn_q[c.ip]
            && let Some(m) = mix("attn_mix.states", "attn_mix.out", q, &mut upd)
        {
            rec["attn_mix"] = m;
        }
        if let Some(m) = mix("ffn_mix.states", "ffn_mix.out", &a.hw.ffn_q[c.ip], &mut upd) {
            rec["ffn_mix"] = m;
        }
        let li = (c.ip >= start && c.ip < end && c.pass.unwrap_or(0) > 0).then(|| c.ip - start);
        if let Some(li) = li {
            if let Some(m) = mix("msg_mix.states", "msg_mix.out", &a.hw.msg_q[li], &mut upd) {
                rec["msg_mix"] = m;
            }
            // Controllers: out = x - tanh(f) x + tanh(g) msg, (g, f) = W_gf cd.
            let rank = a.hw.rank;
            let msgs = one("msg_mix.out");
            let mut ctrls = Vec::new();
            for j in 0..4 {
                let (cd, x, o) = if j < 3 {
                    let cd = one("ctrl.cd").map(|v| &v[j * rank..(j + 1) * rank]);
                    let nq = a.hw.attn_q[c.ip].as_ref().map_or(0, |q| q.len() / d);
                    let row = if l.swa || nq <= 1 { 0 } else { j };
                    let x = if nq > 0 { one("attn_mix.out").map(|v| &v[row * d..(row + 1) * d]) } else { one("in") };
                    (cd, x, entries(es, "ctrl.out").get(j).map(|e| slice(st, e)))
                } else {
                    (one("ffn_ctrl.cd"), one("ffn_mix.out"), one("ffn_ctrl.out"))
                };
                let (Some(cd), Some(x), Some(o), Some(ms)) = (cd, x, o, msgs) else { continue };
                let msg = &ms[j * d..(j + 1) * d];
                let gfw = &a.hw.gf[li][j];
                let (mut g, mut f, mut rec_o) = (vec![0f64; d], vec![0f64; d], vec![0f64; d]);
                for n in 0..d {
                    let gr = &gfw[n * rank..(n + 1) * rank];
                    let fr = &gfw[(d + n) * rank..(d + n + 1) * rank];
                    let gs: f64 = gr.iter().zip(cd).map(|(&a, &b)| a as f64 * b as f64).sum();
                    let fs: f64 = fr.iter().zip(cd).map(|(&a, &b)| a as f64 * b as f64).sum();
                    g[n] = gs.tanh();
                    f[n] = fs.tanh();
                    rec_o[n] = x[n] as f64 - f[n] * x[n] as f64 + g[n] * msg[n] as f64;
                }
                let err = max_abs_diff(&rec_o, o) / o.iter().map(|v| v.abs() as f64).fold(0.0, f64::max).max(1e-30);
                upd("ctrl.out", err);
                let keep: Vec<f64> = f.iter().map(|v| 1.0 - v).collect();
                let fx: f64 = f.iter().zip(x).map(|(a, &b)| (a * b as f64).powi(2)).sum::<f64>().sqrt();
                let gm: f64 = g.iter().zip(msg).map(|(a, &b)| (a * b as f64).powi(2)).sum::<f64>().sqrt();
                let which = ["q", "k", "v", "ffn"][j];
                ctrls.push(json!({"which": which, "gain": dist_summary(&g), "forget": dist_summary(&f), "keep": dist_summary(&keep),
                    "x_norm": norm(x), "msg_norm": norm(msg), "forget_term_norm": fx, "gain_term_norm": gm, "out_norm": norm(o),
                    "cos_x_msg": cos(x, msg), "cos_out_x": cos(o, x), "recompute_rel_err": err}));
            }
            rec["ctrl"] = json!(ctrls);
        }
        // Router: probs = sigmoid(logits); picks = top-k of probs + bias; weights = normalised probs.
        if let Some(lg) = one("router.logits") {
            let probs: Vec<f64> = lg.iter().map(|&v| 1.0 / (1.0 + (-(v as f64)).exp())).collect();
            let sel: Vec<f64> = match &a.hw.probs_b[c.ip] {
                Some(b) => probs.iter().zip(b).map(|(p, &b)| p + b as f64).collect(),
                None => probs.clone(),
            };
            let mut order: Vec<usize> = (0..sel.len()).collect();
            order.sort_by(|&x, &y| sel[y].total_cmp(&sel[x]).then(x.cmp(&y)));
            let k = cfg.n_expert_used;
            let ids: Vec<i32> = es.iter().find(|e| e.key == "router.ids" && e.int).map(|e| st.i[e.off..e.off + e.len].to_vec()).unwrap_or_default();
            let mine: Vec<i32> = order[..k].iter().map(|&i| i as i32).collect();
            let wts = one("router.w").map(|v| v.to_vec()).unwrap_or_default();
            let match_ids = ids == mine;
            upd("router.ids_mismatch", if match_ids { 0.0 } else { 1.0 });
            rec["router"] = json!({"ids": ids, "weights": wts, "sel_top": order[..(k + 1).min(order.len())].iter().map(|&i| json!([i, sel[i], probs[i]])).collect::<Vec<_>>(),
                                   "margin_k_k1": sel[order[k - 1]] - sel[order[k]], "ids_match_recompute": match_ids});
        }
        // Attention: logit_j = q . k_j / sqrt(hd) + rel bias(dist) - sink; the sink logit is 0.
        if let (Some(q), Some(r), Some(sk)) = (one("attn.q"), one("attn.r"), one("attn.sinks")) {
            let (h, kvh, hd, nr) = (cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.n_rel as usize);
            let window = if l.swa { cfg.n_swa as usize } else { 0 };
            let pos = a.pos as usize;
            let lo = if window > 0 && pos + 1 > window { pos + 1 - window } else { 0 };
            let n = pos + 1 - lo;
            let kv_w = kvh * hd;
            let kr = (a.kv)(c.slot, lo, n, false)?;
            let vr = if a.validate { Some((a.kv)(c.slot, lo, n, true)?) } else { None };
            let rb = &a.hw.rel_bias[c.ip];
            let scale = 1.0 / (hd as f64).sqrt();
            let (mut mass, mut ents, mut top_d) = (Vec::new(), Vec::new(), Vec::new());
            let mut err_max = 0.0f64;
            let sparse = window == 0 && std::env::var_os("ASTRAL_TRACE_SPARSE").is_some();
            let mut shadow = Vec::new();
            let gate = one("attn.gate");
            let aout = one("attn.out");
            for hh in 0..h {
                let g = hh / (h / kvh);
                let qh = &q[hh * hd..(hh + 1) * hd];
                let rs = &r[hh * nr..(hh + 1) * nr];
                let mut lg = vec![0f64; n];
                for t in 0..n {
                    let kj = &kr[t * kv_w + g * hd..t * kv_w + (g + 1) * hd];
                    let dot: f64 = qh.iter().zip(kj).map(|(&a, &b)| a as f64 * f16_to_f32(b) as f64).sum();
                    let dist = pos - (lo + t);
                    let bias: f64 = if (dist as u32) < cfg.rel_extent { rb[dist * nr..(dist + 1) * nr].iter().zip(rs).map(|(&a, &b)| a as f64 * b as f64).sum() } else { 0.0 };
                    lg[t] = scale * dot + bias - sk[hh] as f64;
                }
                let m = lg.iter().cloned().fold(0.0f64, f64::max);
                let e: Vec<f64> = lg.iter().map(|v| (v - m).exp()).collect();
                let z = e.iter().sum::<f64>() + (-m).exp();
                let p: Vec<f64> = e.iter().map(|v| v / z).collect();
                let psink = (-m).exp() / z;
                let mut buckets = [psink, 0.0, 0.0, 0.0, 0.0];
                for (t, &pv) in p.iter().enumerate() {
                    let dist = pos - (lo + t);
                    let b = match dist {
                        0 => 1,
                        1..=63 => 2,
                        64..=1023 => 3,
                        _ => 4,
                    };
                    buckets[b] += pv;
                }
                let ent = -(p.iter().chain(std::iter::once(&psink)).filter(|&&v| v > 0.0).map(|v| v * v.ln()).sum::<f64>());
                let (am, _) = p.iter().enumerate().fold((0usize, -1.0f64), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc });
                if sparse {
                    shadow.push(sparse_shadow(&lg, &p, pos, lo, &kr, vr.as_deref(), g, hd, kv_w, qh, scale, sk[hh] as f64, (pos as u64) << 16 ^ (c.slot as u64) << 8 ^ hh as u64));
                }
                mass.push(buckets.to_vec());
                ents.push(ent);
                top_d.push(pos - (lo + am));
                if let (Some(vr), Some(gt), Some(ao)) = (vr.as_ref(), gate, aout) {
                    for dd in 0..hd {
                        let acc: f64 = (0..n).map(|t| p[t] * f16_to_f32(vr[t * kv_w + g * hd + dd]) as f64).sum();
                        let gv = 1.0 / (1.0 + (-(gt[hh * hd + dd] as f64)).exp());
                        let ref_v = ao[hh * hd + dd] as f64;
                        err_max = err_max.max((acc * gv - ref_v).abs());
                    }
                }
            }
            let mean_b: Vec<f64> = (0..5).map(|b| mass.iter().map(|m| m[b]).sum::<f64>() / h as f64).collect();
            let mut at = json!({"window": window, "positions": n, "mass_mean": {"sink": mean_b[0], "self": mean_b[1], "near_1_63": mean_b[2], "mid_64_1023": mean_b[3], "far_1024": mean_b[4]},
                                "per_head": {"sink": mass.iter().map(|m| m[0]).collect::<Vec<_>>(), "far_1024": mass.iter().map(|m| m[4]).collect::<Vec<_>>(), "entropy": ents, "argmax_dist": top_d}});
            if sparse {
                at["sparse"] = json!(shadow);
            }
            if vr.is_some() {
                let scale = aout.map_or(1.0, |o| o.iter().map(|v| v.abs() as f64).fold(0.0, f64::max).max(1e-30));
                at["recompute_rel_err"] = json!(err_max / scale);
                upd("attn.out", err_max / scale);
            }
            rec["attn"] = at;
        }
        if let Some(o) = one("out") {
            rec["stream"] = stream_stats(o);
        }
        if let Some(z) = one("lens") {
            rec["lens_raw"] = readout(z, a.target);
        }
        out.push(rec);
    }
    if let Some(h) = out.last_mut() {
        h["checks"] = worst;
    }
    Ok(out)
}

/// T28 dense shadow (ASTRAL_TRACE_SPARSE=1, global layers): which 64-token blocks of the far
/// history (distance >= 1024) a block-sparse decode could skip, against the exact attention of
/// this head. Block metadata are the mean key and radius (max distance of a key from the mean);
/// the logit bound of a block is scale * (q . mean + |q| * radius) plus its largest relative
/// bias, minus the sink logit. Policies at a budget (share of far blocks kept): the bound, the mean alone, the
/// oracle (true block mass) and a seeded random choice; a threshold rule skips the blocks whose
/// bound mass (block size * exp(bound) / the recent window's partition sum) is at most tau, which
/// bounds the true skipped mass from above. Missed mass is true probability; the output error is
/// |o_kept / (1 - missed) - o| / |o| over the head's value dims (needs the V cache).
#[allow(clippy::too_many_arguments)]
fn sparse_shadow(lg: &[f64], p: &[f64], pos: usize, lo: usize, kr: &[u16], vr: Option<&[u16]>, g: usize, hd: usize, kv_w: usize, qh: &[f32], scale: f64, sink: f64, seed: u64) -> Option<serde_json::Value> {
    const B: usize = 64;
    const RECENT: usize = 1024;
    let n = lg.len();
    // Far tokens: distance >= RECENT; blocks by absolute position, only whole far blocks.
    let far_end = (pos + 1).saturating_sub(RECENT); // absolute positions < far_end are far
    let nb = far_end / B;
    if nb < 2 || lo > 0 {
        return None;
    }
    let qn: f64 = qh.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
    let mut mass = vec![0f64; nb];
    let mut bound = vec![0f64; nb];
    let mut meanlg = vec![0f64; nb];
    let mut kb = vec![0f64; B * hd];
    for b in 0..nb {
        for (i, t) in (b * B..(b + 1) * B).enumerate() {
            for dd in 0..hd {
                kb[i * hd + dd] = f16_to_f32(kr[t * kv_w + g * hd + dd]) as f64;
            }
        }
        let mut mu = vec![0f64; hd];
        for i in 0..B {
            for (dd, m) in mu.iter_mut().enumerate() {
                *m += kb[i * hd + dd] / B as f64;
            }
        }
        let mut rad = 0f64;
        let mut bmax = f64::MIN;
        for (i, t) in (b * B..(b + 1) * B).enumerate() {
            let k = &kb[i * hd..(i + 1) * hd];
            let d2: f64 = k.iter().zip(&mu).map(|(a, m)| (a - m).powi(2)).sum();
            rad = rad.max(d2.sqrt());
            let dot: f64 = k.iter().zip(qh).map(|(a, &q)| a * q as f64).sum();
            bmax = bmax.max(lg[t] - (scale * dot - sink)); // the relative bias (0 past its extent)
            mass[b] += p[t];
        }
        let qmu: f64 = (0..hd).map(|dd| qh[dd] as f64 * mu[dd]).sum();
        bound[b] = scale * (qmu + qn * rad) + bmax - sink;
        meanlg[b] = scale * qmu - sink;
    }
    let far_mass: f64 = mass.iter().sum();
    // Per-block and recent partial sums of p * v (one pass over V), then the value output of a
    // kept set (renormalised) against the exact one.
    let parts: Option<(Vec<Vec<f64>>, Vec<f64>)> = vr.map(|vr| {
        let mut blk = vec![vec![0f64; hd]; nb];
        let mut rec = vec![0f64; hd];
        for t in 0..n {
            let acc = if t < nb * B { &mut blk[t / B] } else { &mut rec };
            for (dd, a) in acc.iter_mut().enumerate() {
                *a += p[t] * f16_to_f32(vr[t * kv_w + g * hd + dd]) as f64;
            }
        }
        (blk, rec)
    });
    let out_err = |keep: &[bool], missed: f64| -> Option<f64> {
        let (blk, rec) = parts.as_ref()?;
        let (mut e2, mut o2) = (0f64, 0f64);
        for dd in 0..hd {
            let (mut full, mut kept) = (rec[dd], rec[dd]);
            for (b, s) in blk.iter().enumerate() {
                full += s[dd];
                if keep[b] {
                    kept += s[dd];
                }
            }
            let kept = kept / (1.0 - missed).max(1e-30);
            e2 += (kept - full).powi(2);
            o2 += full * full;
        }
        Some((e2 / o2.max(1e-300)).sqrt())
    };
    let pick = |score: &[f64], k: usize| -> Vec<bool> {
        let mut ord: Vec<usize> = (0..nb).collect();
        ord.sort_by(|&x, &y| score[y].total_cmp(&score[x]).then(x.cmp(&y)));
        let mut keep = vec![false; nb];
        for &i in &ord[..k] {
            keep[i] = true;
        }
        keep
    };
    let mut rnd = vec![0f64; nb];
    let mut s = seed | 1;
    for r in rnd.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *r = (s >> 11) as f64;
    }
    let missed = |keep: &[bool]| -> f64 { (0..nb).filter(|&b| !keep[b]).map(|b| mass[b]).sum() };
    let mut budgets = serde_json::Map::new();
    for &frac in &[0.0625f64, 0.125, 0.25, 0.5] {
        let k = ((nb as f64 * frac).ceil() as usize).clamp(1, nb);
        let mut pol = serde_json::Map::new();
        for (name, score) in [("bound", &bound), ("mean", &meanlg), ("oracle", &mass), ("random", &rnd)] {
            let keep = pick(score, k);
            let m = missed(&keep);
            pol.insert(name.into(), json!([m, out_err(&keep, m)]));
        }
        budgets.insert(format!("{frac}"), serde_json::Value::Object(pol));
    }
    // Threshold rule against the recent window's partition sum (the sink's logit is 0 here).
    let mr = lg[nb * B..].iter().cloned().fold(0f64, f64::max);
    let zr: f64 = lg[nb * B..].iter().map(|v| (v - mr).exp()).sum::<f64>() + (-mr).exp();
    let mut thr = serde_json::Map::new();
    for &tau in &[1e-3f64, 1e-4] {
        let bm = |s: f64| B as f64 * (s - mr).exp() / zr;
        let keep: Vec<bool> = bound.iter().map(|&s| bm(s) > tau).collect();
        let bsum: f64 = (0..nb).filter(|&b| !keep[b]).map(|b| bm(bound[b])).sum();
        let m = missed(&keep);
        thr.insert(format!("{tau}"), json!({"kept": keep.iter().filter(|&&x| x).count() as f64 / nb as f64, "missed": m, "bound": bsum, "out_err": out_err(&keep, m)}));
    }
    Some(json!({"far_blocks": nb, "far_mass": far_mass, "budget": budgets, "threshold": thr}))
}
