// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! `astral` command-line tool.
//!
//!   astral hw [--bench | --bench-fresh]   print detected hardware as JSON; --bench reuses a
//!                         cached bandwidth result for this GPU/driver/clocks, --bench-fresh re-measures
//!   astral selftest       run every GPU kernel against its CPU reference
//!   astral gguf-info <model> [--tensors]   summarise model weights (metadata, types, layout)
//!   astral verify-embd <file> <eval_callback.log>   match token_embd rows against a llama.cpp dump
//!   astral verify-model <file> <eval_callback.log>   run the prompt tokens and compare every layer slot
//!   astral generate <model> <n> <token ids...>   greedy decode n tokens, report tok/s
//!   astral serve <model> [--checkpoint <dir>] [options]   the OpenAI-compatible server
//!
//! `<model>` is a `.gguf` file or a safetensors checkpoint directory.
//!   astral kbench <mxfp4-gemv | mxfp4-mma | mxfp4-mma-f16> <n> <k> <b> [iters]   time one kernel (for Nsight Compute)
//!   astral version        print version and build information

#![forbid(unsafe_code)]

use astral_core::Result;
use std::process::ExitCode;

const USAGE: &str = "usage: astral <hw [--bench | --bench-fresh] | selftest | gguf-info <model> [--tensors] | verify-embd <file> <log> | verify-model <file> <log> | generate <model> <n> <ids...> | dump-logits <model> <llama kld base> <out> | dump-final-logits <model> <manifest> <out dir> | dump-verify-logits <model> <checkpoint dir> <k> <manifest> <out dir> | trace <model> <manifest> <out dir> | ls-diag <model> <first> <ids...|@file> | spec-generate <model> <checkpoint dir> <k> <n> <ids...|@file> | serve <model> [--checkpoint <dir>] [--no-draft] [--k 6] [--host 0.0.0.0] [--port 8000] [--ctx 8192] [--name inkling-2-mini] | version>";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("hw") => cmd_hw(if args.iter().any(|a| a == "--bench-fresh") {
            astral_hw::BenchMode::Fresh
        } else if args.iter().any(|a| a == "--bench") {
            astral_hw::BenchMode::Cached
        } else {
            astral_hw::BenchMode::Skip
        }),
        Some("selftest") => cmd_selftest(),
        Some("gguf-info") if args.len() >= 2 => cmd_gguf_info(&args[1], args.iter().any(|a| a == "--tensors")),
        Some("verify-embd") if args.len() >= 3 => cmd_verify_embd(&args[1], &args[2]),
        Some("verify-model") if args.len() >= 3 => cmd_verify_model(&args[1], &args[2]),
        Some("score") if args.len() >= 3 => cmd_score(&args[1], &args[2..]),
        Some("dump-logits") if args.len() >= 4 => cmd_dump_logits(&args[1], &args[2], &args[3]),
        Some("qwen-generate") if args.len() >= 4 => cmd_qwen_generate(&args[1], &args[2], &args[3..]),
        Some("generate") if args.len() >= 4 => cmd_generate(&args[1], &args[2], &args[3..]),
        Some("serve") if args.len() >= 2 => cmd_serve(&args[1..]),
        Some("dump-final-logits") if args.len() >= 4 => cmd_dump_final_logits(&args[1], &args[2], &args[3]),
        Some("trace") if args.len() >= 4 => cmd_trace(&args[1], &args[2], &args[3]),
        Some("dump-verify-logits") if args.len() >= 6 => cmd_dump_verify_logits(&args[1], &args[2], &args[3], &args[4], &args[5]),
        Some("cmp-prefill") if args.len() >= 3 => cmd_cmp_prefill(&args[1], &args[2..]),
        Some("ls-diag") if args.len() >= 4 => cmd_ls_diag(&args[1], &args[2], &args[3..]),
        Some("spec-generate") if args.len() >= 6 => cmd_spec_generate(&args[1], &args[2], &args[3], &args[4], &args[5..]),
        Some("kbench") if args.len() >= 5 && args[1] == "mxfp4-gemv" => cmd_kbench_mxfp4(&args[2..], false),
        Some("kbench") if args.len() >= 5 && args[1] == "mxfp4-mma" => cmd_kbench_mxfp4(&args[2..], true),
        Some("kbench") if args.len() >= 5 && args[1] == "mxfp4-mma-f16" => cmd_kbench_mxfp4_f16(&args[2..]),
        Some("version") | Some("--version") | Some("-V") => {
            cmd_version();
            Ok(true)
        }
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

fn cmd_version() {
    println!("astral {}", env!("CARGO_PKG_VERSION"));
    println!("kernels built for: {:?}", astral_kernels::BUILT_ARCHS);
    println!("nvcc: {}", astral_kernels::NVCC_VERSION);
}

fn cmd_hw(mode: astral_hw::BenchMode) -> Result<bool> {
    let report = astral_hw::probe(mode)?;
    println!("{}", serde_json::to_string_pretty(&report).expect("report serialises"));
    Ok(true)
}

fn cmd_selftest() -> Result<bool> {
    let n = astral_gpu::Device::count()?;
    if n == 0 {
        eprintln!("no CUDA devices found");
        return Ok(false);
    }
    let dev = astral_gpu::Device::get(0)?;
    println!("device 0: {} (cc {:?})", dev.name()?, dev.compute_capability()?);
    let ctx = astral_gpu::Context::new(dev)?;
    let results = astral_testing::selftest::run(&ctx)?;
    let mut all = true;
    for r in &results {
        all &= r.pass;
        println!("{:<5} {:<20} {:<20} max_scaled_err={:.3e} (tol {:.1e})", if r.pass { "PASS" } else { "FAIL" }, r.kernel, r.case, r.error.max_scaled, r.tolerance);
        if let Some(info) = &r.info {
            println!("      {info}");
        }
    }
    println!("{} / {} cases passed", results.iter().filter(|r| r.pass).count(), results.len());
    Ok(all)
}

fn cmd_gguf_info(path: &str, list_tensors: bool) -> Result<bool> {
    let g = astral_loader::open_model(path)?;
    let arch = g.architecture().unwrap_or("?").to_string();
    println!("file        {}", g.path.display());
    println!("gguf        v{}  alignment {}  data offset {}  file {} bytes", g.version, g.alignment, g.data_offset, g.file_len);
    println!("arch        {arch}   name {}", g.get("general.name").map(|v| v.summary()).unwrap_or_default());
    println!("tensors     {}   metadata keys {}", g.tensors.len(), g.metadata.len());
    let (gap, data) = g.coverage();
    println!("coverage    {} of {} data bytes are alignment padding", gap, data);
    println!("types:");
    for (ty, (n, bytes)) in g.type_tally() {
        println!("  {ty:<8} {n:>5} tensors  {:>9.3} GB", bytes as f64 / 1e9);
    }
    println!("{arch}.* keys:");
    for (k, v) in &g.metadata {
        if k.starts_with(&format!("{arch}.")) {
            println!("  {k} = {}", v.summary());
        }
    }
    if list_tensors {
        for t in &g.tensors {
            println!("  {:<48} {:<7} {:?}", t.name, t.ty.name(), t.dims);
        }
    }
    Ok(true)
}

fn cmd_verify_embd(path: &str, log_path: &str) -> Result<bool> {
    let g = astral_loader::Gguf::open(path)?;
    let log = std::fs::read_to_string(log_path)?;
    let printed = astral_testing::golden::parse_first(&log, "embd").ok_or_else(|| astral_core::AstralError::Invalid("no `embd` tensor in the dump".into()))?;
    let m = astral_testing::golden::match_embedding_rows(&g, "token_embd.weight", &printed)?;
    println!("printed rows {}  matched token ids {:?}", printed.rows.len(), m.token_ids);
    println!("sum: astral {:.6}  llama.cpp {:.6}  diff {:.3e}", m.our_sum, m.printed_sum, m.our_sum - m.printed_sum);
    let ok = m.pass(1e-5);
    println!("{}", if ok { "PASS" } else { "FAIL" });
    Ok(ok)
}

/// The prompt of the llama.cpp eval-callback dump (identified by `verify-embd`).
const GOLDEN_PROMPT: [u32; 5] = [371, 151451, 149879, 151194, 149884];

fn load_session(path: &str, capacity: u32) -> Result<astral_model::inkling::Session> {
    let ctx = astral_gpu::Context::new(astral_gpu::Device::get(0)?)?;
    let g = astral_loader::open_model(path)?;
    let t0 = std::time::Instant::now();
    let w = astral_model::inkling::Weights::load(&ctx, &g)?;
    eprintln!("loaded {:.2} GB of weights in {:.1} s", w.bytes as f64 / 1e9, t0.elapsed().as_secs_f64());
    astral_model::inkling::Session::new(&ctx, w, capacity)
}

/// Max |a - b| over the printed first/last three values of each row.
fn row_diff(rows: &[Vec<f32>], printed: &astral_testing::golden::PrintedTensor) -> f64 {
    let mut m = 0.0f64;
    for (r, (head, tail)) in rows.iter().zip(&printed.rows) {
        let n = r.len();
        for i in 0..3 {
            m = m.max((r[i] as f64 - head[i]).abs()).max((r[n - 3 + i] as f64 - tail[i]).abs());
        }
    }
    m
}

fn cmd_verify_model(path: &str, log_path: &str) -> Result<bool> {
    let log = std::fs::read_to_string(log_path)?;
    let mut s = load_session(path, 64)?;
    s.capture = true;
    s.set_debug(true);
    let mut dbg: std::collections::BTreeMap<&'static str, Vec<Vec<f32>>> = Default::default();
    let mut order: Vec<&'static str> = Vec::new();
    let mut per_slot: std::collections::BTreeMap<usize, Vec<Vec<f32>>> = Default::default();
    let mut next = 0;
    for &t in &GOLDEN_PROMPT {
        next = s.step(t)?;
        for (slot, v) in s.captured.drain(..) {
            per_slot.entry(slot).or_default().push(v);
        }
        for (name, v) in s.take_debug() {
            if !order.contains(&name) {
                order.push(name);
            }
            dbg.entry(name).or_default().push(v);
        }
    }
    for name in &order {
        let rows = &dbg[name];
        let ours: f64 = rows.iter().flatten().map(|&x| x as f64).sum();
        match astral_testing::golden::parse_first(&log, &format!("{name}-{}", astral_model::inkling::tap_layer())) {
            Some(p) if p.rows.len() == rows.len() => {
                println!("  {name:<16} max|d| {:.2e}  sum astral {ours:>12.4} llama.cpp {:>12.4}", row_diff(rows, &p), p.sum)
            }
            Some(p) => println!("  {name:<16} sum astral {ours:>12.4} llama.cpp {:>12.4} (printed as {} rows)", p.sum, p.rows.len()),
            None => println!("  {name:<16} sum astral {ours:>12.4}  first {:?}", &rows[0][..3]),
        }
    }
    let mut ok = true;
    for (slot, rows) in &per_slot {
        let Some(p) = astral_testing::golden::parse_first(&log, &format!("l_out-{slot}")) else {
            println!("l_out-{slot:<2}  (not in the dump)");
            continue;
        };
        let sum: f64 = rows.iter().flatten().map(|&x| x as f64).sum();
        let diff = row_diff(rows, &p);
        let rel = (sum - p.sum).abs() / p.sum.abs().max(1.0);
        let pass = diff <= 0.05 * rows.iter().flatten().fold(1.0f32, |m, &x| m.max(x.abs())) as f64;
        ok &= pass;
        println!("l_out-{slot:<2}  max|d| {diff:.2e}  sum astral {sum:>12.4} llama.cpp {:>12.4} (rel {rel:.1e})  {}", p.sum, if pass { "ok" } else { "DIFF" });
    }
    let logits = s.logits()?;
    if let Some(p) = astral_testing::golden::parse_first(&log, "result_output") {
        let sum: f64 = logits.iter().map(|&x| x as f64).sum();
        let diff = row_diff(std::slice::from_ref(&logits), &p);
        println!("decode  result_output  max|d| {diff:.2e}  sum astral {sum:.4} llama.cpp {:.4}", p.sum);
    }

    // The same prompt as one batched prefill (llama.cpp's dump is itself a prefill).
    s.reset()?;
    s.set_debug(false);
    let next_p = s.prefill(&GOLDEN_PROMPT)?;
    println!("prefill:");
    let d = s.w.cfg.dim;
    for (slot, v) in s.captured.clone() {
        let Some(p) = astral_testing::golden::parse_first(&log, &format!("l_out-{slot}")) else { continue };
        let rows: Vec<Vec<f32>> = v.chunks(d).map(|c| c.to_vec()).collect();
        let sum: f64 = v.iter().map(|&x| x as f64).sum();
        let diff = row_diff(&rows, &p);
        let pass = diff <= 0.05 * rows.iter().flatten().fold(1.0f32, |m, &x| m.max(x.abs())) as f64;
        ok &= pass;
        println!("l_out-{slot:<2}  max|d| {diff:.2e}  sum astral {sum:>12.4} llama.cpp {:>12.4}  {}", p.sum, if pass { "ok" } else { "DIFF" });
    }
    println!("prefill greedy next token: {next_p} (decode path: {next})");
    ok &= next_p == next;
    let logits = s.logits()?;
    if let Some(p) = astral_testing::golden::parse_first(&log, "result_output") {
        let sum: f64 = logits.iter().map(|&x| x as f64).sum();
        let diff = row_diff(std::slice::from_ref(&logits), &p);
        println!("result_output  max|d| {diff:.2e}  sum astral {sum:.4} llama.cpp {:.4}", p.sum);
        ok &= diff < 0.25;
    }
    println!("greedy next token after the prompt: {next}");
    println!("{}", if ok { "PASS" } else { "FAIL" });
    Ok(ok)
}

/// Teacher-forced: after each token, our greedy pick and the top-2 logit margin.
fn cmd_score(path: &str, ids: &[String]) -> Result<bool> {
    let toks: Vec<u32> = ids.iter().filter_map(|t| t.parse().ok()).collect();
    let mut s = load_session(path, (toks.len() + 1) as u32)?;
    let mut agree = 0;
    for (i, &t) in toks.iter().enumerate() {
        let next = s.step(t)?;
        let l = s.logits()?;
        let mut best = (f32::MIN, 0usize, f32::MIN);
        for (j, &v) in l.iter().enumerate() {
            if v > best.0 {
                best = (v, j, best.0);
            } else if v > best.2 {
                best.2 = v;
            }
        }
        if let Some(&want) = toks.get(i + 1) {
            let ok = next == want;
            agree += ok as usize;
            println!("pos {i:>3}  next {next:>6}  want {want:>6}  {}  margin {:.3}  logit[want] {:.3}", if ok { "  " } else { "XX" }, best.0 - best.2, l[want as usize]);
        }
    }
    println!("agreement {agree}/{}", toks.len() - 1);
    Ok(true)
}

/// `dump-logits <model> <base> <out>`: token-by-token log-probs in the format of llama.cpp's
/// `llama-perplexity --kl-divergence-base` file, on the chunks and tokens read from `<base>`
/// (one written by llama.cpp), so the two engines can be compared position by position.
/// Mirrors b12751 `tools/perplexity/perplexity.cpp`: header `_logits_`, n_ctx, n_vocab,
/// n_chunk, tokens; then per chunk the second half (positions n_ctx/2 .. n_ctx-2), each
/// as [scale f32, min_log_prob f32, n_vocab x u16] padded to an even count. No BOS is added
/// (Inkling's GGUF has add_bos false). With ASTRAL_DUMP_PREFILL=1 each chunk's first half is
/// prefilled in one batch and only the scored positions are decode steps.
fn cmd_dump_logits(path: &str, base: &str, out: &str) -> Result<bool> {
    use std::io::{Read, Write};
    let bad = |m: String| astral_core::AstralError::Invalid(m);
    let mut f = std::io::BufReader::new(std::fs::File::open(base).map_err(|e| bad(format!("{base}: {e}")))?);
    let mut magic = [0u8; 8];
    let mut i32s = [0u8; 12];
    f.read_exact(&mut magic).and_then(|_| f.read_exact(&mut i32s)).map_err(|e| bad(format!("{base}: {e}")))?;
    if &magic != b"_logits_" {
        return Err(bad(format!("{base}: not a llama.cpp logits file")));
    }
    let rd = |i: usize| i32::from_le_bytes(i32s[4 * i..4 * i + 4].try_into().unwrap());
    let (n_ctx, n_vocab, n_chunk) = (rd(0), rd(1), rd(2));
    if n_ctx < 4 || n_vocab < 2 || n_chunk < 1 {
        return Err(bad(format!("{base}: bad header n_ctx {n_ctx} n_vocab {n_vocab} n_chunk {n_chunk}")));
    }
    let (n_ctx, n_vocab, n_chunk) = (n_ctx as usize, n_vocab as usize, n_chunk as usize);
    let mut tb = vec![0u8; n_ctx * n_chunk * 4];
    f.read_exact(&mut tb).map_err(|e| bad(format!("{base}: tokens: {e}")))?;
    let tokens: Vec<u32> = tb.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let mut s = load_session(path, n_ctx as u32 + 1)?;
    // ASTRAL_DUMP_FIRST=k scores positions k..n_ctx-2 instead of the second half (dense
    // long-document diagnosis, T25); the record count then tells readers the first position.
    let first = match std::env::var("ASTRAL_DUMP_FIRST") {
        Ok(v) => v.parse::<usize>().ok().filter(|&k| k >= 1 && k < n_ctx - 1).ok_or_else(|| bad(format!("ASTRAL_DUMP_FIRST={v}: need 1..{}", n_ctx - 1)))?,
        Err(_) => n_ctx / 2,
    };
    let nv = 2 * n_vocab.div_ceil(2) + 4;
    let mut w = std::io::BufWriter::new(std::fs::File::create(out).map_err(|e| bad(format!("{out}: {e}")))?);
    let io = |e: std::io::Error| bad(format!("{out}: {e}"));
    w.write_all(&magic).and_then(|_| w.write_all(&i32s)).and_then(|_| w.write_all(&tb)).map_err(io)?;
    let mut rec = vec![0u16; nv];
    let t0 = std::time::Instant::now();
    // ASTRAL_DUMP_PREFILL=1: the serving path, a batched prefill up to the first scored position,
    // then decode steps.
    let prefill = std::env::var_os("ASTRAL_DUMP_PREFILL").is_some();
    for c in 0..n_chunk {
        s.reset()?;
        let toks = &tokens[c * n_ctx..(c + 1) * n_ctx];
        if prefill {
            s.prefill(&toks[..first])?;
        }
        for (j, &t) in toks[..n_ctx - 1].iter().enumerate().skip(if prefill { first } else { 0 }) {
            s.step(t)?;
            if j < first {
                continue;
            }
            let l = s.logits()?;
            if l.len() != n_vocab {
                return Err(bad(format!("vocab mismatch: astral {} vs file {n_vocab}", l.len())));
            }
            encode_log_probs(&l, &mut rec);
            let bytes: Vec<u8> = rec.iter().flat_map(|v| v.to_le_bytes()).collect();
            w.write_all(&bytes).map_err(io)?;
        }
        eprintln!("chunk {}/{n_chunk} done ({:.1} s)", c + 1, t0.elapsed().as_secs_f64());
    }
    w.flush().map_err(io)?;
    Ok(true)
}

/// llama.cpp's `log_softmax` encoder (perplexity.cpp): log-probs within 16 of the max, as
/// u16 steps of (max - min) / 65535 above `min_log_prob`; slots 0..4 hold scale and min.
fn encode_log_probs(logits: &[f32], rec: &mut [u16]) {
    let mut max = logits[0];
    let mut min = logits[0];
    for &v in &logits[1..] {
        max = max.max(v);
        min = min.min(v);
    }
    min = min.max(max - 16.0);
    let sum: f64 = logits.iter().map(|&v| (v - max).exp() as f64).sum();
    let lse = sum.ln() as f32;
    let min_log_prob = min - max - lse;
    let scale = (max - min) / 65535.0;
    rec.fill(0);
    let (sb, mb) = (scale.to_bits(), min_log_prob.to_bits());
    rec[..4].copy_from_slice(&[sb as u16, (sb >> 16) as u16, mb as u16, (mb >> 16) as u16]);
    if scale != 0.0 {
        let inv = 1.0 / scale;
        for (r, &v) in rec[4..].iter_mut().zip(logits) {
            *r = if v > min { (inv * (v - min)).round_ties_even() as u16 } else { 0 };
        }
    }
}

fn cmd_qwen_generate(dir: &str, n: &str, ids: &[String]) -> Result<bool> {
    let n: usize = n.parse().map_err(|_| astral_core::AstralError::Invalid("n must be a number".into()))?;
    let prompt: Vec<u32> = ids.iter().filter_map(|t| t.parse().ok()).collect();
    let ctx = astral_gpu::Context::new(astral_gpu::Device::get(0)?)?;
    let t0 = std::time::Instant::now();
    let w = astral_model::qwen35::load(&ctx, std::path::Path::new(dir))?;
    eprintln!("loaded {:.2} GB in {:.1} s ({} MXFP4 blocks re-encoded)", w.bytes as f64 / 1e9, t0.elapsed().as_secs_f64(), w.reencoded_blocks);
    let warm: usize = std::env::var("ASTRAL_WARM_PREFILL").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut s = astral_model::qwen35::Session::new(&ctx, w, (prompt.len() + n + 1).max(warm).min(8192) as u32)?;
    if warm > 0 {
        let toks: Vec<u32> = prompt.iter().cycle().take(warm).copied().collect();
        s.prefill(&toks)?;
        s.reset()?;
        let t = std::time::Instant::now();
        s.prefill(&toks)?;
        println!("prefill {warm} tokens: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        s.reset()?;
    }
    let t1 = std::time::Instant::now();
    let mut next = if std::env::var_os("ASTRAL_NO_PREFILL").is_some() {
        let mut x = 0;
        for &t in &prompt {
            x = s.step(t)?;
        }
        x
    } else {
        s.prefill(&prompt)?
    };
    let tp = t1.elapsed().as_secs_f64();
    let mut out = vec![next];
    let t2 = std::time::Instant::now();
    for _ in 1..n {
        next = s.step(next)?;
        out.push(next);
    }
    let td = t2.elapsed().as_secs_f64();
    println!("tokens: {out:?}");
    println!("prompt {} tok in {:.3} s; decode {} tok at {:.1} tok/s", prompt.len(), tp, n.saturating_sub(1), (n.saturating_sub(1)) as f64 / td.max(1e-9));
    Ok(true)
}

fn cmd_generate(path: &str, n: &str, ids: &[String]) -> Result<bool> {
    let n: usize = n.parse().map_err(|_| astral_core::AstralError::Invalid("n must be a number".into()))?;
    let prompt: Vec<u32> = ids.iter().filter_map(|t| t.parse().ok()).collect();
    if prompt.is_empty() {
        return Err(astral_core::AstralError::Invalid("no token ids".into()));
    }
    let warm_len: usize = std::env::var("ASTRAL_WARM_PREFILL").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let cap = std::env::var("ASTRAL_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or((prompt.len() + n + 1).max(warm_len).min(astral_model::inkling::MAX_CONTEXT as usize) as u32);
    let mut s = load_session(path, cap)?;
    if let Ok(w) = std::env::var("ASTRAL_TOUCH") {
        s.touch(&w)?;
    }
    if let Ok(warm) = std::env::var("ASTRAL_WARM_PREFILL") {
        // Time a prefill of `warm` tokens of the prompt repeated (cold start excluded).
        let k: usize = warm.parse().unwrap_or(512);
        let toks: Vec<u32> = prompt.iter().cycle().take(k).copied().collect();
        s.prefill(&toks)?;
        s.reset()?;
        let t = std::time::Instant::now();
        s.prefill(&toks)?;
        println!("prefill {k} tokens: {:.1} ms ({:.0} tok/s)", t.elapsed().as_secs_f64() * 1e3, k as f64 / t.elapsed().as_secs_f64());
        s.reset()?;
    }
    let t0 = std::time::Instant::now();
    let mut next = if std::env::var_os("ASTRAL_NO_PREFILL").is_some() {
        let mut n = 0;
        for &t in &prompt {
            n = s.step(t)?;
        }
        n
    } else {
        s.prefill(&prompt)?
    };
    let tp = t0.elapsed().as_secs_f64();
    let mut out = vec![next];
    let t1 = std::time::Instant::now();
    for _ in 1..n {
        next = s.step(next)?;
        out.push(next);
    }
    let td = t1.elapsed().as_secs_f64();
    println!("tokens: {out:?}");
    println!("prompt {} tok in {:.3} s; decode {} tok at {:.1} tok/s", prompt.len(), tp, n - 1, (n - 1) as f64 / td.max(1e-9));
    Ok(true)
}

/// One manifest line: name, sorted unique lengths, token ids.
type ManifestJob = (String, Vec<usize>, Vec<u32>);

/// Reads a `name<TAB>L1,L2,..<TAB>ids` manifest; returns the jobs and the largest length.
fn read_manifest(manifest: &str) -> Result<(Vec<ManifestJob>, usize)> {
    let bad = |m: String| astral_core::AstralError::Invalid(m);
    let text = std::fs::read_to_string(manifest)?;
    let mut jobs: Vec<ManifestJob> = Vec::new();
    let mut cap = 0;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let mut f = line.splitn(3, '\t');
        let (Some(name), Some(lens), Some(ids)) = (f.next(), f.next(), f.next()) else {
            return Err(bad(format!("{manifest}: expected name<TAB>lengths<TAB>ids, got {line:?}")));
        };
        let ids = ids.split_whitespace().map(|t| t.parse::<u32>().map_err(|_| bad(format!("{name}: bad token id {t:?}")))).collect::<Result<Vec<_>>>()?;
        let mut lens = lens.split(',').map(|t| t.trim().parse::<usize>().map_err(|_| bad(format!("{name}: bad length {t:?}")))).collect::<Result<Vec<_>>>()?;
        lens.sort_unstable();
        lens.dedup();
        match (lens.first(), lens.last()) {
            (Some(&lo), Some(&hi)) if lo > 0 && hi <= ids.len() => cap = cap.max(hi),
            _ => return Err(bad(format!("{name}: lengths must be in 1..={}", ids.len()))),
        }
        jobs.push((name.to_string(), lens, ids));
    }
    if jobs.is_empty() {
        return Err(bad(format!("{manifest}: no sequences")));
    }
    Ok((jobs, cap))
}

/// `dump-final-logits <model> <manifest> <out dir>`: raw logits after the first L tokens of a
/// sequence, once from token-by-token steps (`<name>_<L>.step.f32`) and once from a batched
/// prefill of the same L tokens (`<name>_<L>.prefill.f32`), and for every length after the first
/// along the serving path: prefill the first length, then decode steps (`<name>_<L>.decode.f32`);
/// little-endian f32 over the model vocabulary. Each manifest line is `name<TAB>L1,L2,...<TAB>ids...`. Pairs with
/// `bench/llamacpp_compare/final_logits.cpp` (the same files from llama.cpp) and
/// `logit_compare.py`.
fn cmd_dump_final_logits(path: &str, manifest: &str, out: &str) -> Result<bool> {
    let (jobs, cap) = read_manifest(manifest)?;
    std::fs::create_dir_all(out)?;
    let save = |name: &str, len: usize, kind: &str, v: &[f32]| -> Result<()> {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(format!("{out}/{name}_{len}.{kind}.f32"), bytes)?;
        Ok(())
    };
    let mut s = load_session(path, (cap + 8) as u32)?;
    // ASTRAL_DUMP_KINDS (default "step,prefill,decode") selects the paths, e.g. "prefill,decode"
    // at lengths where stepping from the first token takes too long.
    let kinds = std::env::var("ASTRAL_DUMP_KINDS").unwrap_or_else(|_| "step,prefill,decode".into());
    let want = |k: &str| kinds.split(',').any(|x| x.trim() == k);
    for (name, lens, ids) in &jobs {
        if want("step") {
            s.reset()?;
            let mut next = 0;
            for (i, &t) in ids[..lens[lens.len() - 1]].iter().enumerate() {
                s.step(t)?;
                if lens.get(next) == Some(&(i + 1)) {
                    save(name, i + 1, "step", &s.logits()?)?;
                    next += 1;
                }
            }
        }
        if want("prefill") {
            for &len in lens {
                s.reset()?;
                s.prefill(&ids[..len])?;
                save(name, len, "prefill", &s.logits()?)?;
            }
        }
        if !want("decode") {
            println!("{name}: {} lengths up to {}", lens.len(), lens[lens.len() - 1]);
            continue;
        }
        // The serving path: prefill the first length (the prompt), then one decode step per token.
        s.reset()?;
        s.prefill(&ids[..lens[0]])?;
        let mut next = 1;
        for (i, &t) in ids.iter().enumerate().take(lens[lens.len() - 1]).skip(lens[0]) {
            s.step(t)?;
            if lens.get(next) == Some(&(i + 1)) {
                save(name, i + 1, "decode", &s.logits()?)?;
                next += 1;
            }
        }
        println!("{name}: {} lengths up to {}", lens.len(), lens[lens.len() - 1]);
    }
    Ok(true)
}

/// `ls-diag <model> <first> <ids...|@file>`: loop self-speculation diagnosis (`ASTRAL_LS_DIAG=1`;
/// without it the per-pass picks are empty, the control that the diagnosis leaves the
/// full-depth picks unchanged), teacher-forced. Prefills the first `first` tokens, then feeds the rest one decode
/// step at a time; per step one JSON line with the position, the full-depth greedy pick, the pick
/// after each earlier pass (coda and head on that pass's output, real coda history) and the next
/// token of the sequence (`ASTRAL_LS_GEN=n`: then n more steps fed with the full-depth picks).
/// Ends with the agreement of each pass's pick with the full-depth one.
fn cmd_ls_diag(path: &str, first: &str, ids: &[String]) -> Result<bool> {
    let bad = |m: String| astral_core::AstralError::Invalid(m);
    let prompt = read_ids(ids)?;
    let first: usize = first.parse().ok().filter(|&f: &usize| f >= 1 && f < prompt.len()).ok_or_else(|| bad(format!("first must be in 1..{}", prompt.len())))?;
    // ASTRAL_LS_GEN=n: after the sequence, n more steps on the full-depth greedy picks.
    let extra: usize = std::env::var("ASTRAL_LS_GEN").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut s = load_session(path, (prompt.len() + extra + 8) as u32)?;
    s.prefill(&prompt[..first])?;
    let (mut n, mut agree) = (0usize, Vec::<usize>::new());
    let t0 = std::time::Instant::now();
    let (mut t, mut tok) = (first, prompt[first]);
    loop {
        let r = s.step(tok)?;
        let picks = s.ls_picks()?.unwrap_or_default();
        agree.resize(picks.len(), 0);
        for (a, &p) in agree.iter_mut().zip(&picks) {
            *a += usize::from(p == r);
        }
        n += 1;
        let next = if t + 1 < prompt.len() { Some(prompt[t + 1]) } else if t + 1 < prompt.len() + extra { Some(r) } else { None };
        println!("{{\"pos\":{t},\"final\":{r},\"picks\":{picks:?},\"next\":{},\"generated\":{}}}", next.map_or("null".into(), |v| v.to_string()), t >= prompt.len());
        match next {
            Some(x) => (t, tok) = (t + 1, x),
            None => break,
        }
    }
    let rates: Vec<String> = agree.iter().map(|&a| format!("{:.4}", a as f64 / n.max(1) as f64)).collect();
    eprintln!("ls-diag: {n} steps in {:.1} s; agreement with the full depth by pass: [{}]", t0.elapsed().as_secs_f64(), rates.join(", "));
    Ok(true)
}

/// `cmp-prefill <gguf> <ids...|@file>`: last-token logits of a batched prefill vs feeding the
/// same tokens one decode step at a time (max |diff|, argmax of each).
fn cmd_cmp_prefill(path: &str, ids: &[String]) -> Result<bool> {
    let prompt = read_ids(ids)?;
    let mut s = load_session(path, (prompt.len() + 8) as u32)?;
    for &t in &prompt {
        s.step(t)?;
    }
    let a = s.logits()?;
    s.reset()?;
    s.prefill(&prompt)?;
    let b = s.logits()?;
    let am = |v: &[f32]| v.iter().enumerate().fold((0, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m });
    let diff = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    println!("{} tokens: max |logit diff| {diff:.4}; argmax step {:?} prefill {:?}", prompt.len(), am(&a), am(&b));
    // Where the difference sits: within 16 of the top logit (the tokens with any probability)
    // and over the whole vocabulary, plus KL(step || prefill) in nats.
    let top = am(&a).1;
    let (mut near, mut near_i, mut far_i) = (0f32, 0usize, 0usize);
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        let d = (x - y).abs();
        if *x > top - 16.0 && d > near {
            near = d;
            near_i = i;
        }
        if d >= diff {
            far_i = i;
        }
    }
    let lse = |v: &[f32]| {
        let m = v.iter().fold(f32::MIN, |m, &x| m.max(x)) as f64;
        m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
    };
    let (la, lb) = (lse(&a), lse(&b));
    let kl: f64 = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| {
            let pa = x as f64 - la;
            pa.exp() * (pa - (y as f64 - lb))
        })
        .sum();
    println!(
        "  near-top max |diff| {near:.4} at {near_i} (step {:.3} prefill {:.3}); largest at {far_i} (step {:.3} prefill {:.3}); KL(step||prefill) {kl:.3e}",
        a[near_i], b[near_i], a[far_i], b[far_i]
    );
    Ok(true)
}

/// `serve <model> [--checkpoint <dir>] [--no-draft] [--k N] [--host H] [--port P] [--ctx N]
/// [--name S] [--queue N]`: the OpenAI-compatible server. `<model>` is a `.gguf` file or a
/// safetensors checkpoint directory. The checkpoint directory provides the tokenizer and the
/// DSpark draft (used unless `--no-draft`); it defaults to `<model>` when that is a directory.
fn cmd_serve(a: &[String]) -> Result<bool> {
    let bad = |m: String| astral_core::AstralError::Invalid(m);
    let gguf = a.first().ok_or_else(|| bad("serve: missing model (a .gguf file or a checkpoint directory)".into()))?.clone();
    let (mut ckpt, mut draft, mut k, mut host, mut port, mut ctx, mut name, mut queue) = (None::<String>, true, 6usize, "0.0.0.0".to_string(), 8000u16, 8192u32, "inkling-2-mini".to_string(), 64usize);
    let mut i = 1;
    while i < a.len() {
        let val = |i: usize| a.get(i + 1).cloned().ok_or_else(|| bad(format!("serve: {} needs a value", a[i])));
        match a[i].as_str() {
            "--checkpoint" => {
                ckpt = Some(val(i)?);
                i += 1;
            }
            "--no-draft" => draft = false,
            "--k" => {
                k = val(i)?.parse().map_err(|_| bad("serve: --k must be a number".into()))?;
                i += 1;
            }
            "--host" => {
                host = val(i)?;
                i += 1;
            }
            "--port" => {
                port = val(i)?.parse().map_err(|_| bad("serve: --port must be a number".into()))?;
                i += 1;
            }
            "--ctx" => {
                ctx = val(i)?.parse().map_err(|_| bad("serve: --ctx must be a number".into()))?;
                i += 1;
            }
            "--name" => {
                name = val(i)?;
                i += 1;
            }
            "--queue" => {
                queue = val(i)?.parse().map_err(|_| bad("serve: --queue must be a number".into()))?;
                i += 1;
            }
            o => return Err(bad(format!("serve: unknown option {o}"))),
        }
        i += 1;
    }
    let ckpt = ckpt.or_else(|| std::path::Path::new(&gguf).is_dir().then(|| gguf.clone()));
    let ckpt = std::path::PathBuf::from(ckpt.ok_or_else(|| bad("serve: --checkpoint <dir> (tokenizer and draft) is required with a .gguf model".into()))?);
    let engine = astral_api::engine::EngineConfig { gguf: gguf.into(), draft: draft.then(|| ckpt.clone()), k, ctx, queue };
    astral_api::serve(astral_api::ServerConfig { engine, tokenizer_dir: ckpt, host, port, model_name: name })?;
    Ok(true)
}

/// Token ids from the arguments, or from a file of whitespace-separated ids (`@path`).
fn read_ids(ids: &[String]) -> Result<Vec<u32>> {
    let words: Vec<String> = match ids {
        [one] if one.starts_with('@') => std::fs::read_to_string(&one[1..])?.split_whitespace().map(str::to_string).collect(),
        _ => ids.to_vec(),
    };
    let v: Vec<u32> = words.iter().filter_map(|t| t.trim_matches(|c| c == ',' || c == '[' || c == ']').parse().ok()).collect();
    if v.is_empty() {
        return Err(astral_core::AstralError::Invalid("no token ids".into()));
    }
    Ok(v)
}

/// `trace <model> <manifest> <out dir>`: read-only probes of the looped forward pass
/// (docs/trace.md). Each manifest sequence is fed token by token; the step that consumes token
/// L - 1 for each listed length L is probed. Writes `<out dir>/trace.jsonl` (one record per layer
/// visit and one for the head, per probed position) and the probed step's final logits as
/// `<out dir>/<name>_<L>.trace.f32` (same format as dump-final-logits). The attention output is
/// recomputed from the V cache at the first probed position of each sequence, or at every
/// one with ASTRAL_TRACE_VALIDATE=all.
fn cmd_trace(path: &str, manifest: &str, out: &str) -> Result<bool> {
    use std::io::Write;
    let (jobs, cap) = read_manifest(manifest)?;
    std::fs::create_dir_all(out)?;
    let all = std::env::var("ASTRAL_TRACE_VALIDATE").is_ok_and(|v| v == "all");
    let mut s = load_session(path, (cap + 8) as u32)?;
    let mut f = std::io::BufWriter::new(std::fs::File::create(format!("{out}/trace.jsonl"))?);
    let t0 = std::time::Instant::now();
    // ASTRAL_TRACE_PREFILL=1: prefill up to the first probed step instead of stepping there.
    let prefill = std::env::var_os("ASTRAL_TRACE_PREFILL").is_some();
    for (name, lens, ids) in &jobs {
        s.reset()?;
        let mut next = 0;
        let skip = if prefill { lens[0] - 1 } else { 0 };
        if skip > 0 {
            s.prefill(&ids[..skip])?;
        }
        for (i, &t) in ids[..lens[lens.len() - 1]].iter().enumerate().skip(skip) {
            let probe = lens.get(next) == Some(&(i + 1));
            if probe {
                s.arm_probe()?;
            }
            s.step(t)?;
            if probe {
                let recs = s.probe_report(t, ids.get(i + 1).copied(), all || next == 0)?;
                for mut r in recs {
                    r["seq"] = serde_json::json!(name);
                    r["len"] = serde_json::json!(i + 1);
                    writeln!(f, "{r}")?;
                }
                let bytes: Vec<u8> = s.logits()?.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(format!("{out}/{name}_{}.trace.f32", i + 1), bytes)?;
                next += 1;
            }
        }
        eprintln!("{name}: {} positions ({:.1} s)", lens.len(), t0.elapsed().as_secs_f64());
    }
    f.flush()?;
    Ok(true)
}

/// `dump-verify-logits <model> <checkpoint dir> <k> <manifest> <out dir>`: logits of DSpark verify
/// passes. Each sequence is prefilled up to its first length (the prompt), then speculative
/// rounds of k drafts (k + 1 rows per verify pass; needs ASTRAL_SPEC_FIXED=1) accept along the
/// manifest sequence. Every later length is the output of exactly one verify row whose inputs
/// were all accepted tokens; that row's logits go to `<name>_<L>.verify<k>.f32` and a line
/// `name, length, row, rows` to `<out dir>/verify<k>.tsv`.
fn cmd_dump_verify_logits(path: &str, dir: &str, k: &str, manifest: &str, out: &str) -> Result<bool> {
    let bad = |m: String| astral_core::AstralError::Invalid(m);
    let k: usize = k.parse().map_err(|_| bad("k must be a number".into()))?;
    if std::env::var_os("ASTRAL_SPEC_FIXED").is_none() {
        return Err(bad("dump-verify-logits: set ASTRAL_SPEC_FIXED=1 so every round verifies k drafts".into()));
    }
    let (jobs, cap) = read_manifest(manifest)?;
    std::fs::create_dir_all(out)?;
    let mut s = load_session(path, (cap + k + 16) as u32)?;
    s.enable_spec(std::path::Path::new(dir), k)?;
    let mut index = String::new();
    for (name, lens, ids) in &jobs {
        s.reset()?;
        s.spec_force(ids)?;
        s.spec_begin(&ids[..lens[0]])?;
        let mut p = lens[0];
        let mut want = lens[1..].iter().copied().peekable();
        while want.peek().is_some() {
            let got = s.spec_next()?;
            let c = got.len();
            if c == 0 {
                return Err(bad(format!("{name}: context full at position {p}")));
            }
            let end = ids.len().min(p + 1 + c);
            if p + 1 < end && got[..end - p - 1] != ids[p + 1..end] {
                return Err(bad(format!("{name}: round at {p} did not follow the forced sequence")));
            }
            let rows = s.spec_round_logits()?;
            while let Some(l) = want.next_if(|&l| l <= p + c) {
                let row = l - p - 1;
                let bytes: Vec<u8> = rows[row].iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(format!("{out}/{name}_{l}.verify{k}.f32"), bytes)?;
                index.push_str(&format!("{name}\t{l}\t{row}\t{}\n", k + 1));
            }
            p += c;
        }
        println!("{name}: {} lengths up to {}", lens.len() - 1, lens[lens.len() - 1]);
    }
    std::fs::write(format!("{out}/verify{k}.tsv"), index)?;
    Ok(true)
}

/// `spec-generate <gguf> <native checkpoint dir> <k> <n> <ids... | @file>`: greedy generation
/// with DSpark speculative decoding (k drafts verified per round), then the same with plain
/// decoding for comparison (skipped with ASTRAL_SPEC_NO_CHECK=1).
fn cmd_spec_generate(path: &str, dir: &str, k: &str, n: &str, ids: &[String]) -> Result<bool> {
    let bad = |m: &str| astral_core::AstralError::Invalid(m.into());
    let k: usize = k.parse().map_err(|_| bad("k must be a number"))?;
    let n: usize = n.parse().map_err(|_| bad("n must be a number"))?;
    let prompt = read_ids(ids)?;
    let cap = std::env::var("ASTRAL_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or((prompt.len() + n + 64).min(astral_model::inkling::MAX_CONTEXT as usize) as u32);
    let mut s = load_session(path, cap)?;
    let t = std::time::Instant::now();
    s.enable_spec(std::path::Path::new(dir), k)?;
    // Diagnostics: accept against a fixed continuation (e.g. another engine's output).
    if let Ok(f) = std::env::var("ASTRAL_SPEC_FORCE") {
        let cont = read_ids(&[f])?;
        let mut seq = prompt.clone();
        seq.extend(&cont);
        s.spec_force(&seq)?;
        eprintln!("forced continuation: {} tokens", cont.len());
    }
    eprintln!("draft loaded in {:.1} s", t.elapsed().as_secs_f64());
    let reps: usize = std::env::var("ASTRAL_SPEC_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    let mut spec_out = Vec::new();
    for rep in 0..reps {
        s.reset()?;
        let t0 = std::time::Instant::now();
        let first = s.prefill(&prompt)?;
        let tp = t0.elapsed().as_secs_f64();
        s.reset()?;
        let before = s.spec_stats().unwrap_or_default();
        let t1 = std::time::Instant::now();
        spec_out = s.spec_generate(&prompt, n)?;
        let total = t1.elapsed().as_secs_f64();
        let st = s.spec_stats().unwrap_or_default();
        let (rounds, acc, drafted) = (st.rounds - before.rounds, st.accepted - before.accepted, st.drafted - before.drafted);
        let decode = (total - tp).max(1e-9);
        println!(
            "run {rep}: prefill {} tok {:.1} ms; decode {} tok in {:.3} s = {:.1} tok/s; {rounds} rounds, {:.2} tokens/round, acceptance {:.1}% of {drafted} verified drafts",
            prompt.len(),
            tp * 1e3,
            spec_out.len() - 1,
            decode,
            (spec_out.len() - 1) as f64 / decode,
            (spec_out.len() - 1) as f64 / rounds.max(1) as f64,
            100.0 * acc as f64 / drafted.max(1) as f64
        );
        let _ = first;
    }
    println!("spec tokens: {spec_out:?}");
    if std::env::var_os("ASTRAL_SPEC_NO_CHECK").is_none() {
        s.reset()?;
        let mut next = s.prefill(&prompt)?;
        let mut plain = vec![next];
        let t1 = std::time::Instant::now();
        for _ in 1..n {
            next = s.step(next)?;
            plain.push(next);
        }
        let td = t1.elapsed().as_secs_f64();
        let same = spec_out.iter().zip(&plain).take_while(|(a, b)| a == b).count();
        println!("plain decode {} tok at {:.1} tok/s; spec output equals plain greedy for the first {same} / {} tokens", n - 1, (n - 1) as f64 / td.max(1e-9), plain.len());
        // Teacher-forced check: plain decode along the spec tokens; at every position the spec
        // token must be the plain argmax, up to near-ties (logit gap below the batched-vs-decode
        // numerical noise).
        s.reset()?;
        let mut pred = s.prefill(&prompt)?;
        let mut logits = s.logits()?;
        let (mut miss, mut worst) = (0usize, 0f32);
        for (i, &t) in spec_out.iter().enumerate() {
            if t != pred {
                miss += 1;
                worst = worst.max(logits[pred as usize] - logits[t as usize]);
            }
            if i + 1 < spec_out.len() {
                pred = s.step(t)?;
                logits = s.logits()?;
            }
        }
        println!("teacher-forced: spec token != plain argmax at {miss} / {} positions, largest logit gap {worst:.3}", spec_out.len());
    }
    Ok(true)
}

fn cmd_kbench_mxfp4(a: &[String], mma: bool) -> Result<bool> {
    let p = |i: usize, d: usize| a.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
    let (n, k, b, iters) = (p(0, 16384), p(1, 8192), p(2, 1), p(3, 50));
    let ctx = astral_gpu::Context::new(astral_gpu::Device::get(0)?)?;
    let (name, r) =
        if mma { ("mxfp4 mma", astral_testing::kbench::gemm_mxfp4_mma(&ctx, n, k, b, iters as u32, false)?) } else { ("mxfp4 gemv", astral_testing::kbench::gemv_mxfp4(&ctx, n, k, b, iters as u32)?) };
    println!("{name} n{n} k{k} b{b}: {:.1} us/call, {:.0} GB/s of weights", r.us_per_call, r.weight_gbps);
    Ok(true)
}

fn cmd_kbench_mxfp4_f16(a: &[String]) -> Result<bool> {
    let p = |i: usize, d: usize| a.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
    let (n, k, b, iters) = (p(0, 16384), p(1, 8192), p(2, 1), p(3, 50));
    let ctx = astral_gpu::Context::new(astral_gpu::Device::get(0)?)?;
    let r = astral_testing::kbench::gemm_mxfp4_mma(&ctx, n, k, b, iters as u32, true)?;
    println!("mxfp4 mma f16x n{n} k{k} b{b}: {:.1} us/call, {:.0} GB/s of weights", r.us_per_call, r.weight_gbps);
    Ok(true)
}
