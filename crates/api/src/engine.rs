// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Synthetic Minds LTD
//! The engine thread: owns the GPU session (and the DSpark draft), runs one request at a
//! time from a bounded queue and streams tokens back over a per-request channel. A dropped
//! receiver cancels its request at the next token or round.

use astral_core::{AstralError, Result};
use astral_model::inkling::{Session, Weights};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender, SyncSender, TrySendError};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub gguf: PathBuf,
    /// Native checkpoint directory with `mtp.safetensors` (the DSpark draft); None: no draft.
    pub draft: Option<PathBuf>,
    /// Drafted tokens verified per round.
    pub k: usize,
    /// Context capacity (positions) of the session.
    pub ctx: u32,
    /// Requests waiting beyond the running one.
    pub queue: usize,
}

pub enum Event {
    Token(u32),
    /// The request finished: "stop" (a stop token) or "length".
    Done {
        finish: &'static str,
    },
    Error(String),
}

pub struct Job {
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub stop_ids: Vec<u32>,
    pub use_draft: bool,
    pub tx: Sender<Event>,
}

pub struct Engine {
    tx: SyncSender<Job>,
    pub capacity: u32,
    pub draft: bool,
    pub k: usize,
}

impl Engine {
    /// Loads the model on a new engine thread and returns once it is ready to serve.
    pub fn start(cfg: EngineConfig) -> Result<Arc<Engine>> {
        let (tx, rx) = mpsc::sync_channel::<Job>(cfg.queue.max(1));
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(u32, bool)>>();
        let k = cfg.k;
        std::thread::Builder::new()
            .name("astral-engine".into())
            .spawn(move || {
                let mut s = match load(&cfg) {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let _ = ready_tx.send(Ok((s.capacity(), s.spec_enabled())));
                run(&mut s, rx);
            })
            .map_err(|e| AstralError::Invalid(format!("engine thread: {e}")))?;
        let (capacity, draft) = ready_rx.recv().map_err(|_| AstralError::Invalid("engine thread died while loading".into()))??;
        Ok(Arc::new(Engine { tx, capacity, draft, k }))
    }

    /// Queues a request; `Err` when the queue is full (the caller answers 503).
    pub fn submit(&self, job: Job) -> std::result::Result<(), String> {
        match self.tx.try_send(job) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err("server busy: request queue is full".into()),
            Err(TrySendError::Disconnected(_)) => Err("engine stopped".into()),
        }
    }

    /// Positions a request needs beyond its prompt and output (a speculative round may write
    /// up to k + 1 positions past the last one it keeps).
    pub fn margin(&self) -> usize {
        if self.draft {
            self.k + 2
        } else {
            1
        }
    }
}

fn load(cfg: &EngineConfig) -> Result<Session> {
    let ctx = astral_gpu::Context::new(astral_gpu::Device::get(0)?)?;
    let g = astral_loader::Gguf::open(&cfg.gguf)?;
    let t = std::time::Instant::now();
    let w = Weights::load(&ctx, &g)?;
    eprintln!("astral: loaded {:.2} GB of weights in {:.1} s", w.bytes as f64 / 1e9, t.elapsed().as_secs_f64());
    let mut s = Session::new(&ctx, w, cfg.ctx)?;
    if let Some(dir) = &cfg.draft {
        s.enable_spec(dir, cfg.k)?;
    }
    // Warm-up: capture the decode / speculative graphs before the first request.
    let warm = [200028u32, 760, 6511, 314];
    s.prefill(&warm)?;
    let mut t = 0;
    for _ in 0..4 {
        t = s.step(t)?;
    }
    s.reset()?;
    if s.spec_enabled() {
        s.spec_begin(&warm)?;
        for _ in 0..4 {
            s.spec_next()?;
        }
        s.reset()?;
    }
    Ok(s)
}

fn run(s: &mut Session, rx: Receiver<Job>) {
    while let Ok(job) = rx.recv() {
        if let Err(e) = run_job(s, &job) {
            let _ = job.tx.send(Event::Error(e.to_string()));
        }
    }
}

fn run_job(s: &mut Session, job: &Job) -> Result<()> {
    if job.prompt.is_empty() || job.max_tokens == 0 {
        let _ = job.tx.send(Event::Done { finish: "length" });
        return Ok(());
    }
    s.reset()?;
    let mut n = 0usize;
    // Emits one token; false when the request is over (stop token, length or a cancel).
    let mut emit = |t: u32| -> Option<&'static str> {
        if job.stop_ids.contains(&t) {
            return Some("stop");
        }
        if job.tx.send(Event::Token(t)).is_err() {
            return Some("cancelled");
        }
        n += 1;
        if n >= job.max_tokens {
            return Some("length");
        }
        None
    };
    let finish = if job.use_draft && s.spec_enabled() {
        let first = s.spec_begin(&job.prompt)?;
        let mut fin = emit(first);
        while fin.is_none() {
            let r = s.spec_next()?;
            if r.is_empty() {
                fin = Some("length");
                break;
            }
            for t in r {
                fin = emit(t);
                if fin.is_some() {
                    break;
                }
            }
        }
        fin
    } else {
        let mut t = s.prefill(&job.prompt)?;
        let mut fin = emit(t);
        while fin.is_none() {
            if s.pos >= s.capacity() {
                fin = Some("length");
                break;
            }
            t = s.step(t)?;
            fin = emit(t);
        }
        fin
    };
    if let Some(f) = finish {
        if f != "cancelled" {
            let _ = job.tx.send(Event::Done { finish: f });
        }
    }
    Ok(())
}
