// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Synthetic Minds LTD
//! `astral-runtime`: the engine thread (R-RT-2, ADR-0004).
//!
//! One dedicated OS thread (optionally pinned to a core) owns all model state and runs
//! the step loop with no async and no `.await`. Requests arrive over a **bounded**
//! channel (a full queue is reported to the caller, never grown without limit); tokens
//! leave over per-request channels. A dropped receiver cancels its request at the next
//! step and frees its slot (cancel-on-disconnect, spec 07 §1).
//!
//! The model is anything implementing [`StepModel`]; P1 plugs the real Inkling forward
//! pass in here. Admission is deliberately simple (FIFO up to `max_batch`);
//! `astral-sched` replaces it in P1.

#![forbid(unsafe_code)]

use astral_core::{AstralError, Result};
use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver, Sender, SyncSender, TryRecvError, TrySendError};
use std::thread::JoinHandle;
use std::time::Instant;

pub type TokenId = u32;

#[derive(Debug, Clone)]
pub struct Request {
    pub id: u64,
    pub prompt: Vec<TokenId>,
    pub max_tokens: usize,
    /// Generation stops after emitting any of these.
    pub stop_tokens: Vec<TokenId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Length,
    Stop,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Token(TokenId),
    Done(FinishReason),
    Error(String),
}

/// A sequence as the model sees it in one step.
pub struct SeqView<'a> {
    pub id: u64,
    pub tokens: &'a [TokenId],
    /// Trailing tokens of `tokens` the model has not processed yet (whole prompt on first step).
    pub new_tokens: usize,
}

/// One forward step over a batch: returns the next token for every sequence, in order.
pub trait StepModel: Send + 'static {
    fn step(&mut self, batch: &[SeqView<'_>]) -> Result<Vec<TokenId>>;
    /// Called when a sequence leaves the batch (finished, cancelled or failed) so the
    /// model can free its KV and state.
    fn release(&mut self, _id: u64) {}
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub max_batch: usize,
    pub queue_depth: usize,
    /// Pin the engine thread to this core, if it exists.
    pub pin_core: Option<usize>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self { max_batch: 32, queue_depth: 1024, pin_core: None }
    }
}

#[derive(Debug, Clone, Default)]
pub struct EngineStats {
    pub steps: u64,
    pub tokens_out: u64,
    pub finished: u64,
    pub cancelled: u64,
    pub failed: u64,
    pub max_batch_seen: usize,
    pub pinned: bool,
    /// Host time spent outside `StepModel::step` while the engine had work.
    pub overhead_ns: u128,
    pub step_ns: u128,
}

impl EngineStats {
    /// Mean engine-side host overhead per step, in microseconds.
    pub fn overhead_us_per_step(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.overhead_ns as f64 / self.steps as f64 / 1e3
        }
    }
}

enum Command {
    Submit(Request, Sender<Event>),
    Shutdown,
}

struct Seq {
    req: Request,
    tokens: Vec<TokenId>,
    processed: usize,
    generated: usize,
    out: Sender<Event>,
}

#[derive(Clone, Copy)]
enum Outcome {
    Continue,
    Finished,
    Cancelled,
}

pub struct Engine {
    tx: SyncSender<Command>,
    handle: Option<JoinHandle<EngineStats>>,
}

impl Engine {
    pub fn start<M: StepModel>(model: M, cfg: EngineConfig) -> Result<Self> {
        if cfg.max_batch == 0 || cfg.queue_depth == 0 {
            return Err(AstralError::Invalid("max_batch and queue_depth must be > 0".into()));
        }
        let (tx, rx) = mpsc::sync_channel(cfg.queue_depth);
        let handle = std::thread::Builder::new().name("astral-engine".into()).spawn(move || engine_loop(model, cfg, rx)).map_err(AstralError::Io)?;
        Ok(Self { tx, handle: Some(handle) })
    }

    /// Queues a request. Fails immediately (never blocks) if the queue is full.
    pub fn submit(&self, req: Request) -> Result<Receiver<Event>> {
        let (out_tx, out_rx) = mpsc::channel();
        match self.tx.try_send(Command::Submit(req, out_tx)) {
            Ok(()) => Ok(out_rx),
            Err(TrySendError::Full(_)) => Err(AstralError::Unsupported("engine queue full".into())),
            Err(TrySendError::Disconnected(_)) => Err(AstralError::Invalid("engine stopped".into())),
        }
    }

    /// Stops after in-flight and queued sequences finish; returns statistics.
    pub fn shutdown(mut self) -> EngineStats {
        let _ = self.tx.send(Command::Shutdown);
        self.handle.take().and_then(|h| h.join().ok()).unwrap_or_default()
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if let Some(h) = self.handle.take() {
            let _ = self.tx.send(Command::Shutdown);
            let _ = h.join();
        }
    }
}

fn enqueue(cmd: Command, waiting: &mut VecDeque<Seq>, shutting_down: &mut bool) {
    match cmd {
        Command::Submit(req, out) => {
            let tokens = req.prompt.clone();
            waiting.push_back(Seq { req, tokens, processed: 0, generated: 0, out });
        }
        Command::Shutdown => *shutting_down = true,
    }
}

fn engine_loop<M: StepModel>(mut model: M, cfg: EngineConfig, rx: Receiver<Command>) -> EngineStats {
    let mut stats = EngineStats::default();
    if let Some(core) = cfg.pin_core {
        if let Some(id) = core_affinity::get_core_ids().and_then(|ids| ids.into_iter().find(|c| c.id == core)) {
            stats.pinned = core_affinity::set_for_current(id);
        }
    }
    let mut waiting: VecDeque<Seq> = VecDeque::new();
    let mut running: Vec<Seq> = Vec::new();
    let mut shutting_down = false;

    loop {
        // Block only when idle; otherwise drain the queue without waiting.
        if running.is_empty() && waiting.is_empty() {
            if shutting_down {
                break;
            }
            match rx.recv() {
                Ok(cmd) => enqueue(cmd, &mut waiting, &mut shutting_down),
                Err(_) => break,
            }
        }
        let t0 = Instant::now();
        // Pull at most `queue_depth` into the local queue so backpressure reaches callers.
        while waiting.len() < cfg.queue_depth {
            match rx.try_recv() {
                Ok(cmd) => enqueue(cmd, &mut waiting, &mut shutting_down),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    shutting_down = true;
                    break;
                }
            }
        }
        while running.len() < cfg.max_batch {
            match waiting.pop_front() {
                Some(s) if s.req.max_tokens == 0 || s.tokens.is_empty() => {
                    let _ = s.out.send(Event::Done(FinishReason::Length));
                    stats.finished += 1;
                }
                Some(s) => running.push(s),
                None => break,
            }
        }
        if running.is_empty() {
            continue;
        }
        stats.max_batch_seen = stats.max_batch_seen.max(running.len());

        let views: Vec<SeqView<'_>> = running.iter().map(|s| SeqView { id: s.req.id, tokens: &s.tokens, new_tokens: s.tokens.len() - s.processed }).collect();
        let t1 = Instant::now();
        let result = model.step(&views);
        let t2 = Instant::now();
        drop(views);
        stats.steps += 1;
        stats.step_ns += (t2 - t1).as_nanos();

        let next = match result {
            Ok(next) if next.len() == running.len() => next,
            Ok(next) => {
                let msg = format!("model returned {} tokens for batch of {}", next.len(), running.len());
                fail_all(&mut running, &mut model, &mut stats, msg);
                stats.overhead_ns += (t1 - t0).as_nanos() + t2.elapsed().as_nanos();
                continue;
            }
            Err(e) => {
                fail_all(&mut running, &mut model, &mut stats, e.to_string());
                stats.overhead_ns += (t1 - t0).as_nanos() + t2.elapsed().as_nanos();
                continue;
            }
        };

        let mut outcomes = Vec::with_capacity(running.len());
        for (s, &tok) in running.iter_mut().zip(&next) {
            s.processed = s.tokens.len();
            s.tokens.push(tok);
            s.generated += 1;
            if s.out.send(Event::Token(tok)).is_err() {
                outcomes.push(Outcome::Cancelled);
                continue;
            }
            stats.tokens_out += 1;
            let finish = if s.req.stop_tokens.contains(&tok) {
                Some(FinishReason::Stop)
            } else if s.generated >= s.req.max_tokens {
                Some(FinishReason::Length)
            } else {
                None
            };
            match finish {
                Some(r) => {
                    let _ = s.out.send(Event::Done(r));
                    outcomes.push(Outcome::Finished);
                }
                None => outcomes.push(Outcome::Continue),
            }
        }
        let mut idx = 0;
        running.retain(|s| {
            let o = outcomes[idx];
            idx += 1;
            match o {
                Outcome::Continue => true,
                Outcome::Finished => {
                    stats.finished += 1;
                    model.release(s.req.id);
                    false
                }
                Outcome::Cancelled => {
                    stats.cancelled += 1;
                    model.release(s.req.id);
                    false
                }
            }
        });
        stats.overhead_ns += (t1 - t0).as_nanos() + t2.elapsed().as_nanos();
    }
    stats
}

fn fail_all<M: StepModel>(running: &mut Vec<Seq>, model: &mut M, stats: &mut EngineStats, msg: String) {
    for s in running.drain(..) {
        let _ = s.out.send(Event::Error(msg.clone()));
        model.release(s.req.id);
        stats.failed += 1;
    }
}

/// Deterministic mock: next token = last token + 1 (mod `vocab`). Tracks live sequences.
pub struct CountingModel {
    pub vocab: u32,
    pub live: std::collections::HashSet<u64>,
    pub max_live: usize,
}

impl CountingModel {
    pub fn new(vocab: u32) -> Self {
        Self { vocab, live: Default::default(), max_live: 0 }
    }
}

impl StepModel for CountingModel {
    fn step(&mut self, batch: &[SeqView<'_>]) -> Result<Vec<TokenId>> {
        for s in batch {
            self.live.insert(s.id);
        }
        self.max_live = self.max_live.max(self.live.len());
        Ok(batch.iter().map(|s| (s.tokens.last().unwrap() + 1) % self.vocab).collect())
    }

    fn release(&mut self, id: u64) {
        self.live.remove(&id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(rx: &Receiver<Event>) -> (Vec<TokenId>, Event) {
        let mut toks = Vec::new();
        loop {
            match rx.recv().expect("engine sends Done") {
                Event::Token(t) => toks.push(t),
                end => return (toks, end),
            }
        }
    }

    #[test]
    fn many_concurrent_requests_get_exact_sequences() {
        let engine = Engine::start(CountingModel::new(1_000_000), EngineConfig { max_batch: 8, queue_depth: 256, pin_core: None }).unwrap();
        let rxs: Vec<_> = (0..64u64)
            .map(|i| {
                let req = Request { id: i, prompt: vec![1, (i * 100) as u32], max_tokens: 5 + (i as usize % 7), stop_tokens: vec![] };
                (i, engine.submit(req).unwrap())
            })
            .collect();
        for (i, rx) in &rxs {
            let (toks, end) = collect(rx);
            let n = 5 + (*i as usize % 7);
            let want: Vec<u32> = (1..=n as u32).map(|k| (*i as u32) * 100 + k).collect();
            assert_eq!(toks, want, "request {i}");
            assert_eq!(end, Event::Done(FinishReason::Length));
        }
        let stats = engine.shutdown();
        assert_eq!(stats.finished, 64);
        assert!(stats.max_batch_seen <= 8);
    }

    #[test]
    fn stop_token_ends_generation() {
        let engine = Engine::start(CountingModel::new(1000), EngineConfig::default()).unwrap();
        let rx = engine.submit(Request { id: 1, prompt: vec![10], max_tokens: 100, stop_tokens: vec![13] }).unwrap();
        let (toks, end) = collect(&rx);
        assert_eq!(toks, vec![11, 12, 13]);
        assert_eq!(end, Event::Done(FinishReason::Stop));
    }

    #[test]
    fn dropped_receiver_cancels_and_frees_the_sequence() {
        let engine = Engine::start(CountingModel::new(1000), EngineConfig::default()).unwrap();
        let rx = engine.submit(Request { id: 7, prompt: vec![1], max_tokens: 1_000_000, stop_tokens: vec![] }).unwrap();
        assert!(matches!(rx.recv().unwrap(), Event::Token(2)));
        drop(rx);
        let stats = engine.shutdown();
        assert_eq!(stats.cancelled, 1);
        assert!(stats.tokens_out < 1_000_000);
    }

    #[test]
    fn full_queue_is_reported_not_grown() {
        // A model that blocks until told to proceed keeps the queue from draining.
        struct Gate(std::sync::mpsc::Receiver<()>);
        impl StepModel for Gate {
            fn step(&mut self, batch: &[SeqView<'_>]) -> Result<Vec<TokenId>> {
                let _ = self.0.recv();
                Ok(vec![0; batch.len()])
            }
        }
        let (go, gate) = std::sync::mpsc::channel();
        let engine = Engine::start(Gate(gate), EngineConfig { max_batch: 1, queue_depth: 2, pin_core: None }).unwrap();
        let mut accepted = Vec::new();
        let mut rejected = 0;
        for i in 0..50 {
            match engine.submit(Request { id: i, prompt: vec![1], max_tokens: 1, stop_tokens: vec![] }) {
                Ok(rx) => accepted.push(rx),
                Err(_) => rejected += 1,
            }
        }
        assert!(rejected > 0, "bounded queue must reject when full");
        for _ in 0..accepted.len() + 2 {
            let _ = go.send(());
        }
        drop(go);
        let stats = engine.shutdown();
        assert_eq!(stats.finished as usize, accepted.len());
    }

    #[test]
    fn engine_thread_can_be_pinned() {
        let engine = Engine::start(CountingModel::new(10), EngineConfig { pin_core: Some(0), ..Default::default() }).unwrap();
        let rx = engine.submit(Request { id: 1, prompt: vec![1], max_tokens: 1, stop_tokens: vec![] }).unwrap();
        let _ = collect(&rx);
        let stats = engine.shutdown();
        // Pinning can be refused in restricted environments; it must never fail the engine.
        let _ = stats.pinned;
    }
}
