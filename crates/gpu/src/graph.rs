// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Synthetic Minds LTD
//! CUDA graph capture and replay (R-RT-4). Decode steps are captured once per
//! batch-size bucket and replayed, removing per-kernel CPU launch overhead.

use crate::{check, drv, sys, Context, Stream};
use astral_core::Result;
use std::sync::Arc;

/// An instantiated, launchable graph.
pub struct GraphExec {
    raw: sys::CUgraphExec,
    ctx: Arc<Context>,
}

// SAFETY: executable graph handles may be launched from any thread with the context current.
unsafe impl Send for GraphExec {}

impl Stream {
    /// Captures all work `f` enqueues on this stream into a graph, then instantiates it.
    /// Nothing runs during capture. Buffers used inside must outlive the returned graph.
    pub fn capture<F: FnOnce(&Stream) -> Result<()>>(&self, f: F) -> Result<GraphExec> {
        self.ctx.bind()?;
        let d = drv()?;
        // SAFETY: live stream; thread-local mode so other threads are unaffected.
        check("cuStreamBeginCapture_v2", unsafe { (d.fns.cuStreamBeginCapture_v2)(self.raw, sys::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) })?;
        let body = f(self);
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        // SAFETY: always end the capture, even if `f` failed, so the stream is usable again.
        let end = check("cuStreamEndCapture", unsafe { (d.fns.cuStreamEndCapture)(self.raw, &mut graph) });
        let destroy = |g: sys::CUgraph| {
            if !g.is_null() {
                // SAFETY: graph created by capture, destroyed once.
                unsafe { (d.fns.cuGraphDestroy)(g) };
            }
        };
        if let Err(e) = body.and(end) {
            destroy(graph);
            return Err(e);
        }
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        // SAFETY: valid graph; out-pointer to a local.
        let inst = check("cuGraphInstantiateWithFlags", unsafe { (d.fns.cuGraphInstantiateWithFlags)(&mut exec, graph, 0) });
        destroy(graph);
        inst?;
        Ok(GraphExec { raw: exec, ctx: self.ctx.clone() })
    }
}

impl GraphExec {
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        self.ctx.bind()?;
        let d = drv()?;
        // SAFETY: live executable graph and stream in the same context.
        check("cuGraphLaunch", unsafe { (d.fns.cuGraphLaunch)(self.raw, stream.raw) })
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if let Ok(d) = sys::driver() {
            let _ = self.ctx.bind();
            // SAFETY: destroyed exactly once.
            unsafe { (d.fns.cuGraphExecDestroy)(self.raw) };
        }
    }
}
