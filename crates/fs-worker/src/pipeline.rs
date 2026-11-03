//! The render loop.
//!
//! ```text
//! decoder ─▶ [align] ─▶ process (tiles, OOM backoff) ─▶ interpolate (cut-aware) ─▶ writer ─▶ commit
//!     ▲ backpressure (bounded channels, soft pause blocks the reader)
//! ```
//!
//! Frames lag by one, because interpolating between `i` and `i+1` needs `i+1`.
//! The arrival of the first frame of segment k+1 closes segment k, and that
//! frame's hash becomes segment k's `next_hash` (resume alignment).

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, bounded};
use fs_engine::interp::{FALLBACK_CUT_THRESHOLD, Scheduler, Synth};
use fs_engine::tiling::TileController;
use fs_engine::{EngineError, Frame, InferenceBackend, frame};
use fs_types::pipeline::StageConfig;
use fs_types::worker::{RenderPlan, WorkerEvent, WorkerJob, exit};
use fs_types::{FailureKind, WarningCode};

use crate::control::{Control, StopMode};
use crate::emit::Emitter;
use crate::ffmpeg::{DecodeMsg, Decoder};
use crate::writer::{self, PendingSegment, WriterMsg, WriterResult};

/// Resume alignment: decode this many frames early and search ± this window.
const ALIGN_BACK: u64 = 3;

struct Seg {
    index: u32,
    first_frame: u64,
    in_hash: u64,
    frames: u64,
    out: u64,
}

struct Prev {
    n: u64,
    raw: Frame,
    processed: Frame,
}

pub struct Fatal {
    pub kind: FailureKind,
    pub message: String,
}

impl Fatal {
    fn new(kind: FailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
    pub fn exit_code(&self) -> i32 {
        match self.kind {
            FailureKind::DiskFull => exit::DISK_FULL,
            FailureKind::DeviceLost => exit::DEVICE_LOST,
            FailureKind::LicenseRequired => exit::LICENSE,
            _ => exit::FATAL,
        }
    }
}

impl From<EngineError> for Fatal {
    fn from(e: EngineError) -> Self {
        Self::new(e.failure_kind(), e.to_string())
    }
}

pub enum Outcome {
    Finished,
    Stopped { next_segment: u32 },
}

struct Run<'a> {
    plan: Arc<RenderPlan>,
    ctrl: Arc<Control>,
    emit: &'a Emitter,
    backend: Box<dyn InferenceBackend>,
    tiles: TileController,
    sched: Scheduler,
    writer_tx: crossbeam_channel::Sender<WriterMsg>,
    results: Receiver<WriterResult>,
    frames_in: u64,
    frames_out: u64,
    last_progress: Instant,
    fps_ewma: f32,
    last_frame_at: Instant,
}

impl Run<'_> {
    fn seg_of(&self, n: u64) -> u32 {
        (n / self.plan.segment_frames.max(1)) as u32
    }

    /// Run one engine call under the tile controller and the Eco governor.
    fn engine<T>(
        &mut self,
        mut f: impl FnMut(&mut dyn InferenceBackend, u32) -> Result<T, EngineError>,
    ) -> Result<T, Fatal> {
        let t0 = Instant::now();
        let backend = &mut self.backend;
        let (v, outcome) = self.tiles.run(|tile| f(backend.as_mut(), tile))?;
        if let Some(t) = outcome.reduced_to {
            self.emit.send(&WorkerEvent::Warning {
                code: WarningCode::TileReduced,
                message: format!("VRAM pressure: tile reduced to {t}px"),
            });
        }
        let pause = fs_hw::governor::sleep_after(t0.elapsed(), self.ctrl.duty());
        if !pause.is_zero() {
            std::thread::sleep(pause);
        }
        Ok(v)
    }

    fn send_out(&mut self, seg: &mut Seg, frame: Frame) -> Result<(), Fatal> {
        seg.out += 1;
        self.frames_out += 1;
        self.writer_tx
            .send(WriterMsg::Frame {
                segment: seg.index,
                frame,
            })
            .map_err(|_| Fatal::new(FailureKind::Internal, "writer thread exited"))
    }

    /// Emit `prev` and the intermediates between `prev` and `next`.
    fn emit_prev(
        &mut self,
        seg: &mut Seg,
        prev: &Prev,
        next: Option<(&Frame, &Frame)>,
    ) -> Result<(), Fatal> {
        self.send_out(seg, prev.processed.clone())?;
        let synths = match next {
            Some((next_raw, _)) if self.sched.factor() > 1 => {
                let pixel_cut = frame::difference(&prev.raw, next_raw) > FALLBACK_CUT_THRESHOLD;
                self.sched.between(prev.n, pixel_cut)
            }
            Some(_) => Vec::new(),
            None => self.sched.tail(),
        };
        for s in synths {
            let f = match (s, next) {
                (Synth::Blend(t), Some((_, next_processed))) => {
                    let a = &prev.processed;
                    self.engine(|b, tile| b.interpolate(a, next_processed, t, tile))?
                }
                _ => prev.processed.clone(),
            };
            self.send_out(seg, f)?;
        }
        Ok(())
    }

    fn close(&self, seg: &Seg, next_hash: Option<u64>) -> Result<(), Fatal> {
        self.writer_tx
            .send(WriterMsg::Close(PendingSegment {
                index: seg.index,
                first_frame: seg.first_frame,
                frame_count: seg.frames,
                out_frames: seg.out,
                in_hash: seg.in_hash,
                next_hash,
            }))
            .map_err(|_| Fatal::new(FailureKind::Internal, "writer thread exited"))
    }

    fn drain_results(&self, block_until_closed: bool) -> Result<(), Fatal> {
        let handle = |r: WriterResult| match r {
            WriterResult::Committed(info) => {
                self.emit.send(&WorkerEvent::SegmentCommitted(info));
                Ok(())
            }
            WriterResult::PreviewDone => Ok(()),
            WriterResult::Failed { kind, message } => Err(Fatal::new(kind, message)),
        };
        if block_until_closed {
            for r in self.results.iter() {
                handle(r)?;
            }
        } else {
            while let Ok(r) = self.results.try_recv() {
                handle(r)?;
            }
        }
        Ok(())
    }

    fn progress(&mut self, segment: u32) {
        let now = Instant::now();
        let dt = now
            .duration_since(self.last_frame_at)
            .as_secs_f32()
            .max(1e-6);
        self.last_frame_at = now;
        let inst = 1.0 / dt;
        self.fps_ewma = if self.fps_ewma == 0.0 {
            inst
        } else {
            self.fps_ewma * 0.9 + inst * 0.1
        };
        if now.duration_since(self.last_progress) >= Duration::from_millis(250) {
            self.last_progress = now;
            self.emit.send(&WorkerEvent::Progress {
                frames_in: self.frames_in,
                frames_out: self.frames_out,
                fps: self.fps_ewma,
                segment,
            });
        }
    }
}

/// Pick the inference backend.
///
/// `FS_ENGINE=reference` forces the deterministic CPU reference backend (tests).
/// Otherwise ncnn is used: Vulkan on the planned GPU, or ncnn's CPU path when
/// no real GPU exists. A failure to load models is fatal, never a silent
/// downgrade to the non-AI reference.
fn select_backend(plan: &RenderPlan, emit: &Emitter) -> Result<Box<dyn InferenceBackend>, Fatal> {
    if std::env::var("FS_ENGINE").as_deref() == Ok("reference") {
        return Ok(Box::new(fs_engine::cpu::ReferenceBackend::new(
            &plan.engine.stages,
        )));
    }
    #[cfg(feature = "ncnn")]
    {
        let _ = emit;
        crate::device::ncnn_backend(&plan.engine).map_err(Fatal::from)
    }
    #[cfg(not(feature = "ncnn"))]
    {
        emit.send(&WorkerEvent::Warning {
            code: WarningCode::EncoderFallback,
            message: "this build has no inference runtime; using the non-AI reference backend"
                .into(),
        });
        Ok(Box::new(fs_engine::cpu::ReferenceBackend::new(
            &plan.engine.stages,
        )))
    }
}

pub fn run(job: WorkerJob, ctrl: Arc<Control>, emit: &Emitter) -> Result<Outcome, Fatal> {
    let plan = Arc::new(job.plan);
    let backend = select_backend(&plan, emit)?;
    let max_tile = backend.max_tile();
    emit.send(&WorkerEvent::Ready {
        backend: backend.name().into(),
        device_name: backend.device_name(),
        tile: plan.engine.tile.or(Some(max_tile)),
    });

    let factor = plan
        .engine
        .stages
        .iter()
        .find_map(|s| match s {
            StageConfig::Interpolate { factor, .. } => Some(u32::from(*factor)),
            _ => None,
        })
        .unwrap_or(1);

    std::fs::create_dir_all(&plan.segment_dir)
        .map_err(|e| Fatal::new(FailureKind::Io, format!("segment dir: {e}")))?;

    // --- decoder, with resume alignment ------------------------------------------
    let first = job.start.first_frame;
    let decode_from = match job.start.expected_hash {
        Some(_) => first.saturating_sub(ALIGN_BACK),
        None => first,
    };
    let argv = fs_media::argv::decode(&plan.decode, decode_from);
    let mut decoder = Decoder::spawn(
        &plan.ffmpeg,
        &argv,
        plan.decode.width,
        plan.decode.height,
        plan.decode.frame_format,
        ctrl.clone(),
    )
    .map_err(|e| Fatal::new(FailureKind::Io, format!("could not start decoder: {e}")))?;

    let mut pending: VecDeque<Frame> = VecDeque::new();
    if let Some(expected) = job.start.expected_hash {
        let window = (first - decode_from) + ALIGN_BACK + 1;
        while (pending.len() as u64) < window {
            match decoder.recv() {
                DecodeMsg::Frame(f) => pending.push_back(f),
                _ => break,
            }
        }
        match pending.iter().position(|f| frame::hash(f) == expected) {
            Some(j) => {
                pending.drain(..j);
            }
            None => {
                // Trust the seek (validated frame-exact for CFR timelines).
                pending.drain(..((first - decode_from) as usize).min(pending.len()));
                emit.send(&WorkerEvent::Warning {
                    code: WarningCode::CorruptPackets,
                    message: "resume alignment hash not found; continuing from the seek position"
                        .into(),
                });
            }
        }
    }

    // --- writer -----------------------------------------------------------------
    let (writer_tx, writer_rx) = bounded::<WriterMsg>(4);
    let (res_tx, res_rx) = crossbeam_channel::unbounded();
    let preview = job.preview.clone();
    let mut before = None;
    let writer = match &preview {
        Some(pv) => {
            let dir = std::path::Path::new(&pv.out_dir);
            for sub in ["before", "after"] {
                std::fs::create_dir_all(dir.join(sub))
                    .map_err(|e| Fatal::new(FailureKind::Io, format!("preview dir: {e}")))?;
            }
            // "Before": the decoded source frames, tone-mapped the same way.
            let argv = fs_media::argv::jpeg_sequence(
                plan.decode.width,
                plan.decode.height,
                plan.decode.frame_format,
                &plan.decode.color,
                &dir.join("before").join("%04d.jpg"),
            );
            before = Some(
                crate::ffmpeg::SegmentEncoder::spawn(&plan.ffmpeg, &argv)
                    .map_err(|e| Fatal::new(FailureKind::Io, format!("preview encoder: {e}")))?,
            );
            writer::spawn_preview(
                plan.clone(),
                dir.join("after").join("%04d.jpg"),
                writer_rx,
                res_tx,
            )
        }
        None => writer::spawn(plan.clone(), writer_rx, res_tx),
    };
    let limit = preview.as_ref().map(|p| p.frames);
    let mut limit_hit = false;

    let now = Instant::now();
    let mut r = Run {
        plan: plan.clone(),
        ctrl: ctrl.clone(),
        emit,
        backend,
        tiles: TileController::new(plan.engine.tile.unwrap_or(max_tile)),
        sched: Scheduler::new(factor, plan.engine.scene_cuts.iter().copied()),
        writer_tx,
        results: res_rx,
        frames_in: 0,
        frames_out: 0,
        last_progress: now,
        fps_ewma: 0.0,
        last_frame_at: now,
    };

    let mut n = first;
    let mut prev: Option<Prev> = None;
    let mut seg: Option<Seg> = None;

    let result: Result<Outcome, Fatal> = (|| {
        loop {
            if ctrl.stop_mode() == StopMode::Now {
                let next_segment = seg.as_ref().map_or(job.start.segment, |s| s.index);
                return Ok(Outcome::Stopped { next_segment });
            }
            if limit.is_some_and(|l| r.frames_in >= l) {
                limit_hit = true;
                break; // preview window done: treat as end of stream
            }
            let raw = match pending.pop_front() {
                Some(f) => f,
                None => match decoder.recv() {
                    DecodeMsg::Frame(f) => f,
                    DecodeMsg::Eof => break,
                    DecodeMsg::Error(e) => return Err(Fatal::new(FailureKind::DecodeFailed, e)),
                },
            };
            if let Some(b) = before.as_mut() {
                b.write(&raw.data).map_err(|e| {
                    Fatal::new(FailureKind::Internal, format!("preview encoder: {e}"))
                })?;
            }
            let processed = r.engine(|b, tile| b.process(&raw, tile))?;
            let cur_seg = r.seg_of(n);

            if let Some(p) = prev.take() {
                let s = seg
                    .as_mut()
                    .ok_or_else(|| Fatal::new(FailureKind::Internal, "no segment"))?;
                r.emit_prev(s, &p, Some((&raw, &processed)))?;
                if cur_seg != s.index {
                    r.close(s, Some(frame::hash(&raw)))?;
                    if ctrl.stop_mode() == StopMode::Checkpoint {
                        return Ok(Outcome::Stopped {
                            next_segment: cur_seg,
                        });
                    }
                    seg = None;
                }
            }
            if seg.is_none() {
                seg = Some(Seg {
                    index: cur_seg,
                    first_frame: n,
                    in_hash: frame::hash(&raw),
                    frames: 0,
                    out: 0,
                });
            }
            if let Some(s) = seg.as_mut() {
                s.frames += 1;
            }
            r.frames_in += 1;
            prev = Some(Prev { n, raw, processed });
            n += 1;
            r.progress(cur_seg);
            r.drain_results(false)?;
        }
        // EOF: flush the lagging frame (+ tail holds) and close the last segment.
        if let (Some(p), Some(s)) = (prev.take(), seg.as_mut()) {
            r.emit_prev(s, &p, None)?;
            r.close(s, None)?;
        }
        Ok(Outcome::Finished)
    })();

    // --- shutdown ---------------------------------------------------------------
    let stopped_now = ctrl.stop_mode() == StopMode::Now;
    let finished_cleanly = matches!(result, Ok(Outcome::Finished));
    if result.is_err() || stopped_now {
        let _ = r.writer_tx.send(WriterMsg::Abort); // discard the partial segment
    }
    if !finished_cleanly || limit_hit {
        decoder.kill(); // a preview stops mid-stream on purpose
    }
    if let Some(b) = before.take() {
        let (status, log) = b.finish();
        if result.is_ok() && !status.is_some_and(|s| s.success()) {
            let why = log.lock().map(|l| l.summary()).unwrap_or_default();
            return Err(Fatal::new(
                FailureKind::Internal,
                format!("preview encoder failed: {why}"),
            ));
        }
    }
    let Run {
        emit: emitter,
        results,
        writer_tx,
        frames_in,
        frames_out,
        ..
    } = r;
    // Closing the channel lets the writer finish any queued Close (commit) and exit.
    drop(writer_tx);
    let _ = writer.join();
    let mut final_result = result;
    for res in results.try_iter() {
        match res {
            WriterResult::Committed(info) => emitter.send(&WorkerEvent::SegmentCommitted(info)),
            WriterResult::PreviewDone => {}
            WriterResult::Failed { kind, message } => {
                if final_result.is_ok() {
                    final_result = Err(Fatal::new(kind, message));
                }
            }
        }
    }

    if finished_cleanly && !limit_hit {
        let (status, log) = decoder.finish();
        if let Ok(l) = log.lock() {
            if l.corrupt > 0 {
                emitter.send(&WorkerEvent::Warning {
                    code: WarningCode::CorruptPackets,
                    message: format!(
                        "{} damaged packets were concealed by the decoder",
                        l.corrupt
                    ),
                });
            }
            if !status.is_some_and(|s| s.success()) && final_result.is_ok() {
                let kind = l
                    .worst()
                    .and_then(|i| i.failure_kind())
                    .unwrap_or(FailureKind::DecodeFailed);
                final_result = Err(Fatal::new(kind, format!("decoder failed: {}", l.summary())));
            }
        }
    }
    if matches!(final_result, Ok(Outcome::Finished)) {
        emitter.send(&WorkerEvent::Finished {
            frames_in,
            frames_out,
        });
    }
    final_result
}
