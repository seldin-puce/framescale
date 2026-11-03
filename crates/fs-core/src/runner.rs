//! The job runner: analyze → plan → supervised render (with recovery) →
//! concat → mux → verify. One async task per job.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fs_journal::Journal;
use fs_media::encoders::{self, EncodeTarget, EncoderEnv};
use fs_media::{analyze, mux, plan, probe, rate};
use fs_proc::{Ledger, TreeHandle};
use fs_types::pipeline::{EncoderPreference, StageConfig};
use fs_types::worker::{
    HwDecode, PROTOCOL_VERSION, RenderPlan, StartPoint, WorkerCommand, WorkerEvent, WorkerJob,
};
use fs_types::{
    FailureKind, GpuDevice, JobEvent, JobId, JobProgress, JobState, MediaInfo, PauseMode,
    WarningCode,
};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::janitor::lock_dir;
use crate::recovery::{Action, RecoveryPolicy, WorkerExit};
use crate::storage;

/// Everything a job needs from the app.
#[derive(Debug)]
pub struct Services {
    pub journal: Arc<Mutex<Journal>>,
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
    pub worker: PathBuf,
    /// Temp root; each job gets `<work_root>/<job-id>/`.
    pub work_root: PathBuf,
    pub models_dir: PathBuf,
    pub encoders: EncoderEnv,
    pub device: Option<GpuDevice>,
    pub ledger: Arc<Ledger>,
    pub duty: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JobCommand {
    Pause(PauseMode),
    Resume,
    Cancel,
    SetDuty(f32),
}

pub type EventSink = UnboundedSender<JobEvent>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Intent {
    None,
    HardPause,
    DiskPause,
    Cancel,
}

struct Ctx<'a> {
    id: JobId,
    svc: &'a Services,
    events: &'a EventSink,
    cmds: &'a mut UnboundedReceiver<JobCommand>,
}

/// A step ended early because the user cancelled.
struct Cancelled;

impl Ctx<'_> {
    fn journal<T>(
        &self,
        f: impl FnOnce(&mut Journal) -> fs_journal::Result<T>,
    ) -> Result<T, String> {
        let mut j = self
            .svc
            .journal
            .lock()
            .map_err(|_| "journal lock poisoned".to_owned())?;
        f(&mut j).map_err(|e| e.to_string())
    }

    fn state(&self, state: JobState, reason: Option<String>) {
        let _ = self.journal(|j| j.set_state(self.id, state, reason.as_deref()));
        let _ = self.events.send(JobEvent::StateChanged {
            job: self.id,
            state,
            reason,
        });
    }

    fn warn(&self, code: WarningCode, message: String) {
        tracing::warn!(job = %self.id, ?code, %message);
        let _ = self.journal(|j| j.log_event(self.id, "warn", &format!("{code:?}"), &message));
        let _ = self.events.send(JobEvent::Warning {
            job: self.id,
            code,
            message,
        });
    }

    fn error(&self, kind: FailureKind, message: String) {
        let _ = self.journal(|j| j.log_event(self.id, "error", &format!("{kind:?}"), &message));
        let _ = self.events.send(JobEvent::Error {
            job: self.id,
            kind,
            message,
        });
    }

    /// Block until Resume (true) or Cancel (false). Other commands are ignored.
    async fn wait_for_resume(&mut self) -> bool {
        while let Some(c) = self.cmds.recv().await {
            match c {
                JobCommand::Resume => return true,
                JobCommand::Cancel => return false,
                _ => {}
            }
        }
        false
    }

    /// Sleep that a Cancel interrupts.
    async fn sleep(&mut self, d: Duration) -> Result<(), Cancelled> {
        let t = tokio::time::sleep(d);
        tokio::pin!(t);
        loop {
            tokio::select! {
                () = &mut t => return Ok(()),
                c = self.cmds.recv() => if matches!(c, Some(JobCommand::Cancel) | None) { return Err(Cancelled) },
            }
        }
    }

    /// Run an FFmpeg step to completion; Cancel kills it.
    async fn ffmpeg(
        &mut self,
        argv: &fs_media::Argv,
    ) -> Result<Result<(String, String), String>, Cancelled> {
        let mut cmd = std::process::Command::new(&self.svc.ffmpeg);
        cmd.args(argv.as_slice())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        fs_proc::prepare(&mut cmd);
        let child = match tokio::process::Command::from(cmd)
            .kill_on_drop(true)
            .spawn()
        {
            Ok(c) => c,
            Err(e) => return Ok(Err(format!("could not start ffmpeg: {e}"))),
        };
        let out = child.wait_with_output();
        tokio::pin!(out);
        loop {
            tokio::select! {
                r = &mut out => {
                    return Ok(match r {
                        Ok(o) if o.status.success() => Ok((
                            String::from_utf8_lossy(&o.stdout).into_owned(),
                            String::from_utf8_lossy(&o.stderr).into_owned(),
                        )),
                        Ok(o) => Err(String::from_utf8_lossy(&o.stderr).lines().rev().take(5).collect::<Vec<_>>().join(" | ")),
                        Err(e) => Err(e.to_string()),
                    });
                }
                c = self.cmds.recv() => if matches!(c, Some(JobCommand::Cancel) | None) { return Err(Cancelled) },
            }
        }
    }
}

fn scene_threshold(spec: &fs_types::JobSpec) -> Option<f32> {
    spec.pipeline.stages.iter().find_map(|s| match s {
        StageConfig::Interpolate {
            scene_threshold, ..
        } => Some(*scene_threshold),
        _ => None,
    })
}

async fn probe_async(ffprobe: &Path, input: &Path) -> Result<MediaInfo, String> {
    let (fp, inp) = (ffprobe.to_path_buf(), input.to_path_buf());
    tokio::task::spawn_blocking(move || probe::probe(&fp, &inp))
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())
}

/// Run a job to a terminal or paused state. Returns the final state.
pub async fn run_job(
    id: JobId,
    svc: Arc<Services>,
    events: EventSink,
    mut cmds: UnboundedReceiver<JobCommand>,
) -> JobState {
    let mut ctx = Ctx {
        id,
        svc: &svc,
        events: &events,
        cmds: &mut cmds,
    };
    let state = match run_inner(&mut ctx).await {
        Ok(s) => s,
        Err(msg) => {
            ctx.error(FailureKind::Internal, msg.clone());
            ctx.state(JobState::Failed, Some(msg));
            JobState::Failed
        }
    };
    if state == JobState::Cancelled {
        let _ = std::fs::remove_dir_all(svc.work_root.join(id.to_string()));
    }
    state
}

async fn run_inner(ctx: &mut Ctx<'_>) -> Result<JobState, String> {
    let id = ctx.id;
    let spec = ctx.journal(|j| j.spec(id))?;
    let job_dir = ctx.svc.work_root.join(id.to_string());
    let lock = lock_dir(&job_dir).map_err(|e| e.to_string())?;
    let seg_dir = job_dir.join("segments");
    std::fs::create_dir_all(&seg_dir).map_err(|e| e.to_string())?;

    // --- 1. probe + analyze + plan (a stored plan means we're resuming) -----------
    let mut plan = match ctx.journal(|j| j.plan(id))? {
        Some(p) => p,
        None => {
            ctx.state(JobState::Analyzing, None);
            let info = probe_async(&ctx.svc.ffprobe, Path::new(&spec.input)).await?;
            let conform = rate::conform_rate(
                Some(info.video.r_frame_rate),
                Some(info.video.avg_frame_rate),
            );
            let hw = plan::hw_decode_for(ctx.svc.device.as_ref(), ctx.svc.encoders.os);
            let argv = analyze::argv(
                Path::new(&spec.input),
                info.video.index,
                &hw,
                conform,
                scene_threshold(&spec),
            );
            let (stdout, stderr) = match ctx.ffmpeg(&argv).await {
                Err(Cancelled) => return Ok(JobState::Cancelled),
                Ok(Ok(o)) => o,
                // Hardware decode can fail on exotic sources; analysis retries in software.
                Ok(Err(_)) if hw != HwDecode::None => {
                    let argv = analyze::argv(
                        Path::new(&spec.input),
                        info.video.index,
                        &HwDecode::None,
                        conform,
                        scene_threshold(&spec),
                    );
                    match ctx.ffmpeg(&argv).await {
                        Err(Cancelled) => return Ok(JobState::Cancelled),
                        Ok(r) => r.map_err(|e| format!("analysis failed: {e}"))?,
                    }
                }
                Ok(Err(e)) => return Err(format!("analysis failed: {e}")),
            };
            let mut p = analyze::AnalyzeParser::new();
            stderr.lines().for_each(|l| p.feed_stderr(l));
            stdout.lines().for_each(|l| p.feed_progress(l));
            let analysis = p.finish(conform);

            let segment_dir = seg_dir.to_string_lossy().into_owned();
            let planned = plan::compile(&plan::PlanInputs {
                info: &info,
                analysis: &analysis,
                spec: &spec,
                device: ctx.svc.device.as_ref(),
                encoders: &ctx.svc.encoders,
                ffmpeg: &ctx.svc.ffmpeg.to_string_lossy(),
                segment_dir: &segment_dir,
                models_dir: &ctx.svc.models_dir.to_string_lossy(),
            })
            .map_err(|e| e.to_string())?;
            for (code, msg) in planned.warnings {
                ctx.warn(code, msg);
            }
            ctx.journal(|j| j.set_plan(id, &planned.plan))?;
            planned.plan
        }
    };

    // --- 2. storage preflight -----------------------------------------------------
    let out_path = PathBuf::from(&spec.output.path);
    let estimate = plan::estimate_output_bytes(&plan);
    loop {
        let free = storage::free_space(&out_path).unwrap_or(u64::MAX);
        match storage::preflight(estimate, free) {
            Ok(()) => break,
            Err(s) => {
                let msg = format!(
                    "not enough free space: about {:.1} GB needed, {:.1} GB available",
                    s.required as f64 / 1e9,
                    s.available as f64 / 1e9
                );
                ctx.error(FailureKind::DiskFull, msg.clone());
                ctx.state(JobState::Paused, Some(msg));
                if !ctx.wait_for_resume().await {
                    return Ok(JobState::Cancelled);
                }
            }
        }
    }

    // --- 3. supervised render -------------------------------------------------------
    let mut policy = RecoveryPolicy::new();
    loop {
        let segs = ctx.journal(|j| j.segments(id))?;
        let contiguous = segs
            .iter()
            .enumerate()
            .take_while(|(i, s)| s.index as usize == *i)
            .count();
        if contiguous > 0 && segs[contiguous - 1].next_hash.is_none() {
            break; // the last segment (ended at EOF) is committed
        }
        let rp = ctx.journal(|j| j.resume_point(id, plan.segment_frames))?;
        ctx.state(JobState::Rendering, None);
        let start = StartPoint {
            segment: rp.segment,
            first_frame: rp.first_frame,
            expected_hash: rp.expected_hash,
        };
        let (exit, intent) = run_worker_once(ctx, &plan, start).await?;

        let action = match (policy.decide(&exit), intent) {
            (Action::Stopped, Intent::DiskPause) => Action::PauseForDisk,
            (a, _) => a,
        };
        match action {
            Action::Done => {}
            Action::Stopped => match intent {
                Intent::Cancel => return Ok(JobState::Cancelled),
                _ => {
                    ctx.state(JobState::Paused, Some("paused".into()));
                    if !ctx.wait_for_resume().await {
                        return Ok(JobState::Cancelled);
                    }
                }
            },
            Action::PauseForDisk => {
                let msg = "disk is full: free some space, then resume".to_owned();
                ctx.error(FailureKind::DiskFull, msg.clone());
                ctx.state(JobState::Paused, Some(msg));
                if !ctx.wait_for_resume().await {
                    return Ok(JobState::Cancelled);
                }
            }
            Action::Restart(r) => {
                if r.disable_hw_decode {
                    plan.decode.hw = HwDecode::None;
                    ctx.warn(
                        WarningCode::HwDecodeFallback,
                        "hardware decoding failed; retrying in software".into(),
                    );
                }
                if let Some(scale) = r.tile_scale {
                    let t = ((plan.engine.tile.unwrap_or(512) as f32 * scale) as u32).max(64);
                    plan.engine.tile = Some(t);
                    ctx.warn(
                        WarningCode::TileReduced,
                        format!("GPU fault; retrying with {t}px tiles"),
                    );
                }
                if r.software_encoder {
                    switch_to_software_encoder(ctx, &spec, &mut plan, &seg_dir)?;
                }
                ctx.journal(|j| j.set_plan(id, &plan))?;
                if ctx.sleep(r.delay).await.is_err() {
                    return Ok(JobState::Cancelled);
                }
            }
            Action::Fail(m) => {
                let kind = exit.error.map_or(FailureKind::WorkerCrashed, |(k, _)| k);
                ctx.error(kind, m.clone());
                ctx.state(JobState::Failed, Some(m));
                return Ok(JobState::Failed);
            }
        }
    }

    // --- 4. concat ------------------------------------------------------------------
    ctx.state(JobState::Concatenating, None);
    let segs = ctx.journal(|j| j.segments(id))?;
    let list = seg_dir.join("concat.txt");
    std::fs::write(
        &list,
        fs_media::argv::concat_list(segs.iter().map(|s| s.file_name.as_str())),
    )
    .map_err(|e| e.to_string())?;
    let video = job_dir.join("video.mkv");
    match ctx.ffmpeg(&fs_media::argv::concat(&list, &video)).await {
        Err(Cancelled) => return Ok(JobState::Cancelled),
        Ok(r) => r.map_err(|e| format!("concat failed: {e}"))?,
    };

    // --- 5. mux + verify ----------------------------------------------------------------
    ctx.state(JobState::Muxing, None);
    let info = probe_async(&ctx.svc.ffprobe, Path::new(&spec.input)).await?;
    let part = out_path.with_file_name(format!(
        ".{}.part",
        out_path
            .file_name()
            .map_or_else(|| "output".into(), |n| n.to_string_lossy().into_owned())
    ));
    let m = mux::plan(
        &video,
        &info,
        spec.output.container,
        spec.output.codec,
        &part,
    );
    for (code, msg) in m.warnings.clone() {
        ctx.warn(code, msg);
    }
    match ctx.ffmpeg(&m.argv).await {
        Err(Cancelled) => {
            let _ = std::fs::remove_file(&part);
            return Ok(JobState::Cancelled);
        }
        Ok(r) => r.map_err(|e| format!("mux failed: {e}"))?,
    };
    let out_info = probe_async(&ctx.svc.ffprobe, &part).await?;
    let tolerance = (info.duration_s * 0.01).max(0.5);
    if (out_info.duration_s - info.duration_s).abs() > tolerance {
        let _ = std::fs::remove_file(&part);
        return Err(format!(
            "output duration {:.2}s doesn't match source {:.2}s",
            out_info.duration_s, info.duration_s
        ));
    }
    std::fs::rename(&part, &out_path)
        .map_err(|e| format!("could not move output into place: {e}"))?;

    ctx.state(JobState::Done, None);
    drop(lock);
    let _ = std::fs::remove_dir_all(&job_dir);
    Ok(JobState::Done)
}

/// Encoder fallback: segments from different encoders can't be concatenated
/// losslessly, so committed segments are discarded and the render restarts.
fn switch_to_software_encoder(
    ctx: &Ctx<'_>,
    spec: &fs_types::JobSpec,
    plan: &mut RenderPlan,
    seg_dir: &Path,
) -> Result<(), String> {
    let target = EncodeTarget {
        quality: spec.output.quality,
        width: plan.encode.width,
        height: plan.encode.height,
        rate: plan.encode.rate,
    };
    let sel = encoders::select(
        spec.output.codec,
        EncoderPreference::Software,
        &plan.decode.color,
        target,
        &ctx.svc.encoders,
    )
    .map_err(|e| e.to_string())?;
    ctx.warn(
        WarningCode::EncoderFallback,
        format!(
            "{} failed; switching to {}",
            plan.encode.encoder, sel.encoder
        ),
    );
    plan.encode.encoder = sel.encoder;
    plan.encode.args = sel.args;
    plan.encode.pix_fmt = sel.pix_fmt;
    plan.encode.vaapi_device = sel.vaapi_device;
    plan.encode.hdr = sel.hdr;
    plan.encode.color.bit_depth = sel.out_bit_depth;
    ctx.journal(|j| j.truncate_segments(ctx.id, 0))?;
    if let Ok(entries) = std::fs::read_dir(seg_dir) {
        for e in entries.flatten() {
            let _ = std::fs::remove_file(e.path());
        }
    }
    Ok(())
}

async fn send(stdin: &mut tokio::process::ChildStdin, cmd: &WorkerCommand) {
    if let Ok(line) = serde_json::to_string(cmd) {
        let _ = stdin.write_all(format!("{line}\n").as_bytes()).await;
        let _ = stdin.flush().await;
    }
}

async fn run_worker_once(
    ctx: &mut Ctx<'_>,
    plan: &RenderPlan,
    start: StartPoint,
) -> Result<(WorkerExit, Intent), String> {
    let id = ctx.id;
    let mut cmd = std::process::Command::new(&ctx.svc.worker);
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    fs_proc::prepare(&mut cmd);
    let mut child = tokio::process::Command::from(cmd)
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("could not start fs-worker: {e}"))?;
    let pid = child.id().ok_or("worker exited immediately")?;
    #[cfg(unix)]
    let tree = TreeHandle::new(pid).map_err(|e| e.to_string())?;
    #[cfg(windows)]
    let tree = TreeHandle::new(pid, child.raw_handle().ok_or("no process handle")?)
        .map_err(|e| e.to_string())?;
    if let Err(e) = ctx.svc.ledger.record(pid) {
        tracing::warn!(error = %e, "could not record worker pid");
    }

    let mut stdin = child.stdin.take().ok_or("no worker stdin")?;
    let mut lines = BufReader::new(child.stdout.take().ok_or("no worker stdout")?).lines();
    // Keep the last stderr lines: they explain crashes that never reached the
    // JSON event stream (panics, loader errors, signals).
    let stderr_tail = Arc::new(Mutex::new(std::collections::VecDeque::<String>::new()));
    let stderr_task = child.stderr.take().map(|err| {
        let tail = stderr_tail.clone();
        tokio::spawn(async move {
            let mut l = BufReader::new(err).lines();
            while let Ok(Some(line)) = l.next_line().await {
                tracing::debug!(target: "fs_worker", "{line}");
                if let Ok(mut t) = tail.lock() {
                    t.push_back(line);
                    if t.len() > 20 {
                        t.pop_front();
                    }
                }
            }
        })
    });

    let job = WorkerJob {
        protocol: PROTOCOL_VERSION,
        job: id,
        plan: plan.clone(),
        start: start.clone(),
        duty: ctx.svc.duty,
        preview: None,
    };
    send(&mut stdin, &WorkerCommand::Start(Box::new(job))).await;

    let segments_total = plan.segments_total();
    let mut intent = Intent::None;
    let mut last_error = None;
    let mut committed = 0u32;
    let mut seg_bytes_total = 0u64;
    let out_path = ctx
        .journal(|j| j.spec(id))
        .map(|s| PathBuf::from(s.output.path))
        .unwrap_or_default();

    loop {
        tokio::select! {
            line = lines.next_line() => {
                let Ok(Some(line)) = line else { break };
                let Ok(ev) = serde_json::from_str::<WorkerEvent>(&line) else { continue };
                match ev {
                    WorkerEvent::Progress { frames_in, fps, segment, .. } => {
                        let done = start.first_frame + frames_in;
                        let remaining = plan.total_frames.saturating_sub(done);
                        let _ = ctx.events.send(JobEvent::Progress(JobProgress {
                            job: id,
                            frames_done: done,
                            frames_total: plan.total_frames,
                            fps,
                            eta_s: (fps > 0.01).then(|| remaining as f64 / f64::from(fps)),
                            segment,
                            segments_total,
                        }));
                    }
                    WorkerEvent::SegmentCommitted(s) => {
                        seg_bytes_total += s.bytes;
                        let device = plan.engine.device;
                        ctx.journal(|j| j.commit_segment(id, &s, device))?;
                        committed += 1;
                        // Pause before the disk is actually full (never mid-write ENOSPC).
                        let avg = seg_bytes_total / u64::from(committed);
                        if intent == Intent::None
                            && storage::free_space(&out_path).is_some_and(|f| storage::should_pause(f, avg))
                        {
                            intent = Intent::DiskPause;
                            send(&mut stdin, &WorkerCommand::Stop { checkpoint: true }).await;
                        }
                    }
                    WorkerEvent::Warning { code, message } => ctx.warn(code, message),
                    WorkerEvent::Error { kind, message, .. } => last_error = Some((kind, message)),
                    WorkerEvent::Ready { backend, device_name, tile } => {
                        tracing::info!(job = %id, %backend, ?device_name, ?tile, "worker ready");
                    }
                    WorkerEvent::Hello { .. } | WorkerEvent::Finished { .. } | WorkerEvent::Stopped { .. } => {}
                }
            }
            c = ctx.cmds.recv() => match c {
                Some(JobCommand::Pause(PauseMode::Soft)) => {
                    send(&mut stdin, &WorkerCommand::Pause).await;
                    ctx.state(JobState::Paused, Some("paused (holding GPU memory)".into()));
                }
                Some(JobCommand::Resume) => {
                    send(&mut stdin, &WorkerCommand::Resume).await;
                    ctx.state(JobState::Rendering, None);
                }
                Some(JobCommand::Pause(PauseMode::Hard)) => {
                    intent = Intent::HardPause;
                    // A soft-paused worker must be unpaused to reach its checkpoint.
                    send(&mut stdin, &WorkerCommand::Resume).await;
                    send(&mut stdin, &WorkerCommand::Stop { checkpoint: true }).await;
                }
                Some(JobCommand::Cancel) | None => {
                    intent = Intent::Cancel;
                    send(&mut stdin, &WorkerCommand::Stop { checkpoint: false }).await;
                }
                Some(JobCommand::SetDuty(d)) => send(&mut stdin, &WorkerCommand::SetDuty { duty: d }).await,
            },
        }
    }

    // stdout closed: the worker is exiting. Give it a grace period, then kill the tree.
    let status = match tokio::time::timeout(Duration::from_secs(5), child.wait()).await {
        Ok(s) => s.ok(),
        Err(_) => {
            let _ = tree.kill();
            child.wait().await.ok()
        }
    };
    let _ = ctx.svc.ledger.remove(pid);
    if let Some(t) = stderr_task {
        let _ = tokio::time::timeout(Duration::from_secs(1), t).await;
    }
    let code = status.and_then(|s| s.code());
    if last_error.is_none()
        && !matches!(
            code,
            Some(fs_types::worker::exit::OK | fs_types::worker::exit::STOPPED)
        )
    {
        let tail = stderr_tail
            .lock()
            .map(|t| t.iter().cloned().collect::<Vec<_>>().join(" | "))
            .unwrap_or_default();
        let what = match code {
            Some(c) => format!("worker exited with code {c}"),
            None => "worker was killed by a signal".to_owned(),
        };
        last_error = Some((
            FailureKind::WorkerCrashed,
            if tail.is_empty() {
                what
            } else {
                format!("{what}: {tail}")
            },
        ));
    }
    Ok((
        WorkerExit {
            code: status.and_then(|s| s.code()),
            error: last_error,
            committed,
            start_segment: start.segment,
        },
        intent,
    ))
}
