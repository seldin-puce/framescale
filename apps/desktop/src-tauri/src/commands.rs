//! IPC commands. Every Pro gate is enforced here in Rust (and again in the
//! worker). The UI's lock icons are only hints.

use std::path::PathBuf;
use std::sync::Arc;

use fs_core::JobCommand;
use fs_types::{
    DeviceUuid, EcoMode, GpuDevice, JobEvent, JobId, JobSpec, JobState, JobSummary, MediaInfo,
    PauseMode, Telemetry,
};
use serde::Serialize;
use tauri::State;
use tauri::ipc::Channel;

use crate::state::AppState;

type S<'a> = State<'a, Arc<AppState>>;

/// Error shape the UI can switch on.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CmdError {
    pub code: &'static str,
    pub message: String,
}

type R<T> = Result<T, CmdError>;

fn err(code: &'static str, message: impl Into<String>) -> CmdError {
    CmdError {
        code,
        message: message.into(),
    }
}

fn journal<T>(
    s: &AppState,
    f: impl FnOnce(&mut fs_journal::Journal) -> fs_journal::Result<T>,
) -> R<T> {
    let mut j = s
        .journal
        .lock()
        .map_err(|_| err("internal", "journal lock poisoned"))?;
    f(&mut j).map_err(|e| err("journal", e.to_string()))
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareInfo {
    pub ready: bool,
    pub devices: Vec<GpuDevice>,
    pub selected: Option<DeviceUuid>,
    pub discovery_error: Option<String>,
    pub encoders: Vec<String>,
}

#[tauri::command]
pub fn hardware_info(state: S<'_>) -> HardwareInfo {
    let selected = state.selected_device().map(|d| d.uuid);
    let hw = state.hw.read();
    match hw {
        Ok(h) => HardwareInfo {
            ready: h.encoders.is_some(),
            devices: h.devices.clone(),
            selected,
            discovery_error: h.discovery_error.clone(),
            encoders: h.encoders.iter().flatten().cloned().collect(),
        },
        Err(_) => HardwareInfo {
            ready: false,
            devices: vec![],
            selected,
            discovery_error: None,
            encoders: vec![],
        },
    }
}

/// Manual GPU override (`None` = automatic). Takes effect for the next job.
#[tauri::command]
pub fn set_preferred_device(state: S<'_>, uuid: Option<DeviceUuid>) -> R<()> {
    let mut s = state
        .settings
        .lock()
        .map_err(|_| err("internal", "settings lock"))?;
    s.preferred_device = uuid;
    s.save(&state.paths.settings)
        .map_err(|e| err("io", e.to_string()))
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbeResult {
    pub path: String,
    pub info: Option<MediaInfo>,
    pub error: Option<String>,
}

#[tauri::command]
pub async fn probe_media(state: S<'_>, paths: Vec<String>) -> R<Vec<ProbeResult>> {
    let ffprobe = state.paths.ffprobe.clone();
    tauri::async_runtime::spawn_blocking(move || {
        paths
            .into_iter()
            .map(
                |p| match fs_media::probe::probe(&ffprobe, &PathBuf::from(&p)) {
                    Ok(info) => ProbeResult {
                        path: p,
                        info: Some(info),
                        error: None,
                    },
                    Err(e) => ProbeResult {
                        path: p,
                        info: None,
                        error: Some(e.to_string()),
                    },
                },
            )
            .collect()
    })
    .await
    .map_err(|e| err("internal", e.to_string()))
}

#[tauri::command]
pub fn enqueue_jobs(state: S<'_>, specs: Vec<JobSpec>) -> R<Vec<JobId>> {
    for spec in &specs {
        reject_unsupported(&spec.pipeline)?;
    }
    let mut ids = Vec::with_capacity(specs.len());
    for spec in &specs {
        let id = JobId::new();
        journal(&state, |j| j.insert_job(id, spec))?;
        state.emit(JobEvent::StateChanged {
            job: id,
            state: JobState::Queued,
            reason: None,
        });
        ids.push(id);
    }
    state.queue_changed.notify_one();
    Ok(ids)
}

fn reject_unsupported(pipeline: &fs_types::PipelineSpec) -> R<()> {
    if let Some(s) = pipeline.stages.iter().find(|s| {
        matches!(
            s,
            fs_types::StageConfig::Denoise { .. } | fs_types::StageConfig::FaceFix { .. }
        )
    }) {
        return Err(err(
            "unsupported",
            format!("{:?} isn't available in this build yet", s.kind()),
        ));
    }
    pipeline
        .validate()
        .map_err(|e| err("invalidPipeline", format!("{e:?}")))
}

/// Render a 2 s preview of `path` at `at_s` with the given pipeline. The
/// result's `prefix` is served through the `fs://` protocol.
#[tauri::command]
pub async fn render_preview(
    state: S<'_>,
    path: String,
    at_s: f64,
    seconds: f64,
    pipeline: fs_types::PipelineSpec,
) -> R<fs_types::PreviewResult> {
    reject_unsupported(&pipeline)?;
    let tools = fs_core::preview::PreviewTools {
        ffmpeg: state.paths.ffmpeg.clone(),
        ffprobe: state.paths.ffprobe.clone(),
        worker: state.paths.worker.clone(),
        models_dir: state.paths.models.clone(),
        device: state.selected_device(),
    };
    fs_core::preview::prune(&state.paths.previews, 6);
    tracing::info!(%path, at_s, seconds, stages = pipeline.stages.len(), "rendering preview");
    let res = fs_core::preview::render(
        &tools,
        &PathBuf::from(&path),
        at_s,
        seconds,
        &pipeline,
        &state.paths.previews,
    )
    .await;
    match res {
        Ok(r) => {
            tracing::info!(
                frames = r.after_frames,
                w = r.width,
                h = r.height,
                ms = r.elapsed_ms,
                device = ?r.device,
                "preview ready"
            );
            Ok(r)
        }
        Err(e) => {
            tracing::warn!(error = %e, "preview failed");
            Err(err("preview", e))
        }
    }
}

/// Cached JPEG thumbnail; returns its `fs://` path (e.g. `thumbs/<key>.jpg`).
#[tauri::command]
pub async fn thumbnail(state: S<'_>, path: String) -> R<String> {
    fs_core::preview::thumbnail(
        &state.paths.ffmpeg,
        &state.paths.ffprobe,
        &PathBuf::from(&path),
        &state.paths.thumbs,
    )
    .await
    .map_err(|e| {
        tracing::warn!(%path, error = %e, "thumbnail failed");
        err("thumbnail", e)
    })
}

#[tauri::command]
pub fn list_jobs(state: S<'_>) -> R<Vec<JobSummary>> {
    journal(&state, |j| j.list_jobs())
}

fn active_commands(
    state: &AppState,
    id: JobId,
) -> Option<tokio::sync::mpsc::UnboundedSender<JobCommand>> {
    state
        .active
        .lock()
        .ok()?
        .as_ref()
        .filter(|a| a.id == id)
        .map(|a| a.commands.clone())
}

#[tauri::command]
pub fn pause_job(state: S<'_>, id: JobId, mode: PauseMode) -> R<()> {
    let tx = active_commands(&state, id).ok_or_else(|| err("notRunning", "job is not running"))?;
    tx.send(JobCommand::Pause(mode))
        .map_err(|_| err("notRunning", "job finished"))
}

/// Resume a paused job, or re-queue an interrupted/failed one (it continues
/// from its last committed segment).
#[tauri::command]
pub fn resume_job(state: S<'_>, id: JobId) -> R<()> {
    if let Some(tx) = active_commands(&state, id) {
        return tx
            .send(JobCommand::Resume)
            .map_err(|_| err("notRunning", "job finished"));
    }
    match journal(&state, |j| j.state(id))? {
        JobState::Interrupted | JobState::Failed | JobState::Paused => {
            journal(&state, |j| j.set_state(id, JobState::Queued, None))?;
            state.emit(JobEvent::StateChanged {
                job: id,
                state: JobState::Queued,
                reason: None,
            });
            state.queue_changed.notify_one();
            Ok(())
        }
        s => Err(err("invalidState", format!("cannot resume a {s:?} job"))),
    }
}

#[tauri::command]
pub fn cancel_job(state: S<'_>, id: JobId) -> R<()> {
    if let Some(tx) = active_commands(&state, id) {
        return tx
            .send(JobCommand::Cancel)
            .map_err(|_| err("notRunning", "job finished"));
    }
    journal(&state, |j| j.set_state(id, JobState::Cancelled, None))?;
    let _ = std::fs::remove_dir_all(state.paths.work_root.join(id.to_string()));
    state.emit(JobEvent::StateChanged {
        job: id,
        state: JobState::Cancelled,
        reason: None,
    });
    Ok(())
}

#[tauri::command]
pub fn remove_job(state: S<'_>, id: JobId) -> R<()> {
    if active_commands(&state, id).is_some() {
        return Err(err("running", "cancel the job before removing it"));
    }
    journal(&state, |j| j.delete_job(id))?;
    let _ = std::fs::remove_dir_all(state.paths.work_root.join(id.to_string()));
    Ok(())
}

#[tauri::command]
pub fn reorder_jobs(state: S<'_>, order: Vec<JobId>) -> R<()> {
    journal(&state, |j| j.reorder(&order))?;
    state.queue_changed.notify_one();
    Ok(())
}

#[tauri::command]
pub fn set_eco_mode(state: S<'_>, mode: EcoMode) -> R<()> {
    let mut s = state
        .settings
        .lock()
        .map_err(|_| err("internal", "settings lock"))?;
    s.eco = mode;
    s.save(&state.paths.settings)
        .map_err(|e| err("io", e.to_string()))
}

#[tauri::command]
pub fn disk_free(path: String) -> Option<u64> {
    fs_core::storage::free_space(&PathBuf::from(path))
}

/// Stream job events to the UI until the window goes away.
#[tauri::command]
pub fn subscribe_jobs(state: S<'_>, channel: Channel<JobEvent>) {
    let mut rx = state.job_events.subscribe();
    tauri::async_runtime::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(ev) => {
                    if channel.send(ev).is_err() {
                        return;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => return,
            }
        }
    });
}

/// Stream telemetry (≈2 Hz) to the UI.
#[tauri::command]
pub fn subscribe_telemetry(state: S<'_>, channel: Channel<Telemetry>) {
    let mut rx = state.telemetry.subscribe();
    tauri::async_runtime::spawn(async move {
        while rx.changed().await.is_ok() {
            let t = rx.borrow_and_update().clone();
            if channel.send(t).is_err() {
                return;
            }
        }
    });
}
