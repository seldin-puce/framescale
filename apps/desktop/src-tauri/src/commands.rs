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
pub fn enqueue_jobs(state: S<'_>, mut specs: Vec<JobSpec>) -> R<Vec<JobId>> {
    for spec in &specs {
        reject_unsupported(&spec.pipeline)?;
    }
    // Never overwrite: not an existing file, not another job's pending output.
    let mut taken: std::collections::HashSet<String> = journal(&state, |j| j.list_jobs())?
        .into_iter()
        .filter(|j| !matches!(j.state, JobState::Done | JobState::Cancelled))
        .map(|j| j.spec.output.path)
        .collect();
    for spec in &mut specs {
        spec.output.path = unique_output(&spec.output.path, &taken);
        taken.insert(spec.output.path.clone());
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

/// `clip_framescale_2x.mp4` → `clip_framescale_2x (2).mp4`, … while the name
/// is taken on disk or by a queued job.
fn unique_output(path: &str, taken: &std::collections::HashSet<String>) -> String {
    let free = |c: &std::path::Path| !c.exists() && !taken.contains(c.to_string_lossy().as_ref());
    let p = std::path::Path::new(path);
    if free(p) {
        return path.to_owned();
    }
    let stem = p
        .file_stem()
        .map_or_else(String::new, |s| s.to_string_lossy().into_owned());
    let ext = p
        .extension()
        .map_or_else(String::new, |e| format!(".{}", e.to_string_lossy()));
    (2u32..)
        .map(|i| p.with_file_name(format!("{stem} ({i}){ext}")))
        .find(|c| free(c))
        .map_or_else(|| path.to_owned(), |c| c.to_string_lossy().into_owned())
}

/// What a CodeFormer-ncnn model folder must contain.
const CODEFORMER_FILES: [&str; 4] = [
    "encoder.param",
    "encoder.bin",
    "generator.param",
    "generator.bin",
];

fn check_codeformer(dir: &std::path::Path) -> R<()> {
    let missing: Vec<&str> = CODEFORMER_FILES
        .into_iter()
        .filter(|f| !dir.join(f).is_file())
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(err(
            "faceModel",
            format!("{} is missing {}", dir.display(), missing.join(", ")),
        ))
    }
}

/// Is this a usable CodeFormer-ncnn folder? (Only fixed file names are
/// checked for existence; nothing is read.)
#[tauri::command]
pub fn check_face_model(path: String) -> R<()> {
    check_codeformer(std::path::Path::new(&path))
}

fn reject_unsupported(pipeline: &fs_types::PipelineSpec) -> R<()> {
    for s in &pipeline.stages {
        if let fs_types::StageConfig::FaceFix { model, .. } = s {
            let Some(dir) = model.path.as_deref() else {
                return Err(err(
                    "faceModel",
                    "Face Fix needs your CodeFormer model folder: choose it in the Face Fix stage",
                ));
            };
            check_codeformer(std::path::Path::new(dir))?;
        }
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

/// The output file of a finished job. Paths come from the journal, never from
/// the webview, so these commands can't be used to open arbitrary files.
fn finished_output(state: &AppState, id: JobId) -> R<PathBuf> {
    let (job_state, spec) = journal(state, |j| Ok((j.state(id)?, j.spec(id)?)))?;
    if job_state != JobState::Done {
        return Err(err("notFinished", "this job hasn't finished yet"));
    }
    let path = PathBuf::from(spec.output.path);
    if !path.exists() {
        return Err(err(
            "notFound",
            format!("{} was moved or deleted", path.display()),
        ));
    }
    Ok(path)
}

/// Play the rendered video in the system's default player.
#[tauri::command]
pub fn open_output(app: tauri::AppHandle, state: S<'_>, id: JobId) -> R<()> {
    use tauri_plugin_opener::OpenerExt;
    let path = finished_output(&state, id)?;
    app.opener()
        .open_path(path.to_string_lossy(), None::<&str>)
        .map_err(|e| err("open", e.to_string()))
}

/// Show the rendered video in the file manager, selected.
#[tauri::command]
pub fn reveal_output(app: tauri::AppHandle, state: S<'_>, id: JobId) -> R<()> {
    use tauri_plugin_opener::OpenerExt;
    let path = finished_output(&state, id)?;
    app.opener()
        .reveal_item_in_dir(path)
        .map_err(|e| err("reveal", e.to_string()))
}

/// Play the original source of a job in the system's default player.
#[tauri::command]
pub fn open_source(app: tauri::AppHandle, state: S<'_>, id: JobId) -> R<()> {
    use tauri_plugin_opener::OpenerExt;
    let path = PathBuf::from(journal(&state, |j| j.spec(id))?.input);
    if !path.exists() {
        return Err(err(
            "notFound",
            format!("{} was moved or deleted", path.display()),
        ));
    }
    app.opener()
        .open_path(path.to_string_lossy(), None::<&str>)
        .map_err(|e| err("open", e.to_string()))
}

/// Start comparing a finished render with its source. Frames are served by
/// the `fs://` protocol (`compare/<session>/…`, see `protocol.rs`).
#[tauri::command]
pub async fn compare_open(state: S<'_>, id: JobId, max_width: u32) -> R<fs_types::CompareInfo> {
    let output = finished_output(&state, id)?;
    let source = PathBuf::from(journal(&state, |j| j.spec(id))?.input);
    if !source.exists() {
        return Err(err(
            "notFound",
            format!("the original, {}, was moved or deleted", source.display()),
        ));
    }
    let session = fs_core::compare::CompareSession::open(
        &state.paths.ffmpeg,
        &state.paths.ffprobe,
        &source,
        &output,
        max_width,
    )
    .await
    .map_err(|e| err("compare", e))?;
    let info = session.info.clone();
    if let Ok(mut open) = state.compare.lock() {
        open.push_back(Arc::new(session));
        // One viewer: older sessions only linger while the UI switches over.
        while open.len() > 2 {
            if let Some(old) = open.pop_front() {
                old.close();
            }
        }
    }
    Ok(info)
}

/// Stop a comparison's decoders.
#[tauri::command]
pub fn compare_close(state: S<'_>, session: String) {
    if let Ok(mut open) = state.compare.lock() {
        open.retain(|s| {
            let keep = s.info.session != session;
            if !keep {
                s.close();
            }
            keep
        });
    }
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

#[cfg(test)]
mod tests {
    use super::unique_output;

    #[test]
    fn output_names_never_collide() {
        let dir = std::env::temp_dir().join(format!("fs-unique-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let p = |n: &str| dir.join(n).to_string_lossy().into_owned();
        let none = std::collections::HashSet::new();
        assert_eq!(unique_output(&p("a_2x.mp4"), &none), p("a_2x.mp4"));
        std::fs::write(p("a_2x.mp4"), b"").expect("w");
        assert_eq!(unique_output(&p("a_2x.mp4"), &none), p("a_2x (2).mp4"));
        let taken = [p("a_2x (2).mp4")].into_iter().collect();
        assert_eq!(unique_output(&p("a_2x.mp4"), &taken), p("a_2x (3).mp4"));
        let _ = std::fs::remove_dir_all(dir);
    }
}
