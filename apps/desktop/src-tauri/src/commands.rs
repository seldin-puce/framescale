//! IPC commands. Every Pro gate is enforced here in Rust (and again in the
//! worker). The UI's lock icons are only hints.

use std::path::PathBuf;
use std::sync::Arc;

use fs_core::JobCommand;
use fs_types::license::FREE_BATCH_LIMIT;
use fs_types::{
    DeviceUuid, EcoMode, Entitlements, Feature, GpuDevice, JobEvent, JobId, JobSpec, JobState,
    JobSummary, MediaInfo, PauseMode, Telemetry,
};
use serde::Serialize;
use tauri::State;
use tauri::ipc::Channel;

use crate::state::{AppState, now_secs};

type S<'a> = State<'a, Arc<AppState>>;

/// Error shape the UI can switch on.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CmdError {
    pub code: &'static str,
    pub message: String,
    /// Set when the action needs a Pro feature (opens the activation modal).
    pub feature: Option<Feature>,
}

type R<T> = Result<T, CmdError>;

fn err(code: &'static str, message: impl Into<String>) -> CmdError {
    CmdError {
        code,
        message: message.into(),
        feature: None,
    }
}

fn pro(feature: Feature, message: &str) -> CmdError {
    CmdError {
        code: "proRequired",
        message: message.into(),
        feature: Some(feature),
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
    let ent = state.entitlements();
    let pending = journal(&state, |j| j.list_jobs())?
        .iter()
        .filter(|j| !j.state.is_terminal())
        .count();
    if !ent.has(Feature::BatchUnlimited) && pending + specs.len() > FREE_BATCH_LIMIT {
        return Err(pro(
            Feature::BatchUnlimited,
            &format!("the free tier queues up to {FREE_BATCH_LIMIT} videos at once"),
        ));
    }
    for spec in &specs {
        if let Err(errors) = spec.pipeline.validate() {
            return Err(err("invalidPipeline", format!("{errors:?}")));
        }
        // Early, friendly gate. The worker repeats it against the compiled plan.
        if let Some(f) = ent.missing_for(spec, 0, 0).first() {
            return Err(pro(*f, "this pipeline uses a Pro feature"));
        }
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
    if !state.entitlements().has(Feature::BatchReorder) {
        return Err(pro(
            Feature::BatchReorder,
            "reordering the queue is a Pro feature",
        ));
    }
    journal(&state, |j| j.reorder(&order))?;
    state.queue_changed.notify_one();
    Ok(())
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LicenseInfo {
    pub entitlements: Entitlements,
    pub development_key: bool,
}

#[tauri::command]
pub fn get_license(state: S<'_>) -> LicenseInfo {
    LicenseInfo {
        entitlements: state.entitlements(),
        development_key: state.dev_license_key,
    }
}

#[tauri::command]
pub fn activate_license(state: S<'_>, token: String) -> R<LicenseInfo> {
    let verifier = fs_license::Verifier::embedded();
    let device = fs_license::machine_fingerprint();
    let ent = verifier
        .verify(token.trim(), now_secs(), device.as_deref())
        .map_err(|e| err("licenseInvalid", e.to_string()))?;
    if let Some(dir) = state.paths.license.parent() {
        std::fs::create_dir_all(dir).map_err(|e| err("io", e.to_string()))?;
    }
    std::fs::write(&state.paths.license, token.trim()).map_err(|e| err("io", e.to_string()))?;
    if let Ok(mut t) = state.license_token.write() {
        *t = Some(token.trim().to_owned());
    }
    if let Ok(mut e) = state.entitlements.write() {
        *e = ent;
    }
    Ok(get_license(state))
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
