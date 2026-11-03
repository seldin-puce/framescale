//! Process-wide state and the startup sequence.

use std::collections::{BTreeSet, HashSet};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use fs_core::JobCommand;
use fs_hw::VulkanContext;
use fs_journal::Journal;
use fs_media::encoders::{EncoderEnv, Os};
use fs_proc::Ledger;
use fs_types::{GpuDevice, JobEvent, JobId, Telemetry};
use tokio::sync::{Notify, broadcast, watch};

use crate::paths::Paths;
use crate::settings::Settings;

/// Hardware facts discovered in the background after startup.
#[derive(Debug, Default)]
pub struct Hardware {
    pub devices: Vec<GpuDevice>,
    pub vulkan: Option<Arc<VulkanContext>>,
    pub discovery_error: Option<String>,
    /// Encoders that passed a probe encode on this machine.
    pub encoders: Option<BTreeSet<String>>,
}

#[derive(Debug)]
pub struct ActiveJob {
    pub id: JobId,
    pub commands: tokio::sync::mpsc::UnboundedSender<JobCommand>,
}

pub struct AppState {
    pub paths: Paths,
    pub journal: Arc<Mutex<Journal>>,
    pub ledger: Arc<Ledger>,
    pub hw: RwLock<Hardware>,
    /// Signalled when hardware discovery / encoder probing completes.
    pub hw_ready: Notify,
    pub settings: Mutex<Settings>,
    pub active: Mutex<Option<ActiveJob>>,
    /// Wakes the queue loop (new job, resume, reorder…).
    pub queue_changed: Notify,
    /// Open source-vs-output comparison sessions, by session id.
    pub compare: Mutex<std::collections::VecDeque<Arc<fs_core::compare::CompareSession>>>,
    /// Bumped by every preview render and by `cancel_preview`: only the
    /// newest preview runs.
    pub preview_epoch: AtomicU64,
    /// Wakes the running preview when `preview_epoch` moves on.
    pub preview_cancel: Notify,
    pub job_events: broadcast::Sender<JobEvent>,
    pub telemetry: watch::Sender<Telemetry>,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("paths", &self.paths)
            .finish_non_exhaustive()
    }
}

pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

impl AppState {
    /// Synchronous part of startup: everything the UI needs before first paint.
    pub fn init(paths: Paths) -> Result<Arc<Self>, String> {
        // 1. Zombie reaper: kill survivors of a previous crash before anything else.
        let ledger = Arc::new(Ledger::new(&paths.ledger));
        match ledger.reap_stale() {
            Ok(k) if !k.is_empty() => tracing::warn!(pids = ?k, "reaped orphaned workers"),
            Err(e) => tracing::warn!(error = %e, "PID ledger unreadable"),
            _ => {}
        }

        // 2. Journal: anything that was running when we died is now `interrupted`.
        let journal = Journal::open(&paths.journal).map_err(|e| e.to_string())?;
        let interrupted = journal.mark_interrupted().map_err(|e| e.to_string())?;
        if !interrupted.is_empty() {
            tracing::info!(
                count = interrupted.len(),
                "jobs interrupted by the previous session"
            );
        }

        let (job_events, _) = broadcast::channel(512);
        let (telemetry, _) = watch::channel(Telemetry::default());
        Ok(Arc::new(Self {
            settings: Mutex::new(Settings::load(&paths.settings)),
            paths,
            journal: Arc::new(Mutex::new(journal)),
            ledger,
            hw: RwLock::new(Hardware::default()),
            hw_ready: Notify::new(),
            active: Mutex::new(None),
            queue_changed: Notify::new(),
            compare: Mutex::new(std::collections::VecDeque::new()),
            preview_epoch: AtomicU64::new(0),
            preview_cancel: Notify::new(),
            job_events,
            telemetry,
        }))
    }

    /// Slow startup work, off the UI thread: janitor, GPU discovery, encoder probe.
    pub async fn init_background(self: Arc<Self>) {
        let me = self.clone();
        let _ = tauri::async_runtime::spawn_blocking(move || {
            // Janitor: abandoned temp dirs only. Resumable jobs keep their segments.
            let resumable: HashSet<JobId> = me
                .journal
                .lock()
                .ok()
                .and_then(|j| j.resumable_jobs().ok())
                .unwrap_or_default()
                .into_iter()
                .collect();
            let report = fs_core::janitor::sweep(
                &me.paths.work_root,
                &resumable,
                SystemTime::now(),
                fs_core::janitor::MIN_AGE,
            );
            if !report.deleted.is_empty() {
                tracing::info!(
                    dirs = report.deleted.len(),
                    bytes = report.freed_bytes,
                    "temp janitor freed space"
                );
            }

            // GPU discovery.
            let mut hw = Hardware::default();
            match VulkanContext::discover() {
                Ok((ctx, mut devices)) => {
                    fs_hw::rank(&mut devices);
                    hw.devices = devices;
                    hw.vulkan = Some(Arc::new(ctx));
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Vulkan discovery failed; CPU path only");
                    hw.discovery_error = Some(e.to_string());
                }
            }

            // Encoder probe for the selected device's vendor.
            let selected = me.selected_device_in(&hw.devices);
            let vendor = selected.map(|d| d.vendor);
            let vaapi = selected
                .and_then(|d| d.drm_render_minor)
                .map(|m| format!("/dev/dri/renderD{m}"));
            let candidates = fs_media::encoders::all_candidates(Os::current(), vendor);
            hw.encoders = Some(fs_media::encoders::probe_usable(
                &me.paths.ffmpeg,
                &candidates,
                vaapi.as_deref(),
            ));
            tracing::info!(encoders = ?hw.encoders, "usable encoders");

            if let Ok(mut slot) = me.hw.write() {
                *slot = hw;
            }
        })
        .await;
        self.hw_ready.notify_waiters();
        self.queue_changed.notify_one();
    }

    pub fn hardware_ready(&self) -> bool {
        self.hw.read().is_ok_and(|h| h.encoders.is_some())
    }

    fn selected_device_in<'a>(&self, devices: &'a [GpuDevice]) -> Option<&'a GpuDevice> {
        let pref = self.settings.lock().ok().and_then(|s| s.preferred_device);
        fs_hw::select_default(devices, pref.as_ref())
    }

    pub fn selected_device(&self) -> Option<GpuDevice> {
        let hw = self.hw.read().ok()?;
        self.selected_device_in(&hw.devices).cloned()
    }

    pub fn encoder_env(&self) -> EncoderEnv {
        let device = self.selected_device();
        EncoderEnv {
            os: Os::current(),
            vendor: device.as_ref().map(|d| d.vendor),
            usable: self
                .hw
                .read()
                .ok()
                .and_then(|h| h.encoders.clone())
                .unwrap_or_default(),
            vaapi_device: device
                .and_then(|d| d.drm_render_minor)
                .map(|m| format!("/dev/dri/renderD{m}")),
        }
    }

    pub fn emit(&self, ev: JobEvent) {
        let _ = self.job_events.send(ev);
    }
}
