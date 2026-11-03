//! Filesystem layout of an installed app.

use std::path::PathBuf;

use tauri::Manager;

#[derive(Debug, Clone)]
pub struct Paths {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
    pub worker: PathBuf,
    pub models: PathBuf,
    pub journal: PathBuf,
    pub ledger: PathBuf,
    pub settings: PathBuf,
    pub work_root: PathBuf,
    pub previews: PathBuf,
    pub thumbs: PathBuf,
}

/// Tauri installs sidecars next to the main executable with the target-triple
/// suffix stripped (`binaries/ffmpeg-x86_64-…` → `ffmpeg`). `tauri dev` copies
/// them the same way into `target/<profile>/`.
fn sidecar(name: &str) -> PathBuf {
    let dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(PathBuf::from))
        .unwrap_or_default();
    dir.join(format!("{name}{}", std::env::consts::EXE_SUFFIX))
}

impl Paths {
    pub fn resolve(app: &tauri::AppHandle) -> tauri::Result<Self> {
        let data = app.path().app_data_dir()?;
        let cache = app.path().app_cache_dir()?;
        let resources = app.path().resource_dir()?;
        Ok(Self {
            ffmpeg: sidecar("ffmpeg"),
            ffprobe: sidecar("ffprobe"),
            worker: sidecar("fs-worker"),
            models: resources.join("models"),
            journal: data.join("journal.sqlite3"),
            ledger: data.join("workers.jsonl"),
            settings: data.join("settings.json"),
            work_root: cache.join("jobs"),
            previews: cache.join("previews"),
            thumbs: cache.join("thumbs"),
        })
    }
}
