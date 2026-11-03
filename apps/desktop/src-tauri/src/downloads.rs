//! Models downloaded on demand: `third_party/manifest.toml` entries with
//! `ondemand = true` (CodeFormer, whose license isn't OSI-approved, so the
//! installers don't carry it). The user starts the download after seeing the
//! license; files come from the pinned URLs, are checked against the pinned
//! size and SHA-256, and land in `<app data>/models/<dest>/`.
//!
//! A download fills `<dest>.partial/` and renames it to `<dest>` when every
//! file has verified, so the engine never sees half a model. Files already
//! complete in `.partial` are kept, so an interrupted download resumes.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use fs_types::{DownloadProgress, ModelStatus};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

const MANIFEST: &str = include_str!("../../../../third_party/manifest.toml");

#[derive(Debug, Deserialize)]
struct Manifest {
    models: Vec<Entry>,
}

#[derive(Debug, Clone, Deserialize)]
struct Entry {
    id: String,
    license: String,
    url: String,
    sha256: String,
    dest: String,
    file: Option<String>,
    #[serde(default)]
    ondemand: bool,
    size: Option<u64>,
}

/// One file of an on-demand model.
#[derive(Debug, Clone)]
struct File {
    url: String,
    sha256: String,
    name: String,
    size: u64,
}

#[derive(Debug, Clone)]
struct Model {
    dest: String,
    license: String,
    license_url: String,
    files: Vec<File>,
}

fn model(id: &str) -> Result<Model, String> {
    let m: Manifest = toml::from_str(MANIFEST).map_err(|e| format!("model manifest: {e}"))?;
    let entries: Vec<Entry> = m
        .models
        .into_iter()
        .filter(|e| e.ondemand && e.id == id)
        .collect();
    let first = entries
        .first()
        .ok_or_else(|| format!("'{id}' isn't a downloadable model"))?;
    let files = entries
        .iter()
        .map(|e| {
            Ok(File {
                url: e.url.clone(),
                sha256: e.sha256.to_ascii_lowercase(),
                name: e.file.clone().ok_or("on-demand model without `file`")?,
                size: e.size.ok_or("on-demand model without `size`")?,
            })
        })
        .collect::<Result<Vec<_>, &str>>()?;
    let license_url = files
        .iter()
        .find(|f| f.name == "LICENSE")
        .map_or_else(String::new, |f| github_page(&f.url));
    Ok(Model {
        dest: first.dest.clone(),
        license: first.license.clone(),
        license_url,
        files,
    })
}

/// `raw.githubusercontent.com/o/r/rev/path` → `github.com/o/r/blob/rev/path`
/// (the UI may only open github.com links).
fn github_page(raw: &str) -> String {
    match raw.strip_prefix("https://raw.githubusercontent.com/") {
        Some(rest) => {
            let mut parts = rest.splitn(4, '/');
            match (parts.next(), parts.next(), parts.next(), parts.next()) {
                (Some(o), Some(r), Some(rev), Some(path)) => {
                    format!("https://github.com/{o}/{r}/blob/{rev}/{path}")
                }
                _ => raw.to_owned(),
            }
        }
        None => raw.to_owned(),
    }
}

fn complete(dir: &Path, f: &File) -> bool {
    std::fs::metadata(dir.join(&f.name)).is_ok_and(|m| m.len() == f.size)
}

pub fn status(downloads: &Path, id: &str) -> Result<ModelStatus, String> {
    let m = model(id)?;
    let dir = downloads.join(&m.dest);
    Ok(ModelStatus {
        id: id.to_owned(),
        installed: m.files.iter().all(|f| complete(&dir, f)),
        bytes: m.files.iter().map(|f| f.size).sum(),
        license: m.license,
        license_url: m.license_url,
    })
}

/// Download `id` into `downloads`; `progress` gets the running byte count.
/// Stops with an error when `cancel` is set.
pub async fn download(
    downloads: &Path,
    id: &str,
    cancel: &AtomicBool,
    progress: impl Fn(DownloadProgress),
) -> Result<(), String> {
    let m = model(id)?;
    let done_dir = downloads.join(&m.dest);
    if m.files.iter().all(|f| complete(&done_dir, f)) {
        return Ok(());
    }
    let part = downloads.join(format!("{}.partial", m.dest));
    tokio::fs::create_dir_all(&part)
        .await
        .map_err(|e| format!("can't create {}: {e}", part.display()))?;
    let total: u64 = m.files.iter().map(|f| f.size).sum();
    let mut done: u64 = m
        .files
        .iter()
        .filter(|f| complete(&part, f))
        .map(|f| f.size)
        .sum();
    progress(DownloadProgress { done, total });

    // reqwest uses the process-wide rustls provider (as the updater does).
    let _ = rustls::crypto::ring::default_provider().install_default();
    let client = reqwest::Client::builder()
        .user_agent(concat!("FrameScale-AI/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(std::time::Duration::from_secs(20))
        .build()
        .map_err(|e| format!("network setup failed: {e}"))?;

    for f in &m.files {
        if complete(&part, f) {
            continue;
        }
        fetch(&client, f, &part, cancel, |n| {
            progress(DownloadProgress {
                done: done + n,
                total,
            });
        })
        .await?;
        done += f.size;
    }

    // Every file verified: swap the finished folder in.
    if done_dir.exists() {
        tokio::fs::remove_dir_all(&done_dir)
            .await
            .map_err(|e| format!("can't replace {}: {e}", done_dir.display()))?;
    }
    tokio::fs::rename(&part, &done_dir)
        .await
        .map_err(|e| format!("can't finish the download: {e}"))?;
    progress(DownloadProgress { done: total, total });
    Ok(())
}

async fn fetch(
    client: &reqwest::Client,
    f: &File,
    dir: &Path,
    cancel: &AtomicBool,
    progress: impl Fn(u64),
) -> Result<(), String> {
    let tmp: PathBuf = dir.join(format!("{}.download", f.name));
    let result = async {
        let mut resp = client
            .get(&f.url)
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .map_err(|e| format!("download failed ({}): {e}", f.name))?;
        let mut out = tokio::fs::File::create(&tmp)
            .await
            .map_err(|e| format!("can't write {}: {e}", tmp.display()))?;
        let mut hash = Sha256::new();
        let mut n: u64 = 0;
        let mut last = 0u64;
        while let Some(chunk) = resp
            .chunk()
            .await
            .map_err(|e| format!("download interrupted ({}): {e}", f.name))?
        {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".to_owned());
            }
            n += chunk.len() as u64;
            if n > f.size {
                return Err(format!("{} is larger than expected", f.name));
            }
            hash.update(&chunk);
            out.write_all(&chunk)
                .await
                .map_err(|e| format!("can't write {}: {e}", tmp.display()))?;
            if n - last >= 1 << 20 {
                last = n;
                progress(n);
            }
        }
        out.sync_all()
            .await
            .map_err(|e| format!("can't write {}: {e}", tmp.display()))?;
        drop(out);
        let got: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
        if n != f.size || got != f.sha256 {
            return Err(format!(
                "{} didn't match its published checksum; nothing was installed",
                f.name
            ));
        }
        tokio::fs::rename(&tmp, dir.join(&f.name))
            .await
            .map_err(|e| format!("can't write {}: {e}", f.name))?;
        progress(n);
        Ok(())
    }
    .await;
    if result.is_err() {
        let _ = tokio::fs::remove_file(&tmp).await;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_lists_codeformer() {
        let m = model("codeformer").expect("codeformer is on demand");
        assert_eq!(m.dest, "codeformer");
        let names: Vec<_> = m.files.iter().map(|f| f.name.as_str()).collect();
        for want in [
            "encoder.param",
            "encoder.bin",
            "generator.param",
            "generator.bin",
            "LICENSE",
        ] {
            assert!(names.contains(&want), "{want} missing: {names:?}");
        }
        assert!(m.files.iter().all(|f| f.sha256.len() == 64 && f.size > 0));
        assert!(
            m.license_url
                .starts_with("https://github.com/sczhou/CodeFormer/blob/"),
            "{}",
            m.license_url
        );
        assert!(
            model("realesrgan").is_err(),
            "bundled models aren't downloadable"
        );
    }

    /// The real download (≈189 MB from GitHub): `cargo test -p framescale -- --ignored`.
    #[test]
    #[ignore = "network, 189 MB"]
    fn real_download() {
        let dir = std::env::temp_dir().join(format!("fs-dl-real-{}", std::process::id()));
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let last = std::sync::Mutex::new(DownloadProgress { done: 0, total: 0 });
        let cancel = AtomicBool::new(false);
        rt.block_on(download(&dir, "codeformer", &cancel, |p| {
            *last.lock().expect("lock") = p;
        }))
        .expect("download");
        let p = *last.lock().expect("lock");
        assert_eq!(p.done, p.total);
        assert!(status(&dir, "codeformer").expect("status").installed);
        assert!(
            !dir.join("codeformer.partial").exists(),
            "partial folder renamed"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn status_needs_every_file() {
        let dir = std::env::temp_dir().join(format!("fs-dl-{}", std::process::id()));
        let s = status(&dir, "codeformer").expect("status");
        assert!(!s.installed);
        assert!(s.bytes > 180_000_000);
        assert!(s.license.contains("non-commercial"));
    }
}
