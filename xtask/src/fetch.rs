//! Verified downloads into `.cache/downloads/`.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

use crate::manifest::Artifact;
use crate::util;

/// Download `a` (or reuse the cache) and verify its SHA-256.
/// An empty pin is an error: run `cargo xtask pin` and review the diff first.
pub fn fetch(a: &Artifact) -> Result<PathBuf> {
    if a.sha256.is_empty() {
        bail!(
            "{} is not pinned (sha256 = \"\"); run `cargo xtask pin`",
            a.url
        );
    }
    let path = download(a)?;
    let got = sha256_file(&path)?;
    if !got.eq_ignore_ascii_case(&a.sha256) {
        // Never leave a poisoned file in the cache.
        let _ = std::fs::remove_file(&path);
        bail!(
            "SHA-256 mismatch for {}\n  expected {}\n  got      {got}",
            a.url,
            a.sha256
        );
    }
    Ok(path)
}

/// Download without verification (used by `pin` only).
pub fn download(a: &Artifact) -> Result<PathBuf> {
    let dir = util::cache_dir().join("downloads");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(a.file_name());
    if path.exists() {
        return Ok(path);
    }
    let part = path.with_extension("part");
    eprintln!("  ↓ {}", a.effective_url());
    util::run(
        Command::new("curl")
            .args([
                "--fail",
                "--location",
                "--retry",
                "3",
                "--proto",
                "=https",
                "--tlsv1.2",
            ])
            .args(["--silent", "--show-error", "--output"])
            .arg(&part)
            .arg(a.effective_url()),
    )?;
    std::fs::rename(&part, &path)?;
    Ok(path)
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}
