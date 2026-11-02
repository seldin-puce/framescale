//! Assemble Tauri sidecars: `binaries/<name>-<target-triple>[.exe]`
//! (+ `frameworks/libMoltenVK.dylib` on macOS).

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::manifest::Manifest;
use crate::{archive, fetch, gate, util};

const SIDECARS: [&str; 3] = ["ffmpeg", "ffprobe", "fs-worker"];
const UNIVERSAL: &str = "universal-apple-darwin";
const MAC_ARCHES: [&str; 2] = ["aarch64-apple-darwin", "x86_64-apple-darwin"];

pub fn run(m: &Manifest, triple: &str, stub: bool) -> Result<()> {
    eprintln!("sidecars → {triple}{}", if stub { " (stubs)" } else { "" });
    if stub {
        return write_stubs(triple);
    }
    if triple == UNIVERSAL {
        for arch in MAC_ARCHES {
            build_one(m, arch)?;
        }
        for name in SIDECARS {
            let parts: Vec<PathBuf> = MAC_ARCHES.iter().map(|a| sidecar_path(name, a)).collect();
            util::run(
                Command::new("lipo")
                    .arg("-create")
                    .args(&parts)
                    .arg("-output")
                    .arg(sidecar_path(name, UNIVERSAL)),
            )?;
        }
    } else {
        build_one(m, triple)?;
    }
    if util::is_macos(triple) {
        install_moltenvk(m)?;
    }
    Ok(())
}

pub fn sidecar_path(name: &str, triple: &str) -> PathBuf {
    let suffix = util::exe_suffix(triple);
    util::tauri_dir()
        .join("binaries")
        .join(format!("{name}-{triple}{suffix}"))
}

fn build_one(m: &Manifest, triple: &str) -> Result<()> {
    // 1. FFmpeg + ffprobe
    let bin_dir = ffmpeg_bin_dir(m, triple)?;
    let suffix = util::exe_suffix(triple);
    let ffmpeg = bin_dir.join(format!("ffmpeg{suffix}"));
    if can_execute(triple)? {
        gate::check(&m.ffmpeg.gate, &ffmpeg, triple)?;
    } else {
        eprintln!("  ! cannot execute {triple} binaries on this host; gate deferred to CI");
    }
    for name in ["ffmpeg", "ffprobe"] {
        util::copy_executable(
            &bin_dir.join(format!("{name}{suffix}")),
            &sidecar_path(name, triple),
        )?;
    }

    // 2. fs-worker (our own binary)
    util::cargo(&["build", "--release", "-p", "fs-worker", "--target", triple])?;
    let built = util::root()
        .join("target")
        .join(triple)
        .join("release")
        .join(format!("fs-worker{suffix}"));
    util::copy_executable(&built, &sidecar_path("fs-worker", triple))?;
    eprintln!("  ✓ sidecars installed for {triple}");
    Ok(())
}

/// Returns the directory that holds `ffmpeg[.exe]` and `ffprobe[.exe]` for `triple`.
fn ffmpeg_bin_dir(m: &Manifest, triple: &str) -> Result<PathBuf> {
    if util::is_macos(triple) {
        let out = util::root().join("third_party/ffmpeg/out").join(triple);
        if !out.join("bin/ffmpeg").exists() {
            build_ffmpeg_macos(m, triple, &out)?;
        }
        return Ok(out.join("bin"));
    }
    let artifact = m
        .ffmpeg
        .targets
        .get(triple)
        .with_context(|| format!("no FFmpeg build pinned for {triple} in manifest"))?;
    let archive_path = fetch::fetch(artifact)?;
    let dest = util::cache_dir().join("ffmpeg").join(triple);
    archive::extract_all(&archive_path, &dest)?;
    let suffix = util::exe_suffix(triple);
    let ffmpeg = util::find_file(&dest, &format!("bin/ffmpeg{suffix}"))
        .context("ffmpeg binary not found inside pinned archive")?;
    Ok(ffmpeg.parent().map(Path::to_path_buf).unwrap_or(dest))
}

fn build_ffmpeg_macos(m: &Manifest, triple: &str, out: &Path) -> Result<()> {
    if !cfg!(target_os = "macos") {
        bail!("FFmpeg for {triple} is built from source and requires a macOS host");
    }
    // Fetch + verify all sources up front; the script never downloads anything.
    let src = util::cache_dir().join("ffmpeg-src");
    std::fs::create_dir_all(&src)?;
    let mut all = vec![("ffmpeg", &m.ffmpeg.source.artifact)];
    all.extend(m.ffmpeg.source.deps.iter().map(|(k, v)| (k.as_str(), v)));
    for (name, a) in all {
        let path = fetch::fetch(a)?;
        std::fs::copy(
            &path,
            src.join(format!("{name}.tar.{}", ext(a.file_name()))),
        )?;
    }
    util::run(
        Command::new("bash")
            .arg(util::root().join("third_party/ffmpeg/build-macos.sh"))
            .arg(triple)
            .arg(&src)
            .arg(out),
    )
}

fn ext(file: &str) -> &str {
    file.rsplit('.').next().unwrap_or("gz")
}

/// Extract the pinned ncnn source tree; returns the directory with `CMakeLists.txt`.
/// Idempotent: a stamp file records the extracted archive's hash.
pub fn fetch_ncnn(m: &Manifest) -> Result<PathBuf> {
    let root = util::cache_dir().join("ncnn");
    let src = root.join("src");
    let stamp = root.join("STAMP");
    if std::fs::read_to_string(&stamp).is_ok_and(|s| s.trim() == m.ncnn.sha256) {
        return Ok(src);
    }
    let zip = fetch::fetch(&m.ncnn)?;
    let tmp = root.join("extract");
    archive::extract_all(&zip, &tmp)?;
    // The archive either has files at its root or one top-level folder.
    let cmake = if tmp.join("CMakeLists.txt").exists() {
        tmp.clone()
    } else {
        std::fs::read_dir(&tmp)?
            .flatten()
            .map(|e| e.path())
            .find(|p| p.join("CMakeLists.txt").exists() && p.join("src").join("gpu.h").exists())
            .context("ncnn archive has no top-level CMakeLists.txt")?
    };
    if src.exists() {
        std::fs::remove_dir_all(&src)?;
    }
    std::fs::rename(&cmake, &src)?;
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::write(&stamp, &m.ncnn.sha256)?;
    Ok(src)
}

fn install_moltenvk(m: &Manifest) -> Result<()> {
    let tarball = fetch::fetch(&m.moltenvk)?;
    let dest = util::cache_dir().join("moltenvk");
    archive::extract_all(&tarball, &dest)?;
    // The macOS dylib in the release is already universal (arm64 + x86_64).
    let dylib = util::find_file(&dest, "dylib/macOS/libMoltenVK.dylib")
        .context("libMoltenVK.dylib (macOS) not found in MoltenVK release")?;
    let target = util::tauri_dir().join("frameworks/libMoltenVK.dylib");
    std::fs::copy(&dylib, &target)?;
    eprintln!("  ✓ MoltenVK → {}", target.display());
    Ok(())
}

fn can_execute(triple: &str) -> Result<bool> {
    let host = util::host_triple()?;
    // Apple Silicon hosts run x86_64 binaries via Rosetta 2.
    Ok(host == triple || (util::is_macos(&host) && util::is_macos(triple)))
}

fn write_stubs(triple: &str) -> Result<()> {
    for name in SIDECARS {
        let p = sidecar_path(name, triple);
        if !p.exists() {
            if let Some(dir) = p.parent() {
                std::fs::create_dir_all(dir)?;
            }
            std::fs::write(&p, b"")?;
            util::make_executable(&p)?;
        }
    }
    if util::is_macos(triple) {
        let fw = util::tauri_dir().join("frameworks/libMoltenVK.dylib");
        if !fw.exists() {
            std::fs::write(fw, b"")?;
        }
    }
    Ok(())
}
