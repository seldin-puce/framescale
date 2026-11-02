//! Archive extraction (.zip, .tar, .tar.xz) in pure Rust, path-traversal safe.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};

/// Extract the whole archive into `dest` (cleared first).
pub fn extract_all(archive: &Path, dest: &Path) -> Result<()> {
    if dest.exists() {
        std::fs::remove_dir_all(dest)?;
    }
    std::fs::create_dir_all(dest)?;
    let name = archive.to_string_lossy();
    if name.ends_with(".zip") {
        extract_zip(archive, dest, |p| Some(p.to_path_buf()))
    } else if name.ends_with(".tar.xz") {
        let tar_path = dest.with_extension("tar");
        {
            let mut input = BufReader::new(File::open(archive)?);
            let mut out = BufWriter::new(File::create(&tar_path)?);
            lzma_rs::xz_decompress(&mut input, &mut out)
                .map_err(|e| anyhow!("xz decode {}: {e:?}", archive.display()))?;
        }
        let r = tar::Archive::new(File::open(&tar_path)?).unpack(dest);
        let _ = std::fs::remove_file(&tar_path);
        r.with_context(|| format!("untar {}", archive.display()))
    } else if name.ends_with(".tar") {
        tar::Archive::new(File::open(archive)?)
            .unpack(dest)
            .with_context(|| format!("untar {}", archive.display()))
    } else {
        bail!("unsupported archive type: {name}")
    }
}

/// Extract only zip entries whose path contains one of `markers`. The output
/// path starts at the marker, e.g. `pkg-1.0/models/x.bin` with marker
/// `models/x.` lands at `dest/models/x.bin`.
pub fn extract_zip_selected(archive: &Path, dest: &Path, markers: &[String]) -> Result<usize> {
    let mut count = 0usize;
    extract_zip(archive, dest, |p| {
        let s = p.to_string_lossy().replace('\\', "/");
        let hit = markers.iter().find_map(|m| s.find(m.as_str()))?;
        count += 1;
        Some(PathBuf::from(&s[hit..]))
    })?;
    Ok(count)
}

fn extract_zip(
    archive: &Path,
    dest: &Path,
    mut map: impl FnMut(&Path) -> Option<PathBuf>,
) -> Result<()> {
    let mut zip = zip::ZipArchive::new(File::open(archive)?)
        .with_context(|| format!("open zip {}", archive.display()))?;
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        if entry.is_dir() {
            continue;
        }
        // `enclosed_name` rejects absolute paths and `..` traversal.
        let Some(inner) = entry.enclosed_name() else {
            continue;
        };
        let Some(rel) = map(&inner) else { continue };
        if !rel.components().all(|c| matches!(c, Component::Normal(_))) {
            bail!("refusing suspicious archive path {}", rel.display());
        }
        let out = dest.join(&rel);
        if let Some(dir) = out.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut f = BufWriter::new(File::create(&out)?);
        std::io::copy(&mut entry, &mut f)?;
        #[cfg(unix)]
        if let Some(mode) = entry.unix_mode() {
            use std::os::unix::fs::PermissionsExt;
            drop(f);
            std::fs::set_permissions(&out, std::fs::Permissions::from_mode(mode & 0o755))?;
        }
    }
    Ok(())
}
