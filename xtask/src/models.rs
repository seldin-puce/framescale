//! Bundled model extraction into `apps/desktop/src-tauri/resources/models/<dest>/`;
//! on-demand models (downloaded by the app) into `target/ondemand-models/<dest>/`.

use anyhow::{Context, Result, bail};

use crate::manifest::Manifest;
use crate::{archive, convert, fetch, util};

pub fn run(m: &Manifest) -> Result<()> {
    let bundled = util::tauri_dir().join("resources/models");
    let ondemand = ondemand_root();
    let root = |model: &crate::manifest::Model| {
        if model.ondemand { &ondemand } else { &bundled }
    };
    // Several entries may share a destination (single-file artifacts): clear
    // each destination once, before anything is written into it. An
    // on-demand model's old bundled copy goes too.
    let mut cleared = std::collections::HashSet::new();
    for model in &m.models {
        for dest in [root(model).join(&model.dest), bundled.join(&model.dest)] {
            if cleared.insert(dest.clone()) && dest.exists() {
                std::fs::remove_dir_all(&dest)?;
            }
        }
    }
    for model in &m.models {
        let fetched = fetch::fetch(&model.artifact)?;
        let dest = root(model).join(&model.dest);
        if model.ondemand {
            let (Some(file), Some(size)) = (&model.file, model.size) else {
                bail!("on-demand model '{}' needs `file` and `size`", model.id);
            };
            let got = std::fs::metadata(&fetched)?.len();
            if got != size {
                bail!("on-demand {file}: manifest says {size} bytes, download has {got}");
            }
        }
        if let Some(kind) = &model.convert {
            let stem = model
                .file
                .as_deref()
                .context("a converted model needs `file` (the output name)")?;
            convert::run(kind, &fetched, &dest, stem)
                .with_context(|| format!("converting {}", model.artifact.url))?;
            eprintln!(
                "  ✓ model {:<10} {stem}  (converted: {kind}, {})",
                model.id, model.license
            );
            continue;
        }
        if let Some(file) = &model.file {
            std::fs::create_dir_all(&dest)?;
            std::fs::copy(&fetched, dest.join(file))?;
            let how = if model.ondemand { ", on demand" } else { "" };
            eprintln!(
                "  ✓ model {:<10} {file}  ({}{how})",
                model.id, model.license
            );
            continue;
        }
        let n = archive::extract_zip_selected(&fetched, &dest, &model.select)?;
        if n == 0 {
            bail!(
                "model '{}': no archive entries matched {:?}",
                model.id,
                model.select
            );
        }
        eprintln!(
            "  ✓ model {:<10} {n:>3} files  ({})",
            model.id, model.license
        );
    }
    Ok(())
}

/// Where `cargo xtask models` puts on-demand models (tests and tools read
/// them here; the app downloads its own copy).
pub fn ondemand_root() -> std::path::PathBuf {
    util::root().join("target/ondemand-models")
}
