//! Bundled model extraction into `apps/desktop/src-tauri/resources/models/<dest>/`.

use anyhow::{Result, bail};

use crate::manifest::Manifest;
use crate::{archive, fetch, util};

pub fn run(m: &Manifest) -> Result<()> {
    let root = util::tauri_dir().join("resources/models");
    // Several entries may share a destination (single-file artifacts): clear
    // each destination once, before anything is written into it.
    let mut cleared = std::collections::HashSet::new();
    for model in &m.models {
        let dest = root.join(&model.dest);
        if cleared.insert(dest.clone()) && dest.exists() {
            std::fs::remove_dir_all(&dest)?;
        }
    }
    for model in &m.models {
        let fetched = fetch::fetch(&model.artifact)?;
        let dest = root.join(&model.dest);
        if let Some(file) = &model.file {
            std::fs::create_dir_all(&dest)?;
            std::fs::copy(&fetched, dest.join(file))?;
            eprintln!("  ✓ model {:<10} {file}  ({})", model.id, model.license);
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
