//! Bundled model extraction into `apps/desktop/src-tauri/resources/models/<dest>/`.

use anyhow::{Result, bail};

use crate::manifest::Manifest;
use crate::{archive, fetch, util};

pub fn run(m: &Manifest) -> Result<()> {
    let root = util::tauri_dir().join("resources/models");
    for model in &m.models {
        let zip = fetch::fetch(&model.artifact)?;
        let dest = root.join(&model.dest);
        if dest.exists() {
            std::fs::remove_dir_all(&dest)?;
        }
        let n = archive::extract_zip_selected(&zip, &dest, &model.select)?;
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
