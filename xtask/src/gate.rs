//! FFmpeg license & capability gate. Bundling fails if a build is GPL/nonfree
//! or is missing a filter or encoder the pipeline depends on.

use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

use anyhow::{Result, bail};

use crate::manifest::Gate;
use crate::util;

pub fn check(gate: &Gate, ffmpeg: &Path, triple: &str) -> Result<()> {
    let q = |arg: &str| util::output(Command::new(ffmpeg).args(["-hide_banner", arg]));

    let mut problems = Vec::new();

    let buildconf = q("-buildconf")?;
    let flags: BTreeSet<&str> = buildconf.split_whitespace().collect();
    for f in &gate.forbidden_configure_flags {
        if flags.contains(f.as_str()) {
            problems.push(format!("license: build is configured with {f}"));
        }
    }

    let filters = names(&q("-filters")?);
    let encoders = names(&q("-encoders")?);
    let decoders = names(&q("-decoders")?);

    let per_target = gate.encoders.get(triple).cloned().unwrap_or_default();
    let checks: [(&str, &BTreeSet<String>, Vec<&String>); 3] = [
        ("filter", &filters, gate.required_filters.iter().collect()),
        (
            "encoder",
            &encoders,
            gate.required_encoders_common
                .iter()
                .chain(per_target.iter())
                .collect(),
        ),
        (
            "decoder",
            &decoders,
            gate.required_decoders_common.iter().collect(),
        ),
    ];
    for (kind, have, want) in checks {
        for w in want {
            if !have.contains(w) {
                problems.push(format!("missing {kind}: {w}"));
            }
        }
    }

    if problems.is_empty() {
        eprintln!("  ✓ ffmpeg gate passed for {triple} ({})", ffmpeg.display());
        Ok(())
    } else {
        bail!(
            "ffmpeg gate FAILED for {triple}:\n  - {}",
            problems.join("\n  - ")
        )
    }
}

/// `ffmpeg -filters/-encoders/-decoders` rows look like ` V....D libsvtav1  desc…`:
/// the component name is the second whitespace-separated column.
fn names(listing: &str) -> BTreeSet<String> {
    listing
        .lines()
        .filter_map(|l| l.split_whitespace().nth(1))
        .map(str::to_owned)
        .collect()
}
