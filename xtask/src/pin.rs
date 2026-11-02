//! Trust-on-first-pin: download each unpinned artifact once and write its
//! SHA-256 into the manifest. The manifest is edited as text so comments and
//! layout survive. Review the resulting diff before you commit it.

use anyhow::{Result, bail};

use crate::manifest::{Artifact, Manifest};
use crate::{fetch, util};

/// Byte range of the first `sha256<ws>=<ws>""` at or after `from`. Tolerates
/// column-aligned whitespace; skips keys that already hold a digest.
fn find_empty_pin(text: &str, from: usize) -> Option<(usize, usize)> {
    let mut pos = from;
    while let Some(rel) = text[pos..].find("sha256") {
        let start = pos + rel;
        let rest = text[start + "sha256".len()..].trim_start_matches([' ', '\t']);
        if let Some(rest) = rest.strip_prefix('=') {
            let rest = rest.trim_start_matches([' ', '\t']);
            if rest.starts_with("\"\"") {
                let end = text.len() - rest.len() + 2;
                return Some((start, end));
            }
        }
        pos = start + "sha256".len();
    }
    None
}

pub fn run() -> Result<()> {
    let m = Manifest::load()?;
    let mut unpinned: Vec<&Artifact> = Vec::new();
    unpinned.extend(m.ffmpeg.targets.values());
    unpinned.push(&m.ffmpeg.source.artifact);
    unpinned.extend(m.ffmpeg.source.deps.values());
    unpinned.push(&m.ncnn);
    unpinned.push(&m.moltenvk);
    unpinned.extend(m.models.iter().map(|x| &x.artifact));
    unpinned.retain(|a| a.sha256.is_empty());

    if unpinned.is_empty() {
        eprintln!("all artifacts pinned");
        return Ok(());
    }

    let path = util::manifest_path();
    let mut text = std::fs::read_to_string(&path)?;
    for a in unpinned {
        let file = fetch::download(a)?;
        let digest = fetch::sha256_file(&file)?;
        // Replace the first empty pin that follows this artifact's URL.
        let Some(at) = text.find(&format!("\"{}\"", a.url)) else {
            bail!("URL not found verbatim in manifest: {}", a.url);
        };
        let Some((start, end)) = find_empty_pin(&text, at) else {
            bail!("no empty `sha256 = \"\"` after {}", a.url);
        };
        text.replace_range(start..end, &format!("sha256 = \"{digest}\""));
        eprintln!("  pinned {digest}  {}", a.file_name());
    }
    std::fs::write(&path, text)?;
    eprintln!("manifest updated; review with `git diff third_party/manifest.toml`");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::find_empty_pin;

    #[test]
    fn finds_aligned_and_inline_pins() {
        let t = "url = \"a\"\nsha256  = \"\"\nb = { url = \"b\", sha256 = \"\" }";
        let (s, e) = find_empty_pin(t, 0).expect("aligned");
        assert_eq!(&t[s..e], "sha256  = \"\"");
        let (s, e) = find_empty_pin(t, e).expect("inline");
        assert_eq!(&t[s..e], "sha256 = \"\"");
    }

    #[test]
    fn skips_existing_digest() {
        let t = "sha256 = \"abc\"\nsha256 = \"\"";
        let (s, _) = find_empty_pin(t, 0).expect("second");
        assert_eq!(s, 15);
    }
}
