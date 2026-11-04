//! Tuned tile sizes, remembered per device and models in a small JSON file
//! (`EnginePlan::tile_cache`): `{ "<version> | <profile>": tile }`.
//!
//! Best effort throughout: an unreadable file means tuning again, a failed
//! write means tuning again next time.

use std::path::Path;

use fs_engine::tiling::MIN_TILE;
use serde_json::{Map, Value};

/// Bump to discard every stored size when tiling itself changes.
const VERSION: &str = "v1";

fn key(profile: &str) -> String {
    format!("{VERSION} | {profile}")
}

fn read(path: &Path) -> Map<String, Value> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn lookup(path: &Path, profile: &str) -> Option<u32> {
    read(path)
        .get(&key(profile))?
        .as_u64()
        .and_then(|t| u32::try_from(t).ok())
        .filter(|t| (MIN_TILE..=4096).contains(t))
}

pub fn store(path: &Path, profile: &str, tile: u32) -> std::io::Result<()> {
    let mut m = read(path);
    m.insert(key(profile), tile.into());
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    // Write aside and rename: a preview and a render may finish at once.
    let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
    std::fs::write(&tmp, serde_json::to_vec_pretty(&Value::Object(m))?)?;
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remembers_per_profile_and_ignores_junk() {
        let dir = std::env::temp_dir().join(format!("fs-tune-{}", std::process::id()));
        let f = dir.join("sub").join("tile-tuning.json");
        assert_eq!(lookup(&f, "gpu | upscale:x"), None, "no file yet");
        store(&f, "gpu | upscale:x", 192).expect("store");
        store(&f, "gpu | upscale:y", 384).expect("store");
        assert_eq!(lookup(&f, "gpu | upscale:x"), Some(192));
        assert_eq!(lookup(&f, "gpu | upscale:y"), Some(384));
        assert_eq!(lookup(&f, "other gpu | upscale:x"), None);

        std::fs::write(&f, r#"{"v1 | a": 8, "v1 | b": "256", "v0 | c": 256}"#).expect("write");
        assert_eq!(lookup(&f, "a"), None, "below the minimum tile");
        assert_eq!(lookup(&f, "b"), None, "not a number");
        assert_eq!(lookup(&f, "c"), None, "another version's key");
        std::fs::write(&f, "not json").expect("write");
        assert_eq!(lookup(&f, "a"), None);
        store(&f, "a", 256).expect("a corrupt file is replaced");
        assert_eq!(lookup(&f, "a"), Some(256));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
