//! Typed view of `third_party/manifest.toml`.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::util;

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub ffmpeg: Ffmpeg,
    pub ncnn: Artifact,
    pub moltenvk: Artifact,
    pub models: Vec<Model>,
}

#[derive(Debug, Deserialize)]
pub struct Ffmpeg {
    pub gate: Gate,
    pub targets: BTreeMap<String, Artifact>,
    pub source: FfmpegSource,
}

#[derive(Debug, Deserialize)]
pub struct FfmpegSource {
    #[serde(flatten)]
    pub artifact: Artifact,
    pub deps: BTreeMap<String, Artifact>,
}

#[derive(Debug, Deserialize)]
pub struct Gate {
    pub forbidden_configure_flags: Vec<String>,
    pub required_filters: Vec<String>,
    pub required_encoders_common: Vec<String>,
    pub required_decoders_common: Vec<String>,
    pub encoders: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Artifact {
    pub url: String,
    pub sha256: String,
    /// Our own mirror; preferred over `url` when set.
    #[serde(default)]
    pub mirror: String,
}

impl Artifact {
    pub fn effective_url(&self) -> &str {
        if self.mirror.is_empty() {
            &self.url
        } else {
            &self.mirror
        }
    }

    /// File name used in the download cache (last URL path segment).
    pub fn file_name(&self) -> &str {
        self.url.rsplit('/').next().unwrap_or(&self.url)
    }
}

#[derive(Debug, Deserialize)]
pub struct Model {
    pub id: String,
    pub license: String,
    #[serde(flatten)]
    pub artifact: Artifact,
    pub select: Vec<String>,
    pub dest: String,
}

impl Manifest {
    pub fn load() -> Result<Self> {
        let path = util::manifest_path();
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }
}
