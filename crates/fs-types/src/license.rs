//! Entitlements derived from a verified license token (see `fs-license`).

use serde::{Deserialize, Serialize};
use ts_rs::TS;

use crate::pipeline::{JobSpec, StageKind, VideoCodec};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum Tier {
    Free,
    Pro,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum Feature {
    FaceFix,
    /// Interpolation factor above 2×.
    InterpolateHighFactor,
    ProResOutput,
    /// Output above 4K (3840×2160).
    OutputAbove4k,
    BatchReorder,
    /// More than `FREE_BATCH_LIMIT` queued jobs.
    BatchUnlimited,
    MultiGpu,
    Byom,
}

impl Feature {
    pub const ALL: [Feature; 8] = [
        Self::FaceFix,
        Self::InterpolateHighFactor,
        Self::ProResOutput,
        Self::OutputAbove4k,
        Self::BatchReorder,
        Self::BatchUnlimited,
        Self::MultiGpu,
        Self::Byom,
    ];
}

pub const FREE_BATCH_LIMIT: usize = 3;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct Entitlements {
    pub tier: Tier,
    pub features: Vec<Feature>,
    pub licensee: Option<String>,
    /// Unix seconds; `None` = perpetual.
    #[ts(type = "number | null")]
    pub expires_at: Option<u64>,
}

impl Entitlements {
    pub fn free() -> Self {
        Self {
            tier: Tier::Free,
            features: Vec::new(),
            licensee: None,
            expires_at: None,
        }
    }

    pub fn has(&self, f: Feature) -> bool {
        self.features.contains(&f)
    }

    /// Pro features `spec` would need that these entitlements lack.
    /// The worker calls this as the authoritative gate; the UI only mirrors it.
    pub fn missing_for(&self, spec: &JobSpec, out_w: u32, out_h: u32) -> Vec<Feature> {
        let mut need = Vec::new();
        if spec.pipeline.get(StageKind::FaceFix).is_some() {
            need.push(Feature::FaceFix);
        }
        if spec.pipeline.interpolation_factor() > 2 {
            need.push(Feature::InterpolateHighFactor);
        }
        if matches!(
            spec.output.codec,
            VideoCodec::ProRes422Hq | VideoCodec::ProRes4444
        ) {
            need.push(Feature::ProResOutput);
        }
        if u64::from(out_w) * u64::from(out_h) > 3840 * 2160 {
            need.push(Feature::OutputAbove4k);
        }
        if spec
            .pipeline
            .stages
            .iter()
            .filter_map(|s| s.model())
            .any(|m| m.path.is_some())
        {
            need.push(Feature::Byom);
        }
        need.retain(|f| !self.has(*f));
        need
    }
}
