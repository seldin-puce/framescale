//! The user's processing request: stage chain + output settings.

use serde::{Deserialize, Serialize};
use ts_rs::TS;

use crate::device::DeviceUuid;

/// Stages in their only valid order. Deinterlace must precede the CFR conform
/// (bwdif doubles the rate). Interpolation runs last because it is the
/// cheapest stage per pixel, so running it before upscaling would double the
/// most expensive stage's work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum StageKind {
    Deinterlace,
    Denoise,
    Upscale,
    FaceFix,
    Interpolate,
}

impl StageKind {
    pub const ORDER: [StageKind; 5] = [
        Self::Deinterlace,
        Self::Denoise,
        Self::Upscale,
        Self::FaceFix,
        Self::Interpolate,
    ];
}

/// A model to run: a bundled id, or a user-supplied file (BYOM).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct ModelRef {
    /// Bundled model id (e.g. `realesrgan-x4plus`) or registry id for BYOM.
    pub id: String,
    /// BYOM: absolute path to the user's `.param`/`.onnx` file. Never bundled.
    pub path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
#[ts(export)]
pub enum StageConfig {
    /// FFmpeg `bwdif` (motion-adaptive). Runs in the decoder, not on the GPU path.
    Deinterlace {
        send_field: bool,
    },
    Denoise {
        model: ModelRef,
        strength: u8,
    },
    Upscale {
        model: ModelRef,
        scale: u8,
        /// Noise removal 0–1 for models that blend two checkpoints
        /// (realesr-general-x4v3: 0 keeps grain, 1 removes the most). `None`:
        /// the model's default.
        #[serde(default)]
        denoise: Option<f32>,
    },
    FaceFix {
        model: ModelRef,
        fidelity: f32,
    },
    Interpolate {
        model: ModelRef,
        factor: u8,
        scene_threshold: f32,
    },
}

impl StageConfig {
    pub fn kind(&self) -> StageKind {
        match self {
            Self::Deinterlace { .. } => StageKind::Deinterlace,
            Self::Denoise { .. } => StageKind::Denoise,
            Self::Upscale { .. } => StageKind::Upscale,
            Self::FaceFix { .. } => StageKind::FaceFix,
            Self::Interpolate { .. } => StageKind::Interpolate,
        }
    }

    pub fn model(&self) -> Option<&ModelRef> {
        match self {
            Self::Deinterlace { .. } => None,
            Self::Denoise { model, .. }
            | Self::Upscale { model, .. }
            | Self::FaceFix { model, .. }
            | Self::Interpolate { model, .. } => Some(model),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(
    tag = "code",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
#[ts(export)]
pub enum PipelineError {
    OutOfOrder { stage: StageKind },
    Duplicate { stage: StageKind },
    InvalidParameter { stage: StageKind, message: String },
    Empty,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct PipelineSpec {
    pub stages: Vec<StageConfig>,
}

impl PipelineSpec {
    /// Validate the chain. Returns every problem, not just the first.
    pub fn validate(&self) -> Result<(), Vec<PipelineError>> {
        let mut errs = Vec::new();
        if self.stages.is_empty() {
            errs.push(PipelineError::Empty);
        }
        let mut last: Option<StageKind> = None;
        for s in &self.stages {
            let k = s.kind();
            match last {
                Some(prev) if prev == k => errs.push(PipelineError::Duplicate { stage: k }),
                Some(prev) if prev > k => errs.push(PipelineError::OutOfOrder { stage: k }),
                _ => {}
            }
            last = Some(k);
            let bad = |m: &str| PipelineError::InvalidParameter {
                stage: k,
                message: m.into(),
            };
            match s {
                StageConfig::Denoise { strength, .. } if *strength > 3 => {
                    errs.push(bad("strength must be 0–3"))
                }
                StageConfig::Upscale { scale, .. } if !(2..=4).contains(scale) => {
                    errs.push(bad("scale must be 2, 3 or 4"))
                }
                StageConfig::Upscale {
                    denoise: Some(d), ..
                } if !(0.0..=1.0).contains(d) => errs.push(bad("denoise must be between 0 and 1")),
                StageConfig::FaceFix { fidelity, .. } if !(0.0..=1.0).contains(fidelity) => {
                    errs.push(bad("fidelity must be within 0–1"))
                }
                StageConfig::Interpolate {
                    factor,
                    scene_threshold,
                    ..
                } => {
                    if !(2..=4).contains(factor) {
                        errs.push(bad("factor must be 2, 3 or 4"));
                    }
                    if !(1.0..=100.0).contains(scene_threshold) {
                        errs.push(bad("scene threshold must be within 1–100"));
                    }
                }
                _ => {}
            }
        }
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }

    pub fn get(&self, kind: StageKind) -> Option<&StageConfig> {
        self.stages.iter().find(|s| s.kind() == kind)
    }

    pub fn scale_factor(&self) -> u32 {
        match self.get(StageKind::Upscale) {
            Some(StageConfig::Upscale { scale, .. }) => u32::from(*scale),
            _ => 1,
        }
    }

    pub fn interpolation_factor(&self) -> u32 {
        match self.get(StageKind::Interpolate) {
            Some(StageConfig::Interpolate { factor, .. }) => u32::from(*factor),
            _ => 1,
        }
    }

    pub fn deinterlace(&self) -> Option<bool> {
        match self.get(StageKind::Deinterlace) {
            Some(StageConfig::Deinterlace { send_field }) => Some(*send_field),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum VideoCodec {
    /// HEVC Main10. The LGPL FFmpeg build has no software HEVC encoder, so a
    /// hardware encoder is required.
    Hevc10,
    /// AV1 10-bit: hardware where available, SVT-AV1 otherwise.
    Av1,
    /// H.264 8-bit (hardware only in the LGPL build). For compatibility exports.
    H264,
    /// Apple ProRes 422 HQ (macOS VideoToolbox only).
    ProRes422Hq,
    /// Apple ProRes 4444 (macOS VideoToolbox only).
    ProRes4444,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum Container {
    Mp4,
    Mov,
    Mkv,
}

impl Container {
    pub fn extension(self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Mov => "mov",
            Self::Mkv => "mkv",
        }
    }
    pub fn muxer(self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Mov => "mov",
            Self::Mkv => "matroska",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum EncoderPreference {
    #[default]
    Auto,
    Hardware,
    Software,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct OutputSpec {
    pub path: String,
    pub codec: VideoCodec,
    pub container: Container,
    /// Constant-quality value on a CRF-like 0–51 scale (lower = better).
    pub quality: u8,
    pub encoder: EncoderPreference,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct JobSpec {
    pub input: String,
    pub pipeline: PipelineSpec,
    pub output: OutputSpec,
    /// Explicit device; `None` → the ranked default.
    pub device: Option<DeviceUuid>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m() -> ModelRef {
        ModelRef {
            id: "x".into(),
            path: None,
        }
    }

    #[test]
    fn validates_order_duplicates_and_params() {
        let ok = PipelineSpec {
            stages: vec![
                StageConfig::Upscale {
                    model: m(),
                    scale: 4,
                    denoise: None,
                },
                StageConfig::Interpolate {
                    model: m(),
                    factor: 2,
                    scene_threshold: 10.0,
                },
            ],
        };
        assert!(ok.validate().is_ok());
        assert_eq!(ok.scale_factor(), 4);
        assert_eq!(ok.interpolation_factor(), 2);

        let bad = PipelineSpec {
            stages: vec![
                StageConfig::Interpolate {
                    model: m(),
                    factor: 5,
                    scene_threshold: 10.0,
                },
                StageConfig::Upscale {
                    model: m(),
                    scale: 4,
                    denoise: None,
                },
                StageConfig::Upscale {
                    model: m(),
                    scale: 4,
                    denoise: None,
                },
            ],
        };
        let errs = bad.validate().expect_err("invalid");
        assert!(errs.contains(&PipelineError::OutOfOrder {
            stage: StageKind::Upscale
        }));
        assert!(errs.contains(&PipelineError::Duplicate {
            stage: StageKind::Upscale
        }));
        assert!(
            errs.iter()
                .any(|e| matches!(e, PipelineError::InvalidParameter { .. }))
        );
    }

    #[test]
    fn stage_json_shape_is_tagged_camel_case() {
        let s = StageConfig::Interpolate {
            model: m(),
            factor: 2,
            scene_threshold: 10.0,
        };
        let j = serde_json::to_string(&s).expect("json");
        assert!(
            j.contains(r#""kind":"interpolate""#) && j.contains("sceneThreshold"),
            "{j}"
        );
    }
}
