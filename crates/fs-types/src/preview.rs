//! 2-second preview results (served to the UI through the `fs://` protocol).

use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct PreviewResult {
    /// Protocol path prefix, e.g. `preview/<id>`; frames live at
    /// `<prefix>/before/0000.jpg` and `<prefix>/after/0000.jpg`.
    pub prefix: String,
    pub before_frames: u32,
    pub after_frames: u32,
    pub width: u32,
    pub height: u32,
    pub source_width: u32,
    pub source_height: u32,
    /// Output frame rate (after interpolation).
    pub fps: f64,
    /// Seconds into the source where the preview starts.
    pub start_s: f64,
    pub backend: String,
    pub device: Option<String>,
    pub elapsed_ms: u32,
    /// True when the source is HDR and the preview was tone-mapped for display.
    pub tone_mapped: bool,
}

/// Which video a comparison frame comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum CompareSide {
    /// The original source file.
    Before,
    /// The rendered output.
    After,
}

/// A source-vs-output comparison of a finished render. Both sides are
/// sampled on the output's frame grid (frame n = n / fps), so they stay
/// aligned even when the render interpolated to a higher rate.
///
/// Frames are served at `compare/<session>/<side>/<n>.jpg` (playback size)
/// and `compare/<session>/<side>/full/<n>.jpg` (native resolution).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct CompareInfo {
    pub session: String,
    pub duration_s: f64,
    /// Output frame rate: the comparison's clock.
    pub fps: f64,
    #[ts(type = "number")]
    pub frames: u64,
    /// Playback frame size (both sides).
    pub width: u32,
    pub height: u32,
    pub source_width: u32,
    pub source_height: u32,
    pub source_fps: f64,
    pub output_width: u32,
    pub output_height: u32,
    pub source_path: String,
    pub output_path: String,
    /// HDR sources/outputs are tone-mapped for display only.
    pub tone_mapped: bool,
}
