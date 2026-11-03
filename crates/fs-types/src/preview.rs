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
