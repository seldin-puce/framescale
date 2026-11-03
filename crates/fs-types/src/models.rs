//! Models the app downloads on demand (not in the installers).

use serde::{Deserialize, Serialize};
use ts_rs::TS;

/// Whether an on-demand model is on disk, and what downloading it involves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct ModelStatus {
    pub id: String,
    pub installed: bool,
    /// Download size of all its files.
    #[ts(type = "number")]
    pub bytes: u64,
    /// SPDX-ish license name, e.g. "S-Lab-1.0 (non-commercial)".
    pub license: String,
    /// The license text, for the consent dialog.
    pub license_url: String,
}

/// Download progress, streamed while a model downloads.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct DownloadProgress {
    #[ts(type = "number")]
    pub done: u64,
    #[ts(type = "number")]
    pub total: u64,
}
