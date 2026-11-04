//! Jobs, their lifecycle, and the events the UI receives.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use ts_rs::TS;
use uuid::Uuid;

use crate::pipeline::JobSpec;
use crate::telemetry::EngineStats;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, TS)]
#[ts(export, type = "string")]
pub struct JobId(pub Uuid);

impl JobId {
    /// UUIDv7: time-sortable, so the journal's natural order is creation order.
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
}

impl Default for JobId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl FromStr for JobId {
    type Err = uuid::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Uuid::parse_str(s).map(Self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum JobState {
    Queued,
    Analyzing,
    Rendering,
    Paused,
    /// The app or worker died mid-render; resumable from the journal.
    Interrupted,
    Concatenating,
    Muxing,
    Done,
    Failed,
    Cancelled,
}

impl JobState {
    pub const ALL: [JobState; 10] = [
        Self::Queued,
        Self::Analyzing,
        Self::Rendering,
        Self::Paused,
        Self::Interrupted,
        Self::Concatenating,
        Self::Muxing,
        Self::Done,
        Self::Failed,
        Self::Cancelled,
    ];

    /// Journal (SQL) representation; matches the CHECK constraint.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Analyzing => "analyzing",
            Self::Rendering => "rendering",
            Self::Paused => "paused",
            Self::Interrupted => "interrupted",
            Self::Concatenating => "concatenating",
            Self::Muxing => "muxing",
            Self::Done => "done",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|v| v.as_str() == s)
    }

    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Done | Self::Failed | Self::Cancelled)
    }

    /// States in which a process may be running for this job.
    pub fn is_active(self) -> bool {
        matches!(
            self,
            Self::Analyzing | Self::Rendering | Self::Concatenating | Self::Muxing
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum PauseMode {
    /// Backpressure: the worker stops reading frames. Instant resume, VRAM stays held.
    Soft,
    /// Checkpoint: finish the current segment, exit, free the GPU.
    Hard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum FailureKind {
    VramOom,
    DeviceLost,
    DiskFull,
    EncoderUnavailable,
    DecodeFailed,
    WorkerCrashed,
    InvalidInput,
    Io,
    Internal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum WarningCode {
    /// Dolby Vision RPU / HDR10+ dynamic metadata is invalid after resizing.
    DynamicHdrStripped,
    /// The encoder carries PQ/HLG + BT.2020 tags but not ST 2086 / CLL.
    HdrStaticMetadataTagsOnly,
    VfrConformed,
    AudioTranscoded,
    SubtitlesDropped,
    EncoderFallback,
    HwDecodeFallback,
    CorruptPackets,
    ThermalThrottled,
    TileReduced,
    /// The preferred inference backend couldn't start; the next one runs.
    BackendFallback,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct JobProgress {
    pub job: JobId,
    /// Input-space (conformed) frames processed.
    #[ts(type = "number")]
    pub frames_done: u64,
    #[ts(type = "number")]
    pub frames_total: u64,
    pub fps: f32,
    pub eta_s: Option<f64>,
    pub segment: u32,
    pub segments_total: u32,
    /// Backend, tile size and tile reuse, when the engine reports them.
    pub engine: Option<EngineStats>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
#[ts(export)]
pub enum JobEvent {
    Progress(JobProgress),
    StateChanged {
        job: JobId,
        state: JobState,
        reason: Option<String>,
    },
    Warning {
        job: JobId,
        code: WarningCode,
        message: String,
    },
    Error {
        job: JobId,
        kind: FailureKind,
        message: String,
    },
}

/// Row shown in the queue.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct JobSummary {
    pub id: JobId,
    pub spec: JobSpec,
    pub state: JobState,
    pub queue_pos: u32,
    #[ts(type = "number")]
    pub frames_done: u64,
    #[ts(type = "number | null")]
    pub frames_total: Option<u64>,
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_strings_roundtrip() {
        for s in JobState::ALL {
            assert_eq!(JobState::parse(s.as_str()), Some(s));
        }
    }
}
