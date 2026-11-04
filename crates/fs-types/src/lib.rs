//! Shared domain model and IPC contract types.
//!
//! Every type that crosses a process boundary lives here: UI ⇄ core (Tauri IPC,
//! camelCase JSON) and core ⇄ fs-worker (JSON lines over stdio). Types derive
//! `ts_rs::TS`. `cargo xtask gen-types` exports them to
//! `packages/ipc-types/src/generated`, and CI fails if the output drifts.
#![forbid(unsafe_code)]

pub mod device;
pub mod job;
pub mod media;
pub mod models;
pub mod pipeline;
pub mod preview;
pub mod telemetry;
pub mod worker;

pub use device::{DeviceKind, DeviceUuid, GpuDevice, Vendor};
pub use job::{
    FailureKind, JobEvent, JobId, JobProgress, JobState, JobSummary, PauseMode, WarningCode,
};
pub use media::{
    AnalysisReport, ColorPrimaries, ColorProfile, ColorRange, ContentLightLevel, FieldOrder,
    MasteringDisplay, MatrixCoefficients, MediaInfo, Rational, TransferCharacteristic, VideoStream,
};
pub use models::{DownloadProgress, ModelStatus};
pub use pipeline::{
    Container, EncoderPreference, JobSpec, ModelRef, OutputSpec, PipelineError, PipelineSpec,
    StageConfig, StageKind, VideoCodec,
};
pub use preview::{CompareInfo, CompareSide, PreviewProgress, PreviewResult};
pub use telemetry::{BackendKind, DeviceTelemetry, EcoMode, EcoState, EngineStats, Telemetry};
