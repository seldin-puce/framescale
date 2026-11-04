//! Core ⇄ fs-worker protocol: JSON lines on the worker's stdin/stdout.
//!
//! Frames never travel over these streams. They flow through dedicated pipes
//! between the worker and its FFmpeg children. These messages are Rust ⇄ Rust
//! only and are not exported to TypeScript.

use serde::{Deserialize, Serialize};

use crate::device::DeviceUuid;
use crate::job::{FailureKind, JobId, WarningCode};
use crate::media::{ColorProfile, Rational};
use crate::pipeline::StageConfig;
use crate::telemetry::{BackendKind, EngineStats};

/// 2: `Progress.engine`, `EnginePlan.backend`.
pub const PROTOCOL_VERSION: u32 = 2;

/// Worker process exit codes. The supervisor maps them to recovery actions.
pub mod exit {
    pub const OK: i32 = 0;
    pub const FATAL: i32 = 1;
    pub const USAGE: i32 = 2;
    pub const DEVICE_LOST: i32 = 3;
    pub const DISK_FULL: i32 = 4;
    /// Clean stop after a checkpoint (hard pause / stop request).
    pub const STOPPED: i32 = 5;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FrameFormat {
    /// 8-bit SDR sources.
    Rgb24,
    /// 10/12-bit and HDR sources (little-endian u16 per channel).
    Rgb48,
}

impl FrameFormat {
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgb24 => 3,
            Self::Rgb48 => 6,
        }
    }
    pub fn pix_fmt(self) -> &'static str {
        match self {
            Self::Rgb24 => "rgb24",
            Self::Rgb48 => "rgb48le",
        }
    }
    pub fn frame_bytes(self, w: u32, h: u32) -> usize {
        w as usize * h as usize * self.bytes_per_pixel()
    }
}

/// Hardware decode path. Decoded frames are always downloaded to system
/// memory, because the AI path consumes raw frames from a pipe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "api", rename_all = "camelCase")]
pub enum HwDecode {
    None,
    Cuda,
    D3d11va,
    Vaapi { device: String },
    Qsv,
    VideoToolbox,
}

/// How HDR static metadata (ST 2086 + CLL) survives encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum HdrEmbedding {
    /// SDR output: nothing to carry.
    NotApplicable,
    /// Written into the bitstream via encoder parameters (SVT-AV1).
    Bitstream,
    /// Transfer, primaries and matrix tags only (enough for displays to switch
    /// to HDR). Mastering and CLL values are not embedded.
    TagsOnly,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DecodePlan {
    pub input: String,
    pub stream_index: u32,
    pub hw: HwDecode,
    pub conform_rate: Rational,
    /// `Some(send_field)` → bwdif before the conform.
    pub deinterlace: Option<bool>,
    pub width: u32,
    pub height: u32,
    pub frame_format: FrameFormat,
    pub color: ColorProfile,
    /// Video start time minus container start time. Input `-ss` is relative
    /// to the container start, while frame 0 is the first video frame.
    pub seek_base_s: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EncodePlan {
    /// FFmpeg encoder name (`hevc_nvenc`, `libsvtav1`, …).
    pub encoder: String,
    /// Encoder-specific arguments (rate control, preset, HDR params).
    pub args: Vec<String>,
    /// Pixel format handed to the encoder after the RGB→YUV conversion.
    pub pix_fmt: String,
    /// VAAPI device for `hwupload`, when the encoder is `*_vaapi`.
    pub vaapi_device: Option<String>,
    pub width: u32,
    pub height: u32,
    pub rate: Rational,
    pub color: ColorProfile,
    pub hdr: HdrEmbedding,
    /// Pixel aspect of the source (DVD/DV/HDV are anamorphic). Every stage
    /// scales uniformly, so it carries over to the output unchanged.
    #[serde(default = "square_pixels")]
    pub sample_aspect: Rational,
}

fn square_pixels() -> Rational {
    Rational::new(1, 1)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnginePlan {
    pub stages: Vec<StageConfig>,
    /// Scene cuts in input (conformed) frame indices.
    pub scene_cuts: Vec<u64>,
    pub in_width: u32,
    pub in_height: u32,
    pub out_width: u32,
    pub out_height: u32,
    pub frame_format: FrameFormat,
    pub device: Option<DeviceUuid>,
    /// Initial tile edge in pixels; `None` lets the backend auto-tune.
    pub tile: Option<u32>,
    /// Bundled-model directory (resources/models).
    pub models_dir: String,
    /// Models downloaded on demand (CodeFormer: `ondemand` in
    /// third_party/manifest.toml), in the app's data folder.
    #[serde(default)]
    pub downloads_dir: Option<String>,
    /// JSON file of tuned tile sizes per device and models (app cache). The
    /// worker reads it when `tile` is `None`, and tunes and records the size
    /// when there's no entry yet. `None`: no tuning.
    #[serde(default)]
    pub tile_cache: Option<String>,
    /// Run on this backend instead of the detected one (user override).
    #[serde(default)]
    pub backend: Option<BackendKind>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderPlan {
    pub ffmpeg: String,
    pub segment_dir: String,
    pub decode: DecodePlan,
    pub encode: EncodePlan,
    pub engine: EnginePlan,
    /// Input-space frames after the CFR conform.
    pub total_frames: u64,
    /// Input-space frames per segment (4 s × conform rate).
    pub segment_frames: u64,
}

impl RenderPlan {
    pub fn segments_total(&self) -> u32 {
        self.total_frames.div_ceil(self.segment_frames.max(1)) as u32
    }

    pub fn segment_file(&self, index: u32) -> String {
        format!("seg_{index:05}.mkv")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StartPoint {
    pub segment: u32,
    pub first_frame: u64,
    /// Hash of the expected first input frame (resume alignment), if resuming.
    pub expected_hash: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerJob {
    pub protocol: u32,
    pub job: JobId,
    pub plan: RenderPlan,
    pub start: StartPoint,
    /// Governor duty cycle (0.1–1.0).
    pub duty: f32,
    /// Preview mode: render only `frames` input frames to JPEG sequences
    /// instead of encoded segments.
    #[serde(default)]
    pub preview: Option<PreviewRequest>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreviewRequest {
    /// Input (conformed) frames to process.
    pub frames: u64,
    /// Receives `before/%04d.jpg` and `after/%04d.jpg`.
    pub out_dir: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum WorkerCommand {
    Start(Box<WorkerJob>),
    Pause,
    Resume,
    /// `checkpoint: true` finishes the current segment first (hard pause).
    Stop {
        checkpoint: bool,
    },
    SetDuty {
        duty: f32,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SegmentInfo {
    pub index: u32,
    pub first_frame: u64,
    /// Input-space frames in this segment.
    pub frame_count: u64,
    pub out_frames: u64,
    pub in_hash: u64,
    /// Hash of the first input frame of the *next* segment (the frame whose
    /// arrival closed this one). Resume alignment for segment `index + 1`.
    pub next_hash: Option<u64>,
    pub file_name: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "event",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum WorkerEvent {
    Hello {
        protocol: u32,
        version: String,
    },
    Ready {
        backend: String,
        device_name: Option<String>,
        tile: Option<u32>,
    },
    Progress {
        frames_in: u64,
        frames_out: u64,
        fps: f32,
        segment: u32,
        #[serde(default)]
        engine: Option<EngineStats>,
    },
    SegmentCommitted(SegmentInfo),
    Warning {
        code: WarningCode,
        message: String,
    },
    Error {
        kind: FailureKind,
        message: String,
        fatal: bool,
    },
    Finished {
        frames_in: u64,
        frames_out: u64,
    },
    Stopped {
        next_segment: u32,
    },
}
