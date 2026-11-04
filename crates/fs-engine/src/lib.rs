//! Inference engine: the [`InferenceBackend`] trait the worker drives, plus
//! the backend-independent machinery around it.
//!
//! - [`tiling::TileController`]: VRAM-OOM recovery by halving the tile and
//!   retrying the *same* frame. Tiles also bound per-dispatch time, which
//!   keeps Windows TDR (2 s GPU watchdog) from firing.
//! - [`interp::Scheduler`]: frame-rate multiplication that never morphs across
//!   a hard cut (scdet table + a pixel-difference backstop).
//! - [`frame::hash`]: sampled xxh3 used for frame-exact resume alignment.
//!
//! - [`engine::NetEngine`]: one network on one device, submit/wait with up to
//!   `max_in_flight` tiles outstanding; [`planar::run_tiled`] drives it.
//!   [`engine::ThreadedEngine`] runs any blocking per-tile call that way.
//!
//! Backends:
//! - [`cpu::ReferenceBackend`]: nearest-neighbour scale + linear blend. **Not
//!   AI**: a deterministic reference that exercises the whole pipeline in tests
//!   and CI on machines without a GPU.
//! - [`mock::MockBackend`]: the real tiler and scheduler with a fake network
//!   (`FS_ENGINE=mock`); equals the reference pixel for pixel.
//! - [`ncnn`] (feature `ncnn`, alias `backend-ncnn`): Vulkan inference
//!   everywhere (MoltenVK on macOS), with ncnn's CPU path as a fallback.
//! - [`apple`] (feature `backend-apple`, macOS on Apple Silicon): Core ML for
//!   the pipelines a converted model exists for (scripts/convert_coreml.py).
//! - TensorRT (`backend-tensorrt`): not implemented yet; docs/engine-audit.md
//!   has the plan.
#![deny(unsafe_code)]

#[cfg(all(
    feature = "backend-apple",
    target_os = "macos",
    target_arch = "aarch64"
))]
pub mod apple;
pub mod cpu;
pub mod engine;
pub mod face;
pub mod frame;
pub mod half;
pub mod interp;
pub mod mock;
#[cfg(feature = "ncnn")]
pub mod ncnn;
pub mod planar;
pub mod tiling;

pub use frame::Frame;
use fs_types::FailureKind;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum EngineError {
    #[error("out of device memory at tile {tile}px")]
    OutOfDeviceMemory { tile: u32 },
    #[error("GPU device lost: {0}")]
    DeviceLost(String),
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("{0}")]
    Internal(String),
}

impl EngineError {
    pub fn failure_kind(&self) -> FailureKind {
        match self {
            Self::OutOfDeviceMemory { .. } => FailureKind::VramOom,
            Self::DeviceLost(_) => FailureKind::DeviceLost,
            Self::ModelNotFound(_) | Self::Unsupported(_) => FailureKind::InvalidInput,
            Self::Internal(_) => FailureKind::Internal,
        }
    }
}

/// One backend instance owns one device and the loaded stage models.
pub trait InferenceBackend: Send {
    fn name(&self) -> &'static str;
    fn device_name(&self) -> Option<String>;
    /// Run every spatial stage (denoise → upscale → face fix) on one frame.
    fn process(&mut self, input: &Frame, tile: u32) -> Result<Frame, EngineError>;
    /// Synthesize the frame at `t ∈ (0, 1)` between two processed frames.
    fn interpolate(
        &mut self,
        a: &Frame,
        b: &Frame,
        t: f32,
        tile: u32,
    ) -> Result<Frame, EngineError>;
    /// Default tile for this device and these models; the start point for
    /// [`tiling::TileTuner`].
    fn max_tile(&self) -> u32 {
        512
    }
    /// What a tuned tile size depends on (device and model architectures),
    /// as a key for remembering it. `None`: don't tune (CPU, reference).
    fn tune_profile(&self) -> Option<String> {
        None
    }
    /// Carry unchanged tiles over from the previous frame (on by default
    /// where supported). Off while tile sizes are being timed.
    fn set_reuse(&mut self, _on: bool) {}
    /// One line about the run so far, for the log.
    fn report(&self) -> Option<String> {
        None
    }
    /// Backend, tile and reuse figures for the progress reports.
    fn stats(&self) -> engine::EngineStats {
        engine::EngineStats::default()
    }
}
