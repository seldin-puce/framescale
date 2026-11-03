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
//! Backends:
//! - [`cpu::ReferenceBackend`]: nearest-neighbour scale + linear blend. **Not
//!   AI**: a deterministic reference that exercises the whole pipeline in tests
//!   and CI on machines without a GPU.
//! - [`ncnn`] (feature `ncnn`): Vulkan inference everywhere (MoltenVK on
//!   macOS), with ncnn's CPU path as a fallback. Real-ESRGAN + RIFE v4.6.
//! - `coreml`, `tensorrt`: later, behind the same trait.
#![deny(unsafe_code)]

pub mod cpu;
pub mod face;
pub mod frame;
pub mod interp;
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
    /// Largest tile worth trying on this device (start point for auto-tune).
    fn max_tile(&self) -> u32 {
        512
    }
}
