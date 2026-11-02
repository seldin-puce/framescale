//! ffprobe parsing, typed FFmpeg argv builders, render planning, and
//! progress/stderr parsing.
//!
//! FFmpeg is never linked; it always runs as a sidecar process, which keeps
//! the LGPL boundary clean. Command lines are built from typed plans and never
//! by string concatenation, so a hostile file name can't inject flags: every
//! path is passed as its own argv element.
#![forbid(unsafe_code)]

pub mod analyze;
pub mod argv;
pub mod classify;
pub mod color;
pub mod encoders;
pub mod mux;
pub mod plan;
pub mod probe;
pub mod progress;
pub mod rate;

pub use argv::Argv;
