//! Job orchestration for the desktop host. No Tauri dependency, so it's
//! testable headless.
//!
//! - [`runner::run_job`]: analyze → plan → supervised render → concat → mux → verify
//! - [`recovery`]: what to do when a worker exits (pure, unit-tested)
//! - [`storage`]: preflight + mid-render disk guard
//! - [`janitor`]: startup cleanup of abandoned temp dirs (never resumable ones)
#![deny(unsafe_code)]

pub mod janitor;
pub mod recovery;
pub mod runner;
pub mod storage;

pub use runner::{EventSink, JobCommand, Services, run_job};
