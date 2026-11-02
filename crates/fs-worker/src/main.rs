//! fs-worker: one process per (job, GPU). The Tauri core spawns it as a sidecar.
//!
//! Protocol (Phase 3): JSON-lines control messages on stdin (including the
//! license token, which the worker verifies independently) and JSON-lines
//! events on stdout. Frames never pass through these streams; they move over
//! dedicated pipes between this process and its FFmpeg children.
#![deny(unsafe_code)]

/// Bumped whenever the core ⇄ worker message schema changes incompatibly.
const PROTOCOL_VERSION: u32 = 1;

fn main() {
    // `--handshake` lets the core (and the bundling smoke test) verify that
    // the sidecar binary matches the app build before it trusts it with a job.
    if std::env::args().any(|a| a == "--handshake") {
        println!(
            r#"{{"event":"hello","protocol":{PROTOCOL_VERSION},"version":"{}"}}"#,
            env!("CARGO_PKG_VERSION")
        );
        return;
    }
    eprintln!("fs-worker is launched by FrameScale AI; run with --handshake to self-test.");
    std::process::exit(2);
}
