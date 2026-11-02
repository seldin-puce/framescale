//! `cargo xtask <command>`: FrameScale build tooling.
//!
//! Every external artifact comes from `third_party/manifest.toml` and is
//! verified by SHA-256. Downloads use the system `curl` (present on all CI
//! runners and on Windows 10+), so this tool needs no TLS stack or C toolchain.

mod archive;
mod fetch;
mod gate;
mod manifest;
mod models;
mod pin;
mod sidecars;
mod util;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "xtask", about = "FrameScale AI build tooling")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Everything a fresh checkout needs: sidecars + bundled models.
    Setup(TargetArgs),
    /// Assemble FFmpeg/ffprobe/fs-worker sidecars (+ MoltenVK on macOS) for a target.
    Sidecars {
        #[command(flatten)]
        target: TargetArgs,
        /// Write empty placeholder sidecars (CI lint jobs: tauri-build only checks they exist).
        #[arg(long)]
        stub: bool,
    },
    /// Download and extract the bundled, freely-licensed models.
    Models,
    /// Fill in empty `sha256 = ""` entries in the manifest (trust-on-first-pin).
    Pin,
    /// Run the FFmpeg license/capability gate against an arbitrary ffmpeg binary.
    Gate {
        /// Path to the ffmpeg executable to check.
        ffmpeg: std::path::PathBuf,
        #[command(flatten)]
        target: TargetArgs,
    },
    /// Export ts-rs bindings from fs-types into packages/ipc-types.
    GenTypes,
    /// Check local prerequisites for building the desktop app.
    Doctor,
}

#[derive(Debug, Args)]
struct TargetArgs {
    /// Rust target triple, or `universal-apple-darwin`. Defaults to the host.
    #[arg(long)]
    target: Option<String>,
}

impl TargetArgs {
    fn resolve(&self) -> Result<String> {
        match &self.target {
            Some(t) => Ok(t.clone()),
            None => util::host_triple(),
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let m = manifest::Manifest::load()?;
    match cli.cmd {
        Cmd::Setup(t) => {
            sidecars::run(&m, &t.resolve()?, false)?;
            models::run(&m)
        }
        Cmd::Sidecars { target, stub } => sidecars::run(&m, &target.resolve()?, stub),
        Cmd::Models => models::run(&m),
        Cmd::Pin => pin::run(),
        Cmd::Gate { ffmpeg, target } => gate::check(&m.ffmpeg.gate, &ffmpeg, &target.resolve()?),
        Cmd::GenTypes => util::cargo(&["test", "--quiet", "-p", "fs-types", "export_bindings"]),
        Cmd::Doctor => util::doctor(),
    }
}
