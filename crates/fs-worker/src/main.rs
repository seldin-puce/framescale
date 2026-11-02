//! fs-worker: one process per (job, GPU), spawned by the core as a sidecar.
//!
//! Protocol: JSON lines. `Start` and later control commands arrive on stdin;
//! events go out on stdout. Frames never pass through these streams; they move
//! over dedicated pipes between this process and its FFmpeg children.
#![deny(unsafe_code)]

mod control;
mod emit;
mod ffmpeg;
mod pipeline;
mod writer;

use std::io::BufRead;
use std::time::{SystemTime, UNIX_EPOCH};

use fs_types::FailureKind;
use fs_types::worker::{PROTOCOL_VERSION, WorkerCommand, WorkerEvent, exit};

use crate::emit::Emitter;
use crate::pipeline::Outcome;

fn main() {
    if std::env::args().any(|a| a == "--handshake") {
        println!(
            r#"{{"event":"hello","protocol":{PROTOCOL_VERSION},"version":"{}"}}"#,
            env!("CARGO_PKG_VERSION")
        );
        return;
    }
    std::process::exit(run());
}

fn fatal(emit: &Emitter, kind: FailureKind, message: impl Into<String>, code: i32) -> i32 {
    emit.send(&WorkerEvent::Error {
        kind,
        message: message.into(),
        fatal: true,
    });
    code
}

fn run() -> i32 {
    let emit = Emitter::new();
    emit.send(&WorkerEvent::Hello {
        protocol: PROTOCOL_VERSION,
        version: env!("CARGO_PKG_VERSION").into(),
    });

    // If the core dies, take our FFmpeg children down with us.
    fs_proc::watch_parent(|| {
        fs_proc::kill_own_group();
        std::process::exit(exit::FATAL);
    });

    let mut stdin = std::io::BufReader::new(std::io::stdin());
    let mut first = String::new();
    match stdin.read_line(&mut first) {
        Ok(0) => {
            eprintln!("fs-worker: stdin closed before Start");
            return exit::USAGE;
        }
        Ok(_) => {}
        Err(e) => {
            eprintln!("fs-worker: could not read Start: {e}");
            return exit::USAGE;
        }
    }
    let job = match serde_json::from_str::<WorkerCommand>(&first) {
        Ok(WorkerCommand::Start(job)) => *job,
        Ok(_) => return fatal(&emit, FailureKind::Internal, "expected Start", exit::USAGE),
        Err(e) => {
            return fatal(
                &emit,
                FailureKind::Internal,
                format!("bad Start: {e}"),
                exit::USAGE,
            );
        }
    };
    if job.protocol != PROTOCOL_VERSION {
        return fatal(
            &emit,
            FailureKind::Internal,
            format!(
                "protocol mismatch: core {} / worker {PROTOCOL_VERSION}",
                job.protocol
            ),
            exit::USAGE,
        );
    }

    // Authoritative license gate: verify the token ourselves, against the plan
    // we are about to execute.
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let verifier = fs_license::Verifier::embedded();
    let device = fs_license::machine_fingerprint();
    let (ent, _err) = verifier.resolve(job.license_token.as_deref(), now, device.as_deref());
    let missing: Vec<_> = fs_license::required_features(&job.plan)
        .into_iter()
        .filter(|f| !ent.has(*f))
        .collect();
    if !missing.is_empty() {
        return fatal(
            &emit,
            FailureKind::LicenseRequired,
            format!("Pro features required: {missing:?}"),
            exit::LICENSE,
        );
    }

    let ctrl = control::Control::new(job.duty);
    control::spawn_reader(ctrl.clone(), stdin);

    match pipeline::run(job, ctrl, &emit) {
        Ok(Outcome::Finished) => exit::OK,
        Ok(Outcome::Stopped { next_segment }) => {
            emit.send(&WorkerEvent::Stopped { next_segment });
            exit::STOPPED
        }
        Err(f) => {
            let code = f.exit_code();
            fatal(&emit, f.kind, f.message, code)
        }
    }
}
