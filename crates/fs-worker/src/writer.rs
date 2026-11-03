//! Segment writer thread: feeds the current segment's encoder and commits
//! finished segments atomically (`.part` → fsync → rename).

use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender};
use fs_engine::Frame;
use fs_media::classify::FfmpegIssue;
use fs_types::FailureKind;
use fs_types::worker::{RenderPlan, SegmentInfo};

use crate::ffmpeg::{SegmentEncoder, StderrLog, durable_rename};

#[derive(Debug, Clone)]
pub struct PendingSegment {
    pub index: u32,
    pub first_frame: u64,
    pub frame_count: u64,
    pub out_frames: u64,
    pub in_hash: u64,
    pub next_hash: Option<u64>,
}

pub enum WriterMsg {
    Frame {
        segment: u32,
        frame: Frame,
    },
    Close(PendingSegment),
    /// Kill the in-progress encoder and discard its `.part` file.
    Abort,
}

#[derive(Debug)]
pub enum WriterResult {
    Committed(SegmentInfo),
    /// Preview mode: all frames written.
    PreviewDone,
    Failed {
        kind: FailureKind,
        message: String,
    },
}

fn failure_from(log: &StderrLog, fallback: &str) -> (FailureKind, String) {
    let kind = log
        .worst()
        .and_then(FfmpegIssue::failure_kind)
        .unwrap_or(FailureKind::Internal);
    (kind, format!("{fallback}: {}", log.summary()))
}

fn io_failure(e: &std::io::Error, what: &str) -> (FailureKind, String) {
    let kind = if e.kind() == std::io::ErrorKind::StorageFull {
        FailureKind::DiskFull
    } else {
        FailureKind::Io
    };
    (kind, format!("{what}: {e}"))
}

pub fn spawn(
    plan: Arc<RenderPlan>,
    rx: Receiver<WriterMsg>,
    results: Sender<WriterResult>,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let dir = PathBuf::from(&plan.segment_dir);
        let part_path = |i: u32| dir.join(format!("{}.part", plan.segment_file(i)));
        let mut cur: Option<(u32, SegmentEncoder)> = None;
        let mut failed = false;
        let fail = |kind, message| {
            let _ = results.send(WriterResult::Failed { kind, message });
        };

        for msg in rx {
            match msg {
                WriterMsg::Frame { segment, frame } => {
                    if failed {
                        continue;
                    }
                    if cur.as_ref().is_none_or(|(i, _)| *i != segment) {
                        if let Some((_, old)) = cur.take() {
                            old.kill(); // protocol violation: frames for a new segment without Close
                        }
                        // A crashed or killed earlier attempt can leave a `.part` behind.
                        // It's never committed data, so discard it before re-encoding.
                        let _ = std::fs::remove_file(part_path(segment));
                        let argv = fs_media::argv::encode_segment(
                            &plan.encode,
                            frame.width,
                            frame.height,
                            frame.format,
                            &part_path(segment),
                        );
                        match SegmentEncoder::spawn(&plan.ffmpeg, &argv) {
                            Ok(enc) => cur = Some((segment, enc)),
                            Err(e) => {
                                let (k, m) = io_failure(&e, "could not start encoder");
                                fail(k, m);
                                failed = true;
                                continue;
                            }
                        }
                    }
                    if let Some((_, enc)) = &mut cur
                        && enc.write(&frame.data).is_err()
                    {
                        // Broken pipe: the encoder died. Its stderr says why.
                        if let Some((_, enc)) = cur.take() {
                            let (_, log) = enc.finish();
                            let (k, m) = log.lock().map_or(
                                (FailureKind::Internal, "encoder exited".to_owned()),
                                |l| failure_from(&l, "encoder exited mid-segment"),
                            );
                            fail(k, m);
                        }
                        failed = true;
                    }
                }
                WriterMsg::Close(p) => {
                    if failed {
                        continue;
                    }
                    let Some((idx, enc)) = cur.take() else {
                        fail(
                            FailureKind::Internal,
                            format!("close of segment {} with no frames", p.index),
                        );
                        failed = true;
                        continue;
                    };
                    debug_assert_eq!(idx, p.index);
                    let (status, log) = enc.finish();
                    if !status.is_some_and(|s| s.success()) {
                        let (k, m) = log
                            .lock()
                            .map_or((FailureKind::Internal, "encoder failed".to_owned()), |l| {
                                failure_from(&l, "encoder failed")
                            });
                        fail(k, m);
                        failed = true;
                        continue;
                    }
                    let file_name = plan.segment_file(p.index);
                    let final_path = dir.join(&file_name);
                    match durable_rename(&part_path(p.index), &final_path)
                        .and_then(|()| std::fs::metadata(&final_path))
                    {
                        Ok(meta) => {
                            let _ = results.send(WriterResult::Committed(SegmentInfo {
                                index: p.index,
                                first_frame: p.first_frame,
                                frame_count: p.frame_count,
                                out_frames: p.out_frames,
                                in_hash: p.in_hash,
                                next_hash: p.next_hash,
                                file_name,
                                bytes: meta.len(),
                            }));
                        }
                        Err(e) => {
                            let (k, m) = io_failure(&e, "could not commit segment");
                            fail(k, m);
                            failed = true;
                        }
                    }
                }
                WriterMsg::Abort => {
                    if let Some((i, enc)) = cur.take() {
                        enc.kill();
                        let _ = std::fs::remove_file(part_path(i));
                    }
                    return;
                }
            }
        }
        if let Some((i, enc)) = cur.take() {
            enc.kill();
            let _ = std::fs::remove_file(part_path(i));
        }
    })
}

/// Preview mode: every output frame goes to one JPEG-sequence encoder.
/// Segment boundaries (`Close`) are irrelevant and ignored.
pub fn spawn_preview(
    plan: Arc<RenderPlan>,
    pattern: PathBuf,
    rx: Receiver<WriterMsg>,
    results: Sender<WriterResult>,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let mut enc: Option<SegmentEncoder> = None;
        let fail = |message: String| {
            let _ = results.send(WriterResult::Failed {
                kind: FailureKind::Internal,
                message,
            });
        };
        for msg in rx {
            match msg {
                WriterMsg::Frame { frame, .. } => {
                    if enc.is_none() {
                        let argv = fs_media::argv::jpeg_sequence(
                            frame.width,
                            frame.height,
                            frame.format,
                            &plan.encode.color,
                            &pattern,
                        );
                        match SegmentEncoder::spawn(&plan.ffmpeg, &argv) {
                            Ok(e) => enc = Some(e),
                            Err(e) => return fail(format!("could not start preview encoder: {e}")),
                        }
                    }
                    if let Some(e) = enc.as_mut()
                        && e.write(&frame.data).is_err()
                    {
                        return fail("preview encoder exited early".into());
                    }
                }
                WriterMsg::Close(_) => {}
                WriterMsg::Abort => {
                    if let Some(e) = enc.take() {
                        e.kill();
                    }
                    return;
                }
            }
        }
        if let Some(e) = enc.take() {
            let (status, log) = e.finish();
            if !status.is_some_and(|s| s.success()) {
                let why = log.lock().map(|l| l.summary()).unwrap_or_default();
                return fail(format!("preview encoder failed: {why}"));
            }
        }
        let _ = results.send(WriterResult::PreviewDone);
    })
}
