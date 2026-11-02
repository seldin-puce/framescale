//! FFmpeg child processes owned by the worker: decoder and segment encoders.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, bounded};
use fs_engine::Frame;
use fs_media::Argv;
use fs_media::classify::{FfmpegIssue, classify};
use fs_types::worker::FrameFormat;

use crate::control::Control;

/// Last stderr lines + classified issues, shared with the stderr thread.
#[derive(Debug, Default)]
pub struct StderrLog {
    pub tail: VecDeque<String>,
    pub issues: Vec<FfmpegIssue>,
    pub corrupt: u64,
}

impl StderrLog {
    pub fn worst(&self) -> Option<FfmpegIssue> {
        // Most actionable first.
        use FfmpegIssue::*;
        [
            NoSpace,
            EncoderSessionLimit,
            EncoderUnavailable,
            HwDecodeFailed,
            InvalidInput,
            PermissionDenied,
            OutOfMemory,
        ]
        .into_iter()
        .find(|i| self.issues.contains(i))
    }
    pub fn summary(&self) -> String {
        self.tail
            .iter()
            .rev()
            .take(6)
            .rev()
            .cloned()
            .collect::<Vec<_>>()
            .join(" | ")
    }
}

/// FFmpeg children stay in the worker's process group (no `fs_proc::prepare`),
/// so killing the worker's group takes them down too.
fn command(ffmpeg: &str, argv: &Argv) -> Command {
    let mut c = Command::new(ffmpeg);
    c.args(argv.as_slice());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    c
}

fn spawn_stderr(child: &mut Child) -> (Arc<Mutex<StderrLog>>, Option<JoinHandle<()>>) {
    let log = Arc::new(Mutex::new(StderrLog::default()));
    let handle = child.stderr.take().map(|err| {
        let log = log.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(err).lines().map_while(Result::ok) {
                let Ok(mut l) = log.lock() else { return };
                match classify(&line) {
                    Some(FfmpegIssue::CorruptInput) => l.corrupt += 1,
                    Some(i) if !l.issues.contains(&i) => l.issues.push(i),
                    _ => {}
                }
                l.tail.push_back(line);
                if l.tail.len() > 50 {
                    l.tail.pop_front();
                }
            }
        })
    });
    (log, handle)
}

pub enum DecodeMsg {
    Frame(Frame),
    Eof,
    Error(String),
}

pub struct Decoder {
    child: Child,
    rx: Receiver<DecodeMsg>,
    stderr: Arc<Mutex<StderrLog>>,
    stderr_thread: Option<JoinHandle<()>>,
}

/// Read until `buf` is full or EOF; returns bytes read.
fn read_full(r: &mut impl Read, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        match r.read(&mut buf[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(n)
}

impl Decoder {
    pub fn spawn(
        ffmpeg: &str,
        argv: &Argv,
        width: u32,
        height: u32,
        format: FrameFormat,
        ctrl: Arc<Control>,
    ) -> std::io::Result<Self> {
        let mut child = command(ffmpeg, argv)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let (stderr, stderr_thread) = spawn_stderr(&mut child);
        let mut out = child
            .stdout
            .take()
            .ok_or_else(|| std::io::Error::other("no stdout"))?;
        // Small bound: backpressure reaches FFmpeg quickly (memory ≈ 3 frames).
        let (tx, rx) = bounded(3);
        let frame_bytes = format.frame_bytes(width, height);
        std::thread::Builder::new()
            .name("decode-reader".into())
            .spawn(move || {
                loop {
                    ctrl.wait_if_paused();
                    let mut buf = vec![0u8; frame_bytes];
                    let msg = match read_full(&mut out, &mut buf) {
                        Ok(n) if n == frame_bytes => {
                            DecodeMsg::Frame(Frame::new(width, height, format, buf))
                        }
                        Ok(0) => DecodeMsg::Eof,
                        Ok(n) => DecodeMsg::Error(format!(
                            "decoder produced a partial frame ({n}/{frame_bytes} bytes)"
                        )),
                        Err(e) => DecodeMsg::Error(e.to_string()),
                    };
                    let last = !matches!(msg, DecodeMsg::Frame(_));
                    if tx.send(msg).is_err() || last {
                        return;
                    }
                }
            })?;
        Ok(Self {
            child,
            rx,
            stderr,
            stderr_thread,
        })
    }

    pub fn recv(&self) -> DecodeMsg {
        self.rx.recv().unwrap_or(DecodeMsg::Eof)
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    pub fn finish(mut self) -> (Option<ExitStatus>, Arc<Mutex<StderrLog>>) {
        let status = self.child.wait().ok();
        if let Some(t) = self.stderr_thread.take() {
            let _ = t.join();
        }
        (status, self.stderr.clone())
    }
}

/// One segment's encoder process.
pub struct SegmentEncoder {
    child: Child,
    stdin: Option<ChildStdin>,
    stderr: Arc<Mutex<StderrLog>>,
    stderr_thread: Option<JoinHandle<()>>,
}

impl SegmentEncoder {
    pub fn spawn(ffmpeg: &str, argv: &Argv) -> std::io::Result<Self> {
        let mut child = command(ffmpeg, argv)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()?;
        let stdin = child.stdin.take();
        let (stderr, stderr_thread) = spawn_stderr(&mut child);
        Ok(Self {
            child,
            stdin,
            stderr,
            stderr_thread,
        })
    }

    pub fn write(&mut self, data: &[u8]) -> std::io::Result<()> {
        use std::io::Write;
        match &mut self.stdin {
            Some(s) => s.write_all(data),
            None => Err(std::io::Error::other("encoder stdin closed")),
        }
    }

    /// Close stdin (EOF → the encoder flushes and writes the trailer) and wait.
    pub fn finish(mut self) -> (Option<ExitStatus>, Arc<Mutex<StderrLog>>) {
        drop(self.stdin.take());
        let status = self.child.wait().ok();
        if let Some(t) = self.stderr_thread.take() {
            let _ = t.join();
        }
        (status, self.stderr.clone())
    }

    pub fn kill(mut self) {
        drop(self.stdin.take());
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// fsync a file and (on Unix) its directory, so a committed rename survives power loss.
pub fn durable_rename(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::File::open(from)?.sync_all()?;
    std::fs::rename(from, to)?;
    #[cfg(unix)]
    if let Some(dir) = to.parent() {
        std::fs::File::open(dir)?.sync_all()?;
    }
    Ok(())
}
