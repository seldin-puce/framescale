//! Source-vs-output comparison of a finished render.
//!
//! Webviews can't reliably play our outputs (AV1/HEVC 10-bit, HDR) or many
//! sources (ProRes, MPEG-2, AVI…), so both videos are decoded by our own
//! FFmpeg and shown in the WebGL compare viewer (split, side by side, or
//! either alone).
//!
//! * **Playback**: one FFmpeg per side streams display-sized JPEGs into a
//!   bounded in-memory window around the playhead (about 2 s ahead, 1 s
//!   behind). When nobody asks for frames, backpressure stalls the decoder,
//!   which then uses no CPU. A seek outside the window restarts it there.
//!   Nothing touches the disk.
//! * **Paused**: single frames at native resolution, for honest pixel-peeping.
//!
//! Both sides are sampled on the output's frame grid (frame n = n / fps), so
//! they stay aligned even when the render interpolated to a higher rate.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use tokio::io::AsyncReadExt;
use tokio::sync::Notify;

use fs_media::argv::Argv;
use fs_media::{probe, rate};
use fs_types::media::{ColorProfile, MediaInfo};
use fs_types::{CompareInfo, CompareSide, JobId, Rational};

/// Playback width cap: comparisons are viewed in a viewport, not at 8K.
pub const MAX_WIDTH: u32 = 2560;
/// How long a frame request waits for the decoder before giving up.
const WAIT: Duration = Duration::from_secs(30);
/// Native-resolution stills kept per side.
const STILLS: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameError {
    /// The session was closed (another comparison was opened).
    Closed,
    /// Beyond the last frame of the video.
    PastEnd,
    /// A newer request moved the decoder elsewhere.
    Superseded,
    TimedOut,
    Decode(String),
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Closed => f.write_str("comparison closed"),
            Self::PastEnd => f.write_str("past the end of the video"),
            Self::Superseded => f.write_str("superseded by a newer request"),
            Self::TimedOut => f.write_str("timed out waiting for the decoder"),
            Self::Decode(e) => write!(f, "decode failed: {e}"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Limits {
    /// Frames decoded beyond the furthest requested one.
    ahead: u64,
    /// Frames kept behind it (short back-scrubs are instant).
    behind: u64,
    max_bytes: usize,
}

/// Decoded frames `[lo, lo + frames.len())` of one side's current stream.
#[derive(Default)]
struct Window {
    /// Bumped on every restart; a stream only writes into its own generation.
    epoch: u64,
    lo: u64,
    frames: VecDeque<Arc<[u8]>>,
    bytes: usize,
    /// Furthest frame requested: the decoder may run to `want + ahead`.
    want: u64,
    running: bool,
    error: Option<String>,
    task: Option<tokio::task::JoinHandle<()>>,
    closed: bool,
}

impl Window {
    fn hi(&self) -> u64 {
        self.lo + self.frames.len() as u64
    }

    fn get(&self, n: u64) -> Option<Arc<[u8]>> {
        let i = usize::try_from(n.checked_sub(self.lo)?).ok()?;
        self.frames.get(i).cloned()
    }
}

struct Side {
    input: PathBuf,
    stream: u32,
    color: ColorProfile,
    /// First video frame's offset from the file's start time.
    v0: f64,
    win: Mutex<Window>,
    /// Frames arrived or the stream stopped (wakes waiting requests).
    changed: Notify,
    /// The playhead moved (wakes a decoder held back by the window).
    room: Notify,
    /// Frames at or past this don't exist (learned when a stream ends early).
    end: AtomicU64,
    stills: Mutex<VecDeque<(u64, Arc<[u8]>)>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Side {
    fn new(m: &MediaInfo, input: &Path) -> Self {
        Self {
            input: input.to_path_buf(),
            stream: m.video.index,
            color: m.video.color.clone(),
            v0: (m.video.start_time - m.start_time).max(0.0),
            win: Mutex::new(Window::default()),
            changed: Notify::new(),
            room: Notify::new(),
            end: AtomicU64::new(u64::MAX),
            stills: Mutex::new(VecDeque::new()),
        }
    }

    /// Append a decoded frame once the window has room. `false` = this
    /// stream was superseded or closed and should stop.
    async fn push(&self, epoch: u64, frame: Arc<[u8]>, l: Limits) -> bool {
        loop {
            let room = self.room.notified();
            {
                let mut w = lock(&self.win);
                if w.epoch != epoch || w.closed {
                    return false;
                }
                let hi = w.hi();
                if hi <= w.want || (hi < w.want + l.ahead && w.bytes < l.max_bytes) {
                    w.bytes += frame.len();
                    w.frames.push_back(frame);
                    drop(w);
                    self.changed.notify_waiters();
                    return true;
                }
            }
            room.await;
        }
    }

    /// Note a request for frame `n`: moves the window forward, frees frames
    /// far behind, and lets a held-back decoder continue.
    fn advance(&self, w: &mut Window, n: u64, l: Limits) {
        w.want = w.want.max(n);
        let keep_from = w.want.saturating_sub(l.behind);
        while w.lo < keep_from || (w.bytes > l.max_bytes && w.lo < n) {
            let Some(f) = w.frames.pop_front() else { break };
            w.bytes -= f.len();
            w.lo += 1;
        }
        self.room.notify_one();
    }
}

pub struct CompareSession {
    pub info: CompareInfo,
    ffmpeg: PathBuf,
    fps: Rational,
    size: (u32, u32),
    limits: Limits,
    sides: [Arc<Side>; 2],
}

async fn probe_async(ffprobe: &Path, input: &Path) -> Result<MediaInfo, String> {
    let (fp, inp) = (ffprobe.to_path_buf(), input.to_path_buf());
    tokio::task::spawn_blocking(move || probe::probe(&fp, &inp))
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| format!("{}: {e}", input.display()))
}

fn even(x: f64) -> u32 {
    ((x / 2.0).round() as u32).max(1) * 2
}

fn last_line(stderr: &[u8]) -> String {
    String::from_utf8_lossy(stderr)
        .lines()
        .rfind(|l| !l.trim().is_empty())
        .unwrap_or("FFmpeg failed")
        .to_owned()
}

/// Length of the first complete JPEG in `buf` (FFmpeg's MJPEG pipe is plain
/// concatenated JPEGs), or `None` while it's still incomplete. Walks the
/// marker segments, so bytes inside headers can't be mistaken for EOI.
fn jpeg_len(buf: &[u8]) -> Result<Option<usize>, String> {
    if buf.len() < 2 {
        return Ok(None);
    }
    if buf[0] != 0xFF || buf[1] != 0xD8 {
        return Err("decoder output isn't a JPEG stream".into());
    }
    let mut i = 2;
    loop {
        if i + 1 >= buf.len() {
            return Ok(None);
        }
        if buf[i] != 0xFF {
            return Err(format!("corrupt JPEG marker at byte {i}"));
        }
        let m = buf[i + 1];
        match m {
            0xFF => {
                i += 1; // fill byte
                continue;
            }
            0xD9 => return Ok(Some(i + 2)),
            0x01 | 0xD0..=0xD7 => {
                i += 2; // standalone markers
                continue;
            }
            _ => {}
        }
        if i + 3 >= buf.len() {
            return Ok(None);
        }
        i += 2 + usize::from(u16::from_be_bytes([buf[i + 2], buf[i + 3]]));
        if m == 0xDA {
            // Entropy-coded data runs to the next marker that isn't a
            // stuffed 0xFF00 or a restart marker.
            loop {
                if i + 1 >= buf.len() {
                    return Ok(None);
                }
                if buf[i] == 0xFF && buf[i + 1] != 0x00 && !(0xD0..=0xD7).contains(&buf[i + 1]) {
                    break;
                }
                i += 1;
            }
        }
    }
}

fn command(ffmpeg: &Path, argv: &Argv) -> tokio::process::Command {
    let mut cmd = std::process::Command::new(ffmpeg);
    cmd.args(argv.as_slice())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    fs_proc::prepare(&mut cmd);
    let mut cmd = tokio::process::Command::from(cmd);
    cmd.kill_on_drop(true);
    cmd
}

/// Run one stream into the window. `Ok(true)` = FFmpeg reached the end,
/// `Ok(false)` = superseded.
async fn stream(
    side: &Side,
    epoch: u64,
    ffmpeg: &Path,
    argv: &Argv,
    l: Limits,
) -> Result<bool, String> {
    let mut child = command(ffmpeg, argv).spawn().map_err(|e| e.to_string())?;
    let mut out = child.stdout.take().ok_or("no decoder stdout")?;
    let stderr = child.stderr.take().map(|mut e| {
        tokio::spawn(async move {
            let mut v = Vec::new();
            let _ = e.read_to_end(&mut v).await;
            v
        })
    });
    let mut buf = Vec::with_capacity(1 << 20);
    let mut chunk = vec![0u8; 1 << 20];
    loop {
        while let Some(len) = jpeg_len(&buf)? {
            let frame: Arc<[u8]> = Arc::from(&buf[..len]);
            buf.drain(..len);
            if !side.push(epoch, frame, l).await {
                return Ok(false);
            }
        }
        let k = out.read(&mut chunk).await.map_err(|e| e.to_string())?;
        if k == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..k]);
    }
    let status = child.wait().await.map_err(|e| e.to_string())?;
    if status.success() {
        return Ok(true);
    }
    let err = match stderr {
        Some(h) => h.await.unwrap_or_default(),
        None => Vec::new(),
    };
    Err(last_line(&err))
}

async fn produce(side: Arc<Side>, epoch: u64, ffmpeg: PathBuf, argv: Argv, l: Limits) {
    let r = stream(&side, epoch, &ffmpeg, &argv, l).await;
    let mut w = lock(&side.win);
    if w.epoch != epoch {
        return;
    }
    w.running = false;
    match r {
        // A stream that ends early has found the real end of the video.
        Ok(true) => {
            side.end.fetch_min(w.hi(), Ordering::Relaxed);
        }
        Ok(false) => {}
        Err(e) => w.error = Some(e),
    }
    drop(w);
    side.changed.notify_waiters();
}

impl CompareSession {
    /// Probe both files and size the playback frames to `max_width`.
    pub async fn open(
        ffmpeg: &Path,
        ffprobe: &Path,
        source: &Path,
        output: &Path,
        max_width: u32,
    ) -> Result<Self, String> {
        let (src, out) = (
            probe_async(ffprobe, source).await?,
            probe_async(ffprobe, output).await?,
        );
        let fps = rate::conform_rate(Some(out.video.r_frame_rate), Some(out.video.avg_frame_rate));
        if fps.num <= 0 || fps.den <= 0 {
            return Err(format!("{} has no usable frame rate", output.display()));
        }
        let src_fps =
            rate::conform_rate(Some(src.video.r_frame_rate), Some(src.video.avg_frame_rate));
        let (ow, oh) = (out.video.width.max(2), out.video.height.max(2));
        let sar = out.video.sample_aspect;
        let sar = if sar.num > 0 && sar.den > 0 {
            sar.as_f64()
        } else {
            1.0
        };
        let display_w = f64::from(ow) * sar;
        let width = even(display_w.min(f64::from(max_width.clamp(64, MAX_WIDTH))));
        let height = even(f64::from(width) * f64::from(oh) / display_w);

        let after = Side::new(&out, output);
        let before = Side::new(&src, source);
        let duration = if out.duration_s > 0.0 {
            out.duration_s
        } else {
            src.duration_s
        };
        let frames = out
            .video
            .frame_count
            .filter(|&n| n > 0)
            .unwrap_or_else(|| (((duration - after.v0) * fps.as_f64()).round() as u64).max(1));
        let per_s = fps.as_f64().ceil() as u64;
        Ok(Self {
            info: CompareInfo {
                session: JobId::new().to_string(),
                duration_s: frames as f64 / fps.as_f64(),
                fps: fps.as_f64(),
                frames,
                width,
                height,
                source_width: src.video.width,
                source_height: src.video.height,
                source_fps: src_fps.as_f64(),
                output_width: ow,
                output_height: oh,
                source_path: source.to_string_lossy().into_owned(),
                output_path: output.to_string_lossy().into_owned(),
                tone_mapped: src.video.color.is_hdr() || out.video.color.is_hdr(),
            },
            ffmpeg: ffmpeg.to_path_buf(),
            fps,
            size: (width, height),
            limits: Limits {
                ahead: (per_s * 2).max(16),
                behind: per_s.max(8),
                max_bytes: 192 << 20,
            },
            sides: [Arc::new(before), Arc::new(after)],
        })
    }

    fn side(&self, side: CompareSide) -> &Arc<Side> {
        match side {
            CompareSide::Before => &self.sides[0],
            CompareSide::After => &self.sides[1],
        }
    }

    fn argv(&self, s: &Side, n: u64, frames: u64, size: Option<(u32, u32)>, quality: u8) -> Argv {
        fs_media::argv::compare_frames(
            &s.input,
            s.stream,
            s.v0 + self.fps.frame_time(n),
            frames,
            self.fps,
            size,
            &s.color,
            quality,
        )
    }

    fn restart(&self, s: &Arc<Side>, w: &mut Window, n: u64) {
        if let Some(t) = w.task.take() {
            t.abort(); // drops the child: kill_on_drop
        }
        w.epoch += 1;
        w.lo = n;
        w.frames.clear();
        w.bytes = 0;
        w.want = n;
        w.running = true;
        w.error = None;
        let last = self.info.frames.min(s.end.load(Ordering::Relaxed));
        let argv = self.argv(s, n, last.saturating_sub(n).max(1), Some(self.size), 3);
        w.task = Some(tokio::spawn(produce(
            Arc::clone(s),
            w.epoch,
            self.ffmpeg.clone(),
            argv,
            self.limits,
        )));
    }

    /// Playback-size frame `n` of one side. Waits while the decoder gets
    /// there; restarts it when `n` is outside its window.
    pub async fn frame(&self, side: CompareSide, n: u64) -> Result<Arc<[u8]>, FrameError> {
        let s = self.side(side);
        let deadline = tokio::time::Instant::now() + WAIT;
        let mut joined: Option<u64> = None;
        loop {
            let changed = s.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            {
                let mut w = lock(&s.win);
                if w.closed {
                    return Err(FrameError::Closed);
                }
                if n >= self.info.frames.min(s.end.load(Ordering::Relaxed)) {
                    return Err(FrameError::PastEnd);
                }
                if let Some(f) = w.get(n) {
                    s.advance(&mut w, n, self.limits);
                    return Ok(f);
                }
                let reachable = w.running && n >= w.lo && n < w.hi() + self.limits.ahead;
                match joined {
                    // Someone moved the decoder away: the newest request wins.
                    Some(g) if g != w.epoch && !reachable => return Err(FrameError::Superseded),
                    Some(g) if g == w.epoch && !w.running => {
                        return Err(w
                            .error
                            .clone()
                            .map_or(FrameError::PastEnd, FrameError::Decode));
                    }
                    Some(_) => {}
                    None if !reachable => self.restart(s, &mut w, n),
                    None => {}
                }
                joined = Some(w.epoch);
                s.advance(&mut w, n, self.limits);
            }
            if tokio::time::timeout_at(deadline, changed).await.is_err() {
                return Err(FrameError::TimedOut);
            }
        }
    }

    /// Frame `n` of one side at native resolution (decoded on its own).
    pub async fn still(&self, side: CompareSide, n: u64) -> Result<Arc<[u8]>, FrameError> {
        let s = self.side(side);
        if n >= self.info.frames.min(s.end.load(Ordering::Relaxed)) {
            return Err(FrameError::PastEnd);
        }
        if let Some((_, f)) = lock(&s.stills).iter().find(|(k, _)| *k == n) {
            return Ok(Arc::clone(f));
        }
        let o = command(&self.ffmpeg, &self.argv(s, n, 1, None, 1))
            .output()
            .await
            .map_err(|e| FrameError::Decode(e.to_string()))?;
        if !o.status.success() {
            return Err(FrameError::Decode(last_line(&o.stderr)));
        }
        let len = jpeg_len(&o.stdout)
            .map_err(FrameError::Decode)?
            .ok_or(FrameError::PastEnd)?;
        let f: Arc<[u8]> = Arc::from(&o.stdout[..len]);
        let mut c = lock(&s.stills);
        c.push_back((n, Arc::clone(&f)));
        while c.len() > STILLS {
            c.pop_front();
        }
        Ok(f)
    }

    /// Stop both decoders and fail pending requests.
    pub fn close(&self) {
        for s in &self.sides {
            let mut w = lock(&s.win);
            w.closed = true;
            w.epoch += 1;
            if let Some(t) = w.task.take() {
                t.abort();
            }
            w.frames.clear();
            w.bytes = 0;
            drop(w);
            lock(&s.stills).clear();
            s.changed.notify_waiters();
        }
    }
}

impl std::fmt::Debug for CompareSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompareSession")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

impl Drop for CompareSession {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::jpeg_len;

    #[test]
    fn splits_concatenated_jpegs() {
        // SOI, APP0 (contains FF D9 inside its payload), SOS header, entropy
        // data with a stuffed FF00 and a restart marker, EOI. Then a second image.
        let one: Vec<u8> = [
            &[0xFF, 0xD8][..],
            &[0xFF, 0xE0, 0x00, 0x06, 0xFF, 0xD9, 0x00, 0x00],
            &[0xFF, 0xDA, 0x00, 0x03, 0x01],
            &[0x12, 0xFF, 0x00, 0x34, 0xFF, 0xD3, 0x56],
            &[0xFF, 0xD9],
        ]
        .concat();
        let two = [one.clone(), one.clone()].concat();
        assert_eq!(jpeg_len(&two), Ok(Some(one.len())));
        for cut in 0..one.len() {
            assert_eq!(jpeg_len(&one[..cut]), Ok(None), "cut at {cut}");
        }
        assert!(jpeg_len(&[0x00, 0x01, 0x02]).is_err());
    }
}
