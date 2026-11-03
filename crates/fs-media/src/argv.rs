//! A typed argument vector, and the decode/encode/concat command builders.

use std::ffi::{OsStr, OsString};
use std::fmt;
use std::path::Path;

use fs_types::worker::{DecodePlan, EncodePlan, FrameFormat, HwDecode};

use crate::color;

/// An FFmpeg argument list (excluding the program name).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Argv(Vec<OsString>);

impl Argv {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn arg(&mut self, a: impl AsRef<OsStr>) -> &mut Self {
        self.0.push(a.as_ref().to_owned());
        self
    }
    pub fn args<I, S>(&mut self, it: I) -> &mut Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        self.0.extend(it.into_iter().map(|s| s.as_ref().to_owned()));
        self
    }
    pub fn as_slice(&self) -> &[OsString] {
        &self.0
    }
    pub fn into_vec(self) -> Vec<OsString> {
        self.0
    }
    /// True if `a` appears as a whole argument.
    pub fn contains(&self, a: &str) -> bool {
        self.0.iter().any(|x| x == a)
    }
    /// The argument following flag `flag`, if any.
    pub fn value_of(&self, flag: &str) -> Option<String> {
        let i = self.0.iter().position(|x| x == flag)?;
        self.0.get(i + 1).map(|v| v.to_string_lossy().into_owned())
    }
}

/// Shell-like rendering for logs only. Never pass this to a shell.
impl fmt::Display for Argv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, a) in self.0.iter().enumerate() {
            let s = a.to_string_lossy();
            let quote = s.is_empty() || s.contains([' ', '\'', '"', ';', '(', ')', ',']);
            if i > 0 {
                f.write_str(" ")?;
            }
            if quote {
                write!(f, "'{}'", s.replace('\'', r"'\''"))?
            } else {
                f.write_str(&s)?
            }
        }
        Ok(())
    }
}

/// Common prefix: no stdin, no banner, warnings only, no human stats line,
/// and `-y`. Every file we write is our own temp/segment/`.part` file, and a
/// leftover from a crashed run must never block the retry ("already exists").
fn base(level: &str) -> Argv {
    let mut a = Argv::new();
    a.args([
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-y",
        "-loglevel",
        level,
    ]);
    a
}

/// Hardware decode flags. Frames are downloaded to system memory: we don't set
/// `-hwaccel_output_format`, because the AI path needs raw frames on a pipe.
pub fn hwaccel_args(hw: &HwDecode) -> Vec<String> {
    let v = |xs: &[&str]| xs.iter().map(|s| (*s).to_owned()).collect();
    match hw {
        HwDecode::None => Vec::new(),
        HwDecode::Cuda => v(&["-hwaccel", "cuda"]),
        HwDecode::D3d11va => v(&["-hwaccel", "d3d11va"]),
        HwDecode::Qsv => v(&["-hwaccel", "qsv"]),
        HwDecode::VideoToolbox => v(&["-hwaccel", "videotoolbox"]),
        HwDecode::Vaapi { device } => {
            vec![
                "-hwaccel".into(),
                "vaapi".into(),
                "-hwaccel_device".into(),
                device.clone(),
            ]
        }
    }
}

/// Seconds to pass to input `-ss` so decoding starts at conformed frame `n`.
pub fn seek_seconds(plan: &DecodePlan, n: u64) -> f64 {
    if n == 0 {
        0.0
    } else {
        plan.seek_base_s + plan.conform_rate.frame_time(n)
    }
}

/// The filter chain from decoded source frames to raw RGB for the engine.
pub fn decode_filter(plan: &DecodePlan) -> String {
    let mut f = Vec::new();
    if let Some(send_field) = plan.deinterlace {
        f.push(format!(
            "bwdif=mode={}:parity=auto:deint=all",
            u8::from(send_field)
        ));
    }
    f.push("setpts=PTS-STARTPTS".into());
    f.push(format!("fps={}:round=near", plan.conform_rate));
    f.push(color::to_rgb_filter(
        &plan.color,
        plan.frame_format == FrameFormat::Rgb48,
    ));
    f.push(format!("format={}", plan.frame_format.pix_fmt()));
    f.join(",")
}

/// Decoder: source → conformed raw RGB frames on stdout, starting at conformed
/// frame `start_frame` (approximately; the worker aligns exactly via hashes).
pub fn decode(plan: &DecodePlan, start_frame: u64) -> Argv {
    let mut a = base("warning");
    a.args(hwaccel_args(&plan.hw));
    if start_frame > 0 {
        a.args(["-ss", &format!("{:.6}", seek_seconds(plan, start_frame))]);
    }
    a.arg("-i").arg(&plan.input);
    a.args(["-map", &format!("0:{}", plan.stream_index)]);
    a.args(["-vf", &decode_filter(plan)]);
    a.args([
        "-f",
        "rawvideo",
        "-pix_fmt",
        plan.frame_format.pix_fmt(),
        "pipe:1",
    ]);
    a
}

/// Encoder for one segment: raw RGB on stdin → `<out>` (Matroska).
pub fn encode_segment(
    plan: &EncodePlan,
    in_w: u32,
    in_h: u32,
    in_fmt: FrameFormat,
    out: &Path,
) -> Argv {
    let mut a = base("warning");
    if let Some(dev) = &plan.vaapi_device {
        a.args(["-vaapi_device", dev]);
    }
    a.args(["-f", "rawvideo", "-pix_fmt", in_fmt.pix_fmt()]);
    a.args([
        "-s",
        &format!("{in_w}x{in_h}"),
        "-framerate",
        &plan.rate.to_string(),
    ]);
    a.args(["-i", "pipe:0"]);

    let mut vf = Vec::new();
    if !in_w.is_multiple_of(2) || !in_h.is_multiple_of(2) {
        // 4:2:0 / 4:2:2 need even dimensions; pad by one pixel, never crop.
        vf.push("pad=ceil(iw/2)*2:ceil(ih/2)*2".to_owned());
    }
    let sar = plan.sample_aspect;
    if sar.num > 0 && sar.den > 0 && sar.num != sar.den {
        // Raw frames carry no aspect: without this, anamorphic sources come
        // out squeezed or stretched (PAL 4:3 at 16:15 would play as 5:4).
        vf.push(format!("setsar={}/{}", sar.num, sar.den));
    }
    vf.push(color::from_rgb_filter(&plan.color, &plan.pix_fmt));
    vf.push(format!("format={}", plan.pix_fmt));
    if plan.vaapi_device.is_some() {
        vf.push("hwupload".into());
    }
    a.args(["-vf", &vf.join(",")]);
    a.args(["-c:v", &plan.encoder]).args(&plan.args);
    a.args(color::output_tags(&plan.color));
    a.args(["-an", "-sn", "-dn", "-f", "matroska"]).arg(out);
    a
}

/// Raw RGB frames on stdin → numbered JPEGs for display (`pattern` like `…/%04d.jpg`).
pub fn jpeg_sequence(
    w: u32,
    h: u32,
    fmt: FrameFormat,
    color: &fs_types::media::ColorProfile,
    pattern: &Path,
) -> Argv {
    let mut a = base("warning");
    a.args([
        "-f",
        "rawvideo",
        "-pix_fmt",
        fmt.pix_fmt(),
        "-s",
        &format!("{w}x{h}"),
        "-i",
        "pipe:0",
    ]);
    a.args(["-vf", &color::display_jpeg_filter(color)]);
    a.args([
        "-c:v",
        "mjpeg",
        "-q:v",
        "2",
        "-start_number",
        "0",
        "-f",
        "image2",
    ])
    .arg(pattern);
    a
}

/// Decoding starts this long before the requested time, so the frame on
/// screen at that time is included even for low-frame-rate sources.
const COMPARE_PREROLL_S: f64 = 0.5;

/// Compare player: `frames` frames of `input` on the output's frame grid,
/// starting at media time `at_s` (relative to the file start), as a stream of
/// concatenated JPEGs on stdout. Display-sized when `size` is given, native
/// resolution otherwise; HDR is tone-mapped for display.
///
/// The `fps` filter is anchored at `at_s` instead of the first decoded frame:
/// each output tick shows the latest input frame at or before it, so a seek
/// never shifts the result by one frame (millisecond container timestamps,
/// lower source frame rates).
#[allow(clippy::too_many_arguments)]
pub fn compare_frames(
    input: &Path,
    stream_index: u32,
    at_s: f64,
    frames: u64,
    fps: fs_types::Rational,
    size: Option<(u32, u32)>,
    color: &fs_types::media::ColorProfile,
    quality: u8,
) -> Argv {
    let seek = (at_s - COMPARE_PREROLL_S).max(0.0);
    let mut a = base("error");
    if seek > 0.0 {
        a.args(["-ss", &format!("{seek:.6}")]);
    }
    a.arg("-i").arg(input);
    a.args([
        "-map",
        &format!("0:{stream_index}"),
        "-frames:v",
        &frames.to_string(),
    ]);
    let scale = size.map_or_else(String::new, |(w, h)| {
        format!(",scale={w}:{h}:flags=bicubic")
    });
    let vf = format!(
        "fps=fps={fps}:start_time={:.6}:round=near{scale},{},{}",
        at_s - seek,
        color::to_rgb_filter(color, color.needs_high_precision()),
        color::display_jpeg_filter(color)
    );
    a.args([
        "-vf",
        &vf,
        "-c:v",
        "mjpeg",
        "-qmin",
        "1",
        "-q:v",
        &quality.to_string(),
    ]);
    a.args(["-f", "image2pipe", "pipe:1"]);
    a
}

/// One small JPEG thumbnail from a source file at `at_s` seconds.
pub fn thumbnail(
    input: &Path,
    stream_index: u32,
    at_s: f64,
    color: &fs_types::media::ColorProfile,
    width: u32,
    out: &Path,
) -> Argv {
    let mut a = base("error");
    a.args(["-ss", &format!("{at_s:.3}")]).arg("-i").arg(input);
    a.args(["-map", &format!("0:{stream_index}"), "-frames:v", "1"]);
    let hp = color.needs_high_precision();
    let vf = format!(
        "scale={width}:-2:flags=bicubic,{},{}",
        color::to_rgb_filter(color, hp),
        color::display_jpeg_filter(color)
    );
    a.args(["-vf", &vf, "-c:v", "mjpeg", "-q:v", "4", "-update", "1"])
        .arg(out);
    a
}

/// Concatenate committed segments losslessly (`-c copy`).
pub fn concat(list_file: &Path, out: &Path) -> Argv {
    let mut a = base("warning");
    a.args(["-progress", "pipe:1"]);
    a.args(["-f", "concat", "-safe", "0", "-i"]).arg(list_file);
    a.args(["-map", "0:v:0", "-c", "copy", "-f", "matroska"])
        .arg(out);
    a
}

/// Contents of a concat-demuxer list. Names are relative to the list's
/// directory; single quotes are escaped per the concat demuxer's rules.
pub fn concat_list<'a>(files: impl IntoIterator<Item = &'a str>) -> String {
    files
        .into_iter()
        .map(|f| format!("file '{}'\n", f.replace('\'', r"'\''")))
        .collect()
}

/// Quick capability probe: can this machine actually open `encoder`?
/// (A build may list `av1_nvenc` on a GPU that can't encode AV1, and a distro
/// may ship VAAPI without H.264/HEVC.) 720p, because hardware encoders reject
/// tiny frames (RDNA4 AV1 fails at 256×256 with EINVAL).
pub fn encoder_probe(encoder: &str, pix_fmt: &str, vaapi_device: Option<&str>) -> Argv {
    let mut a = base("error");
    let vaapi = encoder.ends_with("_vaapi");
    if vaapi {
        a.args([
            "-vaapi_device",
            vaapi_device.unwrap_or("/dev/dri/renderD128"),
        ]);
    }
    a.args(["-f", "lavfi", "-i", "testsrc2=s=1280x720:r=24:d=0.25"]);
    let vf = if vaapi {
        format!("format={pix_fmt},hwupload")
    } else {
        format!("format={pix_fmt}")
    };
    a.args(["-vf", &vf, "-frames:v", "4", "-c:v", encoder]);
    a.args(["-f", "null", "-"]);
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs_types::media::*;

    pub(crate) fn sdr() -> ColorProfile {
        ColorProfile {
            primaries: ColorPrimaries::Bt709,
            transfer: TransferCharacteristic::Bt709,
            matrix: MatrixCoefficients::Bt709,
            range: ColorRange::Limited,
            bit_depth: 8,
            mastering: None,
            content_light: None,
            inferred: false,
        }
    }

    fn dplan() -> DecodePlan {
        DecodePlan {
            input: "/media/my clip; rm -rf.mov".into(),
            stream_index: 0,
            hw: HwDecode::Cuda,
            conform_rate: Rational::new(24000, 1001),
            deinterlace: None,
            width: 1920,
            height: 1080,
            frame_format: FrameFormat::Rgb24,
            color: sdr(),
            seek_base_s: 0.5,
        }
    }

    #[test]
    fn decode_keeps_hostile_path_as_single_arg() {
        let a = decode(&dplan(), 0);
        assert_eq!(
            a.value_of("-i").as_deref(),
            Some("/media/my clip; rm -rf.mov")
        );
        assert_eq!(a.value_of("-hwaccel").as_deref(), Some("cuda"));
        assert!(!a.contains("-ss"));
        let vf = a.value_of("-vf").expect("vf");
        assert!(
            vf.starts_with("setpts=PTS-STARTPTS,fps=24000/1001:round=near,zscale="),
            "{vf}"
        );
        assert!(vf.ends_with("format=rgb24"));
    }

    #[test]
    fn decode_seek_is_relative_to_video_start() {
        let a = decode(&dplan(), 24_000);
        // 24000 frames at 23.976 = 1001 s, + 0.5 s video-vs-container start offset.
        assert_eq!(a.value_of("-ss").as_deref(), Some("1001.500000"));
    }

    #[test]
    fn deinterlace_precedes_conform() {
        let mut p = dplan();
        p.deinterlace = Some(true);
        let vf = decode_filter(&p);
        assert!(vf.starts_with("bwdif=mode=1"), "{vf}");
    }

    #[test]
    fn concat_list_escapes_quotes() {
        assert_eq!(
            concat_list(["a.mkv", "it's.mkv"]),
            "file 'a.mkv'\nfile 'it'\\''s.mkv'\n"
        );
    }
}
