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

/// Common prefix: no stdin, no banner, warnings only, and no human stats line.
fn base(level: &str) -> Argv {
    let mut a = Argv::new();
    a.args(["-hide_banner", "-nostdin", "-nostats", "-loglevel", level]);
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
/// (A build may list `av1_nvenc` on a GPU that can't encode AV1.)
pub fn encoder_probe(encoder: &str, pix_fmt: &str) -> Argv {
    let mut a = base("error");
    a.args(["-f", "lavfi", "-i", "color=c=black:s=256x256:r=24:d=0.25"]);
    a.args([
        "-vf",
        &format!("format={pix_fmt}"),
        "-frames:v",
        "4",
        "-c:v",
        encoder,
    ]);
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
