//! Runs the command builders against a real FFmpeg.
//!
//! Set `FS_FFMPEG` and `FS_FFPROBE` (e.g. to the sidecars installed by
//! `cargo xtask setup`). Without them, these tests skip.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use fs_media::{analyze, argv, color, probe, rate};
use fs_types::Rational;
use fs_types::media::{MediaInfo, TransferCharacteristic};
use fs_types::worker::{DecodePlan, EncodePlan, FrameFormat, HdrEmbedding, HwDecode};

fn tools() -> Option<(PathBuf, PathBuf)> {
    let f = std::env::var_os("FS_FFMPEG")?;
    let p = std::env::var_os("FS_FFPROBE")?;
    Some((f.into(), p.into()))
}

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("fs-media-it-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&d).expect("tmp");
    d
}

fn ffmpeg(ff: &Path, args: &[&str]) {
    let out = Command::new(ff)
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args(args)
        .output()
        .expect("spawn");
    assert!(
        out.status.success(),
        "ffmpeg {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// 3 scenes × 2 s @ 24 fps (cuts at frames 48 and 96), audio starting 0 s,
/// video stream starting at 0.25 s (A/V offset that the mux must preserve).
fn make_cuts_clip(ff: &Path, dir: &Path) -> PathBuf {
    let out = dir.join("cuts.mkv");
    ffmpeg(
        ff,
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x240:r=24:d=2",
            "-f",
            "lavfi",
            "-i",
            "mandelbrot=s=320x240:r=24,trim=duration=2",
            "-f",
            "lavfi",
            "-i",
            "smptehdbars=s=320x240:r=24:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=f=440:d=6",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p,setpts=PTS+0.25/TB[v]",
            "-map",
            "[v]",
            "-map",
            "3:a",
            "-c:v",
            "ffv1",
            "-c:a",
            "flac",
            out.to_str().expect("utf8"),
        ],
    );
    out
}

fn probe_info(fp: &Path, input: &Path) -> MediaInfo {
    probe::probe(fp, input).expect("probe")
}

fn decode_plan(info: &MediaInfo, rate: Rational, fmt: FrameFormat) -> DecodePlan {
    DecodePlan {
        input: info.path.clone(),
        stream_index: info.video.index,
        hw: HwDecode::None,
        conform_rate: rate,
        deinterlace: None,
        width: info.video.width,
        height: info.video.height,
        frame_format: fmt,
        color: info.video.color.clone(),
        seek_base_s: fs_media::mux::video_offset(info),
    }
}

fn decode_all(ff: &Path, plan: &DecodePlan, start: u64) -> Vec<Vec<u8>> {
    let a = argv::decode(plan, start);
    let mut child = Command::new(ff)
        .args(a.as_slice())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    let mut raw = Vec::new();
    child
        .stdout
        .take()
        .expect("stdout")
        .read_to_end(&mut raw)
        .expect("read");
    let out = child.wait_with_output().expect("wait");
    assert!(
        out.status.success(),
        "{a}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let fb = plan.frame_format.frame_bytes(plan.width, plan.height);
    assert_eq!(raw.len() % fb, 0, "partial frame");
    raw.chunks(fb).map(<[u8]>::to_vec).collect()
}

#[test]
fn probe_analyze_and_frame_exact_seek() {
    let Some((ff, fp)) = tools() else {
        return eprintln!("skip: FS_FFMPEG/FS_FFPROBE unset");
    };
    let dir = tmpdir("seek");
    let clip = make_cuts_clip(&ff, &dir);

    let info = probe_info(&fp, &clip);
    assert_eq!((info.video.width, info.video.height), (320, 240));
    assert_eq!(info.audio.len(), 1);
    assert!(
        (fs_media::mux::video_offset(&info) - 0.25).abs() < 0.01,
        "video start offset"
    );
    let conform = rate::conform_rate(
        Some(info.video.r_frame_rate),
        Some(info.video.avg_frame_rate),
    );
    assert_eq!(conform, Rational::new(24, 1));

    // ANALYZE: exact frame count + cuts at the scene boundaries.
    let a = analyze::argv(
        &clip,
        info.video.index,
        &HwDecode::None,
        conform,
        Some(10.0),
    );
    let out = Command::new(&ff)
        .args(a.as_slice())
        .output()
        .expect("analyze");
    assert!(out.status.success());
    let mut p = analyze::AnalyzeParser::new();
    String::from_utf8_lossy(&out.stderr)
        .lines()
        .for_each(|l| p.feed_stderr(l));
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .for_each(|l| p.feed_progress(l));
    let report = p.finish(conform);
    assert_eq!(report.total_frames, 144);
    assert_eq!(report.scene_cuts, vec![48, 96]);
    assert!(!report.is_vfr);

    // DECODE: full decode, then a resume-style decode from frame 50 must
    // reproduce frames 50.. bit-exactly (validates seek_base_s + -ss maths).
    let plan = decode_plan(&info, conform, FrameFormat::Rgb24);
    let all = decode_all(&ff, &plan, 0);
    assert_eq!(all.len(), 144);
    let tail = decode_all(&ff, &plan, 50);
    assert_eq!(tail.len(), 94);
    assert!(tail[0] == all[50], "seek landed on the wrong frame");
    assert!(tail[93] == all[143]);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn hdr10_passthrough_through_rgb48_and_svtav1() {
    let Some((ff, fp)) = tools() else {
        return eprintln!("skip: FS_FFMPEG/FS_FFPROBE unset");
    };
    let dir = tmpdir("hdr");
    let src = dir.join("hdr.mkv");
    // 10-bit BT.2020/PQ source. AV1 carries the tags in its bitstream; FFV1 in
    // Matroska would drop transfer/primaries.
    ffmpeg(
        &ff,
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=256x144:r=25:d=1",
            "-vf",
            "format=yuv420p10le,setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc:range=tv",
            "-c:v",
            "libsvtav1",
            "-preset",
            "12",
            src.to_str().expect("utf8"),
        ],
    );
    let info = probe_info(&fp, &src);
    assert!(info.video.color.is_hdr() && info.video.color.bit_depth == 10);

    let plan = decode_plan(&info, Rational::new(25, 1), FrameFormat::Rgb48);
    assert!(plan.color.needs_high_precision());
    let frames = decode_all(&ff, &plan, 0);
    assert_eq!(frames.len(), 25);

    // Encode those RGB48 frames as one segment with SVT-AV1 + HDR params.
    let sel_color = info.video.color.clone();
    let enc = EncodePlan {
        encoder: "libsvtav1".into(),
        args: vec![
            "-preset".into(),
            "12".into(),
            "-crf".into(),
            "35".into(),
            "-svtav1-params".into(),
            "enable-hdr=1".into(),
        ],
        pix_fmt: "yuv420p10le".into(),
        vaapi_device: None,
        width: 256,
        height: 144,
        rate: Rational::new(25, 1),
        color: sel_color,
        hdr: HdrEmbedding::Bitstream,
        sample_aspect: Rational::new(1, 1),
    };
    let seg = dir.join("seg_00000.mkv");
    let a = argv::encode_segment(&enc, 256, 144, FrameFormat::Rgb48, &seg);
    let mut child = Command::new(&ff)
        .args(a.as_slice())
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("enc");
    {
        let mut stdin = child.stdin.take().expect("stdin");
        for f in &frames {
            stdin.write_all(f).expect("write");
        }
    }
    let out = child.wait_with_output().expect("wait");
    assert!(
        out.status.success(),
        "{a}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let o = probe_info(&fp, &seg);
    assert_eq!(o.video.codec, "av1");
    assert_eq!(
        o.video.color.transfer,
        TransferCharacteristic::Pq,
        "transfer must pass through"
    );
    assert_eq!(o.video.color.bit_depth, 10);
    assert!(
        !o.video.color.inferred,
        "all colour tags present: {:?}",
        o.video.color
    );
    assert_eq!(color::output_tags(&o.video.color)[1], "bt2020");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn hdr10_static_metadata_survives_x265_bitstream() {
    use fs_media::encoders::{self, EncodeTarget, EncoderEnv, Os};
    use fs_types::media::{ContentLightLevel, MasteringDisplay};
    use fs_types::pipeline::{EncoderPreference, VideoCodec};

    let Some((ff, fp)) = tools() else {
        return eprintln!("skip: FS_FFMPEG/FS_FFPROBE unset");
    };
    let dir = tmpdir("x265");
    let src = dir.join("hdr.mkv");
    ffmpeg(
        &ff,
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x180:r=25:d=0.4",
            "-vf",
            "format=yuv420p10le,setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc:range=tv",
            "-c:v",
            "libsvtav1",
            "-preset",
            "12",
            src.to_str().expect("utf8"),
        ],
    );
    let info = probe_info(&fp, &src);
    let mut color = info.video.color.clone();
    let md = MasteringDisplay {
        red: [34000, 16000],
        green: [13250, 34500],
        blue: [7500, 3000],
        white: [15635, 16450],
        max_luminance: 10_000_000,
        min_luminance: 50,
    };
    let cll = ContentLightLevel {
        max_cll: 1000,
        max_fall: 400,
    };
    color.mastering = Some(md);
    color.content_light = Some(cll);

    // The real selection path, as a render on Fedora + AMD would take it.
    let env = EncoderEnv {
        os: Os::Linux,
        vendor: None,
        usable: ["libx265".to_owned()].into(),
        vaapi_device: None,
    };
    let rate = Rational::new(25, 1);
    let target = EncodeTarget {
        quality: 28,
        width: 320,
        height: 180,
        rate,
    };
    let sel = encoders::select(
        VideoCodec::Hevc10,
        EncoderPreference::Auto,
        &color,
        target,
        &env,
    )
    .expect("select");
    assert_eq!(sel.encoder, "libx265");

    let frames = decode_all(&ff, &decode_plan(&info, rate, FrameFormat::Rgb48), 0);
    let enc = EncodePlan {
        encoder: sel.encoder,
        args: sel.args,
        pix_fmt: sel.pix_fmt,
        vaapi_device: None,
        width: 320,
        height: 180,
        rate,
        color: fs_types::media::ColorProfile {
            bit_depth: 10,
            ..color
        },
        hdr: sel.hdr,
        sample_aspect: Rational::new(16, 15),
    };
    let out = dir.join("seg.mkv");
    let a = argv::encode_segment(&enc, 320, 180, FrameFormat::Rgb48, &out);
    let mut child = Command::new(&ff)
        .args(a.as_slice())
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("enc");
    {
        let mut stdin = child.stdin.take().expect("stdin");
        for f in &frames {
            stdin.write_all(f).expect("write");
        }
    }
    let o = child.wait_with_output().expect("wait");
    assert!(
        o.status.success(),
        "{a}\n{}",
        String::from_utf8_lossy(&o.stderr)
    );

    // probe() reads the first frame's SEI side data for PQ sources.
    let got = probe_info(&fp, &out);
    assert_eq!(got.video.codec, "hevc");
    assert_eq!(got.video.profile.as_deref(), Some("Main 10"));
    assert_eq!(
        got.video.sample_aspect,
        Rational::new(16, 15),
        "pixel aspect carried"
    );
    assert_eq!(got.video.color.transfer, TransferCharacteristic::Pq);
    assert_eq!(
        got.video.color.mastering,
        Some(md),
        "ST 2086 must be in the bitstream"
    );
    assert_eq!(
        got.video.color.content_light,
        Some(cll),
        "CLL must be in the bitstream"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// `name -> named constants` from an FFmpeg `-h` options listing.
fn parse_options(help: &str) -> std::collections::HashMap<String, Vec<String>> {
    let mut map = std::collections::HashMap::new();
    let mut current: Option<String> = None;
    for line in help.lines() {
        if let Some(rest) = line.strip_prefix("  -") {
            let name = rest
                .split_whitespace()
                .next()
                .unwrap_or_default()
                .to_owned();
            map.entry(name.clone()).or_insert_with(Vec::new);
            current = Some(name);
        } else if line.starts_with("     ") && !line.trim().is_empty() {
            if let (Some(c), Some(tok)) = (&current, line.split_whitespace().next()) {
                map.get_mut(c).expect("current").push(tok.to_owned());
            }
        } else if !line.starts_with(' ') {
            current = None;
        }
    }
    map
}

/// Every `-option value` we pass to every encoder must exist in this FFmpeg
/// build (private encoder option or global codec option), and named values
/// must be valid constants. This catches a wrong option name without needing
/// the GPU (e.g. `-qp` on av1_vaapi, which only reads `-global_quality`).
#[test]
fn every_encoder_option_we_emit_exists_in_ffmpeg() {
    use fs_media::encoders::{self, EncodeTarget, EncoderEnv, Os};
    use fs_types::pipeline::VideoCodec;

    let Some((ff, _)) = tools() else {
        return eprintln!("skip: FS_FFMPEG/FS_FFPROBE unset");
    };
    let help = |args: &[&str]| {
        let o = Command::new(&ff)
            .arg("-hide_banner")
            .args(args)
            .output()
            .expect("ffmpeg -h");
        String::from_utf8_lossy(&o.stdout).into_owned() + &String::from_utf8_lossy(&o.stderr)
    };
    let full = help(&["-h", "full"]);
    let start = full
        .find("AVCodecContext AVOptions:")
        .expect("global codec options");
    let global =
        parse_options(&full[start..full[start..].find("\n\n").map_or(full.len(), |e| start + e)]);

    let mut hdr = probe::parse_streams(
        r#"{"streams":[{"index":0,"codec_type":"video","width":3840,"height":2160,"pix_fmt":"yuv420p10le",
            "color_primaries":"bt2020","color_transfer":"smpte2084","color_space":"bt2020nc","color_range":"tv"}],"format":{}}"#,
        "x",
    )
    .expect("parse")
    .video
    .color;
    hdr.mastering = Some(fs_types::media::MasteringDisplay {
        red: [34000, 16000],
        green: [13250, 34500],
        blue: [7500, 3000],
        white: [15635, 16450],
        max_luminance: 10_000_000,
        min_luminance: 50,
    });
    let target = EncodeTarget {
        quality: 20,
        width: 3840,
        height: 2160,
        rate: Rational::new(24, 1),
    };
    let env = EncoderEnv {
        os: Os::Linux,
        vendor: None,
        usable: Default::default(),
        vaapi_device: None,
    };

    let cases = [
        ("hevc_nvenc", VideoCodec::Hevc10),
        ("av1_nvenc", VideoCodec::Av1),
        ("h264_nvenc", VideoCodec::H264),
        ("hevc_amf", VideoCodec::Hevc10),
        ("av1_amf", VideoCodec::Av1),
        ("h264_amf", VideoCodec::H264),
        ("hevc_qsv", VideoCodec::Hevc10),
        ("av1_qsv", VideoCodec::Av1),
        ("h264_qsv", VideoCodec::H264),
        ("hevc_vaapi", VideoCodec::Hevc10),
        ("av1_vaapi", VideoCodec::Av1),
        ("h264_vaapi", VideoCodec::H264),
        ("hevc_videotoolbox", VideoCodec::Hevc10),
        ("prores_videotoolbox", VideoCodec::ProRes422Hq),
        ("libsvtav1", VideoCodec::Av1),
        ("libx265", VideoCodec::Hevc10),
        ("libx264", VideoCodec::H264),
        ("libkvazaar", VideoCodec::Hevc10),
        ("libopenh264", VideoCodec::H264),
    ];
    let mut problems = Vec::new();
    let mut checked = 0;
    for (enc, codec) in cases {
        let h = help(&["-h", &format!("encoder={enc}")]);
        if h.contains("is not recognized") || !h.contains("Encoder ") {
            eprintln!("skip {enc}: not in this build");
            continue;
        }
        let private = parse_options(&h);
        let color = if codec == VideoCodec::H264 {
            probe::parse_streams(r#"{"streams":[{"index":0,"codec_type":"video","width":1920,"height":1080,"pix_fmt":"yuv420p"}],"format":{}}"#, "x").expect("p").video.color
        } else {
            hdr.clone()
        };
        let sel = encoders::encoder_args(enc, codec, &color, target, &env);
        let a = &sel.args;
        let mut i = 0;
        while i < a.len() {
            if let Some(flag) = a[i].strip_prefix('-') {
                let name = flag.split(':').next().unwrap_or(flag);
                let value = a.get(i + 1).map(String::as_str).unwrap_or("");
                match private.get(name).or_else(|| global.get(name)) {
                    None => problems.push(format!("{enc}: unknown option -{name}")),
                    Some(consts)
                        if !consts.is_empty()
                            && value.parse::<f64>().is_err()
                            && !consts.iter().any(|c| c == value) =>
                    {
                        problems.push(format!("{enc}: -{name} {value} is not one of {consts:?}"))
                    }
                    _ => {}
                }
                checked += 1;
                i += 2;
            } else {
                i += 1;
            }
        }
    }
    eprintln!("checked {checked} encoder options");
    assert!(
        problems.is_empty(),
        "invalid encoder options:\n  {}",
        problems.join("\n  ")
    );
}
