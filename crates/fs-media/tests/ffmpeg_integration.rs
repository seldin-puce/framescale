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
