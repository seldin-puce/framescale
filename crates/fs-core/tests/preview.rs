//! Preview + thumbnail through the real worker, FFmpeg and models.
//! Needs FS_FFMPEG, FS_FFPROBE, FS_WORKER (skips otherwise).

use std::path::{Path, PathBuf};
use std::process::Command;

use fs_core::preview::{self, PreviewTools};
use fs_types::PreviewProgress;
use fs_types::pipeline::{ModelRef, PipelineSpec, StageConfig};

fn env(k: &str) -> Option<PathBuf> {
    std::env::var_os(k).map(Into::into)
}

#[tokio::test(flavor = "multi_thread")]
async fn preview_and_thumbnail() {
    let (Some(ffmpeg), Some(ffprobe), Some(worker)) =
        (env("FS_FFMPEG"), env("FS_FFPROBE"), env("FS_WORKER"))
    else {
        return eprintln!("skip: FS_FFMPEG/FS_FFPROBE/FS_WORKER unset");
    };
    let dir = std::env::temp_dir().join(format!("fs-preview-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("dir");
    let make = |args: &[&str], out: &PathBuf| {
        let st = Command::new(&ffmpeg)
            .args(["-hide_banner", "-loglevel", "error", "-y"])
            .args(args)
            .arg(out)
            .status()
            .expect("gen");
        assert!(st.success());
    };
    let sdr = dir.join("sdr.mkv");
    make(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x180:r=24:d=6",
            "-c:v",
            "libsvtav1",
            "-preset",
            "12",
        ],
        &sdr,
    );
    let hdr = dir.join("hdr.mkv");
    make(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x180:r=24:d=3",
            "-vf",
            "format=yuv420p10le,setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc:range=tv",
            "-c:v",
            "libsvtav1",
            "-preset",
            "12",
        ],
        &hdr,
    );

    let tools = PreviewTools {
        ffmpeg: ffmpeg.clone(),
        ffprobe: ffprobe.clone(),
        worker,
        models_dir: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../apps/desktop/src-tauri/resources/models"),
        downloads_dir: None,
        tile_cache: None,
        device: None,
    };
    let m = |id: &str| ModelRef {
        id: id.into(),
        path: None,
    };
    let pipeline = PipelineSpec {
        stages: vec![
            StageConfig::Upscale {
                model: m("realesr-animevideov3"),
                scale: 2,
                denoise: None,
            },
            StageConfig::Interpolate {
                model: m("rife-v4.6"),
                factor: 2,
                scene_threshold: 10.0,
            },
        ],
    };
    let root = dir.join("previews");
    let seen = std::sync::Mutex::new(Vec::<PreviewProgress>::new());
    let record = |p: PreviewProgress| seen.lock().expect("lock").push(p);
    let r = preview::render(&tools, &sdr, 2.0, 2.0, &pipeline, &root, &record)
        .await
        .expect("preview");
    eprintln!("{r:?}");
    // Progress: known total up front, then engine start, ending complete.
    let seen = seen.into_inner().expect("progress");
    assert!(
        seen.iter().all(|p| p.total == 48 && p.done <= 48),
        "{seen:?}"
    );
    assert!(!seen[0].started, "the first report precedes the engine");
    assert!(seen.iter().any(|p| p.started), "{seen:?}");
    let last = seen.last().expect("reports");
    assert_eq!((last.done, last.total), (48, 48), "{seen:?}");
    assert_eq!(r.before_frames, 48, "2 s at 24 fps");
    assert_eq!(r.after_frames, 96, "×2 interpolation");
    assert_eq!((r.width, r.height), (640, 360));
    assert!((r.start_s - 2.0).abs() < 0.05);
    assert!((r.fps - 48.0).abs() < 1e-9);
    assert!(!r.tone_mapped);
    let first = root
        .join(r.prefix.trim_start_matches("preview/"))
        .join("after/0000.jpg");
    assert!(std::fs::metadata(&first).expect("jpeg").len() > 1000);

    // Near the end: the window is clamped so a full 2 s still renders.
    let late = preview::render(&tools, &sdr, 5.9, 2.0, &pipeline, &root, &|_| {})
        .await
        .expect("late");
    assert_eq!(late.before_frames, 48);

    let up = PipelineSpec {
        stages: vec![StageConfig::Upscale {
            model: m("realesr-animevideov3"),
            scale: 2,
            denoise: None,
        }],
    };
    let h = preview::render(&tools, &hdr, 0.0, 2.0, &up, &root, &|_| {})
        .await
        .expect("hdr preview");
    assert!(h.tone_mapped);
    assert_eq!(h.after_frames, h.before_frames);

    // Denoise (Real-CUGAN) runs at the input size, chained before Upscale.
    let dn = PipelineSpec {
        stages: vec![
            StageConfig::Denoise {
                model: m("realcugan-se"),
                strength: 2,
            },
            StageConfig::Upscale {
                model: m("realesr-animevideov3"),
                scale: 2,
                denoise: None,
            },
        ],
    };
    let d = preview::render(&tools, &sdr, 1.0, 0.5, &dn, &root, &|_| {})
        .await
        .expect("denoise preview");
    assert_eq!(
        (d.width, d.height),
        (640, 360),
        "denoise keeps size, upscale ×2"
    );
    assert_eq!(d.after_frames, d.before_frames);

    // The newer models through the real worker: NAFNet denoise, General v3
    // with its noise blend, RIFE 4.25 (frames padded to 64).
    let newer = PipelineSpec {
        stages: vec![
            StageConfig::Denoise {
                model: m("nafnet-sidd"),
                strength: 2,
            },
            StageConfig::Upscale {
                model: m("realesr-general-x4v3"),
                scale: 2,
                denoise: Some(0.3),
            },
            StageConfig::Interpolate {
                model: m("rife-v4.25"),
                factor: 2,
                scene_threshold: 10.0,
            },
        ],
    };
    let n = preview::render(&tools, &sdr, 1.0, 0.5, &newer, &root, &|_| {})
        .await
        .expect("newer models preview");
    assert_eq!((n.width, n.height), (640, 360));
    assert_eq!(n.after_frames, n.before_frames * 2, "×2 interpolation");

    // A single-frame "still" (shown first while the full 2 s renders).
    let still = preview::render(&tools, &sdr, 3.0, 0.0, &up, &root, &|_| {})
        .await
        .expect("still");
    assert_eq!((still.before_frames, still.after_frames), (1, 1));

    preview::prune(&root, 1);
    assert_eq!(std::fs::read_dir(&root).expect("rd").count(), 1);

    let thumbs = dir.join("thumbs");
    let t1 = preview::thumbnail(&ffmpeg, &ffprobe, &hdr, &thumbs)
        .await
        .expect("thumb");
    assert!(t1.starts_with("thumbs/") && t1.ends_with(".jpg"));
    let t2 = preview::thumbnail(&ffmpeg, &ffprobe, &hdr, &thumbs)
        .await
        .expect("cached");
    assert_eq!(t1, t2);
    assert_eq!(
        std::fs::read_dir(&thumbs).expect("rd").count(),
        1,
        "no temp files left"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// Mean luma of a JPEG, decoded by FFmpeg (1×1 area average).
fn luma(ffmpeg: &Path, jpeg: &[u8], dir: &Path) -> u8 {
    let f = dir.join("probe.jpg");
    std::fs::write(&f, jpeg).expect("write jpeg");
    let o = Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(&f)
        .args([
            "-vf",
            "scale=1:1:flags=area,format=gray",
            "-f",
            "rawvideo",
            "-",
        ])
        .output()
        .expect("decode");
    assert!(
        o.status.success() && o.stdout.len() == 1,
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    o.stdout[0]
}

/// Compare sessions: both sides are sampled on the output's frame grid, stay
/// frame-exact after seeks (also with millisecond MKV timestamps and a lower
/// source rate), serve native-resolution stills, and stop at the real end.
#[tokio::test(flavor = "multi_thread")]
async fn compare_session_streams_aligned_frames() {
    use fs_core::compare::{CompareSession, FrameError};
    use fs_types::CompareSide::{After, Before};

    let (Some(ffmpeg), Some(ffprobe)) = (env("FS_FFMPEG"), env("FS_FFPROBE")) else {
        return eprintln!("skip: FS_FFMPEG/FS_FFPROBE unset");
    };
    let dir = std::env::temp_dir().join(format!("fs-compare-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("dir");
    let make = |args: &[&str], out: &PathBuf| {
        let st = Command::new(&ffmpeg)
            .args(["-hide_banner", "-loglevel", "error", "-y"])
            .args(args)
            .arg(out)
            .status()
            .expect("gen");
        assert!(st.success());
    };
    // Each frame's luma encodes its index: code = N mod 11.
    let coded = |s: &str, r: u32| {
        format!("nullsrc=s={s}:r={r}:d=3,format=yuv420p,geq=lum='16+20*mod(N,11)':cb=128:cr=128")
    };
    // Source 320×180 @ 24 fps (MP4); output 640×360 @ 48 fps (MKV, 1 ms timestamps).
    let src = dir.join("src.mp4");
    make(
        &[
            "-f",
            "lavfi",
            "-i",
            &coded("320x180", 24),
            "-c:v",
            "libx264",
            "-qp",
            "0",
            "-color_range",
            "tv",
        ],
        &src,
    );
    let out = dir.join("out.mkv");
    make(
        &[
            "-f",
            "lavfi",
            "-i",
            &coded("640x360", 48),
            "-c:v",
            "libx264",
            "-qp",
            "0",
            "-color_range",
            "tv",
        ],
        &out,
    );
    // Reference level of each code: the source's first 11 frames, decoded directly.
    let reference = Command::new(&ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(&src)
        .args([
            "-frames:v",
            "11",
            "-vf",
            "scale=1:1:flags=area,format=gray",
            "-f",
            "rawvideo",
            "-",
        ])
        .output()
        .expect("reference")
        .stdout;
    assert_eq!(reference.len(), 11);
    let index = |l: u8| {
        (0..11i64)
            .min_by_key(|&k| (i16::from(reference[k as usize]) - i16::from(l)).abs())
            .unwrap_or(-1)
    };

    let s = CompareSession::open(&ffmpeg, &ffprobe, &src, &out, 4000)
        .await
        .expect("open");
    let i = s.info.clone();
    assert_eq!((i.width, i.height), (640, 360), "capped to the output size");
    assert_eq!((i.output_width, i.source_width), (640, 320));
    assert!((i.fps - 48.0).abs() < 1e-9 && (i.source_fps - 24.0).abs() < 1e-9);
    assert_eq!(i.frames, 144);

    // Output frame n ↔ output index n; source shows frame floor(n / 2).
    let check = |n: u64, b: &[u8], a: &[u8]| {
        assert_eq!(index(luma(&ffmpeg, a, &dir)), (n % 11) as i64, "after #{n}");
        assert_eq!(
            index(luma(&ffmpeg, b, &dir)),
            ((n / 2) % 11) as i64,
            "before #{n}"
        );
    };
    // Sequential playback from a seek to an odd frame, then a far seek back.
    for n in (37..60).chain([5, 6, 7, 120, 143]) {
        let (b, a) = tokio::join!(s.frame(Before, n), s.frame(After, n));
        check(n, &b.expect("before"), &a.expect("after"));
    }
    // Native-resolution stills match the stream.
    for n in [37, 101] {
        let (b, a) = tokio::join!(s.still(Before, n), s.still(After, n));
        check(n, &b.expect("still before"), &a.expect("still after"));
    }
    assert_eq!(s.frame(After, 144).await, Err(FrameError::PastEnd));

    // A request superseded by a far seek gives up instead of fighting it.
    let (late, _) = tokio::join!(s.frame(After, 10), async {
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        s.frame(After, 130).await
    });
    assert!(
        matches!(late, Ok(_) | Err(FrameError::Superseded)),
        "{late:?}"
    );

    s.close();
    assert_eq!(s.frame(After, 0).await, Err(FrameError::Closed));

    // A clip that hasn't been rendered: the source alone, on its own grid,
    // with one decoder behind both sides.
    let s = CompareSession::open_source(&ffmpeg, &ffprobe, &src, 4000)
        .await
        .expect("open source");
    let i = s.info.clone();
    assert_eq!((i.width, i.height, i.frames), (320, 180, 72));
    assert!((i.fps - 24.0).abs() < 1e-9);
    for n in (30..40).chain([3, 71]) {
        let (b, a) = tokio::join!(s.frame(Before, n), s.frame(After, n));
        let (b, a) = (b.expect("before"), a.expect("after"));
        assert_eq!(b, a, "one decoder serves both sides");
        assert_eq!(
            index(luma(&ffmpeg, &b, &dir)),
            (n % 11) as i64,
            "source #{n}"
        );
    }
    let full = s.still(Before, 12).await.expect("still");
    assert_eq!(index(luma(&ffmpeg, &full, &dir)), 1, "source #12");
    assert_eq!(s.frame(Before, 72).await, Err(FrameError::PastEnd));
    let _ = std::fs::remove_dir_all(dir);
}
