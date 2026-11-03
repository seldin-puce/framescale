//! Preview + thumbnail through the real worker, FFmpeg and models.
//! Needs FS_FFMPEG, FS_FFPROBE, FS_WORKER (skips otherwise).

use std::path::PathBuf;
use std::process::Command;

use fs_core::preview::{self, PreviewTools};
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
            },
            StageConfig::Interpolate {
                model: m("rife-v4.6"),
                factor: 2,
                scene_threshold: 10.0,
            },
        ],
    };
    let root = dir.join("previews");
    let r = preview::render(&tools, &sdr, 2.0, 2.0, &pipeline, &root)
        .await
        .expect("preview");
    eprintln!("{r:?}");
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
    let late = preview::render(&tools, &sdr, 5.9, 2.0, &pipeline, &root)
        .await
        .expect("late");
    assert_eq!(late.before_frames, 48);

    let up = PipelineSpec {
        stages: vec![StageConfig::Upscale {
            model: m("realesr-animevideov3"),
            scale: 2,
        }],
    };
    let h = preview::render(&tools, &hdr, 0.0, 2.0, &up, &root)
        .await
        .expect("hdr preview");
    assert!(h.tone_mapped);
    assert_eq!(h.after_frames, h.before_frames);

    // A single-frame "still" (shown first while the full 2 s renders).
    let still = preview::render(&tools, &sdr, 3.0, 0.0, &up, &root)
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
