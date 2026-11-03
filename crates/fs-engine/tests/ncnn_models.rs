//! Runs the real bundled models through ncnn: CPU path always, Vulkan when a
//! device exists (lavapipe in CI containers). Needs `cargo xtask models`.
#![cfg(feature = "ncnn")]

use std::path::PathBuf;

use fs_engine::ncnn::{self, DeviceSel, NcnnBackend};
use fs_engine::{Frame, InferenceBackend};
use fs_types::pipeline::{ModelRef, StageConfig};
use fs_types::worker::{EnginePlan, FrameFormat};

fn models_dir() -> Option<PathBuf> {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../apps/desktop/src-tauri/resources/models");
    d.join("realesrgan").exists().then_some(d)
}

fn plan(stages: Vec<StageConfig>, models: &std::path::Path) -> EnginePlan {
    EnginePlan {
        stages,
        scene_cuts: vec![],
        in_width: 64,
        in_height: 48,
        out_width: 128,
        out_height: 96,
        frame_format: FrameFormat::Rgb24,
        device: None,
        tile: None,
        models_dir: models.to_string_lossy().into(),
    }
}

/// Smooth gradients + a hard diagonal edge: something an SR model must sharpen.
fn test_frame(w: u32, h: u32, shift: u32) -> Frame {
    let mut d = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let xs = x + shift;
            let edge = if xs + y > (w + h) / 2 { 220 } else { 30 };
            d.extend([(xs * 255 / (w + shift)) as u8, edge, (y * 255 / h) as u8]);
        }
    }
    Frame::new(w, h, FrameFormat::Rgb24, d)
}

fn mean_abs_diff(a: &Frame, b: &Frame) -> f32 {
    assert_eq!((a.width, a.height), (b.width, b.height));
    a.data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| (f32::from(*x) - f32::from(*y)).abs())
        .sum::<f32>()
        / a.data.len() as f32
}

fn devices() -> Vec<DeviceSel> {
    let mut v = vec![DeviceSel {
        index: -1,
        name: "CPU".into(),
        vram_bytes: None,
    }];
    if ncnn::init(None) > 0 {
        let g = &ncnn::gpus()[0];
        eprintln!("vulkan device: {} (kind {})", g.name, g.kind);
        v.push(DeviceSel {
            index: g.index,
            name: g.name.clone(),
            vram_bytes: Some(1 << 30),
        });
    }
    v
}

#[test]
fn real_models_on_every_device() {
    let Some(models) = models_dir() else {
        return eprintln!("skip: run `cargo xtask models`");
    };
    let m = |id: &str| ModelRef {
        id: id.into(),
        path: None,
    };
    let src = test_frame(64, 48, 0);

    for dev in devices() {
        eprintln!("== {}", dev.name);
        // AnimeVideo-v3 ×2 (native ×2).
        let mut b = NcnnBackend::new(
            &plan(
                vec![StageConfig::Upscale {
                    model: m("realesr-animevideov3"),
                    scale: 2,
                }],
                &models,
            ),
            dev.clone(),
            4,
        )
        .expect("load animevideov3");
        let big = b.process(&src, 512).expect("sr");
        assert_eq!((big.width, big.height), (128, 96));
        // Tiled vs whole-frame: 10 px context keeps seams to rounding noise.
        let tiled = b.process(&src, 24).expect("tiled sr");
        let seam = mean_abs_diff(&big, &tiled);
        eprintln!("   tile seam mean |Δ| = {seam:.3}");
        assert!(seam < 1.0, "visible tile seams: {seam}");
        // It must actually do something beyond pixel duplication.
        let nn = fs_engine::cpu::ReferenceBackend::new(&[StageConfig::Upscale {
            model: m("x"),
            scale: 2,
        }])
        .process(&src, 0)
        .expect("nn");
        assert!(
            mean_abs_diff(&big, &nn) > 1.0,
            "SR output equals nearest-neighbour"
        );

        // x4plus at ×2: native ×4 then exact area downscale.
        let mut b4 = NcnnBackend::new(
            &plan(
                vec![StageConfig::Upscale {
                    model: m("realesrgan-x4plus"),
                    scale: 2,
                }],
                &models,
            ),
            dev.clone(),
            4,
        )
        .expect("load x4plus");
        let o = b4.process(&src, 512).expect("x4plus");
        assert_eq!((o.width, o.height), (128, 96));

        // RIFE (custom rife.Warp layer), every bundled version: identical
        // inputs → (almost) the same frame; a 6 px move → close to the 3 px
        // midpoint; padding (32, or 64 for 4.25+) is cropped back.
        let moved = test_frame(64, 48, 6);
        let halfway = test_frame(64, 48, 3);
        for id in ["rife-v4.6", "rife-v4.22-lite", "rife-v4.25", "rife-v4.26"] {
            let mut r = NcnnBackend::new(
                &plan(
                    vec![StageConfig::Interpolate {
                        model: m(id),
                        factor: 2,
                        scene_threshold: 10.0,
                    }],
                    &models,
                ),
                dev.clone(),
                4,
            )
            .unwrap_or_else(|e| panic!("load {id}: {e}"));
            let still = r.interpolate(&src, &src, 0.5, 0).expect("rife still");
            let d_still = mean_abs_diff(&still, &src);
            let mid = r.interpolate(&src, &moved, 0.5, 0).expect("rife motion");
            assert_eq!(
                (mid.width, mid.height),
                (64, 48),
                "{id}: padding cropped back"
            );
            let (d_a, d_b, d_half) = (
                mean_abs_diff(&mid, &src),
                mean_abs_diff(&mid, &moved),
                mean_abs_diff(&mid, &halfway),
            );
            eprintln!(
                "   {id}: rife(a, a) |Δ| {d_still:.3}; midpoint |Δ| to a {d_a:.2}, b {d_b:.2}, true midpoint {d_half:.2}"
            );
            assert!(
                d_still < 3.0,
                "{id}: ~identity for identical frames: {d_still}"
            );
            assert!(d_a > 0.5 && d_b > 0.5, "{id}: the midpoint moves");
            assert!(d_half < d_a.min(d_b), "{id}: closer to the true midpoint");
        }

        // Inside the device loop: ncnn::shutdown() below ends Vulkan for good.
        check_real_cugan(&dev, &models);
        check_face_fix(&dev, &models);
    }

    // Face Fix runs the bundled CodeFormer only: other folders are rejected.
    let dev = DeviceSel {
        index: -1,
        name: "CPU".into(),
        vram_bytes: None,
    };
    let err = NcnnBackend::new(
        &plan(
            vec![StageConfig::FaceFix {
                model: ModelRef {
                    id: "codeformer".into(),
                    path: Some("/elsewhere/codeformer".into()),
                },
                fidelity: 0.5,
            }],
            &models,
        ),
        dev,
        1,
    )
    .expect_err("custom face model");
    assert!(err.to_string().contains("not supported"), "{err}");
    ncnn::shutdown();
}

/// Deterministic noise (LCG), roughly ±`amp` levels.
fn noisy(f: &Frame, amp: i32) -> Frame {
    let mut state = 0x2545_f491_u32;
    let data = f
        .data
        .iter()
        .map(|&v| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let n = ((state >> 16) % (2 * amp as u32 + 1)) as i32 - amp;
            (i32::from(v) + n).clamp(0, 255) as u8
        })
        .collect();
    Frame::new(f.width, f.height, f.format, data)
}

fn mean(f: &Frame) -> f32 {
    f.data.iter().map(|&v| f32::from(v)).sum::<f32>() / f.data.len() as f32
}

fn max_abs_diff(a: &Frame, b: &Frame) -> u8 {
    a.data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0)
}

/// Denoise (Real-CUGAN): removes noise at the input size, keeps brightness
/// (SE and Pro), and is seam-free: small tiles with the injected whole-frame
/// SE statistics match a single tile.
fn check_real_cugan(dev: &DeviceSel, models: &std::path::Path) {
    if !models.join("realcugan").exists() {
        return eprintln!("   skip Real-CUGAN: models missing");
    }
    let stage = |id: &str, strength: u8| StageConfig::Denoise {
        model: ModelRef {
            id: id.into(),
            path: None,
        },
        strength,
    };
    let clean = test_frame(160, 120, 0);
    let dirty = noisy(&clean, 24);
    let mut b = NcnnBackend::new(
        &plan(vec![stage("realcugan-se", 3)], models),
        dev.clone(),
        4,
    )
    .expect("load realcugan-se denoise3x");
    let whole = b.process(&dirty, 256).expect("cugan one tile");
    let tiled = b.process(&dirty, 32).expect("cugan 20 tiles");
    assert_eq!(
        (whole.width, whole.height),
        (160, 120),
        "denoise keeps the size"
    );
    let (before, after) = (mean_abs_diff(&dirty, &clean), mean_abs_diff(&whole, &clean));
    let seam = (mean_abs_diff(&whole, &tiled), max_abs_diff(&whole, &tiled));
    eprintln!(
        "   real-cugan se L3: |noisy-clean| {before:.2} -> |out-clean| {after:.2}; tiles vs whole mean {:.3} max {}",
        seam.0, seam.1
    );
    assert!(
        after < before * 0.6,
        "denoise must remove most of the noise: {before} -> {after}"
    );
    assert!((mean(&whole) - mean(&clean)).abs() < 3.0, "brightness kept");
    assert!(
        seam.0 < 0.25 && seam.1 <= 4,
        "tiling must be seam-free: {seam:?}"
    );
    // Odd sizes: the network needs even tiles; the extra column/row is cropped.
    let odd = noisy(&test_frame(161, 119, 0), 24);
    let a = b.process(&odd, 256).expect("odd whole");
    let c = b.process(&odd, 32).expect("odd tiled");
    assert_eq!((a.width, a.height, c.width, c.height), (161, 119, 161, 119));
    let d = mean_abs_diff(&a, &c);
    assert!(d < 0.25, "odd tiles seam-free: {d}");
    assert!(
        mean_abs_diff(&a, &test_frame(161, 119, 0)) < 5.0,
        "odd frame denoised"
    );

    for (id, strength) in [
        ("realcugan-se", 0),
        ("realcugan-pro", 0),
        ("realcugan-pro", 3),
    ] {
        let mut b = NcnnBackend::new(&plan(vec![stage(id, strength)], models), dev.clone(), 4)
            .expect("load realcugan variant");
        let out = b.process(&clean, 256).expect("cugan");
        let (dm, d) = (
            (mean(&out) - mean(&clean)).abs(),
            mean_abs_diff(&out, &clean),
        );
        eprintln!("   {id} s{strength}: mean shift {dm:.2}, |out-clean| {d:.2}");
        assert!(
            dm < 3.0 && d < 12.0,
            "{id} s{strength} must keep the image: shift {dm}, diff {d}"
        );
    }
}

fn env_path(k: &str) -> Option<PathBuf> {
    std::env::var_os(k)
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

/// Decode an image to RGB24 with FFmpeg (`FS_FFMPEG` / `FS_FFPROBE`).
fn load_image(path: &std::path::Path) -> Option<Frame> {
    let (ff, fp) = (env_path("FS_FFMPEG")?, env_path("FS_FFPROBE")?);
    let dims = std::process::Command::new(fp)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .ok()?;
    let dims = String::from_utf8_lossy(&dims.stdout);
    let (w, h) = dims.trim().split_once(',')?;
    let (w, h): (u32, u32) = (w.parse().ok()?, h.parse().ok()?);
    let raw = std::process::Command::new(ff)
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .output()
        .ok()?;
    (raw.stdout.len() == (w * h * 3) as usize)
        .then(|| Frame::new(w, h, FrameFormat::Rgb24, raw.stdout))
}

fn save_png(f: &Frame, path: &std::path::Path) {
    let Some(ff) = env_path("FS_FFMPEG") else {
        return;
    };
    let mut child = std::process::Command::new(ff)
        .args([
            "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
        ])
        .arg(format!("{}x{}", f.width, f.height))
        .args(["-i", "-"])
        .arg(path)
        .stdin(std::process::Stdio::piped())
        .spawn()
        .expect("ffmpeg");
    use std::io::Write;
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(&f.data)
        .expect("write");
    let _ = child.wait();
}

/// Mean per-channel value inside a rectangle.
fn region_mean(f: &Frame, (x0, y0, x1, y1): (u32, u32, u32, u32), c: usize) -> f32 {
    let mut s = 0.0;
    for y in y0..y1 {
        for x in x0..x1 {
            s += f32::from(f.data[((y * f.width + x) * 3) as usize + c]);
        }
    }
    s / ((x1 - x0) * (y1 - y0)) as f32
}

fn region_diff(a: &Frame, b: &Frame, (x0, y0, x1, y1): (u32, u32, u32, u32)) -> f32 {
    let mut s = 0.0;
    for y in y0..y1 {
        for x in x0..x1 {
            for c in 0..3 {
                let i = ((y * a.width + x) * 3) as usize + c;
                s += (f32::from(a.data[i]) - f32::from(b.data[i])).abs();
            }
        }
    }
    s / ((x1 - x0) * (y1 - y0) * 3) as f32
}

/// Face Fix with the bundled CodeFormer. Always: it loads, and a frame with
/// no faces passes through bit-identical. With a real photo (`FS_FACE_IMAGE`;
/// one frontal face around x 150–330, y 40–230 in the 600×337 test still):
/// the face is restored, colours are kept, the background stays
/// bit-identical and the fidelity setting matters.
fn check_face_fix(dev: &DeviceSel, models: &std::path::Path) {
    let stage = |fidelity: f32| StageConfig::FaceFix {
        model: ModelRef {
            id: "codeformer".into(),
            path: None,
        },
        fidelity,
    };
    let run = |src: &Frame, fidelity: f32| {
        let mut b = NcnnBackend::new(&plan(vec![stage(fidelity)], models), dev.clone(), 4)
            .expect("load face fix");
        let t = std::time::Instant::now();
        let out = b.process(src, 256).expect("face fix");
        (out, b.last_faces(), t.elapsed())
    };
    let plain = test_frame(64, 48, 0);
    let (out, faces, _) = run(&plain, 0.5);
    assert_eq!(faces, 0, "no face in a gradient");
    assert_eq!(
        (out.width, out.height, &out.data),
        (plain.width, plain.height, &plain.data),
        "frames without faces pass through"
    );

    let Some(img_path) = env_path("FS_FACE_IMAGE") else {
        return eprintln!("   skip Face Fix on a photo: set FS_FACE_IMAGE");
    };
    let Some(src) = load_image(&img_path) else {
        return eprintln!("   skip Face Fix: can't decode the image (FS_FFMPEG/FS_FFPROBE)");
    };
    let (out, faces, took) = run(&src, 0.5);
    let face = (170, 60, 310, 220);
    let d_face = region_diff(&out, &src, face);
    let corner = (src.width - 60, 0, src.width, 40);
    eprintln!("   face fix: {faces} face(s) in {took:.2?}; face Δ {d_face:.2}");
    if let Some(dir) = img_path.parent() {
        save_png(
            &out,
            &dir.join(format!(
                "fixed-{}.png",
                if dev.index >= 0 { "gpu" } else { "cpu" }
            )),
        );
    }
    assert_eq!(faces, 1, "one frontal face");
    assert!(d_face > 1.0, "the face must change: {d_face}");
    assert!(d_face < 25.0, "restored, not replaced: {d_face}");
    for c in 0..3 {
        let (a, b) = (region_mean(&src, face, c), region_mean(&out, face, c));
        assert!(
            (a - b).abs() < 12.0,
            "channel {c} mean {a} -> {b}: colours must be kept"
        );
    }
    assert_eq!(region_diff(&out, &src, corner), 0.0, "background untouched");
    let (sharp, _, _) = run(&src, 0.0);
    let (faithful, _, _) = run(&src, 1.0);
    let d_fid = region_diff(&sharp, &faithful, face);
    eprintln!("   fidelity 0 vs 1: face Δ {d_fid:.2}");
    assert!(d_fid > 0.5, "fidelity must change the result: {d_fid}");
}
