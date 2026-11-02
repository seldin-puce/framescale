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

        // RIFE v4.6 (custom rife.Warp layer): identical inputs → (almost) the same frame.
        let mut r = NcnnBackend::new(
            &plan(
                vec![StageConfig::Interpolate {
                    model: m("rife-v4.6"),
                    factor: 2,
                    scene_threshold: 10.0,
                }],
                &models,
            ),
            dev.clone(),
            4,
        )
        .expect("load rife");
        let still = r.interpolate(&src, &src, 0.5, 0).expect("rife still");
        let d_still = mean_abs_diff(&still, &src);
        eprintln!("   rife(a, a) mean |Δ| = {d_still:.3}");
        assert!(
            d_still < 3.0,
            "RIFE should be ~identity for identical frames: {d_still}"
        );
        // Moving content: the midpoint should differ from both ends.
        let moved = test_frame(64, 48, 6);
        let mid = r.interpolate(&src, &moved, 0.5, 0).expect("rife motion");
        assert_eq!(
            (mid.width, mid.height),
            (64, 48),
            "padding to 32 is cropped back"
        );
        assert!(mean_abs_diff(&mid, &src) > 0.5 && mean_abs_diff(&mid, &moved) > 0.5);
    }

    // Unsupported stages are rejected, not silently skipped.
    let dev = DeviceSel {
        index: -1,
        name: "CPU".into(),
        vram_bytes: None,
    };
    let err = NcnnBackend::new(
        &plan(
            vec![StageConfig::Denoise {
                model: m("realcugan-se"),
                strength: 1,
            }],
            &models,
        ),
        dev,
        1,
    )
    .expect_err("denoise unsupported");
    assert!(err.to_string().contains("not available"), "{err}");
    ncnn::shutdown();
}
