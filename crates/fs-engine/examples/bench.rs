//! `cargo run --release -p fs-engine --features ncnn --example bench [-- <device-index>…]`
//! Real-model throughput per device (synthetic frames; inference + conversion, no I/O).
use std::path::PathBuf;
use std::time::Instant;

use fs_engine::ncnn::{self, DeviceSel, NcnnBackend};
use fs_engine::{Frame, InferenceBackend};
use fs_types::pipeline::{ModelRef, StageConfig};
use fs_types::worker::{EnginePlan, FrameFormat};

fn frame(w: u32, h: u32, phase: u32) -> Frame {
    let mut d = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let v = ((x + phase) ^ y) as u8;
            d.extend([v, (x * 3 + y) as u8, (y * 2 + phase) as u8]);
        }
    }
    Frame::new(w, h, FrameFormat::Rgb24, d)
}

fn plan(stage: StageConfig, models: &std::path::Path) -> EnginePlan {
    EnginePlan {
        stages: vec![stage],
        scene_cuts: vec![],
        in_width: 0,
        in_height: 0,
        out_width: 0,
        out_height: 0,
        frame_format: FrameFormat::Rgb24,
        device: None,
        tile: None,
        models_dir: models.to_string_lossy().into(),
    }
}

fn main() {
    let models = std::env::var_os("FS_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../apps/desktop/src-tauri/resources/models")
        });
    let n = ncnn::init(None);
    let gpus = ncnn::gpus();
    let wanted: Vec<i32> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let devices: Vec<&ncnn::GpuEntry> = gpus
        .iter()
        .filter(|g| g.kind != 3 && (wanted.is_empty() || wanted.contains(&g.index)))
        .collect();
    println!("{n} Vulkan device(s)");
    let m = |id: &str| ModelRef {
        id: id.into(),
        path: None,
    };

    for g in devices {
        println!("\n== [{}] {}", g.index, g.name);
        let dev = DeviceSel {
            index: g.index,
            name: g.name.clone(),
            vram_bytes: Some(16 << 30),
        };
        let cases: Vec<(&str, StageConfig, (u32, u32), usize)> = vec![
            (
                "AnimeVideo-v3 ×2   1920×1080 → 3840×2160",
                StageConfig::Upscale {
                    model: m("realesr-animevideov3"),
                    scale: 2,
                },
                (1920, 1080),
                6,
            ),
            (
                "Real-ESRGAN x4+ ×4  960×540 → 3840×2160",
                StageConfig::Upscale {
                    model: m("realesrgan-x4plus"),
                    scale: 4,
                },
                (960, 540),
                3,
            ),
            (
                "Real-ESRGAN x4+ ×2 1920×1080 → 3840×2160",
                StageConfig::Upscale {
                    model: m("realesrgan-x4plus"),
                    scale: 2,
                },
                (1920, 1080),
                2,
            ),
            (
                "Real-CUGAN SE denoise  720×576",
                StageConfig::Denoise {
                    model: m("realcugan-se"),
                    strength: 2,
                },
                (720, 576),
                6,
            ),
            (
                "Real-CUGAN SE denoise 1920×1080",
                StageConfig::Denoise {
                    model: m("realcugan-se"),
                    strength: 2,
                },
                (1920, 1080),
                3,
            ),
            (
                "RIFE v4.6           1920×1080",
                StageConfig::Interpolate {
                    model: m("rife-v4.6"),
                    factor: 2,
                    scene_threshold: 10.0,
                },
                (1920, 1080),
                10,
            ),
            (
                "RIFE v4.22 lite     1920×1080",
                StageConfig::Interpolate {
                    model: m("rife-v4.22-lite"),
                    factor: 2,
                    scene_threshold: 10.0,
                },
                (1920, 1080),
                10,
            ),
            (
                "RIFE v4.25          1920×1080",
                StageConfig::Interpolate {
                    model: m("rife-v4.25"),
                    factor: 2,
                    scene_threshold: 10.0,
                },
                (1920, 1080),
                10,
            ),
            (
                "RIFE v4.26          1920×1080",
                StageConfig::Interpolate {
                    model: m("rife-v4.26"),
                    factor: 2,
                    scene_threshold: 10.0,
                },
                (1920, 1080),
                10,
            ),
            (
                "RIFE v4.6           3840×2160",
                StageConfig::Interpolate {
                    model: m("rife-v4.6"),
                    factor: 2,
                    scene_threshold: 10.0,
                },
                (3840, 2160),
                4,
            ),
        ];
        for (label, stage, (w, h), iters) in cases {
            let interp = matches!(stage, StageConfig::Interpolate { .. });
            let mut b = match NcnnBackend::new(&plan(stage, &models), dev.clone(), 8) {
                Ok(b) => b,
                Err(e) => {
                    println!("  {label}: load failed: {e}");
                    continue;
                }
            };
            let tile = std::env::var("FS_TILE")
                .ok()
                .and_then(|t| t.parse().ok())
                .unwrap_or(b.max_tile());
            let (a, c) = (frame(w, h, 0), frame(w, h, 7));
            let run = |b: &mut NcnnBackend| {
                if interp {
                    b.interpolate(&a, &c, 0.5, tile)
                } else {
                    b.process(&a, tile)
                }
            };
            if let Err(e) = run(&mut b) {
                println!("  {label}: {e}");
                continue;
            }
            let t0 = Instant::now();
            for _ in 0..iters {
                run(&mut b).expect("run");
            }
            let per = t0.elapsed().as_secs_f64() / iters as f64;
            println!(
                "  {label}: {:>7.1} ms/frame  {:>6.2} fps   (tile {tile})",
                per * 1e3,
                1.0 / per
            );
        }
    }
    // CPU path (`FS_BENCH_CPU=1`): ncnn's own threads vs one tile per core
    // (`FS_CPU_TILE_WORKERS`, the Windows/MSVC mode).
    if std::env::var_os("FS_BENCH_CPU").is_some() {
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get() as i32);
        println!("\n== CPU ({threads} threads available)");
        let cpu = DeviceSel {
            index: -1,
            name: "CPU".into(),
            vram_bytes: None,
        };
        for (label, stage) in [
            (
                "AnimeVideo-v3 ×2   640×360",
                StageConfig::Upscale {
                    model: m("realesr-animevideov3"),
                    scale: 2,
                },
            ),
            (
                "Real-CUGAN SE denoise 640×360",
                StageConfig::Denoise {
                    model: m("realcugan-se"),
                    strength: 2,
                },
            ),
        ] {
            let mut b =
                NcnnBackend::new(&plan(stage, &models), cpu.clone(), threads).expect("cpu load");
            let a = frame(640, 360, 0);
            let tile = std::env::var("FS_TILE")
                .ok()
                .and_then(|t| t.parse().ok())
                .unwrap_or(b.max_tile());
            b.process(&a, tile).expect("warm-up");
            let t0 = Instant::now();
            for _ in 0..2 {
                b.process(&a, tile).expect("run");
            }
            let per = t0.elapsed().as_secs_f64() / 2.0;
            println!(
                "  {label}: {:>7.1} ms/frame  {:>6.2} fps",
                per * 1e3,
                1.0 / per
            );
        }
    }
    ncnn::shutdown();
}
