//! Core ML vs. ncnn on the same network and frames (macOS, Apple Silicon).
//!
//! ```text
//! coreml_compare <frames.rgb> <w> <h>
//! ```
//!
//! Runs Real-ESRGAN General v3 ×2 (default denoise) over every RGB24 frame in
//! the file, first with ncnn (Vulkan through MoltenVK when `FS_VULKAN_DRIVER`
//! points at `libMoltenVK.dylib`, else ncnn's CPU path), then with Core ML on
//! each compute-unit setting. Prints frames per second and the PSNR of each
//! Core ML output against ncnn's. Models: `FS_MODELS` (default: the bundled
//! folder), with the converted Core ML package in `<FS_MODELS>/coreml/`
//! (scripts/convert_coreml.py). Tile reuse is off: every frame is computed.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() {
    use std::path::PathBuf;
    use std::time::Instant;

    use fs_engine::apple::AppleBackend;
    use fs_engine::ncnn::{self, DeviceSel, NcnnBackend};
    use fs_engine::{Frame, InferenceBackend};
    use fs_types::pipeline::{ModelRef, StageConfig};
    use fs_types::worker::{EnginePlan, FrameFormat};

    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: coreml_compare <frames.rgb> <w> <h>");
        std::process::exit(2);
    }
    let (w, h): (u32, u32) = (a[2].parse().expect("width"), a[3].parse().expect("height"));
    let data = std::fs::read(&a[1]).expect("read frames");
    let size = (w * h * 3) as usize;
    assert!(
        !data.is_empty() && data.len().is_multiple_of(size),
        "input must be {w}×{h} RGB24 frames"
    );
    let frames: Vec<Frame> = data
        .chunks(size)
        .map(|c| Frame::new(w, h, FrameFormat::Rgb24, c.to_vec()))
        .collect();
    let models = std::env::var_os("FS_MODELS").map_or_else(
        || {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../apps/desktop/src-tauri/resources/models")
        },
        PathBuf::from,
    );
    let plan = EnginePlan {
        stages: vec![StageConfig::Upscale {
            model: ModelRef {
                id: "realesr-general-x4v3".into(),
                path: None,
            },
            scale: 2,
            denoise: None,
        }],
        scene_cuts: vec![],
        in_width: w,
        in_height: h,
        out_width: w * 2,
        out_height: h * 2,
        frame_format: FrameFormat::Rgb24,
        device: None,
        tile: None,
        models_dir: models.to_string_lossy().into(),
        downloads_dir: None,
        tile_cache: None,
        backend: None,
    };

    // Run every frame (after one warm-up) and time it.
    let run = |b: &mut dyn InferenceBackend, tile: u32| -> (Vec<Frame>, f64) {
        b.set_reuse(false);
        b.process(&frames[0], tile).expect("warm-up");
        let t0 = Instant::now();
        let out: Vec<Frame> = frames
            .iter()
            .map(|f| b.process(f, tile).expect("process"))
            .collect();
        (out, frames.len() as f64 / t0.elapsed().as_secs_f64())
    };
    let psnr = |a: &[Frame], b: &[Frame]| -> f64 {
        let (mut se, mut n) = (0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            for (p, q) in x.data.iter().zip(&y.data) {
                let d = f64::from(*p) - f64::from(*q);
                se += d * d;
                n += 1.0;
            }
        }
        if se == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0f64 * 255.0 / (se / n)).log10()
        }
    };

    let driver = std::env::var_os("FS_VULKAN_DRIVER").map(PathBuf::from);
    let device = if ncnn::init(driver.as_deref()) > 0 {
        let g = ncnn::gpus().into_iter().next().expect("gpu");
        DeviceSel {
            index: g.index,
            name: g.name,
            vram_bytes: None,
        }
    } else {
        DeviceSel {
            index: -1,
            name: "CPU".into(),
            vram_bytes: None,
        }
    };
    let ncnn_name = device.name.clone();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get() as i32);
    let mut nb = NcnnBackend::new(&plan, device, threads).expect("ncnn");
    let tile = nb.max_tile();
    let (reference, ncnn_fps) = run(&mut nb, tile);
    drop(nb);
    println!("{} frames of {w}×{h}, General v3 ×2", frames.len());
    println!("{:<34} {:>8} {:>16}", "backend", "fps", "PSNR vs ncnn");
    println!(
        "{:<34} {:>8.2} {:>16}",
        format!("ncnn ({ncnn_name}, tile {tile})"),
        ncnn_fps,
        "—"
    );

    for (units, label) in [
        ("all", "Core ML, all units"),
        ("gpu", "Core ML, CPU+GPU"),
        ("ane", "Core ML, CPU+Neural Engine"),
    ] {
        // SAFETY of set_var: single-threaded at this point (no backend alive).
        unsafe { std::env::set_var("FS_COREML_UNITS", units) };
        let mut cb = match AppleBackend::new(&plan) {
            Ok(b) => b,
            Err(e) => {
                println!("{label:<34} unavailable: {e}");
                continue;
            }
        };
        let tile = cb.max_tile();
        let (out, fps) = run(&mut cb, tile);
        let mem = cb.stats();
        println!(
            "{:<34} {:>8.2} {:>13.2} dB   (Metal: {} MB allocated, budget {} MB)",
            format!("{label} (tile {tile})"),
            fps,
            psnr(&reference, &out),
            mem.vram_used.map_or(0, |b| b >> 20),
            mem.vram_budget.map_or(0, |b| b >> 20),
        );
    }
    ncnn::shutdown();
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("coreml_compare runs on macOS on Apple Silicon only");
    std::process::exit(2);
}
