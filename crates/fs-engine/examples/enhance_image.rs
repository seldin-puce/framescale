//! Run the real engine on one image or a few raw frames (RGB24, back to back),
//! e.g. for website before/after pictures and demo videos:
//!
//!   enhance_image <in.rgb> <width> <height> <out.rgb> [--denoise ID LEVEL] [--upscale ID SCALE] [--noise 0-1] [--facefix FIDELITY]
//!
//! Uses the first Vulkan GPU, or the CPU. Models come from `FS_MODELS` or the
//! bundled resources folder.

use std::path::PathBuf;

use fs_engine::ncnn::{self, DeviceSel, NcnnBackend};
use fs_engine::{Frame, InferenceBackend};
use fs_types::pipeline::{ModelRef, StageConfig};
use fs_types::worker::{EnginePlan, FrameFormat};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        eprintln!(
            "usage: enhance_image <in.rgb> <w> <h> <out.rgb> [--denoise ID LEVEL] [--upscale ID SCALE] [--noise 0-1] [--facefix FIDELITY]"
        );
        std::process::exit(2);
    }
    let (w, h): (u32, u32) = (a[2].parse().expect("width"), a[3].parse().expect("height"));
    let model = |id: &str| ModelRef {
        id: id.into(),
        path: None,
    };
    let mut stages = Vec::new();
    let mut i = 5;
    while i + 1 < a.len() {
        match a[i].as_str() {
            "--denoise" => stages.push(StageConfig::Denoise {
                model: model(&a[i + 1]),
                strength: a[i + 2].parse().expect("level"),
            }),
            "--upscale" => stages.push(StageConfig::Upscale {
                model: model(&a[i + 1]),
                scale: a[i + 2].parse().expect("scale"),
                denoise: None,
            }),
            "--noise" => {
                // Noise removal (0–1) for the last Upscale stage (General v3).
                let v: f32 = a[i + 1].parse().expect("noise");
                if let Some(StageConfig::Upscale { denoise, .. }) = stages.last_mut() {
                    *denoise = Some(v);
                }
                i += 2;
                continue;
            }
            "--facefix" => {
                stages.push(StageConfig::FaceFix {
                    model: model("codeformer"),
                    fidelity: a[i + 1].parse().expect("fidelity"),
                });
                i += 2;
                continue;
            }
            other => panic!("unknown option {other}"),
        }
        i += 3;
    }
    let models = std::env::var_os("FS_MODELS").map_or_else(
        || {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../apps/desktop/src-tauri/resources/models")
        },
        PathBuf::from,
    );
    let device = if ncnn::init(None) > 0 {
        let g = &ncnn::gpus()[0];
        DeviceSel {
            index: g.index,
            name: g.name.clone(),
            vram_bytes: Some(4 << 30),
        }
    } else {
        DeviceSel {
            index: -1,
            name: "CPU".into(),
            vram_bytes: None,
        }
    };
    eprintln!("device: {}", device.name);
    let plan = EnginePlan {
        stages,
        scene_cuts: vec![],
        in_width: w,
        in_height: h,
        out_width: w,
        out_height: h,
        frame_format: FrameFormat::Rgb24,
        device: None,
        tile: None,
        models_dir: models.to_string_lossy().into(),
        // CodeFormer (on demand): FS_DOWNLOADS or where `cargo xtask models` puts it.
        downloads_dir: Some(std::env::var("FS_DOWNLOADS").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/ondemand-models").into()
        })),
        tile_cache: None,
        backend: None,
    };
    // One frame, or several back to back (raw video): models load once.
    let data = std::fs::read(&a[1]).expect("read input");
    let size = (w * h * 3) as usize;
    assert!(
        !data.is_empty() && data.len().is_multiple_of(size),
        "input must be {w}×{h} RGB24 frames"
    );
    let mut b = NcnnBackend::new(&plan, device, 8).expect("load models");
    let mut out = Vec::new();
    let (mut ow, mut oh) = (w, h);
    for chunk in data.chunks(size) {
        let f = b
            .process(&Frame::new(w, h, FrameFormat::Rgb24, chunk.to_vec()), 256)
            .expect("process");
        (ow, oh) = (f.width, f.height);
        out.extend_from_slice(&f.data);
    }
    drop(b);
    std::fs::write(&a[4], &out).expect("write output");
    eprintln!("wrote {} frame(s) of {ow}×{oh}", data.len() / size);
    ncnn::shutdown();
}
