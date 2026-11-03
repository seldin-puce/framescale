//! ncnn backend: Vulkan inference on every platform (MoltenVK on macOS), or
//! ncnn's CPU path when no usable GPU exists.
//!
//! Supported stages: Denoise (Real-CUGAN), Upscale (Real-ESRGAN family), Face
//! Fix (YuNet detection + CodeFormer restoration) and Interpolate (RIFE
//! v4.6). Anything else is rejected up front, never silently skipped.
#![allow(unsafe_code)] // cxx bridge

use std::path::{Path, PathBuf};

use cxx::UniquePtr;
use fs_types::pipeline::StageConfig;
use fs_types::worker::EnginePlan;

use crate::face;
use crate::planar::{self, Planar, Rect};
use crate::{EngineError, Frame, InferenceBackend};

#[cxx::bridge(namespace = "fsncnn")]
mod ffi {
    #[derive(Debug, Clone)]
    struct GpuEntry {
        index: i32,
        name: String,
        vendor_id: u32,
        device_id: u32,
        /// 0 discrete, 1 integrated, 2 virtual, 3 cpu (ncnn convention)
        kind: i32,
    }

    unsafe extern "C++" {
        include!("fs-engine/cpp/bridge.h");

        type Model;

        fn init_gpu(driver_path: &str) -> i32;
        fn destroy_gpu();
        fn gpu_list() -> Vec<GpuEntry>;
        fn model_new(gpu: i32, fp16: bool, threads: i32) -> Result<UniquePtr<Model>>;
        fn load(self: Pin<&mut Model>, param: &str, bin: &str) -> Result<()>;
        fn load_text(self: Pin<&mut Model>, param_text: &str, bin: &str) -> Result<()>;
        fn run_outputs(
            self: &Model,
            input: &[f32],
            w: i32,
            h: i32,
            shapes: &mut Vec<i32>,
        ) -> Result<Vec<f32>>;
        fn run_codeformer(
            self: &Model,
            generator: &Model,
            input: &[f32],
            output: &mut [f32],
        ) -> Result<()>;
        fn run_sr(
            self: &Model,
            input: &[f32],
            w: i32,
            h: i32,
            output: &mut [f32],
            out_w: i32,
            out_h: i32,
        ) -> Result<()>;
        fn run_rife(
            self: &Model,
            a: &[f32],
            b: &[f32],
            w: i32,
            h: i32,
            t: f32,
            output: &mut [f32],
        ) -> Result<()>;
        fn cugan_stats(
            self: &Model,
            input: &[f32],
            w: i32,
            h: i32,
            lens: &mut Vec<i32>,
        ) -> Result<Vec<f32>>;
        #[allow(clippy::too_many_arguments)]
        fn run_cugan(
            self: &Model,
            input: &[f32],
            w: i32,
            h: i32,
            gaps: &[f32],
            lens: &[i32],
            output: &mut [f32],
            out_w: i32,
            out_h: i32,
        ) -> Result<()>;
    }
}

pub use ffi::GpuEntry;

// SAFETY: after loading, a model is only used through `run_*`, which are
// `const` and create a fresh ncnn::Extractor per call. Concurrent extractors on
// one ncnn::Net are supported by ncnn. Loading takes `Pin<&mut>` (exclusive).
unsafe impl Send for ffi::Model {}
unsafe impl Sync for ffi::Model {}

/// Create ncnn's Vulkan instance (once per process). Returns the GPU count, or
/// a negative value when Vulkan is unavailable (the CPU path still works).
pub fn init(driver_path: Option<&Path>) -> i32 {
    ffi::init_gpu(
        &driver_path
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default(),
    )
}

pub fn gpus() -> Vec<GpuEntry> {
    ffi::gpu_list()
}

/// Destroy the Vulkan instance. Call after every backend is dropped.
pub fn shutdown() {
    ffi::destroy_gpu();
}

/// Which device to run on.
#[derive(Debug, Clone)]
pub struct DeviceSel {
    /// ncnn GPU index, or -1 for ncnn's CPU path.
    pub index: i32,
    pub name: String,
    pub vram_bytes: Option<u64>,
}

/// Context pixels around each SR tile (Real-ESRGAN's own default).
const SR_PAD: usize = 10;
/// A RIFE flow network and the frame alignment it needs.
struct Rife {
    model: UniquePtr<ffi::Model>,
    align: usize,
}

/// RIFE model directory (`models/rife/<id>`) and alignment: frames are padded
/// to a multiple of the network's coarsest flow scale (32, or 64 for 4.25+).
fn resolve_rife(models: &Path, id: &str) -> Result<(PathBuf, usize), EngineError> {
    let align = match id {
        "rife-v4.6" | "rife-v4.22-lite" => 32,
        "rife-v4.25" | "rife-v4.26" => 64,
        _ => {
            return Err(EngineError::ModelNotFound(format!(
                "unknown interpolation model '{id}'"
            )));
        }
    };
    Ok((models.join("rife").join(id), align))
}

struct Sr {
    model: UniquePtr<ffi::Model>,
    native: usize,
    target: usize,
}

/// Context Real-CUGAN ×2 consumes on each side (valid convolutions:
/// input w + 36 → output 2w), from the reference implementation.
const CUGAN_PAD: usize = 18;
/// The SE statistics pass runs on the whole frame, downscaled to about this
/// many pixels ("very rough" sync, realcugan-ncnn-vulkan's default). Measured
/// on a 1920×1028 photo against exact full-resolution statistics: mean Δ 2.8
/// levels at 512², 3.1 at 256². The CPU path uses 256²: the pass is serial
/// there and dominated the frame time.
const CUGAN_STATS_AREA_GPU: f64 = 512.0 * 512.0;
const CUGAN_STATS_AREA_CPU: f64 = 256.0 * 256.0;

/// Denoise stage: Real-CUGAN ×2 at the chosen denoise level, then an exact
/// area downscale back to the input size. (The ncnn Pro models take and give
/// plain 0–1 RGB like SE; measured, no extra normalization.)
struct Cugan {
    model: UniquePtr<ffi::Model>,
    stats_area: f64,
}

impl Cugan {
    /// SE statistics of the whole frame (from a downscaled copy).
    fn stats(&self, img: &Planar) -> Result<(Vec<f32>, Vec<i32>), EngineError> {
        let f = ((img.w * img.h) as f64 / self.stats_area).sqrt().max(1.0);
        let even = |v: f64| ((v / 2.0).round() as usize).max(1) * 2;
        let (sw, sh) = (
            even(img.w as f64 / f).min(img.w + img.w % 2),
            even(img.h as f64 / f).min(img.h + img.h % 2),
        );
        let small = if sw < img.w && sh < img.h {
            planar::area_resample(img, sw, sh)
        } else {
            img.region(0, 0, sw, sh)
        };
        let p = CUGAN_PAD as isize;
        let input = small.region(-p, -p, small.w + 2 * CUGAN_PAD, small.h + 2 * CUGAN_PAD);
        let mut lens = Vec::new();
        let gaps = self
            .model
            .cugan_stats(&input.data, input.w as i32, input.h as i32, &mut lens)
            .map_err(|e| map_err(&e, 0))?;
        Ok((gaps, lens))
    }

    /// One padded tile (`(w + 36) × (h + 36)`) → `2·(w + 36) × 2·(h + 36)`
    /// as `tiled_into` expects; only the interior (the network output) is used.
    fn tile(
        &self,
        t: &Planar,
        gaps: &[f32],
        lens: &[i32],
        tile: u32,
    ) -> Result<Planar, EngineError> {
        let (w, h) = (t.w - 2 * CUGAN_PAD, t.h - 2 * CUGAN_PAD);
        // The network needs even sizes: extend by one replicated column/row.
        let (ew, eh) = (w + w % 2, h + h % 2);
        let owned;
        let input = if (ew, eh) == (w, h) {
            t
        } else {
            owned = t.region(0, 0, ew + 2 * CUGAN_PAD, eh + 2 * CUGAN_PAD);
            &owned
        };
        let mut net = Planar::zeros(ew * 2, eh * 2);
        self.model
            .run_cugan(
                &input.data,
                input.w as i32,
                input.h as i32,
                gaps,
                lens,
                &mut net.data,
                net.w as i32,
                net.h as i32,
            )
            .map_err(|e| map_err(&e, tile))?;
        let mut out = Planar::zeros(t.w * 2, t.h * 2);
        let r = Rect {
            x: 0,
            y: 0,
            w: w * 2,
            h: h * 2,
        };
        out.blit(&net, r, CUGAN_PAD * 2, CUGAN_PAD * 2);
        Ok(out)
    }

    fn denoise(
        &self,
        img: &Planar,
        tile: u32,
        workers: usize,
        scratch: &mut Planar,
    ) -> Result<Planar, EngineError> {
        let (gaps, lens) = self.stats(img)?;
        planar::tiled_into(img, tile as usize, CUGAN_PAD, 2, workers, scratch, |t| {
            self.tile(t, &gaps, &lens, tile)
        })?;
        Ok(planar::area_resample(scratch, img.w, img.h))
    }
}

/// Faces narrower than this (output pixels) are left alone: restoring them
/// would mostly invent detail.
const MIN_FACE: f32 = 24.0;
/// Bound the per-frame cost on crowd shots.
const MAX_FACES: usize = 12;
/// YuNet confidence needed to restore a face.
const FACE_SCORE: f32 = 0.7;

/// Face Fix: YuNet finds faces and landmarks; each face is aligned to the
/// FFHQ template, restored by CodeFormer and feathered back in. Both models
/// are bundled (`models/yunet`, `models/codeformer`).
struct FaceFix {
    detector: UniquePtr<ffi::Model>,
    encoder: UniquePtr<ffi::Model>,
    generator: UniquePtr<ffi::Model>,
}

impl FaceFix {
    fn load(
        models: &Path,
        fidelity: f32,
        device: &DeviceSel,
        threads: i32,
    ) -> Result<Self, EngineError> {
        let yunet = models.join("yunet");
        let dir = models.join("codeformer");
        let detector = load(
            device,
            threads,
            &yunet.join("yunet_2023mar.ncnn.param"),
            &yunet.join("yunet_2023mar.ncnn.bin"),
        )?;
        let encoder = load(
            device,
            threads,
            &dir.join("encoder.param"),
            &dir.join("encoder.bin"),
        )?;
        let gen_param = dir.join("generator.param");
        let gen_bin = dir.join("generator.bin");
        if !gen_bin.exists() {
            return Err(EngineError::ModelNotFound(gen_bin.display().to_string()));
        }
        let text = std::fs::read_to_string(&gen_param)
            .map_err(|e| EngineError::ModelNotFound(format!("{}: {e}", gen_param.display())))?;
        let (text, fused) = face::with_fidelity(&text, fidelity.clamp(0.0, 1.0));
        if fused == 0 {
            tracing::warn!("CodeFormer generator has no fuse blocks; fidelity setting ignored");
        }
        let mut generator =
            ffi::model_new(device.index, device.index >= 0, threads).map_err(|e| map_err(&e, 0))?;
        generator
            .pin_mut()
            .load_text(&text, &gen_bin.to_string_lossy())
            .map_err(|e| EngineError::ModelNotFound(e.what().to_owned()))?;
        Ok(Self {
            detector,
            encoder,
            generator,
        })
    }

    /// Detect and restore faces in place. Returns how many were restored.
    fn fix(&self, img: &mut Planar) -> Result<usize, EngineError> {
        let (input, scale) = face::letterbox_bgr(img);
        let mut shapes = Vec::new();
        let data = self
            .detector
            .run_outputs(&input.data, input.w as i32, input.h as i32, &mut shapes)
            .map_err(|e| map_err(&e, 0))?;
        let mut heads = Vec::new();
        let mut off = 0;
        for &[w, h, c] in shapes.as_chunks::<3>().0 {
            let (w, h, c) = (w as usize, h as usize, c as usize);
            let n = w * h * c;
            heads.push(face::Head {
                data: &data[off..off + n],
                w,
                h,
                c,
            });
            off += n;
        }
        let faces = face::decode_yunet(&heads, FACE_SCORE);
        let mut restored = 0;
        for f in faces
            .iter()
            .map(|f| f.scaled(scale))
            .filter(|f| f.bbox[2] >= MIN_FACE)
            .take(MAX_FACES)
        {
            let Some(to_face) = face::similarity(&f.kps, &face::TEMPLATE) else {
                continue;
            };
            let Some(to_img) = to_face.invert() else {
                continue;
            };
            let crop = face::warp(img, &to_img, face::FACE, face::FACE);
            let input: Vec<f32> = crop.data.iter().map(|v| v * 2.0 - 1.0).collect();
            let mut out = Planar::zeros(face::FACE, face::FACE);
            self.encoder
                .run_codeformer(&self.generator, &input, &mut out.data)
                .map_err(|e| map_err(&e, 0))?;
            for v in &mut out.data {
                *v = (v.clamp(-1.0, 1.0) + 1.0) * 0.5;
            }
            face::paste(img, &out, &to_face);
            restored += 1;
        }
        Ok(restored)
    }
}

pub struct NcnnBackend {
    denoise: Option<Cugan>,
    face: Option<FaceFix>,
    /// Faces restored in the last processed frame.
    faces: usize,
    /// Tiles in flight on the CPU path: 1 when ncnn threads each layer itself
    /// (OpenMP), one per core when it can't (Windows/MSVC builds).
    cpu_workers: usize,
    /// Reused Real-CUGAN ×2 buffer.
    dn_scratch: Planar,
    sr: Option<Sr>,
    /// Reused SR output buffer (resized on demand).
    scratch: Planar,
    gpu: bool,
    rife: Option<Rife>,
    device_name: String,
    max_tile: u32,
}

impl std::fmt::Debug for NcnnBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NcnnBackend")
            .field("device", &self.device_name)
            .finish_non_exhaustive()
    }
}

fn map_err(e: &cxx::Exception, tile: u32) -> EngineError {
    let m = e.what();
    if m.contains("(ncnn -100)") {
        EngineError::OutOfDeviceMemory { tile }
    } else if m.to_ascii_lowercase().contains("device lost") {
        EngineError::DeviceLost(m.to_owned())
    } else {
        EngineError::Internal(m.to_owned())
    }
}

/// `(param, bin)` for a Real-CUGAN denoise model at a strength 0–3:
/// 0 conservative (restore, keep grain), 1–3 denoise1x…3x. Pro only ships
/// conservative and denoise3x, so 1 → conservative and 2 → denoise3x.
fn resolve_cugan(models: &Path, id: &str, strength: u8) -> Result<(PathBuf, PathBuf), EngineError> {
    let pro = match id {
        "realcugan-se" => false,
        "realcugan-pro" => true,
        other => {
            return Err(EngineError::ModelNotFound(format!(
                "unknown denoise model '{other}'"
            )));
        }
    };
    let variant = match (pro, strength) {
        (false, 0) | (true, 0 | 1) => "up2x-conservative",
        (false, 1) => "up2x-denoise1x",
        (false, 2) => "up2x-denoise2x",
        _ => "up2x-denoise3x",
    };
    let dir = models
        .join("realcugan")
        .join(if pro { "models-pro" } else { "models-se" });
    Ok((
        dir.join(format!("{variant}.param")),
        dir.join(format!("{variant}.bin")),
    ))
}

/// `(param, bin, native_scale)` for an upscale model id at a requested scale.
fn resolve_sr(
    models: &Path,
    id: &str,
    scale: usize,
) -> Result<(PathBuf, PathBuf, usize), EngineError> {
    let (name, native) = match id {
        "realesr-animevideov3" => (format!("realesr-animevideov3-x{scale}"), scale),
        "realesrgan-x4plus" | "realesrgan-x4plus-anime" => (id.to_owned(), 4),
        other => {
            return Err(EngineError::ModelNotFound(format!(
                "unknown upscale model '{other}'"
            )));
        }
    };
    let dir = models.join("realesrgan").join("models");
    Ok((
        dir.join(format!("{name}.param")),
        dir.join(format!("{name}.bin")),
        native,
    ))
}

fn load(
    device: &DeviceSel,
    threads: i32,
    param: &Path,
    bin: &Path,
) -> Result<UniquePtr<ffi::Model>, EngineError> {
    for p in [param, bin] {
        if !p.exists() {
            return Err(EngineError::ModelNotFound(p.display().to_string()));
        }
    }
    let fp16 = device.index >= 0;
    let mut m = ffi::model_new(device.index, fp16, threads).map_err(|e| map_err(&e, 0))?;
    m.pin_mut()
        .load(&param.to_string_lossy(), &bin.to_string_lossy())
        .map_err(|e| EngineError::ModelNotFound(e.what().to_owned()))?;
    Ok(m)
}

impl NcnnBackend {
    pub fn new(plan: &EnginePlan, device: DeviceSel, threads: i32) -> Result<Self, EngineError> {
        let models = Path::new(&plan.models_dir);
        // CPU without ncnn-internal threading: single-threaded models, one tile
        // per core. FS_CPU_TILE_WORKERS forces it (benchmarks, any platform).
        let forced = std::env::var("FS_CPU_TILE_WORKERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        let tile_parallel = device.index < 0 && (forced.is_some() || !cfg!(ncnn_omp));
        // Measured (24-thread Arrow Lake): 8 workers beat 16 and 24; convolution
        // is memory-bound and slow cores hold back the frame.
        let cpu_workers = match forced {
            Some(n) if tile_parallel => n.max(1),
            _ if tile_parallel => (usize::try_from(threads).unwrap_or(1) / 2).clamp(1, 8),
            _ => 1,
        };
        let threads = if tile_parallel { 1 } else { threads };
        let mut sr = None;
        let mut denoise = None;
        let mut face_fix = None;
        let mut rife = None;
        let mut heavy = false;
        for stage in &plan.stages {
            match stage {
                StageConfig::Deinterlace { .. } => {} // handled by FFmpeg in the decoder
                StageConfig::Denoise { model, strength } => {
                    if model.path.is_some() {
                        return Err(EngineError::Unsupported(
                            "custom denoise models are not supported yet".into(),
                        ));
                    }
                    let (param, bin) = resolve_cugan(models, &model.id, *strength)?;
                    denoise = Some(Cugan {
                        model: load(&device, threads, &param, &bin)?,
                        stats_area: if device.index >= 0 {
                            CUGAN_STATS_AREA_GPU
                        } else {
                            CUGAN_STATS_AREA_CPU
                        },
                    });
                }
                StageConfig::FaceFix { model, fidelity } => {
                    if model.path.is_some() {
                        return Err(EngineError::Unsupported(
                            "custom face models are not supported yet".into(),
                        ));
                    }
                    if model.id != "codeformer" {
                        return Err(EngineError::ModelNotFound(format!(
                            "unknown face model '{}'",
                            model.id
                        )));
                    }
                    face_fix = Some(FaceFix::load(models, *fidelity, &device, threads)?);
                }
                StageConfig::Upscale { model, scale } => {
                    if model.path.is_some() {
                        return Err(EngineError::Unsupported(
                            "custom upscale models are not supported yet".into(),
                        ));
                    }
                    let (param, bin, native) = resolve_sr(models, &model.id, usize::from(*scale))?;
                    heavy = native == 4 && model.id.starts_with("realesrgan");
                    sr = Some(Sr {
                        model: load(&device, threads, &param, &bin)?,
                        native,
                        target: usize::from(*scale),
                    });
                }
                StageConfig::Interpolate { model, .. } => {
                    if model.path.is_some() {
                        return Err(EngineError::Unsupported(
                            "custom interpolation models are not supported yet".into(),
                        ));
                    }
                    let (dir, align) = resolve_rife(models, &model.id)?;
                    rife = Some(Rife {
                        model: load(
                            &device,
                            threads,
                            &dir.join("flownet.param"),
                            &dir.join("flownet.bin"),
                        )?,
                        align,
                    });
                }
            }
        }
        // Starting tile; the TileController halves it on OOM and remembers.
        // Measured on RX 9070 XT (RADV): small tiles win. With two in-flight
        // tiles the working set stays cache-resident. AnimeVideo-v3 1080p→4K:
        // 512 px 6.9 fps → 256 px 10.8 fps; x4plus: 400 px 1.0 → 192 px 1.27 fps.
        // A per-device calibration pass can refine this later.
        let gib = device.vram_bytes.unwrap_or(0) >> 30;
        let max_tile = match (device.index >= 0, heavy, gib) {
            (false, _, _) => 128,
            (true, true, g) if g >= 4 => 192,
            (true, true, _) => 128,
            (true, false, g) if g >= 2 => 256,
            (true, false, _) => 192,
        };
        Ok(Self {
            denoise,
            face: face_fix,
            faces: 0,
            cpu_workers,
            dn_scratch: Planar::zeros(1, 1),
            sr,
            scratch: Planar::zeros(1, 1),
            gpu: device.index >= 0,
            rife,
            device_name: device.name,
            max_tile,
        })
    }
}

impl NcnnBackend {
    /// Faces restored in the last frame (Face Fix).
    pub fn last_faces(&self) -> usize {
        self.faces
    }
}

impl InferenceBackend for NcnnBackend {
    fn name(&self) -> &'static str {
        "ncnn"
    }

    fn device_name(&self) -> Option<String> {
        Some(self.device_name.clone())
    }

    fn max_tile(&self) -> u32 {
        self.max_tile
    }

    fn process(&mut self, input: &Frame, tile: u32) -> Result<Frame, EngineError> {
        if self.denoise.is_none() && self.sr.is_none() && self.face.is_none() {
            return Ok(input.clone());
        }
        // Two in-flight tiles keep the GPU busy while the CPU slices/stitches;
        // the output buffer is reused across frames (no 100 MB page-faulting).
        let workers = if self.gpu { 2 } else { self.cpu_workers };
        // Tile-parallel CPU: about eight tiles per worker so slow cores don't
        // leave the rest idle at the end of a frame (measured: 640×360 at 8
        // workers, 64 px tiles 532 ms vs 80 px 641 ms).
        let tile = if workers > 2 {
            let balanced =
                ((input.width * input.height) as f64 / (workers * 8) as f64).sqrt() as u32;
            tile.min((balanced / 16 * 16).max(64))
        } else {
            tile
        };
        let mut img = planar::from_frame(input);
        if let Some(dn) = self.denoise.as_ref() {
            img = dn.denoise(&img, tile, workers, &mut self.dn_scratch)?;
        }
        let Some(sr) = self.sr.as_ref() else {
            if let Some(ff) = self.face.as_ref() {
                self.faces = ff.fix(&mut img)?;
            }
            return Ok(planar::to_frame(&img, input.format));
        };
        let (native, model) = (sr.native, &sr.model);
        planar::tiled_into(
            &img,
            tile as usize,
            SR_PAD,
            native,
            workers,
            &mut self.scratch,
            |t| {
                let mut o = Planar::zeros(t.w * native, t.h * native);
                model
                    .run_sr(
                        &t.data,
                        t.w as i32,
                        t.h as i32,
                        &mut o.data,
                        o.w as i32,
                        o.h as i32,
                    )
                    .map_err(|e| map_err(&e, tile))?;
                Ok(o)
            },
        )?;
        // Face Fix works on the final (upscaled) pixels.
        let frame = if sr.target == native {
            if let Some(ff) = self.face.as_ref() {
                self.faces = ff.fix(&mut self.scratch)?;
            }
            planar::to_frame(&self.scratch, input.format)
        } else {
            let mut out =
                planar::area_resample(&self.scratch, img.w * sr.target, img.h * sr.target);
            if let Some(ff) = self.face.as_ref() {
                self.faces = ff.fix(&mut out)?;
            }
            planar::to_frame(&out, input.format)
        };
        Ok(frame)
    }

    fn interpolate(
        &mut self,
        a: &Frame,
        b: &Frame,
        t: f32,
        tile: u32,
    ) -> Result<Frame, EngineError> {
        let Some(rife) = self.rife.as_ref() else {
            return Err(EngineError::Internal(
                "interpolation requested but RIFE is not loaded".into(),
            ));
        };
        if (a.width, a.height) != (b.width, b.height) {
            return Err(EngineError::Internal(
                "interpolate: frame size mismatch".into(),
            ));
        }
        let pa = planar::pad_to_multiple(&planar::from_frame(a), rife.align);
        let pb = planar::pad_to_multiple(&planar::from_frame(b), rife.align);
        let mut o = Planar::zeros(pa.w, pa.h);
        rife.model
            .run_rife(&pa.data, &pb.data, pa.w as i32, pa.h as i32, t, &mut o.data)
            .map_err(|e| map_err(&e, tile))?;
        Ok(planar::to_frame(
            &o.region(0, 0, a.width as usize, a.height as usize),
            a.format,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_resolution() {
        let m = Path::new("/m");
        let (p, _, n) = resolve_sr(m, "realesr-animevideov3", 3).expect("anime");
        assert!(p.ends_with("realesrgan/models/realesr-animevideov3-x3.param"));
        assert_eq!(n, 3);
        assert_eq!(
            resolve_sr(m, "realesrgan-x4plus", 2).expect("x4").2,
            4,
            "native ×4, area-downscaled"
        );
        assert!(matches!(
            resolve_sr(m, "nope", 2),
            Err(EngineError::ModelNotFound(_))
        ));
        let (dir, align) = resolve_rife(m, "rife-v4.25").expect("4.25");
        assert!(dir.ends_with("rife/rife-v4.25"));
        assert_eq!(align, 64);
        assert_eq!(resolve_rife(m, "rife-v4.6").expect("4.6").1, 32);
        assert!(resolve_rife(m, "rife-v9").is_err());
    }
}
