//! ncnn backend: Vulkan inference on every platform (MoltenVK on macOS), or
//! ncnn's CPU path when no usable GPU exists.
//!
//! Supported stages: Upscale (Real-ESRGAN family) and Interpolate (RIFE v4.6).
//! Denoise (Real-CUGAN) and Face Fix (BYOM) are rejected up front with a clear
//! error rather than silently skipped.
#![allow(unsafe_code)] // cxx bridge

use std::path::{Path, PathBuf};

use cxx::UniquePtr;
use fs_types::pipeline::StageConfig;
use fs_types::worker::EnginePlan;

use crate::planar::{self, Planar};
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
        fn run_sr(
            self: Pin<&mut Model>,
            input: &[f32],
            w: i32,
            h: i32,
            output: &mut [f32],
            out_w: i32,
            out_h: i32,
        ) -> Result<()>;
        fn run_rife(
            self: Pin<&mut Model>,
            a: &[f32],
            b: &[f32],
            w: i32,
            h: i32,
            t: f32,
            output: &mut [f32],
        ) -> Result<()>;
    }
}

pub use ffi::GpuEntry;

// SAFETY: an ncnn::Net may be used from any single thread at a time. The
// backend owns its models exclusively and is moved, never shared.
unsafe impl Send for ffi::Model {}

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
/// RIFE v4 needs dimensions divisible by 32.
const RIFE_ALIGN: usize = 32;

struct Sr {
    model: UniquePtr<ffi::Model>,
    native: usize,
    target: usize,
}

pub struct NcnnBackend {
    sr: Option<Sr>,
    rife: Option<UniquePtr<ffi::Model>>,
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
        let mut sr = None;
        let mut rife = None;
        let mut heavy = false;
        for stage in &plan.stages {
            match stage {
                StageConfig::Deinterlace { .. } => {} // handled by FFmpeg in the decoder
                StageConfig::Denoise { .. } => {
                    return Err(EngineError::Unsupported(
                        "Denoise (Real-CUGAN) is not available in this build yet".into(),
                    ));
                }
                StageConfig::FaceFix { .. } => {
                    return Err(EngineError::Unsupported(
                        "Face Fix (bring-your-own model) is not available in this build yet".into(),
                    ));
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
                    if model.id != "rife-v4.6" {
                        return Err(EngineError::ModelNotFound(format!(
                            "unknown interpolation model '{}'",
                            model.id
                        )));
                    }
                    let dir = models.join("rife").join("rife-v4.6");
                    rife = Some(load(
                        &device,
                        threads,
                        &dir.join("flownet.param"),
                        &dir.join("flownet.bin"),
                    )?);
                }
            }
        }
        // Starting tile; the TileController halves it on OOM and remembers.
        let gib = device.vram_bytes.unwrap_or(0) >> 30;
        let max_tile = match (device.index >= 0, heavy, gib) {
            (false, _, _) => 128,
            (true, true, g) if g >= 8 => 400,
            (true, true, g) if g >= 4 => 256,
            (true, true, _) => 128,
            (true, false, g) if g >= 4 => 512,
            (true, false, _) => 256,
        };
        Ok(Self {
            sr,
            rife,
            device_name: device.name,
            max_tile,
        })
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
        let Some(sr) = self.sr.as_mut() else {
            return Ok(input.clone());
        };
        let img = planar::from_frame(input);
        let native = sr.native;
        let out = planar::tiled(&img, tile as usize, SR_PAD, native, |t| {
            let mut o = Planar::zeros(t.w * native, t.h * native);
            sr.model
                .pin_mut()
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
        })?;
        let out = if sr.target == native {
            out
        } else {
            planar::area_resample(&out, img.w * sr.target, img.h * sr.target)
        };
        Ok(planar::to_frame(&out, input.format))
    }

    fn interpolate(
        &mut self,
        a: &Frame,
        b: &Frame,
        t: f32,
        tile: u32,
    ) -> Result<Frame, EngineError> {
        let Some(rife) = self.rife.as_mut() else {
            return Err(EngineError::Internal(
                "interpolation requested but RIFE is not loaded".into(),
            ));
        };
        if (a.width, a.height) != (b.width, b.height) {
            return Err(EngineError::Internal(
                "interpolate: frame size mismatch".into(),
            ));
        }
        let pa = planar::pad_to_multiple(&planar::from_frame(a), RIFE_ALIGN);
        let pb = planar::pad_to_multiple(&planar::from_frame(b), RIFE_ALIGN);
        let mut o = Planar::zeros(pa.w, pa.h);
        rife.pin_mut()
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
    }
}
