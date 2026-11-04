//! Which inference backend runs a job.
//!
//! An explicit choice wins: `FS_ENGINE` (tests, support) over the plan's
//! `backend` (the user's override). Otherwise the first that opens of:
//! Apple (Apple Silicon, `backend-apple`) → TensorRT (NVIDIA device and the
//! TensorRT runtime installed, `backend-tensorrt`) → ncnn (`backend-ncnn`:
//! Vulkan, or ncnn's CPU path). A backend that fails to open hands over to
//! the next, with a warning (a log line when it merely has no model for this
//! pipeline yet). The last one's error is fatal: there is never a
//! silent downgrade to the non-AI reference. The reference only runs when
//! asked for, or in a build without any inference backend (as before).

use fs_engine::engine::BackendKind;
use fs_engine::mock::{MockBackend, MockConfig};
use fs_engine::{EngineError, InferenceBackend};
use fs_types::WarningCode;
use fs_types::worker::{EnginePlan, WorkerEvent};

use crate::emit::Emitter;

/// `FS_ENGINE`, else the plan's override.
fn explicit(plan: &EnginePlan, emit: &Emitter) -> Option<BackendKind> {
    if let Ok(v) = std::env::var("FS_ENGINE") {
        match BackendKind::parse(&v) {
            Some(k) => return Some(k),
            None => emit.send(&WorkerEvent::Warning {
                code: WarningCode::BackendFallback,
                message: format!("FS_ENGINE={v} isn't a backend; choosing automatically"),
            }),
        }
    }
    plan.backend
}

/// Backends to try, best first.
pub fn candidates(plan: &EnginePlan, emit: &Emitter) -> Vec<BackendKind> {
    if let Some(k) = explicit(plan, emit) {
        return vec![k];
    }
    let mut v = Vec::new();
    if cfg!(all(
        feature = "backend-apple",
        target_os = "macos",
        target_arch = "aarch64"
    )) {
        v.push(BackendKind::Apple);
    }
    #[cfg(feature = "backend-tensorrt")]
    if tensorrt::planned_device_is_nvidia(plan) && tensorrt::probe().is_some() {
        v.push(BackendKind::TensorRt);
    }
    if cfg!(feature = "ncnn") {
        v.push(BackendKind::Ncnn);
    }
    if v.is_empty() {
        v.push(BackendKind::Reference);
    }
    v
}

/// Open one backend for `plan`.
pub fn open(
    kind: BackendKind,
    plan: &EnginePlan,
) -> Result<Box<dyn InferenceBackend>, EngineError> {
    let missing = |what: &str| {
        Err(EngineError::Unsupported(format!(
            "this build has no {what} backend"
        )))
    };
    match kind {
        BackendKind::Reference => Ok(Box::new(fs_engine::cpu::ReferenceBackend::new(
            &plan.stages,
        ))),
        BackendKind::Mock => Ok(Box::new(MockBackend::new(
            &plan.stages,
            MockConfig::from_env(1),
        ))),
        BackendKind::Ncnn => {
            #[cfg(feature = "ncnn")]
            {
                crate::device::ncnn_backend(plan)
            }
            #[cfg(not(feature = "ncnn"))]
            {
                missing("ncnn")
            }
        }
        BackendKind::Apple => {
            #[cfg(all(
                feature = "backend-apple",
                target_os = "macos",
                target_arch = "aarch64"
            ))]
            {
                fs_engine::apple::AppleBackend::new(plan)
                    .map(|b| Box::new(b) as Box<dyn InferenceBackend>)
            }
            #[cfg(not(all(
                feature = "backend-apple",
                target_os = "macos",
                target_arch = "aarch64"
            )))]
            {
                missing("Core ML")
            }
        }
        BackendKind::TensorRt => {
            if cfg!(feature = "backend-tensorrt") {
                unimplemented!(
                    "TensorRT backend: not written yet (docs/engine-audit.md, section 7)"
                )
            }
            missing("TensorRT")
        }
    }
}

/// Open the first backend that works, warning about each one skipped.
pub fn select(plan: &EnginePlan, emit: &Emitter) -> Result<Box<dyn InferenceBackend>, EngineError> {
    let chosen = std::env::var("FS_ENGINE")
        .ok()
        .and_then(|v| BackendKind::parse(&v))
        .or(plan.backend)
        .is_some();
    let list = candidates(plan, emit);
    let last = list.len() - 1;
    for (i, kind) in list.into_iter().enumerate() {
        if kind == BackendKind::Reference && !chosen {
            emit.send(&WorkerEvent::Warning {
                code: WarningCode::EncoderFallback,
                message: "this build has no inference runtime; using the non-AI reference backend"
                    .into(),
            });
        }
        match open(kind, plan) {
            Ok(b) => return Ok(b),
            // Expected (no model for this pipeline on that backend yet): a
            // log line, not a warning in the UI.
            Err(EngineError::Unsupported(why)) if i < last => {
                eprintln!("fs-worker: {kind:?} backend skipped: {why}");
            }
            Err(e) if i < last => emit.send(&WorkerEvent::Warning {
                code: WarningCode::BackendFallback,
                message: format!("{kind:?} backend unavailable ({e}); trying the next one"),
            }),
            Err(e) => return Err(e),
        }
    }
    unreachable!("candidates() is never empty")
}

/// TensorRT is never linked: its runtime is proprietary and large, and the
/// app is GPL (docs/engine-audit.md, section 5). It's found at runtime.
#[cfg(feature = "backend-tensorrt")]
pub mod tensorrt {
    use fs_types::worker::EnginePlan;

    /// Library names, newest major version first. TensorRT 10's Windows
    /// import library is `nvinfer_10.lib` (NVIDIA's zip install guide), so its
    /// DLL is `nvinfer_10.dll`; 11 follows the same pattern, as do the Linux
    /// sonames. Not checked against an installed runtime here (no NVIDIA GPU).
    #[cfg(windows)]
    const NVINFER: &[&str] = &["nvinfer_11.dll", "nvinfer_10.dll"];
    #[cfg(windows)]
    const CUDART: &[&str] = &["cudart64_13.dll", "cudart64_12.dll"];
    #[cfg(all(unix, not(target_os = "macos")))]
    const NVINFER: &[&str] = &["libnvinfer.so.11", "libnvinfer.so.10"];
    #[cfg(all(unix, not(target_os = "macos")))]
    const CUDART: &[&str] = &["libcudart.so.13", "libcudart.so.12"];
    /// No TensorRT on macOS.
    #[cfg(target_os = "macos")]
    const NVINFER: &[&str] = &[];
    #[cfg(target_os = "macos")]
    const CUDART: &[&str] = &[];

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Runtime {
        pub nvinfer: &'static str,
        pub cudart: &'static str,
    }

    #[allow(unsafe_code)]
    fn loads(name: &str) -> bool {
        // SAFETY: loading runs the library's initialisers. These are NVIDIA's
        // runtime libraries, found by name on the system's search path; we
        // call nothing in them and unload them again right away.
        unsafe { libloading::Library::new(name) }.is_ok()
    }

    /// The TensorRT and CUDA runtime libraries this system can load.
    pub fn probe() -> Option<Runtime> {
        let nvinfer = NVINFER.iter().copied().find(|n| loads(n))?;
        let cudart = CUDART.iter().copied().find(|n| loads(n))?;
        Some(Runtime { nvinfer, cudart })
    }

    /// Whether the job's device is an NVIDIA GPU (the best-ranked one when
    /// the plan doesn't name one).
    pub fn planned_device_is_nvidia(plan: &EnginePlan) -> bool {
        let Ok((_, mut devices)) = fs_hw::VulkanContext::discover() else {
            return false;
        };
        let dev = match plan.device {
            Some(u) => devices.into_iter().find(|d| d.uuid == u),
            None => {
                fs_hw::rank(&mut devices);
                devices.into_iter().next()
            }
        };
        dev.is_some_and(|d| d.vendor == fs_types::Vendor::Nvidia)
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn probe_never_panics() {
            // No NVIDIA runtime on CI or the development machine: None there,
            // Some on a machine with TensorRT installed.
            let _ = super::probe();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs_types::pipeline::{ModelRef, StageConfig};
    use fs_types::worker::FrameFormat;

    fn plan(backend: Option<BackendKind>) -> EnginePlan {
        EnginePlan {
            stages: vec![StageConfig::Upscale {
                model: ModelRef {
                    id: "x".into(),
                    path: None,
                },
                scale: 2,
                denoise: None,
            }],
            scene_cuts: vec![],
            in_width: 8,
            in_height: 8,
            out_width: 16,
            out_height: 16,
            frame_format: FrameFormat::Rgb24,
            device: None,
            tile: None,
            models_dir: String::new(),
            downloads_dir: None,
            tile_cache: None,
            backend,
        }
    }

    #[test]
    fn an_explicit_choice_is_the_only_candidate() {
        if std::env::var("FS_ENGINE").is_ok() {
            return; // the environment decides; covered by the e2e tests
        }
        let emit = Emitter::new();
        assert_eq!(
            candidates(&plan(Some(BackendKind::Mock)), &emit),
            vec![BackendKind::Mock]
        );
        let auto = candidates(&plan(None), &emit);
        assert_eq!(
            auto.last(),
            Some(if cfg!(feature = "ncnn") {
                &BackendKind::Ncnn
            } else {
                &BackendKind::Reference
            }),
            "{auto:?}"
        );
        assert!(!auto.contains(&BackendKind::Mock));
    }

    #[test]
    fn mock_and_reference_open_without_a_gpu() {
        let p = plan(None);
        assert_eq!(open(BackendKind::Mock, &p).expect("mock").name(), "mock");
        assert_eq!(
            open(BackendKind::Reference, &p).expect("ref").name(),
            "cpu-reference"
        );
        // Without the feature (or off Apple Silicon) there's no Core ML; with
        // it, there's no Core ML model for this plan's network. Either way the
        // next backend takes over.
        assert!(matches!(
            open(BackendKind::Apple, &p),
            Err(EngineError::Unsupported(_))
        ));
    }
}
