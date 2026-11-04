//! Apple Silicon backend: Core ML (`backend-apple`, macOS on aarch64).
//!
//! - [`CoreMlEngine`] is a [`NetEngine`]: one compiled Core ML program with a
//!   fixed tile shape, run with the **asynchronous** prediction API
//!   (`predictionFromFeatures:options:completionHandler:`), up to
//!   `max_in_flight` tiles at once. Completions arrive on Core ML's queue and
//!   are handed to the owning thread through a channel; `wait_any` / `poll`
//!   keep the submit/wait contract.
//! - Tensors live in IOSurface-backed `OneComponent16Half` pixel buffers from
//!   two `CVPixelBufferPool`s (input, output), wrapped as FP16
//!   `MLMultiArray`s without a copy (`initWithPixelBuffer:shape:`). The output
//!   buffer is passed as the prediction's output backing, so Core ML writes
//!   straight into it. The tiler gets it back as a [`PixelBufferTensor`]
//!   (`Tensor::Device`, this codebase's form of a pixel-buffer frame storage)
//!   and reads it once, when stitching.
//! - [`AppleBackend`] puts it behind the frame-level [`InferenceBackend`]. It
//!   supports what a converted model exists for (today: General v3 ×2 or ×4 at
//!   the app's default denoise, see scripts/convert_coreml.py) and reports
//!   `Unsupported` otherwise, so the worker falls back to ncnn.
//!
//! Why not MPSGraph: it can't load ONNX or a PyTorch export, so each network
//! would have to be rebuilt op by op in Rust. Core ML runs the same network
//! on the GPU *or* the Neural Engine, which MPSGraph can't reach. No
//! measurable advantage justifies that cost.
#![allow(unsafe_code)] // Objective-C, CoreVideo and Metal FFI; each block says why it holds.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};

use block2::RcBlock;
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_foundation::{CFDictionary, CFNumber, CFRetained, CFString, CFType};
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLPredictionOptions,
};
use objc2_core_video::{
    CVPixelBuffer, CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow,
    CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags, CVPixelBufferPool,
    CVPixelBufferUnlockBaseAddress, kCVPixelBufferHeightKey, kCVPixelBufferIOSurfacePropertiesKey,
    kCVPixelBufferPixelFormatTypeKey, kCVPixelBufferWidthKey,
    kCVPixelFormatType_OneComponent16Half, kCVReturnSuccess,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL};
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

use fs_types::pipeline::StageConfig;
use fs_types::worker::EnginePlan;

use crate::engine::{
    BackendKind, Completed, DeviceTensor, EngineStats, HostTensor, NetEngine, SideInputs, Tensor,
    Ticket,
};
use crate::half;
use crate::planar::{self, Planar, TileMemory};
use crate::{EngineError, Frame, InferenceBackend};

/// Tiles in flight: one being written or read on the CPU while Core ML runs
/// the other (the same reasoning as ncnn's two).
const IN_FLIGHT: usize = 2;
/// The only network converted so far, and the denoise strength baked in.
const GENERAL_V3: &str = "realesr-general-x4v3";
const GENERAL_V3_DENOISE: f32 = 0.5;

fn err(what: &str, e: Option<&NSError>) -> EngineError {
    let why = e.map_or_else(
        || "unknown error".to_owned(),
        |e| e.localizedDescription().to_string(),
    );
    EngineError::Internal(format!("Core ML: {what}: {why}"))
}

fn units_from_env() -> MLComputeUnits {
    // FS_COREML_UNITS: benchmarking override; default lets Core ML pick the
    // Neural Engine or the GPU per layer.
    match std::env::var("FS_COREML_UNITS").as_deref() {
        Ok("cpu") => MLComputeUnits::CPUOnly,
        Ok("gpu") => MLComputeUnits::CPUAndGPU,
        Ok("ane") => MLComputeUnits::CPUAndNeuralEngine,
        _ => MLComputeUnits::All,
    }
}

/// An IOSurface-backed `OneComponent16Half` pool of `w × h` buffers.
fn pool(w: usize, h: usize) -> Result<CFRetained<CVPixelBufferPool>, EngineError> {
    let width = CFNumber::new_i64(w as i64);
    let height = CFNumber::new_i64(h as i64);
    let format = CFNumber::new_i64(i64::from(kCVPixelFormatType_OneComponent16Half));
    let iosurface = CFDictionary::<CFString, CFType>::empty();
    // SAFETY: CoreVideo's attribute keys are immutable CFString constants.
    let keys: [&CFString; 4] = unsafe {
        [
            kCVPixelBufferWidthKey,
            kCVPixelBufferHeightKey,
            kCVPixelBufferPixelFormatTypeKey,
            kCVPixelBufferIOSurfacePropertiesKey,
        ]
    };
    let values: [&CFType; 4] = [&width, &height, &format, &iosurface];
    let attrs = CFDictionary::<CFString, CFType>::from_slices(&keys, &values);
    let mut out: *mut CVPixelBufferPool = std::ptr::null_mut();
    // SAFETY: `attrs` holds CFString keys and CF values as CoreVideo expects;
    // `out` is a valid place for the created pool (+1 retained on success).
    let r = unsafe {
        CVPixelBufferPool::create(None, None, Some(attrs.as_opaque()), NonNull::from(&mut out))
    };
    match NonNull::new(out) {
        // SAFETY: Create rule: we own the returned reference.
        Some(p) if r == kCVReturnSuccess => Ok(unsafe { CFRetained::from_raw(p) }),
        _ => Err(EngineError::Internal(format!(
            "CVPixelBufferPoolCreate failed ({r})"
        ))),
    }
}

fn buffer(pool: &CVPixelBufferPool) -> Result<CFRetained<CVPixelBuffer>, EngineError> {
    let mut out: *mut CVPixelBuffer = std::ptr::null_mut();
    // SAFETY: `pool` is a live pool; `out` receives a +1 retained buffer.
    let r = unsafe { CVPixelBufferPool::create_pixel_buffer(None, pool, NonNull::from(&mut out)) };
    match NonNull::new(out) {
        // SAFETY: Create rule: we own the returned reference.
        Some(p) if r == kCVReturnSuccess => Ok(unsafe { CFRetained::from_raw(p) }),
        // Pool allocation failing is memory pressure: let the caller back off.
        _ => Err(EngineError::OutOfDeviceMemory { tile: 0 }),
    }
}

/// Lock `buf`'s base address for the duration of `f(base, bytes_per_row)`.
fn with_locked<R>(
    buf: &CVPixelBuffer,
    read_only: bool,
    f: impl FnOnce(*mut u8, usize) -> R,
) -> Result<R, EngineError> {
    let flags = if read_only {
        CVPixelBufferLockFlags::ReadOnly
    } else {
        CVPixelBufferLockFlags::empty()
    };
    // SAFETY: locking a valid buffer; paired with the unlock below.
    if unsafe { CVPixelBufferLockBaseAddress(buf, flags) } != kCVReturnSuccess {
        return Err(EngineError::Internal(
            "CVPixelBufferLockBaseAddress failed".into(),
        ));
    }
    let base = CVPixelBufferGetBaseAddress(buf).cast::<u8>();
    let stride = CVPixelBufferGetBytesPerRow(buf);
    let r = if base.is_null() {
        Err(EngineError::Internal(
            "pixel buffer has no base address".into(),
        ))
    } else {
        Ok(f(base, stride))
    };
    // SAFETY: unlocks the lock taken above, with the same flags.
    unsafe { CVPixelBufferUnlockBaseAddress(buf, flags) };
    r
}

fn shape(side: usize) -> Retained<NSArray<NSNumber>> {
    NSArray::from_retained_slice(&[
        NSNumber::new_usize(1),
        NSNumber::new_usize(3),
        NSNumber::new_usize(side),
        NSNumber::new_usize(side),
    ])
}

/// FP16 `[1, 3, side, side]` view of a `side × 3·side` pixel buffer.
fn multi_array(buf: &CVPixelBuffer, side: usize) -> Retained<MLMultiArray> {
    // SAFETY: `buf` is OneComponent16Half, `side` wide and `3·side` high, as
    // `initWithPixelBuffer:shape:` requires (last dimension = width, the
    // product of the others = height). The array retains the buffer.
    unsafe { MLMultiArray::initWithPixelBuffer_shape(MLMultiArray::alloc(), buf, &shape(side)) }
}

/// A Core ML output still in its pixel buffer: `side × side` planar FP16, of
/// which the top-left `w × h` is the tile (edge tiles were padded up).
pub struct PixelBufferTensor {
    buf: CFRetained<CVPixelBuffer>,
    side: usize,
    w: usize,
    h: usize,
}

// SAFETY: a CVPixelBuffer is a reference-counted CF object; its retain and
// release are thread-safe, and the tensor only reads it under a lock.
unsafe impl Send for PixelBufferTensor {}

impl std::fmt::Debug for PixelBufferTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PixelBufferTensor({}×{} of {})",
            self.w, self.h, self.side
        )
    }
}

impl DeviceTensor for PixelBufferTensor {
    fn backend(&self) -> BackendKind {
        BackendKind::Apple
    }

    fn dims(&self) -> (usize, usize, usize) {
        (self.w, self.h, 3)
    }

    fn to_host(&self) -> Result<HostTensor, EngineError> {
        let (side, w, h) = (self.side, self.w, self.h);
        with_locked(&self.buf, true, |base, stride| {
            let mut out = Planar::zeros(w, h);
            for c in 0..3 {
                for y in 0..h {
                    // SAFETY: row `c·side + y` < 3·side lies inside the
                    // locked buffer, and holds `side ≥ w` FP16 values.
                    let row = unsafe {
                        std::slice::from_raw_parts(
                            base.add((c * side + y) * stride).cast::<u16>(),
                            w,
                        )
                    };
                    for (x, &v) in row.iter().enumerate() {
                        out.set(c, x, y, half::to_f32(v));
                    }
                }
            }
            out
        })
    }
}

struct Pending {
    input: Planar,
    out: CFRetained<CVPixelBuffer>,
    /// Keeps the input buffer and arrays alive until Core ML is done.
    _keep: (Retained<MLMultiArray>, Retained<MLMultiArray>),
    w: usize,
    h: usize,
}

/// Compile a `.mlpackage` into a temporary `.mlmodelc`. The synchronous
/// call is deprecated in favour of a completion handler, which would only add
/// a wait here.
#[allow(deprecated)]
fn compile(url: &NSURL) -> Result<Retained<NSURL>, Retained<NSError>> {
    // SAFETY: plain Objective-C call on a file URL.
    unsafe { MLModel::compileModelAtURL_error(url) }
}

/// One Core ML program with a fixed `side × side` input on the device Core ML
/// picks (Neural Engine, GPU or CPU per layer; `FS_COREML_UNITS` overrides).
pub struct CoreMlEngine {
    model: Retained<MLModel>,
    side: usize,
    scale: usize,
    in_pool: CFRetained<CVPixelBufferPool>,
    out_pool: CFRetained<CVPixelBufferPool>,
    tx: Sender<(Ticket, Result<(), String>)>,
    rx: Receiver<(Ticket, Result<(), String>)>,
    pending: HashMap<Ticket, Pending>,
    next: u64,
}

// SAFETY: the engine is owned and used by one thread at a time (the
// `NetEngine` contract); it is only moved. Core ML objects may be used from
// any thread, and the completion blocks only touch the channel.
unsafe impl Send for CoreMlEngine {}

impl std::fmt::Debug for CoreMlEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoreMlEngine")
            .field("side", &self.side)
            .field("scale", &self.scale)
            .field("in_flight", &self.pending.len())
            .finish_non_exhaustive()
    }
}

impl CoreMlEngine {
    /// Compile and load `package` (`.mlpackage` or `.mlmodelc`) whose input
    /// is `[1, 3, side, side]` FP16 and output `[1, 3, side·scale, side·scale]`.
    pub fn load(package: &Path, side: usize, scale: usize) -> Result<Self, EngineError> {
        let url = NSURL::fileURLWithPath(&NSString::from_str(&package.to_string_lossy()));
        let compiled = if package.extension().is_some_and(|e| e == "mlmodelc") {
            url
        } else {
            compile(&url)
                .map_err(|e| err(&format!("compiling {}", package.display()), Some(&*e)))?
        };
        // SAFETY: plain Objective-C object creation and setters.
        let config = unsafe { MLModelConfiguration::new() };
        unsafe { config.setComputeUnits(units_from_env()) };
        let model =
            unsafe { MLModel::modelWithContentsOfURL_configuration_error(&compiled, &config) }
                .map_err(|e| err(&format!("loading {}", package.display()), Some(&*e)))?;
        let (tx, rx) = channel();
        Ok(Self {
            model,
            side,
            scale,
            in_pool: pool(side, 3 * side)?,
            out_pool: pool(side * scale, 3 * side * scale)?,
            tx,
            rx,
            pending: HashMap::new(),
            next: 0,
        })
    }

    fn finish(&mut self, ticket: Ticket, r: Result<(), String>) -> Completed {
        let Some(p) = self.pending.remove(&ticket) else {
            return Completed {
                ticket,
                input: None,
                output: Err(EngineError::Internal("Core ML: unknown prediction".into())),
            };
        };
        let output = r
            .map(|()| {
                Tensor::Device(Box::new(PixelBufferTensor {
                    buf: p.out,
                    side: self.side * self.scale,
                    w: p.w * self.scale,
                    h: p.h * self.scale,
                }))
            })
            .map_err(|why| EngineError::Internal(format!("Core ML prediction failed: {why}")));
        Completed {
            ticket,
            input: Some(p.input),
            output,
        }
    }
}

impl NetEngine for CoreMlEngine {
    fn max_in_flight(&self) -> usize {
        IN_FLIGHT
    }

    fn submit(&mut self, input: Tensor, _side: Option<SideInputs>) -> Result<Ticket, EngineError> {
        let input = input.into_host()?;
        let side = self.side;
        if input.w > side || input.h > side {
            return Err(EngineError::Unsupported(format!(
                "{}×{} tile on a model built for {side}×{side}",
                input.w, input.h
            )));
        }
        // Write the tile as planar FP16; an edge tile is padded up to the
        // model's fixed shape by repeating its last row and column.
        let in_buf = buffer(&self.in_pool)?;
        with_locked(&in_buf, false, |base, stride| {
            for c in 0..3 {
                for y in 0..side {
                    // SAFETY: row `c·side + y` < 3·side is inside the locked
                    // buffer and holds `side` FP16 values.
                    let row = unsafe {
                        std::slice::from_raw_parts_mut(
                            base.add((c * side + y) * stride).cast::<u16>(),
                            side,
                        )
                    };
                    let sy = y.min(input.h - 1);
                    for (x, v) in row.iter_mut().enumerate() {
                        *v = half::from_f32(input.at(c, x.min(input.w - 1), sy));
                    }
                }
            }
        })?;
        let out_buf = buffer(&self.out_pool)?;
        let in_arr = multi_array(&in_buf, side);
        let out_arr = multi_array(&out_buf, side * self.scale);

        // SAFETY: plain Objective-C object creation; the dictionaries hold
        // what Core ML documents (a feature value per input name, a
        // multi-array per output backing).
        let provider = unsafe {
            let value = MLFeatureValue::featureValueWithMultiArray(&in_arr);
            let dict = NSDictionary::<NSString, AnyObject>::from_slices(
                &[&*NSString::from_str("input")],
                &[&*value],
            );
            MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &dict,
            )
        }
        .map_err(|e| err("input", Some(&*e)))?;
        let options = unsafe { MLPredictionOptions::new() };
        let backings = NSDictionary::<NSString, AnyObject>::from_slices(
            &[&*NSString::from_str("output")],
            &[&*out_arr],
        );
        unsafe { options.setOutputBackings(&backings) };

        let ticket = Ticket(self.next);
        self.next += 1;
        let tx = self.tx.clone();
        let done = RcBlock::new(
            move |out: *mut ProtocolObject<dyn MLFeatureProvider>, error: *mut NSError| {
                let r = if out.is_null() {
                    // SAFETY: Core ML passes a valid NSError or null.
                    Err(unsafe { error.as_ref() }.map_or_else(
                        || "unknown error".to_owned(),
                        |e| e.localizedDescription().to_string(),
                    ))
                } else {
                    Ok(()) // the result is already in the output backing
                };
                let _ = tx.send((ticket, r));
            },
        );
        self.pending.insert(
            ticket,
            Pending {
                w: input.w,
                h: input.h,
                input,
                out: out_buf,
                _keep: (in_arr, out_arr),
            },
        );
        // SAFETY: the provider, options and block outlive the call (Core ML
        // retains what it keeps); the block is `Fn` and thread-safe.
        unsafe {
            self.model.predictionFromFeatures_options_completionHandler(
                ProtocolObject::from_ref(&*provider),
                &options,
                &done,
            );
        }
        Ok(ticket)
    }

    fn wait_any(&mut self) -> Option<Completed> {
        if self.pending.is_empty() {
            return None;
        }
        // The engine holds a sender itself, so this only fails if Core ML
        // never calls back; it always does, with a result or an error.
        let (ticket, r) = self.rx.recv().ok()?;
        Some(self.finish(ticket, r))
    }

    fn poll(&mut self) -> Option<Completed> {
        if self.pending.is_empty() {
            return None;
        }
        match self.rx.try_recv() {
            Ok((ticket, r)) => Some(self.finish(ticket, r)),
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => None,
        }
    }

    fn device_bytes(&self) -> Option<u64> {
        metal_memory().map(|(used, _)| used)
    }
}

/// `(currentAllocatedSize, recommendedMaxWorkingSetSize)` of the system's
/// Metal device, as Metal reports them for this process. No GPU load: macOS
/// doesn't expose one per process, and we don't make one up.
fn metal_memory() -> Option<(u64, u64)> {
    let dev = MTLCreateSystemDefaultDevice()?;
    Some((
        dev.currentAllocatedSize() as u64,
        dev.recommendedMaxWorkingSetSize(),
    ))
}

/// The converted model for a plan's upscale stage:
/// `<dir>/coreml/realesr-general-x4v3-dn0.50-x<scale>-t<tile>-p<pad>.mlpackage`
/// (or `.mlmodelc`), in the bundled or the download folder.
fn find_model(plan: &EnginePlan, scale: u8) -> Option<(PathBuf, usize, usize)> {
    let dirs = [
        Some(plan.models_dir.as_str()),
        plan.downloads_dir.as_deref(),
    ];
    for dir in dirs.into_iter().flatten() {
        let Ok(entries) = std::fs::read_dir(Path::new(dir).join("coreml")) else {
            continue;
        };
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let stem = name
                .strip_suffix(".mlpackage")
                .or_else(|| name.strip_suffix(".mlmodelc"));
            let Some(rest) =
                stem.and_then(|s| s.strip_prefix(&format!("{GENERAL_V3}-dn0.50-x{scale}-t")))
            else {
                continue;
            };
            let mut it = rest.split("-p");
            if let (Some(t), Some(p), None) = (it.next(), it.next(), it.next())
                && let (Ok(t), Ok(p)) = (t.parse::<usize>(), p.parse::<usize>())
            {
                return Some((e.path(), t, p));
            }
        }
    }
    None
}

/// Core ML behind the frame-level trait, for the pipelines it has models for.
pub struct AppleBackend {
    engine: CoreMlEngine,
    scale: usize,
    tile: usize,
    pad: usize,
    scratch: Planar,
    memory: TileMemory,
    reuse: bool,
}

impl std::fmt::Debug for AppleBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppleBackend")
            .field("scale", &self.scale)
            .field("tile", &self.tile)
            .finish_non_exhaustive()
    }
}

impl AppleBackend {
    /// `Unsupported` when the plan needs anything without a Core ML model
    /// yet; the worker then uses ncnn.
    pub fn new(plan: &EnginePlan) -> Result<Self, EngineError> {
        let mut upscale = None;
        for s in &plan.stages {
            match s {
                StageConfig::Deinterlace { .. } => {} // FFmpeg, before the engine
                StageConfig::Upscale {
                    model,
                    scale,
                    denoise,
                } if model.path.is_none()
                    && model.id == GENERAL_V3
                    && denoise.unwrap_or(GENERAL_V3_DENOISE) == GENERAL_V3_DENOISE =>
                {
                    upscale = Some(*scale);
                }
                other => {
                    return Err(EngineError::Unsupported(format!(
                        "no Core ML model for {:?} yet",
                        other.kind()
                    )));
                }
            }
        }
        let scale =
            upscale.ok_or_else(|| EngineError::Unsupported("Core ML: no upscale stage".into()))?;
        let (path, tile, pad) = find_model(plan, scale).ok_or_else(|| {
            EngineError::Unsupported(format!("no Core ML model for General v3 ×{scale}"))
        })?;
        Ok(Self {
            engine: CoreMlEngine::load(&path, tile + 2 * pad, usize::from(scale))?,
            scale: usize::from(scale),
            tile,
            pad,
            scratch: Planar::zeros(1, 1),
            memory: TileMemory::default(),
            reuse: true,
        })
    }
}

impl InferenceBackend for AppleBackend {
    fn name(&self) -> &'static str {
        "coreml"
    }

    fn device_name(&self) -> Option<String> {
        MTLCreateSystemDefaultDevice().map(|d| format!("{} (Core ML)", d.name()))
    }

    /// The model's shape is fixed, so `tile` (the controller's) is ignored.
    fn process(&mut self, input: &Frame, _tile: u32) -> Result<Frame, EngineError> {
        let img = planar::from_frame(input);
        let memory = self.reuse.then_some(&mut self.memory);
        planar::run_tiled(
            &mut self.engine,
            &img,
            (self.tile, self.tile),
            self.pad,
            self.scale,
            &mut self.scratch,
            memory,
            None,
        )?;
        Ok(planar::to_frame(&self.scratch, input.format))
    }

    fn interpolate(
        &mut self,
        _a: &Frame,
        _b: &Frame,
        _t: f32,
        _tile: u32,
    ) -> Result<Frame, EngineError> {
        Err(EngineError::Unsupported(
            "Core ML: no interpolation model yet".into(),
        ))
    }

    fn max_tile(&self) -> u32 {
        self.tile as u32
    }

    fn set_reuse(&mut self, on: bool) {
        self.reuse = on;
        if !on {
            self.memory.clear();
        }
    }

    fn stats(&self) -> EngineStats {
        let m = &self.memory;
        let mem = metal_memory();
        EngineStats {
            backend: Some(BackendKind::Apple),
            tile: Some(self.tile as u32),
            max_in_flight: Some(IN_FLIGHT as u32),
            vram_used: mem.map(|(used, _)| used),
            vram_budget: mem.map(|(_, budget)| budget),
            gpu_load_pct: None,
            reused_tiles_pct: (m.tiles > 0)
                .then(|| (100.0 * m.reused as f64 / m.tiles as f64) as f32),
        }
    }
}
