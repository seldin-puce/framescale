# Engine audit: the inference path today, and an interface for native backends

Status: **proposal, waiting for approval** (Step 1). Nothing in the code has
changed for this document. File and line references are to `main` at
`81d9fe9`.

## 1. How inference runs today

| | |
|---|---|
| Library | **ncnn** (Tencent), built from the pinned source (`cargo xtask ncnn`) and linked statically into `fs-worker` through a small C++ bridge (`crates/fs-engine/cpp/bridge.{h,cpp}`, `cxx`). |
| GPU API | Vulkan on every platform: native drivers on Windows and Linux, the bundled **MoltenVK** on macOS (`fs-worker/src/device.rs:driver_path`). ncnn's own CPU path is the fallback when no usable GPU exists. |
| Process model | **In-process, inside the worker subprocess.** The Tauri app (`fs-core::runner`) spawns one `fs-worker` per job (and one per preview). The worker owns FFmpeg decode and encode children and talks to the core over JSON lines on stdin and stdout. A worker crash never takes the app down. |
| Feature gate | `fs-engine/ncnn` and `fs-worker/ncnn` (worker default: on). Without it the worker uses the non-AI reference backend and warns. |
| Model loading | `NcnnBackend::new` (`fs-engine/src/ncnn.rs`) loads one `ncnn::Net` per network named in the plan's stages, from the bundled folder `EnginePlan::models_dir` or the download folder `EnginePlan::downloads_dir` (CodeFormer). Weights are blended in memory for General v3's denoise level (DNI) and the param text is edited at load time for ×4→×2 pooling (`pool_output`). FP16 storage on GPUs, FP32 arithmetic. |
| Where tiling happens | **Inside the backend**, in `fs-engine/src/planar.rs` (`tiled_into`, `tiled_into_even`, `tiled_into_reusing`, `tiled_grid`). That code is pure Rust and unit-tested, with no inference dependency. The backend picks the tile size it is given, pads each tile with context (SR 10 px, NAFNet 32 px, Real-CUGAN 18 px), keeps only the interior (no blending: seams are avoided by context, and the tests check the output equals the untiled result), and reuses unchanged tiles across frames. |
| Tile size policy | `fs-engine/src/tiling.rs`: `TileController` halves the tile and retries the same frame on VRAM OOM. `TileTuner` times sizes on the first frames and the worker caches the best one per device and model set (`fs-worker/src/tune.rs`, `<app cache>/tile-tuning.json`). |

Networks and how each runs:

| Stage | Network(s) | Granularity |
|---|---|---|
| Deinterlace | FFmpeg `bwdif` in the decoder | not inference |
| Denoise | NAFNet-SIDD (even tiles, 2× SR tile) or Real-CUGAN ×2 (tiles, plus one whole-frame SE statistics pass, then area-downscaled) | tiled |
| Upscale | Real-ESRGAN x4plus / x2plus (RRDB), General v3 (SRVGG, DNI-blended), AnimeVideo v3 | tiled |
| Face Fix | YuNet detector (640², whole frame), then CodeFormer encoder and generator per face (512²) | whole frame / per face |
| Interpolate | RIFE v4.x, whole frame (padded to 32/64) | whole frame |

## 2. A frame's path from decoder to encoder

Threads in the worker:

| Thread | Owns |
|---|---|
| `decode-reader` (`ffmpeg.rs:Decoder::spawn`) | The FFmpeg decoder's stdout. Reads exactly one frame into a **new `Vec<u8>` per frame**, sends `DecodeMsg::Frame` on a bounded(3) channel. Blocks while soft-paused. |
| main (`pipeline.rs:run`) | The backend (`Box<dyn InferenceBackend>`), `TileController`, `TileTuner`, interpolation `Scheduler`, segment bookkeeping. Strictly **one frame at a time**. |
| scoped tile workers (`planar.rs:tiled_grid`) | Spawned **per stage per frame** (`std::thread::scope`), `workers` of them: 2 on a GPU, up to 8 on CPU. Each slices a tile, calls the network synchronously, sends the result back. The main thread stitches. |
| `par_rows` helpers | Scoped threads for whole-frame conversions (u8/u16 ⇄ f32). |
| writer (`writer.rs:spawn`) | The current segment's FFmpeg encoder stdin. Frames arrive on a bounded(4) channel. Commits segments (`.part` → fsync → rename). |
| `control` | stdin commands (pause, stop, duty). |
| stderr readers | One per FFmpeg child. |

Types and copies for one frame with denoise and upscale on a GPU (`ncnn.rs:process`):

| # | Step | Memory |
|---|---|---|
| 1 | FFmpeg → pipe → `Frame { data: Vec<u8> }` (packed RGB24 or RGB48) | new allocation per frame |
| 2 | `planar::from_frame` → `Planar { data: Vec<f32> }` (planar 0–1) | new allocation, conversion |
| 3 | Per tile: `Planar::region` (tile + context) | new allocation per tile |
| 4 | `wrap_planar` into an `ncnn::Mat`, then upload (ncnn casts to FP16 and packs) | copy and transfer per tile |
| 5 | Download, `unwrap_planar` into a new `Planar` | transfer and copy per tile |
| 6 | NAFNet strength blend; ×3: per-tile area resample on the tile's worker | in place / new |
| 7 | `blit` into the stage's persistent buffer (`dn_scratch`, `scratch`) | copy, buffer reused across frames |
| 8 | Denoise returns `scratch.clone()` | full-frame copy |
| 9 | Face Fix on a copy of the SR output (the buffer carries reused tiles) | full-frame copy, only with Face Fix |
| 10 | `planar::to_frame` → output `Frame` | new allocation, conversion |
| 11 | Interpolation lag: `prev.processed.clone()` is sent to the writer | full output-frame copy |
| 12 | Writer → encoder stdin pipe | kernel copy |

RIFE converts both processed frames back to planar f32 (`from_frame`), pads
them, runs the whole frame and converts back.

The frame never leaves host memory between stages: every network call
uploads its input and downloads its output.

## 3. The existing abstraction and its callers

`fs_engine::InferenceBackend` (`crates/fs-engine/src/lib.rs`), a frame-level,
synchronous, `Send` trait:

```rust
pub trait InferenceBackend: Send {
    fn name(&self) -> &'static str;
    fn device_name(&self) -> Option<String>;
    fn process(&mut self, input: &Frame, tile: u32) -> Result<Frame, EngineError>;
    fn interpolate(&mut self, a: &Frame, b: &Frame, t: f32, tile: u32) -> Result<Frame, EngineError>;
    fn max_tile(&self) -> u32;                 // default tile, tuner start
    fn tune_profile(&self) -> Option<String>;  // key for the tuned-size cache
    fn set_reuse(&mut self, on: bool);         // cross-frame tile reuse
    fn report(&self) -> Option<String>;        // one log line at the end
}
```

Implementations: `NcnnBackend` (`fs-engine/src/ncnn.rs`) and `ReferenceBackend`
(`fs-engine/src/cpu.rs`, deterministic and non-AI: nearest-neighbour ×N plus
linear blend, used by the e2e tests through `FS_ENGINE=reference`).

Callers:
- `fs-worker/src/pipeline.rs`: `select_backend` (`FS_ENGINE=reference`, else
  ncnn through `device::ncnn_backend`), `Run::engine` (wraps every call in
  `TileController::run` and the Eco governor), and the main loop (`process`
  per frame, `interpolate` per synthesized frame).
- `fs-worker/src/device.rs`: `ncnn_backend`, `pick` (maps the planned Vulkan
  device UUID onto ncnn's GPU index, cross-checked by PCI ids).
- `fs-engine/tests/ncnn_models.rs`, `examples/bench.rs`, `examples/enhance_image.rs`.

`EngineError` carries the recovery policy: `OutOfDeviceMemory` halves the
tile in the worker, and `DeviceLost` ends the worker so the core restarts it
with half the tile (`fs-core/src/recovery.rs`).

## 4. Telemetry and progress to the frontend

- **Device telemetry is sampled by the app, not the worker**: `fs-hw::telemetry::Sampler`
  (NVML, sysfs, Windows PDH/D3DKMT, macOS IOKit/IOReport/SMC) every 500 ms in
  `apps/desktop/src-tauri/src/telemetry.rs`. It fills `DeviceTelemetry`
  (load, VRAM used/total, temperature, power) and runs the Eco governor,
  whose duty is pushed to the worker (`WorkerCommand::SetDuty`). The UI
  subscribes with `subscribe_telemetry` (Tauri `Channel`, about 2 Hz).
- **Progress comes from the worker**: `WorkerEvent::Progress { frames_in, frames_out, fps, segment }`
  every 250 ms → `fs-core::runner` → `JobEvent::Progress` → `queue.rs`, which
  also copies `fps` into `Telemetry::render_fps` → `state.emit` → UI.
- The tile size appears only in `WorkerEvent::Ready` (logged by the core) and
  in `TileReduced` warnings. The in-flight count, tuning result and reuse rate
  only reach the worker's stderr, which the core logs at debug level.

## 5. Constraints for new backends

1. **Frames arrive and leave in host memory.** Decode and encode are FFmpeg
   subprocesses on pipes, by design (crash isolation, LGPL/GPL split,
   hardware decode and encode in FFmpeg). Backend-native memory (a CUDA
   pointer, a `CVPixelBuffer`) can therefore only remove copies **between the
   stages inside the engine** (denoise → upscale → face), not at the decoder
   or encoder. Removing those would mean in-process decode and encode
   (libav* or VideoToolbox), which is a separate project.
2. **Several tiles are already in flight, but through threads.** `tiled_grid`
   runs `workers` synchronous calls at once (2 on a GPU), so the call
   pattern is not strictly one tile at a time. It works for ncnn because an
   `ncnn::Net` accepts concurrent extractors (`Model` is `Send + Sync`). It
   does **not** fit engines whose execution state must stay on one thread or
   needs one context per slot (a TensorRT `IExecutionContext` per CUDA
   stream). For those, the scheduler has to own submission order, not the
   network.
3. **The frame loop is sequential**, and measurements say that's not the
   bottleneck: NAFNet + x2plus at SD keeps the RX 9070 XT 97% busy
   (median), with 24, 4 or 2 CPU cores alike. General v3 ×2 at 720p: 92%.
   A new backend should keep **tile-level** parallelism. Frame-level
   pipelining isn't needed to fill the GPU.
4. **Model-specific logic lives next to the ncnn calls**: Real-CUGAN's SE
   statistics injection, NAFNet's strength blend and alignment, x2plus's
   2-pixel alignment, ×4→×2 pooling, General v3's DNI blend, RIFE padding,
   Face Fix's detect → warp → restore → paste. A new backend needs the same
   network graphs in its own format (TensorRT engine from ONNX, Core ML
   `.mlpackage`), and that conversion pipeline (in `xtask`, pinned like the
   ncnn models) is most of the work of adding one.
5. **Error semantics must map** onto `EngineError::{OutOfDeviceMemory, DeviceLost, ModelNotFound, Unsupported, Internal}`:
   recovery depends on them.
6. **Licensing.** The app is GPL-3.0-or-later. TensorRT and CUDA runtimes are
   proprietary: they can't be bundled, and linking them needs an additional
   permission from the copyright holder. SignPath Foundation also requires
   OSI-only components in signed packages. Hence: probe at runtime with
   `libloading`, never link or ship them. Apple's frameworks are system
   libraries, so Core ML or Metal is fine.
7. **Verification.** TensorRT can't be built or run on the development
   machine (AMD GPU). Apple code compiles only on macOS (CI `macos-15` and the
   maintainer's M4).

## 6. Proposed target interface

Two levels, so the pipeline and the existing ncnn code stay as they are and
a new backend only has to implement the low level:

- **`InferenceBackend` stays the pipeline-facing, frame-level trait**
  (unchanged signature; one new `stats()` method with a default). The worker
  loop, `TileController`, tuner, reuse and recovery keep working as today.
- **New `NetEngine`**: one loaded network on one device, owned by one
  thread, with an explicit submit/poll contract. The shared tiler drives it.

```rust
// crates/fs-engine/src/engine.rs (new)

/// Which implementation runs the networks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BackendKind { Apple, TensorRt, Ncnn, Reference, Mock }

/// A tensor in host memory: planar f32 `[c][h][w]` (today's `Planar`).
/// Always available; every engine accepts and returns it.
pub type HostTensor = planar::Planar;

/// Memory an engine can keep on its device between stages. Opaque to the
/// pipeline; `to_host` is the always-available fallback.
pub trait DeviceTensor: Send {
    fn backend(&self) -> BackendKind;
    fn dims(&self) -> (usize, usize, usize); // w, h, c
    fn to_host(&self) -> Result<HostTensor, EngineError>;
}

pub enum Tensor {
    Host(HostTensor),
    Device(Box<dyn DeviceTensor>),
}

/// Ticket for one submitted input; results may complete in any order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ticket(pub u64);

/// One network on one device. `Send`, not `Sync`: the thread that owns it
/// submits and polls; an implementation may keep its own threads, streams
/// or command queues behind that.
pub trait NetEngine: Send {
    /// Inputs this engine can have outstanding at once (≥ 1).
    fn max_in_flight(&self) -> usize;
    /// Queue `input` (with extra named inputs, e.g. Real-CUGAN's statistics).
    /// Never blocks while fewer than `max_in_flight` are outstanding.
    fn submit(&mut self, input: Tensor, extra: &[(&str, &[f32])]) -> Result<Ticket, EngineError>;
    /// Block until one outstanding input finishes; `None` when none are.
    fn wait_any(&mut self) -> Option<(Ticket, Result<Tensor, EngineError>)>;
    /// Non-blocking variant of `wait_any`.
    fn poll(&mut self) -> Option<(Ticket, Result<Tensor, EngineError>)>;
    /// Device memory the engine holds, when it knows.
    fn device_bytes(&self) -> Option<u64> { None }
}

/// What the HUD can show about the engine (sent with progress).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct EngineStats {
    pub backend: Option<BackendKind>,
    pub tile: Option<u32>,
    pub in_flight: Option<u32>,
    pub max_in_flight: Option<u32>,
    /// Backend-reported device memory (CUDA, Metal); the app's sampler still
    /// supplies board-level VRAM and load.
    pub vram_used: Option<u64>,
    pub gpu_load_pct: Option<f32>,
    pub reused_tiles_pct: Option<f32>,
}

// added to InferenceBackend, with a default:
//     fn stats(&self) -> EngineStats { EngineStats::default() }
```

The shared tiler becomes a submit/poll loop over any `NetEngine`:

```rust
// planar.rs: tiled_grid's thread pool is replaced by this loop, same grid,
// same context, same interior crop, same reuse, same error propagation.
pub fn run_tiled(engine: &mut dyn NetEngine, img: &Planar, grid: Grid,
                 out: &mut Planar, memory: Option<&mut TileMemory>,
                 post: impl Fn(&Planar /*in*/, Planar /*out*/) -> Result<Planar, EngineError>)
                 -> Result<TiledReport, EngineError>;
// keep up to engine.max_in_flight() tiles submitted; stitch as they finish
```

ncnn implements `NetEngine` with `ThreadedEngine<F: Fn(&Planar) -> Result<Planar> + Send + Sync>`:
`max_in_flight` worker threads, each running today's blocking extract. Those
threads are long-lived instead of spawned per frame. That is today's
behaviour (2 calls at once on a GPU, `cpu_workers` on CPU) and today's
numerics: each tile goes through the same function, and blits are disjoint.

A TensorRT engine would implement it with N execution contexts on N CUDA
streams (`max_in_flight = N`). Core ML would use async prediction or a
small queue. Both stay behind features and are not part of Step 2.

### Runtime selection

```rust
// crates/fs-worker/src/backend.rs (new; select_backend moves here)
pub fn candidates(plan: &EnginePlan, devices: &[GpuDevice]) -> Vec<BackendKind>;
pub fn open(kind: BackendKind, plan: &EnginePlan) -> Result<Box<dyn InferenceBackend>, EngineError>;
```

Order, first that opens wins. A fallback emits a `Warning`; a model-load
failure on the last candidate stays fatal, as today. It is never a silent
downgrade to the non-AI reference.

1. **Explicit override**: `EnginePlan::backend: Option<BackendKind>`
   (`#[serde(default)]`, from a future app setting), or `FS_ENGINE=<kind>`
   (`reference` keeps working for tests, `mock` is new).
2. **Apple**: `cfg(all(target_os = "macos", target_arch = "aarch64"))` and the
   `backend-apple` feature.
3. **TensorRT**: the `backend-tensorrt` feature, the planned device is NVIDIA,
   and a `libloading` probe finds `nvinfer` and `cudart` (versioned names per
   OS; the list gets verified against NVIDIA's packaging before it is
   written). Never linked: symbols resolved at runtime.
4. **Ncnn**: a Vulkan device (`backend-ncnn`, default on), else ncnn's CPU
   path, as today.

Features: `backend-ncnn = ["ncnn"]` as an alias, so the default build, the
CI command (`--features ncnn`) and `cfg(feature = "ncnn")` stay valid.
`backend-tensorrt` and `backend-apple` start empty and off by default.

### Mock

`MockEngine` (a `NetEngine`, test-only and behind `FS_ENGINE=mock` in the
worker): deterministic per-tile output (nearest-neighbour ×scale plus a
tile-position stamp in debug mode), configurable `max_in_flight`, latency
and completion order (to exercise out-of-order results), and fault
injection (`OutOfDeviceMemory` above a tile size, `DeviceLost` at frame N).
Through `ReferenceBackend`'s frame-level API, it runs the real pipeline end
to end in `fs-worker/tests/e2e.rs`. Today's `ReferenceBackend` stays as is:
the e2e tests compare its pixels bit for bit.

### Telemetry

`WorkerEvent::Progress` gains `engine: Option<EngineStats>`
(`#[serde(default)]`; worker and core ship together and `PROTOCOL_VERSION`
already guards mismatches, so it goes 1 → 2). `fs-core` forwards it in
`JobProgress`, and `queue.rs` puts it on `Telemetry` next to `render_fps`.
The HUD can then show the backend, tile size and in-flight count. Board VRAM
and load keep coming from the app's sampler; an engine's own `vram_used`
overrides it only where the board-level number doesn't exist.

## 7. Migration plan (diff-level outline)

Each step keeps `cargo check`, `cargo clippy -D warnings` and every
existing test green, and is verified bit-identical on a reference clip:
FFV1 output, `framemd5` before and after, on the GPU and the CPU path,
for General v3 ×2, x4plus ×3, NAFNet + x2plus, Real-CUGAN + AnimeVideo v3,
and Face Fix.

1. **`engine.rs` + features (no behaviour change).**
   - `fs-engine/src/engine.rs`: `BackendKind`, `Tensor`, `DeviceTensor`, `Ticket`, `NetEngine`, `EngineStats`.
   - `fs-engine/Cargo.toml`, `fs-worker/Cargo.toml`: `backend-ncnn = ["ncnn"]`, `backend-tensorrt = []`, `backend-apple = []`; worker default `["backend-ncnn"]`.
   - `InferenceBackend::stats()` with a default.
2. **Tiler on `NetEngine` (behaviour identical).**
   - `planar.rs`: `tiled_grid` → `run_tiled` over `&mut dyn NetEngine`; `tiled`, `tiled_into`, `tiled_into_even`, `*_reusing` keep their signatures as thin wrappers (closure → `ThreadedEngine`), so callers and tests don't change.
   - `engine.rs`: `ThreadedEngine` (long-lived workers, bounded queue).
   - Tests: existing tiling tests unchanged, plus scale 1/2/4, edge tiles, out-of-order completion and errors through the mock.
3. **ncnn on `NetEngine` (behaviour identical).**
   - `ncnn.rs`: `Sr`, `Naf`, `Cugan` hold a `ThreadedEngine` per network (built once at load, worker count = today's `workers`). `process` calls `run_tiled`. RIFE and Face Fix stay direct calls (whole frame, one at a time).
   - `NcnnBackend::stats()` fills tile, in-flight and reuse.
4. **Selection.**
   - `fs-worker/src/backend.rs`: `candidates` + `open`; `pipeline.rs::select_backend` calls it; `device.rs` unchanged.
   - `fs-types/src/worker.rs`: `EnginePlan::backend` (`serde(default)`), plus every `EnginePlan` literal.
   - TensorRT probe via `libloading` (workspace dependency; already in `Cargo.lock` transitively).
   - `Apple` and `TensorRt` openers: `unimplemented!("…: Step 3")` behind their features. The default build doesn't compile them.
5. **Mock.** `MockEngine` + `FS_ENGINE=mock`; an e2e test that runs the real worker with the mock and checks frame count, segment commits, OOM halving and resume.
6. **Telemetry.** `EngineStats` in `Progress` → `JobProgress` → `Telemetry`; protocol 1 → 2; ts-rs types regenerated. HUD changes are out of scope unless asked.

Not in Step 2: TensorRT and Core ML implementations themselves, model
conversion for them, in-process decode/encode, and HUD changes.

## 8. Notes on the Step 2 brief

- *"Extract tiling into a reusable module if it isn't shared yet"*: it already
  is (`planar.rs`, 7 tests: seams for any tile size, parallel = sequential,
  error propagation, reuse, area downscale per tile, padding, frame round
  trip). Step 2 adds the `NetEngine`-driven loop and the missing cases
  (scale 1 and 4, edge tiles, out-of-order completion).
- *"Overlap blending"*: today's tiler doesn't blend. It crops each tile's
  interior after enough context, which is exact (the tests compare tiled
  output to untiled). Adding blending would change the output, so it stays
  out unless asked for separately.
- *"Keep `max_in_flight()` tiles submitted, only if the audit shows it
  doesn't already"*: it already keeps 2 in flight on a GPU through threads,
  and the GPU is 92–97% busy. Step 2 moves that into the scheduler (the
  `NetEngine` contract) and doesn't change the frame loop.

## 9. NCNN backend (follow-up brief: "NCNN Vulkan backend")

That brief assumes an `engine/` module with `ModelConfig`, a mock and a
`factory.rs`. None of these exist: they are the Step 2 proposal above, which
isn't approved yet. The NCNN Vulkan backend itself already exists and ships
(`crates/fs-engine/src/ncnn.rs` + `cpp/bridge.{h,cpp}`). Writing a second one
in `engine/ncnn_vulkan.rs` would be the parallel architecture the working
rules forbid. So this section answers each requirement against the existing
backend. One gap turned up (device loss), and it is fixed.

### Bindings: `ncnn-rs` / `ncnn-bind` vs. our C++ shim

Checked on crates.io and in ncnn's headers (pinned `20261020` source):

- `ncnn-rs` and `ncnn-bind` 0.1.1 (one repo, `tpoisonooo/rust-ncnn`) were
  last released on **2021-12-19**. They are `bindgen` (0.59) bindings of
  ncnn's **C API** (`c_api.h`).
- ncnn's C API has no `VkMat`, no `VkCompute`, no `Extractor` overloads on GPU
  memory, and no device properties (vendor or PCI ids, queue counts). Its
  Vulkan surface is `ncnn_option_set_use_vulkan_compute`,
  `ncnn_net_set_vulkan_device` and a pipeline cache. Custom layers in the C
  API can't carry Vulkan pipelines, so our RIFE `Warp` layer (GLSL compiled
  through `compile_spirv_module`) couldn't be ported.
- **Decision: keep the thin C++ shim through `cxx`** (`cxx-build` compiles
  `cpp/bridge.cpp` against the headers of the ncnn we build). It reaches the
  full C++ API (`VulkanDevice`, `VkCompute`, `VkMat`, custom GPU layers,
  `load_param_mem`, in-memory weights), adds about 600 lines of C++, and
  converts C++ exceptions to `Result` at the boundary (no panics across FFI).
  A hand-written C ABI would add an unsafe layer and gain nothing over
  `cxx`.

### Model loading, FP16, device name and VRAM

- `.param` + `.bin` from files, from param text (×4→×2 pooling, Face Fix
  edits) or from in-memory weights (General v3's DNI blend): `Model::load`,
  `load_text`, `load_weights`, `load_text_weights`.
- Options (`Model::Model`): `use_vulkan_compute` on a GPU, **FP16 storage and
  packing on**, **FP16 arithmetic off**. ncnn turns FP16 off by itself on
  devices that can't do it. FP16 arithmetic stays off for accuracy, and the
  numbers below support that: realesrgan-ncnn-vulkan, which leaves it on,
  lands at 38.3 dB from an FP32 reference on x4plus, while ours lands at
  59.4 dB. On the RX 9070 XT it was also slower (the matrix cores already
  run the convolutions in FP16), with a worst-case error of 231/255.
- Device name, vendor and PCI ids come from ncnn (`gpu_list`). VRAM comes from
  our own Vulkan enumeration (`fs-hw::vulkan`: device-local heaps), matched
  to ncnn's index by PCI vendor and device id (`fs-worker/src/device.rs:pick`).
  Live VRAM use and load come from the app's telemetry sampler (section 4).

### Two tiles in flight: what ncnn's public API can express

ncnn's `VkCompute` is `record_upload` / `record_pipeline` / `record_download`
plus **`submit_and_wait()`** and `reset()`. There is no asynchronous submit or
fence to poll in the public API (`command.h`). One thread therefore can't
upload tile N+1 while tile N computes: each `submit_and_wait` blocks until
its command buffer is done.

The closest correct alternative is what the engine does today: **two
threads, each with its own `Extractor` and therefore its own `VkCompute`**,
running tile N and N+1 at once (`planar.rs:tiled_grid`, `workers = 2` on a
GPU). ncnn takes a queue from the device's compute-queue pool for each
submission (`VulkanDevice::acquire_queue`; RADV exposes 4 compute queues),
so one tile's upload and download overlap the other's compute on the GPU.
Measured: GPU busy 92% (General v3 720p) to 97% median (NAFNet + x2plus SD,
also with 2 CPU cores). More slots (3–4) were measured and don't help
(+2% at best, −40% for RRDB-class models through cache pressure), so it
stays at 2. Keeping the output on the GPU across stages is possible
(`Extractor::extract(…, VkMat&, VkCompute&)`) but was measured not to be
worth it (section 5, point 3).

### Normalization and channel order (no `ModelConfig` exists)

Per network, in code next to each call:

| Network | Input | Output |
|---|---|---|
| Real-ESRGAN (x4plus, x2plus, General v3, AnimeVideo v3), Real-CUGAN, NAFNet | planar RGB `[3][h][w]`, 0–1 | same |
| RIFE v4 | two frames planar RGB 0–1, plus a timestep plane | planar RGB 0–1 |
| YuNet | **BGR**, 0–255, letterboxed into 640×640 (`face::letterbox_bgr`) | score, box and landmark heads |
| CodeFormer | planar RGB, **−1…1**, 512×512 aligned face | same |

u8/u16 ⇄ f32 conversion is `planar::from_frame` / `to_frame` (RGB24 and RGB48).

### Errors

`cpp/bridge.cpp` throws, and `ncnn.rs:map_err` maps:
- `(ncnn -100)` (ncnn's allocation failure, VRAM included) →
  `EngineError::OutOfDeviceMemory { tile }` → the worker halves the tile and
  retries the frame;
- "device lost" → `EngineError::DeviceLost` → the worker exits with
  `DEVICE_LOST`, and the core restarts it with half the tile;
- anything else → `EngineError::Internal`.

**Fixed while checking this:** ncnn's `submit_and_wait` returns `-1` for any
`vkWaitForFences` failure, a lost device included, so a TDR reset used to
surface as `Internal`, and the core retried with the same tile. Now every
failed run on a GPU checks the device: `device_lost` creates a fence and
reads its status (`vkCreateFence` / `vkGetFenceStatus` through ncnn's
loader; neither touches a queue, so it is safe while the other tile thread
holds one) and reports "GPU device lost". This is compiled and the tests
pass, but **it hasn't been exercised on a real device loss**, which can't be
provoked safely here. The Vulkan spec lets a lost device report
`VK_ERROR_DEVICE_LOST` from both calls but doesn't require it, so on a
driver that doesn't, behaviour stays as before.

### Selection ("factory")

`fs-worker/src/pipeline.rs:select_backend`: `FS_ENGINE=reference` → the
reference backend; otherwise ncnn. `fs-worker/src/device.rs:ncnn_backend`
initialises Vulkan (bundled MoltenVK on macOS), enumerates GPUs with
`fs-hw`, takes the planned device UUID (or the best-ranked one), maps it to
ncnn's index by PCI ids, and prefers ncnn's own CPU path over a software
Vulkan device (llvmpipe, SwiftShader). Section 6 proposes extending this
with Apple and TensorRT candidates.

### Build

- ncnn **20261020**, the full-source release zip pinned by URL and SHA-256 in
  `third_party/manifest.toml` (`[ncnn]`). `cargo xtask ncnn` (part of
  `cargo xtask setup`) downloads and verifies it into `.cache/ncnn/src`.
- `crates/fs-engine/build.rs` (feature `ncnn`) builds it with CMake: static
  library, `NCNN_VULKAN=ON`, `NCNN_SIMPLEVK=ON` (the Vulkan loader or MoltenVK
  is loaded at runtime, so no Vulkan SDK is needed), glslang bundled, tools,
  examples, Python and tests off. It then compiles `cpp/bridge.cpp` with
  `cxx-build` and links ncnn and glslang statically.
- Overrides: `NCNN_SRC=<dir>` (another source tree) or `NCNN_DIR=<prefix>`
  (a prebuilt install with `include/ncnn` and `lib`).
- First build: ncnn plus glslang take about a minute on 12 threads.
- `cargo check --features backend-ncnn` doesn't work yet, because the feature
  doesn't exist; today's name is `ncnn`. The alias is step 1 of the migration
  plan (section 7).

### Comparison with realesrgan-ncnn-vulkan

`scripts/ncnn-compare.sh` takes the pinned upstream release
(`realesrgan-ncnn-vulkan-20220424-ubuntu.zip`, the same zip the bundled
models come from, so both engines use **identical weights**). It runs the
photo shipped with it through x4plus ×4, and 24 frames of its demo video
through AnimeVideo v3 ×2, with both engines. As an FP32 reference it uses
our engine on ncnn's CPU path (Vulkan hidden).

RX 9070 XT (RADV, Mesa in the build container):

| Case | ours vs FP32 | upstream vs FP32 | ours vs upstream |
|---|---|---|---|
| photo 220×220, x4plus ×4 | **59.36 dB** | 38.33 dB | 38.33 dB |
| video 640×480 ×24, AnimeVideo v3 ×2 | **61.44 dB** | 49.46 dB | 49.52 dB |
| same, tile reuse on | 61.44 dB | 49.46 dB | 49.52 dB |

Upstream itself is the less accurate of the two. With one tile covering the
whole photo the gap stays the same (38.38 dB), so tiling isn't the cause.
The cause is its FP16 arithmetic (ncnn's default; its log reports
`fp16-p/s/a=1/1/1` on this GPU). A fixed "ours vs upstream ≥ 40 dB" test
would therefore measure upstream's rounding, not our correctness. The script
passes when, for every case:

- **ours vs FP32 ≥ 50 dB.** 50 dB is an RMS error of 0.8 of an 8-bit level,
  well below anything visible. What's left is FP16 storage, which both
  engines use.
- **ours vs FP32 ≥ upstream vs FP32.** We're at least as close to the exact
  result as the reference tool.
- **ours vs upstream ≥ 35 dB.** A sanity bound on the gross agreement: same
  network, same weights, nothing systematically wrong such as colour order,
  scale or offset. 35 dB is an RMS of 4.6 levels, below upstream's own
  measured 38.3 dB deviation from FP32, with margin for GPUs whose FP16
  arithmetic rounds differently.
