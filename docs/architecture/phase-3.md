# Phase 3: Core Implementation (status)

## What exists

| Deliverable | Where | Verified by |
|---|---|---|
| (a) Multi-platform GPU discovery | `crates/fs-hw` (`vulkan.rs`, `rank.rs`, `telemetry.rs`) | Real Vulkan run (lavapipe) incl. `VK_EXT_memory_budget`; ranking unit tests; Windows/macOS cross-clippy |
| (b) FFmpeg command builder: VFR→CFR, scene detection, HW flags | `crates/fs-media` (`argv`, `analyze`, `rate`, `color`, `encoders`, `mux`, `plan`) | 30 tests incl. real-FFmpeg integration: exact frame counts, cuts at 48/96, frame-exact seek, HDR10 passthrough |
| (c) STDERR/STDOUT parsing for speed/progress | `fs-media::progress` (`-progress`), `classify` (stderr → typed issues), worker JSON events | Captured real log fixtures |
| (d) Recovery: OOM, disk full, pause/resume via frame offset | `fs-engine::tiling`, `fs-core::{recovery, storage, runner}`, `fs-worker`, `fs-journal` | Policy unit tests; **bit-identical resume** after hard pause (e2e); full job with hard pause → resume |
| (e) Zombie reaper (process groups / Job Objects) | `crates/fs-proc` | Grandchild killed by group kill; PID-ledger reaps orphans but not recycled PIDs |
| UI (a) workspace layout | `apps/desktop/src/components/shell`, `queue`, `inspector` | Headless-Chromium screenshots, zero console errors |
| UI (b) split-screen compare slider | `src/gl/compare.ts`, `components/viewport/CompareStage.tsx` | Dragged + zoomed to 1488 % in a real browser (nearest-neighbour pixel peeping) |
| UI (c) telemetry HUD | `components/telemetry/TelemetryHUD.tsx`, `stores/telemetry.ts` | Live demo telemetry; 4 Hz render cap; canvas sparkline |

## Inference (ncnn backend)

- `crates/fs-engine/build.rs` builds the pinned ncnn (`cargo xtask ncnn`) with
  Vulkan via SimpleVK (no SDK; the loader is dlopen'd) and ncnn's
  self-contained OpenMP on Linux/macOS. `cpp/bridge.cpp` is a cxx bridge.
- Models: Real-ESRGAN x4plus / x4plus-anime (native ×4, exact area-downscale to
  ×2/×3), AnimeVideo-v3 ×2/×3/×4 and General v3 (live action, native ×4, the
  default). "x4plus" at ×2 runs RealESRGAN_x2plus instead: the same RRDBNet on
  pixel-unshuffled input (ncnn `Reorg`), converted at build time
  (`convert = "rrdb"`), ~3.4× faster than ×4 + downscale. Its tiles are kept
  even so each tile's 2×2 unshuffle grid stays in phase with the frame (odd
  tiles doubled the seam error). Checks: our x4plus conversion vs the
  official ncnn export mean 0.14 / max 5 levels; x2plus (GPU fp16) vs
  PyTorch mean 0.86 levels, as close as the official x4plus export (0.63).
  General v3 has no official ncnn export: `cargo xtask models` converts the
  official .pth (`convert = "srvgg"`: the pickle is interpreted with an
  allow-list, never executed) into fp32 ncnn files. Converting AnimeVideo-v3
  the same way reproduces xinntao's own export (mean |Δ| 0.04, max 2 levels).
  Its noise-removal slider blends the general and weak-denoise checkpoints
  (DNI, s·a + (1 − s)·b) when the model loads. RIFE 4.6, 4.22 lite, 4.25 (default) and
  4.26 run through our own `rife.Warp` custom layer: a Vulkan bilinear warp
  (pack1/pack4 shaders after nihui's, fp32 coordinates) with a CPU twin for CPU
  nets. 4.25 and 4.26 need frames padded to a multiple of 64, the rest 32.
- Tiling with 10 px context in Rust (`planar::tiled`). It's bit-exact against
  untiled output for a 1-px-receptive-field model, and ≈0.1/255 mean seam
  error with the real models.
- Device mapping: the worker enumerates Vulkan itself (same loader, same order)
  and cross-checks PCI vendor/device ids against ncnn before trusting an index.
  CPU-type Vulkan devices (llvmpipe) use ncnn's native CPU path instead.
- macOS: `@executable_path/../Frameworks/libMoltenVK.dylib` is passed to
  `create_gpu_instance`. `FS_VULKAN_DRIVER` overrides it anywhere.
- Verified: real models on ncnn CPU + Vulkan (lavapipe); full 10 s job with
  real Real-ESRGAN, hard pause and resume: 12 s on 6 CPU cores.

## Measured performance (RX 9070 XT, Fedora 44, Mesa 26.2.5 RADV)

`cargo run --release -p fs-engine --features ncnn --example bench` (inference
plus CPU-side conversion; excludes decode/encode):

| Workload | fps |
|---|---|
| Real-ESR AnimeVideo-v3 ×2, 1920×1080 → 3840×2160 | **10.9** |
| Real-ESRGAN x4plus ×4, 960×540 → 3840×2160 | **1.31** |
| Real-ESRGAN x4plus ×2, 1920×1080 → 3840×2160 | 0.31 |
| Real-ESRGAN x4plus ×2 via x2plus, 1620×1080 → 3240×2160 (incl. load) | ~0.6 (was 0.18) |
| Real-ESRGAN General v3 ×4, 960×540 → 3840×2160 | **13.5** |
| Real-ESRGAN General v3 ×2, 1920×1080 → 3840×2160 | 2.4 |
| NAFNet-SIDD denoise, 720×576 / 1920×1080 | 26.8 / 7.1 |
| RIFE 4.6, 1920×1080 | 15.6 |
| RIFE 4.22 lite / 4.25 / 4.26, 1920×1080 | 14.5 / 14.1 / 14.3 |
| RIFE 4.6, 3840×2160 | 3.0 |

RIFE was 4.4 fps (4.6) and 2.3 fps (4.25) at 1080p while `rife.Warp` ran on
the CPU: every warp (8 in 4.6, 18 in 4.25) copied its blobs GPU → CPU → GPU.
With the warp on the GPU the remaining time is mostly CPU-side conversion and
padding, so the network version barely matters on this GPU.

How we got there (from 2.7 / 0.56 fps):
- Host Mesa 26.2 vs Debian's 25.0: RDNA4 cooperative matrix exposed; ~1.5×.
- CPU overhead 95 → ~5 ms/frame: slice-based tile extraction, multi-threaded
  packing, a reused output buffer.
- Two concurrent ncnn extractors per GPU: CPU slicing/stitching overlaps GPU work.
- Tile sweep: small tiles win on RDNA4 (256 px light / 192 px heavy).
- FP16 *arithmetic* measured and rejected: slower for SR and up to 231/255 error.

The official `realesrgan-ncnn-vulkan` (2022 build) on the same GPU took 8.8 s for
x4plus 540p→4K (incl. load) vs our 0.76 s/frame. The Arc iGPU is ~60× slower
than the 9070 XT, which the default ranking correctly avoids.

## Bugs real binaries caught (and the fixes)

1. zimg only outputs planar formats. Requesting `rgb24` made FFmpeg insert swscale
   with **BT.601** defaults after zscale. Fixed: zscale → `gbrp` → lossless pack.
2. FFmpeg ≥ 7 takes encoder colour tags from frames: `-color_trc` alone produced
   untagged AV1. Fixed: `setparams` stamps frames after zscale.
3. Matroska's ms timestamps (41/42 ms at 24 fps) made `vfrdet` report 66 % VFR on
   CFR files. Fixed: require a >10 % min/max duration spread.
4. `sysinfo` 0.40 made `System::new()` fallible; a Zustand selector returned a fresh
   `[]` (render loop); a shader varying was off by 2×. All fixed.

## Known open items

- **Denoise (Real-CUGAN)** (done): a 1× stage, Real-CUGAN ×2 at the chosen
  level (SE: conservative, denoise 1x/2x/3x; Pro: conservative, 3x), then an
  exact area downscale. Its SE blocks pool globally (`gap0…gap3`), so tiling
  alone gives each tile its own brightness/contrast. Each frame runs one
  statistics pass on a ≤512² downscale (the "very rough" sync of
  realcugan-ncnn-vulkan) and every tile gets those means injected. Tiled output
  matches one whole-frame tile (GPU max Δ 1/255). The ncnn Pro models use plain
  0–1 RGB (measured; the PyTorch `(x-0.15)/0.7` is already folded in).
- **Face Fix** (done; both models bundled, CodeFormer is non-commercial, D10):
  YuNet (MIT) finds faces and 5 landmarks on a 640² letterbox; each face is
  aligned to the FFHQ 512 template (least-squares similarity), restored by
  CodeFormer (FeiGeChuanShu's ncnn conversion in `models/codeformer`: encoder,
  CPU top-1 codebook lookup, generator) and pasted back with a 10 % feathered
  border. Runs after Upscale on the final pixels.
  The conversion bakes CodeFormer's fidelity `w` into four fuse `Eltwise`
  coefficients (`1, 0.7`); the generator param is rewritten in memory, so the
  fidelity slider is real. Measured: about 240 ms per face on RX 9070 XT,
  2.2 s per face on CPU. Faces aren't tracked across frames (no temporal
  smoothing yet).
- **NAFNet denoise** (done, live action): NAFNet-SIDD-width32 (megvii, MIT;
  trained on real camera noise) is converted at build time like General v3
  (`convert = "nafnet"`, fp16 weights, 58 MB). On the CPU it matches megvii's
  own PyTorch code to max 1 level; on the GPU too. Two ncnn pitfalls on the
  way: its Vulkan `Reduction` dispatches one workgroup row per output, and
  Vulkan caps that at 65535, so a channel reduction over a tile larger than
  256×256 left most pixels uncomputed. And a stored squared feature map can
  overflow fp16. LayerNorm2d therefore runs as our `fs.ChannelNorm` custom
  layer (one thread per pixel, statistics in fp32 registers, 2-D dispatch).
  Its channel attention averages over each tile, so tiles are twice the SR
  tile and split evenly (no thin remainder tiles): mean 0.44 / max 6 levels
  from a whole-frame run, no visible blocks. Strength levels mix 40 / 60 /
  80 / 100 % of the network's output with the input.
- Windows CPU path (done): MSVC builds have no OpenMP (vcomp140.dll would need
  the dynamic CRT), so ncnn runs each call single-threaded and the backend
  runs one tile per core (`cores/2`, at most 8; about 8 tiles per worker).
  Measured on a 24-thread Arrow Lake at 640×360: AnimeVideo-v3 ×2 1376 → 532
  ms/frame, Real-CUGAN 2755 → 1263 ms (2.6× / 2.2×; ncnn's OpenMP on Linux:
  482 / 962 ms). RIFE and Face Fix stay single-threaded there (whole-frame and
  per-face networks); the GPU is the intended path anyway. Linux/macOS keep
  ncnn's built-in simpleomp. Force the tile mode anywhere with
  `FS_CPU_TILE_WORKERS=n` (benchmarks).
- SR pre/post-processing runs on the CPU (u8/u16 ⇄ f32); GPU shaders for it are an
  optimisation for later.
- Measured on AMD RDNA4 only; NVIDIA/Apple may prefer other tile sizes. A
  per-device calibration pass (cache best tile per device+model) is the
  long-term fix.
- **2 s preview rendering** + `fs://` protocol, and real **thumbnails**: the
  viewport shows a labelled procedural sample until then.
- HDR10 *static* metadata: embedded in the bitstream by **x265** (HEVC) and
  SVT-AV1 (AV1), verified by a round-trip test. Hardware encoders stay
  tags-only (warned in the UI);
  SVT-AV1 embeds it in the bitstream.
- macOS: battery state and GPU temperature (IOKit) not yet wired; Linux/Windows are.
- One unexplained `fs-core` job-test failure (worker exited "usage" with no
  message) in ~30 runs; not reproduced since. Failure messages now carry the
  worker's exit code + stderr tail.
- The Tauri host only compiles on CI for Windows/macOS (C dependencies);
  pure-Rust crates are cross-checked locally.
