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
  ×2/×3) and AnimeVideo-v3 ×2/×3/×4. RIFE v4.6 runs through our own `rife.Warp`
  custom layer (CPU bilinear warp; ncnn moves blobs to and from the GPU).
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
| RIFE v4.6, 1920×1080 | 4.5 |
| RIFE v4.6, 3840×2160 | 1.14 |

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

- **Denoise (Real-CUGAN)** and **Face Fix (BYOM)**: shown as "Soon"; the host
  rejects them at enqueue. Real-CUGAN needs its multi-pass cache pipeline.
- Windows CPU path is single-threaded (MSVC OpenMP needs vcomp140.dll).
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
