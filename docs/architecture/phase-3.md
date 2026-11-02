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

- **ncnn inference backend** (`fs-engine/cpp`): the worker currently runs the CPU
  *reference* backend (nearest-neighbour + blend, clearly labelled non-AI). The
  `InferenceBackend` trait, tiling/OOM backoff and interpolation scheduling are
  ready for it.
- **2 s preview rendering** + `fs://` protocol, and real **thumbnails**: the
  viewport shows a labelled procedural sample until then.
- HDR10 *static* metadata on hardware encoders is tags-only (warned in the UI);
  SVT-AV1 embeds it in the bitstream.
- macOS: battery state and GPU temperature (IOKit) not yet wired; Linux/Windows are.
- One unexplained `fs-core` job-test failure (worker exited "usage" with no
  message) in ~30 runs; not reproduced since. Failure messages now carry the
  worker's exit code + stderr tail.
- The Tauri host only compiles on CI for Windows/macOS (C dependencies);
  pure-Rust crates are cross-checked locally.
