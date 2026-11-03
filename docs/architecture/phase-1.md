# Phase 1 — System & UI Architecture (accepted)

This is the condensed reference copy of the Phase 1 design. Decisions marked
**[D#]** were confirmed before Phase 2 started.

## 0. Decisions & corrections to the original brief

| # | Brief | Reality | Decision |
|---|---|---|---|
| D1 | "NCNN natively on Metal/MPS" | ncnn has no Metal backend; it reaches Metal through MoltenVK | ncnn-Vulkan via **MoltenVK** is the baseline everywhere. A native `CoreMlBackend` comes later behind the same `InferenceBackend` trait. |
| D2 | "Zero-copy via FFmpeg pipes" | Pipes always copy; true zero-copy needs in-process decode + inference | **Zero-disk, bounded-copy streaming** by default; the `Engine` trait leaves room for true zero-copy backends. |
| D3 | Stock `*-ncnn-vulkan` CLIs | They only read image files/directories | Custom **`fs-worker`** links libncnn and streams raw frames between FFmpeg pipes. |
| D4 | "Resume from exact frame" | Encoded bitstreams can't be appended mid-GOP | **Segmented encoding, 4 s closed-GOP segments**, committed atomically. Resume restarts at the last committed segment. Segments also enable multi-GPU. |
| D5 | "TensorRT" / "INT8" | ncnn ≠ TensorRT; INT8 hurts super-resolution quality | FP16 everywhere, with cooperative-matrix where exposed. TensorRT is an optional later backend. |
| D6 | FFmpeg licensing | Linking libav* or bundling GPL encoders creates obligations | FFmpeg runs as a **separate process**. ~~LGPL build~~ **Revised with D9:** FrameScale is GPL, so it ships FFmpeg's **GPL build** (libx265: software HEVC Main10 with HDR10 SEI; libx264). CI gate now rejects only `--enable-nonfree`. |
| D7 | ProRes everywhere | `prores_ks` is an unlicensed reverse-engineered implementation | ProRes only via `prores_videotoolbox` on macOS. |
| D8 | Repo tooling | — | pnpm workspace + Cargo workspace (+ `cargo xtask`). |
| **D9** | Freemium + Pro licensing | **Superseded (2026-11-03):** FrameScale is free and open source under **GPL-3.0-or-later**, funded by optional donations (GitHub Sponsors, Ko-fi). | The Ed25519 token layer, Pro gates, activation modal and worker feature checks were removed. All features for everyone. Phase 4 now covers community growth and release distribution instead of Pro conversion. |
| **D10** | Face Fix with CodeFormer | CodeFormer's weights are under the S-Lab License 1.0: use and redistribution for **non-commercial purposes only** | **Downloaded on first use (2026-11-03):** not in the installers, so they stay entirely OSI-licensed (a SignPath Foundation requirement for free Windows code signing). Turning Face Fix on shows the license; with consent the app downloads FeiGeChuanShu's ncnn conversion and S-Lab's LICENSE (189 MB) from the URLs pinned in `third_party/manifest.toml` (`ondemand = true`, SHA-256 and size verified) into `<app data>/models/codeformer/`. Previews and renders with Face Fix are refused until it's there. The GPL doesn't reach the weights. |

## 1. Process architecture

```
Tauri app process (Windows: whole tree in a KILL_ON_JOB_CLOSE Job Object)
├─ WebView (React): UI only; no secrets, no process spawning
│    invoke() ⇄ commands · Channel<T> ← progress/telemetry (≤10 Hz) · fs:// ← preview frames
└─ Rust core: JobManager · HardwareService · LicenseVerifier · StorageGuard/TempJanitor
             · Journal (SQLite WAL) · Supervisor + Reaper
      │ spawn 1 per (job, GPU); stdin JSON control + license; stdout JSON-lines events
      ▼
   fs-worker (own process group; parent watchdog: pidfd / kqueue / job inheritance)
      ffmpeg decode ─raw pipe─▶ reader ▸ convert ▸ GPU stages ▸ convert ▸ writer ─raw pipe─▶ ffmpeg encode (per segment)
```

A crash in the GPU driver or in ncnn kills only the worker. The supervisor then resumes from the journal.

## 2. UI

Three columns: Queue 280 px (220–400) · Viewport (flex) · Inspector 340 px (300–440). TitleBar is 44 px, StatusBar 28 px, minimum window 1280×760.

Component tree:

```
App
├─ AppShell
│  ├─ TitleBar → ProjectSwitcher, DeviceSelector, EcoToggle, LicenseBadge
│  ├─ PanelGroup
│  │  ├─ QueueSidebar → DropZone (Tauri drag-drop), QueueToolbar (ProGate), QueueList (virtualized) → QueueItem
│  │  ├─ Viewport → ViewportToolbar, CompareStage (one WebGL2 canvas: SplitHandle, PixelLoupe),
│  │  │             Timeline → Scrubber, SnippetRangeBrush, SceneCutMarkers, RenderPreviewButton
│  │  └─ InspectorSidebar → PipelineChain (StageNode ×5), StageInspector, OutputSettings,
│  │                        TelemetryHUD → DeviceCard, MemoryBar, FpsSparkline, ThermalGauge, DiskMeter
│  └─ StatusBar
└─ Overlays → ProActivationModal, ResumeRecoveryDialog, StorageWarningDialog, Toaster
```

- **Pipeline:** a linear chain in fixed canonical order (Deinterlace → Denoise → Upscale → FaceFix → Interpolate), not a free-form graph.
- **Telemetry:** written to a ring buffer outside React and read with `useSyncExternalStore` at 4 Hz; sparklines draw to canvas.
- **Split slider:** moving it changes one shader uniform; nothing reflows.
- **Preview:** the 2 s snippet runs through the *same compiled plan* as the final render. JPEG q92 frames for playback, PNG for the paused frame and loupe. HDR is tone-mapped for preview only.
- **Materials:** native Mica on Windows and vibrancy on macOS. Linux uses solid surfaces. `prefers-reduced-transparency` is respected.
- **ProGate:** cosmetic only; enforcement happens in Rust and in the worker.

## 3. Media pipeline

`PROBE → ANALYZE → PLAN → PREFLIGHT → RENDER (segments) → CONCAT → MUX → VERIFY`

- **ANALYZE** is one fast hardware-decoded pass: `idet, vfrdet, fps=R, scale=-2:270, scdet`. It yields interlace detection, VFR detection, and scene cuts in CFR frame-index space.
- **Invariant:** rawvideo pipes carry no timestamps, so **frame index is the clock**. That makes CFR conform mandatory.
- **Color:** `ColorProfile` from ffprobe (untagged ≥720p → BT.709, SD → BT.601). Decode to rgb24/rgb48le with the source matrix and range; the transfer curve passes through untouched. Encode with the same tags plus HDR10 static metadata. DV RPU and HDR10+ are stripped with a warning.
- **Audio:** never decoded. `-c:a copy` plus an explicit A/V start-offset correction at mux, with a fallback when the container is incompatible.
- **RIFE:** at a scene cut, duplicate the frame instead of morphing; a fallback MAD check catches missed cuts.

## 4. Hardware

- **Discovery:** Vulkan (`ash`); devices identified by **UUID**, never by index.
- **Ranking:** discrete ≫ integrated, then VRAM and compute features.
- **Telemetry:** NVML / ADLX / sysfs / IGCL / IOKit + thermalState; `VK_EXT_memory_budget` is vendor-neutral.
- **Tiles:** auto-tuned, cached per (device, driver, model), and bounded so each dispatch stays under 500 ms (TDR-safe).
- **Eco governor:** duty-cycle PI controller on temperature (or a fixed duty). Turns on automatically on battery.

## 5. Backend

Crates: `fs-types`, `fs-hw`, `fs-media`, `fs-proc`, `fs-license`, `fs-journal`, `fs-engine`, `fs-worker`, app.

| Failure | Recovery |
|---|---|
| VRAM OOM | halve tile, retry same frame |
| Device lost / TDR / crash | restart worker from last segment, tile ×0.5, 3 strikes → poison segment |
| Disk full | hard pause + dialog, resume when space is available |
| HW encoder unavailable | software encoder fallback |
| App killed | journal `running` → `interrupted` → resume dialog |
| Resume misalignment | per-segment xxh3 of first input frame, search ±3 frames |

- **Pause:** soft (backpressure) or hard (checkpoint + exit).
- **Temp janitor:** deletes a job dir only if it's unlocked, not resumable, and older than 24 h.
- **Licensing:** ~~Ed25519 Pro tokens~~ removed by D9 (GPL-3.0, no feature gates). BYOM accepts ncnn/ONNX only, never pickles. The shell plugin is not exposed to JS. Updater uses minisign.
