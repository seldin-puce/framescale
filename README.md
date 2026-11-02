# FrameScale AI

A local, cross-platform desktop app for AI video upscaling, restoration, and
frame interpolation. It runs on NVIDIA, AMD, Intel, and Apple Silicon GPUs.
Built on Tauri v2 (Rust + React).

```
framescale/
├── Cargo.toml                  Cargo workspace + pinned Rust dependency BOM + release profiles
├── rust-toolchain.toml         Rust 1.99.1 (pinned)
├── package.json                pnpm workspace root scripts
├── pnpm-workspace.yaml         workspace + supply-chain policy (24 h release-age gate)
├── tsconfig.base.json          strict TS settings shared by every package
│
├── apps/desktop/               ── the shipped application ──
│   ├── src/                    React 19 UI (Tailwind v4 tokens in styles/theme.css)
│   ├── assets/app-icon.svg     icon source → `pnpm tauri icon`
│   └── src-tauri/              Rust host crate `framescale`
│       ├── tauri.conf.json         base config: window, CSP, bundle, sidecars, updater
│       ├── tauri.{macos,windows,linux}.conf.json   per-OS bundling & signing (JSON merge-patch)
│       ├── capabilities/main.json  least-privilege IPC grants (no shell plugin)
│       ├── entitlements.plist      hardened runtime (intentionally empty)
│       ├── scripts/sign-windows.ps1 Authenticode hook (Azure Artifact Signing)
│       ├── binaries/  frameworks/  resources/models/   ← populated by `cargo xtask setup` (git-ignored)
│       └── src/                    lib.rs (builder, window materials), main.rs
│
├── crates/                     ── domain crates (shared by host + worker) ──
│   ├── fs-types                IPC/worker contract types → ts-rs → packages/ipc-types
│   ├── fs-hw                   Vulkan discovery, ranking, telemetry backends
│   ├── fs-media                ffprobe, typed FFmpeg argv builders, progress/stderr parsing
│   ├── fs-proc                 Job Objects / process groups / reaper
│   ├── fs-license              Ed25519 entitlements (linked into host AND worker)
│   ├── fs-journal              SQLite WAL journal (+ migrations/)
│   ├── fs-core                 job runner: analyze → plan → supervised render → concat → mux; recovery policy
│   ├── fs-engine               InferenceBackend, ncnn bridge (cpp/), tiler
│   └── fs-worker               sidecar binary: decode pipe → GPU → segmented encode
│
├── packages/ipc-types/         generated TS mirror of fs-types (CI checks drift)
├── xtask/                      `cargo xtask`: setup, sidecars, models, pin, gate, gen-types, doctor
├── third_party/
│   ├── manifest.toml           every external artifact: URL + SHA-256 (+ mirror)
│   └── ffmpeg/build-macos.sh   LGPL static FFmpeg build for macOS
├── docs/
│   ├── architecture/phase-1.md accepted system design
│   └── build-and-release.md    prerequisites, sidecars, signing, secrets
├── THIRD_PARTY_NOTICES.md      shipped inside the bundle
└── .github/workflows/          ci.yml (lint/test/gate/drift), release.yml (sign + notarize + draft release)
```

Quick start: see [docs/build-and-release.md](docs/build-and-release.md).
