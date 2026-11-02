# Build & Release

## Prerequisites

| Platform | Needs |
|---|---|
| All | Rust (the version in `rust-toolchain.toml` installs automatically), Node ≥ 22.12, pnpm 12, `curl` |
| Linux | C toolchain, `libwebkit2gtk-4.1-dev libsoup-3.0-dev librsvg2-dev libayatana-appindicator3-dev patchelf` (Fedora: `webkit2gtk4.1-devel libsoup3-devel librsvg2-devel libappindicator-gtk3-devel`). Fedora Atomic: work inside a `toolbox`. |
| macOS | Xcode CLT; for the FFmpeg source build also `brew install meson ninja nasm autoconf automake libtool pkg-config cmake` |
| Windows | MSVC Build Tools (C++), WebView2 runtime (preinstalled on Win 11), PowerShell 7 (`pwsh`) for the signing hook |

Run `cargo xtask doctor` to check.

## First checkout

```sh
pnpm install
cargo xtask setup          # verified FFmpeg/ffprobe + fs-worker sidecars, MoltenVK (macOS), bundled models
pnpm dev                   # tauri dev (Vite on :1420 + Rust host)
```

## Sidecars

Tauri resolves `bundle.externalBin` entries by target triple:

```
apps/desktop/src-tauri/binaries/
  ffmpeg-x86_64-unknown-linux-gnu        ffmpeg-x86_64-pc-windows-msvc.exe
  ffprobe-…                              fs-worker-…
  ffmpeg-aarch64-apple-darwin            ffmpeg-universal-apple-darwin   (lipo, only for universal builds)
```

`cargo xtask sidecars --target <triple>` produces these files:

1. **FFmpeg.**
   - Windows/Linux: downloads the pinned LGPL build and verifies its SHA-256.
   - macOS: builds from verified sources (`third_party/ffmpeg/build-macos.sh`).
2. **License/capability gate.** The build fails on `--enable-gpl` or `--enable-nonfree`, or if a filter or encoder the pipeline needs is missing (`[ffmpeg.gate]` in the manifest).
3. **fs-worker.** Builds `fs-worker` for the target in release mode.
4. **macOS only.** Copies `libMoltenVK.dylib` into `frameworks/`. Tauri places it in `Contents/Frameworks` and signs it.

`--stub` writes empty placeholders. `tauri-build` only checks that sidecars exist, so lint and test CI jobs don't need real binaries.

## Third-party pinning

`third_party/manifest.toml` is the only source of external artifacts.

- To add or bump an artifact, set `url`, clear `sha256 = ""`, run `cargo xtask pin`, and review the diff.
- **Before the first public release**, mirror every artifact to our own storage and set `mirror`. BtbN autobuilds are pruned after some weeks.

## Release (tag `vX.Y.Z`)

`.github/workflows/release.yml` builds Linux (on Ubuntu 22.04, for the glibc baseline), Windows, and macOS arm64 + x86_64. It signs everything, notarizes the macOS builds, and creates a **draft** GitHub release with `latest.json` for the updater.

### Secrets

| Secret | Purpose |
|---|---|
| `TAURI_SIGNING_PRIVATE_KEY`, `…_PASSWORD` | Updater signatures (`pnpm tauri signer generate`); put the public key in `plugins.updater.pubkey` |
| `APPLE_CERTIFICATE` (base64 .p12), `APPLE_CERTIFICATE_PASSWORD`, `APPLE_SIGNING_IDENTITY` | Developer ID Application signing |
| `APPLE_API_ISSUER`, `APPLE_API_KEY`, `APPLE_API_KEY_P8` | Notarization via an App Store Connect API key (preferred over Apple ID + app-specific password) |
| `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_SIGNING_METADATA_JSON` | Windows Authenticode via Azure Artifact Signing (HSM-backed; EV-style SmartScreen reputation without a USB token) |

### Signing coverage

- **macOS:** with hardened runtime and *empty* entitlements, the app, every sidecar, and `libMoltenVK.dylib` are signed with the same Team ID, so library validation passes. Notarization is then stapled by the Tauri bundler.
- **Windows:** `scripts/sign-windows.ps1` is invoked for the main exe, the sidecars, and the NSIS installer. If the signing env vars are missing, it skips with a warning (local builds).
- **Linux:** packages are unsigned. Updater artifacts are minisign-signed.

## Items to verify on the first real CI run

These can't be tested on a Linux dev box. Each is called out where it's configured:

- [ ] The `signCommand` working directory resolves `scripts/sign-windows.ps1` (Tauri runs it from `src-tauri/`).
- [ ] macOS cross build x86_64 on arm64 runner: `build-macos.sh` + SVT-AV1 4.3 against FFmpeg 8.1.
- [ ] ncnn's SimpleVK loader finds `@executable_path/../Frameworks/libMoltenVK.dylib`. If it doesn't, fs-worker passes an explicit path (Phase 3).
- [ ] `hevc_videotoolbox` HDR10 metadata propagation, and `prores_videotoolbox` availability on x86_64 Macs.
