# Third-Party Notices — FrameScale AI

FrameScale AI itself is licensed under the GNU General Public License v3.0 or
later (see LICENSE). It bundles the following third-party components, whose
own licenses apply to those components. This file ships inside the application bundle.

> Before the first release, replace this hand-maintained list with generated
> output (`cargo about` for Rust crates plus a pnpm license report) and keep
> this header.

## Executables shipped as separate programs (not linked)

| Component | Version | License | Notes |
|---|---|---|---|
| FFmpeg (`ffmpeg`, `ffprobe`) | 8.1.3 | GPL-3.0-or-later (built with `--enable-gpl --enable-version3`) | Unmodified upstream source: https://ffmpeg.org/releases/ffmpeg-8.1.3.tar.xz. Build configuration is printed by `ffmpeg -buildconf`. Windows/Linux builds: BtbN/FFmpeg-Builds (GPL variant, `autobuild-2026-11-02-12-51`). macOS: `third_party/ffmpeg/build-macos.sh`. Corresponding source is attached to each release. |
| x265 (inside FFmpeg) | 4.2 (macOS); BtbN's pinned build elsewhere | GPL-2.0-or-later | https://download.videolan.org/videolan/x265/ |
| x264 (inside FFmpeg, Windows/Linux) | BtbN's pinned build | GPL-2.0-or-later | https://code.videolan.org/videolan/x264 |
| SVT-AV1 (inside FFmpeg) | 4.3.0 | BSD-3-Clause-Clear + AOM Patent License 1.0 | |
| dav1d (inside FFmpeg) | 1.5.4 | BSD-2-Clause | |
| zimg (inside FFmpeg) | 3.0.6 | WTFPL | |

## Libraries linked into FrameScale binaries

| Component | Version | License |
|---|---|---|
| ncnn | 20261020 | BSD-3-Clause |
| MoltenVK (macOS) | 1.4.2 | Apache-2.0 |
| Tauri and its plugins | 2.x | Apache-2.0 OR MIT |
| SQLite (via rusqlite, bundled) | — | Public Domain |

## Bundled AI models

| Model | License | Source |
|---|---|---|
| Real-ESRGAN (x4plus, x4plus-anime, animevideov3) | BSD-3-Clause | https://github.com/xinntao/Real-ESRGAN |
| Real-CUGAN (se, pro) | MIT | https://github.com/bilibili/ailab / nihui/realcugan-ncnn-vulkan |
| RIFE v4.6 | MIT | https://github.com/hzwer/ECCV2022-RIFE / nihui/rife-ncnn-vulkan |
| YuNet face detector (2023mar), ncnn conversion | MIT, © 2020 Shiqi Yu | https://github.com/opencv/opencv_zoo / huggingface.co/lightcella/face-pipeline-ncnn |

Models such as CodeFormer are **not** distributed with FrameScale AI. Users
who choose to load them (Bring-Your-Own-Model) do so under those models' own
license terms. CodeFormer's weights are under the S-Lab License 1.0
(non-commercial use only); Face Fix loads the ncnn conversion
(github.com/FeiGeChuanShu/CodeFormer-ncnn, `models/` folder) from a folder the
user chooses.
