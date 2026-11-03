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
| `rife.Warp` Vulkan shader (adapted, in fs-engine) | — | MIT, © 2020 nihui (rife-ncnn-vulkan) |

## Bundled AI models

| Model | License | Source |
|---|---|---|
| Real-ESRGAN (x4plus, x4plus-anime, animevideov3) | BSD-3-Clause | https://github.com/xinntao/Real-ESRGAN |
| Real-ESRGAN General v3 (realesr-general-x4v3 + wdn), converted to ncnn at build time from the official v0.2.5.0 weights | BSD-3-Clause, © 2021 Xintao Wang | https://github.com/xinntao/Real-ESRGAN |
| Real-CUGAN (se, pro) | MIT | https://github.com/bilibili/ailab / nihui/realcugan-ncnn-vulkan |
| NAFNet-SIDD-width32, converted to ncnn at build time | MIT, © 2022 megvii-model (its LICENSE, shipped in `models/nafnet/`, also carries BasicSR's Apache-2.0 notice) | https://github.com/megvii-research/NAFNet (weights via huggingface.co/nyanko7/nafnet-models, SHA-256 pinned) |
| RIFE 4.6, 4.22 lite, 4.25, 4.26 | MIT, © 2021 hzwer (weights); ncnn conversions MIT, © 2020 nihui | https://github.com/hzwer/Practical-RIFE / nihui/rife-ncnn-vulkan / TNTwise/rife-ncnn-vulkan (commit `13338e3`) |
| YuNet face detector (2023mar), ncnn conversion | MIT, © 2020 Shiqi Yu | https://github.com/opencv/opencv_zoo / huggingface.co/lightcella/face-pipeline-ncnn |
| CodeFormer (face restoration), ncnn conversion | **S-Lab License 1.0: non-commercial use only**, © 2022 S-Lab | https://github.com/sczhou/CodeFormer (weights) / https://github.com/FeiGeChuanShu/CodeFormer-ncnn (ncnn conversion, commit `cfe0fe2`) |

### CodeFormer is for non-commercial use only

CodeFormer is the one bundled component that is not freely licensed. Its
weights, and the ncnn conversion of them in `models/codeformer/`, may be used
and redistributed for non-commercial purposes only. FrameScale AI is
distributed non-commercially. The GPL covers FrameScale's own code, not these
weights: anyone who redistributes FrameScale commercially must remove
`models/codeformer/` or obtain a license from S-Lab. The conversion repository
states no license of its own. The license also ships as
`models/codeformer/LICENSE`:

```text
S-Lab License 1.0

Copyright 2022 S-Lab

Redistribution and use for non-commercial purpose in source and 
binary forms, with or without modification, are permitted provided 
that the following conditions are met:

1. Redistributions of source code must retain the above copyright 
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright 
   notice, this list of conditions and the following disclaimer in 
   the documentation and/or other materials provided with the 
   distribution.

3. Neither the name of the copyright holder nor the names of its 
   contributors may be used to endorse or promote products derived 
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

In the event that redistribution and/or use for commercial purpose in 
source or binary forms, with or without modification is required, 
please contact the contributor(s) of the work.
```
