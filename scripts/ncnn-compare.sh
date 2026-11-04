#!/usr/bin/env bash
# Compare FrameScale's ncnn engine against xinntao's realesrgan-ncnn-vulkan
# (the pinned 20220424 release in third_party/manifest.toml) on the same
# weights: a photo with Real-ESRGAN x4plus ×4, and frames of the demo video
# shipped in that release with AnimeVideo v3 ×2. Both are also compared with
# an FP32 reference: our engine on ncnn's CPU path (Vulkan hidden).
#
# Pass when, for each case:
#   ours (GPU) vs FP32 reference  >= REF_MIN dB      (default 50)
#   ours (GPU) vs FP32 reference  >= upstream vs FP32 reference
#   ours (GPU) vs upstream        >= UP_MIN dB       (default 35)
# Why these numbers: docs/engine-audit.md, "NCNN backend".
#
# Needs: a built `enhance_image` example (cargo build --release -p fs-engine
# --features ncnn --example enhance_image), FFmpeg/ffprobe (FS_FFMPEG,
# FS_FFPROBE), a Vulkan GPU, and `cargo xtask models` run once (it downloads
# the release zip into .cache/downloads).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
FFMPEG=${FS_FFMPEG:-ffmpeg}
FFPROBE=${FS_FFPROBE:-ffprobe}
OURS=${OURS:-$ROOT/target/release/examples/enhance_image}
REF_MIN=${REF_MIN:-50}
UP_MIN=${UP_MIN:-35}
FRAMES=${FRAMES:-24}
WORK=${WORK:-$ROOT/target/ncnn-compare}

ZIP=$(ls "$ROOT"/.cache/downloads/*-realesrgan-ncnn-vulkan-20220424-ubuntu.zip 2>/dev/null | head -1)
[ -n "$ZIP" ] || { echo "run 'cargo xtask models' first (no realesrgan-ncnn-vulkan zip)"; exit 2; }
rm -rf "$WORK" && mkdir -p "$WORK/upstream" "$WORK/frames" "$WORK/up-frames"
unzip -q "$ZIP" -d "$WORK/upstream"
UP="$WORK/upstream/realesrgan-ncnn-vulkan"
chmod +x "$UP"
ff() { "$FFMPEG" -hide_banner -loglevel error -y "$@"; }
size() { "$FFPROBE" -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x "$1"; }
psnr() { # psnr <w>x<h> <ours.rgb> <upstream input args...>
    local s=$1 ours=$2; shift 2
    "$FFMPEG" -hide_banner -f rawvideo -pix_fmt rgb24 -s "$s" -i "$ours" "$@" \
        -lavfi "[0:v]format=rgb24[a];[1:v]format=rgb24[b];[a][b]psnr" -f null - 2>&1 |
        grep -o 'average:[0-9.inf]*' | cut -d: -f2
}
fp32() { # ours on ncnn's CPU path (FP32 throughout): hide every Vulkan driver
    VK_ICD_FILENAMES=/nonexistent.json VK_DRIVER_FILES=/nonexistent.json FS_TILE_REUSE=0 "$OURS" "$@" 2>/dev/null
}
fail=0
check() { # check <case> <ours-vs-ref> <up-vs-ref> <ours-vs-up>
    local ok
    ok=$(awk -v r="$2" -v u="$3" -v o="$4" -v rm="$REF_MIN" -v um="$UP_MIN" 'function n(x) { return x == "inf" ? 1e9 : x + 0 }
        BEGIN { print (n(r) >= rm && n(r) >= n(u) && n(o) >= um) ? "ok" : "FAIL" }')
    printf "%-30s ours~FP32 %6.2f  upstream~FP32 %6.2f  ours~upstream %6.2f dB  %s\n" "$1" "$2" "$3" "$4" "$ok"
    [ "$ok" = ok ] || fail=1
}

# Same decoded pixels for both: PNG in (lossless), raw RGB24 for ours.
ff -i "$WORK/upstream/input.jpg" "$WORK/photo.png"
ff -i "$WORK/photo.png" -f rawvideo -pix_fmt rgb24 "$WORK/photo.rgb"
WH=$(size "$WORK/photo.png"); W=${WH%x*}; H=${WH#*x}

"$UP" -i "$WORK/photo.png" -o "$WORK/photo-up.png" -n realesrgan-x4plus -s 4 \
    -m "$WORK/upstream/models" -f png >/dev/null 2>&1
FS_TILE_REUSE=0 "$OURS" "$WORK/photo.rgb" "$W" "$H" "$WORK/photo-ours.rgb" \
    --upscale realesrgan-x4plus 4 2>/dev/null
fp32 "$WORK/photo.rgb" "$W" "$H" "$WORK/photo-ref.rgb" --upscale realesrgan-x4plus 4
S1="$((W * 4))x$((H * 4))"
R1=$(psnr "$S1" "$WORK/photo-ours.rgb" -f rawvideo -pix_fmt rgb24 -s "$S1" -i "$WORK/photo-ref.rgb")
U1=$(psnr "$S1" "$WORK/photo-ref.rgb" -i "$WORK/photo-up.png")
O1=$(psnr "$S1" "$WORK/photo-ours.rgb" -i "$WORK/photo-up.png")

ff -i "$WORK/upstream/onepiece_demo.mp4" -frames:v "$FRAMES" "$WORK/frames/%04d.png"
ff -i "$WORK/frames/%04d.png" -f rawvideo -pix_fmt rgb24 "$WORK/video.rgb"
WH=$(size "$WORK/frames/0001.png"); VW=${WH%x*}; VH=${WH#*x}
"$UP" -i "$WORK/frames" -o "$WORK/up-frames" -n realesr-animevideov3 -s 2 \
    -m "$WORK/upstream/models" -f png >/dev/null 2>&1
FS_TILE_REUSE=0 "$OURS" "$WORK/video.rgb" "$VW" "$VH" "$WORK/video-ours.rgb" \
    --upscale realesr-animevideov3 2 2>/dev/null
fp32 "$WORK/video.rgb" "$VW" "$VH" "$WORK/video-ref.rgb" --upscale realesr-animevideov3 2
# Ours with tile reuse on (the default in the app).
"$OURS" "$WORK/video.rgb" "$VW" "$VH" "$WORK/video-reuse.rgb" \
    --upscale realesr-animevideov3 2 2>/dev/null
S2="$((VW * 2))x$((VH * 2))"
R2=$(psnr "$S2" "$WORK/video-ours.rgb" -f rawvideo -pix_fmt rgb24 -s "$S2" -i "$WORK/video-ref.rgb")
U2=$(psnr "$S2" "$WORK/video-ref.rgb" -i "$WORK/up-frames/%04d.png")
O2=$(psnr "$S2" "$WORK/video-ours.rgb" -i "$WORK/up-frames/%04d.png")
R3=$(psnr "$S2" "$WORK/video-reuse.rgb" -f rawvideo -pix_fmt rgb24 -s "$S2" -i "$WORK/video-ref.rgb")
O3=$(psnr "$S2" "$WORK/video-reuse.rgb" -i "$WORK/up-frames/%04d.png")

check "photo ${W}x${H}, x4plus x4" "$R1" "$U1" "$O1"
check "video ${VW}x${VH} x${FRAMES}, animev3 x2" "$R2" "$U2" "$O2"
check "  same, tile reuse on" "$R3" "$U2" "$O3"
exit $fail
