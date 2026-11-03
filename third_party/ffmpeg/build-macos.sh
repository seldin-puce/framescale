#!/usr/bin/env bash
# Build a static GPL FFmpeg (+ ffprobe) for one macOS architecture.
#
#   build-macos.sh <aarch64-apple-darwin|x86_64-apple-darwin> <src-dir> <out-dir>
#
# <src-dir> holds pre-verified tarballs written by `cargo xtask sidecars`:
#   ffmpeg.tar.xz  svt_av1.tar.gz  dav1d.tar.gz  zimg.tar.gz  x265.tar.gz
# This script never downloads anything.
#
# Requires: Xcode CLT, cmake, meson, ninja, nasm, autoconf/automake/libtool, pkg-config.
# GPL build (FrameScale is GPL-3.0-or-later): x265 is GPL-2.0+, the other
# libraries are BSD/ISC/WTFPL. The xtask gate rejects --enable-nonfree.
set -euo pipefail

TRIPLE="$1"; SRC="$(cd "$2" && pwd)"; OUT="$3"
case "$TRIPLE" in
  aarch64-apple-darwin) ARCH=arm64  ;;
  x86_64-apple-darwin)  ARCH=x86_64 ;;
  *) echo "unsupported target $TRIPLE" >&2; exit 2 ;;
esac

mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
WORK="$(mktemp -d -t fs-ffmpeg-XXXXXX)"; trap 'rm -rf "$WORK"' EXIT
PREFIX="$WORK/prefix"; mkdir -p "$PREFIX"
JOBS="$(sysctl -n hw.ncpu)"
: "${MACOSX_DEPLOYMENT_TARGET:=12.0}"; export MACOSX_DEPLOYMENT_TARGET

export CC="clang -arch $ARCH" CXX="clang++ -arch $ARCH"
export CFLAGS="-arch $ARCH -mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET -O3"
export CXXFLAGS="$CFLAGS" LDFLAGS="-arch $ARCH -mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig" PKG_CONFIG_LIBDIR="$PREFIX/lib/pkgconfig"

HOST_ARCH="$(uname -m)"
CROSS=0; [[ "$HOST_ARCH" != "$ARCH" ]] && CROSS=1

unpack() { mkdir -p "$WORK/$1" && tar -xf "$SRC/$2" -C "$WORK/$1" --strip-components=1; }

# --- SVT-AV1 (BSD-3-Clause-Clear) ---------------------------------------------
unpack svt svt_av1.tar.gz
cmake -S "$WORK/svt" -B "$WORK/svt/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_OSX_ARCHITECTURES="$ARCH" -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_APPS=OFF -DBUILD_TESTING=OFF
cmake --build "$WORK/svt/build" -j "$JOBS" && cmake --install "$WORK/svt/build"

# --- dav1d (BSD-2-Clause) -----------------------------------------------------
unpack dav1d dav1d.tar.gz
CROSS_FILE=()
if [[ $CROSS == 1 ]]; then
  cat > "$WORK/cross.ini" <<EOF
[binaries]
c = ['clang', '-arch', '$ARCH']
[host_machine]
system = 'darwin'
cpu_family = '$([[ $ARCH == arm64 ]] && echo aarch64 || echo x86_64)'
cpu = '$ARCH'
endian = 'little'
EOF
  CROSS_FILE=(--cross-file "$WORK/cross.ini")
fi
meson setup "$WORK/dav1d/build" "$WORK/dav1d" "${CROSS_FILE[@]}" \
  --prefix="$PREFIX" --libdir=lib --buildtype=release --default-library=static \
  -Denable_tools=false -Denable_tests=false
ninja -C "$WORK/dav1d/build" && ninja -C "$WORK/dav1d/build" install

# --- zimg (WTFPL) — required for the zscale filter -----------------------------
unpack zimg zimg.tar.gz
( cd "$WORK/zimg" && ./autogen.sh >/dev/null && \
  ./configure --prefix="$PREFIX" --disable-shared --enable-static \
    $([[ $CROSS == 1 ]] && echo "--host=$([[ $ARCH == arm64 ]] && echo aarch64 || echo x86_64)-apple-darwin") && \
  make -j "$JOBS" && make install )

# --- x265 (GPL-2.0+), 10-bit: HEVC Main10 with HDR10 SEI -----------------------
unpack x265 x265.tar.gz
cmake -S "$WORK/x265/source" -B "$WORK/x265/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_OSX_ARCHITECTURES="$ARCH" -DENABLE_SHARED=OFF -DENABLE_CLI=OFF \
  -DHIGH_BIT_DEPTH=ON -DENABLE_HDR10_PLUS=OFF -DENABLE_ASSEMBLY=ON  # NEON / NASM
cmake --build "$WORK/x265/build" -j "$JOBS" && cmake --install "$WORK/x265/build"

# --- FFmpeg (GPL) -------------------------------------------------------------
unpack ffmpeg ffmpeg.tar.xz
CROSS_ARGS=()
[[ $CROSS == 1 ]] && CROSS_ARGS=(--enable-cross-compile --arch="$ARCH" --target-os=darwin)
( cd "$WORK/ffmpeg" && ./configure \
    --prefix="$OUT" "${CROSS_ARGS[@]}" \
    --cc="clang -arch $ARCH" --cxx="clang++ -arch $ARCH" \
    --extra-cflags="-I$PREFIX/include $CFLAGS" --extra-ldflags="-L$PREFIX/lib $LDFLAGS" \
    --pkg-config-flags=--static \
    --enable-static --disable-shared --disable-doc --disable-ffplay --disable-debug \
    --disable-autodetect \
    --enable-videotoolbox --enable-audiotoolbox --enable-coreimage \
    --enable-gpl --enable-libx265 \
    --enable-libsvtav1 --enable-libdav1d --enable-libzimg \
    --enable-zlib --enable-bzlib \
  && make -j "$JOBS" && make install )

echo "FFmpeg for $TRIPLE → $OUT/bin"
