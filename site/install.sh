#!/bin/bash
# FrameScale AI installer for macOS:
#   curl -fsSL https://framescale.org/install.sh | bash
#
# Downloads the latest release's .dmg for this Mac (Apple Silicon or Intel)
# from GitHub, copies the app into /Applications and opens it. Files fetched
# with curl aren't flagged as "downloaded from the internet", so macOS opens
# the app without the "can't be verified" prompt the browser download gets
# (the app isn't notarized: that needs a paid Apple account).
# Source: https://github.com/seldin-puce/framescale/blob/main/site/install.sh
set -euo pipefail

REPO="seldin-puce/framescale"
APP="FrameScale AI.app"
DEST="/Applications"

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
die() { printf '\033[31mError:\033[0m %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || die "this installer is for macOS. Other systems: https://framescale.org/#download"
case "$(uname -m)" in
  arm64) ARCH="aarch64" ;;
  x86_64)
    # Rosetta reports x86_64; prefer the native build on Apple Silicon.
    if [ "$(sysctl -in hw.optional.arm64 2>/dev/null)" = "1" ]; then ARCH="aarch64"; else ARCH="x64"; fi ;;
  *) die "unsupported CPU: $(uname -m)" ;;
esac

say "Finding the latest FrameScale AI release"
URL=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
  | grep -o "\"browser_download_url\": *\"[^\"]*_${ARCH}\.dmg\"" \
  | head -n 1 | sed 's/.*"\(https[^"]*\)"/\1/') || true
[ -n "${URL:-}" ] || die "no macOS ($ARCH) download found in the latest release"

TMP=$(mktemp -d)
MNT="$TMP/mnt"
cleanup() {
  hdiutil detach -quiet "$MNT" 2>/dev/null || true
  rm -rf "$TMP"
}
trap cleanup EXIT

say "Downloading $(basename "$URL" | sed 's/%20/ /g')"
curl -fL --progress-bar -o "$TMP/FrameScale.dmg" "$URL"

say "Installing into $DEST (license: GPL-3.0, https://github.com/$REPO/blob/main/LICENSE)"
mkdir -p "$MNT"
# The disk image carries the license as an agreement, which hdiutil shows and
# asks to accept. Through `curl | bash` nobody can answer, so the attach used
# to fail and nothing was installed. Answer it ("q" leaves the pager, "y"
# agrees), as Homebrew does for such images; the GPL needs no acceptance to
# use the program anyway. Not with -quiet: that cancels the agreement.
printf 'qy\n' | hdiutil attach -nobrowse -readonly -mountpoint "$MNT" "$TMP/FrameScale.dmg" >/dev/null
[ -d "$MNT/$APP" ] || die "the disk image has no $APP"
if pgrep -xq "FrameScale AI"; then
  say "Closing the running FrameScale AI"
  osascript -e 'quit app "FrameScale AI"' 2>/dev/null || true
  sleep 2
fi
SUDO=""
if [ ! -w "$DEST" ] || { [ -e "$DEST/$APP" ] && [ ! -w "$DEST/$APP" ]; }; then
  say "Your password is needed to write to $DEST"
  SUDO="sudo"
fi
$SUDO rm -rf "$DEST/$APP"
$SUDO ditto "$MNT/$APP" "$DEST/$APP"
$SUDO xattr -dr com.apple.quarantine "$DEST/$APP" 2>/dev/null || true

if [ -n "${FRAMESCALE_NO_OPEN:-}" ]; then
  say "Done: $DEST/$APP"
else
  say "Done. Opening FrameScale AI"
  open "$DEST/$APP"
fi
