#!/usr/bin/env bash
# Build the Core ML models the Apple Silicon app bundles
# (apps/desktop/src-tauri/resources/models/coreml), from the pinned General v3
# checkpoints `cargo xtask models` downloads. Needs macOS or Linux with
# Python 3.12 (coremltools has no native parts for newer Pythons yet); on a
# Mac with Homebrew it installs python@3.12 when missing.
#
# Without these models the Apple Silicon build still works: every network
# runs through ncnn, as on the other platforms.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT="$ROOT/apps/desktop/src-tauri/resources/models/coreml"
VENV="${VENV:-$ROOT/.cache/coreml-venv}"

if [ ! -x "$VENV/bin/python" ]; then
  PY="${PYTHON:-}"
  if [ -z "$PY" ]; then
    if command -v python3.12 >/dev/null; then
      PY=python3.12
    elif command -v brew >/dev/null; then
      brew install python@3.12 >/dev/null
      PY="$(brew --prefix python@3.12)/bin/python3.12"
    else
      echo "Python 3.12 not found (set PYTHON=/path/to/python3.12)" >&2
      exit 1
    fi
  fi
  "$PY" -m venv "$VENV"
fi
# The versions the conversion was checked with (docs/engine-audit.md, section 11).
"$VENV/bin/pip" install -q "torch==2.15.0" "coremltools==9.0" "numpy==2.5.4"

GENERAL=$(ls "$ROOT"/.cache/downloads/*-realesr-general-x4v3.pth | head -n 1)
WDN=$(ls "$ROOT"/.cache/downloads/*-realesr-general-wdn-x4v3.pth | head -n 1)
rm -rf "$OUT"
mkdir -p "$OUT"
for scale in 2 4; do
  log="$OUT/.convert-x$scale.log"
  if ! "$VENV/bin/python" -I "$ROOT/scripts/convert_coreml.py" \
      --general "$GENERAL" --wdn "$WDN" --denoise 0.5 --scale "$scale" --tile 256 --pad 10 \
      --out "$OUT/realesr-general-x4v3-dn0.50-x$scale-t256-p10.mlpackage" >"$log" 2>&1; then
    cat "$log" >&2
    exit 1
  fi
  grep -E "^wrote" "$log"
  rm -f "$log"
done
du -sh "$OUT"/*
