#!/usr/bin/env python3
"""Convert Real-ESRGAN General v3 (SRVGGNetCompact) to a Core ML program for
the Apple backend (crates/fs-engine/src/apple.rs).

What it builds, for one fixed tile shape:

    input  "input"  float16 [1, 3, T+2P, T+2P]   planar RGB, 0-1
    output "output" float16 [1, 3, S(T+2P), S(T+2P)]

- The weights are General v3 blended with its weak-denoise twin at the
  denoise strength (`--denoise`, default 0.5): `s·general + (1−s)·wdn`, the
  same DNI blend the ncnn backend does at load time (ncnn.rs `dni`).
- `--scale 2` appends a 2×2 average pool to the native ×4 network: the exact
  area downscale the ncnn backend appends to its graph (ncnn.rs
  `pool_output`). `--scale 4` leaves the native output.
- FP16 weights and FP16 compute (`compute_precision=FLOAT16`), FP16 I/O so
  the app can hand Core ML IOSurface-backed `OneComponent16Half` pixel buffers
  without a conversion pass. Needs macOS 13+ (FP16 I/O), the target is 14.
- Fixed shape (`--tile`, `--pad`): the Neural Engine only runs fixed shapes,
  and the app pads edge tiles up to it.

Usage (Python 3.10+, `pip install torch coremltools numpy`; runs on Linux
or macOS, Core ML prediction itself needs macOS):

    python3 scripts/convert_coreml.py \
        --general .cache/downloads/*-realesr-general-x4v3.pth \
        --wdn     .cache/downloads/*-realesr-general-wdn-x4v3.pth \
        --scale 2 --tile 256 --pad 10 \
        --out apps/desktop/src-tauri/resources/models/coreml/realesr-general-x4v3-x2-t256.mlpackage

`--check PREFIX` additionally writes a fixed test image (`PREFIX.in.rgb`,
RGB24, (T+2P)²) and the PyTorch result for it (`PREFIX.torch.rgb`), to
compare with the ncnn and Core ML backends on the same pixels.
"""

import argparse
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SRVGGNetCompact(nn.Module):
    """Real-ESRGAN's compact network (BSD-3-Clause, xinntao/Real-ESRGAN,
    realesrgan/archs/srvgg_arch.py), with PReLU activations: conv, then
    `num_conv` × (conv + PReLU), a last conv to 3·s² channels, pixel shuffle,
    plus the nearest-upsampled input."""

    def __init__(self, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        body = [nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        body += [nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1)]
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


class Pooled(nn.Module):
    """A native ×4 network averaged 2×2: the exact area downscale to ×2."""

    def __init__(self, net, pool):
        super().__init__()
        self.net, self.pool = net, pool

    def forward(self, x):
        y = self.net(x)
        return F.avg_pool2d(y, self.pool) if self.pool > 1 else y


def load_params(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("params_ema", "params"):
        if key in ckpt:
            return ckpt[key]
    return ckpt


def blended(general, wdn, s):
    a, b = load_params(general), load_params(wdn)
    if a.keys() != b.keys():
        sys.exit("the two checkpoints don't belong to one model")
    return {k: s * a[k] + (1.0 - s) * b[k] for k in a}


def build(args):
    params = blended(args.general, args.wdn, args.denoise)
    num_conv = sum(1 for k in params if k.startswith("body.") and k.endswith(".weight")
                   and params[k].dim() == 4) - 2
    net = SRVGGNetCompact(num_feat=params["body.0.weight"].shape[0], num_conv=num_conv, upscale=4)
    net.load_state_dict(params, strict=True)
    if args.scale not in (2, 4):
        sys.exit("--scale must be 2 (pooled) or 4 (native)")
    return Pooled(net, 4 // args.scale).eval()


def test_tile(side):
    """A deterministic image with edges, gradients and texture, on 8-bit
    levels (so an RGB24 file holds it exactly)."""
    y, x = np.mgrid[0:side, 0:side].astype(np.float32) / side
    r = 0.5 + 0.5 * np.sin(12 * x + 3 * y)
    g = (np.floor(x * 8) + np.floor(y * 8)) % 2 * 0.8 + 0.1
    b = (x * 0.7 + y * 0.3) % 1.0
    img = np.stack([r, g, b])[None]
    return (np.round(img * 255) / 255).astype(np.float32)


def to_rgb24(planar):
    """[1,3,H,W] float 0-1 → interleaved RGB24 bytes (rounded, clamped)."""
    return np.clip(np.round(planar[0].transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8).tobytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--general", required=True, help="realesr-general-x4v3.pth")
    ap.add_argument("--wdn", required=True, help="realesr-general-wdn-x4v3.pth")
    ap.add_argument("--denoise", type=float, default=0.5, help="denoise strength 0-1 (app default 0.5)")
    ap.add_argument("--scale", type=int, default=2, help="2 (pooled ×4) or 4")
    ap.add_argument("--tile", type=int, default=256, help="tile edge without context")
    ap.add_argument("--pad", type=int, default=10, help="context on each side (the app's SR_PAD)")
    ap.add_argument("--out", help="output .mlpackage")
    ap.add_argument("--check", metavar="PREFIX", help="write a test image and the PyTorch result for it")
    args = ap.parse_args()

    model = build(args)
    side = args.tile + 2 * args.pad

    if args.check:
        x = test_tile(side)
        with torch.no_grad():
            y = model(torch.from_numpy(x)).numpy()
        open(args.check + ".in.rgb", "wb").write(to_rgb24(x))
        open(args.check + ".torch.rgb", "wb").write(to_rgb24(y))
        print(f"wrote {args.check}.in.rgb ({side}x{side}) and {args.check}.torch.rgb "
              f"({y.shape[3]}x{y.shape[2]})")

    if args.out:
        import coremltools as ct

        traced = torch.jit.trace(model, torch.from_numpy(test_tile(side)))
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input", shape=(1, 3, side, side), dtype=np.float16)],
            outputs=[ct.TensorType(name="output", dtype=np.float16)],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS14,
        )
        mlmodel.short_description = (
            f"Real-ESRGAN General v3, denoise {args.denoise}, x{args.scale}, "
            f"tile {args.tile}+2x{args.pad} (FrameScale AI)"
        )
        mlmodel.license = "BSD-3-Clause (Real-ESRGAN, xinntao)"
        mlmodel.user_defined_metadata["fs.tile"] = str(args.tile)
        mlmodel.user_defined_metadata["fs.pad"] = str(args.pad)
        mlmodel.user_defined_metadata["fs.scale"] = str(args.scale)
        mlmodel.save(args.out)
        print(f"wrote {args.out}: input [1,3,{side},{side}] → output [1,3,{side * args.scale},{side * args.scale}]")


if __name__ == "__main__":
    main()
