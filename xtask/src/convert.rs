//! Build-time conversion of pinned PyTorch weights into ncnn models, so the app
//! never loads a pickle (`pth` interprets them without executing anything).
//!
//! Each architecture writes its ncnn graph by hand, mirroring the official
//! ncnn exports, and its weights as fp32 (flag 0): two checkpoints of one
//! architecture then share a byte layout, so the engine can blend them (DNI)
//! element by element.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};

use crate::pth::Tensor;

/// An ncnn graph and its weights, written as the layers are added.
#[derive(Default)]
struct Graph {
    layers: Vec<String>,
    blobs: std::collections::BTreeSet<String>,
    bin: Vec<u8>,
    /// Store flagged weights as fp16 (half the size; ncnn's own exports do).
    /// Off where two checkpoints must blend byte-wise (DNI needs fp32).
    half: bool,
}

/// f32 → IEEE half, round to nearest even.
fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    let (half, rem, mid) = if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        (m >> shift, m & ((1 << shift) - 1), 1u32 << (shift - 1))
    } else {
        (((e as u32) << 10) | (man >> 13), man & 0x1fff, 0x1000)
    };
    let r = if rem > mid || (rem == mid && half & 1 == 1) {
        half + 1
    } else {
        half
    };
    sign | r as u16
}

impl Graph {
    fn layer(&mut self, kind: &str, name: &str, inputs: &[&str], outputs: &[&str], params: &str) {
        let mut line = format!("{kind:<24} {name:<24} {} {}", inputs.len(), outputs.len());
        for b in inputs.iter().chain(outputs) {
            line.push(' ');
            line.push_str(b);
        }
        if !params.is_empty() {
            line.push(' ');
            line.push_str(params);
        }
        self.layers.push(line);
        self.blobs.extend(outputs.iter().map(|s| (*s).to_owned()));
    }

    fn raw(&mut self, values: &[f32]) {
        for v in values {
            self.bin.extend_from_slice(&v.to_le_bytes());
        }
    }

    /// A weight blob ncnn loads with a storage flag: 0 = plain fp32,
    /// 0x01306B47 = fp16 (padded to 4 bytes).
    fn flagged(&mut self, values: &[f32]) {
        if self.half {
            self.bin.extend_from_slice(&0x0130_6B47u32.to_le_bytes());
            for v in values {
                self.bin.extend_from_slice(&f32_to_f16(*v).to_le_bytes());
            }
            if values.len() % 2 == 1 {
                self.bin.extend_from_slice(&[0, 0]);
            }
        } else {
            self.bin.extend_from_slice(&0u32.to_le_bytes());
            self.raw(values);
        }
    }

    fn write(&self, dir: &Path, stem: &str) -> Result<()> {
        let mut param = format!("7767517\n{} {}\n", self.layers.len(), self.blobs.len());
        for l in &self.layers {
            writeln!(param, "{l}")?;
        }
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join(format!("{stem}.param")), param)?;
        std::fs::write(dir.join(format!("{stem}.bin")), &self.bin)?;
        Ok(())
    }
}

fn get<'a>(t: &'a BTreeMap<String, Tensor>, key: &str) -> Result<&'a Tensor> {
    t.get(key)
        .with_context(|| format!("weights have no `{key}`"))
}

/// Real-ESRGAN's SRVGGNetCompact (realesr-general-x4v3, realesr-animevideov3):
/// conv + PReLU, `n` × (conv + PReLU), conv to 3·s² channels, PixelShuffle(s),
/// plus the nearest-neighbour upscaled input. Blobs: `data` → `output`.
pub fn srvgg(t: &BTreeMap<String, Tensor>, dir: &Path, stem: &str) -> Result<()> {
    let convs = (0..)
        .map(|i| format!("body.{}.weight", 2 * i))
        .take_while(|k| t.contains_key(k))
        .count();
    ensure!(convs >= 2, "not an SRVGGNetCompact checkpoint");
    let last = get(t, &format!("body.{}.weight", 2 * (convs - 1)))?;
    let out_ch = last.shape[0];
    let scale = (1..=8)
        .find(|s| 3 * s * s == out_ch)
        .context("output channels aren't 3·s²")?;

    let mut g = Graph::default();
    g.layer("Input", "input", &[], &["data"], "");
    g.layer(
        "Split",
        "split_input",
        &["data"],
        &["data_body", "data_skip"],
        "",
    );
    let mut x = "data_body".to_owned();
    for i in 0..convs {
        let w = get(t, &format!("body.{}.weight", 2 * i))?;
        let b = get(t, &format!("body.{}.bias", 2 * i))?;
        ensure!(
            w.shape.len() == 4 && w.shape[2] == 3 && w.shape[3] == 3,
            "body.{}: 3×3 conv expected",
            2 * i
        );
        let out = format!("conv{i}");
        g.layer(
            "Convolution",
            &format!("conv_{i}"),
            &[&x],
            &[&out],
            &format!("0={} 1=3 4=1 5=1 6={}", w.shape[0], w.data.len()),
        );
        g.flagged(&w.data);
        g.raw(&b.data);
        x = out;
        if i + 1 < convs {
            let slope = get(t, &format!("body.{}.weight", 2 * i + 1))?;
            let out = format!("act{i}");
            g.layer(
                "PReLU",
                &format!("prelu_{i}"),
                &[&x],
                &[&out],
                &format!("0={}", slope.data.len()),
            );
            g.raw(&slope.data);
            x = out;
        }
    }
    g.layer(
        "PixelShuffle",
        "shuffle",
        &[&x],
        &["upsampled"],
        &format!("0={scale}"),
    );
    g.layer(
        "Interp",
        "base",
        &["data_skip"],
        &["nearest"],
        &format!("0=1 1={scale}.0 2={scale}.0"),
    );
    g.layer(
        "BinaryOp",
        "add",
        &["upsampled", "nearest"],
        &["output"],
        "0=0",
    );
    g.write(dir, stem)
}

/// NAFNet (megvii, "Simple Baselines for Image Restoration") builder: unique
/// blob names, explicit Splits for every blob read twice (ncnn requires it).
struct Naf<'a> {
    t: &'a BTreeMap<String, Tensor>,
    g: Graph,
    n: usize,
}

impl Naf<'_> {
    fn blob(&mut self, tag: &str) -> String {
        self.n += 1;
        format!("{tag}{}", self.n)
    }

    fn get(&self, key: &str) -> Result<&Tensor> {
        get(self.t, key)
    }

    fn split(&mut self, x: &str) -> (String, String) {
        let (a, b) = (self.blob("s"), self.blob("s"));
        let name = self.blob("split");
        self.g.layer("Split", &name, &[x], &[&a, &b], "");
        (a, b)
    }

    /// Convolution; `groups` > 1 makes it depthwise.
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        x: &str,
        out: usize,
        k: usize,
        stride: usize,
        pad: usize,
        groups: usize,
        w: &[f32],
        b: Option<&[f32]>,
    ) -> String {
        let y = self.blob("c");
        let name = self.blob("conv");
        let bias = u8::from(b.is_some());
        let kind = if groups > 1 {
            "ConvolutionDepthWise"
        } else {
            "Convolution"
        };
        let group = if groups > 1 {
            format!(" 7={groups}")
        } else {
            String::new()
        };
        self.g.layer(
            kind,
            &name,
            &[x],
            &[&y],
            &format!(
                "0={out} 1={k} 3={stride} 4={pad} 5={bias} 6={}{group}",
                w.len()
            ),
        );
        self.g.flagged(w);
        if let Some(b) = b {
            self.g.raw(b);
        }
        y
    }

    fn binary(&mut self, a: &str, b: &str, op: u8) -> String {
        let y = self.blob("b");
        let name = self.blob("bin");
        self.g
            .layer("BinaryOp", &name, &[a, b], &[&y], &format!("0={op}"));
        y
    }

    /// Channel-wise LayerNorm2d without its affine (folded into the next 1×1
    /// convolution): FrameScale's `fs.ChannelNorm` custom layer, which keeps
    /// its statistics in fp32 on the GPU (see fs-engine's bridge.cpp).
    fn norm(&mut self, x: &str) -> String {
        let y = self.blob("n");
        let name = self.blob("norm");
        self.g
            .layer("fs.ChannelNorm", &name, &[x], &[&y], "0=1.000000e-06");
        y
    }

    /// SimpleGate: the two channel halves multiplied.
    fn gate(&mut self, x: &str) -> String {
        let (a, b) = (self.blob("g"), self.blob("g"));
        let name = self.blob("slice");
        self.g
            .layer("Slice", &name, &[x], &[&a, &b], "-23300=2,-233,-233 1=0");
        self.binary(&a, &b, 2)
    }

    /// 1×1 conv whose input is LayerNorm-ed: y = W(γ·n + β) + b.
    fn conv_after_norm(&mut self, x: &str, p: &str, norm: &str, conv: &str) -> Result<String> {
        let (gw, gb) = (
            self.get(&format!("{p}.{norm}.weight"))?,
            self.get(&format!("{p}.{norm}.bias"))?,
        );
        let (w, b) = (
            self.get(&format!("{p}.{conv}.weight"))?,
            self.get(&format!("{p}.{conv}.bias"))?,
        );
        let (o, i) = (w.shape[0], w.shape[1]);
        ensure!(
            w.shape[2] == 1 && gw.data.len() == i,
            "{p}.{conv}: 1×1 conv after norm expected"
        );
        let mut fw = w.data.clone();
        let mut fb = b.data.clone();
        for oo in 0..o {
            for ii in 0..i {
                fw[oo * i + ii] = w.data[oo * i + ii] * gw.data[ii];
                fb[oo] += w.data[oo * i + ii] * gb.data[ii];
            }
        }
        Ok(self.conv(x, o, 1, 1, 0, 1, &fw, Some(&fb)))
    }

    /// 1×1 conv whose output is scaled per channel (β or γ): (Wx + b)·s.
    fn conv_scaled(&mut self, x: &str, p: &str, conv: &str, scale: &str) -> Result<String> {
        let s = self.get(&format!("{p}.{scale}"))?;
        let (w, b) = (
            self.get(&format!("{p}.{conv}.weight"))?,
            self.get(&format!("{p}.{conv}.bias"))?,
        );
        let (o, i) = (w.shape[0], w.shape[1]);
        ensure!(
            w.shape[2] == 1 && s.data.len() == o,
            "{p}.{conv}: scaled 1×1 conv expected"
        );
        let fw: Vec<f32> = (0..o * i).map(|k| w.data[k] * s.data[k / i]).collect();
        let fb: Vec<f32> = (0..o).map(|k| b.data[k] * s.data[k]).collect();
        Ok(self.conv(x, o, 1, 1, 0, 1, &fw, Some(&fb)))
    }

    fn block(&mut self, p: &str, c: usize, inp: &str) -> Result<String> {
        let (ia, ib) = self.split(inp);
        let n1 = self.norm(&ib);
        let x = self.conv_after_norm(&n1, p, "norm1", "conv1")?;
        let (w2, b2) = (
            self.get(&format!("{p}.conv2.weight"))?.clone(),
            self.get(&format!("{p}.conv2.bias"))?.clone(),
        );
        let x = self.conv(&x, 2 * c, 3, 1, 1, 2 * c, &w2.data, Some(&b2.data));
        let x = self.gate(&x);
        // Simplified channel attention: global average → 1×1 conv → scale.
        let (xa, xb) = self.split(&x);
        let pooled = self.blob("pool");
        let name = self.blob("gap");
        self.g
            .layer("Pooling", &name, &[&xa], &[&pooled], "0=1 4=1");
        let (ws, bs) = (
            self.get(&format!("{p}.sca.1.weight"))?.clone(),
            self.get(&format!("{p}.sca.1.bias"))?.clone(),
        );
        let att = self.blob("att");
        let name = self.blob("sca");
        self.g.layer(
            "InnerProduct",
            &name,
            &[&pooled],
            &[&att],
            &format!("0={c} 1=1 2={}", ws.data.len()),
        );
        self.g.flagged(&ws.data);
        self.g.raw(&bs.data);
        let x = self.binary(&xb, &att, 2);
        let x = self.conv_scaled(&x, p, "conv3", "beta")?;
        let y = self.binary(&ia, &x, 0);
        let (ya, yb) = self.split(&y);
        let n2 = self.norm(&yb);
        let x = self.conv_after_norm(&n2, p, "norm2", "conv4")?;
        let x = self.gate(&x);
        let x = self.conv_scaled(&x, p, "conv5", "gamma")?;
        Ok(self.binary(&ya, &x, 0))
    }

    fn blocks(&mut self, p: &str, c: usize, mut x: String) -> Result<String> {
        for b in 0.. {
            let q = format!("{p}.{b}");
            if !self.t.contains_key(&format!("{q}.conv1.weight")) {
                break;
            }
            x = self.block(&q, c, &x)?;
        }
        Ok(x)
    }
}

/// NAFNet image restoration (e.g. NAFNet-SIDD-width32: real camera noise).
/// RGB 0–1 in, RGB out, same size; sides must be multiples of 2^levels (16).
/// Blobs: `data` → `output`.
pub fn nafnet(t: &BTreeMap<String, Tensor>, dir: &Path, stem: &str) -> Result<()> {
    let mut m = Naf {
        t,
        g: Graph {
            half: true,
            ..Graph::default()
        },
        n: 0,
    };
    m.g.layer("Input", "input", &[], &["data"], "");
    let (residual, data) = m.split("data");
    let (iw, ib) = (get(t, "intro.weight")?, get(t, "intro.bias")?);
    let mut c = iw.shape[0];
    let mut x = m.conv(&data, c, 3, 1, 1, 1, &iw.data, Some(&ib.data));
    let mut skips = Vec::new();
    let mut level = 0;
    while t.contains_key(&format!("downs.{level}.weight")) {
        x = m.blocks(&format!("encoders.{level}"), c, x)?;
        let (skip, down) = m.split(&x);
        skips.push(skip);
        let (dw, db) = (
            get(t, &format!("downs.{level}.weight"))?,
            get(t, &format!("downs.{level}.bias"))?,
        );
        x = m.conv(&down, 2 * c, 2, 2, 0, 1, &dw.data, Some(&db.data));
        c *= 2;
        level += 1;
    }
    ensure!(level > 0, "not a NAFNet checkpoint");
    x = m.blocks("middle_blks", c, x)?;
    for l in 0..level {
        let uw = get(t, &format!("ups.{l}.0.weight"))?;
        x = m.conv(&x, 2 * c, 1, 1, 0, 1, &uw.data, None);
        let up = m.blob("up");
        let name = m.blob("shuffle");
        m.g.layer("PixelShuffle", &name, &[&x], &[&up], "0=2");
        c /= 2;
        let skip = skips.pop().context("skip connections")?;
        x = m.binary(&up, &skip, 0);
        x = m.blocks(&format!("decoders.{l}"), c, x)?;
    }
    let (ew, eb) = (get(t, "ending.weight")?, get(t, "ending.bias")?);
    x = m.conv(&x, ew.shape[0], 3, 1, 1, 1, &ew.data, Some(&eb.data));
    m.g.layer("BinaryOp", "residual", &[&x, &residual], &["output"], "0=0");
    m.g.write(dir, stem)
}

/// Run the named converter on a downloaded checkpoint.
pub fn run(kind: &str, pth: &Path, dir: &Path, stem: &str) -> Result<()> {
    let tensors = crate::pth::load(pth)?;
    match kind {
        "srvgg" => srvgg(&tensors, dir, stem),
        "nafnet" => nafnet(&tensors, dir, stem),
        _ => bail!("unknown converter `{kind}`"),
    }
}

#[cfg(test)]
mod tests {
    use super::f32_to_f16;

    #[test]
    fn half_rounding() {
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(65504.0), 0x7bff);
        assert_eq!(f32_to_f16(1e6), 0x7c00, "overflow → inf");
        assert_eq!(f32_to_f16(2f32.powi(-24)), 0x0001, "smallest subnormal");
        assert_eq!(
            f32_to_f16(1.0 + 2f32.powi(-11)),
            0x3c00,
            "tie rounds to even"
        );
        assert_eq!(
            f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)),
            0x3c02,
            "tie rounds to even (up)"
        );
        for x in [0.1f32, -3.75, 1234.5, 6.1e-5, 1e-7] {
            let back = crate::pth::f16_to_f32_pub(f32_to_f16(x));
            assert!((back - x).abs() <= x.abs() * 1e-3 + 6e-8, "{x} → {back}");
        }
    }
}
