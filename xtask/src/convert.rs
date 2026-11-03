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

    /// A weight blob ncnn loads with a storage flag: 0 = plain fp32.
    fn flagged(&mut self, values: &[f32]) {
        self.bin.extend_from_slice(&0u32.to_le_bytes());
        self.raw(values);
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

/// Run the named converter on a downloaded checkpoint.
pub fn run(kind: &str, pth: &Path, dir: &Path, stem: &str) -> Result<()> {
    let tensors = crate::pth::load(pth)?;
    match kind {
        "srvgg" => srvgg(&tensors, dir, stem),
        _ => bail!("unknown converter `{kind}`"),
    }
}
