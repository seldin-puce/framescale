//! Frame ⇄ planar-f32 conversion, seam-free tiling, and area resampling.
//! Pure Rust, independent of any inference runtime, and unit-tested.

use fs_types::worker::FrameFormat;

use crate::engine::{NetEngine, SideInputs, Tensor, ThreadedEngine, Ticket};
use crate::{EngineError, Frame};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

/// Planar RGB image, channel-major (`[c][y][x]`), values nominally 0–1.
#[derive(Debug, Clone, PartialEq)]
pub struct Planar {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Planar {
    pub fn zeros(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            data: vec![0.0; w * h * 3],
        }
    }

    #[inline]
    pub fn at(&self, c: usize, x: usize, y: usize) -> f32 {
        self.data[(c * self.h + y) * self.w + x]
    }

    #[inline]
    pub fn set(&mut self, c: usize, x: usize, y: usize, v: f32) {
        self.data[(c * self.h + y) * self.w + x] = v;
    }

    /// Copy region `[x0, x0+w) × [y0, y0+h)`, replicating edge pixels for
    /// coordinates outside the image (tile pre-padding at borders). Rows are
    /// copied as slices; only the out-of-image margins are filled per pixel.
    pub fn region(&self, x0: isize, y0: isize, w: usize, h: usize) -> Self {
        let mut out = Self::zeros(w, h);
        let sw = self.w as isize;
        // Columns [lead, lead + mid) map 1:1 onto source columns [sx0, sx0 + mid).
        let lead = (-x0).clamp(0, w as isize) as usize;
        let sx0 = x0.max(0).min(sw) as usize;
        let mid = ((x0 + w as isize).min(sw) - x0.max(0)).max(0) as usize;
        for c in 0..3 {
            for y in 0..h {
                let sy = (y0 + y as isize).clamp(0, self.h as isize - 1) as usize;
                let src = &self.data[(c * self.h + sy) * self.w..][..self.w];
                let dst = &mut out.data[(c * h + y) * w..][..w];
                dst[..lead].fill(src[0]);
                dst[lead..lead + mid].copy_from_slice(&src[sx0..sx0 + mid]);
                dst[lead + mid..].fill(src[self.w - 1]);
            }
        }
        out
    }

    /// Paste `src`'s sub-rectangle `r` at `(dx, dy)`.
    pub fn blit(&mut self, src: &Planar, r: Rect, dx: usize, dy: usize) {
        for c in 0..3 {
            for y in 0..r.h {
                let s = (c * src.h + r.y + y) * src.w + r.x;
                let d = (c * self.h + dy + y) * self.w + dx;
                self.data[d..d + r.w].copy_from_slice(&src.data[s..s + r.w]);
            }
        }
    }
}

/// Split work over rows across threads (conversion of 4K frames is ~8 MP).
fn par_rows<T: Send, U: Sync>(
    rows: usize,
    out: &mut [T],
    out_per_row: usize,
    shared: &U,
    f: impl Fn(usize, &mut [T], &U) + Sync,
) {
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .clamp(1, 16);
    let per = rows.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (i, chunk) in out.chunks_mut(per * out_per_row).enumerate() {
            let f = &f;
            s.spawn(move || f(i * per, chunk, shared));
        }
    });
}

pub fn from_frame(f: &Frame) -> Planar {
    let (w, h) = (f.width as usize, f.height as usize);
    let mut p = Planar::zeros(w, h);
    let plane = w * h;
    let (r, rest) = p.data.split_at_mut(plane);
    let (g, b) = rest.split_at_mut(plane);
    for (c, dst) in [r, g, b].into_iter().enumerate() {
        par_rows(h, dst, w, &f.data, |row0, chunk, data| match f.format {
            FrameFormat::Rgb24 => {
                let src = &data[row0 * w * 3..];
                for (i, v) in chunk.iter_mut().enumerate() {
                    *v = f32::from(src[i * 3 + c]) * (1.0 / 255.0);
                }
            }
            FrameFormat::Rgb48 => {
                let src = &data[row0 * w * 6..];
                for (i, v) in chunk.iter_mut().enumerate() {
                    let k = i * 6 + c * 2;
                    *v = f32::from(u16::from_le_bytes([src[k], src[k + 1]])) * (1.0 / 65535.0);
                }
            }
        });
    }
    p
}

pub fn to_frame(p: &Planar, format: FrameFormat) -> Frame {
    let (w, h) = (p.w, p.h);
    let bpp = format.bytes_per_pixel();
    let mut data = vec![0u8; format.frame_bytes(w as u32, h as u32)];
    par_rows(h, &mut data, w * bpp, p, |row0, chunk, p| {
        let plane = p.w * p.h;
        let base = row0 * p.w;
        let (r, g, b) = (
            &p.data[base..],
            &p.data[plane + base..],
            &p.data[2 * plane + base..],
        );
        match format {
            FrameFormat::Rgb24 => {
                let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                for (i, px) in chunk.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                    *px = [q(r[i]), q(g[i]), q(b[i])];
                }
            }
            FrameFormat::Rgb48 => {
                let q = |v: f32| ((v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16).to_le_bytes();
                for (i, px) in chunk.as_chunks_mut::<6>().0.iter_mut().enumerate() {
                    let (x, y, z) = (q(r[i]), q(g[i]), q(b[i]));
                    *px = [x[0], x[1], y[0], y[1], z[0], z[1]];
                }
            }
        }
    });
    Frame::new(w as u32, h as u32, format, data)
}

/// Run a scale-`s` network over `img` in tiles of `tile` px with `pad` px of
/// context on each side. Only each tile's interior is kept, so seams vanish
/// as long as `pad` covers the network's effective receptive field.
pub fn tiled(
    img: &Planar,
    tile: usize,
    pad: usize,
    scale: usize,
    run: impl Fn(&Planar) -> Result<Planar, EngineError> + Sync,
) -> Result<Planar, EngineError> {
    let mut out = Planar::zeros(img.w * scale, img.h * scale);
    tiled_into(img, tile, pad, scale, 1, &mut out, run)?;
    Ok(out)
}

/// [`tiled`] with `workers` concurrent tile runners writing into a reusable
/// `out` buffer (resized if needed). With two or more workers, one tile's
/// CPU-side copies overlap another tile's GPU time. The output is identical
/// regardless of worker count.
pub fn tiled_into(
    img: &Planar,
    tile: usize,
    pad: usize,
    scale: usize,
    workers: usize,
    out: &mut Planar,
    run: impl Fn(&Planar) -> Result<Planar, EngineError> + Sync,
) -> Result<(), EngineError> {
    let tile = tile.max(16);
    tiled_grid(img, (tile, tile), pad, scale, workers, out, None, run)
}

/// Largest change (0–1 scale, any pixel, any channel) in a tile's input,
/// context included, for its previous output to be kept: under one 8-bit
/// level, so a kept tile is what the network would produce to within
/// rounding. Exactly unchanged input (static shots, screen recordings,
/// skipped blocks in compressed video) gives the same output bit for bit.
pub const REUSE_MAX_DIFF: f32 = 1.0 / 255.0;

/// What a tiled stage last computed, tile by tile, across frames: the input
/// each tile's current output came from. [`tiled_into_reusing`] skips tiles
/// whose input hasn't changed and leaves their previous output in place.
#[derive(Debug, Default)]
pub struct TileMemory {
    /// Image size, tile size, pad and scale the inputs belong to.
    key: (usize, usize, usize, usize, usize, usize),
    inputs: Vec<Option<Planar>>,
    /// Tiles seen and tiles reused, since creation.
    pub tiles: u64,
    pub reused: u64,
}

impl TileMemory {
    /// Forget every tile (the output buffer can't be trusted any more).
    pub fn clear(&mut self) {
        self.inputs.clear();
    }
}

/// [`tiled_into`] that only runs tiles whose padded input changed by more
/// than [`REUSE_MAX_DIFF`] since `memory` last ran them. `out` must hold the
/// previous result and not be touched in between; any change of size, tile
/// or scale starts over. Returns the number of tiles reused.
#[allow(clippy::too_many_arguments)]
pub fn tiled_into_reusing(
    img: &Planar,
    tile: usize,
    pad: usize,
    scale: usize,
    workers: usize,
    out: &mut Planar,
    memory: &mut TileMemory,
    run: impl Fn(&Planar) -> Result<Planar, EngineError> + Sync,
) -> Result<usize, EngineError> {
    let tile = tile.max(16);
    let before = memory.reused;
    tiled_grid(
        img,
        (tile, tile),
        pad,
        scale,
        workers,
        out,
        Some(memory),
        run,
    )?;
    Ok((memory.reused - before) as usize)
}

/// [`tiled_into_reusing`] over [`tiled_into_even`]'s grid.
#[allow(clippy::too_many_arguments)]
pub fn tiled_into_even_reusing(
    img: &Planar,
    tile: usize,
    pad: usize,
    scale: usize,
    workers: usize,
    out: &mut Planar,
    memory: &mut TileMemory,
    run: impl Fn(&Planar) -> Result<Planar, EngineError> + Sync,
) -> Result<usize, EngineError> {
    let before = memory.reused;
    let grid = even_grid(img, tile);
    tiled_grid(img, grid, pad, scale, workers, out, Some(memory), run)?;
    Ok((memory.reused - before) as usize)
}

/// The grid [`tiled_into_even`] uses: equal tiles of at most `tile` px a side.
pub fn even_grid(img: &Planar, tile: usize) -> (usize, usize) {
    let tile = tile.max(16);
    let even = |n: usize| n.div_ceil(n.div_ceil(tile)).max(1);
    (even(img.w), even(img.h))
}

/// Whether `a` and `b` differ nowhere by more than `limit`.
fn within(a: &Planar, b: &Planar, limit: f32) -> bool {
    (a.w, a.h) == (b.w, b.h)
        && a.data
            .iter()
            .zip(&b.data)
            .all(|(x, y)| (x - y).abs() <= limit)
}

/// [`tiled_into`] with the frame split into equal tiles of at most `tile` px
/// a side, instead of full tiles plus a thin remainder. For networks whose
/// tiles see tile-wide statistics (NAFNet's channel attention): every tile
/// then averages over a similar area, which keeps neighbours consistent.
pub fn tiled_into_even(
    img: &Planar,
    tile: usize,
    pad: usize,
    scale: usize,
    workers: usize,
    out: &mut Planar,
    run: impl Fn(&Planar) -> Result<Planar, EngineError> + Sync,
) -> Result<(), EngineError> {
    tiled_grid(
        img,
        even_grid(img, tile),
        pad,
        scale,
        workers,
        out,
        None,
        run,
    )
}

/// The scoped-thread form every closure-based tiling call goes through:
/// `workers` threads run `run`, [`run_tiled`] schedules and stitches.
#[allow(clippy::too_many_arguments)]
fn tiled_grid(
    img: &Planar,
    grid: (usize, usize),
    pad: usize,
    scale: usize,
    workers: usize,
    out: &mut Planar,
    memory: Option<&mut TileMemory>,
    run: impl Fn(&Planar) -> Result<Planar, EngineError> + Sync,
) -> Result<(), EngineError> {
    let f = |t: &Planar, _: Option<&SideInputs>| run(t);
    std::thread::scope(|s| {
        let mut engine = ThreadedEngine::scoped(s, workers, &f);
        run_tiled(&mut engine, img, grid, pad, scale, out, memory, None)
        // `engine` drops here, before the scope joins its threads.
    })
}

/// Run a scale-`scale` network over `img` on `engine`, tile by tile on a
/// `(tw, th)` grid with `pad` px of context around each tile, keeping up to
/// `engine.max_in_flight()` tiles submitted and stitching each tile's
/// interior into `out` as it finishes (in any order). With `memory`, tiles
/// whose padded input is unchanged since the last call keep their output
/// (see [`TileMemory`]). `side` goes with every tile. The first error stops
/// further submissions; tiles already running finish and are discarded.
#[allow(clippy::too_many_arguments)]
pub fn run_tiled(
    engine: &mut dyn NetEngine,
    img: &Planar,
    (tw, th): (usize, usize),
    pad: usize,
    scale: usize,
    out: &mut Planar,
    mut memory: Option<&mut TileMemory>,
    side: Option<&SideInputs>,
) -> Result<(), EngineError> {
    let (tw, th) = (tw.max(1), th.max(1));
    if (out.w, out.h) != (img.w * scale, img.h * scale) {
        *out = Planar::zeros(img.w * scale, img.h * scale);
        if let Some(m) = memory.as_deref_mut() {
            m.clear();
        }
    }
    let tiles: Vec<(usize, usize, usize, usize)> = (0..img.h)
        .step_by(th)
        .flat_map(|ty| (0..img.w).step_by(tw).map(move |tx| (tx, ty)))
        .map(|(tx, ty)| (tx, ty, tw.min(img.w - tx), th.min(img.h - ty)))
        .collect();
    if let Some(m) = memory.as_deref_mut() {
        let key = (img.w, img.h, tw, th, pad, scale);
        if m.key != key || m.inputs.len() != tiles.len() {
            m.key = key;
            m.inputs = vec![None; tiles.len()];
        }
    }
    let remember = memory.is_some();
    let previous: &[Option<Planar>] = memory.as_deref().map_or(&[], |m| &m.inputs);
    let slots = engine.max_in_flight().max(1);
    let mut in_flight: Vec<(Ticket, usize)> = Vec::with_capacity(slots);
    let (mut next, mut reused) = (0usize, 0u64);
    let mut first_err: Option<EngineError> = None;
    let mut learned: Vec<(usize, Planar)> = Vec::new();

    loop {
        // Keep every slot busy.
        while first_err.is_none() && in_flight.len() < slots && next < tiles.len() {
            let i = next;
            next += 1;
            let (x, y, w, h) = tiles[i];
            let input = img.region(
                x as isize - pad as isize,
                y as isize - pad as isize,
                w + 2 * pad,
                h + 2 * pad,
            );
            let unchanged = previous
                .get(i)
                .and_then(Option::as_ref)
                .is_some_and(|p| within(p, &input, REUSE_MAX_DIFF));
            if unchanged {
                reused += 1;
                continue;
            }
            match engine.submit(Tensor::Host(input), side.cloned()) {
                Ok(t) => in_flight.push((t, i)),
                Err(e) => first_err = Some(e),
            }
        }
        if in_flight.is_empty() {
            break;
        }
        let Some(done) = engine.wait_any() else {
            first_err.get_or_insert(EngineError::Internal(
                "engine lost track of submitted tiles".into(),
            ));
            break;
        };
        let Some(pos) = in_flight.iter().position(|(t, _)| *t == done.ticket) else {
            // Not ours: the engine failed as a whole (its threads are gone).
            first_err.get_or_insert(
                done.output
                    .err()
                    .unwrap_or_else(|| EngineError::Internal("unknown tile ticket".into())),
            );
            break;
        };
        let (_, i) = in_flight.swap_remove(pos);
        let (x, y, w, h) = tiles[i];
        let (iw, ih) = (w + 2 * pad, h + 2 * pad);
        match done.output.and_then(Tensor::into_host) {
            Ok(res) if res.w != iw * scale || res.h != ih * scale => {
                first_err.get_or_insert(EngineError::Internal(format!(
                    "model returned {}×{} for a {iw}×{ih} tile at ×{scale}",
                    res.w, res.h
                )));
            }
            Ok(res) if first_err.is_none() => {
                let interior = Rect {
                    x: pad * scale,
                    y: pad * scale,
                    w: w * scale,
                    h: h * scale,
                };
                out.blit(&res, interior, x * scale, y * scale);
                if let (true, Some(input)) = (remember, done.input) {
                    learned.push((i, input));
                }
            }
            Ok(_) => {} // after an error: discarded
            Err(e) => {
                first_err.get_or_insert(e);
            }
        }
    }
    if let Some(m) = memory {
        if first_err.is_some() {
            // Part of `out` may be stale: start over next time.
            m.clear();
        } else {
            for (i, input) in learned {
                m.inputs[i] = Some(input);
            }
            m.tiles += tiles.len() as u64;
            m.reused += reused;
        }
    }
    first_err.map_or(Ok(()), Err)
}

/// Exact area-average resample for downscaling by any ratio ≥ 1, e.g. a
/// native ×4 model reduced to the requested ×2 or ×3. Each output pixel is the
/// coverage-weighted mean of the source pixels under it, so there's no aliasing.
pub fn area_resample(src: &Planar, w: usize, h: usize) -> Planar {
    if (w, h) == (src.w, src.h) {
        return src.clone();
    }
    // Separable: first X, then Y. Weights are computed once per axis.
    let weights = |n_src: usize, n_dst: usize| -> Vec<Vec<(usize, f32)>> {
        let r = n_src as f64 / n_dst as f64;
        (0..n_dst)
            .map(|o| {
                let (a, b) = (o as f64 * r, (o + 1) as f64 * r);
                let mut v = Vec::new();
                let mut i = a.floor() as usize;
                while (i as f64) < b && i < n_src {
                    let cover = (b.min(i as f64 + 1.0) - a.max(i as f64)) / r;
                    if cover > 0.0 {
                        v.push((i, cover as f32));
                    }
                    i += 1;
                }
                v
            })
            .collect()
    };
    let wx = weights(src.w, w);
    let wy = weights(src.h, h);
    let mut tmp = vec![0f32; 3 * src.h * w];
    for c in 0..3 {
        for y in 0..src.h {
            for (x, ws) in wx.iter().enumerate() {
                tmp[(c * src.h + y) * w + x] = ws.iter().map(|&(i, k)| src.at(c, i, y) * k).sum();
            }
        }
    }
    let mut out = Planar::zeros(w, h);
    for c in 0..3 {
        for (y, ws) in wy.iter().enumerate() {
            for x in 0..w {
                let v: f32 = ws
                    .iter()
                    .map(|&(i, k)| tmp[(c * src.h + i) * w + x] * k)
                    .sum();
                out.set(c, x, y, v);
            }
        }
    }
    out
}

/// Pad to the next multiple of `m` (edge replicate). Returns the padded image.
pub fn pad_to_multiple(img: &Planar, m: usize) -> Planar {
    let w = img.w.div_ceil(m) * m;
    let h = img.h.div_ceil(m) * m;
    if (w, h) == (img.w, img.h) {
        img.clone()
    } else {
        img.region(0, 0, w, h)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn ramp(w: usize, h: usize) -> Planar {
        let mut p = Planar::zeros(w, h);
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    p.set(c, x, y, ((x * 7 + y * 13 + c * 29) % 97) as f32 / 97.0);
                }
            }
        }
        p
    }

    #[test]
    fn reuses_unchanged_tiles_and_reruns_changed_ones() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        // A ×2 "network" with a context-dependent result (local mean), so a
        // tile that missed a change in its padding would show it.
        let net = |t: &Planar| {
            let mut o = Planar::zeros(t.w * 2, t.h * 2);
            for c in 0..3 {
                for y in 0..o.h {
                    for x in 0..o.w {
                        let (sx, sy) = (x / 2, y / 2);
                        let l = t.at(c, sx.saturating_sub(1), sy);
                        let r = t.at(c, (sx + 1).min(t.w - 1), sy);
                        o.set(c, x, y, (l + t.at(c, sx, sy) + r) / 3.0);
                    }
                }
            }
            o
        };
        let runs = AtomicUsize::new(0);
        let counted = |t: &Planar| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(net(t))
        };
        let fresh = |img: &Planar| tiled(img, 16, 4, 2, |t| Ok(net(t))).expect("tiles");

        let mut img = ramp(50, 40); // 4 × 3 = 12 tiles
        let (mut out, mut mem) = (Planar::zeros(1, 1), TileMemory::default());
        let n = tiled_into_reusing(&img, 16, 4, 2, 2, &mut out, &mut mem, counted).expect("a");
        assert_eq!((n, runs.swap(0, Ordering::Relaxed)), (0, 12));
        assert_eq!(out, fresh(&img));

        // Same frame: nothing runs, the output stays.
        let n = tiled_into_reusing(&img, 16, 4, 2, 2, &mut out, &mut mem, counted).expect("b");
        assert_eq!((n, runs.swap(0, Ordering::Relaxed)), (12, 0));
        assert_eq!(out, fresh(&img));

        // A sub-level change is kept; a real one reruns the tiles whose
        // padded input contains it: (17, 5) sits in tile (16, 0) and in the
        // 4 px padding of tile (0, 0).
        img.set(1, 30, 30, img.at(1, 30, 30) + 0.5 / 255.0);
        let n = tiled_into_reusing(&img, 16, 4, 2, 2, &mut out, &mut mem, counted).expect("c");
        assert_eq!((n, runs.swap(0, Ordering::Relaxed)), (12, 0));
        img.set(0, 17, 5, 0.1);
        let n = tiled_into_reusing(&img, 16, 4, 2, 2, &mut out, &mut mem, counted).expect("d");
        assert_eq!((n, runs.swap(0, Ordering::Relaxed)), (10, 2));
        let worst = out
            .data
            .iter()
            .zip(&fresh(&img).data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(worst <= REUSE_MAX_DIFF, "off by {worst}");

        // Another tile size starts over.
        let n = tiled_into_reusing(&img, 24, 4, 2, 2, &mut out, &mut mem, counted).expect("e");
        assert_eq!((n, runs.swap(0, Ordering::Relaxed)), (0, 6));
        assert_eq!(out, tiled(&img, 24, 4, 2, |t| Ok(net(t))).expect("tiles"));
        assert_eq!((mem.tiles, mem.reused), (12 * 4 + 6, 34));
    }

    #[test]
    fn area_downscale_per_tile_equals_the_whole_frame() {
        // A stand-in native ×4 model (nearest neighbour), asked for ×2 and ×3:
        // downscaling each padded tile must match downscaling the whole
        // upscaled frame (what the SR stage did before).
        let up4 = |t: &Planar| {
            let mut o = Planar::zeros(t.w * 4, t.h * 4);
            for c in 0..3 {
                for y in 0..o.h {
                    for x in 0..o.w {
                        o.set(c, x, y, t.at(c, x / 4, y / 4));
                    }
                }
            }
            o
        };
        let img = ramp(37, 23);
        for target in [2, 3] {
            let whole = area_resample(&up4(&img), img.w * target, img.h * target);
            let mut tiled = Planar::zeros(1, 1);
            tiled_into(&img, 16, 10, target, 2, &mut tiled, |t| {
                Ok(area_resample(&up4(t), t.w * target, t.h * target))
            })
            .expect("tiles");
            assert_eq!((tiled.w, tiled.h), (whole.w, whole.h));
            let worst = whole
                .data
                .iter()
                .zip(&tiled.data)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(worst < 1e-5, "×{target}: off by {worst}");
        }
    }

    /// Nearest-neighbour ×s "network": its receptive field is one pixel, so
    /// tiled output must be *identical* to untiled output.
    fn nn(s: usize) -> impl Fn(&Planar) -> Result<Planar, EngineError> + Sync {
        move |p| {
            let mut o = Planar::zeros(p.w * s, p.h * s);
            for c in 0..3 {
                for y in 0..o.h {
                    for x in 0..o.w {
                        o.set(c, x, y, p.at(c, x / s, y / s));
                    }
                }
            }
            Ok(o)
        }
    }

    #[test]
    fn tiling_is_seamless_for_any_tile_size() {
        let img = ramp(53, 37); // deliberately not multiples of the tile
        let whole = tiled(&img, 1024, 0, 2, nn(2)).expect("whole");
        for (tile, pad) in [(16, 0), (17, 3), (32, 10), (40, 1)] {
            assert_eq!(
                tiled(&img, tile, pad, 2, nn(2)).expect("tiled"),
                whole,
                "tile {tile} pad {pad}"
            );
        }
    }

    #[test]
    fn parallel_tiling_matches_sequential_and_reuses_buffer() {
        let img = ramp(97, 61);
        let seq = tiled(&img, 20, 4, 2, nn(2)).expect("seq");
        let mut out = Planar::zeros(1, 1);
        for workers in [2, 3, 8] {
            tiled_into(&img, 20, 4, 2, workers, &mut out, nn(2)).expect("par");
            assert_eq!(out, seq, "{workers} workers");
        }
        // Errors propagate and stop the other workers.
        let failing = |_: &Planar| -> Result<Planar, EngineError> {
            Err(EngineError::OutOfDeviceMemory { tile: 20 })
        };
        assert_eq!(
            tiled_into(&img, 20, 4, 2, 3, &mut out, failing),
            Err(EngineError::OutOfDeviceMemory { tile: 20 })
        );
    }

    #[test]
    fn frame_roundtrip_8_and_16_bit() {
        for fmt in [FrameFormat::Rgb24, FrameFormat::Rgb48] {
            let bytes = fmt.frame_bytes(5, 3);
            let data: Vec<u8> = (0..bytes).map(|i| (i * 37 % 251) as u8).collect();
            let f = Frame::new(5, 3, fmt, data);
            assert_eq!(to_frame(&from_frame(&f), fmt), f, "{fmt:?}");
        }
    }

    #[test]
    fn area_resample_preserves_mean_and_handles_thirds() {
        let img = ramp(12, 12);
        let half = area_resample(&img, 6, 6);
        let mean = |p: &Planar| p.data.iter().sum::<f32>() / p.data.len() as f32;
        assert!((mean(&half) - mean(&img)).abs() < 1e-4);
        // 2×2 box average exactly.
        let expect = (img.at(0, 0, 0) + img.at(0, 1, 0) + img.at(0, 0, 1) + img.at(0, 1, 1)) / 4.0;
        assert!((half.at(0, 0, 0) - expect).abs() < 1e-6);
        // ×4 → ×3 (ratio 4/3) keeps the mean too.
        let third = area_resample(&img, 9, 9);
        assert!((mean(&third) - mean(&img)).abs() < 1e-4);
    }

    #[test]
    fn padding_replicates_edges() {
        let img = ramp(33, 17);
        let p = pad_to_multiple(&img, 32);
        assert_eq!((p.w, p.h), (64, 32));
        assert_eq!(p.at(1, 63, 31), img.at(1, 32, 16));
    }
}
