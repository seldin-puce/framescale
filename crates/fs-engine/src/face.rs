//! Face Fix geometry: YuNet decoding, 5-point similarity alignment to the
//! FFHQ 512 template (CodeFormer's input), warping and feathered paste-back.
//! Pure Rust; the networks themselves run through the ncnn bridge.

use crate::planar::{self, Planar};

/// CodeFormer's input size.
pub const FACE: usize = 512;

/// FFHQ 5-point template at 512 px (facexlib): eye, eye, nose tip, mouth
/// corner, mouth corner, image-left point first.
pub const TEMPLATE: [[f32; 2]; 5] = [
    [192.981_38, 239.947_08],
    [318.902_77, 240.193_6],
    [256.634_16, 314.019_35],
    [201.261_17, 371.410_43],
    [313.089_05, 371.151_18],
];

/// YuNet's (fixed) input size and head strides.
pub const DETECT: usize = 640;
const STRIDES: [usize; 3] = [8, 16, 32];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Face {
    pub score: f32,
    /// `x, y, w, h` in image pixels.
    pub bbox: [f32; 4],
    /// Eyes, nose tip, mouth corners (image-left first), in image pixels.
    pub kps: [[f32; 2]; 5],
}

impl Face {
    /// Where the nose tip sits between the eyes, along the eye line: about
    /// 0.5 for a face looking at the camera, towards 0 or 1 (or beyond) as
    /// the head turns into profile.
    pub fn nose_position(&self) -> f32 {
        let [l, r, n, _, _] = self.kps;
        let (ex, ey) = (r[0] - l[0], r[1] - l[1]);
        let d2 = ex * ex + ey * ey;
        if d2 < 1e-6 {
            return 0.0;
        }
        ((n[0] - l[0]) * ex + (n[1] - l[1]) * ey) / d2
    }
}

/// 2×3 affine map `(x, y) → (a·x + b·y + tx, c·x + d·y + ty)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine(pub [f32; 6]);

impl Affine {
    pub fn apply(&self, [x, y]: [f32; 2]) -> [f32; 2] {
        let [a, b, tx, c, d, ty] = self.0;
        [a * x + b * y + tx, c * x + d * y + ty]
    }

    pub fn invert(&self) -> Option<Self> {
        let [a, b, tx, c, d, ty] = self.0;
        let det = a * d - b * c;
        if det.abs() < 1e-12 {
            return None;
        }
        let (ia, ib, ic, id) = (d / det, -b / det, -c / det, a / det);
        Some(Self([
            ia,
            ib,
            -(ia * tx + ib * ty),
            ic,
            id,
            -(ic * tx + id * ty),
        ]))
    }
}

/// Least-squares similarity (rotation, uniform scale, translation) mapping
/// `src` points onto `dst`, like OpenCV's `estimateAffinePartial2D`.
pub fn similarity(src: &[[f32; 2]], dst: &[[f32; 2]]) -> Option<Affine> {
    let n = src.len().min(dst.len());
    if n < 2 {
        return None;
    }
    let mean = |p: &[[f32; 2]]| {
        let (sx, sy) = p[..n]
            .iter()
            .fold((0.0, 0.0), |(x, y), q| (x + q[0], y + q[1]));
        (sx / n as f32, sy / n as f32)
    };
    let ((mx, my), (mu, mv)) = (mean(src), mean(dst));
    let (mut na, mut nb, mut den) = (0.0f32, 0.0f32, 0.0f32);
    for (s, d) in src[..n].iter().zip(&dst[..n]) {
        let (x, y, u, v) = (s[0] - mx, s[1] - my, d[0] - mu, d[1] - mv);
        na += x * u + y * v;
        nb += x * v - y * u;
        den += x * x + y * y;
    }
    if den < 1e-9 {
        return None;
    }
    let (a, b) = (na / den, nb / den);
    // u = a·x − b·y + tx, v = b·x + a·y + ty
    Some(Affine([
        a,
        -b,
        mu - (a * mx - b * my),
        b,
        a,
        mv - (b * mx + a * my),
    ]))
}

/// Bilinear sample of channel `c` at `(x, y)` (pixel centres on integers),
/// clamped to the image edges.
fn sample(img: &Planar, c: usize, x: f32, y: f32) -> f32 {
    let xf = x.clamp(0.0, (img.w - 1) as f32);
    let yf = y.clamp(0.0, (img.h - 1) as f32);
    let (x0, y0) = (xf.floor() as usize, yf.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(img.w - 1), (y0 + 1).min(img.h - 1));
    let (fx, fy) = (xf - x0 as f32, yf - y0 as f32);
    let top = img.at(c, x0, y0) * (1.0 - fx) + img.at(c, x1, y0) * fx;
    let bottom = img.at(c, x0, y1) * (1.0 - fx) + img.at(c, x1, y1) * fx;
    top * (1.0 - fy) + bottom * fy
}

/// A `w×h` image whose pixel `p` samples `img` at `to_src(p)`.
pub fn warp(img: &Planar, to_src: &Affine, w: usize, h: usize) -> Planar {
    let mut out = Planar::zeros(w, h);
    for y in 0..h {
        for x in 0..w {
            let [sx, sy] = to_src.apply([x as f32, y as f32]);
            for c in 0..3 {
                out.set(c, x, y, sample(img, c, sx, sy));
            }
        }
    }
    out
}

/// Resize with exact area averaging when shrinking, bilinear otherwise.
fn resize(img: &Planar, w: usize, h: usize) -> Planar {
    if w <= img.w && h <= img.h {
        return planar::area_resample(img, w, h);
    }
    let (sx, sy) = (img.w as f32 / w as f32, img.h as f32 / h as f32);
    warp(
        img,
        &Affine([sx, 0.0, 0.5 * sx - 0.5, 0.0, sy, 0.5 * sy - 0.5]),
        w,
        h,
    )
}

/// The frame fitted into `DETECT×DETECT` (aspect kept, padded right and
/// bottom) as BGR 0–255, YuNet's input, and the frame pixels per input pixel.
pub fn letterbox_bgr(img: &Planar) -> (Planar, f32) {
    let s = (DETECT as f32 / img.w as f32).min(DETECT as f32 / img.h as f32);
    let w = ((img.w as f32 * s).round() as usize).clamp(1, DETECT);
    let h = ((img.h as f32 * s).round() as usize).clamp(1, DETECT);
    let small = resize(img, w, h);
    let mut out = Planar::zeros(DETECT, DETECT);
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                out.set(c, x, y, small.at(2 - c, x, y) * 255.0);
            }
        }
    }
    (out, img.w as f32 / w as f32)
}

/// One detector output: data with ncnn's `(w, h, c)` shape.
#[derive(Debug, Clone, Copy)]
pub struct Head<'a> {
    pub data: &'a [f32],
    pub w: usize,
    pub h: usize,
    pub c: usize,
}

impl Head<'_> {
    /// `(rows, channels)` if this head has `rows` anchors, either layout.
    fn value(&self, rows: usize, idx: usize, k: usize) -> f32 {
        if self.h == rows {
            self.data[idx * self.w + k] // [rows][ch]
        } else {
            self.data[k * rows + idx] // [ch][rows]
        }
    }

    fn channels(&self, rows: usize) -> Option<usize> {
        let total = self.w * self.h * self.c;
        (total.is_multiple_of(rows) && (self.h == rows || self.w == rows)).then_some(total / rows)
    }
}

/// Decode YuNet's heads (OpenCV's `FaceDetectorYN` scheme). Heads are told
/// apart by shape: per stride, two 1-channel score maps (class, objectness;
/// the score is their geometric mean), a 4-channel box and 10-channel
/// landmarks. Coordinates are in detector-input pixels.
pub fn decode_yunet(heads: &[Head<'_>], min_score: f32) -> Vec<Face> {
    let mut faces = Vec::new();
    for stride in STRIDES {
        let cols = DETECT / stride;
        let rows = cols * cols;
        let of = |ch: usize| heads.iter().filter(move |h| h.channels(rows) == Some(ch));
        let scores: Vec<_> = of(1).collect();
        let (Some(bbox), Some(kps)) = (of(4).next(), of(10).next()) else {
            continue;
        };
        if scores.len() != 2 {
            continue;
        }
        for idx in 0..rows {
            let (r, c) = ((idx / cols) as f32, (idx % cols) as f32);
            let cls = scores[0].value(rows, idx, 0).clamp(0.0, 1.0);
            let obj = scores[1].value(rows, idx, 0).clamp(0.0, 1.0);
            let score = (cls * obj).sqrt();
            if score < min_score {
                continue;
            }
            let s = stride as f32;
            let cx = (c + bbox.value(rows, idx, 0)) * s;
            let cy = (r + bbox.value(rows, idx, 1)) * s;
            let w = bbox.value(rows, idx, 2).exp() * s;
            let h = bbox.value(rows, idx, 3).exp() * s;
            let mut pts = [[0.0f32; 2]; 5];
            for (n, p) in pts.iter_mut().enumerate() {
                *p = [
                    (kps.value(rows, idx, 2 * n) + c) * s,
                    (kps.value(rows, idx, 2 * n + 1) + r) * s,
                ];
            }
            faces.push(Face {
                score,
                bbox: [cx - w / 2.0, cy - h / 2.0, w, h],
                kps: pts,
            });
        }
    }
    nms(faces, 0.3)
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x0 = a[0].max(b[0]);
    let y0 = a[1].max(b[1]);
    let x1 = (a[0] + a[2]).min(b[0] + b[2]);
    let y1 = (a[1] + a[3]).min(b[1] + b[3]);
    let inter = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
    let union = a[2] * a[3] + b[2] * b[3] - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// Greedy non-maximum suppression, best first.
pub fn nms(mut faces: Vec<Face>, threshold: f32) -> Vec<Face> {
    faces.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<Face> = Vec::new();
    for f in faces {
        if keep.iter().all(|k| iou(&k.bbox, &f.bbox) <= threshold) {
            keep.push(f);
        }
    }
    keep
}

impl Face {
    /// The same face in a frame `scale`× larger than the detector input.
    pub fn scaled(&self, scale: f32) -> Self {
        Self {
            score: self.score,
            bbox: self.bbox.map(|v| v * scale),
            kps: self.kps.map(|[x, y]| [x * scale, y * scale]),
        }
    }
}

/// Blend a restored `FACE×FACE` face back into `img`; `to_face` maps image
/// coordinates to face coordinates. The border is feathered over 10 % of the
/// face so the seam doesn't show.
pub fn paste(img: &mut Planar, face: &Planar, to_face: &Affine) {
    let Some(to_img) = to_face.invert() else {
        return;
    };
    let n = FACE as f32 - 1.0;
    let corners = [[0.0, 0.0], [n, 0.0], [0.0, n], [n, n]].map(|p| to_img.apply(p));
    let lo = |i: usize| {
        corners
            .iter()
            .map(|p| p[i])
            .fold(f32::INFINITY, f32::min)
            .floor()
            .max(0.0) as usize
    };
    let hi = |i: usize, len: usize| {
        (corners
            .iter()
            .map(|p| p[i])
            .fold(f32::NEG_INFINITY, f32::max)
            .ceil()
            .max(0.0) as usize
            + 1)
        .min(len)
    };
    let (x0, x1, y0, y1) = (lo(0), hi(0, img.w), lo(1), hi(1, img.h));
    let feather = FACE as f32 * 0.1;
    for y in y0..y1 {
        for x in x0..x1 {
            let [u, v] = to_face.apply([x as f32, y as f32]);
            let edge = u.min(v).min(n - u).min(n - v);
            if edge <= 0.0 {
                continue;
            }
            let t = (edge / feather).min(1.0);
            let w = t * t * (3.0 - 2.0 * t); // smoothstep
            for c in 0..3 {
                let o = img.at(c, x, y);
                img.set(c, x, y, o + (sample(face, c, u, v) - o) * w);
            }
        }
    }
}

/// Set CodeFormer's fidelity weight `w` in a converted generator's param
/// text: its four fuse blocks are `Eltwise` sums `dec + w·residual`
/// (`-23301=2,1,w`). Returns the text and how many blocks were set.
pub fn with_fidelity(param: &str, w: f32) -> (String, usize) {
    let mut n = 0;
    let lines: Vec<String> = param
        .lines()
        .map(|line| {
            if !line.starts_with("Eltwise") {
                return line.to_owned();
            }
            let parts: Vec<String> = line
                .split_whitespace()
                .map(|t| {
                    if t.starts_with("-23301=2,") {
                        n += 1;
                        format!("-23301=2,1.000000,{w:.6}")
                    } else {
                        t.to_owned()
                    }
                })
                .collect();
            parts.join(" ")
        })
        .collect();
    (lines.join("\n") + "\n", n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nose_position_tells_frontal_from_profile() {
        let face = |kps| Face {
            score: 1.0,
            bbox: [0.0, 0.0, 100.0, 100.0],
            kps,
        };
        let frontal = face(TEMPLATE).nose_position();
        assert!((frontal - 0.5).abs() < 0.05, "{frontal}");
        let mut turned = TEMPLATE;
        turned[2][0] = TEMPLATE[1][0] + 15.0; // nose past the far eye
        assert!(face(turned).nose_position() > 1.0);
    }

    #[test]
    fn similarity_recovers_a_known_transform() {
        let truth = Affine([0.8 * 0.6, -0.8 * 0.8, 40.0, 0.8 * 0.8, 0.8 * 0.6, -12.0]);
        let src = [
            [10.0, 20.0],
            [200.0, 25.0],
            [100.0, 140.0],
            [30.0, 230.0],
            [190.0, 210.0],
        ];
        let dst = src.map(|p| truth.apply(p));
        let m = similarity(&src, &dst).expect("fit");
        for (a, b) in m.0.iter().zip(truth.0) {
            assert!((a - b).abs() < 1e-3, "{m:?} vs {truth:?}");
        }
        let back = m.invert().expect("inv");
        let p = back.apply(m.apply([77.0, 13.0]));
        assert!((p[0] - 77.0).abs() < 1e-3 && (p[1] - 13.0).abs() < 1e-3);
    }

    #[test]
    fn warp_then_paste_is_identity_inside_the_face() {
        let mut img = Planar::zeros(300, 200);
        for c in 0..3 {
            for y in 0..200 {
                for x in 0..300 {
                    let v =
                        0.5 + 0.4 * (x as f32 / 17.0 + c as f32).sin() * (y as f32 / 23.0).cos();
                    img.set(c, x, y, v);
                }
            }
        }
        let orig = img.clone();
        // Face square: a 120 px face rotated a little.
        let to_face = similarity(
            &[
                [100.0, 80.0],
                [160.0, 82.0],
                [130.0, 115.0],
                [105.0, 140.0],
                [158.0, 141.0],
            ],
            &TEMPLATE,
        )
        .expect("align");
        let crop = warp(&img, &to_face.invert().expect("inv"), FACE, FACE);
        paste(&mut img, &crop, &to_face);
        // Pasting the face's own crop back changes (almost) nothing.
        let diff = img
            .data
            .iter()
            .zip(&orig.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(diff < 0.03, "max diff {diff}");
        // Far from the face nothing is touched.
        assert_eq!(img.at(0, 5, 5), orig.at(0, 5, 5));
    }

    #[test]
    fn yunet_decoding_and_nms() {
        // One strong anchor at stride 32, cell (row 3, col 5); everything else 0.
        let (rows8, rows16, rows32) = (80 * 80, 40 * 40, 20 * 20);
        let mut bufs: Vec<(Vec<f32>, usize)> = Vec::new();
        for rows in [rows8, rows16, rows32] {
            for ch in [1usize, 1, 4, 10] {
                bufs.push((vec![0.0; rows * ch], ch));
            }
        }
        let idx = 3 * 20 + 5;
        bufs[8].0[idx] = 0.9; // cls_32
        bufs[9].0[idx] = 0.9; // obj_32
        bufs[10].0[idx * 4..idx * 4 + 4].copy_from_slice(&[0.5, 0.5, 0.0, 0.0]);
        let heads: Vec<Head<'_>> = bufs
            .iter()
            .map(|(d, ch)| Head {
                data: d,
                w: *ch,
                h: d.len() / ch,
                c: 1,
            })
            .collect();
        let faces = decode_yunet(&heads, 0.5);
        assert_eq!(faces.len(), 1);
        let f = faces[0];
        assert!((f.score - 0.9).abs() < 1e-5);
        assert_eq!(
            f.bbox,
            [(5.5 * 32.0) - 16.0, (3.5 * 32.0) - 16.0, 32.0, 32.0]
        );
        assert_eq!(f.kps[0], [5.0 * 32.0, 3.0 * 32.0]);

        let a = Face {
            score: 0.9,
            bbox: [0.0, 0.0, 10.0, 10.0],
            kps: [[0.0; 2]; 5],
        };
        let b = Face {
            score: 0.8,
            bbox: [1.0, 1.0, 10.0, 10.0],
            ..a
        };
        let c = Face {
            score: 0.7,
            bbox: [50.0, 50.0, 10.0, 10.0],
            ..a
        };
        assert_eq!(nms(vec![b, c, a], 0.3), vec![a, c]);
    }

    #[test]
    fn fidelity_rewrites_the_fuse_blocks() {
        let p = "7767517\n2 3\nEltwise Add_585 2 1 a b c 0=1 -23301=2,1.000000e+00,7.000000e-01\nConvolution x 1 1 c d 0=3\n";
        let (t, n) = with_fidelity(p, 0.25);
        assert_eq!(n, 1);
        assert!(
            t.contains("-23301=2,1.000000,0.250000") && t.contains("Convolution x 1 1 c d 0=3")
        );
    }
}
