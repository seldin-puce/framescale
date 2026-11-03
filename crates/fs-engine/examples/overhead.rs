//! CPU-side cost of one 1080p→4K SR frame with the network replaced by a
//! trivial ×2 copy: conversion + tiling + stitching + packing, no inference.
use std::time::Instant;

use fs_engine::Frame;
use fs_engine::planar::{self, Planar};
use fs_types::worker::FrameFormat;

fn main() {
    let (w, h) = (1920u32, 1080u32);
    let f = Frame::new(
        w,
        h,
        FrameFormat::Rgb24,
        (0..w * h * 3).map(|i| (i % 251) as u8).collect(),
    );
    for _ in 0..2 {
        let t0 = Instant::now();
        let p = planar::from_frame(&f);
        let t1 = Instant::now();
        let out = planar::tiled(&p, 512, 10, 2, |t| {
            let mut o = Planar::zeros(t.w * 2, t.h * 2);
            o.data[0] = 0.5; // like the real path: ncnn writes the buffer
            Ok(o)
        })
        .expect("tiled");
        let t2 = Instant::now();
        let fr = planar::to_frame(&out, FrameFormat::Rgb24);
        let t3 = Instant::now();
        println!(
            "from_frame {:>6.1} ms | tiling+stitch {:>6.1} ms | to_frame {:>6.1} ms | total {:>6.1} ms ({} B out)",
            (t1 - t0).as_secs_f64() * 1e3,
            (t2 - t1).as_secs_f64() * 1e3,
            (t3 - t2).as_secs_f64() * 1e3,
            (t3 - t0).as_secs_f64() * 1e3,
            fr.data.len()
        );
    }
}
