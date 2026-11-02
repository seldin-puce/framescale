//! Parser for FFmpeg's machine-readable `-progress` output (key=value blocks,
//! each terminated by `progress=continue|end`).

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProgressSnapshot {
    pub frame: u64,
    pub fps: f32,
    /// Output timestamp in microseconds.
    pub out_time_us: i64,
    pub total_size: u64,
    /// Realtime multiple (`speed=1.5x`), when known.
    pub speed: Option<f32>,
    pub end: bool,
}

#[derive(Debug, Default)]
pub struct ProgressParser {
    cur: ProgressSnapshot,
}

impl ProgressParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one line. Returns a complete snapshot at each block terminator.
    pub fn feed(&mut self, line: &str) -> Option<ProgressSnapshot> {
        let (k, v) = line.trim().split_once('=')?;
        let v = v.trim();
        match k {
            "frame" => self.cur.frame = v.parse().unwrap_or(self.cur.frame),
            "fps" => self.cur.fps = v.parse().unwrap_or(0.0),
            // `out_time_ms` is misnamed upstream and is actually µs. Prefer `out_time_us`.
            "out_time_us" | "out_time_ms" => {
                if let Ok(x) = v.parse() {
                    self.cur.out_time_us = x;
                }
            }
            "total_size" => self.cur.total_size = v.parse().unwrap_or(self.cur.total_size),
            "speed" => self.cur.speed = v.trim_end_matches('x').trim().parse().ok(),
            "progress" => {
                self.cur.end = v == "end";
                return Some(self.cur.clone());
            }
            _ => {}
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_blocks() {
        let text = "frame=120\nfps=59.94\nstream_0_0_q=-0.0\ntotal_size=1048576\nout_time_us=5005000\nout_time_ms=5005000\nout_time=00:00:05.005000\nspeed=2.5x\nprogress=continue\nframe=240\nfps=60.1\nspeed=N/A\nprogress=end\n";
        let mut p = ProgressParser::new();
        let snaps: Vec<_> = text.lines().filter_map(|l| p.feed(l)).collect();
        assert_eq!(snaps.len(), 2);
        assert_eq!(snaps[0].frame, 120);
        assert_eq!(snaps[0].out_time_us, 5_005_000);
        assert_eq!(snaps[0].speed, Some(2.5));
        assert!(!snaps[0].end);
        assert_eq!(snaps[1].frame, 240);
        assert_eq!(snaps[1].speed, None);
        assert!(snaps[1].end);
    }
}
