//! End-to-end: the real fs-worker binary + the real FFmpeg.
//!
//! Requires `FS_FFMPEG` / `FS_FFPROBE`; skips otherwise. Uses a lossless
//! (FFV1) segment encoder so pixels can be compared exactly.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use fs_media::encoders::{EncoderEnv, Os};
use fs_media::{analyze, mux, plan, probe, rate};
use fs_types::JobId;
use fs_types::pipeline::*;
use fs_types::worker::{
    PROTOCOL_VERSION, RenderPlan, SegmentInfo, StartPoint, WorkerCommand, WorkerEvent, WorkerJob,
    exit,
};

struct Tools {
    ffmpeg: PathBuf,
    ffprobe: PathBuf,
}

fn tools() -> Option<Tools> {
    Some(Tools {
        ffmpeg: std::env::var_os("FS_FFMPEG")?.into(),
        ffprobe: std::env::var_os("FS_FFPROBE")?.into(),
    })
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("fs-worker-e2e-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("tmp");
    d
}

fn ff(t: &Tools, args: &[&str]) {
    let o = Command::new(&t.ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args(args)
        .output()
        .expect("ffmpeg");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
}

/// 3 scenes × 2 s @ 24 fps (cuts at 48, 96), 6 s audio, video starts at +0.25 s.
fn source(t: &Tools, dir: &Path) -> PathBuf {
    let p = dir.join("src.mkv");
    ff(
        t,
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=160x120:r=24:d=2",
            "-f",
            "lavfi",
            "-i",
            "mandelbrot=s=160x120:r=24,trim=duration=2",
            "-f",
            "lavfi",
            "-i",
            "smptehdbars=s=160x120:r=24:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=f=440:d=6.25",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p,setsar=16/15,setpts=PTS+0.25/TB[v]",
            "-map",
            "[v]",
            "-map",
            "3:a",
            "-c:v",
            "ffv1",
            "-c:a",
            "flac",
            p.to_str().expect("utf8"),
        ],
    );
    p
}

fn make_plan(t: &Tools, src: &Path, seg_dir: &Path) -> (RenderPlan, fs_types::MediaInfo) {
    let info = probe::probe(&t.ffprobe, src).expect("probe");
    let conform = rate::conform_rate(
        Some(info.video.r_frame_rate),
        Some(info.video.avg_frame_rate),
    );
    let a = analyze::argv(
        src,
        info.video.index,
        &fs_types::worker::HwDecode::None,
        conform,
        Some(10.0),
    );
    let out = Command::new(&t.ffmpeg)
        .args(a.as_slice())
        .output()
        .expect("analyze");
    let mut p = analyze::AnalyzeParser::new();
    String::from_utf8_lossy(&out.stderr)
        .lines()
        .for_each(|l| p.feed_stderr(l));
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .for_each(|l| p.feed_progress(l));
    let analysis = p.finish(conform);
    assert_eq!(analysis.scene_cuts, vec![48, 96]);

    let model = ModelRef {
        id: "reference".into(),
        path: None,
    };
    let spec = JobSpec {
        input: src.to_string_lossy().into(),
        pipeline: PipelineSpec {
            stages: vec![
                StageConfig::Upscale {
                    model: model.clone(),
                    scale: 2,
                    denoise: None,
                },
                StageConfig::Interpolate {
                    model,
                    factor: 2,
                    scene_threshold: 10.0,
                },
            ],
        },
        output: OutputSpec {
            path: "unused".into(),
            codec: VideoCodec::Av1,
            container: Container::Mkv,
            quality: 20,
            encoder: EncoderPreference::Auto,
        },
        device: None,
    };
    let env = EncoderEnv {
        os: Os::current(),
        vendor: None,
        usable: ["libsvtav1".to_owned()].into(),
        vaapi_device: None,
    };
    let mut planned = plan::compile(&plan::PlanInputs {
        info: &info,
        analysis: &analysis,
        spec: &spec,
        device: None,
        encoders: &env,
        ffmpeg: t.ffmpeg.to_str().expect("utf8"),
        segment_dir: seg_dir.to_str().expect("utf8"),
        models_dir: "",
    })
    .expect("plan")
    .plan;
    // Lossless segments so we can compare pixels exactly.
    planned.encode.encoder = "ffv1".into();
    planned.encode.args = vec![];
    planned.encode.pix_fmt = "yuv420p".into();
    (planned, info)
}

#[derive(Default)]
struct RunLog {
    committed: Vec<SegmentInfo>,
    finished: Option<(u64, u64)>,
    stopped: Option<u32>,
    exit: i32,
    errors: Vec<String>,
}

fn run_worker(plan: &RenderPlan, start: StartPoint, stop_after_ready: bool) -> RunLog {
    let mut child = Command::new(env!("CARGO_BIN_EXE_fs-worker"))
        // Deterministic non-AI backend: these tests check pixels bit-exactly.
        .env("FS_ENGINE", "reference")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("worker");
    let mut stdin = child.stdin.take().expect("stdin");
    let job = WorkerJob {
        protocol: PROTOCOL_VERSION,
        job: JobId::new(),
        plan: plan.clone(),
        start,
        duty: 1.0,
        preview: None,
    };
    writeln!(
        stdin,
        "{}",
        serde_json::to_string(&WorkerCommand::Start(Box::new(job))).expect("json")
    )
    .expect("send");
    let mut log = RunLog::default();
    for line in BufReader::new(child.stdout.take().expect("stdout")).lines() {
        let line = line.expect("line");
        match serde_json::from_str::<WorkerEvent>(&line).expect("event") {
            WorkerEvent::Ready { .. } if stop_after_ready => {
                writeln!(
                    stdin,
                    "{}",
                    serde_json::to_string(&WorkerCommand::Stop { checkpoint: true }).expect("j")
                )
                .expect("stop");
            }
            WorkerEvent::SegmentCommitted(s) => log.committed.push(s),
            WorkerEvent::Finished {
                frames_in,
                frames_out,
            } => log.finished = Some((frames_in, frames_out)),
            WorkerEvent::Stopped { next_segment } => log.stopped = Some(next_segment),
            WorkerEvent::Error { message, .. } => log.errors.push(message),
            _ => {}
        }
    }
    drop(stdin);
    log.exit = child.wait().expect("wait").code().unwrap_or(-1);
    log
}

fn framemd5(t: &Tools, file: &Path) -> Vec<String> {
    let o = Command::new(&t.ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(file)
        .args(["-map", "0:v:0", "-f", "framemd5", "-"])
        .output()
        .expect("framemd5");
    String::from_utf8_lossy(&o.stdout)
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| l.rsplit(',').next().map(|s| s.trim().to_owned()))
        .collect()
}

fn concat(t: &Tools, dir: &Path, segs: &[SegmentInfo], out: &Path) {
    let list = dir.join("list.txt");
    std::fs::write(
        &list,
        fs_media::argv::concat_list(segs.iter().map(|s| s.file_name.as_str())),
    )
    .expect("list");
    let a = fs_media::argv::concat(&list, out);
    let o = Command::new(&t.ffmpeg)
        .args(a.as_slice())
        .output()
        .expect("concat");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
}

#[test]
fn full_run_cut_holds_and_bit_exact_resume() {
    let Some(t) = tools() else {
        return eprintln!("skip: FS_FFMPEG/FS_FFPROBE unset");
    };
    let dir = tmp("main");
    let src = source(&t, &dir);

    // --- 1. uninterrupted run -------------------------------------------------------
    let seg_a = dir.join("a");
    let (plan_a, info) = make_plan(&t, &src, &seg_a);
    // Regression: a `.part` left by a killed earlier attempt must not block the
    // render (FFmpeg used to refuse with "already exists").
    std::fs::create_dir_all(&seg_a).expect("seg dir");
    std::fs::write(seg_a.join("seg_00000.mkv.part"), b"stale partial segment").expect("stale");
    assert_eq!(plan_a.segment_frames, 96);
    let a = run_worker(
        &plan_a,
        StartPoint {
            segment: 0,
            first_frame: 0,
            expected_hash: None,
        },
        false,
    );
    assert!(a.errors.is_empty(), "{:?}", a.errors);
    assert_eq!(a.exit, exit::OK);
    assert_eq!(a.finished, Some((144, 288)), "factor × N frames");
    assert_eq!(
        a.committed
            .iter()
            .map(|s| (s.index, s.frame_count, s.out_frames))
            .collect::<Vec<_>>(),
        vec![(0, 96, 192), (1, 48, 96)]
    );
    assert!(a.committed[0].next_hash.is_some() && a.committed[1].next_hash.is_none());

    let full_a = dir.join("a.mkv");
    concat(&t, &seg_a, &a.committed, &full_a);
    let md5_a = framemd5(&t, &full_a);
    assert_eq!(md5_a.len(), 288);
    // Cut between input 47 and 48 → output 95 is a hold of output 94.
    assert_eq!(md5_a[95], md5_a[94], "must not morph across the cut");
    // Inside a scene, the intermediate is a real blend.
    assert_ne!(md5_a[3], md5_a[2]);

    // --- 2. hard pause after segment 0, then resume ---------------------------------
    let seg_b = dir.join("b");
    let (plan_b, _) = make_plan(&t, &src, &seg_b);
    let b1 = run_worker(
        &plan_b,
        StartPoint {
            segment: 0,
            first_frame: 0,
            expected_hash: None,
        },
        true,
    );
    assert_eq!(b1.exit, exit::STOPPED, "{:?}", b1.errors);
    assert_eq!(b1.stopped, Some(1));
    assert_eq!(b1.committed.len(), 1);
    assert!(
        !seg_b.join("seg_00001.mkv.part").exists(),
        "no partial segment left behind"
    );

    let resume = StartPoint {
        segment: 1,
        first_frame: 96,
        expected_hash: b1.committed[0].next_hash,
    };
    let b2 = run_worker(&plan_b, resume, false);
    assert_eq!(b2.exit, exit::OK, "{:?}", b2.errors);
    assert_eq!(b2.committed.len(), 1);
    assert_eq!(b2.committed[0].index, 1);

    let mut segs = b1.committed.clone();
    segs.extend(b2.committed.clone());
    let full_b = dir.join("b.mkv");
    concat(&t, &seg_b, &segs, &full_b);
    assert_eq!(
        framemd5(&t, &full_b),
        md5_a,
        "resumed render must be bit-identical"
    );

    // --- 3. final mux: audio back, A/V offset preserved -----------------------------
    let out = dir.join("final.mkv");
    let m = mux::plan(&full_b, &info, Container::Mkv, VideoCodec::Av1, &out);
    let o = Command::new(&t.ffmpeg)
        .args(m.argv.as_slice())
        .output()
        .expect("mux");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let fin = probe::probe(&t.ffprobe, &out).expect("probe final");
    assert_eq!(fin.audio.len(), 1);
    assert_eq!((fin.video.width, fin.video.height), (320, 240));
    assert_eq!(
        fin.video.sample_aspect,
        fs_types::Rational::new(16, 15),
        "anamorphic pixel aspect must survive segments, concat and mux"
    );
    assert!(
        (fin.video.start_time - fin.audio[0].start_time - 0.25).abs() < 0.02,
        "video must start 0.25 s after audio, got v={} a={}",
        fin.video.start_time,
        fin.audio[0].start_time
    );
    let _ = std::fs::remove_dir_all(dir);
}
