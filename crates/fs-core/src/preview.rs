//! 2-second previews: the *same* compiled pipeline and engine as a real render,
//! applied to a short window and written as JPEG sequences the UI can scrub.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Instant;

use fs_media::encoders::{EncoderEnv, Os};
use fs_media::{plan, probe, rate};
use fs_types::media::{AnalysisReport, FieldOrder};
use fs_types::pipeline::{
    Container, EncoderPreference, JobSpec, OutputSpec, PipelineSpec, VideoCodec,
};
use fs_types::worker::{
    PROTOCOL_VERSION, PreviewRequest, StartPoint, WorkerCommand, WorkerEvent, WorkerJob,
};
use fs_types::{GpuDevice, JobId, PreviewResult};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

pub const PREVIEW_SECONDS: f64 = 2.0;

#[derive(Debug, Clone)]
pub struct PreviewTools {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
    pub worker: PathBuf,
    pub models_dir: PathBuf,
    pub device: Option<GpuDevice>,
}

fn count_jpegs(dir: &Path) -> u32 {
    std::fs::read_dir(dir)
        .map(|it| {
            it.flatten()
                .filter(|e| e.path().extension().is_some_and(|x| x == "jpg"))
                .count() as u32
        })
        .unwrap_or(0)
}

/// Render `seconds` (≤ [`PREVIEW_SECONDS`]; a single frame if tiny) of `input`
/// starting at `at_s` into `<root>/<id>/`. Returns the protocol prefix
/// `preview/<id>` plus metadata.
pub async fn render(
    tools: &PreviewTools,
    input: &Path,
    at_s: f64,
    seconds: f64,
    pipeline: &PipelineSpec,
    root: &Path,
) -> Result<PreviewResult, String> {
    let started = Instant::now();
    let (fp, inp) = (tools.ffprobe.clone(), input.to_path_buf());
    let info = tokio::task::spawn_blocking(move || probe::probe(&fp, &inp))
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let conform = rate::conform_rate(
        Some(info.video.r_frame_rate),
        Some(info.video.avg_frame_rate),
    );
    let total = plan::estimate_frames(&info, conform).max(1);

    // A full ANALYZE pass is unnecessary for 2 s: no scene cuts in the window,
    // frame count from the container duration.
    let analysis = AnalysisReport {
        conform_rate: conform,
        vfr_ratio: 0.0,
        is_vfr: false,
        field_order: FieldOrder::Progressive,
        scene_cuts: vec![],
        total_frames: total,
    };
    // The encoder is unused in preview mode; any always-available one satisfies the planner.
    let spec = JobSpec {
        input: input.to_string_lossy().into(),
        pipeline: pipeline.clone(),
        output: OutputSpec {
            path: String::new(),
            codec: VideoCodec::Av1,
            container: Container::Mkv,
            quality: 30,
            encoder: EncoderPreference::Software,
        },
        device: None,
    };
    let env = EncoderEnv {
        os: Os::current(),
        vendor: None,
        usable: ["libsvtav1".to_owned()].into(),
        vaapi_device: None,
    };

    let id = JobId::new().to_string();
    let out_dir = root.join(&id);
    let planned = plan::compile(&plan::PlanInputs {
        info: &info,
        analysis: &analysis,
        spec: &spec,
        device: tools.device.as_ref(),
        encoders: &env,
        ffmpeg: &tools.ffmpeg.to_string_lossy(),
        segment_dir: &out_dir.to_string_lossy(),
        models_dir: &tools.models_dir.to_string_lossy(),
    })
    .map_err(|e| e.to_string())?
    .plan;

    // `seconds` 0 → a single-frame still (shown first while the full clip renders).
    let seconds = seconds.clamp(0.0, PREVIEW_SECONDS);
    let window = (seconds * planned.decode.conform_rate.as_f64())
        .round()
        .max(1.0) as u64;
    let start = planned
        .decode
        .conform_rate
        .frame_at(at_s.max(0.0))
        .min(total.saturating_sub(window));
    let start_s = planned.decode.conform_rate.frame_time(start);
    let (width, height) = (planned.encode.width, planned.encode.height);
    let fps = planned.encode.rate.as_f64();
    let tone_mapped = planned.decode.color.is_hdr();

    let mut cmd = std::process::Command::new(&tools.worker);
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    fs_proc::prepare(&mut cmd);
    let mut child = tokio::process::Command::from(cmd)
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("could not start fs-worker: {e}"))?;
    let job = WorkerJob {
        protocol: PROTOCOL_VERSION,
        job: JobId::new(),
        plan: planned,
        start: StartPoint {
            segment: 0,
            first_frame: start,
            expected_hash: None,
        },
        duty: 1.0,
        preview: Some(PreviewRequest {
            frames: window,
            out_dir: out_dir.to_string_lossy().into(),
        }),
    };
    let mut stdin = child.stdin.take().ok_or("no worker stdin")?;
    let line =
        serde_json::to_string(&WorkerCommand::Start(Box::new(job))).map_err(|e| e.to_string())?;
    stdin
        .write_all(format!("{line}\n").as_bytes())
        .await
        .map_err(|e| e.to_string())?;
    stdin.flush().await.map_err(|e| e.to_string())?;

    let mut backend = String::from("unknown");
    let mut device = None;
    let mut error = None;
    let mut lines = BufReader::new(child.stdout.take().ok_or("no worker stdout")?).lines();
    while let Ok(Some(l)) = lines.next_line().await {
        match serde_json::from_str::<WorkerEvent>(&l) {
            Ok(WorkerEvent::Ready {
                backend: b,
                device_name,
                ..
            }) => {
                backend = b;
                device = device_name;
            }
            Ok(WorkerEvent::Error { message, .. }) => error = Some(message),
            _ => {}
        }
    }
    drop(stdin);
    let status = child.wait().await.map_err(|e| e.to_string())?;
    if !status.success() {
        let _ = std::fs::remove_dir_all(&out_dir);
        return Err(error.unwrap_or_else(|| format!("preview worker exited with {status}")));
    }
    let result = PreviewResult {
        prefix: format!("preview/{id}"),
        before_frames: count_jpegs(&out_dir.join("before")),
        after_frames: count_jpegs(&out_dir.join("after")),
        width,
        height,
        source_width: info.video.width,
        source_height: info.video.height,
        fps,
        start_s,
        backend,
        device,
        elapsed_ms: started.elapsed().as_millis() as u32,
        tone_mapped,
    };
    if result.after_frames == 0 {
        return Err("preview produced no frames".into());
    }
    Ok(result)
}

/// Keep only the newest `keep` previews (each is tens of MB of JPEGs).
pub fn prune(root: &Path, keep: usize) {
    let Ok(rd) = std::fs::read_dir(root) else {
        return;
    };
    let mut dirs: Vec<_> = rd
        .flatten()
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .collect();
    dirs.sort_by_key(|d| std::cmp::Reverse(d.0));
    for (_, d) in dirs.into_iter().skip(keep) {
        let _ = std::fs::remove_dir_all(d);
    }
}

/// Thumbnail at ~10 % into the clip, cached by (path, size, mtime).
pub async fn thumbnail(
    ffmpeg: &Path,
    ffprobe: &Path,
    input: &Path,
    root: &Path,
) -> Result<String, String> {
    let meta = std::fs::metadata(input).map_err(|e| e.to_string())?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    let key = format!(
        "{:016x}",
        xxh(&format!("{}|{}|{mtime}", input.display(), meta.len()))
    );
    let out = root.join(format!("{key}.jpg"));
    let rel = format!("thumbs/{key}.jpg");
    if out.exists() {
        return Ok(rel);
    }
    std::fs::create_dir_all(root).map_err(|e| e.to_string())?;
    let (fp, inp) = (ffprobe.to_path_buf(), input.to_path_buf());
    let info = tokio::task::spawn_blocking(move || probe::probe(&fp, &inp))
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let at = (info.duration_s * 0.1).min(30.0);
    let tmp = root.join(format!("{key}.tmp.jpg"));
    let argv = fs_media::argv::thumbnail(input, info.video.index, at, &info.video.color, 320, &tmp);
    let mut cmd = std::process::Command::new(ffmpeg);
    cmd.args(argv.as_slice())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    fs_proc::prepare(&mut cmd);
    let o = tokio::process::Command::from(cmd)
        .output()
        .await
        .map_err(|e| e.to_string())?;
    if !o.status.success() || !tmp.exists() {
        return Err(String::from_utf8_lossy(&o.stderr)
            .lines()
            .last()
            .unwrap_or("thumbnail failed")
            .to_owned());
    }
    std::fs::rename(&tmp, &out).map_err(|e| e.to_string())?;
    Ok(rel)
}

/// FNV-1a: a tiny stable cache-key hash (no extra dependency).
fn xxh(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}
