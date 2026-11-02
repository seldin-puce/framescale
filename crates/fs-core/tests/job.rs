//! A whole job through `run_job` with the real worker and FFmpeg.
//! Needs FS_FFMPEG, FS_FFPROBE and FS_WORKER (path to a built fs-worker).

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};

use fs_core::{JobCommand, Services, run_job};
use fs_media::encoders::{EncoderEnv, Os};
use fs_types::pipeline::*;
use fs_types::{JobEvent, JobId, JobState, PauseMode};

fn env(k: &str) -> Option<PathBuf> {
    std::env::var_os(k).map(Into::into)
}

#[tokio::test(flavor = "multi_thread")]
async fn full_job_with_hard_pause_and_resume() {
    let (Some(ffmpeg), Some(ffprobe), Some(worker)) =
        (env("FS_FFMPEG"), env("FS_FFPROBE"), env("FS_WORKER"))
    else {
        return eprintln!("skip: FS_FFMPEG/FS_FFPROBE/FS_WORKER unset");
    };
    let dir = std::env::temp_dir().join(format!("fs-core-job-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("dir");
    let src = dir.join("src.mp4");
    let st = Command::new(&ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=192x108:r=25:d=10",
            "-f",
            "lavfi",
            "-i",
            "sine=f=330:d=10",
            "-c:v",
            "libsvtav1",
            "-preset",
            "12",
            "-c:a",
            "aac",
            "-shortest",
        ])
        .arg(&src)
        .status()
        .expect("gen");
    assert!(st.success());

    let journal = Arc::new(Mutex::new(
        fs_journal::Journal::open_in_memory().expect("journal"),
    ));
    let svc = Arc::new(Services {
        journal: journal.clone(),
        ffmpeg: ffmpeg.clone(),
        ffprobe,
        worker,
        work_root: dir.join("work"),
        models_dir: dir.join("models"),
        encoders: EncoderEnv {
            os: Os::current(),
            vendor: None,
            usable: ["libsvtav1".to_owned()].into(),
            vaapi_device: None,
        },
        device: None,
        license_token: None,
        ledger: Arc::new(fs_proc::Ledger::new(dir.join("ledger.jsonl"))),
        duty: 1.0,
    });
    let out = dir.join("out.mp4");
    let m = ModelRef {
        id: "reference".into(),
        path: None,
    };
    let spec = JobSpec {
        input: src.to_string_lossy().into(),
        pipeline: PipelineSpec {
            stages: vec![StageConfig::Upscale { model: m, scale: 2 }],
        },
        output: OutputSpec {
            path: out.to_string_lossy().into(),
            codec: VideoCodec::Av1,
            container: Container::Mp4,
            quality: 30,
            encoder: EncoderPreference::Auto,
        },
        device: None,
    };
    let id = JobId::new();
    journal
        .lock()
        .expect("lock")
        .insert_job(id, &spec)
        .expect("insert");

    let (ev_tx, mut ev_rx) = tokio::sync::mpsc::unbounded_channel();
    let (cmd_tx, cmd_rx) = tokio::sync::mpsc::unbounded_channel();
    let task = tokio::spawn(run_job(id, svc.clone(), ev_tx, cmd_rx));

    // Hard-pause as soon as the first segment is rendering, then resume.
    let mut paused_once = false;
    let mut states = Vec::new();
    while let Some(ev) = ev_rx.recv().await {
        match ev {
            JobEvent::Progress(_) if !paused_once => {
                paused_once = true;
                cmd_tx
                    .send(JobCommand::Pause(PauseMode::Hard))
                    .expect("pause");
            }
            JobEvent::StateChanged { state, .. } => {
                states.push(state);
                if state == JobState::Paused {
                    cmd_tx.send(JobCommand::Resume).expect("resume");
                }
            }
            JobEvent::Error { message, .. } => panic!("job error: {message}"),
            _ => {}
        }
    }
    let final_state = task.await.expect("join");
    assert_eq!(final_state, JobState::Done, "states: {states:?}");
    assert!(
        states.contains(&JobState::Paused),
        "hard pause happened: {states:?}"
    );
    assert_eq!(states.last(), Some(&JobState::Done));

    let info = fs_media::probe::probe(&env("FS_FFPROBE").expect("p"), &out).expect("probe out");
    assert_eq!((info.video.width, info.video.height), (384, 216));
    assert_eq!(info.audio.len(), 1);
    assert!(
        (info.duration_s - 10.0).abs() < 0.5,
        "duration {}",
        info.duration_s
    );
    assert!(
        !dir.join("work").join(id.to_string()).exists(),
        "temp dir cleaned up"
    );
    assert_eq!(
        journal.lock().expect("l").state(id).expect("s"),
        JobState::Done
    );
    let _ = std::fs::remove_dir_all(dir);
}
