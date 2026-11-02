//! The batch queue: runs queued jobs one at a time, in queue order.

use std::sync::Arc;

use fs_core::{JobCommand, Services};
use fs_types::{JobEvent, JobState};

use crate::state::{ActiveJob, AppState};

pub async fn run(state: Arc<AppState>) {
    // Planning needs the encoder probe; wait for hardware discovery once.
    while !state.hardware_ready() {
        // Bounded wait: a notification sent before we started waiting is not lost.
        let _ = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            state.hw_ready.notified(),
        )
        .await;
    }
    loop {
        let next = state
            .journal
            .lock()
            .ok()
            .and_then(|j| j.list_jobs().ok())
            .and_then(|jobs| jobs.into_iter().find(|j| j.state == JobState::Queued));
        let Some(job) = next else {
            state.queue_changed.notified().await;
            continue;
        };

        let svc = Arc::new(Services {
            journal: state.journal.clone(),
            ffmpeg: state.paths.ffmpeg.clone(),
            ffprobe: state.paths.ffprobe.clone(),
            worker: state.paths.worker.clone(),
            work_root: state.paths.work_root.clone(),
            models_dir: state.paths.models.clone(),
            encoders: state.encoder_env(),
            device: state.selected_device(),
            license_token: state.license_token.read().ok().and_then(|t| t.clone()),
            ledger: state.ledger.clone(),
            duty: state.telemetry.borrow().eco.duty.max(0.1),
        });

        let (cmd_tx, cmd_rx) = tokio::sync::mpsc::unbounded_channel::<JobCommand>();
        let (ev_tx, mut ev_rx) = tokio::sync::mpsc::unbounded_channel::<JobEvent>();
        if let Ok(mut a) = state.active.lock() {
            *a = Some(ActiveJob {
                id: job.id,
                commands: cmd_tx,
            });
        }
        let forward = {
            let state = state.clone();
            tauri::async_runtime::spawn(async move {
                while let Some(ev) = ev_rx.recv().await {
                    if let JobEvent::Progress(p) = &ev {
                        state.telemetry.send_modify(|t| t.render_fps = Some(p.fps));
                    }
                    state.emit(ev);
                }
            })
        };
        let final_state = fs_core::run_job(job.id, svc, ev_tx, cmd_rx).await;
        let _ = forward.await;
        state.telemetry.send_modify(|t| t.render_fps = None);
        if let Ok(mut a) = state.active.lock() {
            *a = None;
        }
        tracing::info!(job = %job.id, ?final_state, "job finished");
        if final_state == JobState::Cancelled {
            if let Ok(j) = state.journal.lock() {
                let _ = j.set_state(job.id, JobState::Cancelled, None);
            }
            state.emit(JobEvent::StateChanged {
                job: job.id,
                state: JobState::Cancelled,
                reason: None,
            });
        }
    }
}
