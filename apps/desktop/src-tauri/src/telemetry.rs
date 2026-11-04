//! Telemetry loop: samples the selected GPU every 500 ms, runs the Eco
//! governor, and pushes duty changes to the active worker.

use std::sync::Arc;
use std::time::{Duration, Instant};

use fs_core::JobCommand;
use fs_hw::governor::Governor;
use fs_hw::telemetry::Sampler;
use fs_types::{DeviceUuid, EcoState, Telemetry};

use crate::state::AppState;

pub const PERIOD: Duration = Duration::from_millis(500);

pub async fn run(state: Arc<AppState>) {
    let initial_mode = state.settings.lock().map(|s| s.eco).unwrap_or_default();
    let mut governor = Governor::new(initial_mode);
    let mut sampler: Option<(DeviceUuid, Sampler)> = None;
    let mut last_sent_duty = 1.0f32;
    let mut last = Instant::now();
    let mut tick = tokio::time::interval(PERIOD);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tick.tick().await;
        let dt = last.elapsed().as_secs_f32();
        last = Instant::now();

        // (Re)build the sampler if the selected device changed.
        let device = state.selected_device();
        match (&device, &sampler) {
            (Some(d), Some((u, _))) if *u == d.uuid => {}
            (Some(d), _) => {
                let vk = state.hw.read().ok().and_then(|h| h.vulkan.clone());
                sampler = Some((d.uuid, Sampler::for_device(d.clone(), vk)));
            }
            (None, _) => sampler = None,
        }
        let mode = state.settings.lock().map(|s| s.eco).unwrap_or_default();
        if mode != governor.mode() {
            governor.set_mode(mode);
        }

        let dev = sampler
            .as_mut()
            .map(|(_, s)| s.sample())
            .unwrap_or_default();
        let on_battery = fs_hw::power::on_battery();
        let duty = governor.update(dev.temperature_c, on_battery == Some(true), dt);

        // Push duty to the running worker only when it changed meaningfully.
        if (duty - last_sent_duty).abs() >= 0.05 {
            last_sent_duty = duty;
            if let Ok(a) = state.active.lock()
                && let Some(job) = a.as_ref()
            {
                let _ = job.commands.send(JobCommand::SetDuty(duty));
            }
        }

        let disk_free_bytes = dirs_for_disk(&state).and_then(|p| fs_core::storage::free_space(&p));
        state.telemetry.send_modify(|t| {
            *t = Telemetry {
                at_ms: crate::state::now_secs() * 1000,
                device: dev,
                render_fps: t.render_fps,
                engine: t.engine.clone(),
                eco: EcoState {
                    mode,
                    duty,
                    throttling: governor.is_throttling(),
                    on_battery: on_battery == Some(true),
                },
                disk_free_bytes,
            };
        });
    }
}

/// The volume that matters: the active job's output, else the temp root.
fn dirs_for_disk(state: &AppState) -> Option<std::path::PathBuf> {
    let active = state
        .active
        .lock()
        .ok()
        .and_then(|a| a.as_ref().map(|j| j.id));
    if let Some(id) = active
        && let Ok(j) = state.journal.lock()
        && let Ok(spec) = j.spec(id)
    {
        return Some(spec.output.path.into());
    }
    Some(state.paths.work_root.clone())
}
