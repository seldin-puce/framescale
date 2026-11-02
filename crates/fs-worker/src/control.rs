//! Commands arriving on stdin after `Start`: pause/resume, stop, duty.

use std::io::BufRead;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use fs_types::worker::WorkerCommand;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopMode {
    None,
    /// Finish the current segment, then exit (hard pause).
    Checkpoint,
    /// Abort immediately; the in-progress segment is discarded.
    Now,
}

#[derive(Debug)]
pub struct Control {
    paused: Mutex<bool>,
    cv: Condvar,
    stop: AtomicU8,
    duty_bits: AtomicU32,
}

impl Control {
    pub fn new(duty: f32) -> Arc<Self> {
        Arc::new(Self {
            paused: Mutex::new(false),
            cv: Condvar::new(),
            stop: AtomicU8::new(0),
            duty_bits: AtomicU32::new(duty.clamp(0.1, 1.0).to_bits()),
        })
    }

    pub fn set_paused(&self, p: bool) {
        if let Ok(mut g) = self.paused.lock() {
            *g = p;
        }
        self.cv.notify_all();
    }

    /// Soft pause: block the decode reader. FFmpeg then blocks on its full
    /// pipe, so the whole pipeline idles without losing any state.
    pub fn wait_if_paused(&self) {
        let Ok(mut g) = self.paused.lock() else {
            return;
        };
        while *g && self.stop_mode() == StopMode::None {
            g = match self.cv.wait(g) {
                Ok(g) => g,
                Err(_) => return,
            };
        }
    }

    pub fn request_stop(&self, checkpoint: bool) {
        let m = if checkpoint { 1 } else { 2 };
        // Never downgrade an immediate stop to a checkpoint.
        self.stop.fetch_max(m, Ordering::SeqCst);
        self.cv.notify_all();
    }

    pub fn stop_mode(&self) -> StopMode {
        match self.stop.load(Ordering::SeqCst) {
            0 => StopMode::None,
            1 => StopMode::Checkpoint,
            _ => StopMode::Now,
        }
    }

    pub fn duty(&self) -> f32 {
        f32::from_bits(self.duty_bits.load(Ordering::Relaxed))
    }

    pub fn set_duty(&self, d: f32) {
        self.duty_bits
            .store(d.clamp(0.1, 1.0).to_bits(), Ordering::Relaxed);
    }
}

/// Read further commands. If stdin closes, the core is gone: kill our own
/// group (FFmpeg children included) and exit, so we never become a zombie.
pub fn spawn_reader(ctrl: Arc<Control>, input: impl BufRead + Send + 'static) {
    std::thread::Builder::new()
        .name("control".into())
        .spawn(move || {
            for line in input.lines() {
                let Ok(line) = line else { break };
                match serde_json::from_str::<WorkerCommand>(&line) {
                    Ok(WorkerCommand::Pause) => ctrl.set_paused(true),
                    Ok(WorkerCommand::Resume) => ctrl.set_paused(false),
                    Ok(WorkerCommand::Stop { checkpoint }) => ctrl.request_stop(checkpoint),
                    Ok(WorkerCommand::SetDuty { duty }) => ctrl.set_duty(duty),
                    Ok(WorkerCommand::Start(_)) | Err(_) => {} // ignore duplicates / garbage
                }
            }
            fs_proc::kill_own_group();
            std::process::exit(fs_types::worker::exit::FATAL);
        })
        .ok();
}
