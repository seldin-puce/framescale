//! JSON-lines events on stdout (the only thing ever written to stdout).

use std::io::Write;
use std::sync::{Arc, Mutex};

use fs_types::worker::WorkerEvent;

#[derive(Clone, Debug)]
pub struct Emitter(Arc<Mutex<std::io::Stdout>>);

impl Emitter {
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(std::io::stdout())))
    }

    pub fn send(&self, ev: &WorkerEvent) {
        let Ok(line) = serde_json::to_string(ev) else {
            return;
        };
        if let Ok(mut out) = self.0.lock() {
            // A closed stdout means the core is gone; the watchdog handles exit.
            let _ = writeln!(out, "{line}").and_then(|()| out.flush());
        }
    }
}
