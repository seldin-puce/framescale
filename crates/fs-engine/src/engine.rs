//! The contract between the shared tiler and a network running on some
//! device ([`NetEngine`]), plus [`ThreadedEngine`], which runs any blocking
//! per-tile function on a few threads (how ncnn keeps two tiles in flight).
//!
//! The pipeline-facing, frame-level trait stays [`crate::InferenceBackend`];
//! a backend uses `NetEngine`s and [`crate::planar::run_tiled`] inside it.
//! See docs/engine-audit.md, section 6.

use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::sync::{Arc, Mutex};
use std::thread::{JoinHandle, Scope};

pub use fs_types::{BackendKind, EngineStats};

use crate::EngineError;
use crate::planar::Planar;

/// A tensor in host memory: planar f32 RGB `[3][h][w]`, nominally 0–1.
/// Every engine accepts and returns it.
pub type HostTensor = Planar;

/// Memory an engine keeps on its device (a CUDA buffer, a `CVPixelBuffer`).
/// Opaque to the tiler; [`DeviceTensor::to_host`] is the fallback that always
/// works. No engine produces one yet: frames enter and leave through FFmpeg
/// pipes in host memory (docs/engine-audit.md, section 5).
pub trait DeviceTensor: Send {
    fn backend(&self) -> BackendKind;
    /// Width, height, channels.
    fn dims(&self) -> (usize, usize, usize);
    fn to_host(&self) -> Result<HostTensor, EngineError>;
}

pub enum Tensor {
    Host(HostTensor),
    Device(Box<dyn DeviceTensor>),
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Host(t) => write!(f, "Host({}×{})", t.w, t.h),
            Self::Device(d) => write!(f, "Device({:?} {:?})", d.backend(), d.dims()),
        }
    }
}

impl Tensor {
    pub fn into_host(self) -> Result<HostTensor, EngineError> {
        match self {
            Self::Host(t) => Ok(t),
            Self::Device(d) => d.to_host(),
        }
    }
}

/// Named extra inputs shared by every tile of one call, e.g. Real-CUGAN's
/// whole-frame SE statistics (`gap0`, `gap1`, … in order).
pub type SideInputs = Arc<Vec<(String, Vec<f32>)>>;

/// One submitted input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ticket(pub u64);

/// A finished submission.
#[derive(Debug)]
pub struct Completed {
    pub ticket: Ticket,
    /// The input, handed back when the engine no longer needs it (the tiler
    /// keeps it to recognise an unchanged tile next frame).
    pub input: Option<HostTensor>,
    pub output: Result<Tensor, EngineError>,
}

/// One network on one device. `Send`, not `Sync`: the thread that owns it
/// submits and waits. Behind that an engine may use threads, streams or
/// command queues of its own.
pub trait NetEngine: Send {
    /// Inputs that can be outstanding at once (≥ 1). The tiler never submits
    /// more than this.
    fn max_in_flight(&self) -> usize;
    /// Queue one input. Doesn't block while fewer than `max_in_flight` are
    /// outstanding.
    fn submit(&mut self, input: Tensor, side: Option<SideInputs>) -> Result<Ticket, EngineError>;
    /// Block until an outstanding input finishes, in any order. `None` when
    /// nothing is outstanding.
    fn wait_any(&mut self) -> Option<Completed>;
    /// [`NetEngine::wait_any`] without blocking.
    fn poll(&mut self) -> Option<Completed>;
    /// Device memory the engine holds, when it can tell.
    fn device_bytes(&self) -> Option<u64> {
        None
    }
}

/// The per-tile work a [`ThreadedEngine`] runs: one input (and the call's
/// side inputs) to one output, blocking.
pub type TileFn<'f> =
    dyn Fn(&HostTensor, Option<&SideInputs>) -> Result<HostTensor, EngineError> + Send + Sync + 'f;

type Job = (Ticket, HostTensor, Option<SideInputs>);

/// Runs a blocking per-tile function on `slots` threads, so `slots` tiles are
/// in flight. For ncnn that's the closest correct form of double buffering:
/// its public `VkCompute` only has `submit_and_wait`, so overlap comes from
/// two threads with their own extractors, which take separate queues
/// (docs/engine-audit.md, section 9).
///
/// [`ThreadedEngine::spawn`] owns its threads (joined on drop);
/// [`ThreadedEngine::scoped`] borrows a function inside `std::thread::scope`,
/// for callers whose closure borrows per-call state.
pub struct ThreadedEngine<'scope> {
    jobs: Option<Sender<Job>>,
    done: Receiver<Completed>,
    slots: usize,
    outstanding: usize,
    next: u64,
    owned: Vec<JoinHandle<()>>,
    _scope: PhantomData<&'scope ()>,
}

fn worker(f: &TileFn<'_>, jobs: &Mutex<Receiver<Job>>, done: &Sender<Completed>) {
    loop {
        let job = match jobs.lock() {
            Ok(rx) => rx.recv(),
            Err(_) => return,
        };
        let Ok((ticket, input, side)) = job else {
            return; // the engine was dropped
        };
        // A panicking network must not leave the tiler waiting forever.
        let output = catch_unwind(AssertUnwindSafe(|| f(&input, side.as_ref())))
            .unwrap_or_else(|_| Err(EngineError::Internal("network call panicked".into())))
            .map(Tensor::Host);
        let c = Completed {
            ticket,
            input: Some(input),
            output,
        };
        if done.send(c).is_err() {
            return;
        }
    }
}

impl std::fmt::Debug for ThreadedEngine<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThreadedEngine")
            .field("slots", &self.slots)
            .field("outstanding", &self.outstanding)
            .finish_non_exhaustive()
    }
}

impl ThreadedEngine<'static> {
    /// Long-lived threads named `fs-<name>`, running `f`.
    pub fn spawn(name: &str, slots: usize, f: Arc<TileFn<'static>>) -> Result<Self, EngineError> {
        let slots = slots.max(1);
        let (jobs, rx) = channel::<Job>();
        let (done_tx, done) = channel();
        let rx = Arc::new(Mutex::new(rx));
        let mut owned = Vec::with_capacity(slots);
        for i in 0..slots {
            let (f, rx, done_tx) = (f.clone(), rx.clone(), done_tx.clone());
            let h = std::thread::Builder::new()
                .name(format!("fs-{name}-{i}"))
                .spawn(move || worker(&*f, &rx, &done_tx))
                .map_err(|e| EngineError::Internal(format!("can't start {name} thread: {e}")))?;
            owned.push(h);
        }
        Ok(Self {
            jobs: Some(jobs),
            done,
            slots,
            outstanding: 0,
            next: 0,
            owned,
            _scope: PhantomData,
        })
    }
}

impl<'scope> ThreadedEngine<'scope> {
    /// Threads inside `s`, running a borrowed `f`. Drop the engine before
    /// the scope ends: that's what lets its threads exit.
    pub fn scoped<'env, F>(s: &'scope Scope<'scope, 'env>, slots: usize, f: &'env F) -> Self
    where
        F: Fn(&HostTensor, Option<&SideInputs>) -> Result<HostTensor, EngineError> + Sync,
    {
        let slots = slots.max(1);
        let (jobs, rx) = channel::<Job>();
        let (done_tx, done) = channel();
        let rx = Arc::new(Mutex::new(rx));
        for _ in 0..slots {
            let (rx, done_tx) = (rx.clone(), done_tx.clone());
            s.spawn(move || {
                let f = |t: &HostTensor, side: Option<&SideInputs>| f(t, side);
                worker(&f, &rx, &done_tx);
            });
        }
        Self {
            jobs: Some(jobs),
            done,
            slots,
            outstanding: 0,
            next: 0,
            owned: Vec::new(),
            _scope: PhantomData,
        }
    }

    fn completed(&mut self, c: Result<Completed, ()>) -> Option<Completed> {
        match c {
            Ok(c) => {
                self.outstanding -= 1;
                Some(c)
            }
            Err(()) => {
                // Every thread is gone with work outstanding.
                self.outstanding = 0;
                Some(Completed {
                    ticket: Ticket(u64::MAX),
                    input: None,
                    output: Err(EngineError::Internal("network threads exited".into())),
                })
            }
        }
    }
}

impl NetEngine for ThreadedEngine<'_> {
    fn max_in_flight(&self) -> usize {
        self.slots
    }

    fn submit(&mut self, input: Tensor, side: Option<SideInputs>) -> Result<Ticket, EngineError> {
        let input = input.into_host()?;
        let ticket = Ticket(self.next);
        let sent = self
            .jobs
            .as_ref()
            .is_some_and(|j| j.send((ticket, input, side)).is_ok());
        if !sent {
            return Err(EngineError::Internal("network threads exited".into()));
        }
        self.next += 1;
        self.outstanding += 1;
        Ok(ticket)
    }

    fn wait_any(&mut self) -> Option<Completed> {
        if self.outstanding == 0 {
            return None;
        }
        let c = self.done.recv().map_err(|_| ());
        self.completed(c)
    }

    fn poll(&mut self) -> Option<Completed> {
        if self.outstanding == 0 {
            return None;
        }
        match self.done.try_recv() {
            Ok(c) => self.completed(Ok(c)),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => self.completed(Err(())),
        }
    }
}

impl Drop for ThreadedEngine<'_> {
    fn drop(&mut self) {
        // Closing the job channel ends every worker's loop.
        drop(self.jobs.take());
        for h in self.owned.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn double(t: &HostTensor, side: Option<&SideInputs>) -> Result<HostTensor, EngineError> {
        let add = side.and_then(|s| s.first()).map_or(0.0, |(_, v)| v[0]);
        let mut o = t.clone();
        o.data.iter_mut().for_each(|v| *v = *v * 2.0 + add);
        Ok(o)
    }

    #[test]
    fn owned_threads_run_every_submission_once() {
        let mut e = ThreadedEngine::spawn("test", 3, Arc::new(double)).expect("spawn");
        assert_eq!(e.max_in_flight(), 3);
        assert!(
            e.wait_any().is_none() && e.poll().is_none(),
            "nothing outstanding"
        );
        let side: SideInputs = Arc::new(vec![("bias".into(), vec![1.0])]);
        let mut want = std::collections::HashMap::new();
        for i in 0..3 {
            let mut t = Planar::zeros(2, 2);
            t.data.fill(i as f32);
            let ticket = e
                .submit(Tensor::Host(t), Some(side.clone()))
                .expect("submit");
            want.insert(ticket, i as f32 * 2.0 + 1.0);
        }
        for _ in 0..3 {
            let c = e.wait_any().expect("one finishes");
            let out = c.output.and_then(Tensor::into_host).expect("ok");
            assert!(out.data.iter().all(|&v| v == want[&c.ticket]));
            assert!(c.input.is_some(), "input handed back");
        }
        assert!(e.wait_any().is_none());
    }

    #[test]
    fn scoped_threads_borrow_and_panics_become_errors() {
        let offset = 0.5f32; // borrowed by the closure
        let f = |t: &HostTensor, _: Option<&SideInputs>| -> Result<HostTensor, EngineError> {
            assert!(t.w != 7, "boom");
            let mut o = t.clone();
            o.data.iter_mut().for_each(|v| *v += offset);
            Ok(o)
        };
        std::thread::scope(|s| {
            let mut e = ThreadedEngine::scoped(s, 2, &f);
            e.submit(Tensor::Host(Planar::zeros(2, 2)), None)
                .expect("a");
            e.submit(Tensor::Host(Planar::zeros(7, 1)), None)
                .expect("b");
            let mut results: Vec<_> = (0..2)
                .map(|_| e.wait_any().expect("done").output.map(|_| ()))
                .collect();
            results.sort_by_key(Result::is_err);
            assert_eq!(results[0], Ok(()));
            assert_eq!(
                results[1],
                Err(EngineError::Internal("network call panicked".into()))
            );
        });
    }
}
