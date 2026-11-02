//! Process supervision and the zombie reaper.
//!
//! Guarantee: no fs-worker or FFmpeg process outlives the app, even when the
//! app is force-killed. The layers:
//!
//! | Layer | Windows | Linux | macOS |
//! |---|---|---|---|
//! | Tree | app joins a `KILL_ON_JOB_CLOSE` job ([`install_kill_on_close_job`]); each worker gets a nested job ([`TreeHandle`]) | worker leads its own process group | same as Linux |
//! | Parent death | kernel closes the job handle → whole tree killed | `PR_SET_PDEATHSIG(SIGKILL)` + [`watch_parent`] | [`watch_parent`] (no PDEATHSIG) |
//! | Last resort | broken pipes: FFmpeg exits on EPIPE/EOF | same | same |
//! | Next launch | [`Ledger::reap_stale`] kills survivors matched by (pid, start time, exe) | same | same |
#![deny(unsafe_code)]

mod ledger;

use std::io;
use std::process::Command;
use std::time::Duration;

pub use ledger::{Ledger, LedgerEntry};

/// Configure `cmd` so the child leads a new process tree tied to our lifetime.
///
/// Linux: `PR_SET_PDEATHSIG` fires when the *thread* that forked the child
/// exits, not the process. Spawn workers from a long-lived thread (the tokio
/// runtime's blocking pool is fine because its threads outlive jobs, and
/// [`watch_parent`] in the worker is the backstop).
pub fn prepare(cmd: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
        #[cfg(target_os = "linux")]
        {
            let parent = std::process::id();
            #[allow(unsafe_code)]
            // SAFETY: the closure runs between fork and exec, so it only makes
            // async-signal-safe calls (prctl, getppid) and allocates nothing.
            unsafe {
                cmd.pre_exec(move || {
                    nix::sys::prctl::set_pdeathsig(Some(nix::sys::signal::Signal::SIGKILL))
                        .map_err(io::Error::from)?;
                    // The parent may have died before prctl took effect.
                    if nix::unistd::getppid().as_raw() as u32 != parent {
                        return Err(io::Error::other("parent exited during spawn"));
                    }
                    Ok(())
                });
            }
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // No console window flashes for sidecars.
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
}

/// A handle to a child's whole process tree.
#[derive(Debug)]
pub struct TreeHandle {
    pid: u32,
    #[cfg(windows)]
    job: win::Job,
}

impl TreeHandle {
    /// Unix: the child was started with [`prepare`], so its pgid equals its pid.
    #[cfg(unix)]
    pub fn new(pid: u32) -> io::Result<Self> {
        Ok(Self { pid })
    }

    /// Windows: create a nested kill-on-close job and assign the child to it.
    /// Do this **before** the worker receives `Start`, because it only spawns
    /// FFmpeg after that, so every grandchild lands in the job too.
    #[cfg(windows)]
    pub fn new(pid: u32, process: std::os::windows::io::RawHandle) -> io::Result<Self> {
        let job = win::Job::kill_on_close()?;
        job.assign(process)?;
        Ok(Self { pid, job })
    }

    pub fn pid(&self) -> u32 {
        self.pid
    }

    /// Ask the tree to exit (SIGTERM to the group). On Windows there is no
    /// tree-wide graceful signal, so callers send `Stop` over the worker's stdin.
    pub fn terminate(&self) -> io::Result<()> {
        #[cfg(unix)]
        {
            signal_group(self.pid, nix::sys::signal::Signal::SIGTERM)
        }
        #[cfg(windows)]
        {
            Ok(())
        }
    }

    /// Kill the whole tree immediately.
    pub fn kill(&self) -> io::Result<()> {
        #[cfg(unix)]
        {
            signal_group(self.pid, nix::sys::signal::Signal::SIGKILL)
        }
        #[cfg(windows)]
        {
            self.job.terminate()
        }
    }

    /// `terminate`, wait up to `grace` for `exited()` to become true, then `kill`.
    pub fn shutdown(&self, grace: Duration, mut exited: impl FnMut() -> bool) -> io::Result<()> {
        self.terminate()?;
        let deadline = std::time::Instant::now() + grace;
        while std::time::Instant::now() < deadline {
            if exited() {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        self.kill()
    }
}

#[cfg(unix)]
fn signal_group(pgid: u32, sig: nix::sys::signal::Signal) -> io::Result<()> {
    use nix::errno::Errno;
    match nix::sys::signal::killpg(nix::unistd::Pid::from_raw(pgid as i32), sig) {
        Ok(()) | Err(Errno::ESRCH) => Ok(()), // already gone
        Err(e) => Err(e.into()),
    }
}

/// Windows: place the *current* process in a kill-on-close job. Every process
/// we spawn afterwards (workers, FFmpeg, WebView2 helpers) inherits it, and
/// when the app dies for any reason the kernel closes the last handle and
/// kills the whole tree. Other platforms: no-op.
pub fn install_kill_on_close_job() -> io::Result<()> {
    #[cfg(windows)]
    {
        let job = win::Job::kill_on_close()?;
        job.assign_current()?;
        // Intentionally leaked: the handle must live exactly as long as we do.
        std::mem::forget(job);
    }
    Ok(())
}

/// In a child process: call `on_parent_exit` when our parent dies.
/// Detects reparenting (`getppid()` changes) with a 250 ms poll, which is
/// portable to macOS where `PR_SET_PDEATHSIG` doesn't exist.
pub fn watch_parent(on_parent_exit: impl FnOnce() + Send + 'static) {
    #[cfg(unix)]
    {
        let original = nix::unistd::getppid();
        std::thread::Builder::new()
            .name("parent-watchdog".into())
            .spawn(move || {
                loop {
                    std::thread::sleep(Duration::from_millis(250));
                    if nix::unistd::getppid() != original {
                        on_parent_exit();
                        return;
                    }
                }
            })
            .ok();
    }
    #[cfg(windows)]
    {
        // Job objects already guarantee this on Windows.
        let _ = on_parent_exit;
    }
}

/// In a worker: kill our own process group (our FFmpeg children). Used by
/// the parent watchdog right before exiting.
pub fn kill_own_group() {
    #[cfg(unix)]
    {
        let _ = signal_group(std::process::id(), nix::sys::signal::Signal::SIGKILL);
    }
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod win {
    use std::io;
    use std::os::windows::io::RawHandle;

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
        SetInformationJobObject, TerminateJobObject,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    #[derive(Debug)]
    pub struct Job(HANDLE);

    // SAFETY: a job handle may be used from any thread.
    unsafe impl Send for Job {}
    unsafe impl Sync for Job {}

    impl Job {
        pub fn kill_on_close() -> io::Result<Self> {
            // SAFETY: null security attributes and name are allowed.
            let h = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            if h.is_null() {
                return Err(io::Error::last_os_error());
            }
            let job = Self(h);
            // SAFETY: zeroed is a valid initial state for this POD struct.
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            // SAFETY: valid job handle; pointer/size describe `info`.
            let ok = unsafe {
                SetInformationJobObject(
                    job.0,
                    JobObjectExtendedLimitInformation,
                    (&raw const info).cast(),
                    size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
            };
            if ok == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(job)
        }

        pub fn assign(&self, process: RawHandle) -> io::Result<()> {
            // SAFETY: both handles are valid; nested jobs are supported on Windows 8+.
            if unsafe { AssignProcessToJobObject(self.0, process as HANDLE) } == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        }

        pub fn assign_current(&self) -> io::Result<()> {
            // SAFETY: pseudo-handle of the current process.
            self.assign(unsafe { GetCurrentProcess() } as RawHandle)
        }

        pub fn terminate(&self) -> io::Result<()> {
            // SAFETY: valid job handle.
            if unsafe { TerminateJobObject(self.0, 1) } == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            // SAFETY: we own the handle. Closing it kills the tree (KILL_ON_JOB_CLOSE).
            unsafe { CloseHandle(self.0) };
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::process::Stdio;

    #[test]
    fn group_kill_takes_down_grandchildren() {
        // sh spawns a grandchild `sleep`, then waits. Killing the group must
        // kill both, otherwise the grandchild would be the classic zombie FFmpeg.
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "sleep 30 & echo $!; wait"])
            .stdout(Stdio::piped());
        prepare(&mut cmd);
        let mut child = cmd.spawn().expect("spawn");
        let mut line = String::new();
        std::io::BufRead::read_line(
            &mut std::io::BufReader::new(child.stdout.take().expect("stdout")),
            &mut line,
        )
        .expect("read");
        let grandchild: i32 = line.trim().parse().expect("pid");

        let tree = TreeHandle::new(child.id()).expect("tree");
        tree.shutdown(Duration::from_millis(500), || {
            matches!(child.try_wait(), Ok(Some(_)))
        })
        .expect("shutdown");
        let _ = child.wait();
        std::thread::sleep(Duration::from_millis(100));
        let alive = nix::sys::signal::kill(nix::unistd::Pid::from_raw(grandchild), None).is_ok();
        assert!(!alive, "grandchild {grandchild} survived the group kill");
    }
}
