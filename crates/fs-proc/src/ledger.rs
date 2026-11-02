//! PID ledger: a record of every worker we spawned, so the next launch can
//! kill survivors of a crash.
//!
//! PIDs get recycled, so an entry only matches a running process when the pid,
//! the process start time *and* the executable name all agree. We never kill
//! an unrelated process that happens to have reused the pid.

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub pid: u32,
    /// Process start time (seconds since epoch, as reported by the OS).
    pub start_time: u64,
    /// Executable file name (not the full path: bundles move between installs).
    pub exe: String,
}

#[derive(Debug)]
pub struct Ledger {
    path: PathBuf,
}

fn lookup(pid: u32) -> Option<(System, Pid)> {
    let mut sys = System::new().ok()?;
    let p = Pid::from_u32(pid);
    sys.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[p]),
        true,
        ProcessRefreshKind::nothing().with_exe(UpdateKind::Always),
    );
    sys.process(p)?;
    Some((sys, p))
}

fn exe_name(path: Option<&Path>) -> String {
    path.and_then(Path::file_name)
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

impl Ledger {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    fn read(&self) -> Vec<LedgerEntry> {
        let Ok(f) = std::fs::File::open(&self.path) else {
            return Vec::new();
        };
        io::BufReader::new(f)
            .lines()
            .map_while(Result::ok)
            .filter_map(|l| serde_json::from_str(&l).ok())
            .collect()
    }

    fn write_all(&self, entries: &[LedgerEntry]) -> io::Result<()> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = self.path.with_extension("tmp");
        {
            let mut f = io::BufWriter::new(std::fs::File::create(&tmp)?);
            for e in entries {
                writeln!(f, "{}", serde_json::to_string(e).map_err(io::Error::other)?)?;
            }
            f.flush()?;
            f.get_ref().sync_all()?;
        }
        std::fs::rename(tmp, &self.path)
    }

    /// Record a freshly spawned process.
    pub fn record(&self, pid: u32) -> io::Result<LedgerEntry> {
        let (sys, p) = lookup(pid).ok_or_else(|| io::Error::other("process vanished"))?;
        let proc_ = sys
            .process(p)
            .ok_or_else(|| io::Error::other("process vanished"))?;
        let entry = LedgerEntry {
            pid,
            start_time: proc_.start_time(),
            exe: exe_name(proc_.exe()),
        };
        let mut all = self.read();
        all.retain(|e| e.pid != pid);
        all.push(entry.clone());
        self.write_all(&all)?;
        Ok(entry)
    }

    /// Forget a process that exited normally.
    pub fn remove(&self, pid: u32) -> io::Result<()> {
        let mut all = self.read();
        let before = all.len();
        all.retain(|e| e.pid != pid);
        if all.len() != before {
            self.write_all(&all)
        } else {
            Ok(())
        }
    }

    /// On startup: kill every recorded process that is *provably* ours and
    /// still running, then clear the ledger. Returns the killed pids.
    pub fn reap_stale(&self) -> io::Result<Vec<u32>> {
        let mut killed = Vec::new();
        for e in self.read() {
            let Some((sys, p)) = lookup(e.pid) else {
                continue;
            };
            let Some(proc_) = sys.process(p) else {
                continue;
            };
            if proc_.start_time() == e.start_time && exe_name(proc_.exe()) == e.exe {
                tracing::warn!(pid = e.pid, exe = %e.exe, "reaping orphaned process from a previous run");
                // Workers lead their process group: take FFmpeg children down too.
                #[cfg(unix)]
                let _ = nix::sys::signal::killpg(
                    nix::unistd::Pid::from_raw(e.pid as i32),
                    nix::sys::signal::Signal::SIGKILL,
                );
                proc_.kill();
                killed.push(e.pid);
            }
        }
        self.write_all(&[])?;
        Ok(killed)
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn reaps_only_matching_processes() {
        let dir = std::env::temp_dir().join(format!("fs-proc-ledger-{}", std::process::id()));
        let ledger = Ledger::new(dir.join("workers.jsonl"));

        let mut cmd = Command::new("sleep");
        cmd.arg("30");
        crate::prepare(&mut cmd);
        let mut orphan = cmd.spawn().expect("spawn");
        let e = ledger.record(orphan.id()).expect("record");
        assert_eq!(e.exe, "sleep");

        // A recycled-pid impostor: same pid, wrong start time. Must be left alone.
        let me = std::process::id();
        let fake = LedgerEntry {
            pid: me,
            start_time: 1,
            exe: "sleep".into(),
        };
        let mut all = ledger.read();
        all.push(fake);
        ledger.write_all(&all).expect("write");

        let killed = ledger.reap_stale().expect("reap");
        assert_eq!(killed, vec![orphan.id()]);
        assert!(
            orphan.wait().expect("wait").code().is_none(),
            "killed by signal"
        );
        assert!(ledger.read().is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }
}
