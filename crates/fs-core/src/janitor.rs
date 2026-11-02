//! Temp janitor: on startup, delete abandoned job directories. Never deletes
//! (a) a directory whose `.lock` is held by a live runner, or (b) one that
//! belongs to a resumable job, because those segments are the crash-recovery data.

use std::collections::HashSet;
use std::fs::File;
use std::path::Path;
use std::time::{Duration, SystemTime};

use fs_types::JobId;

pub const LOCK_FILE: &str = ".lock";
pub const MIN_AGE: Duration = Duration::from_secs(24 * 3600);

/// Hold this for the lifetime of a running job.
#[derive(Debug)]
pub struct DirLock(#[allow(dead_code)] File);

pub fn lock_dir(dir: &Path) -> std::io::Result<DirLock> {
    std::fs::create_dir_all(dir)?;
    let f = File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open(dir.join(LOCK_FILE))?;
    f.try_lock()
        .map_err(|e| std::io::Error::other(format!("job directory in use: {e}")))?;
    Ok(DirLock(f))
}

fn is_locked(dir: &Path) -> bool {
    let Ok(f) = File::options().write(true).open(dir.join(LOCK_FILE)) else {
        return false;
    };
    f.try_lock().is_err()
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub deleted: Vec<String>,
    pub kept_resumable: usize,
    pub kept_locked: usize,
    pub kept_young: usize,
    pub freed_bytes: u64,
}

fn dir_size(p: &Path) -> u64 {
    std::fs::read_dir(p)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| {
            let path = e.path();
            if path.is_dir() {
                dir_size(&path)
            } else {
                e.metadata().map_or(0, |m| m.len())
            }
        })
        .sum()
}

pub fn sweep(
    root: &Path,
    resumable: &HashSet<JobId>,
    now: SystemTime,
    min_age: Duration,
) -> Report {
    let mut r = Report::default();
    let Ok(entries) = std::fs::read_dir(root) else {
        return r;
    };
    for e in entries.flatten() {
        let path = e.path();
        if !path.is_dir() {
            continue;
        }
        let name = e.file_name().to_string_lossy().into_owned();
        if name
            .parse::<JobId>()
            .is_ok_and(|id| resumable.contains(&id))
        {
            r.kept_resumable += 1;
            continue;
        }
        if is_locked(&path) {
            r.kept_locked += 1;
            continue;
        }
        let age = e
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|m| now.duration_since(m).ok());
        if age.is_none_or(|a| a < min_age) {
            r.kept_young += 1;
            continue;
        }
        let size = dir_size(&path);
        if std::fs::remove_dir_all(&path).is_ok() {
            r.freed_bytes += size;
            r.deleted.push(name);
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_resumable_locked_and_young_dirs() {
        let root = std::env::temp_dir().join(format!("fs-janitor-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (resumable, locked, abandoned) = (JobId::new(), JobId::new(), JobId::new());
        for id in [resumable, locked, abandoned] {
            std::fs::create_dir_all(root.join(id.to_string())).expect("mk");
            std::fs::write(root.join(id.to_string()).join("seg_00000.mkv"), [0u8; 100]).expect("w");
        }
        let _held = lock_dir(&root.join(locked.to_string())).expect("lock");
        let resumable_set: HashSet<_> = [resumable].into();

        // Everything is "young" right now.
        let young = sweep(&root, &resumable_set, SystemTime::now(), MIN_AGE);
        assert_eq!(young.kept_young, 1);
        assert!(young.deleted.is_empty());

        // Pretend a day passed.
        let later = SystemTime::now() + Duration::from_secs(2 * 24 * 3600);
        let r = sweep(&root, &resumable_set, later, MIN_AGE);
        assert_eq!(r.deleted, vec![abandoned.to_string()]);
        assert_eq!((r.kept_resumable, r.kept_locked), (1, 1));
        assert!(r.freed_bytes >= 100);
        assert!(root.join(resumable.to_string()).exists());
        let _ = std::fs::remove_dir_all(root);
    }
}
