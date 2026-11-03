//! Crash-consistent job and segment journal (SQLite, WAL, `synchronous=FULL`).
//!
//! The journal is the source of truth for resume. A segment exists only after
//! (1) the worker fsync'd and renamed its file and (2) [`Journal::commit_segment`]
//! returned. A crash between the two leaves an unjournaled file, which simply
//! gets re-rendered and overwritten, so the protocol is idempotent.
#![forbid(unsafe_code)]

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use fs_types::worker::{RenderPlan, SegmentInfo};
use fs_types::{DeviceUuid, JobId, JobSpec, JobState, JobSummary};
use rusqlite::{Connection, OptionalExtension, params};

const MIGRATION_1: &str = include_str!("../migrations/0001_init.sql");
const SCHEMA_VERSION: i64 = 1;

#[derive(Debug, thiserror::Error)]
pub enum JournalError {
    #[error(transparent)]
    Sql(#[from] rusqlite::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("journal schema v{0} is newer than this app (v{SCHEMA_VERSION}); refusing to touch it")]
    TooNew(i64),
    #[error("corrupt journal row: {0}")]
    Corrupt(String),
    #[error("job {0} not found")]
    NotFound(JobId),
}

pub type Result<T> = std::result::Result<T, JournalError>;

/// Where to restart a job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumePoint {
    pub segment: u32,
    pub first_frame: u64,
    pub expected_hash: Option<u64>,
}

#[derive(Debug)]
pub struct Journal {
    conn: Connection,
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64)
}

fn hex(x: u64) -> String {
    format!("{x:016x}")
}

fn unhex(s: &str) -> Result<u64> {
    u64::from_str_radix(s, 16).map_err(|_| JournalError::Corrupt(format!("bad hash {s:?}")))
}

/// Stable hash of a plan's JSON (resume refuses if the plan changed).
pub fn plan_hash(plan: &RenderPlan) -> Result<String> {
    Ok(hex(xxhash_rust::xxh3::xxh3_64(
        serde_json::to_string(plan)?.as_bytes(),
    )))
}

impl Journal {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| JournalError::Corrupt(e.to_string()))?;
        }
        Self::init(Connection::open(path)?)
    }

    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "FULL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        let mut j = Self { conn };
        j.migrate()?;
        Ok(j)
    }

    fn migrate(&mut self) -> Result<()> {
        let v: i64 = self
            .conn
            .pragma_query_value(None, "user_version", |r| r.get(0))?;
        if v > SCHEMA_VERSION {
            return Err(JournalError::TooNew(v));
        }
        if v < 1 {
            let tx = self.conn.transaction()?;
            tx.execute_batch(MIGRATION_1)?;
            tx.commit()?;
        }
        Ok(())
    }

    // --- jobs ----------------------------------------------------------------

    pub fn insert_job(&self, id: JobId, spec: &JobSpec) -> Result<()> {
        let pos: i64 = self.conn.query_row(
            "SELECT COALESCE(MAX(queue_pos), -1) + 1 FROM jobs",
            [],
            |r| r.get(0),
        )?;
        let t = now_ms();
        self.conn.execute(
            "INSERT INTO jobs (id, spec_json, state, queue_pos, created_at, updated_at)
             VALUES (?1, ?2, 'queued', ?3, ?4, ?4)",
            params![id.to_string(), serde_json::to_string(spec)?, pos, t],
        )?;
        Ok(())
    }

    pub fn spec(&self, id: JobId) -> Result<JobSpec> {
        let json: String = self
            .conn
            .query_row(
                "SELECT spec_json FROM jobs WHERE id = ?1",
                [id.to_string()],
                |r| r.get(0),
            )
            .optional()?
            .ok_or(JournalError::NotFound(id))?;
        Ok(serde_json::from_str(&json)?)
    }

    pub fn set_plan(&self, id: JobId, plan: &RenderPlan) -> Result<()> {
        let n = self.conn.execute(
            "UPDATE jobs SET plan_json = ?2, plan_hash = ?3, total_frames = ?4, segment_frames = ?5,
                    updated_at = ?6 WHERE id = ?1",
            params![
                id.to_string(),
                serde_json::to_string(plan)?,
                plan_hash(plan)?,
                plan.total_frames as i64,
                plan.segment_frames as i64,
                now_ms()
            ],
        )?;
        if n == 0 {
            Err(JournalError::NotFound(id))
        } else {
            Ok(())
        }
    }

    pub fn plan(&self, id: JobId) -> Result<Option<RenderPlan>> {
        let json: Option<String> = self
            .conn
            .query_row(
                "SELECT plan_json FROM jobs WHERE id = ?1",
                [id.to_string()],
                |r| r.get(0),
            )
            .optional()?
            .flatten();
        Ok(json.map(|j| serde_json::from_str(&j)).transpose()?)
    }

    pub fn set_state(&self, id: JobId, state: JobState, error: Option<&str>) -> Result<()> {
        let n = self.conn.execute(
            "UPDATE jobs SET state = ?2, error = ?3, updated_at = ?4 WHERE id = ?1",
            params![id.to_string(), state.as_str(), error, now_ms()],
        )?;
        if n == 0 {
            Err(JournalError::NotFound(id))
        } else {
            Ok(())
        }
    }

    pub fn state(&self, id: JobId) -> Result<JobState> {
        let s: String = self
            .conn
            .query_row(
                "SELECT state FROM jobs WHERE id = ?1",
                [id.to_string()],
                |r| r.get(0),
            )
            .optional()?
            .ok_or(JournalError::NotFound(id))?;
        JobState::parse(&s).ok_or_else(|| JournalError::Corrupt(format!("state {s:?}")))
    }

    /// At startup: anything that was running when we died is `interrupted`.
    pub fn mark_interrupted(&self) -> Result<Vec<JobId>> {
        let mut stmt = self.conn.prepare(
            "UPDATE jobs SET state = 'interrupted', updated_at = ?1
             WHERE state IN ('analyzing','rendering','concatenating','muxing') RETURNING id",
        )?;
        let ids = stmt
            .query_map([now_ms()], |r| r.get::<_, String>(0))?
            .filter_map(|r| r.ok()?.parse().ok())
            .collect();
        Ok(ids)
    }

    pub fn list_jobs(&self) -> Result<Vec<JobSummary>> {
        let mut stmt = self.conn.prepare(
            "SELECT j.id, j.spec_json, j.state, j.queue_pos, j.total_frames, j.error,
                    COALESCE((SELECT SUM(frame_count) FROM segments s WHERE s.job_id = j.id), 0)
             FROM jobs j ORDER BY j.queue_pos",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, Option<i64>>(4)?,
                r.get::<_, Option<String>>(5)?,
                r.get::<_, i64>(6)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (id, spec, state, pos, total, error, done) = row?;
            out.push(JobSummary {
                id: id
                    .parse()
                    .map_err(|_| JournalError::Corrupt(format!("job id {id:?}")))?,
                spec: serde_json::from_str(&spec)?,
                state: JobState::parse(&state)
                    .ok_or_else(|| JournalError::Corrupt(state.clone()))?,
                queue_pos: pos as u32,
                frames_done: done as u64,
                frames_total: total.map(|t| t as u64),
                error,
            });
        }
        Ok(out)
    }

    /// Persist a new queue order (Pro: batch reordering).
    pub fn reorder(&mut self, order: &[JobId]) -> Result<()> {
        let tx = self.conn.transaction()?;
        for (pos, id) in order.iter().enumerate() {
            tx.execute(
                "UPDATE jobs SET queue_pos = ?2 WHERE id = ?1",
                params![id.to_string(), pos as i64],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn delete_job(&self, id: JobId) -> Result<()> {
        self.conn
            .execute("DELETE FROM jobs WHERE id = ?1", [id.to_string()])?;
        Ok(())
    }

    // --- segments --------------------------------------------------------------

    pub fn commit_segment(
        &mut self,
        id: JobId,
        seg: &SegmentInfo,
        device: Option<DeviceUuid>,
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT OR REPLACE INTO segments
               (job_id, idx, first_frame, frame_count, out_frames, in_hash, next_hash,
                file_name, bytes, device_uuid, committed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                id.to_string(),
                seg.index,
                seg.first_frame as i64,
                seg.frame_count as i64,
                seg.out_frames as i64,
                hex(seg.in_hash),
                seg.next_hash.map(hex),
                seg.file_name,
                seg.bytes as i64,
                device.map(|d| d.to_string()),
                now_ms()
            ],
        )?;
        tx.execute(
            "UPDATE jobs SET updated_at = ?2 WHERE id = ?1",
            params![id.to_string(), now_ms()],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn segments(&self, id: JobId) -> Result<Vec<SegmentInfo>> {
        let mut stmt = self.conn.prepare(
            "SELECT idx, first_frame, frame_count, out_frames, in_hash, next_hash, file_name, bytes
             FROM segments WHERE job_id = ?1 ORDER BY idx",
        )?;
        let rows = stmt.query_map([id.to_string()], |r| {
            Ok((
                r.get::<_, u32>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, Option<String>>(5)?,
                r.get::<_, String>(6)?,
                r.get::<_, i64>(7)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (index, first, count, outf, inh, next, file_name, bytes) = row?;
            out.push(SegmentInfo {
                index,
                first_frame: first as u64,
                frame_count: count as u64,
                out_frames: outf as u64,
                in_hash: unhex(&inh)?,
                next_hash: next.as_deref().map(unhex).transpose()?,
                file_name,
                bytes: bytes as u64,
            });
        }
        Ok(out)
    }

    /// Where to resume: the first gap in the contiguous run of committed
    /// segments starting at 0.
    pub fn resume_point(&self, id: JobId, segment_frames: u64) -> Result<ResumePoint> {
        let segs = self.segments(id)?;
        let mut next = 0u32;
        let mut expected = None;
        for s in &segs {
            if s.index != next {
                break;
            }
            next += 1;
            expected = s.next_hash;
        }
        Ok(ResumePoint {
            segment: next,
            first_frame: u64::from(next) * segment_frames,
            expected_hash: if next == 0 { None } else { expected },
        })
    }

    /// Drop segments at or after `from` (e.g. after a plan change).
    pub fn truncate_segments(&self, id: JobId, from: u32) -> Result<()> {
        self.conn.execute(
            "DELETE FROM segments WHERE job_id = ?1 AND idx >= ?2",
            params![id.to_string(), from],
        )?;
        Ok(())
    }

    pub fn log_event(&self, id: JobId, level: &str, kind: &str, detail: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO job_events (job_id, at, level, kind, detail) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id.to_string(), now_ms(), level, kind, detail],
        )?;
        Ok(())
    }

    /// Job ids whose temp directories must survive the janitor.
    pub fn resumable_jobs(&self) -> Result<Vec<JobId>> {
        let mut stmt = self.conn.prepare(
            "SELECT id FROM jobs WHERE state IN ('queued','analyzing','rendering','paused','interrupted','concatenating','muxing')",
        )?;
        let ids = stmt
            .query_map([], |r| r.get::<_, String>(0))?
            .filter_map(|r| r.ok()?.parse().ok())
            .collect();
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs_types::pipeline::*;

    fn spec() -> JobSpec {
        JobSpec {
            input: "/in.mov".into(),
            pipeline: PipelineSpec {
                stages: vec![StageConfig::Upscale {
                    model: ModelRef {
                        id: "x".into(),
                        path: None,
                    },
                    scale: 2,
                    denoise: None,
                }],
            },
            output: OutputSpec {
                path: "/out.mkv".into(),
                codec: VideoCodec::Av1,
                container: Container::Mkv,
                quality: 20,
                encoder: EncoderPreference::Auto,
            },
            device: None,
        }
    }

    fn seg(i: u32, next: Option<u64>) -> SegmentInfo {
        SegmentInfo {
            index: i,
            first_frame: u64::from(i) * 96,
            frame_count: 96,
            out_frames: 192,
            in_hash: 0xdead_0000 + u64::from(i),
            next_hash: next,
            file_name: format!("seg_{i:05}.mkv"),
            bytes: 1000,
        }
    }

    #[test]
    fn lifecycle_resume_and_interrupt() {
        let mut j = Journal::open_in_memory().expect("open");
        let id = JobId::new();
        j.insert_job(id, &spec()).expect("insert");
        assert_eq!(j.state(id).expect("state"), JobState::Queued);
        assert_eq!(
            j.resume_point(id, 96).expect("rp"),
            ResumePoint {
                segment: 0,
                first_frame: 0,
                expected_hash: None
            }
        );

        j.set_state(id, JobState::Rendering, None).expect("set");
        j.commit_segment(id, &seg(0, Some(0xaaaa)), None)
            .expect("c0");
        j.commit_segment(id, &seg(1, Some(0xbbbb)), None)
            .expect("c1");
        // Segment 3 committed out of order (e.g. second GPU) but 2 missing → resume at 2.
        j.commit_segment(id, &seg(3, Some(0xdddd)), None)
            .expect("c3");
        let rp = j.resume_point(id, 96).expect("rp");
        assert_eq!(
            rp,
            ResumePoint {
                segment: 2,
                first_frame: 192,
                expected_hash: Some(0xbbbb)
            }
        );

        // Re-committing is idempotent (crash between rename and journal).
        j.commit_segment(id, &seg(1, Some(0xbbbb)), None)
            .expect("again");
        assert_eq!(j.segments(id).expect("segs").len(), 3);

        // App crash → interrupted on next start.
        let interrupted = j.mark_interrupted().expect("mark");
        assert_eq!(interrupted, vec![id]);
        assert_eq!(j.state(id).expect("state"), JobState::Interrupted);
        assert_eq!(j.resumable_jobs().expect("r"), vec![id]);

        let rows = j.list_jobs().expect("list");
        assert_eq!(rows[0].frames_done, 288);
    }

    #[test]
    fn reorder_and_delete_cascade() {
        let mut j = Journal::open_in_memory().expect("open");
        let (a, b) = (JobId::new(), JobId::new());
        j.insert_job(a, &spec()).expect("a");
        j.insert_job(b, &spec()).expect("b");
        j.reorder(&[b, a]).expect("reorder");
        let ids: Vec<_> = j
            .list_jobs()
            .expect("list")
            .into_iter()
            .map(|r| r.id)
            .collect();
        assert_eq!(ids, vec![b, a]);
        j.commit_segment(a, &seg(0, None), None).expect("seg");
        j.delete_job(a).expect("del");
        assert!(j.segments(a).expect("segs").is_empty(), "segments cascade");
    }

    #[test]
    fn refuses_newer_schema() {
        let conn = Connection::open_in_memory().expect("mem");
        conn.pragma_update(None, "user_version", 99)
            .expect("pragma");
        assert!(matches!(Journal::init(conn), Err(JournalError::TooNew(99))));
    }
}
