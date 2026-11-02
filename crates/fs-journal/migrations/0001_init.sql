-- Journal schema v1. Applied in a single transaction; WAL mode set at open.
PRAGMA user_version = 1;

CREATE TABLE jobs (
  id            TEXT PRIMARY KEY,          -- UUIDv7 (time-sortable)
  source_path   TEXT    NOT NULL,
  output_path   TEXT    NOT NULL,
  plan_json     TEXT    NOT NULL,          -- compiled PipelinePlan (immutable once rendering starts)
  plan_hash     TEXT    NOT NULL,          -- resume is refused if the plan changed
  state         TEXT    NOT NULL CHECK (state IN
                  ('queued','analyzing','rendering','paused','interrupted',
                   'concatenating','muxing','done','failed','cancelled')),
  total_frames  INTEGER,                   -- in CFR output space
  segment_frames INTEGER NOT NULL,         -- 4 s × conform rate (decision: 4 s segments)
  queue_pos     INTEGER NOT NULL,
  error         TEXT,
  created_at    INTEGER NOT NULL,
  updated_at    INTEGER NOT NULL
);

CREATE TABLE segments (
  job_id        TEXT    NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  idx           INTEGER NOT NULL,
  first_frame   INTEGER NOT NULL,          -- input-space frame index
  frame_count   INTEGER NOT NULL,
  in_hash       TEXT    NOT NULL,          -- xxh3 of downsampled first input frame (resume alignment)
  file_name     TEXT    NOT NULL,
  bytes         INTEGER NOT NULL,
  device_uuid   TEXT    NOT NULL,
  committed_at  INTEGER NOT NULL,
  PRIMARY KEY (job_id, idx)
);

CREATE TABLE job_events (
  job_id  TEXT    NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  at      INTEGER NOT NULL,
  level   TEXT    NOT NULL,
  kind    TEXT    NOT NULL,
  detail  TEXT
);

CREATE INDEX jobs_state ON jobs(state);
CREATE INDEX job_events_job ON job_events(job_id, at);
