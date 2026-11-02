-- Journal schema v1. Applied in one transaction; WAL + synchronous=FULL set at open.

CREATE TABLE jobs (
  id             TEXT PRIMARY KEY,          -- UUIDv7 (time-sortable)
  spec_json      TEXT    NOT NULL,          -- JobSpec as submitted
  plan_json      TEXT,                      -- compiled RenderPlan (set after ANALYZE)
  plan_hash      TEXT,                      -- resume is refused if the plan changed
  state          TEXT    NOT NULL CHECK (state IN
                   ('queued','analyzing','rendering','paused','interrupted',
                    'concatenating','muxing','done','failed','cancelled')),
  total_frames   INTEGER,                   -- conformed input frames
  segment_frames INTEGER,                   -- 4 s × conform rate
  queue_pos      INTEGER NOT NULL,
  error          TEXT,
  created_at     INTEGER NOT NULL,
  updated_at     INTEGER NOT NULL
);

CREATE TABLE segments (
  job_id        TEXT    NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  idx           INTEGER NOT NULL,
  first_frame   INTEGER NOT NULL,           -- input-space frame index
  frame_count   INTEGER NOT NULL,
  out_frames    INTEGER NOT NULL,
  in_hash       TEXT    NOT NULL,           -- xxh3 of this segment's first input frame (hex)
  next_hash     TEXT,                       -- xxh3 of the next segment's first input frame (hex)
  file_name     TEXT    NOT NULL,
  bytes         INTEGER NOT NULL,
  device_uuid   TEXT,
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

PRAGMA user_version = 1;
