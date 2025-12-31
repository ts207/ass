PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  title TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('system','user','assistant')),
  content TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(conversation_id) REFERENCES conversations(id)
);

CREATE INDEX IF NOT EXISTS idx_messages_convo_time
ON messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS reminders (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  title TEXT NOT NULL,
  due_at TEXT NOT NULL,
  due_at_utc TEXT,
  rrule TEXT,
  notes TEXT,
  status TEXT NOT NULL DEFAULT 'scheduled',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  fired_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_reminders_user_due
ON reminders(user_id, due_at);

CREATE TABLE IF NOT EXISTS code_progress (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  topic TEXT NOT NULL,
  notes TEXT,
  evidence_path TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_code_user_time
ON code_progress(user_id, created_at);
CREATE TABLE IF NOT EXISTS agent_sessions (
  user_id TEXT NOT NULL,
  agent_name TEXT NOT NULL CHECK(agent_name IN ('life','ds')),
  conversation_id TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (user_id, agent_name)
);

CREATE TABLE IF NOT EXISTS courses (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  course_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS course_progress (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  course_id TEXT NOT NULL,
  lesson_key TEXT NOT NULL,
  status TEXT NOT NULL,
  score REAL,
  attempts INTEGER DEFAULT 0,
  updated_at TEXT NOT NULL,
  UNIQUE(user_id, course_id, lesson_key)
);

CREATE INDEX IF NOT EXISTS idx_course_progress_user_course
ON course_progress(user_id, course_id);

CREATE TABLE IF NOT EXISTS ds_progress (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  topic TEXT NOT NULL,
  score REAL,
  notes TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ds_progress_user_time
ON ds_progress(user_id, created_at);

CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  name TEXT NOT NULL,
  path TEXT NOT NULL,
  schema_json TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  goal TEXT NOT NULL,
  approach_json TEXT,
  metrics_json TEXT,
  artifact_path TEXT,
  created_at TEXT NOT NULL
);
