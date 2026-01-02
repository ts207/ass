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
  agent_name TEXT NOT NULL CHECK(agent_name IN ('life','ds','health','code')),
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
-- =========================
-- ChatGPT Export (Graph) Memory
-- Lossless storage of ChatGPT conversations.json + indexes for retrieval
-- =========================

-- 1) Conversation metadata
CREATE TABLE IF NOT EXISTS chatgpt_conversations (
  id TEXT PRIMARY KEY,
  title TEXT,
  create_time INTEGER,
  update_time INTEGER
);

-- 2) Conversation graph nodes
CREATE TABLE IF NOT EXISTS chatgpt_nodes (
  node_id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  parent_id TEXT,
  role TEXT,                     -- 'user'/'assistant'/'system'/etc
  text TEXT,                     -- joined parts
  create_time INTEGER,
  is_message INTEGER NOT NULL DEFAULT 1, -- 0 for structural/null nodes
  main_child_id TEXT,            -- chosen for mainline traversal (computed in importer)
  agent TEXT,                    -- 'life'/'ds'/'general' (optional)
  FOREIGN KEY(conversation_id) REFERENCES chatgpt_conversations(id)
);

CREATE INDEX IF NOT EXISTS idx_chatgpt_nodes_conv
  ON chatgpt_nodes(conversation_id);

CREATE INDEX IF NOT EXISTS idx_chatgpt_nodes_parent
  ON chatgpt_nodes(parent_id);

CREATE INDEX IF NOT EXISTS idx_chatgpt_nodes_agent
  ON chatgpt_nodes(agent);

-- 3) Full-text search over node text (fast candidate generation)
-- Note: FTS5 is a separate virtual table; keep it simple.
CREATE VIRTUAL TABLE IF NOT EXISTS chatgpt_nodes_fts
USING fts5(
  node_id,
  conversation_id,
  title,
  agent,
  text
);

-- 4) Embeddings for reranking (store as BLOB; cosine computed in Python)
CREATE TABLE IF NOT EXISTS chatgpt_node_embeddings (
  node_id TEXT PRIMARY KEY,
  dim INTEGER NOT NULL,
  vec BLOB NOT NULL,
  FOREIGN KEY(node_id) REFERENCES chatgpt_nodes(node_id)
);

-- Optional: keep embeddings consistent with nodes
CREATE TRIGGER IF NOT EXISTS trg_chatgpt_nodes_delete_embedding
AFTER DELETE ON chatgpt_nodes
BEGIN
  DELETE FROM chatgpt_node_embeddings WHERE node_id = OLD.node_id;
  DELETE FROM chatgpt_nodes_fts WHERE node_id = OLD.node_id;
END;

-- =========================
-- User Profile (Stable Memory)
-- Key facts/preferences/goals explicitly saved by the user.
-- =========================
CREATE TABLE IF NOT EXISTS user_profiles (
  user_id TEXT PRIMARY KEY,
  profile_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
