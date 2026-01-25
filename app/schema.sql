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

CREATE TABLE IF NOT EXISTS conversation_summaries (
  conversation_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  agent TEXT NOT NULL,
  summary TEXT NOT NULL,
  message_count INTEGER NOT NULL DEFAULT 0,
  last_message_id TEXT,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(conversation_id) REFERENCES conversations(id)
);

CREATE INDEX IF NOT EXISTS idx_summaries_user_agent
ON conversation_summaries(user_id, agent);

CREATE TABLE IF NOT EXISTS reminders (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  title TEXT NOT NULL,
  due_at TEXT NOT NULL,
  due_at_utc TEXT,
  rrule TEXT,
  channels_json TEXT,
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
  agent_name TEXT NOT NULL CHECK(agent_name IN ('life','ds','health','code','general')),
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

-- =========================
-- Permissions / Consent Gates
-- =========================
CREATE TABLE IF NOT EXISTS user_permissions (
  user_id TEXT PRIMARY KEY,
  mode TEXT NOT NULL CHECK(mode IN ('read','write')),
  allow_network INTEGER NOT NULL DEFAULT 0,
  allow_fs_read INTEGER NOT NULL DEFAULT 0,
  allow_fs_write INTEGER NOT NULL DEFAULT 0,
  allow_shell INTEGER NOT NULL DEFAULT 0,
  allow_exec INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL
);

-- =========================
-- Audit Log
-- =========================
CREATE TABLE IF NOT EXISTS audit_log (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  tool TEXT NOT NULL,
  payload_json TEXT,
  result_json TEXT,
  status TEXT NOT NULL,
  error TEXT,
  duration_ms INTEGER,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_user_time
ON audit_log(user_id, created_at);

CREATE INDEX IF NOT EXISTS idx_audit_tool_time
ON audit_log(tool, created_at);

-- =========================
-- Turn Memory Usage (last-turn debug + UI)
-- =========================
CREATE TABLE IF NOT EXISTS turn_memory_usage (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  agent TEXT NOT NULL,
  node_id TEXT,
  rank INTEGER,
  score REAL,
  snippet TEXT,
  meta_json TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(conversation_id) REFERENCES conversations(id)
);

CREATE INDEX IF NOT EXISTS idx_turn_memory_usage_convo_time
ON turn_memory_usage(conversation_id, created_at);

CREATE INDEX IF NOT EXISTS idx_turn_memory_usage_turn
ON turn_memory_usage(conversation_id, turn_id);

-- =========================
-- Turn Tool Usage (last-turn debug + audit)
-- =========================
CREATE TABLE IF NOT EXISTS turn_tool_usage (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  agent TEXT NOT NULL,
  tool_name TEXT,
  input_json TEXT,
  output_json TEXT,
  status TEXT NOT NULL,
  error TEXT,
  duration_ms INTEGER,
  created_at TEXT NOT NULL,
  FOREIGN KEY(conversation_id) REFERENCES conversations(id)
);

CREATE INDEX IF NOT EXISTS idx_turn_tool_usage_convo_time
ON turn_tool_usage(conversation_id, created_at);

CREATE INDEX IF NOT EXISTS idx_turn_tool_usage_turn
ON turn_tool_usage(conversation_id, turn_id);

-- =========================
-- Turn Token Usage (last-turn debug + audit)
-- =========================
CREATE TABLE IF NOT EXISTS turn_token_usage (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  agent TEXT NOT NULL,
  model TEXT NOT NULL,
  prompt_tokens INTEGER,
  completion_tokens INTEGER,
  total_tokens INTEGER,
  tool_calls INTEGER,
  created_at TEXT NOT NULL,
  FOREIGN KEY(conversation_id) REFERENCES conversations(id)
);

CREATE INDEX IF NOT EXISTS idx_turn_token_usage_turn
ON turn_token_usage(conversation_id, turn_id);

-- =========================
-- Turn Router Decisions (routing + tool need)
-- =========================
CREATE TABLE IF NOT EXISTS turn_router_decisions (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  agent TEXT NOT NULL,
  need_tools INTEGER,
  task_type TEXT,
  confidence REAL,
  proposed_tools_json TEXT,
  decision_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(conversation_id) REFERENCES conversations(id)
);

CREATE INDEX IF NOT EXISTS idx_turn_router_decisions_turn
ON turn_router_decisions(conversation_id, turn_id);

-- =========================
-- Tool Policies (scoped permissions)
-- =========================
CREATE TABLE IF NOT EXISTS tool_policies (
  user_id TEXT NOT NULL,
  agent TEXT NOT NULL,
  tool_name TEXT NOT NULL,
  allow INTEGER NOT NULL,
  constraints_json TEXT,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (user_id, agent, tool_name)
);

CREATE INDEX IF NOT EXISTS idx_tool_policies_user_agent
ON tool_policies(user_id, agent);

-- =========================
-- Life Manager: calendar, tasks, contacts, docs, finance
-- =========================
CREATE TABLE IF NOT EXISTS calendar_events (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  title TEXT NOT NULL,
  start_at TEXT NOT NULL,
  start_at_utc TEXT,
  end_at TEXT NOT NULL,
  end_at_utc TEXT,
  location TEXT,
  notes TEXT,
  status TEXT NOT NULL DEFAULT 'scheduled',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_user_start
ON calendar_events(user_id, start_at_utc);

CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  title TEXT NOT NULL,
  notes TEXT,
  priority INTEGER,
  due_at TEXT,
  due_at_utc TEXT,
  rrule TEXT,
  status TEXT NOT NULL DEFAULT 'open',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_user_due
ON tasks(user_id, due_at_utc);

CREATE INDEX IF NOT EXISTS idx_tasks_user_status
ON tasks(user_id, status);

CREATE TABLE IF NOT EXISTS contacts (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  name TEXT NOT NULL,
  email TEXT,
  phone TEXT,
  notes TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_contacts_user_name
ON contacts(user_id, name);

CREATE INDEX IF NOT EXISTS idx_contacts_user_email
ON contacts(user_id, email);

CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_user_time
ON documents(user_id, updated_at);

CREATE TABLE IF NOT EXISTS email_drafts (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  to_json TEXT NOT NULL,
  subject TEXT NOT NULL,
  body TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'draft',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_email_drafts_user_time
ON email_drafts(user_id, updated_at);

CREATE TABLE IF NOT EXISTS expenses (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  amount REAL NOT NULL,
  currency TEXT NOT NULL,
  category TEXT,
  merchant TEXT,
  notes TEXT,
  occurred_at TEXT,
  occurred_at_utc TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_expenses_user_time
ON expenses(user_id, occurred_at_utc);

-- =========================
-- Health: metrics, meds, appointments, meals, workouts
-- =========================
CREATE TABLE IF NOT EXISTS health_metrics (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  metric TEXT NOT NULL,
  value REAL NOT NULL,
  unit TEXT,
  recorded_at TEXT,
  recorded_at_utc TEXT,
  notes TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metrics_user_metric_time
ON health_metrics(user_id, metric, recorded_at_utc);

CREATE TABLE IF NOT EXISTS medication_schedules (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  medication TEXT NOT NULL,
  dose TEXT,
  unit TEXT,
  times_json TEXT NOT NULL,
  start_date TEXT,
  end_date TEXT,
  notes TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS appointments (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  event_id TEXT NOT NULL,
  provider TEXT,
  reason TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(event_id) REFERENCES calendar_events(id)
);

CREATE INDEX IF NOT EXISTS idx_appointments_user_time
ON appointments(user_id, updated_at);

CREATE TABLE IF NOT EXISTS meals (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  summary TEXT NOT NULL,
  calories REAL,
  protein_g REAL,
  carbs_g REAL,
  fat_g REAL,
  recorded_at TEXT,
  recorded_at_utc TEXT,
  notes TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_meals_user_time
ON meals(user_id, recorded_at_utc);

CREATE TABLE IF NOT EXISTS workouts (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  workout_type TEXT NOT NULL,
  duration_min REAL,
  intensity TEXT,
  calories REAL,
  recorded_at TEXT,
  recorded_at_utc TEXT,
  notes TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_workouts_user_time
ON workouts(user_id, recorded_at_utc);

-- =========================
-- DS: experiment tracking (runs)
-- =========================
CREATE TABLE IF NOT EXISTS ds_runs (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  name TEXT NOT NULL,
  params_json TEXT,
  metrics_json TEXT,
  notes TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ds_runs_user_time
ON ds_runs(user_id, created_at);
