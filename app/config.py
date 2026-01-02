from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "assistant.sqlite3"

MODEL = "gpt-5.2"  # change to your preferred model
EMBEDDING_MODEL = "text-embedding-3-small"  # default for ChatGPT export memory
SYSTEM_INSTRUCTIONS = "You are a helpful assistant."

# Context budgeting (token-aware trimming/summarization).
# Set this to your model's actual context length if different.
MODEL_CONTEXT_TOKENS = 8_192
REQUEST_BUDGET_FRACTION = 0.7

# Auto-injected long-term memory (from imported ChatGPT export)
MEMORY_INJECT_K = 6
MEMORY_INJECT_CANDIDATE_LIMIT = 250
MEMORY_INJECT_MAX_TOKENS = 3_000

# Auto-injected stable profile memory (user-controlled)
PROFILE_INJECT_MAX_TOKENS = 600

# simple MVP limit: last K messages (not tokens)
MAX_HISTORY_MESSAGES = 40
