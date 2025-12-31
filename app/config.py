from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "assistant.sqlite3"

MODEL = "gpt-5"  # change to your preferred model
SYSTEM_INSTRUCTIONS = "You are a helpful assistant."

# simple MVP limit: last K messages (not tokens)
MAX_HISTORY_MESSAGES = 24
