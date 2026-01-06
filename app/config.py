from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "assistant.sqlite3"

MODEL = "gpt-5.2"  # change to your preferred model
EMBEDDING_MODEL = "text-embedding-3-small"  # default for ChatGPT export memory
SYSTEM_INSTRUCTIONS = "You are a helpful assistant."

# Context budgeting (token-aware trimming/summarization).
# Set this to your model's actual context length if different.
MODEL_CONTEXT_TOKENS = 128_000
REQUEST_BUDGET_FRACTION = 0.7

# Auto-injected long-term memory (from imported ChatGPT export)
MEMORY_INJECT_K = 6
MEMORY_INJECT_CANDIDATE_LIMIT = 250
MEMORY_INJECT_MAX_TOKENS = 3_000
MEMORY_MMR_LAMBDA = 0.65
MEMORY_MMR_CANDIDATE_POOL = 60
MEMORY_RECENCY_HALF_LIFE_DAYS = 180.0

# Auto-injected stable profile memory (user-controlled)
PROFILE_INJECT_MAX_TOKENS = 600

# simple MVP limit: last K messages (not tokens)
MAX_HISTORY_MESSAGES = 400

# Helper utilities for token budgeting and sensible defaults.
# Use these to compute how much of the model context to reserve for prompts/history
# and to recommend how large injected memory/profile chunks should be given the selected model.
DEFAULT_EXPECTED_REPLY_TOKENS = 2048


def compute_request_budget(context_tokens: int = MODEL_CONTEXT_TOKENS,
                           fraction: float = REQUEST_BUDGET_FRACTION,
                           expected_reply_tokens: int = DEFAULT_EXPECTED_REPLY_TOKENS) -> int:
    """
    Return a safe token budget for the request (system + history + injected memory),
    ensuring there is space left for an expected reply.
    """
    budget = int(context_tokens * fraction)
    # Never allow the budget to consume the whole context; reserve space for reply
    safe = max(0, min(budget, context_tokens - expected_reply_tokens))
    return safe


def recommend_memory_inject_max(context_tokens: int = MODEL_CONTEXT_TOKENS) -> int:
    """
    Recommend a sensible default for MEMORY_INJECT_MAX_TOKENS based on the model context size.
    """
    if context_tokens >= 200_000:
        return 6_000
    if context_tokens >= 100_000:
        return 3_000
    if context_tokens >= 50_000:
        return 1_500
    if context_tokens >= 25_000:
        return 800
    return 300
