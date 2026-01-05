from typing import Any, Dict, List

from app.db import get_user_profile, upsert_user_profile


def _deep_merge_profile(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base or {})
    for key, value in (updates or {}).items():
        if key in out and isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = _deep_merge_profile(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def set_profile_tool(
    conn,
    *,
    user_id: str,
    profile: Dict[str, Any],
    replace: bool = False,
    remove_keys: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Persist stable, user-approved facts/preferences/goals that should be injected every turn.
    """
    if not isinstance(profile, dict):
        raise ValueError("profile must be an object/dict")

    existing = get_user_profile(conn, user_id)
    if replace:
        merged = dict(profile)
    else:
        merged = _deep_merge_profile(existing, profile)

    if remove_keys:
        for k in remove_keys:
            if isinstance(k, str):
                merged.pop(k, None)

    upsert_user_profile(conn, user_id, merged)
    return {"user_id": user_id, "profile": merged}
