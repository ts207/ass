#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_html_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    marker = "var jsonData = "
    start = text.find(marker)
    if start == -1:
        raise RuntimeError("Could not find jsonData in HTML export.")
    start += len(marker)
    i = start
    n = len(text)
    while i < n and text[i].isspace():
        i += 1
    if i >= n or text[i] not in "[{":
        raise RuntimeError("Invalid jsonData start in HTML export.")
    depth = 0
    in_str = False
    escape = False
    for j in range(i, n):
        ch = text[j]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                json_str = text[i : j + 1].strip()
                return json.loads(json_str)
    raise RuntimeError("Could not parse jsonData from HTML export.")


def _normalize_conversations(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [c for c in data if isinstance(c, dict)]
    if isinstance(data, dict):
        for key in ("conversations", "items", "data"):
            v = data.get(key)
            if isinstance(v, list):
                return [c for c in v if isinstance(c, dict)]
        if "mapping" in data and isinstance(data["mapping"], dict):
            return [data]
    return []


def _extract_text(message: Dict[str, Any]) -> str:
    if not message:
        return ""
    content = message.get("content") or {}
    parts = content.get("parts")
    if isinstance(parts, list) and parts:
        return "\n".join(str(p) for p in parts if p is not None).strip()
    if isinstance(content.get("text"), str):
        return content["text"].strip()
    if content.get("content_type") == "multimodal_text" and isinstance(content.get("parts"), list):
        return "\n".join(str(p) for p in content.get("parts") if p is not None).strip()
    return ""


def _ts_to_iso(ts: Any) -> str | None:
    if ts is None:
        return None
    try:
        val = float(ts)
    except Exception:
        return None
    try:
        return datetime.fromtimestamp(val, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _message_features(text: str) -> Dict[str, Any]:
    s = text or ""
    words = re.findall(r"[A-Za-z0-9_']+", s)
    return {
        "chars": len(s),
        "words": len(words),
        "has_codeblock": "```" in s,
        "has_url": bool(re.search(r"https?://|www\\.", s)),
        "is_question": s.strip().endswith("?"),
    }

TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "life": ["reminder", "schedule", "calendar", "task", "habit", "routine", "plan"],
    "health": ["symptom", "doctor", "medication", "medicine", "pain", "sleep", "diet", "workout", "exercise"],
    "code": ["python", "javascript", "typescript", "java", "c++", "c#", "rust", "sql", "bash", "traceback", "stack trace", "exception"],
    "ds": ["pandas", "numpy", "sklearn", "scikit", "regression", "classification", "dataset", "feature", "hyperparameter", "train", "test set"],
    "finance": ["budget", "expense", "invoice", "tax", "rent", "salary", "cost"],
    "creative": ["story", "poem", "lyrics", "novel", "character", "rewrite"],
}


def _tag_topic(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    best_topic = "other"
    best_score = 0
    for topic in sorted(TOPIC_KEYWORDS.keys()):
        score = 0
        for kw in TOPIC_KEYWORDS[topic]:
            if kw in t:
                score += 1
        if score > best_score:
            best_score = score
            best_topic = topic
    if best_score == 0:
        return {"topic": "other", "topic_score": 0}
    return {"topic": best_topic, "topic_score": best_score}


def _sessionize_messages(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    work = work.sort_values(["conversation_id", "timestamp_dt", "message_id"])
    work["prev_time"] = work.groupby("conversation_id")["timestamp_dt"].shift()
    work["gap_min"] = (work["timestamp_dt"] - work["prev_time"]).dt.total_seconds() / 60.0
    work["new_session"] = work["prev_time"].isna() | (work["gap_min"] > 30)
    work["session_num"] = work.groupby("conversation_id")["new_session"].cumsum()
    work["session_id"] = work["conversation_id"].astype(str) + "_s" + work["session_num"].astype(str)
    return work


def _build_sessions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "session_id",
                "conversation_id",
                "start_time",
                "end_time",
                "duration_min",
                "message_count",
                "user_messages",
                "assistant_messages",
                "total_chars",
            ]
        )
    work = df.copy()
    if "session_id" not in work.columns:
        work = _sessionize_messages(work)
    work["is_user"] = work["role"] == "user"
    work["is_assistant"] = work["role"] == "assistant"
    grouped = work.groupby(["conversation_id", "session_id"], as_index=False)
    out = grouped.agg(
        start_time=("timestamp_dt", "min"),
        end_time=("timestamp_dt", "max"),
        message_count=("message_id", "count"),
        user_messages=("is_user", "sum"),
        assistant_messages=("is_assistant", "sum"),
        total_chars=("chars", "sum"),
    )
    out["duration_min"] = (out["end_time"] - out["start_time"]).dt.total_seconds() / 60.0
    return out[
        [
            "session_id",
            "conversation_id",
            "start_time",
            "end_time",
            "duration_min",
            "message_count",
            "user_messages",
            "assistant_messages",
            "total_chars",
        ]
    ]


def _build_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "message_count", "user_messages", "assistant_messages", "total_chars", "avg_chars"])
    work = df.copy()
    work["date"] = work["timestamp_dt"].dt.date
    work["is_user"] = work["role"] == "user"
    work["is_assistant"] = work["role"] == "assistant"
    grouped = work.groupby("date", as_index=False)
    out = grouped.agg(
        message_count=("message_id", "count"),
        user_messages=("is_user", "sum"),
        assistant_messages=("is_assistant", "sum"),
        total_chars=("chars", "sum"),
        avg_chars=("chars", "mean"),
    )
    return out


def _build_conversation_timelines(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "conversation_id",
                "start_time",
                "end_time",
                "duration_min",
                "message_count",
                "user_messages",
                "assistant_messages",
                "total_chars",
                "active_days",
            ]
        )
    work = df.copy()
    work["date"] = work["timestamp_dt"].dt.date
    work["is_user"] = work["role"] == "user"
    work["is_assistant"] = work["role"] == "assistant"
    grouped = work.groupby("conversation_id", as_index=False)
    out = grouped.agg(
        start_time=("timestamp_dt", "min"),
        end_time=("timestamp_dt", "max"),
        message_count=("message_id", "count"),
        user_messages=("is_user", "sum"),
        assistant_messages=("is_assistant", "sum"),
        total_chars=("chars", "sum"),
        active_days=("date", "nunique"),
    )
    out["duration_min"] = (out["end_time"] - out["start_time"]).dt.total_seconds() / 60.0
    return out[
        [
            "conversation_id",
            "start_time",
            "end_time",
            "duration_min",
            "message_count",
            "user_messages",
            "assistant_messages",
            "total_chars",
            "active_days",
        ]
    ]


def _compute_response_times(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "conversation_id",
                "user_message_id",
                "assistant_message_id",
                "user_time",
                "assistant_time",
                "response_time_sec",
                "session_id",
                "topic",
            ]
        )
    work = df.copy()
    work = work.sort_values(["conversation_id", "timestamp_dt", "message_id"])
    work["next_role"] = work.groupby("conversation_id")["role"].shift(-1)
    work["next_time"] = work.groupby("conversation_id")["timestamp_dt"].shift(-1)
    work["next_message_id"] = work.groupby("conversation_id")["message_id"].shift(-1)
    mask = (work["role"] == "user") & (work["next_role"] == "assistant")
    base_cols = ["conversation_id", "message_id", "timestamp_dt", "next_message_id", "next_time"]
    if "session_id" in work.columns:
        base_cols.append("session_id")
    if "topic" in work.columns:
        base_cols.append("topic")
    out = work.loc[mask, base_cols].copy()
    out["response_time_sec"] = (out["next_time"] - out["timestamp_dt"]).dt.total_seconds()
    out = out[out["response_time_sec"] >= 0]
    out = out.rename(
        columns={
            "message_id": "user_message_id",
            "timestamp_dt": "user_time",
            "next_message_id": "assistant_message_id",
            "next_time": "assistant_time",
        }
    )
    cols = [
        "conversation_id",
        "user_message_id",
        "assistant_message_id",
        "user_time",
        "assistant_time",
        "response_time_sec",
    ]
    if "session_id" in out.columns:
        cols.append("session_id")
    if "topic" in out.columns:
        cols.append("topic")
    return out[cols]


def _topic_entropy(counts: pd.Series) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts / float(total)
    return float(-sum(p * math.log(p, 2) for p in probs if p > 0))


def _build_topic_drift(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "conversation_id",
                "total_messages",
                "distinct_topics",
                "dominant_topic",
                "dominant_topic_share",
                "topic_entropy",
                "topic_switches",
                "topic_switch_rate",
            ]
        )
    work = df.copy()
    work = work.sort_values(["conversation_id", "timestamp_dt", "message_id"])
    out_rows = []
    for convo_id, grp in work.groupby("conversation_id"):
        topics = grp["topic"].fillna("other")
        counts = topics.value_counts()
        dominant_topic = str(counts.index[0]) if not counts.empty else "other"
        dominant_share = float(counts.iloc[0]) / float(counts.sum()) if not counts.empty else 0.0
        switches = int((topics != topics.shift()).sum() - 1) if len(topics) > 1 else 0
        total_msgs = int(len(topics))
        switch_rate = float(switches) / float(total_msgs - 1) if total_msgs > 1 else 0.0
        out_rows.append(
            {
                "conversation_id": convo_id,
                "total_messages": total_msgs,
                "distinct_topics": int(counts.size),
                "dominant_topic": dominant_topic,
                "dominant_topic_share": dominant_share,
                "topic_entropy": _topic_entropy(counts),
                "topic_switches": switches,
                "topic_switch_rate": switch_rate,
            }
        )
    return pd.DataFrame(out_rows)


def _response_time_stats_by_topic(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "topic" not in df.columns:
        return pd.DataFrame(columns=["topic", "count", "mean_sec", "median_sec", "p90_sec", "p95_sec"])
    grouped = df.groupby("topic")
    out = grouped["response_time_sec"].agg(
        count="count",
        mean_sec="mean",
        median_sec="median",
        p90_sec=lambda s: s.quantile(0.9),
        p95_sec=lambda s: s.quantile(0.95),
    ).reset_index()
    return out.sort_values(["count", "topic"], ascending=[False, True])


def _response_time_stats_by_session_bucket(
    response_times: pd.DataFrame,
    sessions: pd.DataFrame,
) -> pd.DataFrame:
    if response_times.empty or sessions.empty or "session_id" not in response_times.columns:
        return pd.DataFrame(columns=["session_bucket", "count", "mean_sec", "median_sec", "p90_sec", "p95_sec"])
    dur = sessions["duration_min"].dropna()
    if dur.empty:
        return pd.DataFrame(columns=["session_bucket", "count", "mean_sec", "median_sec", "p90_sec", "p95_sec"])
    q1 = float(dur.quantile(0.33))
    q2 = float(dur.quantile(0.66))
    bucket_map = sessions[["session_id", "duration_min"]].copy()
    bucket_map["session_bucket"] = "long"
    bucket_map.loc[bucket_map["duration_min"] <= q2, "session_bucket"] = "medium"
    bucket_map.loc[bucket_map["duration_min"] <= q1, "session_bucket"] = "short"
    merged = response_times.merge(bucket_map[["session_id", "session_bucket"]], on="session_id", how="left")
    grouped = merged.groupby("session_bucket")["response_time_sec"].agg(
        count="count",
        mean_sec="mean",
        median_sec="median",
        p90_sec=lambda s: s.quantile(0.9),
        p95_sec=lambda s: s.quantile(0.95),
    ).reset_index()
    return grouped.sort_values(["count", "session_bucket"], ascending=[False, True])


def _build_activity_profile(messages: pd.DataFrame) -> Dict[str, Any]:
    if messages.empty or "timestamp_dt" not in messages.columns:
        return {"hour_counts": [], "weekday_counts": [], "top_hours": [], "top_weekdays": []}

    work = messages.dropna(subset=["timestamp_dt"]).copy()
    if work.empty:
        return {"hour_counts": [], "weekday_counts": [], "top_hours": [], "top_weekdays": []}

    hour_counts = work["timestamp_dt"].dt.hour.value_counts().sort_index()
    hour_total = int(hour_counts.sum())
    hour_list = []
    for hour, count in hour_counts.items():
        hour_list.append(
            {"hour": int(hour), "count": int(count), "share": (float(count) / hour_total) if hour_total else 0.0}
        )

    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_counts = work["timestamp_dt"].dt.dayofweek.value_counts().sort_index()
    weekday_total = int(weekday_counts.sum())
    weekday_list = []
    for idx, count in weekday_counts.items():
        name = weekdays[int(idx)] if int(idx) < len(weekdays) else str(idx)
        weekday_list.append(
            {"weekday": name, "count": int(count), "share": (float(count) / weekday_total) if weekday_total else 0.0}
        )

    top_hours = sorted(hour_list, key=lambda x: (-x["count"], x["hour"]))[:3]
    top_weekdays = sorted(weekday_list, key=lambda x: (-x["count"], x["weekday"]))[:3]
    return {
        "hour_counts": hour_list,
        "weekday_counts": weekday_list,
        "top_hours": top_hours,
        "top_weekdays": top_weekdays,
    }


def _build_user_profile(
    *,
    source_used: str,
    outdir: Path,
    conversations: pd.DataFrame,
    messages: pd.DataFrame,
    sessions: pd.DataFrame,
    daily: pd.DataFrame,
    topic_counts: pd.DataFrame,
    session_topic_summary: pd.DataFrame,
    role_transitions: pd.DataFrame,
    conversation_timelines: pd.DataFrame,
    response_times: pd.DataFrame,
    topic_drift: pd.DataFrame,
    response_time_by_topic: pd.DataFrame,
    response_time_by_session_bucket: pd.DataFrame,
) -> Dict[str, Any]:
    total_messages = int(len(messages))
    total_conversations = int(len(conversations))
    total_sessions = int(len(sessions))
    active_days = int(len(daily))

    ts_min = messages["timestamp_dt"].min() if "timestamp_dt" in messages.columns else None
    ts_max = messages["timestamp_dt"].max() if "timestamp_dt" in messages.columns else None
    date_range = {
        "start": str(ts_min) if pd.notna(ts_min) else None,
        "end": str(ts_max) if pd.notna(ts_max) else None,
    }

    role_counts = messages["role"].value_counts(dropna=False).to_dict()
    role_stats = []
    for role, count in sorted(role_counts.items(), key=lambda x: (-x[1], str(x[0]))):
        role_stats.append(
            {
                "role": str(role),
                "count": int(count),
                "share": (float(count) / total_messages) if total_messages else 0.0,
            }
        )

    msg_len_mean = float(messages["chars"].mean()) if total_messages else 0.0
    msg_len_median = float(messages["chars"].median()) if total_messages else 0.0
    word_len_mean = float(messages["words"].mean()) if total_messages else 0.0
    word_len_median = float(messages["words"].median()) if total_messages else 0.0
    skew_ratio = (msg_len_mean / msg_len_median) if msg_len_median else 0.0

    topic_list = []
    for _, row in topic_counts.sort_values(["count", "topic"], ascending=[False, True]).iterrows():
        topic_list.append(
            {"topic": str(row["topic"]), "count": int(row["count"]), "share": float(row["share"])}
        )

    top_topics = [t for t in topic_list if t["topic"] != "other"][:3]

    activity_profile = _build_activity_profile(messages)

    session_stats = {
        "avg_duration_min": float(sessions["duration_min"].mean()) if total_sessions else 0.0,
        "median_duration_min": float(sessions["duration_min"].median()) if total_sessions else 0.0,
        "avg_messages_per_session": float(sessions["message_count"].mean()) if total_sessions else 0.0,
        "max_duration_min": float(sessions["duration_min"].max()) if total_sessions else 0.0,
    }

    conv_count = int(len(conversation_timelines))
    conv_stats = {
        "avg_duration_min": float(conversation_timelines["duration_min"].mean()) if conv_count else 0.0,
        "median_duration_min": float(conversation_timelines["duration_min"].median()) if conv_count else 0.0,
        "max_duration_min": float(conversation_timelines["duration_min"].max()) if conv_count else 0.0,
        "multi_day_conversations": int((conversation_timelines["active_days"] > 1).sum()) if conv_count else 0,
    }

    resp_count = int(len(response_times))
    response_stats = {
        "count": resp_count,
        "mean_sec": float(response_times["response_time_sec"].mean()) if resp_count else 0.0,
        "median_sec": float(response_times["response_time_sec"].median()) if resp_count else 0.0,
        "p90_sec": float(response_times["response_time_sec"].quantile(0.9)) if resp_count else 0.0,
        "p95_sec": float(response_times["response_time_sec"].quantile(0.95)) if resp_count else 0.0,
        "max_sec": float(response_times["response_time_sec"].max()) if resp_count else 0.0,
    }

    drift_count = int(len(topic_drift))
    drift_stats = {
        "avg_switch_rate": float(topic_drift["topic_switch_rate"].mean()) if drift_count else 0.0,
        "avg_entropy": float(topic_drift["topic_entropy"].mean()) if drift_count else 0.0,
        "high_drift_conversations": int((topic_drift["topic_switch_rate"] >= 0.5).sum()) if drift_count else 0,
    }

    question_rate = float(messages["is_question"].mean()) if total_messages else 0.0
    code_rate = float(messages["has_codeblock"].mean()) if total_messages else 0.0
    url_rate = float(messages["has_url"].mean()) if total_messages else 0.0

    insights = [
        f"Message length is skewed (median {msg_len_median:.2f} chars vs mean {msg_len_mean:.2f}; ratio {skew_ratio:.2f}).",
        f"Questions appear in {question_rate * 100:.2f}% of messages; code blocks in {code_rate * 100:.2f}%; URLs in {url_rate * 100:.2f}%.",
        f"Top tagged interests beyond 'other': {', '.join([t['topic'] for t in top_topics]) or 'n/a'}.",
        f"Median session duration {session_stats['median_duration_min']:.2f} min with {session_stats['avg_messages_per_session']:.2f} messages per session on average.",
        f"Response time median {response_stats['median_sec']:.2f}s; p90 {response_stats['p90_sec']:.2f}s.",
        f"Topic drift is moderate (avg switch rate {drift_stats['avg_switch_rate']:.2f}; high-drift conversations {drift_stats['high_drift_conversations']}).",
        f"Multi-day conversations: {conv_stats['multi_day_conversations']} of {conv_count} ({(conv_stats['multi_day_conversations'] / conv_count * 100.0) if conv_count else 0.0:.2f}%).",
        f"Most active UTC hours: {', '.join([str(h['hour']) for h in activity_profile['top_hours']]) or 'n/a'}.",
        f"Most active weekdays (UTC): {', '.join([d['weekday'] for d in activity_profile['top_weekdays']]) or 'n/a'}.",
    ]

    return {
        "source_used": source_used,
        "run_id": outdir.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_range": date_range,
        "counts": {
            "conversations": total_conversations,
            "messages": total_messages,
            "sessions": total_sessions,
            "active_days": active_days,
        },
        "message_length": {
            "mean_chars": msg_len_mean,
            "median_chars": msg_len_median,
            "mean_words": word_len_mean,
            "median_words": word_len_median,
        },
        "role_distribution": role_stats,
        "topic_distribution": topic_list,
        "session_stats": session_stats,
        "conversation_stats": conv_stats,
        "response_time_stats": response_stats,
        "response_time_by_topic": response_time_by_topic.to_dict(orient="records"),
        "response_time_by_session_bucket": response_time_by_session_bucket.to_dict(orient="records"),
        "topic_drift": drift_stats,
        "activity_profile_utc": activity_profile,
        "insights": insights,
        "limitations": [
            "Profile is derived from message logs and timestamps; no external context or demographics are inferred.",
            "Topic tagging is keyword-based and coarse; 'other' may include diverse content.",
            "Response times reflect message order in exports, not necessarily live latency.",
        ],
    }


def _write_user_profile_md(profile: Dict[str, Any], path: Path) -> None:
    lines = []
    lines.append("# Derived User Profile")
    lines.append("")
    lines.append(f"Run: `{profile.get('run_id')}`")
    lines.append(f"Source: `{profile.get('source_used')}`")
    lines.append(f"Generated: {profile.get('generated_at_utc')}")
    lines.append("")
    lines.append("## Counts")
    counts = profile.get("counts", {})
    lines.append(f"- conversations: {counts.get('conversations')}")
    lines.append(f"- messages: {counts.get('messages')}")
    lines.append(f"- sessions: {counts.get('sessions')}")
    lines.append(f"- active_days: {counts.get('active_days')}")
    lines.append("")
    lines.append("## Insights")
    for item in profile.get("insights", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Activity (UTC)")
    activity = profile.get("activity_profile_utc", {})
    top_hours = ", ".join([str(h.get("hour")) for h in activity.get("top_hours", [])]) or "n/a"
    top_days = ", ".join([d.get("weekday") for d in activity.get("top_weekdays", [])]) or "n/a"
    lines.append(f"- top_hours: {top_hours}")
    lines.append(f"- top_weekdays: {top_days}")
    lines.append("")
    lines.append("## Topic distribution (top 10)")
    for row in profile.get("topic_distribution", [])[:10]:
        lines.append(f"- {row.get('topic')}: {row.get('count')} ({row.get('share') * 100:.2f}%)")
    lines.append("")
    lines.append("## Limitations")
    for item in profile.get("limitations", []):
        lines.append(f"- {item}")
    path.write_text("\n".join(lines), encoding="utf-8")

def _plot_volume_by_day(daily: pd.DataFrame, path: Path) -> None:
    if daily.empty:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(pd.to_datetime(daily["date"]), daily["message_count"], marker="o", linewidth=1)
    plt.title("Messages by Day (UTC)")
    plt.xlabel("Date")
    plt.ylabel("Messages")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_session_duration_hist(sessions: pd.DataFrame, path: Path) -> None:
    if sessions.empty:
        return
    vals = sessions["duration_min"].dropna()
    if vals.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=30, color="#1f77b4", alpha=0.8)
    plt.title("Session Duration Histogram (minutes)")
    plt.xlabel("Duration (min)")
    plt.ylabel("Sessions")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_response_time_hist(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    vals = df["response_time_sec"].dropna()
    vals = vals[vals >= 0]
    if vals.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(vals / 60.0, bins=40, color="#4c72b0", alpha=0.8)
    plt.title("Response Time Histogram (minutes)")
    plt.xlabel("Minutes")
    plt.ylabel("Responses")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_user_vs_assistant(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    subset = df[df["role"].isin(["user", "assistant"])]
    if subset.empty:
        return
    means = subset.groupby("role")["chars"].mean()
    plt.figure(figsize=(5, 4))
    means.plot(kind="bar", color=["#ff7f0e", "#2ca02c"])
    plt.title("Average Message Length by Role")
    plt.ylabel("Average Characters")
    plt.xlabel("Role")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _write_report(
    *,
    out_path: Path,
    source_used: str,
    source_reason: str,
    conversations: pd.DataFrame,
    messages: pd.DataFrame,
    sessions: pd.DataFrame,
    daily: pd.DataFrame,
    topic_counts: pd.DataFrame,
    session_topic_summary: pd.DataFrame,
    role_transitions: pd.DataFrame,
    conversation_timelines: pd.DataFrame,
    response_times: pd.DataFrame,
    topic_drift: pd.DataFrame,
    response_time_by_topic: pd.DataFrame,
    response_time_by_session_bucket: pd.DataFrame,
    user_profile: Dict[str, Any],
) -> None:
    n_convos = len(conversations)
    n_msgs = len(messages)
    missing_text = int(messages["text"].isna().sum())
    missing_ts = int(messages["timestamp"].isna().sum())
    missing_role = int(messages["role"].isna().sum())
    with_ts = n_msgs - missing_ts
    role_counts = messages["role"].value_counts(dropna=False).to_dict()
    user_msgs = int(role_counts.get("user", 0))
    assistant_msgs = int(role_counts.get("assistant", 0))
    system_msgs = int(role_counts.get("system", 0))
    avg_chars = float(messages["chars"].mean()) if n_msgs else 0.0
    med_chars = float(messages["chars"].median()) if n_msgs else 0.0
    codeblock_rate = float(messages["has_codeblock"].mean()) if n_msgs else 0.0
    url_rate = float(messages["has_url"].mean()) if n_msgs else 0.0
    question_rate = float(messages["is_question"].mean()) if n_msgs else 0.0

    session_count = len(sessions)
    avg_session_msgs = float(sessions["message_count"].mean()) if session_count else 0.0
    med_session_duration = float(sessions["duration_min"].median()) if session_count else 0.0
    max_session_duration = float(sessions["duration_min"].max()) if session_count else 0.0

    days_active = len(daily)
    peak_day_msgs = int(daily["message_count"].max()) if days_active else 0
    peak_day = str(daily.loc[daily["message_count"].idxmax(), "date"]) if days_active else "n/a"

    conv_count = len(conversation_timelines)
    avg_conv_duration = float(conversation_timelines["duration_min"].mean()) if conv_count else 0.0
    med_conv_duration = float(conversation_timelines["duration_min"].median()) if conv_count else 0.0
    max_conv_duration = float(conversation_timelines["duration_min"].max()) if conv_count else 0.0
    multi_day_convos = int((conversation_timelines["active_days"] > 1).sum()) if conv_count else 0

    resp_count = len(response_times)
    if resp_count:
        resp_mean = float(response_times["response_time_sec"].mean())
        resp_median = float(response_times["response_time_sec"].median())
        resp_p90 = float(response_times["response_time_sec"].quantile(0.9))
        resp_p95 = float(response_times["response_time_sec"].quantile(0.95))
        resp_max = float(response_times["response_time_sec"].max())
    else:
        resp_mean = resp_median = resp_p90 = resp_p95 = resp_max = 0.0

    top_topic = "n/a"
    top_topic_count = 0
    top_topic_share = 0.0
    if not topic_counts.empty:
        top_row = topic_counts.sort_values(["count", "topic"], ascending=[False, True]).iloc[0]
        top_topic = str(top_row["topic"])
        top_topic_count = int(top_row["count"])
        top_topic_share = float(top_row["share"])

    longest_topic = "n/a"
    longest_topic_duration = 0.0
    if not session_topic_summary.empty:
        row = session_topic_summary.sort_values(
            ["avg_duration_min", "topic"], ascending=[False, True]
        ).iloc[0]
        longest_topic = str(row["topic"])
        longest_topic_duration = float(row["avg_duration_min"])

    top_transition = "n/a"
    top_transition_count = 0
    if not role_transitions.empty:
        row = role_transitions.sort_values(["count", "from_role", "to_role"], ascending=[False, True, True]).iloc[0]
        top_transition = f"{row['from_role']} -> {row['to_role']}"
        top_transition_count = int(row["count"])

    drift_count = len(topic_drift)
    avg_switch_rate = float(topic_drift["topic_switch_rate"].mean()) if drift_count else 0.0
    avg_entropy = float(topic_drift["topic_entropy"].mean()) if drift_count else 0.0
    high_drift = int((topic_drift["topic_switch_rate"] >= 0.5).sum()) if drift_count else 0

    observations = [
        f"Total conversations: {n_convos}.",
        f"Total messages: {n_msgs} (user={user_msgs}, assistant={assistant_msgs}, system={system_msgs}).",
        f"Messages with timestamps: {with_ts} ({with_ts / n_msgs * 100:.2f}% of messages)." if n_msgs else "Messages with timestamps: 0.",
        f"Average message length: {avg_chars:.2f} chars; median: {med_chars:.2f} chars.",
        f"Code block rate: {codeblock_rate * 100:.2f}% of messages.",
        f"URL rate: {url_rate * 100:.2f}% of messages.",
        f"Question rate: {question_rate * 100:.2f}% of messages.",
        f"Total sessions (gap>30m): {session_count}.",
        f"Median session duration: {med_session_duration:.2f} min; max: {max_session_duration:.2f} min.",
        f"Average messages per session: {avg_session_msgs:.2f}.",
        f"Active days: {days_active}; peak day {peak_day} with {peak_day_msgs} messages.",
        f"Top message topic: {top_topic} with {top_topic_count} messages ({top_topic_share * 100:.2f}%).",
        f"Longest sessions by topic: {longest_topic} (avg {longest_topic_duration:.2f} min).",
        f"Most common role transition: {top_transition} ({top_transition_count} transitions).",
        f"Conversations spanning >1 day: {multi_day_convos} ({(multi_day_convos / conv_count * 100.0) if conv_count else 0.0:.2f}%).",
        f"Response times (user->assistant): count={resp_count}, median={resp_median:.2f}s, p90={resp_p90:.2f}s.",
        f"Avg topic switch rate per conversation: {avg_switch_rate:.2f}.",
        f"Avg topic entropy per conversation: {avg_entropy:.2f}.",
        f"High topic-drift conversations (switch rate >=0.5): {high_drift}.",
    ]

    lines = []
    lines.append("# ChatGPT Export Analysis Report")
    lines.append("")
    lines.append(f"Source used: `{source_used}`")
    lines.append(f"Reason: {source_reason}")
    lines.append("")
    lines.append("## Row counts")
    lines.append(f"- conversations: {n_convos}")
    lines.append(f"- messages: {n_msgs}")
    lines.append(f"- sessions: {session_count}")
    lines.append(f"- daily rows: {days_active}")
    lines.append("")
    lines.append("## Missingness")
    if n_msgs:
        lines.append(f"- messages.text missing: {missing_text} ({missing_text / n_msgs * 100:.2f}%)")
        lines.append(f"- messages.timestamp missing: {missing_ts} ({missing_ts / n_msgs * 100:.2f}%)")
        lines.append(f"- messages.role missing: {missing_role} ({missing_role / n_msgs * 100:.2f}%)")
    else:
        lines.append("- messages.text missing: 0")
        lines.append("- messages.timestamp missing: 0")
        lines.append("- messages.role missing: 0")
    lines.append("")
    lines.append("## Key stats (exact)")
    lines.append(f"- avg_chars: {avg_chars:.2f}")
    lines.append(f"- median_chars: {med_chars:.2f}")
    lines.append(f"- codeblock_rate: {codeblock_rate:.6f}")
    lines.append(f"- url_rate: {url_rate:.6f}")
    lines.append(f"- question_rate: {question_rate:.6f}")
    lines.append(f"- avg_session_msgs: {avg_session_msgs:.2f}")
    lines.append(f"- median_session_duration_min: {med_session_duration:.2f}")
    lines.append(f"- max_session_duration_min: {max_session_duration:.2f}")
    if not topic_counts.empty:
        lines.append(f"- top_topic: {top_topic}")
        lines.append(f"- top_topic_share: {top_topic_share:.6f}")
    if not role_transitions.empty:
        lines.append(f"- top_role_transition: {top_transition}")
    if conv_count:
        lines.append(f"- avg_conversation_duration_min: {avg_conv_duration:.2f}")
        lines.append(f"- median_conversation_duration_min: {med_conv_duration:.2f}")
        lines.append(f"- max_conversation_duration_min: {max_conv_duration:.2f}")
    if resp_count:
        lines.append(f"- response_time_mean_sec: {resp_mean:.2f}")
        lines.append(f"- response_time_median_sec: {resp_median:.2f}")
        lines.append(f"- response_time_p90_sec: {resp_p90:.2f}")
        lines.append(f"- response_time_p95_sec: {resp_p95:.2f}")
        lines.append(f"- response_time_max_sec: {resp_max:.2f}")
    if drift_count:
        lines.append(f"- avg_topic_switch_rate: {avg_switch_rate:.4f}")
        lines.append(f"- avg_topic_entropy: {avg_entropy:.4f}")
    lines.append("")
    lines.append("## Topic tagging")
    if topic_counts.empty:
        lines.append("- No topics available.")
    else:
        for _, row in topic_counts.sort_values(["count", "topic"], ascending=[False, True]).head(10).iterrows():
            lines.append(f"- {row['topic']}: {int(row['count'])} ({row['share'] * 100:.2f}%)")
    lines.append("")
    lines.append("## Session comparisons by topic (dominant topic per session)")
    if session_topic_summary.empty:
        lines.append("- No session topic summary available.")
    else:
        for _, row in session_topic_summary.sort_values(["session_count", "topic"], ascending=[False, True]).iterrows():
            lines.append(
                f"- {row['topic']}: sessions={int(row['session_count'])}, "
                f"avg_msgs={row['avg_message_count']:.2f}, avg_duration_min={row['avg_duration_min']:.2f}"
            )
    lines.append("")
    lines.append("## Role transitions (top 10)")
    if role_transitions.empty:
        lines.append("- No role transitions available.")
    else:
        for _, row in role_transitions.head(10).iterrows():
            lines.append(f"- {row['from_role']} -> {row['to_role']}: {int(row['count'])}")
    lines.append("")
    lines.append("## Conversation timelines")
    if conversation_timelines.empty:
        lines.append("- No conversation timeline summary available.")
    else:
        lines.append(f"- conversations: {conv_count}")
        lines.append(f"- avg_duration_min: {avg_conv_duration:.2f}")
        lines.append(f"- median_duration_min: {med_conv_duration:.2f}")
        lines.append(f"- max_duration_min: {max_conv_duration:.2f}")
        lines.append(f"- multi_day_conversations: {multi_day_convos}")
    lines.append("")
    lines.append("## Response time distribution (user -> assistant)")
    if resp_count == 0:
        lines.append("- No response times available.")
    else:
        lines.append(f"- count: {resp_count}")
        lines.append(f"- mean_sec: {resp_mean:.2f}")
        lines.append(f"- median_sec: {resp_median:.2f}")
        lines.append(f"- p90_sec: {resp_p90:.2f}")
        lines.append(f"- p95_sec: {resp_p95:.2f}")
        lines.append(f"- max_sec: {resp_max:.2f}")
    lines.append("")
    lines.append("## Response times by topic (top 10)")
    if response_time_by_topic.empty:
        lines.append("- No response-time topic breakdown available.")
    else:
        for _, row in response_time_by_topic.head(10).iterrows():
            lines.append(
                f"- {row['topic']}: count={int(row['count'])}, median_sec={row['median_sec']:.2f}, p90_sec={row['p90_sec']:.2f}"
            )
    lines.append("")
    lines.append("## Response times by session length bucket")
    if response_time_by_session_bucket.empty:
        lines.append("- No response-time session bucket breakdown available.")
    else:
        for _, row in response_time_by_session_bucket.iterrows():
            lines.append(
                f"- {row['session_bucket']}: count={int(row['count'])}, median_sec={row['median_sec']:.2f}, p90_sec={row['p90_sec']:.2f}"
            )
    lines.append("")
    lines.append("## Topic drift (per conversation)")
    if topic_drift.empty:
        lines.append("- No topic drift metrics available.")
    else:
        lines.append(f"- avg_switch_rate: {avg_switch_rate:.2f}")
        lines.append(f"- avg_entropy: {avg_entropy:.2f}")
        lines.append(f"- high_drift_conversations: {high_drift}")
    lines.append("")
    lines.append("## User profile (derived)")
    for item in user_profile.get("insights", [])[:10]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Observations")
    for i, obs in enumerate(observations, 1):
        lines.append(f"{i}. {obs}")
    lines.append("")
    lines.append("## Limitations")
    lines.append("- Sessions rely on message timestamps; missing timestamps reduce session accuracy.")
    lines.append("- The export graph is flattened to messages; branch structure is not represented in sequences.")
    lines.append("- HTML parsing depends on the jsonData marker; malformed exports may reduce coverage.")
    lines.append("")
    lines.append("## Next analyses")
    lines.append("- Refine topic taxonomy (expand keywords or add domain-specific labels).")
    lines.append("- Segment response-time distributions by topic or session length.")
    lines.append("- Add per-conversation topic drift metrics.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze ChatGPT export data.")
    parser.add_argument("--json", required=True, help="Path to conversations.json")
    parser.add_argument("--html", required=True, help="Path to chat.html (fallback)")
    parser.add_argument("--outdir", required=True, help="Output directory (data/exports/<run_id>)")
    args = parser.parse_args()

    json_path = Path(args.json)
    html_path = Path(args.html)
    outdir = Path(args.outdir)

    source_used = "conversations.json"
    source_reason = ""

    data = None
    if json_path.exists():
        try:
            data = _load_json(json_path)
            convs = _normalize_conversations(data)
            if not convs:
                raise RuntimeError("Parsed JSON but found zero conversations.")
            source_reason = f"Parsed JSON successfully with {len(convs)} conversations."
        except Exception as e:
            data = None
            source_reason = f"JSON parse failed or incomplete: {e}"
    else:
        source_reason = "JSON file missing."

    if data is None:
        if not html_path.exists():
            raise SystemExit("No usable input found: conversations.json and chat.html are missing or invalid.")
        data = _load_html_json(html_path)
        convs = _normalize_conversations(data)
        source_used = "chat.html"
        source_reason = f"Used HTML fallback with {len(convs)} conversations. {source_reason}".strip()
    else:
        convs = _normalize_conversations(data)

    conversations_rows: List[Dict[str, Any]] = []
    message_rows: List[Dict[str, Any]] = []

    for conv in convs:
        conv_id = conv.get("id") or conv.get("conversation_id") or conv.get("uuid")
        if not conv_id:
            conv_id = f"conv_{len(conversations_rows) + 1}"
        conversations_rows.append(
            {
                "conversation_id": str(conv_id),
                "title": (conv.get("title") or "").strip(),
                "create_time": _ts_to_iso(conv.get("create_time")),
                "update_time": _ts_to_iso(conv.get("update_time")),
            }
        )
        mapping = conv.get("mapping") or {}
        if not isinstance(mapping, dict):
            continue
        for node_id, node in mapping.items():
            if not isinstance(node, dict):
                continue
            message = node.get("message") or {}
            if not message:
                continue
            role = None
            author = message.get("author") or {}
            if isinstance(author, dict):
                role = author.get("role")
            text = _extract_text(message)
            message_rows.append(
                {
                    "message_id": str(message.get("id") or node_id),
                    "conversation_id": str(conv_id),
                    "role": role,
                    "timestamp": _ts_to_iso(message.get("create_time")),
                    "text": text,
                }
            )

    conversations_df = pd.DataFrame(conversations_rows)
    messages_df = pd.DataFrame(message_rows)
    if messages_df.empty:
        messages_df = pd.DataFrame(columns=["message_id", "conversation_id", "role", "timestamp", "text"])

    features = messages_df["text"].apply(_message_features)
    features_df = pd.DataFrame(list(features))
    topics_df = pd.DataFrame(list(messages_df["text"].apply(_tag_topic)))
    messages_features_df = pd.concat([messages_df, features_df, topics_df], axis=1)
    messages_features_df["topic"] = messages_features_df["topic"].fillna("other")
    messages_features_df["timestamp_dt"] = pd.to_datetime(messages_features_df["timestamp"], errors="coerce", utc=True)

    sessionized_df = _sessionize_messages(messages_features_df.dropna(subset=["timestamp_dt"]))
    sessions_df = _build_sessions(sessionized_df)
    daily_df = _build_daily(messages_features_df.dropna(subset=["timestamp_dt"]))

    if messages_features_df.empty:
        topic_counts_df = pd.DataFrame(columns=["topic", "count", "share"])
    else:
        topic_counts_df = messages_features_df["topic"].value_counts(dropna=False).reset_index()
        topic_counts_df.columns = ["topic", "count"]
        topic_counts_df["share"] = topic_counts_df["count"] / float(len(messages_features_df))

    if sessionized_df.empty:
        session_topic_summary = pd.DataFrame(
            columns=["topic", "session_count", "avg_duration_min", "avg_message_count", "avg_total_chars"]
        )
    else:
        topic_counts = (
            sessionized_df.groupby(["session_id", "conversation_id", "topic"], as_index=False)
            .size()
            .rename(columns={"size": "message_count"})
        )
        topic_counts = topic_counts.sort_values(
            ["session_id", "message_count", "topic"], ascending=[True, False, True]
        )
        dominant = topic_counts.drop_duplicates("session_id", keep="first").rename(
            columns={"message_count": "topic_message_count"}
        )
        session_topic = sessions_df.merge(
            dominant[["session_id", "topic", "topic_message_count"]],
            on="session_id",
            how="left",
        )
        session_topic_summary = session_topic.groupby("topic", as_index=False).agg(
            session_count=("session_id", "count"),
            avg_duration_min=("duration_min", "mean"),
            avg_message_count=("message_count", "mean"),
            avg_total_chars=("total_chars", "mean"),
        )

    if sessionized_df.empty:
        role_transitions = pd.DataFrame(columns=["from_role", "to_role", "count"])
    else:
        work = sessionized_df.sort_values(["conversation_id", "timestamp_dt", "message_id"])
        work["next_role"] = work.groupby("conversation_id")["role"].shift(-1)
        role_transitions = (
            work.dropna(subset=["role", "next_role"])
            .groupby(["role", "next_role"])
            .size()
            .reset_index(name="count")
            .rename(columns={"role": "from_role", "next_role": "to_role"})
            .sort_values("count", ascending=False)
        )

    outdir.mkdir(parents=True, exist_ok=True)
    normalized_dir = outdir / "normalized"
    features_dir = outdir / "features"
    report_dir = outdir / "report"
    figures_dir = report_dir / "figures"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    conversations_df.to_parquet(normalized_dir / "conversations.parquet", index=False)
    messages_df.to_parquet(normalized_dir / "messages.parquet", index=False)
    messages_features_df.to_parquet(features_dir / "messages_features.parquet", index=False)
    sessions_df.to_csv(features_dir / "session_features.csv", index=False)
    daily_df.to_csv(features_dir / "daily_features.csv", index=False)
    topic_counts_df.to_csv(features_dir / "topic_message_counts.csv", index=False)
    session_topic_summary.to_csv(features_dir / "session_topic_summary.csv", index=False)
    role_transitions.to_csv(features_dir / "role_transitions.csv", index=False)

    conversation_timelines = _build_conversation_timelines(sessionized_df)
    response_times = _compute_response_times(sessionized_df)
    topic_drift = _build_topic_drift(sessionized_df)
    response_time_by_topic = _response_time_stats_by_topic(response_times)
    response_time_by_session_bucket = _response_time_stats_by_session_bucket(response_times, sessions_df)
    conversation_timelines.to_csv(features_dir / "conversation_timelines.csv", index=False)
    response_times.to_csv(features_dir / "response_times.csv", index=False)
    topic_drift.to_csv(features_dir / "conversation_topic_drift.csv", index=False)
    response_time_by_topic.to_csv(features_dir / "response_time_by_topic.csv", index=False)
    response_time_by_session_bucket.to_csv(features_dir / "response_time_by_session_bucket.csv", index=False)

    _plot_volume_by_day(daily_df, figures_dir / "volume_by_day.png")
    _plot_session_duration_hist(sessions_df, figures_dir / "session_duration_hist.png")
    _plot_user_vs_assistant(messages_features_df, figures_dir / "user_vs_assistant_length.png")
    _plot_response_time_hist(response_times, figures_dir / "response_time_hist.png")

    user_profile = _build_user_profile(
        source_used=source_used,
        outdir=outdir,
        conversations=conversations_df,
        messages=messages_features_df,
        sessions=sessions_df,
        daily=daily_df,
        topic_counts=topic_counts_df,
        session_topic_summary=session_topic_summary,
        role_transitions=role_transitions,
        conversation_timelines=conversation_timelines,
        response_times=response_times,
        topic_drift=topic_drift,
        response_time_by_topic=response_time_by_topic,
        response_time_by_session_bucket=response_time_by_session_bucket,
    )
    (report_dir / "user_profile.json").write_text(
        json.dumps(user_profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_user_profile_md(user_profile, report_dir / "user_profile.md")

    report_path = report_dir / "report.md"
    _write_report(
        out_path=report_path,
        source_used=source_used,
        source_reason=source_reason,
        conversations=conversations_df,
        messages=messages_features_df,
        sessions=sessions_df,
        daily=daily_df,
        topic_counts=topic_counts_df,
        session_topic_summary=session_topic_summary,
        role_transitions=role_transitions,
        conversation_timelines=conversation_timelines,
        response_times=response_times,
        topic_drift=topic_drift,
        response_time_by_topic=response_time_by_topic,
        response_time_by_session_bucket=response_time_by_session_bucket,
        user_profile=user_profile,
    )

    print(str(outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
