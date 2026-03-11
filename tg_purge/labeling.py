"""ML label management data layer for tg-bot-detector.

Provides functions to bootstrap label annotations from heuristic scores,
persist them to JSON with appropriate permissions, reload them, and compute
aggregate statistics.  No external dependencies — stdlib only.

Label taxonomy
--------------
  "bot"       — score >= 4  (high-confidence bot signal)
  "human"     — score == 0  (no bot signal detected)
  "unlabeled" — score 1-3   (ambiguous; needs human review)

File format
-----------
  {
    "channel": "@chan",
    "version": 1,
    "labels": {
      "<user_id as str>": {
        "label": "bot" | "human" | "unlabeled",
        "source": "heuristic_bootstrap" | "human",
        "timestamp": "<ISO-8601 string>"
      },
      ...
    }
  }

Security
--------
  Label files contain real Telegram user IDs (PII).
  Parent directories are created with chmod 700 (best-effort).
  Files are written with chmod 600 (best-effort).
"""

import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Score thresholds
# ---------------------------------------------------------------------------

# Minimum score that classifies a subscriber as a bot.
_BOT_THRESHOLD = 4


def _score_to_label(score: int) -> str:
    """Map a heuristic score to a label string.

    Args:
        score: Integer score from score_user().

    Returns:
        "bot" if score >= _BOT_THRESHOLD,
        "human" if score == 0,
        "unlabeled" otherwise (1 <= score < _BOT_THRESHOLD).
    """
    if score >= _BOT_THRESHOLD:
        return "bot"
    if score == 0:
        return "human"
    return "unlabeled"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bootstrap_labels(users: dict, scored: dict) -> dict:
    """Derive initial labels from heuristic scores.

    Only users that appear in *both* `users` and `scored` receive a label.
    Users present in `users` but missing from `scored` are silently skipped
    (they have not been scored yet and carry no signal).

    Args:
        users:  Mapping of user_id (int) → User object.
        scored: Mapping of user_id (int) → (score: int, reasons: list[str]).

    Returns:
        dict mapping user_id (int) → {
            "label":     str,
            "source":    "heuristic_bootstrap",
            "timestamp": ISO-8601 UTC string,
        }
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    result = {}

    for uid, (score, _reasons) in scored.items():
        # Only include users that exist in the users mapping.
        if uid not in users:
            continue
        result[uid] = {
            "label": _score_to_label(score),
            "source": "heuristic_bootstrap",
            "timestamp": now_iso,
        }

    return result


def save_labels(labels: dict, channel: str, path: str) -> None:
    """Persist labels to a JSON file with restrictive permissions.

    Parent directories are created automatically if they do not exist.
    The directory is chmod 700 (best-effort) and the file is chmod 600
    (best-effort) because they contain real Telegram user IDs (PII).

    JSON stores dict keys as strings; load_labels() converts them back to
    int user IDs on read.

    Args:
        labels:  Mapping of user_id (int) → label info dict.
        channel: Telegram channel identifier string (stored in JSON header).
        path:    Absolute or relative file path for the output JSON.
    """
    file_path = Path(path)
    parent = file_path.parent

    # Create parent directory tree.
    parent.mkdir(parents=True, exist_ok=True)

    # Best-effort: restrict directory permissions to owner only.
    try:
        os.chmod(str(parent), 0o700)
    except OSError:
        pass

    # Build the JSON payload.  User IDs are int keys in memory; JSON
    # serialises them as strings — load_labels() reverses this.
    payload = {
        "channel": channel,
        "version": 1,
        "labels": {str(uid): info for uid, info in labels.items()},
    }

    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Best-effort: restrict file permissions to owner read/write only.
    try:
        os.chmod(str(file_path), 0o600)
    except OSError:
        pass


def load_labels(path: str) -> dict:
    """Load labels from a JSON file produced by save_labels().

    JSON stores user IDs as string keys; this function converts them back to
    int so callers can look up entries by numeric user ID without
    explicit casting.

    Args:
        path: Path to the JSON label file.

    Returns:
        dict with keys "channel" (str), "version" (int), "labels" (dict
        mapping int user_id → label info).  If the file does not exist,
        returns an empty structure:
          {"channel": "", "version": 1, "labels": {}}.
    """
    file_path = Path(path)

    if not file_path.exists():
        return {"channel": "", "version": 1, "labels": {}}

    raw = json.loads(file_path.read_text(encoding="utf-8"))

    # Convert string keys back to int user IDs.
    raw_labels = raw.get("labels", {})
    int_labels = {int(k): v for k, v in raw_labels.items()}

    return {
        "channel": raw.get("channel", ""),
        "version": raw.get("version", 1),
        "labels": int_labels,
    }


def label_stats(labels: dict) -> dict:
    """Compute aggregate statistics over a label mapping.

    Args:
        labels: Mapping of user_id → {"label": str, "source": str, ...}.
                Typically the "labels" sub-dict returned by load_labels().

    Returns:
        dict with the following integer counts:
          "bot"          — entries with label == "bot"
          "human"        — entries with label == "human"
          "unlabeled"    — entries with label == "unlabeled"
          "total"        — total number of entries (bot + human + unlabeled)
          "human_labeled"— entries where source == "human" (analyst-reviewed)
    """
    bot_count = 0
    human_count = 0
    unlabeled_count = 0
    human_labeled_count = 0

    for _uid, info in labels.items():
        label = info.get("label", "unlabeled")
        source = info.get("source", "")

        if label == "bot":
            bot_count += 1
        elif label == "human":
            human_count += 1
        else:
            # Treat any unknown label as unlabeled to be safe.
            unlabeled_count += 1

        if source == "human":
            human_labeled_count += 1

    return {
        "bot": bot_count,
        "human": human_count,
        "unlabeled": unlabeled_count,
        "total": bot_count + human_count + unlabeled_count,
        "human_labeled": human_labeled_count,
    }
