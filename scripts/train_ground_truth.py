#!/usr/bin/env python3
"""
Train ML model using ground-truth labels instead of heuristic-derived labels.

Ground truth sources:
  - Departed users (admin log): confirmed bots — they left during the exodus
  - Labeled humans (labels.json): manually curated human ground truth

This script solves the circular label problem where the old pipeline trained
on labels derived from the heuristic scorer, making F1=1.0 meaningless.

Data flow:
  1. Load feature-validation-full.json (raw fields for 3,583 bots + 3,069 humans)
  2. Load cached features.json (47-feature vectors from bootstrap enumeration)
  3. For users with cached vectors, reuse those (richer: temporal, cohort data)
  4. For remaining users, convert raw validation fields to feature vectors
  5. Check labels.json for any human-reviewed entries to merge
  6. Run stratified k-fold cross-validation for meaningful metrics
  7. Train final model on all data and save

IMPORTANT: Status fields for departed bots are contaminated — leaving the
channel counts as a login, so bots appear "recently active". The model will
learn that status is unreliable, which is actually desirable for generalization.

Usage:
    python scripts/train_ground_truth.py \
        --validation-data output/feature-validation-full.json \
        --features-cache datasets/leviathan_news/features.json \
        --labels-path datasets/leviathan_news/labels.json \
        --output-dir models
"""

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path so tg_purge is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tg_purge.features import (
    FEATURE_KEYS, _count_emoji, _has_crypto_keywords,
    _name_username_similarity,
)
from tg_purge.scoring import _AIRDROP_TOKENS
from tg_purge.ml import train_model, ml_available


def raw_to_feature_vector(raw: dict) -> dict:
    """Convert raw validation fields to a 51-feature ML vector.

    Maps the flat dict from validate_new_features.py's extract_all_fields()
    to the FEATURE_KEYS schema used by features.py's extract_features().

    Features that require data not present in the raw fields (temporal,
    cohort, photo quality) are set to their "unknown" defaults (-1 or 0).

    Args:
        raw: Dict from feature-validation-full.json with keys like
            has_photo, has_username, photo_dc_id, status_type, etc.

    Returns:
        Dict[str, float] with exactly the keys in FEATURE_KEYS.
    """
    first_name = raw.get("first_name", "") or ""

    # --- Name analysis ---
    # Digit ratio: fraction of digit characters in first_name.
    if first_name:
        digit_ratio = sum(c.isdigit() for c in first_name) / len(first_name)
    else:
        digit_ratio = 0.0

    # Script count: number of distinct writing systems in first_name.
    has_latin = bool(re.search(r"[a-zA-Z]", first_name))
    has_cyrillic = bool(re.search(r"[\u0400-\u04FF]", first_name))
    has_arabic = bool(re.search(r"[\u0600-\u06FF]", first_name))
    has_cjk = bool(re.search(r"[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF]", first_name))
    script_count = sum([has_latin, has_cyrillic, has_arabic, has_cjk])

    # --- Status one-hot encoding ---
    # The raw data stores status as the Telethon class name (e.g., "UserStatusRecently").
    # Map to one-hot columns matching features.py's schema.
    status_type = raw.get("status_type", "None")
    status_map = {
        "None":                1 if status_type == "None" else 0,
        "UserStatusEmpty":     1 if status_type == "UserStatusEmpty" else 0,
        "UserStatusOnline":    1 if status_type == "UserStatusOnline" else 0,
        "UserStatusRecently":  1 if status_type == "UserStatusRecently" else 0,
        "UserStatusLastWeek":  1 if status_type == "UserStatusLastWeek" else 0,
        "UserStatusLastMonth": 1 if status_type == "UserStatusLastMonth" else 0,
        "UserStatusOffline":   1 if status_type == "UserStatusOffline" else 0,
    }
    # "None" and "UserStatusEmpty" both map to status_empty.
    status_empty = float(status_map["None"] or status_map["UserStatusEmpty"])

    # Days since last seen: approximation from status type.
    # We don't have the actual timestamp, so use the standard approximations.
    if status_type in ("None", "UserStatusEmpty"):
        days_since = -1.0
    elif status_type == "UserStatusOnline":
        days_since = 0.0
    elif status_type == "UserStatusRecently":
        days_since = 1.0
    elif status_type == "UserStatusLastWeek":
        days_since = 7.0
    elif status_type == "UserStatusLastMonth":
        days_since = 30.0
    elif status_type == "UserStatusOffline":
        # No timestamp available in raw data; use -1 as unknown.
        days_since = -1.0
    else:
        days_since = -1.0

    # --- Photo metadata ---
    dc_id = raw.get("photo_dc_id", 0) or 0

    features = {
        # Account flags
        "is_deleted":       float(bool(raw.get("is_deleted", False))),
        "is_bot":           float(bool(raw.get("is_bot_api", False))),
        "is_scam":          float(bool(raw.get("is_scam", False))),
        "is_fake":          float(bool(raw.get("is_fake", False))),
        "is_restricted":    float(bool(raw.get("is_restricted", False))),
        "is_premium":       float(bool(raw.get("is_premium", False))),
        "has_emoji_status": float(bool(raw.get("has_emoji_status", False))),

        # Profile completeness
        "has_photo":          float(bool(raw.get("has_photo", False))),
        "has_username":       float(bool(raw.get("has_username", False))),
        "has_last_name":      float(bool(raw.get("has_last_name", False))),
        "first_name_length":  float(len(first_name)),
        "name_digit_ratio":   digit_ratio,
        "script_count":       float(script_count),

        # Name analysis — emoji, crypto keywords, name/username similarity.
        "name_emoji_count":   float(_count_emoji(
            first_name + " " + (raw.get("last_name", "") or "")
        )),
        "name_has_crypto_kw": float(_has_crypto_keywords(
            first_name + " " + (raw.get("last_name", "") or "")
        )),
        "name_airdrop_token_count": float(sum(
            1 for token in _AIRDROP_TOKENS
            if token in (first_name + " " + (raw.get("last_name", "") or "")).lower()
        )),
        "name_username_sim":  _name_username_similarity(
            first_name,
            raw.get("last_name", "") or "",
            raw.get("username", "") or "",
        ),

        # Activity status (one-hot)
        "status_empty":       status_empty,
        "status_online":      float(status_map["UserStatusOnline"]),
        "status_recently":    float(status_map["UserStatusRecently"]),
        "status_last_week":   float(status_map["UserStatusLastWeek"]),
        "status_last_month":  float(status_map["UserStatusLastMonth"]),
        "status_offline":     float(status_map["UserStatusOffline"]),
        "days_since_last_seen": days_since,

        # Temporal (not available in validation data — use defaults)
        "is_spike_join":    0.0,
        "days_since_join":  -1.0,
        "join_hour_utc":    -1.0,
        "join_day_of_week": -1.0,

        # Cohort (not available — defaults)
        "is_cohort_member":          0.0,
        "cohort_size":               0.0,
        "cohort_join_spread_hours":  0.0,
        "cohort_profile_similarity": 0.0,

        # Photo metadata
        "photo_dc_id":      float(dc_id),
        "photo_has_video":  float(bool(raw.get("photo_has_video", False))),
        "photo_dc_is_1":    float(dc_id == 1),
        "photo_dc_is_5":    float(dc_id == 5),

        # Photo quality (not available — defaults)
        "photo_file_size":    0.0,
        "photo_edge_std":     0.0,
        "photo_lum_variance": 0.0,
        "photo_sat_mean":     0.0,

        # Extended profile signals
        "has_custom_color":            float(bool(raw.get("has_custom_color", False))),
        "has_profile_color":           float(bool(raw.get("has_profile_color", False))),
        "has_stories":                 float(bool(raw.get("has_stories", False))),
        "stories_unavailable":         float(bool(raw.get("stories_unavailable", False))),
        "has_contact_require_premium": float(bool(raw.get("has_contact_require_premium", False))),
        "usernames_count":             float(raw.get("usernames_count", 0) or 0),
        "is_verified":                 float(bool(raw.get("is_verified", False))),
        "has_lang_code":               float(bool(raw.get("has_lang_code", False))),
        "has_paid_messages":           float(bool(raw.get("has_paid_messages", False))),
        "is_stars_subscriber":         0.0,  # Not available in validation data.

        # Heuristic score
        "heuristic_score": float(raw.get("heuristic_score", 0) or 0),
    }

    # Verify all FEATURE_KEYS are present.
    missing = set(FEATURE_KEYS) - set(features.keys())
    if missing:
        raise ValueError(f"Missing feature keys after conversion: {missing}")

    return features


def _exit_profile_to_raw(profile: dict, heuristic_score: int) -> dict:
    """Convert an exit monitor JSONL profile to the raw format expected by
    raw_to_feature_vector().

    The exit monitor (monitor_leaves.py) serializes Telethon User objects with
    different key names than validate_new_features.py. This function maps
    between the two schemas so departed users from live monitoring can be
    ingested into the training pipeline as confirmed bot ground truth.

    Field mapping:
      JSONL profile key       → raw_to_feature_vector() expected key
      ─────────────────────────────────────────────────────────────────
      id                      → user_id
      photo (bool)            → has_photo
      username (str|null)     → has_username, username
      first_name              → first_name
      last_name               → last_name, has_last_name
      status_type             → status_type
      deleted                 → is_deleted
      bot                     → is_bot_api
      verified                → is_verified
      restricted              → is_restricted
      scam                    → is_scam
      fake                    → is_fake
      premium                 → is_premium
      emoji_status (bool)     → has_emoji_status
      photo_meta.dc_id        → photo_dc_id
      photo_meta.has_video    → photo_has_video
      color (bool)            → has_custom_color
      profile_color (bool)    → has_profile_color
      usernames_count         → usernames_count
      stories_max_id          → has_stories
      contact_require_premium → has_contact_require_premium

    Args:
        profile: Dict from the "profile" field of an exit monitor JSONL record.
        heuristic_score: Heuristic score from the JSONL record.

    Returns:
        Dict compatible with raw_to_feature_vector().
    """
    photo_meta = profile.get("photo_meta") or {}

    return {
        "user_id": profile.get("id"),
        "first_name": profile.get("first_name") or "",
        "last_name": profile.get("last_name") or "",
        "username": profile.get("username") or "",
        "has_photo": bool(profile.get("photo", False)),
        "has_username": bool(profile.get("username")),
        "has_last_name": bool(profile.get("last_name")),
        "is_deleted": bool(profile.get("deleted", False)),
        "is_bot_api": bool(profile.get("bot", False)),
        "is_scam": bool(profile.get("scam", False)),
        "is_fake": bool(profile.get("fake", False)),
        "is_restricted": bool(profile.get("restricted", False)),
        "is_premium": bool(profile.get("premium", False)),
        "is_verified": bool(profile.get("verified", False)),
        "has_emoji_status": bool(profile.get("emoji_status", False)),
        "status_type": profile.get("status_type") or "None",
        "photo_dc_id": photo_meta.get("dc_id", 0) or 0,
        "photo_has_video": bool(photo_meta.get("has_video", False)),
        "has_custom_color": bool(profile.get("color", False)),
        "has_profile_color": bool(profile.get("profile_color", False)),
        "usernames_count": profile.get("usernames_count", 0) or 0,
        "has_stories": bool((profile.get("stories_max_id") or 0) > 0),
        "stories_unavailable": False,  # Not tracked in exit monitor.
        "has_contact_require_premium": bool(
            profile.get("contact_require_premium", False)
        ),
        "has_lang_code": False,  # Not tracked in exit monitor.
        "has_paid_messages": False,  # Not tracked in exit monitor.
        "heuristic_score": heuristic_score or 0,
    }


def load_exit_data(paths: list) -> list:
    """Load confirmed bot departures from exit monitor JSONL files.

    Reads one or more JSONL files produced by monitor_leaves.py. Each line
    is a departure event with a full profile snapshot and heuristic score.
    Profiles are converted to the raw dict format expected by
    raw_to_feature_vector().

    Filters out:
      - Records with no profile (user was gone before snapshot)
      - Deleted accounts (no useful profile features)
      - Duplicate user IDs (keeps first occurrence)

    Args:
        paths: List of file paths to JSONL exit monitor files.

    Returns:
        List of raw dicts suitable for raw_to_feature_vector(), each
        representing a confirmed bot (departed during exodus).
    """
    results = []
    seen_ids = set()
    skipped_no_profile = 0
    skipped_deleted = 0
    skipped_duplicate = 0

    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"  Warning: exit data file not found: {path}", file=sys.stderr)
            continue

        with open(p, "r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(
                        f"  Warning: malformed JSON at {path}:{line_num}: {e}",
                        file=sys.stderr,
                    )
                    continue

                profile = record.get("profile")
                if not profile:
                    skipped_no_profile += 1
                    continue

                if profile.get("deleted", False):
                    skipped_deleted += 1
                    continue

                uid = profile.get("id")
                if uid in seen_ids:
                    skipped_duplicate += 1
                    continue
                seen_ids.add(uid)

                raw = _exit_profile_to_raw(
                    profile, record.get("heuristic_score", 0)
                )
                results.append(raw)

    print(
        f"Exit data: {len(results)} departed bots from {len(paths)} file(s) "
        f"(skipped: {skipped_no_profile} no-profile, {skipped_deleted} deleted, "
        f"{skipped_duplicate} duplicate)",
        file=sys.stderr,
    )
    return results


def load_validation_data(path: str) -> tuple:
    """Load ground-truth data from feature-validation-full.json.

    Args:
        path: Path to the validation JSON produced by validate_new_features.py.

    Returns:
        Tuple of (bot_data: list[dict], human_data: list[dict]) where each
        dict contains raw field values keyed by field name (user_id, has_photo, etc.).
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    bot_data = data.get("bot_data", [])
    human_data = data.get("human_data", [])
    print(
        f"Validation data: {len(bot_data)} bots, {len(human_data)} humans",
        file=sys.stderr,
    )
    return bot_data, human_data


def load_feature_cache(path: str) -> dict:
    """Load cached feature vectors from features.json.

    Args:
        path: Path to features.json (from label --bootstrap).

    Returns:
        Dict of user_id_str -> feature_vector_dict. Empty dict if file missing.
    """
    p = Path(path)
    if not p.exists():
        print(f"No feature cache at {path}", file=sys.stderr)
        return {}

    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    cache = data.get("features", {})
    print(f"Feature cache: {len(cache)} cached vectors", file=sys.stderr)

    # Pad cached vectors that are missing newly-added feature keys.
    # This handles the transition from 47 → 50 features without requiring
    # a full re-bootstrap. Missing name analysis features default to 0.0
    # (no emoji, no crypto keywords, no similarity signal).
    if cache:
        sample = next(iter(cache.values()))
        missing_keys = set(FEATURE_KEYS) - set(sample.keys())
        if missing_keys:
            print(
                f"  Padding {len(missing_keys)} missing feature(s) in cache: "
                f"{sorted(missing_keys)}",
                file=sys.stderr,
            )
            for vec in cache.values():
                for key in missing_keys:
                    vec.setdefault(key, 0.0)

    return cache


def load_human_reviewed(path: str) -> dict:
    """Load human-reviewed labels from a dedicated reviews file.

    Reads from a standalone human_reviews.json file that is separate from
    the bootstrap-managed labels.json. This file is never overwritten by
    label --bootstrap, so human reviews are safe from accidental loss.

    The file format is:
      {
        "reviews": {
          "<user_id_str>": {"label": "bot"|"human", "timestamp": "..."},
          ...
        }
      }

    Falls back to scanning labels.json for source="human" entries if the
    dedicated reviews file doesn't exist.

    Args:
        path: Path to human_reviews.json (or labels.json as fallback).

    Returns:
        Dict of user_id_str -> label ("bot" or "human") for human-reviewed entries.
    """
    p = Path(path)

    # Try dedicated human reviews file first.
    # Convention: same directory as the path arg, named human_reviews.json.
    reviews_file = p.parent / "human_reviews.json" if p.name != "human_reviews.json" else p

    if reviews_file.exists():
        with open(reviews_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        reviews = data.get("reviews", {})
        reviewed = {}
        for uid_str, info in reviews.items():
            label = info.get("label", "unlabeled") if isinstance(info, dict) else info
            if label in ("bot", "human"):
                reviewed[uid_str] = label

        if reviewed:
            label_counts = Counter(reviewed.values())
            print(
                f"Human reviews file: {reviews_file} ({len(reviewed)} entries: "
                f"{', '.join(f'{v} {k}' for k, v in label_counts.items())})",
                file=sys.stderr,
            )
        return reviewed

    # Fallback: scan labels.json for source="human" entries.
    if not p.exists():
        print(f"No reviews file at {reviews_file} or {path}", file=sys.stderr)
        return {}

    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    labels = data.get("labels", {})
    reviewed = {}
    for uid_str, info in labels.items():
        source = info.get("source", "")
        if source == "human":
            reviewed[uid_str] = info.get("label", "unlabeled")

    n_reviewed = len(reviewed)
    if n_reviewed > 0:
        label_counts = Counter(reviewed.values())
        print(
            f"Human-reviewed labels (from labels.json fallback): {n_reviewed} "
            f"({', '.join(f'{v} {k}' for k, v in label_counts.items())})",
            file=sys.stderr,
        )
    else:
        print(f"No human-reviewed labels found", file=sys.stderr)

    return reviewed



# Features that differ systematically between cached (from bootstrap) and
# converted (from raw validation) vectors due to data availability, NOT
# because of real bot/human differences. These leak the data source into
# the model and must be neutralized to prevent artificial separation.
#
# - Temporal features: Cached vectors have real join dates from channel
#   participants. Converted vectors have -1 (no join date available for
#   departed users). The model would learn "if days_since_join == -1 → bot".
# - days_since_last_seen: Cached vectors use actual timestamps from
#   status.was_online. Converted vectors use rough approximations (1.0 for
#   recently, -1.0 for offline). Different precision leaks the source.
# - Cohort features: Only populated during full enumeration. Always 0 in
#   converted vectors.
# - Photo quality: Only populated when photos are downloaded. Always 0 in
#   converted vectors (and in cached vectors too, currently).
LEAKED_FEATURES = {
    "days_since_join":           -1.0,
    "join_hour_utc":             -1.0,
    "join_day_of_week":          -1.0,
    "is_spike_join":              0.0,
    "days_since_last_seen":      -1.0,
    # Status features: exit monitoring data captures bots at the moment
    # the farm activates them to leave — 97.6% show "online". This makes
    # status_online/status_recently appear as bot signals, when in normal
    # conditions they're human signals. Neutralize to prevent the model
    # from learning "active = bot".
    "status_empty":               0.0,
    "status_online":              0.0,
    "status_recently":            0.0,
    "status_last_week":           0.0,
    "status_last_month":          0.0,
    "status_offline":             0.0,
    # Cohort and photo quality: only populated during specific analysis
    # passes, not available for all data sources.
    "is_cohort_member":           0.0,
    "cohort_size":                0.0,
    "cohort_join_spread_hours":   0.0,
    "cohort_profile_similarity":  0.0,
    "photo_file_size":            0.0,
    "photo_edge_std":             0.0,
    "photo_lum_variance":         0.0,
    "photo_sat_mean":             0.0,
}


def _neutralize_leaked_features(feature_vec: dict) -> dict:
    """Set leaked features to their default "unknown" values.

    When bots and humans come from different data sources (converted vs
    cached), some features systematically differ because of data
    availability, not because of real bot/human differences. Setting these
    to uniform defaults prevents the model from learning source artifacts.

    Args:
        feature_vec: Feature vector dict to neutralize (modified in place).

    Returns:
        The same dict with leaked features set to defaults.
    """
    for key, default_val in LEAKED_FEATURES.items():
        if key in feature_vec:
            feature_vec[key] = default_val
    return feature_vec


def build_training_set(
    bot_data: list,
    human_data: list,
    feature_cache: dict,
    human_reviewed: dict,
    include_bootstrap: bool = False,
    bootstrap_bot_threshold: int = 4,
    bootstrap_human_threshold: int = 0,
) -> tuple:
    """Build aligned feature vectors and labels from all data sources.

    Priority for feature vectors:
      1. Cached vectors from features.json (have richer profile data)
      2. Converted vectors from raw validation fields

    All vectors are post-processed to neutralize features that differ
    due to data source (cached vs converted), not bot/human differences.
    This prevents data leakage from the feature extraction pipeline.

    Priority for labels:
      1. Human-reviewed labels (highest confidence)
      2. Ground-truth source labels (departed = bot, labeled = human)
      3. High-confidence bootstrap labels (score >= threshold, optional)

    Args:
        bot_data: List of raw dicts for departed bots.
        human_data: List of raw dicts for labeled humans.
        feature_cache: Dict of uid_str -> cached feature vector.
        human_reviewed: Dict of uid_str -> human-reviewed label.
        include_bootstrap: If True, add high-confidence bootstrap users
            from the feature cache (using heuristic_score thresholds).
        bootstrap_bot_threshold: Min heuristic_score to label as bot (default 4).
        bootstrap_human_threshold: Max heuristic_score to label as human (default 0).

    Returns:
        Tuple of (features: list[dict], labels: list[str], stats: dict).
    """
    features = []
    labels = []

    # Track statistics for reporting.
    stats = {
        "cached_bot": 0,
        "converted_bot": 0,
        "cached_human": 0,
        "converted_human": 0,
        "human_reviewed_added": 0,
        "human_reviewed_override": 0,
        "skipped_deleted": 0,
        "bootstrap_bot": 0,
        "bootstrap_human": 0,
    }

    # Set of user IDs already added (prevent duplicates).
    seen_uids = set()

    # --- Process departed bots ---
    for raw in bot_data:
        uid = str(raw.get("user_id", ""))
        if not uid or uid in seen_uids:
            continue

        # Skip deleted accounts — they have no useful profile features.
        if raw.get("is_deleted", False):
            stats["skipped_deleted"] += 1
            continue

        seen_uids.add(uid)

        # Determine label: human review overrides ground-truth source.
        label = human_reviewed.get(uid, "bot")
        if uid in human_reviewed and label != "bot":
            stats["human_reviewed_override"] += 1

        # Get feature vector: prefer cached, fall back to conversion.
        if uid in feature_cache:
            vec = dict(feature_cache[uid])  # Copy to avoid mutating cache.
            stats["cached_bot"] += 1
        else:
            vec = raw_to_feature_vector(raw)
            stats["converted_bot"] += 1

        # Neutralize features that leak data source identity.
        _neutralize_leaked_features(vec)
        features.append(vec)
        labels.append(label)

    # --- Process labeled humans ---
    for raw in human_data:
        uid = str(raw.get("user_id", ""))
        if not uid or uid in seen_uids:
            continue

        if raw.get("is_deleted", False):
            stats["skipped_deleted"] += 1
            continue

        seen_uids.add(uid)

        label = human_reviewed.get(uid, "human")
        if uid in human_reviewed and label != "human":
            stats["human_reviewed_override"] += 1

        if uid in feature_cache:
            vec = dict(feature_cache[uid])  # Copy.
            stats["cached_human"] += 1
        else:
            vec = raw_to_feature_vector(raw)
            stats["converted_human"] += 1

        _neutralize_leaked_features(vec)
        features.append(vec)
        labels.append(label)

    # --- Add human-reviewed entries not yet in the dataset ---
    # These are users that were manually reviewed but aren't in the
    # departed/labeled sets. They need cached feature vectors.
    for uid_str, label in human_reviewed.items():
        if uid_str in seen_uids:
            continue
        if label not in ("bot", "human"):
            continue
        if uid_str not in feature_cache:
            continue  # Can't add without features.

        seen_uids.add(uid_str)
        vec = dict(feature_cache[uid_str])
        _neutralize_leaked_features(vec)
        features.append(vec)
        labels.append(label)
        stats["human_reviewed_added"] += 1

    # --- Add high-confidence bootstrap users ---
    # Users with very high or very low heuristic scores are reliable enough
    # to use as training data. Score >= 4 means multiple strong bot signals
    # (deleted, scam, no-status + no-photo, etc.). Score 0 means a clean
    # profile with no bot indicators. These expand the training set without
    # introducing ambiguous labels.
    #
    # No data leakage concern: all bootstrap users come from the same
    # feature cache (same data source), so temporal/cohort features are
    # consistently populated.
    if include_bootstrap:
        for uid_str, vec_orig in feature_cache.items():
            if uid_str in seen_uids:
                continue

            # Skip deleted accounts — no useful profile features.
            if vec_orig.get("is_deleted", 0) == 1:
                continue

            score = vec_orig.get("heuristic_score", -1)

            # Human review overrides bootstrap label.
            if uid_str in human_reviewed:
                label = human_reviewed[uid_str]
            elif score >= bootstrap_bot_threshold:
                label = "bot"
            elif score <= bootstrap_human_threshold:
                label = "human"
            else:
                continue  # Ambiguous score — skip.

            seen_uids.add(uid_str)
            vec = dict(vec_orig)
            _neutralize_leaked_features(vec)
            features.append(vec)
            labels.append(label)

            if label == "bot":
                stats["bootstrap_bot"] += 1
            else:
                stats["bootstrap_human"] += 1

    # When bootstrap labels are included, drop heuristic_score from all
    # vectors to prevent circularity. Bootstrap labels are derived from
    # heuristic_score (score >= threshold → bot), so keeping it as a feature
    # lets the model trivially learn the labeling rule instead of real patterns.
    if include_bootstrap:
        for vec in features:
            vec.pop("heuristic_score", None)
        stats["dropped_heuristic_score"] = True

    return features, labels, stats


def cross_validate(features: list, labels: list, n_folds: int = 5) -> dict:
    """Run stratified k-fold cross-validation for meaningful metrics.

    Unlike the simple 80/20 split in train_model(), k-fold CV gives a
    more reliable estimate of generalization performance by averaging
    metrics across multiple splits.

    Args:
        features: List of feature dicts.
        labels: List of "bot"/"human" label strings.
        n_folds: Number of CV folds (default 5).

    Returns:
        Dict with per-fold and mean metrics.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    from tg_purge.ml import _features_to_array, _get_available_models, _train_single_backend

    # Convert to arrays.
    feature_names = sorted(features[0].keys())
    X = _features_to_array(features, feature_names)
    y = np.array([1 if lbl == "bot" else 0 for lbl in labels], dtype=np.int32)

    # Pick best available backend.
    backends = _get_available_models()
    if not backends:
        return {"error": "No ML backend available"}
    # Use the first (highest priority) backend.
    backend_name, backend_cls = backends[0]

    # Cross-validation.
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Compute class weight for this fold.
        n_neg = int(np.sum(y_train == 0))
        n_pos = int(np.sum(y_train == 1))
        scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

        model = _train_single_backend(backend_name, backend_cls, X_train, y_train, scale_pos_weight)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred_int = [int(v) for v in y_pred]

        try:
            auc = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            auc = 0.0

        fold_result = {
            "fold": fold_idx + 1,
            "f1": float(f1_score(y_test, y_pred_int, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred_int, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_int, zero_division=0)),
            "auc_roc": auc,
            "n_test": len(y_test),
            "n_test_bot": int(np.sum(y_test == 1)),
            "n_test_human": int(np.sum(y_test == 0)),
        }
        fold_metrics.append(fold_result)

        print(
            f"  Fold {fold_idx + 1}/{n_folds}: "
            f"F1={fold_result['f1']:.4f}  "
            f"P={fold_result['precision']:.4f}  "
            f"R={fold_result['recall']:.4f}  "
            f"AUC={fold_result['auc_roc']:.4f}",
            file=sys.stderr,
        )

    # Compute mean and std of each metric across folds.
    mean_metrics = {}
    for key in ("f1", "precision", "recall", "auc_roc"):
        vals = [f[key] for f in fold_metrics]
        mean_metrics[f"mean_{key}"] = sum(vals) / len(vals)
        # Standard deviation.
        mean_val = mean_metrics[f"mean_{key}"]
        std_val = (sum((v - mean_val) ** 2 for v in vals) / len(vals)) ** 0.5
        mean_metrics[f"std_{key}"] = std_val

    return {
        "algorithm": backend_name,
        "n_folds": n_folds,
        "folds": fold_metrics,
        "mean_metrics": mean_metrics,
    }


def print_feature_importance(features: list, labels: list, top_n: int = 20):
    """Train on full data and print feature importance ranking.

    Uses the best available backend to train on all data, then extracts
    feature importance scores. Helps validate which features the model
    actually relies on for discrimination.

    Args:
        features: List of feature dicts.
        labels: List of "bot"/"human" label strings.
        top_n: Number of top features to display.
    """
    import numpy as np
    from tg_purge.ml import _features_to_array, _get_available_models, _train_single_backend

    feature_names = sorted(features[0].keys())
    X = _features_to_array(features, feature_names)
    y = np.array([1 if lbl == "bot" else 0 for lbl in labels], dtype=np.int32)

    backends = _get_available_models()
    if not backends:
        return

    name, cls = backends[0]
    n_neg = int(np.sum(y == 0))
    n_pos = int(np.sum(y == 1))
    scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

    model = _train_single_backend(name, cls, X, y, scale_pos_weight)

    # Extract feature importance — available from all backends.
    if name == "lightgbm":
        importance = model.feature_importances_
    elif name == "xgboost":
        importance = model.feature_importances_
    else:  # sklearn_rf
        importance = model.feature_importances_

    # Sort by importance (descending).
    indices = np.argsort(importance)[::-1]

    print(f"\n{'=' * 60}")
    print(f"FEATURE IMPORTANCE (top {top_n}, {name})")
    print(f"{'=' * 60}")
    print(f"{'Rank':>4s}  {'Feature':40s}  {'Importance':>10s}")
    print(f"{'─' * 4}  {'─' * 40}  {'─' * 10}")

    for rank, idx in enumerate(indices[:top_n], 1):
        print(f"{rank:>4d}  {feature_names[idx]:40s}  {importance[idx]:>10.4f}")


def save_ground_truth_dataset(
    features: list,
    labels: list,
    output_dir: str,
    stats: dict,
):
    """Save the ground-truth training data for reproducibility.

    Writes two files:
      - ground_truth_labels.json: label assignments with source tracking
      - ground_truth_features.json: feature vectors keyed by index

    These files serve as the authoritative training data for the model,
    separate from the heuristic-derived labels.json.

    Args:
        features: List of feature dicts.
        labels: List of "bot"/"human" label strings.
        output_dir: Directory to write the files.
        stats: Build statistics from build_training_set().
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dataset = {
        "created": datetime.now(timezone.utc).isoformat(),
        "source": "ground_truth",
        "description": (
            "Ground-truth training data: departed users as confirmed bots, "
            "labeled humans as confirmed humans. Not derived from heuristic scorer."
        ),
        "stats": stats,
        "n_samples": len(features),
        "n_bot": sum(1 for l in labels if l == "bot"),
        "n_human": sum(1 for l in labels if l == "human"),
        "features": features,
        "labels": labels,
    }

    dataset_path = str(output / "ground_truth_dataset.json")
    with open(dataset_path, "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, indent=2, default=str)

    print(f"\nGround-truth dataset saved to: {dataset_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Train ML model on ground-truth labels (departed bots + labeled humans)."
    )
    parser.add_argument(
        "--validation-data",
        dest="validation_data",
        default="output/feature-validation-full.json",
        help="Path to feature-validation-full.json from validate_new_features.py.",
    )
    parser.add_argument(
        "--features-cache",
        dest="features_cache",
        default="datasets/leviathan_news/features.json",
        help="Path to features.json (cached feature vectors from bootstrap).",
    )
    parser.add_argument(
        "--labels-path",
        dest="labels_path",
        default="datasets/leviathan_news/labels.json",
        help="Path to labels.json (check for human-reviewed entries).",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default="models",
        help="Directory to save the trained model.",
    )
    parser.add_argument(
        "--channel",
        default="@leviathan_news",
        help="Channel name for model file naming.",
    )
    parser.add_argument(
        "--cv-folds",
        dest="cv_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--save-dataset",
        dest="save_dataset",
        action="store_true",
        help="Save the assembled ground-truth dataset to output-dir.",
    )
    parser.add_argument(
        "--exit-data",
        dest="exit_data",
        nargs="+",
        default=[],
        help="Path(s) to exit monitor JSONL files (confirmed bot departures).",
    )
    parser.add_argument(
        "--backend",
        choices=["lightgbm", "xgboost", "sklearn_rf"],
        default=None,
        help="Force a specific ML backend instead of auto-selecting the best.",
    )
    parser.add_argument(
        "--include-bootstrap",
        dest="include_bootstrap",
        action="store_true",
        help="Include high-confidence bootstrap labels (score>=4 bot, score<=0 human).",
    )
    parser.add_argument(
        "--bootstrap-bot-threshold",
        dest="bootstrap_bot_threshold",
        type=int,
        default=4,
        help="Min heuristic score to auto-label as bot (default: 4).",
    )
    parser.add_argument(
        "--bootstrap-human-threshold",
        dest="bootstrap_human_threshold",
        type=int,
        default=0,
        help="Max heuristic score to auto-label as human (default: 0).",
    )
    args = parser.parse_args()

    # --- Check ML dependencies ---
    if not ml_available():
        print(
            "ML dependencies not installed. Run:\n"
            "  pip install scikit-learn lightgbm",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Load all data sources ---
    print("Loading data sources...", file=sys.stderr)
    bot_data, human_data = load_validation_data(args.validation_data)
    feature_cache = load_feature_cache(args.features_cache)
    human_reviewed = load_human_reviewed(args.labels_path)

    # --- Load exit monitoring data (confirmed bot departures) ---
    if args.exit_data:
        exit_bots = load_exit_data(args.exit_data)
        bot_data.extend(exit_bots)
        print(
            f"  → Bot data after exit merge: {len(bot_data)} total",
            file=sys.stderr,
        )

    # --- Build training set ---
    print("\nBuilding training set...", file=sys.stderr)
    features, labels, stats = build_training_set(
        bot_data, human_data, feature_cache, human_reviewed,
        include_bootstrap=args.include_bootstrap,
        bootstrap_bot_threshold=args.bootstrap_bot_threshold,
        bootstrap_human_threshold=args.bootstrap_human_threshold,
    )

    n_bot = sum(1 for l in labels if l == "bot")
    n_human = sum(1 for l in labels if l == "human")
    print(f"\nTraining set assembled:", file=sys.stderr)
    print(f"  Total:           {len(features)} samples", file=sys.stderr)
    print(f"  Bots:            {n_bot}", file=sys.stderr)
    print(f"  Humans:          {n_human}", file=sys.stderr)
    print(f"  From cache:      {stats['cached_bot']} bot + {stats['cached_human']} human", file=sys.stderr)
    print(f"  Converted:       {stats['converted_bot']} bot + {stats['converted_human']} human", file=sys.stderr)
    print(f"  Skipped deleted: {stats['skipped_deleted']}", file=sys.stderr)
    if stats["human_reviewed_added"] > 0:
        print(f"  Human-reviewed:  {stats['human_reviewed_added']} added", file=sys.stderr)
    if stats["human_reviewed_override"] > 0:
        print(f"  Overrides:       {stats['human_reviewed_override']} labels changed by human review", file=sys.stderr)
    if stats["bootstrap_bot"] > 0 or stats["bootstrap_human"] > 0:
        print(
            f"  Bootstrap:       {stats['bootstrap_bot']} bot + "
            f"{stats['bootstrap_human']} human (high-confidence heuristic)",
            file=sys.stderr,
        )
    if stats.get("dropped_heuristic_score"):
        print(
            "  NOTE: heuristic_score dropped to prevent circular labels",
            file=sys.stderr,
        )

    if len(features) < 10:
        print("Error: fewer than 10 samples. Check data paths.", file=sys.stderr)
        sys.exit(1)
    if n_bot == 0 or n_human == 0:
        print("Error: need both bot and human labels.", file=sys.stderr)
        sys.exit(1)

    # --- Save dataset if requested ---
    if args.save_dataset:
        save_ground_truth_dataset(features, labels, args.output_dir, stats)

    # --- Cross-validation ---
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"STRATIFIED {args.cv_folds}-FOLD CROSS-VALIDATION", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    cv_results = cross_validate(features, labels, n_folds=args.cv_folds)

    if "error" in cv_results:
        print(f"CV failed: {cv_results['error']}", file=sys.stderr)
        sys.exit(1)

    mean = cv_results["mean_metrics"]
    print(f"\nMean metrics ({cv_results['algorithm']}, {args.cv_folds}-fold):", file=sys.stderr)
    print(f"  F1:        {mean['mean_f1']:.4f} +/- {mean['std_f1']:.4f}", file=sys.stderr)
    print(f"  Precision: {mean['mean_precision']:.4f} +/- {mean['std_precision']:.4f}", file=sys.stderr)
    print(f"  Recall:    {mean['mean_recall']:.4f} +/- {mean['std_recall']:.4f}", file=sys.stderr)
    print(f"  AUC-ROC:   {mean['mean_auc_roc']:.4f} +/- {mean['std_auc_roc']:.4f}", file=sys.stderr)

    # --- Feature importance ---
    print_feature_importance(features, labels)

    # --- Train final model on all data ---
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"TRAINING FINAL MODEL ON ALL DATA", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    result = train_model(
        features,
        labels,
        output_dir=args.output_dir,
        channel=args.channel,
        backend=args.backend,
    )

    if not result.get("success"):
        print(f"Training failed: {result.get('error', 'unknown')}", file=sys.stderr)
        sys.exit(1)

    metrics = result["metrics"]
    print(f"\nFinal model (80/20 holdout):", file=sys.stderr)
    print(f"  Algorithm:  {result['algorithm']}", file=sys.stderr)
    print(f"  F1:         {metrics['f1']:.4f}", file=sys.stderr)
    print(f"  Precision:  {metrics['precision']:.4f}", file=sys.stderr)
    print(f"  Recall:     {metrics['recall']:.4f}", file=sys.stderr)
    print(f"  AUC-ROC:    {metrics['auc_roc']:.4f}", file=sys.stderr)
    print(f"\n  Model:      {result['model_file']}", file=sys.stderr)
    print(f"  Metadata:   {result['metadata_file']}", file=sys.stderr)

    # --- Compare with CV metrics ---
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"COMPARISON", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"  CV mean F1:      {mean['mean_f1']:.4f} +/- {mean['std_f1']:.4f}", file=sys.stderr)
    print(f"  Holdout F1:      {metrics['f1']:.4f}", file=sys.stderr)
    # Flag if holdout is suspiciously higher than CV (possible overfit).
    if metrics["f1"] > mean["mean_f1"] + 2 * mean["std_f1"]:
        print(
            "  WARNING: Holdout F1 is >2 sigma above CV mean — possible overfit.",
            file=sys.stderr,
        )
    print(
        f"\n  NOTE: Status features are contaminated for departed bots "
        f"(leaving = login).\n"
        f"  The model should not rely heavily on status_* features.\n"
        f"  Check feature importance above to verify.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
