#!/usr/bin/env python3
"""
Validate new ML features using labeled humans and ALL departed bots.

Uses ground-truth human labels from labels.json (not random subscribers)
and fetches live User objects for both groups to extract extended MTProto
fields (DC, color, stories, etc.).

IMPORTANT: Status fields for departed bots are unreliable — the act of
leaving the channel updates their last-seen timestamp, so they appear
"recently active" even if they were dormant before leaving.

Usage:
    python scripts/validate_new_features.py \
        --channel @leviathan_news \
        --departed-ids output/admin-log-departures-20260312.jsonl \
        --labels-path datasets/leviathan_news/labels.json
"""

import argparse
import asyncio
import json
import signal
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telethon import TelegramClient

from tg_purge.config import load_config
from tg_purge.scoring import score_user


def extract_all_fields(user):
    """Extract every field we care about from a User object.

    Returns a flat dict of feature values for statistical analysis.
    Does NOT include status fields — those are extracted separately
    since they're unreliable for departed bots.
    """
    photo = getattr(user, "photo", None)

    # Photo metadata — dc_id and has_video are available inline
    # without downloading the image. DC distribution is a strong
    # bot signal (bot farms cluster on specific DCs).
    dc_id = 0
    has_video = False
    if photo is not None:
        dc_id = getattr(photo, "dc_id", 0) or 0
        has_video = bool(getattr(photo, "has_video", False))

    # Status info — included for completeness but flagged as unreliable
    # for departed bots (leaving = login = status update).
    status = getattr(user, "status", None)
    status_type = type(status).__name__ if status else "None"

    first = user.first_name or ""

    return {
        "user_id": user.id,
        "first_name": first,
        # --- Profile flags ---
        "has_photo": bool(photo),
        "has_username": bool(user.username),
        "has_last_name": bool(user.last_name),
        "is_premium": bool(getattr(user, "premium", False)),
        "has_emoji_status": bool(getattr(user, "emoji_status", None)),
        "is_deleted": bool(getattr(user, "deleted", False)),
        "is_bot_api": bool(getattr(user, "bot", False)),
        "is_scam": bool(getattr(user, "scam", False)),
        "is_fake": bool(getattr(user, "fake", False)),
        "is_restricted": bool(getattr(user, "restricted", False)),
        # --- Status (unreliable for departed bots) ---
        "status_type": status_type,
        # --- Photo metadata ---
        "photo_dc_id": dc_id,
        "photo_has_video": has_video,
        # --- Extended profile (MTProto layer 160+) ---
        "has_custom_color": bool(getattr(user, "color", None)),
        "has_profile_color": bool(getattr(user, "profile_color", None)),
        "has_stories": bool(getattr(user, "stories_max_id", None)),
        "stories_unavailable": bool(getattr(user, "stories_unavailable", False)),
        "has_contact_require_premium": bool(getattr(user, "contact_require_premium", False)),
        "usernames_count": len(getattr(user, "usernames", None) or []),
        "is_verified": bool(getattr(user, "verified", False)),
        "has_lang_code": bool(getattr(user, "lang_code", None)),
        "lang_code": getattr(user, "lang_code", None) or "",
        "has_paid_messages": bool(getattr(user, "send_paid_messages_stars", None)),
        # --- Heuristic score ---
        "heuristic_score": score_user(user)[0],
    }


async def fetch_users_by_ids(client, user_ids, label):
    """Fetch live User objects for a list of user IDs.

    Fetches ALL IDs (no cap). Skips deleted accounts and errors.
    Prints progress every 100 users to stderr.

    Args:
        client: Connected TelegramClient.
        user_ids: List of integer user IDs to fetch.
        label: Label string for progress messages (e.g., "humans", "bots").

    Returns:
        Dict of user_id -> User for accessible, non-deleted accounts.
    """
    users = {}
    errors = 0

    for i, uid in enumerate(user_ids):
        try:
            user = await client.get_entity(uid)
            # Skip fully deleted accounts — they have no useful profile data.
            if not getattr(user, "deleted", False):
                users[uid] = user
        except Exception:
            errors += 1

        if (i + 1) % 100 == 0:
            print(
                f"  [{label}] {i + 1}/{len(user_ids)} checked, "
                f"{len(users)} accessible, {errors} errors",
                file=sys.stderr,
            )
        # Rate limit: 0.3s between requests to avoid flood waits.
        await asyncio.sleep(0.3)

    print(
        f"  [{label}] Done: {len(users)} accessible out of "
        f"{len(user_ids)} ({errors} errors)",
        file=sys.stderr,
    )
    return users


def print_comparison(bot_data, human_data):
    """Print full statistical comparison of all fields.

    Status fields are flagged with a warning since departed bots have
    contaminated status (leaving = login = status update).
    """

    print(f"\n{'=' * 90}")
    print(f"FEATURE VALIDATION: BOTS ({len(bot_data)}) vs HUMANS ({len(human_data)})")
    print(f"{'=' * 90}")
    print(f"  Bot source:   ALL departed users (admin log)")
    print(f"  Human source: Labeled humans (labels.json ground truth)")

    # Boolean fields — compare prevalence rates
    bool_fields = [
        ("has_photo", "Has profile photo"),
        ("has_username", "Has @username"),
        ("has_last_name", "Has last name"),
        ("is_premium", "Has Premium"),
        ("has_emoji_status", "Has emoji status"),
        ("photo_has_video", "Has video profile"),
        ("has_custom_color", "Has custom color"),
        ("has_profile_color", "Has profile color"),
        ("has_stories", "Has posted stories"),
        ("stories_unavailable", "Stories unavailable"),
        ("has_contact_require_premium", "Requires premium to contact"),
        ("is_verified", "Telegram verified"),
        ("has_lang_code", "Has language code"),
        ("has_paid_messages", "Charges for messages"),
    ]

    print(f"\n{'Field':40s}  {'BOTS':>10s}  {'HUMANS':>10s}  {'Delta':>10s}  {'Signal':>8s}")
    print(f"{'─' * 40}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 8}")

    for field, label in bool_fields:
        bot_count = sum(1 for d in bot_data if d.get(field))
        human_count = sum(1 for d in human_data if d.get(field))
        bot_pct = bot_count / max(len(bot_data), 1) * 100
        human_pct = human_count / max(len(human_data), 1) * 100
        delta = bot_pct - human_pct

        abs_delta = abs(delta)
        if abs_delta > 30:
            strength = "STRONG"
        elif abs_delta > 10:
            strength = "MEDIUM"
        elif abs_delta > 3:
            strength = "WEAK"
        else:
            strength = "NONE"

        print(f"{label:40s}  {bot_pct:>9.1f}%  {human_pct:>9.1f}%  {delta:>+9.1f}%  {strength:>8s}")

    # Numeric fields
    import statistics

    print(f"\n{'─' * 90}")
    print(f"NUMERIC DISTRIBUTIONS")
    print(f"{'─' * 90}")

    print(f"\n{'Field':40s}  {'BOTS mean':>10s}  {'BOTS med':>10s}  {'HUMANS mean':>10s}  {'HUMANS med':>10s}")
    print(f"{'─' * 40}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}")

    for field, label in [("usernames_count", "Collectible usernames"),
                          ("heuristic_score", "Heuristic score")]:
        bot_vals = [d[field] for d in bot_data]
        human_vals = [d[field] for d in human_data]
        print(f"{label:40s}  {statistics.mean(bot_vals):>10.2f}  "
              f"{statistics.median(bot_vals):>10.1f}  "
              f"{statistics.mean(human_vals):>10.2f}  "
              f"{statistics.median(human_vals):>10.1f}")

    # DC distribution
    print(f"\n{'─' * 90}")
    print(f"DATA CENTER DISTRIBUTION (photo storage)")
    print(f"{'─' * 90}")

    bot_dcs = Counter(d["photo_dc_id"] for d in bot_data if d["photo_dc_id"] > 0)
    human_dcs = Counter(d["photo_dc_id"] for d in human_data if d["photo_dc_id"] > 0)
    bot_with_photo = sum(bot_dcs.values())
    human_with_photo = sum(human_dcs.values())

    print(f"  Users with photos: BOTS={bot_with_photo}, HUMANS={human_with_photo}")
    all_dcs = sorted(set(bot_dcs.keys()) | set(human_dcs.keys()))
    for dc in all_dcs:
        b = bot_dcs.get(dc, 0)
        h = human_dcs.get(dc, 0)
        b_pct = b / max(bot_with_photo, 1) * 100
        h_pct = h / max(human_with_photo, 1) * 100
        print(f"  DC {dc}:  BOTS {b_pct:5.1f}% ({b:4d})   HUMANS {h_pct:5.1f}% ({h:4d})")

    # Language code distribution
    print(f"\n{'─' * 90}")
    print(f"LANGUAGE CODE DISTRIBUTION (top 10)")
    print(f"{'─' * 90}")

    bot_langs = Counter(d["lang_code"] for d in bot_data if d["lang_code"])
    human_langs = Counter(d["lang_code"] for d in human_data if d["lang_code"])

    all_langs = set(list(dict(bot_langs.most_common(10)).keys()) +
                    list(dict(human_langs.most_common(10)).keys()))
    bot_with_lang = sum(bot_langs.values())
    human_with_lang = sum(human_langs.values())
    print(f"  Users with lang_code: BOTS={bot_with_lang}/{len(bot_data)}, "
          f"HUMANS={human_with_lang}/{len(human_data)}")

    for lang in sorted(all_langs, key=lambda l: -(bot_langs.get(l, 0) + human_langs.get(l, 0))):
        b = bot_langs.get(lang, 0)
        h = human_langs.get(lang, 0)
        b_pct = b / max(bot_with_lang, 1) * 100
        h_pct = h / max(human_with_lang, 1) * 100
        print(f"  {lang or '(empty)':6s}:  BOTS {b_pct:5.1f}% ({b:4d})   HUMANS {h_pct:5.1f}% ({h:4d})")

    # Status type distribution — with warning
    print(f"\n{'─' * 90}")
    print(f"STATUS TYPE DISTRIBUTION")
    print(f"  ⚠ Bot status is UNRELIABLE — leaving the channel counts")
    print(f"    as a login, so bots appear 'recently active'.")
    print(f"{'─' * 90}")

    bot_status = Counter(d["status_type"] for d in bot_data)
    human_status = Counter(d["status_type"] for d in human_data)
    all_statuses = sorted(set(bot_status.keys()) | set(human_status.keys()))
    for s in all_statuses:
        b_pct = bot_status.get(s, 0) / max(len(bot_data), 1) * 100
        h_pct = human_status.get(s, 0) / max(len(human_data), 1) * 100
        print(f"  {s:30s}  BOTS {b_pct:5.1f}%   HUMANS {h_pct:5.1f}%")


def save_results(bot_data, human_data, output_path):
    """Save raw validation data to JSON. Runs synchronously with SIGINT
    blocked so partial results are never lost on Ctrl+C.
    """
    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "bot_source": "departed users (admin log)",
                "human_source": "labeled humans (labels.json)",
                "bot_count": len(bot_data),
                "human_count": len(human_data),
                "bot_data": bot_data,
                "human_data": human_data,
            }, f, indent=2, default=str)
        print(f"\nRaw data saved to: {output_path}", file=sys.stderr)
    finally:
        signal.signal(signal.SIGINT, original_handler)


async def run(args):
    config = load_config(args.config)
    if args.channel:
        config.default_channel = args.channel
    if args.session_path:
        config.session_path = args.session_path
    config.validate_credentials()

    # Load departed user IDs from admin log JSONL.
    departed_ids = []
    with open(args.departed_ids) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                departed_ids.append(int(r["user_id"]))
    # Deduplicate — same user may appear multiple times in the log.
    departed_ids = list(dict.fromkeys(departed_ids))
    print(f"Loaded {len(departed_ids)} unique departed IDs", file=sys.stderr)

    # Load labeled human IDs from labels.json.
    with open(args.labels_path) as f:
        labels_data = json.load(f)
    human_ids = [
        int(uid) for uid, info in labels_data["labels"].items()
        if info["label"] == "human"
    ]
    print(f"Loaded {len(human_ids)} labeled human IDs", file=sys.stderr)

    # Connect to Telegram and fetch live User objects for both groups.
    client = TelegramClient(config.session_path, int(config.api_id), config.api_hash)
    await client.start()
    me = await client.get_me()
    print(f"Connected as: {me.first_name}", file=sys.stderr)

    try:
        print(f"\nFetching ALL {len(departed_ids)} departed users...", file=sys.stderr)
        bot_users = await fetch_users_by_ids(client, departed_ids, "bots")

        print(f"\nFetching ALL {len(human_ids)} labeled humans...", file=sys.stderr)
        human_users = await fetch_users_by_ids(client, human_ids, "humans")
    finally:
        await client.disconnect()

    # Extract features — all sync from here, safe from CancelledError.
    print(f"\nExtracting features...", file=sys.stderr)
    bot_data = [extract_all_fields(u) for u in bot_users.values()]
    human_data = [extract_all_fields(u) for u in human_users.values()]

    print_comparison(bot_data, human_data)

    output = args.output or "output/feature-validation-full.json"
    save_results(bot_data, human_data, output)


def main():
    parser = argparse.ArgumentParser(
        description="Validate ML features: labeled humans vs ALL departed bots."
    )
    parser.add_argument("--channel", required=True)
    parser.add_argument(
        "--departed-ids", dest="departed_ids", required=True,
        help="JSONL of departed users (admin log dump).",
    )
    parser.add_argument(
        "--labels-path", dest="labels_path", required=True,
        help="Path to labels.json with ground-truth human labels.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--session-path", dest="session_path", default=None)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
