"""
Interactive label review for tg-bot-detector.

Loads unlabeled users from datasets/<channel>/labels.json, fetches their
current profiles via Telethon, and presents them in batches for human
review. Updated labels are saved back to labels.json with source="human".

Prioritizes score-3 users first (closest to bot threshold, most
informative for ML), then score-2, then score-1.

Usage:
    .venv/bin/python scripts/review_labels.py --channel @leviathan_news
    .venv/bin/python scripts/review_labels.py --channel @leviathan_news --batch-size 20
    .venv/bin/python scripts/review_labels.py --channel @leviathan_news --score 2

Controls during review:
    b  = bot
    h  = human
    s  = skip (keep unlabeled)
    q  = quit and save progress
"""

import asyncio
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path so tg_purge is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tg_purge.config import load_config
from tg_purge.client import create_client, resolve_channel
from tg_purge.utils import channel_slug


def _format_user(user_data: dict, index: int, total: int) -> str:
    """Format a single user for terminal display.

    Renders a compact block with all relevant profile signals so the
    reviewer can make a quick bot/human decision.

    Args:
        user_data: Dict with user profile fields (id, first_name, etc.).
        index:     1-based position in the current review batch.
        total:     Total number of users in the current batch.

    Returns:
        Formatted multi-line string for terminal output.
    """
    # Build the name display string.
    name_parts = [user_data.get("first_name", "")]
    if user_data.get("last_name"):
        name_parts.append(user_data["last_name"])
    name = " ".join(name_parts).strip() or "(no name)"

    username = user_data.get("username", "")
    username_display = f"@{username}" if username else "—"

    # Collect signal indicators as compact tags.
    signals = []
    if user_data.get("photo"):
        signals.append("📷 photo")
    else:
        signals.append("no photo")

    if user_data.get("premium"):
        signals.append("⭐ premium")

    if user_data.get("deleted"):
        signals.append("🗑 DELETED")

    if user_data.get("bot_flag"):
        signals.append("🤖 BOT API")

    if user_data.get("spike"):
        signals.append("📈 spike join")

    status = user_data.get("status", "unknown")
    days = user_data.get("days_joined", -1)
    h_score = user_data.get("h_score", 0)

    return (
        f"\n{'─' * 60}\n"
        f"  [{index}/{total}]  ID: {user_data['id']}\n"
        f"  Name:     {name}\n"
        f"  Username: {username_display}\n"
        f"  Status:   {status}  |  Joined: {days}d ago  |  Score: {h_score}\n"
        f"  Signals:  {', '.join(signals)}\n"
        f"{'─' * 60}"
    )


async def fetch_user_profiles(client, channel, user_ids: list, features: dict) -> list:
    """Fetch live profiles for a list of user IDs.

    Uses GetParticipantRequest (one at a time) since GetUsersRequest with
    access_hash=0 doesn't resolve users we haven't interacted with directly.
    GetParticipantRequest works because we have the channel context.

    Args:
        client:   Connected TelegramClient.
        channel:  Resolved channel entity.
        user_ids: List of integer user IDs to fetch.
        features: Dict of uid_str -> feature dict for heuristic score lookup.

    Returns:
        List of dicts with user profile fields ready for display.
    """
    from telethon.tl.functions.channels import GetParticipantRequest

    results = []
    for i, uid in enumerate(user_ids):
        try:
            p = await client(GetParticipantRequest(channel, uid))
            u = p.users[0] if p.users else None
            if u:
                feat = features.get(str(u.id), {})
                status_type = type(u.status).__name__ if u.status else "None"
                results.append({
                    "id": u.id,
                    "first_name": u.first_name or "",
                    "last_name": u.last_name or "",
                    "username": u.username or "",
                    "photo": bool(u.photo),
                    "premium": bool(u.premium),
                    "deleted": bool(u.deleted),
                    "bot_flag": bool(u.bot),
                    "status": status_type,
                    "h_score": int(feat.get("heuristic_score", 0)),
                    "spike": bool(feat.get("is_spike_join")),
                    "days_joined": int(feat.get("days_since_join", -1)),
                })
        except Exception as e:
            # User may have left the channel or been deleted.
            print(f"  Could not fetch {uid}: {e}", file=sys.stderr)

        # Rate limit — GetParticipantRequest is lighter than search but
        # still subject to Telegram flood protection.
        if (i + 1) % 30 == 0:
            await asyncio.sleep(1.0)
        else:
            await asyncio.sleep(0.15)

        # Progress indicator for large batches.
        if (i + 1) % 10 == 0:
            print(
                f"  Fetching profiles... {i + 1}/{len(user_ids)}",
                file=sys.stderr,
            )

    return results


def save_updated_labels(labels_data: dict, path: Path) -> None:
    """Write updated labels back to disk with PII-safe permissions.

    Args:
        labels_data: Full labels dict with "channel", "version", "labels" keys.
        path:        Path to the labels.json file.
    """
    # Convert int keys back to strings for JSON serialization.
    payload = {
        "channel": labels_data["channel"],
        "version": labels_data["version"],
        "labels": {str(k): v for k, v in labels_data["labels"].items()},
    }

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    try:
        os.chmod(str(path), stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


async def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Interactive label review for unlabeled users"
    )
    parser.add_argument("--channel", required=True, help="Channel identifier")
    parser.add_argument("--config", default=None, help="Path to config.toml")
    parser.add_argument(
        "--batch-size", type=int, default=30, dest="batch_size",
        help="Number of users to review per batch (default: 30)",
    )
    parser.add_argument(
        "--score", type=int, default=None,
        help="Only review users with this heuristic score (default: all unlabeled, highest first)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    channel_name = args.channel
    slug = channel_slug(channel_name)

    # Load labels and features.
    labels_path = Path("datasets") / slug / "labels.json"
    features_path = Path("datasets") / slug / "features.json"

    if not labels_path.exists():
        print(f"Labels file not found: {labels_path}", file=sys.stderr)
        sys.exit(1)
    if not features_path.exists():
        print(f"Features file not found: {features_path}", file=sys.stderr)
        sys.exit(1)

    with open(labels_path, "r", encoding="utf-8") as fh:
        raw_labels = json.load(fh)
    # Keep int keys internally for consistency with labeling.py.
    labels_data = {
        "channel": raw_labels.get("channel", ""),
        "version": raw_labels.get("version", 1),
        "labels": {int(k): v for k, v in raw_labels.get("labels", {}).items()},
    }

    with open(features_path, "r", encoding="utf-8") as fh:
        features = json.load(fh).get("features", {})

    # Find unlabeled users and sort by score descending (score 3 first).
    unlabeled = []
    for uid, info in labels_data["labels"].items():
        if info.get("label") != "unlabeled":
            continue
        feat = features.get(str(uid), {})
        h_score = int(feat.get("heuristic_score", 0))
        if args.score is not None and h_score != args.score:
            continue
        unlabeled.append((uid, h_score))

    # Sort: highest score first (most informative for model improvement).
    unlabeled.sort(key=lambda x: x[1], reverse=True)

    if not unlabeled:
        print("No unlabeled users to review.", file=sys.stderr)
        sys.exit(0)

    print(
        f"\nFound {len(unlabeled)} unlabeled users to review.\n"
        f"Score distribution: "
        + ", ".join(
            f"score {s}: {sum(1 for _, sc in unlabeled if sc == s)}"
            for s in sorted(set(sc for _, sc in unlabeled), reverse=True)
        ),
        file=sys.stderr,
    )

    # Connect to Telegram for profile fetching.
    client = await create_client(config)
    try:
        channel = await resolve_channel(client, channel_name)

        batch_start = 0
        total_reviewed = 0
        total_labeled = 0
        quit_requested = False

        while batch_start < len(unlabeled) and not quit_requested:
            batch = unlabeled[batch_start:batch_start + args.batch_size]
            batch_ids = [uid for uid, _ in batch]

            print(
                f"\n{'=' * 60}\n"
                f"  Fetching batch {batch_start // args.batch_size + 1} "
                f"({len(batch)} users, starting at #{batch_start + 1})...\n"
                f"{'=' * 60}",
                file=sys.stderr,
            )

            # Fetch live profiles for this batch.
            profiles = await fetch_user_profiles(
                client, channel, batch_ids, features,
            )

            # Present each user for review.
            for i, profile in enumerate(profiles):
                print(_format_user(profile, i + 1, len(profiles)))
                print("  Label: [b]ot  [h]uman  [s]kip  [q]uit+save")

                while True:
                    try:
                        choice = input("  > ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        # Ctrl+C or Ctrl+D — save and exit.
                        choice = "q"

                    if choice in ("b", "bot"):
                        new_label = "bot"
                        break
                    elif choice in ("h", "human"):
                        new_label = "human"
                        break
                    elif choice in ("s", "skip", ""):
                        new_label = None
                        break
                    elif choice in ("q", "quit"):
                        quit_requested = True
                        new_label = None
                        break
                    else:
                        print("  Invalid input. Use: b, h, s, or q")

                if quit_requested:
                    break

                total_reviewed += 1

                if new_label is not None:
                    # Update the label in-memory.
                    uid = profile["id"]
                    labels_data["labels"][uid] = {
                        "label": new_label,
                        "source": "human",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    total_labeled += 1
                    print(f"  → Labeled as {new_label.upper()}")

            batch_start += args.batch_size

            # Save after each batch to avoid losing progress.
            if total_labeled > 0:
                save_updated_labels(labels_data, labels_path)
                print(
                    f"\n  Saved progress: {total_labeled} labeled, "
                    f"{total_reviewed} reviewed.",
                    file=sys.stderr,
                )

    finally:
        await client.disconnect()

    # Final save.
    if total_labeled > 0:
        save_updated_labels(labels_data, labels_path)

    # Print summary.
    from tg_purge.labeling import label_stats
    stats = label_stats(labels_data["labels"])
    print(
        f"\n{'=' * 60}\n"
        f"  Review complete!\n"
        f"  Reviewed: {total_reviewed}  |  Labeled: {total_labeled}\n\n"
        f"  Updated label statistics:\n"
        f"    Total:          {stats['total']}\n"
        f"    Bot:            {stats['bot']}\n"
        f"    Human:          {stats['human']}\n"
        f"    Unlabeled:      {stats['unlabeled']}\n"
        f"    Human-reviewed: {stats['human_labeled']}\n"
        f"{'=' * 60}\n\n"
        f"  To retrain the model:\n"
        f"    .venv/bin/python -m tg_purge ml train --channel {channel_name}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    asyncio.run(main())
