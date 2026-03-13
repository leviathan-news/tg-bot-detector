#!/usr/bin/env python3
"""
Real-time monitor for users leaving a Telegram channel.

Polls the channel admin log every --poll-interval seconds for new
leave/kick/ban events. ChatAction events don't fire for broadcast
channels, so admin log polling is the only reliable approach.

Tracks the most recent event ID to avoid duplicates across polls.
Each new departure is scored and written to a JSONL file immediately.

Usage:
    python scripts/monitor_leaves.py --channel @leviathan_news
    python scripts/monitor_leaves.py --channel @leviathan_news --poll-interval 30
    python scripts/monitor_leaves.py --channel @leviathan_news --output output/leaves.jsonl

Output format (JSONL, one JSON object per line):
    {
        "timestamp": "2026-03-12T16:04:40+00:00",
        "event": "ChannelAdminLogEventActionParticipantLeave",
        "user_id": 123456789,
        "profile": { ... full user attributes ... } | null,
        "heuristic_score": 4,
        "reasons": ["no_photo", "no_username", ...]
    }

Designed to run indefinitely in a tmux session. Ctrl+C exits cleanly.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telethon import TelegramClient
from telethon.tl.functions.channels import GetAdminLogRequest, GetFullChannelRequest
from telethon.tl.types import ChannelAdminLogEventsFilter

from tg_purge.config import load_config
from tg_purge.scoring import score_user


def serialize_user(user):
    """Extract all relevant attributes from a Telethon User object into a dict.

    Captures the full profile snapshot so we have it even after the user
    leaves or deletes their account. Includes all fields used by
    score_user() plus additional metadata for ML feature extraction.
    """
    if user is None:
        return None

    status_type = type(user.status).__name__ if user.status else None

    # Extract photo metadata — dc_id and has_video are available without
    # downloading the actual image and provide strong bot-detection signals.
    photo = getattr(user, "photo", None)
    photo_meta = None
    if photo is not None:
        photo_meta = {
            "photo_id": getattr(photo, "photo_id", None),
            "dc_id": getattr(photo, "dc_id", None),
            "has_video": bool(getattr(photo, "has_video", False)),
        }

    return {
        "id": user.id,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "username": user.username,
        "phone": None,  # Never log phone numbers (PII)
        "photo": user.photo is not None,
        "photo_meta": photo_meta,
        "status_type": status_type,
        "bot": user.bot,
        "deleted": user.deleted,
        "verified": getattr(user, "verified", False),
        "restricted": user.restricted,
        "scam": user.scam,
        "fake": user.fake,
        "premium": user.premium,
        "emoji_status": user.emoji_status is not None,
        # Additional fields for ML features
        "color": getattr(user, "color", None) is not None,
        "profile_color": getattr(user, "profile_color", None) is not None,
        "usernames_count": len(getattr(user, "usernames", None) or []),
        "stories_hidden": bool(getattr(user, "stories_hidden", False)),
        "stories_max_id": getattr(user, "stories_max_id", None) or 0,
        "contact_require_premium": bool(getattr(user, "contact_require_premium", False)),
    }


async def poll_admin_log(client, channel, min_id, limit=100):
    """Fetch new admin log events since min_id.

    Returns (events, users_map) where events are sorted newest-first
    and users_map is {user_id: User} from the response.
    Only fetches leave/kick/ban events.
    """
    result = await client(GetAdminLogRequest(
        channel=channel,
        q="",
        min_id=min_id,  # Only events with ID > min_id
        max_id=0,        # No upper bound
        limit=limit,
        events_filter=ChannelAdminLogEventsFilter(
            join=False,
            leave=True,
            ban=True,
            kick=True,
            invite=False,
            unkick=False,
            unban=False,
        ),
        admins=[],
    ))
    users_map = {u.id: u for u in result.users}
    return result.events, users_map


async def run(args):
    """Main async loop. Connects to Telegram, then polls the admin log
    at regular intervals for new departure events.
    """
    config = load_config(args.config)
    if args.channel:
        config.default_channel = args.channel
    if args.session_path:
        config.session_path = args.session_path

    config.validate_credentials()

    client = TelegramClient(
        config.session_path,
        int(config.api_id),
        config.api_hash,
    )
    await client.start()

    me = await client.get_me()
    print(f"Connected as: {me.first_name}", file=sys.stderr)

    channel_id = args.channel or config.default_channel
    entity = await client.get_entity(channel_id)

    try:
        full = await client(GetFullChannelRequest(entity))
        sub_count = full.full_chat.participants_count
    except Exception:
        sub_count = getattr(entity, "participants_count", "?")

    print(f"Channel: {entity.title}", file=sys.stderr)
    fmt_count = f"{sub_count:,}" if isinstance(sub_count, int) else str(sub_count)
    print(f"Subscribers: {fmt_count}", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)
    print(f"Poll interval: {args.poll_interval}s", file=sys.stderr)
    print(f"Monitoring for leave events via admin log polling...", file=sys.stderr)

    # Track the highest event ID we've seen so we only process new events
    # Start by fetching the latest event to get the current high-water mark
    initial_events, _ = await poll_admin_log(client, entity, min_id=0, limit=1)
    last_seen_id = initial_events[0].id if initial_events else 0
    print(f"Starting from admin log event ID: {last_seen_id}", file=sys.stderr)

    stats = {"total": 0, "likely_bot": 0, "start_time": time.time()}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            await asyncio.sleep(args.poll_interval)

            try:
                events, users_map = await poll_admin_log(
                    client, entity, min_id=last_seen_id
                )
            except Exception as e:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"Poll error: {e}",
                    file=sys.stderr,
                )
                continue

            if not events:
                # No new events — print periodic heartbeat every 5 polls
                elapsed = (time.time() - stats["start_time"]) / 3600
                if stats["total"] == 0 or int(elapsed * 60) % 5 == 0:
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"STATUS: {stats['total']} departures logged "
                        f"({stats['likely_bot']} likely bots) in {elapsed:.1f}h",
                        file=sys.stderr,
                    )
                continue

            # Process new events (they come newest-first, so reverse for
            # chronological processing)
            new_count = 0
            for event in reversed(events):
                if event.id <= last_seen_id:
                    continue  # Already processed

                user = users_map.get(event.user_id)
                profile = serialize_user(user)

                if user:
                    score, reasons = score_user(user)
                else:
                    score, reasons = None, []

                action_type = type(event.action).__name__

                record = {
                    "timestamp": event.date.isoformat(),
                    "event": action_type,
                    "user_id": event.user_id,
                    "profile": profile,
                    "heuristic_score": score,
                    "reasons": reasons,
                }

                # Append to JSONL and flush immediately
                with open(args.output, "a") as f:
                    f.write(json.dumps(record) + "\n")
                    f.flush()

                stats["total"] += 1
                if score is not None and score >= 3:
                    stats["likely_bot"] += 1
                new_count += 1

                # Print to stderr for tmux visibility
                name = "???"
                if profile:
                    name = profile.get("first_name") or "???"
                    if profile.get("last_name"):
                        name += " " + profile["last_name"]
                event_short = "LEFT" if "Leave" in action_type else action_type
                print(
                    f"[{event.date.strftime('%H:%M:%S')}] "
                    f"{event_short}: {name} (id={event.user_id}) "
                    f"score={score} reasons={reasons}",
                    file=sys.stderr,
                )

            # Update high-water mark to the newest event we processed
            if events:
                last_seen_id = max(e.id for e in events)

            if new_count > 0:
                elapsed = (time.time() - stats["start_time"]) / 3600
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"+{new_count} new | Total: {stats['total']} "
                    f"({stats['likely_bot']} likely bots) in {elapsed:.1f}h",
                    file=sys.stderr,
                )

    except KeyboardInterrupt:
        elapsed = (time.time() - stats["start_time"]) / 3600
        print(
            f"\nStopped. {stats['total']} departures logged "
            f"({stats['likely_bot']} likely bots) in {elapsed:.1f}h. "
            f"Data saved to {args.output}",
            file=sys.stderr,
        )
    finally:
        await client.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="Monitor users leaving a Telegram channel via admin log polling."
    )
    parser.add_argument(
        "--channel",
        required=True,
        help="Target channel username (e.g., @leviathan_news) or numeric ID.",
    )
    parser.add_argument(
        "--output",
        default=f"output/leaves-{datetime.now().strftime('%Y%m%d')}.jsonl",
        help="Output JSONL file path (default: output/leaves-YYYYMMDD.jsonl).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to TOML config file.",
    )
    parser.add_argument(
        "--session-path",
        default=None,
        help="Override session file path (use a separate copy to avoid "
             "SQLite locks when another tg-purge process is running).",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between admin log polls (default: 60).",
    )
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
