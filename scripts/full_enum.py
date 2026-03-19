#!/usr/bin/env python3
"""
Comprehensive subscriber enumeration using every available method.

Combines multiple strategies to maximize coverage beyond the standard
search-based enumeration:

1. Telethon iter_participants (aggressive) — tries hundreds of search
   prefixes internally, handles pagination and deduplication
2. Extended Unicode base queries — Hindi, Thai, Korean, Georgian, etc.
3. Emoji/special character prefixes — catches emoji-name accounts
4. Depth-5 recursive expansion on high-yield prefixes
5. ChannelParticipantsRecent with offset pagination

Results are merged, deduplicated, scored (heuristic + ML hybrid),
and exported to CSV.

Usage:
    python scripts/full_enum.py --channel @leviathan_news \
        --output output/full-enum.csv --delay 0.5
"""

import argparse
import asyncio
import csv
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telethon.tl.functions.channels import GetParticipantsRequest
from telethon.tl.types import (
    ChannelParticipant,
    ChannelParticipantsBots,
    ChannelParticipantsRecent,
    ChannelParticipantsSearch,
)
from telethon.errors import FloodWaitError

from tg_purge.config import load_config
from tg_purge.client import create_client, resolve_channel
from tg_purge.scoring import score_user, format_name, status_label


# Extended base queries covering more Unicode blocks.
EXTENDED_QUERIES = [
    # Latin
    *list("abcdefghijklmnopqrstuvwxyz"),
    # Cyrillic (full)
    *[chr(c) for c in range(0x0430, 0x0450)],  # а-я
    # Arabic (common)
    *[chr(c) for c in range(0x0627, 0x064B)],   # ا-ي
    # CJK common surnames
    "\u674e", "\u738b", "\u5f20", "\u5218", "\u9648",  # 李王张刘陈
    "\u6768", "\u8d75", "\u9ec4", "\u5468", "\u5434",  # 杨赵黄周吴
    # Hindi/Devanagari
    *[chr(c) for c in range(0x0905, 0x0940)],   # अ-ि
    # Thai
    *[chr(c) for c in range(0x0E01, 0x0E30)],   # ก-ะ
    # Korean Hangul syllables (common)
    "\uAC00", "\uB098", "\uB2E4", "\uB77C", "\uB9C8",  # 가나다라마
    "\uBC14", "\uC0AC", "\uC544", "\uC790", "\uCC28",  # 바사아자차
    # Georgian
    *[chr(c) for c in range(0x10D0, 0x10F1)],   # ა-ჰ
    # Numbers
    *list("0123456789"),
    # Common emoji prefixes (catches emoji-first-name accounts)
    "🔥", "💎", "🚀", "⚡", "🌟", "✨", "💰", "🎯", "👑", "🐾",
    "🆙", "🌱", "🍅", "🥠", "💧", "🦴", "🦆", "💠", "🐍", "🐸",
    # Special characters
    ".", "@", "_", "-", "~", "$", "0x",
    # Empty string — returns a different set
    "",
]

EXPANSION_CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789")
RESULT_CAP = 200


async def fetch_search(client, channel, query, limit=200):
    """Single search query, returns (users_list, participants_list)."""
    result = await client(GetParticipantsRequest(
        channel=channel,
        filter=ChannelParticipantsSearch(query),
        offset=0,
        limit=limit,
        hash=0,
    ))
    return result.users, result.participants


async def fetch_recent_paginated(client, channel, delay=0.5):
    """Paginate ChannelParticipantsRecent with offset until exhausted."""
    all_users = {}
    all_participants = {}
    offset = 0
    while True:
        try:
            result = await client(GetParticipantsRequest(
                channel=channel,
                filter=ChannelParticipantsRecent(),
                offset=offset,
                limit=200,
                hash=0,
            ))
        except FloodWaitError as e:
            print(f"  FloodWait {e.seconds}s on Recent offset={offset}", flush=True)
            await asyncio.sleep(e.seconds + 2)
            continue

        if not result.users:
            break

        new = 0
        for user in result.users:
            if user.id not in all_users:
                all_users[user.id] = user
                new += 1
        for p in result.participants:
            if p.user_id not in all_participants:
                all_participants[p.user_id] = p

        # If we got fewer than 200 or no new users, we're done.
        if len(result.users) < 200 or new == 0:
            break

        offset += len(result.users)
        await asyncio.sleep(delay)

    return all_users, all_participants


async def main():
    parser = argparse.ArgumentParser(description="Full enumeration of all channel subscribers")
    parser.add_argument("--channel", required=True, help="Channel username or ID")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between queries")
    parser.add_argument("--max-depth", type=int, default=5, help="Max recursive expansion depth")
    parser.add_argument("--safelist", default=None, help="Safelist CSV path")
    parser.add_argument("--session-path", default=None, help="Session file path")
    parser.add_argument("--config", default=None, help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.session_path:
        config.session_path = args.session_path

    interrupted = False
    def _shutdown(signum, frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGHUP, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    client = await create_client(config)
    channel = await resolve_channel(client, args.channel)
    sub_count = getattr(channel, "participants_count", None)
    print(f"Channel: {channel.title}")
    print(f"Subscribers: {sub_count:,}")

    all_users = {}
    join_dates = {}
    start = time.time()

    # ── Phase 1: ChannelParticipantsRecent with pagination ──
    print(f"\nPhase 1: Recent participants (paginated)...", flush=True)
    recent_users, recent_parts = await fetch_recent_paginated(client, channel, args.delay)
    for uid, user in recent_users.items():
        all_users[uid] = user
    for uid, p in recent_parts.items():
        if isinstance(p, ChannelParticipant) and hasattr(p, "date") and p.date:
            join_dates[uid] = p.date
    print(f"  Recent: {len(recent_users)} users", flush=True)

    # ── Phase 2: Bots ──
    print(f"\nPhase 2: Bot accounts...", flush=True)
    try:
        result = await client(GetParticipantsRequest(
            channel=channel,
            filter=ChannelParticipantsBots(),
            offset=0, limit=200, hash=0,
        ))
        for user in result.users:
            all_users[user.id] = user
        print(f"  Bots: {len(result.users)}", flush=True)
    except Exception as e:
        print(f"  Bots: error — {e}", flush=True)
    await asyncio.sleep(args.delay)

    # ── Phase 3: Extended search with deep recursive expansion ──
    print(f"\nPhase 3: Extended search ({len(EXTENDED_QUERIES)} base queries, "
          f"max_depth={args.max_depth})...", flush=True)

    # Deduplicate base queries.
    seen_queries = set()
    from collections import deque
    work_queue = deque()
    for q in EXTENDED_QUERIES:
        if q not in seen_queries:
            seen_queries.add(q)
            work_queue.append((q, 0))

    total_queries = len(work_queue)
    completed = 0

    while work_queue and not interrupted:
        query, depth = work_queue.popleft()

        try:
            users, participants = await fetch_search(client, channel, query)

            new = 0
            for user in users:
                if user.id not in all_users:
                    all_users[user.id] = user
                    new += 1

            for p in participants:
                if p.user_id not in join_dates:
                    if isinstance(p, ChannelParticipant) and hasattr(p, "date") and p.date:
                        join_dates[p.user_id] = p.date

            # Recursive expansion if we hit the cap.
            if len(users) >= RESULT_CAP and depth < args.max_depth:
                for ch in EXPANSION_CHARS:
                    sub_q = query + ch
                    if sub_q not in seen_queries:
                        seen_queries.add(sub_q)
                        work_queue.append((sub_q, depth + 1))
                        total_queries += 1

        except FloodWaitError as e:
            wait = e.seconds + 2
            print(f"\n  FloodWait {e.seconds}s at query {completed}/{total_queries} — sleeping {wait}s...",
                  flush=True)
            await asyncio.sleep(wait)
            # Re-queue this query.
            work_queue.appendleft((query, depth))
            continue
        except Exception as e:
            pass  # Skip failed queries.

        completed += 1
        if completed % 50 == 0:
            elapsed = time.time() - start
            pct = len(all_users) / sub_count * 100 if sub_count else 0
            print(f"  ...{completed}/{total_queries} queries, "
                  f"{len(all_users)} users ({pct:.1f}% coverage)",
                  flush=True)

        await asyncio.sleep(args.delay)

    # ── Phase 4: Telethon iter_participants aggressive ──
    # This uses Telethon's built-in aggressive enumeration which tries
    # additional search strategies we might have missed.
    if not interrupted:
        print(f"\nPhase 4: Telethon aggressive enumeration...", flush=True)
        before = len(all_users)
        try:
            async for user in client.iter_participants(
                channel, aggressive=True
            ):
                if interrupted:
                    break
                if user.id not in all_users:
                    all_users[user.id] = user
        except FloodWaitError as e:
            print(f"  FloodWait {e.seconds}s — skipping aggressive enum", flush=True)
        except Exception as e:
            print(f"  Aggressive enum error: {e}", flush=True)
        after = len(all_users)
        print(f"  Aggressive: +{after - before} new users", flush=True)

    await client.disconnect()

    # ── Scoring ──
    elapsed = time.time() - start
    pct = len(all_users) / sub_count * 100 if sub_count else 0
    print(f"\nEnumeration complete: {len(all_users)} users "
          f"({pct:.1f}% of {sub_count:,}) in {elapsed:.0f}s", flush=True)

    print(f"\nScoring {len(all_users)} users...", flush=True)

    # Auto-detect spike windows.
    from tg_purge.clustering import detect_spike_windows
    spike_windows = detect_spike_windows(join_dates) if join_dates else []
    if spike_windows:
        print(f"  {len(spike_windows)} spike windows detected")

    # Score all users.
    scored = []
    for uid, user in all_users.items():
        s, reasons = score_user(
            user,
            join_date=join_dates.get(uid),
            spike_windows=spike_windows,
        )
        scored.append((user, s, reasons))

    scored.sort(key=lambda x: -x[1])

    # ── ML scoring ──
    ml_predictions = {}
    try:
        from tg_purge.features import extract_features
        from tg_purge.ml import ml_available, predict
        import glob

        if ml_available():
            model_path = None
            for pattern in ["models/*_lightgbm.model", "models/*_xgboost.model"]:
                matches = sorted(glob.glob(pattern))
                if matches:
                    model_path = matches[-1]
                    break

            if model_path:
                print(f"Running ML inference ({model_path})...", flush=True)

                # Photo quality from stripped thumbnails.
                photo_cache = {}
                try:
                    from tg_purge.photo_analysis import extract_photo_quality
                    for uid, user in all_users.items():
                        pq = extract_photo_quality(user)
                        if pq:
                            photo_cache[uid] = pq
                    print(f"  Photo quality: {len(photo_cache)} users")
                except ImportError:
                    pass

                feature_vecs = []
                uid_order = []
                for uid, user in all_users.items():
                    feats = extract_features(
                        user,
                        join_date=join_dates.get(uid),
                        spike_windows=spike_windows,
                        photo_quality=photo_cache.get(uid),
                    )
                    feature_vecs.append(feats)
                    uid_order.append(uid)

                preds = predict(feature_vecs, model_path)
                for uid, pred in zip(uid_order, preds):
                    ml_predictions[uid] = pred

                ml_bots = sum(1 for p in preds if p["label"] == "bot")
                print(f"  ML: {ml_bots} bots out of {len(preds)}")
    except Exception as e:
        print(f"  ML scoring failed: {e}", file=sys.stderr)

    # ── Export ──
    print(f"\nExporting to {args.output}...", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "user_id", "name", "username", "heuristic_score",
            "ml_probability", "ml_label", "status", "signals",
        ])
        for user, s, reasons in scored:
            pred = ml_predictions.get(user.id, {})
            prob = f"{pred['probability']:.4f}" if "probability" in pred else ""
            label = pred.get("label", "")
            writer.writerow([
                user.id, format_name(user), user.username or "",
                s, prob, label, status_label(user),
                "; ".join(reasons),
            ])

    from collections import Counter
    scores = Counter(s for _, s, _ in scored)
    print(f"\nTotal: {len(scored)} users exported")
    print(f"Score distribution:")
    for s in sorted(scores.keys(), reverse=True):
        print(f"  Score {s}: {scores[s]}")

    if ml_predictions:
        ml_bots = sum(1 for _, s, _ in scored if ml_predictions.get(_.id if hasattr(_, 'id') else 0, {}).get("label") == "bot")
        # Recount properly
        mb = sum(1 for u, s, r in scored if ml_predictions.get(u.id, {}).get("label") == "bot")
        print(f"ML bots: {mb}")


if __name__ == "__main__":
    asyncio.run(main())
